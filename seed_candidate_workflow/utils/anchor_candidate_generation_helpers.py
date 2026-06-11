from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from seed_candidate_workflow.utils import graph_structure_helpers as gh
from seed_candidate_workflow.utils.config_run_fields import resolve_graph_id
from seed_candidate_workflow.utils.anchor_candidate_rare_artifact_helpers import (
    _load_seed_pairs,
    _resolve_latest_seed_dir,
    generate_candidates_rare_artifact_v1,
)
from seed_candidate_workflow.utils.anchor_candidate_semantic_reciprocal_helpers import (
    _compute_mutual_topk_cosine_candidates,
    _pair_time_gap_seconds,
)
from seed_candidate_workflow.utils.anchor_candidate_component_expansion_helpers import (
    generate_component_expansion_v1,
)
from seed_candidate_workflow.utils.anchor_candidate_2hop_bounded_helpers import (
    generate_candidates_2hop_bounded_v1,
)
from seed_candidate_workflow.utils.anchor_candidate_shared_stem_highconf_helpers import (
    generate_candidates_shared_stem_highconf_v1,
)
from seed_candidate_workflow.utils.anchor_candidate_body_similarity_helpers import (
    BODY_GENERATOR_NAMES,
    build_semantic_band_pool_for_body_generators,
    generate_body_char4gram_jaccard_highconf_v1,
    generate_body_token_jaccard_highconf_v1,
    prepare_body_feature_store,
)
from seed_candidate_workflow.utils.body_similarity_progress import progress_from_cfg
from seed_candidate_workflow.utils.anchor_candidate_semantic_mid_support_helpers import (
    generate_semantic_mid_core_support_v1,
    generate_semantic_mid_sender_support_v1,
    generate_semantic_mid_senderlocalpart_support_v1,
    generate_semantic_mid_stem_support_v1,
)
from seed_candidate_workflow.utils.pair_similarity_features import load_misp_text_catalog_for_pairs
from seed_candidate_workflow.utils.anchor_candidate_eval_helpers import run_candidate_evaluation_report
from seed_candidate_workflow.utils.anchor_graph_helpers import load_anchor_graph_artifacts, load_embedding_vectors

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None


def _pair_key(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a <= b else (b, a)


def _safe_float(x: Any) -> float:
    v = pd.to_numeric(x, errors="coerce")
    return float(v) if pd.notna(v) else float("nan")


def _as_windows_extended_path(path: Path) -> str:
    """Return a path string safe for open()/to_csv on Windows (MAX_PATH bypass)."""
    resolved = path.resolve()
    if os.name != "nt":
        return str(resolved)
    s = str(resolved)
    if s.startswith("\\\\?\\"):
        return s
    if s.startswith("\\\\"):
        return "\\\\?\\UNC\\" + s[2:]
    return "\\\\?\\" + s


def _write_candidate_csv(df: pd.DataFrame, path: Path) -> Path:
    """Write candidate CSV, ensuring parent directory exists (Windows-safe)."""
    p = Path(path).resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(_as_windows_extended_path(p), index=False)
    return p


def _pairs_from_df(df: pd.DataFrame) -> set[tuple[str, str]]:
    if df.empty or not {"email_i", "email_j"}.issubset(df.columns):
        return set()
    out: set[tuple[str, str]] = set()
    for a, b in zip(df["email_i"].astype(str).tolist(), df["email_j"].astype(str).tolist(), strict=False):
        if a == b:
            continue
        out.add(_pair_key(a, b))
    return out


def _aggregate_pair_metric_max(
    df: pd.DataFrame,
    metric_col: str,
) -> dict[tuple[str, str], float]:
    if df.empty or metric_col not in df.columns:
        return {}
    out: dict[tuple[str, str], float] = {}
    for a, b, m in zip(
        df["email_i"].astype(str).tolist(),
        df["email_j"].astype(str).tolist(),
        df[metric_col].tolist(),
        strict=False,
    ):
        if a == b:
            continue
        k = _pair_key(a, b)
        v = _safe_float(m)
        if pd.notna(v):
            out[k] = max(out.get(k, float("-inf")), v)
    return out


def _aggregate_pair_metric_min(
    df: pd.DataFrame,
    metric_col: str,
) -> dict[tuple[str, str], float]:
    if df.empty or metric_col not in df.columns:
        return {}
    out: dict[tuple[str, str], float] = {}
    for a, b, m in zip(
        df["email_i"].astype(str).tolist(),
        df["email_j"].astype(str).tolist(),
        df[metric_col].tolist(),
        strict=False,
    ):
        if a == b:
            continue
        k = _pair_key(a, b)
        v = _safe_float(m)
        if pd.notna(v):
            out[k] = min(out.get(k, float("inf")), v)
    return out


def run_anchor_candidate_generation(config: dict[str, Any]) -> dict[str, Any]:
    run_cfg = config.get("run") or {}
    cand_root = config.get("candidates") or {}
    generators = cand_root.get("generators") or []
    eval_cfg = config.get("evaluation") or {}
    seed_cfg = config.get("seed") or {}
    out_cfg = config.get("output") or {}

    project_root = gh.find_project_root()
    graph_id = resolve_graph_id(run_cfg)
    anchor_output_root_raw = str(run_cfg.get("anchor_output_root") or "").strip()
    if anchor_output_root_raw:
        anchor_output_root = Path(anchor_output_root_raw).expanduser().resolve()
        anchor_run_dir = (anchor_output_root / graph_id).resolve()
    else:
        anchor_run_dir = (
            project_root / "seed_candidate_workflow" / "output" / "graph_bundles" / graph_id / "anchor" / graph_id
        ).resolve()
    if not anchor_run_dir.is_dir():
        raise FileNotFoundError(f"Anchor graph run directory not found: {anchor_run_dir}")

    nodes_df, edges_df, _cand, _summary, _g = load_anchor_graph_artifacts(anchor_run_dir, load_graph_pickle=False)
    nodes_df["external_id"] = nodes_df["external_id"].astype(str)
    edges_df["email_a"] = edges_df["email_a"].astype(str)
    edges_df["email_b"] = edges_df["email_b"].astype(str)

    seed_output_root_raw = str(run_cfg.get("seed_output_root") or "").strip()
    seed_output_root = (
        Path(seed_output_root_raw).expanduser().resolve()
        if seed_output_root_raw
        else (project_root / "seed_candidate_workflow" / "output" / "graph_bundles" / graph_id / "seed").resolve()
    )
    seed_stage_prefix = str(seed_cfg.get("seed_stage_name_prefix") or "seed_generation_")
    seed_stage_dir_override = str(run_cfg.get("seed_stage_dir") or "").strip()
    if seed_stage_dir_override:
        seed_dir = Path(seed_stage_dir_override).expanduser().resolve()
        if not seed_dir.is_dir():
            raise FileNotFoundError(f"run.seed_stage_dir is not a directory: {seed_dir}")
    else:
        seed_dir = _resolve_latest_seed_dir(
            seed_output_root=seed_output_root,
            graph_id=graph_id,
            seed_stage_name_prefix=seed_stage_prefix,
        )
    seed_edges_all_csv = seed_dir / "seed_edges_all.csv"
    seed_pairs = _load_seed_pairs(seed_edges_all_csv)
    out_root_raw = str(out_cfg.get("output_root") or "").strip()
    out_root = (
        Path(out_root_raw).expanduser().resolve()
        if out_root_raw
        else (project_root / "seed_candidate_workflow" / "output" / "graph_bundles" / graph_id / "candidate").resolve()
    )
    stage_name = str(out_cfg.get("stage_name") or "candidate_generation")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = (out_root / graph_id / f"{stage_name}_{stamp}").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_probe = out_dir / ".write_probe"
    _write_probe.write_text("", encoding="utf-8")
    _write_probe.unlink(missing_ok=True)

    if not isinstance(generators, list) or not generators:
        raise ValueError("candidates.generators must be a non-empty list")

    p_anchor_run_cfg = anchor_run_dir / "anchor_graph_run_config.json"
    if not p_anchor_run_cfg.is_file():
        raise FileNotFoundError(f"Missing anchor_graph_run_config.json: {p_anchor_run_cfg}")
    run_meta = json.loads(p_anchor_run_cfg.read_text(encoding="utf-8"))
    build_cfg = run_meta.get("config") or {}
    inputs = build_cfg.get("inputs") or {}

    embedding_source = str(inputs.get("embedding_source") or "cache_or_compute")
    prefer_translated_for_compute = bool(inputs.get("prefer_translated_for_compute", True))
    tfidf_max_features = int(inputs.get("tfidf_max_features", 4096))
    embeddings_json_raw = inputs.get("embeddings_json")
    embeddings_json = None
    if embeddings_json_raw not in (None, ""):
        embeddings_json = Path(str(embeddings_json_raw)).expanduser()
        if not embeddings_json.is_absolute():
            embeddings_json = (project_root / embeddings_json).resolve()

    id_to_vec, emb_meta = load_embedding_vectors(
        nodes_df=nodes_df,
        embeddings_json=embeddings_json,
        embedding_source=embedding_source,
        prefer_translated_for_compute=prefer_translated_for_compute,
        tfidf_max_features=tfidf_max_features,
    )

    _body_gen_names = {"body_token_jaccard_highconf_v1", "body_char4gram_jaccard_highconf_v1"}
    text_catalog_shared: dict[str, dict[str, str]] = {}
    text_catalog_meta_shared: dict[str, Any] = {}
    if any(
        bool(g.get("enabled", True)) and str(g.get("name") or "").strip().lower() in _body_gen_names
        for g in generators
        if isinstance(g, dict)
    ):
        text_catalog_shared, text_catalog_meta_shared = load_misp_text_catalog_for_pairs(
            project_root=project_root
        )

    ts_map: dict[str, float] = {}
    if "ts" in nodes_df.columns:
        for eid, ts in zip(nodes_df["external_id"].astype(str).tolist(), nodes_df["ts"].tolist(), strict=False):
            v = pd.to_numeric(ts, errors="coerce")
            ts_map[eid] = float(v) if pd.notna(v) else float("nan")

    per_gen_outputs: list[dict[str, Any]] = []
    generator_cfg_map: dict[str, dict[str, Any]] = {}
    union_pairs: set[tuple[str, str]] = set()
    pair_sources: dict[str, set[tuple[str, str]]] = {
        "seed": set(seed_pairs),
        "rare_artifact": set(),
        "shared_stem_highconf": set(),
        "semantic_mid_sender": set(),
        "semantic_mid_core": set(),
        "semantic_mid_stem": set(),
        "semantic_mid_senderlocalpart": set(),
        "body_token_jaccard_highconf": set(),
        "body_char4gram_jaccard_highconf": set(),
        "semantic": set(),
        "component": set(),
        "twohop": set(),
    }
    rare_rarity_max: dict[tuple[str, str], float] = {}
    semantic_cosine_max: dict[tuple[str, str], float] = {}
    component_cosine_max: dict[tuple[str, str], float] = {}
    twohop_rarity_max: dict[tuple[str, str], float] = {}
    pair_time_gap_min: dict[tuple[str, str], float] = {}

    deferred_body_gens: list[tuple[str, dict[str, Any]]] = []

    pbar_total = 5 + int(len(generators))
    pbar = tqdm(total=pbar_total, desc=f"Anchor candidate generation [{graph_id}]") if tqdm is not None else None
    if pbar is not None:
        pbar.update(1)                                
        pbar.update(1)                                  
        pbar.update(1)                     
    for g in generators:
        if not isinstance(g, dict):
            if pbar is not None:
                pbar.update(1)
            continue
        name = str(g.get("name") or "").strip().lower()
        enabled = bool(g.get("enabled", True))
        cfg = g.get("config") or {}
        if name:
            generator_cfg_map[name] = {"enabled": enabled, "config": cfg}
        if not enabled or not name:
            if pbar is not None:
                pbar.update(1)
            continue

        if name in BODY_GENERATOR_NAMES:
            deferred_body_gens.append((name, cfg))
            if pbar is not None:
                pbar.update(1)
            continue

        if name == "seed_backbone_v1":
            rows = [{"email_i": a, "email_j": b, "source": "seed_backbone"} for a, b in sorted(seed_pairs)]
            df = pd.DataFrame(rows)
            p = out_dir / "candidates_seed_backbone.csv"
            _write_candidate_csv(df, p)
            pairs = _pairs_from_df(df)
            union_pairs |= pairs
            per_gen_outputs.append({"name": name, "enabled": True, "csv": str(p), "n_rows": int(len(df)), "n_pairs": int(len(pairs))})
            if pbar is not None:
                pbar.update(1)
            continue

        if name == "rare_artifact_v1":
            df, diag = generate_candidates_rare_artifact_v1(
                nodes_df=nodes_df,
                edges_df=edges_df,
                seed_pairs=None,
                generator_cfg=cfg,
            )
            p = out_dir / "candidates_rare_artifact.csv"
            _write_candidate_csv(df, p)
            pairs = _pairs_from_df(df)
            union_pairs |= pairs
            pair_sources["rare_artifact"] |= pairs
            rare_rarity_max.update(_aggregate_pair_metric_max(df, "rarity_score"))
            tmin = _aggregate_pair_metric_min(df, "time_gap_seconds")
            for k, v in tmin.items():
                pair_time_gap_min[k] = min(pair_time_gap_min.get(k, float("inf")), v)
            per_gen_outputs.append({"name": name, "enabled": True, "csv": str(p), "n_rows": int(len(df)), "n_pairs": int(len(pairs)), "diagnostics": diag})
            if pbar is not None:
                pbar.update(1)
            continue

        if name == "shared_stem_highconf_v1":
            df, diag = generate_candidates_shared_stem_highconf_v1(
                nodes_df=nodes_df,
                edges_df=edges_df,
                seed_pairs=None,
                generator_cfg=cfg,
            )
            p = out_dir / "candidates_shared_stem_highconf.csv"
            _write_candidate_csv(df, p)
            pairs = _pairs_from_df(df)
            union_pairs |= pairs
            pair_sources["shared_stem_highconf"] |= pairs
            rare_rarity_max.update(_aggregate_pair_metric_max(df, "rarity_score"))
            tmin = _aggregate_pair_metric_min(df, "time_gap_seconds")
            for k, v in tmin.items():
                pair_time_gap_min[k] = min(pair_time_gap_min.get(k, float("inf")), v)
            per_gen_outputs.append(
                {
                    "name": name,
                    "enabled": True,
                    "csv": str(p),
                    "n_rows": int(len(df)),
                    "n_pairs": int(len(pairs)),
                    "diagnostics": diag,
                }
            )
            if pbar is not None:
                pbar.update(1)
            continue

        if name == "semantic_mid_sender_support_v1":
            df, diag = generate_semantic_mid_sender_support_v1(
                nodes_df=nodes_df,
                id_to_vec=id_to_vec,
                generator_cfg=cfg,
            )
            p = out_dir / "candidates_mid_sender.csv"
            _write_candidate_csv(df, p)
            pairs = _pairs_from_df(df)
            union_pairs |= pairs
            pair_sources["semantic_mid_sender"] |= pairs
            semantic_cosine_max.update(_aggregate_pair_metric_max(df, "cosine"))
            per_gen_outputs.append(
                {
                    "name": name,
                    "enabled": True,
                    "csv": str(p),
                    "n_rows": int(len(df)),
                    "n_pairs": int(len(pairs)),
                    "diagnostics": diag,
                }
            )
            if pbar is not None:
                pbar.update(1)
            continue

        if name == "semantic_mid_core_support_v1":
            df, diag = generate_semantic_mid_core_support_v1(
                nodes_df=nodes_df,
                id_to_vec=id_to_vec,
                generator_cfg=cfg,
            )
            p = out_dir / "candidates_mid_core.csv"
            _write_candidate_csv(df, p)
            pairs = _pairs_from_df(df)
            union_pairs |= pairs
            pair_sources["semantic_mid_core"] |= pairs
            semantic_cosine_max.update(_aggregate_pair_metric_max(df, "cosine"))
            per_gen_outputs.append(
                {
                    "name": name,
                    "enabled": True,
                    "csv": str(p),
                    "n_rows": int(len(df)),
                    "n_pairs": int(len(pairs)),
                    "diagnostics": diag,
                }
            )
            if pbar is not None:
                pbar.update(1)
            continue

        if name == "semantic_mid_stem_support_v1":
            df, diag = generate_semantic_mid_stem_support_v1(
                nodes_df=nodes_df,
                id_to_vec=id_to_vec,
                generator_cfg=cfg,
            )
            p = out_dir / "candidates_mid_stem.csv"
            _write_candidate_csv(df, p)
            pairs = _pairs_from_df(df)
            union_pairs |= pairs
            pair_sources["semantic_mid_stem"] |= pairs
            semantic_cosine_max.update(_aggregate_pair_metric_max(df, "cosine"))
            per_gen_outputs.append(
                {
                    "name": name,
                    "enabled": True,
                    "csv": str(p),
                    "n_rows": int(len(df)),
                    "n_pairs": int(len(pairs)),
                    "diagnostics": diag,
                }
            )
            if pbar is not None:
                pbar.update(1)
            continue

        if name == "semantic_mid_senderlocalpart_support_v1":
            df, diag = generate_semantic_mid_senderlocalpart_support_v1(
                nodes_df=nodes_df,
                id_to_vec=id_to_vec,
                generator_cfg=cfg,
            )
            p = out_dir / "candidates_semantic_mid_senderlocalpart_support.csv"
            _write_candidate_csv(df, p)
            pairs = _pairs_from_df(df)
            union_pairs |= pairs
            pair_sources["semantic_mid_senderlocalpart"] |= pairs
            semantic_cosine_max.update(_aggregate_pair_metric_max(df, "cosine"))
            per_gen_outputs.append(
                {
                    "name": name,
                    "enabled": True,
                    "csv": str(p),
                    "n_rows": int(len(df)),
                    "n_pairs": int(len(pairs)),
                    "diagnostics": diag,
                }
            )
            if pbar is not None:
                pbar.update(1)
            continue

        if name == "semantic_reciprocal_v1":
            semantic_top_k = int(cfg.get("semantic_top_k", 50))
            semantic_min_cos = float(cfg.get("semantic_min_cos", 0.9))
            time_gating_enabled = bool(cfg.get("time_gating_enabled", False))
            max_time_gap_seconds = cfg.get("max_time_gap_seconds")
            if time_gating_enabled and max_time_gap_seconds is None:
                max_time_gap_seconds = 86400.0

            mutual_df, _neighbor_info = _compute_mutual_topk_cosine_candidates(
                node_ids=nodes_df["external_id"].astype(str).tolist(),
                id_to_vec=id_to_vec,
                semantic_top_k=semantic_top_k,
                semantic_min_cos=semantic_min_cos,
            )
            if not mutual_df.empty:
                mutual_df["time_gap_seconds"] = [
                    _pair_time_gap_seconds(ts_map, a, b)
                    for a, b in zip(
                        mutual_df["email_i"].astype(str).tolist(),
                        mutual_df["email_j"].astype(str).tolist(),
                        strict=False,
                    )
                ]
                if time_gating_enabled:
                    mutual_df = mutual_df[
                        pd.to_numeric(mutual_df["time_gap_seconds"], errors="coerce").fillna(float("inf"))
                        <= float(max_time_gap_seconds)
                    ].copy()
                mutual_df = mutual_df.drop_duplicates(subset=["email_i", "email_j", "source"], keep="first").reset_index(drop=True)

            p = out_dir / "candidates_semantic.csv"
            mutual_df.to_csv(p, index=False)
            pairs = _pairs_from_df(mutual_df)
            union_pairs |= pairs
            pair_sources["semantic"] |= pairs
            semantic_cosine_max.update(_aggregate_pair_metric_max(mutual_df, "cosine"))
            tmin = _aggregate_pair_metric_min(mutual_df, "time_gap_seconds")
            for k, v in tmin.items():
                pair_time_gap_min[k] = min(pair_time_gap_min.get(k, float("inf")), v)
            per_gen_outputs.append(
                {
                    "name": name,
                    "enabled": True,
                    "csv": str(p),
                    "n_rows": int(len(mutual_df)),
                    "n_pairs": int(len(pairs)),
                    "semantic_top_k": semantic_top_k,
                    "semantic_min_cos": semantic_min_cos,
                    "time_gating_enabled": time_gating_enabled,
                    "max_time_gap_seconds": max_time_gap_seconds,
                }
            )
            if pbar is not None:
                pbar.update(1)
            continue

        if name == "component_expansion_v1":
            result = generate_component_expansion_v1(
                nodes_df=nodes_df,
                id_to_vec=id_to_vec,
                seed_dir=seed_dir,
                generator_cfg=cfg,
                out_dir=out_dir,
            )
            pairs = set(result.get("pairs_set") or set())
            union_pairs |= pairs
            pair_sources["component"] |= pairs
            comp_df_path = result.get("candidates_component_expanded_csv")
            if comp_df_path:
                comp_df = pd.read_csv(comp_df_path)
                component_cosine_max.update(_aggregate_pair_metric_max(comp_df, "cosine"))
                tmin = _aggregate_pair_metric_min(comp_df, "time_gap_seconds")
                for k, v in tmin.items():
                    pair_time_gap_min[k] = min(pair_time_gap_min.get(k, float("inf")), v)
            per_gen_outputs.append(
                {
                    "name": name,
                    "enabled": True,
                    "component_links_csv": result.get("component_links_csv"),
                    "candidates_component_expanded_csv": result.get("candidates_component_expanded_csv"),
                    "n_component_links": int(result.get("n_component_links", 0)),
                    "n_rows": int(result.get("n_candidate_rows", 0)),
                    "n_pairs": int(result.get("n_candidate_pairs", 0)),
                }
            )
            if pbar is not None:
                pbar.update(1)
            continue

        if name == "2hop_bounded_v1":
            result = generate_candidates_2hop_bounded_v1(
                nodes_df=nodes_df,
                id_to_vec=id_to_vec,
                ts_map=ts_map,
                seed_pairs=seed_pairs,
                seed_dir=seed_dir,
                generator_cfg=cfg,
                out_dir=out_dir,
            )
            pairs = set(result.get("pairs_set") or set())
            union_pairs |= pairs
            pair_sources["twohop"] |= pairs
            twohop_df_path = result.get("candidates_2hop_csv")
            if twohop_df_path:
                twohop_df = pd.read_csv(twohop_df_path)
                twohop_rarity_max.update(_aggregate_pair_metric_max(twohop_df, "rarity_score"))
                tmin = _aggregate_pair_metric_min(twohop_df, "time_gap_seconds")
                for k, v in tmin.items():
                    pair_time_gap_min[k] = min(pair_time_gap_min.get(k, float("inf")), v)
            per_gen_outputs.append(
                {
                    "name": name,
                    "enabled": True,
                    "csv": result.get("candidates_2hop_csv"),
                    "n_rows": int(result.get("n_rows", 0)),
                    "n_pairs": int(result.get("n_pairs", 0)),
                    "diagnostics": result.get("diagnostics"),
                }
            )
            if pbar is not None:
                pbar.update(1)
            continue

        raise ValueError(f"Unknown candidate generator: {name!r}")

    body_generation_summary: dict[str, Any] = {}
    if deferred_body_gens:
        prior_pair_pool = set(union_pairs)
        first_body_cfg = deferred_body_gens[0][1]
        body_progress = progress_from_cfg(first_body_cfg, graph_id=graph_id)
        body_progress.message(
            f"Starting body Jaccard generators ({len(deferred_body_gens)} rules, "
            f"prior_pool={len(prior_pair_pool):,} pairs)"
        )
        semantic_band_pool = build_semantic_band_pool_for_body_generators(
            nodes_df=nodes_df,
            id_to_vec=id_to_vec,
            generator_cfg=first_body_cfg,
            progress=body_progress,
        )
        shared_store, cache_diag, _catalog = prepare_body_feature_store(
            nodes_df=nodes_df,
            generator_cfg=first_body_cfg,
            project_root=project_root,
            graph_id=graph_id,
            text_catalog=text_catalog_shared,
            progress=body_progress,
        )
        body_generation_summary = {
            "n_deferred_generators": int(len(deferred_body_gens)),
            "n_prior_pair_pool": int(len(prior_pair_pool)),
            "n_semantic_band_pool": int(len(semantic_band_pool)),
            "cache": cache_diag,
        }
        for gen_idx, (body_name, body_cfg) in enumerate(deferred_body_gens, start=1):
            body_progress.message(f"Generator {gen_idx}/{len(deferred_body_gens)}: {body_name}")
            if body_name == "body_token_jaccard_highconf_v1":
                df, diag = generate_body_token_jaccard_highconf_v1(
                    nodes_df=nodes_df,
                    generator_cfg=body_cfg,
                    project_root=project_root,
                    graph_id=graph_id,
                    text_catalog=text_catalog_shared,
                    prior_pair_pool=prior_pair_pool,
                    semantic_band_pool=semantic_band_pool,
                    body_feature_store=shared_store,
                    cache_diag_preload=cache_diag,
                    progress=body_progress,
                )
                p = out_dir / "candidates_body_token_jaccard_highconf.csv"
                pair_key = "body_token_jaccard_highconf"
            elif body_name == "body_char4gram_jaccard_highconf_v1":
                df, diag = generate_body_char4gram_jaccard_highconf_v1(
                    nodes_df=nodes_df,
                    generator_cfg=body_cfg,
                    project_root=project_root,
                    graph_id=graph_id,
                    text_catalog=text_catalog_shared,
                    prior_pair_pool=prior_pair_pool,
                    semantic_band_pool=semantic_band_pool,
                    body_feature_store=shared_store,
                    cache_diag_preload=cache_diag,
                    progress=body_progress,
                )
                p = out_dir / "candidates_body_char4gram_jaccard_highconf.csv"
                pair_key = "body_char4gram_jaccard_highconf"
            else:
                continue
            _write_candidate_csv(df, p)
            pairs = _pairs_from_df(df)
            union_pairs |= pairs
            pair_sources[pair_key] |= pairs
            per_gen_outputs.append(
                {
                    "name": body_name,
                    "enabled": True,
                    "csv": str(p),
                    "n_rows": int(len(df)),
                    "n_pairs": int(len(pairs)),
                    "diagnostics": diag,
                }
            )
        body_progress.message("Body Jaccard generators finished")

    missing_seed_pairs = seed_pairs - union_pairs
    if missing_seed_pairs:
        raise AssertionError(
            f"Candidate union is missing {len(missing_seed_pairs)} seed pairs. "
            f"Enable seed_backbone_v1 or adjust generators."
        )

    members_path = seed_dir / "seed_union_component_members.csv"
    comp_map: dict[str, int] = {}
    if members_path.is_file():
        mdf = pd.read_csv(members_path)
        if not mdf.empty and {"external_id", "component_id"}.issubset(mdf.columns):
            for eid, cid in zip(mdf["external_id"].astype(str).tolist(), mdf["component_id"].tolist(), strict=False):
                comp_map[str(eid)] = int(pd.to_numeric(cid, errors="coerce")) if pd.notna(pd.to_numeric(cid, errors="coerce")) else -1

    candidate_union_rows: list[dict[str, Any]] = []
    for email_i, email_j in sorted(union_pairs):
        from_seed = (email_i, email_j) in pair_sources["seed"]
        from_rare = (email_i, email_j) in pair_sources["rare_artifact"]
        from_stem_hi = (email_i, email_j) in pair_sources["shared_stem_highconf"]
        from_mid_sender = (email_i, email_j) in pair_sources["semantic_mid_sender"]
        from_mid_core = (email_i, email_j) in pair_sources["semantic_mid_core"]
        from_mid_stem = (email_i, email_j) in pair_sources["semantic_mid_stem"]
        from_mid_sender_lp = (email_i, email_j) in pair_sources["semantic_mid_senderlocalpart"]
        from_body_tok = (email_i, email_j) in pair_sources["body_token_jaccard_highconf"]
        from_body_c4 = (email_i, email_j) in pair_sources["body_char4gram_jaccard_highconf"]
        from_sem = (email_i, email_j) in pair_sources["semantic"]
        from_comp = (email_i, email_j) in pair_sources["component"]
        from_2hop = (email_i, email_j) in pair_sources["twohop"]
        cid_i = int(comp_map.get(email_i, -1))
        cid_j = int(comp_map.get(email_j, -1))
        same_seed_component = bool(cid_i >= 0 and cid_j >= 0 and cid_i == cid_j)
        time_gap = pair_time_gap_min.get((email_i, email_j), float("nan"))
        candidate_union_rows.append(
            {
                "email_i": email_i,
                "email_j": email_j,
                "from_seed": bool(from_seed),
                "from_rare_artifact": bool(from_rare),
                "from_shared_stem_highconf": bool(from_stem_hi),
                "from_semantic_mid_sender_support": bool(from_mid_sender),
                "from_semantic_mid_core_support": bool(from_mid_core),
                "from_semantic_mid_stem_support": bool(from_mid_stem),
                "from_semantic_mid_senderlocalpart_support": bool(from_mid_sender_lp),
                "from_body_token_jaccard_highconf": bool(from_body_tok),
                "from_body_char4gram_jaccard_highconf": bool(from_body_c4),
                "from_semantic": bool(from_sem),
                "from_component": bool(from_comp),
                "from_2hop": bool(from_2hop),
                "source_count": int(
                    sum(
                        [
                            from_seed,
                            from_rare,
                            from_stem_hi,
                            from_mid_sender,
                            from_mid_core,
                            from_mid_stem,
                            from_mid_sender_lp,
                            from_body_tok,
                            from_body_c4,
                            from_sem,
                            from_comp,
                            from_2hop,
                        ]
                    )
                ),
                "rare_artifact_rarity_max": rare_rarity_max.get((email_i, email_j), float("nan")),
                "semantic_cosine_max": semantic_cosine_max.get((email_i, email_j), float("nan")),
                "component_cosine_max": component_cosine_max.get((email_i, email_j), float("nan")),
                "twohop_rarity_max": twohop_rarity_max.get((email_i, email_j), float("nan")),
                "time_gap_seconds_min": float(time_gap) if pd.notna(time_gap) else float("nan"),
                "email_i_seed_component_id": int(cid_i),
                "email_j_seed_component_id": int(cid_j),
                "both_in_seed_components": bool(cid_i >= 0 and cid_j >= 0),
                "same_seed_component": same_seed_component,
            }
        )
    candidate_union_df = pd.DataFrame(candidate_union_rows)
    p_candidate_union = out_dir / "candidate_union.csv"
    _write_candidate_csv(candidate_union_df, p_candidate_union)
    if pbar is not None:
        pbar.update(1)

    eval_outputs = None
    if bool(eval_cfg.get("enabled", True)):
        eval_outputs = run_candidate_evaluation_report(
            project_root=project_root,
            graph_id=graph_id,
            out_dir=out_dir,
            seed_dir=seed_dir,
            candidate_union_df=candidate_union_df,
            seed_pairs=seed_pairs,
            total_emails=int(len(nodes_df)),
            eval_cfg=eval_cfg,
            generator_configs=generator_cfg_map,
            generator_outputs=per_gen_outputs,
            full_candidate_generation_config=config,
        )
    if pbar is not None:
        pbar.update(1)

    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "graph_id": graph_id,
        "anchor_run_dir": str(anchor_run_dir),
        "seed_stage_dir": str(seed_dir),
        "generators_run": per_gen_outputs,
        "body_jaccard_generation": body_generation_summary or None,
        "union_invariant": {
            "n_seed_pairs": int(len(seed_pairs)),
            "n_union_pairs": int(len(union_pairs)),
            "n_missing_seed_pairs": int(len(missing_seed_pairs)),
        },
        "candidate_union": {
            "csv": str(p_candidate_union),
            "n_rows": int(len(candidate_union_df)),
            "n_from_seed": int(candidate_union_df["from_seed"].sum()) if not candidate_union_df.empty else 0,
            "n_from_rare_artifact": int(candidate_union_df["from_rare_artifact"].sum()) if not candidate_union_df.empty else 0,
            "n_from_semantic": int(candidate_union_df["from_semantic"].sum()) if not candidate_union_df.empty else 0,
            "n_from_semantic_mid_sender_support": int(
                candidate_union_df["from_semantic_mid_sender_support"].sum()
            )
            if not candidate_union_df.empty
            and "from_semantic_mid_sender_support" in candidate_union_df.columns
            else 0,
            "n_from_semantic_mid_core_support": int(
                candidate_union_df["from_semantic_mid_core_support"].sum()
            )
            if not candidate_union_df.empty
            and "from_semantic_mid_core_support" in candidate_union_df.columns
            else 0,
            "n_from_semantic_mid_stem_support": int(
                candidate_union_df["from_semantic_mid_stem_support"].sum()
            )
            if not candidate_union_df.empty
            and "from_semantic_mid_stem_support" in candidate_union_df.columns
            else 0,
            "n_from_component": int(candidate_union_df["from_component"].sum()) if not candidate_union_df.empty else 0,
            "n_from_2hop": int(candidate_union_df["from_2hop"].sum()) if not candidate_union_df.empty else 0,
            "n_from_semantic_mid_senderlocalpart_support": int(
                candidate_union_df["from_semantic_mid_senderlocalpart_support"].sum()
            )
            if not candidate_union_df.empty
            and "from_semantic_mid_senderlocalpart_support" in candidate_union_df.columns
            else 0,
            "n_from_body_token_jaccard_highconf": int(
                candidate_union_df["from_body_token_jaccard_highconf"].sum()
            )
            if not candidate_union_df.empty
            and "from_body_token_jaccard_highconf" in candidate_union_df.columns
            else 0,
            "n_from_body_char4gram_jaccard_highconf": int(
                candidate_union_df["from_body_char4gram_jaccard_highconf"].sum()
            )
            if not candidate_union_df.empty
            and "from_body_char4gram_jaccard_highconf" in candidate_union_df.columns
            else 0,
        },
        "candidate_eval": eval_outputs,
        "embedding_meta": emb_meta,
        "text_catalog_meta": text_catalog_meta_shared,
    }
    p_summary = out_dir / "anchor_candidates_summary.json"
    p_summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    if pbar is not None:
        pbar.update(1)
        pbar.close()

    return {
        "output_dir": str(out_dir),
        "candidate_union_csv": str(p_candidate_union),
        "candidate_eval_summary_json": None if not eval_outputs else eval_outputs.get("summary_json"),
        "candidate_source_ablation_csv": None if not eval_outputs else eval_outputs.get("ablation_csv"),
        "candidate_manual_review_csv": None if not eval_outputs else eval_outputs.get("manual_review_csv"),
        "candidate_oracle_ceiling_csv": None if not eval_outputs else eval_outputs.get("oracle_ceiling_csv"),
        "candidate_eval_notes_txt": None if not eval_outputs else eval_outputs.get("notes_txt"),
        "candidate_eval_readiness": None if not eval_outputs else eval_outputs.get("readiness"),
        "summary_json": str(p_summary),
    }

