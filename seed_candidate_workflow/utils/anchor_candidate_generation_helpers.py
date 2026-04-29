from __future__ import annotations

import json
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
    # Output dir
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

    if not isinstance(generators, list) or not generators:
        raise ValueError("candidates.generators must be a non-empty list")

    # Pull embedding config snapshot from anchor_graph_run_config.json if semantic generator is enabled.
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
        "semantic": set(),
        "component": set(),
        "twohop": set(),
    }
    rare_rarity_max: dict[tuple[str, str], float] = {}
    semantic_cosine_max: dict[tuple[str, str], float] = {}
    component_cosine_max: dict[tuple[str, str], float] = {}
    twohop_rarity_max: dict[tuple[str, str], float] = {}
    pair_time_gap_min: dict[tuple[str, str], float] = {}

    pbar_total = 5 + int(len(generators))
    pbar = tqdm(total=pbar_total, desc=f"Anchor candidate generation [{graph_id}]") if tqdm is not None else None
    if pbar is not None:
        pbar.update(1)  # loaded anchor + seed context
        pbar.update(1)  # output dir + config resolution
        pbar.update(1)  # embeddings loaded
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

        if name == "seed_backbone_v1":
            rows = [{"email_i": a, "email_j": b, "source": "seed_backbone"} for a, b in sorted(seed_pairs)]
            df = pd.DataFrame(rows)
            p = out_dir / "candidates_seed_backbone.csv"
            df.to_csv(p, index=False)
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
            df.to_csv(p, index=False)
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

    # Enforce required invariant on the union of enabled generator outputs.
    missing_seed_pairs = seed_pairs - union_pairs
    if missing_seed_pairs:
        raise AssertionError(
            f"Candidate union is missing {len(missing_seed_pairs)} seed pairs. "
            f"Enable seed_backbone_v1 or adjust generators."
        )

    # Build candidate union table.
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
                "from_semantic": bool(from_sem),
                "from_component": bool(from_comp),
                "from_2hop": bool(from_2hop),
                "source_count": int(sum([from_seed, from_rare, from_sem, from_comp, from_2hop])),
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
    candidate_union_df.to_csv(p_candidate_union, index=False)
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

    # Summary
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "graph_id": graph_id,
        "anchor_run_dir": str(anchor_run_dir),
        "seed_stage_dir": str(seed_dir),
        "generators_run": per_gen_outputs,
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
            "n_from_component": int(candidate_union_df["from_component"].sum()) if not candidate_union_df.empty else 0,
            "n_from_2hop": int(candidate_union_df["from_2hop"].sum()) if not candidate_union_df.empty else 0,
        },
        "candidate_eval": eval_outputs,
        "embedding_meta": emb_meta,
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

