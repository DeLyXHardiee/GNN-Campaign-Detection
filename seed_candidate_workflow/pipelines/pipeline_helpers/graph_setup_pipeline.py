from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - optional dependency
    tqdm = None

from seed_candidate_workflow.utils.anchor_candidate_generation_helpers import run_anchor_candidate_generation
from seed_candidate_workflow.utils.anchor_graph_helpers import build_anchor_graph
from seed_candidate_workflow.utils.anchor_seed_helpers import run_anchor_seed_generation
from seed_candidate_workflow.utils.pair_graph_contract import (
    GRAPH_KIND_SEED_CANDIDATE,
    GRAPH_KIND_SEMANTIC_SHARD,
    ensure_unscored_contract,
)
from seed_candidate_workflow.utils.pair_training_dataset_helpers import build_pair_training_dataset


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _resolve_path(project_root: Path, raw: str | Path) -> Path:
    p = Path(str(raw)).expanduser()
    if not p.is_absolute():
        p = (project_root / p).resolve()
    else:
        p = p.resolve()
    return p

def _default_graph_meta_json_from_pipeline(project_root: Path) -> Path | None:
    """
    When experiment config omits setup.paths.pair_training.graph_meta_json,
    use the hetero .meta.json alongside pipeline_config graph_pt_path_override /
    default_hetero_graph_pt_path so pair training email indices stay aligned with the GNN graph.
    """
    try:
        import sys

        core_root = project_root / "core"
        if str(core_root) not in sys.path:
            sys.path.insert(0, str(core_root))
        from config.pipeline_config import default_hetero_graph_pt_path

        pt = Path(default_hetero_graph_pt_path(project_root=project_root))
        meta = pt.with_suffix(".meta.json")
        return meta.resolve()
    except Exception:
        return None


def _reliable_negative_pool_from_pipeline(project_root: Path) -> dict[str, Any] | None:
    """Load ``pair_training.reliable_negative_pool`` from repo-root ``pipeline_config.json``."""
    p = (project_root / "pipeline_config.json").resolve()
    if not p.is_file():
        return None
    cfg = _read_json(p)
    pool = (cfg.get("pair_training") or {}).get("reliable_negative_pool")
    return dict(pool) if isinstance(pool, dict) else None


def _resolve_latest_stage_dir(stage_root: Path, stage_prefix: str) -> Path:
    if not stage_root.is_dir():
        raise FileNotFoundError(f"Stage root not found: {stage_root}")
    dirs = [p for p in stage_root.iterdir() if p.is_dir() and p.name.startswith(stage_prefix)]
    if not dirs:
        raise FileNotFoundError(f"No stage directories with prefix {stage_prefix!r} under {stage_root}")
    return max(dirs, key=lambda p: p.stat().st_mtime)


def _resolve_bundle_paths(*, graph_bundle_root: Path, graph_id: str) -> dict[str, Path]:
    bundle_root = (graph_bundle_root / graph_id).resolve()
    return {
        "bundle_root": bundle_root,
        "anchor_root": bundle_root / "anchor",
        "seed_root": bundle_root / "seed",
        "candidate_root": bundle_root / "candidate",
        "seed_candidate_root": bundle_root / "seed_candidate",
        "semantic_shard_root": bundle_root / "semantic_shard",
        "pair_training_root": bundle_root / "pair_training",
    }


def _validate_setup_cfg(setup_cfg: dict[str, Any]) -> None:
    policy_cfg = dict(setup_cfg.get("policy") or {})
    on_missing = str(policy_cfg.get("on_missing") or "build").strip().lower()
    on_present = str(policy_cfg.get("on_present") or "reuse").strip().lower()
    if on_missing not in {"build", "fail"}:
        raise ValueError(f"Unsupported setup.policy.on_missing={on_missing!r}")
    if on_present not in {"reuse", "rebuild"}:
        raise ValueError(f"Unsupported setup.policy.on_present={on_present!r}")
    paths_cfg = dict(setup_cfg.get("paths") or {})
    semantic_shard_cfg = dict(paths_cfg.get("semantic_shard") or {})
    if semantic_shard_cfg:
        required = ("embeddings_json", "graph_pt", "meta_json")
        missing = [k for k in required if not str(semantic_shard_cfg.get(k) or "").strip()]
        if missing:
            raise ValueError(f"setup.paths.semantic_shard missing required keys: {missing}")
        step1 = dict(semantic_shard_cfg.get("step1") or {})
        step2 = dict(semantic_shard_cfg.get("step2") or {})
        if step1:
            if int(step1.get("min_cluster_size", 2)) < 1:
                raise ValueError("setup.paths.semantic_shard.step1.min_cluster_size must be >= 1")
            ms = step1.get("min_samples")
            if ms not in (None, "") and int(ms) < 1:
                raise ValueError("setup.paths.semantic_shard.step1.min_samples must be >= 1 or null")
        if step2:
            if int(step2.get("semantic_top_k", 8)) < 1:
                raise ValueError("setup.paths.semantic_shard.step2.semantic_top_k must be >= 1")
            smc = float(step2.get("semantic_min_cos", 0.72))
            if smc < 0.0 or smc > 1.0:
                raise ValueError("setup.paths.semantic_shard.step2.semantic_min_cos must be in [0,1]")
            switches = step2.get("infra_channel_switches")
            if switches is not None and not isinstance(switches, dict):
                raise ValueError("setup.paths.semantic_shard.step2.infra_channel_switches must be an object")
            logical = step2.get("shard_edge_scoring_logical")
            if logical is not None and not isinstance(logical, dict):
                raise ValueError("setup.paths.semantic_shard.step2.shard_edge_scoring_logical must be an object")


def _build_seed_candidate_pairgraph(
    *,
    candidate_union_csv: Path,
    graph_id: str,
    out_dir: Path,
) -> dict[str, str]:
    union = pd.read_csv(candidate_union_csv, low_memory=False)
    if union.empty:
        pair_df = pd.DataFrame(
            {
                "email_i": pd.Series(dtype=str),
                "email_j": pd.Series(dtype=str),
                "graph_kind": pd.Series(dtype=str),
                "graph_id": pd.Series(dtype=str),
                "from_seed": pd.Series(dtype=bool),
                "from_semantic": pd.Series(dtype=bool),
                "from_rare_artifact": pd.Series(dtype=bool),
                "from_component": pd.Series(dtype=bool),
                "from_2hop": pd.Series(dtype=bool),
                "source_count": pd.Series(dtype=int),
            }
        )
    else:
        pair_df = union.copy()
        pair_df["graph_kind"] = GRAPH_KIND_SEED_CANDIDATE
        pair_df["graph_id"] = graph_id
        pair_df["from_seed"] = pair_df.get("from_seed", False)
        pair_df["from_semantic"] = pair_df.get("from_semantic", False)
        pair_df["from_rare_artifact"] = pair_df.get("from_rare_artifact", False)
        pair_df["from_component"] = pair_df.get("from_component", False)
        pair_df["from_2hop"] = pair_df.get("from_2hop", False)
    pair_df = ensure_unscored_contract(pair_df)

    out_dir.mkdir(parents=True, exist_ok=True)
    p_pair = out_dir / "seed_candidate_pairgraph_unscored.csv"
    p_summary = out_dir / "seed_candidate_graph_summary.json"
    pair_df.to_csv(p_pair, index=False)
    p_summary.write_text(
        json.dumps(
            {
                "graph_id": graph_id,
                "candidate_union_csv": str(candidate_union_csv),
                "pairgraph_unscored_csv": str(p_pair),
                "n_pairs": int(len(pair_df)),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return {
        "pairgraph_unscored_csv": str(p_pair),
        "summary_json": str(p_summary),
    }


def _build_semantic_shard_pairgraph(
    *,
    project_root: Path,
    graph_id: str,
    out_dir: Path,
    stage_cfg: dict[str, Any],
) -> dict[str, str]:
    from seed_candidate_workflow.utils import semantic_shard_graph_helpers as s2
    from seed_candidate_workflow.utils import semantic_shard_helpers as ssh
    from seed_candidate_workflow.utils import graph_structure_helpers as gh

    required = ("embeddings_json", "graph_pt", "meta_json")
    missing = [k for k in required if not str(stage_cfg.get(k) or "").strip()]
    if missing:
        raise ValueError(f"setup.paths.semantic_shard missing required keys: {missing}")

    embeddings_json = _resolve_path(project_root, stage_cfg["embeddings_json"])
    graph_pt = _resolve_path(project_root, stage_cfg["graph_pt"])
    meta_json = _resolve_path(project_root, stage_cfg["meta_json"])
    popular_domains_path_raw = str(stage_cfg.get("popular_domains_path") or "").strip()
    popular_domains_path = _resolve_path(project_root, popular_domains_path_raw) if popular_domains_path_raw else None
    to_undirected = bool(stage_cfg.get("to_undirected", True))

    step1_cfg = dict(stage_cfg.get("step1") or {})
    min_cluster_size = int(step1_cfg.get("min_cluster_size", 2))
    min_samples = step1_cfg.get("min_samples")
    min_samples = None if min_samples in (None, "") else int(min_samples)
    fallback_cosine_distance_threshold = float(step1_cfg.get("fallback_cosine_distance_threshold", 0.22))
    noise_as_singleton = bool(step1_cfg.get("noise_as_singleton", True))

    step2_cfg = dict(stage_cfg.get("step2") or {})
    semantic_top_k = int(step2_cfg.get("semantic_top_k", 8))
    semantic_min_cos = float(step2_cfg.get("semantic_min_cos", 0.72))
    show_progress = bool(step2_cfg.get("show_progress", False))
    include_routing_channels_in_graph = bool(step2_cfg.get("include_routing_channels_in_graph", False))
    candidate_infra_channels = tuple(step2_cfg.get("candidate_infra_channels") or ())
    scoring_infra_channels = tuple(step2_cfg.get("scoring_infra_channels") or ())
    semantic_weight = float(step2_cfg.get("semantic_weight", 0.45))
    infra_weight = float(step2_cfg.get("infra_weight", 0.45))
    temporal_weight = float(step2_cfg.get("temporal_weight", 0.10))
    channel_weights = dict(step2_cfg.get("channel_weights") or {})
    shard_edge_scoring_logical = dict(step2_cfg.get("shard_edge_scoring_logical") or {})
    infra_channel_switches = dict(step2_cfg.get("infra_channel_switches") or {})

    pbar = tqdm(total=13, desc=f"Semantic shard setup [{graph_id}]") if tqdm is not None else None
    try:
        _payload, id_to_semantic, _summary = ssh.load_transformer_cache(embeddings_json)
        if pbar is not None:
            pbar.update(1)
        # Notebook parity: shard clustering should run on embeddings intersected with graph email IDs.
        graph_external_ids = set(gh.email_external_id_list(gh.load_meta(meta_json)))
        id_to_semantic = {eid: vec for eid, vec in id_to_semantic.items() if eid in graph_external_ids}
        if not id_to_semantic:
            raise ValueError("No aligned embeddings after intersecting with graph external IDs")
        if pbar is not None:
            pbar.update(1)
        clustered_df = ssh.cluster_semantic_shards_hdbscan(
            id_to_vec=id_to_semantic,
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            fallback_cosine_distance_threshold=fallback_cosine_distance_threshold,
        )
        if pbar is not None:
            pbar.update(1)
        assignments_df = ssh.build_shard_assignments(clustered_df, noise_as_singleton=noise_as_singleton)
        if pbar is not None:
            pbar.update(1)
        shard_summary_df, overall_df = ssh.shard_quality_tables(assignments_df, id_to_semantic, gt_label_map={})
        if pbar is not None:
            pbar.update(1)

        step1_out_dir = out_dir / "step1"
        step2_out_dir = out_dir / "step2"
        step1_saved = ssh.save_shard_step1_artifacts(
            output_dir=step1_out_dir,
            assignments_df=assignments_df,
            shard_summary_df=shard_summary_df,
            overall_df=overall_df,
        )
        if pbar is not None:
            pbar.update(1)

        email_df, _benign_diag = s2.load_email_level_inputs(
            graph_pt=graph_pt,
            meta_json=meta_json,
            to_undirected=to_undirected,
            popular_domains_path=popular_domains_path,
        )
        if pbar is not None:
            pbar.update(1)
        logical_to_col = {
            "url": "url_set",
            "sender_email_domain": "sender_email_domain_set",
            "domain": "domain_set",
            "stem": "stem_set",
            "sender": "sender_set",
            "attachment": "attachment_set",
            "html_structure_fingerprint": "html_structure_fingerprint_set",
            "origin_ip": "origin_ip_set",
            "helo_host": "helo_host_set",
            "return_path_email": "return_path_email_set",
            "return_path_domain": "return_path_domain_set",
        }
        default_switches: dict[str, dict[str, bool]] = {
            "url": {"enabled": True, "routing": False, "candidate": True, "scoring": True},
            "sender_email_domain": {"enabled": True, "routing": False, "candidate": True, "scoring": True},
            "domain": {"enabled": True, "routing": False, "candidate": True, "scoring": True},
            "stem": {"enabled": True, "routing": False, "candidate": True, "scoring": True},
            "sender": {"enabled": True, "routing": False, "candidate": True, "scoring": True},
            "attachment": {"enabled": True, "routing": False, "candidate": True, "scoring": True},
            "html_structure_fingerprint": {"enabled": False, "routing": False, "candidate": True, "scoring": True},
            "origin_ip": {"enabled": False, "routing": True, "candidate": True, "scoring": True},
            "helo_host": {"enabled": False, "routing": True, "candidate": True, "scoring": True},
            "return_path_email": {"enabled": False, "routing": True, "candidate": True, "scoring": True},
            "return_path_domain": {"enabled": False, "routing": True, "candidate": True, "scoring": True},
        }
        for k, v in infra_channel_switches.items():
            if k not in default_switches or not isinstance(v, dict):
                continue
            m = dict(default_switches[k])
            m.update({kk: bool(vv) for kk, vv in v.items() if kk in {"enabled", "routing", "candidate", "scoring"}})
            default_switches[k] = m
        enabled_logical: list[str] = []
        for logical, cfg in default_switches.items():
            if not cfg.get("enabled", False):
                continue
            if cfg.get("routing", False) and not include_routing_channels_in_graph:
                continue
            enabled_logical.append(logical)
        available_cols = set(email_df.columns.tolist())
        enabled_cols = [logical_to_col[k] for k in enabled_logical if logical_to_col.get(k) in available_cols]
        candidate_cols = (
            [c for c in candidate_infra_channels if c in available_cols]
            if candidate_infra_channels
            else [logical_to_col[k] for k in enabled_logical if default_switches.get(k, {}).get("candidate", True) and logical_to_col.get(k) in available_cols]
        )
        scoring_cols = (
            [c for c in scoring_infra_channels if c in available_cols]
            if scoring_infra_channels
            else [logical_to_col[k] for k in enabled_logical if default_switches.get(k, {}).get("scoring", True) and logical_to_col.get(k) in available_cols]
        )
        if not shard_edge_scoring_logical:
            shard_edge_scoring_logical = {
                "url": {"enabled": True, "weight": 1.00, "scoring_mode": "legacy"},
                "sender_email_domain": {"enabled": True, "weight": 0.85, "scoring_mode": "legacy"},
                "domain": {"enabled": True, "weight": 0.60, "scoring_mode": "legacy"},
                "stem": {"enabled": True, "weight": 0.55, "scoring_mode": "legacy"},
                "sender": {"enabled": True, "weight": 0.50, "scoring_mode": "legacy"},
                "attachment": {"enabled": True, "weight": 0.50, "scoring_mode": "legacy"},
                "html_structure_fingerprint": {"enabled": True, "weight": 0.45, "scoring_mode": "legacy"},
                "origin_ip": {
                    "enabled": True,
                    "weight": 0.35,
                    "scoring_mode": "routed",
                    "idf_exponent": 1.35,
                    "idf_scale": 1.10,
                    "max_shard_df": 100,
                    "contribution_cap": 0.09,
                },
                "helo_host": {
                    "enabled": True,
                    "weight": 0.30,
                    "scoring_mode": "routed",
                    "idf_exponent": 1.25,
                    "idf_scale": 1.00,
                    "max_shard_df": 80,
                    "contribution_cap": 0.07,
                },
                "return_path_email": {
                    "enabled": True,
                    "weight": 0.40,
                    "scoring_mode": "routed",
                    "idf_exponent": 1.20,
                    "idf_scale": 1.00,
                    "max_shard_df": 150,
                    "contribution_cap": 0.14,
                },
                "return_path_domain": {
                    "enabled": True,
                    "weight": 0.42,
                    "scoring_mode": "routed",
                    "idf_exponent": 1.15,
                    "idf_scale": 1.00,
                    "max_shard_df": 180,
                    "contribution_cap": 0.16,
                },
            }
        score_specs = s2.resolve_shard_edge_channel_scoring(
            scoring_channels_logical=enabled_logical,
            scoring_spec_by_logical=shard_edge_scoring_logical,
            logical_to_col=lambda x: logical_to_col.get(str(x), ""),
            available_infra_cols=available_cols,
            default_legacy_weight=0.55,
        )
        shard_nodes_df, shard_centroids = s2.build_shard_nodes(
            assignments_df=assignments_df,
            id_to_semantic=id_to_semantic,
            email_df=email_df,
            gt_label_map={},
            infra_channels=tuple(sorted(set(enabled_cols) | set(candidate_cols) | set(scoring_cols))),
        )
        if pbar is not None:
            pbar.update(1)
        cand_kwargs: dict[str, Any] = {
            "semantic_top_k": semantic_top_k,
            "semantic_min_cos": semantic_min_cos,
            "show_progress": show_progress,
        }
        if candidate_cols:
            cand_kwargs["candidate_infra_channels"] = tuple(candidate_cols)
        candidate_df = s2.build_candidate_edges(shard_nodes_df, shard_centroids, **cand_kwargs)
        if pbar is not None:
            pbar.update(1)
        edge_kwargs: dict[str, Any] = {
            "semantic_weight": semantic_weight,
            "infra_weight": infra_weight,
            "temporal_weight": temporal_weight,
        }
        if scoring_cols:
            edge_kwargs["scoring_infra_channels"] = tuple(scoring_cols)
        if score_specs:
            edge_kwargs["channel_scoring"] = score_specs
        if channel_weights:
            edge_kwargs["channel_weights"] = channel_weights
        edges_df = s2.build_weighted_edges(
            shard_nodes_df=shard_nodes_df,
            centroid_mat=shard_centroids,
            candidate_df=candidate_df,
            **edge_kwargs,
        )
        if pbar is not None:
            pbar.update(1)
        step2_saved = s2.save_step2_graph_artifacts(
            output_dir=step2_out_dir,
            shard_nodes_df=shard_nodes_df,
            centroid_mat=shard_centroids,
            edges_df=edges_df,
            candidate_df=candidate_df,
        )
        if pbar is not None:
            pbar.update(1)

        out_dir.mkdir(parents=True, exist_ok=True)
        p_edges_weighted = out_dir / "semantic_shard_edges_weighted.csv"
        p_assign = out_dir / "semantic_shard_assignments.csv"
        p_nodes = out_dir / "semantic_shard_nodes.csv"
        p_candidates = out_dir / "semantic_shard_candidates.csv"
        p_pair = out_dir / "semantic_shard_pairgraph_unscored.csv"
        p_summary = out_dir / "semantic_shard_graph_summary.json"

        edges_df.to_csv(p_edges_weighted, index=False)
        assignments_df.to_csv(p_assign, index=False)
        shard_nodes_df.to_csv(p_nodes, index=False)
        candidate_df.to_csv(p_candidates, index=False)
        if pbar is not None:
            pbar.update(1)
    finally:
        if pbar is not None:
            pbar.close()

    if edges_df.empty:
        pair_df = pd.DataFrame(
            {
                "email_i": pd.Series(dtype=str),
                "email_j": pd.Series(dtype=str),
                "graph_kind": pd.Series(dtype=str),
                "graph_id": pd.Series(dtype=str),
                "from_seed": pd.Series(dtype=bool),
                "from_semantic": pd.Series(dtype=bool),
                "from_rare_artifact": pd.Series(dtype=bool),
                "from_component": pd.Series(dtype=bool),
                "from_2hop": pd.Series(dtype=bool),
                "source_count": pd.Series(dtype=int),
            }
        )
    else:
        pair_df = edges_df.copy()
        pair_df["email_i"] = pair_df["shard_a"].astype(str)
        pair_df["email_j"] = pair_df["shard_b"].astype(str)
        pair_df["graph_kind"] = GRAPH_KIND_SEMANTIC_SHARD
        pair_df["graph_id"] = graph_id
        pair_df["from_seed"] = False
        pair_df["from_semantic"] = False
        pair_df["from_rare_artifact"] = False
        pair_df["from_component"] = False
        pair_df["from_2hop"] = False
        pair_df["source_count"] = 0
    pair_df = ensure_unscored_contract(pair_df)
    pair_df.to_csv(p_pair, index=False)
    p_summary.write_text(
        json.dumps(
            {
                "graph_id": graph_id,
                "pairgraph_unscored_csv": str(p_pair),
                "weighted_edges_csv": str(p_edges_weighted),
                "assignments_csv": str(p_assign),
                "nodes_csv": str(p_nodes),
                "candidates_csv": str(p_candidates),
                "n_pairs": int(len(pair_df)),
                "step1_artifacts": step1_saved,
                "step2_artifacts": step2_saved,
                "enabled_infra_channels_cols": enabled_cols,
                "candidate_infra_channels_cols": candidate_cols,
                "scoring_infra_channels_cols": scoring_cols,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return {
        "pairgraph_unscored_csv": str(p_pair),
        "weighted_edges_csv": str(p_edges_weighted),
        "assignments_csv": str(p_assign),
        "nodes_csv": str(p_nodes),
        "candidates_csv": str(p_candidates),
        "summary_json": str(p_summary),
        "step1_dir": str(step1_out_dir),
        "step2_dir": str(step2_out_dir),
    }


def run_graph_setup(
    *,
    project_root: Path,
    graph_id: str,
    graph_bundle_root: Path,
    setup_cfg: dict[str, Any],
) -> dict[str, Any]:
    _validate_setup_cfg(setup_cfg)
    enable_cfg = dict(setup_cfg.get("enable") or {})
    policy_cfg = dict(setup_cfg.get("policy") or {})
    paths_cfg = dict(setup_cfg.get("paths") or {})
    pair_training_cfg = dict(paths_cfg.get("pair_training") or {})
    semantic_shard_cfg = dict(paths_cfg.get("semantic_shard") or {})

    on_missing = str(policy_cfg.get("on_missing") or "build").strip().lower()
    on_present = str(policy_cfg.get("on_present") or "reuse").strip().lower()
    if on_missing not in {"build", "fail"}:
        raise ValueError(f"Unsupported setup.policy.on_missing={on_missing!r}")
    if on_present not in {"reuse", "rebuild"}:
        raise ValueError(f"Unsupported setup.policy.on_present={on_present!r}")

    bundle = _resolve_bundle_paths(graph_bundle_root=graph_bundle_root, graph_id=graph_id)
    bundle["bundle_root"].mkdir(parents=True, exist_ok=True)

    anchor_cfg_path = _resolve_path(project_root, paths_cfg.get("anchor_config") or "seed_candidate_workflow/configs/anchor_graph.default.json")
    seed_cfg_path = _resolve_path(project_root, paths_cfg.get("seed_config") or "seed_candidate_workflow/configs/anchor_seed.default.json")
    candidate_cfg_path = _resolve_path(project_root, paths_cfg.get("candidate_config") or "seed_candidate_workflow/configs/anchor_candidate_generation.default.json")

    anchor_run_dir = bundle["anchor_root"] / graph_id
    anchor_edges_csv = anchor_run_dir / "anchor_graph_edges_unscored.csv"
    anchor_exists = anchor_edges_csv.is_file()

    seed_stage_prefix = "seed_generation_"
    candidate_stage_prefix = "candidate_generation_"
    seed_stage_dir = None
    candidate_stage_dir = None

    component_actions: dict[str, str] = {}
    outputs: dict[str, Any] = {}

    anchor_enabled = bool(enable_cfg.get("anchor", True))
    seed_enabled = bool(enable_cfg.get("seed", True))
    candidate_enabled = bool(enable_cfg.get("candidate", True))
    seed_candidate_enabled = bool(enable_cfg.get("seed_candidate", True))
    pair_enabled = bool(enable_cfg.get("pair_training", True))
    need_pair_training = pair_enabled
    need_seed_candidate = seed_candidate_enabled or need_pair_training
    need_candidate = candidate_enabled or need_seed_candidate
    need_seed = seed_enabled or need_candidate
    need_anchor = anchor_enabled or need_seed

    # Anchor
    if not anchor_enabled:
        if not anchor_exists and need_anchor:
            raise FileNotFoundError("setup.enable.anchor=false but anchor artifact is missing")
        component_actions["anchor"] = "reused_existing" if anchor_exists else "skipped"
    else:
        should_build_anchor = (not anchor_exists and on_missing == "build") or (anchor_exists and on_present == "rebuild")
        if not anchor_exists and on_missing == "fail":
            raise FileNotFoundError(f"Anchor artifact missing and on_missing=fail: {anchor_edges_csv}")
        if should_build_anchor:
            bundle["anchor_root"].mkdir(parents=True, exist_ok=True)
            anchor_cfg = _read_json(anchor_cfg_path)
            anchor_cfg.setdefault("run", {})
            anchor_cfg["run"]["graph_id"] = graph_id
            anchor_cfg.setdefault("persistence", {})
            anchor_cfg["persistence"]["output_dir"] = str(bundle["anchor_root"])
            outputs["anchor"] = build_anchor_graph(anchor_cfg)
            component_actions["anchor"] = "built"
        else:
            component_actions["anchor"] = "reused_existing"

    # Seed
    if seed_enabled:
        bundle["seed_root"].mkdir(parents=True, exist_ok=True)
        seed_cfg = _read_json(seed_cfg_path)
        seed_cfg.setdefault("run", {})
        seed_cfg["run"]["graph_id"] = graph_id
        seed_cfg["run"]["anchor_output_root"] = str(bundle["anchor_root"])
        seed_cfg.setdefault("output", {})
        seed_cfg["output"]["output_root"] = str(bundle["seed_root"])
        seed_stage_prefix = str(seed_cfg["output"].get("stage_name") or "seed_generation") + "_"
        existing_seed = None
        try:
            existing_seed = _resolve_latest_stage_dir(bundle["seed_root"] / graph_id, seed_stage_prefix)
        except FileNotFoundError:
            existing_seed = None
        if existing_seed is None and on_missing == "fail":
            raise FileNotFoundError("Seed stage missing and on_missing=fail")
        should_build_seed = (existing_seed is None and on_missing == "build") or (existing_seed is not None and on_present == "rebuild")
        if should_build_seed:
            outputs["seed"] = run_anchor_seed_generation(seed_cfg)
            seed_stage_dir = Path(str(outputs["seed"]["output_dir"])).resolve()
            component_actions["seed"] = "built"
        else:
            seed_stage_dir = existing_seed
            component_actions["seed"] = "reused_existing"
    else:
        if need_seed:
            seed_stage_dir = _resolve_latest_stage_dir(bundle["seed_root"] / graph_id, seed_stage_prefix)
            component_actions["seed"] = "reused_existing"
        else:
            component_actions["seed"] = "skipped"

    # Candidate
    if candidate_enabled:
        bundle["candidate_root"].mkdir(parents=True, exist_ok=True)
        candidate_cfg = _read_json(candidate_cfg_path)
        candidate_cfg.setdefault("run", {})
        candidate_cfg["run"]["graph_id"] = graph_id
        candidate_cfg["run"]["anchor_output_root"] = str(bundle["anchor_root"])
        candidate_cfg["run"]["seed_output_root"] = str(bundle["seed_root"])
        candidate_cfg["run"]["seed_stage_dir"] = str(seed_stage_dir)
        candidate_cfg.setdefault("output", {})
        candidate_cfg["output"]["output_root"] = str(bundle["candidate_root"])
        candidate_stage_prefix = str(candidate_cfg["output"].get("stage_name") or "candidate_generation") + "_"
        existing_cand = None
        try:
            existing_cand = _resolve_latest_stage_dir(bundle["candidate_root"] / graph_id, candidate_stage_prefix)
        except FileNotFoundError:
            existing_cand = None
        if existing_cand is None and on_missing == "fail":
            raise FileNotFoundError("Candidate stage missing and on_missing=fail")
        should_build_candidate = (existing_cand is None and on_missing == "build") or (existing_cand is not None and on_present == "rebuild")
        if should_build_candidate:
            outputs["candidate"] = run_anchor_candidate_generation(candidate_cfg)
            candidate_stage_dir = Path(str(outputs["candidate"]["output_dir"])).resolve()
            component_actions["candidate"] = "built"
        else:
            candidate_stage_dir = existing_cand
            component_actions["candidate"] = "reused_existing"
    else:
        if need_candidate:
            candidate_stage_dir = _resolve_latest_stage_dir(bundle["candidate_root"] / graph_id, candidate_stage_prefix)
            component_actions["candidate"] = "reused_existing"
        else:
            component_actions["candidate"] = "skipped"

    p_candidate_union = (
        (candidate_stage_dir / "candidate_union.csv")
        if candidate_stage_dir is not None
        else (bundle["candidate_root"] / graph_id / "candidate_union.csv")
    )
    p_seed_all = (
        (seed_stage_dir / "seed_edges_all.csv")
        if seed_stage_dir is not None
        else (bundle["seed_root"] / graph_id / "seed_edges_all.csv")
    )
    if need_seed_candidate:
        if not p_candidate_union.is_file():
            raise FileNotFoundError(f"Missing candidate_union.csv: {p_candidate_union}")
        if not p_seed_all.is_file():
            raise FileNotFoundError(f"Missing seed_edges_all.csv: {p_seed_all}")

    # Seed-candidate pairgraph
    seed_candidate_dir = bundle["seed_candidate_root"] / graph_id
    p_seed_candidate = seed_candidate_dir / "seed_candidate_pairgraph_unscored.csv"
    if seed_candidate_enabled:
        if p_seed_candidate.is_file() and on_present == "reuse":
            component_actions["seed_candidate"] = "reused_existing"
        else:
            seed_candidate_dir.mkdir(parents=True, exist_ok=True)
            outputs["seed_candidate"] = _build_seed_candidate_pairgraph(
                candidate_union_csv=p_candidate_union,
                graph_id=graph_id,
                out_dir=seed_candidate_dir,
            )
            component_actions["seed_candidate"] = "built"
    else:
        if need_seed_candidate and not p_seed_candidate.is_file():
            raise FileNotFoundError("setup.enable.seed_candidate=false but seed_candidate pairgraph is missing")
        component_actions["seed_candidate"] = "reused_existing" if p_seed_candidate.is_file() else "skipped"

    # Pair training
    p_pair_training = bundle["pair_training_root"] / graph_id / "pair_training_dataset.csv"
    if pair_enabled:
        if p_pair_training.is_file() and on_present == "reuse":
            component_actions["pair_training"] = "reused_existing"
        else:
            p_graph_meta = pair_training_cfg.get("graph_meta_json")
            if str(p_graph_meta or "").strip():
                graph_meta_path = _resolve_path(project_root, p_graph_meta)
            else:
                graph_meta_path = _default_graph_meta_json_from_pipeline(project_root)
            pt_out_dir = bundle["pair_training_root"] / graph_id
            pt_out_dir.mkdir(parents=True, exist_ok=True)
            rn_pool = _reliable_negative_pool_from_pipeline(project_root)
            outputs["pair_training"] = build_pair_training_dataset(
                seed_edges_all_csv=p_seed_all,
                candidate_union_csv=p_candidate_union,
                output_dir=pt_out_dir,
                graph_meta_json=graph_meta_path,
                graph_id=graph_id,
                project_root=project_root,
                reliable_negative_pool=rn_pool,
            )
            component_actions["pair_training"] = "built"
    else:
        if need_pair_training and not p_pair_training.is_file():
            raise FileNotFoundError("setup.enable.pair_training=false but pair_training_dataset.csv is missing")
        component_actions["pair_training"] = "reused_existing" if p_pair_training.is_file() else "skipped"

    # Semantic shard graph
    semantic_shard_enabled = bool(enable_cfg.get("semantic_shard", False))
    semantic_shard_dir = bundle["semantic_shard_root"] / graph_id
    p_sem_pair = semantic_shard_dir / "semantic_shard_pairgraph_unscored.csv"
    if semantic_shard_enabled:
        if p_sem_pair.is_file() and on_present == "reuse":
            component_actions["semantic_shard"] = "reused_existing"
        else:
            semantic_shard_dir.mkdir(parents=True, exist_ok=True)
            outputs["semantic_shard"] = _build_semantic_shard_pairgraph(
                project_root=project_root,
                graph_id=graph_id,
                out_dir=semantic_shard_dir,
                stage_cfg=semantic_shard_cfg,
            )
            component_actions["semantic_shard"] = "built"
    else:
        if p_sem_pair.is_file():
            component_actions["semantic_shard"] = "reused_existing"

    return {
        "graph_id": graph_id,
        "bundle_root": str(bundle["bundle_root"]),
        "paths": {
            "anchor_run_dir": str(anchor_run_dir),
            "seed_stage_dir": str(seed_stage_dir),
            "candidate_stage_dir": str(candidate_stage_dir),
            "seed_candidate_pairgraph_csv": str(p_seed_candidate),
            "semantic_shard_pairgraph_csv": str(p_sem_pair),
            "pair_training_dataset_csv": str(p_pair_training),
            "candidate_union_csv": str(p_candidate_union),
            "seed_edges_all_csv": str(p_seed_all),
        },
        "actions": component_actions,
        "outputs": outputs,
    }

