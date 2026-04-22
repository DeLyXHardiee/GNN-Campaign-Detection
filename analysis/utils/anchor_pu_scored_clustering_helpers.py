"""
PU-trained pair scorer -> weighted email graph -> anchor community sweep + eval.

Inference-only: loads checkpoint, scores the fixed pair-training universe, merges
``seed_edges_all.csv``, writes CSVs + JSON summaries under a timestamped bundle directory
(``{stage_name}_{UTC}/`` under ``output.output_parent_dir``), reuses
``run_anchor_multi_gt_community_sweep``, and optionally runs PU threshold retention on the
full ``pu_scored_candidate_edges_all.csv`` universe.
"""

from __future__ import annotations

import json
import math
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from analysis.utils import graph_structure_helpers as gh
from analysis.utils.anchor_candidate_eval_helpers import _pair
from analysis.utils.anchor_candidate_rare_artifact_helpers import _resolve_latest_seed_dir
from analysis.utils.anchor_graph_community_helpers import (
    _build_weighted_email_graph,
    run_anchor_multi_gt_community_sweep,
)
from analysis.utils.pu_threshold_retention_analysis import run_pu_threshold_retention_analysis
from analysis.utils.anchor_graph_helpers import load_anchor_graph_artifacts


def _ensure_gnn_on_path() -> None:
    root = gh.find_project_root()
    for p in (str(root), str(root / "core"), str(root / "core" / "GNN")):
        if p not in sys.path:
            sys.path.insert(0, p)


def _null_json(x: Any) -> Any:
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return None
    if isinstance(x, dict):
        return {k: _null_json(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_null_json(v) for v in x]
    return x


def _load_json_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _quantiles_dict_np(x: np.ndarray, qs: tuple[float, ...]) -> dict[str, float | None]:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {f"q{int(q * 100)}": None for q in qs}
    return {f"q{int(q * 100)}": float(np.quantile(x, q)) for q in qs}


def _seed_pairs_from_csv(seed_csv: Path) -> set[tuple[str, str]]:
    df = pd.read_csv(seed_csv, low_memory=False)
    if df.empty or "email_i" not in df.columns or "email_j" not in df.columns:
        return set()
    out: set[tuple[str, str]] = set()
    for a, b in zip(df["email_i"].astype(str), df["email_j"].astype(str)):
        out.add(_pair(a, b))
    return out


def _flag_col(df: pd.DataFrame, name: str, default: bool = False) -> pd.Series:
    if name not in df.columns:
        return pd.Series(default, index=df.index, dtype=bool)
    return df[name].fillna(False).astype(bool)


def _graph_diag_subset(
    *,
    node_ids: list[str],
    edges_df: pd.DataFrame,
    min_edge_weight: float,
    weight_col: str = "edge_weight",
) -> dict[str, Any]:
    import networkx as nx

    g = _build_weighted_email_graph(
        node_ids=node_ids,
        edges_df=edges_df,
        weight_col=weight_col,
        min_edge_weight=float(min_edge_weight),
    )
    n = len(node_ids)
    deg = dict(g.degree())
    isolated = int(sum(1 for nid in node_ids if deg.get(nid, 0) == 0))
    if g.number_of_edges() == 0:
        return {
            "n_edges_after_min_edge_weight": 0,
            "n_isolated_nodes_in_anchor_set": isolated,
            "largest_connected_component_size": 1,
            "largest_component_fraction_of_anchor_nodes": float(1 / max(1, n)),
        }

    comps = list(nx.connected_components(g))
    lcc = max(len(c) for c in comps)
    return {
        "n_edges_after_min_edge_weight": int(g.number_of_edges()),
        "n_isolated_nodes_in_anchor_set": isolated,
        "largest_connected_component_size": int(lcc),
        "largest_component_fraction_of_anchor_nodes": float(lcc / max(1, n)),
    }


def _load_baseline_best_v_by_gt(summary_path: Path | None) -> dict[str, Any]:
    if summary_path is None or not summary_path.is_file():
        return {}
    try:
        data = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    out: dict[str, dict[str, Any]] = {}
    for row in data.get("best_rows_by_gt") or []:
        slug = str(row.get("gt_slug") or "")
        br = row.get("best_row") or {}
        if slug:
            out[slug] = {
                "v_measure": br.get("v_measure"),
                "homogeneity": br.get("homogeneity"),
                "completeness": br.get("completeness"),
                "min_edge_weight": br.get("min_edge_weight"),
                "method": br.get("method"),
                "resolution": br.get("resolution"),
                "n_eval": br.get("n_eval"),
                "coverage_gt": br.get("coverage_gt"),
            }
    return out


def _load_oracle_ceiling_rows(csv_path: Path | None) -> list[dict[str, Any]]:
    if csv_path is None or not csv_path.is_file():
        return []
    df = pd.read_csv(csv_path, low_memory=False)
    if df.empty:
        return []
    rows = df.to_dict(orient="records")
    slim: list[dict[str, Any]] = []
    for r in rows:
        slim.append(
            {
                "gt_path": r.get("gt_path"),
                "v_measure": r.get("v_measure"),
                "homogeneity": r.get("homogeneity"),
                "completeness": r.get("completeness"),
                "n_eval": r.get("n_eval"),
            }
        )
    return slim


def _resolve_weight_mode(
    graph_cfg: dict[str, Any],
) -> tuple[str, str, str]:
    """
    Returns (weight_mode, transform_label, transform_note).

    Supported:
      - raw_score
      - raw_score_squared
      - raw_score_cubed
    """
    mode = str(graph_cfg.get("weight_mode") or "raw_score").strip().lower()
    if mode == "raw_score":
        return mode, "identity", "edge_weight = pu_score"
    if mode == "raw_score_squared":
        return mode, "square", "edge_weight = pu_score ** 2"
    if mode == "raw_score_cubed":
        return mode, "cube", "edge_weight = pu_score ** 3"
    raise ValueError(
        "Unsupported graph_construction.weight_mode: "
        f"{mode!r}. Expected one of: raw_score, raw_score_squared, raw_score_cubed"
    )


def _apply_non_seed_weight_transform(
    pu_scores: pd.Series,
    *,
    weight_mode: str,
) -> pd.Series:
    s = pd.to_numeric(pu_scores, errors="coerce")
    if weight_mode == "raw_score":
        return s
    if weight_mode == "raw_score_squared":
        return s.pow(2)
    if weight_mode == "raw_score_cubed":
        return s.pow(3)
    raise ValueError(
        "Unsupported graph_construction.weight_mode: "
        f"{weight_mode!r}. Expected one of: raw_score, raw_score_squared, raw_score_cubed"
    )


def run_anchor_pu_scored_clustering_stage(config: dict[str, Any]) -> dict[str, Any]:
    _ensure_gnn_on_path()
    from analysis.utils.pair_score_separation import load_pair_supervision_for_inference, score_pair_rows
    from src.pair_train import load_pair_training_dataframe, resolve_pair_dataset_csv

    config = dict(config)
    pipeline_config_path = str(config.pop("_pipeline_config_path", "") or "").strip() or None

    pu_cfg = config.get("pu_run") or {}
    run_cfg = config.get("run") or {}
    graph_cfg = config.get("graph_construction") or {}
    eval_cfg = config.get("evaluation") or {}
    comm_cfg_in = config.get("community") or {}
    comm_path = str(config.get("community_config_path") or "").strip()
    out_cfg = config.get("output") or {}
    pu_comm_detect = config.get("community_detection") or {}
    use_edge_weights_in_partitioning = bool(
        pu_comm_detect.get("use_edge_weights_in_partitioning", True)
    )

    project_root = gh.find_project_root()

    pu_run_dir = Path(str(pu_cfg.get("run_dir") or "")).expanduser()
    if not pu_run_dir.is_absolute():
        pu_run_dir = (project_root / pu_run_dir).resolve()
    else:
        pu_run_dir = pu_run_dir.resolve()
    if not pu_run_dir.is_dir():
        raise FileNotFoundError(f"pu_run.run_dir not found: {pu_run_dir}")

    graph_pt = Path(str(pu_cfg.get("graph_pt") or "")).expanduser()
    if not graph_pt.is_absolute():
        graph_pt = (project_root / graph_pt).resolve()
    else:
        graph_pt = graph_pt.resolve()
    if not graph_pt.is_file():
        raise FileNotFoundError(f"pu_run.graph_pt not found: {graph_pt}")

    checkpoint_name = str(pu_cfg.get("checkpoint", "best_model.pt"))
    device = str(pu_cfg.get("device", "cpu"))
    to_undirected = not bool(pu_cfg.get("no_to_undirected", False))

    raw_pair = str(pu_cfg.get("pair_dataset_csv") or "").strip()
    if raw_pair:
        pair_csv = Path(raw_pair).expanduser()
        if not pair_csv.is_absolute():
            pair_csv = (project_root / pair_csv).resolve()
        else:
            pair_csv = pair_csv.resolve()
    else:
        tc_path = pu_run_dir / "training_config.json"
        tc = json.loads(tc_path.read_text(encoding="utf-8"))
        raw_ds = tc.get("pair_dataset_csv")
        if not raw_ds:
            raise ValueError("pu_run.pair_dataset_csv missing and not in training_config.json")
        pair_csv = resolve_pair_dataset_csv(str(raw_ds), project_root=project_root)

    score_rule_version = str(graph_cfg.get("score_rule_version") or "pu_sigmoid_v1")
    # Backward compatible alias: allow both seed_edge_weight and seed_weight.
    seed_edge_weight = float(
        graph_cfg.get("seed_edge_weight", graph_cfg.get("seed_weight", 1.0))
    )
    weight_mode, transform_label, transform_note = _resolve_weight_mode(graph_cfg)
    export_non_seed_min = float(graph_cfg.get("export_clustering_edges_non_seed_min_pu_score", 0.0))
    prefilter_thresholds = [float(x) for x in (graph_cfg.get("non_seed_pu_thresholds_for_summary") or [0.0, 0.1, 0.2, 0.3])]

    graph_run_id = str(run_cfg.get("graph_run_id") or "").strip()
    if not graph_run_id:
        raise ValueError("run.graph_run_id is required")

    anchor_output_root = Path(
        run_cfg.get("anchor_output_root") or (project_root / "analysis" / "output" / "anchor_graph")
    ).expanduser().resolve()
    anchor_run_dir = (anchor_output_root / graph_run_id).resolve()
    nodes_df, _ea, _cand, anchor_summary, _g = load_anchor_graph_artifacts(anchor_run_dir, load_graph_pickle=False)
    node_ids = [str(x) for x in nodes_df["external_id"].astype(str).tolist()]
    node_set = set(node_ids)

    seed_stage_dir_override = str(run_cfg.get("seed_stage_dir") or "").strip()
    if seed_stage_dir_override:
        seed_dir = Path(seed_stage_dir_override).expanduser()
        if not seed_dir.is_absolute():
            seed_dir = (project_root / seed_dir).resolve()
        else:
            seed_dir = seed_dir.resolve()
        if not seed_dir.is_dir():
            raise FileNotFoundError(f"run.seed_stage_dir not found: {seed_dir}")
    else:
        seed_output_root = Path(
            run_cfg.get("seed_output_root") or (project_root / "analysis" / "output" / "anchor_seeds")
        ).expanduser().resolve()
        seed_prefix = str(run_cfg.get("seed_stage_name_prefix") or "seed_generation_")
        seed_dir = _resolve_latest_seed_dir(
            seed_output_root=seed_output_root,
            graph_run_id=graph_run_id,
            seed_stage_name_prefix=seed_prefix,
        )
    p_seed_all = seed_dir / "seed_edges_all.csv"
    if not p_seed_all.is_file():
        raise FileNotFoundError(f"Missing seed_edges_all.csv: {p_seed_all}")
    seed_pair_set = _seed_pairs_from_csv(p_seed_all)

    out_parent = str(out_cfg.get("output_parent_dir") or "").strip()
    if out_parent:
        artifact_parent = Path(out_parent).expanduser()
        if not artifact_parent.is_absolute():
            artifact_parent = (project_root / artifact_parent).resolve()
        else:
            artifact_parent = artifact_parent.resolve()
    else:
        artifact_parent = (pu_run_dir / "pu_scored_clustering").resolve()

    if comm_path:
        comm_base_path = Path(comm_path).expanduser()
        if not comm_base_path.is_absolute():
            comm_base_path = (project_root / comm_base_path).resolve()
        else:
            comm_base_path = comm_base_path.resolve()
        community_cfg = _load_json_config(comm_base_path)
    else:
        community_cfg = deepcopy(comm_cfg_in)
        if not community_cfg.get("ground_truth", {}).get("paths"):
            raise ValueError("community_config_path or community.ground_truth.paths is required")

    community_cfg = deepcopy(community_cfg)
    community_cfg.setdefault("run", {})
    community_cfg["run"]["graph_run_id"] = graph_run_id
    community_cfg["run"]["anchor_output_root"] = str(anchor_output_root)
    community_cfg.setdefault("output", {})
    community_cfg["output"]["stage_name"] = str(
        community_cfg["output"].get("stage_name") or "pu_scored_community_sweep"
    )
    bundle_stage_name = str(community_cfg["output"]["stage_name"])
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    bundle_dir = artifact_parent / f"{bundle_stage_name}_{stamp}"
    bundle_dir.mkdir(parents=True, exist_ok=True)

    df, pair_stats = load_pair_training_dataframe(pair_csv)
    df_work = df.reset_index(drop=True)
    df_work["_row"] = np.arange(len(df_work), dtype=np.int64)

    bundle = load_pair_supervision_for_inference(
        run_dir=pu_run_dir,
        graph_pt=graph_pt,
        checkpoint_name=checkpoint_name,
        device=device,
        to_undirected=to_undirected,
    )
    scored_tup = score_pair_rows(
        model=bundle["model"],
        pair_scorer=bundle["pair_scorer"],
        data_cpu=bundle["data_cpu"],
        df_work=df_work,
        device=bundle["device"],
        fanout=bundle["fanout"],
        pair_batch_size=bundle["pair_batch_size"],
        max_unique_emails=bundle["max_unique_emails"],
        with_logits=True,
    )
    if not isinstance(scored_tup, tuple):
        raise RuntimeError("score_pair_rows(..., with_logits=True) must return (pu_score, pu_logit)")
    pu_score, pu_logit = scored_tup

    is_seed_col = _flag_col(df_work, "is_seed_pair")
    from_seed_ds = _flag_col(df_work, "from_seed")
    pair_keys = {_pair(str(a), str(b)) for a, b in zip(df_work["email_i"], df_work["email_j"])}
    in_seed_file = np.array(
        [
            _pair(str(df_work["email_i"].iloc[k]), str(df_work["email_j"].iloc[k])) in seed_pair_set
            for k in range(len(df_work))
        ],
        dtype=bool,
    )
    trusted_seed = (is_seed_col.to_numpy() | from_seed_ds.to_numpy() | in_seed_file).astype(bool)

    rows_all: list[dict[str, Any]] = []
    base_cols = [
        "email_i",
        "email_j",
        "graph_email_idx_i",
        "graph_email_idx_j",
        "pair_status",
        "is_seed_pair",
        "is_candidate_pair",
        "from_seed",
        "from_rare_artifact",
        "from_semantic",
        "from_component",
        "from_2hop",
        "source_count",
    ]
    # Pair-training / candidate context for downstream retention & diagnostics (pair_train schema).
    seed_ctx_cols = [
        "seed_component_i",
        "seed_component_j",
        "same_seed_component_flag",
        "cross_seed_component_flag",
    ]

    def _row_seed_ctx(r: pd.Series) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for c in seed_ctx_cols:
            if c not in df_work.columns:
                out[c] = False if c.endswith("_flag") else None
                continue
            v = r.get(c)
            if c.endswith("_flag"):
                out[c] = bool(v) if pd.notna(v) else False
            else:
                nv = pd.to_numeric(v, errors="coerce")
                out[c] = None if pd.isna(nv) else int(nv)
        return out

    for k in range(len(df_work)):
        r = df_work.iloc[k]
        rec: dict[str, Any] = {"score_rule_version": score_rule_version}
        for c in base_cols:
            rec[c] = r[c] if c in df_work.columns else (False if c.startswith("from_") or c in ("is_seed_pair", "is_candidate_pair") else None)
        if "is_candidate_pair" not in df_work.columns:
            rec["is_candidate_pair"] = True
        rec.update(_row_seed_ctx(r))
        rec["pu_logit"] = float(pu_logit[k]) if np.isfinite(pu_logit[k]) else None
        rec["pu_score"] = float(pu_score[k]) if np.isfinite(pu_score[k]) else None
        rec["trusted_seed_edge"] = bool(trusted_seed[k])
        rows_all.append(rec)

    for pk in seed_pair_set:
        if pk in pair_keys:
            continue
        a, b = pk
        if a not in node_set or b not in node_set:
            continue
        rows_all.append(
            {
                "email_i": a,
                "email_j": b,
                "graph_email_idx_i": None,
                "graph_email_idx_j": None,
                "pair_status": "positive",
                "is_seed_pair": True,
                "is_candidate_pair": False,
                "from_seed": True,
                "from_rare_artifact": False,
                "from_semantic": False,
                "from_component": False,
                "from_2hop": False,
                "source_count": 0,
                "seed_component_i": None,
                "seed_component_j": None,
                "same_seed_component_flag": False,
                "cross_seed_component_flag": False,
                "score_rule_version": score_rule_version,
                "pu_logit": None,
                "pu_score": None,
                "trusted_seed_edge": True,
            }
        )

    df_export = pd.DataFrame(rows_all)
    p_all = bundle_dir / "pu_scored_candidate_edges_all.csv"
    df_export.to_csv(p_all, index=False)

    # --- clustering edge table (canonical, dedup) ---
    ce_rows: list[dict[str, Any]] = []
    for _, r in df_export.iterrows():
        a, b = _pair(str(r["email_i"]), str(r["email_j"]))
        ts = bool(r.get("trusted_seed_edge"))
        ps = r.get("pu_score")
        ps_f = float(ps) if ps is not None and not (isinstance(ps, float) and math.isnan(ps)) else float("nan")
        ce_rows.append(
            {
                "email_a": a,
                "email_b": b,
                "pu_score": ps_f,
                "pu_logit": r.get("pu_logit"),
                "from_seed": ts,
                "from_rare_artifact": bool(r.get("from_rare_artifact", False)),
                "from_semantic": bool(r.get("from_semantic", False)),
                "from_component": bool(r.get("from_component", False)),
                "from_2hop": bool(r.get("from_2hop", False)),
                "source_count": int(pd.to_numeric(r.get("source_count"), errors="coerce") or 0),
                "score_rule_version": score_rule_version,
            }
        )
    cdf = pd.DataFrame(ce_rows)
    cdf["__sort_seed"] = cdf["from_seed"].astype(int)
    cdf["__pu"] = pd.to_numeric(cdf["pu_score"], errors="coerce").fillna(-1.0)
    cdf = cdf.sort_values(["email_a", "email_b", "__sort_seed", "__pu"], ascending=[True, True, False, False])
    cdf = cdf.drop(columns=["__sort_seed", "__pu"])
    cdf = cdf.drop_duplicates(subset=["email_a", "email_b"], keep="first")
    non_seed_weight_series = _apply_non_seed_weight_transform(
        cdf["pu_score"], weight_mode=weight_mode
    )
    cdf["edge_weight"] = np.where(
        cdf["from_seed"], seed_edge_weight, non_seed_weight_series
    )
    mask_keep = cdf["from_seed"] | (
        pd.to_numeric(cdf["pu_score"], errors="coerce").ge(export_non_seed_min)
        & pd.to_numeric(cdf["pu_score"], errors="coerce").notna()
    )
    cdf_kept = cdf.loc[mask_keep].copy()

    p_cluster = bundle_dir / "pu_scored_clustering_edges.csv"
    out_cols = [
        "email_a",
        "email_b",
        "edge_weight",
        "from_seed",
        "from_semantic",
        "from_rare_artifact",
        "from_component",
        "from_2hop",
        "source_count",
        "score_rule_version",
    ]
    cdf_kept[out_cols].to_csv(p_cluster, index=False)

    finite = np.isfinite(pu_score)
    n_finite = int(finite.sum())
    n_fail = int(len(df_work) - n_finite)
    # rows_all aligned with df_export rows - use df_export trusted_seed_edge
    ts_series = df_export["trusted_seed_edge"].astype(bool)
    ps_series = pd.to_numeric(df_export["pu_score"], errors="coerce")
    non_seed_weight_for_summary = _apply_non_seed_weight_transform(
        ps_series, weight_mode=weight_mode
    )
    pos_mask = df_export["pair_status"].astype(str).str.lower() == "positive" if "pair_status" in df_export.columns else pd.Series(False, index=df_export.index)
    unl_mask = df_export["pair_status"].astype(str).str.lower() == "unlabeled" if "pair_status" in df_export.columns else pd.Series(False, index=df_export.index)

    qs = (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95)

    def _qser(mask: pd.Series) -> dict[str, float | None]:
        v = ps_series.loc[mask & ps_series.notna()].to_numpy(dtype=np.float64)
        return _quantiles_dict_np(v, qs)

    post_by_t: dict[str, Any] = {}
    for t in prefilter_thresholds:
        sub = cdf[(cdf["from_seed"]) | (pd.to_numeric(cdf["pu_score"], errors="coerce") >= float(t))]
        touched: set[str] = set()
        for aa, bb in zip(sub["email_a"].astype(str), sub["email_b"].astype(str), strict=False):
            touched.add(aa)
            touched.add(bb)
        non_seed = sub[~sub["from_seed"]]
        src_ctr: dict[str, int] = {}
        for lbl, col in [
            ("from_semantic", "from_semantic"),
            ("from_rare_artifact", "from_rare_artifact"),
            ("from_component", "from_component"),
            ("from_2hop", "from_2hop"),
        ]:
            src_ctr[lbl] = int(non_seed[col].fillna(False).astype(bool).sum())
        post_by_t[str(t)] = {
            "non_seed_pu_score_min": t,
            "n_edges_kept_total": int(len(sub)),
            "n_non_seed_edges_kept": int((~sub["from_seed"]).sum()),
            "n_seed_edges_kept": int(sub["from_seed"].sum()),
            "n_unique_emails_touched": int(len(touched)),
            "non_seed_source_breakdown": src_ctr,
        }

    created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    graph_summary: dict[str, Any] = {
        "metadata": {
            "created_at_utc": created_at,
            "artifact_parent_dir": str(artifact_parent),
            "bundle_output_dir": str(bundle_dir),
            "pu_run_dir": str(pu_run_dir),
            "checkpoint": str(pu_run_dir / "models" / checkpoint_name),
            "pair_dataset_csv": str(pair_csv),
            "graph_pt": str(graph_pt),
            "pipeline_config_path": pipeline_config_path,
        },
        "inputs": {
            "graph_run_id": graph_run_id,
            "anchor_run_dir": str(anchor_run_dir),
            "seed_edges_all_path": str(p_seed_all),
            "seed_stage_dir": str(seed_dir),
            "total_scored_pair_rows": int(len(df_work)),
            "total_export_rows_including_seed_only": int(len(df_export)),
            "total_seed_pairs_in_seed_file": int(len(seed_pair_set)),
            "pair_load_stats": pair_stats,
        },
        "score_stats": {
            "n_finite_pu_scores_on_pair_rows": n_finite,
            "n_failed_or_missing_inference_on_pair_rows": n_fail,
            "weight_mode": weight_mode,
            "non_seed_weight_transform": transform_label,
            "quantiles_pu_score_all_rows_with_score": _qser(pd.Series(True, index=df_export.index)),
            "quantiles_pu_score_trusted_seed_rows": _qser(ts_series),
            "quantiles_pu_score_non_trusted_seed_rows": _qser(~ts_series),
            "quantiles_pu_score_pair_status_positive": _qser(pos_mask),
            "quantiles_pu_score_pair_status_unlabeled": _qser(unl_mask),
            "quantiles_non_seed_weight_after_transform": _quantiles_dict_np(
                non_seed_weight_for_summary.loc[(~ts_series) & non_seed_weight_for_summary.notna()].to_numpy(dtype=np.float64),
                qs,
            ),
        },
        "graph_construction_rule": {
            "weight_mode": weight_mode,
            "non_seed_weight_transform": transform_label,
            "seeds": (
                "All pairs in trusted_seed_edge (is_seed_pair OR from_seed in dataset OR pair in seed_edges_all.csv) "
                "get edge_weight = seed_edge_weight; PU scores ignored for seed weight."
            ),
            "non_seeds": (
                f"{transform_note}; rows without finite pu_score are omitted unless trusted seed."
            ),
            "export_csv_non_seed_floor": export_non_seed_min,
            "seed_edge_weight": seed_edge_weight,
            "seed_weight_rule": "fixed_constant",
            "dedup": "Canonical (email_a, email_b) via sorted pair; sort key prefers from_seed then max pu_score; edge_weight uses seed rule above.",
            "score_rule_version": score_rule_version,
        },
        "thresholds_used": {
            "non_seed_pu_prefilter_thresholds_documented_in_post_threshold_graph_by_threshold": prefilter_thresholds,
            "community_sweep_min_edge_weights": list(config.get("community_sweep_weight_thresholds", [0.0, 0.1, 0.2, 0.3])),
        },
        "community_detection": {
            "use_edge_weights_in_partitioning": use_edge_weights_in_partitioning,
            "description": (
                "When true, Louvain/Leiden use edge_weight in the objective (PU probabilities on non-seeds). "
                "When false, min_edge_weight still filters on edge_weight, but each surviving edge is treated "
                "as weight 1.0 inside the partitioner (PU scores act only as a cutoff, not as tie-breaking strength)."
            ),
        },
        "pre_threshold_graph": {
            "n_unique_canonical_edges": int(len(cdf)),
            "n_seed_edges": int(cdf["from_seed"].sum()),
            "n_non_seed_edges": int((~cdf["from_seed"]).sum()),
        },
        "post_threshold_graph_by_threshold": post_by_t,
        "export_clustering_edges_csv": {
            "path": str(p_cluster),
            "n_rows": int(len(cdf_kept)),
            "n_seed": int(cdf_kept["from_seed"].sum()),
            "n_non_seed": int((~cdf_kept["from_seed"]).sum()),
            "export_non_seed_min_pu_score_applied": export_non_seed_min,
        },
        "notes": [
            "post_threshold_graph_by_threshold counts use deduped canonical edges before export_non_seed_min_pu_score filter.",
            "Community sweep applies min_edge_weight to all edges; seeds use weight=1.0 so they survive thresholds below 1.0.",
            "Sweep min_edge_weight values align with non-seed PU score cutoffs when non-seed edge_weight equals pu_score.",
            "See community_detection.use_edge_weights_in_partitioning in this JSON and in anchor_community_multi_gt_summary.json.",
        ],
    }
    p_graph_sum = bundle_dir / "pu_scored_graph_summary.json"
    p_graph_sum.write_text(json.dumps(_null_json(graph_summary), indent=2, ensure_ascii=False), encoding="utf-8")

    community_cfg["run"]["custom_edges_csv"] = str(p_cluster.resolve())
    community_cfg["run"].pop("community_output_parent_dir", None)
    community_cfg["run"]["community_bundle_out_dir"] = str(bundle_dir.resolve())
    sweep_merged = community_cfg.setdefault("sweep", {})
    sweep_merged["weight_thresholds"] = list(config.get("community_sweep_weight_thresholds", [0.0, 0.1, 0.2, 0.3]))
    sweep_merged["use_edge_weights_in_partitioning"] = use_edge_weights_in_partitioning

    comm_res = run_anchor_multi_gt_community_sweep(community_cfg)

    # --- eval summary ---
    per_threshold_rows: list[dict[str, Any]] = []
    best_vm = -1.0
    best_row: dict[str, Any] | None = None
    for gt_out in comm_res.get("per_ground_truth_outputs") or []:
        p_csv = Path(str(gt_out.get("sweep_csv") or ""))
        gt_slug = str(gt_out.get("gt_slug") or "")
        if not p_csv.is_file():
            continue
        sdf = pd.read_csv(p_csv, low_memory=False)
        for _, row in sdf.iterrows():
            d = {
                "gt_slug": gt_slug,
                "gt_path": str(gt_out.get("gt_path", "")),
                "min_edge_weight": float(row.get("min_edge_weight", 0.0)),
                "method": str(row.get("method", "")),
                "resolution": float(row.get("resolution", 0.0)),
                "homogeneity": float(row["homogeneity"]) if pd.notna(row.get("homogeneity")) else None,
                "completeness": float(row["completeness"]) if pd.notna(row.get("completeness")) else None,
                "v_measure": float(row["v_measure"]) if pd.notna(row.get("v_measure")) else None,
                "n_eval": int(row["n_eval"]) if pd.notna(row.get("n_eval")) else None,
                "coverage_gt": float(row["coverage_gt"]) if pd.notna(row.get("coverage_gt")) else None,
                "n_edges_after_threshold": int(row["n_edges_after_threshold"])
                if pd.notna(row.get("n_edges_after_threshold"))
                else None,
                "n_communities": int(row["n_communities"]) if pd.notna(row.get("n_communities")) else None,
            }
            diag = _graph_diag_subset(
                node_ids=node_ids,
                edges_df=cdf_kept[out_cols],
                min_edge_weight=float(d["min_edge_weight"]),
            )
            d["graph_diag_at_min_edge_weight"] = diag
            per_threshold_rows.append(d)
            vm = d["v_measure"]
            if vm is not None and vm > best_vm:
                best_vm = vm
                best_row = dict(d)

    def _eval_path(key: str) -> Path | None:
        raw = str(eval_cfg.get(key) or "").strip()
        if not raw:
            return None
        p = Path(raw).expanduser()
        if not p.is_absolute():
            p = (project_root / p).resolve()
        else:
            p = p.resolve()
        return p

    baseline_anchor = _eval_path("baseline_anchor_community_summary_json")
    handcrafted = _eval_path("handcrafted_scored_clustering_summary_json")
    oracle_csv = _eval_path("candidate_oracle_ceiling_csv")

    eval_summary: dict[str, Any] = {
        "metadata": {
            "created_at_utc": created_at,
            "artifact_parent_dir": str(artifact_parent),
            "bundle_output_dir": str(bundle_dir),
            "pu_run_dir": str(pu_run_dir),
            "community_sweep_output_dir": comm_res.get("output_dir"),
            "community_multi_gt_summary_json": comm_res.get("summary_json"),
            "use_edge_weights_in_partitioning": use_edge_weights_in_partitioning,
        },
        "best_by_v_measure": best_row,
        "per_threshold_results": per_threshold_rows,
        "baseline_comparison": {
            "baseline_anchor_community_summary_json": str(baseline_anchor) if baseline_anchor else None,
            "baseline_best_v_measure_by_gt_slug": _load_baseline_best_v_by_gt(
                baseline_anchor if baseline_anchor and baseline_anchor.is_file() else None
            ),
            "handcrafted_scored_clustering_summary_json": str(handcrafted) if handcrafted else None,
            "handcrafted_best_v_measure_by_gt_slug": _load_baseline_best_v_by_gt(
                handcrafted if handcrafted and handcrafted.is_file() else None
            ),
        },
        "oracle_comparison": {
            "candidate_oracle_ceiling_csv": str(oracle_csv) if oracle_csv else None,
            "oracle_ceiling_rows": _load_oracle_ceiling_rows(oracle_csv if oracle_csv and oracle_csv.is_file() else None),
        },
        "notes": [
            "best_by_v_measure is the single sweep row (any GT) with highest v_measure.",
            "Populate evaluation.* paths in config to compare against prior anchor / handcrafted / oracle CSVs.",
            "For baseline_comparison JSONs, use anchor_community_multi_gt_summary.json from a prior community sweep (same schema as this stage's community_summary_json), not scored_clustering_graph_summary.json.",
            "PU threshold retention (full pu_scored_candidate_edges_all.csv) is written under bundle_output_dir/pu_ret/ when threshold_retention.enabled is true (plots in pu_ret/plots/).",
        ],
    }

    ret_cfg = config.get("threshold_retention") or {}
    retention_out: dict[str, Any] | None = None
    if bool(ret_cfg.get("enabled", True)):
        gt_cfg_rt = community_cfg.get("ground_truth") or {}
        gt_paths_raw_rt = gt_cfg_rt.get("paths") or []
        if isinstance(gt_paths_raw_rt, list) and gt_paths_raw_rt:
            gt_paths_rt: list[Path] = []
            for raw in gt_paths_raw_rt:
                p_gt = Path(str(raw)).expanduser()
                if not p_gt.is_absolute():
                    p_gt = (project_root / p_gt).resolve()
                else:
                    p_gt = p_gt.resolve()
                if not p_gt.is_file():
                    raise FileNotFoundError(f"Ground truth file not found: {p_gt}")
                gt_paths_rt.append(p_gt)
            thr_raw = ret_cfg.get("thresholds")
            thr_list = (
                [float(x) for x in thr_raw]
                if isinstance(thr_raw, list) and thr_raw
                else list(config.get("community_sweep_weight_thresholds", [0.0, 0.1, 0.2, 0.3]))
            )
            retention_out = run_pu_threshold_retention_analysis(
                scored_pairs_csv=p_all,
                gt_paths=gt_paths_rt,
                thresholds=thr_list,
                output_dir=bundle_dir / "pu_ret",
                keep_seeds_always=bool(ret_cfg.get("keep_seeds_always", True)),
                make_plots=bool(ret_cfg.get("make_plots", True)),
            )
    eval_summary["pu_threshold_retention"] = retention_out

    p_eval = bundle_dir / "pu_scored_graph_eval_summary.json"
    p_eval.write_text(json.dumps(_null_json(eval_summary), indent=2, ensure_ascii=False), encoding="utf-8")

    return {
        "pipeline_config_path": pipeline_config_path,
        "artifact_parent_dir": str(artifact_parent),
        "output_dir": str(bundle_dir),
        "pu_scored_candidate_edges_all_csv": str(p_all.resolve()),
        "pu_scored_clustering_edges_csv": str(p_cluster.resolve()),
        "pu_scored_graph_summary_json": str(p_graph_sum.resolve()),
        "pu_scored_graph_eval_summary_json": str(p_eval.resolve()),
        "community_sweep_output_dir": comm_res.get("output_dir"),
        "community_summary_json": comm_res.get("summary_json"),
        "pu_threshold_retention_summary_json": retention_out.get("summary_json") if retention_out else None,
        "pu_threshold_retention_csv": retention_out.get("summary_csv") if retention_out else None,
        "anchor_summary": anchor_summary,
    }
