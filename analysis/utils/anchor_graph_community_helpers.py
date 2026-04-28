from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd

from analysis.utils import community_eval_contract as cec
from analysis.utils import community_sweep_driver as csd
from analysis.utils import graph_structure_helpers as gh
from analysis.utils.config_run_fields import resolve_graph_id
from analysis.utils import raw_gnn_notebook as rn
from analysis.utils.graph_scorer_registry import SCORER_REGISTRY
from analysis.utils.anchor_candidate_rare_artifact_helpers import _resolve_latest_seed_dir
from analysis.utils.anchor_graph_helpers import load_anchor_graph_artifacts

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - optional dependency
    tqdm = None


def _slugify(s: str) -> str:
    t = re.sub(r"[^A-Za-z0-9._-]+", "_", str(s).strip())
    t = re.sub(r"_+", "_", t).strip("_.-")
    return t or "unknown"


def _gt_slug(gt_path: Path) -> str:
    return _slugify(gt_path.stem)


def _normalize_sort_metric(raw: str | None) -> str:
    """Map config value to column name: homogeneity | completeness | v_measure."""
    if raw is None or (isinstance(raw, str) and not str(raw).strip()):
        return "homogeneity"
    s = str(raw).strip().lower().replace("-", "_")
    aliases = {
        "homogeneity": "homogeneity",
        "homogeniety": "homogeneity",
        "completeness": "completeness",
        "v_measure": "v_measure",
        "vmeasure": "v_measure",
    }
    if s not in aliases:
        raise ValueError(
            f"Invalid sweep.sort_by metric {raw!r}. "
            f"Use one of: homogeneity, completeness, v_measure"
        )
    return aliases[s]


def _sort_tiebreakers(primary: str) -> list[str]:
    """Descending sort key: primary first, then stable tie-breakers."""
    return cec.metric_sort_columns(primary)


def _load_anchor_run(
    *,
    graph_id: str,
    anchor_output_root: Path,
) -> tuple[Path, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    run_dir = (anchor_output_root / graph_id).expanduser().resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Anchor graph run directory not found: {run_dir}")
    nodes_df, edges_df, _cand, summary, _g = load_anchor_graph_artifacts(
        run_dir, load_graph_pickle=False
    )
    req_node_cols = {"external_id"}
    req_edge_cols = {"email_i", "email_j"}
    miss_n = sorted(c for c in req_node_cols if c not in nodes_df.columns)
    miss_e = sorted(c for c in req_edge_cols if c not in edges_df.columns)
    if miss_n:
        raise ValueError(f"Anchor nodes missing required columns: {miss_n}")
    if miss_e:
        raise ValueError(f"Anchor edges missing required columns: {miss_e}")
    nodes_df = nodes_df.copy()
    edges_df = edges_df.copy()
    nodes_df["external_id"] = nodes_df["external_id"].astype(str)
    edges_df["email_i"] = edges_df["email_i"].astype(str)
    edges_df["email_j"] = edges_df["email_j"].astype(str)
    return run_dir, nodes_df, edges_df, summary


def _build_weighted_email_graph(
    *,
    node_ids: list[str],
    edges_df: pd.DataFrame,
    weight_col: str,
    min_edge_weight: float,
    use_edge_weights_in_partitioning: bool = True,
    apply_threshold_filter: bool = True,
) -> nx.Graph:
    g = nx.Graph()
    g.add_nodes_from(node_ids)
    if edges_df.empty:
        return g
    if apply_threshold_filter:
        if weight_col not in edges_df.columns:
            return g
        use = edges_df[pd.to_numeric(edges_df[weight_col], errors="coerce") >= float(min_edge_weight)].copy()
    else:
        use = edges_df.copy()
    for _, r in use.iterrows():
        a, b = str(r["email_a"]), str(r["email_b"])
        if a not in g or b not in g or a == b:
            continue
        w = float(r[weight_col]) if use_edge_weights_in_partitioning else 1.0
        if g.has_edge(a, b):
            # deterministic and conservative: keep max weight if duplicates appear.
            g[a][b]["weight"] = max(float(g[a][b]["weight"]), w)
        else:
            g.add_edge(a, b, weight=w)
    return g


def _run_leiden_partition(
    *,
    node_ids: list[str],
    edges_df: pd.DataFrame,
    weight_col: str,
    min_edge_weight: float,
    resolution: float,
    seed: int,
    use_edge_weights_in_partitioning: bool = True,
    apply_threshold_filter: bool = True,
) -> tuple[dict[str, int], dict[str, Any]]:
    import igraph as ig
    import leidenalg as la

    id_to_idx = {eid: i for i, eid in enumerate(node_ids)}
    if edges_df.empty:
        use = edges_df.iloc[0:0]
    else:
        if apply_threshold_filter:
            if weight_col not in edges_df.columns:
                use = edges_df.iloc[0:0]
            else:
                use = edges_df[pd.to_numeric(edges_df[weight_col], errors="coerce") >= float(min_edge_weight)]
        else:
            use = edges_df

    pair_w: dict[tuple[int, int], float] = {}
    for _, r in use.iterrows():
        a, b = str(r["email_a"]), str(r["email_b"])
        if a not in id_to_idx or b not in id_to_idx:
            continue
        i, j = id_to_idx[a], id_to_idx[b]
        if i == j:
            continue
        u, v = (i, j) if i < j else (j, i)
        w = float(r[weight_col]) if use_edge_weights_in_partitioning else 1.0
        pair_w[(u, v)] = max(pair_w.get((u, v), 0.0), w)

    edges_list = list(pair_w.keys())
    weights = [pair_w[e] for e in edges_list]
    g = ig.Graph(n=len(node_ids), edges=edges_list, directed=False)
    if weights:
        g.es["weight"] = weights

    part = la.find_partition(
        g,
        la.RBConfigurationVertexPartition,
        weights="weight" if weights else None,
        resolution_parameter=float(resolution),
        n_iterations=-1,
        seed=int(seed),
    )
    membership = list(part.membership)
    email_to_comm = {node_ids[i]: int(membership[i]) for i in range(len(node_ids))}
    info = {
        "method_requested": "leiden",
        "method_used": "leiden",
        "n_nodes": int(g.vcount()),
        "n_edges_after_threshold": int(g.ecount()),
        "n_communities": int(len(set(email_to_comm.values()))),
    }
    return email_to_comm, info


def run_weighted_email_community_detection(
    *,
    node_ids: list[str],
    edges_df: pd.DataFrame,
    method: str,
    resolution: float,
    min_edge_weight: float,
    weight_col: str = "edge_weight",
    seed: int = 0,
    use_edge_weights_in_partitioning: bool = True,
    apply_threshold_filter: bool = True,
) -> tuple[dict[str, int], dict[str, Any]]:
    m = str(method).strip().lower()
    wcol = str(weight_col)
    if m == "leiden":
        return _run_leiden_partition(
            node_ids=node_ids,
            edges_df=edges_df,
            weight_col=wcol,
            min_edge_weight=min_edge_weight,
            resolution=resolution,
            seed=seed,
            use_edge_weights_in_partitioning=use_edge_weights_in_partitioning,
            apply_threshold_filter=apply_threshold_filter,
        )

    g = _build_weighted_email_graph(
        node_ids=node_ids,
        edges_df=edges_df,
        weight_col=wcol,
        min_edge_weight=min_edge_weight,
        use_edge_weights_in_partitioning=use_edge_weights_in_partitioning,
        apply_threshold_filter=apply_threshold_filter,
    )
    if m != "louvain":
        raise ValueError(f"Unsupported method: {method!r} (expected: louvain, leiden)")
    used_method = "louvain"
    try:
        comms = nx.algorithms.community.louvain_communities(
            g,
            weight="weight",
            resolution=float(resolution),
            seed=int(seed),
        )
    except Exception:
        comms = nx.algorithms.community.greedy_modularity_communities(g, weight="weight")
        used_method = "greedy_fallback"

    email_to_comm: dict[str, int] = {}
    for cid, members in enumerate(comms):
        for eid in members:
            email_to_comm[str(eid)] = int(cid)
    next_id = max(email_to_comm.values(), default=-1) + 1
    for eid in node_ids:
        if eid not in email_to_comm:
            email_to_comm[eid] = int(next_id)
            next_id += 1

    info = {
        "method_requested": m,
        "method_used": used_method,
        "n_nodes": int(g.number_of_nodes()),
        "n_edges_after_threshold": int(g.number_of_edges()),
        "n_communities": int(len(set(email_to_comm.values()))),
    }
    return email_to_comm, info


def map_email_predictions(node_ids: list[str], email_to_comm: dict[str, int]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "external_id": [str(e) for e in node_ids],
            "pred_community": [int(email_to_comm.get(str(e), -1)) for e in node_ids],
        }
    )


def evaluate_external_metrics(
    *,
    pred_map: dict[str, int],
    gt_label_map: dict[str, Any],
) -> dict[str, float]:
    m = cec.evaluate_external_metrics(
        gt_label_map=gt_label_map,
        pred_label_map=pred_map,
        n_predictions_total=len(pred_map),
    )
    return {
        "n_eval": m["n_eval"],
        "homogeneity": m["homogeneity"],
        "completeness": m["completeness"],
        "v_measure": m["v_measure"],
        "coverage_gt": m["coverage_gt"],
    }


def _pred_map_from_assignment(node_ids: list[str], email_to_comm: dict[str, int]) -> dict[str, int]:
    return {str(eid): int(email_to_comm[str(eid)]) for eid in node_ids}


def _run_sweep_communities_once(
    *,
    node_ids: list[str],
    edges_df: pd.DataFrame,
    methods: list[str],
    weight_thresholds: list[float],
    resolutions: list[float],
    weight_col: str,
    seed: int,
    use_edge_weights_in_partitioning: bool = True,
    apply_threshold_filter: bool = True,
) -> list[dict[str, Any]]:
    """
    Run community detection once per (method, threshold, resolution).
    Returns rows with partition as pred_map (no ground truth).
    """
    combos: list[tuple[str, float, float]] = []
    for m in methods:
        for w in weight_thresholds:
            for r in resolutions:
                combos.append((str(m), float(w), float(r)))
    iterator = tqdm(combos, desc="sweep settings") if (tqdm is not None) else combos
    out: list[dict[str, Any]] = []
    for method, w, r in iterator:
        email_to_comm, info = run_weighted_email_community_detection(
            node_ids=node_ids,
            edges_df=edges_df,
            method=method,
            resolution=float(r),
            min_edge_weight=float(w),
            weight_col=weight_col,
            seed=seed,
            use_edge_weights_in_partitioning=use_edge_weights_in_partitioning,
            apply_threshold_filter=apply_threshold_filter,
        )
        pred_map = _pred_map_from_assignment(node_ids, email_to_comm)
        out.append(
            {
                "method": str(info["method_used"]),
                "weight_col": weight_col,
                "use_edge_weights_in_partitioning": bool(use_edge_weights_in_partitioning),
                "resolution": float(r),
                "min_edge_weight": float(w),
                "n_edges_after_threshold": float(info["n_edges_after_threshold"]),
                "n_communities": float(info["n_communities"]),
                "_pred_map": pred_map,
            }
        )
    return out


def _metrics_for_gt(
    sweep_partitions: list[dict[str, Any]],
    gt_label_map: dict[str, Any],
    *,
    sort_by: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for part in sweep_partitions:
        pred_map = part["_pred_map"]
        m = evaluate_external_metrics(pred_map=pred_map, gt_label_map=gt_label_map)
        rows.append(
            {
                "method": part["method"],
                "weight_col": part["weight_col"],
                "use_edge_weights_in_partitioning": bool(
                    part.get("use_edge_weights_in_partitioning", True)
                ),
                "resolution": part["resolution"],
                "min_edge_weight": part["min_edge_weight"],
                "n_edges_after_threshold": part["n_edges_after_threshold"],
                "n_communities": part["n_communities"],
                "n_eval": m["n_eval"],
                "coverage_gt": m["coverage_gt"],
                "homogeneity": m["homogeneity"],
                "completeness": m["completeness"],
                "v_measure": m["v_measure"],
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    d = df.copy()
    cols = _sort_tiebreakers(sort_by)
    for c in cols:
        d[f"_{c}"] = pd.to_numeric(d[c], errors="coerce")
    sort_keys = [f"_{c}" for c in cols]
    ascending = [False] * len(sort_keys)
    d = d.sort_values(sort_keys, ascending=ascending)
    d = d.drop(columns=sort_keys)
    d = d.reset_index(drop=True)
    return d


def _best_row(df: pd.DataFrame, metric: str = "v_measure") -> dict[str, Any]:
    r = cec.best_sweep_metric_row(df, metric=metric)
    return r.to_dict() if not r.empty else {}


def _validate_output_contract(df: pd.DataFrame, *, sort_by: str) -> None:
    if df.empty:
        return
    required = {
        "method",
        "resolution",
        "min_edge_weight",
        "n_edges_after_threshold",
        "n_communities",
        "n_eval",
        "coverage_gt",
        "homogeneity",
        "completeness",
        "v_measure",
    }
    missing = sorted(c for c in required if c not in df.columns)
    if missing:
        raise ValueError(f"Sweep output missing required columns: {missing}")
    cols = _sort_tiebreakers(sort_by)
    prev_key: tuple[float, ...] | None = None
    for i in range(len(df)):
        key = tuple(
            float(pd.to_numeric(df.iloc[i][c], errors="coerce"))
            for c in cols
        )
        # Descending: each row must be >= the next (best row first).
        if prev_key is not None and prev_key < key:
            raise AssertionError(
                f"Per-GT sweep CSV must be lexicographically descending by {cols}; "
                f"row {i - 1} vs {i}"
            )
        prev_key = key


def run_anchor_multi_gt_community_sweep(config: dict[str, Any]) -> dict[str, Any]:
    run_cfg = config.get("run") or {}
    sweep_cfg = config.get("sweep") or {}
    gt_cfg = config.get("ground_truth") or {}
    out_cfg = config.get("output") or {}

    project_root = gh.find_project_root()
    graph_id = resolve_graph_id(run_cfg)
    anchor_output_root_raw = str(run_cfg.get("anchor_output_root") or "").strip()
    if anchor_output_root_raw:
        anchor_output_root = Path(anchor_output_root_raw).expanduser().resolve()
    else:
        anchor_output_root = (
            project_root / "analysis" / "output" / "graph_bundles" / graph_id / "anchor"
        ).resolve()

    score_mode_raw = str(sweep_cfg.get("score_mode") or "").strip().lower()
    use_scoring = bool(score_mode_raw)
    use_pre_scored_edges = score_mode_raw in {"pre_scored", "weighted_pre_scored"}
    score_params = dict(sweep_cfg.get("score_params") or {})

    custom_edges_csv_raw = str(run_cfg.get("custom_edges_csv") or "").strip()
    custom_edges_resolved: str | None = None
    if custom_edges_csv_raw:
        p_edges = Path(custom_edges_csv_raw).expanduser()
        if not p_edges.is_absolute():
            p_edges = (project_root / p_edges).resolve()
        else:
            p_edges = p_edges.resolve()
        if not p_edges.is_file():
            raise FileNotFoundError(f"run.custom_edges_csv not found: {p_edges}")
        edges_df = pd.read_csv(p_edges, low_memory=False)
        req = {"email_i", "email_j"}
        miss = sorted(c for c in req if c not in edges_df.columns)
        if miss:
            raise ValueError(f"custom_edges_csv missing columns {miss}; required {sorted(req)}")
        run_dir = (anchor_output_root / graph_id).expanduser().resolve()
        if not run_dir.is_dir():
            raise FileNotFoundError(f"Anchor graph run directory not found: {run_dir}")
        p_nodes = run_dir / "anchor_graph_nodes.csv"
        if not p_nodes.is_file():
            raise FileNotFoundError(f"Missing anchor nodes CSV: {p_nodes}")
        nodes_df = pd.read_csv(p_nodes, low_memory=False)
        p_summary = run_dir / "anchor_graph_summary.json"
        anchor_summary = (
            json.loads(p_summary.read_text(encoding="utf-8")) if p_summary.is_file() else {}
        )
        nodes_df = nodes_df.copy()
        nodes_df["external_id"] = nodes_df["external_id"].astype(str)
        edges_df = edges_df.copy()
        edges_df["email_i"] = edges_df["email_i"].astype(str)
        edges_df["email_j"] = edges_df["email_j"].astype(str)
        custom_edges_resolved = str(p_edges)
    else:
        run_dir, nodes_df, edges_df, anchor_summary = _load_anchor_run(
            graph_id=graph_id,
            anchor_output_root=anchor_output_root,
        )
        # Anchor-run loaded edges are unscored by default after refactor.
        if "email_i" not in edges_df.columns or "email_j" not in edges_df.columns:
            if "email_a" in edges_df.columns and "email_b" in edges_df.columns:
                edges_df["email_i"] = edges_df["email_a"].astype(str)
                edges_df["email_j"] = edges_df["email_b"].astype(str)
            else:
                raise ValueError("Anchor edge table must contain email_i/email_j.")

    if use_scoring:
        if not use_pre_scored_edges and score_mode_raw not in SCORER_REGISTRY:
            raise ValueError(
                f"Unknown sweep.score_mode {score_mode_raw!r}. Available: {sorted(SCORER_REGISTRY)}"
            )
        if use_pre_scored_edges:
            if "edge_weight" not in edges_df.columns:
                raise ValueError(
                    "sweep.score_mode='pre_scored' requires custom edges with edge_weight."
                )
            scored_df = edges_df.copy()
        elif score_mode_raw == "seed_candidate_handcrafted_v1":
            seed_edges_csv_raw = str(run_cfg.get("seed_edges_csv") or "").strip()
            if seed_edges_csv_raw:
                p_seed = Path(seed_edges_csv_raw).expanduser()
                if not p_seed.is_absolute():
                    p_seed = (project_root / p_seed).resolve()
                else:
                    p_seed = p_seed.resolve()
            else:
                seed_stage_dir_override = str(run_cfg.get("seed_stage_dir") or "").strip()
                if seed_stage_dir_override:
                    seed_dir = Path(seed_stage_dir_override).expanduser()
                    if not seed_dir.is_absolute():
                        seed_dir = (project_root / seed_dir).resolve()
                    else:
                        seed_dir = seed_dir.resolve()
                else:
                    seed_output_root_raw = str(run_cfg.get("seed_output_root") or "").strip()
                    seed_output_root = (
                        Path(seed_output_root_raw).expanduser().resolve()
                        if seed_output_root_raw
                        else (project_root / "analysis" / "output" / "graph_bundles" / graph_id / "seed").resolve()
                    )
                    seed_prefix = str(run_cfg.get("seed_stage_name_prefix") or "seed_generation_")
                    seed_dir = _resolve_latest_seed_dir(
                        seed_output_root=seed_output_root,
                        graph_id=graph_id,
                        seed_stage_name_prefix=seed_prefix,
                    )
                p_seed = seed_dir / "seed_edges_all.csv"
            if not p_seed.is_file():
                raise FileNotFoundError(f"seed edges CSV not found for handcrafted scoring: {p_seed}")
            seed_edges_df = pd.read_csv(p_seed, low_memory=False)
            scored_df = SCORER_REGISTRY[score_mode_raw](
                candidate_union_df=edges_df,
                seed_edges_df=seed_edges_df,
                scoring_cfg=score_params,
            )
        elif score_mode_raw == "seed_candidate_pu_v1":
            scored_all_df, _scored_thr_df = SCORER_REGISTRY[score_mode_raw](
                candidate_union_df=edges_df,
                scoring_cfg=score_params,
                score_mode="seed_candidate_pu_v1",
            )
            scored_df = scored_all_df
        else:
            raise ValueError(f"Unsupported score_mode for community sweep: {score_mode_raw}")
        edges_df = scored_df.copy()
        edges_df["email_i"] = edges_df["email_i"].astype(str)
        edges_df["email_j"] = edges_df["email_j"].astype(str)
        edges_df["email_a"] = edges_df["email_i"]
        edges_df["email_b"] = edges_df["email_j"]
    else:
        # Unweighted Option A: topology-only community detection.
        edges_df = edges_df.copy()
        edges_df["email_i"] = edges_df["email_i"].astype(str)
        edges_df["email_j"] = edges_df["email_j"].astype(str)
        edges_df["email_a"] = edges_df["email_i"]
        edges_df["email_b"] = edges_df["email_j"]
        if "edge_weight" in edges_df.columns:
            edges_df = edges_df.drop(columns=["edge_weight"])
    node_ids = [str(x) for x in nodes_df["external_id"].astype(str).tolist()]

    methods = [str(x).strip().lower() for x in (sweep_cfg.get("methods") or ["louvain", "leiden"]) if str(x).strip()]
    weight_thresholds = [float(x) for x in (sweep_cfg.get("weight_thresholds") or [0.0])]
    resolutions = [float(x) for x in (sweep_cfg.get("resolutions") or [1.0])]
    weight_col = str(sweep_cfg.get("weight_col") or "edge_weight")
    seed = int(sweep_cfg.get("seed", 0))
    sort_by = _normalize_sort_metric(sweep_cfg.get("sort_by"))
    use_edge_weights_in_partitioning = bool(sweep_cfg.get("use_edge_weights_in_partitioning", True)) if use_scoring else False
    apply_threshold_filter = bool(use_scoring)
    if not use_scoring and weight_thresholds != [0.0]:
        weight_thresholds = [0.0]

    gt_paths_raw = gt_cfg.get("paths") or []
    if not isinstance(gt_paths_raw, list) or not gt_paths_raw:
        raise ValueError("ground_truth.paths must be a non-empty list")
    gt_paths: list[Path] = []
    for raw in gt_paths_raw:
        p = Path(str(raw)).expanduser()
        if not p.is_absolute():
            p = project_root / p
        p = p.resolve()
        if not p.is_file():
            raise FileNotFoundError(f"Ground truth file not found: {p}")
        gt_paths.append(p)

    out_root = Path(
        out_cfg.get("output_root")
        or (project_root / "analysis" / "output" / "anchor_community")
    ).expanduser().resolve()
    stage_name = str(out_cfg.get("stage_name") or "community_sweep")
    created_at_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    bundle_out_raw = str(run_cfg.get("community_bundle_out_dir") or "").strip()
    if bundle_out_raw:
        out_dir = Path(bundle_out_raw).expanduser()
        if not out_dir.is_absolute():
            out_dir = (project_root / out_dir).resolve()
        else:
            out_dir = out_dir.resolve()
    else:
        community_parent = str(run_cfg.get("community_output_parent_dir") or "").strip()
        if community_parent:
            parent = Path(community_parent).expanduser()
            if not parent.is_absolute():
                parent = (project_root / parent).resolve()
            else:
                parent = parent.resolve()
            out_dir = parent / f"{stage_name}_{stamp}"
        else:
            out_dir = out_root / graph_id / f"{stage_name}_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    sweep_partitions = _run_sweep_communities_once(
        node_ids=node_ids,
        edges_df=edges_df,
        methods=methods,
        weight_thresholds=weight_thresholds,
        resolutions=resolutions,
        weight_col=weight_col,
        seed=seed,
        use_edge_weights_in_partitioning=use_edge_weights_in_partitioning,
        apply_threshold_filter=apply_threshold_filter,
    )
    n_graph_nodes = len(node_ids)

    def _per_gt(gt_path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
        gt_label_map, _eid_row, _campaign_to_members = rn.load_ground_truth_structures(gt_path)
        sweep_df = _metrics_for_gt(sweep_partitions, gt_label_map, sort_by=sort_by)
        best = _best_row(sweep_df, metric=sort_by)
        return sweep_df, {"gt_label_map": gt_label_map, "best_row": best}

    def _write_gt(gt_path_raw: str, sweep_df: pd.DataFrame, best_info: dict[str, Any]) -> dict[str, Any]:
        gt_path = Path(gt_path_raw)
        gt_label_map = dict(best_info.get("gt_label_map") or {})
        best = dict(best_info.get("best_row") or {})
        gt_slug = _gt_slug(gt_path)
        if not sweep_df.empty:
            _validate_output_contract(sweep_df, sort_by=sort_by)
        p_csv = out_dir / f"anchor_community_sweep__{gt_slug}.csv"
        sweep_df.to_csv(p_csv, index=False)
        gt_ids = {str(k) for k in gt_label_map.keys()}
        graph_ids = {str(x) for x in node_ids}
        n_intersection = len(gt_ids & graph_ids)
        best_payload = {
            "graph_id": graph_id,
            "anchor_run_dir": str(run_dir),
            "gt_path": str(gt_path),
            "gt_slug": gt_slug,
            "created_at_utc": created_at_utc,
            "n_graph_nodes": n_graph_nodes,
            "n_gt_labeled_emails": len(gt_label_map),
            "n_gt_ids_in_graph": n_intersection,
            "labeled_in_graph_fraction": float(n_intersection / max(1, n_graph_nodes)),
            "sort_by": sort_by,
            "best_row": best,
        }
        p_best = out_dir / f"anchor_community_best__{gt_slug}.json"
        p_best.write_text(json.dumps(best_payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return {
            "gt_path": str(gt_path),
            "gt_slug": gt_slug,
            "sweep_csv": str(p_csv),
            "best_json": str(p_best),
            "n_rows": int(len(sweep_df)),
            "best_row": best,
        }

    per_gt_outputs, best_rows = csd.run_multi_gt_sweep(
        gt_paths=[str(p) for p in gt_paths],
        per_gt_sweep=lambda p: _per_gt(Path(p)),
        write_per_gt=_write_gt,
    )

    summary = {
        "created_at_utc": created_at_utc,
        "graph_id": graph_id,
        "anchor_run_dir": str(run_dir),
        "custom_edges_csv": custom_edges_resolved,
        "score_mode": (score_mode_raw or None),
        "use_edge_weights_in_partitioning": use_edge_weights_in_partitioning,
        "apply_threshold_filter": apply_threshold_filter,
        "n_graph_nodes": n_graph_nodes,
        "anchor_summary_json": str((run_dir / "anchor_graph_summary.json").resolve()),
        "anchor_run_config_json": str((run_dir / "anchor_graph_run_config.json").resolve()),
        "anchor_summary": anchor_summary,
        "methods": methods,
        "weight_thresholds": weight_thresholds,
        "weight_threshold_behavior": (
            "active_threshold_filtering" if apply_threshold_filter else "disabled_in_unweighted_mode_option_a"
        ),
        "resolutions": resolutions,
        "weight_col": weight_col,
        "seed": seed,
        "sort_by": sort_by,
        "n_sweep_settings": len(sweep_partitions),
        "ground_truth_paths": [str(p) for p in gt_paths],
        "per_ground_truth_outputs": per_gt_outputs,
        "best_rows_by_gt": best_rows,
    }
    p_summary = out_dir / "anchor_community_multi_gt_summary.json"
    p_summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return {
        "output_dir": str(out_dir),
        "summary_json": str(p_summary),
        "per_ground_truth_outputs": per_gt_outputs,
    }

