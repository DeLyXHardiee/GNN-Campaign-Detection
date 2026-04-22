from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.metrics import completeness_score, homogeneity_score, v_measure_score

from analysis.utils import graph_structure_helpers as gh
from analysis.utils import raw_gnn_notebook as rn
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
    all_m = ("homogeneity", "completeness", "v_measure")
    rest = [m for m in all_m if m != primary]
    if primary == "homogeneity":
        order_rest = ["v_measure", "completeness"]
    elif primary == "completeness":
        order_rest = ["v_measure", "homogeneity"]
    else:
        order_rest = ["homogeneity", "completeness"]
    # Keep only metrics that are in rest (always two)
    return [primary] + [m for m in order_rest if m in rest]


def _load_anchor_run(
    *,
    graph_run_id: str,
    anchor_output_root: Path,
) -> tuple[Path, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    run_dir = (anchor_output_root / graph_run_id).expanduser().resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Anchor graph run directory not found: {run_dir}")
    nodes_df, edges_df, _cand, summary, _g = load_anchor_graph_artifacts(
        run_dir, load_graph_pickle=False
    )
    req_node_cols = {"external_id"}
    req_edge_cols = {"email_a", "email_b", "edge_weight"}
    miss_n = sorted(c for c in req_node_cols if c not in nodes_df.columns)
    miss_e = sorted(c for c in req_edge_cols if c not in edges_df.columns)
    if miss_n:
        raise ValueError(f"Anchor nodes missing required columns: {miss_n}")
    if miss_e:
        raise ValueError(f"Anchor edges missing required columns: {miss_e}")
    nodes_df = nodes_df.copy()
    edges_df = edges_df.copy()
    nodes_df["external_id"] = nodes_df["external_id"].astype(str)
    edges_df["email_a"] = edges_df["email_a"].astype(str)
    edges_df["email_b"] = edges_df["email_b"].astype(str)
    return run_dir, nodes_df, edges_df, summary


def _build_weighted_email_graph(
    *,
    node_ids: list[str],
    edges_df: pd.DataFrame,
    weight_col: str,
    min_edge_weight: float,
    use_edge_weights_in_partitioning: bool = True,
) -> nx.Graph:
    g = nx.Graph()
    g.add_nodes_from(node_ids)
    if edges_df.empty or weight_col not in edges_df.columns:
        return g
    use = edges_df[pd.to_numeric(edges_df[weight_col], errors="coerce") >= float(min_edge_weight)].copy()
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
) -> tuple[dict[str, int], dict[str, Any]]:
    import igraph as ig
    import leidenalg as la

    id_to_idx = {eid: i for i, eid in enumerate(node_ids)}
    if edges_df.empty or weight_col not in edges_df.columns:
        use = edges_df.iloc[0:0]
    else:
        use = edges_df[pd.to_numeric(edges_df[weight_col], errors="coerce") >= float(min_edge_weight)]

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
        )

    g = _build_weighted_email_graph(
        node_ids=node_ids,
        edges_df=edges_df,
        weight_col=wcol,
        min_edge_weight=min_edge_weight,
        use_edge_weights_in_partitioning=use_edge_weights_in_partitioning,
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
    gt = {str(k): v for k, v in gt_label_map.items()}
    pred = {str(k): int(v) for k, v in pred_map.items()}
    common = sorted(set(gt.keys()) & set(pred.keys()))
    if not common:
        return {
            "n_eval": 0.0,
            "homogeneity": float("nan"),
            "completeness": float("nan"),
            "v_measure": float("nan"),
            "coverage_gt": 0.0,
        }
    y_true = [gt[e] for e in common]
    y_pred = [pred[e] for e in common]
    return {
        "n_eval": float(len(common)),
        "homogeneity": float(homogeneity_score(y_true, y_pred)),
        "completeness": float(completeness_score(y_true, y_pred)),
        "v_measure": float(v_measure_score(y_true, y_pred)),
        "coverage_gt": float(len(common) / max(1, len(gt))),
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
    if df.empty:
        return {}
    d = df.copy()
    d["_m"] = pd.to_numeric(d[metric], errors="coerce")
    d = d[np.isfinite(d["_m"])]
    if d.empty:
        return {}
    return d.sort_values("_m", ascending=False).iloc[0].drop(labels=["_m"], errors="ignore").to_dict()


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
    graph_run_id = str(run_cfg.get("graph_run_id") or "").strip()
    if not graph_run_id:
        raise ValueError("run.graph_run_id is required")
    anchor_output_root = Path(
        run_cfg.get("anchor_output_root")
        or (project_root / "analysis" / "output" / "anchor_graph")
    ).expanduser().resolve()

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
        req = {"email_a", "email_b", "edge_weight"}
        miss = sorted(c for c in req if c not in edges_df.columns)
        if miss:
            raise ValueError(f"custom_edges_csv missing columns {miss}; required {sorted(req)}")
        run_dir = (anchor_output_root / graph_run_id).expanduser().resolve()
        if not run_dir.is_dir():
            raise FileNotFoundError(f"Anchor graph run directory not found: {run_dir}")
        nodes_df, _edges_anchor, _cand, anchor_summary, _g = load_anchor_graph_artifacts(
            run_dir, load_graph_pickle=False
        )
        nodes_df = nodes_df.copy()
        nodes_df["external_id"] = nodes_df["external_id"].astype(str)
        edges_df = edges_df.copy()
        edges_df["email_a"] = edges_df["email_a"].astype(str)
        edges_df["email_b"] = edges_df["email_b"].astype(str)
        custom_edges_resolved = str(p_edges)
    else:
        run_dir, nodes_df, edges_df, anchor_summary = _load_anchor_run(
            graph_run_id=graph_run_id,
            anchor_output_root=anchor_output_root,
        )
    node_ids = [str(x) for x in nodes_df["external_id"].astype(str).tolist()]

    methods = [str(x).strip().lower() for x in (sweep_cfg.get("methods") or ["louvain", "leiden"]) if str(x).strip()]
    weight_thresholds = [float(x) for x in (sweep_cfg.get("weight_thresholds") or [0.0])]
    resolutions = [float(x) for x in (sweep_cfg.get("resolutions") or [1.0])]
    weight_col = str(sweep_cfg.get("weight_col") or "edge_weight")
    seed = int(sweep_cfg.get("seed", 0))
    sort_by = _normalize_sort_metric(sweep_cfg.get("sort_by"))
    use_edge_weights_in_partitioning = bool(sweep_cfg.get("use_edge_weights_in_partitioning", True))

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
            out_dir = out_root / graph_run_id / f"{stage_name}_{stamp}"
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
    )
    n_graph_nodes = len(node_ids)

    per_gt_outputs: list[dict[str, Any]] = []
    best_rows: list[dict[str, Any]] = []
    for gt_path in gt_paths:
        gt_label_map, _eid_row, _campaign_to_members = rn.load_ground_truth_structures(gt_path)
        sweep_df = _metrics_for_gt(sweep_partitions, gt_label_map, sort_by=sort_by)
        gt_slug = _gt_slug(gt_path)
        if not sweep_df.empty:
            _validate_output_contract(sweep_df, sort_by=sort_by)
        p_csv = out_dir / f"anchor_community_sweep__{gt_slug}.csv"
        sweep_df.to_csv(p_csv, index=False)

        best = _best_row(sweep_df, metric=sort_by)
        gt_ids = {str(k) for k in gt_label_map.keys()}
        graph_ids = {str(x) for x in node_ids}
        n_intersection = len(gt_ids & graph_ids)
        best_payload = {
            "graph_run_id": graph_run_id,
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
        per_gt_outputs.append(
            {
                "gt_path": str(gt_path),
                "gt_slug": gt_slug,
                "sweep_csv": str(p_csv),
                "best_json": str(p_best),
                "n_rows": int(len(sweep_df)),
            }
        )
        if best:
            best_rows.append({"gt_path": str(gt_path), "gt_slug": gt_slug, **best})

    summary = {
        "created_at_utc": created_at_utc,
        "graph_run_id": graph_run_id,
        "anchor_run_dir": str(run_dir),
        "custom_edges_csv": custom_edges_resolved,
        "use_edge_weights_in_partitioning": use_edge_weights_in_partitioning,
        "n_graph_nodes": n_graph_nodes,
        "anchor_summary_json": str((run_dir / "anchor_graph_summary.json").resolve()),
        "anchor_run_config_json": str((run_dir / "anchor_graph_run_config.json").resolve()),
        "anchor_summary": anchor_summary,
        "methods": methods,
        "weight_thresholds": weight_thresholds,
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

