from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from analysis.utils import completeness_fragmentation_helpers as cfh
from analysis.utils import community_eval_contract as cec
from analysis.utils import graph_structure_helpers as gh
from analysis.utils import raw_gnn_notebook as rn


def load_step2_artifacts(
    step2_dir: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    d = Path(step2_dir).expanduser().resolve()
    nodes = pd.read_csv(d / "semantic_shard_step2_nodes.csv")
    edges = pd.read_csv(d / "semantic_shard_step2_edges_weighted.csv")
    candidates = pd.read_csv(d / "semantic_shard_step2_candidates.csv")
    nodes["shard_id"] = nodes["shard_id"].astype(str)
    edges["shard_a"] = edges["shard_a"].astype(str)
    edges["shard_b"] = edges["shard_b"].astype(str)
    return nodes, edges, candidates


def _build_nx_graph(
    shard_ids: list[str],
    edges_df: pd.DataFrame,
    *,
    weight_col: str = "edge_weight",
    min_edge_weight: float = 0.0,
):
    import networkx as nx

    g = nx.Graph()
    g.add_nodes_from(shard_ids)
    if edges_df.empty or weight_col not in edges_df.columns:
        return g
    wcol = str(weight_col)
    use = edges_df[edges_df[wcol] >= float(min_edge_weight)].copy()
    for _, r in use.iterrows():
        g.add_edge(
            str(r["shard_a"]),
            str(r["shard_b"]),
            weight=float(r[wcol]),
        )
    return g


def _run_leiden_partition(
    shard_ids: list[str],
    edges_df: pd.DataFrame,
    *,
    weight_col: str,
    min_edge_weight: float,
    resolution: float,
    seed: int,
) -> tuple[dict[str, int], dict[str, Any]]:
    import igraph as ig
    import leidenalg as la

    id_to_idx = {str(sid): i for i, sid in enumerate(shard_ids)}
    wcol = str(weight_col)
    if edges_df.empty or wcol not in edges_df.columns:
        use = edges_df.iloc[0:0]
    else:
        use = edges_df[edges_df[wcol] >= float(min_edge_weight)]

    pair_w: dict[tuple[int, int], float] = {}
    for _, r in use.iterrows():
        a, b = str(r["shard_a"]), str(r["shard_b"])
        if a not in id_to_idx or b not in id_to_idx:
            continue
        i, j = id_to_idx[a], id_to_idx[b]
        if i == j:
            continue
        u, v = (i, j) if i < j else (j, i)
        w = float(r[wcol])
        pair_w[(u, v)] = max(pair_w.get((u, v), 0.0), w)

    edges_list = list(pair_w.keys())
    weights = [pair_w[e] for e in edges_list]
    g = ig.Graph(n=len(shard_ids), edges=edges_list, directed=False)
    if weights:
        g.es["weight"] = weights

    partition = la.find_partition(
        g,
        la.RBConfigurationVertexPartition,
        weights="weight" if weights else None,
        resolution_parameter=float(resolution),
        n_iterations=-1,
        seed=int(seed),
    )
    membership = list(partition.membership)
    shard_to_comm = {str(shard_ids[i]): int(membership[i]) for i in range(len(shard_ids))}

    info = {
        "method_requested": "leiden",
        "method_used": "leiden",
        "weight_col": wcol,
        "n_nodes": int(g.vcount()),
        "n_edges_after_threshold": int(g.ecount()),
        "n_communities": int(len(set(shard_to_comm.values()))),
    }
    return shard_to_comm, info


def run_weighted_community_detection(
    shard_ids: list[str],
    edges_df: pd.DataFrame,
    *,
    method: str,
    resolution: float,
    min_edge_weight: float,
    weight_col: str = "edge_weight",
    seed: int = 0,
) -> tuple[dict[str, int], dict[str, Any]]:
    import networkx as nx

    m = str(method).lower().strip()
    wcol = str(weight_col)

    if m == "leiden":
        shard_to_comm, info = _run_leiden_partition(
            shard_ids,
            edges_df,
            weight_col=wcol,
            min_edge_weight=min_edge_weight,
            resolution=resolution,
            seed=seed,
        )
        return shard_to_comm, info

    g = _build_nx_graph(
        shard_ids, edges_df, weight_col=wcol, min_edge_weight=min_edge_weight
    )

    if m not in {"louvain", "greedy"}:
        raise ValueError(
            f"Unsupported community detection method: {method!r} "
            f"(expected one of: louvain, greedy, leiden)"
        )

    used_method = m
    communities = None
    if m == "louvain":
        try:
            communities = nx.algorithms.community.louvain_communities(
                g,
                weight="weight",
                resolution=float(resolution),
                seed=int(seed),
            )
        except Exception:
            communities = nx.algorithms.community.greedy_modularity_communities(
                g,
                weight="weight",
            )
            used_method = "greedy_fallback"
    else:
        communities = nx.algorithms.community.greedy_modularity_communities(
            g,
            weight="weight",
        )

    shard_to_comm: dict[str, int] = {}
    for cid, members in enumerate(communities):
        for sid in members:
            shard_to_comm[str(sid)] = int(cid)
    next_id = max(shard_to_comm.values(), default=-1) + 1
    for sid in shard_ids:
        if sid not in shard_to_comm:
            shard_to_comm[sid] = int(next_id)
            next_id += 1

    info = {
        "method_requested": m,
        "method_used": used_method,
        "weight_col": wcol,
        "n_nodes": int(g.number_of_nodes()),
        "n_edges_after_threshold": int(g.number_of_edges()),
        "n_communities": int(len(set(shard_to_comm.values()))),
    }
    return shard_to_comm, info


def map_shards_to_email_predictions(
    assignments_df: pd.DataFrame,
    shard_to_comm: dict[str, int],
) -> pd.DataFrame:
    out = assignments_df.copy()
    out["external_id"] = out["external_id"].astype(str)
    out["shard_id"] = out["shard_id"].astype(str)
    out["pred_community"] = out["shard_id"].map(lambda s: int(shard_to_comm.get(s, -1)))
    return out


def evaluate_external_metrics(
    email_pred_df: pd.DataFrame,
    gt_label_map: dict[str, Any],
) -> dict[str, float]:
    pred = {
        str(r["external_id"]): int(r["pred_community"])
        for _, r in email_pred_df.iterrows()
    }
    m = cec.evaluate_external_metrics(
        gt_label_map=gt_label_map,
        pred_label_map=pred,
        n_predictions_total=len(email_pred_df),
    )
    return {
        **m,
        "coverage_assignments": m["coverage_predictions"],
    }


def run_community_sweep(
    *,
    assignments_df: pd.DataFrame,
    shard_ids: list[str],
    edges_df: pd.DataFrame,
    gt_label_map: dict[str, Any],
    method: str,
    resolution_values: list[float],
    min_edge_weight_values: list[float],
    weight_col: str = "edge_weight",
    seed: int = 0,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    email_preds_by_key: dict[str, pd.DataFrame] = {}
    info_by_key: dict[str, dict[str, Any]] = {}
    wcol = str(weight_col)
    key_suffix = "" if wcol == "edge_weight" else f"_{wcol}"
    for w in min_edge_weight_values:
        for r in resolution_values:
            key = f"w{w:.3f}_r{r:.3f}{key_suffix}"
            shard_to_comm, info = run_weighted_community_detection(
                shard_ids=shard_ids,
                edges_df=edges_df,
                method=method,
                resolution=float(r),
                min_edge_weight=float(w),
                weight_col=wcol,
                seed=int(seed),
            )
            email_pred_df = map_shards_to_email_predictions(assignments_df, shard_to_comm)
            m = evaluate_external_metrics(email_pred_df, gt_label_map)
            row = {
                "setting_key": key,
                "method_requested": method,
                "method_used": info["method_used"],
                "weight_col": wcol,
                "resolution": float(r),
                "min_edge_weight": float(w),
                "n_edges_after_threshold": float(info["n_edges_after_threshold"]),
                "n_communities": float(info["n_communities"]),
                "n_eval": m["n_eval"],
                "coverage_gt": m["coverage_gt"],
                "homogeneity": m["homogeneity"],
                "completeness": m["completeness"],
                "v_measure": m["v_measure"],
            }
            rows.append(row)
            email_preds_by_key[key] = email_pred_df
            info_by_key[key] = info
    return pd.DataFrame(rows), email_preds_by_key, info_by_key


def best_sweep_metric_row(sweep_df: pd.DataFrame, metric: str = "v_measure") -> pd.Series:
    """Return the sweep row with highest finite ``metric`` (default V-measure)."""
    return cec.best_sweep_metric_row(sweep_df=sweep_df, metric=metric)


def load_method1_refined_edges(method1_dir: str | Path) -> pd.DataFrame:
    """Load Method 1 refined edge list written by ``save_method1_edge_refinement_artifacts``."""
    d = Path(method1_dir).expanduser().resolve()
    p = d / "semantic_shard_step2_edges_refined.csv"
    if not p.is_file():
        raise FileNotFoundError(f"Missing Method 1 refined edges: {p}")
    df = pd.read_csv(p)
    df["shard_a"] = df["shard_a"].astype(str)
    df["shard_b"] = df["shard_b"].astype(str)
    return df


def load_baseline_rows(
    *,
    raw_sweep_csv: str | Path,
    transformer_sweep_csv: str | Path,
) -> pd.DataFrame:
    def _best_row(p: str | Path, name: str) -> dict[str, Any]:
        d = pd.read_csv(Path(p).expanduser().resolve())
        if d.empty:
            return {"baseline": name}
        r = d.sort_values("v_measure", ascending=False).iloc[0].to_dict()
        return {
            "baseline": name,
            "homogeneity": float(r.get("homogeneity", np.nan)),
            "completeness": float(r.get("completeness", np.nan)),
            "v_measure": float(r.get("v_measure", np.nan)),
            "n_clusters": float(r.get("n_clusters", np.nan)),
            "coverage_ground_truth": float(r.get("coverage_ground_truth", np.nan)),
            "model": r.get("model"),
            "embedding_mode": r.get("embedding_mode"),
        }

    return pd.DataFrame(
        [
            _best_row(raw_sweep_csv, "RAW_graph_email_x"),
            _best_row(transformer_sweep_csv, "Transformer_subject_body"),
        ]
    )


def rebuild_raw_prediction_map(
    *,
    graph_pt: str | Path,
    meta_json: str | Path,
    to_undirected: bool,
    gt_label_map: dict[str, Any],
    raw_sweep_csv: str | Path,
) -> tuple[dict[str, int], dict[str, Any]]:
    meta = gh.load_meta(meta_json)
    data = gh.load_hetero(graph_pt, to_undirected=to_undirected)
    ext = gh.email_external_id_list(meta)
    # Build RAW embedding map directly (avoid dependency on src.* import paths in notebooks).
    if "email" not in data.node_types:
        raise ValueError("Node type 'email' not found in graph.")
    email_x = data["email"].x
    if email_x is None:
        raise ValueError("data['email'].x is missing; cannot rebuild RAW baseline map.")
    email_vecs = email_x.detach().cpu().numpy()
    if len(ext) != len(email_vecs):
        raise ValueError(
            f"external_id length ({len(ext)}) != number of email rows ({len(email_vecs)})."
        )
    id_to_emb = {
        str(eid): email_vecs[i].copy()
        for i, eid in enumerate(ext)
    }
    mcs, ms = cfh.load_raw_hdbscan_params(Path(raw_sweep_csv))
    try:
        sorted_ids, labels, pred_map, metrics = cfh.build_raw_predictions(
            id_to_emb,
            gt_label_map,
            min_cluster_size=int(mcs),
            min_samples=ms,
        )
        _ = sorted_ids, labels
        return pred_map, metrics
    except ModuleNotFoundError as e:
        # Notebook environments sometimes lack optional clustering deps like `hdbscan`.
        # Fallback: use KMeans over L2-normalized vectors (cosine-ish in practice).
        if "hdbscan" not in str(e).lower():
            raise

        # Sort to make label assignment stable across runs.
        sorted_ids = sorted(id_to_emb.keys())
        X = np.stack([id_to_emb[eid] for eid in sorted_ids]).astype(np.float32)
        # L2-normalize for cosine-ish similarity with Euclidean KMeans.
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        Xn = X / norms

        # Heuristic: aim for a moderate number of clusters.
        N = Xn.shape[0]
        k = int(np.clip(round(np.sqrt(N)), 2, 50))

        from sklearn.cluster import KMeans

        km = KMeans(n_clusters=k, n_init="auto", random_state=0)
        labels = km.fit_predict(Xn)
        pred_map = rn.eid_label_map(sorted_ids, labels)
        metrics = rn.external_scores_subset(sorted_ids, labels, gt_label_map)
        return pred_map, metrics


def proto_prediction_map_from_email_df(email_pred_df: pd.DataFrame) -> dict[str, int]:
    return {
        str(r["external_id"]): int(r["pred_community"])
        for _, r in email_pred_df.iterrows()
    }


def fragmentation_compare_table(
    *,
    campaign_to_members: dict[Any, list[str]],
    raw_pred_map: dict[str, int],
    proto_pred_map: dict[str, int],
) -> pd.DataFrame:
    fr = cfh.campaign_fragmentation_df(campaign_to_members, raw_pred_map).rename(
        columns={
            "num_pred_clusters": "num_pred_clusters_raw",
            "dominant_fraction": "dominant_fraction_raw",
            "fragmentation_score": "fragmentation_score_raw",
        }
    )
    fp = cfh.campaign_fragmentation_df(campaign_to_members, proto_pred_map).rename(
        columns={
            "num_pred_clusters": "num_pred_clusters_proto",
            "dominant_fraction": "dominant_fraction_proto",
            "fragmentation_score": "fragmentation_score_proto",
        }
    )
    keep_r = [
        "campaign_id",
        "campaign_size",
        "num_pred_clusters_raw",
        "dominant_fraction_raw",
        "fragmentation_score_raw",
    ]
    keep_p = [
        "campaign_id",
        "num_pred_clusters_proto",
        "dominant_fraction_proto",
        "fragmentation_score_proto",
    ]
    m = fr[keep_r].merge(fp[keep_p], on="campaign_id", how="outer")
    m["delta_completeness_proxy"] = m["dominant_fraction_proto"] - m["dominant_fraction_raw"]
    m["delta_fragmentation"] = m["fragmentation_score_proto"] - m["fragmentation_score_raw"]
    m["delta_num_pred_clusters"] = m["num_pred_clusters_proto"] - m["num_pred_clusters_raw"]
    return m.sort_values("delta_completeness_proxy", ascending=False).reset_index(drop=True)


def split_campaign_merge_outcomes(
    frag_cmp_df: pd.DataFrame,
) -> pd.DataFrame:
    # Only campaigns split under RAW baseline.
    d = frag_cmp_df[frag_cmp_df["num_pred_clusters_raw"] > 1].copy()
    if d.empty:
        return pd.DataFrame(columns=["outcome", "n_campaigns", "fraction"])
    outcome = []
    for _, r in d.iterrows():
        if float(r["delta_completeness_proxy"]) > 1e-9 and float(r["delta_num_pred_clusters"]) < 0:
            outcome.append("improved_merge")
        elif float(r["delta_completeness_proxy"]) < -1e-9 and float(r["delta_num_pred_clusters"]) > 0:
            outcome.append("worse_split")
        elif float(r["delta_completeness_proxy"]) > 1e-9:
            outcome.append("improved_other")
        elif float(r["delta_completeness_proxy"]) < -1e-9:
            outcome.append("worse_other")
        else:
            outcome.append("no_change")
    d["merge_outcome"] = outcome
    vc = d["merge_outcome"].value_counts(dropna=False)
    out = vc.rename_axis("outcome").reset_index(name="n_campaigns")
    out["fraction"] = out["n_campaigns"] / max(1, len(d))
    return out


def campaign_drilldown_tables(
    *,
    campaign_id: Any,
    campaign_to_members: dict[Any, list[str]],
    assignments_df: pd.DataFrame,
    raw_pred_map: dict[str, int],
    proto_pred_map: dict[str, int],
    edges_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    members = [str(x) for x in campaign_to_members.get(campaign_id, [])]
    adf = assignments_df.copy()
    adf["external_id"] = adf["external_id"].astype(str)
    sub = adf[adf["external_id"].isin(set(members))].copy()
    sub["raw_pred"] = sub["external_id"].map(raw_pred_map)
    sub["proto_pred"] = sub["external_id"].map(proto_pred_map)

    # For the involved shard pairs, show edge evidence if present.
    involved = sorted(set(sub["shard_id"].astype(str).tolist()))
    e = edges_df.copy()
    e["shard_a"] = e["shard_a"].astype(str)
    e["shard_b"] = e["shard_b"].astype(str)
    pair_df = e[
        e["shard_a"].isin(set(involved)) & e["shard_b"].isin(set(involved))
    ].copy()
    cols = [
        "shard_a",
        "shard_b",
        "edge_weight",
        "edge_weight_refined",
        "edge_trust",
        "centroid_cosine",
        "infra_score",
        "temporal_score",
        "shared_url_count",
        "shared_sender_email_domain_count",
        "shared_domain_count",
        "shared_stem_count",
        "shared_sender_count",
    ]
    keep = [c for c in cols if c in pair_df.columns]
    sort_col = sort_edge_column if sort_edge_column in pair_df.columns else "edge_weight"
    pair_df = pair_df[keep].sort_values(sort_col, ascending=False) if not pair_df.empty else pair_df
    return sub, pair_df


def save_step3_outputs(
    *,
    output_dir: str | Path,
    sweep_df: pd.DataFrame,
    best_setting_row: dict[str, Any],
    best_email_pred_df: pd.DataFrame,
    best_shard_comm_df: pd.DataFrame,
) -> dict[str, str]:
    out = Path(output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    p_sweep = out / "semantic_shard_step3_sweep_results.csv"
    p_best_json = out / "semantic_shard_step3_best_setting.json"
    p_email = out / "semantic_shard_step3_email_predictions_best.csv"
    p_shard = out / "semantic_shard_step3_shard_communities_best.csv"
    sweep_df.to_csv(p_sweep, index=False)
    p_best_json.write_text(json.dumps(best_setting_row, indent=2), encoding="utf-8")
    best_email_pred_df.to_csv(p_email, index=False)
    best_shard_comm_df.to_csv(p_shard, index=False)
    return {
        "sweep_csv": str(p_sweep),
        "best_setting_json": str(p_best_json),
        "best_email_predictions_csv": str(p_email),
        "best_shard_communities_csv": str(p_shard),
    }
