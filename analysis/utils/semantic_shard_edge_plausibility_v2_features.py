"""
Build tabular edge features for Method 1 V2 plausibility MLP.

Excludes ``edge_weight`` and GT/evaluation node fields. Local graph stats use unweighted adjacency only.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd

from analysis.utils.semantic_shard_edge_refinement_method1 import build_method1_edge_feature_frame

# Never use these node fields for training (evaluation / leakage).
NODE_GT_BLOCKLIST = frozenset(
    {
        "n_members_with_gt",
        "n_gt_campaigns_touched",
        "dominant_campaign",
        "dominant_campaign_fraction",
    }
)

# Optional noisy / huge text columns on nodes — skip for tabular MLP.
NODE_SKIP_COLUMNS = frozenset(
    {
        "shard_id",
        "member_external_ids",
        "sender_set",
        "sender_email_domain_set",
        "url_set",
        "domain_set",
        "stem_set",
        "attachment_set",
    }
)


def _numeric_node_columns(nodes_df: pd.DataFrame) -> list[str]:
    cols: list[str] = []
    for c in nodes_df.columns:
        if c in NODE_SKIP_COLUMNS or c in NODE_GT_BLOCKLIST:
            continue
        if pd.api.types.is_numeric_dtype(nodes_df[c]):
            cols.append(c)
    return cols


def compute_unweighted_local_graph_features(edges_df: pd.DataFrame) -> dict[str, np.ndarray]:
    """
    Local graph statistics from **unweighted** adjacency (Step 2 edge list only).

    For each edge (a,b):
    - common neighbor count (excluding a,b)
    - Jaccard similarity of neighbor sets (excluding endpoints)
    - embeddedness: |N(a)∩N(b)| / min(deg(a),deg(b)) with deg = unweighted neighbor count
    """
    edges_df = edges_df.copy()
    edges_df["shard_a"] = edges_df["shard_a"].astype(str)
    edges_df["shard_b"] = edges_df["shard_b"].astype(str)
    adj: dict[str, set[str]] = defaultdict(set)
    for _, r in edges_df.iterrows():
        a, b = str(r["shard_a"]), str(r["shard_b"])
        if a == b:
            continue
        adj[a].add(b)
        adj[b].add(a)

    cn_list: list[int] = []
    jac_list: list[float] = []
    emb_list: list[float] = []

    for _, r in edges_df.iterrows():
        a, b = str(r["shard_a"]), str(r["shard_b"])
        na = adj.get(a, set()) - {b}
        nb = adj.get(b, set()) - {a}
        inter = na & nb
        union = na | nb
        cn = len(inter)
        jac = float(cn) / float(max(1, len(union)))
        da = max(1, len(adj.get(a, set())))
        db = max(1, len(adj.get(b, set())))
        emb = float(cn) / float(min(da, db))
        cn_list.append(cn)
        jac_list.append(jac)
        emb_list.append(emb)

    cn_arr = np.asarray(cn_list, dtype=np.float64)
    jac_arr = np.asarray(jac_list, dtype=np.float64)
    emb_arr = np.asarray(emb_list, dtype=np.float64)

    # Rank-normalize to [0,1] for scale with other features (unsupervised, global on this graph).
    def _rank01(x: np.ndarray) -> np.ndarray:
        n = len(x)
        if n == 0:
            return x
        order = np.argsort(x)
        ranks = np.empty(n, dtype=np.float64)
        ranks[order] = np.arange(n, dtype=np.float64)
        return ranks / max(1.0, float(n - 1))

    return {
        "v2_local_common_n_rank": _rank01(cn_arr),
        "v2_local_neighbor_jaccard": jac_arr,  # already in [0,1]
        "v2_local_embeddedness_rank": _rank01(emb_arr),
    }


def compute_infra_dominance(edges_df: pd.DataFrame, feat_frame: dict[str, Any]) -> np.ndarray:
    """
    Per-edge max(infra_contrib) / sum(infra_contrib) in [0,1]. High => single channel dominates.
    """
    contrib_cols: list[str] = list(feat_frame.get("infra_contrib_cols") or [])
    if not contrib_cols:
        return np.zeros(len(edges_df), dtype=np.float64)
    e = feat_frame["edges_df"]
    mat = e[contrib_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    s = np.maximum(mat.sum(axis=1), 1e-12)
    mx = np.max(mat, axis=1)
    return np.clip(mx / s, 0.0, 1.0)


def build_v2_edge_feature_table(
    edges_df: pd.DataFrame,
    nodes_df: pd.DataFrame,
    cfg: Any,
) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    """
    Build numeric feature matrix for the MLP.

    Returns:
        features_df: columns = feature names, aligned row-wise with edges_df
        feature_names: list of column names in fixed order
        manifest: metadata for JSON (groups for perturbation, stats, etc.)
    """
    edges_df = edges_df.copy()
    edges_df["shard_a"] = edges_df["shard_a"].astype(str)
    edges_df["shard_b"] = edges_df["shard_b"].astype(str)

    feat_frame = build_method1_edge_feature_frame(edges_df, weight_col="edge_weight")

    # Edge-local numeric columns: all numeric except ids and edge_weight.
    exclude = {"shard_a", "shard_b", "edge_weight"}
    ex_extra = getattr(cfg, "extra_edge_columns_exclude", ()) or ()
    exclude |= set(ex_extra)

    edge_feature_cols: list[str] = []
    for c in edges_df.columns:
        if c in exclude:
            continue
        if pd.api.types.is_numeric_dtype(edges_df[c]):
            edge_feature_cols.append(c)
    edge_feature_cols = sorted(edge_feature_cols)

    X_edge = edges_df[edge_feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    # Local graph (unweighted only)
    local = compute_unweighted_local_graph_features(edges_df)
    local_df = pd.DataFrame(local)

    # Hub dominance (input + hub penalty)
    hub_dom = compute_infra_dominance(edges_df, feat_frame)

    # Node joins: safe numeric columns only
    nodes_df = nodes_df.copy()
    nodes_df["shard_id"] = nodes_df["shard_id"].astype(str)
    node_num = _numeric_node_columns(nodes_df)
    n_indexed = nodes_df.set_index("shard_id")

    pair_feats: dict[str, np.ndarray] = {}
    for c in node_num:
        ser = pd.to_numeric(n_indexed[c], errors="coerce")
        va = edges_df["shard_a"].map(ser)
        vb = edges_df["shard_b"].map(ser)
        va = pd.to_numeric(va, errors="coerce")
        vb = pd.to_numeric(vb, errors="coerce")
        pair_feats[f"nodepair_min_{c}"] = np.minimum(va.fillna(0.0), vb.fillna(0.0)).to_numpy()
        pair_feats[f"nodepair_max_{c}"] = np.maximum(va.fillna(0.0), vb.fillna(0.0)).to_numpy()
        pair_feats[f"nodepair_absdiff_{c}"] = (va.fillna(0.0) - vb.fillna(0.0)).abs().to_numpy()

    pair_df = pd.DataFrame(pair_feats)

    hub_df = pd.DataFrame({"v2_infra_dominance": hub_dom})

    features_df = pd.concat([X_edge, local_df, pair_df, hub_df], axis=1)
    feature_names = list(features_df.columns)

    # Column groups for perturbation (by name prefix / exact)
    semantic_cols = [c for c in feature_names if c == "centroid_cosine"]
    temporal_cols = [c for c in feature_names if c.startswith("temporal")]
    infra_cols = [
        c
        for c in feature_names
        if c.startswith("shared_")
        or c.startswith("infra_contrib")
        or c.startswith("url_jaccard")
        or c.startswith("domain_jaccard")
        or c.startswith("stem_jaccard")
        or c.startswith("sender_jaccard")
        or c.startswith("sender_email_domain_jaccard")
        or c == "infra_score"
    ]
    local_cols = [c for c in feature_names if c.startswith("v2_local_")]
    hub_cols = [c for c in feature_names if c == "v2_infra_dominance"]
    nodepair_cols = [c for c in feature_names if c.startswith("nodepair_")]

    manifest: dict[str, Any] = {
        "n_edges": int(len(edges_df)),
        "edge_feature_columns": edge_feature_cols,
        "local_feature_columns": list(local_df.columns),
        "nodepair_feature_columns": [c for c in pair_df.columns],
        "hub_feature_columns": hub_cols,
        "excluded_edge_columns": sorted(exclude),
        "blocked_node_columns": sorted(NODE_GT_BLOCKLIST),
        "perturb_groups": {
            "semantic": semantic_cols,
            "temporal": temporal_cols,
            "infra": infra_cols,
            "local_graph": local_cols,
            "hub": hub_cols,
            "nodepair": nodepair_cols,
        },
    }
    return features_df, feature_names, manifest
