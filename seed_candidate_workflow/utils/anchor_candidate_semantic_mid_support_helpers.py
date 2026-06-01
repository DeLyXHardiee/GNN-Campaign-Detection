"""
Medium-semantic-band candidate families (0.85 <= cosine < 0.90) with shared-evidence support.

Candidate-only broadening rules; uses the same embedding vectors and core-channel overlap
semantics as pair_training_dataset_helpers._add_shared_attribute_pair_features.
"""

from __future__ import annotations

from typing import Any, Callable, Literal

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

CORE_CHANNEL_NODE_COLS: tuple[tuple[str, str], ...] = (
    ("sender_set", "sender"),
    ("stem_set", "stem"),
    ("url_set", "url"),
    ("attachment_set", "attachment"),
    ("sender_email_domain_set", "sender_domain"),
    ("domain_set", "domain"),
)

SupportMode = Literal["sender", "core", "stem"]


def _to_set_cell(v: Any) -> set[str]:
    if isinstance(v, set):
        return {str(x) for x in v if str(x).strip()}
    if isinstance(v, list):
        return {str(x) for x in v if str(x).strip()}
    if isinstance(v, str) and v.strip():
        return {v.strip()}
    return set()


def build_nodes_core_sets(nodes_df: pd.DataFrame) -> dict[str, dict[str, set[str]]]:
    keep = [c for c, _ in CORE_CHANNEL_NODE_COLS if c in nodes_df.columns]
    out: dict[str, dict[str, set[str]]] = {}
    for _, row in nodes_df.iterrows():
        eid = str(row.get("external_id") or "").strip()
        if not eid:
            continue
        out[eid] = {col: _to_set_cell(row.get(col)) for col in keep}
    return out


def n_shared_core_channels(eid_a: str, eid_b: str, nodes_by_email: dict[str, dict[str, set[str]]]) -> int:
    na = nodes_by_email.get(eid_a)
    nb = nodes_by_email.get(eid_b)
    if na is None or nb is None:
        return 0
    n = 0
    for col, _base in CORE_CHANNEL_NODE_COLS:
        if col not in na or col not in nb:
            continue
        if na[col] & nb[col]:
            n += 1
    return n


def has_shared_channel(
    eid_a: str,
    eid_b: str,
    nodes_by_email: dict[str, dict[str, set[str]]],
    *,
    node_col: str,
) -> bool:
    na = nodes_by_email.get(eid_a)
    nb = nodes_by_email.get(eid_b)
    if na is None or nb is None:
        return False
    return bool((na.get(node_col) or set()) & (nb.get(node_col) or set()))


def _compute_direct_cosine_band_pairs(
    *,
    node_ids: list[str],
    id_to_vec: dict[str, np.ndarray],
    semantic_top_k: int,
    semantic_min_cos: float,
    semantic_max_cos_exclusive: float,
) -> dict[tuple[str, str], float]:
    """
    Directed kNN scan; return canonical pair -> max cosine in [min, max_exclusive).
    """
    semantic_node_ids = [eid for eid in node_ids if eid in id_to_vec]
    if len(semantic_node_ids) < 2 or semantic_top_k <= 0:
        return {}

    emb = np.stack([id_to_vec[eid] for eid in semantic_node_ids]).astype(np.float32)
    n = emb.shape[0]
    k_plus = min(int(semantic_top_k) + 1, n)

    nn = NearestNeighbors(n_neighbors=k_plus, metric="cosine", algorithm="brute")
    nn.fit(emb)
    dists, neigh = nn.kneighbors(emb, return_distance=True)

    lo = float(semantic_min_cos)
    hi = float(semantic_max_cos_exclusive)
    pair_cos: dict[tuple[str, str], float] = {}

    for local_i in range(n):
        i_eid = semantic_node_ids[local_i]
        rank = 0
        for local_j, dist in zip(neigh[local_i], dists[local_i], strict=False):
            j_eid = semantic_node_ids[int(local_j)]
            if j_eid == i_eid:
                continue
            cs = float(1.0 - float(dist))
            if cs < lo or cs >= hi:
                continue
            rank += 1
            if rank > int(semantic_top_k):
                break
            a, b = (i_eid, j_eid) if i_eid <= j_eid else (j_eid, i_eid)
            prev = pair_cos.get((a, b))
            if prev is None or cs > prev:
                pair_cos[(a, b)] = cs

    return pair_cos


def _support_predicate(mode: SupportMode) -> Callable[[str, str, dict[str, dict[str, set[str]]]], bool]:
    if mode == "sender":
        return lambda a, b, nb: has_shared_channel(a, b, nb, node_col="sender_set")
    if mode == "stem":
        return lambda a, b, nb: has_shared_channel(a, b, nb, node_col="stem_set")
    if mode == "core":
        return lambda a, b, nb: n_shared_core_channels(a, b, nb) >= 1
    raise ValueError(f"Unknown support mode: {mode!r}")


def generate_semantic_mid_support_v1(
    *,
    nodes_df: pd.DataFrame,
    id_to_vec: dict[str, np.ndarray],
    generator_cfg: dict[str, Any],
    support_mode: SupportMode,
    source_label: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Emit candidate pairs in the medium semantic band with a channel-support predicate.
    """
    semantic_top_k = int(generator_cfg.get("semantic_top_k", 50))
    semantic_min_cos = float(generator_cfg.get("semantic_min_cos", 0.85))
    semantic_max_cos_exclusive = float(generator_cfg.get("semantic_max_cos_exclusive", 0.90))
    max_candidate_rows = int(generator_cfg.get("max_candidate_rows", 500_000))

    nodes_df = nodes_df.copy()
    nodes_df["external_id"] = nodes_df["external_id"].astype(str)
    nodes_by_email = build_nodes_core_sets(nodes_df)
    node_ids = nodes_df["external_id"].tolist()

    band_pairs = _compute_direct_cosine_band_pairs(
        node_ids=node_ids,
        id_to_vec=id_to_vec,
        semantic_top_k=semantic_top_k,
        semantic_min_cos=semantic_min_cos,
        semantic_max_cos_exclusive=semantic_max_cos_exclusive,
    )

    pred = _support_predicate(support_mode)
    rows: list[dict[str, Any]] = []
    n_band = len(band_pairs)
    n_support_fail = 0
    for (a, b), cos in sorted(band_pairs.items(), key=lambda x: (-x[1], x[0][0], x[0][1])):
        if not pred(a, b, nodes_by_email):
            n_support_fail += 1
            continue
        rows.append(
            {
                "email_i": a,
                "email_j": b,
                "source": source_label,
                "cosine": float(cos),
                "semantic_min_cos": semantic_min_cos,
                "semantic_max_cos_exclusive": semantic_max_cos_exclusive,
                "n_shared_core_channels": int(n_shared_core_channels(a, b, nodes_by_email)),
                "has_shared_sender": bool(has_shared_channel(a, b, nodes_by_email, node_col="sender_set")),
                "has_shared_stem": bool(has_shared_channel(a, b, nodes_by_email, node_col="stem_set")),
            }
        )
        if len(rows) >= max_candidate_rows:
            break

    df = pd.DataFrame(rows)
    diag = {
        "support_mode": support_mode,
        "semantic_min_cos": semantic_min_cos,
        "semantic_max_cos_exclusive": semantic_max_cos_exclusive,
        "semantic_top_k": semantic_top_k,
        "n_pairs_in_band": int(n_band),
        "n_pairs_support_pass": int(len(df)),
        "n_pairs_support_fail": int(n_support_fail),
        "max_candidate_rows": max_candidate_rows,
        "truncated": bool(len(rows) >= max_candidate_rows),
    }
    return df, diag


def generate_semantic_mid_sender_support_v1(
    *,
    nodes_df: pd.DataFrame,
    id_to_vec: dict[str, np.ndarray],
    generator_cfg: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    return generate_semantic_mid_support_v1(
        nodes_df=nodes_df,
        id_to_vec=id_to_vec,
        generator_cfg=generator_cfg,
        support_mode="sender",
        source_label="semantic_mid_sender_support_v1",
    )


def generate_semantic_mid_core_support_v1(
    *,
    nodes_df: pd.DataFrame,
    id_to_vec: dict[str, np.ndarray],
    generator_cfg: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    return generate_semantic_mid_support_v1(
        nodes_df=nodes_df,
        id_to_vec=id_to_vec,
        generator_cfg=generator_cfg,
        support_mode="core",
        source_label="semantic_mid_core_support_v1",
    )


def generate_semantic_mid_stem_support_v1(
    *,
    nodes_df: pd.DataFrame,
    id_to_vec: dict[str, np.ndarray],
    generator_cfg: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    return generate_semantic_mid_support_v1(
        nodes_df=nodes_df,
        id_to_vec=id_to_vec,
        generator_cfg=generator_cfg,
        support_mode="stem",
        source_label="semantic_mid_stem_support_v1",
    )


def _nodes_sender_dict(nodes_df: pd.DataFrame) -> dict[str, dict[str, set[str]]]:
    out: dict[str, dict[str, set[str]]] = {}
    if "sender_set" not in nodes_df.columns:
        return out
    for _, row in nodes_df.iterrows():
        eid = str(row.get("external_id") or "").strip()
        if not eid:
            continue
        out[eid] = {"sender_set": _to_set_cell(row.get("sender_set"))}
    return out


def generate_semantic_mid_senderlocalpart_support_v1(
    *,
    nodes_df: pd.DataFrame,
    id_to_vec: dict[str, np.ndarray],
    generator_cfg: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Medium semantic band (default 0.85 <= cosine < 0.90) with normalized sender local-part
    similarity >= threshold (default 0.7). Distinct from ``semantic_mid_sender_support_v1``
    (exact shared sender channel only).
    """
    from seed_candidate_workflow.utils.pair_similarity_features import (
        sender_localpart_norm_jaccard_for_nodes,
    )

    semantic_top_k = int(generator_cfg.get("semantic_top_k", 50))
    semantic_min_cos = float(generator_cfg.get("semantic_min_cos", 0.85))
    semantic_max_cos_exclusive = float(generator_cfg.get("semantic_max_cos_exclusive", 0.90))
    min_sender_lp = float(generator_cfg.get("min_sender_localpart_norm_jaccard", 0.7))
    max_candidate_rows = int(generator_cfg.get("max_candidate_rows", 500_000))

    nodes_df = nodes_df.copy()
    nodes_df["external_id"] = nodes_df["external_id"].astype(str)
    nodes_by_sender = _nodes_sender_dict(nodes_df)
    node_ids = nodes_df["external_id"].tolist()

    band_pairs = _compute_direct_cosine_band_pairs(
        node_ids=node_ids,
        id_to_vec=id_to_vec,
        semantic_top_k=semantic_top_k,
        semantic_min_cos=semantic_min_cos,
        semantic_max_cos_exclusive=semantic_max_cos_exclusive,
    )

    rows: list[dict[str, Any]] = []
    n_band = len(band_pairs)
    n_lp_fail = 0
    for (a, b), cos in sorted(band_pairs.items(), key=lambda x: (-x[1], x[0][0], x[0][1])):
        na = nodes_by_sender.get(a) or {}
        nb = nodes_by_sender.get(b) or {}
        lp_sim = sender_localpart_norm_jaccard_for_nodes(na, nb)
        if lp_sim < min_sender_lp:
            n_lp_fail += 1
            continue
        rows.append(
            {
                "email_i": a,
                "email_j": b,
                "source": "semantic_mid_senderlocalpart_support_v1",
                "cosine": float(cos),
                "sender_localpart_norm_jaccard": float(lp_sim),
                "semantic_min_cos": semantic_min_cos,
                "semantic_max_cos_exclusive": semantic_max_cos_exclusive,
            }
        )
        if len(rows) >= max_candidate_rows:
            break

    df = pd.DataFrame(rows)
    return df, {
        "semantic_min_cos": semantic_min_cos,
        "semantic_max_cos_exclusive": semantic_max_cos_exclusive,
        "min_sender_localpart_norm_jaccard": min_sender_lp,
        "semantic_top_k": semantic_top_k,
        "n_pairs_in_band": int(n_band),
        "n_pairs_sender_localpart_pass": int(len(df)),
        "n_pairs_sender_localpart_fail": int(n_lp_fail),
        "max_candidate_rows": max_candidate_rows,
        "truncated": bool(len(rows) >= max_candidate_rows),
    }
