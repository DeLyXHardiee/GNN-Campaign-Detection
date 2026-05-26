"""Line graph over candidate email-email pairs (edge-node = one candidate pair)."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd
import torch


def _rank_values_for_pruning(
    pairs_df: pd.DataFrame,
    node_ids: list[int],
    *,
    rank_column: str,
) -> list[float]:
    """Higher is better for keeping incident edge-nodes under top-k."""
    if rank_column in pairs_df.columns:
        s = pd.to_numeric(pairs_df.loc[node_ids, rank_column], errors="coerce")
        return [float(x) if pd.notna(x) else float("-inf") for x in s.tolist()]
    if "source_count" in pairs_df.columns:
        s = pd.to_numeric(pairs_df.loc[node_ids, "source_count"], errors="coerce").fillna(0)
        return [float(x) for x in s.tolist()]
    return [float(-node_id) for node_id in node_ids]


def _prune_incident_list(
    pairs_df: pd.DataFrame,
    node_ids: list[int],
    *,
    max_neighbors_per_endpoint: int | None,
    rank_column: str,
) -> tuple[list[int], bool]:
    if max_neighbors_per_endpoint is None or max_neighbors_per_endpoint <= 0:
        return list(node_ids), False
    if len(node_ids) <= max_neighbors_per_endpoint:
        return list(node_ids), False
    scores = _rank_values_for_pruning(pairs_df, node_ids, rank_column=rank_column)
    order = sorted(range(len(node_ids)), key=lambda i: (-scores[i], node_ids[i]))
    kept = [node_ids[i] for i in order[: int(max_neighbors_per_endpoint)]]
    return kept, True


def build_candidate_edge_line_graph(
    pairs_df: pd.DataFrame,
    *,
    max_neighbors_per_endpoint: int | None = 64,
    rank_column: str = "semantic_cosine_max",
    make_undirected: bool = True,
) -> tuple[torch.Tensor, pd.DataFrame, dict[str, Any]]:
    """
    Build a line graph on candidate pairs: nodes are pairs; edges connect pairs sharing an email.

    ``edge_node_id`` equals the row position in ``pairs_df`` (0 .. N-1).
    """
    if "email_i" not in pairs_df.columns or "email_j" not in pairs_df.columns:
        raise ValueError("pairs_df must contain email_i and email_j columns")

    n = int(len(pairs_df))
    work = pairs_df.reset_index(drop=True)
    edge_node_meta = pd.DataFrame(
        {
            "edge_node_id": np.arange(n, dtype=np.int64),
            "email_i": work["email_i"].astype(str).tolist(),
            "email_j": work["email_j"].astype(str).tolist(),
            "row_index": np.arange(n, dtype=np.int64),
        }
    )

    email_to_nodes: dict[str, list[int]] = defaultdict(list)
    for node_id in range(n):
        ei = str(work.at[node_id, "email_i"])
        ej = str(work.at[node_id, "email_j"])
        email_to_nodes[ei].append(node_id)
        email_to_nodes[ej].append(node_id)

    max_before = 0
    max_after = 0
    num_pruned_lists = 0
    edge_set: set[tuple[int, int]] = set()

    for _email, incident in email_to_nodes.items():
        max_before = max(max_before, len(incident))
        pruned, was_pruned = _prune_incident_list(
            work,
            incident,
            max_neighbors_per_endpoint=max_neighbors_per_endpoint,
            rank_column=rank_column,
        )
        if was_pruned:
            num_pruned_lists += 1
        max_after = max(max_after, len(pruned))
        for i in range(len(pruned)):
            for j in range(i + 1, len(pruned)):
                u, v = int(pruned[i]), int(pruned[j])
                if u == v:
                    continue
                edge_set.add((u, v))
                if make_undirected:
                    edge_set.add((v, u))

    if not edge_set:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    else:
        src, dst = zip(*sorted(edge_set))
        edge_index = torch.tensor([src, dst], dtype=torch.long)

    if edge_index.numel() == 0:
        degree = torch.zeros(n, dtype=torch.long)
    else:
        degree = torch.zeros(n, dtype=torch.long)
        ones = torch.ones(edge_index.size(1), dtype=torch.long)
        degree.index_add_(0, edge_index[0], ones)

    stats: dict[str, Any] = {
        "num_edge_nodes": n,
        "num_line_edges": int(edge_index.size(1)),
        "num_unique_email_endpoints": int(len(email_to_nodes)),
        "max_incident_edges_before_pruning": int(max_before),
        "max_incident_edges_after_pruning": int(max_after),
        "num_pruned_endpoint_lists": int(num_pruned_lists),
        "max_neighbors_per_endpoint": max_neighbors_per_endpoint,
        "rank_column": str(rank_column),
        "make_undirected": bool(make_undirected),
        "mean_degree": float(degree.float().mean().item()) if n > 0 else 0.0,
        "max_degree": int(degree.max().item()) if n > 0 else 0,
        "isolated_nodes": int((degree == 0).sum().item()),
    }
    return edge_index, edge_node_meta, stats
