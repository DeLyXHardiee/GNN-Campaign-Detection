"""Smoke tests for candidate-edge line graph builder."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import torch

_GNN = Path(__file__).resolve().parents[1]
if str(_GNN) not in sys.path:
    sys.path.insert(0, str(_GNN))

from src.edge_candidate_line_graph import build_candidate_edge_line_graph


def _adjacency_set(edge_index: torch.Tensor) -> set[tuple[int, int]]:
    ei = edge_index.cpu().numpy()
    return {(int(ei[0, k]), int(ei[1, k])) for k in range(ei.shape[1])}


def test_shared_endpoint_adjacency_triangle():
    df = pd.DataFrame(
        {
            "email_i": ["a", "a", "b"],
            "email_j": ["b", "c", "c"],
            "semantic_cosine_max": [0.9, 0.5, 0.7],
        }
    )
    edge_index, meta, stats = build_candidate_edge_line_graph(df, max_neighbors_per_endpoint=None)
    assert len(meta) == 3
    assert stats["num_edge_nodes"] == 3
    adj = _adjacency_set(edge_index)
    assert (0, 1) in adj and (1, 0) in adj
    assert (0, 2) in adj and (2, 0) in adj
    assert (1, 2) in adj and (2, 1) in adj
    assert (0, 0) not in adj


def test_top_k_prunes_hub():
    rows = [{"email_i": "hub", "email_j": f"x{i}", "semantic_cosine_max": float(i) / 10.0} for i in range(10)]
    df = pd.DataFrame(rows)
    edge_index, _meta, stats = build_candidate_edge_line_graph(
        df,
        max_neighbors_per_endpoint=3,
        rank_column="semantic_cosine_max",
    )
    assert stats["num_pruned_endpoint_lists"] >= 1
    assert stats["max_incident_edges_after_pruning"] <= 3
    assert stats["num_line_edges"] < stats["num_edge_nodes"] * (stats["num_edge_nodes"] - 1)


def test_edge_node_id_matches_row_order():
    df = pd.DataFrame({"email_i": ["e1", "e2"], "email_j": ["e2", "e3"]})
    _edge_index, meta, _stats = build_candidate_edge_line_graph(df)
    assert meta["edge_node_id"].tolist() == [0, 1]
    assert meta["row_index"].tolist() == [0, 1]


def test_split_masks_after_reset_index_splits():
    from src.edge_pair_gnn_train import _ensure_edge_node_ids_on_splits, _split_masks_from_subframes

    df = pd.DataFrame(
        {
            "email_i": ["a", "b", "c"],
            "email_j": ["b", "c", "d"],
            "is_positive": [True, False, False],
            "is_unlabeled": [False, True, True],
            "is_reliable_negative": [False, False, False],
        }
    )
    df["_edge_node_id"] = [0, 1, 2]
    train_df = df.iloc[[0, 2]].reset_index(drop=True)
    val_df = df.iloc[[1]].reset_index(drop=True)
    test_df = df.iloc[[]].reset_index(drop=True)
    df2, train_df2, val_df2, test_df2 = _ensure_edge_node_ids_on_splits(df, train_df, val_df, test_df)
    train_mask, val_mask, test_mask = _split_masks_from_subframes(
        len(df2), train_df2, val_df2, test_df2
    )
    assert train_mask.tolist() == [True, False, True]
    assert val_mask.tolist() == [False, True, False]
    assert not test_mask.any()
