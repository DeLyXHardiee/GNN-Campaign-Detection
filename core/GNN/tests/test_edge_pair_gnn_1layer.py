"""1-layer Edge-GNN with MLP-compatible local head and concat combine mode."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

_GNN = Path(__file__).resolve().parents[1]
if str(_GNN) not in sys.path:
    sys.path.insert(0, str(_GNN))

from src.edge_pair_gnn import (  # noqa: E402
    COMBINE_MODE_CONCAT_LOCAL_GRAPH,
    LOCAL_HEAD_EMAIL_PAIR_MLP_COMPATIBLE,
    EdgePairGnnModel,
    build_edge_pair_gnn_model,
)


def test_one_layer_mlp_compatible_concat_returns_logits() -> None:
    in_dim = 21
    training_cfg = {
        "pair_scorer_hidden_dim": 256,
        "pair_scorer_dropout": 0.2,
        "edge_gnn": {
            "num_gnn_layers": 1,
            "local_head": LOCAL_HEAD_EMAIL_PAIR_MLP_COMPATIBLE,
            "hidden_dim": 256,
            "dropout": 0.2,
            "combine_mode": COMBINE_MODE_CONCAT_LOCAL_GRAPH,
        },
    }
    model = build_edge_pair_gnn_model(in_dim, training_cfg)
    assert model.num_gnn_layers == 1
    assert model.uses_mlp_compatible_local_graph
    assert model.output_mlp.in_features == 512

    x = torch.randn(4, in_dim)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
    logits = model(x, edge_index)
    assert logits.shape == (4,)


def test_one_layer_uses_edge_index() -> None:
    model = EdgePairGnnModel(
        4,
        hidden_dim=8,
        num_gnn_layers=1,
        dropout=0.0,
        local_head=LOCAL_HEAD_EMAIL_PAIR_MLP_COMPATIBLE,
        combine_mode=COMBINE_MODE_CONCAT_LOCAL_GRAPH,
    )
    x = torch.randn(3, 4)
    no_edges = torch.zeros((2, 0), dtype=torch.long)
    with_edges = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    out0 = model(x, no_edges)
    out1 = model(x, with_edges)
    assert out0.shape == (3,)
    assert out1.shape == (3,)
    assert not torch.allclose(out0, out1)


def test_tiny_line_graph_forward() -> None:
    model = build_edge_pair_gnn_model(
        3,
        {
            "edge_gnn": {
                "num_gnn_layers": 1,
                "local_head": LOCAL_HEAD_EMAIL_PAIR_MLP_COMPATIBLE,
                "hidden_dim": 4,
                "dropout": 0.0,
                "combine_mode": COMBINE_MODE_CONCAT_LOCAL_GRAPH,
            }
        },
    )
    x = torch.randn(2, 3)
    edge_index = torch.tensor([[0], [1]], dtype=torch.long)
    logits = model(x, edge_index)
    assert logits.shape == (2,)
    stats = model.graph_representation_stats(x, edge_index)
    assert "h_local_norm_mean" in stats
    assert "h_graph_norm_mean" in stats
