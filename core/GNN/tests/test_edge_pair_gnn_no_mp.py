"""EdgePairGnnModel with num_gnn_layers=0 (no message passing)."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

_GNN = Path(__file__).resolve().parents[1]
if str(_GNN) not in sys.path:
    sys.path.insert(0, str(_GNN))

from src.edge_pair_gnn import (  # noqa: E402
    LOCAL_HEAD_EMAIL_PAIR_MLP_COMPATIBLE,
    EdgePairGnnModel,
    build_edge_pair_gnn_model,
)
from src.pair_scorer import EmailPairMLPScorer  # noqa: E402


def test_num_gnn_layers_zero_no_convs_and_ignores_edges() -> None:
    model = EdgePairGnnModel(4, hidden_dim=8, num_gnn_layers=0, dropout=0.0)
    assert model.num_gnn_layers == 0
    assert len(model.convs) == 0

    x = torch.randn(5, 4)
    edge_index = torch.zeros((2, 0), dtype=torch.long)
    logits_a = model(x, edge_index)

    # Random edges must not change output when num_gnn_layers=0.
    edge_index2 = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
    logits_b = model(x, edge_index2)
    assert logits_a.shape == (5,)
    assert torch.allclose(logits_a, logits_b)


def test_mlp_compatible_head_reuses_email_pair_scorer() -> None:
    in_dim = 21
    training_cfg = {
        "pair_scorer_type": "email_pair_mlp",
        "pair_scorer_hidden_dim": 256,
        "pair_scorer_dropout": 0.2,
        "pair_scorer_use_embedding_features": False,
        "pair_scorer_use_explicit_features": True,
        "edge_gnn": {
            "num_gnn_layers": 0,
            "local_head": LOCAL_HEAD_EMAIL_PAIR_MLP_COMPATIBLE,
            "hidden_dim": 256,
            "dropout": 0.2,
        },
    }
    model = build_edge_pair_gnn_model(in_dim, training_cfg)
    assert model.num_gnn_layers == 0
    assert model.local_head == LOCAL_HEAD_EMAIL_PAIR_MLP_COMPATIBLE
    assert isinstance(model.pair_scorer, EmailPairMLPScorer)
    assert model.pair_scorer._in_dim == in_dim
    assert model.pair_scorer.net[0].out_features == 256

    x = torch.randn(4, in_dim)
    edge_index = torch.zeros((2, 0), dtype=torch.long)
    logits = model(x, edge_index)
    assert logits.shape == (4,)


def test_num_gnn_layers_positive_uses_message_passing() -> None:
    model = EdgePairGnnModel(4, hidden_dim=8, num_gnn_layers=1, dropout=0.0)
    assert len(model.convs) == 1
    x = torch.randn(3, 4)
    no_edges = torch.zeros((2, 0), dtype=torch.long)
    with_edges = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    # With no edges, SAGE still runs; outputs may differ from multi-edge case.
    out0 = model(x, no_edges)
    out1 = model(x, with_edges)
    assert out0.shape == (3,)
    assert out1.shape == (3,)
