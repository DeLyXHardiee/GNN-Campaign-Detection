"""Tests for explicit-only pair scorer (no embedding features)."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

_GNN = Path(__file__).resolve().parents[1]
if str(_GNN) not in sys.path:
    sys.path.insert(0, str(_GNN))

from src.pair_scorer import EmailPairMLPScorer, build_email_pair_mlp_scorer


def test_explicit_only_forward_ignores_embeddings():
    scorer = EmailPairMLPScorer(
        embed_dim=1,
        pair_feat_dim=4,
        mlp_hidden_dim=16,
        use_explicit_pair_features=True,
        use_embedding_features=False,
    )
    feats = torch.randn(3, 4)
    logits = scorer(torch.zeros(3, 1), torch.zeros(3, 1), feats)
    assert logits.shape == (3,)


def test_build_explicit_only_from_training_cfg():
    scorer = build_email_pair_mlp_scorer(
        embed_dim=1,
        pair_feat_dim=5,
        training_cfg={
            "pair_scorer_use_explicit_features": True,
            "pair_scorer_use_embedding_features": False,
            "pair_scorer_hidden_dim": 32,
            "pair_scorer_dropout": 0.1,
        },
    )
    assert scorer.input_feature_dim == 5
    assert scorer.use_embedding_features is False
