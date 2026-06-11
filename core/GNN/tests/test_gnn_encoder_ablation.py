"""Tests for reversible GNN encoder ablation helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

_GNN = Path(__file__).resolve().parents[1]
if str(_GNN) not in sys.path:
    sys.path.insert(0, str(_GNN))

from src.gnn_encoder_ablation import (
    EmailGnnInputAdapter,
    GnnEncoderAblationConfig,
    apply_hetero_edge_dropout,
    parse_gnn_encoder_ablation,
)
from src.model import PairGnnEncoder, build_pair_gnn_encoder


def test_email_adapter_projects_128_32_to_32_64():
    adapter = EmailGnnInputAdapter(
        semantic_in=128,
        semantic_out=32,
        other_in=32,
        other_out=64,
        semantic_dropout=0.30,
    )
    x = torch.randn(5, 160)
    adapter.eval()
    out = adapter(x)
    assert out.shape == (5, 96)


def test_edge_dropout_keeps_at_least_one_edge():
    ei = torch.tensor([[0, 1, 2, 3], [1, 0, 2, 3]])
    out = apply_hetero_edge_dropout(
        {("email", "has_url", "url"): ei},
        default_p=0.99,
        training=True,
    )
    assert out[("email", "has_url", "url")].size(1) >= 1


def test_parse_gnn_encoder_ablation_defaults_off():
    cfg = parse_gnn_encoder_ablation(None)
    assert cfg.enabled is False


def test_build_pair_gnn_encoder_types():
    meta = (["email", "url"], [("email", "has_url", "url"), ("url", "rev", "email")])
    off = build_pair_gnn_encoder(
        meta,
        {"hidden": 16, "out_dim": 16, "layers": 2, "dropout": 0.0, "gnn_encoder_ablation": {"enabled": False}},
    )
    from src.model import HeteroSAGE

    assert isinstance(off, HeteroSAGE)
    on = build_pair_gnn_encoder(
        meta,
        {
            "hidden": 16,
            "out_dim": 16,
            "layers": 2,
            "dropout": 0.0,
            "gnn_encoder_ablation": {
                "enabled": True,
                "email_semantic_proj_dim": 32,
                "email_nonsemantic_proj_dim": 64,
                "email_baked_semantic_dim": 128,
                "email_baked_nonsemantic_dim": 32,
            },
        },
    )
    assert isinstance(on, PairGnnEncoder)
    assert on.email_adapter is not None
    assert on.email_adapter.out_dim == 96
