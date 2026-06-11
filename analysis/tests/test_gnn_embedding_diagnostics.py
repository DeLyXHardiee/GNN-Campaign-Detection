"""Unit tests for GNN embedding diagnostics helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.utils.gnn_embedding_diagnostics import (
    _auroc_auprc,
    _cosine_l2,
    build_recommendations,
    summarize_cosine_calibration,
    summarize_pairwise_by_relation,
)


def test_cosine_l2_orthogonal():
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    c, l2 = _cosine_l2(a, b)
    assert abs(c) < 1e-6
    assert abs(l2 - np.sqrt(2)) < 1e-6


def test_auroc_perfect_separation():
    y = np.array([0] * 10 + [1] * 10)
    s = np.array([0.1] * 10 + [0.9] * 10)
    m = _auroc_auprc(y, s)
    assert m["auroc"] is not None and m["auroc"] >= 0.99


def test_summarize_pairwise_by_relation():
    df = pd.DataFrame(
        {
            "email_i": ["a", "b", "c", "d"],
            "email_j": ["b", "c", "d", "e"],
            "gt_relation": ["same_campaign", "same_campaign", "cross_campaign", "cross_campaign"],
            "gnn_encoder_cosine": [0.9, 0.85, 0.2, 0.15],
        }
    )
    tab, block = summarize_pairwise_by_relation(df, source_names=["gnn_encoder"])
    assert not tab.empty
    assert "gnn_encoder" in block["by_source"]
    assert block["by_source"]["gnn_encoder"]["same_minus_cross_mean_cosine"] > 0


def test_cosine_calibration_buckets():
    df = pd.DataFrame(
        {
            "gnn_encoder_cosine": [0.3, 0.6, 0.8, 0.92],
            "gt_relation": ["same_campaign", "cross_campaign", "same_campaign", "same_campaign"],
        }
    )
    out = summarize_cosine_calibration(df, primary_source="gnn_encoder")
    assert "0.85_0.95" in out
    assert out["0.85_0.95"]["n_pairs"] == 1


def test_build_recommendations_smoke():
    rec = build_recommendations(
        pairwise_block={"by_source": {"gnn_encoder": {"separation": {"auroc": 0.8}}}},
        retrieval_df=pd.DataFrame(),
        primary_source="gnn_encoder",
        encoder_meta={"pair_encoder_backend": "gnn"},
        suspicious_nonedge={"n_high_cosine_nonedges": 10},
        probe_summary={},
    )
    assert "gnn_embedding_recommendations" not in rec
    assert "A_are_gnn_embeddings_useful" in rec
