"""Lightweight sanity checks for Method 1 edge refinement."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from seed_candidate_workflow.utils.semantic_shard_edge_refinement_method1 import (  # noqa: E402
    Method1RefinementConfig,
    run_method1_edge_refinement_pipeline,
    synthetic_method1_sanity_check,
)
from seed_candidate_workflow.utils.semantic_shard_step3_helpers import run_weighted_community_detection  # noqa: E402


def test_synthetic_method1_pipeline():
    out = synthetic_method1_sanity_check()
    assert out["ok"] is True


def test_method1_ablation_no_perturbation():
    edges = pd.DataFrame(
        [
            {
                "shard_a": "a",
                "shard_b": "b",
                "centroid_cosine": 0.1,
                "infra_score": 0.2,
                "temporal_score": 0.3,
                "edge_weight": 1.0,
            }
        ]
    )
    cfg = Method1RefinementConfig(
        use_perturbation_stability=False,
        use_local_structure=False,
    )
    refined, _, _ = run_method1_edge_refinement_pipeline(edges, cfg=cfg, output_dir=None)
    assert len(refined) == 1
    assert 0 <= float(refined["edge_trust"].iloc[0]) <= 1


@pytest.mark.parametrize("method", ["louvain", "leiden"])
def test_leiden_and_louvain_on_triangle(method):
    shard_ids = ["s0", "s1", "s2"]
    edges_df = pd.DataFrame(
        [
            {"shard_a": "s0", "shard_b": "s1", "edge_weight": 1.0},
            {"shard_a": "s1", "shard_b": "s2", "edge_weight": 1.0},
            {"shard_a": "s0", "shard_b": "s2", "edge_weight": 1.0},
        ]
    )
    m, info = run_weighted_community_detection(
        shard_ids,
        edges_df,
        method=method,
        resolution=1.0,
        min_edge_weight=0.0,
        weight_col="edge_weight",
        seed=0,
    )
    assert len(m) == 3
    assert info["n_edges_after_threshold"] == 3
    if method == "leiden":
        assert info["method_used"] == "leiden"


def test_trust_gamma_stores_calibrated_column():
    edges = pd.DataFrame(
        [
            {
                "shard_a": "a",
                "shard_b": "b",
                "centroid_cosine": 0.5,
                "infra_score": 0.5,
                "temporal_score": 0.5,
                "edge_weight": 1.0,
            }
        ]
    )
    cfg = Method1RefinementConfig(
        use_perturbation_stability=False,
        use_local_structure=False,
        trust_gamma=0.5,
        blend_rule="multiplicative",
    )
    refined, _, _ = run_method1_edge_refinement_pipeline(edges, cfg=cfg, output_dir=None)
    assert "edge_trust_calibrated" in refined.columns
    tr = float(refined["edge_trust"].iloc[0])
    tc = float(refined["edge_trust_calibrated"].iloc[0])
    assert tc == pytest.approx(tr**0.5, rel=1e-5, abs=1e-5)


def test_convex_blend_weights_in_zero_one():
    edges = pd.DataFrame(
        [
            {"shard_a": "a", "shard_b": "b", "centroid_cosine": 0.2, "infra_score": 0.3, "temporal_score": 0.4, "edge_weight": 0.8},
            {"shard_a": "b", "shard_b": "c", "centroid_cosine": 0.9, "infra_score": 0.8, "temporal_score": 0.7, "edge_weight": 0.2},
        ]
    )
    cfg = Method1RefinementConfig(
        use_perturbation_stability=False,
        use_local_structure=False,
        blend_rule="convex",
        convex_alpha=0.5,
        trust_gamma=1.0,
    )
    refined, _, _ = run_method1_edge_refinement_pipeline(edges, cfg=cfg, output_dir=None)
    wr = refined["edge_weight_refined"].astype(float)
    assert (wr >= 0).all() and (wr <= 1.0 + 1e-9).all()
    assert "edge_weight_orig_norm_convex" in refined.columns
