"""Tests for high-scoring cross-campaign unlabeled analysis helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from seed_candidate_workflow.utils.pair_high_cross_unlabeled_analysis import (
    HighCrossThresholds,
    _aggregate_likely_explanations,
    _build_high_cross_minimal_explicit_support_summary,
    _build_high_cross_vs_high_same_digest,
    _build_semantic_body_path_coverage_summary,
    _cohort_coverage_block,
    _row_has_minimal_explicit_support,
    _row_likely_explanation_tags,
    high_score_unlabeled_cohort_masks,
)


def test_high_score_unlabeled_cohort_masks():
    th = HighCrossThresholds(high_cross_score_min=0.8, mid_cross_score_min=0.7)
    scores = np.array([0.9, 0.85, 0.75, 0.5, 0.95])
    same = np.array([True, True, False, True, False])
    cross = np.array([False, False, True, False, True])
    unl = np.array([True, True, True, True, True])
    m = high_score_unlabeled_cohort_masks(
        same_eval=same,
        cross_eval=cross,
        unl_eval=unl,
        scores=scores,
        thresholds=th,
    )
    assert m["high_same_unlabeled"].sum() == 2
    assert m["high_cross_unlabeled"].sum() == 1
    assert m["mid_cross_unlabeled"].sum() == 1


def test_row_likely_explanation_tags_semantic_weak_support():
    row = pd.Series(
        {
            "semantic_cosine_max": 0.95,
            "n_shared_core_channels": 0,
            "body_token_jaccard": 0.05,
        }
    )
    tags = _row_likely_explanation_tags(row)
    assert "high_semantic_weak_shared_support" in tags


def test_aggregate_likely_explanations():
    df = pd.DataFrame(
        [
            {"semantic_cosine_max": 0.95, "n_shared_core_channels": 0},
            {"semantic_cosine_max": 0.92, "n_shared_core_channels": 0},
        ]
    )
    out = _aggregate_likely_explanations(df)
    assert out["n_pairs"] == 2
    assert out["top_tags"][0]["fraction"] >= 0.5


def test_row_minimal_explicit_support():
    weak = pd.Series(
        {
            "has_shared_sender": False,
            "has_shared_url": False,
            "n_shared_core_channels": 0,
            "source_count": 1,
        }
    )
    strong = pd.Series(
        {
            "has_shared_sender": True,
            "n_shared_core_channels": 2,
            "source_count": 2,
        }
    )
    assert _row_has_minimal_explicit_support(weak) is True
    assert _row_has_minimal_explicit_support(strong) is False


def test_semantic_body_path_coverage_summary():
    df_cross = pd.DataFrame(
        [
            {"semantic_cosine_max": 0.9, "body_only_token_jaccard": 0.1, "path_token_jaccard_combined": 0.2},
            {"body_only_token_jaccard": 0.2},
        ]
    )
    df_same = pd.DataFrame(
        [{"semantic_cosine_max": 0.8, "body_only_token_jaccard": 0.3, "path_token_jaccard_combined": 0.1}]
    )
    out = _build_semantic_body_path_coverage_summary(df_cross=df_cross, df_same=df_same)
    cross = out["by_cohort"]["high_cross_unlabeled"]
    assert cross["n_semantic_cosine_available"] == 1
    assert cross["n_body_only_features_available"] == 2
    assert cross["n_path_features_available"] == 1


def test_minimal_explicit_support_summary_fraction():
    base = {
        "from_semantic": False,
        "from_rare_artifact": False,
        "from_2hop": False,
        "from_component": False,
        "same_seed_component_flag": False,
        "cross_seed_component_flag": True,
    }
    df_cross = pd.DataFrame(
        [
            {
                **base,
                "email_i": "a",
                "email_j": "b",
                "has_shared_sender": False,
                "n_shared_core_channels": 0,
                "source_count": 1,
                "score": 0.9,
            },
            {
                **base,
                "email_i": "c",
                "email_j": "d",
                "has_shared_sender": True,
                "n_shared_core_channels": 1,
                "source_count": 2,
                "score": 0.85,
            },
        ]
    )
    out = _build_high_cross_minimal_explicit_support_summary(
        df_cross=df_cross,
        nodes_by_email={},
    )
    assert out["n_pairs"] == 1
    assert out["fraction_of_high_cross_cohort"] == 0.5


def test_high_cross_vs_high_same_digest_nonempty():
    comparisons = {
        "high_cross_vs_high_same": {
            "marginal": {
                "ranked_separators_top15": [
                    {
                        "metric_group": "shared_artifact_boolean",
                        "metric_name": "has_shared_sender",
                        "difference_left_minus_right": -0.2,
                    }
                ]
            },
            "body_path_signal_comparison": {"interpretation_notes": ["high_cross lower body_only"]},
        }
    }
    coverage = {
        "by_cohort": {
            "high_cross_unlabeled": _cohort_coverage_block(
                pd.DataFrame([{"semantic_cosine_max": np.nan}]), "high_cross_unlabeled"
            ),
            "high_same_unlabeled": _cohort_coverage_block(
                pd.DataFrame([{"semantic_cosine_max": 0.9}]), "high_same_unlabeled"
            ),
        }
    }
    digest = _build_high_cross_vs_high_same_digest(
        comparisons=comparisons,
        coverage_summary=coverage,
        minimal_support_summary={"fraction_of_high_cross_cohort": 0.6},
        profile={"readable_bullets": ["cross-seed dominant"]},
        likely_explanations={"top_tags": [{"tag": "high_latent_low_explicit", "fraction": 0.5}]},
    )
    assert digest["headline"]
    assert len(digest["distinguishing_factors"]) >= 2
