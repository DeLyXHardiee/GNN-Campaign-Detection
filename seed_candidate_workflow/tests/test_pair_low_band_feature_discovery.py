"""Unit tests for low-band feature discovery stats helpers."""

from __future__ import annotations

import pandas as pd

from seed_candidate_workflow.utils.pair_low_band_feature_discovery import (
    _alignment_from_means,
    _compare_feature_series,
    _combined_recommendation_score,
    _jaccard,
    _threshold_scorecard,
    _verdict_from_diff,
)


def test_jaccard_empty():
    assert _jaccard(set(), set()) is None
    assert _jaccard({"a"}, {"a"}) == 1.0


def test_verdict_from_diff():
    assert _verdict_from_diff(0.1) == "promising_same_enriched"
    assert _verdict_from_diff(-0.1) == "promising_cross_enriched"
    assert _verdict_from_diff(0.01) == "weak_separator"


def test_compare_boolean_feature():
    same = pd.Series([1, 1, 0, 1])
    cross = pd.Series([0, 0, 1, 0])
    row = _compare_feature_series(same, cross, feature="subject_normalized_exact_match", family="subject")
    assert row["same_mean"] == 0.75
    assert row["cross_mean"] == 0.25
    assert row["difference_mean_same_minus_cross"] == 0.5
    assert row["verdict"] == "promising_same_enriched"


def test_threshold_scorecard_lift():
    same = pd.Series([0.2, 0.5, 0.8, 0.9])
    cross = pd.Series([0.1, 0.15, 0.2, 0.25])
    rows = _threshold_scorecard(same, cross, feature="subject_token_jaccard", thresholds=[0.5])
    assert len(rows) == 1
    assert rows[0]["n_same_captured"] == 3
    assert rows[0]["n_cross_captured"] == 0


def test_alignment_margin_closer_to_positive():
    align = _alignment_from_means(
        low_same_mean=0.8,
        positive_mean=0.85,
        cross_unlabeled_mean=0.1,
        low_cross_mean=0.05,
    )
    assert align["low_same_is_closer_to_positive_than_to_cross_unlabeled"] is True
    assert align["alignment_margin_vs_cross_unlabeled"] > 0

    score = _combined_recommendation_score(
        low_band_separation=0.7,
        alignment_margin=align["alignment_margin_vs_cross_unlabeled"],
        positive_mean=0.85,
        low_same_mean=0.8,
        cross_unlabeled_mean=0.1,
        low_cross_mean=0.05,
    )
    assert score is not None and score > 0
