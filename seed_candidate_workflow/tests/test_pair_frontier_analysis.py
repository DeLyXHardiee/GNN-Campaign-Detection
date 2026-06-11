"""Tests for full pair frontier analysis (low/mid/high unlabeled cohorts)."""

from __future__ import annotations

import numpy as np

from seed_candidate_workflow.utils.pair_frontier_analysis import (
    COHORT_NAMES,
    FrontierThresholds,
    frontier_unlabeled_cohort_masks,
)
from seed_candidate_workflow.utils.pair_mid_band_frontier import score_band_masks


def test_frontier_cohort_masks_six_buckets():
    th = FrontierThresholds(
        low_score_max=0.15,
        mid_score_min=0.15,
        mid_score_max=0.50,
        high_score_min=0.80,
    )
    scores = np.array([0.10, 0.20, 0.85, 0.12, 0.40, 0.90], dtype=float)
    same = np.array([True, True, True, False, False, False], dtype=bool)
    cross = ~same
    unl = np.ones(6, dtype=bool)
    masks = frontier_unlabeled_cohort_masks(
        same_eval=same, cross_eval=cross, unl_eval=unl, scores=scores, thresholds=th
    )
    assert set(masks.keys()) == set(COHORT_NAMES)
    assert masks["low_same_unlabeled"].tolist() == [True, False, False, False, False, False]
    assert masks["mid_same_unlabeled"].tolist() == [False, True, False, False, False, False]
    assert masks["high_same_unlabeled"].tolist() == [False, False, True, False, False, False]
    assert masks["low_cross_unlabeled"].tolist() == [False, False, False, True, False, False]
    assert masks["mid_cross_unlabeled"].tolist() == [False, False, False, False, True, False]
    assert masks["high_cross_unlabeled"].tolist() == [False, False, False, False, False, True]


def test_high_band_aligns_with_score_band_masks():
    th = FrontierThresholds(high_score_min=0.80)
    scores = np.array([0.79, 0.81], dtype=float)
    bands = score_band_masks(scores, thresholds=th)
    assert bands["high"].tolist() == [False, True]
