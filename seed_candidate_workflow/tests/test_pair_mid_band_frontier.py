"""Tests for mid-band frontier score separation analysis."""

from __future__ import annotations

import numpy as np

from seed_candidate_workflow.utils.pair_mid_band_frontier import (
    MidBandThresholds,
    score_band_masks,
)


def test_score_band_masks_low_mid_disjoint_at_boundary():
    th = MidBandThresholds(low_score_max=0.15, mid_score_min=0.15, mid_score_max=0.50)
    scores = np.array([0.0, 0.15, 0.16, 0.50, 0.51, np.nan], dtype=float)
    bands = score_band_masks(scores, thresholds=th)
    assert bands["low"].tolist() == [True, True, False, False, False, False]
    assert bands["mid"].tolist() == [False, False, True, True, False, False]


def test_score_band_masks_community_cut_zone():
    th = MidBandThresholds(community_cut_score_min=0.30, community_cut_score_max=0.50)
    scores = np.array([0.29, 0.35, 0.50, 0.51], dtype=float)
    bands = score_band_masks(scores, thresholds=th)
    assert bands["community_cut"].tolist() == [False, True, True, False]
