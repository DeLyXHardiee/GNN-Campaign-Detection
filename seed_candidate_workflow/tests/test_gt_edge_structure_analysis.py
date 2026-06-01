"""Tests for GT edge-structure analysis utilities."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from seed_candidate_workflow.utils.gt_edge_structure_analysis import (
    _candidate_rule_scorecard,
    _cmp_condition,
    _cosine_bucket_analysis,
    _eval_rule_expr,
    _generate_candidate_rule_recommendations,
    _rule_scorecard,
    _safe_enrichment,
)


def test_safe_enrichment():
    assert _safe_enrichment(0.4, 0.2) == pytest.approx(2.0)
    assert _safe_enrichment(0.4, 0.0) is None


def test_cmp_condition_counts():
    cond = np.array([True, True, False, True, False])
    same = np.array([True, True, True, False, False])
    cross = ~same
    stats = _cmp_condition(cond, same, cross)
    assert stats["same_rate"] == pytest.approx(2 / 3)
    assert stats["cross_rate"] == pytest.approx(0.5)
    assert stats["precision_like"] == pytest.approx(2 / 3)
    assert stats["support_total"] == 3


def test_eval_rule_semantic_and_sender():
    df = pd.DataFrame(
        {
            "has_shared_sender": [True, False, True],
            "semantic_cosine": [0.94, 0.96, 0.80],
            "gt_same_campaign": [True, True, False],
        }
    )
    mask = _eval_rule_expr(df, "semantic_ge_0_93_AND_shared_sender")
    assert mask.tolist() == [True, False, False]


def test_cosine_bucket_analysis():
    df = pd.DataFrame(
        {
            "semantic_cosine": [0.80, 0.88, 0.92, 0.96, 0.99],
            "gt_same_campaign": [True, True, True, False, False],
        }
    )
    buckets = (
        ("cosine_lt_0_85", None, 0.85),
        ("cosine_ge_0_95", 0.95, None),
    )
    analysis, rows = _cosine_bucket_analysis(df, buckets=buckets, gt_file="gt.json")
    assert len(analysis["buckets"]) == 2
    assert len(rows) == 2
    low_bucket = next(b for b in analysis["buckets"] if b["bucket"] == "cosine_lt_0_85")
    assert low_bucket["support_total"] == 1


def test_rule_scorecard_min_support():
    n = 100
    df = pd.DataFrame(
        {
            "has_shared_sender": [True] * 50 + [False] * 50,
            "gt_same_campaign": [True] * 60 + [False] * 40,
            "semantic_cosine": np.linspace(0.7, 0.99, n),
        }
    )
    rules = [("shared_sender", "shared_sender")]
    scorecard, _ = _rule_scorecard(
        df, rules, gt_file="t.json", min_support=10
    )
    assert len(scorecard) == 1
    assert scorecard[0]["support_total"] == 50


def test_candidate_rule_scorecard_novelty_metrics():
    df = pd.DataFrame(
        {
            "email_i": ["a", "b", "c", "d", "e", "f"],
            "email_j": ["b", "c", "d", "e", "f", "g"],
            "gt_same_campaign": [True, True, True, False, False, False],
            "has_shared_sender": [True, True, False, True, False, False],
            "has_shared_stem": [False, True, True, False, False, True],
            "has_shared_sender_domain": [False] * 6,
            "has_shared_url": [False] * 6,
            "has_shared_domain": [False] * 6,
            "has_shared_attachment": [False] * 6,
            "n_shared_core_channels": [1, 2, 1, 1, 0, 1],
            "semantic_cosine": [0.96, 0.91, 0.94, 0.92, 0.70, 0.86],
            "from_semantic": [False, True, False, True, False, False],
            "from_2hop": [False, False, False, False, True, False],
            "from_component": [False, False, False, False, False, True],
            "from_rare_artifact": [False] * 6,
            "from_shared_stem_highconf": [False] * 6,
            "in_current_candidate_union": [True, True, False, True, False, False],
        }
    )
    scorecard, _ = _candidate_rule_scorecard(df, gt_file="gt.json", min_support=1)
    sem93 = next(r for r in scorecard if r["rule_name"] == "semantic_ge_0_93")
    assert sem93["same_pairs_captured"] == 2
    assert sem93["same_pairs_new_not_in_union"] == 1
    assert sem93["precision_like_new"] == pytest.approx(1.0)

    recs = _generate_candidate_rule_recommendations(
        scorecard, min_new_same_pairs=1, union_joined=True
    )
    assert "recommended_seed_like_additions" in recs
    assert "recommended_candidate_broadening" in recs


def test_eval_rule_semantic_band_and_n_shared():
    df = pd.DataFrame(
        {
            "has_shared_sender": [True, False],
            "semantic_cosine": [0.87, 0.91],
            "n_shared_core_channels": [1, 2],
            "gt_same_campaign": [True, False],
        }
    )
    band = _eval_rule_expr(df, "semantic_band_0_85_0_90_AND_shared_sender")
    assert band.tolist() == [True, False]
    n2 = _eval_rule_expr(df, "n_shared_core_channels_ge_2")
    assert n2.tolist() == [False, True]


def test_summary_schema_keys(tmp_path: Path):
    """Light integration: analyze_gt_file with mocked pair dataframe path."""
    from seed_candidate_workflow.utils import gt_edge_structure_analysis as mod

    df = pd.DataFrame(
        {
            "email_i": ["a", "b", "c", "d"],
            "email_j": ["b", "c", "d", "e"],
            "gt_same_campaign": [True, True, False, False],
            "has_shared_sender": [True, False, True, False],
            "has_shared_stem": [False, True, False, False],
            "has_shared_sender_domain": [False, False, True, True],
            "has_shared_url": [False, False, False, False],
            "has_shared_domain": [False, False, False, False],
            "has_shared_attachment": [True, False, False, False],
            "has_shared_received_host": [False, False, False, False],
            "has_shared_return_path_domain": [False, False, False, False],
            "has_shared_origin_ip": [False, False, False, False],
            "n_shared_core_channels": [1, 1, 1, 1],
            "semantic_cosine": [0.96, 0.88, 0.91, 0.70],
        }
    )

    same_mask, cross_mask = mod._masks_from_df(df)
    assert int(same_mask.sum()) == 2
    assert int(cross_mask.sum()) == 2

    core, _ = mod._channel_marginals(df, channels=["sender"], gt_file="gt.json")
    assert core[0]["channel"] == "sender"

    recs = mod._generate_recommendations(
        channel_marginals=core,
        cosine_analysis={"contamination_slope_note": "test"},
        scorecard=[],
        joint={"top_by_enrichment": []},
        frontier={"ambiguous_overlap": []},
        config_audit={},
    )
    assert "seed_recommendations" in recs
