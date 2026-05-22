"""Tests for candidate-family scorecard utilities."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from seed_candidate_workflow.utils.candidate_family_rules import eval_family_rule_expr
from seed_candidate_workflow.utils.candidate_family_scorecard import (
    CandidateFamilySpec,
    RecommendationThresholds,
    _assign_recommendation,
    _cohens_d,
    _learnability_block,
    _pairs_to_edges_df,
    score_one_family,
)
from seed_candidate_workflow.utils.candidate_family_scorecard_catalog import build_rich_v1_catalog


def test_cohens_d_separation():
    same = np.array([1.0, 1.1, 0.9, 1.05])
    cross = np.array([0.2, 0.3, 0.25, 0.15])
    d = _cohens_d(same, cross)
    assert d is not None
    assert d > 1.0


def test_assign_recommendation_promising():
    act, _reason = _assign_recommendation(
        n_new_same=20,
        n_new_cross=2,
        precision_like_new=0.9,
        cross_new_capture_rate=0.01,
        oracle_v_gain=0.02,
        graph_only_v_gain=0.005,
        learnability_score=0.4,
        already_in_graph=False,
        th=RecommendationThresholds(),
    )
    assert act == "promising_for_learning"


def test_assign_recommendation_too_clean():
    act, _reason = _assign_recommendation(
        n_new_same=20,
        n_new_cross=1,
        precision_like_new=0.95,
        cross_new_capture_rate=0.01,
        oracle_v_gain=0.02,
        graph_only_v_gain=0.019,
        learnability_score=0.4,
        already_in_graph=False,
        th=RecommendationThresholds(),
    )
    assert act == "too_clean_graph_only"


def test_score_one_family_synthetic(tmp_path: Path):
    gt_df = pd.DataFrame(
        {
            "email_i": ["a", "a", "b", "c"],
            "email_j": ["b", "c", "c", "d"],
            "gt_same_campaign": [True, False, False, False],
            "semantic_cosine": [0.95, 0.96, 0.70, 0.50],
            "has_shared_sender": [True, True, False, False],
            "has_shared_stem": [False, False, False, False],
            "n_shared_core_channels": [1, 1, 0, 0],
            "in_current_candidate_union": [False, False, False, False],
        }
    )
    gt_map = {"a": "C1", "b": "C1", "c": "C2", "d": "C3"}
    graph_pairs = {("a", "b")}  # only same-campaign edge in graph
    spec = CandidateFamilySpec(
        family_name="sem_sender",
        rule_expression="semantic_ge_0_90_AND_shared_sender",
        category="test",
    )
    oracle_baseline = {
        "homogeneity": 0.5,
        "completeness": 0.5,
        "v_measure": 0.5,
    }
    graph_baseline = {"homogeneity": 0.5, "completeness": 0.5, "v_measure": 0.5}

    class _Cfg:
        thresholds = RecommendationThresholds(
            min_new_same_pairs=1,
            weak_gain_max_new_same=0,
            min_oracle_v_gain=0.0,
            min_learnability_score=0.0,
        )
        community_method = "louvain"
        community_resolution = 1.0
        community_seed = 0

    row = score_one_family(
        spec,
        gt_file="gt.json",
        gt_df=gt_df,
        gt_map=gt_map,
        graph_pairs=graph_pairs,
        candidate_union_df=None,
        n_same_total=1,
        n_cross_total=3,
        oracle_baseline=oracle_baseline,
        graph_baseline=graph_baseline,
        cfg=_Cfg(),  # type: ignore[arg-type]
    )
    assert row["n_new_same_pairs"] >= 0
    assert "recommended_action" in row
    assert row["family_name"] == "sem_sender"


def test_eval_family_rule_with_duplicate_columns():
    base = pd.DataFrame(
        {
            "semantic_cosine_max": [0.95, 0.50],
            "has_shared_sender": [True, False],
        }
    )
    dup = pd.concat([base, base[["semantic_cosine_max"]]], axis=1)
    hits = eval_family_rule_expr(dup, "semantic_ge_0_90")
    assert int(hits.sum()) == 1


def test_pairs_to_edges_df_empty():
    df = _pairs_to_edges_df(set())
    assert list(df.columns) == ["email_a", "email_b", "edge_weight"]


def test_eval_family_rule_path_and_time():
    df = pd.DataFrame(
        {
            "path_token_jaccard_combined": [0.5, 0.1],
            "time_gap_seconds_min": [86400.0, 86400.0 * 10],
            "from_2hop": [True, False],
            "has_shared_sender": [True, False],
        }
    )
    m1 = eval_family_rule_expr(df, "path_token_jaccard_combined_ge_0_4")
    assert m1.tolist() == [True, False]
    m2 = eval_family_rule_expr(df, "time_gap_le_7d")
    assert m2.tolist() == [True, False]


def test_rich_catalog_size():
    families, skipped = build_rich_v1_catalog()
    assert len(families) >= 90
    assert len(skipped) >= 1
