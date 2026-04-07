"""Smoke tests for teacher scoring and diagnostic helpers (no full Step 3 sweep)."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.utils.semantic_shard_edge_teacher_score import (
    TEACHER_WEIGHT_COL,
    build_teacher_scored_edges,
    save_teacher_bundle,
)
from analysis.utils.semantic_shard_method2_v2_diagnostics import (
    merge_edge_score_frame,
    pairwise_correlation_table,
    score_column_summary,
    topk_overlap_scores,
)


def _tiny_edges():
    return pd.DataFrame(
        [
            {
                "shard_a": "s0",
                "shard_b": "s1",
                "centroid_cosine": 0.9,
                "infra_score": 0.8,
                "temporal_score": 0.7,
                "shared_url_count": 1,
                "shared_url_idf_sum": 0.5,
                "infra_contrib_url": 0.5,
                "url_jaccard": 0.2,
                "edge_weight": 0.9,
            },
            {
                "shard_a": "s1",
                "shard_b": "s2",
                "centroid_cosine": 0.2,
                "infra_score": 0.9,
                "temporal_score": 0.1,
                "shared_url_count": 0,
                "shared_url_idf_sum": 0.0,
                "infra_contrib_url": 0.9,
                "url_jaccard": 0.0,
                "edge_weight": 0.5,
            },
        ]
    )


def test_teacher_scored_edges_columns():
    e = _tiny_edges()
    t = build_teacher_scored_edges(e)
    assert TEACHER_WEIGHT_COL in t.columns
    assert "view_semantic" in t.columns
    assert len(t) == 2
    assert t[TEACHER_WEIGHT_COL].between(0.0, 1.0).all()


def test_save_teacher_bundle(tmp_path):
    e = _tiny_edges()
    paths = save_teacher_bundle(e, run_id="pytest_teacher", output_root=str(tmp_path))
    assert Path(paths["scored_edges_csv"]).is_file()


def test_merge_and_correlations():
    e = _tiny_edges()
    t = build_teacher_scored_edges(e)
    v2 = e.copy()
    v2["edge_plausibility"] = [0.8, 0.3]
    m = merge_edge_score_frame(e, teacher_scored=t, v2_scored=v2)
    assert TEACHER_WEIGHT_COL in m.columns
    assert "edge_plausibility" in m.columns
    tab = pairwise_correlation_table(
        m, ["edge_weight", TEACHER_WEIGHT_COL, "edge_plausibility"]
    )
    assert len(tab) == 3
    s = score_column_summary(m[TEACHER_WEIGHT_COL], "teacher")
    assert "teacher__mean" in s
    o = topk_overlap_scores(
        m["edge_plausibility"].to_numpy(),
        m[TEACHER_WEIGHT_COL].to_numpy(),
        ks=[1, 2],
    )
    assert len(o) == 2
