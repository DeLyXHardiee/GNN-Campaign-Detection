"""Tests for low-band manual review HTML helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from seed_candidate_workflow.utils.pair_score_separation import (
    _classify_low_band_review_regime,
    _export_low_band_pairs_for_manual_review,
    _low_band_review_prompt,
    _pair_metric_groups_html,
)
from seed_candidate_workflow.utils.pair_score_separation_output_layout import (
    ExportFlags,
    ensure_pair_score_separation_layout,
)


def test_classify_low_band_review_regime_score_zero_suffix():
    same_zero = pd.Series({"gt_relation": "same_campaign", "score": 0.0})
    assert _classify_low_band_review_regime(same_zero) == "same_campaign_low__score_zero"
    cross_pos = pd.Series({"gt_relation": "cross_campaign", "score": 0.12})
    assert _classify_low_band_review_regime(cross_pos) == "cross_campaign_low"


def test_low_band_review_prompt_by_relation():
    assert "same campaign" in _low_band_review_prompt(
        pd.Series({"gt_relation": "same_campaign"})
    ).lower()
    assert "cross campaign" in _low_band_review_prompt(
        pd.Series({"gt_relation": "cross_campaign"})
    ).lower()


def test_pair_metric_groups_html_includes_body_jaccard():
    row = pd.Series(
        {
            "body_token_jaccard": 0.42,
            "body_only_token_jaccard": 0.1,
            "semantic_cosine_max": 0.9,
            "semantic_cosine": 0.9,
            "source_count": 2,
        }
    )
    html_out = _pair_metric_groups_html(row)
    assert "body_token_jaccard" in html_out
    assert "body_only_token_jaccard" in html_out


def test_pair_metric_groups_semantic_cosine_uses_display_when_csv_max_missing():
    """2-hop-only rows often lack semantic_cosine_max; review uses semantic_cosine."""
    row = pd.Series(
        {
            "semantic_cosine_max": float("nan"),
            "semantic_cosine": 0.876,
            "semantic_cosine_for_display": 0.876,
            "semantic_cosine_source": "embedding_cache",
            "source_count": 1,
        }
    )
    html_out = _pair_metric_groups_html(row)
    assert "<strong>semantic_cosine</strong> 0.876" in html_out


def test_export_low_band_empty_writes_html(tmp_path: Path):
    layout = ensure_pair_score_separation_layout(tmp_path)
    out = _export_low_band_pairs_for_manual_review(
        df_low_band=pd.DataFrame(),
        layout=layout,
        email_text_by_eid={},
        email_text_meta={"status": "skipped"},
        preview_chars=80,
        wrap_width=40,
        low_score_max=0.4,
        export_flags=ExportFlags(),
    )
    html = out["paths"]["low_band_unlabeled_pairs_for_review_html"]
    assert Path(html).is_file()
    assert "review_html" in str(html).replace("\\", "/")
    text = Path(html).read_text(encoding="utf-8")
    assert "Low-score unlabeled pairs" in text
    assert "No pairs in this cohort" in text
