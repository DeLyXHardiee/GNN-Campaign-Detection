"""Tests for low-band manual review HTML helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from seed_candidate_workflow.utils.pair_score_separation import (
    _classify_low_band_review_regime,
    _export_low_band_pairs_for_manual_review,
    _low_band_review_prompt,
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


def test_export_low_band_empty_writes_html(tmp_path: Path):
    out = _export_low_band_pairs_for_manual_review(
        df_low_band=pd.DataFrame(),
        out_dir=tmp_path,
        email_text_by_eid={},
        email_text_meta={"status": "skipped"},
        preview_chars=80,
        wrap_width=40,
        low_score_max=0.4,
    )
    html = out["paths"]["low_band_unlabeled_pairs_for_review_html"]
    assert Path(html).is_file()
    text = Path(html).read_text(encoding="utf-8")
    assert "Low-score unlabeled pairs" in text
