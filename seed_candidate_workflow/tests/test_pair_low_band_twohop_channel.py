"""Tests for low-band 2-hop channel analysis helpers."""

from __future__ import annotations

import pandas as pd

from seed_candidate_workflow.utils.pair_low_band_twohop_channel import (
    build_channel_summary_table,
    build_twohop_channel_recommendations,
    normalize_twohop_channel,
    twohop_records_from_evidence_list,
)


def test_normalize_twohop_channel_routing():
    assert normalize_twohop_channel(artifact_type="routing", path_type="email_routing_clue_email") == "routing"
    assert normalize_twohop_channel(artifact_type="html_structure_fingerprint", path_type="") == "html_fp"


def test_channel_summary_same_cross_split():
    df = pd.DataFrame(
        [
            {
                "email_i": "a",
                "email_j": "b",
                "gt_relation": "same_campaign",
                "from_2hop": True,
                "score": 0.1,
                "twohop_via_routing": True,
                "twohop_via_html_fp": False,
                "twohop_channels": "routing",
                "semantic_cosine_max": 0.5,
                "source_count": 1,
                "twohop_rarity_max": 1.2,
            },
            {
                "email_i": "c",
                "email_j": "d",
                "gt_relation": "cross_campaign",
                "from_2hop": True,
                "score": 0.05,
                "twohop_via_routing": True,
                "twohop_via_html_fp": False,
                "twohop_channels": "routing",
                "semantic_cosine_max": 0.4,
                "source_count": 1,
                "twohop_rarity_max": 1.1,
            },
            {
                "email_i": "e",
                "email_j": "f",
                "gt_relation": "same_campaign",
                "from_2hop": True,
                "score": 0.2,
                "twohop_via_routing": False,
                "twohop_via_html_fp": True,
                "twohop_channels": "html_fp",
                "semantic_cosine_max": 0.95,
                "source_count": 2,
                "twohop_rarity_max": 1.5,
            },
        ]
    )
    rows, meta = build_channel_summary_table(df)
    assert meta["n_same_low_unlabeled"] == 2
    routing = next(r for r in rows if r["twohop_channel"] == "routing")
    assert routing["n_same_low"] == 1
    assert routing["n_cross_low"] == 1
    html = next(r for r in rows if r["twohop_channel"] == "html_fp")
    assert html["n_same_low"] == 1
    recs = build_twohop_channel_recommendations(rows)
    assert "likely_too_noisy_for_2hop_generation" in recs or "potentially_useful_but_require_corroboration" in recs


def test_twohop_records_from_evidence_list():
    recs = twohop_records_from_evidence_list(
        [
            {"source_family": "2hop", "artifact_type": "stem", "path_type": "email_url_template_email"},
            {"source_family": "semantic", "cosine": 0.9},
        ]
    )
    assert len(recs) == 1
    assert recs[0]["twohop_channel"] == "stem"
