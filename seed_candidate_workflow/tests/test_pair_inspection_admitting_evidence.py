"""Tests for pair inspection admitting-evidence helpers."""

from __future__ import annotations

import json

import pandas as pd

from seed_candidate_workflow.utils.pair_inspection_admitting_evidence import (
    enrich_inspection_with_admitting_evidence,
    format_admitting_line,
    load_admitting_evidence_index,
    pair_key,
)


def test_format_2hop_admitting_line():
    line = format_admitting_line(
        {
            "source_family": "2hop",
            "artifact_type": "stem",
            "artifact_value": "/reset-password/",
            "path_type": "email_url_template_email",
            "artifact_degree": 4,
            "rarity_score": 1.27,
            "reason_code": "C1",
        }
    )
    assert "2hop via stem" in line
    assert "/reset-password/" in line
    assert "rarity=1.270" in line


def test_enrich_inspection_adds_2hop_when_index_present():
    pk = pair_key("a@x", "b@y")
    index = {
        pk: [
            {
                "source_family": "2hop",
                "artifact_type": "url",
                "artifact_value": "https://evil.example/login",
                "rarity_score": 1.1,
                "artifact_degree": 3,
            }
        ]
    }
    df = pd.DataFrame(
        [
            {
                "email_i": "a@x",
                "email_j": "b@y",
                "from_2hop": True,
                "from_semantic": False,
                "has_shared_url": False,
                "shared_url_values": "",
            }
        ]
    )
    out = enrich_inspection_with_admitting_evidence(df, evidence_index=index)
    assert "2hop via url" in out.iloc[0]["admitting_evidence_lines"]
    assert out.iloc[0]["shared_artifacts_brief"] != "none"
    recs = json.loads(out.iloc[0]["admitting_evidence_json"])
    assert recs[0]["artifact_type"] == "url"


def test_load_admitting_evidence_from_tmp_csvs(tmp_path):
    cand = tmp_path / "candidate_generation_test"
    cand.mkdir()
    hop = cand / "candidates_2hop.csv"
    pd.DataFrame(
        [
            {
                "email_i": "e1",
                "email_j": "e2",
                "intermediary_artifact_type": "sender_domain",
                "intermediary_artifact_value": "example.com",
                "intermediary_degree": 2,
                "rarity_score": 1.5,
                "path_type": "email_sender_domain_pattern_email",
                "reason_code": "C1",
            }
        ]
    ).to_csv(hop, index=False)
    index, meta = load_admitting_evidence_index(
        candidate_generation_dir=cand,
        seed_generation_dir=None,
    )
    assert meta["status"] == "ok"
    assert pair_key("e1", "e2") in index
