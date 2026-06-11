"""Tests for bridge candidate explainability helpers."""

from __future__ import annotations

import pandas as pd

from seed_candidate_workflow.utils.bridge_candidate_experiment import merge_retrieval_hits
from seed_candidate_workflow.utils.bridge_candidate_review import (
    _score_band,
    build_bridge_band_analysis,
    build_bridge_suspicious_high_score_analysis,
    bridge_feature_population_diagnostics,
    load_misp_node_sets_by_email,
)


def test_merge_retrieval_hits_min_rank():
    a = {
        ("e1", "e2"): {
            "retrieval_channels": {"semantic"},
            "retrieval_semantic_cosine": 0.88,
            "retrieval_body_only_token_jaccard": None,
            "retrieval_path_token_jaccard": None,
            "retrieval_semantic_rank": 5,
            "retrieval_body_only_rank": None,
            "retrieval_path_rank": None,
        }
    }
    b = {
        ("e1", "e2"): {
            "retrieval_channels": {"semantic"},
            "retrieval_semantic_cosine": 0.9,
            "retrieval_body_only_token_jaccard": None,
            "retrieval_path_token_jaccard": None,
            "retrieval_semantic_rank": 2,
            "retrieval_body_only_rank": None,
            "retrieval_path_rank": None,
        }
    }
    m = merge_retrieval_hits(a, b)
    assert m[("e1", "e2")]["retrieval_semantic_rank"] == 2
    assert m[("e1", "e2")]["retrieval_semantic_cosine"] == 0.9


def test_score_band_labels():
    assert _score_band(0.95) == "high_bridge"
    assert _score_band(0.6) == "mid_bridge"
    assert _score_band(0.1) == "low_bridge"


def test_build_bridge_band_analysis_cohorts():
    df = pd.DataFrame(
        {
            "email_i": ["a", "b", "c"],
            "email_j": ["d", "e", "f"],
            "score": [0.95, 0.6, 0.1],
            "retrieval_channels": ["semantic", "body_only", "path"],
            "n_retrieval_channels": [1, 1, 1],
            "embedding_cosine_subj_body": [0.9, 0.5, 0.2],
        }
    )
    out = build_bridge_band_analysis(df)
    assert out["cohorts"]["high_bridge"]["n_pairs"] == 1
    assert out["cohorts"]["low_bridge"]["n_pairs"] == 1


def test_bridge_feature_population_diagnostics():
    df = pd.DataFrame(
        {
            "email_i": ["a"],
            "email_j": ["b"],
            "score": [0.5],
            "retrieval_channels": ["semantic"],
            "retrieval_semantic_rank": [3],
            "has_shared_sender": [False],
            "shared_sender_values": [""],
        }
    )
    diag = bridge_feature_population_diagnostics(df)
    assert diag["n_bridge_pairs"] == 1
    assert diag["retrieval_provenance_columns"]["retrieval_semantic_rank"]["present"] is True


def test_load_misp_node_sets_parses_url_dict_components(monkeypatch, tmp_path):
    """parse_url_components returns dict keys (domain/stem), not attribute objects."""
    fake_events = [
        {
            "external_id": "ext@test",
            "senders": ["user@Example.COM"],
            "urls": ["https://sub.example.com/path/to/file"],
            "attachments": [],
        }
    ]

    def _fake_load(_path):
        return fake_events

    def _fake_parse(events):
        return events

    monkeypatch.setattr(
        "analysis.scripts.misp_email_text_catalog.load_misp_events_list",
        _fake_load,
    )
    monkeypatch.setattr("graph.common.parse_misp_events", _fake_parse)

    misp_json = tmp_path / "fake_misp.json"
    misp_json.write_text("[]", encoding="utf-8")
    nodes, meta = load_misp_node_sets_by_email(
        project_root=tmp_path,
        misp_json_path=misp_json,
    )
    assert meta.get("available") is True
    assert "ext@test" in nodes
    assert "example.com" in nodes["ext@test"]["domain_set"]
    assert nodes["ext@test"]["stem_set"]


def test_suspicious_high_score_splits_latent_explained():
    df = pd.DataFrame(
        {
            "email_i": ["a", "b", "c"],
            "email_j": ["d", "e", "f"],
            "score": [0.95, 0.92, 0.91],
            "retrieval_channels": ["semantic", "semantic", "body_only"],
            "semantic_cosine_max": [0.1, 0.1, 0.1],
            "body_token_jaccard": [0.0, 0.0, 0.0],
            "path_token_jaccard_combined": [0.0, 0.0, 0.0],
            "n_shared_core_channels": [0, 0, 0],
            "scorer_encoder_cosine": [0.9, 0.2, 0.3],
            "embedding_cosine_subj_body": [0.85, 0.1, 0.1],
        }
    )
    out = build_bridge_suspicious_high_score_analysis(df)
    assert out["n_high_score"] == 3
    assert out["latent_explained_high_score"]["n"] >= 1
    assert out["suspicious_weak_explicit_and_latent"]["n"] >= 1
