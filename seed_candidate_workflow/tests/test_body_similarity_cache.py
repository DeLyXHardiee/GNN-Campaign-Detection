"""Tests for persistent body similarity email-level cache."""

from __future__ import annotations

from pathlib import Path

from seed_candidate_workflow.utils.body_similarity_cache import (
    build_or_load_email_body_feature_store,
    manifest_is_fresh,
)


def test_body_similarity_cache_round_trip(tmp_path: Path):
    catalog = {
        "e1": {"body": "alpha beta gamma delta"},
        "e2": {"body": "alpha beta gamma delta epsilon"},
    }
    misp = tmp_path / "fake_misp.json"
    misp.write_text("{}", encoding="utf-8")

    store1, diag1 = build_or_load_email_body_feature_store(
        email_ids=["e1", "e2"],
        text_catalog=catalog,
        graph_id="test_graph",
        misp_json_path=misp,
        cache_root=tmp_path / "cache",
    )
    assert diag1["cache_status"] == "rebuilt"
    assert store1.token_jaccard("e1", "e2") > 0.0

    store2, diag2 = build_or_load_email_body_feature_store(
        email_ids=["e1", "e2"],
        text_catalog=catalog,
        graph_id="test_graph",
        misp_json_path=misp,
        cache_root=tmp_path / "cache",
    )
    assert diag2["cache_status"] == "hit"
    assert store2.token_jaccard("e1", "e2") == store1.token_jaccard("e1", "e2")


def test_body_similarity_cache_reuses_across_graph_ids(tmp_path: Path):
    """Content fingerprint ignores graph_id so different bundle ids share one cache."""
    catalog = {
        "e1": {"body": "hello world"},
        "e2": {"body": "hello there"},
    }
    misp = tmp_path / "fake_misp.json"
    misp.write_text("{}", encoding="utf-8")
    cache_root = tmp_path / "cache"

    _, diag_a = build_or_load_email_body_feature_store(
        email_ids=["e1", "e2"],
        text_catalog=catalog,
        graph_id="graph_bundle_A",
        misp_json_path=misp,
        cache_root=cache_root,
    )
    assert diag_a["cache_status"] == "rebuilt"

    _, diag_b = build_or_load_email_body_feature_store(
        email_ids=["e1", "e2"],
        text_catalog=catalog,
        graph_id="graph_bundle_B",
        misp_json_path=misp,
        cache_root=cache_root,
    )
    assert diag_b["cache_status"] == "hit"
    assert "by_content" in str(diag_b.get("cache_dir_resolved") or diag_b.get("cache_dir") or "")


def test_body_similarity_cache_invalidates_on_email_set_change(tmp_path: Path):
    catalog = {"e1": {"body": "hello world"}}
    misp = tmp_path / "fake_misp.json"
    misp.write_text("{}", encoding="utf-8")
    cache_root = tmp_path / "cache"

    _, diag1 = build_or_load_email_body_feature_store(
        email_ids=["e1"],
        text_catalog=catalog,
        graph_id="g",
        misp_json_path=misp,
        cache_root=cache_root,
    )
    assert diag1["cache_status"] == "rebuilt"

    _, diag2 = build_or_load_email_body_feature_store(
        email_ids=["e1", "e2"],
        text_catalog={**catalog, "e2": {"body": "other text"}},
        graph_id="g",
        misp_json_path=misp,
        cache_root=cache_root,
    )
    assert diag2["cache_status"] == "rebuilt"


def test_manifest_is_fresh_requires_matching_misp_fingerprint(tmp_path: Path):
    misp = tmp_path / "misp.json"
    misp.write_text("{}", encoding="utf-8")
    from seed_candidate_workflow.utils.body_similarity_cache import cache_manifest_payload

    expected = cache_manifest_payload(
        graph_id="g",
        misp_json_path=misp,
        email_ids=["a"],
        min_token_len=2,
        char_n=4,
    )
    stale = dict(expected)
    stale["email_id_hash"] = "deadbeef"
    assert manifest_is_fresh(stale, expected=expected) is False
    assert manifest_is_fresh(expected, expected=expected) is True
