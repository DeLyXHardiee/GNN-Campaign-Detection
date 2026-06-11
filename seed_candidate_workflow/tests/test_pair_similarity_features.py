"""Tests for shared pair similarity features."""

from __future__ import annotations

from seed_candidate_workflow.utils.pair_similarity_features import (
    RESCUE_ALIGNED_SCORER_FEATURE_COLS,
    SCORER_PAIR_NUMERIC_FEATURE_COLS,
    attach_path_jaccard_features_to_dataframe,
    body_char4gram_jaccard_from_bodies,
    body_only_char4gram_jaccard_from_bodies,
    body_only_token_jaccard_from_bodies,
    body_token_jaccard_from_bodies,
    compute_body_only_pair_features,
    path_jaccard_features_for_node_pair,
    path_token_jaccard_combined_for_nodes,
    sender_localpart_norm_jaccard_for_nodes,
    sender_localpart_norm_similarity,
    strip_url_like_tokens_from_body,
)


def test_path_jaccard_features_for_node_pair():
    na = {"url_set": set(), "stem_set": {"/login/reset"}}
    nb = {"url_set": set(), "stem_set": {"/login/reset", "/login/reset-extra"}}
    feats = path_jaccard_features_for_node_pair(na, nb)
    assert feats["path_token_jaccard_combined"] is not None
    assert feats["url_path_token_jaccard"] is not None
    assert feats["stem_path_token_jaccard"] is not None
    assert feats["path_token_jaccard_combined"] > 0.0


def test_attach_path_jaccard_features_to_dataframe():
    import pandas as pd

    nodes = {
        "a": {"url_set": set(), "stem_set": {"/a/b"}},
        "b": {"url_set": set(), "stem_set": {"/a/b", "/a/b/c"}},
    }
    df = pd.DataFrame({"email_i": ["a"], "email_j": ["b"]})
    out = attach_path_jaccard_features_to_dataframe(df, nodes_by_email=nodes)
    assert out["path_token_jaccard_combined"].notna().iloc[0]
    assert out["url_path_token_jaccard"].notna().iloc[0]


def test_path_token_jaccard_combined_shared_stem_and_url():
    na = {"url_set": set(), "stem_set": {"/login/reset"}}
    nb = {"url_set": set(), "stem_set": {"/login/reset"}}
    nc = {"url_set": set(), "stem_set": {"/foo"}}
    ab = path_token_jaccard_combined_for_nodes(na, nb)
    ac = path_token_jaccard_combined_for_nodes(na, nc)
    assert ab > ac
    assert ab > 0.0


def test_sender_localpart_norm_similarity():
    assert sender_localpart_norm_similarity("user123", "user456") == 1.0
    assert sender_localpart_norm_similarity("alice", "bob") < 0.5


def test_body_token_jaccard_identical_bodies():
    body = "Hello world this is a test email body with enough tokens"
    assert body_token_jaccard_from_bodies(body, body) == 1.0


def test_body_char4gram_jaccard_empty_default():
    assert body_char4gram_jaccard_from_bodies("", "short") == 0.0


def test_sender_localpart_norm_jaccard_for_nodes():
    na = {"sender_set": {"Alice <alice@corp.com>"}}
    nb = {"sender_set": {"Bob <alice@corp.com>"}}
    nc = {"sender_set": {"x@other.com"}}
    assert sender_localpart_norm_jaccard_for_nodes(na, nb) == 1.0
    assert sender_localpart_norm_jaccard_for_nodes(na, nc) == 0.0


def test_strip_url_like_tokens_from_body():
    body = "Please visit https://evil.com/foo/bar?id=1 now"
    stripped = strip_url_like_tokens_from_body(body)
    assert "https://" not in stripped
    assert "evil.com" not in stripped
    assert "please" in stripped.lower()


def test_body_only_ignores_url_tokens():
    b1 = "hello world https://evil.com/foo/bar"
    b2 = "hello world https://other.net/baz/qux"
    assert strip_url_like_tokens_from_body(b1) == strip_url_like_tokens_from_body(b2)
    assert body_only_token_jaccard_from_bodies(b1, b2) == 1.0
    assert body_token_jaccard_from_bodies(b1, b2) < 1.0


def test_compute_body_only_pair_features():
    catalog = {
        "e1": {"body": "hello https://x.com/a world"},
        "e2": {"body": "hello https://x.com/b world"},
    }
    feats = compute_body_only_pair_features(
        email_i="e1", email_j="e2", text_catalog=catalog
    )
    assert feats["body_only_token_jaccard"] == 1.0
    assert feats["body_only_char4gram_jaccard"] >= 0.0


def test_scorer_pair_numeric_feature_cols_include_rescue_aligned():
    for col in RESCUE_ALIGNED_SCORER_FEATURE_COLS:
        assert col in SCORER_PAIR_NUMERIC_FEATURE_COLS


def test_pair_train_numeric_cols_include_rescue_aligned():
    from pathlib import Path

    src = Path("core/GNN/src/pair_train.py").read_text(encoding="utf-8")
    for col in RESCUE_ALIGNED_SCORER_FEATURE_COLS:
        assert f'"{col}"' in src
    assert '"body_token_jaccard"' in src
    assert '"body_char4gram_jaccard"' in src
