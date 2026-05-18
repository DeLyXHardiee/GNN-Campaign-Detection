"""Tests for shared pair similarity features."""

from __future__ import annotations

from seed_candidate_workflow.utils.pair_similarity_features import (
    path_token_jaccard_combined_for_nodes,
    sender_localpart_norm_jaccard_for_nodes,
    sender_localpart_norm_similarity,
)


def test_path_token_jaccard_combined_shared_stem_and_url():
    na = {
        "url_set": {"https://evil.example.com/login/reset"},
        "stem_set": {"/login/reset"},
    }
    nb = {
        "url_set": {"https://evil.example.com/login/reset?id=1"},
        "stem_set": {"/login/reset"},
    }
    nc = {"url_set": {"https://other.example.com/foo"}, "stem_set": {"/foo"}}
    ab = path_token_jaccard_combined_for_nodes(na, nb)
    ac = path_token_jaccard_combined_for_nodes(na, nc)
    assert ab > ac
    assert ab > 0.0


def test_sender_localpart_norm_similarity():
    assert sender_localpart_norm_similarity("user123", "user456") == 1.0
    assert sender_localpart_norm_similarity("alice", "bob") < 0.5


def test_sender_localpart_norm_jaccard_for_nodes():
    na = {"sender_set": {"Alice <alice@corp.com>"}}
    nb = {"sender_set": {"Bob <alice@corp.com>"}}
    nc = {"sender_set": {"x@other.com"}}
    assert sender_localpart_norm_jaccard_for_nodes(na, nb) == 1.0
    assert sender_localpart_norm_jaccard_for_nodes(na, nc) == 0.0
