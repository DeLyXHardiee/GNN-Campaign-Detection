"""Tests for bridge candidate retrieval helpers."""

from __future__ import annotations

from seed_candidate_workflow.utils.bridge_candidate_experiment import (
    canonical_pair,
    load_connected_pair_keys,
    merge_retrieval_hits,
)


def test_canonical_pair_orders_and_rejects_self():
    assert canonical_pair("b", "a") == ("a", "b")
    assert canonical_pair("x", "x") is None


def test_merge_retrieval_hits_unions_channels():
    a = {
        ("e1", "e2"): {
            "retrieval_channels": {"semantic"},
            "retrieval_semantic_cosine": 0.9,
            "retrieval_body_only_token_jaccard": None,
            "retrieval_path_token_jaccard": None,
        }
    }
    b = {
        ("e1", "e2"): {
            "retrieval_channels": {"body_only"},
            "retrieval_semantic_cosine": None,
            "retrieval_body_only_token_jaccard": 0.3,
            "retrieval_path_token_jaccard": None,
        }
    }
    m = merge_retrieval_hits(a, b)
    assert m[("e1", "e2")]["retrieval_channels"] == {"semantic", "body_only"}
    assert m[("e1", "e2")]["retrieval_semantic_cosine"] == 0.9
    assert m[("e1", "e2")].get("retrieval_semantic_rank") is None


def test_load_connected_pair_keys_from_csvs(tmp_path):
    cu = tmp_path / "candidate_union.csv"
    cu.write_text("email_i,email_j\na,b\n", encoding="utf-8")
    connected = load_connected_pair_keys(candidate_union_csv=cu, seed_edges_csv=None)
    assert ("a", "b") in connected
