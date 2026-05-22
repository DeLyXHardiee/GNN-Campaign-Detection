"""Tests for body-similarity and sender-localpart candidate generators."""

from __future__ import annotations

import numpy as np
import pandas as pd

from seed_candidate_workflow.utils.anchor_candidate_body_similarity_helpers import (
    generate_body_char4gram_jaccard_highconf_v1,
    generate_body_token_jaccard_highconf_v1,
)
from seed_candidate_workflow.utils.anchor_candidate_semantic_mid_support_helpers import (
    generate_semantic_mid_senderlocalpart_support_v1,
)
from seed_candidate_workflow.utils.pair_similarity_features import (
    body_char4gram_jaccard_from_bodies,
    body_token_jaccard_from_bodies,
)


def test_body_token_jaccard_highconf_emits_shared_body_pair():
    body = "alpha beta gamma delta epsilon zeta eta theta iota"
    catalog = {
        "a": {"body": body},
        "b": {"body": body + " extra"},
        "c": {"body": "completely unrelated vocabulary here"},
    }
    nodes = pd.DataFrame({"external_id": ["a", "b", "c"]})
    df, diag = generate_body_token_jaccard_highconf_v1(
        nodes_df=nodes,
        generator_cfg={
            "min_body_token_jaccard": 0.25,
            "max_candidate_rows": 1000,
            "use_body_similarity_cache": False,
            "use_filtered_inverted_index": True,
            "show_progress": False,
        },
        project_root=None,
        text_catalog=catalog,
    )
    assert diag["status"] == "ok"
    assert not df.empty
    pairs = {tuple(sorted(x)) for x in zip(df["email_i"], df["email_j"], strict=False)}
    assert ("a", "b") in pairs
    assert body_token_jaccard_from_bodies(body, body + " extra") >= 0.25


def test_body_char4gram_jaccard_highconf_threshold():
    body = "aaaaaaaaaaaaaaaa"
    catalog = {"a": {"body": body}, "b": {"body": body}}
    nodes = pd.DataFrame({"external_id": ["a", "b"]})
    df, diag = generate_body_char4gram_jaccard_highconf_v1(
        nodes_df=nodes,
        generator_cfg={
            "min_body_char4gram_jaccard": 0.25,
            "max_candidate_rows": 100,
            "use_body_similarity_cache": False,
            "show_progress": False,
        },
        project_root=None,
        text_catalog=catalog,
    )
    assert diag["status"] == "ok"
    assert len(df) == 1
    assert float(df.iloc[0]["body_char4gram_jaccard"]) == 1.0


def test_body_generators_use_prior_pool_when_provided():
    catalog = {
        "a": {"body": "shared campaign body text alpha beta gamma"},
        "b": {"body": "shared campaign body text alpha beta gamma delta"},
        "c": {"body": "unrelated vocabulary completely different"},
    }
    nodes = pd.DataFrame({"external_id": ["a", "b", "c"]})
    prior = {tuple(sorted(("a", "b")))}
    df, diag = generate_body_token_jaccard_highconf_v1(
        nodes_df=nodes,
        generator_cfg={
            "min_body_token_jaccard": 0.25,
            "max_candidate_rows": 100,
            "use_body_similarity_cache": False,
            "use_filtered_inverted_index": False,
            "show_progress": False,
        },
        project_root=None,
        text_catalog=catalog,
        prior_pair_pool=prior,
    )
    assert diag["n_prior_pool_pairs"] == 1
    pairs = {tuple(sorted(x)) for x in zip(df["email_i"], df["email_j"], strict=False)}
    assert ("a", "b") in pairs
    assert ("a", "c") not in pairs and ("b", "c") not in pairs


def test_semantic_mid_senderlocalpart_requires_norm_similarity():
    nodes = pd.DataFrame(
        [
            {"external_id": "a", "sender_set": {"Alice <user123@corp.com>"}},
            {"external_id": "b", "sender_set": {"Bob <user456@corp.com>"}},
            {"external_id": "c", "sender_set": {"x@other.com"}},
        ]
    )
    v = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    id_to_vec = {
        "a": np.concatenate([v, np.zeros(2, dtype=np.float32)]),
        "b": np.concatenate([v * 0.88, np.zeros(2, dtype=np.float32)]),
        "c": np.concatenate([v * 0.87, np.zeros(2, dtype=np.float32)]),
    }
    for k in list(id_to_vec):
        id_to_vec[k] = id_to_vec[k] / np.linalg.norm(id_to_vec[k])

    df, diag = generate_semantic_mid_senderlocalpart_support_v1(
        nodes_df=nodes,
        id_to_vec=id_to_vec,
        generator_cfg={
            "semantic_top_k": 10,
            "semantic_min_cos": 0.85,
            "semantic_max_cos_exclusive": 0.90,
            "min_sender_localpart_norm_jaccard": 0.7,
        },
    )
    if not df.empty:
        assert (df["cosine"] >= 0.85).all()
        assert (df["cosine"] < 0.90).all()
        assert (df["sender_localpart_norm_jaccard"] >= 0.7).all()
    assert diag["min_sender_localpart_norm_jaccard"] == 0.7
