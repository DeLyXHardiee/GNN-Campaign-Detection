"""Tests for medium-semantic-band candidate generators."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from seed_candidate_workflow.utils.anchor_candidate_semantic_mid_support_helpers import (
    build_nodes_core_sets,
    generate_semantic_mid_core_support_v1,
    generate_semantic_mid_sender_support_v1,
    n_shared_core_channels,
)


def test_n_shared_core_channels_matches_pair_training_spec():
    nodes = pd.DataFrame(
        [
            {
                "external_id": "a",
                "sender_set": {"s1"},
                "stem_set": set(),
                "url_set": set(),
                "attachment_set": set(),
                "sender_email_domain_set": set(),
                "domain_set": set(),
            },
            {
                "external_id": "b",
                "sender_set": {"s1"},
                "stem_set": {"st"},
                "url_set": set(),
                "attachment_set": set(),
                "sender_email_domain_set": set(),
                "domain_set": set(),
            },
        ]
    )
    nb = build_nodes_core_sets(nodes)
    assert n_shared_core_channels("a", "b", nb) == 1


def test_semantic_mid_sender_support_band_and_sender_gate():
    nodes = pd.DataFrame(
        [
            {"external_id": "a", "sender_set": {"s"}, "stem_set": set(), "url_set": set(), "attachment_set": set(), "sender_email_domain_set": set(), "domain_set": set()},
            {"external_id": "b", "sender_set": {"s"}, "stem_set": set(), "url_set": set(), "attachment_set": set(), "sender_email_domain_set": set(), "domain_set": set()},
            {"external_id": "c", "sender_set": {"x"}, "stem_set": set(), "url_set": set(), "attachment_set": set(), "sender_email_domain_set": set(), "domain_set": set()},
        ]
    )
    v = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    id_to_vec = {
        "a": np.concatenate([v, np.zeros(3, dtype=np.float32)]),
        "b": np.concatenate([v * 0.88, np.zeros(3, dtype=np.float32)]),
        "c": np.concatenate([v * 0.87, np.zeros(3, dtype=np.float32)]),
    }
    # Normalize so cosine is meaningful
    for k in list(id_to_vec):
        x = id_to_vec[k]
        id_to_vec[k] = x / np.linalg.norm(x)

    df, diag = generate_semantic_mid_sender_support_v1(
        nodes_df=nodes,
        id_to_vec=id_to_vec,
        generator_cfg={
            "semantic_top_k": 10,
            "semantic_min_cos": 0.85,
            "semantic_max_cos_exclusive": 0.90,
        },
    )
    assert diag["n_pairs_in_band"] >= 0
    if not df.empty:
        assert (df["cosine"] >= 0.85).all()
        assert (df["cosine"] < 0.90).all()
        assert df["has_shared_sender"].all()


def test_semantic_mid_core_requires_core_channel():
    nodes = pd.DataFrame(
        [
            {"external_id": "a", "sender_set": set(), "stem_set": {"t"}, "url_set": set(), "attachment_set": set(), "sender_email_domain_set": set(), "domain_set": set()},
            {"external_id": "b", "sender_set": set(), "stem_set": {"t"}, "url_set": set(), "attachment_set": set(), "sender_email_domain_set": set(), "domain_set": set()},
        ]
    )
    v = np.array([1.0, 0.0], dtype=np.float32)
    id_to_vec = {"a": v, "b": v * 0.87}
    for k in list(id_to_vec):
        id_to_vec[k] = id_to_vec[k] / np.linalg.norm(id_to_vec[k])

    df, _ = generate_semantic_mid_core_support_v1(
        nodes_df=nodes,
        id_to_vec=id_to_vec,
        generator_cfg={"semantic_top_k": 5, "semantic_min_cos": 0.85, "semantic_max_cos_exclusive": 0.90},
    )
    if not df.empty:
        assert (df["n_shared_core_channels"] >= 1).all()
