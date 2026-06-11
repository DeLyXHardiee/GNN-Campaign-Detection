"""Tests for GT-edge-informed seed rules and n_shared_core_channels."""

from __future__ import annotations

import pandas as pd
import pytest

from seed_candidate_workflow.utils.anchor_seed_helpers import (
    generate_semantic_sender_seed_edges_v1,
    generate_semantic_strong_seed_edges_v1,
)
from seed_candidate_workflow.utils.pair_training_dataset_helpers import (
    _add_shared_attribute_pair_features,
)


def test_semantic_strong_seed_rule():
    edges = pd.DataFrame(
        {
            "email_a": ["a", "b", "c"],
            "email_b": ["b", "c", "d"],
            "semantic_score": [0.94, 0.91, 0.88],
        }
    )
    out = generate_semantic_strong_seed_edges_v1(
        nodes_df=pd.DataFrame(),
        edges_df=edges,
        generator_cfg={"min_semantic_score": 0.93},
    )
    assert len(out) == 1
    assert out.iloc[0]["seed_generator"] == "semantic_strong_v1"
    assert out.iloc[0]["rule_id"] == "semantic_ge_0_93"


def test_semantic_sender_seed_rule():
    edges = pd.DataFrame(
        {
            "email_a": ["a", "b"],
            "email_b": ["b", "c"],
            "semantic_score": [0.91, 0.88],
            "has_sender_overlap": [True, True],
        }
    )
    out = generate_semantic_sender_seed_edges_v1(
        nodes_df=pd.DataFrame(),
        edges_df=edges,
        generator_cfg={},
    )
    assert len(out) == 1
    assert out.iloc[0]["seed_generator"] == "semantic_sender_seed_v1"


def test_n_shared_core_channels():
    df = pd.DataFrame({"email_i": ["a"], "email_j": ["b"]})
    nodes = {
        "a": {
            "sender_set": {"s1"},
            "stem_set": {"st"},
            "url_set": set(),
            "attachment_set": set(),
            "sender_email_domain_set": set(),
            "domain_set": set(),
        },
        "b": {
            "sender_set": {"s1"},
            "stem_set": {"st"},
            "url_set": {"u"},
            "attachment_set": set(),
            "sender_email_domain_set": set(),
            "domain_set": set(),
        },
    }
    out = _add_shared_attribute_pair_features(df=df, nodes_by_email=nodes)
    assert int(out.iloc[0]["n_shared_core_channels"]) == 2
