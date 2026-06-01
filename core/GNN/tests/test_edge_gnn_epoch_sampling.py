"""Edge-GNN per-epoch train sampling parity with MLP pair supervision."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import torch

_GNN = Path(__file__).resolve().parents[1]
if str(_GNN) not in sys.path:
    sys.path.insert(0, str(_GNN))

from src.edge_pair_gnn_train import (  # noqa: E402
    _build_epoch_train_edge_indices,
    _parse_epoch_sampling_flags,
)
from src.pair_train import _easy_positive_mask, _hard_positive_mask, _hard_unlabeled_mask, _safe_bool_series  # noqa: E402


def _synthetic_train_df(n_pos: int, n_unl: int) -> pd.DataFrame:
    rows: list[dict] = []
    for i in range(n_pos):
        rows.append(
            {
                "email_i": f"p{i}",
                "email_j": f"p{i}_b",
                "is_positive": True,
                "is_unlabeled": False,
                "is_reliable_negative": False,
                "cluster_i": i % 3,
                "cluster_j": (i + 1) % 3,
                "cluster_pair_key": f"{i % 3}|{(i + 1) % 3}",
            }
        )
    for i in range(n_unl):
        rows.append(
            {
                "email_i": f"u{i}",
                "email_j": f"u{i}_b",
                "is_positive": False,
                "is_unlabeled": True,
                "is_reliable_negative": False,
                "cluster_i": 10 + (i % 4),
                "cluster_j": 10 + ((i + 1) % 4),
                "cluster_pair_key": f"{10 + (i % 4)}|{10 + ((i + 1) % 4)}",
            }
        )
    return pd.DataFrame(rows)


def test_epoch_sampling_applies_train_balance_2_to_1() -> None:
    train_df = _synthetic_train_df(n_pos=20, n_unl=100)
    full = train_df.copy()
    full["_edge_node_id"] = range(len(full))
    lookup = full[["email_i", "email_j", "_edge_node_id"]]

    training_cfg = {
        "semantic_cluster_sampling": {"enabled": True},
        "cluster_redundancy_control": {"enabled": False},
        "train_balance": {
            "enabled": True,
            "mode": "target_pos_to_unl_ratio",
            "target_pos_to_unl_ratio": 2.0,
            "shuffle_each_epoch": True,
        },
        "easy_positive_capping": {"enabled": False},
        "hard_positive_emphasis": {"enabled": False},
        "hard_unlabeled_emphasis": {"enabled": False},
        "reliable_negative_emphasis": {"enabled": False},
        "pair_split_seed": 42,
    }
    epoch_sampling = _parse_epoch_sampling_flags(training_cfg)
    assert epoch_sampling["balance_enabled"] is True

    hard_pos = _hard_positive_mask(
        train_df,
        cross_seed_component_only=True,
        require_from_2hop=True,
        max_source_count=None,
        exclude_from_rare_artifact=False,
        require_not_same_seed_component=True,
    )
    hard_unl = _hard_unlabeled_mask(
        train_df,
        cross_seed_component_only=True,
        require_from_2hop=True,
        max_source_count=None,
        exclude_from_rare_artifact=True,
        require_not_same_seed_component=False,
        require_from_semantic_false=False,
    )
    easy_pos = _easy_positive_mask(
        train_df,
        same_seed_component_only=True,
        min_semantic_cosine=None,
        min_source_count=None,
        or_rule_across_conditions=True,
    )
    rn = _safe_bool_series(train_df, "is_reliable_negative", default=False)

    idx, diag = _build_epoch_train_edge_indices(
        train_df=train_df,
        edge_id_lookup=lookup,
        training_cfg=training_cfg,
        split_seed=42,
        epoch=1,
        epoch_sampling=epoch_sampling,
        hard_pos_mask_train=hard_pos,
        hard_unl_mask_train=hard_unl,
        easy_pos_mask_train=easy_pos,
        reliable_neg_mask_train=rn,
        rn_supervision_active=False,
    )

    assert isinstance(idx, torch.Tensor)
    assert idx.numel() < len(train_df)
    ratio = float(diag.get("effective_pos_to_unl_ratio", 0.0))
    assert 1.8 <= ratio <= 2.2
    assert diag.get("balance_cap_applied_to") == "unlabeled"


def test_epoch_sampling_caps_positives_when_pos_dominate() -> None:
    train_df = _synthetic_train_df(n_pos=200, n_unl=50)
    full = train_df.copy()
    full["_edge_node_id"] = range(len(full))
    lookup = full[["email_i", "email_j", "_edge_node_id"]]

    training_cfg = {
        "semantic_cluster_sampling": {"enabled": True},
        "cluster_redundancy_control": {"enabled": False},
        "train_balance": {
            "enabled": True,
            "mode": "target_pos_to_unl_ratio",
            "target_pos_to_unl_ratio": 2.0,
            "shuffle_each_epoch": True,
        },
        "easy_positive_capping": {"enabled": False},
        "hard_positive_emphasis": {"enabled": False},
        "hard_unlabeled_emphasis": {"enabled": False},
        "reliable_negative_emphasis": {"enabled": False},
        "pair_split_seed": 7,
    }
    epoch_sampling = _parse_epoch_sampling_flags(training_cfg)
    hard_pos = _hard_positive_mask(
        train_df,
        cross_seed_component_only=True,
        require_from_2hop=True,
        max_source_count=None,
        exclude_from_rare_artifact=False,
        require_not_same_seed_component=True,
    )
    hard_unl = _hard_unlabeled_mask(
        train_df,
        cross_seed_component_only=True,
        require_from_2hop=True,
        max_source_count=None,
        exclude_from_rare_artifact=True,
        require_not_same_seed_component=False,
        require_from_semantic_false=False,
    )
    easy_pos = _easy_positive_mask(
        train_df,
        same_seed_component_only=True,
        min_semantic_cosine=None,
        min_source_count=None,
        or_rule_across_conditions=True,
    )
    rn = _safe_bool_series(train_df, "is_reliable_negative", default=False)

    _idx, diag = _build_epoch_train_edge_indices(
        train_df=train_df,
        edge_id_lookup=lookup,
        training_cfg=training_cfg,
        split_seed=7,
        epoch=1,
        epoch_sampling=epoch_sampling,
        hard_pos_mask_train=hard_pos,
        hard_unl_mask_train=hard_unl,
        easy_pos_mask_train=easy_pos,
        reliable_neg_mask_train=rn,
        rn_supervision_active=False,
    )

    assert diag.get("balance_cap_applied_to") == "positive"
    tb = dict(diag.get("train_balance") or {})
    assert int(tb.get("n_pos_after", 0)) == 100                
    assert int(tb.get("n_unl_after", 0)) == 50
