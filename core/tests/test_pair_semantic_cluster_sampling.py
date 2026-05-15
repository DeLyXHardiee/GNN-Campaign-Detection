from __future__ import annotations

import pandas as pd

import pytest

from core.GNN.src.pair_semantic_cluster_sampling import (
    CLUSTER_SPLIT_ASSIGNMENT_GREEDY,
    CLUSTER_SPLIT_ASSIGNMENT_LABEL_AWARE,
    CLUSTER_SPLIT_ASSIGNMENT_RANDOM,
    annotate_pair_rows_with_semantic_clusters,
    build_train_epoch_cluster_aware,
    split_pairs_by_disjoint_semantic_clusters,
)


def _mini_pairs() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "email_i": ["e1", "e1", "e2", "e3", "e4"],
            "email_j": ["e2", "e3", "e3", "e4", "e5"],
            "graph_email_idx_i": [0, 0, 1, 2, 3],
            "graph_email_idx_j": [1, 2, 2, 3, 4],
            "pair_status": ["positive", "positive", "unlabeled", "positive", "unlabeled"],
            "is_positive": [True, True, False, True, False],
            "is_unlabeled": [False, False, True, False, True],
            "is_reliable_negative": [False] * 5,
        }
    )


def test_annotate_cluster_pair_key():
    email_to_cluster = {"e1": 1, "e2": 1, "e3": 2, "e4": 3, "e5": 3}
    out = annotate_pair_rows_with_semantic_clusters(_mini_pairs(), email_to_cluster)
    assert out.loc[0, "cluster_pair_key"] == "1|1"
    assert out.loc[1, "cluster_pair_key"] == "1|2"


def test_disjoint_cluster_split_no_leak():
    email_to_cluster = {f"e{i}": i // 2 for i in range(1, 6)}
    df = _mini_pairs()
    train, val, test, meta = split_pairs_by_disjoint_semantic_clusters(
        df,
        email_to_cluster,
        val_ratio=0.2,
        test_ratio=0.2,
        split_seed=42,
    )
    assert meta["split_mode"] == "disjoint_semantic_clusters"
    assert meta["cluster_split_assignment_strategy"] == CLUSTER_SPLIT_ASSIGNMENT_LABEL_AWARE
    assert meta["cross_split_pair_policy"] == "drop"
    assert meta["hygiene"]["no_cluster_leakage_across_splits"] is True
    assert meta["watertight_semantic_cluster_split"] is True
    assert len(train) + len(val) + len(test) <= len(df)
    train_clusters = set(train["cluster_i"]) | set(train["cluster_j"])
    val_clusters = set(val["cluster_i"]) | set(val["cluster_j"])
    test_clusters = set(test["cluster_i"]) | set(test["cluster_j"])
    assert train_clusters.isdisjoint(val_clusters)
    assert train_clusters.isdisjoint(test_clusters)
    assert val_clusters.isdisjoint(test_clusters)


def test_cluster_cap_and_balance_preserves_some_of_each_class():
    email_to_cluster = {"e1": 1, "e2": 1, "e3": 2, "e4": 3, "e5": 4}
    train = annotate_pair_rows_with_semantic_clusters(_mini_pairs(), email_to_cluster)
    # duplicate many rows same cluster pair
    dup = pd.concat([train] * 20, ignore_index=True)
    dup["is_positive"] = dup["pair_status"] == "positive"
    dup["is_unlabeled"] = dup["pair_status"] == "unlabeled"
    out, diag = build_train_epoch_cluster_aware(
        dup,
        redundancy_cfg={
            "enabled": True,
            "max_rows_per_cluster_pair_per_epoch": 2,
            "shuffle_each_epoch": True,
        },
        balance_cfg={
            "enabled": True,
            "mode": "target_pos_to_unl_ratio",
            "target_pos_to_unl_ratio": 1.0,
            "shuffle_each_epoch": True,
        },
        epoch_seed=7,
    )
    assert diag["n_pos_effective_epoch"] > 0
    assert diag["n_unl_effective_epoch"] > 0
    assert int(diag.get("n_rows_skipped_cluster_pair_cap_positive", 0)) > 0
    assert len(out) < len(dup)
    assert int(diag["max_rows_per_cluster_pair_epoch"]) <= 4


def _skewed_pairs_df() -> pd.DataFrame:
    """One giant cluster vs many tiny clusters — greedy should balance row mass better."""
    rows = []
    for _ in range(500):
        rows.append(
            {
                "email_i": "big_a",
                "email_j": "big_b",
                "graph_email_idx_i": 0,
                "graph_email_idx_j": 1,
                "pair_status": "positive",
                "is_positive": True,
                "is_unlabeled": False,
                "is_reliable_negative": False,
            }
        )
    for i in range(20):
        rows.append(
            {
                "email_i": f"t{i}a",
                "email_j": f"t{i}b",
                "graph_email_idx_i": 10 + i,
                "graph_email_idx_j": 30 + i,
                "pair_status": "unlabeled",
                "is_positive": False,
                "is_unlabeled": True,
                "is_reliable_negative": False,
            }
        )
    return pd.DataFrame(rows)


def test_cross_split_pairs_dropped_not_routed_to_train():
    df = pd.DataFrame(
        {
            "email_i": ["a", "b"],
            "email_j": ["b", "a"],
            "graph_email_idx_i": [0, 1],
            "graph_email_idx_j": [1, 0],
            "pair_status": ["positive", "positive"],
            "is_positive": [True, True],
            "is_unlabeled": [False, False],
            "is_reliable_negative": [False, False],
        }
    )
    email_to_cluster = {"a": 1, "b": 2}
    train, val, test, meta = split_pairs_by_disjoint_semantic_clusters(
        df,
        email_to_cluster,
        val_ratio=0.5,
        test_ratio=0.0,
        split_seed=1,
    )
    assert meta["n_rows_cross_split_dropped"] == 2
    assert meta["n_rows_cross_split_routed_to_train"] == 0
    assert len(train) + len(val) + len(test) == 0
    assert meta["hygiene"]["no_cluster_leakage_across_splits"] is True


def test_semantic_cluster_rejects_train_only_policy():
    with pytest.raises(ValueError, match="train_only"):
        split_pairs_by_disjoint_semantic_clusters(
            _mini_pairs(),
            {"e1": 1, "e2": 1, "e3": 2, "e4": 3, "e5": 4},
            val_ratio=0.1,
            test_ratio=0.1,
            split_seed=42,
            cross_split_pair_policy="train_only",
        )


def test_label_aware_split_includes_unlabeled_in_val_on_skew():
    pos_rows = []
    unl_rows = []
    email_to_cluster = {}
    for i in range(40):
        eid = f"p{i}"
        email_to_cluster[eid] = 100
        pos_rows.append(
            {
                "email_i": eid,
                "email_j": f"p{i}_b",
                "graph_email_idx_i": i,
                "graph_email_idx_j": 1000 + i,
                "pair_status": "positive",
                "is_positive": True,
                "is_unlabeled": False,
                "is_reliable_negative": False,
            }
        )
        email_to_cluster[f"p{i}_b"] = 100
    for i in range(40):
        eid = f"u{i}"
        cid = 200 + i
        email_to_cluster[eid] = cid
        email_to_cluster[f"u{i}_b"] = cid
        unl_rows.append(
            {
                "email_i": eid,
                "email_j": f"u{i}_b",
                "graph_email_idx_i": 200 + i,
                "graph_email_idx_j": 3000 + i,
                "pair_status": "unlabeled",
                "is_positive": False,
                "is_unlabeled": True,
                "is_reliable_negative": False,
            }
        )
    df = pd.DataFrame(pos_rows + unl_rows)
    _, val_la, _, meta_la = split_pairs_by_disjoint_semantic_clusters(
        df,
        email_to_cluster,
        val_ratio=0.1,
        test_ratio=0.1,
        split_seed=42,
        cluster_split_assignment_strategy=CLUSTER_SPLIT_ASSIGNMENT_LABEL_AWARE,
    )
    _, val_rm, _, meta_rm = split_pairs_by_disjoint_semantic_clusters(
        df,
        email_to_cluster,
        val_ratio=0.1,
        test_ratio=0.1,
        split_seed=42,
        cluster_split_assignment_strategy=CLUSTER_SPLIT_ASSIGNMENT_GREEDY,
    )
    n_unl_la = int(val_la["is_unlabeled"].sum()) if len(val_la) else 0
    n_unl_rm = int(val_rm["is_unlabeled"].sum()) if len(val_rm) else 0
    assert n_unl_la >= n_unl_rm


def test_greedy_row_mass_split_more_balanced_than_random_on_skew():
    email_to_cluster = {"big_a": 100, "big_b": 100}
    for i in range(20):
        email_to_cluster[f"t{i}a"] = i
        email_to_cluster[f"t{i}b"] = i

    df = _skewed_pairs_df()
    n = len(df)
    _, _, _, meta_greedy = split_pairs_by_disjoint_semantic_clusters(
        df,
        email_to_cluster,
        val_ratio=0.1,
        test_ratio=0.1,
        split_seed=42,
        cluster_split_assignment_strategy=CLUSTER_SPLIT_ASSIGNMENT_LABEL_AWARE,
    )
    _, _, _, meta_random = split_pairs_by_disjoint_semantic_clusters(
        df,
        email_to_cluster,
        val_ratio=0.1,
        test_ratio=0.1,
        split_seed=42,
        cluster_split_assignment_strategy=CLUSTER_SPLIT_ASSIGNMENT_RANDOM,
    )

    def _row_imbalance(meta: dict) -> float:
        ach = meta["balance_diagnostics"]["achieved_split_ratios_by_pair_rows"]
        req = meta["balance_diagnostics"]["requested_split_ratios"]
        return sum(abs(ach[s] - req[s]) for s in ("train", "val", "test"))

    assert _row_imbalance(meta_greedy) < _row_imbalance(meta_random)
    assert _row_imbalance(meta_greedy) <= _row_imbalance(meta_random) + 0.05
    assert meta_greedy["n_rows_train"] + meta_greedy["n_rows_val"] + meta_greedy["n_rows_test"] <= n
