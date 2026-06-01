"""
Semantic-cluster metadata for pair training: split hygiene and per-epoch redundancy control.

Uses BERT semantic-cluster membership (e.g. from ``semantic_supernode_clusters.csv``) as a
family id layer on **email-level** pair rows — does not change graph structure.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

CLUSTER_SPLIT_ASSIGNMENT_GREEDY = "greedy_row_mass_balanced"
CLUSTER_SPLIT_ASSIGNMENT_LABEL_AWARE = "greedy_label_aware_balanced"
CLUSTER_SPLIT_ASSIGNMENT_RANDOM = "random_cluster_count_split"
CLUSTER_SPLIT_ASSIGNMENT_CHOICES = (
    CLUSTER_SPLIT_ASSIGNMENT_LABEL_AWARE,
    CLUSTER_SPLIT_ASSIGNMENT_GREEDY,
    CLUSTER_SPLIT_ASSIGNMENT_RANDOM,
)

CLUSTER_SPLIT_CROSS_DROP = "drop"
CLUSTER_SPLIT_CROSS_TRAIN_ONLY = "train_only"

GROUP_SOURCE_SEMANTIC_CLUSTER = "semantic_cluster"

MASS_KEYS = ("total", "positive", "unlabeled")
SplitName = Literal["train", "val", "test"]
SPLIT_ORDER: tuple[SplitName, ...] = ("train", "val", "test")


def _project_root_from_here() -> Path:
    return Path(__file__).resolve().parents[3]


def resolve_mapping_csv(raw: str | Path, *, project_root: Path | None = None) -> Path:
    p = Path(str(raw)).expanduser()
    if not p.is_absolute():
        root = project_root or _project_root_from_here()
        p = (root / p).resolve()
    return p.resolve()


def load_email_to_semantic_cluster(
    mapping_csv: Path,
    *,
    email_column: str = "email_id",
    cluster_column: str = "cluster_id",
) -> dict[str, int]:
    """``email_id -> cluster_id`` (int)."""
    df = pd.read_csv(mapping_csv, usecols=[email_column, cluster_column], low_memory=False)
    if df.empty:
        return {}
    out: dict[str, int] = {}
    for eid, cid in zip(
        df[email_column].astype(str).str.strip(),
        pd.to_numeric(df[cluster_column], errors="coerce"),
        strict=False,
    ):
        if not eid or pd.isna(cid):
            continue
        out[eid] = int(cid)
    return out


def annotate_pair_rows_with_semantic_clusters(
    df: pd.DataFrame,
    email_to_cluster: dict[str, int],
    *,
    email_i_col: str = "email_i",
    email_j_col: str = "email_j",
    missing_cluster_policy: str = "singleton_per_email",
) -> pd.DataFrame:
    """
    Add ``cluster_i``, ``cluster_j``, ``cluster_pair_key`` (string ``"a|b"`` with a <= b).

    Emails absent from ``email_to_cluster`` get a synthetic cluster id per email when
    ``missing_cluster_policy == "singleton_per_email"`` (stable hash bucket).
    """
    out = df.copy()
    ei = out[email_i_col].astype(str).str.strip()
    ej = out[email_j_col].astype(str).str.strip()

    def _cid(eid: str, counter: dict[str, int], next_neg: list[int]) -> int:
        if eid in email_to_cluster:
            return int(email_to_cluster[eid])
        if missing_cluster_policy == "singleton_per_email":
            if eid not in counter:
                counter[eid] = next_neg[0]
                next_neg[0] -= 1
            return counter[eid]
        return -1

    synth: dict[str, int] = {}
    next_neg = [-1]

    ci = [_cid(e, synth, next_neg) for e in ei]
    cj = [_cid(e, synth, next_neg) for e in ej]
    out["cluster_i"] = ci
    out["cluster_j"] = cj

    keys: list[str] = []
    for a, b in zip(ci, cj, strict=False):
        lo, hi = (int(a), int(b)) if int(a) <= int(b) else (int(b), int(a))
        keys.append(f"{lo}|{hi}")
    out["cluster_pair_key"] = keys
    return out


def _compute_cluster_endpoint_row_mass(work: pd.DataFrame) -> dict[int, dict[str, int]]:
    """
    Per-cluster endpoint incidence mass: each pair row adds 1 to both endpoint clusters.

    Also tracks positive / unlabeled incidence when those columns exist.
    """
    mass: dict[int, dict[str, int]] = {}
    has_pos = "is_positive" in work.columns
    has_unl = "is_unlabeled" in work.columns

    def _bump(cid: int, *, pos: bool, unl: bool) -> None:
        slot = mass.setdefault(int(cid), {"total": 0, "positive": 0, "unlabeled": 0})
        slot["total"] += 1
        if pos:
            slot["positive"] += 1
        if unl:
            slot["unlabeled"] += 1

    for row in work.itertuples(index=False):
        ci = int(row.cluster_i)
        cj = int(row.cluster_j)
        pos = bool(getattr(row, "is_positive", False)) if has_pos else False
        unl = bool(getattr(row, "is_unlabeled", False)) if has_unl else False
        _bump(ci, pos=pos, unl=unl)
        _bump(cj, pos=pos, unl=unl)
    return mass


def _assign_clusters_random_count(
    unique_clusters: list[int],
    *,
    val_ratio: float,
    test_ratio: float,
    rng: np.random.Generator,
) -> tuple[set[int], set[int], set[int], dict[str, Any]]:
    """Legacy: shuffle clusters and slice by cluster count (ignores row mass)."""
    shuffled = list(unique_clusters)
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_test_c = int(np.floor(n * test_ratio))
    n_val_c = int(np.floor(n * val_ratio))
    test_clusters = set(shuffled[:n_test_c])
    val_clusters = set(shuffled[n_test_c : n_test_c + n_val_c])
    train_clusters = set(shuffled[n_test_c + n_val_c :])
    diag = {
        "n_clusters_assigned_train": len(train_clusters),
        "n_clusters_assigned_val": len(val_clusters),
        "n_clusters_assigned_test": len(test_clusters),
    }
    return train_clusters, val_clusters, test_clusters, diag


def _split_mass_targets(
    cluster_mass: dict[int, dict[str, int]],
    *,
    val_ratio: float,
    test_ratio: float,
) -> tuple[dict[SplitName, dict[str, float]], dict[str, float]]:
    train_ratio = max(0.0, 1.0 - float(val_ratio) - float(test_ratio))
    ratios: dict[SplitName, float] = {
        "train": train_ratio,
        "val": float(val_ratio),
        "test": float(test_ratio),
    }
    dataset_totals = {
        k: float(sum(int(m.get(k, 0)) for m in cluster_mass.values())) for k in MASS_KEYS
    }
    targets: dict[SplitName, dict[str, float]] = {}
    for split in SPLIT_ORDER:
        targets[split] = {k: dataset_totals[k] * ratios[split] for k in MASS_KEYS}
    return targets, dataset_totals


def _greedy_score_after_assign(
    loads: dict[SplitName, dict[str, float]],
    targets: dict[SplitName, dict[str, float]],
    split: SplitName,
    add: dict[str, float],
    *,
    label_aware: bool,
) -> float:
    """Sum of squared relative errors vs target after hypothetically adding ``add`` mass."""
    keys = MASS_KEYS if label_aware else ("total",)
    score = 0.0
    for key in keys:
        target = targets[split][key]
        if target < 1e-12:
            continue
        after = (loads[split][key] + add.get(key, 0.0)) / target
        score += (after - 1.0) ** 2
    return score


def _assign_clusters_greedy_balanced(
    unique_clusters: list[int],
    cluster_mass: dict[int, dict[str, int]],
    *,
    val_ratio: float,
    test_ratio: float,
    rng: np.random.Generator,
    label_aware: bool,
) -> tuple[set[int], set[int], set[int], dict[str, Any]]:
    """Greedy cluster assignment minimizing imbalance on endpoint row mass (label-aware optional)."""
    targets, dataset_totals = _split_mass_targets(cluster_mass, val_ratio=val_ratio, test_ratio=test_ratio)
    loads: dict[SplitName, dict[str, float]] = {
        s: {k: 0.0 for k in MASS_KEYS} for s in SPLIT_ORDER
    }
    assignment: dict[int, SplitName] = {}

    order_rng = np.random.default_rng(int(rng.integers(0, 2**31 - 1)))
    tie_keys = {cid: int(order_rng.integers(0, 2**31 - 1)) for cid in unique_clusters}
    sorted_clusters = sorted(
        unique_clusters,
        key=lambda cid: (
            -int(cluster_mass.get(cid, {}).get("total", 0)),
            -int(cluster_mass.get(cid, {}).get("unlabeled", 0)),
            tie_keys[cid],
            cid,
        ),
    )

    per_cluster_log: list[dict[str, Any]] = []
    for cid in sorted_clusters:
        m = cluster_mass.get(cid, {"total": 0, "positive": 0, "unlabeled": 0})
        add = {k: float(m.get(k, 0)) for k in MASS_KEYS}
        best_score = float("inf")
        best_split: SplitName = "train"
        for split in SPLIT_ORDER:
            score = _greedy_score_after_assign(loads, targets, split, add, label_aware=label_aware)
            if score < best_score - 1e-12 or (
                abs(score - best_score) <= 1e-12
                and (loads[split]["total"], SPLIT_ORDER.index(split))
                < (loads[best_split]["total"], SPLIT_ORDER.index(best_split))
            ):
                best_score = score
                best_split = split
        assignment[cid] = best_split
        for key in MASS_KEYS:
            loads[best_split][key] += add[key]
        per_cluster_log.append(
            {
                "cluster_id": int(cid),
                "assigned_split": best_split,
                "row_mass_total": int(m.get("total", 0)),
                "row_mass_positive": int(m.get("positive", 0)),
                "row_mass_unlabeled": int(m.get("unlabeled", 0)),
                "assignment_imbalance_score": float(best_score),
            }
        )

    train_clusters = {cid for cid, s in assignment.items() if s == "train"}
    val_clusters = {cid for cid, s in assignment.items() if s == "val"}
    test_clusters = {cid for cid, s in assignment.items() if s == "test"}

    train_ratio = max(0.0, 1.0 - float(val_ratio) - float(test_ratio))

    def _mass_stats(split: SplitName, cluster_set: set[int]) -> dict[str, Any]:
        return {
            "n_clusters": len(cluster_set),
            "assigned_row_mass_total": float(loads[split]["total"]),
            "assigned_row_mass_positive": float(loads[split]["positive"]),
            "assigned_row_mass_unlabeled": float(loads[split]["unlabeled"]),
            "target_row_mass_total": float(targets[split]["total"]),
            "target_row_mass_positive": float(targets[split]["positive"]),
            "target_row_mass_unlabeled": float(targets[split]["unlabeled"]),
            "achieved_mass_ratio_total_vs_dataset": float(
                loads[split]["total"] / max(dataset_totals["total"], 1.0)
            ),
            "achieved_mass_ratio_positive_vs_dataset": float(
                loads[split]["positive"] / max(dataset_totals["positive"], 1.0)
            ),
            "achieved_mass_ratio_unlabeled_vs_dataset": float(
                loads[split]["unlabeled"] / max(dataset_totals["unlabeled"], 1.0)
            ),
            "requested_mass_ratio": float(
                {"train": train_ratio, "val": val_ratio, "test": test_ratio}[split]
            ),
        }

    top_by_mass = sorted(per_cluster_log, key=lambda x: -x["row_mass_total"])[:10]
    return (
        train_clusters,
        val_clusters,
        test_clusters,
        {
            "greedy_objective": "label_aware_squared_relative_error"
            if label_aware
            else "total_mass_squared_relative_error",
            "greedy_dataset_endpoint_mass_totals": dataset_totals,
            "greedy_targets_by_split": targets,
            "cluster_allocation_by_split": {
                "train": _mass_stats("train", train_clusters),
                "val": _mass_stats("val", val_clusters),
                "test": _mass_stats("test", test_clusters),
            },
            "max_cluster_row_mass_per_split": {
                "train": max(
                    (x["row_mass_total"] for x in per_cluster_log if x["assigned_split"] == "train"),
                    default=0,
                ),
                "val": max(
                    (x["row_mass_total"] for x in per_cluster_log if x["assigned_split"] == "val"),
                    default=0,
                ),
                "test": max(
                    (x["row_mass_total"] for x in per_cluster_log if x["assigned_split"] == "test"),
                    default=0,
                ),
            },
            "top_clusters_by_row_mass": top_by_mass,
            "per_cluster_assignment": per_cluster_log,
        },
    )


def _assign_clusters_greedy_row_mass(
    unique_clusters: list[int],
    cluster_mass: dict[int, dict[str, int]],
    *,
    val_ratio: float,
    test_ratio: float,
    rng: np.random.Generator,
) -> tuple[set[int], set[int], set[int], dict[str, Any]]:
    return _assign_clusters_greedy_balanced(
        unique_clusters,
        cluster_mass,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        rng=rng,
        label_aware=False,
    )


def _assign_clusters_greedy_label_aware(
    unique_clusters: list[int],
    cluster_mass: dict[int, dict[str, int]],
    *,
    val_ratio: float,
    test_ratio: float,
    rng: np.random.Generator,
) -> tuple[set[int], set[int], set[int], dict[str, Any]]:
    return _assign_clusters_greedy_balanced(
        unique_clusters,
        cluster_mass,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        rng=rng,
        label_aware=True,
    )


def _resolve_cluster_split_assignment_strategy(raw: str | None) -> str:
    s = str(raw or CLUSTER_SPLIT_ASSIGNMENT_LABEL_AWARE).strip().lower()
    if s in CLUSTER_SPLIT_ASSIGNMENT_CHOICES:
        return s
    raise ValueError(
        f"Unknown cluster_split_assignment_strategy: {raw!r}; "
        f"choices: {', '.join(CLUSTER_SPLIT_ASSIGNMENT_CHOICES)}."
    )


def _resolve_cross_split_pair_policy(
    raw: str | None,
    *,
    group_source: str,
) -> str:
    gs = str(group_source or GROUP_SOURCE_SEMANTIC_CLUSTER).strip().lower()
    policy = str(raw or "").strip().lower()
    if gs == GROUP_SOURCE_SEMANTIC_CLUSTER:
        if not policy:
            return CLUSTER_SPLIT_CROSS_DROP
        if policy == CLUSTER_SPLIT_CROSS_TRAIN_ONLY:
            raise ValueError(
                "cluster_split_hygiene.cross_split_pair_policy='train_only' is not allowed when "
                "group_source='semantic_cluster' (it leaks clusters into train). Use 'drop'."
            )
        if policy != CLUSTER_SPLIT_CROSS_DROP:
            raise ValueError(
                f"Unsupported cross_split_pair_policy for semantic_cluster: {raw!r}; use 'drop'."
            )
        return CLUSTER_SPLIT_CROSS_DROP
    if not policy:
        return CLUSTER_SPLIT_CROSS_TRAIN_ONLY
    if policy not in (CLUSTER_SPLIT_CROSS_DROP, CLUSTER_SPLIT_CROSS_TRAIN_ONLY):
        raise ValueError(
            f"Unknown cross_split_pair_policy: {raw!r}; use 'drop' or 'train_only'."
        )
    return policy


def _pair_label_counts(frame: pd.DataFrame) -> dict[str, int]:
    out = {
        "n_rows": int(len(frame)),
        "n_positive": 0,
        "n_unlabeled": 0,
        "n_reliable_negative": 0,
    }
    if frame.empty:
        return out
    if "is_positive" in frame.columns:
        out["n_positive"] = int(frame["is_positive"].fillna(False).astype(bool).sum())
    if "is_unlabeled" in frame.columns:
        out["n_unlabeled"] = int(frame["is_unlabeled"].fillna(False).astype(bool).sum())
    if "is_reliable_negative" in frame.columns:
        out["n_reliable_negative"] = int(frame["is_reliable_negative"].fillna(False).astype(bool).sum())
    return out


def _cluster_sets_disjoint(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> dict[str, Any]:
    train_c = set(train["cluster_i"].astype(int)) | set(train["cluster_j"].astype(int)) if len(train) else set()
    val_c = set(val["cluster_i"].astype(int)) | set(val["cluster_j"].astype(int)) if len(val) else set()
    test_c = set(test["cluster_i"].astype(int)) | set(test["cluster_j"].astype(int)) if len(test) else set()
    leak_tv = train_c & val_c
    leak_tt = train_c & test_c
    leak_vt = val_c & test_c
    return {
        "no_cluster_leakage_across_splits": not (leak_tv or leak_tt or leak_vt),
        "n_shared_clusters_train_val": len(leak_tv),
        "n_shared_clusters_train_test": len(leak_tt),
        "n_shared_clusters_val_test": len(leak_vt),
    }


def _balance_diagnostics(
    *,
    n_kept_rows: int,
    train_counts: dict[str, int],
    val_counts: dict[str, int],
    test_counts: dict[str, int],
    val_ratio: float,
    test_ratio: float,
    imbalance_relative_tolerance: float = 0.15,
    min_unlabeled_per_eval_split: int = 5,
) -> dict[str, Any]:
    train_ratio = max(0.0, 1.0 - val_ratio - test_ratio)
    requested = {"train": train_ratio, "val": val_ratio, "test": test_ratio}
    achieved_rows = {
        "train": train_counts["n_rows"] / max(n_kept_rows, 1),
        "val": val_counts["n_rows"] / max(n_kept_rows, 1),
        "test": test_counts["n_rows"] / max(n_kept_rows, 1),
    }
    n_pos_total = train_counts["n_positive"] + val_counts["n_positive"] + test_counts["n_positive"]
    n_unl_total = train_counts["n_unlabeled"] + val_counts["n_unlabeled"] + test_counts["n_unlabeled"]
    achieved_pos = {
        "train": train_counts["n_positive"] / max(n_pos_total, 1),
        "val": val_counts["n_positive"] / max(n_pos_total, 1),
        "test": test_counts["n_positive"] / max(n_pos_total, 1),
    }
    achieved_unl = {
        "train": train_counts["n_unlabeled"] / max(n_unl_total, 1),
        "val": val_counts["n_unlabeled"] / max(n_unl_total, 1),
        "test": test_counts["n_unlabeled"] / max(n_unl_total, 1),
    }
    warnings: list[str] = []
    for split in ("train", "val", "test"):
        req = requested[split]
        ach = achieved_rows[split]
        if req > 0 and abs(ach - req) / req > imbalance_relative_tolerance:
            warnings.append(
                f"{split}: pair-row ratio achieved={ach:.3f} vs requested={req:.3f} "
                f"(relative error {abs(ach - req) / req:.3f} > {imbalance_relative_tolerance})"
            )
    for split in ("val", "test"):
        n_unl = {"val": val_counts["n_unlabeled"], "test": test_counts["n_unlabeled"]}[split]
        n_pos = {"val": val_counts["n_positive"], "test": test_counts["n_positive"]}[split]
        n_rows = {"val": val_counts["n_rows"], "test": test_counts["n_rows"]}[split]
        if n_rows > 0 and n_unl < min_unlabeled_per_eval_split:
            warnings.append(
                f"{split}: only {n_unl} unlabeled rows (min recommended {min_unlabeled_per_eval_split})"
            )
        if n_rows > 0 and n_pos > 0 and n_unl == 0:
            warnings.append(f"{split}: no unlabeled rows — eval split is all-positive")
        if n_rows > 0 and n_unl > 0 and n_pos / max(n_unl, 1) > 20.0:
            warnings.append(
                f"{split}: extreme positive:unlabeled skew ({n_pos} pos / {n_unl} unl)"
            )
    return {
        "requested_split_ratios": requested,
        "achieved_split_ratios_by_pair_rows": achieved_rows,
        "achieved_split_ratios_by_positives": achieved_pos,
        "achieved_split_ratios_by_unlabeled": achieved_unl,
        "imbalance_warnings": warnings,
        "imbalance_relative_tolerance": float(imbalance_relative_tolerance),
        "min_unlabeled_per_eval_split": int(min_unlabeled_per_eval_split),
    }


def _route_pairs_to_splits(
    work: pd.DataFrame,
    *,
    train_clusters: set[int],
    val_clusters: set[int],
    test_clusters: set[int],
    cross_split_pair_policy: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, int]]:
    def _cluster_split(cid: int) -> SplitName:
        if cid in test_clusters:
            return "test"
        if cid in val_clusters:
            return "val"
        return "train"

    split_labels: list[str | None] = []
    n_cross = 0
    n_cross_dropped = 0
    n_cross_to_train = 0
    policy = str(cross_split_pair_policy).strip().lower()
    for _, row in work.iterrows():
        si = _cluster_split(int(row["cluster_i"]))
        sj = _cluster_split(int(row["cluster_j"]))
        if si != sj:
            n_cross += 1
            if policy == CLUSTER_SPLIT_CROSS_DROP:
                split_labels.append(None)
                n_cross_dropped += 1
            elif policy == CLUSTER_SPLIT_CROSS_TRAIN_ONLY:
                split_labels.append("train")
                n_cross_to_train += 1
            else:
                order = {"test": 0, "val": 1, "train": 2}
                split_labels.append(min((si, sj), key=lambda s: order.get(s, 2)))
        else:
            split_labels.append(si)
    splits = pd.Series(split_labels, index=work.index)
    train_df = work.loc[splits == "train"].reset_index(drop=True)
    val_df = work.loc[splits == "val"].reset_index(drop=True)
    test_df = work.loc[splits == "test"].reset_index(drop=True)
    return train_df, val_df, test_df, {
        "n_rows_input_before_routing": int(len(work)),
        "n_rows_kept_after_routing": int(len(train_df) + len(val_df) + len(test_df)),
        "n_rows_cross_split_endpoints": int(n_cross),
        "n_rows_cross_split_dropped": int(n_cross_dropped),
        "n_rows_cross_split_routed_to_train": int(n_cross_to_train),
    }


def _assert_watertight_semantic_cluster_splits(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    *,
    hygiene: dict[str, Any],
) -> None:
    if hygiene.get("no_cluster_leakage_across_splits"):
        return
    leak_tv = hygiene.get("n_shared_clusters_train_val", "?")
    leak_tt = hygiene.get("n_shared_clusters_train_test", "?")
    leak_vt = hygiene.get("n_shared_clusters_val_test", "?")
    raise ValueError(
        "Semantic-cluster split hygiene failed: cluster ids appear in more than one split's "
        f"pair rows (train∩val={leak_tv}, train∩test={leak_tt}, val∩test={leak_vt}). "
        "Use cross_split_pair_policy='drop' and greedy_label_aware_balanced assignment."
    )


def split_pairs_by_disjoint_semantic_clusters(
    df: pd.DataFrame,
    email_to_cluster: dict[str, int],
    *,
    val_ratio: float,
    test_ratio: float,
    split_seed: int,
    cross_split_pair_policy: str | None = None,
    cluster_split_assignment_strategy: str | None = None,
    group_source: str = GROUP_SOURCE_SEMANTIC_CLUSTER,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """
    Assign whole semantic clusters to train/val/test; route pair rows accordingly.

    For ``group_source='semantic_cluster'`` (default): cross-split endpoint pairs are
    **dropped** (watertight). Cluster assignment defaults to label-aware greedy balancing.
    """
    if val_ratio < 0 or test_ratio < 0 or val_ratio + test_ratio >= 1.0:
        raise ValueError("val_ratio and test_ratio must be non-negative and sum to < 1.")

    gs = str(group_source or GROUP_SOURCE_SEMANTIC_CLUSTER).strip().lower()
    cross_policy = _resolve_cross_split_pair_policy(cross_split_pair_policy, group_source=gs)
    strategy = _resolve_cluster_split_assignment_strategy(cluster_split_assignment_strategy)
    work = annotate_pair_rows_with_semantic_clusters(df, email_to_cluster)
    unique_clusters = sorted(
        set(work["cluster_i"].astype(int).tolist()) | set(work["cluster_j"].astype(int).tolist())
    )
    n_clusters = len(unique_clusters)
    cluster_mass = _compute_cluster_endpoint_row_mass(work)
    rng = np.random.default_rng(int(split_seed))

    if strategy == CLUSTER_SPLIT_ASSIGNMENT_RANDOM:
        train_clusters, val_clusters, test_clusters, assign_diag = _assign_clusters_random_count(
            unique_clusters,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            rng=rng,
        )
    elif strategy == CLUSTER_SPLIT_ASSIGNMENT_GREEDY:
        train_clusters, val_clusters, test_clusters, assign_diag = _assign_clusters_greedy_row_mass(
            unique_clusters,
            cluster_mass,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            rng=rng,
        )
    else:
        train_clusters, val_clusters, test_clusters, assign_diag = _assign_clusters_greedy_label_aware(
            unique_clusters,
            cluster_mass,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            rng=rng,
        )

    train_df, val_df, test_df, route_diag = _route_pairs_to_splits(
        work,
        train_clusters=train_clusters,
        val_clusters=val_clusters,
        test_clusters=test_clusters,
        cross_split_pair_policy=cross_policy,
    )

    train_counts = _pair_label_counts(train_df)
    val_counts = _pair_label_counts(val_df)
    test_counts = _pair_label_counts(test_df)
    n_kept = int(train_counts["n_rows"] + val_counts["n_rows"] + test_counts["n_rows"])
    balance_diag = _balance_diagnostics(
        n_kept_rows=n_kept,
        train_counts=train_counts,
        val_counts=val_counts,
        test_counts=test_counts,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )
    hygiene = _cluster_sets_disjoint(train_df, val_df, test_df)
    _assert_watertight_semantic_cluster_splits(train_df, val_df, test_df, hygiene=hygiene)

    strategy_desc = {
        CLUSTER_SPLIT_ASSIGNMENT_LABEL_AWARE: (
            "greedy_label_aware_balanced: assign each whole cluster to the split that minimizes "
            "sum of squared relative errors for total, positive, and unlabeled endpoint row mass."
        ),
        CLUSTER_SPLIT_ASSIGNMENT_GREEDY: (
            "greedy_row_mass_balanced: minimize squared relative error for total endpoint row mass only."
        ),
        CLUSTER_SPLIT_ASSIGNMENT_RANDOM: (
            "random_cluster_count_split: shuffle cluster ids and slice by cluster count."
        ),
    }[strategy]

    meta: dict[str, Any] = {
        "split_mode": "disjoint_semantic_clusters",
        "group_source": gs,
        "cluster_split_assignment_strategy": strategy,
        "cluster_split_assignment_description": strategy_desc,
        "split_seed": int(split_seed),
        "requested_val_ratio": float(val_ratio),
        "requested_test_ratio": float(test_ratio),
        "n_unique_clusters": n_clusters,
        "n_clusters_train": len(train_clusters),
        "n_clusters_val": len(val_clusters),
        "n_clusters_test": len(test_clusters),
        "cross_split_pair_policy": cross_policy,
        "cross_split_pair_policy_note": (
            "drop: cross-split semantic-cluster pairs excluded (watertight)."
            if cross_policy == CLUSTER_SPLIT_CROSS_DROP
            else "train_only: cross-split pairs routed to train (not watertight for clusters)."
        ),
        "cluster_mass_endpoint_incidence_note": (
            "Each pair row increments total/positive/unlabeled mass for both endpoint clusters."
        ),
        "cluster_assignment": assign_diag,
        "pair_counts_after_routing": {
            "train": train_counts,
            "val": val_counts,
            "test": test_counts,
        },
        "balance_diagnostics": balance_diag,
        "hygiene": hygiene,
        "watertight_semantic_cluster_split": bool(hygiene.get("no_cluster_leakage_across_splits")),
        "n_rows_train": train_counts["n_rows"],
        "n_rows_val": val_counts["n_rows"],
        "n_rows_test": test_counts["n_rows"],
        "n_positive_train": train_counts["n_positive"],
        "n_positive_val": val_counts["n_positive"],
        "n_positive_test": test_counts["n_positive"],
        "n_unlabeled_train": train_counts["n_unlabeled"],
        "n_unlabeled_val": val_counts["n_unlabeled"],
        "n_unlabeled_test": test_counts["n_unlabeled"],
        **route_diag,
    }
    return train_df, val_df, test_df, meta


def _cap_rows_per_group(
    sub: pd.DataFrame,
    group_col: str,
    max_rows: int,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Randomly keep up to ``max_rows`` per group value; return kept df and skip counts per group."""
    if max_rows <= 0 or sub.empty:
        return sub.iloc[0:0].copy(), {}
    skipped: dict[str, int] = {}
    parts: list[pd.DataFrame] = []
    for key, grp in sub.groupby(group_col, sort=False):
        g = grp
        n = len(g)
        if n > max_rows:
            skipped[str(key)] = int(n - max_rows)
            g = g.sample(n=max_rows, replace=False, random_state=int(rng.integers(0, 2**31 - 1)))
        parts.append(g)
    if not parts:
        return sub.iloc[0:0].copy(), skipped
    return pd.concat(parts, axis=0, ignore_index=True), skipped


def _balance_pos_unlabeled(
    pos_df: pd.DataFrame,
    unl_df: pd.DataFrame,
    *,
    target_pos_to_unl_ratio: float,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Subsample the larger class (with replacement=False) to approach target pos:unl count ratio.
    """
    n_pos = len(pos_df)
    n_unl = len(unl_df)
    ratio = float(max(target_pos_to_unl_ratio, 1e-9))
    diag: dict[str, Any] = {
        "n_pos_before": n_pos,
        "n_unl_before": n_unl,
        "target_pos_to_unl_ratio": ratio,
    }
    if n_pos == 0 or n_unl == 0:
        out = pd.concat([pos_df, unl_df], axis=0, ignore_index=True)
        diag["n_pos_after"] = n_pos
        diag["n_unl_after"] = n_unl
        diag["effective_ratio"] = float("nan")
        return out, diag

    # target: n_pos / n_unl = ratio  =>  n_unl_target = n_pos / ratio
    n_unl_target = int(round(n_pos / ratio))
    n_pos_target = int(round(n_unl * ratio))

    if n_unl > n_unl_target and n_unl_target > 0:
        unl_keep = unl_df.sample(n=n_unl_target, replace=False, random_state=int(rng.integers(0, 2**31 - 1)))
        pos_keep = pos_df
    elif n_pos > n_pos_target and n_pos_target > 0:
        pos_keep = pos_df.sample(n=n_pos_target, replace=False, random_state=int(rng.integers(0, 2**31 - 1)))
        unl_keep = unl_df
    else:
        pos_keep = pos_df
        unl_keep = unl_df

    out = pd.concat([pos_keep, unl_keep], axis=0, ignore_index=True)
    n_pos_a = len(pos_keep)
    n_unl_a = len(unl_keep)
    diag["n_pos_after"] = n_pos_a
    diag["n_unl_after"] = n_unl_a
    diag["effective_ratio"] = float(n_pos_a / max(1, n_unl_a))
    return out, diag


def build_train_epoch_pos_unl_balance(
    train_df: pd.DataFrame,
    *,
    balance_cfg: dict[str, Any],
    epoch_seed: int,
    include_reliable_negative_in_epoch: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Per-epoch train subsample to target pos:unlabeled count ratio (no semantic clusters).

    Uses the same ``_balance_pos_unlabeled`` logic as cluster-aware sampling but without
    redundancy caps or cluster_pair_key requirements.
    """
    enabled_bal = bool(balance_cfg.get("enabled", False))
    shuffle = bool(balance_cfg.get("shuffle_each_epoch", True))
    rng = np.random.default_rng(int(epoch_seed))
    diag: dict[str, Any] = {
        "enabled_train_balance": enabled_bal,
        "semantic_cluster_sampling_required": False,
        "epoch_seed": int(epoch_seed),
        "n_train_rows_input": int(len(train_df)),
    }
    if not enabled_bal:
        out = train_df.copy()
        if shuffle:
            out = out.sample(frac=1.0, random_state=int(epoch_seed)).reset_index(drop=True)
        diag["n_train_rows_epoch_effective"] = int(len(out))
        return out, diag

    pos_mask = train_df["is_positive"].fillna(False).astype(bool)
    unl_mask = train_df["is_unlabeled"].fillna(False).astype(bool)
    rn_mask = train_df.get("is_reliable_negative", pd.Series(False, index=train_df.index)).fillna(False).astype(bool)
    pos_df = train_df.loc[pos_mask].copy()
    unl_df = train_df.loc[unl_mask].copy()
    rn_df = train_df.loc[rn_mask].copy() if include_reliable_negative_in_epoch else train_df.iloc[0:0].copy()

    mode = str(balance_cfg.get("mode", "target_pos_to_unl_ratio")).strip().lower()
    if mode != "target_pos_to_unl_ratio":
        raise ValueError(f"Unsupported train_balance.mode: {mode!r}")
    target_ratio = float(balance_cfg.get("target_pos_to_unl_ratio", 1.0))
    core, balance_diag = _balance_pos_unlabeled(pos_df, unl_df, target_pos_to_unl_ratio=target_ratio, rng=rng)
    out = pd.concat([core, rn_df], axis=0, ignore_index=True)
    diag["train_balance"] = balance_diag
    diag["sampling_without_replacement"] = True

    if shuffle:
        out = out.sample(frac=1.0, random_state=int(epoch_seed) + 7).reset_index(drop=True)

    n_pos_e = int(_safe_bool_count(out, "is_positive"))
    n_unl_e = int(_safe_bool_count(out, "is_unlabeled"))
    diag["n_train_rows_epoch_effective"] = int(len(out))
    diag["n_pos_effective_epoch"] = n_pos_e
    diag["n_unl_effective_epoch"] = n_unl_e
    diag["effective_pos_to_unl_ratio"] = float(n_pos_e / max(1, n_unl_e))
    return out, diag


def build_train_epoch_cluster_aware(
    train_df: pd.DataFrame,
    *,
    redundancy_cfg: dict[str, Any],
    balance_cfg: dict[str, Any],
    epoch_seed: int,
    include_reliable_negative_in_epoch: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Per-epoch train sample: cap cluster-pair redundancy within each label, then balance P:U.

    Expects ``cluster_pair_key`` (and optionally ``cluster_i``) on ``train_df``.
    """
    enabled_red = bool(redundancy_cfg.get("enabled", False))
    enabled_bal = bool(balance_cfg.get("enabled", False))
    shuffle = bool(redundancy_cfg.get("shuffle_each_epoch", True) or balance_cfg.get("shuffle_each_epoch", True))

    rng = np.random.default_rng(int(epoch_seed))
    diag: dict[str, Any] = {
        "enabled_redundancy_control": enabled_red,
        "enabled_train_balance": enabled_bal,
        "epoch_seed": int(epoch_seed),
        "n_train_rows_input": int(len(train_df)),
    }

    if not enabled_red and not enabled_bal:
        out = train_df.copy()
        if shuffle:
            out = out.sample(frac=1.0, random_state=int(epoch_seed)).reset_index(drop=True)
        diag["n_train_rows_epoch_effective"] = int(len(out))
        return out, diag

    pos_mask = train_df["is_positive"].fillna(False).astype(bool)
    unl_mask = train_df["is_unlabeled"].fillna(False).astype(bool)
    rn_mask = train_df.get("is_reliable_negative", pd.Series(False, index=train_df.index)).fillna(False).astype(bool)
    pos_df = train_df.loc[pos_mask].copy()
    unl_df = train_df.loc[unl_mask].copy()
    rn_df = train_df.loc[rn_mask].copy()

    max_pair = redundancy_cfg.get("max_rows_per_cluster_pair_per_epoch")
    max_pair = int(max_pair) if max_pair is not None else None
    max_single = redundancy_cfg.get("max_rows_per_single_cluster_per_epoch")
    max_single = int(max_single) if max_single is not None else None

    pair_skips_pos: dict[str, int] = {}
    pair_skips_unl: dict[str, int] = {}
    single_skips_pos: dict[str, int] = {}
    single_skips_unl: dict[str, int] = {}

    if enabled_red:
        if max_pair is not None and max_pair > 0:
            pos_df, pair_skips_pos = _cap_rows_per_group(pos_df, "cluster_pair_key", max_pair, rng)
            unl_df, pair_skips_unl = _cap_rows_per_group(unl_df, "cluster_pair_key", max_pair, rng)
        if max_single is not None and max_single > 0:
            # cap rows sharing same endpoint cluster (either side)
            def _cap_single(sub: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
                if sub.empty:
                    return sub, {}
                sub = sub.copy()
                sub["_single_key"] = sub.apply(
                    lambda r: str(min(int(r["cluster_i"]), int(r["cluster_j"]))),
                    axis=1,
                )
                out_sub, sk = _cap_rows_per_group(sub, "_single_key", max_single, rng)
                return out_sub.drop(columns=["_single_key"], errors="ignore"), sk

            pos_df, single_skips_pos = _cap_single(pos_df)
            unl_df, single_skips_unl = _cap_single(unl_df)

    diag["n_pos_after_cluster_caps"] = int(len(pos_df))
    diag["n_unl_after_cluster_caps"] = int(len(unl_df))
    diag["n_rows_skipped_cluster_pair_cap_positive"] = int(sum(pair_skips_pos.values()))
    diag["n_rows_skipped_cluster_pair_cap_unlabeled"] = int(sum(pair_skips_unl.values()))
    diag["n_rows_skipped_single_cluster_cap_positive"] = int(sum(single_skips_pos.values()))
    diag["n_rows_skipped_single_cluster_cap_unlabeled"] = int(sum(single_skips_unl.values()))

    balance_diag: dict[str, Any] = {}
    if enabled_bal:
        mode = str(balance_cfg.get("mode", "target_pos_to_unl_ratio")).strip().lower()
        if mode != "target_pos_to_unl_ratio":
            raise ValueError(f"Unsupported train_balance.mode: {mode!r}")
        target_ratio = float(balance_cfg.get("target_pos_to_unl_ratio", 1.0))
        core, balance_diag = _balance_pos_unlabeled(pos_df, unl_df, target_pos_to_unl_ratio=target_ratio, rng=rng)
        parts = [core]
        if include_reliable_negative_in_epoch and len(rn_df):
            parts.append(rn_df)
        out = pd.concat(parts, axis=0, ignore_index=True)
    else:
        parts = [pos_df, unl_df]
        if include_reliable_negative_in_epoch:
            parts.append(rn_df)
        out = pd.concat(parts, axis=0, ignore_index=True)

    diag["train_balance"] = balance_diag

    if shuffle:
        out = out.sample(frac=1.0, random_state=int(epoch_seed) + 7).reset_index(drop=True)

    n_pos_e = int(_safe_bool_count(out, "is_positive"))
    n_unl_e = int(_safe_bool_count(out, "is_unlabeled"))
    diag["n_train_rows_epoch_effective"] = int(len(out))
    diag["n_pos_effective_epoch"] = n_pos_e
    diag["n_unl_effective_epoch"] = n_unl_e
    diag["effective_pos_to_unl_ratio"] = float(n_pos_e / max(1, n_unl_e))

    if "cluster_pair_key" in out.columns and len(out):
        vc = out["cluster_pair_key"].value_counts()
        diag["n_unique_cluster_pair_keys_epoch"] = int(vc.shape[0])
        diag["mean_rows_per_cluster_pair_epoch"] = float(vc.mean())
        diag["max_rows_per_cluster_pair_epoch"] = int(vc.max())
        top = vc.head(5)
        diag["top_cluster_pair_keys_by_count"] = {str(k): int(v) for k, v in top.items()}
    else:
        diag["n_unique_cluster_pair_keys_epoch"] = 0

    return out, diag


def _safe_bool_count(df: pd.DataFrame, col: str) -> int:
    if col not in df.columns:
        return 0
    return int(df[col].fillna(False).astype(bool).sum())


def summarize_train_cluster_inventory(train_df: pd.DataFrame) -> dict[str, Any]:
    """Setup-level stats for train split."""
    out: dict[str, Any] = {"n_train_rows": int(len(train_df))}
    if "cluster_pair_key" not in train_df.columns:
        return out
    out["n_unique_semantic_clusters"] = int(
        len(set(train_df["cluster_i"].astype(int)) | set(train_df["cluster_j"].astype(int)))
    )
    out["n_unique_cluster_pair_keys"] = int(train_df["cluster_pair_key"].nunique())
    out["n_train_positive"] = int(train_df["is_positive"].sum()) if "is_positive" in train_df.columns else 0
    out["n_train_unlabeled"] = int(train_df["is_unlabeled"].sum()) if "is_unlabeled" in train_df.columns else 0
    return out


def write_cluster_sampling_artifact(run_dir: Path, payload: dict[str, Any], filename: str = "pair_cluster_sampling_diagnostic.json") -> str:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / filename
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return str(path)
