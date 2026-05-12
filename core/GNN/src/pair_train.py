"""
Email-email pair supervision training (parallel to graph-native link prediction).

Uses NeighborLoader on the hetero evidence graph anchored on email endpoints.
Supports nnPU (default) or legacy placeholder BCE for debugging (``pair_loss_type``).
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from .model import HeteroSAGE
from .model_io import select_device
from .pair_graph_sampling import (
    PairEndpointHeteroSample,
    collect_pair_sampling_diagnostics,
    sample_hetero_around_pair_endpoints,
)
from .pair_scorer import EmailPairMLPScorer, build_email_pair_mlp_scorer, count_scorer_parameters
from .pu_loss import (
    PAIR_LOSS_NNPU_WITH_RELIABLE_NEGATIVES,
    PAIR_LOSS_PLACEHOLDER_BCE,
    aggregate_epoch_pu_stats,
    compute_pair_loss,
    exclusive_pair_masks,
    resolve_pair_loss_type,
)

# Legacy config string (used only when ``pair_loss_type`` is omitted / inferred).
PLACEHOLDER_LOSS_BCE_POS_VS_UNLABELED_AS_NEG = "bce_pos_vs_unlabeled_as_neg"

PAIR_FEATURE_BOOL_COLS = [
    "from_seed",
    "from_rare_artifact",
    "from_semantic",
    "from_component",
    "from_2hop",
    "same_seed_component_flag",
    "cross_seed_component_flag",
    "has_shared_sender",
    "has_shared_stem",
    "has_shared_url",
    "has_shared_attachment",
    "has_shared_sender_domain",
    "has_shared_domain",
]

PAIR_FEATURE_NUMERIC_COLS = [
    "source_count",
    "semantic_cosine_max",
    "rare_artifact_rarity_max",
    "twohop_rarity_max",
    "component_cosine_max",
    "time_gap_seconds_min",
    "shared_sender_count",
    "shared_stem_count",
    "shared_url_count",
    "shared_attachment_count",
    "shared_sender_domain_count",
    "shared_domain_count",
]
# Raw seed_component_* ids excluded: arbitrary identifiers, not continuous features.

PAIR_FEATURE_COLUMNS = PAIR_FEATURE_BOOL_COLS + PAIR_FEATURE_NUMERIC_COLS


def _project_root_from_here() -> Path:
    return Path(__file__).resolve().parents[3]


def resolve_pair_dataset_csv(raw: str | Path, *, project_root: Path | None = None) -> Path:
    p = Path(str(raw)).expanduser()
    if not p.is_absolute():
        root = project_root or _project_root_from_here()
        p = (root / p).resolve()
    return p.resolve()


def _safe_float(v: Any, default: float = 0.0) -> float:
    x = pd.to_numeric(v, errors="coerce")
    if pd.isna(x):
        return float(default)
    return float(x)


def _safe_bool01(v: Any) -> float:
    if isinstance(v, (bool, np.bool_)):
        return 1.0 if v else 0.0
    if pd.isna(v):
        return 0.0
    s = str(v).strip().lower()
    if s in ("1", "true", "yes"):
        return 1.0
    return 0.0


def build_pair_feature_matrix(df: pd.DataFrame) -> np.ndarray:
    """
    Deterministic float matrix (N, F). Booleans -> 0/1; missing numerics -> 0.0.
    """
    rows: list[list[float]] = []
    for _, r in df.iterrows():
        row: list[float] = []
        for c in PAIR_FEATURE_BOOL_COLS:
            row.append(_safe_bool01(r.get(c, 0)))
        for c in PAIR_FEATURE_NUMERIC_COLS:
            row.append(_safe_float(r.get(c), 0.0))
        rows.append(row)
    return np.asarray(rows, dtype=np.float32)


def load_pair_training_dataframe(
    csv_path: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load pair_training_dataset.csv; keep rows with both graph indices present."""
    df = pd.read_csv(csv_path)
    stats: dict[str, Any] = {"n_rows_read": int(len(df))}
    req = {"email_i", "email_j", "graph_email_idx_i", "graph_email_idx_j", "pair_status"}
    missing = req - set(df.columns)
    if missing:
        raise ValueError(f"pair dataset missing columns: {sorted(missing)}")
    gi = pd.to_numeric(df["graph_email_idx_i"], errors="coerce")
    gj = pd.to_numeric(df["graph_email_idx_j"], errors="coerce")
    ok = gi.notna() & gj.notna()
    df = df.loc[ok].copy()
    df["graph_email_idx_i"] = gi.loc[ok].astype(np.int64)
    df["graph_email_idx_j"] = gj.loc[ok].astype(np.int64)
    stats["n_rows_after_graph_index_filter"] = int(len(df))
    st = df["pair_status"].astype(str).str.lower()
    df["is_positive"] = st == "positive"
    df["is_reliable_negative"] = st == "reliable_negative"
    df["is_unlabeled"] = st == "unlabeled"
    stats["n_positive"] = int(df["is_positive"].sum())
    stats["n_unlabeled"] = int(df["is_unlabeled"].sum())
    stats["n_reliable_negative"] = int(df["is_reliable_negative"].sum())
    stats["fraction_reliable_negative"] = float(stats["n_reliable_negative"] / max(1, int(len(df))))
    return df, stats


def split_pairs_train_val_test(
    df: pd.DataFrame,
    *,
    val_ratio: float,
    test_ratio: float,
    split_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Deterministic shuffle then contiguous slices (reproducible given split_seed).
    """
    if val_ratio < 0 or test_ratio < 0 or val_ratio + test_ratio >= 1.0:
        raise ValueError("val_ratio and test_ratio must be non-negative and sum to < 1.")
    rng = np.random.default_rng(int(split_seed))
    idx = np.arange(len(df))
    rng.shuffle(idx)
    n = len(idx)
    n_test = int(np.floor(n * test_ratio))
    n_val = int(np.floor(n * val_ratio))
    i_test = idx[:n_test]
    i_val = idx[n_test : n_test + n_val]
    i_train = idx[n_test + n_val :]
    return df.iloc[i_train].reset_index(drop=True), df.iloc[i_val].reset_index(drop=True), df.iloc[i_test].reset_index(drop=True)


@dataclass
class PairBatchDiag:
    n_pairs: int
    n_unique_emails: int
    n_pairs_mapped_ok: int
    n_pairs_missing_endpoint: int


def _chunk_pair_indices_for_unique_cap(
    df: pd.DataFrame,
    gi: np.ndarray,
    gj: np.ndarray,
    max_unique: int,
) -> list[tuple[int, int]]:
    """
    Greedy pack row ranges [s, e) so each chunk's unique endpoint count <= max_unique.
    """
    n = len(df)
    chunks: list[tuple[int, int]] = []
    s = 0
    while s < n:
        uniq: set[int] = set()
        e = s
        while e < n:
            a, b = int(gi[e]), int(gj[e])
            cand = len(uniq | {a, b})
            if cand > max_unique and e > s:
                break
            uniq.update((a, b))
            e += 1
        if e == s:
            e = s + 1
        chunks.append((s, e))
        s = e
    return chunks


def iter_pair_batches(
    df: pd.DataFrame,
    pair_batch_size: int,
    max_unique_emails: int,
) -> Iterator[tuple[pd.DataFrame, np.ndarray, np.ndarray]]:
    """Yield (chunk_df, gi, gj) arrays aligned to chunk rows."""
    n = len(df)
    for start in range(0, n, pair_batch_size):
        chunk = df.iloc[start : start + pair_batch_size]
        gi = chunk["graph_email_idx_i"].to_numpy(dtype=np.int64, copy=False)
        gj = chunk["graph_email_idx_j"].to_numpy(dtype=np.int64, copy=False)
        chunk_r = chunk.reset_index(drop=True)
        gi_r = chunk_r["graph_email_idx_i"].to_numpy(dtype=np.int64, copy=False)
        gj_r = chunk_r["graph_email_idx_j"].to_numpy(dtype=np.int64, copy=False)
        for s, e in _chunk_pair_indices_for_unique_cap(chunk_r, gi_r, gj_r, max_unique_emails):
            sub = chunk_r.iloc[s:e].reset_index(drop=True)
            yield sub, gi_r[s:e], gj_r[s:e]


def count_pair_batches(df: pd.DataFrame, pair_batch_size: int, max_unique_emails: int) -> int:
    """Number of graph batches per full pass over ``df`` (matches ``iter_pair_batches``)."""
    return sum(1 for _ in iter_pair_batches(df, pair_batch_size, max_unique_emails))


def _safe_bool_series(df: pd.DataFrame, col: str, default: bool = False) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype=bool)
    return df[col].fillna(default).astype(bool)


def _safe_numeric_series(df: pd.DataFrame, col: str, default: float = float("nan")) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype=np.float64)
    return pd.to_numeric(df[col], errors="coerce")


def _hard_positive_mask(
    df: pd.DataFrame,
    *,
    cross_seed_component_only: bool,
    require_from_2hop: bool,
    max_source_count: int | None,
    exclude_from_rare_artifact: bool,
    require_not_same_seed_component: bool,
) -> pd.Series:
    m = _safe_bool_series(df, "is_positive", default=False)
    if cross_seed_component_only:
        m = m & _safe_bool_series(df, "cross_seed_component_flag", default=False)
    if require_from_2hop:
        m = m & _safe_bool_series(df, "from_2hop", default=False)
    if max_source_count is not None:
        sc = _safe_numeric_series(df, "source_count")
        m = m & sc.le(float(max_source_count))
    if exclude_from_rare_artifact:
        m = m & (~_safe_bool_series(df, "from_rare_artifact", default=False))
    if require_not_same_seed_component:
        m = m & (~_safe_bool_series(df, "same_seed_component_flag", default=False))
    return m.fillna(False).astype(bool)


def _hard_unlabeled_mask(
    df: pd.DataFrame,
    *,
    cross_seed_component_only: bool,
    require_from_2hop: bool,
    max_source_count: int | None,
    exclude_from_rare_artifact: bool,
    require_not_same_seed_component: bool,
    require_from_semantic_false: bool,
) -> pd.Series:
    m = _safe_bool_series(df, "is_unlabeled", default=False)
    if cross_seed_component_only:
        m = m & _safe_bool_series(df, "cross_seed_component_flag", default=False)
    if require_from_2hop:
        m = m & _safe_bool_series(df, "from_2hop", default=False)
    if max_source_count is not None:
        sc = _safe_numeric_series(df, "source_count")
        m = m & sc.le(float(max_source_count))
    if exclude_from_rare_artifact:
        m = m & (~_safe_bool_series(df, "from_rare_artifact", default=False))
    if require_not_same_seed_component:
        m = m & (~_safe_bool_series(df, "same_seed_component_flag", default=False))
    if require_from_semantic_false:
        m = m & (~_safe_bool_series(df, "from_semantic", default=False))
    return m.fillna(False).astype(bool)


def _oversample_extra_chunks(
    base_df: pd.DataFrame,
    mask: pd.Series,
    oversample_factor: float,
    sample_seed: int,
) -> list[pd.DataFrame]:
    """Return extra dataframes to concat after base_df (factor 1.0 => [])."""
    if oversample_factor <= 1.0 or int(mask.sum()) == 0:
        return []
    hard_df = base_df.loc[mask].copy()
    extra_mult = int(np.floor(max(0.0, oversample_factor - 1.0)))
    frac = float(max(0.0, oversample_factor - 1.0 - extra_mult))
    extras: list[pd.DataFrame] = [hard_df.copy() for _ in range(extra_mult)]
    if frac > 1e-9 and len(hard_df) > 0:
        n_extra_frac = int(np.floor(frac * len(hard_df)))
        if n_extra_frac > 0:
            extras.append(
                hard_df.sample(n=n_extra_frac, replace=False, random_state=int(sample_seed)).copy()
            )
    return extras


def _easy_positive_mask(
    df: pd.DataFrame,
    *,
    same_seed_component_only: bool,
    min_semantic_cosine: float | None,
    min_source_count: int | None,
    or_rule_across_conditions: bool,
) -> pd.Series:
    """
    Easy positives: ``is_positive`` plus configurable evidence of being an ``obvious`` same-campaign pair.

    ``same_seed_component_only`` means: include ``same_seed_component_flag`` as one sub-condition (not ``only`` positives
    in that regime). When false, that branch is omitted.
    """
    pos = _safe_bool_series(df, "is_positive", default=False)
    clauses: list[pd.Series] = []
    if same_seed_component_only:
        clauses.append(_safe_bool_series(df, "same_seed_component_flag", default=False))
    if min_semantic_cosine is not None:
        sem = _safe_numeric_series(df, "semantic_cosine_max")
        clauses.append(sem.ge(float(min_semantic_cosine)).fillna(False).astype(bool))
    if min_source_count is not None:
        sc = _safe_numeric_series(df, "source_count")
        clauses.append(sc.ge(float(min_source_count)).fillna(False).astype(bool))
    if not clauses:
        return pd.Series(False, index=df.index, dtype=bool)
    if or_rule_across_conditions:
        m = clauses[0].copy()
        for c in clauses[1:]:
            m = m | c
    else:
        m = clauses[0].copy()
        for c in clauses[1:]:
            m = m & c
    return (pos & m.fillna(False)).astype(bool)


def _apply_easy_positive_cap(
    train_df: pd.DataFrame,
    easy_mask: pd.Series,
    retain_fraction: float,
    sample_seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Randomly retain ``retain_fraction`` of easy-positive rows each call (all other rows unchanged).

    At least one easy-positive row is kept whenever any exist, even for very small fractions.
    """
    m = easy_mask.reindex(train_df.index).fillna(False).astype(bool)
    pos = _safe_bool_series(train_df, "is_positive", default=False)
    n_total = int(len(train_df))
    n_easy = int(m.sum())
    n_non_easy_pos = int((pos & ~m).sum())
    rt = float(np.clip(retain_fraction, 0.0, 1.0))

    if n_easy == 0:
        out = train_df.copy().reset_index(drop=True)
        return out, {
            "n_train_rows_after_easy_cap": int(len(out)),
            "n_train_rows_before_easy_cap": n_total,
            "n_easy_positives_in_rule": 0,
            "n_easy_positives_retained_this_epoch": 0,
            "n_easy_positives_dropped_this_epoch": 0,
            "n_non_easy_positive_rows_retained": n_non_easy_pos,
            "retain_fraction_requested": float(retain_fraction),
        }

    if rt >= 1.0 - 1e-12:
        n_keep = n_easy
    elif rt <= 0.0:
        n_keep = 1
    else:
        n_keep = max(1, min(n_easy, int(round(rt * n_easy))))

    easy_df = train_df.loc[m]
    rest = train_df.loc[~m]
    if n_keep >= n_easy:
        kept = easy_df.copy()
    else:
        kept = easy_df.sample(n=n_keep, random_state=int(sample_seed), replace=False)
    out = pd.concat([rest, kept], axis=0, ignore_index=True)
    return out, {
        "n_train_rows_after_easy_cap": int(len(out)),
        "n_easy_positives_in_rule": n_easy,
        "n_easy_positives_retained_this_epoch": int(len(kept)),
        "n_easy_positives_dropped_this_epoch": int(n_easy - len(kept)),
        "n_non_easy_positive_rows_retained": n_non_easy_pos,
        "retain_fraction_requested": float(retain_fraction),
        "retain_fraction_effective": float(len(kept) / max(1, n_easy)),
        "n_train_rows_before_easy_cap": n_total,
    }


def _estimate_train_rows_after_easy_cap(
    n_total: int, n_easy: int, retain_frac: float, enabled: bool
) -> int:
    """Expected train row count after easy-positive capping (before hard-pos / hard-unl extras)."""
    if not enabled or n_easy <= 0:
        return int(n_total)
    rf = float(np.clip(retain_frac, 0.0, 1.0))
    if rf >= 1.0 - 1e-12:
        nk = n_easy
    elif rf <= 0.0:
        nk = 1
    else:
        nk = max(1, min(n_easy, int(round(rf * n_easy))))
    return int(n_total - n_easy + nk)


def _build_train_df_epoch_emphasis(
    train_df: pd.DataFrame,
    *,
    easy_pos_mask: pd.Series,
    epc_enabled: bool,
    epc_downsample_fraction: float,
    hard_pos_mask: pd.Series,
    hpe_enabled: bool,
    hpe_oversample_factor: float,
    hard_unl_mask: pd.Series,
    hue_enabled: bool,
    hue_oversample_factor: float,
    reliable_neg_mask: pd.Series,
    rne_enabled: bool,
    rne_oversample_factor: float,
    shuffle_each_epoch: bool,
    epoch_seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Per-epoch training row stream (pair_supervision only).

    Order (intentional):
      1) Start from ``train_df``.
      2) If easy-positive capping is enabled, downsample easy-positive rows (all non-easy rows kept).
      3) Append hard-positive oversample chunks from the **original** ``train_df`` (optional).
      4) Append hard-unlabeled oversample chunks from the **original** ``train_df`` (optional).
      5) Append reliable-negative oversample chunks from the **original** ``train_df`` (optional).
      6) Shuffle the concatenated rows if configured.
    """
    n_train = int(len(train_df))
    if epc_enabled:
        df_core, epc_sub = _apply_easy_positive_cap(
            train_df,
            easy_pos_mask,
            retain_fraction=float(epc_downsample_fraction),
            sample_seed=int(epoch_seed) + 101,
        )
        epc_diag: dict[str, Any] = {"enabled": True, **epc_sub}
    else:
        df_core = train_df.copy()
        em = easy_pos_mask.reindex(train_df.index).fillna(False).astype(bool)
        n_easy_rule = int(em.sum())
        epc_diag = {
            "enabled": False,
            "n_train_rows_after_easy_cap": n_train,
            "n_easy_positives_in_rule": n_easy_rule,
            "n_easy_positives_retained_this_epoch": n_easy_rule,
            "n_easy_positives_dropped_this_epoch": 0,
            "n_non_easy_positive_rows_retained": int(
                (_safe_bool_series(train_df, "is_positive") & ~em).sum()
            ),
            "n_train_rows_before_easy_cap": n_train,
        }

    pos_extra = _oversample_extra_chunks(
        train_df,
        hard_pos_mask,
        hpe_oversample_factor if hpe_enabled else 1.0,
        epoch_seed,
    )
    unl_extra = _oversample_extra_chunks(
        train_df,
        hard_unl_mask,
        hue_oversample_factor if hue_enabled else 1.0,
        epoch_seed + 17,
    )
    neg_extra = _oversample_extra_chunks(
        train_df,
        reliable_neg_mask,
        rne_oversample_factor if rne_enabled else 1.0,
        epoch_seed + 31,
    )
    n_unl_extra = int(sum(len(x) for x in unl_extra))
    n_neg_extra = int(sum(len(x) for x in neg_extra))
    n_hard_unl_base = int(hard_unl_mask.sum())
    n_rn_base = int(reliable_neg_mask.sum())
    parts: list[pd.DataFrame] = [df_core.reset_index(drop=True), *pos_extra, *unl_extra, *neg_extra]
    out = pd.concat(parts, axis=0, ignore_index=True)
    if shuffle_each_epoch:
        out = out.sample(frac=1.0, random_state=int(epoch_seed)).reset_index(drop=True)
    else:
        out = out.reset_index(drop=True)
    diag: dict[str, Any] = {
        "per_epoch_row_pipeline": (
            "train_df -> [easy_positive_cap] -> [+hard_positive_extras] -> [+hard_unlabeled_extras] "
            "-> [+reliable_negative_extras] -> [shuffle]"
        ),
        "n_train_rows_base": n_train,
        "n_train_rows_epoch_effective": int(len(out)),
        "shuffle_each_epoch": bool(shuffle_each_epoch),
        "easy_positive_capping": {
            **epc_diag,
            "n_hard_unlabeled_rows_in_core_before_unl_extras": n_hard_unl_base,
            "n_hard_unlabeled_oversample_extra_rows": n_unl_extra,
            "n_reliable_negative_rows_in_core_before_rn_extras": n_rn_base,
            "n_reliable_negative_oversample_extra_rows": n_neg_extra,
        },
        "hard_positive": {
            "enabled": bool(hpe_enabled),
            "requested_oversample_factor": float(hpe_oversample_factor),
            "n_hard_rows_base": int(hard_pos_mask.sum()),
            "n_extra_rows_appended": int(sum(len(x) for x in pos_extra)),
        },
        "hard_unlabeled": {
            "enabled": bool(hue_enabled),
            "requested_oversample_factor": float(hue_oversample_factor),
            "n_hard_rows_base": int(hard_unl_mask.sum()),
            "n_extra_rows_appended": n_unl_extra,
        },
        "reliable_negative_emphasis": {
            "enabled": bool(rne_enabled),
            "requested_oversample_factor": float(rne_oversample_factor),
            "n_reliable_negative_rows_base": int(reliable_neg_mask.sum()),
            "n_extra_rows_appended": n_neg_extra,
        },
    }
    return out, diag


def forward_encoder_and_pair_logits(
    model: HeteroSAGE,
    pair_scorer: EmailPairMLPScorer,
    sample: PairEndpointHeteroSample,
    pair_feats: torch.Tensor | None,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, PairBatchDiag, Any]:
    """
    HeteroSAGE on ``sample.hetero_batch``, then pair logits from local endpoint indices.

    Invalid pair rows (missing endpoint in subgraph) still receive logits from clamped
    gather indices; loss must mask them (``pair_ok_mask``). Gradients from masked-out
    logits are zero.
    """
    hetero_batch = sample.hetero_batch.to(device)
    h = model(hetero_batch.x_dict, hetero_batch.edge_index_dict)
    z_all = h["email"]
    ok_mask_t = sample.pair_ok_mask.to(device)
    li = sample.pair_local_i.to(device).clamp(min=0)
    lj = sample.pair_local_j.to(device).clamp(min=0)
    z_i = z_all[li]
    z_j = z_all[lj]
    if pair_scorer.use_explicit_pair_features:
        logits = pair_scorer(z_i, z_j, pair_feats.to(device) if pair_feats is not None else None)
    else:
        logits = pair_scorer(z_i, z_j, None)
    cov = sample.coverage
    diag = PairBatchDiag(
        n_pairs=cov.n_pairs_requested,
        n_unique_emails=int(len(sample.global_to_local_email)),
        n_pairs_mapped_ok=cov.n_both_endpoints_present,
        n_pairs_missing_endpoint=cov.n_pairs_requested - cov.n_both_endpoints_present,
    )
    return logits, ok_mask_t, diag, cov


def _quantile_dict(scores: list[float], *, prefix: str) -> dict[str, float]:
    if not scores:
        return {f"{prefix}_q10": float("nan"), f"{prefix}_q50": float("nan"), f"{prefix}_q90": float("nan")}
    a = np.asarray(scores, dtype=np.float64)
    return {
        f"{prefix}_q10": float(np.quantile(a, 0.1)),
        f"{prefix}_q50": float(np.quantile(a, 0.5)),
        f"{prefix}_q90": float(np.quantile(a, 0.9)),
    }


PAIR_METRICS_HEADER = [
    "epoch",
    "pair_loss_type",
    "pi_p",
    "train_loss",
    "val_loss",
    "train_placeholder_acc",
    "val_placeholder_acc",
    "train_epoch_mean_r_p_pos",
    "train_epoch_mean_r_p_neg",
    "train_epoch_mean_r_u_neg",
    "train_epoch_mean_neg_risk_raw",
    "train_epoch_mean_neg_risk_after_nn",
    "val_epoch_mean_r_p_pos",
    "val_epoch_mean_r_p_neg",
    "val_epoch_mean_r_u_neg",
    "val_epoch_mean_neg_risk_raw",
    "val_epoch_mean_neg_risk_after_nn",
    "train_epoch_mean_pos_prob",
    "train_epoch_mean_unl_prob",
    "train_epoch_score_separation",
    "val_epoch_mean_pos_prob",
    "val_epoch_mean_unl_prob",
    "val_epoch_score_separation",
    "train_epoch_sum_n_positive",
    "train_epoch_sum_n_unlabeled",
    "val_epoch_sum_n_positive",
    "val_epoch_sum_n_unlabeled",
    "val_pos_score_q10",
    "val_pos_score_q50",
    "val_pos_score_q90",
    "val_unl_score_q10",
    "val_unl_score_q50",
    "val_unl_score_q90",
    "val_frac_pos_above_threshold",
    "val_frac_unl_above_threshold",
    "val_separation_at_threshold",
    "train_min_recoverable_pair_frac",
    "val_avg_map_rate",
    "train_effective_n_rows_epoch",
    "train_effective_n_positive_epoch",
    "train_effective_n_unlabeled_epoch",
    "train_effective_n_hard_unlabeled_epoch",
    "train_effective_n_nonhard_unlabeled_epoch",
    "train_easy_pos_retained_epoch",
    "train_easy_pos_dropped_epoch",
    "train_non_easy_pos_retained_epoch",
    "train_hard_unl_oversample_extra_rows_epoch",
    "train_epoch_sum_n_reliable_negative",
    "val_epoch_sum_n_reliable_negative",
    "train_epoch_mean_neg_supervised_bce",
    "val_epoch_mean_neg_supervised_bce",
    "train_effective_n_reliable_negative_epoch",
    "train_rn_oversample_extra_rows_epoch",
]


def _csv_scalar(v: Any) -> Any:
    if isinstance(v, float) and (v != v or np.isnan(v)):
        return ""
    return v


def metrics_row_pair_training(
    epoch: int,
    *,
    pair_loss_type: str,
    pi_p: float,
    train_loss: float,
    val_loss: float,
    train_pu: dict[str, Any],
    val_pu: dict[str, Any],
    train_agg: dict[str, Any],
    val_agg: dict[str, Any],
) -> list[Any]:
    def g(d: dict[str, Any], k: str) -> Any:
        return _csv_scalar(d.get(k, ""))

    return [
        epoch,
        pair_loss_type,
        pi_p,
        train_loss,
        val_loss,
        g(train_pu, "epoch_placeholder_acc"),
        g(val_pu, "epoch_placeholder_acc"),
        g(train_pu, "epoch_mean_r_p_pos"),
        g(train_pu, "epoch_mean_r_p_neg"),
        g(train_pu, "epoch_mean_r_u_neg"),
        g(train_pu, "epoch_mean_neg_risk_raw"),
        g(train_pu, "epoch_mean_neg_risk_after_nn"),
        g(val_pu, "epoch_mean_r_p_pos"),
        g(val_pu, "epoch_mean_r_p_neg"),
        g(val_pu, "epoch_mean_r_u_neg"),
        g(val_pu, "epoch_mean_neg_risk_raw"),
        g(val_pu, "epoch_mean_neg_risk_after_nn"),
        g(train_pu, "epoch_mean_pos_prob"),
        g(train_pu, "epoch_mean_unl_prob"),
        g(train_pu, "epoch_score_separation"),
        g(val_pu, "epoch_mean_pos_prob"),
        g(val_pu, "epoch_mean_unl_prob"),
        g(val_pu, "epoch_score_separation"),
        g(train_pu, "epoch_sum_n_positive"),
        g(train_pu, "epoch_sum_n_unlabeled"),
        g(val_pu, "epoch_sum_n_positive"),
        g(val_pu, "epoch_sum_n_unlabeled"),
        g(val_pu, "val_pos_score_q10"),
        g(val_pu, "val_pos_score_q50"),
        g(val_pu, "val_pos_score_q90"),
        g(val_pu, "val_unl_score_q10"),
        g(val_pu, "val_unl_score_q50"),
        g(val_pu, "val_unl_score_q90"),
        g(val_pu, "val_frac_pos_above_threshold"),
        g(val_pu, "val_frac_unl_above_threshold"),
        g(val_pu, "val_separation_at_threshold"),
        train_agg.get("min_recoverable_pair_fraction", ""),
        val_agg.get("avg_pair_endpoint_map_rate", ""),
        g(train_agg, "n_effective_train_rows_epoch"),
        g(train_agg, "n_effective_positive_rows_epoch"),
        g(train_agg, "n_effective_unlabeled_rows_epoch"),
        g(train_agg, "n_effective_hard_unlabeled_rows_epoch"),
        g(train_agg, "n_effective_non_hard_unlabeled_rows_epoch"),
        g(train_agg, "n_easy_positive_retained_epoch"),
        g(train_agg, "n_easy_positive_dropped_epoch"),
        g(train_agg, "n_non_easy_positive_retained_epoch"),
        g(train_agg, "n_hard_unlabeled_oversample_extra_rows_epoch"),
        g(train_pu, "epoch_sum_n_reliable_negative"),
        g(val_pu, "epoch_sum_n_reliable_negative"),
        g(train_pu, "epoch_mean_neg_supervised_bce"),
        g(val_pu, "epoch_mean_neg_supervised_bce"),
        g(train_agg, "n_effective_reliable_negative_rows_epoch"),
        g(train_agg, "n_reliable_negative_oversample_extra_rows_epoch"),
    ]


def train_pair_epoch(
    *,
    model: HeteroSAGE,
    pair_scorer: EmailPairMLPScorer,
    optimizer: torch.optim.Optimizer,
    data_cpu: Any,
    df: pd.DataFrame,
    device: torch.device,
    pair_batch_size: int,
    max_unique_emails: int,
    fanout: list[int],
    pair_loss_type: str,
    pi_p: float,
    pu_non_negative: bool,
    reliable_negative_loss_weight: float = 1.0,
    pair_assert_full_endpoint_coverage: bool = False,
    pair_skip_graph_batch_if_any_endpoint_missing: bool = False,
    pair_log_mapping_misses: bool = True,
    pair_tqdm: bool = True,
    tqdm_total_batches: int | None = None,
    tqdm_desc: str = "train pairs",
) -> tuple[float, dict[str, float], dict[str, float]]:
    model.train()
    pair_scorer.train()
    total_loss = 0.0
    n_batches = 0
    batch_pu_diags: list[dict[str, Any]] = []
    sum_unique = 0.0
    sum_mapped = 0.0
    sum_pairs = 0.0
    seed_sizes: list[int] = []
    het_nodes: list[int] = []
    het_edges: list[int] = []
    email_nodes: list[int] = []
    recover_fracs: list[float] = []
    n_skipped_incomplete = 0
    n_miss_rows = 0
    catastrophic = False
    logged_miss = False

    batch_iter = iter_pair_batches(df, pair_batch_size, max_unique_emails)
    if pair_tqdm:
        batch_iter = tqdm(
            batch_iter,
            total=tqdm_total_batches,
            desc=tqdm_desc,
            unit="batch",
            leave=False,
            mininterval=0.3,
        )
    for chunk, gi, gj in batch_iter:
        sample = sample_hetero_around_pair_endpoints(data_cpu, gi, gj, fanout)
        cov = sample.coverage
        if pair_assert_full_endpoint_coverage and cov.n_both_endpoints_present < cov.n_pairs_requested:
            raise AssertionError(
                f"pair_assert_full_endpoint_coverage: batch has missing endpoints "
                f"(ok={cov.n_both_endpoints_present}/{cov.n_pairs_requested}); "
                f"missing_i_only={cov.n_missing_i_only} missing_j_only={cov.n_missing_j_only} "
                f"missing_both={cov.n_missing_both_endpoints}"
            )
        if pair_skip_graph_batch_if_any_endpoint_missing and cov.n_both_endpoints_present < cov.n_pairs_requested:
            n_skipped_incomplete += 1
            continue
        if pair_log_mapping_misses and not logged_miss and cov.n_pairs_requested > cov.n_both_endpoints_present:
            print(
                f"[pair_supervision][mapping] example batch: requested_pairs={cov.n_pairs_requested} "
                f"both_present={cov.n_both_endpoints_present} missing_i_only={cov.n_missing_i_only} "
                f"missing_j_only={cov.n_missing_j_only} missing_both={cov.n_missing_both_endpoints} "
                f"(fraction_usable={cov.frac_usable_pairs:.4f})"
            )
            logged_miss = True
        n_miss_rows += cov.n_pairs_requested - cov.n_both_endpoints_present
        if cov.frac_usable_pairs == 0.0 and cov.n_pairs_requested > 0:
            catastrophic = True

        feats: torch.Tensor | None = None
        if pair_scorer.use_explicit_pair_features:
            feats = torch.from_numpy(build_pair_feature_matrix(chunk))

        is_pos = torch.tensor(chunk["is_positive"].values, dtype=torch.bool, device=device)
        is_unl = torch.tensor(chunk["is_unlabeled"].values, dtype=torch.bool, device=device)
        if "is_reliable_negative" in chunk.columns:
            is_neg = torch.tensor(chunk["is_reliable_negative"].values, dtype=torch.bool, device=device)
        else:
            is_neg = torch.zeros(len(chunk), dtype=torch.bool, device=device)

        optimizer.zero_grad()
        logits, ok_m, diag, _ = forward_encoder_and_pair_logits(
            model, pair_scorer, sample, feats, device
        )
        if not ok_m.any():
            continue
        loss, batch_diag = compute_pair_loss(
            logits[ok_m],
            is_pos[ok_m],
            is_unl[ok_m],
            pair_loss_type,
            pi_p=pi_p,
            pu_non_negative=pu_non_negative,
            is_reliable_negative=is_neg[ok_m],
            reliable_negative_loss_weight=reliable_negative_loss_weight,
        )
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())
        n_batches += 1
        batch_pu_diags.append(batch_diag)
        sum_unique += diag.n_unique_emails
        sum_mapped += diag.n_pairs_mapped_ok / max(1, diag.n_pairs)
        sum_pairs += diag.n_pairs
        seed_sizes.append(int(sample.seed_global_email_indices.numel()))
        het_nodes.append(sample.hetero_total_nodes)
        het_edges.append(sample.hetero_total_edges)
        email_nodes.append(sample.n_email_nodes_in_batch)
        recover_fracs.append(float(cov.frac_usable_pairs))

    def _mean(xs: list[float | int]) -> float:
        return float(sum(xs) / len(xs)) if xs else 0.0

    agg: dict[str, float | bool | int] = {
        "avg_unique_emails_per_batch": float(sum_unique / max(n_batches, 1)),
        "avg_pair_endpoint_map_rate": float(sum_mapped / max(n_batches, 1)),
        "avg_pairs_per_batch": float(sum_pairs / max(n_batches, 1)),
        "avg_unique_seed_emails_per_graph_batch": _mean(seed_sizes),
        "avg_sampled_hetero_total_nodes": _mean(het_nodes),
        "avg_sampled_hetero_total_edges": _mean(het_edges),
        "avg_n_email_nodes_in_sampled_batch": _mean(email_nodes),
        "avg_recoverable_pair_fraction": _mean(recover_fracs),
        "min_recoverable_pair_fraction": float(min(recover_fracs)) if recover_fracs else 0.0,
        "max_recoverable_pair_fraction": float(max(recover_fracs)) if recover_fracs else 0.0,
        "n_graph_batches_with_grad": int(n_batches),
        "n_skipped_batches_incomplete_endpoints": int(n_skipped_incomplete),
        "n_pair_rows_missing_endpoint_across_epoch": int(n_miss_rows),
        "any_catastrophic_batch": bool(catastrophic),
    }
    pu_epoch = aggregate_epoch_pu_stats(batch_pu_diags, pair_loss_type)
    return total_loss / max(n_batches, 1), pu_epoch, agg


@torch.no_grad()
def eval_pair_epoch(
    *,
    model: HeteroSAGE,
    pair_scorer: EmailPairMLPScorer,
    data_cpu: Any,
    df: pd.DataFrame,
    device: torch.device,
    pair_batch_size: int,
    max_unique_emails: int,
    fanout: list[int],
    pair_loss_type: str,
    pi_p: float,
    pu_non_negative: bool,
    reliable_negative_loss_weight: float = 1.0,
    pair_eval_threshold: float = 0.5,
    pair_assert_full_endpoint_coverage: bool = False,
    pair_skip_graph_batch_if_any_endpoint_missing: bool = False,
    pair_tqdm: bool = True,
    tqdm_total_batches: int | None = None,
    tqdm_desc: str = "eval pairs",
) -> tuple[float, dict[str, float], dict[str, float]]:
    model.eval()
    pair_scorer.eval()
    total_loss = 0.0
    n_batches = 0
    batch_pu_diags: list[dict[str, Any]] = []
    all_pos_scores: list[float] = []
    all_unl_scores: list[float] = []
    sum_unique = 0.0
    sum_mapped = 0.0
    sum_pairs = 0.0
    seed_sizes: list[int] = []
    het_nodes: list[int] = []
    het_edges: list[int] = []
    email_nodes: list[int] = []
    recover_fracs: list[float] = []
    catastrophic = False

    batch_iter = iter_pair_batches(df, pair_batch_size, max_unique_emails)
    if pair_tqdm:
        batch_iter = tqdm(
            batch_iter,
            total=tqdm_total_batches,
            desc=tqdm_desc,
            unit="batch",
            leave=False,
            mininterval=0.3,
        )
    for chunk, gi, gj in batch_iter:
        sample = sample_hetero_around_pair_endpoints(data_cpu, gi, gj, fanout)
        cov = sample.coverage
        if pair_assert_full_endpoint_coverage and cov.n_both_endpoints_present < cov.n_pairs_requested:
            raise AssertionError(
                f"pair_assert_full_endpoint_coverage (eval): ok={cov.n_both_endpoints_present}/"
                f"{cov.n_pairs_requested}"
            )
        if pair_skip_graph_batch_if_any_endpoint_missing and cov.n_both_endpoints_present < cov.n_pairs_requested:
            continue
        if cov.frac_usable_pairs == 0.0 and cov.n_pairs_requested > 0:
            catastrophic = True

        feats: torch.Tensor | None = None
        if pair_scorer.use_explicit_pair_features:
            feats = torch.from_numpy(build_pair_feature_matrix(chunk))
        is_pos = torch.tensor(chunk["is_positive"].values, dtype=torch.bool, device=device)
        is_unl = torch.tensor(chunk["is_unlabeled"].values, dtype=torch.bool, device=device)
        if "is_reliable_negative" in chunk.columns:
            is_neg = torch.tensor(chunk["is_reliable_negative"].values, dtype=torch.bool, device=device)
        else:
            is_neg = torch.zeros(len(chunk), dtype=torch.bool, device=device)

        logits, ok_m, diag, _ = forward_encoder_and_pair_logits(
            model, pair_scorer, sample, feats, device
        )
        if not ok_m.any():
            continue
        loss, batch_diag = compute_pair_loss(
            logits[ok_m],
            is_pos[ok_m],
            is_unl[ok_m],
            pair_loss_type,
            pi_p=pi_p,
            pu_non_negative=pu_non_negative,
            is_reliable_negative=is_neg[ok_m],
            reliable_negative_loss_weight=reliable_negative_loss_weight,
        )
        total_loss += float(loss.item())
        n_batches += 1
        batch_pu_diags.append(batch_diag)
        probs = torch.sigmoid(logits[ok_m])
        pk, uk = exclusive_pair_masks(is_pos[ok_m], is_unl[ok_m])
        if pk.any():
            all_pos_scores.extend(probs[pk].detach().cpu().tolist())
        if uk.any():
            all_unl_scores.extend(probs[uk].detach().cpu().tolist())
        sum_unique += diag.n_unique_emails
        sum_mapped += diag.n_pairs_mapped_ok / max(1, diag.n_pairs)
        sum_pairs += diag.n_pairs
        seed_sizes.append(int(sample.seed_global_email_indices.numel()))
        het_nodes.append(sample.hetero_total_nodes)
        het_edges.append(sample.hetero_total_edges)
        email_nodes.append(sample.n_email_nodes_in_batch)
        recover_fracs.append(float(cov.frac_usable_pairs))

    def _mean(xs: list[float | int]) -> float:
        return float(sum(xs) / len(xs)) if xs else 0.0

    agg: dict[str, float | bool | int] = {
        "avg_unique_emails_per_batch": float(sum_unique / max(n_batches, 1)),
        "avg_pair_endpoint_map_rate": float(sum_mapped / max(n_batches, 1)),
        "avg_pairs_per_batch": float(sum_pairs / max(n_batches, 1)),
        "avg_unique_seed_emails_per_graph_batch": _mean(seed_sizes),
        "avg_sampled_hetero_total_nodes": _mean(het_nodes),
        "avg_sampled_hetero_total_edges": _mean(het_edges),
        "avg_n_email_nodes_in_sampled_batch": _mean(email_nodes),
        "avg_recoverable_pair_fraction": _mean(recover_fracs),
        "min_recoverable_pair_fraction": float(min(recover_fracs)) if recover_fracs else 0.0,
        "max_recoverable_pair_fraction": float(max(recover_fracs)) if recover_fracs else 0.0,
        "n_graph_batches": int(n_batches),
        "any_catastrophic_batch": bool(catastrophic),
    }
    pu_epoch = aggregate_epoch_pu_stats(batch_pu_diags, pair_loss_type)
    pu_epoch.update(_quantile_dict(all_pos_scores, prefix="val_pos_score"))
    pu_epoch.update(_quantile_dict(all_unl_scores, prefix="val_unl_score"))
    thr = float(pair_eval_threshold)
    if all_pos_scores:
        ap = np.asarray(all_pos_scores, dtype=np.float64)
        pu_epoch["val_frac_pos_above_threshold"] = float((ap >= thr).mean())
    else:
        pu_epoch["val_frac_pos_above_threshold"] = float("nan")
    if all_unl_scores:
        au = np.asarray(all_unl_scores, dtype=np.float64)
        pu_epoch["val_frac_unl_above_threshold"] = float((au >= thr).mean())
    else:
        pu_epoch["val_frac_unl_above_threshold"] = float("nan")
    fp = pu_epoch.get("val_frac_pos_above_threshold", float("nan"))
    fu = pu_epoch.get("val_frac_unl_above_threshold", float("nan"))
    if fp == fp and fu == fu:
        pu_epoch["val_separation_at_threshold"] = float(fp - fu)
    else:
        pu_epoch["val_separation_at_threshold"] = float("nan")
    return total_loss / max(n_batches, 1), pu_epoch, agg


@torch.no_grad()
def probe_pair_forward_shapes(
    model: HeteroSAGE,
    pair_scorer: EmailPairMLPScorer,
    data_cpu: Any,
    df: pd.DataFrame,
    device: torch.device,
    pair_batch_size: int,
    max_unique_emails: int,
    fanout: list[int],
) -> dict[str, Any]:
    """One forward on the first non-empty pair batch — validates shapes and endpoint recovery."""
    if len(df) == 0:
        return {"error": "empty_dataframe"}
    model.eval()
    pair_scorer.eval()
    for chunk, gi, gj in iter_pair_batches(df, pair_batch_size, max_unique_emails):
        sample = sample_hetero_around_pair_endpoints(data_cpu, gi, gj, fanout)
        feats: torch.Tensor | None = None
        if pair_scorer.use_explicit_pair_features:
            feats = torch.from_numpy(build_pair_feature_matrix(chunk))
        logits, ok_m, _diag, cov = forward_encoder_and_pair_logits(
            model, pair_scorer, sample, feats, device
        )
        meta_shape = list(feats.shape) if feats is not None else None
        return {
            "probe_n_pair_rows": int(len(gi)),
            "logits_shape": list(logits.shape),
            "n_usable_pairs": int(ok_m.sum().item()),
            "pair_metadata_tensor_shape": meta_shape,
            "encoder_email_embed_dim": int(pair_scorer.embed_dim),
            "scorer_mlp_input_dim": int(pair_scorer.input_feature_dim),
            "recoverable_pair_fraction_this_batch": float(cov.frac_usable_pairs),
            "pair_ok_mask_shape": list(ok_m.shape),
        }
    return {"error": "no_batches"}


def save_pair_training_checkpoint(
    *,
    save_dir: Path,
    filename: str,
    model: HeteroSAGE,
    pair_scorer: EmailPairMLPScorer,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_loss: float,
    encoder_config: dict[str, Any],
    data_metadata: Any,
    torch_seed: int,
    pair_training_config: dict[str, Any],
    patience_counter: int,
    best_val: float,
    best_model_state: dict,
    best_pair_scorer_state: dict,
    training_params: dict[str, Any],
) -> Path:
    save_dir.mkdir(parents=True, exist_ok=True)
    path = save_dir / filename
    payload = {
        "training_objective": "pair_supervision",
        "epoch": int(epoch),
        "val_loss": float(val_loss),
        "model_state_dict": model.state_dict(),
        "pair_scorer_state_dict": pair_scorer.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "encoder_config": encoder_config,
        "data_metadata": data_metadata,
        "torch_seed": int(torch_seed),
        "pair_training_config": pair_training_config,
        "patience_counter": int(patience_counter),
        "best_val": float(best_val),
        "best_model_state_dict": best_model_state,
        "best_pair_scorer_state_dict": best_pair_scorer_state,
        "training_params": training_params,
    }
    torch.save(payload, path)  # nosemgrep: trailofbits.python.pickles-in-pytorch.pickles-in-pytorch
    return path


def run_pair_training(
    *,
    DEVICE: torch.device,
    TORCH_SEED: int,
    data: Any,
    training_cfg: dict[str, Any],
    run_dir: str | Path,
    runs_parent: str | Path | None,
    models_subdir: str,
    metrics_csv: str,
    training_config_json: str,
    project_root: Path | None = None,
) -> dict[str, Any]:
    """
    Parallel training path for email-email pair supervision.

    Default loss: nnPU (``pair_loss_type=nnpu``). Legacy placeholder BCE remains for debugging.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = run_dir / models_subdir
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    root = project_root or _project_root_from_here()
    raw_csv = training_cfg.get("pair_dataset_csv")
    if not raw_csv:
        raise ValueError("pair_dataset_csv is required in training_cfg for pair_supervision.")
    csv_path = resolve_pair_dataset_csv(str(raw_csv), project_root=root)

    pair_batch_size = int(training_cfg.get("pair_batch_size", 64))
    max_unique = int(training_cfg.get("pair_max_unique_emails_per_graph_batch", 2048))
    fanout = list(training_cfg.get("pair_fanout") or training_cfg.get("fanout") or [25, 15])
    val_ratio = float(training_cfg.get("pair_val_ratio", training_cfg.get("val_ratio", 0.1)))
    test_ratio = float(training_cfg.get("pair_test_ratio", training_cfg.get("test_ratio", 0.1)))
    split_seed = int(training_cfg.get("pair_split_seed", training_cfg.get("torch_seed", 42)))
    pair_loss_type = resolve_pair_loss_type(training_cfg)
    placeholder_mode = str(
        training_cfg.get("pair_placeholder_loss_mode") or PLACEHOLDER_LOSS_BCE_POS_VS_UNLABELED_AS_NEG
    )
    pi_p = float(training_cfg.get("pu_class_prior", training_cfg.get("pi_p", 0.1)))
    pu_non_negative = bool(training_cfg.get("pu_non_negative", True))
    pair_eval_threshold = float(training_cfg.get("pair_eval_threshold", 0.5))
    use_explicit_pair_feats = bool(training_cfg.get("pair_scorer_use_explicit_features", True))
    pair_assert_full_endpoint_coverage = bool(training_cfg.get("pair_assert_full_endpoint_coverage", False))
    pair_skip_graph_batch_if_any_endpoint_missing = bool(
        training_cfg.get("pair_skip_graph_batch_if_any_endpoint_missing", False)
    )
    pair_log_mapping_misses = bool(training_cfg.get("pair_log_mapping_misses", True))
    pair_tqdm = bool(training_cfg.get("pair_tqdm_batches", True))
    raw_diag_max = training_cfg.get("pair_sampling_diag_max_batches", 200)
    if raw_diag_max is None or (isinstance(raw_diag_max, str) and raw_diag_max.lower() in ("", "none", "null")):
        pair_sampling_diag_max_batches: int | None = None
    else:
        pair_sampling_diag_max_batches = int(raw_diag_max)
    hpe_cfg = training_cfg.get("hard_positive_emphasis") or {}
    hpe_enabled = bool(hpe_cfg.get("enabled", False))
    hpe_oversample_factor = float(hpe_cfg.get("oversample_factor", 1.0))
    hpe_cross_seed_component_only = bool(hpe_cfg.get("cross_seed_component_only", True))
    hpe_require_from_2hop = bool(hpe_cfg.get("require_from_2hop", True))
    hpe_max_source_count_raw = hpe_cfg.get("max_source_count")
    hpe_max_source_count: int | None
    if hpe_max_source_count_raw is None or str(hpe_max_source_count_raw).strip().lower() in ("", "none", "null"):
        hpe_max_source_count = None
    else:
        hpe_max_source_count = int(hpe_max_source_count_raw)
    hpe_exclude_from_rare_artifact = bool(hpe_cfg.get("exclude_from_rare_artifact", False))
    hpe_require_not_same_seed_component = bool(hpe_cfg.get("require_not_same_seed_component", True))
    hpe_shuffle_each_epoch = bool(hpe_cfg.get("shuffle_each_epoch", True))

    hue_cfg = training_cfg.get("hard_unlabeled_emphasis") or {}
    hue_enabled = bool(hue_cfg.get("enabled", False))
    hue_oversample_factor = float(hue_cfg.get("oversample_factor", 1.0))
    hue_cross_seed_component_only = bool(hue_cfg.get("cross_seed_component_only", True))
    hue_require_from_2hop = bool(hue_cfg.get("require_from_2hop", True))
    hue_max_source_count_raw = hue_cfg.get("max_source_count")
    hue_max_source_count: int | None
    if hue_max_source_count_raw is None or str(hue_max_source_count_raw).strip().lower() in ("", "none", "null"):
        hue_max_source_count = None
    else:
        hue_max_source_count = int(hue_max_source_count_raw)
    hue_exclude_from_rare_artifact = bool(hue_cfg.get("exclude_from_rare_artifact", True))
    hue_require_not_same_seed_component = bool(hue_cfg.get("require_not_same_seed_component", False))
    hue_require_from_semantic_false = bool(hue_cfg.get("require_from_semantic_false", False))
    hue_shuffle_each_epoch = bool(hue_cfg.get("shuffle_each_epoch", True))

    epc_cfg = training_cfg.get("easy_positive_capping") or {}
    epc_enabled = bool(epc_cfg.get("enabled", False))
    epc_downsample_fraction = float(epc_cfg.get("downsample_fraction", 1.0))
    epc_same_seed_component_only = bool(epc_cfg.get("same_seed_component_only", True))
    epc_min_sem_raw = epc_cfg.get("min_semantic_cosine")
    epc_min_semantic_cosine: float | None
    if epc_min_sem_raw is None or str(epc_min_sem_raw).strip().lower() in ("", "none", "null"):
        epc_min_semantic_cosine = None
    else:
        epc_min_semantic_cosine = float(epc_min_sem_raw)
    epc_min_sc_raw = epc_cfg.get("min_source_count")
    epc_min_source_count: int | None
    if epc_min_sc_raw is None or str(epc_min_sc_raw).strip().lower() in ("", "none", "null"):
        epc_min_source_count = None
    else:
        epc_min_source_count = int(epc_min_sc_raw)
    epc_or_rule_across_conditions = bool(epc_cfg.get("or_rule_across_conditions", True))
    epc_shuffle_each_epoch = bool(epc_cfg.get("shuffle_each_epoch", True))

    rne_cfg = training_cfg.get("reliable_negative_emphasis") or {}
    rne_enabled = bool(rne_cfg.get("enabled", False))
    rne_oversample_factor = float(rne_cfg.get("oversample_factor", 1.0))
    rne_shuffle_each_epoch = bool(rne_cfg.get("shuffle_each_epoch", True))
    reliable_negative_loss_weight = float(training_cfg.get("reliable_negative_loss_weight", 1.0))

    shuffle_train_epoch = (
        (hpe_enabled and hpe_shuffle_each_epoch)
        or (hue_enabled and hue_shuffle_each_epoch)
        or (epc_enabled and epc_shuffle_each_epoch)
        or (rne_enabled and rne_shuffle_each_epoch)
    )

    epochs = int(training_cfg["epochs"])
    lr = float(training_cfg["lr"])
    wd = float(training_cfg["wd"])
    hidden = int(training_cfg["hidden"])
    out_dim = int(training_cfg["out_dim"])
    layers = int(training_cfg["layers"])
    dropout = float(training_cfg["dropout"])
    early_stopping_patience = int(training_cfg["early_stopping_patience"])
    lr_reduce_patience = int(training_cfg["lr_reduce_patience"])
    lr_reduce_factor = float(training_cfg["lr_reduce_factor"])
    lr_reduce_min = float(training_cfg["lr_reduce_min"])
    model_save_name = str(training_cfg["model_save_name"])

    torch.manual_seed(TORCH_SEED)
    np.random.seed(TORCH_SEED)

    df, load_stats = load_pair_training_dataframe(csv_path)
    train_df, val_df, test_df = split_pairs_train_val_test(
        df, val_ratio=val_ratio, test_ratio=test_ratio, split_seed=split_seed
    )
    hard_pos_mask_train = _hard_positive_mask(
        train_df,
        cross_seed_component_only=hpe_cross_seed_component_only,
        require_from_2hop=hpe_require_from_2hop,
        max_source_count=hpe_max_source_count,
        exclude_from_rare_artifact=hpe_exclude_from_rare_artifact,
        require_not_same_seed_component=hpe_require_not_same_seed_component,
    )
    hard_unl_mask_train = _hard_unlabeled_mask(
        train_df,
        cross_seed_component_only=hue_cross_seed_component_only,
        require_from_2hop=hue_require_from_2hop,
        max_source_count=hue_max_source_count,
        exclude_from_rare_artifact=hue_exclude_from_rare_artifact,
        require_not_same_seed_component=hue_require_not_same_seed_component,
        require_from_semantic_false=hue_require_from_semantic_false,
    )
    easy_pos_mask_train = _easy_positive_mask(
        train_df,
        same_seed_component_only=epc_same_seed_component_only,
        min_semantic_cosine=epc_min_semantic_cosine,
        min_source_count=epc_min_source_count,
        or_rule_across_conditions=epc_or_rule_across_conditions,
    )
    reliable_neg_mask_train = _safe_bool_series(train_df, "is_reliable_negative", default=False)

    n_train_batches: int | None = None
    n_val_batches: int | None = None
    n_test_batches: int | None = None
    if pair_tqdm:
        n_train_batches = count_pair_batches(train_df, pair_batch_size, max_unique)
        n_val_batches = count_pair_batches(val_df, pair_batch_size, max_unique)
        n_test_batches = count_pair_batches(test_df, pair_batch_size, max_unique)

    feat_dim_columns = len(PAIR_FEATURE_COLUMNS)
    pair_feat_dim_for_scorer = int(feat_dim_columns) if use_explicit_pair_feats else 0

    data_cpu = data.to("cpu")
    metadata = data_cpu.metadata()
    sampling_diagnostics = collect_pair_sampling_diagnostics(
        data_cpu,
        train_df,
        pair_batch_iter=iter_pair_batches,
        pair_batch_size=pair_batch_size,
        max_unique_emails=max_unique,
        fanout=fanout,
        max_batches=pair_sampling_diag_max_batches,
        assert_full_endpoint_coverage=pair_assert_full_endpoint_coverage,
    )

    model = HeteroSAGE(metadata=metadata, hidden=hidden, out=out_dim, layers=layers, dropout=dropout).to(DEVICE)
    pair_scorer = build_email_pair_mlp_scorer(
        embed_dim=out_dim,
        pair_feat_dim=pair_feat_dim_for_scorer,
        training_cfg=training_cfg,
    ).to(DEVICE)
    probe_shapes = probe_pair_forward_shapes(
        model,
        pair_scorer,
        data_cpu,
        train_df,
        DEVICE,
        pair_batch_size,
        max_unique,
        fanout,
    )
    scorer_type = str(training_cfg.get("pair_scorer_type", "email_pair_mlp")).lower().strip()
    scorer_param_count = count_scorer_parameters(pair_scorer)

    setup_summary: dict[str, Any] = {
        "metadata": {
            "created_at_utc": datetime.now().isoformat(timespec="seconds"),
            "pair_dataset_csv": str(csv_path),
            "load_stats": load_stats,
        },
        "split": {
            "pair_val_ratio": val_ratio,
            "pair_test_ratio": test_ratio,
            "pair_split_seed": split_seed,
            "n_train_pairs": int(len(train_df)),
            "n_val_pairs": int(len(val_df)),
            "n_test_pairs": int(len(test_df)),
            "train_positive": int(train_df["is_positive"].sum()),
            "train_unlabeled": int(train_df["is_unlabeled"].sum()),
            "train_reliable_negative": int(train_df["is_reliable_negative"].sum()),
            "val_positive": int(val_df["is_positive"].sum()),
            "val_unlabeled": int(val_df["is_unlabeled"].sum()),
            "val_reliable_negative": int(val_df["is_reliable_negative"].sum()),
            "test_positive": int(test_df["is_positive"].sum()),
            "test_unlabeled": int(test_df["is_unlabeled"].sum()),
            "test_reliable_negative": int(test_df["is_reliable_negative"].sum()),
            "split_note": (
                "Train/val/test use one shuffled index permutation over all rows (positives, unlabeled, "
                "reliable_negative together). Counts above show how many reliable negatives landed in each split."
            ),
        },
        "batching": {
            "pair_batch_size": pair_batch_size,
            "pair_max_unique_emails_per_graph_batch": max_unique,
            "pair_fanout": fanout,
            "pair_tqdm_batches": pair_tqdm,
            "n_train_pair_batches_per_epoch": n_train_batches,
            "n_val_pair_batches_per_epoch": n_val_batches,
            "n_test_pair_batches_per_pass": n_test_batches,
        },
        "sampling_config": {
            "pair_sampling_diag_max_batches": pair_sampling_diag_max_batches,
            "pair_assert_full_endpoint_coverage": pair_assert_full_endpoint_coverage,
            "pair_skip_graph_batch_if_any_endpoint_missing": pair_skip_graph_batch_if_any_endpoint_missing,
            "pair_log_mapping_misses": pair_log_mapping_misses,
            "neighbor_loader_input_nodes": "('email', unique_seed_indices_from_pairs)",
            "hetero_schema_note": "Full hetero evidence graph (all node/edge types); no email-only collapse.",
        },
        "hard_positive_emphasis": {
            "enabled": bool(hpe_enabled),
            "oversample_factor": float(hpe_oversample_factor),
            "cross_seed_component_only": bool(hpe_cross_seed_component_only),
            "require_from_2hop": bool(hpe_require_from_2hop),
            "max_source_count": hpe_max_source_count,
            "exclude_from_rare_artifact": bool(hpe_exclude_from_rare_artifact),
            "require_not_same_seed_component": bool(hpe_require_not_same_seed_component),
            "shuffle_each_epoch": bool(hpe_shuffle_each_epoch),
            "n_hard_positive_rows_train_base": int(hard_pos_mask_train.sum()),
            "n_positive_rows_train_base": int(train_df["is_positive"].sum()),
            "fraction_hard_among_train_positives": float(
                hard_pos_mask_train.sum() / max(1, int(train_df["is_positive"].sum()))
            ),
            "n_train_rows_base": int(len(train_df)),
            "n_train_rows_effective_if_enabled": int(
                len(train_df) + max(0.0, hpe_oversample_factor - 1.0) * int(hard_pos_mask_train.sum())
            ),
        },
        "hard_unlabeled_emphasis": {
            "enabled": bool(hue_enabled),
            "oversample_factor": float(hue_oversample_factor),
            "cross_seed_component_only": bool(hue_cross_seed_component_only),
            "require_from_2hop": bool(hue_require_from_2hop),
            "max_source_count": hue_max_source_count,
            "exclude_from_rare_artifact": bool(hue_exclude_from_rare_artifact),
            "require_not_same_seed_component": bool(hue_require_not_same_seed_component),
            "require_from_semantic_false": bool(hue_require_from_semantic_false),
            "shuffle_each_epoch": bool(hue_shuffle_each_epoch),
            "n_hard_unlabeled_rows_train_base": int(hard_unl_mask_train.sum()),
            "n_unlabeled_rows_train_base": int(train_df["is_unlabeled"].sum()),
            "fraction_hard_among_train_unlabeled": float(
                hard_unl_mask_train.sum() / max(1, int(train_df["is_unlabeled"].sum()))
            ),
            "n_train_rows_base": int(len(train_df)),
            "n_train_rows_effective_if_enabled": int(
                len(train_df) + max(0.0, hue_oversample_factor - 1.0) * int(hard_unl_mask_train.sum())
            ),
        },
        "easy_positive_capping": {
            "enabled": bool(epc_enabled),
            "downsample_fraction": float(epc_downsample_fraction),
            "downsample_fraction_note": "Fraction of **easy** positive rows **retained** each epoch (stochastic).",
            "same_seed_component_only": bool(epc_same_seed_component_only),
            "min_semantic_cosine": epc_min_semantic_cosine,
            "min_source_count": epc_min_source_count,
            "or_rule_across_conditions": bool(epc_or_rule_across_conditions),
            "shuffle_each_epoch": bool(epc_shuffle_each_epoch),
            "n_train_positive_rows": int(train_df["is_positive"].sum()),
            "n_easy_positive_rows_in_rule": int(easy_pos_mask_train.sum()),
            "n_non_easy_positive_rows": int(
                (
                    _safe_bool_series(train_df, "is_positive")
                    & ~easy_pos_mask_train.reindex(train_df.index).fillna(False)
                ).sum()
            ),
            "hard_unlabeled_emphasis_also_enabled": bool(hue_enabled),
            "rule_has_no_subconditions": bool(
                not epc_same_seed_component_only
                and epc_min_semantic_cosine is None
                and epc_min_source_count is None
            ),
            "n_train_rows_after_easy_cap_estimate": int(
                _estimate_train_rows_after_easy_cap(
                    len(train_df),
                    int(easy_pos_mask_train.sum()),
                    epc_downsample_fraction,
                    epc_enabled,
                )
            ),
        },
        "row_emphasis_combined": {
            "n_train_rows_effective_combined_estimate": int(
                _estimate_train_rows_after_easy_cap(
                    len(train_df),
                    int(easy_pos_mask_train.sum()),
                    epc_downsample_fraction,
                    epc_enabled,
                )
                + max(0.0, hpe_oversample_factor - 1.0) * int(hard_pos_mask_train.sum())
                + max(0.0, hue_oversample_factor - 1.0) * int(hard_unl_mask_train.sum())
                + max(0.0, rne_oversample_factor - 1.0) * int(reliable_neg_mask_train.sum())
            ),
        },
        "sampling_diagnostics": sampling_diagnostics,
        "batch_coverage_stats": {
            "avg_unique_seed_emails_per_graph_batch": sampling_diagnostics.get(
                "avg_unique_seed_emails_per_graph_batch"
            ),
            "avg_sampled_hetero_total_nodes": sampling_diagnostics.get("avg_sampled_hetero_total_nodes"),
            "avg_sampled_hetero_total_edges": sampling_diagnostics.get("avg_sampled_hetero_total_edges"),
            "avg_n_email_nodes_in_sampled_batch": sampling_diagnostics.get("avg_n_email_nodes_in_sampled_batch"),
            "avg_recoverable_pair_fraction": sampling_diagnostics.get("avg_recoverable_pair_fraction"),
            "min_recoverable_pair_fraction": sampling_diagnostics.get("min_recoverable_pair_fraction"),
            "max_recoverable_pair_fraction": sampling_diagnostics.get("max_recoverable_pair_fraction"),
            "any_batch_catastrophic_endpoint_failure": sampling_diagnostics.get(
                "any_batch_catastrophic_endpoint_failure"
            ),
        },
        "pair_feature_columns_ordered": list(PAIR_FEATURE_COLUMNS),
        "pair_feature_dim_from_columns": feat_dim_columns,
        "pair_scorer_use_explicit_features": use_explicit_pair_feats,
        "pair_feature_dim_passed_to_scorer": pair_feat_dim_for_scorer,
        "pair_feature_missing_policy": "numeric NaN -> 0.0; bool unknown -> 0",
        "pair_feature_note": (
            "Explicit pair features exclude raw seed_component_i/j (identifier-like). "
            "Flags same_seed_component_flag / cross_seed_component_flag are retained. "
            "Shared-attribute overlap features (sender/stem/url/attachment/sender_domain/domain) "
            "are included as both counts and booleans."
        ),
        "loss_objective": {
            "pair_loss_type": pair_loss_type,
            "pu_class_prior": pi_p,
            "pu_non_negative": pu_non_negative,
            "pair_eval_threshold": pair_eval_threshold,
            "pair_placeholder_loss_mode_legacy": placeholder_mode,
            "nnpu_formulation": "L = pi_p * R_p^+ + max(0, R_u^- - pi_p * R_p^-) with logistic BCE terms.",
            "reliable_negative_loss_weight": float(reliable_negative_loss_weight),
            "nnpu_with_reliable_negatives_note": (
                "When pair_loss_type=nnpu_with_reliable_negatives: L = nnPU(P, U excluding N) + "
                "lambda_neg * mean BCE(logit, 0) over reliable-negative rows only."
            ),
        },
        "reliable_negative_training": {
            "reliable_negative_loss_weight": float(reliable_negative_loss_weight),
            "reliable_negative_emphasis": {
                "enabled": bool(rne_enabled),
                "oversample_factor": float(rne_oversample_factor),
                "shuffle_each_epoch": bool(rne_shuffle_each_epoch),
            },
            "note": (
                "Reliable-negative rows come from pair_training_dataset.csv (pair_status=reliable_negative). "
                "Rebuild that CSV with seed_candidate_workflow/scripts/build_pair_training_dataset.py --pipeline-json pointing "
                "to pipeline_config.json so pair_training.reliable_negative_pool is applied."
            ),
        },
        "placeholder_loss_note": (
            "Only used when pair_loss_type=placeholder_bce: mis-specified BCE (unlabeled as negatives)."
        ),
        "scorer_config": {
            "pair_scorer_type": scorer_type,
            "pair_scorer_hidden_dim": int(training_cfg.get("pair_scorer_hidden_dim", 256)),
            "pair_scorer_dropout": float(training_cfg.get("pair_scorer_dropout", 0.2)),
            "pair_scorer_use_explicit_features": use_explicit_pair_feats,
            "encoder_email_embed_dim": out_dim,
        },
        "scorer_parameter_count": scorer_param_count,
        "scorer_input_shapes": probe_shapes,
        "notes": [
            "Substep 3: neighborhoods sampled with torch_geometric.NeighborLoader on hetero data, "
            "seeded at unique email endpoints from the pair batch.",
            "Substep 4: EmailPairMLPScorer combines z_i, z_j, |z_i-z_j|, z_i*z_j, and optional metadata.",
            "PU: default pair_loss_type=nnpu (Kiryo-style non-negative risk); unlabeled rows are not negatives.",
            "Per-epoch train rows (when enabled): easy_positive_capping (retain fraction of easy positives) -> "
            "hard_positive oversample extras -> hard_unlabeled oversample extras -> "
            "reliable_negative oversample extras -> optional shuffle.",
            "Reliable-negative rows are never treated as unlabeled for nnPU; optional reliable_negative_emphasis "
            "duplicates N rows after hard-unlabeled extras.",
        ],
    }
    (run_dir / "pair_training_setup_summary.json").write_text(
        json.dumps(setup_summary, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    opt = torch.optim.AdamW(
        list(model.parameters()) + list(pair_scorer.parameters()),
        lr=lr,
        weight_decay=wd,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt,
        mode="min",
        factor=lr_reduce_factor,
        patience=lr_reduce_patience,
        min_lr=lr_reduce_min,
    )

    encoder_config = {
        "hidden": hidden,
        "out_dim": out_dim,
        "layers": layers,
        "dropout": dropout,
        "pair_scorer_type": scorer_type,
        "pair_feature_dim_passed_to_scorer": pair_feat_dim_for_scorer,
        "pair_scorer_hidden_dim": int(training_cfg.get("pair_scorer_hidden_dim", 256)),
        "pair_scorer_dropout": float(training_cfg.get("pair_scorer_dropout", 0.2)),
        "pair_scorer_use_explicit_features": use_explicit_pair_feats,
        "pair_loss_type": pair_loss_type,
        "pu_class_prior": pi_p,
        "pu_non_negative": pu_non_negative,
        "pair_eval_threshold": pair_eval_threshold,
        "reliable_negative_loss_weight": float(reliable_negative_loss_weight),
    }
    pair_training_config = {
        "pair_dataset_csv": str(csv_path),
        "pair_batch_size": pair_batch_size,
        "pair_fanout": fanout,
        "pair_max_unique_emails_per_graph_batch": max_unique,
        "pair_loss_type": pair_loss_type,
        "pu_class_prior": pi_p,
        "pu_non_negative": pu_non_negative,
        "pair_eval_threshold": pair_eval_threshold,
        "pair_placeholder_loss_mode": placeholder_mode,
        "pair_val_ratio": val_ratio,
        "pair_test_ratio": test_ratio,
        "pair_split_seed": split_seed,
        "pair_sampling_diag_max_batches": pair_sampling_diag_max_batches,
        "pair_assert_full_endpoint_coverage": pair_assert_full_endpoint_coverage,
        "pair_skip_graph_batch_if_any_endpoint_missing": pair_skip_graph_batch_if_any_endpoint_missing,
        "pair_log_mapping_misses": pair_log_mapping_misses,
        "pair_scorer_use_explicit_features": use_explicit_pair_feats,
        "pair_scorer_hidden_dim": int(training_cfg.get("pair_scorer_hidden_dim", 256)),
        "pair_scorer_dropout": float(training_cfg.get("pair_scorer_dropout", 0.2)),
        "pair_tqdm_batches": pair_tqdm,
        "hard_positive_emphasis": {
            "enabled": bool(hpe_enabled),
            "oversample_factor": float(hpe_oversample_factor),
            "cross_seed_component_only": bool(hpe_cross_seed_component_only),
            "require_from_2hop": bool(hpe_require_from_2hop),
            "max_source_count": hpe_max_source_count,
            "exclude_from_rare_artifact": bool(hpe_exclude_from_rare_artifact),
            "require_not_same_seed_component": bool(hpe_require_not_same_seed_component),
            "shuffle_each_epoch": bool(hpe_shuffle_each_epoch),
        },
        "hard_unlabeled_emphasis": {
            "enabled": bool(hue_enabled),
            "oversample_factor": float(hue_oversample_factor),
            "cross_seed_component_only": bool(hue_cross_seed_component_only),
            "require_from_2hop": bool(hue_require_from_2hop),
            "max_source_count": hue_max_source_count,
            "exclude_from_rare_artifact": bool(hue_exclude_from_rare_artifact),
            "require_not_same_seed_component": bool(hue_require_not_same_seed_component),
            "require_from_semantic_false": bool(hue_require_from_semantic_false),
            "shuffle_each_epoch": bool(hue_shuffle_each_epoch),
        },
        "easy_positive_capping": {
            "enabled": bool(epc_enabled),
            "downsample_fraction": float(epc_downsample_fraction),
            "same_seed_component_only": bool(epc_same_seed_component_only),
            "min_semantic_cosine": epc_min_semantic_cosine,
            "min_source_count": epc_min_source_count,
            "or_rule_across_conditions": bool(epc_or_rule_across_conditions),
            "shuffle_each_epoch": bool(epc_shuffle_each_epoch),
        },
        "reliable_negative_emphasis": {
            "enabled": bool(rne_enabled),
            "oversample_factor": float(rne_oversample_factor),
            "shuffle_each_epoch": bool(rne_shuffle_each_epoch),
        },
        "reliable_negative_loss_weight": float(reliable_negative_loss_weight),
    }
    training_params = {
        "lr": lr,
        "wd": wd,
        "target_epochs": epochs,
        "lr_reduce_patience": lr_reduce_patience,
        "lr_reduce_factor": lr_reduce_factor,
        "lr_reduce_min": lr_reduce_min,
    }
    with open(run_dir / training_config_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "training_objective": "pair_supervision",
                "torch_seed": TORCH_SEED,
                **encoder_config,
                **pair_training_config,
                "epochs": epochs,
                "early_stopping_patience": early_stopping_patience,
            },
            f,
            indent=2,
            default=str,
        )

    metrics_csv_path = os.path.join(run_dir, metrics_csv)
    with open(metrics_csv_path, mode="w", newline="") as f:
        csv.writer(f).writerow(PAIR_METRICS_HEADER)

    best_val = float("inf")
    patience_counter = 0
    best_state: dict[str, Any] | None = None

    print(f"[pair_supervision] Starting pair training | pair_loss_type={pair_loss_type} pi_p={pi_p}")
    print(f"[pair_supervision] train/val/test pairs: {len(train_df)}/{len(val_df)}/{len(test_df)}")
    print(
        f"[pair_supervision] reliable_negative (train/val/test): "
        f"{int(train_df['is_reliable_negative'].sum())}/"
        f"{int(val_df['is_reliable_negative'].sum())}/"
        f"{int(test_df['is_reliable_negative'].sum())} | "
        f"dataset_total_RN={load_stats.get('n_reliable_negative', 'n/a')} | "
        f"lambda_neg={reliable_negative_loss_weight}"
    )
    print(
        f"[pair_supervision] controls | hard_unlabeled_enabled={hue_enabled} | "
        f"reliable_negative_emphasis_enabled={rne_enabled} | "
        f"rn_oversample_factor={rne_oversample_factor:.3f}"
    )
    print(
        f"[pair_supervision] pair_metadata_dim={feat_dim_columns} "
        f"scorer_explicit_feats={use_explicit_pair_feats} "
        f"scorer_in_dim={pair_scorer.input_feature_dim} "
        f"scorer_params={scorer_param_count}"
    )
    print(f"[pair_supervision] probe forward: {probe_shapes}")
    print(
        f"[pair_supervision] sampling preflight: avg_recoverable_frac="
        f"{sampling_diagnostics.get('avg_recoverable_pair_fraction', 'n/a')} "
        f"catastrophic={sampling_diagnostics.get('any_batch_catastrophic_endpoint_failure', 'n/a')}"
    )
    if hpe_enabled:
        print(
            f"[pair_supervision] hard_positive_emphasis enabled | hard_pos={int(hard_pos_mask_train.sum())}/"
            f"{int(train_df['is_positive'].sum())} train positives | oversample_factor={hpe_oversample_factor:.3f} "
            f"| shuffle_each_epoch={hpe_shuffle_each_epoch}"
        )
    if hue_enabled:
        print(
            f"[pair_supervision] hard_unlabeled_emphasis enabled | hard_unl={int(hard_unl_mask_train.sum())}/"
            f"{int(train_df['is_unlabeled'].sum())} train unlabeled | oversample_factor={hue_oversample_factor:.3f} "
            f"| shuffle_each_epoch={hue_shuffle_each_epoch}"
        )
    if epc_enabled:
        n_pos_tr = int(train_df["is_positive"].sum())
        n_easy = int(easy_pos_mask_train.sum())
        print(
            f"[pair_supervision] easy_positive_capping enabled | easy_pos={n_easy}/{n_pos_tr} train positives "
            f"| retain_fraction={epc_downsample_fraction:.3f} | shuffle_each_epoch={epc_shuffle_each_epoch}"
        )
    if rne_enabled:
        print(
            f"[pair_supervision] reliable_negative_emphasis enabled | train_RN={int(reliable_neg_mask_train.sum())} "
            f"| oversample_factor={rne_oversample_factor:.3f} | shuffle_each_epoch={rne_shuffle_each_epoch}"
        )

    if pair_tqdm and n_train_batches is not None:
        tqdm_train_msg = f"train={n_train_batches}"
        if hpe_enabled or hue_enabled or epc_enabled or rne_enabled:
            tdf_probe, _ = _build_train_df_epoch_emphasis(
                train_df,
                easy_pos_mask=easy_pos_mask_train,
                epc_enabled=epc_enabled,
                epc_downsample_fraction=epc_downsample_fraction,
                hard_pos_mask=hard_pos_mask_train,
                hpe_enabled=hpe_enabled,
                hpe_oversample_factor=hpe_oversample_factor,
                hard_unl_mask=hard_unl_mask_train,
                hue_enabled=hue_enabled,
                hue_oversample_factor=hue_oversample_factor,
                reliable_neg_mask=reliable_neg_mask_train,
                rne_enabled=rne_enabled,
                rne_oversample_factor=rne_oversample_factor,
                shuffle_each_epoch=shuffle_train_epoch,
                epoch_seed=split_seed + 1,
            )
            n_train_batches_probe = count_pair_batches(tdf_probe, pair_batch_size, max_unique)
            tqdm_train_msg = f"train_base={n_train_batches} train_epoch={n_train_batches_probe}"
        print(
            f"[pair_supervision] tqdm: pair-batches per pass — "
            f"{tqdm_train_msg} val={n_val_batches} test={n_test_batches}"
        )

    for epoch in range(1, epochs + 1):
        train_df_epoch, emphasis_epoch_diag = _build_train_df_epoch_emphasis(
            train_df,
            easy_pos_mask=easy_pos_mask_train,
            epc_enabled=epc_enabled,
            epc_downsample_fraction=epc_downsample_fraction,
            hard_pos_mask=hard_pos_mask_train,
            hpe_enabled=hpe_enabled,
            hpe_oversample_factor=hpe_oversample_factor,
            hard_unl_mask=hard_unl_mask_train,
            hue_enabled=hue_enabled,
            hue_oversample_factor=hue_oversample_factor,
            reliable_neg_mask=reliable_neg_mask_train,
            rne_enabled=rne_enabled,
            rne_oversample_factor=rne_oversample_factor,
            shuffle_each_epoch=shuffle_train_epoch,
            epoch_seed=split_seed + epoch,
        )
        n_train_batches_epoch = (
            count_pair_batches(train_df_epoch, pair_batch_size, max_unique) if pair_tqdm else None
        )
        n_pos_epoch = int(_safe_bool_series(train_df_epoch, "is_positive", default=False).sum())
        n_unl_epoch = int(_safe_bool_series(train_df_epoch, "is_unlabeled", default=False).sum())
        n_rn_epoch = int(_safe_bool_series(train_df_epoch, "is_reliable_negative", default=False).sum())
        n_hard_pos_epoch = int(
            _hard_positive_mask(
                train_df_epoch,
                cross_seed_component_only=hpe_cross_seed_component_only,
                require_from_2hop=hpe_require_from_2hop,
                max_source_count=hpe_max_source_count,
                exclude_from_rare_artifact=hpe_exclude_from_rare_artifact,
                require_not_same_seed_component=hpe_require_not_same_seed_component,
            ).sum()
        )
        n_hard_unl_epoch = int(
            _hard_unlabeled_mask(
                train_df_epoch,
                cross_seed_component_only=hue_cross_seed_component_only,
                require_from_2hop=hue_require_from_2hop,
                max_source_count=hue_max_source_count,
                exclude_from_rare_artifact=hue_exclude_from_rare_artifact,
                require_not_same_seed_component=hue_require_not_same_seed_component,
                require_from_semantic_false=hue_require_from_semantic_false,
            ).sum()
        )
        tr_loss, tr_pu, tr_agg = train_pair_epoch(
            model=model,
            pair_scorer=pair_scorer,
            optimizer=opt,
            data_cpu=data_cpu,
            df=train_df_epoch,
            device=DEVICE,
            pair_batch_size=pair_batch_size,
            max_unique_emails=max_unique,
            fanout=fanout,
            pair_loss_type=pair_loss_type,
            pi_p=pi_p,
            pu_non_negative=pu_non_negative,
            reliable_negative_loss_weight=reliable_negative_loss_weight,
            pair_assert_full_endpoint_coverage=pair_assert_full_endpoint_coverage,
            pair_skip_graph_batch_if_any_endpoint_missing=pair_skip_graph_batch_if_any_endpoint_missing,
            pair_log_mapping_misses=pair_log_mapping_misses,
            pair_tqdm=pair_tqdm,
            tqdm_total_batches=n_train_batches_epoch,
            tqdm_desc=f"train {epoch}/{epochs}",
        )
        tr_agg["n_effective_train_rows_epoch"] = int(len(train_df_epoch))
        tr_agg["n_effective_positive_rows_epoch"] = int(n_pos_epoch)
        tr_agg["n_effective_reliable_negative_rows_epoch"] = int(n_rn_epoch)
        tr_agg["n_effective_hard_positive_rows_epoch"] = int(n_hard_pos_epoch)
        tr_agg["n_effective_non_hard_positive_rows_epoch"] = int(max(0, n_pos_epoch - n_hard_pos_epoch))
        tr_agg["n_effective_unlabeled_rows_epoch"] = int(n_unl_epoch)
        tr_agg["n_effective_hard_unlabeled_rows_epoch"] = int(n_hard_unl_epoch)
        tr_agg["n_effective_non_hard_unlabeled_rows_epoch"] = int(max(0, n_unl_epoch - n_hard_unl_epoch))
        tr_agg["hard_positive_emphasis_enabled"] = bool(hpe_enabled)
        tr_agg["hard_positive_oversample_factor"] = float(hpe_oversample_factor)
        tr_agg["hard_unlabeled_emphasis_enabled"] = bool(hue_enabled)
        tr_agg["hard_unlabeled_oversample_factor"] = float(hue_oversample_factor)
        tr_agg["row_emphasis_epoch_diag"] = emphasis_epoch_diag
        tr_agg["hard_positive_epoch_diag"] = emphasis_epoch_diag
        ecap = emphasis_epoch_diag.get("easy_positive_capping", {})
        tr_agg["easy_positive_capping_enabled"] = bool(epc_enabled)
        tr_agg["n_easy_positive_retained_epoch"] = int(ecap.get("n_easy_positives_retained_this_epoch", 0))
        tr_agg["n_easy_positive_dropped_epoch"] = int(ecap.get("n_easy_positives_dropped_this_epoch", 0))
        tr_agg["n_non_easy_positive_retained_epoch"] = int(ecap.get("n_non_easy_positive_rows_retained", 0))
        tr_agg["n_hard_unlabeled_oversample_extra_rows_epoch"] = int(
            ecap.get("n_hard_unlabeled_oversample_extra_rows", 0)
        )
        tr_agg["n_reliable_negative_oversample_extra_rows_epoch"] = int(
            ecap.get("n_reliable_negative_oversample_extra_rows", 0)
        )
        tr_agg["reliable_negative_emphasis_enabled"] = bool(rne_enabled)
        tr_agg["reliable_negative_oversample_factor"] = float(rne_oversample_factor)
        va_loss, va_pu, va_agg = eval_pair_epoch(
            model=model,
            pair_scorer=pair_scorer,
            data_cpu=data_cpu,
            df=val_df,
            device=DEVICE,
            pair_batch_size=pair_batch_size,
            max_unique_emails=max_unique,
            fanout=fanout,
            pair_loss_type=pair_loss_type,
            pi_p=pi_p,
            pu_non_negative=pu_non_negative,
            reliable_negative_loss_weight=reliable_negative_loss_weight,
            pair_eval_threshold=pair_eval_threshold,
            pair_assert_full_endpoint_coverage=pair_assert_full_endpoint_coverage,
            pair_skip_graph_batch_if_any_endpoint_missing=pair_skip_graph_batch_if_any_endpoint_missing,
            pair_tqdm=pair_tqdm,
            tqdm_total_batches=n_val_batches,
            tqdm_desc=f"val {epoch}/{epochs}",
        )
        sep_t = tr_pu.get("epoch_score_separation", float("nan"))
        sep_v = va_pu.get("epoch_score_separation", float("nan"))
        sep_ts = f"{sep_t:.4f}" if isinstance(sep_t, float) and sep_t == sep_t else "n/a"
        sep_vs = f"{sep_v:.4f}" if isinstance(sep_v, float) and sep_v == sep_v else "n/a"
        if pair_loss_type == PAIR_LOSS_PLACEHOLDER_BCE:
            ph_t = tr_pu.get("epoch_placeholder_acc", float("nan"))
            ph_v = va_pu.get("epoch_placeholder_acc", float("nan"))
            extra = (
                f"placeholder_acc tr/val "
                f"{(f'{ph_t:.4f}' if isinstance(ph_t, float) and ph_t == ph_t else 'n/a')}/"
                f"{(f'{ph_v:.4f}' if isinstance(ph_v, float) and ph_v == ph_v else 'n/a')}"
            )
        elif pair_loss_type == PAIR_LOSS_NNPU_WITH_RELIABLE_NEGATIVES:
            mn_t = tr_pu.get("epoch_mean_neg_supervised_bce", float("nan"))
            mn_v = va_pu.get("epoch_mean_neg_supervised_bce", float("nan"))
            mns = lambda x: f"{x:.4f}" if isinstance(x, float) and x == x else "n/a"
            extra = (
                f"nnPU+RN mean R_p+ tr/val {tr_pu.get('epoch_mean_r_p_pos', float('nan')):.4f}/"
                f"{va_pu.get('epoch_mean_r_p_pos', float('nan')):.4f} | "
                f"mean R_u- tr/val {tr_pu.get('epoch_mean_r_u_neg', float('nan')):.4f}/"
                f"{va_pu.get('epoch_mean_r_u_neg', float('nan')):.4f} | "
                f"mean_neg_BCE tr/val {mns(mn_t)}/{mns(mn_v)}"
            )
        else:
            extra = (
                f"nnPU mean R_p+ tr/val {tr_pu.get('epoch_mean_r_p_pos', float('nan')):.4f}/"
                f"{va_pu.get('epoch_mean_r_p_pos', float('nan')):.4f} | "
                f"mean R_u- tr/val {tr_pu.get('epoch_mean_r_u_neg', float('nan')):.4f}/"
                f"{va_pu.get('epoch_mean_r_u_neg', float('nan')):.4f}"
            )
        print(
            f"[pair_supervision] epoch {epoch:03d} | train loss {tr_loss:.4f} | val loss {va_loss:.4f} | "
            f"sep(P-U prob) tr {sep_ts} val {sep_vs} | {extra} | "
            f"uniq emails/batch {tr_agg['avg_unique_emails_per_batch']:.1f} | "
            f"train min recoverable pair frac {tr_agg.get('min_recoverable_pair_fraction', 0):.4f} | "
            f"epoch rows pos={n_pos_epoch} unl={n_unl_epoch} rn={n_rn_epoch} | "
            f"hard_pos/nonhard_pos {n_hard_pos_epoch}/{max(0, n_pos_epoch - n_hard_pos_epoch)} | "
            f"hard_unl/nonhard_unl {n_hard_unl_epoch}/{max(0, n_unl_epoch - n_hard_unl_epoch)} | "
            f"easy_pos kept/drop {tr_agg.get('n_easy_positive_retained_epoch', 0)}/"
            f"{tr_agg.get('n_easy_positive_dropped_epoch', 0)} | "
            f"non_easy_pos {tr_agg.get('n_non_easy_positive_retained_epoch', 0)} | "
            f"hard_unl_extra {tr_agg.get('n_hard_unlabeled_oversample_extra_rows_epoch', 0)} | "
            f"rn_extra {tr_agg.get('n_reliable_negative_oversample_extra_rows_epoch', 0)} | "
            f"total_rows {tr_agg.get('n_effective_train_rows_epoch', 0)}"
        )

        with open(metrics_csv_path, mode="a", newline="") as f:
            csv.writer(f).writerow(
                metrics_row_pair_training(
                    epoch,
                    pair_loss_type=pair_loss_type,
                    pi_p=pi_p,
                    train_loss=tr_loss,
                    val_loss=va_loss,
                    train_pu=tr_pu,
                    val_pu=va_pu,
                    train_agg=tr_agg,
                    val_agg=va_agg,
                )
            )

        if va_loss < best_val:
            best_val = va_loss
            patience_counter = 0
            best_state = {
                "model": model.state_dict(),
                "pair_scorer": pair_scorer.state_dict(),
            }
            save_pair_training_checkpoint(
                save_dir=ckpt_dir,
                filename=model_save_name,
                model=model,
                pair_scorer=pair_scorer,
                optimizer=opt,
                epoch=epoch,
                val_loss=va_loss,
                encoder_config=encoder_config,
                data_metadata=metadata,
                torch_seed=TORCH_SEED,
                pair_training_config=pair_training_config,
                patience_counter=patience_counter,
                best_val=best_val,
                best_model_state=best_state["model"],
                best_pair_scorer_state=best_state["pair_scorer"],
                training_params=training_params,
            )
            print(f"[pair_supervision] saved best checkpoint -> {ckpt_dir / model_save_name}")
        else:
            patience_counter += 1

        scheduler.step(va_loss)
        if patience_counter >= early_stopping_patience:
            print(f"[pair_supervision] early stopping at epoch {epoch}")
            break

    if best_state:
        model.load_state_dict(best_state["model"])
        pair_scorer.load_state_dict(best_state["pair_scorer"])
    te_loss, te_pu, te_agg = eval_pair_epoch(
        model=model,
        pair_scorer=pair_scorer,
        data_cpu=data_cpu,
        df=test_df,
        device=DEVICE,
        pair_batch_size=pair_batch_size,
        max_unique_emails=max_unique,
        fanout=fanout,
        pair_loss_type=pair_loss_type,
        pi_p=pi_p,
        pu_non_negative=pu_non_negative,
        reliable_negative_loss_weight=reliable_negative_loss_weight,
        pair_eval_threshold=pair_eval_threshold,
        pair_assert_full_endpoint_coverage=pair_assert_full_endpoint_coverage,
        pair_skip_graph_batch_if_any_endpoint_missing=pair_skip_graph_batch_if_any_endpoint_missing,
        pair_tqdm=pair_tqdm,
        tqdm_total_batches=n_test_batches,
        tqdm_desc="test",
    )
    def _fmt4(x: Any) -> str:
        if isinstance(x, float) and x == x:
            return f"{x:.4f}"
        return "n/a"

    print(
        f"[pair_supervision] [test] loss {te_loss:.4f} | mean pos prob {_fmt4(te_pu.get('epoch_mean_pos_prob', float('nan')))} "
        f"| mean unl prob {_fmt4(te_pu.get('epoch_mean_unl_prob', float('nan')))} | sep {_fmt4(te_pu.get('epoch_score_separation', float('nan')))} | "
        f"map_rate {te_agg['avg_pair_endpoint_map_rate']:.4f}"
    )

    setup_summary["final_test"] = {
        "test_loss": te_loss,
        "test_pu_metrics": te_pu,
        "test_sampling_diag": te_agg,
        "train_val_forward_ok": True,
    }
    (run_dir / "pair_training_setup_summary.json").write_text(
        json.dumps(setup_summary, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    return {
        "model": model,
        "pair_scorer": pair_scorer,
        "run_dir": str(run_dir),
        "best_checkpoint_path": str(ckpt_dir / model_save_name),
        "setup_summary_path": str(run_dir / "pair_training_setup_summary.json"),
    }
