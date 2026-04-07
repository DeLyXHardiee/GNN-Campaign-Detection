"""
Bucket-based ranking supervision for Method 1 V2 / **V2.1 precision** variant (HS-LI split).

Unsupervised buckets from views + V2 feature table (no GT). ``hard_neg`` is split into
``hard_neg_hsli`` (high-semantic / clearly low-infra false bridges + ≥1 risk cue) and
``hard_neg_other`` (remaining false-bridge negatives). ``hard_neg`` in masks is the union for
compatibility. ``strong_pos`` uses stricter multi-evidence safe-positive rules.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from analysis.utils.semantic_shard_edge_plausibility_v2_config import EdgePlausibilityV2Config
from analysis.utils.semantic_shard_edge_plausibility_v2_pairs import sample_ranking_pairs


def _col(
    features_df: pd.DataFrame,
    views_df: pd.DataFrame,
    name: str,
    *,
    default: float = 0.0,
) -> np.ndarray:
    if name in features_df.columns:
        return pd.to_numeric(features_df[name], errors="coerce").fillna(default).to_numpy(dtype=np.float64)
    if name in views_df.columns:
        return pd.to_numeric(views_df[name], errors="coerce").fillna(default).to_numpy(dtype=np.float64)
    return np.full(len(features_df), default, dtype=np.float64)


def _view_spread(sem: np.ndarray, inf: np.ndarray, tmp: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    m = np.stack([sem, inf, tmp], axis=1)
    mu = m.mean(axis=1)
    sd = m.std(axis=1)
    return sd / (np.abs(mu) + eps)


def _summ(x: np.ndarray) -> dict[str, float]:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"mean": float("nan"), "std": float("nan")}
    return {
        "mean": float(x.mean()),
        "std": float(x.std()),
        "p10": float(np.quantile(x, 0.1)),
        "p90": float(np.quantile(x, 0.9)),
    }


def summarize_high_sem_low_infra_regime(
    sem: np.ndarray,
    infv: np.ndarray,
    masks: dict[str, np.ndarray],
    thr_sem_high: float,
    thr_inf_false_max: float,
) -> dict[str, Any]:
    """
    High-semantic / low-infrastructure regime (oracle-motivated false-bridge danger zone).

    Uses the **same** infra cutoff as the false-bridge core bucket (``thr_inf_false_max``).
    """
    finite = np.isfinite(sem) & np.isfinite(infv)
    reg = finite & (sem >= thr_sem_high) & (infv <= thr_inf_false_max)
    assigned = masks["strong_neg"] | masks["hard_neg"] | masks["strong_pos"]
    return {
        "definition": "view_semantic >= thr_semantic_high AND view_infra <= thr_infra_false_bridge_max",
        "n_edges": int(reg.sum()),
        "n_overlap_false_bridge_bucket": int((reg & masks["hard_neg"]).sum()),
        "n_overlap_safe_pos_bucket": int((reg & masks["strong_pos"]).sum()),
        "n_overlap_strong_neg_bucket": int((reg & masks["strong_neg"]).sum()),
        "n_regime_unassigned": int((reg & ~assigned).sum()),
    }


def split_regime_plausibility_stats(
    sem: np.ndarray,
    infv: np.ndarray,
    scores: np.ndarray,
    thr_sem_high: float,
    thr_inf_false_max: float,
    idx: np.ndarray | None,
) -> dict[str, Any]:
    """Mean/median ``edge_plausibility`` on the HS-LI regime, optionally restricted to a split index set."""
    if idx is not None and len(idx) == 0:
        return {"n_edges": 0, "mean_edge_plausibility": float("nan"), "median_edge_plausibility": float("nan")}
    if idx is not None:
        ii = np.asarray(idx, dtype=np.int64)
        sem = sem[ii]
        infv = infv[ii]
        scores = scores[ii]
    finite = np.isfinite(sem) & np.isfinite(infv) & np.isfinite(scores)
    reg = finite & (sem >= thr_sem_high) & (infv <= thr_inf_false_max)
    if not reg.any():
        return {"n_edges": 0, "mean_edge_plausibility": float("nan"), "median_edge_plausibility": float("nan")}
    s = scores[reg]
    return {
        "n_edges": int(reg.sum()),
        "mean_edge_plausibility": float(np.mean(s)),
        "median_edge_plausibility": float(np.median(s)),
    }


def _bucket_counts_for_indices(masks: dict[str, np.ndarray], idx: np.ndarray) -> dict[str, int]:
    if idx.size == 0:
        return {
            "strong_pos": 0,
            "hard_neg_hsli": 0,
            "hard_neg_other": 0,
            "hard_neg": 0,
            "strong_neg": 0,
            "unassigned": 0,
            "n_edges_in_split": 0,
            "safe_pos": 0,
            "false_bridge_neg": 0,
        }
    i = np.asarray(idx, dtype=np.int64)
    sp = int(masks["strong_pos"][i].sum())
    hhsli = int(masks["hard_neg_hsli"][i].sum())
    hoth = int(masks["hard_neg_other"][i].sum())
    hn = int(masks["hard_neg"][i].sum())
    sn = int(masks["strong_neg"][i].sum())
    m = int(len(i))
    return {
        "strong_pos": sp,
        "hard_neg_hsli": hhsli,
        "hard_neg_other": hoth,
        "hard_neg": hn,
        "strong_neg": sn,
        "unassigned": int(m - sp - hn - sn),
        "n_edges_in_split": m,
        "safe_pos": sp,
        "false_bridge_neg": hn,
    }


def _positive_subpath_counts_for_indices(
    robust_core: np.ndarray,
    rarity_path: np.ndarray,
    hard_bridge_pos: np.ndarray,
    backup_teacher: np.ndarray,
    strong_pos: np.ndarray,
    idx: np.ndarray,
) -> dict[str, int]:
    if idx.size == 0:
        return {
            "robust_core_in_strong_pos": 0,
            "rarity_path_in_strong_pos": 0,
            "hard_bridge_in_strong_pos": 0,
            "backup_teacher_in_strong_pos": 0,
            "backup_teacher_predicate_n": 0,
            "backup_teacher_in_strong_pos_n": 0,
        }
    i = np.asarray(idx, dtype=np.int64)
    sp = strong_pos[i]
    rc = robust_core[i]
    rp = rarity_path[i]
    hb = hard_bridge_pos[i]
    bt = backup_teacher[i]
    return {
        "robust_core_in_strong_pos": int((rc & sp).sum()),
        "rarity_path_in_strong_pos": int((rp & sp).sum()),
        "hard_bridge_in_strong_pos": int((hb & sp).sum()),
        "backup_teacher_in_strong_pos": int((bt & sp).sum()),
        "backup_teacher_predicate_n": int(backup_teacher[i].sum()),
        "backup_teacher_in_strong_pos_n": int((bt & sp).sum()),
    }


def compute_ranking_bucket_masks(
    features_df: pd.DataFrame,
    views_df: pd.DataFrame,
    agreement: np.ndarray,
    hub_raw: np.ndarray,
    cfg: EdgePlausibilityV2Config,
    *,
    train_idx: np.ndarray | None = None,
    val_idx: np.ndarray | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """
    Boolean masks (n_edges,) — assignment order: **strong_neg**, **hard_neg_hsli**,
    **hard_neg_other**, **strong_pos**. ``hard_neg`` is ``hard_neg_hsli | hard_neg_other`` (union).

    ``hard_neg_hsli``: high semantic + **clearly** low infra (stricter quantile than the loose
    false-bridge band) and at least one structural / corroboration risk cue.

    ``strong_pos``: stricter safe-positive multi-evidence definition (raised infra floor, local
    support, hub cap, multi-view floor, tighter view-spread cap).
    """
    n = len(features_df)
    sem = _col(features_df, views_df, "view_semantic")
    infv = _col(features_df, views_df, "view_infra")
    tmpv = _col(features_df, views_df, "view_temporal")
    mv_min = np.minimum(np.minimum(sem, infv), tmpv)
    mv_mid = np.median(np.stack([sem, infv, tmpv], axis=1), axis=1)

    hub = np.asarray(hub_raw, dtype=np.float64).ravel()
    if hub.shape[0] != n:
        hub = np.zeros(n, dtype=np.float64)

    dom = _col(features_df, views_df, "v2_infra_dominance")

    loc_emb = _col(features_df, views_df, "v2_local_embeddedness_rank")
    loc_cn = _col(features_df, views_df, "v2_local_common_n_rank")
    loc_support = np.maximum(loc_emb, loc_cn)

    idf = _col(features_df, views_df, "shared_url_idf_sum")
    shared_url = _col(features_df, views_df, "shared_url_count")

    spread = _view_spread(sem, infv, tmpv)
    agree = np.asarray(agreement, dtype=np.float64).ravel()

    def q(x: np.ndarray, p: float) -> float:
        x = x[np.isfinite(x)]
        if x.size == 0:
            return 0.0
        return float(np.quantile(x, np.clip(p, 0.0, 1.0)))

    qh = float(cfg.bucket_q_semantic_high)
    ql = float(cfg.bucket_q_semantic_low)
    qm = float(cfg.bucket_q_semantic_mid)
    min_c = int(cfg.bucket_min_per_class)

    thr_inf_high = q(infv, float(cfg.bucket_q_view_infra_high))
    thr_hub_high = q(hub, float(cfg.bucket_q_hub_high))
    thr_hub_low = q(hub, float(cfg.bucket_q_hub_low_for_positive))
    thr_hub_mid = q(hub, (float(cfg.bucket_q_hub_high) + float(cfg.bucket_q_hub_low_for_positive)) / 2.0)
    thr_loc_high = q(loc_support, float(cfg.bucket_q_local_support_high))
    thr_loc_mid = q(loc_support, float(cfg.bucket_q_local_support_mid))
    thr_loc_low = q(loc_support, float(cfg.bucket_q_local_support_low))
    thr_spread_low = q(spread, float(cfg.bucket_q_view_spread_low))
    thr_spread_high = q(spread, float(getattr(cfg, "bucket_q_view_spread_high", 0.65)))
    thr_mv_low = q(mv_min, float(cfg.bucket_q_mv_min_low))
    thr_mv_floor = q(mv_min, float(cfg.bucket_q_mv_min_floor_pos))
    thr_idf_high = q(idf, float(cfg.bucket_q_idf_high))
    thr_idf_hard_b = q(idf, float(getattr(cfg, "bucket_q_idf_hard_bridge", 0.72)))
    thr_ag_backup = q(agree, float(cfg.bucket_q_agreement_backup_high))
    thr_shared_hi = q(shared_url, float(cfg.bucket_q_shared_url_high))
    thr_dom_high = q(dom, float(getattr(cfg, "bucket_q_infra_dominance_high", 0.72)))

    relax_scale = 1.0
    rounds_done = 0
    thr_inf_false_max = thr_inf_safe_min = thr_inf_mid_aux = 0.0
    idf_lo_gen = 0.0

    for rounds_done in range(int(cfg.bucket_relaxation_rounds) + 1):
        thr_sem_high = q(sem, min(qh * relax_scale, 0.95))
        thr_sem_low = q(sem, max(ql / max(relax_scale, 0.5), 0.05))
        thr_sem_mid = q(sem, qm)

        thr_inf_false_max = q(infv, float(getattr(cfg, "bucket_q_false_bridge_max_infra", 0.40)))
        thr_inf_hsli_core = q(
            infv, float(getattr(cfg, "bucket_q_hsli_core_infra_max", 0.32))
        )
        thr_inf_safe_min = q(infv, float(getattr(cfg, "bucket_q_safe_min_infra", 0.50)))
        thr_inf_mid_aux = q(infv, 0.46)
        idf_lo_gen = q(idf, 0.35)

        # --- Strong negatives: clearly weak / noisy / infra-spam (not the HS-LI false-bridge target)
        base_low = (
            (sem <= thr_sem_low)
            & (mv_min <= thr_mv_low)
            & ((loc_support <= thr_loc_low) | (hub >= thr_hub_high))
        )
        high_inf_weak_sem = (
            (infv >= thr_inf_high) & (sem <= thr_sem_mid) & (hub >= thr_hub_mid)
        )
        unstable_weak = (
            (spread >= thr_spread_high)
            & (mv_min <= thr_mv_low)
            & (sem <= thr_sem_mid)
        )
        strong_neg = base_low | high_inf_weak_sem | unstable_weak

        # --- False-bridge negatives (full union, before HS-LI split)
        false_hs_li = (sem >= thr_sem_high) & (infv <= thr_inf_false_max)
        false_hi_sem_weak_loc = (sem >= thr_sem_high) & (loc_support <= thr_loc_mid)
        false_hi_sem_hub = (sem >= thr_sem_high) & (hub >= thr_hub_mid)
        false_hi_sem_dom = (sem >= thr_sem_high) & (dom >= thr_dom_high)
        false_generic_url = (
            (sem >= thr_sem_mid)
            & (shared_url >= thr_shared_hi)
            & (idf <= idf_lo_gen)
        )
        hard_neg_full = (
            false_hs_li
            | false_hi_sem_weak_loc
            | false_hi_sem_hub
            | false_hi_sem_dom
            | false_generic_url
        ) & (~strong_neg)

        # HS-LI core: clearly low infra (stricter than false-bridge band) + ≥1 distrust cue
        risk_cue_hsli = (
            (loc_support <= thr_loc_mid)
            | (hub >= thr_hub_mid)
            | (dom >= thr_dom_high)
            | false_generic_url
            | ((mv_min <= thr_mv_low) & (sem >= thr_sem_high))
            | ((spread >= thr_spread_high) & (sem >= thr_sem_high))
        )
        hard_neg_hsli = (
            (sem >= thr_sem_high)
            & (infv <= thr_inf_hsli_core)
            & risk_cue_hsli
            & (~strong_neg)
        )
        hard_neg_other = hard_neg_full & (~hard_neg_hsli)
        hard_neg = hard_neg_hsli | hard_neg_other

        # --- Safe positives: multi-evidence; optional hard-bridge bucket (moderate sem + rare IDF + support)
        robust_core = (
            (sem >= thr_sem_high)
            & (infv >= thr_inf_safe_min)
            & (hub <= thr_hub_low)
            & (loc_support >= thr_loc_high)
            & (mv_min >= thr_mv_floor)
            & (spread <= thr_spread_low)
        )
        rarity_path = (
            (idf >= thr_idf_high)
            & (sem >= thr_sem_mid)
            & (infv >= thr_inf_mid_aux)
            & (hub <= thr_hub_mid)
            & (mv_mid >= thr_sem_low)
            & (loc_support >= thr_loc_mid)
        )
        hard_bridge_pos = (
            (sem >= thr_sem_mid)
            & (sem < thr_sem_high)
            & (idf >= thr_idf_hard_b)
            & (infv >= thr_inf_mid_aux)
            & (loc_support >= thr_loc_high)
            & (hub <= thr_hub_mid)
            & (mv_min >= thr_mv_floor)
        )
        backup_teacher = (
            (agree >= thr_ag_backup)
            & (infv >= thr_inf_safe_min)
            & (hub <= thr_hub_low)
            & (loc_support >= thr_loc_mid)
            & (mv_min >= thr_mv_floor)
        )
        if not bool(getattr(cfg, "bucket_use_backup_teacher", False)):
            backup_teacher = np.zeros(n, dtype=bool)

        strong_pos = (robust_core | rarity_path | hard_bridge_pos | backup_teacher) & (~strong_neg) & (~hard_neg)

        if (
            int(strong_neg.sum()) >= min_c
            and int(hard_neg.sum()) >= min_c
            and int(strong_pos.sum()) >= min_c
        ):
            break
        relax_scale *= float(cfg.bucket_relaxation_factor)

    masks = {
        "strong_neg": strong_neg,
        "hard_neg_hsli": hard_neg_hsli,
        "hard_neg_other": hard_neg_other,
        "hard_neg": hard_neg,
        "strong_pos": strong_pos,
    }

    meta: dict[str, Any] = {
        "mode": "buckets_v2_precision",
        "bucket_use_backup_teacher": bool(getattr(cfg, "bucket_use_backup_teacher", False)),
        "relax_rounds_used": int(rounds_done),
        "relax_scale_final": float(relax_scale),
        "n_edges": int(n),
        "precision_bucket_legend": {
            "strong_pos": "safe_positive_multi_evidence_strict",
            "hard_neg_hsli": "false_bridge_hsli_clear_low_infra_plus_risk_cue",
            "hard_neg_other": "false_bridge_non_hsli_or_loose_hsli",
            "hard_neg": "hard_neg_hsli_union_hard_neg_other",
            "strong_neg": "strong_negative_clearly_weak",
        },
        "ranking_pair_legend": {
            "pos_vs_hard_neg_hsli": "safe_positive_gt_hard_neg_hsli",
            "pos_vs_hard_neg_other": "safe_positive_gt_hard_neg_other",
            "pos_vs_strong_neg": "safe_positive_gt_strong_negative",
            "hard_neg_hsli_vs_strong_neg": "hard_neg_hsli_gt_strong_negative",
        },
        "counts": {
            "strong_neg": int(strong_neg.sum()),
            "hard_neg_hsli": int(hard_neg_hsli.sum()),
            "hard_neg_other": int(hard_neg_other.sum()),
            "hard_neg": int(hard_neg.sum()),
            "strong_pos": int(strong_pos.sum()),
            "unassigned": int(n - (strong_neg | hard_neg | strong_pos).sum()),
            "safe_pos": int(strong_pos.sum()),
            "false_bridge_neg": int(hard_neg.sum()),
            "hard_bridge_pos_predicate": int(hard_bridge_pos.sum()),
            "hard_bridge_in_safe_pos": int((hard_bridge_pos & strong_pos).sum()),
        },
        "thresholds": {
            "thr_semantic_high": float(thr_sem_high),
            "thr_semantic_low": float(thr_sem_low),
            "thr_semantic_mid": float(thr_sem_mid),
            "thr_view_infra_high": float(thr_inf_high),
            "thr_infra_false_bridge_max": float(thr_inf_false_max),
            "thr_infra_hsli_core_max": float(thr_inf_hsli_core),
            "thr_infra_safe_min": float(thr_inf_safe_min),
            "thr_hub_high": float(thr_hub_high),
            "thr_hub_low_pos": float(thr_hub_low),
            "thr_hub_mid": float(thr_hub_mid),
            "thr_local_support_high": float(thr_loc_high),
            "thr_local_support_mid": float(thr_loc_mid),
            "thr_local_support_low": float(thr_loc_low),
            "thr_view_spread_low": float(thr_spread_low),
            "thr_view_spread_high": float(thr_spread_high),
            "thr_mv_min_low": float(thr_mv_low),
            "thr_mv_min_floor_pos": float(thr_mv_floor),
            "thr_idf_high": float(thr_idf_high),
            "thr_idf_hard_bridge": float(thr_idf_hard_b),
            "thr_agreement_backup": float(thr_ag_backup),
            "thr_shared_url_high": float(thr_shared_hi),
            "thr_infra_dominance_high": float(thr_dom_high),
        },
        "feature_summaries": {
            "view_semantic": _summ(sem),
            "view_infra": _summ(infv),
            "view_temporal": _summ(tmpv),
            "v2_infra_dominance": _summ(hub),
            "local_support_max_rank": _summ(loc_support),
            "view_spread": _summ(spread),
            "agreement": _summ(agree),
        },
        "high_sem_low_infra_regime_static": summarize_high_sem_low_infra_regime(
            sem,
            infv,
            masks,
            float(thr_sem_high),
            float(thr_inf_false_max),
        ),
    }

    full_idx = np.arange(n, dtype=np.int64)
    train_arr = np.asarray(train_idx, dtype=np.int64) if train_idx is not None else None
    val_arr = np.asarray(val_idx, dtype=np.int64) if val_idx is not None else None
    meta["bucket_counts_by_split"] = {
        "full": _bucket_counts_for_indices(masks, full_idx),
        "train": _bucket_counts_for_indices(masks, train_arr) if train_arr is not None else None,
        "val": _bucket_counts_for_indices(masks, val_arr) if val_arr is not None else None,
    }
    meta["positive_subpath_counts_by_split"] = {
        "full": _positive_subpath_counts_for_indices(
            robust_core, rarity_path, hard_bridge_pos, backup_teacher, strong_pos, full_idx
        ),
        "train": (
            _positive_subpath_counts_for_indices(
                robust_core, rarity_path, hard_bridge_pos, backup_teacher, strong_pos, train_arr
            )
            if train_arr is not None
            else None
        ),
        "val": (
            _positive_subpath_counts_for_indices(
                robust_core, rarity_path, hard_bridge_pos, backup_teacher, strong_pos, val_arr
            )
            if val_arr is not None
            else None
        ),
    }
    return masks, meta


def pool_for_split(mask: np.ndarray, split_idx: np.ndarray) -> np.ndarray:
    """Row indices in ``split_idx`` where ``mask`` is True."""
    if split_idx.size == 0:
        return np.array([], dtype=np.int64)
    si = np.asarray(split_idx, dtype=np.int64)
    return si[mask[si]].astype(np.int64)


def pools_from_masks(
    masks: dict[str, np.ndarray],
    split_idx: np.ndarray,
) -> dict[str, np.ndarray]:
    keys = ("strong_pos", "strong_neg", "hard_neg_hsli", "hard_neg_other", "hard_neg")
    return {k: pool_for_split(masks[k], split_idx) for k in keys}


def sample_bucket_ranking_pairs(
    *,
    pools: dict[str, np.ndarray],
    rng: np.random.Generator,
    n_pairs: int,
    cfg: EdgePlausibilityV2Config,
) -> tuple[np.ndarray, np.ndarray, dict[str, int], np.ndarray]:
    ps = pools["strong_pos"]
    sn = pools["strong_neg"]
    hhsli = pools["hard_neg_hsli"]
    hoth = pools["hard_neg_other"]

    f_hsli = float(cfg.ranking_frac_pos_vs_hard_neg_hsli)
    f_oth = float(cfg.ranking_frac_pos_vs_hard_neg_other)
    f_psn = float(cfg.ranking_frac_pos_vs_strong_neg)
    f_hss = float(cfg.ranking_frac_hard_neg_hsli_vs_strong_neg)
    s = f_hsli + f_oth + f_psn + f_hss
    if s <= 0:
        f_hsli, f_oth, f_psn, f_hss = 0.55, 0.15, 0.12, 0.18
    else:
        f_hsli, f_oth, f_psn, f_hss = f_hsli / s, f_oth / s, f_psn / s, f_hss / s

    n_hsli = int(round(n_pairs * f_hsli))
    n_oth = int(round(n_pairs * f_oth))
    n_psn = int(round(n_pairs * f_psn))
    n_hss = max(0, n_pairs - n_hsli - n_oth - n_psn)

    idx_hi = np.zeros(n_pairs, dtype=np.int64)
    idx_lo = np.zeros(n_pairs, dtype=np.int64)
    margins = np.full(n_pairs, float(cfg.ranking_margin), dtype=np.float64)
    m_hsli = float(cfg.ranking_margin_hsli)
    m_std = float(cfg.ranking_margin)
    counts: dict[str, int] = {
        "pos_vs_hard_neg_hsli": 0,
        "pos_vs_hard_neg_other": 0,
        "pos_vs_strong_neg": 0,
        "hard_neg_hsli_vs_strong_neg": 0,
        "hard_neg_other_vs_strong_neg": 0,
    }

    def draw_pair(hi_pool: np.ndarray, lo_pool: np.ndarray) -> tuple[int, int] | None:
        if len(hi_pool) < 1 or len(lo_pool) < 1:
            return None
        for _ in range(60):
            ih = int(rng.choice(hi_pool))
            il = int(rng.choice(lo_pool))
            if ih != il:
                return ih, il
        return None

    t = 0
    for _ in range(n_hsli):
        if t >= n_pairs:
            break
        pr = draw_pair(ps, hhsli)
        if pr is None:
            break
        idx_hi[t], idx_lo[t] = pr
        margins[t] = m_hsli
        counts["pos_vs_hard_neg_hsli"] += 1
        t += 1
    for _ in range(n_oth):
        if t >= n_pairs:
            break
        pr = draw_pair(ps, hoth)
        if pr is None:
            break
        idx_hi[t], idx_lo[t] = pr
        margins[t] = m_std
        counts["pos_vs_hard_neg_other"] += 1
        t += 1
    for _ in range(n_psn):
        if t >= n_pairs:
            break
        pr = draw_pair(ps, sn)
        if pr is None:
            break
        idx_hi[t], idx_lo[t] = pr
        margins[t] = m_std
        counts["pos_vs_strong_neg"] += 1
        t += 1
    for _ in range(n_hss):
        if t >= n_pairs:
            break
        pr = draw_pair(hhsli, sn)
        if pr is None:
            break
        idx_hi[t], idx_lo[t] = pr
        margins[t] = m_std
        counts["hard_neg_hsli_vs_strong_neg"] += 1
        t += 1

    # Top up with preferred pairs if budget left
    while t < n_pairs:
        pr = draw_pair(ps, hhsli)
        tag = "pos_vs_hard_neg_hsli"
        margin = m_hsli
        if pr is None:
            pr = draw_pair(ps, hoth)
            tag = "pos_vs_hard_neg_other"
            margin = m_std
        if pr is None:
            pr = draw_pair(ps, sn)
            tag = "pos_vs_strong_neg"
            margin = m_std
        if pr is None:
            pr = draw_pair(hhsli, sn)
            tag = "hard_neg_hsli_vs_strong_neg"
            margin = m_std
        if pr is None:
            pr = draw_pair(hoth, sn)
            tag = "hard_neg_other_vs_strong_neg"
            margin = m_std
        if pr is None:
            break
        idx_hi[t], idx_lo[t] = pr
        margins[t] = margin
        counts[tag] = counts.get(tag, 0) + 1
        t += 1

    return idx_hi, idx_lo, counts, margins


def bucket_pools_usable(pools: dict[str, np.ndarray]) -> bool:
    ps = pools["strong_pos"]
    sn = pools["strong_neg"]
    hhsli = pools["hard_neg_hsli"]
    hoth = pools["hard_neg_other"]
    hn = pools["hard_neg"]
    if len(ps) >= 1 and len(hhsli) >= 1:
        return True
    if len(ps) >= 1 and len(hoth) >= 1:
        return True
    if len(ps) >= 1 and len(sn) >= 1:
        return True
    if len(hhsli) >= 1 and len(sn) >= 1:
        return True
    if len(hoth) >= 1 and len(sn) >= 1:
        return True
    if len(ps) >= 1 and len(hn) >= 1:
        return True
    if len(hn) >= 1 and len(sn) >= 1:
        return True
    return False


def sample_ranking_pairs_hybrid(
    *,
    edges_reset: pd.DataFrame,
    agreement: np.ndarray,
    pools: dict[str, np.ndarray],
    rng: np.random.Generator,
    cfg: EdgePlausibilityV2Config,
    index_pool: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], np.ndarray]:
    """
    Bucket pairs when pools are usable; otherwise fall back to legacy teacher-stratified pairs.

    Returns ``(idx_hi, idx_lo, meta, margins)`` where ``margins[i]`` is the ranking margin for pair ``i``.
    """
    n_p = int(cfg.n_ranking_pairs_per_batch)
    m_def = float(cfg.ranking_margin)

    if getattr(cfg, "ranking_supervision_mode", "buckets") != "buckets":
        ih, il = sample_ranking_pairs(
            edges_reset,
            agreement,
            rng,
            n_pairs=n_p,
            n_quantile_bins=cfg.quantile_bins,
            fraction_endpoint_hard=cfg.fraction_endpoint_hard_pairs,
            index_pool=index_pool,
        )
        return ih, il, {"mode": "legacy_teacher", "n_legacy": n_p}, np.full(n_p, m_def, dtype=np.float64)

    if not bucket_pools_usable(pools):
        ih, il = sample_ranking_pairs(
            edges_reset,
            agreement,
            rng,
            n_pairs=n_p,
            n_quantile_bins=cfg.quantile_bins,
            fraction_endpoint_hard=cfg.fraction_endpoint_hard_pairs,
            index_pool=index_pool,
        )
        return ih, il, {"mode": "fallback_teacher", "n_fallback": n_p}, np.full(n_p, m_def, dtype=np.float64)

    ih, il, cts, margins = sample_bucket_ranking_pairs(pools=pools, rng=rng, n_pairs=n_p, cfg=cfg)
    ps = pools["strong_pos"]
    sn = pools["strong_neg"]
    hhsli = pools["hard_neg_hsli"]
    hoth = pools["hard_neg_other"]
    hn = pools["hard_neg"]
    m_hsli = float(cfg.ranking_margin_hsli)
    bad = idx_hi_lo_invalid(ih, il)
    if bad.any():
        for t in np.flatnonzero(bad):
            for _ in range(40):
                ok = None
                mar = m_def
                if len(ps) >= 1 and len(hhsli) >= 1:
                    a, b = int(rng.choice(ps)), int(rng.choice(hhsli))
                    if a != b:
                        ok, mar = (a, b), m_hsli
                if ok is None and len(ps) >= 1 and len(hoth) >= 1:
                    a, b = int(rng.choice(ps)), int(rng.choice(hoth))
                    if a != b:
                        ok = (a, b)
                if ok is None and len(ps) >= 1 and len(sn) >= 1:
                    a, b = int(rng.choice(ps)), int(rng.choice(sn))
                    if a != b:
                        ok = (a, b)
                if ok is None and len(hhsli) >= 1 and len(sn) >= 1:
                    a, b = int(rng.choice(hhsli)), int(rng.choice(sn))
                    if a != b:
                        ok = (a, b)
                if ok is None and len(hoth) >= 1 and len(sn) >= 1:
                    a, b = int(rng.choice(hoth)), int(rng.choice(sn))
                    if a != b:
                        ok = (a, b)
                if ok is None and len(ps) >= 1 and len(hn) >= 1:
                    a, b = int(rng.choice(ps)), int(rng.choice(hn))
                    if a != b:
                        ok = (a, b)
                if ok is None and len(hn) >= 1 and len(sn) >= 1:
                    a, b = int(rng.choice(hn)), int(rng.choice(sn))
                    if a != b:
                        ok = (a, b)
                if ok is not None:
                    ih[t], il[t] = ok
                    margins[t] = mar
                    break
    if idx_hi_lo_invalid(ih, il).any():
        ih, il = sample_ranking_pairs(
            edges_reset,
            agreement,
            rng,
            n_pairs=n_p,
            n_quantile_bins=cfg.quantile_bins,
            fraction_endpoint_hard=cfg.fraction_endpoint_hard_pairs,
            index_pool=index_pool,
        )
        return (
            ih,
            il,
            {"mode": "fallback_teacher_partial_bucket", "n_fallback": n_p},
            np.full(n_p, m_def, dtype=np.float64),
        )
    out: dict[str, Any] = {"mode": "buckets", **cts}
    return ih, il, out, margins


def idx_hi_lo_invalid(ih: np.ndarray, il: np.ndarray) -> np.ndarray:
    return (ih == il) | (ih < 0) | (il < 0)
