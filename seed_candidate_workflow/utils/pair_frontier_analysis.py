"""
Full unlabeled-pair frontier analysis (low / mid / high bands) for pair score separation.

Analysis-only: compares GT-covered unlabeled cohorts and surfaces body/path/support,
provenance, 2-hop channels, and joint separators that explain score regimes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from seed_candidate_workflow.utils.pair_low_band_twohop_channel import (
    TWOHOP_CHANNELS,
    attach_twohop_channel_columns,
    extend_bool_terms_for_low_band_channels,
    low_band_twohop_joint_rule_names,
)
from seed_candidate_workflow.utils.pair_mid_band_frontier import (
    MidBandThresholds,
    _build_same_vs_cross_marginal,
    _build_two_cohort_marginal,
    _community_cut_zone_analysis,
    _export_mid_band_review_html,
    score_band_masks,
)
from seed_candidate_workflow.utils.pair_score_separation_output_layout import (
    ExportFlags,
    rel_to_root,
)
from seed_candidate_workflow.utils.pair_same_unlabeled_rescued_collapsed import (
    BODY_COMPARISON_COLS,
    EXTRA_PROVENANCE_COLS,
    PATH_FEATURE_COLS,
    _backfill_body_features_from_text,
    _build_body_vs_body_only_comparison,
    _cohort_feature_stats,
    _enrich_path_features_from_nodes,
    _merge_pair_features_from_eval,
)
from seed_candidate_workflow.utils.scorer_diagnostics_rules import (
    BINARY_CONDITION_RULES_DEFAULT,
    FEATURE_KEYS_DEFAULT,
    PROVENANCE_KEYS_DEFAULT,
    SHARED_EVIDENCE_KEYS_DEFAULT,
)

_PSE: Any = None


def _pse() -> Any:
    global _PSE
    if _PSE is None:
        from seed_candidate_workflow.utils import pair_score_separation as mod

        _PSE = mod
    return _PSE


FrontierThresholds = MidBandThresholds

COHORT_NAMES: tuple[str, ...] = (
    "low_same_unlabeled",
    "mid_same_unlabeled",
    "high_same_unlabeled",
    "low_cross_unlabeled",
    "mid_cross_unlabeled",
    "high_cross_unlabeled",
)

SIMILARITY_FEATURE_KEYS: tuple[str, ...] = (
    "semantic_cosine_max",
    "body_token_jaccard",
    "body_char4gram_jaccard",
    "body_only_token_jaccard",
    "body_only_char4gram_jaccard",
    "path_token_jaccard_combined",
    "url_path_token_jaccard",
    "stem_path_token_jaccard",
    "sender_localpart_norm_jaccard",
)

SHARED_BOOLEAN_KEYS: tuple[str, ...] = (
    "has_shared_sender",
    "has_shared_stem",
    "has_shared_url",
    "has_shared_attachment",
    "has_shared_sender_domain",
    "has_shared_domain",
    "has_shared_html_fp",
)

COUNT_SUPPORT_KEYS: tuple[str, ...] = (
    "source_count",
    "n_shared_core_channels",
    "shared_sender_count",
    "shared_stem_count",
    "shared_url_count",
    "shared_attachment_count",
    "shared_sender_domain_count",
    "shared_domain_count",
    "time_gap_seconds_min",
)

PROVENANCE_KEYS: tuple[str, ...] = (
    "from_seed",
    *PROVENANCE_KEYS_DEFAULT,
    *EXTRA_PROVENANCE_COLS,
)

CANDIDATE_FAMILY_PROVENANCE_KEYS: tuple[str, ...] = (
    "from_body_token_jaccard_highconf",
    "from_body_char4gram_jaccard_highconf",
    "from_semantic_mid_senderlocalpart_support",
    "from_semantic_mid_sender_support",
    "from_semantic_mid_stem_support",
    "from_semantic_mid_core_support",
    "from_shared_stem_highconf",
)

FRONTIER_MARGINAL_FEATURE_KEYS: tuple[str, ...] = tuple(
    dict.fromkeys([*SIMILARITY_FEATURE_KEYS, *COUNT_SUPPORT_KEYS, *FEATURE_KEYS_DEFAULT])
)

BODY_PATH_COLS: tuple[str, ...] = tuple(
    dict.fromkeys([*BODY_COMPARISON_COLS, *PATH_FEATURE_COLS])
)

FRONTIER_EXTRA_JOINT_RULES: tuple[str, ...] = (
    "semantic_ge_0_90",
    "semantic_ge_0_90_AND_n_shared_core_channels_ge_1",
    "semantic_ge_0_90_AND_shared_sender",
    "semantic_ge_0_90_AND_shared_html_fp",
    "from_2hop_AND_shared_html_fp",
    "from_2hop_AND_source_count_eq_1",
    "from_2hop_AND_NOT_from_semantic",
    "source_count_eq_1_AND_shared_html_fp",
    "shared_html_fp_AND_NOT_shared_sender",
    "semantic_ge_0_90_AND_shared_stem",
    "n_shared_core_channels_ge_1_AND_shared_sender",
    "same_seed_component_flag_AND_path_token_jaccard_combined_ge_0_25",
    "body_only_token_jaccard_ge_0_25_AND_path_token_jaccard_combined_ge_0_25",
    "body_only_token_jaccard_ge_0_25_AND_NOT_from_semantic",
    "path_token_jaccard_combined_ge_0_25",
    "url_path_token_jaccard_ge_0_25",
    "body_only_char4gram_jaccard_ge_0_25",
    "semantic_ge_0_90_AND_body_only_token_jaccard_ge_0_25",
    "source_count_eq_1_AND_twohop_via_html_fp",
)


def _as_float_array(scores: np.ndarray | pd.Series) -> np.ndarray:
    return np.asarray(pd.to_numeric(np.asarray(scores), errors="coerce"), dtype=np.float64)


def frontier_unlabeled_cohort_masks(
    *,
    same_eval: np.ndarray,
    cross_eval: np.ndarray,
    unl_eval: np.ndarray,
    scores: np.ndarray,
    thresholds: FrontierThresholds,
) -> dict[str, np.ndarray]:
    """Six GT-covered unlabeled cohort masks on the eval-aligned score array."""
    bands = score_band_masks(scores, thresholds=thresholds)
    low, mid, high = bands["low"], bands["mid"], bands["high"]
    return {
        "low_same_unlabeled": same_eval & unl_eval & low,
        "mid_same_unlabeled": same_eval & unl_eval & mid,
        "high_same_unlabeled": same_eval & unl_eval & high,
        "low_cross_unlabeled": cross_eval & unl_eval & low,
        "mid_cross_unlabeled": cross_eval & unl_eval & mid,
        "high_cross_unlabeled": cross_eval & unl_eval & high,
    }


def _classify_frontier_review_regime(row: pd.Series) -> str:
    return str(row.get("frontier_cohort") or row.get("cohort") or "frontier_unknown")


def _frontier_review_prompt(row: pd.Series) -> str:
    cohort = _classify_frontier_review_regime(row)
    prompts = {
        "low_same_unlabeled": (
            "Low-band same-campaign unlabeled: collapsed score. "
            "What body/path/support is missing vs mid/high?"
        ),
        "mid_same_unlabeled": (
            "Mid-band same-campaign unlabeled: uncertain. "
            "What would promote this toward high/rescued?"
        ),
        "high_same_unlabeled": (
            "High-band same-campaign unlabeled: rescued-like score. "
            "What evidence defines a good unlabeled edge?"
        ),
        "mid_cross_unlabeled": (
            "Mid-band cross-campaign unlabeled: should stay low at community cut. "
            "What distinguishes this from mid same?"
        ),
        "low_cross_unlabeled": "Low-band cross-campaign unlabeled.",
        "high_cross_unlabeled": "High-band cross-campaign unlabeled (dangerous FP risk).",
    }
    return prompts.get(cohort, "Frontier unlabeled pair — inspect evidence.")


def _build_cohort_inspection_df(
    *,
    df_eval: pd.DataFrame,
    row_mask: np.ndarray,
    gt_path: Path,
    label_map: dict[str, Any],
    gt_relation: str,
    cohort: str,
    nodes_by_email: dict[str, dict[str, set[str]]],
    evidence_index: dict[tuple[str, str], list[dict[str, Any]]] | None,
) -> pd.DataFrame:
    pse = _pse()
    df = pse._build_high_band_inspection_dataframe(
        df_eval=df_eval,
        row_mask=row_mask,
        gt_path=gt_path,
        label_map=label_map,
        gt_relation=gt_relation,
        nodes_by_email=nodes_by_email,
        cohort=cohort,
    )
    if df.empty:
        return df
    eval_sub = df_eval.loc[row_mask].copy()
    df = _merge_pair_features_from_eval(df, eval_sub)
    df = _enrich_path_features_from_nodes(df, nodes_by_email)
    df = _backfill_body_features_from_text(df)
    df["frontier_cohort"] = cohort
    if evidence_index:
        from seed_candidate_workflow.utils.pair_inspection_admitting_evidence import (
            enrich_inspection_with_admitting_evidence,
        )

        df = enrich_inspection_with_admitting_evidence(df, evidence_index=evidence_index)
    df = attach_twohop_channel_columns(df, evidence_index=evidence_index)
    return df


def _feature_population_diagnostics(
    cohort_dfs: dict[str, pd.DataFrame],
    *,
    df_eval: pd.DataFrame,
) -> dict[str, Any]:
    expected_numeric = list(FRONTIER_MARGINAL_FEATURE_KEYS)
    expected_bool_prov = list(PROVENANCE_KEYS)
    expected_shared = list(SHARED_BOOLEAN_KEYS) + list(SHARED_EVIDENCE_KEYS_DEFAULT)
    expected_twohop = [f"twohop_via_{ch}" for ch in TWOHOP_CHANNELS]
    expected_candidate = list(CANDIDATE_FAMILY_PROVENANCE_KEYS)

    eval_cols = set(df_eval.columns)
    candidate_in_eval = [c for c in expected_candidate if c in eval_cols]
    candidate_missing = [c for c in expected_candidate if c not in eval_cols]

    by_cohort: dict[str, Any] = {}
    for name, df in cohort_dfs.items():
        n = int(len(df))
        entry: dict[str, Any] = {"n_pairs": n, "columns_present": sorted(df.columns.tolist())}
        for group_name, keys in (
            ("similarity_features", expected_numeric[: len(SIMILARITY_FEATURE_KEYS)]),
            ("count_support_features", COUNT_SUPPORT_KEYS),
            ("provenance_flags", expected_bool_prov),
            ("shared_artifact_flags", expected_shared),
            ("twohop_channels", expected_twohop),
        ):
            stats: dict[str, Any] = {}
            for k in keys:
                if k not in df.columns:
                    stats[k] = {"present": False, "n_non_null": 0}
                elif n == 0:
                    stats[k] = {"present": True, "n_non_null": 0}
                else:
                    col = df[k]
                    if col.dtype == bool or col.dtype == object:
                        nn = int(col.fillna(False).astype(bool).sum())
                    else:
                        nn = int(pd.to_numeric(col, errors="coerce").notna().sum())
                    stats[k] = {"present": True, "n_non_null": nn, "fraction_non_null": float(nn / n)}
            entry[group_name] = stats
        by_cohort[name] = entry

    return {
        "expected_similarity_features": list(SIMILARITY_FEATURE_KEYS),
        "expected_shared_boolean_columns": list(SHARED_BOOLEAN_KEYS),
        "candidate_family_provenance_in_pair_table": candidate_in_eval,
        "candidate_family_provenance_unavailable": candidate_missing,
        "note_on_candidate_families": (
            "Body-Jaccard / sender-localpart candidate provenance appears only when "
            "those columns exist on the pair training/eval dataframe."
            if candidate_missing
            else "All listed candidate-family provenance columns found on eval frame."
        ),
        "by_cohort": by_cohort,
    }


def _build_body_path_signal_comparison(
    *,
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    comparison: str,
    left_label: str,
    right_label: str,
) -> dict[str, Any]:
    left_stats = _cohort_feature_stats(
        left_df, np.ones(len(left_df), dtype=bool) if len(left_df) else np.array([], dtype=bool), BODY_PATH_COLS
    )
    right_stats = _cohort_feature_stats(
        right_df,
        np.ones(len(right_df), dtype=bool) if len(right_df) else np.array([], dtype=bool),
        BODY_PATH_COLS,
    )
    threshold_rates: dict[str, Any] = {}
    for col in BODY_PATH_COLS:
        thr = 0.25
        for label, sub in ((left_label, left_df), (right_label, right_df)):
            if col not in sub.columns or sub.empty:
                threshold_rates[f"{label}_{col}_ge_{thr}"] = None
            else:
                s = pd.to_numeric(sub[col], errors="coerce")
                threshold_rates[f"{label}_{col}_ge_{thr}"] = float((s >= thr).mean())

    body_vs_body_only = _build_body_vs_body_only_comparison(
        rescued_df=right_df if "high" in right_label or "rescued" in right_label else left_df,
        collapsed_df=left_df if "low" in left_label else right_df,
    )
    notes: list[str] = []
    for col in ("body_only_token_jaccard", "path_token_jaccard_combined"):
        lm = (left_stats.get(col) or {}).get("mean")
        rm = (right_stats.get(col) or {}).get("mean")
        if lm is not None and rm is not None:
            if float(rm) > float(lm) + 0.05:
                notes.append(f"{right_label} higher mean {col} (+{float(rm) - float(lm):.3f}) vs {left_label}.")
            elif float(lm) > float(rm) + 0.05:
                notes.append(f"{left_label} higher mean {col} vs {right_label}.")
    if not notes:
        notes.append("Compare raw body_* vs body_only_* and path_* means across cohorts.")

    return {
        "comparison": comparison,
        "left_cohort": left_label,
        "right_cohort": right_label,
        "feature_means_left": left_stats,
        "feature_means_right": right_stats,
        "threshold_rate_ge_0_25": threshold_rates,
        "body_vs_body_only_notes": body_vs_body_only.get("interpretation_notes", []),
        "interpretation_notes": notes[:10],
    }


def _extend_marginal_with_full_features(
    marginal: dict[str, Any],
    *,
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    comparison: str,
    left_label: str,
    right_label: str,
    gt_path: Path,
    nodes_by_email: dict[str, dict[str, set[str]]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Augment two-cohort marginal with shared flags, twohop, and dataset columns."""
    lcol = f"{left_label}_value"
    rcol = f"{right_label}_value"
    extra_rows: list[dict[str, Any]] = []

    def _frac(df: pd.DataFrame, col: str) -> float | None:
        if col not in df.columns or df.empty:
            return None
        if df[col].dtype == bool:
            return float(df[col].fillna(False).astype(bool).mean())
        s = pd.to_numeric(df[col], errors="coerce")
        if s.notna().any() and s.max() <= 1.0:
            return float(s.fillna(0).astype(bool).mean())
        return float((s > 0).mean()) if s.notna().any() else None

    for k in SHARED_BOOLEAN_KEYS:
        lv, rv = _frac(left_df, k), _frac(right_df, k)
        diff = (lv - rv) if lv is not None and rv is not None else None
        extra_rows.append(
            {
                "gt_path": str(gt_path.resolve()),
                "comparison": comparison,
                "metric_group": "shared_artifact_boolean",
                "metric_name": k,
                lcol: lv,
                rcol: rv,
                "difference_left_minus_right": diff,
                "abs_difference": abs(diff) if diff is not None else None,
            }
        )

    for k in [f"twohop_via_{ch}" for ch in TWOHOP_CHANNELS]:
        lv, rv = _frac(left_df, k), _frac(right_df, k)
        diff = (lv - rv) if lv is not None and rv is not None else None
        extra_rows.append(
            {
                "gt_path": str(gt_path.resolve()),
                "comparison": comparison,
                "metric_group": "twohop_channel",
                "metric_name": k,
                lcol: lv,
                rcol: rv,
                "difference_left_minus_right": diff,
                "abs_difference": abs(diff) if diff is not None else None,
            }
        )

    for k in CANDIDATE_FAMILY_PROVENANCE_KEYS:
        if k not in left_df.columns and k not in right_df.columns:
            continue
        lv, rv = _frac(left_df, k), _frac(right_df, k)
        diff = (lv - rv) if lv is not None and rv is not None else None
        extra_rows.append(
            {
                "gt_path": str(gt_path.resolve()),
                "comparison": comparison,
                "metric_group": "candidate_family_provenance",
                "metric_name": k,
                lcol: lv,
                rcol: rv,
                "difference_left_minus_right": diff,
                "abs_difference": abs(diff) if diff is not None else None,
            }
        )

    ranked = list(marginal.get("ranked_separators_top15") or []) + extra_rows
    ranked = [r for r in ranked if r.get("abs_difference") is not None]
    ranked.sort(key=lambda r: float(r["abs_difference"]), reverse=True)
    marginal = dict(marginal)
    marginal["ranked_separators_top15"] = ranked[:15]
    marginal["ranked_separators_top25_extended"] = ranked[:25]
    all_rows = extra_rows
    return marginal, all_rows


def _build_frontier_joint_two_cohort(
    *,
    gt_path: Path,
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    left_mask_eval: np.ndarray,
    right_mask_eval: np.ndarray,
    df_eval: pd.DataFrame,
    comparison: str,
    band_kind: str,
    value_key_left: str,
    value_key_right: str,
    nodes_by_email: dict[str, dict[str, set[str]]],
    evidence_index: dict[tuple[str, str], list[dict[str, Any]]] | None,
    marginal_sep: dict[str, Any] | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Joint separator analysis between two cohort masks on df_eval."""
    extra = tuple(FRONTIER_EXTRA_JOINT_RULES) + low_band_twohop_joint_rule_names()
    joint, rows = _pse()._build_band_joint_separator_for_gt(
        gt_path=gt_path,
        df_eval=df_eval,
        same_band_mask_eval=left_mask_eval,
        cross_band_mask_eval=right_mask_eval,
        band_kind=band_kind,
        band_thresholds={"comparison": comparison},
        nodes_by_email=nodes_by_email,
        value_key_same=value_key_left,
        value_key_cross=value_key_right,
        focus=comparison,
        extra_joint_rules=extra,
        marginal_sep=marginal_sep,
        include_recommendations=False,
        twohop_channel_analysis=True,
        evidence_index=evidence_index,
    )
    joint["comparison"] = comparison
    return joint, rows


def _cohort_summary_block(
    df: pd.DataFrame,
    *,
    cohort: str,
    nodes_by_email: dict[str, dict[str, set[str]]],
    n_total_eval: int,
) -> dict[str, Any]:
    block = _pse()._summarize_group(gdf=df, n_total_eval=n_total_eval, nodes_by_email=nodes_by_email)
    block["cohort"] = cohort
    block["n_pairs"] = int(len(df))
    if not df.empty:
        block["score_mean"] = float(pd.to_numeric(df["score"], errors="coerce").mean())
        block["score_median"] = float(pd.to_numeric(df["score"], errors="coerce").median())
    return block


def _build_edge_profile(
    *,
    cohort_name: str,
    df: pd.DataFrame,
    nodes_by_email: dict[str, dict[str, set[str]]],
) -> dict[str, Any]:
    if df.empty:
        return {"cohort": cohort_name, "n_pairs": 0, "summary": "no pairs in cohort"}
    n = int(len(df))
    summ = _pse()._summarize_group(gdf=df, n_total_eval=n, nodes_by_email=nodes_by_email)
    top_feats: list[str] = []
    for k in SIMILARITY_FEATURE_KEYS:
        fs = (summ.get("feature_summaries") or {}).get(k) or {}
        m = fs.get("mean")
        if m is not None:
            top_feats.append(f"{k} mean={float(m):.3f}")
    prov_bits: list[str] = []
    for k in PROVENANCE_KEYS[:8]:
        fr = ((summ.get("provenance") or {}).get(k) or {}).get("fraction")
        if fr is not None and float(fr) > 0.15:
            prov_bits.append(f"{k}={float(fr):.0%}")
    twohop_bits: list[str] = []
    for ch in TWOHOP_CHANNELS:
        col = f"twohop_via_{ch}"
        if col in df.columns:
            fr = float(df[col].fillna(False).astype(bool).mean())
            if fr > 0.1:
                twohop_bits.append(f"{col}={fr:.0%}")
    lines = [
        f"{cohort_name}: n={n}, score median={summ.get('score_median') or pd.to_numeric(df['score']).median():.3f}",
    ]
    if top_feats:
        lines.append("Similarity: " + "; ".join(top_feats[:5]))
    if prov_bits:
        lines.append("Provenance: " + ", ".join(prov_bits[:6]))
    if twohop_bits:
        lines.append("2-hop: " + ", ".join(twohop_bits[:5]))
    path_m = (summ.get("feature_summaries") or {}).get("path_token_jaccard_combined", {}).get("mean")
    body_only_m = (summ.get("feature_summaries") or {}).get("body_only_token_jaccard", {}).get("mean")
    if path_m is not None:
        lines.append(f"path_token_jaccard_combined mean={float(path_m):.3f}")
    if body_only_m is not None:
        lines.append(f"body_only_token_jaccard mean={float(body_only_m):.3f}")
    return {
        "cohort": cohort_name,
        "n_pairs": n,
        "bullet_summary": " | ".join(lines),
        "feature_summaries": summ.get("feature_summaries"),
        "provenance": summ.get("provenance"),
        "shared_evidence": summ.get("shared_evidence"),
    }


def _build_promotion_path_analysis(
    *,
    low_vs_mid_marginal: dict[str, Any] | None,
    mid_vs_high_marginal: dict[str, Any] | None,
    low_vs_mid_joint: dict[str, Any] | None,
    mid_vs_high_joint: dict[str, Any] | None,
    low_vs_mid_body_path: dict[str, Any] | None,
    mid_vs_high_body_path: dict[str, Any] | None,
) -> dict[str, Any]:
    def _top(marg: dict[str, Any] | None, n: int = 8) -> list[str]:
        if not marg:
            return []
        out: list[str] = []
        for r in (marg.get("ranked_separators_top15") or marg.get("ranked_separators_top25_extended") or [])[:n]:
            out.append(f"{r.get('metric_group')}:{r.get('metric_name')} Δ={r.get('difference_left_minus_right')}")
        return out

    def _top_joint(j: dict[str, Any] | None, n: int = 6) -> list[str]:
        if not j:
            return []
        names: list[str] = []
        for r in (j.get("ranked_joint_separators_top15") or [])[:n]:
            names.append(str(r.get("condition_name") or r.get("metric_name")))
        return names

    low_to_mid_signals = _top(low_vs_mid_marginal) + _top_joint(low_vs_mid_joint, 4)
    mid_to_high_signals = _top(mid_vs_high_marginal) + _top_joint(mid_vs_high_joint, 4)

    interpretation: list[str] = []
    if low_to_mid_signals:
        interpretation.append(
            "Low → mid: pairs gain " + "; ".join(low_to_mid_signals[:5]) + " (marginal/joint separators)."
        )
    if mid_to_high_signals:
        interpretation.append(
            "Mid → high: rescued-like edges show " + "; ".join(mid_to_high_signals[:5]) + "."
        )
    if low_vs_mid_body_path:
        interpretation.extend((low_vs_mid_body_path.get("interpretation_notes") or [])[:3])
    if mid_vs_high_body_path:
        interpretation.extend((mid_vs_high_body_path.get("interpretation_notes") or [])[:3])

    return {
        "low_to_mid": {
            "marginal_top_separators": _top(low_vs_mid_marginal),
            "joint_top_conditions": _top_joint(low_vs_mid_joint),
            "body_path_comparison": low_vs_mid_body_path,
        },
        "mid_to_high": {
            "marginal_top_separators": _top(mid_vs_high_marginal),
            "joint_top_conditions": _top_joint(mid_vs_high_joint),
            "body_path_comparison": mid_vs_high_body_path,
        },
        "interpretation": interpretation[:12],
    }


def _generate_frontier_recommendations(
    *,
    profiles: dict[str, Any],
    comparisons: dict[str, Any],
    promotion: dict[str, Any],
    population_diag: dict[str, Any],
) -> dict[str, Any]:
    good = profiles.get("good_unlabeled_edge_profile") or {}
    mid = profiles.get("mid_unlabeled_edge_profile") or {}
    bad = profiles.get("bad_unlabeled_edge_profile") or {}

    interventions: list[str] = []
    mid_high = comparisons.get("mid_same_vs_high_same") or {}
    body_high = (mid_high.get("body_path_signal_comparison") or {}).get("interpretation_notes") or []
    if body_high:
        interventions.append(
            "Scorer/graph: strengthen path + body-only features for promotion to high band — "
            + str(body_high[0])
        )

    prom = promotion.get("interpretation") or []
    if prom:
        interventions.append("Promotion path: " + prom[0])

    if population_diag.get("candidate_family_provenance_unavailable"):
        interventions.append(
            "Export candidate-family provenance columns on pair_training_dataset for richer frontier diagnostics."
        )

    return {
        "A_what_makes_a_good_high_unlabeled_edge": good.get("bullet_summary"),
        "B_what_makes_a_middle_band_unlabeled_edge": mid.get("bullet_summary"),
        "C_what_makes_a_bad_low_collapsed_unlabeled_edge": bad.get("bullet_summary"),
        "D_likely_next_interventions": interventions[:10] or [
            "Inspect pair_frontier_analysis_joint_summary.json ranked_joint_separators and HTML cohort reviews."
        ],
        "good_unlabeled_edge_profile": good,
        "mid_unlabeled_edge_profile": mid,
        "bad_unlabeled_edge_profile": bad,
        "mid_cross_note": (profiles.get("mid_cross_unlabeled_profile") or {}).get("bullet_summary"),
    }


def _write_legacy_mid_band_summary(
    *,
    layout: dict[str, Path],
    gt_path: Path,
    payload: dict[str, Any],
    comparisons: dict[str, Any],
    thresholds: FrontierThresholds,
    suffix: str,
) -> dict[str, str]:
    """Backward-compatible pair_mid_band_frontier_* JSON paths."""
    core_json = layout["core_json"]
    debug_csv = layout["debug_csv"]
    summary_path = core_json / f"pair_mid_band_frontier_summary{suffix}.json"
    joint_path = core_json / f"pair_mid_band_frontier_joint_summary{suffix}.json"
    table_path = debug_csv / f"pair_mid_band_frontier_table{suffix}.csv"

    counts = payload.get("cohort_counts") or {}
    legacy = {
        "gt_path": str(gt_path),
        "thresholds": payload.get("thresholds"),
        "counts": {
            "n_mid_same_unlabeled": counts.get("mid_same_unlabeled"),
            "n_mid_cross_unlabeled": counts.get("mid_cross_unlabeled"),
            "n_rescued_same_unlabeled": counts.get("high_same_unlabeled"),
            "n_low_same_unlabeled": counts.get("low_same_unlabeled"),
        },
        "primary_comparison_same_vs_cross": comparisons.get("mid_same_vs_mid_cross"),
        "secondary_comparison_mid_same_vs_rescued_same": comparisons.get("mid_same_vs_high_same"),
        "optional_comparison_low_same_vs_mid_same": comparisons.get("mid_same_vs_low_same"),
        "community_cut_zone_analysis": payload.get("community_cut_zone_analysis"),
        "mid_band_frontier_recommendations": payload.get("frontier_recommendations"),
        "artifact_paths": {
            "summary_json": rel_to_root(layout, summary_path),
            "joint_summary_json": rel_to_root(layout, joint_path),
            "frontier_summary_json": payload.get("artifact_paths", {}).get("frontier_summary_json"),
        },
        "superseded_by": "pair_frontier_analysis_summary.json",
    }
    joint_legacy = {
        "gt_path": str(gt_path),
        "same_vs_cross_joint": (comparisons.get("mid_same_vs_mid_cross") or {}).get("joint"),
        "mid_vs_rescued_joint": (comparisons.get("mid_same_vs_high_same") or {}).get("joint"),
    }
    summary_path.write_text(json.dumps(legacy, indent=2, default=str), encoding="utf-8")
    joint_path.write_text(json.dumps(joint_legacy, indent=2, default=str), encoding="utf-8")
    rows = payload.get("table_rows") or []
    pd.DataFrame(rows).to_csv(table_path, index=False)
    return {
        "summary_path": str(summary_path),
        "joint_summary_path": str(joint_path),
        "table_path": str(table_path),
    }


def run_pair_frontier_analysis(
    *,
    df_work: pd.DataFrame,
    scores: np.ndarray,
    gt_path: Path,
    label_map: dict[str, Any],
    layout: dict[str, Path],
    nodes_by_email: dict[str, dict[str, set[str]]],
    evidence_index: dict[tuple[str, str], list[dict[str, Any]]] | None,
    email_text_by_eid: dict[str, dict[str, str]] | None,
    thresholds: FrontierThresholds | None = None,
    export_flags: ExportFlags | None = None,
    filename_suffix: str = "",
    write_legacy_mid_band: bool = True,
) -> dict[str, Any]:
    """Run full low/mid/high frontier analysis for one GT file."""
    flags = export_flags or ExportFlags()
    th = thresholds or FrontierThresholds()
    gt_path = Path(gt_path).resolve()
    core_json = layout["core_json"]
    debug_csv = layout["debug_csv"]
    review_html = layout["review_html"]
    suffix = filename_suffix or ""

    ei = df_work["email_i"].astype(str).values
    ej = df_work["email_j"].astype(str).values
    n = len(df_work)
    camp_i = np.array([label_map.get(str(ei[k])) for k in range(n)], dtype=object)
    camp_j = np.array([label_map.get(str(ej[k])) for k in range(n)], dtype=object)
    both = np.array([camp_i[k] is not None and camp_j[k] is not None for k in range(n)], dtype=bool)
    scored = np.isfinite(_as_float_array(scores))
    eval_mask = both & scored
    same_eval = eval_mask & (camp_i == camp_j)
    cross_eval = eval_mask & (camp_i != camp_j)
    unl_eval = (
        df_work["pair_status"].astype(str).str.lower().eq("unlabeled").to_numpy()
        if "pair_status" in df_work.columns
        else np.zeros(n, dtype=bool)
    )
    unl_eval = eval_mask & unl_eval

    df_eval = df_work.loc[eval_mask].copy()
    df_eval["score"] = scores[eval_mask]
    same_e = same_eval[eval_mask]
    cross_e = cross_eval[eval_mask]

    cohort_masks = frontier_unlabeled_cohort_masks(
        same_eval=same_e,
        cross_eval=cross_e,
        unl_eval=unl_eval[eval_mask],
        scores=scores[eval_mask],
        thresholds=th,
    )

    cohort_dfs: dict[str, pd.DataFrame] = {}
    gt_rel = {
        "low_same_unlabeled": "same_campaign",
        "mid_same_unlabeled": "same_campaign",
        "high_same_unlabeled": "same_campaign",
        "low_cross_unlabeled": "cross_campaign",
        "mid_cross_unlabeled": "cross_campaign",
        "high_cross_unlabeled": "cross_campaign",
    }
    for cname in COHORT_NAMES:
        cohort_dfs[cname] = _build_cohort_inspection_df(
            df_eval=df_eval,
            row_mask=cohort_masks[cname],
            gt_path=gt_path,
            label_map=label_map,
            gt_relation=gt_rel[cname],
            cohort=cname,
            nodes_by_email=nodes_by_email,
            evidence_index=evidence_index,
        )

    cohort_counts = {k: int(cohort_masks[k].sum()) for k in COHORT_NAMES}
    population_diag = _feature_population_diagnostics(cohort_dfs, df_eval=df_eval)

    n_eval = int(len(df_eval))
    cohort_summaries = {
        c: _cohort_summary_block(cohort_dfs[c], cohort=c, nodes_by_email=nodes_by_email, n_total_eval=n_eval)
        for c in COHORT_NAMES
    }

    comparisons: dict[str, Any] = {}
    table_rows: list[dict[str, Any]] = []
    joint_payloads: dict[str, Any] = {}

    df_mid_same = cohort_dfs["mid_same_unlabeled"]
    df_mid_cross = cohort_dfs["mid_cross_unlabeled"]
    marg_a, rows_a = _build_same_vs_cross_marginal(
        gt_path=gt_path,
        same_df=df_mid_same,
        cross_df=df_mid_cross,
        nodes_by_email=nodes_by_email,
        band_kind="mid",
    )
    marg_a, extra_a = _extend_marginal_with_full_features(
        marg_a,
        left_df=df_mid_same,
        right_df=df_mid_cross,
        comparison="mid_same_unlabeled_vs_mid_cross_unlabeled",
        left_label="mid_same",
        right_label="mid_cross",
        gt_path=gt_path,
        nodes_by_email=nodes_by_email,
    )
    rows_a.extend(extra_a)
    joint_a, jrows_a = _build_frontier_joint_two_cohort(
        gt_path=gt_path,
        left_df=df_mid_same,
        right_df=df_mid_cross,
        left_mask_eval=cohort_masks["mid_same_unlabeled"],
        right_mask_eval=cohort_masks["mid_cross_unlabeled"],
        df_eval=df_eval,
        comparison="mid_same_unlabeled_vs_mid_cross_unlabeled",
        band_kind="mid",
        value_key_left="mid_same_value",
        value_key_right="mid_cross_value",
        nodes_by_email=nodes_by_email,
        evidence_index=evidence_index,
        marginal_sep=marg_a,
    )
    body_a = _build_body_path_signal_comparison(
        left_df=df_mid_same,
        right_df=df_mid_cross,
        comparison="mid_same_unlabeled_vs_mid_cross_unlabeled",
        left_label="mid_same",
        right_label="mid_cross",
    )
    comparisons["mid_same_vs_mid_cross"] = {
        "comparison_id": "A_mid_same_vs_mid_cross",
        "marginal": marg_a,
        "joint": joint_a,
        "body_path_signal_comparison": body_a,
    }
    table_rows.extend(rows_a)
    table_rows.extend(jrows_a)
    joint_payloads["mid_same_vs_mid_cross"] = joint_a

    df_high_same = cohort_dfs["high_same_unlabeled"]
    marg_b, rows_b = _build_two_cohort_marginal(
        gt_path=gt_path,
        left_df=df_mid_same,
        right_df=df_high_same,
        comparison="mid_same_unlabeled_vs_high_same_unlabeled",
        left_label="mid_same",
        right_label="high_same",
        nodes_by_email=nodes_by_email,
    )
    marg_b, extra_b = _extend_marginal_with_full_features(
        marg_b,
        left_df=df_mid_same,
        right_df=df_high_same,
        comparison="mid_same_unlabeled_vs_high_same_unlabeled",
        left_label="mid_same",
        right_label="high_same",
        gt_path=gt_path,
        nodes_by_email=nodes_by_email,
    )
    rows_b.extend(extra_b)
    joint_b, jrows_b = _build_frontier_joint_two_cohort(
        gt_path=gt_path,
        left_df=df_mid_same,
        right_df=df_high_same,
        left_mask_eval=cohort_masks["mid_same_unlabeled"],
        right_mask_eval=cohort_masks["high_same_unlabeled"],
        df_eval=df_eval,
        comparison="mid_same_unlabeled_vs_high_same_unlabeled",
        band_kind="mid_vs_high",
        value_key_left="mid_same_value",
        value_key_right="high_same_value",
        nodes_by_email=nodes_by_email,
        evidence_index=evidence_index,
        marginal_sep=marg_b,
    )
    body_b = _build_body_path_signal_comparison(
        left_df=df_mid_same,
        right_df=df_high_same,
        comparison="mid_same_unlabeled_vs_high_same_unlabeled",
        left_label="mid_same",
        right_label="high_same",
    )
    comparisons["mid_same_vs_high_same"] = {
        "comparison_id": "B_mid_same_vs_high_same",
        "marginal": marg_b,
        "joint": joint_b,
        "body_path_signal_comparison": body_b,
    }
    table_rows.extend(rows_b)
    table_rows.extend(jrows_b)
    joint_payloads["mid_same_vs_high_same"] = joint_b

    df_low_same = cohort_dfs["low_same_unlabeled"]
    marg_c, rows_c = _build_two_cohort_marginal(
        gt_path=gt_path,
        left_df=df_low_same,
        right_df=df_mid_same,
        comparison="low_same_unlabeled_vs_mid_same_unlabeled",
        left_label="low_same",
        right_label="mid_same",
        nodes_by_email=nodes_by_email,
    )
    marg_c, extra_c = _extend_marginal_with_full_features(
        marg_c,
        left_df=df_low_same,
        right_df=df_mid_same,
        comparison="low_same_unlabeled_vs_mid_same_unlabeled",
        left_label="low_same",
        right_label="mid_same",
        gt_path=gt_path,
        nodes_by_email=nodes_by_email,
    )
    rows_c.extend(extra_c)
    joint_c, jrows_c = _build_frontier_joint_two_cohort(
        gt_path=gt_path,
        left_df=df_low_same,
        right_df=df_mid_same,
        left_mask_eval=cohort_masks["low_same_unlabeled"],
        right_mask_eval=cohort_masks["mid_same_unlabeled"],
        df_eval=df_eval,
        comparison="low_same_unlabeled_vs_mid_same_unlabeled",
        band_kind="low_vs_mid",
        value_key_left="low_same_value",
        value_key_right="mid_same_value",
        nodes_by_email=nodes_by_email,
        evidence_index=evidence_index,
        marginal_sep=marg_c,
    )
    body_c = _build_body_path_signal_comparison(
        left_df=df_low_same,
        right_df=df_mid_same,
        comparison="low_same_unlabeled_vs_mid_same_unlabeled",
        left_label="low_same",
        right_label="mid_same",
    )
    comparisons["mid_same_vs_low_same"] = {
        "comparison_id": "C_mid_same_vs_low_same",
        "marginal": marg_c,
        "joint": joint_c,
        "body_path_signal_comparison": body_c,
        "note": "marginal rows use left=low, right=mid (positive difference favors mid over low)",
    }
    table_rows.extend(rows_c)
    table_rows.extend(jrows_c)
    joint_payloads["mid_same_vs_low_same"] = joint_c

    promotion = _build_promotion_path_analysis(
        low_vs_mid_marginal=marg_c,
        mid_vs_high_marginal=marg_b,
        low_vs_mid_joint=joint_c,
        mid_vs_high_joint=joint_b,
        low_vs_mid_body_path=body_c,
        mid_vs_high_body_path=body_b,
    )

    profiles = {
        "good_unlabeled_edge_profile": _build_edge_profile(
            cohort_name="high_same_unlabeled",
            df=df_high_same,
            nodes_by_email=nodes_by_email,
        ),
        "mid_unlabeled_edge_profile": _build_edge_profile(
            cohort_name="mid_same_unlabeled",
            df=df_mid_same,
            nodes_by_email=nodes_by_email,
        ),
        "bad_unlabeled_edge_profile": _build_edge_profile(
            cohort_name="low_same_unlabeled",
            df=df_low_same,
            nodes_by_email=nodes_by_email,
        ),
        "mid_cross_unlabeled_profile": _build_edge_profile(
            cohort_name="mid_cross_unlabeled",
            df=df_mid_cross,
            nodes_by_email=nodes_by_email,
        ),
    }

    community_cut = _community_cut_zone_analysis(
        df_eval=df_eval,
        scores=scores[eval_mask],
        same_eval=same_e,
        cross_eval=cross_e,
        unl_eval=unl_eval[eval_mask],
        thresholds=th,
    )

    recommendations = _generate_frontier_recommendations(
        profiles=profiles,
        comparisons=comparisons,
        promotion=promotion,
        population_diag=population_diag,
    )

    summary_path = core_json / f"pair_frontier_analysis_summary{suffix}.json"
    joint_path = core_json / f"pair_frontier_analysis_joint_summary{suffix}.json"
    table_path = debug_csv / f"pair_frontier_analysis_table{suffix}.csv"

    pd.DataFrame(table_rows).to_csv(table_path, index=False)

    review_paths: dict[str, str] = {}
    email_catalog = email_text_by_eid if email_text_by_eid is not None else {}
    if email_catalog is not None:
        html_exports = {
            "mid_same_review_html": (
                "pair_mid_band_same_unlabeled_for_review",
                df_mid_same,
                "Mid-band same-campaign unlabeled (frontier)",
            ),
            "mid_cross_review_html": (
                "pair_mid_band_cross_unlabeled_for_review",
                df_mid_cross,
                "Mid-band cross-campaign unlabeled (frontier)",
            ),
            "high_same_review_html": (
                "pair_high_band_same_unlabeled_for_review",
                df_high_same,
                "High-band same-campaign unlabeled (rescued-like)",
            ),
            "low_same_review_html": (
                "pair_low_band_same_unlabeled_for_review",
                df_low_same,
                "Low-band same-campaign unlabeled (collapsed-like)",
            ),
        }
        for key, (stem, df_cohort, title) in html_exports.items():
            if df_cohort.empty and key.startswith("high"):
                continue
            review_paths[key] = _export_mid_band_review_html(
                df_pairs=df_cohort,
                layout=layout,
                email_text_by_eid=email_catalog,
                out_name=f"{stem}{suffix}.html",
                title=title,
                subtitle=f"Cohort {df_cohort['frontier_cohort'].iloc[0] if not df_cohort.empty else 'empty'} — inspect body/path/2-hop/provenance.",
                export_flags=flags,
                regime_fn=_classify_frontier_review_regime,
                review_prompt_fn=_frontier_review_prompt,
            )

    payload: dict[str, Any] = {
        "gt_path": str(gt_path),
        "analysis_kind": "pair_frontier_analysis",
        "thresholds": {
            "low_score_max_inclusive": th.low_score_max,
            "mid_score_min_exclusive": th.mid_score_min,
            "mid_score_max_inclusive": th.mid_score_max,
            "high_score_min_exclusive": th.high_score_min,
            "community_cut_score_range": [th.community_cut_score_min, th.community_cut_score_max],
        },
        "cohort_definitions": {
            c: {
                "description": c.replace("_", " "),
                "n_pairs": cohort_counts[c],
                "score_band": (
                    "low"
                    if c.startswith("low")
                    else "high"
                    if c.startswith("high")
                    else "mid"
                ),
                "gt_relation": "same_campaign" if "same" in c else "cross_campaign",
                "pair_status": "unlabeled",
            }
            for c in COHORT_NAMES
        },
        "cohort_counts": cohort_counts,
        "cohort_summaries": cohort_summaries,
        "primary_comparisons": comparisons,
        "promotion_path_analysis": promotion,
        "edge_profiles": profiles,
        "frontier_feature_population_diagnostics": population_diag,
        "frontier_recommendations": recommendations,
        "community_cut_zone_analysis": community_cut,
        "table_rows": table_rows,
        "artifact_paths": {},
    }
    payload["artifact_paths"] = {
        "frontier_summary_json": rel_to_root(layout, summary_path),
        "frontier_joint_summary_json": rel_to_root(layout, joint_path),
        "frontier_table_csv": rel_to_root(layout, table_path),
        **{k: rel_to_root(layout, Path(v)) for k, v in review_paths.items()},
    }

    summary_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    with open(joint_path, "w", encoding="utf-8") as jf:
        json.dump(
            {"gt_path": str(gt_path), "comparisons": joint_payloads, "n_joint_table_rows": len(table_rows)},
            jf,
            indent=2,
            default=str,
        )

    legacy_paths: dict[str, str] = {}
    if write_legacy_mid_band:
        legacy_paths = _write_legacy_mid_band_summary(
            layout=layout,
            gt_path=gt_path,
            payload=payload,
            comparisons={
                "mid_same_vs_mid_cross": comparisons["mid_same_vs_mid_cross"],
                "mid_same_vs_high_same": comparisons["mid_same_vs_high_same"],
                "mid_same_vs_low_same": comparisons["mid_same_vs_low_same"],
            },
            thresholds=th,
            suffix=suffix,
        )

    digest = {
        "cohort_counts": cohort_counts,
        "top_comparison": "mid_same_vs_mid_cross",
        "n_mid_same": cohort_counts.get("mid_same_unlabeled"),
        "n_high_same": cohort_counts.get("high_same_unlabeled"),
        "n_low_same": cohort_counts.get("low_same_unlabeled"),
        "top_recommendation": (recommendations.get("D_likely_next_interventions") or [""])[0],
    }

    return {
        "summary_path": str(summary_path),
        "joint_summary_path": str(joint_path),
        "table_path": str(table_path),
        "paths": {**review_paths, **legacy_paths},
        "legacy_mid_band": legacy_paths,
        "digest": digest,
        "payload": payload,
    }
