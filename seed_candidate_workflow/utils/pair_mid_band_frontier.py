"""
Mid-band frontier analysis for pair score separation (analysis-only).

Compares GT-covered unlabeled pairs in the uncertain score middle band:
same-campaign vs cross-campaign, mid vs rescued same-campaign, and optional low vs mid.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from seed_candidate_workflow.utils.pair_low_band_twohop_channel import (
    extend_bool_terms_for_low_band_channels,
    low_band_twohop_joint_rule_names,
)
from seed_candidate_workflow.utils.pair_score_separation_output_layout import (
    ExportFlags,
    rel_to_root,
)
from seed_candidate_workflow.utils.scorer_diagnostics_rules import (
    BINARY_CONDITION_RULES_DEFAULT,
    CANDIDATE_RULES_DEFAULT,
    FEATURE_KEYS_DEFAULT,
    PROVENANCE_KEYS_DEFAULT,
    SEMANTIC_BUCKET_RULES_DEFAULT,
    SHARED_EVIDENCE_KEYS_DEFAULT,
)

_PSE: Any = None


def _pse() -> Any:
    global _PSE
    if _PSE is None:
        from seed_candidate_workflow.utils import pair_score_separation as mod

        _PSE = mod
    return _PSE


MID_BAND_PROVENANCE_KEYS: tuple[str, ...] = (
    "from_seed",
    *PROVENANCE_KEYS_DEFAULT,
)

MID_BAND_EXTRA_JOINT_RULES: tuple[str, ...] = (
    "semantic_ge_0_90",
    "semantic_ge_0_90_AND_n_shared_core_channels_ge_1",
    "semantic_ge_0_90_AND_shared_sender",
    "semantic_ge_0_90_AND_shared_html_fp",
    "from_2hop_AND_shared_html_fp",
    "from_2hop_AND_source_count_eq_1",
    "source_count_eq_1_AND_shared_html_fp",
    "shared_html_fp_AND_NOT_shared_sender",
    "semantic_ge_0_90_AND_shared_stem",
    "n_shared_core_channels_ge_1_AND_shared_sender",
)

MID_BAND_MARGINAL_FEATURE_KEYS: tuple[str, ...] = tuple(
    dict.fromkeys(
        [
            *FEATURE_KEYS_DEFAULT,
            "shared_sender_count",
            "shared_stem_count",
            "shared_url_count",
            "shared_attachment_count",
            "shared_sender_domain_count",
            "shared_domain_count",
            "n_shared_core_channels",
        ]
    )
)


@dataclass(frozen=True)
class MidBandThresholds:
    low_score_max: float = 0.15
    mid_score_min: float = 0.15
    mid_score_max: float = 0.50
    high_score_min: float = 0.80
    rescued_score_min: float = 0.80
    community_cut_score_min: float = 0.30
    community_cut_score_max: float = 0.50


def _as_float_array(scores: np.ndarray | pd.Series) -> np.ndarray:
    """Coerce scores to float ndarray (works for ndarray or Series input)."""
    return np.asarray(pd.to_numeric(np.asarray(scores), errors="coerce"), dtype=np.float64)


def score_band_masks(
    scores: np.ndarray,
    *,
    thresholds: MidBandThresholds,
) -> dict[str, np.ndarray]:
    """Finite-score band masks (low/mid/high/rescued/collapsed/community_cut)."""
    sv = _as_float_array(scores)
    finite = np.isfinite(sv)
    t = thresholds
    return {
        "finite": finite,
        "low": finite & (sv >= 0.0) & (sv <= float(t.low_score_max)),
        "mid": finite & (sv > float(t.mid_score_min)) & (sv <= float(t.mid_score_max)),
        "high": finite & (sv > float(t.high_score_min)) & (sv <= 1.0),
        "rescued": finite & (sv >= float(t.rescued_score_min)),
        "collapsed": finite & (sv <= 0.10),
        "community_cut": finite
        & (sv >= float(t.community_cut_score_min))
        & (sv <= float(t.community_cut_score_max)),
    }


def _classify_mid_band_review_regime(row: pd.Series) -> str:
    rel = str(row.get("gt_relation") or "")
    if rel == "same_campaign":
        return "mid_same_unlabeled"
    if rel == "cross_campaign":
        return "mid_cross_unlabeled"
    return "mid_unknown_unlabeled"


def _mid_band_review_prompt(row: pd.Series) -> str:
    rel = str(row.get("gt_relation") or "")
    if rel == "same_campaign":
        return (
            "Mid-band same-campaign unlabeled: model is uncertain. "
            "What body/path/semantic/support signals should push this toward rescued (high) scores?"
        )
    if rel == "cross_campaign":
        return (
            "Mid-band cross-campaign unlabeled: should stay below community cut (~0.4). "
            "What signals distinguish this from mid-band same-campaign pairs?"
        )
    return "Mid-band unlabeled pair â€” inspect frontier evidence."


def _summarize_cohort(
    gdf: pd.DataFrame,
    *,
    n_total_eval: int,
    nodes_by_email: dict[str, dict[str, set[str]]],
) -> dict[str, Any]:
    return _pse()._summarize_group(
        gdf=gdf,
        n_total_eval=n_total_eval,
        nodes_by_email=nodes_by_email,
    )


def _compare_fraction_metric(*, same_v: Any, cross_v: Any) -> dict[str, Any]:
    return _pse()._compare_fraction_metric(same_v=same_v, cross_v=cross_v)


def _build_same_vs_cross_marginal(
    *,
    gt_path: Path,
    same_df: pd.DataFrame,
    cross_df: pd.DataFrame,
    nodes_by_email: dict[str, dict[str, set[str]]],
    band_kind: str = "mid",
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    n_same = int(len(same_df))
    n_cross = int(len(cross_df))
    n_band = n_same + n_cross
    same_s = _summarize_cohort(same_df, n_total_eval=n_band, nodes_by_email=nodes_by_email)
    cross_s = _summarize_cohort(cross_df, n_total_eval=n_band, nodes_by_email=nodes_by_email)
    same_col = f"same_{band_kind}_value"
    cross_col = f"cross_{band_kind}_value"
    out: dict[str, Any] = {
        "gt_path": str(gt_path.resolve()),
        "comparison": f"mid_same_unlabeled_vs_mid_cross_unlabeled",
        "band_kind": band_kind,
        "counts": {
            "n_same_campaign_mid_unlabeled": n_same,
            "n_cross_campaign_mid_unlabeled": n_cross,
            "n_total_mid_band_unlabeled": n_band,
        },
    }
    rows: list[dict[str, Any]] = []

    for k in MID_BAND_PROVENANCE_KEYS:
        sv = ((same_s.get("provenance") or {}).get(k) or {}).get("fraction")
        cv = ((cross_s.get("provenance") or {}).get(k) or {}).get("fraction")
        if sv is None and k in same_df.columns and n_same:
            sv = float(same_df[k].fillna(False).astype(bool).mean())
        if cv is None and k in cross_df.columns and n_cross:
            cv = float(cross_df[k].fillna(False).astype(bool).mean())
        cmp = _compare_fraction_metric(same_v=sv, cross_v=cv)
        rows.append(
            {
                "gt_path": str(gt_path.resolve()),
                "comparison": out["comparison"],
                "metric_group": "provenance",
                "metric_name": k,
                same_col: cmp["same_value"],
                cross_col: cmp["cross_value"],
                "difference": cmp["difference_same_minus_cross"],
                "abs_difference": cmp["abs_difference"],
            }
        )

    for k in MID_BAND_MARGINAL_FEATURE_KEYS:
        ssum = (same_s.get("feature_summaries") or {}).get(k) or {}
        csum = (cross_s.get("feature_summaries") or {}).get(k) or {}
        ms = _pse()._safe_float(ssum.get("mean"))
        mc = _pse()._safe_float(csum.get("mean"))
        diff = (ms - mc) if (ms is not None and mc is not None) else None
        rows.append(
            {
                "gt_path": str(gt_path.resolve()),
                "comparison": out["comparison"],
                "metric_group": "feature_mean",
                "metric_name": k,
                same_col: ms,
                cross_col: mc,
                "difference": diff,
                "abs_difference": abs(diff) if diff is not None else None,
            }
        )

    for k in SHARED_EVIDENCE_KEYS_DEFAULT:
        sv = ((same_s.get("shared_evidence") or {}).get(k) or {}).get("fraction_edges_with_at_least_1")
        cv = ((cross_s.get("shared_evidence") or {}).get(k) or {}).get("fraction_edges_with_at_least_1")
        cmp = _compare_fraction_metric(same_v=sv, cross_v=cv)
        rows.append(
            {
                "gt_path": str(gt_path.resolve()),
                "comparison": out["comparison"],
                "metric_group": "shared_evidence",
                "metric_name": k,
                same_col: cmp["same_value"],
                cross_col: cmp["cross_value"],
                "difference": cmp["difference_same_minus_cross"],
                "abs_difference": cmp["abs_difference"],
            }
        )

    ranked = [r for r in rows if r.get("abs_difference") is not None]
    ranked.sort(key=lambda r: float(r["abs_difference"]), reverse=True)
    out["ranked_separators_top15"] = ranked[:15]
    out["ranked_separators_favoring_same_top10"] = [
        r for r in ranked if r.get("difference") is not None and float(r["difference"]) > 0
    ][:10]
    out["ranked_separators_favoring_cross_top10"] = [
        r for r in ranked if r.get("difference") is not None and float(r["difference"]) < 0
    ][:10]
    return out, rows


def _build_two_cohort_marginal(
    *,
    gt_path: Path,
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    comparison: str,
    left_label: str,
    right_label: str,
    nodes_by_email: dict[str, dict[str, set[str]]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Generic marginal comparison (e.g. mid_same vs rescued_same)."""
    n_l = int(len(left_df))
    n_r = int(len(right_df))
    n_tot = n_l + n_r
    left_s = _summarize_cohort(left_df, n_total_eval=n_tot, nodes_by_email=nodes_by_email)
    right_s = _summarize_cohort(right_df, n_total_eval=n_tot, nodes_by_email=nodes_by_email)
    lcol = f"{left_label}_value"
    rcol = f"{right_label}_value"
    out: dict[str, Any] = {
        "gt_path": str(gt_path.resolve()),
        "comparison": comparison,
        "counts": {f"n_{left_label}": n_l, f"n_{right_label}": n_r},
    }
    rows: list[dict[str, Any]] = []

    for k in MID_BAND_PROVENANCE_KEYS:
        lv = ((left_s.get("provenance") or {}).get(k) or {}).get("fraction")
        rv = ((right_s.get("provenance") or {}).get(k) or {}).get("fraction")
        if lv is None and k in left_df.columns and n_l:
            lv = float(left_df[k].fillna(False).astype(bool).mean())
        if rv is None and k in right_df.columns and n_r:
            rv = float(right_df[k].fillna(False).astype(bool).mean())
        diff = (lv - rv) if (lv is not None and rv is not None) else None
        rows.append(
            {
                "gt_path": str(gt_path.resolve()),
                "comparison": comparison,
                "metric_group": "provenance",
                "metric_name": k,
                lcol: lv,
                rcol: rv,
                "difference_left_minus_right": diff,
                "abs_difference": abs(diff) if diff is not None else None,
            }
        )

    for k in MID_BAND_MARGINAL_FEATURE_KEYS:
        ls = (left_s.get("feature_summaries") or {}).get(k) or {}
        rs = (right_s.get("feature_summaries") or {}).get(k) or {}
        lm = _pse()._safe_float(ls.get("mean"))
        rm = _pse()._safe_float(rs.get("mean"))
        diff = (lm - rm) if (lm is not None and rm is not None) else None
        rows.append(
            {
                "gt_path": str(gt_path.resolve()),
                "comparison": comparison,
                "metric_group": "feature_mean",
                "metric_name": k,
                lcol: lm,
                rcol: rm,
                "difference_left_minus_right": diff,
                "abs_difference": abs(diff) if diff is not None else None,
            }
        )

    ranked = [r for r in rows if r.get("abs_difference") is not None]
    ranked.sort(key=lambda r: float(r["abs_difference"]), reverse=True)
    out["ranked_separators_top15"] = ranked[:15]
    return out, rows


def _community_cut_zone_analysis(
    *,
    df_eval: pd.DataFrame,
    scores: np.ndarray,
    same_eval: np.ndarray,
    cross_eval: np.ndarray,
    unl_eval: np.ndarray,
    thresholds: MidBandThresholds,
) -> dict[str, Any]:
    """Describe pairs in the community decision zone (~0.30â€“0.50 by default)."""
    bands = score_band_masks(scores, thresholds=thresholds)
    cc = bands["community_cut"]
    sub = df_eval.loc[cc].copy()
    sub["score"] = scores[cc]
    if sub.empty:
        return {
            "community_cut_score_range": [
                float(thresholds.community_cut_score_min),
                float(thresholds.community_cut_score_max),
            ],
            "n_pairs_in_zone": 0,
            "note": "no_gt_covered_pairs_in_community_cut_zone",
        }

    same_cc = same_eval & cc
    cross_cc = cross_eval & cc
    unl_cc = unl_eval & cc
    st = sub["pair_status"].astype(str).str.lower() if "pair_status" in sub.columns else pd.Series(dtype=str)
    pos_cc = cc & (
        df_eval["pair_status"].astype(str).str.lower().eq("positive").to_numpy()
        if "pair_status" in df_eval.columns
        else np.zeros(len(df_eval), dtype=bool)
    )

    prov: dict[str, float] = {}
    for col in ("from_2hop", "from_semantic", "from_component", "from_rare_artifact"):
        if col in sub.columns:
            prov[col] = float(sub[col].fillna(False).astype(bool).mean())

    score_s = pd.to_numeric(sub["score"], errors="coerce")
    return {
        "community_cut_score_range": [
            float(thresholds.community_cut_score_min),
            float(thresholds.community_cut_score_max),
        ],
        "mid_band_score_range": [float(thresholds.mid_score_min), float(thresholds.mid_score_max)],
        "n_pairs_in_zone": int(len(sub)),
        "n_same_campaign_in_zone": int(same_cc.sum()),
        "n_cross_campaign_in_zone": int(cross_cc.sum()),
        "n_unlabeled_in_zone": int(unl_cc.sum()),
        "n_positive_in_zone": int(pos_cc.sum()),
        "score_mean": float(score_s.mean()) if score_s.notna().any() else None,
        "score_median": float(score_s.median()) if score_s.notna().any() else None,
        "provenance_fractions_in_zone": prov,
        "top_provenance_combos": (
            sub["provenance_combo"].astype(str).value_counts().head(8).to_dict()
            if "provenance_combo" in sub.columns
            else {}
        ),
        "interpretation": (
            "Pairs in this zone are near typical PU community edge thresholds (~0.4). "
            "Mid-band same vs cross comparisons explain which evidence keeps cross edges "
            "from being promoted while same-campaign edges remain uncertain."
        ),
    }


def _generate_mid_band_recommendations(
    *,
    same_vs_cross_marginal: dict[str, Any],
    same_vs_cross_joint: dict[str, Any],
    mid_vs_rescued_marginal: dict[str, Any] | None,
    mid_vs_rescued_joint: dict[str, Any] | None,
    community_cut: dict[str, Any],
) -> dict[str, Any]:
    def _top_lines(block: dict[str, Any] | None, key: str, n: int = 8) -> list[str]:
        if not block:
            return []
        out: list[str] = []
        for r in (block.get(key) or [])[:n]:
            name = r.get("condition_name") or r.get("metric_name") or "?"
            grp = r.get("metric_group") or r.get("analysis_section") or ""
            diff = r.get("difference") or r.get("difference_same_minus_cross") or r.get(
                "difference_left_minus_right"
            )
            out.append(f"{grp}:{name} (Î”={diff})")
        return out

    a_same_cross = _top_lines(same_vs_cross_joint, "ranked_joint_separators_favoring_same_top10")
    a_cross = _top_lines(same_vs_cross_joint, "ranked_joint_separators_favoring_cross_top10")
    b_lift = _top_lines(mid_vs_rescued_joint, "ranked_joint_separators_top15") if mid_vs_rescued_joint else []

    next_steps: list[str] = []
    if a_same_cross:
        next_steps.append(
            "Mid-band same-campaign pairs are enriched for: "
            + "; ".join(a_same_cross[:5])
            + ". Consider pair-scorer features or loss terms that reward these combinations in the middle band."
        )
    if a_cross:
        next_steps.append(
            "Mid-band cross-campaign pairs remain dangerous when: "
            + "; ".join(a_cross[:5])
            + ". Use as negative ranking targets or bridge-recovery guards."
        )
    if b_lift:
        next_steps.append(
            "To lift mid-band same toward rescued: emphasize signals where rescued differs from mid â€” "
            + "; ".join(b_lift[:5])
        )
    if int(community_cut.get("n_pairs_in_zone") or 0) > 0:
        next_steps.append(
            f"Community cut zone holds {community_cut.get('n_pairs_in_zone')} GT-covered pairs "
            f"({community_cut.get('n_same_campaign_in_zone')} same / {community_cut.get('n_cross_campaign_in_zone')} cross). "
            "Align PU threshold sweeps and mid-band training focus with this band."
        )
    if not next_steps:
        next_steps.append(
            "Inspect pair_mid_band_frontier_joint_summary.json ranked_joint_separators and HTML review cohorts."
        )

    return {
        "what_distinguishes_mid_band_same_from_mid_band_cross": {
            "marginal_top": same_vs_cross_marginal.get("ranked_separators_favoring_same_top10", [])[:8],
            "joint_favoring_same": a_same_cross,
            "joint_favoring_cross": a_cross,
        },
        "what_distinguishes_mid_band_same_from_rescued_same": {
            "marginal_top": (mid_vs_rescued_marginal or {}).get("ranked_separators_top15", [])[:8],
            "joint_top": b_lift,
        },
        "community_cut_zone_implications": community_cut.get("interpretation"),
        "implied_next_steps": next_steps,
    }


def _export_mid_band_review_html(
    *,
    df_pairs: pd.DataFrame,
    layout: dict[str, Path],
    email_text_by_eid: dict[str, dict[str, str]],
    out_name: str,
    title: str,
    subtitle: str,
    export_flags: ExportFlags,
    regime_fn: Any | None = None,
    review_prompt_fn: Any | None = None,
) -> str:
    review_html = layout["review_html"]
    html_path = review_html / out_name
    pse = _pse()
    if df_pairs.empty:
        pse._write_pairs_for_review_html(
            df_pairs,
            out_path=html_path,
            email_text_by_eid=email_text_by_eid,
            title=title,
            subtitle=subtitle + " (no pairs in cohort)",
        )
        return str(html_path)

    df_review = pse._enrich_pairs_with_email_text(
        df_pairs,
        email_text_by_eid=email_text_by_eid,
        preview_chars=500,
        regime_fn=regime_fn or _classify_mid_band_review_regime,
        review_prompt_fn=review_prompt_fn or _mid_band_review_prompt,
    )
    df_review = df_review.sort_values(
        ["gt_relation", "score", "email_i", "email_j"],
        ascending=[True, False, True, True],
        na_position="last",
    ).reset_index(drop=True)
    pse._write_pairs_for_review_html(
        df_review,
        out_path=html_path,
        email_text_by_eid=email_text_by_eid,
        title=title,
        subtitle=subtitle,
        review_prompt="Mid-band frontier manual review.",
        gt_note="mid-band unlabeled",
        filter_column="fp_regime",
    )
    if export_flags.emit_debug_csv:
        debug_csv = layout["debug_csv"] / f"debug_{out_name.replace('.html', '.csv')}"
        df_review.to_csv(debug_csv, index=False)
    return str(html_path)


def run_mid_band_frontier_analysis(
    *,
    df_work: pd.DataFrame,
    scores: np.ndarray,
    gt_path: Path,
    label_map: dict[str, Any],
    layout: dict[str, Path],
    nodes_by_email: dict[str, dict[str, set[str]]],
    evidence_index: dict[tuple[str, str], list[dict[str, Any]]] | None,
    email_text_by_eid: dict[str, dict[str, str]] | None,
    thresholds: MidBandThresholds | None = None,
    export_flags: ExportFlags | None = None,
    include_low_vs_mid: bool = True,
    filename_suffix: str = "",
) -> dict[str, Any]:
    """
    Run mid-band frontier analysis for one GT file (delegates to full frontier analysis).
    """
    from seed_candidate_workflow.utils.pair_frontier_analysis import run_pair_frontier_analysis

    out = run_pair_frontier_analysis(
        df_work=df_work,
        scores=scores,
        gt_path=gt_path,
        label_map=label_map,
        layout=layout,
        nodes_by_email=nodes_by_email,
        evidence_index=evidence_index,
        email_text_by_eid=email_text_by_eid,
        thresholds=thresholds,
        export_flags=export_flags,
        filename_suffix=filename_suffix,
        write_legacy_mid_band=True,
    )
    legacy = out.get("legacy_mid_band") or {}
    return {
        "summary_path": legacy.get("summary_path") or out.get("summary_path"),
        "joint_summary_path": legacy.get("joint_summary_path") or out.get("joint_summary_path"),
        "table_path": legacy.get("table_path") or out.get("table_path"),
        "paths": out.get("paths") or {},
        "digest": out.get("digest"),
        "payload": out.get("payload"),
        "frontier_summary_path": out.get("summary_path"),
        "frontier_joint_summary_path": out.get("joint_summary_path"),
    }
