"""
Rescued vs collapsed same-campaign unlabeled pair analysis (post-training diagnostics).

Extends pair score separation with bucket comparisons, joint separators, html-fp frontier
analysis, recommendations, and dedicated HTML review exports.
"""

from __future__ import annotations

import html
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from seed_candidate_workflow.utils.pair_low_band_twohop_channel import (
    attach_twohop_channel_columns,
    extend_bool_terms_for_low_band_channels,
    low_band_twohop_joint_rule_names,
)
from seed_candidate_workflow.utils.scorer_diagnostics_rules import (
    BINARY_CONDITION_RULES_DEFAULT,
    SEMANTIC_BUCKET_RULES_DEFAULT,
)

_PSE: Any = None


def _pse() -> Any:
    """Lazy import to avoid pulling torch via pair_score_separation at module load."""
    global _PSE
    if _PSE is None:
        from seed_candidate_workflow.utils import pair_score_separation as mod

        _PSE = mod
    return _PSE

EXTRA_PROVENANCE_COLS: tuple[str, ...] = (
    "from_shared_stem_highconf",
    "from_semantic_mid_sender_support",
    "from_semantic_mid_core_support",
    "from_semantic_mid_stem_support",
    "from_semantic_mid_senderlocalpart_support",
    "from_body_token_jaccard_highconf",
    "from_body_char4gram_jaccard_highconf",
)

EXTRA_FEATURE_COLS: tuple[str, ...] = (
    "body_token_jaccard",
    "body_char4gram_jaccard",
    "sender_localpart_norm_jaccard",
    "path_token_jaccard_combined",
    "url_path_token_jaccard",
    "stem_path_token_jaccard",
    "subject_token_jaccard",
    "subject_char4gram_jaccard",
    "body_only_token_jaccard",
    "body_only_char4gram_jaccard",
    "shared_sender_count",
    "shared_stem_count",
    "shared_url_count",
    "shared_attachment_count",
    "shared_sender_domain_count",
    "shared_domain_count",
    "n_shared_core_channels",
)

HTML_FP_SEMANTIC_LO = 0.75
HTML_FP_SEMANTIC_HI = 0.85

PATH_FEATURE_COLS: tuple[str, ...] = (
    "path_token_jaccard_combined",
    "url_path_token_jaccard",
    "stem_path_token_jaccard",
)

BODY_COMPARISON_COLS: tuple[str, ...] = (
    "body_token_jaccard",
    "body_char4gram_jaccard",
    "body_only_token_jaccard",
    "body_only_char4gram_jaccard",
)

HTML_FP_COHORT_FEATURE_COLS: tuple[str, ...] = (
    "semantic_cosine_max",
    "body_token_jaccard",
    "body_char4gram_jaccard",
    "body_only_token_jaccard",
    "body_only_char4gram_jaccard",
    "sender_localpart_norm_jaccard",
    *PATH_FEATURE_COLS,
    "source_count",
    "time_gap_seconds_min",
)

RANKING_EXCLUDED_EXACT: frozenset[str] = frozenset(
    {
        "gt_path",
        "gt_name",
        "gt_relation",
        "cohort",
        "pair_status",
        "gt_campaign_i",
        "gt_campaign_j",
        "email_i",
        "email_j",
        "same_unlabeled_bucket",
        "provenance_combo",
        "inspection_badges",
        "twohop_channel_badges",
        "gt_review_note",
        "fp_regime",
        "score",
        "_row",
    }
)

RANKING_EXCLUDED_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^gt_", re.I),
    re.compile(r"campaign", re.I),
    re.compile(r"^email_", re.I),
    re.compile(r"^n_pairs_", re.I),
    re.compile(r"_for_display$", re.I),
    re.compile(r"^subject_", re.I),
    re.compile(r"^body_preview", re.I),
)


BASE_INSPECTION_FEATURE_COLS: tuple[str, ...] = (
    "semantic_cosine_max",
    "component_cosine_max",
    "twohop_rarity_max",
    "rare_artifact_rarity_max",
    "time_gap_seconds_min",
    "source_count",
)


def _core_explanatory_feature_cols() -> tuple[str, ...]:
    """Pair-evidence whitelist (mirrors pair_score_separation inspection + training table)."""
    return tuple(dict.fromkeys([*BASE_INSPECTION_FEATURE_COLS, *EXTRA_FEATURE_COLS]))


def _is_ranking_excluded_column(col: str) -> bool:
    """GT labels, identifiers, and review metadata must not rank as explanatory separators."""
    if col in RANKING_EXCLUDED_EXACT:
        return True
    return any(p.search(col) for p in RANKING_EXCLUDED_PATTERNS)


def _filter_ranked_separator_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for r in rows:
        mg = str(r.get("metric_group") or "")
        mn = str(r.get("metric_name") or r.get("condition_name") or "")
        if mg == "feature_mean" and _is_ranking_excluded_column(mn):
            continue
        if "gt_campaign" in mn or mn.startswith("gt_"):
            continue
        out.append(r)
    return out


def _provenance_cols_in_df(df: pd.DataFrame) -> list[str]:
    cols = list(_pse()._INSPECTION_PROVENANCE_COLS) + list(EXTRA_PROVENANCE_COLS)
    return [c for c in cols if c in df.columns]


def _bool_series(df: pd.DataFrame, col: str) -> pd.Series:
    """Boolean column as Series; all-false if column missing (avoids DataFrame.get scalar default)."""
    if col not in df.columns:
        return pd.Series(False, index=df.index, dtype=bool)
    return df[col].fillna(False).astype(bool)


def _is_continuous_feature_column(df: pd.DataFrame, col: str) -> bool:
    """True for float/int feature columns; excludes bool flags (has_shared_*, twohop_via_*, etc.)."""
    if col not in df.columns or col in ("score", "_row", "same_unlabeled_bucket"):
        return False
    s = df[col]
    if pd.api.types.is_bool_dtype(s):
        return False
    return bool(pd.api.types.is_numeric_dtype(s))


def _feature_cols_in_df(df: pd.DataFrame) -> list[str]:
    """Whitelist of pair-evidence features only (no GT leakage via auto-discovered numeric cols)."""
    return [
        c
        for c in _core_explanatory_feature_cols()
        if _is_continuous_feature_column(df, c) and not _is_ranking_excluded_column(c)
    ]


def _num_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def _merge_pair_features_from_eval(df_ins: pd.DataFrame, df_eval: pd.DataFrame) -> pd.DataFrame:
    """
    Copy pair-training features onto inspection rows.

    ``_build_high_band_inspection_dataframe`` only forwards a reduced feature subset;
    body/path/sender columns live on ``df_eval`` and must be merged explicitly.
    """
    if df_ins.empty or df_eval.empty:
        return df_ins
    out = df_ins.copy()
    cols = [c for c in _core_explanatory_feature_cols() if c in df_eval.columns]
    if len(out) == len(df_eval):
        for c in cols:
            out[c] = _num_series(df_eval, c).to_numpy()
        return out
    eval_key = df_eval["email_i"].astype(str) + "\0" + df_eval["email_j"].astype(str)
    ins_key = out["email_i"].astype(str) + "\0" + out["email_j"].astype(str)
    feat = df_eval.copy()
    feat["_pair_key"] = eval_key.to_numpy()
    for c in cols:
        sub = feat[["_pair_key", c]].drop_duplicates("_pair_key", keep="first")
        mapped = ins_key.map(sub.set_index("_pair_key")[c])
        out[c] = pd.to_numeric(mapped, errors="coerce").to_numpy()
    return out


def _backfill_body_features_from_text(df: pd.DataFrame) -> pd.DataFrame:
    """Use MISP-derived body_only_* when pair-table body columns are absent."""
    out = df.copy()
    for dst, src in (
        ("body_token_jaccard", "body_only_token_jaccard"),
        ("body_char4gram_jaccard", "body_only_char4gram_jaccard"),
    ):
        if src not in out.columns:
            continue
        src_s = _num_series(out, src)
        if dst not in out.columns:
            out[dst] = src_s
        else:
            dst_s = _num_series(out, dst)
            out[dst] = dst_s.where(dst_s.notna(), src_s)
    return out


def _enrich_path_features_from_nodes(
    df: pd.DataFrame,
    nodes_by_email: dict[str, dict[str, set[str]]],
) -> pd.DataFrame:
    """Compute path Jaccard from anchor url_set/stem_set (not on pair_training_dataset)."""
    from seed_candidate_workflow.utils.pair_similarity_features import (
        attach_path_jaccard_features_to_dataframe,
    )

    if df.empty:
        return df
    nodes = {str(k): v for k, v in (nodes_by_email or {}).items()}
    return attach_path_jaccard_features_to_dataframe(df, nodes_by_email=nodes, prefer_existing=True)


def _cohort_feature_stats(df: pd.DataFrame, mask: np.ndarray, cols: tuple[str, ...]) -> dict[str, Any]:
    sub = df.loc[mask] if mask is not None else df
    out: dict[str, Any] = {}
    for col in cols:
        if col not in sub.columns:
            out[col] = {"mean": None, "median": None, "n_non_null": 0, "n_rows": int(len(sub))}
        else:
            stats = _pse()._safe_float_stats(_num_series(sub, col))
            stats["n_rows"] = int(len(sub))
            out[col] = stats
    return out


def _build_path_feature_population_diagnostics(
    *,
    df_same: pd.DataFrame,
    rescued_mask: np.ndarray,
    collapsed_mask: np.ndarray,
    frontier_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    expected = list(PATH_FEATURE_COLS)
    found = [c for c in expected if c in df_same.columns]
    missing = [c for c in expected if c not in df_same.columns]

    def _nn(col: str, mask: np.ndarray) -> dict[str, int]:
        if col not in df_same.columns:
            return {"n_non_null": 0, "n_rows": int(mask.sum())}
        s = _num_series(df_same.loc[mask], col)
        return {"n_non_null": int(s.notna().sum()), "n_rows": int(mask.sum())}

    cohorts: dict[str, Any] = {
        "all_same_unlabeled": {c: _nn(c, np.ones(len(df_same), dtype=bool)) for c in expected},
        "rescued_same_unlabeled": {c: _nn(c, rescued_mask) for c in expected},
        "collapsed_same_unlabeled": {c: _nn(c, collapsed_mask) for c in expected},
    }
    if frontier_mask is not None:
        cohorts["html_fp_frontier_band"] = {c: _nn(c, frontier_mask) for c in expected}

    rescued_means = {
        c: (_cohort_feature_stats(df_same, rescued_mask, (c,)).get(c) or {}).get("mean")
        for c in found
    }
    collapsed_means = {
        c: (_cohort_feature_stats(df_same, collapsed_mask, (c,)).get(c) or {}).get("mean")
        for c in found
    }
    mean_comparison = {
        c: {
            "rescued_mean": rescued_means.get(c),
            "collapsed_mean": collapsed_means.get(c),
            "difference_rescued_minus_collapsed": (
                (rescued_means[c] - collapsed_means[c])
                if rescued_means.get(c) is not None and collapsed_means.get(c) is not None
                else None
            ),
        }
        for c in found
    }

    return {
        "expected_path_feature_columns": expected,
        "found_path_feature_columns": found,
        "missing_path_feature_columns": missing,
        "non_null_counts_by_cohort": cohorts,
        "mean_comparison_rescued_vs_collapsed": mean_comparison,
    }


def _build_body_vs_body_only_comparison(
    *,
    rescued_df: pd.DataFrame,
    collapsed_df: pd.DataFrame,
    cross_low_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Compare raw body vs URL-stripped body-only similarity across cohorts."""
    cols = list(BODY_COMPARISON_COLS)
    rescued_stats = _cohort_feature_stats(rescued_df, np.ones(len(rescued_df), dtype=bool), tuple(cols))
    collapsed_stats = _cohort_feature_stats(
        collapsed_df, np.ones(len(collapsed_df), dtype=bool), tuple(cols)
    )
    notes: list[str] = []
    for raw, only in (
        ("body_token_jaccard", "body_only_token_jaccard"),
        ("body_char4gram_jaccard", "body_only_char4gram_jaccard"),
    ):
        r_raw = (rescued_stats.get(raw) or {}).get("mean")
        r_only = (rescued_stats.get(only) or {}).get("mean")
        c_raw = (collapsed_stats.get(raw) or {}).get("mean")
        c_only = (collapsed_stats.get(only) or {}).get("mean")
        if all(v is not None for v in (r_raw, r_only, c_raw, c_only)):
            rescued_lift_only = float(r_only) - float(r_raw)
            collapsed_lift_only = float(c_only) - float(c_raw)
            if rescued_lift_only > collapsed_lift_only + 0.03:
                notes.append(
                    f"Rescued pairs show stronger body-only than raw `{raw}` lift "
                    f"(+{rescued_lift_only:.3f} vs +{collapsed_lift_only:.3f} for collapsed)."
                )
            elif float(c_raw) > float(c_only) + 0.05 and float(r_only) <= float(r_raw):
                notes.append(
                    f"Collapsed pairs have higher raw `{raw}` than body-only; rescued pairs do not — "
                    "raw body overlap may be noisy for collapse."
                )
    if not notes:
        notes.append(
            "Inspect rescued vs collapsed means for raw body_* vs body_only_* columns; "
            "body-only similarity often tracks rescue better when URLs dominate raw body text."
        )
    out: dict[str, Any] = {
        "feature_columns": cols,
        "rescued_same_unlabeled": rescued_stats,
        "collapsed_same_unlabeled": collapsed_stats,
        "interpretation_notes": notes[:8],
    }
    if cross_low_df is not None and not cross_low_df.empty:
        out["cross_campaign_low_unlabeled"] = _cohort_feature_stats(
            cross_low_df, np.ones(len(cross_low_df), dtype=bool), tuple(cols)
        )
    return out


def _build_single_source_frontier_summary(
    df_same: pd.DataFrame,
    collapsed_mask: np.ndarray,
) -> dict[str, Any]:
    """Collapsed same-unlabeled pairs with single-source / weak 2-hop support."""
    collapsed = df_same.loc[collapsed_mask]
    if collapsed.empty:
        return {"n_collapsed_same_unlabeled": 0, "n_single_source": 0}

    sc = _num_series(collapsed, "source_count")
    single = collapsed.loc[sc.eq(1).fillna(False)]
    n_single = int(len(single))
    n_coll = int(len(collapsed))

    def _top_counts(series: pd.Series, n: int = 8) -> dict[str, int]:
        if series.empty:
            return {}
        return series.astype(str).value_counts().head(n).to_dict()

    prov_top = _top_counts(single.get("provenance_combo", pd.Series(dtype=object)))
    ch_cols = [c for c in single.columns if c.startswith("twohop_via_")]
    channel_fracs: dict[str, float] = {}
    for c in ch_cols:
        channel_fracs[c] = float(_bool_series(single, c).mean()) if n_single else 0.0

    if n_single:
        body_path_inside = _cohort_feature_stats(
            single,
            np.ones(n_single, dtype=bool),
            (
                "body_token_jaccard",
                "body_only_token_jaccard",
                "body_char4gram_jaccard",
                "body_only_char4gram_jaccard",
                *PATH_FEATURE_COLS,
            ),
        )
    else:
        body_path_inside = {}

    no_sem = _bool_col_array(single, "from_semantic") if n_single else np.array([], dtype=bool)
    weak_2hop = _bool_col_array(single, "from_2hop") if n_single else np.array([], dtype=bool)
    html_fp = (
        _bool_series(single, "has_shared_html_fp") | _bool_series(single, "twohop_via_html_fp")
        if n_single
        else pd.Series(dtype=bool)
    )

    return {
        "definition": "collapsed same-campaign unlabeled with source_count == 1",
        "n_collapsed_same_unlabeled": n_coll,
        "n_single_source_collapsed": n_single,
        "fraction_single_source_among_collapsed": float(n_single / n_coll) if n_coll else None,
        "fraction_without_from_semantic": float((~no_sem).mean()) if len(no_sem) else None,
        "fraction_from_2hop": float(weak_2hop.mean()) if len(weak_2hop) else None,
        "fraction_html_fp_signal": float(html_fp.mean()) if n_single else None,
        "top_provenance_combos": prov_top,
        "twohop_channel_fractions": dict(
            sorted(channel_fracs.items(), key=lambda kv: kv[1], reverse=True)[:10]
        ),
        "feature_summaries_inside_single_source": body_path_inside,
    }


def _bool_col_array(df: pd.DataFrame, col: str) -> np.ndarray:
    return _bool_series(df, col).to_numpy()


def _build_feature_population_diagnostics(
    *,
    df_same: pd.DataFrame,
    rescued_mask: np.ndarray,
    collapsed_mask: np.ndarray,
    frontier_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    expected = list(_core_explanatory_feature_cols())
    found = [c for c in expected if c in df_same.columns]
    missing = [c for c in expected if c not in df_same.columns]
    excluded_present = sorted(
        c for c in df_same.columns if _is_ranking_excluded_column(str(c)) and c in df_same.columns
    )

    def _nn(col: str, mask: np.ndarray | None) -> dict[str, int]:
        if col not in df_same.columns:
            return {"n_non_null": 0, "n_rows": int(mask.sum()) if mask is not None else len(df_same)}
        sub = df_same.loc[mask] if mask is not None else df_same
        s = _num_series(sub, col)
        return {"n_non_null": int(s.notna().sum()), "n_rows": int(len(sub))}

    key_features = [
        "body_token_jaccard",
        "body_char4gram_jaccard",
        "body_only_token_jaccard",
        "body_only_char4gram_jaccard",
        "semantic_cosine_max",
        "sender_localpart_norm_jaccard",
        *PATH_FEATURE_COLS,
        "source_count",
    ]
    cohorts: dict[str, Any] = {
        "all_same_unlabeled": {},
        "rescued_same_unlabeled": {},
        "collapsed_same_unlabeled": {},
    }
    if frontier_mask is not None:
        cohorts["html_fp_frontier_band"] = {}
    for label, mask in (
        ("all_same_unlabeled", np.ones(len(df_same), dtype=bool)),
        ("rescued_same_unlabeled", rescued_mask),
        ("collapsed_same_unlabeled", collapsed_mask),
    ):
        if label == "all_same_unlabeled":
            m = mask
        else:
            m = mask
        cohorts[label] = {c: _nn(c, m) for c in key_features}
    if frontier_mask is not None:
        cohorts["html_fp_frontier_band"] = {c: _nn(c, frontier_mask) for c in key_features}

    return {
        "expected_feature_columns": expected,
        "found_feature_columns": found,
        "missing_feature_columns": missing,
        "excluded_from_ranking_present_in_frame": excluded_present,
        "non_null_counts_by_cohort": cohorts,
    }


def assign_same_unlabeled_buckets(
    scores: np.ndarray,
    *,
    collapsed_max: float = 0.10,
    rescued_min: float = 0.80,
) -> np.ndarray:
    """Return object array: collapsed_same_unlabeled | mid_same_unlabeled | rescued_same_unlabeled | other."""
    s = pd.Series(pd.to_numeric(np.asarray(scores, dtype=float), errors="coerce"))
    out = np.array(["other"] * len(s), dtype=object)
    finite = s.notna().to_numpy()
    sv = s.to_numpy(dtype=float)
    out[finite & (sv <= float(collapsed_max))] = "collapsed_same_unlabeled"
    out[finite & (sv >= float(rescued_min))] = "rescued_same_unlabeled"
    mid = finite & (sv > float(collapsed_max)) & (sv < float(rescued_min))
    out[mid] = "mid_same_unlabeled"
    return out


def _compare_rescued_collapsed_fraction(*, rescued_v: Any, collapsed_v: Any) -> dict[str, Any]:
    r = float(rescued_v) if rescued_v is not None and pd.notna(rescued_v) else None
    c = float(collapsed_v) if collapsed_v is not None and pd.notna(collapsed_v) else None
    diff = (r - c) if (r is not None and c is not None) else None
    enrich = (r / c) if (r is not None and c is not None and c > 0) else None
    return {
        "rescued_fraction": r,
        "collapsed_fraction": c,
        "difference_rescued_minus_collapsed": diff,
        "abs_difference": abs(diff) if diff is not None else None,
        "enrichment_rescued_over_collapsed": enrich,
        "favors": (
            "rescued"
            if diff is not None and diff > 0
            else "collapsed"
            if diff is not None and diff < 0
            else "tie"
        ),
    }


def _summarize_bucket_group(
    gdf: pd.DataFrame,
    *,
    n_total_same_unlabeled: int,
    nodes_by_email: dict[str, dict[str, set[str]]],
) -> dict[str, Any]:
    base = _pse()._summarize_group(
        gdf=gdf, n_total_eval=n_total_same_unlabeled, nodes_by_email=nodes_by_email
    )
    n = int(len(gdf))
    prov_cols = _provenance_cols_in_df(gdf) if n else []
    extra_prov: dict[str, Any] = {}
    for k in prov_cols:
        if k in base.get("provenance", {}):
            continue
        if k in gdf.columns:
            v = int(gdf[k].fillna(False).astype(bool).sum())
            extra_prov[k] = {"count": v, "fraction": float(v / n) if n else None}
    if extra_prov:
        base.setdefault("provenance", {}).update(extra_prov)

    feat_cols = _feature_cols_in_df(gdf) if n else []
    for c in feat_cols:
        if c not in (base.get("feature_summaries") or {}):
            base.setdefault("feature_summaries", {})[c] = _pse()._safe_float_stats(gdf[c])

    if n and "has_shared_html_fp" in gdf.columns:
        base["html_fp"] = {
            "fraction_has_shared_html_fp": float(gdf["has_shared_html_fp"].fillna(False).astype(bool).mean()),
            "fraction_twohop_via_html_fp": float(
                gdf["twohop_via_html_fp"].fillna(False).astype(bool).mean()
                if "twohop_via_html_fp" in gdf.columns
                else 0.0
            ),
        }
    return base


def _marginal_comparison_table(
    *,
    gt_path: Path,
    rescued_df: pd.DataFrame,
    collapsed_df: pd.DataFrame,
    mid_df: pd.DataFrame,
    nodes_by_email: dict[str, dict[str, set[str]]],
    n_same_unlabeled: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rescued_s = _summarize_bucket_group(
        rescued_df, n_total_same_unlabeled=n_same_unlabeled, nodes_by_email=nodes_by_email
    )
    collapsed_s = _summarize_bucket_group(
        collapsed_df, n_total_same_unlabeled=n_same_unlabeled, nodes_by_email=nodes_by_email
    )
    mid_s = _summarize_bucket_group(
        mid_df, n_total_same_unlabeled=n_same_unlabeled, nodes_by_email=nodes_by_email
    )

    rows: list[dict[str, Any]] = []

    def _add_rows(metric_group: str, metric_name: str, rv: Any, cv: Any) -> None:
        cmp = _compare_rescued_collapsed_fraction(rescued_v=rv, collapsed_v=cv)
        rows.append(
            {
                "gt_path": str(gt_path.resolve()),
                "metric_group": metric_group,
                "metric_name": metric_name,
                "rescued_value": cmp["rescued_fraction"],
                "collapsed_value": cmp["collapsed_fraction"],
                "difference_rescued_minus_collapsed": cmp["difference_rescued_minus_collapsed"],
                "enrichment_rescued_over_collapsed": cmp["enrichment_rescued_over_collapsed"],
                "abs_difference": cmp["abs_difference"],
                "favors": cmp["favors"],
            }
        )

    for k, rv in (rescued_s.get("provenance") or {}).items():
        cv = ((collapsed_s.get("provenance") or {}).get(k) or {}).get("fraction")
        _add_rows("provenance", k, (rv or {}).get("fraction"), cv)

    for k in _feature_cols_in_df(rescued_df if not rescued_df.empty else collapsed_df):
        rfeat = (rescued_s.get("feature_summaries") or {}).get(k) or {}
        cfeat = (collapsed_s.get("feature_summaries") or {}).get(k) or {}
        _add_rows("feature_mean", k, rfeat.get("mean"), cfeat.get("mean"))

    for k, rv in (rescued_s.get("shared_evidence") or {}).items():
        if k.startswith("n_"):
            continue
        cv = ((collapsed_s.get("shared_evidence") or {}).get(k) or {}).get("fraction_edges_with_at_least_1")
        _add_rows("shared_evidence", k, (rv or {}).get("fraction_edges_with_at_least_1"), cv)

    ranked = [r for r in rows if r.get("abs_difference") is not None]
    ranked.sort(key=lambda r: float(r["abs_difference"]), reverse=True)
    ranked = _filter_ranked_separator_rows(ranked)

    summary = {
        "gt_path": str(gt_path.resolve()),
        "counts": {
            "n_same_campaign_unlabeled_total": int(n_same_unlabeled),
            "n_rescued_same_unlabeled": int(len(rescued_df)),
            "n_collapsed_same_unlabeled": int(len(collapsed_df)),
            "n_mid_same_unlabeled": int(len(mid_df)),
        },
        "rescued_group_summary": rescued_s,
        "collapsed_group_summary": collapsed_s,
        "mid_group_summary": mid_s,
        "ranked_marginal_separators_top20": ranked[:20],
        "ranked_marginal_separators_favoring_rescued_top10": [
            r for r in ranked if r.get("favors") == "rescued"
        ][:10],
        "ranked_marginal_separators_favoring_collapsed_top10": [
            r for r in ranked if r.get("favors") == "collapsed"
        ][:10],
    }
    return summary, rows


def _build_joint_separators_rescued_vs_collapsed(
    *,
    gt_path: Path,
    df_same: pd.DataFrame,
    rescued_mask: np.ndarray,
    collapsed_mask: np.ndarray,
    nodes_by_email: dict[str, dict[str, set[str]]],
    evidence_index: dict[tuple[str, str], list[dict[str, Any]]] | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Joint rules: fraction of rescued vs collapsed same-unlabeled pairs matching each condition."""
    out_rows: list[dict[str, Any]] = []
    n_rescued = int(rescued_mask.sum())
    n_collapsed = int(collapsed_mask.sum())
    if len(df_same) == 0:
        empty = {
            "gt_path": str(gt_path.resolve()),
            "focus": "rescued_vs_collapsed_same_campaign_unlabeled",
            "counts": {"n_rescued": 0, "n_collapsed": 0},
            "ranked_joint_separators_top15": [],
        }
        return empty, out_rows

    df = df_same.copy()
    n = len(df)

    def _bool_col(name: str) -> np.ndarray:
        if name not in df.columns:
            return np.zeros(n, dtype=bool)
        return df[name].fillna(False).astype(bool).to_numpy()

    fs = _bool_col("from_semantic")
    f2 = _bool_col("from_2hop")
    fcomp = _bool_col("from_component")
    fra = _bool_col("from_rare_artifact")
    def _num_col(name: str) -> pd.Series:
        if name not in df.columns:
            return pd.Series(np.nan, index=df.index, dtype=float)
        return pd.to_numeric(df[name], errors="coerce")

    sem = _num_col("semantic_cosine_max")
    body_tok = _num_col("body_token_jaccard")
    body_c4 = _num_col("body_char4gram_jaccard")
    body_only_tok = _num_col("body_only_token_jaccard")
    body_only_c4 = _num_col("body_only_char4gram_jaccard")
    path_comb = _num_col("path_token_jaccard_combined")
    url_path = _num_col("url_path_token_jaccard")
    subj_tok = _num_col("subject_token_jaccard")

    if "has_shared_html_fp" in df.columns:
        has_shared_html_fp = df["has_shared_html_fp"].fillna(False).astype(bool).to_numpy()
    else:
        has_shared_html_fp = np.zeros(n, dtype=bool)

    n_core = np.zeros(n, dtype=np.int32)
    for _ac, short in _pse()._PAIR_SHARED_CHANNEL_DEFS:
        col = f"has_shared_{short}"
        if col in df.columns:
            n_core += df[col].fillna(False).astype(bool).astype(np.int32).to_numpy()

    bool_terms: dict[str, np.ndarray] = {
        "from_semantic": fs,
        "from_2hop": f2,
        "from_component": fcomp,
        "from_rare_artifact": fra,
        "shared_html_fp": has_shared_html_fp,
        "twohop_via_html_fp": _bool_col("twohop_via_html_fp"),
        "semantic_ge_0_80": sem.ge(0.80).fillna(False).to_numpy(),
        "semantic_0_75_to_0_85": (sem.ge(0.75) & sem.lt(0.85)).fillna(False).to_numpy(),
        "body_token_jaccard_ge_0_25": body_tok.ge(0.25).fillna(False).to_numpy(),
        "body_char4gram_jaccard_ge_0_25": body_c4.ge(0.25).fillna(False).to_numpy(),
        "body_high_semantic_mid": (
            (body_tok.ge(0.25) | body_c4.ge(0.25)) & sem.ge(0.75) & sem.lt(0.90)
        )
        .fillna(False)
        .to_numpy(),
        "body_only_token_jaccard_ge_0_25": body_only_tok.ge(0.25).fillna(False).to_numpy(),
        "body_only_char4gram_jaccard_ge_0_25": body_only_c4.ge(0.25).fillna(False).to_numpy(),
        "body_only_high_raw_low": (
            body_only_tok.ge(0.25).fillna(False) & body_tok.lt(0.15).fillna(True)
        ).to_numpy(),
        "path_token_jaccard_combined_ge_0_25": path_comb.ge(0.25).fillna(False).to_numpy(),
        "url_path_token_jaccard_ge_0_25": url_path.ge(0.25).fillna(False).to_numpy(),
        "subject_low_body_high": (subj_tok.lt(0.20) & body_tok.ge(0.25)).fillna(False).to_numpy(),
        "subject_low_body_only_high": (subj_tok.lt(0.20) & body_only_tok.ge(0.25)).fillna(False).to_numpy(),
        "n_shared_core_channels_ge_1": n_core >= 1,
        "n_shared_core_channels_ge_2": n_core >= 2,
    }
    sc = _num_col("source_count")
    bool_terms["source_count_eq_1"] = sc.eq(1).fillna(False).to_numpy(dtype=bool)
    bool_terms["source_count_ge_2"] = sc.ge(2).fillna(False).to_numpy(dtype=bool)

    for col in _provenance_cols_in_df(df):
        if col.startswith("from_") and col not in bool_terms:
            bool_terms[col] = df[col].fillna(False).astype(bool).to_numpy()

    bool_terms = extend_bool_terms_for_low_band_channels(
        bool_terms,
        df,
        nodes_by_email=nodes_by_email,
        evidence_index=evidence_index,
    )
    extra_rules = (
        "twohop_via_html_fp_AND_semantic_0_75_to_0_85",
        "twohop_via_html_fp_AND_NOT_semantic_ge_0_80",
        "shared_html_fp_AND_semantic_0_75_to_0_85",
        "from_2hop_AND_source_count_eq_1",
        "from_2hop_AND_twohop_via_html_fp",
        "body_token_jaccard_ge_0_25_AND_NOT_from_semantic",
        "body_char4gram_jaccard_ge_0_25_AND_semantic_0_75_to_0_85",
        "from_semantic_AND_shared_html_fp",
        "from_semantic_AND_NOT_shared_html_fp",
        "source_count_eq_1_AND_from_2hop_AND_NOT_from_semantic",
        "path_token_jaccard_combined_ge_0_25_AND_body_only_token_jaccard_ge_0_25",
        "body_only_token_jaccard_ge_0_25_AND_NOT_from_semantic",
    )

    def _eval_rule(expr: str) -> np.ndarray:
        toks = expr.split("_AND_")
        out_m = np.ones(n, dtype=bool)
        for tok in toks:
            neg = tok.startswith("NOT_")
            key = tok[4:] if neg else tok
            base = bool_terms.get(key)
            if base is None:
                return np.zeros(n, dtype=bool)
            out_m &= ~base if neg else base
        return out_m

    rule_names = list(BINARY_CONDITION_RULES_DEFAULT) + list(extra_rules) + list(
        low_band_twohop_joint_rule_names()
    )
    seen: set[str] = set()
    unique_rules: list[str] = []
    for r in rule_names:
        if r not in seen:
            seen.add(r)
            unique_rules.append(r)

    bin_out: dict[str, Any] = {}
    for name in unique_rules:
        cond = _eval_rule(name)
        cmp = _pse()._cmp_from_masks(
            cond_same=cond,
            base_same=rescued_mask,
            cond_cross=cond,
            base_cross=collapsed_mask,
            value_key_same="rescued_fraction",
            value_key_cross="collapsed_fraction",
        )
        bin_out[name] = cmp
        favors = (
            "rescued"
            if cmp.get("difference_same_minus_cross") is not None and float(cmp["difference_same_minus_cross"]) > 0
            else "collapsed"
            if cmp.get("difference_same_minus_cross") is not None and float(cmp["difference_same_minus_cross"]) < 0
            else "tie"
        )
        out_rows.append(
            {
                "gt_path": str(gt_path.resolve()),
                "analysis_section": "binary_joint_comparisons",
                "condition_name": name,
                "rescued_fraction": cmp.get("rescued_fraction"),
                "collapsed_fraction": cmp.get("collapsed_fraction"),
                "difference_rescued_minus_collapsed": cmp.get("difference_same_minus_cross"),
                "abs_difference": cmp.get("abs_difference"),
                "enrichment_rescued_over_collapsed": cmp.get("enrichment_same_over_cross"),
                "favors": favors,
            }
        )

    sem_out: dict[str, Any] = {}
    for bname, lo, hi in SEMANTIC_BUCKET_RULES_DEFAULT:
        mask = np.ones(n, dtype=bool)
        if lo is not None:
            mask &= sem.ge(float(lo)).fillna(False).to_numpy()
        if hi is not None:
            mask &= sem.lt(float(hi)).fillna(False).to_numpy()
        cmp = _pse()._cmp_from_masks(
            cond_same=mask,
            base_same=rescued_mask,
            cond_cross=mask,
            base_cross=collapsed_mask,
            value_key_same="rescued_fraction",
            value_key_cross="collapsed_fraction",
        )
        sem_out[bname] = cmp
        out_rows.append(
            {
                "gt_path": str(gt_path.resolve()),
                "analysis_section": "semantic_bucket_analysis",
                "condition_name": bname,
                **{k: cmp.get(k) for k in ("rescued_fraction", "collapsed_fraction")},
                "difference_rescued_minus_collapsed": cmp.get("difference_same_minus_cross"),
                "abs_difference": cmp.get("abs_difference"),
                "favors": (
                    "rescued"
                    if cmp.get("difference_same_minus_cross") is not None
                    and float(cmp["difference_same_minus_cross"]) > 0
                    else "collapsed"
                ),
            }
        )
        for suffix, extra in (
            ("_AND_twohop_via_html_fp", mask & bool_terms["twohop_via_html_fp"]),
            ("_AND_shared_html_fp", mask & has_shared_html_fp),
            ("_AND_from_2hop", mask & f2),
            ("_AND_source_count_eq_1", mask & bool_terms["source_count_eq_1"]),
        ):
            cname = f"{bname}{suffix}"
            cmp2 = _pse()._cmp_from_masks(
                cond_same=extra,
                base_same=rescued_mask,
                cond_cross=extra,
                base_cross=collapsed_mask,
                value_key_same="rescued_fraction",
                value_key_cross="collapsed_fraction",
            )
            sem_out[cname] = cmp2
            out_rows.append(
                {
                    "gt_path": str(gt_path.resolve()),
                    "analysis_section": "semantic_bucket_analysis",
                    "condition_name": cname,
                    "rescued_fraction": cmp2.get("rescued_fraction"),
                    "collapsed_fraction": cmp2.get("collapsed_fraction"),
                    "difference_rescued_minus_collapsed": cmp2.get("difference_same_minus_cross"),
                    "abs_difference": cmp2.get("abs_difference"),
                    "favors": (
                        "rescued"
                        if cmp2.get("difference_same_minus_cross") is not None
                        and float(cmp2["difference_same_minus_cross"]) > 0
                        else "collapsed"
                    ),
                }
            )

    ranked = [r for r in out_rows if r.get("abs_difference") is not None]
    ranked.sort(key=lambda r: float(r["abs_difference"]), reverse=True)
    ranked = _filter_ranked_separator_rows(ranked)

    payload = {
        "gt_path": str(gt_path.resolve()),
        "focus": "rescued_vs_collapsed_same_campaign_unlabeled",
        "counts": {"n_rescued": n_rescued, "n_collapsed": n_collapsed, "n_same_unlabeled": int(n)},
        "binary_joint_comparisons": bin_out,
        "semantic_bucket_analysis": sem_out,
        "ranked_joint_separators_top15": ranked[:15],
        "ranked_joint_separators_favoring_rescued_top10": [r for r in ranked if r.get("favors") == "rescued"][:10],
        "ranked_joint_separators_favoring_collapsed_top10": [
            r for r in ranked if r.get("favors") == "collapsed"
        ][:10],
    }
    return payload, out_rows


def _analyze_html_fp_frontier(
    *,
    df_same: pd.DataFrame,
    rescued_mask: np.ndarray,
    collapsed_mask: np.ndarray,
    cross_unlabeled_mask: np.ndarray | None,
    df_eval_cross: pd.DataFrame | None,
    nodes_by_email: dict[str, dict[str, set[str]]] | None = None,
) -> dict[str, Any]:
    sem = pd.to_numeric(
        df_same["semantic_cosine_max"] if "semantic_cosine_max" in df_same.columns else np.nan,
        errors="coerce",
    )
    html_fp = _bool_series(df_same, "has_shared_html_fp")
    twohop_hf = _bool_series(df_same, "twohop_via_html_fp")
    frontier = html_fp & sem.ge(HTML_FP_SEMANTIC_LO) & sem.lt(HTML_FP_SEMANTIC_HI)

    def _cohort_stats(mask: np.ndarray, label: str, *, df: pd.DataFrame | None = None) -> dict[str, Any]:
        frame = df_same if df is None else df
        sub = frame.loc[mask]
        if sub.empty:
            return {"cohort": label, "n_pairs": 0}
        sc = pd.to_numeric(sub["score"], errors="coerce")
        feat_stats = {
            col: _pse()._safe_float_stats(_num_series(sub, col)) for col in HTML_FP_COHORT_FEATURE_COLS
        }
        return {
            "cohort": label,
            "n_pairs": int(len(sub)),
            "n_html_fp_or_twohop": int(
                (_bool_series(sub, "has_shared_html_fp") | _bool_series(sub, "twohop_via_html_fp")).sum()
            ),
            "fraction_collapsed_score_le_0_10": float((sc <= 0.10).mean()),
            "fraction_rescued_score_ge_0_80": float((sc >= 0.80).mean()),
            "feature_summaries": feat_stats,
            "fraction_source_count_eq_1": float(_num_series(sub, "source_count").eq(1).fillna(False).mean()),
            "fraction_from_2hop": float(_bool_series(sub, "from_2hop").mean()),
            "fraction_twohop_via_html_fp": float(_bool_series(sub, "twohop_via_html_fp").mean()),
        }

    out: dict[str, Any] = {
        "definition": {
            "html_fp_frontier_semantic_band": [HTML_FP_SEMANTIC_LO, HTML_FP_SEMANTIC_HI],
            "collapsed_score_max": 0.10,
            "rescued_score_min": 0.80,
        },
        "counts": {
            "n_same_unlabeled": int(len(df_same)),
            "n_shared_html_fp": int(html_fp.sum()),
            "n_twohop_via_html_fp": int(twohop_hf.sum()),
            "n_html_fp_frontier_band": int(frontier.sum()),
            "n_collapsed_html_fp_frontier": int((frontier & collapsed_mask).sum()),
            "n_rescued_html_fp_frontier": int((frontier & rescued_mask).sum()),
        },
        "cohorts": [
            _cohort_stats(collapsed_mask & (html_fp | twohop_hf), "collapsed_same_with_html_fp_signal"),
            _cohort_stats(rescued_mask & (html_fp | twohop_hf), "rescued_same_with_html_fp_signal"),
            _cohort_stats(frontier & collapsed_mask, "collapsed_same_html_fp_frontier"),
            _cohort_stats(frontier & rescued_mask, "rescued_same_html_fp_frontier"),
            _cohort_stats(frontier & ~(collapsed_mask | rescued_mask), "mid_same_html_fp_frontier"),
        ],
    }
    if df_eval_cross is not None and not df_eval_cross.empty and nodes_by_email:
        cross = df_eval_cross.copy()
        if not cross.empty:
            cross["has_shared_html_fp"] = False
            for i, r in cross.iterrows():
                detail = _pse()._pair_shared_evidence_detail(
                    str(r["email_i"]), str(r["email_j"]), nodes_by_email
                )
                cross.at[i, "has_shared_html_fp"] = bool(detail.get("has_shared_html_fp"))
            cross_html = cross["has_shared_html_fp"].fillna(False).astype(bool).to_numpy()
            out["cohorts"].append(
                _cohort_stats(cross_html, "cross_campaign_unlabeled_shared_html_fp", df=cross)
            )
    return out


def _compute_text_similarity_columns(
    df: pd.DataFrame,
    email_text_by_eid: dict[str, dict[str, str]],
) -> pd.DataFrame:
    from seed_candidate_workflow.utils.pair_similarity_features import (
        body_only_char4gram_jaccard_from_bodies,
        body_only_token_jaccard_from_bodies,
        char_ngrams_text,
        jaccard_similarity,
        tokenize_text,
    )

    if df.empty or not email_text_by_eid:
        return df
    out = df.copy()
    subj_tok: list[float | None] = []
    subj_c4: list[float | None] = []
    body_only_tok: list[float | None] = []
    body_only_c4: list[float | None] = []

    for _, r in out.iterrows():
        ei, ej = str(r["email_i"]), str(r["email_j"])
        ti = email_text_by_eid.get(ei) or {}
        tj = email_text_by_eid.get(ej) or {}
        si, sj = str(ti.get("subject") or ""), str(tj.get("subject") or "")
        bi, bj = str(ti.get("body") or ""), str(tj.get("body") or "")
        ts_i = tokenize_text(si, min_len=2)
        ts_j = tokenize_text(sj, min_len=2)
        subj_tok.append(jaccard_similarity(ts_i, ts_j))
        subj_c4.append(
            jaccard_similarity(char_ngrams_text(si, 4), char_ngrams_text(sj, 4))
        )
        body_only_tok.append(body_only_token_jaccard_from_bodies(bi, bj))
        body_only_c4.append(body_only_char4gram_jaccard_from_bodies(bi, bj))

    out["subject_token_jaccard"] = subj_tok
    out["subject_char4gram_jaccard"] = subj_c4
    out["body_only_token_jaccard"] = body_only_tok
    out["body_only_char4gram_jaccard"] = body_only_c4
    return out


def _attach_inspection_badges(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    badges: list[str] = []
    for _, r in out.iterrows():
        flags: list[str] = []
        bucket = str(r.get("same_unlabeled_bucket") or "")
        if bucket == "collapsed_same_unlabeled":
            flags.append("same-campaign collapsed")
        elif bucket == "rescued_same_unlabeled":
            flags.append("same-campaign rescued")
        sem = pd.to_numeric(r.get("semantic_cosine_for_display", r.get("semantic_cosine_max")), errors="coerce")
        html_fp = bool(r.get("has_shared_html_fp")) or bool(r.get("twohop_via_html_fp"))
        if html_fp and pd.notna(sem) and HTML_FP_SEMANTIC_LO <= float(sem) < HTML_FP_SEMANTIC_HI:
            flags.append("html_fp frontier")
        if bool(r.get("from_2hop")) and pd.to_numeric(r.get("source_count"), errors="coerce") == 1:
            flags.append("single-source 2hop")
        body_t = pd.to_numeric(r.get("body_token_jaccard"), errors="coerce")
        if pd.notna(sem) and pd.notna(body_t) and float(body_t) >= 0.25 and 0.75 <= float(sem) < 0.90:
            flags.append("body-sim high / semantic mid")
        if pd.notna(body_t) and body_t >= 0.25:
            subj_t = pd.to_numeric(r.get("subject_token_jaccard"), errors="coerce")
            if pd.notna(subj_t) and subj_t < 0.15:
                flags.append("body-high subject-low")
        badges.append("|".join(flags))
    out["inspection_badges"] = badges
    return out


def _classify_rescued_collapsed_regime(row: pd.Series) -> str:
    return str(row.get("same_unlabeled_bucket") or "same_campaign_unlabeled")


def _rescued_collapsed_review_prompt(row: pd.Series) -> str:
    b = str(row.get("same_unlabeled_bucket") or "")
    if b == "rescued_same_unlabeled":
        return "Rescued same-campaign unlabeled pair (high score). What evidence did the model use successfully?"
    if b == "collapsed_same_unlabeled":
        return "Collapsed same-campaign unlabeled pair (very low score). What signal is missing or overridden?"
    return "Mid-band same-campaign unlabeled — borderline model behavior."


def _generate_rescued_vs_collapsed_recommendations(
    *,
    marginal: dict[str, Any],
    joint: dict[str, Any],
    html_fp: dict[str, Any],
) -> dict[str, Any]:
    signals_used: list[str] = []
    signals_missing: list[str] = []
    interventions: list[str] = []

    for r in marginal.get("ranked_marginal_separators_favoring_rescued_top10") or []:
        mg, mn = r.get("metric_group"), r.get("metric_name")
        if _is_ranking_excluded_column(str(mn or "")):
            continue
        if mg == "provenance" and r.get("rescued_value", 0) > 0.3:
            signals_used.append(f"Provenance `{mn}` is common among rescued pairs.")
        if mg == "feature_mean" and r.get("difference_rescued_minus_collapsed", 0) > 0.05:
            signals_used.append(f"Higher `{mn}` among rescued vs collapsed same-unlabeled pairs.")

    for r in marginal.get("ranked_marginal_separators_favoring_collapsed_top10") or []:
        mg, mn = r.get("metric_group"), r.get("metric_name")
        if _is_ranking_excluded_column(str(mn or "")):
            continue
        if mg == "provenance" and r.get("collapsed_value", 0) > 0.3:
            signals_missing.append(f"Collapsed pairs often have `{mn}` without sufficient lift.")
        if "twohop_via_html_fp" in str(mn) or "html_fp" in str(mn):
            signals_missing.append(f"HTML-fingerprint / 2-hop html_fp pattern `{mn}` skews collapsed.")

    for r in joint.get("ranked_joint_separators_favoring_collapsed_top10") or []:
        cn = str(r.get("condition_name") or "")
        if "html_fp" in cn and "semantic" in cn:
            signals_missing.append(f"Joint rule `{cn}` marks collapsed same-unlabeled concentration.")
        if "source_count_eq_1" in cn and "2hop" in cn:
            signals_missing.append(f"Single-source 2-hop rule `{cn}` associates with collapse.")

    for r in joint.get("ranked_joint_separators_favoring_rescued_top10") or []:
        cn = str(r.get("condition_name") or "")
        if "body" in cn and "jaccard" in cn:
            signals_used.append(f"Body-similarity joint rule `{cn}` aligns with rescued pairs.")

    n_collapsed_hf = int((html_fp.get("counts") or {}).get("n_collapsed_html_fp_frontier") or 0)
    if n_collapsed_hf > 0:
        interventions.append(
            "html_fp frontier: add features or loss weight for shared html_structure_fingerprint "
            "when semantic cosine is in [0.75, 0.85) — many collapsed same-unlabeled pairs sit here."
        )
    if not interventions:
        interventions.append(
            "Inspect ranked_joint_separators and html_fp_frontier cohorts in the HTML review export."
        )
    if not signals_used:
        signals_used.append("Multi-source provenance and strong body/path/sender features correlate with rescue.")
    if not signals_missing:
        signals_missing.append("Weak semantic + html_fp-only 2-hop support correlates with persistent collapse.")

    return {
        "signals_model_uses_successfully": signals_used[:12],
        "signals_present_in_collapsed_but_underused": signals_missing[:12],
        "suggested_next_interventions": interventions[:10],
    }


def build_same_unlabeled_inspection_frame(
    *,
    df_work: pd.DataFrame,
    scores: np.ndarray,
    same_unlabeled_mask: np.ndarray,
    gt_path: Path,
    label_map: dict[str, Any],
    nodes_by_email: dict[str, dict[str, set[str]]],
    rescued_min: float,
    collapsed_max: float,
    email_text_by_eid: dict[str, dict[str, str]] | None = None,
) -> pd.DataFrame:
    """Full inspection rows for GT same-campaign unlabeled pairs with bucket labels."""
    eval_mask = same_unlabeled_mask & np.isfinite(scores)
    if not bool(np.any(eval_mask)):
        return pd.DataFrame()
    df_eval = df_work.loc[eval_mask].copy()
    df_eval["score"] = scores[eval_mask]
    df_ins = _pse()._build_high_band_inspection_dataframe(
        df_eval=df_eval,
        row_mask=np.ones(len(df_eval), dtype=bool),
        gt_path=gt_path,
        label_map=label_map,
        gt_relation="same_campaign",
        nodes_by_email=nodes_by_email,
        cohort="same_campaign_unlabeled",
    )
    if df_ins.empty:
        return df_ins

    df_ins = _merge_pair_features_from_eval(df_ins, df_eval)

    buckets = assign_same_unlabeled_buckets(
        df_ins["score"].to_numpy(),
        collapsed_max=collapsed_max,
        rescued_min=rescued_min,
    )
    df_ins["same_unlabeled_bucket"] = buckets
    df_ins["pair_status"] = "unlabeled"

    if email_text_by_eid:
        df_ins = _compute_text_similarity_columns(df_ins, email_text_by_eid)
        df_ins = _backfill_body_features_from_text(df_ins)

    df_ins = _enrich_path_features_from_nodes(df_ins, nodes_by_email)

    for _ac, short in _pse()._PAIR_SHARED_CHANNEL_DEFS:
        if f"has_shared_{short}" not in df_ins.columns:
            extras = []
            for _, r in df_ins.iterrows():
                d = _pse()._pair_shared_evidence_detail(
                    str(r["email_i"]), str(r["email_j"]), nodes_by_email
                )
                extras.append(bool(d.get(f"has_shared_{short}")))
            df_ins[f"has_shared_{short}"] = extras

    return df_ins


def _rescued_collapsed_artifact_stem(filename_suffix: str = "") -> str:
    return f"pair_same_unlabeled_rescued_vs_collapsed{filename_suffix}"


def run_same_unlabeled_rescued_vs_collapsed_analysis(
    *,
    df_work: pd.DataFrame,
    scores: np.ndarray,
    gt_path: Path,
    label_map: dict[str, Any],
    out_dir: Path,
    nodes_by_email: dict[str, dict[str, set[str]]],
    evidence_index: dict[tuple[str, str], list[dict[str, Any]]] | None,
    email_text_by_eid: dict[str, dict[str, str]] | None = None,
    rescued_score_min: float = 0.80,
    collapsed_score_max: float = 0.10,
    email_text_preview_chars: int = 500,
    email_text_wrap_width: int = 88,
    cross_unlabeled_mask: np.ndarray | None = None,
    nodes_by_email_for_cross: dict[str, dict[str, set[str]]] | None = None,
    filename_suffix: str = "",
    export_flags: Any = None,
) -> dict[str, Any]:
    """
    Write rescued-vs-collapsed artifacts under ``out_dir`` (pair_score_separation folder).
    """
    from seed_candidate_workflow.utils.pair_score_separation_output_layout import (
        ExportFlags,
        ensure_pair_score_separation_layout,
        rel_to_root,
    )

    out_dir = Path(out_dir).resolve()
    layout = ensure_pair_score_separation_layout(out_dir)
    flags = export_flags if export_flags is not None else ExportFlags()
    core_json = layout["core_json"]
    debug_json = layout["debug_json"]
    debug_csv = layout["debug_csv"]
    review_html = layout["review_html"]
    art = _rescued_collapsed_artifact_stem(filename_suffix)

    ei = df_work["email_i"].astype(str).values
    ej = df_work["email_j"].astype(str).values
    n = len(df_work)
    camp_i = np.array([label_map.get(str(ei[k])) for k in range(n)], dtype=object)
    camp_j = np.array([label_map.get(str(ej[k])) for k in range(n)], dtype=object)
    both = np.array([camp_i[k] is not None and camp_j[k] is not None for k in range(n)], dtype=bool)
    unl = (
        df_work["pair_status"].astype(str).str.lower().eq("unlabeled").to_numpy()
        if "pair_status" in df_work.columns
        else np.zeros(n, dtype=bool)
    )
    same_unl = both & (camp_i == camp_j) & unl & np.isfinite(scores)

    df_same = build_same_unlabeled_inspection_frame(
        df_work=df_work,
        scores=scores,
        same_unlabeled_mask=same_unl,
        gt_path=gt_path,
        label_map=label_map,
        nodes_by_email=nodes_by_email,
        rescued_min=rescued_score_min,
        collapsed_max=collapsed_score_max,
        email_text_by_eid=email_text_by_eid,
    )

    if df_same.empty:
        summary = {
            "status": "empty",
            "gt_path": str(gt_path.resolve()),
            "thresholds": {
                "collapsed_same_unlabeled_max_score": float(collapsed_score_max),
                "rescued_same_unlabeled_min_score": float(rescued_score_min),
            },
            "counts": {"n_same_campaign_unlabeled": 0},
        }
        summary_path = core_json / f"{art}_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return {
            "summary_path": str(summary_path),
            "status": "empty",
            "primary_outputs": {
                "rescued_vs_collapsed_summary_json": rel_to_root(layout, summary_path),
            },
        }

    df_same = attach_twohop_channel_columns(df_same, evidence_index=evidence_index)
    rescued_mask = df_same["same_unlabeled_bucket"].eq("rescued_same_unlabeled").to_numpy()
    collapsed_mask = df_same["same_unlabeled_bucket"].eq("collapsed_same_unlabeled").to_numpy()
    mid_mask = df_same["same_unlabeled_bucket"].eq("mid_same_unlabeled").to_numpy()

    marginal, table_rows = _marginal_comparison_table(
        gt_path=gt_path,
        rescued_df=df_same.loc[rescued_mask],
        collapsed_df=df_same.loc[collapsed_mask],
        mid_df=df_same.loc[mid_mask],
        nodes_by_email=nodes_by_email,
        n_same_unlabeled=int(len(df_same)),
    )

    joint_payload, joint_rows = _build_joint_separators_rescued_vs_collapsed(
        gt_path=gt_path,
        df_same=df_same,
        rescued_mask=rescued_mask,
        collapsed_mask=collapsed_mask,
        nodes_by_email=nodes_by_email,
        evidence_index=evidence_index,
    )

    df_cross_eval = None
    df_cross_low = None
    if cross_unlabeled_mask is not None:
        cross_eval = cross_unlabeled_mask & np.isfinite(scores)
        if cross_eval.any():
            df_cross_eval = df_work.loc[cross_eval].copy()
            df_cross_eval["score"] = scores[cross_eval]
            cross_nodes = nodes_by_email_for_cross or nodes_by_email
            df_cross_eval = _merge_pair_features_from_eval(df_cross_eval, df_cross_eval)
            df_cross_eval = _enrich_path_features_from_nodes(df_cross_eval, cross_nodes)
            if email_text_by_eid:
                df_cross_eval = _compute_text_similarity_columns(df_cross_eval, email_text_by_eid)
                df_cross_eval = _backfill_body_features_from_text(df_cross_eval)
            cross_sc = pd.to_numeric(df_cross_eval["score"], errors="coerce")
            df_cross_low = df_cross_eval.loc[cross_sc.le(0.40)].copy()

    sem_all = _num_series(df_same, "semantic_cosine_max")
    html_fp_mask = _bool_series(df_same, "has_shared_html_fp") & sem_all.ge(HTML_FP_SEMANTIC_LO) & sem_all.lt(
        HTML_FP_SEMANTIC_HI
    )

    html_fp = _analyze_html_fp_frontier(
        df_same=df_same,
        rescued_mask=rescued_mask,
        collapsed_mask=collapsed_mask,
        cross_unlabeled_mask=cross_unlabeled_mask,
        df_eval_cross=df_cross_eval,
        nodes_by_email=nodes_by_email_for_cross or nodes_by_email,
    )

    feature_population_diagnostics = _build_feature_population_diagnostics(
        df_same=df_same,
        rescued_mask=rescued_mask,
        collapsed_mask=collapsed_mask,
        frontier_mask=html_fp_mask.to_numpy(),
    )
    path_feature_population_diagnostics = _build_path_feature_population_diagnostics(
        df_same=df_same,
        rescued_mask=rescued_mask,
        collapsed_mask=collapsed_mask,
        frontier_mask=html_fp_mask.to_numpy(),
    )
    body_vs_body_only_comparison = _build_body_vs_body_only_comparison(
        rescued_df=df_same.loc[rescued_mask],
        collapsed_df=df_same.loc[collapsed_mask],
        cross_low_df=df_cross_low,
    )
    single_source_frontier_summary = _build_single_source_frontier_summary(
        df_same, collapsed_mask
    )

    recommendations = _generate_rescued_vs_collapsed_recommendations(
        marginal=marginal,
        joint=joint_payload,
        html_fp=html_fp,
    )

    thresholds = {
        "collapsed_same_unlabeled_max_score": float(collapsed_score_max),
        "rescued_same_unlabeled_min_score": float(rescued_score_min),
        "mid_same_unlabeled_between": [float(collapsed_score_max), float(rescued_score_min)],
    }

    summary = {
        "status": "ok",
        "gt_path": str(gt_path.resolve()),
        "thresholds": thresholds,
        "bucket_definitions": {
            "collapsed_same_unlabeled": f"score <= {collapsed_score_max}",
            "rescued_same_unlabeled": f"score >= {rescued_score_min}",
            "mid_same_unlabeled": f"{collapsed_score_max} < score < {rescued_score_min}",
        },
        "marginal_comparison": marginal,
        "joint_separator_analysis": joint_payload,
        "same_unlabeled_html_fp_frontier_analysis": html_fp,
        "feature_population_diagnostics": feature_population_diagnostics,
        "path_feature_population_diagnostics": path_feature_population_diagnostics,
        "body_vs_body_only_comparison": body_vs_body_only_comparison,
        "single_source_frontier_summary": single_source_frontier_summary,
        "rescued_vs_collapsed_recommendations": recommendations,
    }

    summary_path = core_json / f"{art}_summary.json"
    table_path = debug_csv / f"detail_{art}_table.csv"
    joint_path = debug_json / f"detail_{art}_joint_summary.json"

    paths: dict[str, str] = {"summary_path": str(summary_path)}
    if flags.emit_debug_csv:
        pd.DataFrame(table_rows).to_csv(table_path, index=False)
        paths["table_path"] = str(table_path)
    if flags.emit_debug_json:
        with open(joint_path, "w", encoding="utf-8") as f:
            json.dump(joint_payload, f, indent=2, default=str)
        paths["joint_path"] = str(joint_path)
    summary["output_layout"] = {k: rel_to_root(layout, v) for k, v in layout.items() if k != "root"}

    if email_text_by_eid is not None:
        from seed_candidate_workflow.utils.pair_inspection_admitting_evidence import (
            enrich_inspection_with_admitting_evidence,
        )

        df_review = enrich_inspection_with_admitting_evidence(
            df_same, evidence_index=evidence_index or {}
        )
        df_review = _attach_inspection_badges(df_review)
        df_review = _pse()._enrich_pairs_with_email_text(
            df_review,
            email_text_by_eid=email_text_by_eid,
            preview_chars=email_text_preview_chars,
            regime_fn=_classify_rescued_collapsed_regime,
            review_prompt_fn=_rescued_collapsed_review_prompt,
        )
        df_review = _pse()._inject_semantic_cosine_for_manual_review(df_review)

        if flags.emit_debug_csv:
            review_csv = debug_csv / f"debug_{art}_pairs_for_review.csv"
            df_review.to_csv(review_csv, index=False)
            paths["review_csv"] = str(review_csv)

        gt_display = str(gt_path.resolve())
        banner = _rescued_collapsed_page_banner_html(
            thresholds=thresholds,
            gt_path=gt_display,
        )

        for bucket, slug, is_primary in (
            ("rescued_same_unlabeled", "rescued", True),
            ("collapsed_same_unlabeled", "collapsed", True),
            ("mid_same_unlabeled", "mid", False),
        ):
            sub = df_review[df_review["same_unlabeled_bucket"] == bucket]
            if sub.empty:
                continue
            if is_primary:
                sub_html = review_html / f"pair_same_unlabeled_{slug}_for_review{filename_suffix}.html"
                _write_rescued_collapsed_review_html(
                    sub,
                    out_path=sub_html,
                    email_text_by_eid=email_text_by_eid,
                    thresholds=thresholds,
                    gt_path=gt_display,
                    title_suffix=f" — {slug}",
                )
                paths[f"review_html_{slug}"] = str(sub_html)
            elif flags.emit_debug_html:
                sub_html = debug_csv / f"debug_pair_same_unlabeled_{slug}_for_review{filename_suffix}.html"
                _write_rescued_collapsed_review_html(
                    sub,
                    out_path=sub_html,
                    email_text_by_eid=email_text_by_eid,
                    thresholds=thresholds,
                    gt_path=gt_display,
                    title_suffix=f" — {slug}",
                )
                paths[f"review_html_{slug}"] = str(sub_html)

        if flags.emit_debug_html:
            debug_combined = debug_csv / f"debug_{art}_ALL_BUCKETS.html"
            _write_rescued_collapsed_review_html(
                df_review,
                out_path=debug_combined,
                email_text_by_eid=email_text_by_eid,
                thresholds=thresholds,
                gt_path=gt_display,
                title_suffix=" — all buckets (debug)",
                page_banner_html=banner,
            )
            paths["review_html_debug_combined"] = str(debug_combined)

    summary["primary_outputs"] = {
        "rescued_vs_collapsed_summary_json": rel_to_root(layout, summary_path),
        **{
            k: rel_to_root(layout, Path(paths[f"review_html_{slug}"]))
            for k, slug in (
                ("rescued_review_html", "rescued"),
                ("collapsed_review_html", "collapsed"),
            )
            if paths.get(f"review_html_{slug}")
        },
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    return paths


def _rescued_collapsed_page_banner_html(
    *,
    thresholds: dict[str, Any],
    gt_path: str,
) -> str:
    collapsed = thresholds.get("collapsed_same_unlabeled_max_score")
    rescued = thresholds.get("rescued_same_unlabeled_min_score")
    return (
        '<div class="page-banner">'
        f"<strong>Rescued threshold</strong> score ≥ {html.escape(str(rescued))} · "
        f"<strong>Collapsed threshold</strong> score ≤ {html.escape(str(collapsed))} · "
        f"<strong>GT file</strong> {html.escape(gt_path)}"
        "</div>"
    )


def _write_rescued_collapsed_review_html(
    df_pairs: pd.DataFrame,
    *,
    out_path: Path,
    email_text_by_eid: dict[str, dict[str, str]],
    thresholds: dict[str, Any],
    gt_path: str = "",
    title_suffix: str = "",
    page_banner_html: str | None = None,
) -> None:
    """HTML review with bucket filters — shared low-band pair-card layout."""
    pse = _pse()
    title = f"Same-campaign unlabeled: rescued vs collapsed{title_suffix}"
    banner = page_banner_html or _rescued_collapsed_page_banner_html(
        thresholds=thresholds,
        gt_path=gt_path,
    )
    subtitle = (
        f"Collapsed ≤ {thresholds.get('collapsed_same_unlabeled_max_score')}; "
        f"rescued ≥ {thresholds.get('rescued_same_unlabeled_min_score')}."
    )
    pse._write_pairs_for_review_html(
        df_pairs,
        out_path=out_path,
        email_text_by_eid=email_text_by_eid,
        title=title,
        subtitle=subtitle,
        review_prompt="Compare rescue-aligned similarity features vs score collapse.",
        gt_note="same_campaign unlabeled",
        filter_column="same_unlabeled_bucket",
        page_banner_html=banner,
        max_main_width="56rem",
    )
