"""
Rule-expression evaluator for candidate-family scorecard (extends gt_edge_structure rules).

Supports numeric thresholds (ge/lt/le), time gaps (3d, 7d, 14d), and enriched pair columns.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd

from seed_candidate_workflow.utils.gt_edge_structure_analysis import (
    _build_bool_terms,
    _eval_rule_expr as _eval_rule_expr_legacy,
)

_COMPARE_RE = re.compile(r"^(.+)_(ge|gt|le|lt)_(.+)$")


def _parse_threshold(raw: str) -> float:
    s = str(raw).strip().lower().replace(",", ".")
    if s.endswith("d"):
        days = float(s[:-1])
        return float(days * 86400.0)
    if "_" in s and s.replace("_", "").replace(".", "").isdigit():
        return float(s.replace("_", "."))
    return float(s)


def _parse_time_gap_token(key: str) -> tuple[str, str, float] | None:
    """Match time_gap_le_7d or time_gap_seconds_min_le_7d."""
    m = re.match(r"^time_gap(?:_seconds_min)?_(le|lt|ge|gt)_(.+)$", key)
    if not m:
        return None
    op, thr_raw = m.group(1), m.group(2)
    return "time_gap_seconds_min", op, _parse_threshold(thr_raw)


def _as_series(df: pd.DataFrame, col: str, *, default: Any = np.nan) -> pd.Series:
    """Return a single Series even when duplicate column names exist after enrichment joins."""
    if col not in df.columns:
        return pd.Series([default] * len(df), index=df.index)
    val = df[col]
    if isinstance(val, pd.DataFrame):
        val = val.iloc[:, 0]
    return val


def _numeric_column(df: pd.DataFrame, col: str) -> np.ndarray:
    return pd.to_numeric(_as_series(df, col), errors="coerce").to_numpy(dtype=np.float64)


def _bool_column(df: pd.DataFrame, col: str) -> np.ndarray:
    return _as_series(df, col, default=False).fillna(False).astype(bool).to_numpy()


def _resolve_semantic_cosine(df: pd.DataFrame) -> np.ndarray:
    if "semantic_cosine" in df.columns:
        v = _numeric_column(df, "semantic_cosine")
        if np.any(np.isfinite(v)):
            return v
    return _numeric_column(df, "semantic_cosine_max")


def _eval_single_token(df: pd.DataFrame, key: str, n: int) -> np.ndarray:
    """Evaluate one AND-term; returns bool array length n."""
    tg = _parse_time_gap_token(key)
    if tg is not None:
        col, op, thr = tg
        vals = _numeric_column(df, col)
        fin = np.isfinite(vals)
        if op in ("ge", "gt"):
            return fin & (vals >= thr)
        return fin & (vals <= thr)

    m = _COMPARE_RE.match(key)
    if m:
        col, op, thr_s = m.group(1), m.group(2), m.group(3)
        thr = _parse_threshold(thr_s)
        if col == "semantic":
            vals = _resolve_semantic_cosine(df)
        elif col == "body_cosine":
            vals = _numeric_column(df, "body_cosine")
        elif col == "subject_cosine":
            vals = _numeric_column(df, "subject_cosine")
        else:
            vals = _numeric_column(df, col)
        fin = np.isfinite(vals)
        if op in ("ge", "gt"):
            return fin & (vals >= thr)
        return fin & (vals < thr if op == "lt" else vals <= thr)

    bool_terms = _build_bool_terms(df)
    cos = _resolve_semantic_cosine(df)

    if key == "semantic_ge_0_90":
        return np.isfinite(cos) & (cos >= 0.90)
    if key == "semantic_ge_0_92":
        return np.isfinite(cos) & (cos >= 0.92)
    if key == "semantic_ge_0_93":
        return np.isfinite(cos) & (cos >= 0.93)
    if key == "semantic_ge_0_95":
        return np.isfinite(cos) & (cos >= 0.95)
    if key == "semantic_band_0_85_0_90":
        return np.isfinite(cos) & (cos >= 0.85) & (cos < 0.90)
    if key == "n_shared_core_channels_ge_1":
        ncol = pd.to_numeric(df.get("n_shared_core_channels"), errors="coerce").fillna(0)
        return (ncol >= 1).to_numpy()
    if key == "n_shared_core_channels_ge_2":
        ncol = pd.to_numeric(df.get("n_shared_core_channels"), errors="coerce").fillna(0)
        return (ncol >= 2).to_numpy()
    if key == "support_count_excl_domain_and_root_stem_ge_1":
        v = _numeric_column(df, "support_count_excl_domain_and_root_stem")
        return np.isfinite(v) & (v >= 1)
    if key == "support_count_excl_domain_and_root_stem_ge_2":
        v = _numeric_column(df, "support_count_excl_domain_and_root_stem")
        return np.isfinite(v) & (v >= 2)
    if key == "strong_support_count_ge_1":
        v = _numeric_column(df, "strong_support_count")
        return np.isfinite(v) & (v >= 1)
    if key == "shared_domain_without_other_support":
        if "shared_domain_without_strong_support" in df.columns:
            return _bool_column(df, "shared_domain_without_strong_support")
        return np.zeros(n, dtype=bool)
    if key == "shared_url_or_stem_without_sender":
        if "shared_url_or_stem_without_sender" in df.columns:
            return _bool_column(df, "shared_url_or_stem_without_sender")
        return np.zeros(n, dtype=bool)
    if key == "sender_exact_match":
        if "sender_exact_match" in df.columns:
            return _bool_column(df, "sender_exact_match")
        if "has_shared_sender" in df.columns:
            return _bool_column(df, "has_shared_sender")
        return np.zeros(n, dtype=bool)
    if key == "sender_localpart_exact_match":
        if "sender_localpart_exact_match" in df.columns:
            return _bool_column(df, "sender_localpart_exact_match")
        return np.zeros(n, dtype=bool)
    if key == "sender_domain_exact_match":
        if "sender_domain_exact_match" in df.columns:
            return _bool_column(df, "sender_domain_exact_match")
        if "has_shared_sender_domain" in df.columns:
            return _bool_column(df, "has_shared_sender_domain")
        return np.zeros(n, dtype=bool)
    if key == "direct_shared_html_fp":
        if "direct_shared_html_fp" in df.columns:
            return _bool_column(df, "direct_shared_html_fp")
        if "shared_html_fp" in df.columns:
            return _bool_column(df, "shared_html_fp")
        return np.zeros(n, dtype=bool)
    if key == "twohop_via_html_fp":
        if "twohop_via_html_fp" in df.columns:
            return _bool_column(df, "twohop_via_html_fp")
        f2 = bool_terms.get("from_2hop", np.zeros(n, dtype=bool))
        hfp = bool_terms.get("shared_html_fp", np.zeros(n, dtype=bool))
        return f2 & hfp
    if key == "same_url_provider_family":
        if "same_registrable_domain" in df.columns:
            return _bool_column(df, "same_registrable_domain")
        return np.zeros(n, dtype=bool)
    if key == "shared_nontrivial_stem":
        if "shared_stem_nontrivial" in df.columns:
            return _bool_column(df, "shared_stem_nontrivial")
        if "has_shared_stem" in df.columns:
            return _bool_column(df, "has_shared_stem")
        return np.zeros(n, dtype=bool)
    if key in bool_terms:
        return bool_terms[key]
    if key in df.columns:
        return _bool_column(df, key)

    return np.zeros(n, dtype=bool)


def eval_family_rule_expr(df: pd.DataFrame, expr: str) -> np.ndarray:
    """Evaluate AND/NOT rule on enriched GT pair dataframe."""
    n = len(df)
    if n == 0:
        return np.zeros(0, dtype=bool)

    if "_AND_" not in expr and not _needs_extended(expr):
        try:
            return _eval_rule_expr_legacy(df, expr)
        except Exception:
            pass

    terms = expr.split("_AND_")
    out = np.ones(n, dtype=bool)
    for tok in terms:
        neg = tok.startswith("NOT_")
        key = tok[4:] if neg else tok
        base = _eval_single_token(df, key, n)
        out &= ~base if neg else base
    return out


def _needs_extended(expr: str) -> bool:
    extended_markers = (
        "path_token",
        "body_",
        "subject_",
        "sender_localpart_norm",
        "rarity_weighted",
        "time_gap",
        "twohop_via",
        "direct_shared_html",
        "support_count_excl",
        "strong_support",
        "same_registrable",
        "shared_nontrivial",
    )
    return any(m in expr for m in extended_markers)


def rule_requires_columns(expr: str) -> list[str]:
    """Rough column dependency check for skip reporting."""
    cols: list[str] = []
    for marker, col in (
        ("path_token_jaccard_combined", "path_token_jaccard_combined"),
        ("url_path_token_jaccard", "url_path_token_jaccard"),
        ("stem_path_token_jaccard", "stem_path_token_jaccard"),
        ("body_token_jaccard", "body_token_jaccard"),
        ("body_char4gram_jaccard", "body_char4gram_jaccard"),
        ("body_cosine", "body_cosine"),
        ("subject_token_jaccard", "subject_token_jaccard"),
        ("subject_cosine", "subject_cosine"),
        ("sender_localpart_norm_jaccard", "sender_localpart_norm_jaccard"),
        ("rarity_weighted_support_sum", "rarity_weighted_support_sum"),
        ("twohop_via_html_fp", "twohop_via_html_fp"),
        ("same_query_key", "same_query_key_set"),
    ):
        if marker in expr:
            cols.append(col)
    return cols


def columns_available_for_rule(df: pd.DataFrame, expr: str) -> tuple[bool, list[str]]:
    missing: list[str] = []
    for col in rule_requires_columns(expr):
        if col not in df.columns:
            missing.append(col)
        elif col == "twohop_via_html_fp" and df[col].isna().all():
            missing.append(f"{col}(all_nan)")
    return (len(missing) == 0, missing)
