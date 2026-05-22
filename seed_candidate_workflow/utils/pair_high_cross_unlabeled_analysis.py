"""
High-scoring cross-campaign unlabeled pair analysis for pair_score_separation.

Compares dangerous GNN-only false positives (high cross unlabeled) against
high-scoring same-campaign unlabeled pairs on explicit, shared, provenance, and
optional latent diagnostics.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from seed_candidate_workflow.utils.pair_frontier_analysis import (
    COUNT_SUPPORT_KEYS,
    SHARED_BOOLEAN_KEYS,
    SIMILARITY_FEATURE_KEYS,
    _build_body_path_signal_comparison,
    _build_cohort_inspection_df,
    _build_edge_profile,
    _build_frontier_joint_two_cohort,
    _extend_marginal_with_full_features,
    _feature_population_diagnostics,
)
from seed_candidate_workflow.utils.pair_low_band_twohop_channel import low_band_twohop_joint_rule_names
from seed_candidate_workflow.utils.pair_mid_band_frontier import (
    _build_same_vs_cross_marginal,
    _build_two_cohort_marginal,
)
from seed_candidate_workflow.utils.pair_score_separation_output_layout import (
    ExportFlags,
    rel_to_root,
)

_PSE: Any = None


def _pse() -> Any:
    global _PSE
    if _PSE is None:
        from seed_candidate_workflow.utils import pair_score_separation as mod

        _PSE = mod
    return _PSE


@dataclass(frozen=True)
class HighCrossThresholds:
    """Score cutoffs for high-cross unlabeled analysis."""

    high_cross_score_min: float = 0.80
    mid_cross_score_min: float = 0.70


HIGH_CROSS_COHORTS: tuple[str, ...] = (
    "high_cross_unlabeled",
    "high_same_unlabeled",
    "mid_cross_unlabeled",
)

HIGH_CROSS_PRIMARY_COHORTS: tuple[str, ...] = (
    "high_cross_unlabeled",
    "high_same_unlabeled",
)

BODY_ONLY_FEATURE_KEYS: tuple[str, ...] = (
    "body_only_token_jaccard",
    "body_only_char4gram_jaccard",
)

PATH_FEATURE_KEYS: tuple[str, ...] = (
    "path_token_jaccard_combined",
    "url_path_token_jaccard",
    "stem_path_token_jaccard",
)

SEMANTIC_COSINE_KEYS: tuple[str, ...] = (
    "semantic_cosine_max",
    "semantic_cosine",
    "semantic_cosine_for_display",
)

LATENT_FEATURE_KEYS: tuple[str, ...] = (
    "gnn_encoder_cosine",
    "gnn_encoder_l2",
    "static_subj_body_cosine",
)

SHARED_COUNT_KEYS: tuple[str, ...] = (
    "shared_sender_count",
    "shared_stem_count",
    "shared_url_count",
    "shared_attachment_count",
    "shared_sender_domain_count",
    "shared_domain_count",
)


def _as_float_array(scores: np.ndarray | pd.Series) -> np.ndarray:
    return np.asarray(pd.to_numeric(np.asarray(scores), errors="coerce"), dtype=np.float64)


def high_score_unlabeled_cohort_masks(
    *,
    same_eval: np.ndarray,
    cross_eval: np.ndarray,
    unl_eval: np.ndarray,
    scores: np.ndarray,
    thresholds: HighCrossThresholds,
) -> dict[str, np.ndarray]:
    s = _as_float_array(scores)
    hi = s >= float(thresholds.high_cross_score_min)
    mid_lo = float(thresholds.mid_cross_score_min)
    mid_hi = float(thresholds.high_cross_score_min)
    mid = (s >= mid_lo) & (s < mid_hi)
    return {
        "high_cross_unlabeled": cross_eval & unl_eval & hi,
        "high_same_unlabeled": same_eval & unl_eval & hi,
        "mid_cross_unlabeled": cross_eval & unl_eval & mid,
    }


def _classify_high_cross_review_regime(row: pd.Series) -> str:
    return str(row.get("analysis_cohort") or row.get("cohort") or "high_cross_unknown")


def _high_cross_review_prompt(row: pd.Series) -> str:
    cohort = _classify_high_cross_review_regime(row)
    prompts = {
        "high_cross_unlabeled": (
            "High-score cross-campaign unlabeled: likely GNN-only false positive. "
            "Why is latent similarity high without same-campaign GT?"
        ),
        "high_same_unlabeled": (
            "High-score same-campaign unlabeled: rescued-like true positive pattern. "
            "Compare explicit/support vs the high-cross cohort."
        ),
        "mid_cross_unlabeled": (
            "Mid-high cross unlabeled: below primary high threshold but elevated. "
            "What separates these from the dangerous high-cross band?"
        ),
    }
    return prompts.get(cohort, "High-score unlabeled pair — inspect failure mode.")


def _cosine_l2_pair(
    a: np.ndarray | None, b: np.ndarray | None
) -> tuple[float | None, float | None]:
    if a is None or b is None:
        return None, None
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na <= 0 or nb <= 0:
        return None, None
    cos = float(np.dot(a, b) / (na * nb))
    l2 = float(np.linalg.norm(a - b))
    return cos, l2


def attach_latent_diagnostics_to_cohort_df(
    df: pd.DataFrame,
    *,
    project_root: Path,
    run_dir: Path | None,
    graph_pt: Path | None,
    inference_bundle: dict[str, Any] | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Add optional GNN/static embedding cosines for cohort rows only."""
    meta: dict[str, Any] = {
        "attempted": True,
        "gnn_encoder_cosine": {"present": False},
        "static_subj_body_cosine": {"present": False},
    }
    if df.empty:
        return df, meta

    out = df.copy()
    out["gnn_encoder_cosine"] = np.nan
    out["gnn_encoder_l2"] = np.nan
    out["static_subj_body_cosine"] = np.nan

    id_to_gnn: dict[str, np.ndarray] = {}
    id_to_static: dict[str, np.ndarray] = {}
    gnn_err: str | None = None
    static_err: str | None = None

    if inference_bundle is not None:
        try:
            from seed_candidate_workflow.utils.raw_gnn_notebook import load_email_external_ids
            from src.clustering.clustering_helpers import extract_email_embeddings

            data_cpu = inference_bundle.get("data_cpu")
            model = inference_bundle.get("model")
            device = inference_bundle.get("device")
            if model is not None and data_cpu is not None and graph_pt is not None:
                meta_path = Path(graph_pt).with_suffix(".meta.json")
                external_ids = [str(x) for x in load_email_external_ids(meta_path)]
                id_to_gnn = extract_email_embeddings(
                    model, data_cpu, device, external_ids
                )
                meta["gnn_encoder_cosine"]["present"] = bool(id_to_gnn)
                meta["gnn_encoder_source"] = "inference_bundle_gnn"
        except Exception as exc:
            gnn_err = str(exc)

    try:
        from src.clustering.clustering_helpers import (
            load_transformer_subject_body_embeddings_from_cache,
        )

        emb_path = project_root / "core" / "utils" / "embeddings" / "output" / "embeddings.json"
        if emb_path.is_file():
            id_to_static = load_transformer_subject_body_embeddings_from_cache(
                embeddings_json_path=emb_path
            )
            meta["static_subj_body_cosine"]["present"] = bool(id_to_static)
            meta["static_subj_body_path"] = str(emb_path)
    except Exception as exc:
        static_err = str(exc)

    if gnn_err:
        meta["gnn_encoder_error"] = gnn_err
    if static_err:
        meta["static_subj_body_error"] = static_err

    cos_gnn: list[float | None] = []
    cos_static: list[float | None] = []
    l2_gnn: list[float | None] = []
    for _, r in out.iterrows():
        ei, ej = str(r["email_i"]), str(r["email_j"])
        vi = id_to_gnn.get(ei)
        vj = id_to_gnn.get(ej)
        c, l2 = _cosine_l2_pair(vi, vj)
        cos_gnn.append(c)
        l2_gnn.append(l2)
        si = id_to_static.get(ei)
        sj = id_to_static.get(ej)
        cs, _ = _cosine_l2_pair(si, sj)
        cos_static.append(cs)

    out["gnn_encoder_cosine"] = cos_gnn
    out["gnn_encoder_l2"] = l2_gnn
    out["static_subj_body_cosine"] = cos_static
    return out, meta


def _row_likely_explanation_tags(row: pd.Series) -> list[str]:
    tags: list[str] = []
    sem = pd.to_numeric(row.get("semantic_cosine_max"), errors="coerce")
    body = pd.to_numeric(row.get("body_token_jaccard"), errors="coerce")
    body_only = pd.to_numeric(row.get("body_only_token_jaccard"), errors="coerce")
    path = pd.to_numeric(row.get("path_token_jaccard_combined"), errors="coerce")
    n_shared = pd.to_numeric(row.get("n_shared_core_channels"), errors="coerce")
    gnn_cos = pd.to_numeric(row.get("gnn_encoder_cosine"), errors="coerce")

    if pd.notna(sem) and float(sem) >= 0.90:
        if pd.isna(n_shared) or float(n_shared) < 1:
            tags.append("high_semantic_weak_shared_support")
        else:
            tags.append("high_semantic_with_some_support")
    if pd.notna(body_only) and float(body_only) >= 0.35:
        if pd.isna(body) or float(body) < float(body_only) + 0.05:
            tags.append("body_only_drives_similarity")
    if pd.notna(path) and float(path) >= 0.25:
        if pd.isna(n_shared) or float(n_shared) < 1:
            tags.append("path_similarity_without_shared_artifacts")
    if bool(row.get("from_2hop")) and not bool(row.get("from_semantic")):
        tags.append("twohop_without_semantic_provenance")
    if bool(row.get("cross_seed_component_flag")):
        tags.append("cross_seed_component_context")
    if pd.notna(gnn_cos) and float(gnn_cos) >= 0.85:
        weak_explicit = True
        for c in ("body_token_jaccard", "path_token_jaccard_combined", "semantic_cosine_max"):
            v = pd.to_numeric(row.get(c), errors="coerce")
            if pd.notna(v) and float(v) >= 0.25:
                weak_explicit = False
                break
        if weak_explicit:
            tags.append("high_latent_low_explicit")
    if bool(row.get("has_shared_html_fp")) and not bool(row.get("has_shared_sender")):
        tags.append("html_fingerprint_without_sender")
    if not tags:
        tags.append("unclassified_high_score_pattern")
    return tags


def _aggregate_likely_explanations(df_cross: pd.DataFrame) -> dict[str, Any]:
    if df_cross.empty:
        return {"n_pairs": 0, "top_tags": [], "by_tag_fraction": {}}
    tag_counts: dict[str, int] = {}
    for _, r in df_cross.iterrows():
        for t in _row_likely_explanation_tags(r):
            tag_counts[t] = tag_counts.get(t, 0) + 1
    n = int(len(df_cross))
    fracs = {k: float(v / n) for k, v in sorted(tag_counts.items(), key=lambda x: -x[1])}
    top = [
        {"tag": k, "count": v, "fraction": float(v / n)}
        for k, v in sorted(tag_counts.items(), key=lambda x: -x[1])[:12]
    ]
    return {"n_pairs": n, "top_tags": top, "by_tag_fraction": fracs}


def _col_non_null_count(df: pd.DataFrame, col: str) -> int:
    if col not in df.columns or df.empty:
        return 0
    series = df[col]
    if series.dtype == bool or series.dtype == object:
        return int(series.fillna(False).astype(bool).sum())
    return int(pd.to_numeric(series, errors="coerce").notna().sum())


def _row_has_semantic_cosine(row: pd.Series) -> bool:
    for k in SEMANTIC_COSINE_KEYS:
        v = pd.to_numeric(row.get(k), errors="coerce")
        if pd.notna(v):
            return True
    return False


def _row_has_body_only_features(row: pd.Series) -> bool:
    for k in BODY_ONLY_FEATURE_KEYS:
        v = pd.to_numeric(row.get(k), errors="coerce")
        if pd.notna(v):
            return True
    return False


def _row_has_path_features(row: pd.Series) -> bool:
    for k in PATH_FEATURE_KEYS:
        v = pd.to_numeric(row.get(k), errors="coerce")
        if pd.notna(v):
            return True
    return False


def _cohort_coverage_block(df: pd.DataFrame, cohort_name: str) -> dict[str, Any]:
    n = int(len(df))
    if n == 0:
        return {
            "cohort": cohort_name,
            "n_pairs": 0,
            "n_semantic_cosine_available": 0,
            "fraction_semantic_cosine_available": 0.0,
            "n_body_only_features_available": 0,
            "fraction_body_only_features_available": 0.0,
            "n_path_features_available": 0,
            "fraction_path_features_available": 0.0,
            "n_all_three_available": 0,
            "fraction_all_three_available": 0.0,
            "interpretation": "empty cohort",
        }
    sem_mask = df.apply(_row_has_semantic_cosine, axis=1)
    body_mask = df.apply(_row_has_body_only_features, axis=1)
    path_mask = df.apply(_row_has_path_features, axis=1)
    all_three = sem_mask & body_mask & path_mask
    n_sem = int(sem_mask.sum())
    n_body = int(body_mask.sum())
    n_path = int(path_mask.sum())
    n_all = int(all_three.sum())
    notes: list[str] = []
    if n_sem < n * 0.25:
        notes.append("semantic cosine is sparse — do not over-interpret semantic means")
    if n_body >= n * 0.8 and n_sem < n * 0.25:
        notes.append("body-only/path features are better observed than semantic cosine")
    if n_path >= n * 0.5 and n_sem < n * 0.25:
        notes.append("path signals may dominate visible support where semantic is missing")
    if n_all == 0:
        notes.append("no pair has semantic + body-only + path all populated")
    return {
        "cohort": cohort_name,
        "n_pairs": n,
        "n_semantic_cosine_available": n_sem,
        "fraction_semantic_cosine_available": float(n_sem / n),
        "n_body_only_features_available": n_body,
        "fraction_body_only_features_available": float(n_body / n),
        "n_path_features_available": n_path,
        "fraction_path_features_available": float(n_path / n),
        "n_all_three_available": n_all,
        "fraction_all_three_available": float(n_all / n),
        "interpretation_notes": notes,
    }


def _build_semantic_body_path_coverage_summary(
    *,
    df_cross: pd.DataFrame,
    df_same: pd.DataFrame,
) -> dict[str, Any]:
    cross = _cohort_coverage_block(df_cross, "high_cross_unlabeled")
    same = _cohort_coverage_block(df_same, "high_same_unlabeled")
    bullets: list[str] = []
    bullets.append(
        f"High-cross: semantic available on {cross['n_semantic_cosine_available']}/{cross['n_pairs']} "
        f"({100 * cross['fraction_semantic_cosine_available']:.0f}%), "
        f"body-only on {cross['n_body_only_features_available']}/{cross['n_pairs']}, "
        f"path on {cross['n_path_features_available']}/{cross['n_pairs']}, "
        f"all three on {cross['n_all_three_available']}/{cross['n_pairs']}."
    )
    bullets.append(
        f"High-same: semantic available on {same['n_semantic_cosine_available']}/{same['n_pairs']} "
        f"({100 * same['fraction_semantic_cosine_available']:.0f}%), "
        f"body-only on {same['n_body_only_features_available']}/{same['n_pairs']}, "
        f"path on {same['n_path_features_available']}/{same['n_pairs']}."
    )
    if cross["fraction_semantic_cosine_available"] < same["fraction_semantic_cosine_available"] - 0.05:
        bullets.append(
            "Semantic cosine is less often populated on high-cross than high-same — "
            "cross-cohort semantic comparisons may be under-observed."
        )
    return {
        "by_cohort": {"high_cross_unlabeled": cross, "high_same_unlabeled": same},
        "readable_bullets": bullets,
    }


def _row_has_minimal_explicit_support(row: pd.Series) -> bool:
    """True when pair has little/no shared explicit artifacts (suspicious latent-only regime)."""
    for k in SHARED_BOOLEAN_KEYS:
        if bool(row.get(k)):
            return False
    for k in SHARED_COUNT_KEYS:
        v = pd.to_numeric(row.get(k), errors="coerce")
        if pd.notna(v) and float(v) > 0:
            return False
    n_shared = pd.to_numeric(row.get("n_shared_core_channels"), errors="coerce")
    if pd.notna(n_shared) and float(n_shared) >= 1:
        return False
    source_count = pd.to_numeric(row.get("source_count"), errors="coerce")
    if pd.notna(source_count) and float(source_count) >= 2:
        return False
    return True


def _provenance_pattern_summary(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        return {}
    out: dict[str, Any] = {}
    for k in (
        "from_semantic",
        "from_2hop",
        "from_seed",
        "same_seed_component_flag",
        "cross_seed_component_flag",
    ):
        if k not in df.columns:
            continue
        col = df[k]
        if col.dtype == bool:
            out[k] = {"count": int(col.fillna(False).astype(bool).sum()), "fraction": float(col.fillna(False).astype(bool).mean())}
        else:
            s = pd.to_numeric(col, errors="coerce")
            if s.notna().any() and s.max() <= 1.0:
                out[k] = {"count": int((s > 0).sum()), "fraction": float((s > 0).mean())}
    if "source_count" in df.columns:
        sc = pd.to_numeric(df["source_count"], errors="coerce")
        out["source_count_mean"] = float(sc.mean()) if sc.notna().any() else None
        out["source_count_eq_1_fraction"] = float((sc == 1).mean()) if sc.notna().any() else None
    return out


def _latent_summary_for_df(df: pd.DataFrame) -> dict[str, Any]:
    block: dict[str, Any] = {}
    for col in LATENT_FEATURE_KEYS:
        if col not in df.columns or df.empty:
            block[col] = {"n_non_null": 0, "mean": None}
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        nn = int(s.notna().sum())
        block[col] = {
            "n_non_null": nn,
            "fraction_non_null": float(nn / len(df)) if len(df) else 0.0,
            "mean": float(s.mean()) if nn else None,
        }
    return block


def _build_high_cross_minimal_explicit_support_summary(
    *,
    df_cross: pd.DataFrame,
    nodes_by_email: dict[str, dict[str, set[str]]],
) -> dict[str, Any]:
    n_cross = int(len(df_cross))
    if n_cross == 0:
        return {
            "cohort_id": "high_cross_minimal_explicit_support",
            "definition": (
                "Subset of high_cross_unlabeled with no shared sender/url/stem/attachment/domain flags, "
                "no positive shared counts, n_shared_core_channels < 1, and source_count < 2 when present."
            ),
            "n_pairs": 0,
            "fraction_of_high_cross_cohort": 0.0,
        }
    mask = df_cross.apply(_row_has_minimal_explicit_support, axis=1)
    df_min = df_cross.loc[mask].copy()
    n_min = int(len(df_min))
    edge_prof = _build_edge_profile(
        cohort_name="high_cross_minimal_explicit_support",
        df=df_min,
        nodes_by_email=nodes_by_email,
    )
    return {
        "cohort_id": "high_cross_minimal_explicit_support",
        "definition": (
            "Subset of high_cross_unlabeled with no shared sender/url/stem/attachment/domain flags, "
            "no positive shared counts, n_shared_core_channels < 1, and source_count < 2 when present."
        ),
        "n_pairs": n_min,
        "fraction_of_high_cross_cohort": float(n_min / n_cross),
        "semantic_body_path_coverage": _cohort_coverage_block(df_min, "high_cross_minimal_explicit_support"),
        "edge_profile": edge_prof,
        "body_path_means": {
            k: (edge_prof.get("feature_summaries") or {}).get(k, {}).get("mean")
            for k in (*BODY_ONLY_FEATURE_KEYS, *PATH_FEATURE_KEYS, "body_token_jaccard")
        },
        "latent_diagnostics": _latent_summary_for_df(df_min),
        "likely_explanation_tags": _aggregate_likely_explanations(df_min),
        "provenance_patterns": _provenance_pattern_summary(df_min),
        "readable_bullets": [
            f"{n_min}/{n_cross} ({100 * n_min / n_cross:.0f}%) of high-cross unlabeled pairs have minimal explicit support.",
            edge_prof.get("bullet_summary", ""),
        ],
    }


def _build_focused_high_cross_population_diagnostics(
    *,
    cohort_dfs: dict[str, pd.DataFrame],
    df_eval: pd.DataFrame,
    latent_meta: dict[str, Any],
) -> dict[str, Any]:
    base = _feature_population_diagnostics(cohort_dfs, df_eval=df_eval)
    tracked = list(SIMILARITY_FEATURE_KEYS) + list(COUNT_SUPPORT_KEYS) + list(SHARED_BOOLEAN_KEYS)
    tracked += list(SHARED_COUNT_KEYS) + list(LATENT_FEATURE_KEYS)
    by_cohort_focus: dict[str, Any] = {}
    for cname in HIGH_CROSS_PRIMARY_COHORTS:
        df = cohort_dfs.get(cname, pd.DataFrame())
        n = int(len(df))
        feat_stats: dict[str, Any] = {}
        for k in tracked:
            feat_stats[k] = {
                "present": k in df.columns,
                "n_non_null": _col_non_null_count(df, k),
                "fraction_non_null": float(_col_non_null_count(df, k) / n) if n else 0.0,
            }
        by_cohort_focus[cname] = {"n_pairs": n, "features": feat_stats, "latent_attach_meta": latent_meta.get(cname)}
    base["tracked_features"] = tracked
    base["high_cross_primary_cohorts"] = by_cohort_focus
    return base


def _build_high_cross_vs_high_same_digest(
    *,
    comparisons: dict[str, Any],
    coverage_summary: dict[str, Any],
    minimal_support_summary: dict[str, Any],
    profile: dict[str, Any],
    likely_explanations: dict[str, Any],
) -> dict[str, Any]:
    factors: list[str] = []
    keyed: dict[str, Any] = {}

    marg = (comparisons.get("high_cross_vs_high_same") or {}).get("marginal") or {}
    for r in (marg.get("ranked_separators_top15") or [])[:10]:
        mg = str(r.get("metric_group") or "")
        mn = str(r.get("metric_name") or "")
        diff = r.get("difference_left_minus_right")
        if diff is None:
            continue
        d = float(diff)
        direction = "higher on high-cross" if d > 0 else "higher on high-same"
        factors.append(f"{mg}:{mn} — {direction} (Δ={d:+.3f}, cross is LEFT)")
        keyed[f"{mg}.{mn}"] = {"difference_cross_minus_same": d, "direction": direction}

    cross_cov = (coverage_summary.get("by_cohort") or {}).get("high_cross_unlabeled") or {}
    same_cov = (coverage_summary.get("by_cohort") or {}).get("high_same_unlabeled") or {}
    if cross_cov.get("fraction_cross_seed_component_flag") is None:
        pass
    if cross_cov.get("fraction_semantic_cosine_available", 1) < 0.3:
        factors.append(
            f"Semantic cosine is only available on ~{100 * float(cross_cov.get('fraction_semantic_cosine_available', 0)):.0f}% "
            "of high-cross pairs — interpret semantic failure mode cautiously."
        )

    frac_min = float(minimal_support_summary.get("fraction_of_high_cross_cohort") or 0)
    if frac_min >= 0.4:
        factors.append(
            f"A large share ({100 * frac_min:.0f}%) of high-cross pairs are in the minimal-explicit-support regime "
            "(unsupported latent / weak visible artifacts)."
        )
        keyed["minimal_explicit_support_fraction"] = frac_min

    for bullet in (profile.get("readable_bullets") or [])[:4]:
        if bullet not in factors:
            factors.append(str(bullet))

    top_tags = likely_explanations.get("top_tags") or []
    if top_tags:
        tag_line = ", ".join(
            f"{t['tag']} ({100 * float(t['fraction']):.0f}% on cross)" for t in top_tags[:5]
        )
        factors.append(f"Automated tags on high-cross: {tag_line}.")

    body_cmp = (comparisons.get("high_cross_vs_high_same") or {}).get("body_path_signal_comparison") or {}
    for note in (body_cmp.get("interpretation_notes") or [])[:3]:
        if note not in factors:
            factors.append(str(note))

    if not factors:
        factors.append("Inspect pair_high_cross_unlabeled_analysis_table.csv and review HTML.")

    headline = (
        "High-cross unlabeled pairs score like rescued same-campaign edges but are GT cross-campaign; "
        "they are usually cross-seed-component with weak shared artifacts and sparse semantic coverage."
    )
    return {
        "headline": headline,
        "distinguishing_factors": factors[:12],
        "key_numeric_signals": keyed,
        "coverage_contrast": {
            "high_cross_semantic_fraction": cross_cov.get("fraction_semantic_cosine_available"),
            "high_same_semantic_fraction": same_cov.get("fraction_semantic_cosine_available"),
        },
    }


def _prepare_high_cross_review_df(df_pairs: pd.DataFrame) -> pd.DataFrame:
    if df_pairs.empty:
        return df_pairs.copy()
    out = df_pairs.copy()
    if "semantic_cosine" not in out.columns and "semantic_cosine_max" in out.columns:
        out["semantic_cosine"] = pd.to_numeric(out["semantic_cosine_max"], errors="coerce")
    if "what_made_it_high" not in out.columns:
        out["what_made_it_high"] = out.get("analysis_cohort", "high_cross_unlabeled")
    if "fp_regime" not in out.columns:
        out["fp_regime"] = out["what_made_it_high"]
    return out


def _export_high_cross_review_html(
    *,
    df_pairs: pd.DataFrame,
    layout: dict[str, Path],
    email_text_by_eid: dict[str, dict[str, str]],
    out_name: str,
    title: str,
    subtitle: str,
    export_flags: ExportFlags,
    review_prompt_fn: Any,
) -> str:
    pse = _pse()
    review_html = layout["review_html"]
    html_path = review_html / out_name
    df = _prepare_high_cross_review_df(df_pairs)
    if df.empty:
        pse._write_pairs_for_review_html(
            df,
            out_path=html_path,
            email_text_by_eid=email_text_by_eid,
            title=title,
            subtitle=subtitle + " (no pairs in cohort)",
        )
        return str(html_path)

    def _regime(row: pd.Series) -> str:
        w = str(row.get("what_made_it_high") or "").strip()
        if w:
            return w
        return str(row.get("analysis_cohort") or "high_cross_unlabeled")

    df_review = pse._enrich_pairs_with_email_text(
        df,
        email_text_by_eid=email_text_by_eid,
        preview_chars=500,
        regime_fn=_regime,
        review_prompt_fn=review_prompt_fn,
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
        review_prompt="High-score unlabeled pair manual review.",
        gt_note="high-score unlabeled",
        filter_column="what_made_it_high",
    )
    if export_flags.emit_debug_csv:
        debug_csv = layout["debug_csv"] / f"debug_{out_name.replace('.html', '.csv')}"
        df_review.to_csv(debug_csv, index=False)
    return str(html_path)


def _build_high_cross_unlabeled_profile(
    *,
    df_cross: pd.DataFrame,
    df_same: pd.DataFrame,
    comparisons: dict[str, Any],
    likely_explanations: dict[str, Any],
    nodes_by_email: dict[str, dict[str, set[str]]],
) -> dict[str, Any]:
    cross_prof = _build_edge_profile(
        cohort_name="high_cross_unlabeled",
        df=df_cross,
        nodes_by_email=nodes_by_email,
    )
    same_prof = _build_edge_profile(
        cohort_name="high_same_unlabeled",
        df=df_same,
        nodes_by_email=nodes_by_email,
    )

    bullets: list[str] = []
    bullets.append(cross_prof.get("bullet_summary", "high_cross: (empty)"))
    bullets.append(same_prof.get("bullet_summary", "high_same: (empty)"))

    main_cmp = comparisons.get("high_cross_vs_high_same") or {}
    marg = main_cmp.get("marginal") or {}
    top_sep = (marg.get("ranked_separators_top15") or [])[:6]
    if top_sep:
        sep_bits = [
            f"{r.get('metric_group')}:{r.get('metric_name')} Δ={r.get('difference_left_minus_right')}"
            for r in top_sep
        ]
        bullets.append(
            "Cross vs same (high band): cross cohort is LEFT; positive Δ means higher on cross — "
            + "; ".join(sep_bits)
        )

    body_cmp = main_cmp.get("body_path_signal_comparison") or {}
    for note in (body_cmp.get("interpretation_notes") or [])[:4]:
        bullets.append(str(note))

    top_tags = likely_explanations.get("top_tags") or []
    if top_tags:
        dominant = ", ".join(
            f"{t['tag']} ({100 * float(t['fraction']):.0f}%)" for t in top_tags[:5]
        )
        bullets.append(f"Dominant failure tags on high-cross: {dominant}.")

    diagnosis = (
        "High-scoring cross-campaign unlabeled pairs are cross-GT edges the scorer ranks like "
        "rescued same-campaign unlabeled edges. "
    )
    if top_tags:
        top0 = top_tags[0]["tag"]
        if "semantic" in top0 or "body_only" in top0:
            diagnosis += (
                "The main failure regime looks like semantic/text or body-only similarity without "
                "campaign-disambiguating explicit support."
            )
        elif "latent" in top0 or "twohop" in top0:
            diagnosis += (
                "The main failure regime looks like graph/latent similarity without explicit disambiguation."
            )
        else:
            diagnosis += f"Top automated tag: {top0}."
    else:
        diagnosis += "Inspect marginal/joint separators and HTML cohort reviews."

    return {
        "n_high_cross_unlabeled": int(len(df_cross)),
        "n_high_same_unlabeled": int(len(df_same)),
        "high_cross_edge_profile": cross_prof,
        "high_same_edge_profile": same_prof,
        "readable_bullets": bullets,
        "readable_diagnosis": diagnosis,
    }


def _generate_high_cross_recommendations(
    *,
    profile: dict[str, Any],
    likely_explanations: dict[str, Any],
    comparisons: dict[str, Any],
    population_diag: dict[str, Any],
    coverage_summary: dict[str, Any],
    minimal_support_summary: dict[str, Any],
    digest: dict[str, Any],
    thresholds: HighCrossThresholds,
    run_encoder_hint: str | None,
) -> dict[str, Any]:
    tags = likely_explanations.get("top_tags") or []
    tag_names = [str(t.get("tag")) for t in tags[:5]]
    interventions: list[str] = []

    why = profile.get("readable_diagnosis") or "See high_cross_unlabeled_profile."
    frac_min = float(minimal_support_summary.get("fraction_of_high_cross_cohort") or 0)
    cross_cov = (coverage_summary.get("by_cohort") or {}).get("high_cross_unlabeled") or {}
    sem_frac = float(cross_cov.get("fraction_semantic_cosine_available") or 0)

    failure_modes: list[str] = []
    if frac_min >= 0.35:
        failure_modes.append("unsupported_or_minimal_explicit_support")
    if sem_frac < 0.35:
        failure_modes.append("semantic_under_observed_on_cross")
    if any("semantic" in t for t in tag_names):
        failure_modes.append("semantic_confusion")
    if any("body_only" in t for t in tag_names):
        failure_modes.append("body_only_text_similarity")
    if any("twohop" in t for t in tag_names):
        failure_modes.append("graph_neighborhood_or_2hop_channels")
    if any("latent" in t for t in tag_names):
        failure_modes.append("support_free_latent_similarity")
    if any("path" in t for t in tag_names):
        failure_modes.append("path_similarity_without_campaign_support")
    if not failure_modes:
        failure_modes.append("mixed_or_unclear — use HTML review")

    primary_failure = failure_modes[0] if failure_modes else "mixed_or_unclear"
    explicit_mostly_absent = frac_min >= 0.35

    if "unsupported_or_minimal_explicit_support" in failure_modes or "support_free_latent_similarity" in failure_modes:
        interventions.append(
            "Teach scorer/GNN to down-rank cross-seed high scores when n_shared_core_channels=0 and shared artifacts are absent."
        )
    if "semantic_under_observed_on_cross" in failure_modes:
        interventions.append(
            "Check latent diagnostics (gnn_encoder_cosine) and pair-table semantic coverage before attributing failure to semantic cosine."
        )
    if "semantic_confusion" in failure_modes or "body_only_text_similarity" in failure_modes:
        interventions.append(
            "Add explicit disambiguation (hybrid GNN+explicit scorer) for community cuts — do not rely on GNN-only for bridges."
        )
    if "graph_neighborhood_or_2hop_channels" in failure_modes:
        interventions.append(
            "Audit noisy 2-hop / component channels; consider graph cleaning or down-weighting generic neighborhoods."
        )
    interventions.append(
        "Do not add bridge edges on high-score cross unlabeled pairs without explicit disambiguation."
    )
    if run_encoder_hint and "explicit_only" in (run_encoder_hint or "").lower():
        interventions.append("N/A — run is explicit-only; re-run on GNN-only ablation for latent FP diagnosis.")
    elif run_encoder_hint and "gnn" in (run_encoder_hint or "").lower():
        interventions.append(
            "GNN-only ablation: add explicit disambiguators to scorer or hybrid ensemble with explicit-only MLP."
        )

    lm_cross = (population_diag.get("latent_diagnostics") or {}).get("high_cross_unlabeled") or {}
    latent_in_pop = (
        (population_diag.get("high_cross_primary_cohorts") or {})
        .get("high_cross_unlabeled", {})
        .get("features", {})
        .get("gnn_encoder_cosine", {})
    )
    missing_latent = not bool(
        lm_cross.get("gnn_encoder_cosine", {}).get("present")
        or latent_in_pop.get("n_non_null", 0)
    )

    next_focus: str
    if explicit_mostly_absent and ("latent" in primary_failure or frac_min >= 0.5):
        next_focus = "reject_cross_seed_unsupported_latent_similarity"
    elif "semantic_confusion" in failure_modes and sem_frac >= 0.2:
        next_focus = "explicit_disambiguation_for_semantic_cross_scores"
    elif missing_latent:
        next_focus = "attach_latent_diagnostics_then_re_review"
    else:
        next_focus = "inspect_high_cross_html_and_minimal_support_cohort"

    return {
        "headline": digest.get("headline") or why,
        "primary_failure_mode": primary_failure,
        "explicit_support_mostly_absent": explicit_mostly_absent,
        "minimal_explicit_support_fraction": frac_min,
        "semantic_coverage_fraction_high_cross": sem_frac,
        "A_why_high_score_cross_errors": why,
        "B_dominant_failure_modes": failure_modes,
        "C_implied_next_steps": interventions[:8],
        "D_recommended_next_focus": next_focus,
        "thresholds": {
            "high_cross_score_min": thresholds.high_cross_score_min,
            "mid_cross_score_min": thresholds.mid_cross_score_min,
        },
        "run_encoder_hint": run_encoder_hint,
        "latent_diagnostics_available": not missing_latent,
    }


def _write_high_cross_artifacts(
    *,
    summary: dict[str, Any],
    table_rows: list[dict[str, Any]],
    joint_payload: dict[str, Any],
    layout: dict[str, Path],
    filename_suffix: str,
    export_flags: ExportFlags,
) -> dict[str, str]:
    core_json = layout["core_json"]
    debug_csv = layout["debug_csv"]
    debug_json = layout["debug_json"]
    suffix = filename_suffix or ""
    summary_path = core_json / f"pair_high_cross_unlabeled_analysis_summary{suffix}.json"
    joint_path = debug_json / f"pair_high_cross_unlabeled_analysis_joint_summary{suffix}.json"
    table_path = debug_csv / f"pair_high_cross_unlabeled_analysis_table{suffix}.csv"

    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    joint_path.write_text(json.dumps(joint_payload, indent=2, default=str), encoding="utf-8")
    pd.DataFrame(table_rows).to_csv(table_path, index=False)

    paths = {
        "summary_path": str(summary_path),
        "joint_summary_path": str(joint_path),
        "table_path": str(table_path),
    }
    if export_flags.emit_debug_json:
        debug_json.joinpath(f"pair_high_cross_unlabeled_analysis_comparisons{suffix}.json").write_text(
            json.dumps(summary.get("comparisons") or {}, indent=2, default=str),
            encoding="utf-8",
        )
    return paths


def run_pair_high_cross_unlabeled_analysis(
    *,
    df_work: pd.DataFrame,
    scores: np.ndarray,
    gt_path: Path,
    label_map: dict[str, Any],
    layout: dict[str, Path],
    nodes_by_email: dict[str, dict[str, set[str]]],
    evidence_index: dict[tuple[str, str], list[dict[str, Any]]] | None,
    email_text_by_eid: dict[str, dict[str, str]] | None,
    thresholds: HighCrossThresholds | None = None,
    export_flags: ExportFlags | None = None,
    filename_suffix: str = "",
    project_root: Path | None = None,
    run_dir: Path | None = None,
    graph_pt: Path | None = None,
    inference_bundle: dict[str, Any] | None = None,
    run_encoder_hint: str | None = None,
) -> dict[str, Any]:
    """Dedicated high-score cross unlabeled analysis for one GT file."""
    flags = export_flags or ExportFlags()
    th = thresholds or HighCrossThresholds()
    gt_path = Path(gt_path).resolve()
    root = (project_root or Path(__file__).resolve().parents[2]).resolve()
    core_json = layout["core_json"]
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

    cohort_masks = high_score_unlabeled_cohort_masks(
        same_eval=same_e,
        cross_eval=cross_e,
        unl_eval=unl_eval[eval_mask],
        scores=scores[eval_mask],
        thresholds=th,
    )
    cohort_counts = {k: int(cohort_masks[k].sum()) for k in HIGH_CROSS_COHORTS}

    gt_rel = {
        "high_cross_unlabeled": "cross_campaign",
        "high_same_unlabeled": "same_campaign",
        "mid_cross_unlabeled": "cross_campaign",
    }
    cohort_dfs: dict[str, pd.DataFrame] = {}
    for cname in HIGH_CROSS_COHORTS:
        df_c = _build_cohort_inspection_df(
            df_eval=df_eval,
            row_mask=cohort_masks[cname],
            gt_path=gt_path,
            label_map=label_map,
            gt_relation=gt_rel[cname],
            cohort=cname,
            nodes_by_email=nodes_by_email,
            evidence_index=evidence_index,
        )
        df_c["analysis_cohort"] = cname
        cohort_dfs[cname] = df_c

    latent_meta: dict[str, Any] = {}
    for cname in ("high_cross_unlabeled", "high_same_unlabeled"):
        cohort_dfs[cname], lm = attach_latent_diagnostics_to_cohort_df(
            cohort_dfs[cname],
            project_root=root,
            run_dir=run_dir,
            graph_pt=graph_pt,
            inference_bundle=inference_bundle,
        )
        latent_meta[cname] = lm

    for _, df_c in cohort_dfs.items():
        if df_c.empty:
            continue
        tags_col = [_row_likely_explanation_tags(r) for _, r in df_c.iterrows()]
        df_c["likely_explanation_tags"] = [",".join(t) for t in tags_col]
        df_c["what_made_it_high"] = [
            t[0] if t else "unclassified" for t in tags_col
        ]

    df_cross = cohort_dfs["high_cross_unlabeled"]
    df_same = cohort_dfs["high_same_unlabeled"]
    df_mid_cross = cohort_dfs["mid_cross_unlabeled"]

    population_diag = _build_focused_high_cross_population_diagnostics(
        cohort_dfs=cohort_dfs,
        df_eval=df_eval,
        latent_meta=latent_meta,
    )
    population_diag["latent_diagnostics"] = latent_meta
    for cname, lm in latent_meta.items():
        if cname in population_diag.get("by_cohort", {}):
            for col in LATENT_FEATURE_KEYS:
                nn = 0
                n_c = int(cohort_counts.get(cname, 0))
                if col in cohort_dfs.get(cname, pd.DataFrame()).columns and n_c:
                    nn = int(pd.to_numeric(cohort_dfs[cname][col], errors="coerce").notna().sum())
                population_diag["by_cohort"][cname].setdefault("latent_features", {})[col] = {
                    "present": col in cohort_dfs.get(cname, pd.DataFrame()).columns,
                    "n_non_null": nn,
                    "fraction_non_null": float(nn / n_c) if n_c else 0.0,
                }

    coverage_summary = _build_semantic_body_path_coverage_summary(
        df_cross=df_cross,
        df_same=df_same,
    )
    minimal_support_summary = _build_high_cross_minimal_explicit_support_summary(
        df_cross=df_cross,
        nodes_by_email=nodes_by_email,
    )

    comparisons: dict[str, Any] = {}
    table_rows: list[dict[str, Any]] = []
    joint_payload: dict[str, Any] = {}

    # Main: high cross vs high same
    marg_main, rows_main = _build_same_vs_cross_marginal(
        gt_path=gt_path,
        same_df=df_same,
        cross_df=df_cross,
        nodes_by_email=nodes_by_email,
        band_kind="high_cross",
    )
    marg_main, extra_main = _extend_marginal_with_full_features(
        marg_main,
        left_df=df_cross,
        right_df=df_same,
        comparison="high_cross_unlabeled_vs_high_same_unlabeled",
        left_label="high_cross",
        right_label="high_same",
        gt_path=gt_path,
        nodes_by_email=nodes_by_email,
    )
    rows_main.extend(extra_main)
    joint_main, jrows_main = _build_frontier_joint_two_cohort(
        gt_path=gt_path,
        left_df=df_cross,
        right_df=df_same,
        left_mask_eval=cohort_masks["high_cross_unlabeled"],
        right_mask_eval=cohort_masks["high_same_unlabeled"],
        df_eval=df_eval,
        comparison="high_cross_unlabeled_vs_high_same_unlabeled",
        band_kind="high_cross",
        value_key_left="high_cross_value",
        value_key_right="high_same_value",
        nodes_by_email=nodes_by_email,
        evidence_index=evidence_index,
        marginal_sep=marg_main,
    )
    body_main = _build_body_path_signal_comparison(
        left_df=df_cross,
        right_df=df_same,
        comparison="high_cross_unlabeled_vs_high_same_unlabeled",
        left_label="high_cross",
        right_label="high_same",
    )
    comparisons["high_cross_vs_high_same"] = {
        "comparison_id": "high_cross_vs_high_same",
        "marginal": marg_main,
        "joint": joint_main,
        "body_path_signal_comparison": body_main,
    }
    table_rows.extend(rows_main)
    table_rows.extend(jrows_main)
    joint_payload["high_cross_vs_high_same"] = joint_main

    # Optional: high cross vs mid cross
    if int(cohort_masks["mid_cross_unlabeled"].sum()) > 0:
        marg_mid, rows_mid = _build_two_cohort_marginal(
            gt_path=gt_path,
            left_df=df_mid_cross,
            right_df=df_cross,
            comparison="mid_cross_unlabeled_vs_high_cross_unlabeled",
            left_label="mid_cross",
            right_label="high_cross",
            nodes_by_email=nodes_by_email,
        )
        marg_mid, extra_mid = _extend_marginal_with_full_features(
            marg_mid,
            left_df=df_mid_cross,
            right_df=df_cross,
            comparison="mid_cross_unlabeled_vs_high_cross_unlabeled",
            left_label="mid_cross",
            right_label="high_cross",
            gt_path=gt_path,
            nodes_by_email=nodes_by_email,
        )
        rows_mid.extend(extra_mid)
        joint_mid, jrows_mid = _build_frontier_joint_two_cohort(
            gt_path=gt_path,
            left_df=df_mid_cross,
            right_df=df_cross,
            left_mask_eval=cohort_masks["mid_cross_unlabeled"],
            right_mask_eval=cohort_masks["high_cross_unlabeled"],
            df_eval=df_eval,
            comparison="mid_cross_unlabeled_vs_high_cross_unlabeled",
            band_kind="mid_vs_high_cross",
            value_key_left="mid_cross_value",
            value_key_right="high_cross_value",
            nodes_by_email=nodes_by_email,
            evidence_index=evidence_index,
            marginal_sep=marg_mid,
        )
        comparisons["mid_cross_vs_high_cross"] = {
            "comparison_id": "mid_cross_vs_high_cross",
            "marginal": marg_mid,
            "joint": joint_mid,
        }
        table_rows.extend(rows_mid)
        table_rows.extend(jrows_mid)
        joint_payload["mid_cross_vs_high_cross"] = joint_mid

    likely_explanations = _aggregate_likely_explanations(df_cross)
    profile = _build_high_cross_unlabeled_profile(
        df_cross=df_cross,
        df_same=df_same,
        comparisons=comparisons,
        likely_explanations=likely_explanations,
        nodes_by_email=nodes_by_email,
    )
    digest = _build_high_cross_vs_high_same_digest(
        comparisons=comparisons,
        coverage_summary=coverage_summary,
        minimal_support_summary=minimal_support_summary,
        profile=profile,
        likely_explanations=likely_explanations,
    )
    recommendations = _generate_high_cross_recommendations(
        profile=profile,
        likely_explanations=likely_explanations,
        comparisons=comparisons,
        population_diag=population_diag,
        coverage_summary=coverage_summary,
        minimal_support_summary=minimal_support_summary,
        digest=digest,
        thresholds=th,
        run_encoder_hint=run_encoder_hint,
    )

    review_paths: dict[str, str] = {}
    catalog = email_text_by_eid or {}
    review_paths["high_cross_unlabeled_for_review_html"] = _export_high_cross_review_html(
        df_pairs=df_cross,
        layout=layout,
        email_text_by_eid=catalog,
        out_name=f"pair_high_cross_unlabeled_for_review{suffix}.html",
        title="High-score cross-campaign unlabeled pairs",
        subtitle=(
            f"GT cross + unlabeled + score >= {th.high_cross_score_min:.2f} — "
            "GNN-only false-positive risk inspection."
        ),
        export_flags=flags,
        review_prompt_fn=_high_cross_review_prompt,
    )
    review_paths["high_same_unlabeled_for_review_html"] = _export_high_cross_review_html(
        df_pairs=df_same,
        layout=layout,
        email_text_by_eid=catalog,
        out_name=f"pair_high_same_unlabeled_for_review{suffix}.html",
        title="High-score same-campaign unlabeled pairs",
        subtitle=(
            f"GT same + unlabeled + score >= {th.high_cross_score_min:.2f} — "
            "comparison cohort for rescued-like edges."
        ),
        export_flags=flags,
        review_prompt_fn=_high_cross_review_prompt,
    )

    summary = {
        "analysis_kind": "pair_high_cross_unlabeled_analysis",
        "gt_path": str(gt_path),
        "thresholds": {
            "high_cross_score_min": th.high_cross_score_min,
            "mid_cross_score_min": th.mid_cross_score_min,
            "definition_high_cross_unlabeled": (
                "GT-covered, pair_status=unlabeled, gt_relation=cross_campaign, "
                f"score >= {th.high_cross_score_min}"
            ),
            "definition_high_same_unlabeled": (
                "GT-covered, pair_status=unlabeled, gt_relation=same_campaign, "
                f"score >= {th.high_cross_score_min}"
            ),
        },
        "cohort_counts": cohort_counts,
        "high_cross_unlabeled_profile": profile,
        "high_cross_likely_explanations": likely_explanations,
        "high_cross_feature_population_diagnostics": population_diag,
        "semantic_body_path_coverage_summary": coverage_summary,
        "high_cross_minimal_explicit_support": minimal_support_summary,
        "high_cross_vs_high_same_digest": digest,
        "comparisons": comparisons,
        "high_cross_unlabeled_recommendations": recommendations,
        "review_html_paths": review_paths,
        "run_encoder_hint": run_encoder_hint,
    }

    artifact_paths = _write_high_cross_artifacts(
        summary=summary,
        table_rows=table_rows,
        joint_payload=joint_payload,
        layout=layout,
        filename_suffix=suffix,
        export_flags=flags,
    )

    run_digest = {
        "n_high_cross_unlabeled": cohort_counts.get("high_cross_unlabeled", 0),
        "n_high_same_unlabeled": cohort_counts.get("high_same_unlabeled", 0),
        "readable_diagnosis": profile.get("readable_diagnosis"),
        "top_failure_tags": (likely_explanations.get("top_tags") or [])[:5],
        "recommendation_headline": recommendations.get("headline"),
        "high_cross_vs_high_same_headline": digest.get("headline"),
        "minimal_explicit_support_fraction": minimal_support_summary.get(
            "fraction_of_high_cross_cohort"
        ),
    }

    return {
        "gt_path": str(gt_path),
        "gt_name": gt_path.name,
        "digest": run_digest,
        "cohort_counts": cohort_counts,
        "paths": {
            **artifact_paths,
            **{k: v for k, v in review_paths.items()},
        },
        "summary_path": artifact_paths["summary_path"],
        "joint_summary_path": artifact_paths["joint_summary_path"],
        "table_path": artifact_paths["table_path"],
        "high_cross_unlabeled_profile": profile,
        "high_cross_unlabeled_recommendations": recommendations,
    }