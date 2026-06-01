"""
Low-band feature discovery with positive-aware alignment.

Scores candidate pair features for:
  - separating low-score same-campaign unlabeled vs low-score cross-campaign unlabeled
  - aligning low-score same-campaign unlabeled toward training positives vs cross unlabeled

Analysis-only (no training / graph / seed changes).
"""

from __future__ import annotations

import argparse
import difflib
import json
import math
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_REPO_ROOT / "core") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "core"))
_GNN_ROOT = _REPO_ROOT / "core" / "GNN"
if str(_GNN_ROOT) not in sys.path:
    sys.path.insert(0, str(_GNN_ROOT))

from seed_candidate_workflow.utils.anchor_graph_helpers import load_anchor_graph_artifacts
from seed_candidate_workflow.utils.pair_score_separation import (
    _load_anchor_nodes_by_email,
    _load_email_text_catalog,
    _resolve_default_misp_json_path,
    load_pair_supervision_for_inference,
    score_pair_rows,
)
from seed_candidate_workflow.utils.raw_gnn_notebook import load_ground_truth_structures


_TOKEN_RE = re.compile(r"[a-z0-9]+", re.I)
_SENDER_DISPLAY_RE = re.compile(r"^(.+?)\s*<([^>]+)>$")
_DIGITS_RE = re.compile(r"\d+")
_ROOT_STEM = "/"
_STRONG_CHANNELS = ("sender_set", "url_set", "attachment_set", "sender_email_domain_set", "stem_set")
_WEAK_DOMAIN_COL = "domain_set"
_INFORMATIVE_BODY_MIN_DF_FRAC = 0.02                                          
_LOW_BAND_SEP_MIN = 0.05
_ALIGNMENT_MARGIN_MIN = 0.02
_EXCLUDE_FROM_TOP_RECOMMENDATIONS = frozenset(
    {
        "semantic_cosine_max",
        "time_gap_seconds_min",
        "shared_sender",
        "shared_domain",
        "shared_stem_any",
        "score",
        "anchor_context_missing",
    }
)


def _tokenize(text: str, *, min_len: int = 2) -> set[str]:
    return {t.lower() for t in _TOKEN_RE.findall(str(text or "")) if len(t) >= min_len}


def _char_ngrams(text: str, n: int) -> set[str]:
    s = re.sub(r"\s+", " ", str(text or "").lower()).strip()
    if len(s) < n:
        return set()
    return {s[i : i + n] for i in range(len(s) - n + 1)}


def _jaccard(a: set[str], b: set[str]) -> float | None:
    if not a and not b:
        return None
    u = a | b
    if not u:
        return None
    return float(len(a & b) / len(u))


def _overlap_count(a: set[str], b: set[str]) -> int:
    return int(len(a & b))


def _len_ratio(a: str, b: str) -> float | None:
    la, lb = len(a), len(b)
    if la == 0 and lb == 0:
        return None
    hi = max(la, lb, 1)
    return float(min(la, lb) / hi)


def _levenshtein_ratio(a: str, b: str) -> float | None:
    if not a and not b:
        return None
    if not a or not b:
        return 0.0
    return float(difflib.SequenceMatcher(None, a, b).ratio())


def _normalize_subject(s: str) -> str:
    s = str(s or "").lower().strip()
    s = re.sub(r"^(re|fw|fwd):\s*", "", s, flags=re.I)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def _parse_sender_parts(sender: str) -> tuple[str, str, str]:
    """Return (local_part, domain, display_name)."""
    s = str(sender or "").strip()
    m = _SENDER_DISPLAY_RE.match(s)
    if m:
        display = m.group(1).strip().strip('"')
        addr = m.group(2).strip().lower()
    else:
        display = ""
        addr = s.lower()
    if "@" in addr:
        local, dom = addr.split("@", 1)
    else:
        local, dom = addr, ""
    return local, dom, display


def _normalize_localpart(local: str) -> str:
    return _DIGITS_RE.sub("", str(local or "").lower())


def _parse_url_path_tokens(url: str) -> tuple[str, list[str], int]:
    """Return (registrable_domain, path_tokens, path_depth)."""
    from core.feature_set_extraction.url_extraction_utils import parse_url_host_and_registrable_domain
    from core.preprocessing.utils.url_extractor import parse_url_components

    u = str(url or "").strip()
    if not u:
        return "", [], 0
    _host, reg, ok = parse_url_host_and_registrable_domain(u)
    reg = reg.lower() if ok else ""
    comp = parse_url_components(u)
    stem = str(comp.get("stem") or "").strip()
    parts = [p for p in stem.split("/") if p and p != _ROOT_STEM]
    depth = len(parts)
    tokens: list[str] = []
    for p in parts:
        p_norm = re.sub(r"\d{4,}", "<id>", p.lower())
        for t in re.split(r"[/_.-]+", p_norm):
            t = t.strip()
            if t and t not in ("<id>",):
                tokens.append(t)
    return reg, tokens, depth


def _nontrivial_stems(stems: set[str]) -> set[str]:
    return {s for s in stems if s and s != _ROOT_STEM}


def _load_anchor_nodes_extended(
    *,
    pair_csv: Path,
    project_root: Path,
    anchor_run_dir: Path | None,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    base, meta = _load_anchor_nodes_by_email(
        pair_csv=pair_csv, project_root=project_root, explicit_anchor_run_dir=anchor_run_dir
    )
    if meta.get("status") != "ok":
        return {}, meta
    run_dir = Path(str(meta["anchor_run_dir"]))
    nodes_df, _, _, _, _ = load_anchor_graph_artifacts(run_dir, load_graph_pickle=False)
    ts_by_eid: dict[str, float] = {}
    if "external_id" in nodes_df.columns and "ts" in nodes_df.columns:
        for _, r in nodes_df[["external_id", "ts"]].iterrows():
            eid = str(r["external_id"]).strip()
            v = pd.to_numeric(r["ts"], errors="coerce")
            ts_by_eid[eid] = float(v) if pd.notna(v) else float("nan")
    out: dict[str, dict[str, Any]] = {}
    for eid, row in base.items():
        ext = dict(row)
        ext["ts"] = ts_by_eid.get(eid, float("nan"))
        out[eid] = ext
    meta["has_ts"] = bool(ts_by_eid)
    return out, meta


def _build_artifact_df_maps(nodes_by_email: dict[str, dict[str, Any]]) -> dict[str, dict[str, int]]:
    """Per-column value -> document frequency (email count)."""
    cols = [
        "url_set",
        "sender_set",
        "attachment_set",
        "sender_email_domain_set",
        "domain_set",
        "stem_set",
    ]
    maps: dict[str, Counter[str]] = {c: Counter() for c in cols}
    for row in nodes_by_email.values():
        for c in cols:
            for v in row.get(c) or set():
                maps[c][str(v)] += 1
    n = max(1, len(nodes_by_email))
    return {c: {k: int(v) for k, v in ctr.items()} for c, ctr in maps.items()}


def _idf_weight(df_count: int, n_docs: int) -> float:
    return float(math.log((1.0 + n_docs) / (1.0 + max(1, df_count))))


def _pair_channel_sets(
    ei: str, ej: str, nodes: dict[str, dict[str, Any]]
) -> dict[str, set[str]] | None:
    na = nodes.get(ei)
    nb = nodes.get(ej)
    if na is None or nb is None:
        return None
    return {c: set(na.get(c) or set()) & set(nb.get(c) or set()) for c in na if c.endswith("_set")}


def _compute_pair_features_row(
    *,
    ei: str,
    ej: str,
    row: pd.Series,
    nodes: dict[str, dict[str, Any]],
    text_i: dict[str, str],
    text_j: dict[str, str],
    df_maps: dict[str, dict[str, int]],
    body_rare_tokens: set[str],
    n_docs: int,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    ch = _pair_channel_sets(ei, ej, nodes)
    if ch is None:
        out["anchor_context_missing"] = True
        return out
    out["anchor_context_missing"] = False

    sem = float(pd.to_numeric(row.get("semantic_cosine_max"), errors="coerce"))
    out["semantic_cosine_max"] = sem if np.isfinite(sem) else np.nan
    tg = float(pd.to_numeric(row.get("time_gap_seconds_min"), errors="coerce"))
    out["time_gap_seconds_min"] = tg if np.isfinite(tg) else np.nan

    shared_url = ch.get("url_set") or set()
    shared_sender = ch.get("sender_set") or set()
    shared_attachment = ch.get("attachment_set") or set()
    shared_sdom = ch.get("sender_email_domain_set") or set()
    shared_domain = ch.get("domain_set") or set()
    shared_stem_raw = ch.get("stem_set") or set()
    shared_stem_nt = _nontrivial_stems(shared_stem_raw)

    out["shared_sender"] = int(len(shared_sender) > 0)
    out["shared_attachment"] = int(len(shared_attachment) > 0)
    out["shared_url"] = int(len(shared_url) > 0)
    out["shared_sender_domain"] = int(len(shared_sdom) > 0)
    out["shared_domain"] = int(len(shared_domain) > 0)
    out["shared_stem_any"] = int(len(shared_stem_raw) > 0)
    out["shared_stem_nontrivial"] = int(len(shared_stem_nt) > 0)
    out["shared_stem_only_slash"] = int(shared_stem_raw == {_ROOT_STEM})
    out["shared_stem_nontrivial_count"] = int(len(shared_stem_nt))

    strong = sum(
        [
            len(shared_sender) > 0,
            len(shared_attachment) > 0,
            len(shared_url) > 0,
            len(shared_sdom) > 0,
            len(shared_stem_nt) > 0,
        ]
    )
    out["strong_support_count"] = int(strong)
    domain_only = int(len(shared_domain) > 0 and strong == 0)
    out["shared_domain_without_strong_support"] = domain_only
    out["support_count_excl_domain_and_root_stem"] = int(
        sum(
            [
                len(shared_sender) > 0,
                len(shared_attachment) > 0,
                len(shared_url) > 0,
                len(shared_sdom) > 0,
                len(shared_stem_nt) > 0,
            ]
        )
    )
    out["shared_url_or_stem_without_sender"] = int(
        (len(shared_url) > 0 or len(shared_stem_nt) > 0) and len(shared_sender) == 0
    )

    rw = 0.0
    for col, inter in (
        ("sender_set", shared_sender),
        ("attachment_set", shared_attachment),
        ("url_set", shared_url),
        ("sender_email_domain_set", shared_sdom),
        ("stem_set", shared_stem_nt),
    ):
        for v in inter:
            df_c = (df_maps.get(col) or {}).get(v, 1)
            rw += _idf_weight(df_c, n_docs)
    out["rarity_weighted_support_sum"] = float(rw)

    si = _normalize_subject(text_i.get("subject", ""))
    sj = _normalize_subject(text_j.get("subject", ""))
    ti = _tokenize(si, min_len=1)
    tj = _tokenize(sj, min_len=1)
    out["subject_normalized_exact_match"] = int(si == sj and bool(si))
    out["subject_token_jaccard"] = _jaccard(ti, tj)
    out["subject_token_overlap_count"] = float(_overlap_count(ti, tj))
    out["subject_len_ratio"] = _len_ratio(si, sj)
    out["subject_char3gram_jaccard"] = _jaccard(_char_ngrams(si, 3), _char_ngrams(sj, 3))
    out["subject_levenshtein_ratio"] = _levenshtein_ratio(si, sj)

    bi = str(text_i.get("body") or "")
    bj = str(text_j.get("body") or "")
    from seed_candidate_workflow.utils.pair_similarity_features import (
        body_char4gram_jaccard_from_bodies,
        body_token_jaccard_from_bodies,
        tokenize_text,
    )

    bt_i = tokenize_text(bi, min_len=2)
    bt_j = tokenize_text(bj, min_len=2)
    out["body_token_jaccard"] = body_token_jaccard_from_bodies(bi, bj)
    out["body_token_overlap_count"] = float(_overlap_count(bt_i, bt_j))
    out["body_char4gram_jaccard"] = body_char4gram_jaccard_from_bodies(bi, bj)
    br_i = bt_i & body_rare_tokens
    br_j = bt_j & body_rare_tokens
    out["body_rare_token_jaccard"] = _jaccard(br_i, br_j)

    url_tokens_i: list[str] = []
    url_tokens_j: list[str] = []
    regs_i: set[str] = set()
    regs_j: set[str] = set()
    depths_i: list[int] = []
    depths_j: list[int] = []
    na = nodes.get(ei) or {}
    nb = nodes.get(ej) or {}
    for u in na.get("url_set") or set():
        reg, toks, dep = _parse_url_path_tokens(u)
        if reg:
            regs_i.add(reg)
        url_tokens_i.extend(toks)
        depths_i.append(dep)
    for u in nb.get("url_set") or set():
        reg, toks, dep = _parse_url_path_tokens(u)
        if reg:
            regs_j.add(reg)
        url_tokens_j.extend(toks)
        depths_j.append(dep)
    path_i = set(url_tokens_i)
    path_j = set(url_tokens_j)
    stem_tok_i = set()
    stem_tok_j = set()
    for st in _nontrivial_stems(na.get("stem_set") or set()):
        for t in re.split(r"[/_.-]+", st.lower()):
            if t:
                stem_tok_i.add(t)
    for st in _nontrivial_stems(nb.get("stem_set") or set()):
        for t in re.split(r"[/_.-]+", st.lower()):
            if t:
                stem_tok_j.add(t)
    path_union_i = path_i | stem_tok_i
    path_union_j = path_j | stem_tok_j
    from seed_candidate_workflow.utils.pair_similarity_features import (
        path_token_jaccard_combined_for_nodes,
        sender_localpart_norm_jaccard_for_nodes,
    )

    out["same_registrable_domain"] = int(bool(regs_i & regs_j))
    out["url_path_token_jaccard"] = _jaccard(path_i, path_j)
    out["stem_path_token_jaccard"] = _jaccard(stem_tok_i, stem_tok_j)
    out["path_token_jaccard_combined"] = path_token_jaccard_combined_for_nodes(na, nb)
    out["path_token_overlap_count"] = float(_overlap_count(path_union_i, path_union_j))
    if depths_i and depths_j:
        out["same_path_depth"] = int(min(depths_i) == min(depths_j))
    else:
        out["same_path_depth"] = np.nan
    norm_urls_i = {str(u).strip().lower() for u in (na.get("url_set") or set())}
    norm_urls_j = {str(u).strip().lower() for u in (nb.get("url_set") or set())}
    out["normalized_url_jaccard"] = _jaccard(norm_urls_i, norm_urls_j)

    def _first_sender(row: dict[str, Any]) -> str:
        ss = row.get("sender_set") or set()
        return str(next(iter(ss), "")) if ss else ""

    li_i, di_i, disp_i = _parse_sender_parts(_first_sender(na))
    li_j, dj_j, disp_j = _parse_sender_parts(_first_sender(nb))
    out["sender_exact_match"] = int(len(shared_sender) > 0)
    out["sender_localpart_exact_match"] = int(bool(li_i and li_j and li_i == li_j))
    out["sender_domain_exact_match"] = int(bool(di_i and dj_j and di_i == dj_j))
    out["sender_localpart_norm_jaccard"] = sender_localpart_norm_jaccard_for_nodes(na, nb)
    out["sender_display_jaccard"] = _jaccard(
        _tokenize(disp_i, min_len=1), _tokenize(disp_j, min_len=1)
    )

    tsi = float(na.get("ts", float("nan")))
    tsj = float(nb.get("ts", float("nan")))
    if np.isfinite(tsi) and np.isfinite(tsj) and tsi > 0 and tsj > 0:
        gap = abs(tsi - tsj)
        out["anchor_time_gap_seconds"] = float(gap)
        out["log_anchor_time_gap_seconds"] = float(math.log1p(gap))
        dti = datetime.fromtimestamp(tsi, tz=timezone.utc)
        dtj = datetime.fromtimestamp(tsj, tz=timezone.utc)
        out["same_day_utc"] = int(dti.date() == dtj.date())
        out["same_week_utc"] = int(dti.isocalendar()[:2] == dtj.isocalendar()[:2])
    else:
        out["anchor_time_gap_seconds"] = np.nan
        out["log_anchor_time_gap_seconds"] = np.nan
        out["same_day_utc"] = np.nan
        out["same_week_utc"] = np.nan
    if np.isfinite(tg):
        out["log_time_gap_seconds_min"] = float(math.log1p(tg))
    else:
        out["log_time_gap_seconds_min"] = np.nan

    out["ch_shared_sender_or_attachment"] = int(
        len(shared_sender) > 0 or len(shared_attachment) > 0
    )
    out["ch_shared_stem_and_not_sender"] = int(len(shared_stem_nt) > 0 and len(shared_sender) == 0)
    out["ch_shared_domain_without_strong_support"] = int(domain_only)
    out["ch_semantic_ge_0_90"] = int(np.isfinite(sem) and sem >= 0.90)
    out["ch_semantic_ge_0_93"] = int(np.isfinite(sem) and sem >= 0.93)
    weak_only = int(
        strong == 0
        and (
            len(shared_domain) > 0
            or (np.isfinite(sem) and sem >= 0.90)
        )
    )
    out["ch_semantic_ge_0_90_and_weak_support_only"] = int(
        np.isfinite(sem) and sem >= 0.90 and strong == 0
    )
    out["ch_semantic_ge_0_90_and_strong_support"] = int(
        np.isfinite(sem) and sem >= 0.90 and strong >= 1
    )

    return out


def _build_body_rare_token_set(
    email_text: dict[str, dict[str, str]], *, max_df_frac: float = _INFORMATIVE_BODY_MIN_DF_FRAC
) -> set[str]:
    ctr: Counter[str] = Counter()
    for rec in email_text.values():
        ctr.update(_tokenize(str(rec.get("body") or ""), min_len=2))
    if not ctr:
        return set()
    n = len(email_text)
    cutoff = max(1, int(n * max_df_frac))
    return {t for t, c in ctr.items() if c <= cutoff}


def _compare_feature_series(
    same: pd.Series, cross: pd.Series, *, feature: str, family: str
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "feature": feature,
        "feature_family": family,
        "n_same": int(len(same)),
        "n_cross": int(len(cross)),
    }
    if len(same) == 0 and len(cross) == 0:
        row["verdict"] = "no_data"
        return row

    if same.dtype == bool or cross.dtype == bool or feature.startswith("ch_") or feature.endswith("_match"):
        ss = pd.to_numeric(same, errors="coerce").fillna(0).astype(float)
        cs = pd.to_numeric(cross, errors="coerce").fillna(0).astype(float)
        same_rate = float(ss.mean()) if len(ss) else None
        cross_rate = float(cs.mean()) if len(cs) else None
        row["same_mean"] = same_rate
        row["cross_mean"] = cross_rate
        row["same_median"] = same_rate
        row["cross_median"] = cross_rate
        row["difference_mean_same_minus_cross"] = (
            (same_rate - cross_rate) if (same_rate is not None and cross_rate is not None) else None
        )
        row["enrichment_same_over_cross"] = (
            (same_rate / cross_rate)
            if (same_rate is not None and cross_rate is not None and cross_rate > 0)
            else None
        )
        row["abs_difference_mean"] = (
            abs(row["difference_mean_same_minus_cross"])
            if row["difference_mean_same_minus_cross"] is not None
            else None
        )
        row["verdict"] = _verdict_from_diff(row["difference_mean_same_minus_cross"])
        return row

    s = pd.to_numeric(same, errors="coerce")
    c = pd.to_numeric(cross, errors="coerce")
    s_ok = s[s.notna()]
    c_ok = c[c.notna()]
    row["n_same_non_null"] = int(s_ok.shape[0])
    row["n_cross_non_null"] = int(c_ok.shape[0])
    row["same_mean"] = float(s_ok.mean()) if not s_ok.empty else None
    row["cross_mean"] = float(c_ok.mean()) if not c_ok.empty else None
    row["same_median"] = float(s_ok.median()) if not s_ok.empty else None
    row["cross_median"] = float(c_ok.median()) if not c_ok.empty else None
    if row["same_mean"] is not None and row["cross_mean"] is not None:
        row["difference_mean_same_minus_cross"] = float(row["same_mean"] - row["cross_mean"])
        row["enrichment_same_over_cross"] = (
            float(row["same_mean"] / row["cross_mean"]) if row["cross_mean"] != 0 else None
        )
        row["abs_difference_mean"] = abs(row["difference_mean_same_minus_cross"])
    else:
        row["difference_mean_same_minus_cross"] = None
        row["enrichment_same_over_cross"] = None
        row["abs_difference_mean"] = None
    row["verdict"] = _verdict_from_diff(row.get("difference_mean_same_minus_cross"))
    return row


def _verdict_from_diff(diff: float | None) -> str:
    if diff is None:
        return "insufficient_data"
    if diff > _LOW_BAND_SEP_MIN:
        return "promising_same_enriched"
    if diff < -_LOW_BAND_SEP_MIN:
        return "promising_cross_enriched"
    return "weak_separator"


def _series_stats(series: pd.Series) -> dict[str, Any]:
    v = pd.to_numeric(series, errors="coerce")
    ok = v[v.notna()]
    return {
        "n": int(len(series)),
        "n_non_null": int(ok.shape[0]),
        "mean": float(ok.mean()) if not ok.empty else None,
        "median": float(ok.median()) if not ok.empty else None,
    }


def _alignment_from_means(
    *,
    low_same_mean: float | None,
    positive_mean: float | None,
    cross_unlabeled_mean: float | None,
    low_cross_mean: float | None,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "abs_low_same_minus_positive_mean": None,
        "abs_low_same_minus_cross_unlabeled_mean": None,
        "abs_low_same_minus_low_cross_mean": None,
        "low_same_is_closer_to_positive_than_to_cross_unlabeled": None,
        "low_same_is_closer_to_positive_than_to_low_cross": None,
        "alignment_margin_vs_cross_unlabeled": None,
        "alignment_margin_vs_low_cross": None,
    }
    if low_same_mean is None:
        return out

    def _abs_dist(a: float | None, b: float | None) -> float | None:
        if a is None or b is None:
            return None
        return float(abs(a - b))

    d_pos = _abs_dist(low_same_mean, positive_mean)
    d_cross = _abs_dist(low_same_mean, cross_unlabeled_mean)
    d_low_cross = _abs_dist(low_same_mean, low_cross_mean)
    out["abs_low_same_minus_positive_mean"] = d_pos
    out["abs_low_same_minus_cross_unlabeled_mean"] = d_cross
    out["abs_low_same_minus_low_cross_mean"] = d_low_cross

    if d_pos is not None and d_cross is not None:
        out["low_same_is_closer_to_positive_than_to_cross_unlabeled"] = bool(d_pos < d_cross)
        out["alignment_margin_vs_cross_unlabeled"] = float(d_cross - d_pos)
    if d_pos is not None and d_low_cross is not None:
        out["low_same_is_closer_to_positive_than_to_low_cross"] = bool(d_pos < d_low_cross)
        out["alignment_margin_vs_low_cross"] = float(d_low_cross - d_pos)
    return out


def _combined_recommendation_score(
    *,
    low_band_separation: float | None,
    alignment_margin: float | None,
    positive_mean: float | None,
    low_same_mean: float | None,
    cross_unlabeled_mean: float | None,
    low_cross_mean: float | None,
) -> float | None:
    if low_band_separation is None:
        return None
    scale_vals = [
        abs(x)
        for x in (positive_mean, low_same_mean, cross_unlabeled_mean, low_cross_mean, low_band_separation)
        if x is not None and np.isfinite(x)
    ]
    scale = max(1.0, *scale_vals) if scale_vals else 1.0
    norm_sep = float(low_band_separation) / scale
    norm_align = float(max(0.0, alignment_margin or 0.0)) / scale
    return float(0.55 * norm_sep + 0.45 * norm_align)


def _build_feature_analysis_row(
    *,
    feature: str,
    family: str,
    feat_df: pd.DataFrame,
    group_masks: dict[str, np.ndarray],
) -> dict[str, Any]:
    low_same = feat_df.iloc[group_masks["low_same_unlabeled"]][feature]
    low_cross = feat_df.iloc[group_masks["low_cross_unlabeled"]][feature]
    low_cmp = _compare_feature_series(low_same, low_cross, feature=feature, family=family)

    row: dict[str, Any] = {
        "feature": feature,
        "feature_family": family,
    }
    row.update({f"low_band_{k}": v for k, v in low_cmp.items() if k not in ("feature", "feature_family")})

    group_keys = (
        "positive",
        "same_unlabeled",
        "cross_unlabeled",
        "low_same_unlabeled",
        "low_cross_unlabeled",
        "high_same_unlabeled",
        "high_cross_unlabeled",
    )
    stats_by_group: dict[str, dict[str, Any]] = {}
    for gk in group_keys:
        st = _series_stats(feat_df.iloc[group_masks[gk]][feature])
        stats_by_group[gk] = st
        prefix = gk
        row[f"{prefix}_n"] = st["n"]
        row[f"{prefix}_n_non_null"] = st["n_non_null"]
        row[f"{prefix}_mean"] = st["mean"]
        row[f"{prefix}_median"] = st["median"]

    pos_m = stats_by_group["positive"]["mean"]
    same_m = stats_by_group["same_unlabeled"]["mean"]
    cross_m = stats_by_group["cross_unlabeled"]["mean"]
    low_same_m = stats_by_group["low_same_unlabeled"]["mean"]
    low_cross_m = stats_by_group["low_cross_unlabeled"]["mean"]

    row["low_band_separation_abs_mean_diff"] = row.get("low_band_abs_difference_mean")
    row["low_band_same_minus_cross_mean"] = row.get("low_band_difference_mean_same_minus_cross")
    row["low_band_verdict"] = row.get("low_band_verdict")

    align = _alignment_from_means(
        low_same_mean=low_same_m,
        positive_mean=pos_m,
        cross_unlabeled_mean=cross_m,
        low_cross_mean=low_cross_m,
    )
    row.update(align)

    row["combined_recommendation_score"] = _combined_recommendation_score(
        low_band_separation=row.get("low_band_separation_abs_mean_diff"),
        alignment_margin=row.get("alignment_margin_vs_cross_unlabeled"),
        positive_mean=pos_m,
        low_same_mean=low_same_m,
        cross_unlabeled_mean=cross_m,
        low_cross_mean=low_cross_m,
    )

    margin = row.get("alignment_margin_vs_cross_unlabeled")
    sep = row.get("low_band_separation_abs_mean_diff")
    closer = row.get("low_same_is_closer_to_positive_than_to_cross_unlabeled")
    verdict = row.get("low_band_verdict")

    separates = bool(
        verdict == "promising_same_enriched"
        or (sep is not None and sep >= _LOW_BAND_SEP_MIN)
    )
    aligned = bool(
        closer is True
        and margin is not None
        and margin > _ALIGNMENT_MARGIN_MIN
    )

    if separates and aligned:
        row["feature_recommendation_category"] = "high_confidence_add"
    elif separates:
        row["feature_recommendation_category"] = "separator_only_not_positive_aligned"
    else:
        row["feature_recommendation_category"] = "too_weak_or_unstable"

    row["positive_aligned"] = aligned
    row["low_band_separates"] = separates
    return row


def _generate_alignment_recommendations(
    table: pd.DataFrame,
    *,
    counts: dict[str, int],
) -> dict[str, Any]:
    ranked = table.sort_values(
        "combined_recommendation_score",
        ascending=False,
        na_position="last",
    )
    ranked = ranked[
        ~ranked["feature"].isin(_EXCLUDE_FROM_TOP_RECOMMENDATIONS)
        & ranked["combined_recommendation_score"].notna()
    ]

    high_conf = ranked[ranked["feature_recommendation_category"] == "high_confidence_add"]
    sep_only = ranked[ranked["feature_recommendation_category"] == "separator_only_not_positive_aligned"]
    too_weak = ranked[ranked["feature_recommendation_category"] == "too_weak_or_unstable"]

    def _pack(sub: pd.DataFrame, n: int) -> list[dict[str, Any]]:
        cols = [
            "feature",
            "feature_family",
            "combined_recommendation_score",
            "low_band_separation_abs_mean_diff",
            "alignment_margin_vs_cross_unlabeled",
            "low_same_is_closer_to_positive_than_to_cross_unlabeled",
            "positive_mean",
            "low_same_unlabeled_mean",
            "cross_unlabeled_mean",
            "low_cross_unlabeled_mean",
            "low_band_verdict",
        ]
        cols = [c for c in cols if c in sub.columns]
        return sub.head(n)[cols].to_dict(orient="records")

    best_next = _pack(high_conf, 8)
    if len(best_next) < 3:
        extra = _pack(
            sep_only[~sep_only["feature"].isin({r["feature"] for r in best_next})],
            6 - len(best_next),
        )
        best_next = best_next + extra

    return {
        "cohort_counts": counts,
        "feature_alignment_recommendations": {
            "high_confidence_features_to_add": _pack(high_conf, 12),
            "separator_only_not_clearly_positive_aligned": _pack(sep_only, 12),
            "too_weak_or_unstable": _pack(too_weak, 15),
        },
        "best_candidate_features_to_add_next": best_next[:6],
        "ranking_note": (
            "combined_recommendation_score = 0.55 * normalized_low_band_separation "
            "+ 0.45 * max(0, normalized_alignment_margin_vs_cross_unlabeled). "
            "high_confidence_add requires low-band separation and low-same closer to positives than cross unlabeled."
        ),
    }


def _generate_recommendations(
    table: pd.DataFrame,
    scorecard: pd.DataFrame,
    alignment: dict[str, Any],
) -> dict[str, Any]:
    prom = table[table.get("low_band_verdict", table.get("verdict")) == "promising_same_enriched"].copy()
    if "low_band_separation_abs_mean_diff" in prom.columns:
        prom = prom.sort_values("low_band_separation_abs_mean_diff", ascending=False, na_position="last")
    elif "abs_difference_mean" in prom.columns:
        prom = prom.sort_values("abs_difference_mean", ascending=False, na_position="last")

    weak = table[
        table.get("feature_recommendation_category", pd.Series(dtype=str)) == "too_weak_or_unstable"
    ]

    top_new: list[dict[str, Any]] = []
    ranked = table.sort_values("combined_recommendation_score", ascending=False, na_position="last")
    for _, r in ranked.iterrows():
        feat = str(r["feature"])
        if feat in _EXCLUDE_FROM_TOP_RECOMMENDATIONS:
            continue
        if r.get("feature_recommendation_category") != "high_confidence_add":
            continue
        top_new.append(
            {
                "feature": feat,
                "feature_family": r.get("feature_family"),
                "combined_recommendation_score": r.get("combined_recommendation_score"),
                "low_band_separation_abs_mean_diff": r.get("low_band_separation_abs_mean_diff"),
                "alignment_margin_vs_cross_unlabeled": r.get("alignment_margin_vs_cross_unlabeled"),
                "positive_mean": r.get("positive_mean"),
                "low_same_unlabeled_mean": r.get("low_same_unlabeled_mean"),
                "cross_unlabeled_mean": r.get("cross_unlabeled_mean"),
            }
        )
        if len(top_new) >= 8:
            break

    composite: list[str] = []
    for name in (
        "ch_semantic_ge_0_90_and_strong_support",
        "ch_shared_sender_or_attachment",
        "shared_url_or_stem_without_sender",
        "shared_domain_without_strong_support",
    ):
        sub = table[table["feature"] == name]
        if not sub.empty and sub.iloc[0].get("feature_recommendation_category") == "high_confidence_add":
            composite.append(name)

    refine_existing: list[dict[str, Any]] = []
    for feat, msg in (
        ("shared_stem_only_slash", "Exclude trivial '/' stem overlap from stem-based features and support counts."),
        (
            "shared_domain_without_strong_support",
            "Plain shared domain without sender/url/stem is cross-enriched; use as negative pattern.",
        ),
        ("semantic_cosine_max", "Semantic cosine alone is insufficient in the low band; combine with subject/path/support."),
        ("shared_stem_nontrivial", "Nontrivial stem overlap is a strong same-campaign signal in the low band."),
        ("subject_token_jaccard", "Subject similarity separates low-score same vs cross and aligns low-same toward positives."),
        ("path_token_jaccard_combined", "URL/path token overlap enriches low-same toward positives."),
        ("strong_support_count", "Strong-channel support count aligns low-same with positives."),
    ):
        sub = table[table["feature"] == feat]
        if not sub.empty:
            refine_existing.append({"feature": feat, "note": msg, "stats": sub.iloc[0].to_dict()})

    best_thr: list[dict[str, Any]] = []
    if not scorecard.empty:
        sc = scorecard.copy()
        sc = sc[sc["n_same_captured"].fillna(0).astype(float) >= 3]
        sc = sc.sort_values(
            ["lift_vs_cross_rate", "same_capture_rate"],
            ascending=[False, False],
            na_position="last",
        )
        best_thr = sc.head(12).to_dict(orient="records")

    legacy = {
        "top_new_candidate_mlp_features": top_new[:8],
        "top_composite_patterns_to_add": composite,
        "existing_features_to_refine_or_downweight": refine_existing,
        "promising_numeric_thresholds": best_thr,
        "features_unlikely_to_help": weak.head(10).to_dict(orient="records"),
    }
    legacy.update(alignment)
    legacy["implementation_priority"] = [
        f"{i + 1}. Add {r['feature']} (combined_score={r.get('combined_recommendation_score')})"
        for i, r in enumerate(alignment.get("best_candidate_features_to_add_next", [])[:5])
    ] or [
        "1. Re-run analysis; no high-confidence positive-aligned features met thresholds.",
    ]
    return legacy


def _threshold_scorecard(
    same: pd.Series,
    cross: pd.Series,
    *,
    feature: str,
    thresholds: list[float],
) -> list[dict[str, Any]]:
    s = pd.to_numeric(same, errors="coerce")
    c = pd.to_numeric(cross, errors="coerce")
    ns = int(s.notna().sum())
    nc = int(c.notna().sum())
    rows: list[dict[str, Any]] = []
    for thr in thresholds:
        hit_s = s >= thr
        hit_c = c >= thr
        n_s = int(hit_s.sum())
        n_c = int(hit_c.sum())
        prec = float(n_s / max(1, n_s + n_c))
        rows.append(
            {
                "feature": feature,
                "threshold": float(thr),
                "n_same_captured": n_s,
                "n_cross_captured": n_c,
                "same_capture_rate": float(n_s / ns) if ns > 0 else None,
                "cross_capture_rate": float(n_c / nc) if nc > 0 else None,
                "precision_among_captured": prec,
                "lift_vs_cross_rate": (
                    float((n_s / ns) / (n_c / nc))
                    if ns > 0 and nc > 0 and n_c > 0
                    else None
                ),
            }
        )
    return rows


_FEATURE_FAMILIES: dict[str, str] = {
    "subject_normalized_exact_match": "subject",
    "subject_token_jaccard": "subject",
    "subject_token_overlap_count": "subject",
    "subject_len_ratio": "subject",
    "subject_char3gram_jaccard": "subject",
    "subject_levenshtein_ratio": "subject",
    "body_token_jaccard": "body_text",
    "body_token_overlap_count": "body_text",
    "body_char4gram_jaccard": "body_text",
    "body_rare_token_jaccard": "body_text",
    "same_registrable_domain": "url_path",
    "url_path_token_jaccard": "url_path",
    "stem_path_token_jaccard": "url_path",
    "path_token_jaccard_combined": "url_path",
    "path_token_overlap_count": "url_path",
    "same_path_depth": "url_path",
    "normalized_url_jaccard": "url_path",
    "shared_stem_nontrivial_count": "url_path",
    "shared_stem_only_slash": "url_path",
    "sender_exact_match": "sender",
    "sender_localpart_exact_match": "sender",
    "sender_domain_exact_match": "sender",
    "sender_localpart_norm_jaccard": "sender",
    "sender_display_jaccard": "sender",
    "log_time_gap_seconds_min": "temporal",
    "log_anchor_time_gap_seconds": "temporal",
    "same_day_utc": "temporal",
    "same_week_utc": "temporal",
    "anchor_time_gap_seconds": "temporal",
    "strong_support_count": "rarity_support",
    "rarity_weighted_support_sum": "rarity_support",
    "support_count_excl_domain_and_root_stem": "rarity_support",
    "shared_domain_without_strong_support": "rarity_support",
    "shared_url_or_stem_without_sender": "rarity_support",
    "ch_shared_sender_or_attachment": "channel_pattern",
    "ch_shared_stem_and_not_sender": "channel_pattern",
    "ch_shared_domain_without_strong_support": "channel_pattern",
    "ch_semantic_ge_0_90": "channel_pattern",
    "ch_semantic_ge_0_93": "channel_pattern",
    "ch_semantic_ge_0_90_and_strong_support": "channel_pattern",
    "ch_semantic_ge_0_90_and_weak_support_only": "channel_pattern",
    "semantic_cosine_max": "existing_baseline",
    "time_gap_seconds_min": "existing_baseline",
}


def run_low_band_feature_discovery(
    *,
    run_dir: Path,
    graph_pt: Path,
    pair_csv: Path,
    gt_path: Path,
    output_dir: Path | None = None,
    checkpoint_name: str = "best_model.pt",
    device: str = "cpu",
    to_undirected: bool = True,
    low_score_max: float = 0.4,
    anchor_run_dir: Path | None = None,
    misp_json_path: Path | None = None,
    misp_translated_json_path: Path | None = None,
) -> dict[str, Any]:
    from src.pair_train import load_pair_training_dataframe

    project_root = _REPO_ROOT
    run_dir = run_dir.resolve()
    graph_pt = graph_pt.resolve()
    pair_csv = pair_csv.resolve()
    gt_path = gt_path.resolve()
    out_dir = (output_dir or (run_dir / "pair_score_separation")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    df, _ = load_pair_training_dataframe(pair_csv)
    df_work = df.reset_index(drop=True)
    df_work["_row"] = np.arange(len(df_work), dtype=np.int64)

    bundle = load_pair_supervision_for_inference(
        run_dir=run_dir,
        graph_pt=graph_pt,
        checkpoint_name=checkpoint_name,
        device=device,
        to_undirected=to_undirected,
    )
    scores = score_pair_rows(
        model=bundle["model"],
        pair_scorer=bundle["pair_scorer"],
        data_cpu=bundle["data_cpu"],
        df_work=df_work,
        device=bundle["device"],
        fanout=bundle["fanout"],
        pair_batch_size=bundle["pair_batch_size"],
        max_unique_emails=bundle["max_unique_emails"],
        pair_feature_columns=bundle.get("pair_feature_columns"),
    )
    scored = np.isfinite(scores)
    df_work["score"] = scores

    label_map, _, _ = load_ground_truth_structures(gt_path)
    label_map = {str(k): v for k, v in label_map.items()}
    ei = df_work["email_i"].astype(str).values
    ej = df_work["email_j"].astype(str).values
    n = len(df_work)
    camp_i = np.array([label_map.get(ei[k]) for k in range(n)], dtype=object)
    camp_j = np.array([label_map.get(ej[k]) for k in range(n)], dtype=object)
    both = np.array([camp_i[k] is not None and camp_j[k] is not None for k in range(n)], dtype=bool)
    same_mask = both & (camp_i == camp_j)
    cross_mask = both & (camp_i != camp_j)
    status = (
        df_work["pair_status"].astype(str).str.lower()
        if "pair_status" in df_work.columns
        else pd.Series(["unlabeled"] * n)
    )
    pos_mask = status.eq("positive").to_numpy()
    unl_mask = status.eq("unlabeled").to_numpy()
    low = df_work["score"].ge(0.0) & df_work["score"].le(float(low_score_max)) & scored
    high = df_work["score"].gt(float(low_score_max)) & scored

    same_unl = same_mask & unl_mask
    cross_unl = cross_mask & unl_mask
    same_low_unl = same_unl & low
    cross_low_unl = cross_unl & low
    same_high_unl = same_unl & high
    cross_high_unl = cross_unl & high

    analysis_mask = pos_mask | same_unl | cross_unl
    idx = np.where(analysis_mask)[0]

    nodes, anchor_meta = _load_anchor_nodes_extended(
        pair_csv=pair_csv, project_root=project_root, anchor_run_dir=anchor_run_dir
    )
    if misp_json_path is None:
        misp_json_path = _resolve_default_misp_json_path(project_root)
    email_text, text_meta = _load_email_text_catalog(
        project_root=project_root,
        misp_json_path=misp_json_path,
        misp_translated_json_path=misp_translated_json_path,
    )
    df_maps = _build_artifact_df_maps(nodes)
    body_rare = _build_body_rare_token_set(email_text)

    feat_rows: list[dict[str, Any]] = []
    for i in idx:
        r = df_work.iloc[int(i)]
        ei_s, ej_s = str(r["email_i"]), str(r["email_j"])
        feats = _compute_pair_features_row(
            ei=ei_s,
            ej=ej_s,
            row=r,
            nodes=nodes,
            text_i=email_text.get(ei_s, {}),
            text_j=email_text.get(ej_s, {}),
            df_maps=df_maps,
            body_rare_tokens=body_rare,
            n_docs=len(nodes),
        )
        feats["email_i"] = ei_s
        feats["email_j"] = ej_s
        feats["score"] = float(r["score"])
        feats["_pair_index"] = int(i)
        feat_rows.append(feats)

    feat_df = pd.DataFrame(feat_rows)
    pair_idx = feat_df["_pair_index"].to_numpy(dtype=np.int64)

    group_masks = {
        k: np.asarray(v, dtype=bool)
        for k, v in {
            "positive": pos_mask[pair_idx],
            "same_unlabeled": same_unl[pair_idx],
            "cross_unlabeled": cross_unl[pair_idx],
            "low_same_unlabeled": same_low_unl[pair_idx],
            "low_cross_unlabeled": cross_low_unl[pair_idx],
            "high_same_unlabeled": same_high_unl[pair_idx],
            "high_cross_unlabeled": cross_high_unl[pair_idx],
        }.items()
    }

    meta_cols = {"email_i", "email_j", "score", "_pair_index", "anchor_context_missing"}
    feature_cols = [c for c in feat_df.columns if c not in meta_cols]

    table_rows: list[dict[str, Any]] = []
    scorecard_rows: list[dict[str, Any]] = []
    low_same_df = feat_df.iloc[group_masks["low_same_unlabeled"]]
    low_cross_df = feat_df.iloc[group_masks["low_cross_unlabeled"]]

    for col in feature_cols:
        fam = _FEATURE_FAMILIES.get(col, "other")
        table_rows.append(
            _build_feature_analysis_row(
                feature=col,
                family=fam,
                feat_df=feat_df,
                group_masks=group_masks,
            )
        )
        if col not in ("anchor_time_gap_seconds", "log_anchor_time_gap_seconds"):
            s_num = pd.to_numeric(low_same_df[col], errors="coerce")
            if s_num.notna().sum() >= 10:
                qs = [0.1, 0.25, 0.5, 0.75, 0.9]
                thr_list = sorted({float(s_num.quantile(q)) for q in qs if s_num.notna().any()})
                scorecard_rows.extend(
                    _threshold_scorecard(
                        low_same_df[col], low_cross_df[col], feature=col, thresholds=thr_list
                    )
                )

    table = pd.DataFrame(table_rows)
    table = table.sort_values("combined_recommendation_score", ascending=False, na_position="last")
    scorecard = pd.DataFrame(scorecard_rows)

    counts = {
        "n_pairs_scored": int(scored.sum()),
        "n_positive": int(pos_mask.sum()),
        "n_same_campaign_unlabeled": int(same_unl.sum()),
        "n_cross_campaign_unlabeled": int(cross_unl.sum()),
        "n_same_campaign_low_unlabeled": int(same_low_unl.sum()),
        "n_cross_campaign_low_unlabeled": int(cross_low_unl.sum()),
        "n_same_campaign_high_unlabeled": int(same_high_unl.sum()),
        "n_cross_campaign_high_unlabeled": int(cross_high_unl.sum()),
        "n_pairs_feature_computed": int(len(feat_df)),
    }
    alignment = _generate_alignment_recommendations(table, counts=counts)
    recommendations = _generate_recommendations(table, scorecard, alignment)

    alignment_scorecard = table[
        [
            "feature",
            "feature_family",
            "feature_recommendation_category",
            "combined_recommendation_score",
            "low_band_separation_abs_mean_diff",
            "alignment_margin_vs_cross_unlabeled",
            "low_same_is_closer_to_positive_than_to_cross_unlabeled",
            "positive_mean",
            "same_unlabeled_mean",
            "cross_unlabeled_mean",
            "low_same_unlabeled_mean",
            "low_cross_unlabeled_mean",
            "high_same_unlabeled_mean",
            "high_cross_unlabeled_mean",
            "low_band_verdict",
            "positive_aligned",
            "low_band_separates",
        ]
    ].copy()

    summary = {
        "analysis": "low_band_feature_discovery_with_positive_alignment",
        "cohort_definition": {
            "low_score_band": [0.0, float(low_score_max)],
            "high_score_band": [float(low_score_max), 1.0],
            "positive_group": "pair_status == positive (seed pairs)",
            "same_unlabeled_group": "GT same-campaign, pair_status unlabeled",
            "cross_unlabeled_group": "GT cross-campaign, pair_status unlabeled",
        },
        "run_dir": str(run_dir),
        "graph_pt": str(graph_pt),
        "pair_csv": str(pair_csv),
        "gt_path": str(gt_path),
        "checkpoint": str(bundle["checkpoint_path"]),
        "counts": counts,
        "anchor_context": anchor_meta,
        "email_text_catalog": text_meta,
        "recommendations": recommendations,
        "feature_alignment_recommendations": alignment["feature_alignment_recommendations"],
        "best_candidate_features_to_add_next": alignment["best_candidate_features_to_add_next"],
        "top_ranked_by_combined_score": alignment_scorecard.head(25).to_dict(orient="records"),
        "top_low_band_separators": table.sort_values(
            "low_band_separation_abs_mean_diff", ascending=False, na_position="last"
        )
        .head(20)
        .to_dict(orient="records"),
    }

    summary_path = out_dir / "pair_low_band_feature_discovery_summary.json"
    table_path = out_dir / "pair_low_band_feature_discovery_table.csv"
    scorecard_path = out_dir / "pair_low_band_feature_threshold_scorecard.csv"
    alignment_path = out_dir / "pair_feature_alignment_scorecard.csv"

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    table.to_csv(table_path, index=False)
    alignment_scorecard.to_csv(alignment_path, index=False)
    if not scorecard.empty:
        scorecard.to_csv(scorecard_path, index=False)

    return {
        "summary_path": str(summary_path),
        "table_path": str(table_path),
        "alignment_scorecard_path": str(alignment_path),
        "scorecard_path": str(scorecard_path) if not scorecard.empty else None,
        "recommendations": recommendations,
        "best_candidate_features_to_add_next": alignment["best_candidate_features_to_add_next"],
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Low-band candidate feature discovery for pair MLP.")
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--graph-pt", type=Path, required=True)
    p.add_argument("--gt-path", type=Path, required=True, help="Ground-truth JSON (dedup or full).")
    p.add_argument("--pair-csv", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--checkpoint", type=str, default="best_model.pt")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--low-score-max", type=float, default=0.4)
    p.add_argument("--anchor-run-dir", type=Path, default=None)
    p.add_argument("--misp-json-path", type=Path, default=None)
    p.add_argument("--misp-translated-json-path", type=Path, default=None)
    p.add_argument("--no-to-undirected", action="store_true")
    args = p.parse_args(argv)

    run_dir = args.run_dir.resolve()
    pair_csv = args.pair_csv
    if pair_csv is None:
        cfg_path = run_dir / "training_config.json"
        if not cfg_path.is_file():
            raise SystemExit(f"Missing training_config.json in {run_dir}; pass --pair-csv")
        with open(cfg_path, encoding="utf-8") as f:
            tc = json.load(f)
        raw = tc.get("pair_dataset_csv")
        if not raw:
            raise SystemExit("pair_dataset_csv missing from training_config.json")
        pair_csv = Path(raw)
        if not pair_csv.is_absolute():
            pair_csv = (_REPO_ROOT / pair_csv).resolve()

    out = run_low_band_feature_discovery(
        run_dir=run_dir,
        graph_pt=args.graph_pt,
        pair_csv=pair_csv,
        gt_path=args.gt_path,
        output_dir=args.output_dir,
        checkpoint_name=args.checkpoint,
        device=args.device,
        to_undirected=not args.no_to_undirected,
        low_score_max=float(args.low_score_max),
        anchor_run_dir=args.anchor_run_dir,
        misp_json_path=args.misp_json_path,
        misp_translated_json_path=args.misp_translated_json_path,
    )
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
