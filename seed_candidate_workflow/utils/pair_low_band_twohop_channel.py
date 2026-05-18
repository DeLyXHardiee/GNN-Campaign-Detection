"""
Low-band 2-hop channel breakdown: summaries, joint rules, and recommendations.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from seed_candidate_workflow.utils.pair_inspection_admitting_evidence import pair_key

TWOHOP_CHANNELS: tuple[str, ...] = (
    "routing",
    "html_fp",
    "sender",
    "sender_domain",
    "url",
    "stem",
    "attachment",
)

_ARTIFACT_TYPE_TO_CHANNEL: dict[str, str] = {
    "url": "url",
    "stem": "stem",
    "attachment": "attachment",
    "html_structure_fingerprint": "html_fp",
    "html_fp": "html_fp",
    "sender": "sender",
    "sender_domain": "sender_domain",
    "routing": "routing",
    "received_host": "routing",
}

_PATH_HINT_TO_CHANNEL: tuple[tuple[str, str], ...] = (
    ("routing", "routing"),
    ("received_host", "routing"),
    ("html_fp", "html_fp"),
    ("html_structure", "html_fp"),
    ("sender_domain", "sender_domain"),
    ("sender_pattern", "sender"),
    ("url_template", "stem"),
    ("attachment", "attachment"),
)


def normalize_twohop_channel(*, artifact_type: str, path_type: str = "") -> str:
    at = str(artifact_type or "").strip().lower()
    pt = str(path_type or "").strip().lower()
    if at in _ARTIFACT_TYPE_TO_CHANNEL:
        return _ARTIFACT_TYPE_TO_CHANNEL[at]
    for hint, ch in _PATH_HINT_TO_CHANNEL:
        if hint in pt or hint in at:
            return ch
    return at.replace(" ", "_") if at else "unknown"


def twohop_records_from_evidence_list(recs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for r in recs:
        if str(r.get("source_family") or "") != "2hop":
            continue
        ch = normalize_twohop_channel(
            artifact_type=str(r.get("artifact_type") or ""),
            path_type=str(r.get("path_type") or ""),
        )
        row = dict(r)
        row["twohop_channel"] = ch
        out.append(row)
    out.sort(
        key=lambda x: (
            float(pd.to_numeric(x.get("rarity_score"), errors="coerce") or -1.0),
            str(x.get("artifact_type") or ""),
        ),
        reverse=True,
    )
    return out


def parse_admitting_evidence_json(raw: Any) -> list[dict[str, Any]]:
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return []
    if isinstance(raw, list):
        return [dict(x) for x in raw if isinstance(x, dict)]
    s = str(raw).strip()
    if not s:
        return []
    try:
        obj = json.loads(s)
        if isinstance(obj, list):
            return [dict(x) for x in obj if isinstance(x, dict)]
    except (json.JSONDecodeError, TypeError):
        pass
    return []


def attach_twohop_channel_columns(
    df: pd.DataFrame,
    *,
    evidence_index: dict[tuple[str, str], list[dict[str, Any]]] | None = None,
) -> pd.DataFrame:
    """Add per-pair 2-hop channel flags and compact admitting metadata."""
    if df.empty:
        return df.copy()

    out = df.copy()
    channels_col: list[str] = []
    primary_col: list[str] = []
    types_col: list[str] = []
    values_col: list[str] = []
    degrees_col: list[str] = []
    rarities_col: list[str] = []
    reasons_col: list[str] = []
    twohop_json_col: list[str] = []
    badge_col: list[str] = []
    via_flags: dict[str, list[bool]] = {f"twohop_via_{ch}": [] for ch in TWOHOP_CHANNELS}

    for _, r in out.iterrows():
        pk = pair_key(r["email_i"], r["email_j"])
        if "admitting_evidence_json" in r.index and pd.notna(r.get("admitting_evidence_json")):
            recs = parse_admitting_evidence_json(r.get("admitting_evidence_json"))
        elif evidence_index is not None:
            recs = list(evidence_index.get(pk) or [])
        else:
            recs = []
        hop = twohop_records_from_evidence_list(recs)
        chans = sorted({str(x.get("twohop_channel") or "unknown") for x in hop})
        chans = [c for c in chans if c != "unknown"] or (["unknown"] if hop else [])
        primary = chans[0] if chans else ""
        if hop:
            primary = str(
                max(
                    hop,
                    key=lambda x: float(pd.to_numeric(x.get("rarity_score"), errors="coerce") or -1.0),
                ).get("twohop_channel")
                or primary
            )

        for ch in TWOHOP_CHANNELS:
            via_flags[f"twohop_via_{ch}"].append(ch in chans)

        channels_col.append("|".join(chans))
        primary_col.append(primary)
        types_col.append("|".join(str(x.get("artifact_type") or "") for x in hop[:8]))
        values_col.append("|".join(str(x.get("artifact_value") or "")[:80] for x in hop[:5]))
        degrees_col.append("|".join(str(int(pd.to_numeric(x.get("artifact_degree"), errors="coerce") or 0)) for x in hop[:5]))
        rarities_col.append(
            "|".join(
                f"{float(pd.to_numeric(x.get('rarity_score'), errors='coerce')):.3f}"
                for x in hop[:5]
                if pd.notna(pd.to_numeric(x.get("rarity_score"), errors="coerce"))
            )
        )
        reasons_col.append("|".join(sorted({str(x.get("reason_code") or "") for x in hop if x.get("reason_code")})))
        twohop_json_col.append(json.dumps(hop[:12], ensure_ascii=False, default=str))
        badge_col.append(_channel_badges_for_row(r, chans))

    for ch in TWOHOP_CHANNELS:
        out[f"twohop_via_{ch}"] = via_flags[f"twohop_via_{ch}"]
    out["twohop_channels"] = channels_col
    out["twohop_primary_channel"] = primary_col
    out["twohop_artifact_types"] = types_col
    out["twohop_artifact_values"] = values_col
    out["twohop_artifact_degrees"] = degrees_col
    out["twohop_artifact_rarities"] = rarities_col
    out["twohop_reason_codes"] = reasons_col
    out["twohop_admitting_evidence_json"] = twohop_json_col
    out["twohop_channel_badges"] = badge_col
    return out


def _channel_badges_for_row(row: pd.Series, twohop_channels: list[str]) -> str:
    badges: list[str] = []
    for ch in twohop_channels:
        if ch in TWOHOP_CHANNELS:
            badges.append(f"2hop:{ch}")
    if bool(row.get("from_2hop")) and not twohop_channels:
        badges.append("2hop:unknown")
    if "routing" in twohop_channels and not any(
        bool(row.get(f"has_shared_{c}")) for c in ("sender", "url", "stem", "attachment", "html_fp")
    ):
        badges.append("shared routing only")
    if "html_fp" in twohop_channels:
        badges.append("shared html fp")
    if "sender_domain" in twohop_channels and not bool(row.get("has_shared_sender")):
        badges.append("shared sender_domain only")
    if bool(row.get("from_2hop")) and int(pd.to_numeric(row.get("source_count"), errors="coerce") or 0) == 1:
        if "weak infrastructure clue" not in badges:
            badges.append("weak infrastructure clue")
    return "|".join(badges)


def channel_bool_columns() -> tuple[str, ...]:
    return tuple(f"twohop_via_{ch}" for ch in TWOHOP_CHANNELS)


def low_band_twohop_joint_rule_names() -> tuple[str, ...]:
    """Joint separator rule expressions (keys must exist in bool_terms)."""
    rules: list[str] = []
    for ch in TWOHOP_CHANNELS:
        key = f"twohop_via_{ch}"
        rules.extend(
            [
                key,
                f"{key}_AND_NOT_shared_sender",
                f"{key}_AND_shared_sender",
                f"{key}_AND_semantic_ge_0_90",
                f"{key}_AND_NOT_semantic_ge_0_90",
                f"{key}_AND_source_count_eq_1",
                f"{key}_AND_same_seed_component_flag",
                f"{key}_AND_cross_seed_component_flag",
                f"{key}_AND_shared_sender_domain",
                f"{key}_AND_shared_html_fp",
                f"{key}_AND_shared_received_host",
            ]
        )
    return tuple(rules)


def _mean_or_none(s: pd.Series) -> float | None:
    v = pd.to_numeric(s, errors="coerce").dropna()
    return float(v.mean()) if not v.empty else None


def build_channel_summary_table(
    df_low_unlabeled: pd.DataFrame,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Per 2-hop channel: same vs cross counts/rates among low-band unlabeled pairs
  that have that channel as an admitting 2-hop path.
    """
    if df_low_unlabeled.empty:
        return [], {"status": "empty"}

    df = df_low_unlabeled.copy()
    if "twohop_channels" not in df.columns:
        df = attach_twohop_channel_columns(df)

    same = df[df["gt_relation"].astype(str) == "same_campaign"]
    cross = df[df["gt_relation"].astype(str) == "cross_campaign"]
    n_same_all = int(len(same))
    n_cross_all = int(len(cross))
    hop_same = same[same["from_2hop"].fillna(False).astype(bool)] if "from_2hop" in same.columns else same.iloc[0:0]
    hop_cross = cross[cross["from_2hop"].fillna(False).astype(bool)] if "from_2hop" in cross.columns else cross.iloc[0:0]

    rows: list[dict[str, Any]] = []
    for ch in TWOHOP_CHANNELS:
        col = f"twohop_via_{ch}"
        if col not in df.columns:
            continue
        m_same = same[col].fillna(False).astype(bool)
        m_cross = cross[col].fillna(False).astype(bool)
        ns = int(m_same.sum())
        nc = int(m_cross.sum())
        n_tot = ns + nc
        row: dict[str, Any] = {
            "twohop_channel": ch,
            "n_same_low": ns,
            "n_cross_low": nc,
            "n_low_total": n_tot,
            "fraction_of_same_low_unlabeled": float(ns / n_same_all) if n_same_all else None,
            "fraction_of_cross_low_unlabeled": float(nc / n_cross_all) if n_cross_all else None,
            "same_rate_among_channel_pairs": float(ns / n_tot) if n_tot else None,
            "cross_rate_among_channel_pairs": float(nc / n_tot) if n_tot else None,
            "same_cross_ratio": float(ns / nc) if nc else None,
            "mean_score_same": _mean_or_none(same.loc[m_same, "score"]),
            "mean_score_cross": _mean_or_none(cross.loc[m_cross, "score"]),
            "mean_semantic_same": _mean_or_none(
                same.loc[m_same, "semantic_cosine_for_display"]
                if "semantic_cosine_for_display" in same.columns
                else same.loc[m_same, "semantic_cosine_max"]
            ),
            "mean_semantic_cross": _mean_or_none(
                cross.loc[m_cross, "semantic_cosine_for_display"]
                if "semantic_cosine_for_display" in cross.columns
                else cross.loc[m_cross, "semantic_cosine_max"]
            ),
            "mean_source_count_same": _mean_or_none(same.loc[m_same, "source_count"]),
            "mean_source_count_cross": _mean_or_none(cross.loc[m_cross, "source_count"]),
            "mean_rarity_same": _mean_or_none(same.loc[m_same, "twohop_rarity_max"]),
            "mean_rarity_cross": _mean_or_none(cross.loc[m_cross, "twohop_rarity_max"]),
            "fraction_of_from_2hop_same": float(ns / len(hop_same)) if len(hop_same) else None,
            "fraction_of_from_2hop_cross": float(nc / len(hop_cross)) if len(hop_cross) else None,
        }
        row["recommendation_tag"] = _recommendation_tag_for_channel(row)
        rows.append(row)

    meta = {
        "status": "ok",
        "n_same_low_unlabeled": n_same_all,
        "n_cross_low_unlabeled": n_cross_all,
        "n_from_2hop_same_low": int(len(hop_same)),
        "n_from_2hop_cross_low": int(len(hop_cross)),
    }
    return rows, meta


def _recommendation_tag_for_channel(row: dict[str, Any]) -> str:
    ns = int(row.get("n_same_low") or 0)
    nc = int(row.get("n_cross_low") or 0)
    n = ns + nc
    if n < 3:
        return "insufficient_data"
    same_rate = row.get("same_rate_among_channel_pairs")
    cross_rate = row.get("cross_rate_among_channel_pairs")
    if same_rate is None or cross_rate is None:
        return "ambiguous"
    sr, cr = float(same_rate), float(cross_rate)
    # Cross-heavy among low-band pairs with this channel
    if cr >= 0.65 and sr <= 0.45:
        return "likely_too_noisy"
    if sr >= 0.55 and cr <= 0.45:
        return "potentially_strong"
    if sr >= 0.45 and cr >= 0.45:
        return "ambiguous_needs_corroboration"
    if sr > cr + 0.12:
        return "lean_keep_with_support"
    if cr > sr + 0.12:
        return "lean_tighten_or_disable"
    return "ambiguous_needs_corroboration"


def build_twohop_channel_recommendations(
    channel_rows: list[dict[str, Any]],
    *,
    joint_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    likely_noisy: list[str] = []
    needs_corroboration: list[str] = []
    potentially_strong: list[str] = []
    insufficient: list[str] = []
    notes: list[str] = []

    for r in channel_rows:
        ch = str(r.get("twohop_channel") or "")
        tag = str(r.get("recommendation_tag") or "")
        if tag == "likely_too_noisy":
            likely_noisy.append(ch)
        elif tag == "potentially_strong":
            potentially_strong.append(ch)
        elif tag in {"ambiguous_needs_corroboration", "lean_keep_with_support", "lean_tighten_or_disable"}:
            needs_corroboration.append(ch)
        elif tag == "insufficient_data":
            insufficient.append(ch)

    if joint_payload:
        favor_cross = joint_payload.get("ranked_joint_separators_favoring_cross_top10") or []
        for e in favor_cross[:5]:
            name = str(e.get("condition_name") or "")
            if name.startswith("twohop_via_routing"):
                if "routing" not in likely_noisy:
                    likely_noisy.append("routing")
                notes.append(f"Joint separator favors cross: {name}")
            if name.startswith("twohop_via_html_fp") and "html_fp" not in needs_corroboration:
                notes.append(f"html_fp cross-leaning joint rule: {name}")

    def _dedup(xs: list[str]) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for x in xs:
            if x and x not in seen:
                seen.add(x)
                out.append(x)
        return out

    return {
        "likely_too_noisy_for_2hop_generation": _dedup(likely_noisy),
        "potentially_useful_but_require_corroboration": _dedup(needs_corroboration),
        "potentially_strong_keep": _dedup(potentially_strong),
        "insufficient_low_band_support": _dedup(insufficient),
        "notes": notes[:20],
        "suggested_actions": _suggested_actions(likely_noisy, needs_corroboration, potentially_strong),
    }


def _suggested_actions(noisy: list[str], corroboration: list[str], strong: list[str]) -> list[str]:
    actions: list[str] = []
    for ch in noisy:
        if ch == "routing":
            actions.append(
                "Consider disabling or heavily tightening 2-hop routing (received_host) — "
                "low-band cross-campaign pairs dominate this channel."
            )
        else:
            actions.append(f"Consider disabling 2-hop channel '{ch}' or raising min_idf / lowering max_degree.")
    for ch in corroboration:
        if ch == "sender_domain":
            actions.append(
                "Keep sender_domain 2-hop only with corroboration (e.g. shared_sender or semantic>=0.90)."
            )
        elif ch == "html_fp":
            actions.append(
                "Treat html_fp 2-hop as support-only: require semantic>=0.90 or shared_sender before trusting."
            )
        else:
            actions.append(f"Gate 2-hop channel '{ch}' with shared_sender and/or semantic>=0.90.")
    for ch in strong:
        actions.append(f"Channel '{ch}' looks relatively same-campaign-enriched in the low band — candidate to keep.")
    return actions


def extend_bool_terms_for_low_band_channels(
    bool_terms: dict[str, np.ndarray],
    df_eval: pd.DataFrame,
    *,
    nodes_by_email: dict[str, dict[str, set[str]]],
    evidence_index: dict[tuple[str, str], list[dict[str, Any]]] | None = None,
) -> dict[str, np.ndarray]:
    """Add semantic/source-count/direct html_fp/routing and twohop_via_* columns."""
    n = len(df_eval)
    sem = pd.to_numeric(df_eval.get("semantic_cosine_max"), errors="coerce")
    bool_terms = dict(bool_terms)
    bool_terms["semantic_ge_0_90"] = sem.ge(0.90).fillna(False).to_numpy()

    sc = pd.to_numeric(df_eval.get("source_count"), errors="coerce")
    bool_terms["source_count_eq_1"] = sc.eq(1).fillna(False).to_numpy(dtype=bool)
    bool_terms["source_count_ge_2"] = sc.ge(2).fillna(False).to_numpy(dtype=bool)

    has_rh = np.zeros(n, dtype=bool)
    has_hfp = np.zeros(n, dtype=bool)
    for i, r in enumerate(df_eval.itertuples(index=False)):
        a, b = str(getattr(r, "email_i")), str(getattr(r, "email_j"))
        na, nb = nodes_by_email.get(a), nodes_by_email.get(b)
        if na is None or nb is None:
            continue
        has_rh[i] = bool((na.get("received_host_set") or set()) & (nb.get("received_host_set") or set()))
        has_hfp[i] = bool(
            (na.get("html_structure_fingerprint_set") or set())
            & (nb.get("html_structure_fingerprint_set") or set())
        )
    bool_terms["shared_received_host"] = has_rh
    bool_terms["shared_html_fp"] = has_hfp

    df_chan = attach_twohop_channel_columns(df_eval, evidence_index=evidence_index)
    for ch in TWOHOP_CHANNELS:
        col = f"twohop_via_{ch}"
        if col in df_chan.columns:
            bool_terms[col] = df_chan[col].fillna(False).astype(bool).to_numpy()
        else:
            bool_terms[col] = np.zeros(n, dtype=bool)
    return bool_terms
