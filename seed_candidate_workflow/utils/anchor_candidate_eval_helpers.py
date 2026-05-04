from __future__ import annotations

import json
import math
from collections import Counter, OrderedDict, defaultdict
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.metrics import completeness_score, homogeneity_score, v_measure_score

from seed_candidate_workflow.utils import raw_gnn_notebook as rn
from seed_candidate_workflow.utils.anchor_seed_helpers import _b_cubed_precision


def _pair(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a <= b else (b, a)


def _pairs_from_df(df: pd.DataFrame) -> set[tuple[str, str]]:
    if df.empty or not {"email_i", "email_j"}.issubset(df.columns):
        return set()
    out: set[tuple[str, str]] = set()
    for a, b in zip(df["email_i"].astype(str).tolist(), df["email_j"].astype(str).tolist(), strict=False):
        if a == b:
            continue
        out.add(_pair(a, b))
    return out


def _degrees_from_pairs(all_emails: list[str], pairs: set[tuple[str, str]]) -> dict[str, int]:
    deg = {str(e): 0 for e in all_emails}
    for a, b in pairs:
        deg[a] = deg.get(a, 0) + 1
        deg[b] = deg.get(b, 0) + 1
    return deg


def _quantiles(xs: list[float]) -> dict[str, float]:
    if not xs:
        return {"avg": 0.0, "median": 0.0, "p90": 0.0, "p95": 0.0, "max": 0.0}
    arr = np.asarray(xs, dtype=np.float64)
    return {
        "avg": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p90": float(np.quantile(arr, 0.90)),
        "p95": float(np.quantile(arr, 0.95)),
        "max": float(np.max(arr)),
    }


def _safe_float(x: Any) -> float:
    v = pd.to_numeric(x, errors="coerce")
    return float(v) if pd.notna(v) else float("nan")


def _resolve_gt_paths(project_root: Path, gt_cfg: dict[str, Any]) -> list[Path]:
    raw = gt_cfg.get("paths") or []
    out: list[Path] = []
    for x in raw:
        p = Path(str(x)).expanduser()
        if not p.is_absolute():
            p = (project_root / p).resolve()
        if p.is_file():
            out.append(p)
    return out


def _resolve_gt_paths_with_seed_fallback(
    *,
    project_root: Path,
    gt_cfg: dict[str, Any],
    seed_dir: Path,
) -> tuple[list[Path], dict[str, Any]]:
    primary = _resolve_gt_paths(project_root, gt_cfg)
    if primary:
        return primary, {"mode": "config_paths", "n_paths": int(len(primary))}
    seed_summary = seed_dir / "anchor_seed_summary.json"
    fallback: list[Path] = []
    if seed_summary.is_file():
        try:
            s = json.loads(seed_summary.read_text(encoding="utf-8"))
            for row in (s.get("gt_eval") or []):
                p_raw = row.get("gt_path")
                if not p_raw:
                    continue
                p = Path(str(p_raw)).expanduser()
                if not p.is_absolute():
                    p = (project_root / p).resolve()
                if p.is_file():
                    fallback.append(p)
        except Exception:
            fallback = []
    out: list[Path] = []
    seen: set[str] = set()
    for p in fallback:
        k = str(p.resolve())
        if k in seen:
            continue
        seen.add(k)
        out.append(p)
    return out, {"mode": "seed_summary_fallback", "seed_summary_path": str(seed_summary), "n_paths": int(len(out))}


def _load_gt_maps(gt_paths: list[Path]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for p in gt_paths:
        lm, _eid, _camp = rn.load_ground_truth_structures(p)
        out[str(p)] = {str(k): v for k, v in lm.items()}
    return out


def _gt_positive_pairs(gt_map: dict[str, Any]) -> set[tuple[str, str]]:
    camp_to_ids: dict[str, list[str]] = defaultdict(list)
    for eid, camp in gt_map.items():
        camp_to_ids[str(camp)].append(str(eid))
    out: set[tuple[str, str]] = set()
    for ids in camp_to_ids.values():
        if len(ids) < 2:
            continue
        ids_sorted = sorted(set(ids))
        for a, b in combinations(ids_sorted, 2):
            out.add((a, b))
    return out


def _b_cubed_recall_from_members(
    *,
    members_df: pd.DataFrame,
    gt_label_map: dict[str, Any],
) -> float:
    """B-cubed recall averaged over GT-covered emails (symmetric to precision over true clusters)."""
    if members_df.empty:
        return float("nan")
    d = members_df.copy()
    d["external_id"] = d["external_id"].astype(str)
    d["gt_label"] = d["external_id"].map({str(k): v for k, v in gt_label_map.items()})
    d = d[d["gt_label"].notna()].copy()
    if d.empty:
        return float("nan")
    d["component_id"] = d["component_id"].astype(int)
    gt_sizes = d.groupby("gt_label", dropna=False).size().rename("gt_n")
    cross = d.groupby(["gt_label", "component_id"], dropna=False).size().rename("n").reset_index()
    cross = cross.merge(gt_sizes.reset_index(), on="gt_label", how="left")
    cross["rec"] = pd.to_numeric(cross["n"], errors="coerce") / pd.to_numeric(cross["gt_n"], errors="coerce")
    weighted = float((cross["rec"] * cross["n"]).sum())
    n_eval = float(len(d))
    return float(weighted / max(1.0, n_eval))


def _null_json_floats(x: Any) -> Any:
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return None
    if isinstance(x, dict):
        return {k: _null_json_floats(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_null_json_floats(v) for v in x]
    return x


def _failure_regime_row(*, regime_name: str, n_good: int, n_bad: int) -> dict[str, Any]:
    n = int(n_good + n_bad)
    return {
        "regime_name": str(regime_name),
        "n_pairs": n,
        "n_oracle_good": int(n_good),
        "n_oracle_bad": int(n_bad),
        "oracle_good_fraction": float(n_good / n) if n else None,
        "oracle_bad_fraction": float(n_bad / n) if n else None,
    }


def _failure_regime_rows_from_masks(
    base: pd.DataFrame,
    *,
    regime_defs: list[tuple[str, pd.Series]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, m in regime_defs:
        mm = m.reindex(base.index).fillna(False).astype(bool)
        sub = base.loc[mm]
        rows.append(
            _failure_regime_row(
                regime_name=name,
                n_good=int(sub["oracle_good"].sum()),
                n_bad=int(sub["oracle_bad"].sum()),
            )
        )
    return rows


def _failure_regime_n_pairs(rows: list[dict[str, Any]], name: str) -> int:
    for r in rows:
        if r.get("regime_name") == name:
            return int(r.get("n_pairs") or 0)
    return 0


def _compute_failure_regime_diagnostics_for_gt(
    *,
    gt_path: str,
    gt_map: dict[str, Any],
    union_df: pd.DataFrame,
    scored_edges_df: pd.DataFrame | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """
    Diagnostic-only: compare oracle-good vs oracle-bad candidate edges among GT-covered pairs.
    """
    gt_map_s = {str(k): v for k, v in gt_map.items()}
    csv_rows: list[dict[str, Any]] = []

    empty_diag: dict[str, Any] = {
        "gt_path": str(gt_path),
        "n_gt_covered_candidate_pairs": 0,
        "n_oracle_good_pairs": 0,
        "n_oracle_bad_pairs": 0,
        "source_provenance_regimes": [],
        "semantic_infra_regimes": [],
        "time_gap_regimes": [],
        "source_count_regimes": [],
        "score_regimes": None,
        "score_regimes_unavailable_reason": None,
        "pu_pull_viability_signals": {},
        "viability_label": "weak",
    }

    if union_df.empty or not {"email_i", "email_j"}.issubset(union_df.columns):
        empty_diag["score_regimes_unavailable_reason"] = "no candidate union"
        return empty_diag, csv_rows

    df = union_df.copy()
    df["email_i"] = df["email_i"].astype(str)
    df["email_j"] = df["email_j"].astype(str)
    df["gt_i"] = df["email_i"].map(lambda x: gt_map_s.get(x))
    df["gt_j"] = df["email_j"].map(lambda x: gt_map_s.get(x))
    covered = df["gt_i"].notna() & df["gt_j"].notna()
    base = df.loc[covered].copy()
    if base.empty:
        out = dict(empty_diag)
        out["score_regimes_unavailable_reason"] = "no_gt_covered_candidate_pairs"
        return out, csv_rows

    base["oracle_good"] = base["gt_i"] == base["gt_j"]
    base["oracle_bad"] = base["gt_i"] != base["gt_j"]

    n_good = int(base["oracle_good"].sum())
    n_bad = int(base["oracle_bad"].sum())
    n_cov = int(len(base))
    bad_rate_all = float(n_bad / n_cov) if n_cov else 0.0
    good_frac_all = float(n_good / n_cov) if n_cov else 0.0

    fs = base.get("from_seed", False).astype(bool)
    fr = base.get("from_rare_artifact", False).astype(bool)
    fsem = base.get("from_semantic", False).astype(bool)
    fc = base.get("from_component", False).astype(bool)
    fh = base.get("from_2hop", False).astype(bool)
    both_sc = base.get("both_in_seed_components", False).astype(bool)
    same_sc = base.get("same_seed_component", False).astype(bool)
    semantic_only = (
        fsem & ~fr & ~fc & ~fh & ~fs
    )
    rare_only = fr & ~fsem & ~fc & ~fh & ~fs
    twohop_only = fh & ~fsem & ~fc & ~fr & ~fs
    component_only = fc & ~fsem & ~fh & ~fr & ~fs
    sem_plus_non = fsem & (fr | fc | fh | fs)
    scnt = pd.to_numeric(base.get("source_count"), errors="coerce")
    multi_source = scnt.notna() & scnt.ge(2)
    infra_only = ~fsem & (fs | fr | fc | fh)
    cross_component = both_sc & ~same_sc
    internal_same_seed_component = same_sc & both_sc

    prov_defs: list[tuple[str, pd.Series]] = [
        ("seed", fs),
        ("semantic_only", semantic_only),
        ("rare_artifact_only", rare_only),
        ("2hop_only", twohop_only),
        ("component_only", component_only),
        ("semantic_plus_non_semantic", sem_plus_non),
        ("multi_source_2plus", multi_source),
        ("infra_only", infra_only),
        ("cross_component_candidate_edges", cross_component),
        ("internal_same_seed_component", internal_same_seed_component),
    ]
    source_provenance_regimes = _failure_regime_rows_from_masks(base, regime_defs=prov_defs)

    cos = pd.to_numeric(base.get("semantic_cosine_max"), errors="coerce")
    has_ns = fs | fr | fc | fh
    sem_infra_name = pd.Series(index=base.index, dtype=object)
    # Priority-ordered semantic / infra interaction regimes
    m_high_sem_only = fsem & cos.ge(0.97) & ~has_ns
    m_high_with_ns = cos.ge(0.97) & has_ns
    m_mid_sem_with_ns = fsem & cos.ge(0.94) & cos.lt(0.97) & has_ns
    m_mid_sem_only = fsem & cos.ge(0.94) & cos.lt(0.97) & ~has_ns
    m_low_band = fsem & cos.ge(0.90) & cos.lt(0.94)
    m_very_low_sem = fsem & (cos.lt(0.90) | cos.isna())
    m_infra_low_sem = ~fsem & has_ns
    m_residual = ~(m_high_sem_only | m_high_with_ns | m_mid_sem_with_ns | m_mid_sem_only | m_low_band | m_very_low_sem | m_infra_low_sem)
    sem_infra_name.loc[m_high_sem_only] = "high_semantic_cos_ge_0.97_no_non_semantic_support"
    sem_infra_name.loc[m_high_with_ns] = "high_semantic_cos_ge_0.97_with_non_semantic_support"
    sem_infra_name.loc[m_mid_sem_with_ns] = "medium_semantic_cos_0.94_0.97_with_non_semantic_support"
    sem_infra_name.loc[m_mid_sem_only] = "medium_semantic_cos_0.94_0.97_semantic_only_no_non_semantic"
    sem_infra_name.loc[m_low_band] = "semantic_cos_0.90_0.94"
    sem_infra_name.loc[m_very_low_sem] = "low_semantic_cos_lt_0.90_or_missing"
    sem_infra_name.loc[m_infra_low_sem] = "low_semantic_signal_infra_supported"
    sem_infra_name.loc[m_residual] = "other_mixed_or_unclassified"
    sem_order = [
        "high_semantic_cos_ge_0.97_no_non_semantic_support",
        "high_semantic_cos_ge_0.97_with_non_semantic_support",
        "medium_semantic_cos_0.94_0.97_with_non_semantic_support",
        "medium_semantic_cos_0.94_0.97_semantic_only_no_non_semantic",
        "semantic_cos_0.90_0.94",
        "low_semantic_cos_lt_0.90_or_missing",
        "low_semantic_signal_infra_supported",
        "other_mixed_or_unclassified",
    ]
    semantic_infra_regimes: list[dict[str, Any]] = []
    for nm in sem_order:
        subm = sem_infra_name == nm
        sub = base.loc[subm]
        semantic_infra_regimes.append(
            _failure_regime_row(
                regime_name=nm,
                n_good=int(sub["oracle_good"].sum()),
                n_bad=int(sub["oracle_bad"].sum()),
            )
        )

    tg = pd.to_numeric(base.get("time_gap_seconds_min"), errors="coerce")
    tg_missing = tg.isna() | (tg == np.inf) | (tg == -np.inf)
    tg0 = ~tg_missing & (tg < 86400.0)
    tg1 = ~tg_missing & (tg >= 86400.0) & (tg < 7.0 * 86400.0)
    tg2 = ~tg_missing & (tg >= 7.0 * 86400.0) & (tg < 30.0 * 86400.0)
    tg3 = ~tg_missing & (tg >= 30.0 * 86400.0)
    time_defs: list[tuple[str, pd.Series]] = [
        ("same_day_or_near_zero_lt_1d", tg0),
        ("time_gap_1_to_7_days", tg1),
        ("time_gap_7_to_30_days", tg2),
        ("time_gap_gt_30_days", tg3),
        ("time_gap_missing", tg_missing),
    ]
    time_gap_regimes = _failure_regime_rows_from_masks(base, regime_defs=time_defs)

    sc_missing = scnt.isna()
    sc1 = ~sc_missing & (scnt == 1)
    sc2 = ~sc_missing & (scnt == 2)
    sc3p = ~sc_missing & (scnt >= 3)
    sc_defs: list[tuple[str, pd.Series]] = [
        ("source_count_1", sc1),
        ("source_count_2", sc2),
        ("source_count_3_plus", sc3p),
        ("source_count_missing", sc_missing),
    ]
    source_count_regimes = _failure_regime_rows_from_masks(base, regime_defs=sc_defs)

    score_regimes: list[dict[str, Any]] | None = None
    score_note: str | None = None
    score_separates = False
    if scored_edges_df is None or scored_edges_df.empty:
        score_note = "scored_clustering_edges_all.csv not found or empty at candidate-eval output dir"
    else:
        se = scored_edges_df.copy()
        if not {"email_a", "email_b", "edge_weight"}.issubset(se.columns):
            score_note = "scored edges CSV missing email_a/email_b/edge_weight"
        else:
            se["email_a"] = se["email_a"].astype(str)
            se["email_b"] = se["email_b"].astype(str)
            se["_pk"] = [str(_pair(a, b)) for a, b in zip(se["email_a"], se["email_b"], strict=False)]
            pk_base = [str(_pair(a, b)) for a, b in zip(base["email_i"], base["email_j"], strict=False)]
            base = base.copy()
            base["_pk"] = pk_base
            wmap = se.drop_duplicates("_pk").set_index("_pk")["edge_weight"]
            base["edge_weight_scored"] = base["_pk"].map(wmap)
            ew = pd.to_numeric(base["edge_weight_scored"], errors="coerce")
            hit = ew.notna()
            if int(hit.sum()) < 5:
                score_note = f"too_few_scored_edges_joined n_matched={int(hit.sum())}"
            else:
                ew_hit = ew.loc[hit]
                q1 = float(ew_hit.quantile(1.0 / 3.0))
                q2 = float(ew_hit.quantile(2.0 / 3.0))
                low = hit & (ew <= q1)
                mid = hit & (ew > q1) & (ew <= q2)
                high = hit & (ew > q2)
                miss = ~hit
                score_defs: list[tuple[str, pd.Series]] = [
                    ("scored_weight_low_tertile", low),
                    ("scored_weight_mid_tertile", mid),
                    ("scored_weight_high_tertile", high),
                    ("scored_weight_missing_join", miss),
                ]
                score_regimes = _failure_regime_rows_from_masks(base, regime_defs=score_defs)
                g_w = ew.loc[base["oracle_good"] & hit]
                b_w = ew.loc[base["oracle_bad"] & hit]
                if len(g_w) >= 5 and len(b_w) >= 5:
                    score_separates = float(g_w.mean()) > float(b_w.mean()) + 0.01

    def _bad_frac(rows: list[dict[str, Any]], name: str) -> float | None:
        for r in rows:
            if r.get("regime_name") == name:
                n = int(r.get("n_pairs") or 0)
                if n < 1:
                    return None
                return float(r.get("n_oracle_bad", 0)) / float(n)
        return None

    def _good_frac(rows: list[dict[str, Any]], name: str) -> float | None:
        for r in rows:
            if r.get("regime_name") == name:
                n = int(r.get("n_pairs") or 0)
                if n < 1:
                    return None
                return float(r.get("n_oracle_good", 0)) / float(n)
        return None

    min_n = 15
    bf_sem_only = _bad_frac(source_provenance_regimes, "semantic_only")
    bf_high_sem_only = _bad_frac(semantic_infra_regimes, "high_semantic_cos_ge_0.97_no_non_semantic_support")
    gf_multi = _good_frac(source_provenance_regimes, "multi_source_2plus")
    bf_cross = _bad_frac(source_provenance_regimes, "cross_component_candidate_edges")
    bf_internal = _bad_frac(source_provenance_regimes, "internal_same_seed_component")
    bf_30d = _bad_frac(time_gap_regimes, "time_gap_gt_30_days")

    semantic_only_bad_bridge_enrichment = bool(
        bf_sem_only is not None and bad_rate_all > 0 and bf_sem_only >= bad_rate_all * 1.2 and (bf_sem_only - bad_rate_all) >= 0.03
    )
    n_high_sem_only = _failure_regime_n_pairs(semantic_infra_regimes, "high_semantic_cos_ge_0.97_no_non_semantic_support")
    high_cos_low_support_bad_bridge_enrichment = bool(
        bf_high_sem_only is not None
        and bad_rate_all > 0
        and bf_high_sem_only >= bad_rate_all * 1.15
        and n_high_sem_only >= min_n
    )
    multi_source_good_edge_enrichment = bool(
        gf_multi is not None and good_frac_all > 0 and gf_multi >= good_frac_all * 1.1
    )
    large_time_gap_bad_bridge_enrichment = bool(
        bf_30d is not None and bad_rate_all > 0 and bf_30d >= bad_rate_all * 1.15
    )
    n_cross = _failure_regime_n_pairs(source_provenance_regimes, "cross_component_candidate_edges")
    n_internal = _failure_regime_n_pairs(source_provenance_regimes, "internal_same_seed_component")
    if (
        bf_cross is not None
        and bf_internal is not None
        and n_internal >= min_n
        and bf_internal > 0
    ):
        cross_vs_internal = bf_cross >= bf_internal * 1.1
    else:
        cross_vs_internal = bf_cross is not None and bad_rate_all > 0 and bf_cross >= bad_rate_all * 1.15
    cross_component_bad_bridge_enrichment = bool(bf_cross is not None and cross_vs_internal and n_cross >= min_n)
    current_score_partially_separates_good_bad = bool(score_separates) if score_regimes is not None else False

    pos = sum(
        [
            semantic_only_bad_bridge_enrichment,
            high_cos_low_support_bad_bridge_enrichment,
            multi_source_good_edge_enrichment,
            large_time_gap_bad_bridge_enrichment,
            cross_component_bad_bridge_enrichment,
        ]
    )
    neg_same_shape = True
    spreads: list[float] = []
    for r in semantic_infra_regimes:
        n = int(r.get("n_pairs") or 0)
        obf = r.get("oracle_bad_fraction")
        if n >= min_n and obf is not None and isinstance(obf, (int, float)) and not (isinstance(obf, float) and math.isnan(obf)):
            spreads.append(float(obf))
    if spreads:
        neg_same_shape = (max(spreads) - min(spreads)) < 0.03
    if pos >= 3:
        viability_label = "strong"
    elif pos >= 2:
        viability_label = "moderate"
    elif pos == 1 and not neg_same_shape:
        viability_label = "moderate"
    else:
        viability_label = "weak"

    pu_pull_viability_signals: dict[str, Any] = {
        "semantic_only_bad_bridge_enrichment": semantic_only_bad_bridge_enrichment,
        "high_cos_low_support_bad_bridge_enrichment": high_cos_low_support_bad_bridge_enrichment,
        "multi_source_good_edge_enrichment": multi_source_good_edge_enrichment,
        "large_time_gap_bad_bridge_enrichment": large_time_gap_bad_bridge_enrichment,
        "cross_component_bad_bridge_enrichment": cross_component_bad_bridge_enrichment,
        "current_score_partially_separates_good_bad": current_score_partially_separates_good_bad,
    }

    out: dict[str, Any] = {
        "gt_path": str(gt_path),
        "n_gt_covered_candidate_pairs": n_cov,
        "n_oracle_good_pairs": n_good,
        "n_oracle_bad_pairs": n_bad,
        "source_provenance_regimes": source_provenance_regimes,
        "semantic_infra_regimes": semantic_infra_regimes,
        "time_gap_regimes": time_gap_regimes,
        "source_count_regimes": source_count_regimes,
        "score_regimes": score_regimes,
        "score_regimes_unavailable_reason": score_note,
        "pu_pull_viability_signals": pu_pull_viability_signals,
        "viability_label": viability_label,
    }

    for family, rows in [
        ("source_provenance_regimes", source_provenance_regimes),
        ("semantic_infra_regimes", semantic_infra_regimes),
        ("time_gap_regimes", time_gap_regimes),
        ("source_count_regimes", source_count_regimes),
        ("score_regimes", score_regimes or []),
    ]:
        for r in rows:
            csv_rows.append(
                {
                    "gt_path": str(gt_path),
                    "regime_family": family,
                    "regime_name": r.get("regime_name"),
                    "n_pairs": r.get("n_pairs"),
                    "n_oracle_good": r.get("n_oracle_good"),
                    "n_oracle_bad": r.get("n_oracle_bad"),
                    "oracle_good_fraction": r.get("oracle_good_fraction"),
                    "oracle_bad_fraction": r.get("oracle_bad_fraction"),
                }
            )

    return out, csv_rows


def _compute_candidate_oracle_ceiling_for_gt(
    *,
    gt_path: str,
    gt_map: dict[str, Any],
    union_pairs: set[tuple[str, str]],
) -> dict[str, Any]:
    """
    Oracle ceiling: connected components on GT-covered nodes using only candidate edges
    that connect two emails in the same GT campaign.
    """
    gt_map_s = {str(k): v for k, v in gt_map.items()}
    n_eval = int(len(gt_map_s))
    if n_eval == 0:
        return {
            "gt_path": str(gt_path),
            "n_eval": 0,
            "n_gt_campaigns": 0,
            "n_candidate_pairs_total": int(len(union_pairs)),
            "n_oracle_same_campaign_candidate_pairs": 0,
            "n_eval_emails_touched_by_oracle_edges": 0,
            "pct_eval_emails_touched_by_oracle_edges": None,
            "homogeneity": None,
            "completeness": None,
            "v_measure": None,
            "b_cubed_precision": None,
            "b_cubed_recall": None,
            "n_predicted_components": 0,
            "singleton_rate": None,
            "largest_component_size": 0,
            "pct_eval_emails_in_largest_component": None,
            "oracle_campaign_reconnect_rate": None,
            "n_gt_campaigns_size_ge_2": 0,
            "n_gt_campaigns_reconnected_by_oracle_edges": 0,
            "pct_campaigns_fully_connected_among_size_ge_2": None,
            "pct_campaigns_partially_connected_among_size_ge_2": None,
            "pct_campaigns_singleton_fractured_among_size_ge_2": None,
            "ceiling_strength_label": "weak",
            "interpretation": "Empty GT label map.",
        }
    campaign_to_ids: dict[str, set[str]] = defaultdict(set)
    for eid, camp in gt_map_s.items():
        campaign_to_ids[str(camp)].add(str(eid))
    n_gt_campaigns = int(len(campaign_to_ids))
    n_candidate_pairs_total = int(len(union_pairs))

    oracle_edges: list[tuple[str, str]] = []
    for a, b in union_pairs:
        if a not in gt_map_s or b not in gt_map_s:
            continue
        if gt_map_s[a] == gt_map_s[b]:
            oracle_edges.append((a, b))
    n_oracle_same_campaign_candidate_pairs = int(len(oracle_edges))

    touched_oracle: set[str] = set()
    for a, b in oracle_edges:
        touched_oracle.add(a)
        touched_oracle.add(b)
    n_eval_emails_touched_by_oracle_edges = int(len(set(gt_map_s.keys()) & touched_oracle))
    pct_eval_emails_touched_by_oracle_edges = (
        float(n_eval_emails_touched_by_oracle_edges / max(1, n_eval)) if n_eval else None
    )

    g = nx.Graph()
    g.add_nodes_from(gt_map_s.keys())
    g.add_edges_from(oracle_edges)
    comps = list(nx.connected_components(g))
    n_predicted_components = int(len(comps))
    comp_sizes = [len(c) for c in comps]
    largest_component_size = int(max(comp_sizes)) if comp_sizes else 0
    n_singleton_components = int(sum(1 for s in comp_sizes if s == 1))
    singleton_rate = float(n_singleton_components / max(1, n_predicted_components))
    pct_eval_emails_in_largest_component = float(largest_component_size / max(1, n_eval)) if n_eval else None

    pred_map: dict[str, int] = {}
    for i, comp in enumerate(comps):
        for n in comp:
            pred_map[str(n)] = int(i)
    eval_ids = sorted(gt_map_s.keys())
    y_true = [str(gt_map_s[e]) for e in eval_ids]
    y_pred = [int(pred_map[e]) for e in eval_ids]

    hom = comp_m = vm = float("nan")
    if n_eval >= 2:
        try:
            hom = float(homogeneity_score(y_true, y_pred))
            comp_m = float(completeness_score(y_true, y_pred))
            vm = float(v_measure_score(y_true, y_pred))
        except ValueError:
            hom = comp_m = vm = float("nan")

    members_df = pd.DataFrame([{"external_id": e, "component_id": pred_map[e]} for e in eval_ids])
    bc = _b_cubed_precision(members_df=members_df, gt_label_map=gt_map_s)
    b_cubed_precision = float(bc.get("b_cubed_precision", float("nan")))
    b_cubed_recall = float(_b_cubed_recall_from_members(members_df=members_df, gt_label_map=gt_map_s))

    campaigns_2plus = [ids for ids in campaign_to_ids.values() if len(ids) >= 2]
    n_camp_2p = int(len(campaigns_2plus))
    n_campaigns_reconnected = 0
    n_fully_connected = 0
    n_partially_connected = 0
    n_singleton_fractured = 0
    for ids in campaigns_2plus:
        sub = g.subgraph(ids).copy()
        wccs = list(nx.connected_components(sub)) if ids else []
        mx = max((len(c) for c in wccs), default=0)
        if mx >= 2:
            n_campaigns_reconnected += 1
        if len(ids) >= 2 and mx == len(ids):
            n_fully_connected += 1
        elif mx >= 2:
            n_partially_connected += 1
        else:
            n_singleton_fractured += 1
    oracle_campaign_reconnect_rate = float(n_campaigns_reconnected / max(1, n_camp_2p)) if n_camp_2p else None
    pct_campaigns_fully_connected = float(n_fully_connected / max(1, n_camp_2p)) if n_camp_2p else None
    pct_campaigns_partially_connected = float(n_partially_connected / max(1, n_camp_2p)) if n_camp_2p else None
    pct_campaigns_singleton_fractured = float(n_singleton_fractured / max(1, n_camp_2p)) if n_camp_2p else None

    if not math.isnan(vm):
        if vm >= 0.65 and (comp_m >= 0.45 if not math.isnan(comp_m) else False):
            ceiling_strength_label = "strong"
        elif vm >= 0.4 or (not math.isnan(comp_m) and comp_m >= 0.35):
            ceiling_strength_label = "moderate"
        else:
            ceiling_strength_label = "weak"
    else:
        ceiling_strength_label = "weak"

    return {
        "gt_path": str(gt_path),
        "n_eval": n_eval,
        "n_gt_campaigns": n_gt_campaigns,
        "n_candidate_pairs_total": n_candidate_pairs_total,
        "n_oracle_same_campaign_candidate_pairs": n_oracle_same_campaign_candidate_pairs,
        "n_eval_emails_touched_by_oracle_edges": n_eval_emails_touched_by_oracle_edges,
        "pct_eval_emails_touched_by_oracle_edges": pct_eval_emails_touched_by_oracle_edges,
        "homogeneity": hom,
        "completeness": comp_m,
        "v_measure": vm,
        "b_cubed_precision": b_cubed_precision,
        "b_cubed_recall": b_cubed_recall,
        "n_predicted_components": n_predicted_components,
        "singleton_rate": singleton_rate,
        "largest_component_size": largest_component_size,
        "pct_eval_emails_in_largest_component": pct_eval_emails_in_largest_component,
        "oracle_campaign_reconnect_rate": oracle_campaign_reconnect_rate,
        "n_gt_campaigns_size_ge_2": n_camp_2p,
        "n_gt_campaigns_reconnected_by_oracle_edges": n_campaigns_reconnected,
        "pct_campaigns_fully_connected_among_size_ge_2": pct_campaigns_fully_connected,
        "pct_campaigns_partially_connected_among_size_ge_2": pct_campaigns_partially_connected,
        "pct_campaigns_singleton_fractured_among_size_ge_2": pct_campaigns_singleton_fractured,
        "ceiling_strength_label": ceiling_strength_label,
        "interpretation": (
            "Oracle uses only candidate-universe edges that are GT same-campaign; "
            "predicted clusters are connected components (singleton if no oracle edge)."
        ),
    }


def _pair_precision_labeled(pairs: set[tuple[str, str]], gt_map: dict[str, Any]) -> float:
    if not pairs:
        return float("nan")
    num = 0
    den = 0
    for a, b in pairs:
        la = gt_map.get(a)
        lb = gt_map.get(b)
        if la is None or lb is None:
            continue
        den += 1
        if la == lb:
            num += 1
    if den == 0:
        return float("nan")
    return float(num / den)


def _read_optional_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    return pd.read_csv(path)


def _variant_pairs(
    *,
    included: set[str],
    source_pairs: dict[str, set[tuple[str, str]]],
) -> set[tuple[str, str]]:
    out: set[tuple[str, str]] = set()
    for s in included:
        out |= source_pairs.get(s, set())
    return out


def run_candidate_evaluation_report(
    *,
    project_root: Path,
    graph_id: str,
    out_dir: Path,
    seed_dir: Path,
    candidate_union_df: pd.DataFrame,
    seed_pairs: set[tuple[str, str]],
    total_emails: int,
    eval_cfg: dict[str, Any] | None = None,
    generator_configs: dict[str, dict[str, Any]] | None = None,
    generator_outputs: list[dict[str, Any]] | None = None,
    full_candidate_generation_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = eval_cfg or {}
    gt_cfg = cfg.get("ground_truth") or {}
    manual_cfg = cfg.get("manual_review") or {}
    readiness_cfg = cfg.get("readiness") or {}
    manual_seed = int(manual_cfg.get("random_seed", 1337))
    per_group_n = int(manual_cfg.get("per_group_n", 25))

    # Required inputs
    seed_edges_all_csv = seed_dir / "seed_edges_all.csv"
    seed_members_csv = seed_dir / "seed_union_component_members.csv"
    seed_components_csv = seed_dir / "seed_union_components.csv"
    candidate_union_csv = out_dir / "candidate_union.csv"

    # Optional sources
    p_rare = out_dir / "candidates_rare_artifact.csv"
    p_sem = out_dir / "candidates_semantic.csv"
    p_comp = out_dir / "candidates_component_expanded.csv"
    p_2hop = out_dir / "candidates_2hop.csv"

    df_rare = _read_optional_csv(p_rare)
    df_sem = _read_optional_csv(p_sem)
    df_comp = _read_optional_csv(p_comp)
    df_2hop = _read_optional_csv(p_2hop)
    seed_members_df = _read_optional_csv(seed_members_csv)

    union_df = candidate_union_df.copy()
    if union_df.empty:
        union_df = pd.DataFrame(
            columns=[
                "email_i",
                "email_j",
                "from_seed",
                "from_rare_artifact",
                "from_semantic",
                "from_component",
                "from_2hop",
                "source_count",
            ]
        )
    if not union_df.empty:
        union_df["email_i"] = union_df["email_i"].astype(str)
        union_df["email_j"] = union_df["email_j"].astype(str)

    union_pairs = _pairs_from_df(union_df)
    union_emails = sorted(set(union_df.get("email_i", pd.Series(dtype=str)).astype(str)).union(set(union_df.get("email_j", pd.Series(dtype=str)).astype(str))))
    all_pairs_total = int(total_emails * (total_emails - 1) // 2)
    deg_union = _degrees_from_pairs([str(i) for i in union_emails], union_pairs)
    deg_vals_union = list(deg_union.values())

    # source families (from union flags, seed explicit)
    source_pairs: dict[str, set[tuple[str, str]]] = {
        "seed": set(seed_pairs),
        "rare_artifact": set(),
        "semantic": set(),
        "component": set(),
        "2hop": set(),
    }
    for _, r in union_df.iterrows():
        p = _pair(str(r["email_i"]), str(r["email_j"]))
        if bool(r.get("from_seed", False)):
            source_pairs["seed"].add(p)
        if bool(r.get("from_rare_artifact", False)):
            source_pairs["rare_artifact"].add(p)
        if bool(r.get("from_semantic", False)):
            source_pairs["semantic"].add(p)
        if bool(r.get("from_component", False)):
            source_pairs["component"].add(p)
        if bool(r.get("from_2hop", False)):
            source_pairs["2hop"].add(p)

    source_present = [s for s, p in source_pairs.items() if len(p) > 0]

    # GT setup (must be resolved before metadata references gt_resolution)
    gt_paths, gt_resolution = _resolve_gt_paths_with_seed_fallback(
        project_root=project_root,
        gt_cfg=gt_cfg,
        seed_dir=seed_dir,
    )
    gt_maps = _load_gt_maps(gt_paths)
    # Metadata
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "graph_id": str(graph_id),
        "seed_input_paths": {
            "seed_edges_all_csv": str(seed_edges_all_csv),
            "seed_union_component_members_csv": str(seed_members_csv),
            "seed_union_components_csv": str(seed_components_csv),
        },
        "candidate_input_paths": {
            "candidate_union_csv": str(candidate_union_csv),
            "candidates_rare_artifact_csv": str(p_rare) if p_rare.is_file() else None,
            "candidates_semantic_csv": str(p_sem) if p_sem.is_file() else None,
            "candidates_component_expanded_csv": str(p_comp) if p_comp.is_file() else None,
            "candidates_2hop_csv": str(p_2hop) if p_2hop.is_file() else None,
        },
        "total_emails": int(total_emails),
        "all_pairs_total": int(all_pairs_total),
        "unavailable_optional_sources": [
            name
            for name, p in [
                ("rare_artifact", p_rare),
                ("semantic", p_sem),
                ("component", p_comp),
                ("2hop", p_2hop),
            ]
            if not p.is_file()
        ],
        "gt_resolution": gt_resolution,
    }

    # candidate_universe
    quant = _quantiles([float(x) for x in deg_vals_union])
    n_zero = int(max(0, total_emails - len(union_emails)))
    b_1_5 = int(sum(1 for x in deg_vals_union if 1 <= x <= 5))
    b_6_20 = int(sum(1 for x in deg_vals_union if 6 <= x <= 20))
    b_21_50 = int(sum(1 for x in deg_vals_union if 21 <= x <= 50))
    b_51_100 = int(sum(1 for x in deg_vals_union if 51 <= x <= 100))
    b_100p = int(sum(1 for x in deg_vals_union if x > 100))
    candidate_universe = {
        "n_candidate_pairs_total": int(len(union_pairs)),
        "n_unique_emails_in_candidates": int(len(union_emails)),
        "pct_emails_in_candidates": float(len(union_emails) / max(1, total_emails)),
        "all_pairs_total": int(all_pairs_total),
        "reduction_ratio_vs_all_pairs": float(len(union_pairs) / max(1, all_pairs_total)),
        "avg_candidates_per_email": float(quant["avg"]),
        "median_candidates_per_email": float(quant["median"]),
        "p90_candidates_per_email": float(quant["p90"]),
        "p95_candidates_per_email": float(quant["p95"]),
        "max_candidates_per_email": float(quant["max"]),
        "candidate_count_distribution": {
            "count_0": int(n_zero),
            "pct_0": float(n_zero / max(1, total_emails)),
            "count_1_5": int(b_1_5),
            "pct_1_5": float(b_1_5 / max(1, total_emails)),
            "count_6_20": int(b_6_20),
            "pct_6_20": float(b_6_20 / max(1, total_emails)),
            "count_21_50": int(b_21_50),
            "pct_21_50": float(b_21_50 / max(1, total_emails)),
            "count_51_100": int(b_51_100),
            "pct_51_100": float(b_51_100 / max(1, total_emails)),
            "count_gt_100": int(b_100p),
            "pct_gt_100": float(b_100p / max(1, total_emails)),
        },
    }

    # per_source
    per_source: dict[str, dict[str, Any]] = {}
    for src in ["seed", "rare_artifact", "semantic", "component", "2hop"]:
        pairs = source_pairs.get(src, set())
        touched = sorted(set(a for a, _ in pairs).union(set(b for _, b in pairs)))
        deg = _degrees_from_pairs(touched, pairs)
        dvals = list(deg.values())
        q = _quantiles([float(x) for x in dvals])
        others = set().union(*[p for s, p in source_pairs.items() if s != src])
        source_only = pairs - others
        item: dict[str, Any] = {
            "source_name": src,
            "n_candidate_pairs": int(len(pairs)),
            "n_unique_emails_touched": int(len(touched)),
            "avg_candidates_per_touched_email": float(q["avg"]),
            "median_candidates_per_touched_email": float(q["median"]),
            "source_only_pairs_count": int(len(source_only)),
            "source_participation_rate_in_union": float(len(pairs) / max(1, len(union_pairs))),
        }
        if src == "rare_artifact" and not df_rare.empty:
            item["score_summary"] = {
                "rarity_score_mean": float(pd.to_numeric(df_rare.get("rarity_score"), errors="coerce").dropna().mean())
                if "rarity_score" in df_rare.columns else float("nan"),
                "rarity_score_median": float(pd.to_numeric(df_rare.get("rarity_score"), errors="coerce").dropna().median())
                if "rarity_score" in df_rare.columns else float("nan"),
            }
            if "time_gap_seconds" in df_rare.columns:
                x = pd.to_numeric(df_rare["time_gap_seconds"], errors="coerce").dropna()
                item["time_gap_summary"] = {"mean": float(x.mean()) if len(x) else float("nan"), "median": float(x.median()) if len(x) else float("nan")}
        if src == "semantic" and not df_sem.empty:
            item["score_summary"] = {
                "cosine_mean": float(pd.to_numeric(df_sem.get("cosine"), errors="coerce").dropna().mean())
                if "cosine" in df_sem.columns else float("nan"),
                "cosine_median": float(pd.to_numeric(df_sem.get("cosine"), errors="coerce").dropna().median())
                if "cosine" in df_sem.columns else float("nan"),
            }
            if "time_gap_seconds" in df_sem.columns:
                x = pd.to_numeric(df_sem["time_gap_seconds"], errors="coerce").dropna()
                item["time_gap_summary"] = {"mean": float(x.mean()) if len(x) else float("nan"), "median": float(x.median()) if len(x) else float("nan")}
        if src == "component" and not df_comp.empty:
            if "time_gap_seconds" in df_comp.columns:
                x = pd.to_numeric(df_comp["time_gap_seconds"], errors="coerce").dropna()
                item["time_gap_summary"] = {"mean": float(x.mean()) if len(x) else float("nan"), "median": float(x.median()) if len(x) else float("nan")}
        if src == "2hop" and not df_2hop.empty:
            item["score_summary"] = {
                "rarity_score_mean": float(pd.to_numeric(df_2hop.get("rarity_score"), errors="coerce").dropna().mean())
                if "rarity_score" in df_2hop.columns else float("nan"),
                "rarity_score_median": float(pd.to_numeric(df_2hop.get("rarity_score"), errors="coerce").dropna().median())
                if "rarity_score" in df_2hop.columns else float("nan"),
            }
            if "intermediary_degree" in df_2hop.columns:
                y = pd.to_numeric(df_2hop["intermediary_degree"], errors="coerce").dropna()
                item["degree_summary"] = {"mean": float(y.mean()) if len(y) else float("nan"), "median": float(y.median()) if len(y) else float("nan")}
            if "time_gap_seconds" in df_2hop.columns:
                x = pd.to_numeric(df_2hop["time_gap_seconds"], errors="coerce").dropna()
                item["time_gap_summary"] = {"mean": float(x.mean()) if len(x) else float("nan"), "median": float(x.median()) if len(x) else float("nan")}

        gt_src: list[dict[str, Any]] = []
        for gt_path, gt_map in gt_maps.items():
            gt_pos = _gt_positive_pairs(gt_map)
            gt_recovered = len(pairs & gt_pos)
            gt_src.append(
                {
                    "gt_path": gt_path,
                    "gt_positive_pair_recall": float(gt_recovered / max(1, len(gt_pos))),
                    "gt_positive_pairs_recovered": int(gt_recovered),
                    "gt_labeled_pair_precision": float(_pair_precision_labeled(pairs, gt_map)),
                    "gt_covered_emails_touched": int(len(set(gt_map.keys()) & set(touched))),
                }
            )
        if gt_src:
            item["gt"] = gt_src
        per_source[src] = item

    # source_overlap
    overlap_matrix: dict[str, dict[str, int]] = {}
    for a in ["seed", "rare_artifact", "semantic", "component", "2hop"]:
        overlap_matrix[a] = {}
        for b in ["seed", "rare_artifact", "semantic", "component", "2hop"]:
            overlap_matrix[a][b] = int(len(source_pairs[a] & source_pairs[b]))
    source_unique = {
        s: int(len(source_pairs[s] - set().union(*[p for k, p in source_pairs.items() if k != s])))
        for s in source_pairs
    }
    n_exact_1 = int((pd.to_numeric(union_df.get("source_count"), errors="coerce").fillna(0) == 1).sum()) if not union_df.empty else 0
    n_exact_2 = int((pd.to_numeric(union_df.get("source_count"), errors="coerce").fillna(0) == 2).sum()) if not union_df.empty else 0
    n_exact_3 = int((pd.to_numeric(union_df.get("source_count"), errors="coerce").fillna(0) == 3).sum()) if not union_df.empty else 0
    n_exact_4p = int((pd.to_numeric(union_df.get("source_count"), errors="coerce").fillna(0) >= 4).sum()) if not union_df.empty else 0
    semantic_plus_non = int(
        (
            union_df.get("from_semantic", False).astype(bool)
            & (
                union_df.get("from_rare_artifact", False).astype(bool)
                | union_df.get("from_component", False).astype(bool)
                | union_df.get("from_2hop", False).astype(bool)
            )
        ).sum()
    ) if not union_df.empty else 0
    semantic_only = int(
        (
            union_df.get("from_semantic", False).astype(bool)
            & ~union_df.get("from_rare_artifact", False).astype(bool)
            & ~union_df.get("from_component", False).astype(bool)
            & ~union_df.get("from_2hop", False).astype(bool)
            & ~union_df.get("from_seed", False).astype(bool)
        ).sum()
    ) if not union_df.empty else 0
    infra_only = int(
        (
            ~union_df.get("from_semantic", False).astype(bool)
            & (
                union_df.get("from_rare_artifact", False).astype(bool)
                | union_df.get("from_component", False).astype(bool)
                | union_df.get("from_2hop", False).astype(bool)
                | union_df.get("from_seed", False).astype(bool)
            )
        ).sum()
    ) if not union_df.empty else 0
    source_overlap = {
        "pair_overlap_matrix": overlap_matrix,
        "unique_pair_contribution_per_source": source_unique,
        "n_pairs_supported_by_exactly_1_source": int(n_exact_1),
        "n_pairs_supported_by_exactly_2_sources": int(n_exact_2),
        "n_pairs_supported_by_exactly_3_sources": int(n_exact_3),
        "n_pairs_supported_by_4_plus_sources": int(n_exact_4p),
        "n_candidates_semantic_plus_non_semantic": int(semantic_plus_non),
        "n_candidates_semantic_only": int(semantic_only),
        "n_candidates_infra_only": int(infra_only),
    }

    # gt_eval
    gt_eval: list[dict[str, Any]] = []
    union_deg_all = _degrees_from_pairs([str(x) for x in union_emails], union_pairs)
    for gt_path, gt_map in gt_maps.items():
        gt_ids = set(map(str, gt_map.keys()))
        touched = set(union_deg_all.keys()) & gt_ids
        gt_pos = _gt_positive_pairs(gt_map)
        recovered_union = union_pairs & gt_pos
        campaign_to_ids: dict[str, set[str]] = defaultdict(set)
        for eid, camp in gt_map.items():
            campaign_to_ids[str(camp)].add(str(eid))
        campaign_touch_at1 = 0
        campaign_touch_at2 = 0
        campaign_seeded_candidate = 0
        campaign_cross_component = 0
        for ids in campaign_to_ids.values():
            camp_pairs = set(combinations(sorted(ids), 2))
            recovered = union_pairs & set(_pair(a, b) for a, b in camp_pairs)
            if len(recovered) >= 1:
                campaign_touch_at1 += 1
            if len(recovered) >= 2:
                campaign_touch_at2 += 1
            if any((a, b) in source_pairs["seed"] or (a, b) in source_pairs["seed"] for a, b in recovered):
                campaign_seeded_candidate += 1
            if not union_df.empty:
                sub = union_df[
                    union_df["email_i"].isin(ids) & union_df["email_j"].isin(ids)
                    & union_df.get("both_in_seed_components", False).astype(bool)
                    & ~union_df.get("same_seed_component", False).astype(bool)
                ]
                if len(sub) > 0:
                    campaign_cross_component += 1
        gt_pos_total = int(len(gt_pos))
        gt_src_recall = {
            s: float(len(source_pairs[s] & gt_pos) / max(1, gt_pos_total))
            for s in ["seed", "rare_artifact", "semantic", "component", "2hop"]
        }
        gt_deg = {eid: union_deg_all.get(eid, 0) for eid in gt_ids}
        gt_eval.append(
            {
                "gt_path": gt_path,
                "gt_labeled_emails_total": int(len(gt_ids)),
                "gt_campaigns_total": int(len(campaign_to_ids)),
                "gt_positive_pairs_total": int(gt_pos_total),
                "gt_labeled_emails_touched_by_candidates": int(len(touched)),
                "pct_gt_labeled_emails_touched_by_candidates": float(len(touched) / max(1, len(gt_ids))),
                "candidate_pair_completeness": float(len(recovered_union) / max(1, gt_pos_total)),
                "candidate_pair_recall_by_source": gt_src_recall,
                "candidate_union_pair_precision_labeled": float(_pair_precision_labeled(union_pairs, gt_map)),
                "campaign_touch_distribution": {
                    "pct_gt_campaigns_with_at_least_1_candidate_pair": float(campaign_touch_at1 / max(1, len(campaign_to_ids))),
                    "pct_gt_campaigns_with_at_least_2_candidate_pairs": float(campaign_touch_at2 / max(1, len(campaign_to_ids))),
                    "pct_gt_campaigns_with_at_least_1_candidate_involving_seeded_email": float(campaign_seeded_candidate / max(1, len(campaign_to_ids))),
                    "pct_gt_campaigns_with_at_least_1_cross_component_candidate": float(campaign_cross_component / max(1, len(campaign_to_ids))),
                },
                "candidate_budget_on_gt_emails": {
                    "avg_candidates_per_gt_email": float(np.mean(list(gt_deg.values()))) if gt_deg else 0.0,
                    "median_candidates_per_gt_email": float(np.median(list(gt_deg.values()))) if gt_deg else 0.0,
                },
            }
        )

    candidate_oracle_ceiling: list[dict[str, Any]] = []
    if gt_maps:
        for gt_path, gt_map in gt_maps.items():
            row = _compute_candidate_oracle_ceiling_for_gt(
                gt_path=str(gt_path),
                gt_map=gt_map,
                union_pairs=union_pairs,
            )
            candidate_oracle_ceiling.append(_null_json_floats(row))

    p_scored_all = out_dir / "scored_clustering_edges_all.csv"
    scored_edges_for_diag: pd.DataFrame | None = None
    if p_scored_all.is_file():
        try:
            scored_edges_for_diag = pd.read_csv(p_scored_all)
        except Exception:
            scored_edges_for_diag = None

    failure_regime_diagnostics: list[dict[str, Any]] = []
    failure_regime_csv_rows: list[dict[str, Any]] = []
    if gt_maps:
        for gt_path, gt_map in gt_maps.items():
            diag, fr_rows = _compute_failure_regime_diagnostics_for_gt(
                gt_path=str(gt_path),
                gt_map=gt_map,
                union_df=union_df,
                scored_edges_df=scored_edges_for_diag,
            )
            failure_regime_diagnostics.append(_null_json_floats(diag))
            failure_regime_csv_rows.extend(fr_rows)

    # ablations
    present_non_seed = [s for s in ["rare_artifact", "semantic", "component", "2hop"] if len(source_pairs.get(s, set())) > 0]
    variants_def: list[tuple[str, set[str]]] = [
        ("full_union", set(["seed"] + present_non_seed)),
        ("no_semantic", set(["seed"] + [s for s in present_non_seed if s != "semantic"])),
        ("no_rare_artifact", set(["seed"] + [s for s in present_non_seed if s != "rare_artifact"])),
        ("no_component", set(["seed"] + [s for s in present_non_seed if s != "component"])),
        ("no_2hop", set(["seed"] + [s for s in present_non_seed if s != "2hop"])),
        ("rare_artifact_only", set(["seed", "rare_artifact"]) & set(["seed"] + present_non_seed)),
        ("semantic_only", set(["seed", "semantic"]) & set(["seed"] + present_non_seed)),
        ("rare_artifact_plus_semantic", set(["seed", "rare_artifact", "semantic"]) & set(["seed"] + present_non_seed)),
        ("rare_artifact_plus_semantic_plus_component", set(["seed", "rare_artifact", "semantic", "component"]) & set(["seed"] + present_non_seed)),
    ]
    # dedupe variants by effective source set
    seen_variant_sets: set[tuple[str, ...]] = set()
    variants: list[tuple[str, set[str]]] = []
    for name, inc in variants_def:
        key = tuple(sorted(inc))
        if key in seen_variant_sets:
            continue
        seen_variant_sets.add(key)
        variants.append((name, inc))

    full_pairs = _variant_pairs(included=dict(variants).get("full_union", set(source_pairs.keys())), source_pairs=source_pairs)
    ablations: list[dict[str, Any]] = []
    ablation_csv_rows: list[dict[str, Any]] = []
    for idx, (name, included) in enumerate(variants):
        pairs = _variant_pairs(included=included, source_pairs=source_pairs)
        touched = sorted(set(a for a, _ in pairs).union(set(b for _, b in pairs)))
        deg = _degrees_from_pairs(touched, pairs)
        med = float(np.median(list(deg.values()))) if deg else 0.0
        avg = float(np.mean(list(deg.values()))) if deg else 0.0
        ab: dict[str, Any] = {
            "variant_name": name,
            "included_sources": sorted(included),
            "n_candidate_pairs": int(len(pairs)),
            "n_unique_emails": int(len(touched)),
            "avg_candidates_per_email": float(avg),
            "median_candidates_per_email": float(med),
            "reduction_ratio": float(len(pairs) / max(1, all_pairs_total)),
        }
        gt_rows = []
        for g in gt_eval:
            gt_path = str(g["gt_path"])
            gt_map = gt_maps.get(gt_path, {})
            gt_pos = _gt_positive_pairs(gt_map)
            campaigns = defaultdict(set)
            for eid, camp in gt_map.items():
                campaigns[str(camp)].add(str(eid))
            camp_touch = int(sum(1 for ids in campaigns.values() if len(_pairs_from_df(pd.DataFrame([{"email_i": a, "email_j": b} for a, b in pairs if a in ids and b in ids]))) > 0))
            touched_gt = int(len(set(touched) & set(gt_map.keys())))
            gt_rows.append(
                {
                    "gt_path": gt_path,
                    "candidate_pair_completeness": float(len(pairs & gt_pos) / max(1, len(gt_pos))),
                    "gt_campaigns_touched_pct": float(camp_touch / max(1, len(campaigns))),
                    "gt_labeled_emails_touched_pct": float(touched_gt / max(1, len(gt_map))),
                }
            )
        ab["gt"] = gt_rows
        if name != "full_union":
            full_gt0 = (gt_eval[0]["candidate_pair_completeness"] if gt_eval else float("nan"))
            cur_gt0 = (gt_rows[0]["candidate_pair_completeness"] if gt_rows else float("nan"))
            delta_first_gt = (float(cur_gt0 - full_gt0) if pd.notna(full_gt0) and pd.notna(cur_gt0) else None)
            ab["delta_vs_full_union"] = {
                "delta_candidate_pairs": int(len(pairs) - len(full_pairs)),
                "delta_candidate_pair_completeness_first_gt": delta_first_gt,
            }
        else:
            ab["delta_vs_full_union"] = {"delta_candidate_pairs": 0, "delta_candidate_pair_completeness_first_gt": 0.0}
        ablations.append(ab)
        ablation_csv_rows.append(
            {
                "variant_name": name,
                "n_candidate_pairs": int(len(pairs)),
                "n_unique_emails": int(len(touched)),
                "avg_candidates_per_email": float(avg),
                "median_candidates_per_email": float(med),
                "reduction_ratio": float(len(pairs) / max(1, all_pairs_total)),
                "gt_pair_completeness_if_available": float(gt_rows[0]["candidate_pair_completeness"]) if gt_rows else None,
                "gt_campaign_touch_if_available": float(gt_rows[0]["gt_campaigns_touched_pct"]) if gt_rows else None,
                "delta_vs_full_union_if_available": (
                    ab["delta_vs_full_union"]["delta_candidate_pair_completeness_first_gt"]
                    if name != "full_union"
                    else 0.0
                ),
            }
        )

    # seed_candidate_compatibility
    if seed_members_df.empty:
        seed_members_df = pd.DataFrame(columns=["external_id", "component_id", "component_size", "is_singleton"])
    if not seed_members_df.empty:
        seed_members_df["external_id"] = seed_members_df["external_id"].astype(str)
        seed_members_df["component_id"] = pd.to_numeric(seed_members_df["component_id"], errors="coerce").fillna(-1).astype(int)
        seed_members_df["component_size"] = pd.to_numeric(seed_members_df["component_size"], errors="coerce").fillna(1).astype(int)
        if "is_singleton" not in seed_members_df.columns:
            seed_members_df["is_singleton"] = seed_members_df["component_size"].eq(1)
    seeded_emails = set(seed_members_df.get("external_id", pd.Series(dtype=str)).astype(str))
    non_seed_pairs = union_pairs - seed_pairs
    seeded_non_seed_deg = _degrees_from_pairs(sorted(seeded_emails), set([p for p in non_seed_pairs if p[0] in seeded_emails or p[1] in seeded_emails]))
    seeded_with_1 = int(sum(1 for _, d in seeded_non_seed_deg.items() if d >= 1))
    seeded_with_5 = int(sum(1 for _, d in seeded_non_seed_deg.items() if d >= 5))

    comp_to_ids = (
        seed_members_df.groupby("component_id", dropna=False)["external_id"].apply(lambda s: sorted(set(s.astype(str)))).to_dict()
        if not seed_members_df.empty else {}
    )
    non_singleton_comp_ids = [int(cid) for cid, ids in comp_to_ids.items() if len(ids) >= 2]
    singleton_comp_ids = [int(cid) for cid, ids in comp_to_ids.items() if len(ids) == 1]
    comp_outward_counts: list[int] = []
    for cid in non_singleton_comp_ids:
        ids = set(comp_to_ids[cid])
        c = sum(1 for a, b in non_seed_pairs if (a in ids and b not in ids) or (b in ids and a not in ids))
        comp_outward_counts.append(c)
    singleton_counts: list[int] = []
    for cid in singleton_comp_ids:
        ids = set(comp_to_ids[cid])
        c = sum(1 for a, b in non_seed_pairs if a in ids or b in ids)
        singleton_counts.append(c)
    cross_component = int(
        (
            union_df.get("both_in_seed_components", False).astype(bool)
            & ~union_df.get("same_seed_component", False).astype(bool)
        ).sum()
    ) if not union_df.empty else 0
    internal_same = int(union_df.get("same_seed_component", False).astype(bool).sum()) if not union_df.empty else 0
    involving_non_seeded = int((~union_df.get("both_in_seed_components", False).astype(bool)).sum()) if not union_df.empty else 0

    seed_candidate_compatibility = {
        "n_seed_pairs_total": int(len(seed_pairs)),
        "n_seed_pairs_present_in_candidate_union": int(len(seed_pairs & union_pairs)),
        "seed_backbone_invariant_holds": bool(seed_pairs.issubset(union_pairs)),
        "n_non_seed_candidate_pairs": int(len(non_seed_pairs)),
        "candidate_to_seed_pair_ratio": float(len(non_seed_pairs) / max(1, len(seed_pairs))),
        "n_seeded_emails_total": int(len(seeded_emails)),
        "n_seeded_emails_with_at_least_1_non_seed_candidate": int(seeded_with_1),
        "pct_seeded_emails_with_at_least_1_non_seed_candidate": float(seeded_with_1 / max(1, len(seeded_emails))),
        "n_seeded_emails_with_at_least_5_non_seed_candidates": int(seeded_with_5),
        "pct_seeded_emails_with_at_least_5_non_seed_candidates": float(seeded_with_5 / max(1, len(seeded_emails))),
        "component_outward_expansion": {
            "n_non_singleton_seed_components": int(len(non_singleton_comp_ids)),
            "n_with_at_least_1_outward_candidate": int(sum(1 for x in comp_outward_counts if x >= 1)),
            "pct_with_at_least_1_outward_candidate": float(sum(1 for x in comp_outward_counts if x >= 1) / max(1, len(comp_outward_counts))),
            "avg_outward_candidate_count_per_component": float(np.mean(comp_outward_counts)) if comp_outward_counts else 0.0,
            "median_outward_candidate_count_per_component": float(np.median(comp_outward_counts)) if comp_outward_counts else 0.0,
        },
        "singleton_expansion": {
            "n_singleton_seed_components": int(len(singleton_comp_ids)),
            "n_with_at_least_1_candidate": int(sum(1 for x in singleton_counts if x >= 1)),
            "pct_with_at_least_1_candidate": float(sum(1 for x in singleton_counts if x >= 1) / max(1, len(singleton_counts))),
            "avg_candidates_per_singleton": float(np.mean(singleton_counts)) if singleton_counts else 0.0,
            "median_candidates_per_singleton": float(np.median(singleton_counts)) if singleton_counts else 0.0,
        },
        "cross_component_structure": {
            "n_candidate_pairs_connecting_different_seed_components": int(cross_component),
            "n_candidate_pairs_internal_same_seed_component": int(internal_same),
            "n_candidate_pairs_involving_at_least_one_non_seeded_email": int(involving_non_seeded),
            "cross_component_to_internal_ratio": float(cross_component / max(1, internal_same)),
        },
    }

    silver_eval: dict[str, Any] = {"enabled": False}
    if full_candidate_generation_config is not None:
        sb_cfg = (full_candidate_generation_config.get("evaluation") or {}).get("silver_hidden_link_benchmark") or {}
        if bool(sb_cfg.get("enabled", False)):
            try:
                from seed_candidate_workflow.utils.anchor_candidate_silver_hidden_link_helpers import (
                    run_silver_hidden_link_benchmark,
                )

                silver_pack = run_silver_hidden_link_benchmark(
                    project_root=project_root,
                    graph_id=graph_id,
                    main_out_dir=out_dir,
                    original_seed_dir=seed_dir,
                    full_generation_config=full_candidate_generation_config,
                    main_candidate_union_df=candidate_union_df,
                )
                silver_eval = silver_pack.get("silver_hidden_link_eval") or {"enabled": False}
            except Exception as exc:  # pragma: no cover - defensive benchmark isolation
                silver_eval = {
                    "enabled": True,
                    "benchmark_invalid": True,
                    "error": str(exc),
                    "notes": "Silver benchmark raised an exception; treat as invalid for go/no-go until fixed.",
                }

    # diagnostics
    n_union = max(1, len(union_pairs))
    semantic_only_pair_fraction = float(semantic_only / n_union)
    multi_source_pair_fraction = float(
        ((pd.to_numeric(union_df.get("source_count"), errors="coerce").fillna(0) >= 2).sum() / n_union)
    ) if not union_df.empty else 0.0
    hub_risk_warning = bool(candidate_universe["p95_candidates_per_email"] > float(readiness_cfg.get("hub_p95_threshold", 100.0)))
    oversparse_warning = bool(seed_candidate_compatibility["pct_seeded_emails_with_at_least_1_non_seed_candidate"] < float(readiness_cfg.get("min_seeded_email_nonseed_pct", 0.50)))
    overbroad_warning = bool(candidate_universe["avg_candidates_per_email"] > float(readiness_cfg.get("max_avg_candidates_per_email", 80.0)))
    semantic_dominance_warning = bool(semantic_only_pair_fraction > float(readiness_cfg.get("semantic_only_max_fraction", 0.70)))
    source_redundancy_warning = bool(source_overlap["n_pairs_supported_by_exactly_1_source"] > int(0.90 * len(union_pairs))) if union_pairs else False
    seed_backbone_warning = bool(not seed_candidate_compatibility["seed_backbone_invariant_holds"])
    singleton_isolation_warning = bool(seed_candidate_compatibility["singleton_expansion"]["pct_with_at_least_1_candidate"] < float(readiness_cfg.get("min_singleton_expansion_pct", 0.30)))
    diagnostics = {
        "semantic_only_pair_fraction": semantic_only_pair_fraction,
        "multi_source_pair_fraction": multi_source_pair_fraction,
        "hub_risk_warning": hub_risk_warning,
        "oversparse_warning": oversparse_warning,
        "overbroad_warning": overbroad_warning,
        "semantic_dominance_warning": semantic_dominance_warning,
        "source_redundancy_warning": source_redundancy_warning,
        "seed_backbone_warning": seed_backbone_warning,
        "singleton_isolation_warning": singleton_isolation_warning,
    }

    # generator status (called/completed/output/rows/reason)
    cfg_map = {k: dict(v) for k, v in (generator_configs or {}).items()}
    out_map = {}
    for row in (generator_outputs or []):
        name = str(row.get("name") or "").strip().lower()
        if name:
            out_map[name] = row
    label_to_name = {
        "rare_artifact": "rare_artifact_v1",
        "semantic": "semantic_reciprocal_v1",
        "component": "component_expansion_v1",
        "2hop": "2hop_bounded_v1",
    }
    generator_status: dict[str, dict[str, Any]] = {}
    for label, name in label_to_name.items():
        cfg = cfg_map.get(name, {})
        out = out_map.get(name, {})
        enabled = bool(cfg.get("enabled", False))
        csv_path = out.get("csv") or out.get("candidates_component_expanded_csv")
        if label == "component":
            csv_path = out.get("candidates_component_expanded_csv")
        called = bool(enabled and bool(out))
        completed = bool(called)
        rows = int(out.get("n_rows", out.get("n_candidate_rows", 0)) or 0)
        output_written = bool(csv_path and Path(str(csv_path)).is_file())
        zero_reason = None
        if enabled and rows == 0:
            zero_reason = "completed_with_zero_rows"
        elif not enabled:
            zero_reason = "disabled_in_config"
        generator_status[label] = {
            "generator_name": name,
            "enabled_in_config": enabled,
            "called": called,
            "completed_successfully": completed,
            "output_file_written": output_written,
            "rows_emitted": rows,
            "zero_rows_reason": zero_reason,
            "output_path": str(csv_path) if csv_path else None,
            "diagnostics": out.get("diagnostics"),
        }
    diagnostics["generator_status"] = generator_status

    # readiness
    blocking_reason_if_not_ready: list[str] = []
    key_positive_signals: list[str] = []
    key_warnings: list[str] = []
    if not seed_candidate_compatibility["seed_backbone_invariant_holds"]:
        blocking_reason_if_not_ready.append("seed_backbone_invariant_failed")
    min_ratio = float(readiness_cfg.get("min_candidate_to_seed_ratio", 1.5))
    if seed_candidate_compatibility["n_non_seed_candidate_pairs"] <= 0:
        blocking_reason_if_not_ready.append("candidate_universe_not_broader_than_seeds")
    elif seed_candidate_compatibility["candidate_to_seed_pair_ratio"] <= min_ratio:
        blocking_reason_if_not_ready.append("candidate_broadening_ratio_too_low")
    if len(source_pairs.get("component", set())) == 0 and len(source_pairs.get("2hop", set())) == 0:
        key_warnings.append("candidate_sources_missing_component_and_2hop")
    if gt_eval:
        mean_compl = float(np.mean([g["candidate_pair_completeness"] for g in gt_eval]))
        if mean_compl < float(readiness_cfg.get("min_gt_pair_completeness", 0.10)):
            blocking_reason_if_not_ready.append("gt_pair_completeness_too_low")
        else:
            key_positive_signals.append(f"gt_pair_completeness={mean_compl:.3f}")
    if diagnostics["oversparse_warning"]:
        key_warnings.append("seeded_emails_insufficiently_expanded")
    if diagnostics["overbroad_warning"]:
        key_warnings.append("candidate_budget_too_broad")
    if diagnostics["semantic_dominance_warning"]:
        key_warnings.append("semantic_only_dominance_high")
    if diagnostics["hub_risk_warning"]:
        key_warnings.append("candidate_hub_concentration_risk")
    if seed_candidate_compatibility["candidate_to_seed_pair_ratio"] > 2.0:
        key_positive_signals.append("candidate_universe_broader_than_seeds")
    if not diagnostics["seed_backbone_warning"]:
        key_positive_signals.append("seed_backbone_invariant_holds")

    if isinstance(silver_eval, dict) and bool(silver_eval.get("enabled")):
        if silver_eval.get("benchmark_invalid") or silver_eval.get("error"):
            key_warnings.append("silver_hidden_link_benchmark_invalid_or_errored_check_leak_and_logs")
        lc = silver_eval.get("leak_checks") or {}
        if lc.get("held_out_seed_leak_count"):
            key_warnings.append(
                f"silver_held_out_edges_still_present_in_benchmark_seed_edges_all_count={int(lc.get('held_out_seed_leak_count', 0))}"
            )
        if lc.get("seed_source_leak_warning"):
            key_warnings.append("silver_held_out_recovered_via_seed_source_treat_as_leak")
        ur = silver_eval.get("universe_recovery") or {}
        rec = ur.get("union_recall_on_held_out_silver")
        if isinstance(rec, (int, float)) and not math.isnan(float(rec)) and not silver_eval.get("benchmark_invalid"):
            if float(rec) >= 0.55:
                key_positive_signals.append(f"silver_hidden_link_union_recall_on_held_out={float(rec):.3f}")
            elif float(rec) < 0.25:
                key_warnings.append(f"silver_hidden_link_union_recall_on_held_out_low={float(rec):.3f}")
        pct_sem_only = silver_eval.get("pct_recovered_held_out_semantic_only_of_union_recovered_edges")
        urec_n = ur.get("union_recovered_edge_count")
        if (
            isinstance(pct_sem_only, (int, float))
            and not math.isnan(float(pct_sem_only))
            and isinstance(urec_n, int)
            and urec_n > 0
            and float(pct_sem_only) > 0.75
        ):
            key_warnings.append("silver_hidden_link_recovery_overly_dependent_on_semantic_only_paths")

    for oc_row in candidate_oracle_ceiling:
        vm = oc_row.get("v_measure")
        cm = oc_row.get("completeness")
        ne = int(oc_row.get("n_eval") or 0)
        if isinstance(vm, (int, float)) and not (isinstance(vm, float) and math.isnan(float(vm))):
            if float(vm) >= 0.5:
                key_positive_signals.append(
                    f"candidate_oracle_ceiling_v_measure={float(vm):.3f} ({Path(str(oc_row.get('gt_path', ''))).name})"
                )
            elif float(vm) < 0.25 and ne >= 10:
                key_warnings.append(
                    f"candidate_oracle_ceiling_v_measure_low={float(vm):.3f} ({Path(str(oc_row.get('gt_path', ''))).name})"
                )
        if (
            isinstance(cm, (int, float))
            and not (isinstance(cm, float) and math.isnan(float(cm)))
            and float(cm) < 0.2
            and ne >= 20
        ):
            key_warnings.append(
                f"candidate_oracle_ceiling_completeness_low={float(cm):.3f} ({Path(str(oc_row.get('gt_path', ''))).name})"
            )

    ready_for_next_stage = len(blocking_reason_if_not_ready) == 0
    readiness = {
        "ready_for_next_stage": bool(ready_for_next_stage),
        "ready_for_pu_dataset_construction": bool(ready_for_next_stage),
        "blocking_reason_if_not_ready": blocking_reason_if_not_ready,
        "key_positive_signals": key_positive_signals,
        "key_warnings": key_warnings,
    }
    # manual review sample
    sample_df = union_df.copy()
    if sample_df.empty:
        sample_df = pd.DataFrame(columns=["email_i", "email_j"])
    sample_df["source_flags"] = sample_df.apply(
        lambda r: "|".join(
            [
                s
                for s, col in [
                    ("seed", "from_seed"),
                    ("rare_artifact", "from_rare_artifact"),
                    ("semantic", "from_semantic"),
                    ("component", "from_component"),
                    ("2hop", "from_2hop"),
                ]
                if bool(r.get(col, False))
            ]
        ),
        axis=1,
    ) if not sample_df.empty else pd.Series(dtype=str)
    sample_df["semantic_involved"] = sample_df.get("from_semantic", False).astype(bool) if not sample_df.empty else False
    sample_df["semantic_only"] = (
        sample_df.get("from_semantic", False).astype(bool)
        & ~sample_df.get("from_rare_artifact", False).astype(bool)
        & ~sample_df.get("from_component", False).astype(bool)
        & ~sample_df.get("from_2hop", False).astype(bool)
        & ~sample_df.get("from_seed", False).astype(bool)
    ) if not sample_df.empty else False
    sample_df["seed_status_i"] = sample_df.get("email_i_seed_component_id", -1).fillna(-1).astype(int).ge(0) if not sample_df.empty else False
    sample_df["seed_status_j"] = sample_df.get("email_j_seed_component_id", -1).fillna(-1).astype(int).ge(0) if not sample_df.empty else False
    sample_df["seed_component_i"] = sample_df.get("email_i_seed_component_id", -1) if not sample_df.empty else -1
    sample_df["seed_component_j"] = sample_df.get("email_j_seed_component_id", -1) if not sample_df.empty else -1
    sample_df["same_seed_component_flag"] = sample_df.get("same_seed_component", False).astype(bool) if not sample_df.empty else False
    sample_df["time_gap_if_available"] = sample_df.get("time_gap_seconds_min", float("nan")) if not sample_df.empty else float("nan")

    # optional GT labels for sample (first GT only for deterministic annotation)
    gt_first = gt_eval[0]["gt_path"] if gt_eval else None
    gt_first_map = gt_maps.get(gt_first, {}) if gt_first else {}
    if not sample_df.empty and gt_first_map:
        sample_df["gt_campaign_i_if_available"] = sample_df["email_i"].map(gt_first_map)
        sample_df["gt_campaign_j_if_available"] = sample_df["email_j"].map(gt_first_map)
        sample_df["gt_same_campaign_if_available"] = (
            sample_df["gt_campaign_i_if_available"].notna()
            & sample_df["gt_campaign_j_if_available"].notna()
            & (sample_df["gt_campaign_i_if_available"] == sample_df["gt_campaign_j_if_available"])
        )
    else:
        sample_df["gt_campaign_i_if_available"] = np.nan
        sample_df["gt_campaign_j_if_available"] = np.nan
        sample_df["gt_same_campaign_if_available"] = np.nan

    groups: list[tuple[str, pd.Series]] = []
    if not sample_df.empty:
        groups.append(("rare_artifact", sample_df.get("from_rare_artifact", False).astype(bool)))
        groups.append(("semantic_only", sample_df.get("semantic_only", False).astype(bool)))
        groups.append(("component_expanded", sample_df.get("from_component", False).astype(bool)))
        groups.append(("2hop", sample_df.get("from_2hop", False).astype(bool)))
        groups.append(("cross_component", sample_df.get("both_in_seed_components", False).astype(bool) & ~sample_df.get("same_seed_component", False).astype(bool)))
        groups.append(("singleton_seed_involving", (sample_df.get("seed_component_i", -1).astype(int) >= 0) ^ (sample_df.get("seed_component_j", -1).astype(int) >= 0)))
        groups.append(("multi_source", pd.to_numeric(sample_df.get("source_count"), errors="coerce").fillna(0).ge(2)))

    sample_rows: list[pd.DataFrame] = []
    for i, (name, mask) in enumerate(groups):
        sub = sample_df[mask].copy()
        if sub.empty:
            continue
        n = min(per_group_n, len(sub))
        pick = sub.sample(n=n, random_state=manual_seed + i).copy()
        pick["sample_group"] = name
        sample_rows.append(pick)
    manual_review_df = pd.concat(sample_rows, axis=0, ignore_index=True) if sample_rows else pd.DataFrame(columns=list(sample_df.columns) + ["sample_group"])
    manual_review_cols = [
        "sample_group",
        "email_i",
        "email_j",
        "source_flags",
        "source_count",
        "semantic_involved",
        "semantic_only",
        "time_gap_if_available",
        "seed_status_i",
        "seed_status_j",
        "seed_component_i",
        "seed_component_j",
        "same_seed_component_flag",
        "gt_same_campaign_if_available",
        "gt_campaign_i_if_available",
        "gt_campaign_j_if_available",
    ]
    for c in manual_review_cols:
        if c not in manual_review_df.columns:
            manual_review_df[c] = np.nan
    manual_review_df = manual_review_df[manual_review_cols]
    p_manual = out_dir / "candidate_manual_review_sample.csv"
    manual_review_df.to_csv(p_manual, index=False)

    # Write ablation csv
    ablation_df = pd.DataFrame(ablation_csv_rows)
    p_ablation = out_dir / "candidate_source_ablation.csv"
    ablation_df.to_csv(p_ablation, index=False)

    p_oracle_ceiling = out_dir / "candidate_oracle_ceiling.csv"
    if candidate_oracle_ceiling:
        oracle_rows_flat = [{k: v for k, v in r.items() if k != "interpretation"} for r in candidate_oracle_ceiling]
        pd.DataFrame(oracle_rows_flat).to_csv(p_oracle_ceiling, index=False)
    else:
        pd.DataFrame(
            columns=[
                "gt_path",
                "n_eval",
                "n_gt_campaigns",
                "n_candidate_pairs_total",
                "n_oracle_same_campaign_candidate_pairs",
                "v_measure",
                "completeness",
                "oracle_campaign_reconnect_rate",
                "ceiling_strength_label",
            ]
        ).to_csv(p_oracle_ceiling, index=False)

    p_failure_regimes = out_dir / "candidate_failure_regimes.csv"
    _fr_cols = [
        "gt_path",
        "regime_family",
        "regime_name",
        "n_pairs",
        "n_oracle_good",
        "n_oracle_bad",
        "oracle_good_fraction",
        "oracle_bad_fraction",
    ]
    fr_df = pd.DataFrame(failure_regime_csv_rows, columns=_fr_cols) if failure_regime_csv_rows else pd.DataFrame(columns=_fr_cols)
    fr_df.to_csv(p_failure_regimes, index=False)

    # notes
    strongest_src = max(per_source.items(), key=lambda kv: kv[1].get("source_only_pairs_count", 0))[0] if per_source else "n/a"
    suspicious_src = max(per_source.items(), key=lambda kv: kv[1].get("n_candidate_pairs", 0))[0] if per_source else "n/a"
    notes_lines = [
        "What looks good",
        f"- Seed backbone invariant: {seed_candidate_compatibility['seed_backbone_invariant_holds']}",
        f"- Candidate/seed ratio: {seed_candidate_compatibility['candidate_to_seed_pair_ratio']:.3f}",
        "",
        "What looks risky",
        f"- Semantic-only fraction: {semantic_only_pair_fraction:.3f}",
        f"- p95 candidates/email: {candidate_universe['p95_candidates_per_email']:.2f}",
        "",
        "Should we move on?",
        f"- ready_for_next_stage={readiness['ready_for_next_stage']}",
        "",
        f"Which source is strongest",
        f"- {strongest_src}",
        "",
        f"Which source is most suspicious",
        f"- {suspicious_src}",
        "",
        "Whether singleton expansion looks healthy",
        f"- singleton_pct_with_at_least_1_candidate={seed_candidate_compatibility['singleton_expansion']['pct_with_at_least_1_candidate']:.3f}",
        "",
        "Whether semantic dominance is a concern",
        f"- semantic_dominance_warning={diagnostics['semantic_dominance_warning']}",
    ]
    if isinstance(silver_eval, dict) and bool(silver_eval.get("enabled")):
        ur = silver_eval.get("universe_recovery") or {}
        lc = silver_eval.get("leak_checks") or {}
        notes_lines.extend(
            [
                "",
                "Silver hidden-link benchmark (Step 4.4)",
                f"- benchmark_invalid={silver_eval.get('benchmark_invalid')}",
                f"- union_recall_on_held_out_silver={ur.get('union_recall_on_held_out_silver')}",
                f"- recovery_strength_label={silver_eval.get('recovery_strength_label')}",
                f"- held_out_seed_leak_count={lc.get('held_out_seed_leak_count')}",
                f"- held_out_recovered_via_seed_source_count={lc.get('held_out_recovered_via_seed_source_count')}",
                f"- seed_source_leak_warning={lc.get('seed_source_leak_warning')}",
                f"- benchmark_candidate_output_dir={silver_eval.get('benchmark_candidate_output_dir')}",
                f"- silver_hidden_link_summary_json={out_dir / 'silver_hidden_link_summary.json'}",
            ]
        )
    if candidate_oracle_ceiling:
        notes_lines.extend(["", "Candidate-universe oracle ceiling (GT-oracle edges only)"])
        for oc in candidate_oracle_ceiling:
            notes_lines.extend(
                [
                    f"- gt={Path(str(oc.get('gt_path', ''))).name}",
                    f"  v_measure={oc.get('v_measure')} completeness={oc.get('completeness')} "
                    f"oracle_campaign_reconnect_rate={oc.get('oracle_campaign_reconnect_rate')}",
                    f"  ceiling_strength_label={oc.get('ceiling_strength_label')}",
                ]
            )
    if failure_regime_diagnostics:
        notes_lines.extend(["", "Failure-regime / bad-bridge diagnostic (GT-covered candidate pairs)"])
        for fr in failure_regime_diagnostics:
            sig = (fr.get("pu_pull_viability_signals") or {}) if isinstance(fr.get("pu_pull_viability_signals"), dict) else {}
            notes_lines.extend(
                [
                    f"- gt={Path(str(fr.get('gt_path', ''))).name}",
                    f"  n_gt_covered_candidate_pairs={fr.get('n_gt_covered_candidate_pairs')} "
                    f"oracle_good={fr.get('n_oracle_good_pairs')} oracle_bad={fr.get('n_oracle_bad_pairs')}",
                    f"  pu_pull_viability_label={fr.get('viability_label')}",
                    f"  semantic_only_bad_enrichment={sig.get('semantic_only_bad_bridge_enrichment')} "
                    f"high_cos_low_support_bad={sig.get('high_cos_low_support_bad_bridge_enrichment')} "
                    f"multi_source_good={sig.get('multi_source_good_edge_enrichment')}",
                ]
            )
    p_notes = out_dir / "candidate_eval_notes.txt"
    p_notes.write_text("\n".join(notes_lines), encoding="utf-8")

    # JSON summary (ordered top-level contract)
    summary = OrderedDict()
    summary["metadata"] = metadata
    summary["candidate_universe"] = candidate_universe
    summary["per_source"] = per_source
    summary["source_overlap"] = source_overlap
    summary["gt_eval"] = gt_eval
    summary["candidate_oracle_ceiling"] = candidate_oracle_ceiling
    summary["failure_regime_diagnostics"] = failure_regime_diagnostics
    summary["ablations"] = ablations
    summary["seed_candidate_compatibility"] = seed_candidate_compatibility
    summary["silver_hidden_link_eval"] = silver_eval
    summary["diagnostics"] = {
        **diagnostics,
        "candidate_manual_review_sample_csv": str(p_manual),
        "candidate_oracle_ceiling_csv": str(p_oracle_ceiling),
        "candidate_failure_regimes_csv": str(p_failure_regimes),
    }
    summary["readiness"] = readiness

    p_summary = out_dir / "candidate_eval_summary.json"
    p_summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    return {
        "summary_json": str(p_summary),
        "ablation_csv": str(p_ablation),
        "manual_review_csv": str(p_manual),
        "oracle_ceiling_csv": str(p_oracle_ceiling),
        "failure_regimes_csv": str(p_failure_regimes),
        "notes_txt": str(p_notes),
        "readiness": readiness,
    }

