"""
Oracle / headroom analysis for semantic shards (GT used only for evaluation — never for training).

Maps shards to GT via **labeled emails inside each shard**:
- dominant campaign = argmax count of GT labels among labeled members
- dominant_fraction = count(dominant) / n_labeled_in_shard
- Shards with n_labeled == 0 are **unlabeled** for association rules
"""

from __future__ import annotations

import ast
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import completeness_score, homogeneity_score, v_measure_score
from sklearn.preprocessing import LabelEncoder

from analysis.utils.semantic_shard_edge_teacher_score import TEACHER_WEIGHT_COL, build_teacher_scored_edges
from analysis.utils.semantic_shard_step3_helpers import evaluate_external_metrics, map_shards_to_email_predictions

# ---------------------------------------------------------------------------
# Paths & loading
# ---------------------------------------------------------------------------


def load_step1_assignments(path: str | Path) -> pd.DataFrame:
    p = Path(path).expanduser().resolve()
    df = pd.read_csv(p)
    df["external_id"] = df["external_id"].astype(str)
    df["shard_id"] = df["shard_id"].astype(str)
    return df


def merge_scored_edge_columns(
    base_edges: pd.DataFrame,
    *,
    method1_edges: pd.DataFrame | None = None,
    v2_edges: pd.DataFrame | None = None,
    teacher_edges: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Left-merge optional score columns onto baseline edges (undirected key min/max shard ids)."""
    out = base_edges.copy()
    out["shard_a"] = out["shard_a"].astype(str)
    out["shard_b"] = out["shard_b"].astype(str)
    out["_ek"] = (
        np.minimum(out["shard_a"], out["shard_b"])
        + "||"
        + np.maximum(out["shard_a"], out["shard_b"])
    )

    def _merge_extra(df: pd.DataFrame | None, cols: list[str], prefix: str) -> None:
        nonlocal out
        if df is None:
            return
        d = df.copy()
        d["shard_a"] = d["shard_a"].astype(str)
        d["shard_b"] = d["shard_b"].astype(str)
        d["_ek"] = (
            np.minimum(d["shard_a"], d["shard_b"])
            + "||"
            + np.maximum(d["shard_a"], d["shard_b"])
        )
        take = [c for c in cols if c in d.columns]
        if not take:
            return
        sub = d[["_ek"] + take].drop_duplicates(subset=["_ek"])
        sub = sub.rename(columns={c: f"{prefix}{c}" for c in take})
        out = out.merge(sub, on="_ek", how="left")

    if method1_edges is not None:
        _merge_extra(method1_edges, ["edge_weight_refined"], "m1_")
    if v2_edges is not None:
        _merge_extra(v2_edges, ["edge_plausibility"], "v2_")
    if teacher_edges is not None:
        _merge_extra(teacher_edges, [TEACHER_WEIGHT_COL], "t_")

    return out.drop(columns=["_ek"], errors="ignore")


# ---------------------------------------------------------------------------
# Shard ↔ GT (labeled subset)
# ---------------------------------------------------------------------------


def build_shard_gt_summary(
    assignments_df: pd.DataFrame,
    gt_label_map: dict[str, Any],
    *,
    min_labeled_for_stats: int = 1,
) -> pd.DataFrame:
    """
    Per-shard GT statistics on the labeled subset (emails appearing in ``gt_label_map``).
    """
    a = assignments_df.copy()
    a["external_id"] = a["external_id"].astype(str)
    a["shard_id"] = a["shard_id"].astype(str)
    gt = {str(k): v for k, v in gt_label_map.items()}

    rows: list[dict[str, Any]] = []
    for sid, g in a.groupby("shard_id"):
        eids = g["external_id"].tolist()
        n_total = len(eids)
        labeled = [e for e in eids if e in gt]
        n_lab = len(labeled)
        if n_lab < min_labeled_for_stats:
            rows.append(
                {
                    "shard_id": sid,
                    "n_members_total": n_total,
                    "n_labeled": n_lab,
                    "dominant_campaign": None,
                    "dominant_count": 0,
                    "dominant_fraction": float("nan"),
                    "n_distinct_gt_campaigns": 0,
                    "gt_entropy": float("nan"),
                    "is_pure_labeled": False,
                }
            )
            continue
        counts = Counter(gt[e] for e in labeled)
        dom_c, dom_n = counts.most_common(1)[0]
        probs = np.array([counts[c] for c in counts], dtype=np.float64) / n_lab
        ent = float(-np.sum(probs * np.log(probs + 1e-15)))
        rows.append(
            {
                "shard_id": sid,
                "n_members_total": n_total,
                "n_labeled": n_lab,
                "dominant_campaign": dom_c,
                "dominant_count": int(dom_n),
                "dominant_fraction": float(dom_n / n_lab),
                "n_distinct_gt_campaigns": int(len(counts)),
                "gt_entropy": ent,
                "is_pure_labeled": bool(len(counts) == 1),
            }
        )
    return pd.DataFrame(rows)


def metrics_pure_shards_as_clusters(
    assignments_df: pd.DataFrame,
    gt_label_map: dict[str, Any],
) -> dict[str, Any]:
    """Each shard is one predicted cluster (factorized); compare to email GT on intersection."""
    a = assignments_df.copy()
    a["external_id"] = a["external_id"].astype(str)
    gt = {str(k): v for k, v in gt_label_map.items()}
    sub = a[a["external_id"].isin(gt.keys())].copy()
    if sub.empty:
        return {
            "n_eval": 0,
            "homogeneity": float("nan"),
            "completeness": float("nan"),
            "v_measure": float("nan"),
            "n_shards_with_labeled": 0,
        }
    y_pred = LabelEncoder().fit_transform(sub["shard_id"].astype(str))
    y_true_raw = [gt[e] for e in sub["external_id"]]
    # Encode mixed-type campaign ids as contiguous integers
    y_true_codes = LabelEncoder().fit_transform([str(x) for x in y_true_raw])
    return {
        "n_eval": int(len(sub)),
        "homogeneity": float(homogeneity_score(y_true_codes, y_pred)),
        "completeness": float(completeness_score(y_true_codes, y_pred)),
        "v_measure": float(v_measure_score(y_true_codes, y_pred)),
        "n_shards_with_labeled": int(sub["shard_id"].nunique()),
    }


def top_mixed_shards(shard_summary: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    lab = shard_summary[shard_summary["n_labeled"] > 0].copy()
    mixed = lab[~lab["is_pure_labeled"]].sort_values(
        ["dominant_fraction", "n_labeled"], ascending=[True, False]
    )
    return mixed.head(int(n)).reset_index(drop=True)


def shard_purity_global_summary(shard_summary: pd.DataFrame, gt_label_map: dict[str, Any]) -> dict[str, Any]:
    """Aggregate shard purity stats (labeled-weighted)."""
    s = shard_summary.copy()
    lab = s[s["n_labeled"] > 0]
    if lab.empty:
        return {"n_mixed_shards": 0, "frac_labeled_in_pure_shards": float("nan")}
    pure = lab[lab["is_pure_labeled"]]
    n_lab_total = int(lab["n_labeled"].sum())
    n_pure_emails = int(pure["n_labeled"].sum()) if not pure.empty else 0
    mixed = lab[~lab["is_pure_labeled"]]
    w_dom = float((lab["n_labeled"] * lab["dominant_fraction"].fillna(0)).sum() / max(1, n_lab_total))
    return {
        "n_shards_with_labeled": int(len(lab)),
        "n_pure_shards": int(len(pure)),
        "n_mixed_shards": int(len(mixed)),
        "frac_labeled_in_pure_shards": float(n_pure_emails / max(1, n_lab_total)),
        "weighted_avg_dominant_fraction": w_dom,
    }


# ---------------------------------------------------------------------------
# Oracle merge maps
# ---------------------------------------------------------------------------


def _shard_to_oracle_community_ids(
    shard_summary: pd.DataFrame,
    *,
    purity_threshold: float,
    graph_edges: pd.DataFrame | None,
) -> dict[str, int]:
    """
    Map shard_id -> integer community.

    If graph_edges is None (**Oracle A**): merge all shards with
    ``dominant_fraction >= purity_threshold`` that share the same ``dominant_campaign``.
    Other shards each get a unique community.

    If graph_edges is set (**Oracle B**): start from an empty graph on all shards; add a Step-2 edge
    only if both endpoints are pure (same threshold) **and** same ``dominant_campaign``.
    Connected components define communities; impure / unlabeled shards are isolates (unique ids).
    """
    import networkx as nx

    s = shard_summary.set_index("shard_id")
    next_id = 0
    shard_to_comm: dict[str, int] = {}

    all_shards = [str(x) for x in shard_summary["shard_id"].astype(str)]
    pure_rows = shard_summary[
        (shard_summary["n_labeled"] > 0) & (shard_summary["dominant_fraction"] >= purity_threshold)
    ]
    pure_shards = set(pure_rows["shard_id"].astype(str))

    if graph_edges is None:
        camp_to_comm: dict[Any, int] = {}
        for _, r in pure_rows.iterrows():
            sid = str(r["shard_id"])
            c = r["dominant_campaign"]
            if c not in camp_to_comm:
                camp_to_comm[c] = next_id
                next_id += 1
            shard_to_comm[sid] = camp_to_comm[c]
    else:
        h = nx.Graph()
        for sid in all_shards:
            h.add_node(sid)
        e = graph_edges.copy()
        for _, r in e.iterrows():
            u, v = str(r["shard_a"]), str(r["shard_b"])
            if u == v:
                continue
            if u not in pure_shards or v not in pure_shards:
                continue
            cu = s.loc[u, "dominant_campaign"] if u in s.index else None
            cv = s.loc[v, "dominant_campaign"] if v in s.index else None
            if cu is None or cv is None or cu != cv:
                continue
            h.add_edge(u, v)
        for comp in nx.connected_components(h):
            cid = next_id
            next_id += 1
            for sid in comp:
                shard_to_comm[str(sid)] = cid

    for sid in all_shards:
        if sid not in shard_to_comm:
            shard_to_comm[sid] = next_id
            next_id += 1
    return shard_to_comm


def oracle_metrics_from_shard_map(
    assignments_df: pd.DataFrame,
    shard_to_comm: dict[str, int],
    gt_label_map: dict[str, Any],
) -> dict[str, Any]:
    email_pred = map_shards_to_email_predictions(assignments_df, shard_to_comm)
    m = evaluate_external_metrics(email_pred, gt_label_map)
    n_comm = len(set(shard_to_comm.values()))
    n_shard = len(shard_to_comm)
    return {
        **m,
        "n_communities": float(n_comm),
        "n_shards": float(n_shard),
        "n_shards_merged_proxy": float(n_shard - n_comm),
    }


# ---------------------------------------------------------------------------
# Edge taxonomy
# ---------------------------------------------------------------------------


EDGE_TAXONOMY_SAME = "same_campaign_edge"
EDGE_TAXONOMY_CROSS = "cross_campaign_edge"
EDGE_TAXONOMY_AMBIG = "ambiguous_edge"


def label_candidate_edges_taxonomy(
    edges_df: pd.DataFrame,
    shard_summary: pd.DataFrame,
    *,
    min_labeled_per_endpoint: int = 1,
    min_dominant_fraction: float = 0.7,
) -> pd.DataFrame:
    """
    Classify each undirected edge using shard dominant GT and thresholds.

    - Both endpoints need n_labeled >= min_labeled_per_endpoint and dominant_fraction >= min_dominant_fraction
    - same if dominant_campaign matches
    - cross if both confident and differ
    - else ambiguous
    """
    s = shard_summary.set_index("shard_id")
    out = edges_df.copy()
    out["shard_a"] = out["shard_a"].astype(str)
    out["shard_b"] = out["shard_b"].astype(str)

    def _row_side(sid: str) -> tuple[Any, float, int, bool]:
        if sid not in s.index:
            return None, 0.0, 0, False
        r = s.loc[sid]
        nlab = int(r["n_labeled"])
        dom = r["dominant_campaign"]
        frac = float(r["dominant_fraction"]) if pd.notna(r["dominant_fraction"]) else 0.0
        ok = nlab >= min_labeled_per_endpoint and frac >= min_dominant_fraction
        return dom, frac, nlab, ok

    tax: list[str] = []
    da_list: list[Any] = []
    db_list: list[Any] = []
    for _, r in out.iterrows():
        da, fa, na, oka = _row_side(str(r["shard_a"]))
        db, fb, nb, okb = _row_side(str(r["shard_b"]))
        da_list.append(da)
        db_list.append(db)
        if not oka or not okb:
            tax.append(EDGE_TAXONOMY_AMBIG)
        elif da is not None and db is not None and da == db:
            tax.append(EDGE_TAXONOMY_SAME)
        elif da is not None and db is not None and da != db:
            tax.append(EDGE_TAXONOMY_CROSS)
        else:
            tax.append(EDGE_TAXONOMY_AMBIG)
    out["edge_taxonomy"] = tax
    out["shard_a_dominant_campaign"] = da_list
    out["shard_b_dominant_campaign"] = db_list
    return out


# ---------------------------------------------------------------------------
# Feature / score summaries
# ---------------------------------------------------------------------------

DEFAULT_EDGE_FEATURE_COLS = [
    "centroid_cosine",
    "infra_score",
    "temporal_score",
    "shared_url_count",
    "shared_url_idf_sum",
    "url_jaccard",
    "edge_weight",
    "v2_infra_dominance",
    "v2_local_embeddedness_rank",
    "v2_local_common_n_rank",
]


def summarize_numeric_by_taxonomy(
    edges_labeled: pd.DataFrame,
    cols: list[str],
    *,
    taxonomy_col: str = "edge_taxonomy",
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for tax in edges_labeled[taxonomy_col].dropna().unique():
        sub = edges_labeled[edges_labeled[taxonomy_col] == tax]
        row: dict[str, Any] = {"edge_taxonomy": str(tax), "n_edges": int(len(sub))}
        for c in cols:
            if c not in sub.columns:
                continue
            x = pd.to_numeric(sub[c], errors="coerce").dropna()
            if x.empty:
                row[f"{c}_mean"] = float("nan")
                row[f"{c}_median"] = float("nan")
            else:
                row[f"{c}_mean"] = float(x.mean())
                row[f"{c}_median"] = float(x.median())
        rows.append(row)
    return pd.DataFrame(rows)


def regime_slice_summary(edges_labeled: pd.DataFrame, *, taxonomy_col: str = "edge_taxonomy") -> pd.DataFrame:
    """Heuristic cross-tabs: low sem / high infra, etc."""
    e = edges_labeled.copy()
    need = ["centroid_cosine", "infra_score"]
    if not all(c in e.columns for c in need):
        return pd.DataFrame()
    sem = pd.to_numeric(e["centroid_cosine"], errors="coerce")
    inf = pd.to_numeric(e["infra_score"], errors="coerce")
    e["_lo_sem"] = sem < sem.median()
    e["_hi_inf"] = inf > inf.median()
    rows = []
    for name, mask in [
        ("low_sem_high_infra", e["_lo_sem"] & e["_hi_inf"]),
        ("high_sem_low_infra", ~e["_lo_sem"] & ~e["_hi_inf"]),
        ("low_sem_low_infra", e["_lo_sem"] & ~e["_hi_inf"]),
        ("high_sem_high_infra", ~e["_lo_sem"] & e["_hi_inf"]),
    ]:
        sub = e[mask & sem.notna() & inf.notna()]
        for tax in [EDGE_TAXONOMY_SAME, EDGE_TAXONOMY_CROSS, EDGE_TAXONOMY_AMBIG]:
            k = sub[sub[taxonomy_col] == tax]
            rows.append({"regime": name, "edge_taxonomy": tax, "n_edges": int(len(k))})
    return pd.DataFrame(rows)


def regime_blend_by_taxonomy(
    edges_labeled: pd.DataFrame,
    *,
    taxonomy_col: str = "edge_taxonomy",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Pivot regime_slice_summary into wide tables for notebooks.

    Returns (counts, frac_within_taxonomy):
    - counts: rows = regime, columns = taxonomy, cell = edge count (among edges with finite
      centroid_cosine and infra_score; medians for lo/hi splits are global on that subset).
    - frac_within_taxonomy: same shape; each column sums to 1.0 (mix of regimes **within**
      that edge class). Compare same_campaign vs cross_campaign columns to see whether
      “successful” bridges look structurally different from harmful ones.
    """
    long = regime_slice_summary(edges_labeled, taxonomy_col=taxonomy_col)
    if long.empty:
        return pd.DataFrame(), pd.DataFrame()
    pivot = long.pivot(index="regime", columns="edge_taxonomy", values="n_edges").fillna(0.0)
    col_order = [
        c for c in (EDGE_TAXONOMY_SAME, EDGE_TAXONOMY_CROSS, EDGE_TAXONOMY_AMBIG) if c in pivot.columns
    ]
    pivot = pivot[col_order].astype(int)
    col_sums = pivot.sum(axis=0)
    frac = pivot.div(col_sums.replace(0, np.nan), axis=1)
    return pivot, frac


def score_overlap_and_precision_at_k(
    edges_labeled: pd.DataFrame,
    score_col: str,
    *,
    taxonomy_col: str = "edge_taxonomy",
    k_list: tuple[int, ...] = (100, 500, 1000),
) -> dict[str, Any]:
    if score_col not in edges_labeled.columns:
        return {"score_col": score_col, "error": "missing_column"}
    e = edges_labeled.copy()
    s = pd.to_numeric(e[score_col], errors="coerce")
    m = taxonomy_col
    same = e[m] == EDGE_TAXONOMY_SAME
    cross = e[m] == EDGE_TAXONOMY_CROSS
    out: dict[str, Any] = {"score_col": score_col}
    for label, mask in [("same_campaign", same), ("cross_campaign", cross)]:
        x = s[mask].dropna()
        out[f"{label}_n"] = int(len(x))
        out[f"{label}_mean"] = float(x.mean()) if len(x) else float("nan")
        out[f"{label}_median"] = float(x.median()) if len(x) else float("nan")
    # precision@k on labeled same vs cross (exclude ambiguous from ranking pool or keep all?)
    pool = e[s.notna() & e[m].isin([EDGE_TAXONOMY_SAME, EDGE_TAXONOMY_CROSS])].copy()
    pool["_s"] = pd.to_numeric(pool[score_col], errors="coerce")
    pool = pool[pool["_s"].notna()].sort_values("_s", ascending=False)
    n = len(pool)
    for k in k_list:
        kk = min(k, n)
        if kk == 0:
            out[f"precision_at_{k}_same"] = float("nan")
            continue
        top = pool.head(kk)
        out[f"precision_at_{k}_same"] = float((top[m] == EDGE_TAXONOMY_SAME).mean())
    return out


def attach_shard_context(edges_df: pd.DataFrame, shard_summary: pd.DataFrame) -> pd.DataFrame:
    """Add per-endpoint shard size / dominant GT / purity from ``shard_summary``."""
    sa = shard_summary.rename(
        columns={c: ("shard_a" if c == "shard_id" else f"a_{c}") for c in shard_summary.columns}
    )
    sb = shard_summary.rename(
        columns={c: ("shard_b" if c == "shard_id" else f"b_{c}") for c in shard_summary.columns}
    )
    out = edges_df.merge(sa, on="shard_a", how="left")
    out = out.merge(sb, on="shard_b", how="left")
    return out


def extract_priority_edge_tables(
    edges_enriched: pd.DataFrame,
    *,
    score_cols: list[str],
    taxonomy_col: str = "edge_taxonomy",
    top_n: int = 50,
) -> dict[str, pd.DataFrame]:
    """
    Missed bridges: same-campaign, low score on primary scorer.
    False bridges: cross-campaign, high score.
    """
    out: dict[str, pd.DataFrame] = {}
    e = edges_enriched.copy()
    primary = score_cols[0] if score_cols else "edge_weight"
    if primary not in e.columns:
        return out
    s = pd.to_numeric(e[primary], errors="coerce")
    same = e[e[taxonomy_col] == EDGE_TAXONOMY_SAME].assign(_s=s)
    cross = e[e[taxonomy_col] == EDGE_TAXONOMY_CROSS].assign(_s=s)
    out["missed_bridges_low_score"] = same[same["_s"].notna()].nsmallest(top_n, "_s").drop(
        columns=["_s"], errors="ignore"
    )
    out["false_bridges_high_score"] = cross[cross["_s"].notna()].nlargest(top_n, "_s").drop(
        columns=["_s"], errors="ignore"
    )
    return out


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------


def oracle_scenarios_comparison_df(
    *,
    pure_shard_metrics: dict[str, Any],
    oracle_a: dict[str, Any],
    oracle_b: dict[str, Any],
    baseline_sweep_best: dict[str, Any] | None = None,
    teacher_sweep_best: dict[str, Any] | None = None,
    v2_sweep_best: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Single table for CSV export comparing ceilings vs optional production sweeps."""

    def _row(name: str, d: dict[str, Any]) -> dict[str, Any]:
        return {
            "scenario": name,
            "homogeneity": d.get("homogeneity"),
            "completeness": d.get("completeness"),
            "v_measure": d.get("v_measure"),
            "n_eval": d.get("n_eval"),
            "n_communities": d.get("n_communities"),
            "coverage_gt": d.get("coverage_gt"),
        }

    rows = [
        _row("1_pure_shards_as_clusters", pure_shard_metrics),
        _row("2_oracle_A_merge_same_campaign_unconstrained", oracle_a),
        _row("3_oracle_B_merge_same_campaign_graph_constrained", oracle_b),
    ]
    if baseline_sweep_best:
        rows.append(_row("4_baseline_sweep_best_row", baseline_sweep_best))
    if teacher_sweep_best:
        rows.append(_row("5_teacher_sweep_best_row", teacher_sweep_best))
    if v2_sweep_best:
        rows.append(_row("6_v2_sweep_best_row", v2_sweep_best))
    return pd.DataFrame(rows)


def write_oracle_headroom_summary_md(
    path: str | Path,
    *,
    pure_shard_metrics: dict[str, Any],
    oracle_a: dict[str, Any],
    oracle_b: dict[str, Any],
    purity_global: dict[str, Any],
    taxonomy_counts: pd.Series,
    extra_bullets: list[str] | None = None,
) -> None:
    lines = [
        "# Semantic shard oracle / headroom summary",
        "",
        "**Ground truth is used only for this offline analysis — not for model training.**",
        "",
        "## 1. Pure shards (Stage 1) as clusters",
        "",
        f"- n_eval (labeled emails): {pure_shard_metrics.get('n_eval', 0)}",
        f"- homogeneity: {pure_shard_metrics.get('homogeneity', float('nan')):.6f}",
        f"- completeness: {pure_shard_metrics.get('completeness', float('nan')):.6f}",
        f"- V-measure: {pure_shard_metrics.get('v_measure', float('nan')):.6f}",
        "",
        "## 2. Oracle A — perfect same-campaign merge (no graph constraint)",
        "",
        f"- homogeneity: {oracle_a.get('homogeneity', float('nan')):.6f}",
        f"- completeness: {oracle_a.get('completeness', float('nan')):.6f}",
        f"- V-measure: {oracle_a.get('v_measure', float('nan')):.6f}",
        f"- n_communities: {oracle_a.get('n_communities', float('nan'))}",
        f"- n_shards_merged_proxy: {oracle_a.get('n_shards_merged_proxy', float('nan'))}",
        "",
        "## 3. Oracle B — graph-constrained same-campaign merge",
        "",
        f"- homogeneity: {oracle_b.get('homogeneity', float('nan')):.6f}",
        f"- completeness: {oracle_b.get('completeness', float('nan')):.6f}",
        f"- V-measure: {oracle_b.get('v_measure', float('nan')):.6f}",
        f"- n_communities: {oracle_b.get('n_communities', float('nan'))}",
        f"- n_shards_merged_proxy: {oracle_b.get('n_shards_merged_proxy', float('nan'))}",
        "",
        "## 4. Shard purity (labeled)",
        "",
        f"- n_mixed_shards: {purity_global.get('n_mixed_shards', 'n/a')}",
        f"- frac_labeled_in_pure_shards: {purity_global.get('frac_labeled_in_pure_shards', float('nan'))}",
        f"- weighted_avg_dominant_fraction: {purity_global.get('weighted_avg_dominant_fraction', float('nan'))}",
        "",
        "## 5. Candidate edge taxonomy counts",
        "",
    ]
    if taxonomy_counts is not None and len(taxonomy_counts):
        for k, v in taxonomy_counts.items():
            lines.append(f"- {k}: {int(v)}")
    else:
        lines.append("- (no taxonomy)")
    lines.append("")
    lines.append("## 6. Headroom (qualitative)")
    lines.append("")
    lines.append(
        "- Compare Oracle A vs pure shards: room left if Stage 1 shards are impure or fragmented across campaigns."
    )
    lines.append(
        "- Compare Oracle B vs Oracle A: **graph connectivity** gap (same-campaign shards missing edges)."
    )
    lines.append(
        "- Compare production sweep best V-measure vs Oracle B: **merge scoring / method** gap."
    )
    if extra_bullets:
        lines.append("")
        for b in extra_bullets:
            lines.append(f"- {b}")
    Path(path).write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Shard-link diagnostics (edge_analysis_df)
# ---------------------------------------------------------------------------

LINK_LABEL_FROM_TAXONOMY: dict[str, str] = {
    EDGE_TAXONOMY_SAME: "same",
    EDGE_TAXONOMY_CROSS: "cross",
    EDGE_TAXONOMY_AMBIG: "ambiguous",
}

def _channel_title(base: str) -> str:
    if base == "sender_email_domain":
        return "Sender-domain"
    return base.replace("_", " ").title()


def infer_shared_overlap_specs(edge_df: pd.DataFrame) -> list[tuple[str, str, str]]:
    """
    Infer overlap channels from `shared_*_count` columns.
    Returns list of (base, count_col, has_flag_col).
    """
    specs: list[tuple[str, str, str]] = []
    for c in edge_df.columns:
        m = re.fullmatch(r"shared_(.+)_count", str(c))
        if not m:
            continue
        base = str(m.group(1))
        has_col = f"has_{base}_overlap"
        if base == "sender_email_domain":
            has_col = "has_sender_domain_overlap"
        specs.append((base, c, has_col))
    specs.sort(key=lambda x: x[0])
    return specs

_SEM_BODY_SUBJECT_TRY_PAIRS: tuple[tuple[str, str], ...] = (
    ("body_centroid_cosine", "subject_centroid_cosine"),
    ("centroid_cosine_body", "centroid_cosine_subject"),
    ("body_semantic_score", "subject_semantic_score"),
    ("semantic_body", "semantic_subject"),
)


def find_body_subject_semantic_columns(df: pd.DataFrame) -> tuple[str | None, str | None]:
    """Return (body_col, subject_col) if both exist, else (None, None)."""
    cols = set(df.columns)
    for b, s in _SEM_BODY_SUBJECT_TRY_PAIRS:
        if b in cols and s in cols:
            return b, s
    return None, None


def evidence_binary_column_specs(edge_df: pd.DataFrame) -> list[tuple[str, str]]:
    """(plot_label, column_name) for infra/temporal/local/optional semantic channel flags."""
    order: list[tuple[str, str]] = [
        ("URL overlap", "has_url_overlap"),
        ("Domain overlap", "has_domain_overlap"),
        ("Sender overlap", "has_sender_overlap"),
        ("Sender-domain overlap", "has_sender_domain_overlap"),
        ("Stem overlap", "has_stem_overlap"),
        ("Temporal support", "has_temporal_support"),
        ("Local support", "has_local_support"),
        ("Body semantic", "has_body_semantic"),
        ("Subject semantic", "has_subject_semantic"),
    ]
    out: list[tuple[str, str]] = []
    for lab, c in order:
        if c not in edge_df.columns:
            continue
        if pd.api.types.is_bool_dtype(edge_df[c]):
            out.append((lab, c))
    known = {c for _, c in out}
    for c in edge_df.columns:
        if not str(c).startswith("has_") or not str(c).endswith("_overlap"):
            continue
        if c in known or not pd.api.types.is_bool_dtype(edge_df[c]):
            continue
        base = str(c)[4:-8]
        out.append((_channel_title(base) + " overlap", str(c)))
    return out


def build_edge_analysis_dataframe(
    edges_df: pd.DataFrame,
    edges_taxonomy_labeled: pd.DataFrame,
) -> pd.DataFrame:
    """
    One row per Step-2 edge with GT-derived ``link_label`` and boolean evidence flags.

    Body/subject semantic flags are populated only when matching columns exist on ``edges_df``
    (Step-2 exports are usually **aggregate** ``centroid_cosine`` only).
    """
    tax_cols = [
        "shard_a",
        "shard_b",
        "edge_taxonomy",
        "shard_a_dominant_campaign",
        "shard_b_dominant_campaign",
    ]
    tax_cols = [c for c in tax_cols if c in edges_taxonomy_labeled.columns]
    tax_sub = edges_taxonomy_labeled[tax_cols].drop_duplicates(subset=["shard_a", "shard_b"])
    out = edges_df.merge(tax_sub, on=["shard_a", "shard_b"], how="left")
    out["shard_a"] = out["shard_a"].astype(str)
    out["shard_b"] = out["shard_b"].astype(str)

    tax = out["edge_taxonomy"].astype(str) if "edge_taxonomy" in out.columns else pd.Series("", index=out.index)
    out["link_label"] = tax.map(LINK_LABEL_FROM_TAXONOMY).fillna("ambiguous")
    out.loc[out["edge_taxonomy"].isna(), "link_label"] = "ambiguous"

    overlap_specs = infer_shared_overlap_specs(out)
    for _, count_col, flag in overlap_specs:
        if count_col in out.columns:
            v = pd.to_numeric(out[count_col], errors="coerce").fillna(0)
            out[flag] = v > 0
        else:
            out[flag] = False

    if "temporal_overlap" in out.columns:
        out["has_temporal_support"] = pd.to_numeric(out["temporal_overlap"], errors="coerce").fillna(0) > 0
    elif "temporal_score" in out.columns:
        out["has_temporal_support"] = pd.to_numeric(out["temporal_score"], errors="coerce").fillna(0) >= 0.5
    else:
        out["has_temporal_support"] = False

    if "v2_local_embeddedness_rank" in out.columns:
        r = pd.to_numeric(out["v2_local_embeddedness_rank"], errors="coerce")
        med = r.median()
        out["has_local_support"] = r.notna() & (r >= med)
    elif "v2_local_common_n_rank" in out.columns:
        r = pd.to_numeric(out["v2_local_common_n_rank"], errors="coerce")
        med = r.median()
        out["has_local_support"] = r.notna() & (r >= med)

    body_col, subj_col = find_body_subject_semantic_columns(out)
    thr = 1e-12
    if body_col and subj_col:
        bs = pd.to_numeric(out[body_col], errors="coerce")
        ss = pd.to_numeric(out[subj_col], errors="coerce")
        hb = bs.fillna(0) > thr
        hs = ss.fillna(0) > thr
        out["has_body_semantic"] = hb
        out["has_subject_semantic"] = hs
        both = hb & hs
        body_only = hb & ~hs
        subj_only = ~hb & hs
        cat = np.full(len(out), "neither", dtype=object)
        cat[both.to_numpy()] = "both"
        cat[body_only.to_numpy()] = "body_only"
        cat[subj_only.to_numpy()] = "subject_only"
        out["semantic_channel_category"] = cat
    else:
        out["semantic_channel_category"] = "unavailable"

    count_cols = [flag for _, _, flag in overlap_specs if flag in out.columns]
    count_cols += ["has_temporal_support"]
    if "has_local_support" in out.columns:
        count_cols.append("has_local_support")
    if body_col and subj_col:
        count_cols += ["has_body_semantic", "has_subject_semantic"]
    count_cols = list(dict.fromkeys(count_cols))
    out["n_evidence_channels"] = out[count_cols].fillna(False).astype(bool).sum(axis=1).astype(int)

    if "centroid_cosine" in out.columns:
        out["semantic_aggregate"] = pd.to_numeric(out["centroid_cosine"], errors="coerce")
    if "infra_score" in out.columns:
        out["infra_aggregate"] = pd.to_numeric(out["infra_score"], errors="coerce")

    return out


# ---------------------------------------------------------------------------
# Cross-edge inspection (shared artifact previews from shard node sets)
# ---------------------------------------------------------------------------

def infer_cross_edge_channel_specs(edge_df: pd.DataFrame) -> tuple[dict[str, Any], ...]:
    specs: list[dict[str, Any]] = []
    for base, count_col, has_col in infer_shared_overlap_specs(edge_df):
        specs.append(
            {
                "key": base,
                "title": f"{_channel_title(base)} overlap",
                "has_col": has_col,
                "nodes_col": f"{base}_set",
                "count_col": count_col,
                "idf_col": f"shared_{base}_idf_sum",
            }
        )
    return tuple(specs)


def parse_shard_set_cell(raw: Any) -> set[str]:
    """Parse list/set literals stored in ``semantic_shard_step2_nodes.csv`` set columns."""
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return set()
    s = str(raw).strip()
    if not s or s.lower() in ("nan", "none", "[]"):
        return set()
    try:
        v = ast.literal_eval(s)
        if isinstance(v, (list, tuple, set)):
            return {str(x).strip() for x in v if str(x).strip()}
    except (ValueError, SyntaxError, TypeError):
        pass
    try:
        v = json.loads(s)
        if isinstance(v, list):
            return {str(x).strip() for x in v if str(x).strip()}
    except (ValueError, TypeError, json.JSONDecodeError):
        pass
    return set()


def build_shard_set_index(nodes_df: pd.DataFrame, col: str) -> dict[str, set[str]]:
    if col not in nodes_df.columns or "shard_id" not in nodes_df.columns:
        return {}
    out: dict[str, set[str]] = {}
    for _, r in nodes_df[["shard_id", col]].iterrows():
        out[str(r["shard_id"])] = parse_shard_set_cell(r[col])
    return out


def format_shared_values_preview(intersection: set[str], *, max_values: int = 5) -> str:
    if not intersection:
        return ""
    ordered = sorted(intersection)
    head = ordered[: int(max_values)]
    extra = len(ordered) - len(head)
    body = "; ".join(head)
    if extra > 0:
        body = f"{body} (+{extra} more)"
    return body


def parse_shard_member_external_ids(raw: Any) -> list[str]:
    """Parse ``member_external_ids`` list literal from ``semantic_shard_step2_nodes.csv``."""
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return []
    s = str(raw).strip()
    if not s or s.lower() in ("nan", "none", "[]"):
        return []
    try:
        v = ast.literal_eval(s)
        if isinstance(v, (list, tuple, set)):
            return [str(x).strip() for x in v if str(x).strip()]
    except (ValueError, SyntaxError, TypeError):
        return []
    return []


def build_shard_member_ids_index(
    nodes_df: pd.DataFrame,
    *,
    col: str = "member_external_ids",
) -> dict[str, list[str]]:
    if col not in nodes_df.columns or "shard_id" not in nodes_df.columns:
        return {}
    out: dict[str, list[str]] = {}
    for _, r in nodes_df[["shard_id", col]].iterrows():
        out[str(r["shard_id"])] = parse_shard_member_external_ids(r[col])
    return out


def format_member_ids_preview(member_ids: list[str], *, max_ids: int | None = None) -> str:
    """
    One ``external_id`` per line (easier to copy than a single long line).

    If ``max_ids`` is ``None`` or <= 0, include every id. Otherwise cap with a ``(+N more)`` tail.
    """
    if not member_ids:
        return ""
    if max_ids is None or int(max_ids) <= 0:
        return "\n".join(member_ids)
    head = member_ids[: int(max_ids)]
    extra = len(member_ids) - len(head)
    body = "\n".join(head)
    if extra > 0:
        body = f"{body}\n... (+{extra} more)"
    return body


def cross_edge_channel_presence_summary(
    edge_cross: pd.DataFrame,
    *,
    specs: tuple[dict[str, Any], ...] | None = None,
) -> pd.DataFrame:
    """Per-channel counts among cross-labeled edges."""
    n = int(len(edge_cross))
    rows: list[dict[str, Any]] = []
    specs = infer_cross_edge_channel_specs(edge_cross) if specs is None else specs
    for spec in specs:
        hc = spec["has_col"]
        if hc not in edge_cross.columns:
            continue
        c = int(edge_cross[hc].astype(bool).sum())
        rows.append(
            {
                "channel": spec["title"],
                "n_cross_with_channel": c,
                "frac_of_cross": float(c / n) if n else 0.0,
            }
        )
    return pd.DataFrame(rows)


def sample_cross_edges_for_channel_inspection(
    edge_cross: pd.DataFrame,
    shard_summary: pd.DataFrame,
    nodes_df: pd.DataFrame,
    spec: dict[str, Any],
    *,
    top_n: int = 12,
    max_values_preview: int = 5,
    max_member_ids_preview: int | None = None,
    member_ids_col: str = "member_external_ids",
) -> tuple[pd.DataFrame, str | None]:
    """
    Return a compact table for manual inspection and an optional note if shared values
    cannot be resolved from node artifacts.

    When ``member_ids_col`` is present on ``nodes_df``, adds ``src_member_ids_preview`` /
    ``dst_member_ids_preview``: one ``external_id`` per line. If ``max_member_ids_preview``
    is ``None`` (default), every member id is included; set a positive int to cap.
    """
    hc = spec["has_col"]
    if hc not in edge_cross.columns:
        return pd.DataFrame(), f"Column {hc!r} missing on edge table."

    sub = edge_cross[edge_cross[hc].astype(bool)].copy()
    if sub.empty:
        return pd.DataFrame(), None

    nodes_col = spec["nodes_col"]
    set_idx: dict[str, set[str]] | None = None
    note: str | None = None
    if nodes_col in nodes_df.columns:
        set_idx = build_shard_set_index(nodes_df, nodes_col)
    else:
        note = (
            f"Column {nodes_col!r} not found on shard nodes table — showing overlap counts only "
            "(no raw shared token list)."
        )

    member_idx = build_shard_member_ids_index(nodes_df, col=member_ids_col)
    if not member_idx:
        extra_m = (
            f"Column {member_ids_col!r} missing or empty on shard nodes — email external_id previews omitted."
        )
        note = f"{note} {extra_m}" if note else extra_m

    count_col = spec.get("count_col")
    idf_col = spec.get("idf_col")

    def _shared_set(row: pd.Series) -> set[str]:
        if set_idx is None:
            return set()
        a, b = str(row["shard_a"]), str(row["shard_b"])
        return set_idx.get(a, set()) & set_idx.get(b, set())

    if set_idx is not None:
        sub["_shared_inter"] = sub.apply(_shared_set, axis=1)
        sub["shared_values_preview"] = sub["_shared_inter"].apply(
            lambda s: format_shared_values_preview(s, max_values=max_values_preview)
        )
        sub = sub.drop(columns=["_shared_inter"])
    else:
        sub["shared_values_preview"] = ""

    ss = shard_summary.set_index("shard_id")
    camp_cols = {"src": "shard_a_dominant_campaign", "dst": "shard_b_dominant_campaign"}
    for side, sk in [("src", "shard_a"), ("dst", "shard_b")]:
        sub[f"{side}_shard"] = sub[sk].astype(str)
        cc = camp_cols[side]
        if cc in sub.columns:
            sub[f"{side}_dominant_campaign"] = sub[cc]
        else:
            sub[f"{side}_dominant_campaign"] = sub[sk].map(ss["dominant_campaign"])
        sub[f"{side}_dominant_fraction"] = sub[sk].map(ss["dominant_fraction"])

    nm_a = sub["shard_a"].map(ss["n_members_total"])
    nm_b = sub["shard_b"].map(ss["n_members_total"])
    if "size" in nodes_df.columns:
        ns = nodes_df.set_index("shard_id")["size"]
        sub["src_size"] = sub["shard_a"].map(ns).fillna(nm_a)
        sub["dst_size"] = sub["shard_b"].map(ns).fillna(nm_b)
    else:
        sub["src_size"] = nm_a
        sub["dst_size"] = nm_b

    if "semantic_aggregate" in sub.columns:
        sem = pd.to_numeric(sub["semantic_aggregate"], errors="coerce")
    elif "centroid_cosine" in sub.columns:
        sem = pd.to_numeric(sub["centroid_cosine"], errors="coerce")
    else:
        sem = pd.Series(0.0, index=sub.index)
    sub["_sort_sem"] = sem
    sub["_sort_cnt"] = (
        pd.to_numeric(sub[count_col], errors="coerce").fillna(0) if count_col and count_col in sub.columns else 0.0
    )
    sub["_sort_idf"] = (
        pd.to_numeric(sub[idf_col], errors="coerce").fillna(0) if idf_col and idf_col in sub.columns else 0.0
    )
    sub = sub.sort_values(
        by=["_sort_sem", "_sort_cnt", "_sort_idf"],
        ascending=[False, False, False],
    ).head(int(top_n))
    sub = sub.drop(columns=["_sort_sem", "_sort_cnt", "_sort_idf"], errors="ignore")

    if member_idx:
        sub["src_member_ids_preview"] = sub["shard_a"].map(
            lambda sid: format_member_ids_preview(member_idx.get(str(sid), []), max_ids=max_member_ids_preview)
        )
        sub["dst_member_ids_preview"] = sub["shard_b"].map(
            lambda sid: format_member_ids_preview(member_idx.get(str(sid), []), max_ids=max_member_ids_preview)
        )
    else:
        sub["src_member_ids_preview"] = ""
        sub["dst_member_ids_preview"] = ""

    disp_cols = [
        "src_shard",
        "dst_shard",
        "src_member_ids_preview",
        "dst_member_ids_preview",
        "src_dominant_campaign",
        "dst_dominant_campaign",
        "src_dominant_fraction",
        "dst_dominant_fraction",
        "src_size",
        "dst_size",
        "semantic_aggregate",
        "infra_aggregate",
    ]
    if count_col and count_col in sub.columns:
        disp_cols.append(count_col)
    if idf_col and idf_col in sub.columns:
        disp_cols.append(idf_col)
    disp_cols.append("shared_values_preview")
    disp_cols = [c for c in disp_cols if c in sub.columns]
    return sub[disp_cols].reset_index(drop=True), note


# ---------------------------------------------------------------------------
# Concrete overlap artifact diagnostics (shard-link × shared artifact)
# ---------------------------------------------------------------------------

def infer_artifact_overlap_specs(edge_df: pd.DataFrame, nodes_df: pd.DataFrame) -> tuple[dict[str, Any], ...]:
    specs: list[dict[str, Any]] = []
    for base, count_col, _ in infer_shared_overlap_specs(edge_df):
        nodes_col = f"{base}_set"
        if nodes_col not in nodes_df.columns:
            continue
        specs.append(
            {
                "key": base,
                "display": _channel_title(base),
                "nodes_col": nodes_col,
                "count_col": count_col,
                "idf_col": f"shared_{base}_idf_sum",
            }
        )
    return tuple(specs)


def build_artifact_shard_frequency_maps(
    nodes_df: pd.DataFrame,
    specs: tuple[dict[str, Any], ...] | None = None,
) -> dict[str, dict[str, int]]:
    """For each artifact type: value -> #shard nodes whose set contains that value."""
    if specs is None:
        inferred = [
            {
                "key": str(c)[:-4],
                "nodes_col": str(c),
            }
            for c in nodes_df.columns
            if str(c).endswith("_set")
        ]
        specs = tuple(inferred)
    out: dict[str, dict[str, int]] = {}
    for spec in specs:
        ncol = spec["nodes_col"]
        if ncol not in nodes_df.columns:
            continue
        idx = build_shard_set_index(nodes_df, ncol)
        c: Counter[str] = Counter()
        for st in idx.values():
            for v in st:
                c[str(v).strip()] += 1
        out[spec["key"]] = dict(c)
    return out


def expand_labeled_edges_to_artifact_incidence(
    edge_df: pd.DataFrame,
    nodes_df: pd.DataFrame,
    *,
    specs: tuple[dict[str, Any], ...] | None = None,
) -> pd.DataFrame:
    specs = infer_artifact_overlap_specs(edge_df, nodes_df) if specs is None else specs
    """
    One row per (shard link, concrete shared artifact). Intersection is taken from
    ``nodes_df`` set columns (parsed via ``parse_shard_set_cell``).
    """
    need = {"shard_a", "shard_b", "link_label"}
    missing = need - set(edge_df.columns)
    if missing:
        raise ValueError(f"edge_df missing columns: {sorted(missing)}")

    copy_cols = [
        c
        for c in (
            "shard_a",
            "shard_b",
            "link_label",
            "semantic_aggregate",
            "infra_aggregate",
            "edge_taxonomy",
        )
        if c in edge_df.columns
    ]
    rows: list[dict[str, Any]] = []
    for spec in specs:
        ncol = spec["nodes_col"]
        if ncol not in nodes_df.columns:
            continue
        idx = build_shard_set_index(nodes_df, ncol)
        if not idx:
            continue
        k = spec["key"]
        cc = spec.get("count_col")
        ic = spec.get("idf_col")
        for _, er in edge_df.iterrows():
            a, b = str(er["shard_a"]), str(er["shard_b"])
            inter = idx.get(a, set()) & idx.get(b, set())
            if not inter:
                continue
            base = {c: er[c] for c in copy_cols if c in er.index}
            ec = float(er[cc]) if cc and cc in er.index and pd.notna(er[cc]) else float("nan")
            ei = float(er[ic]) if ic and ic in er.index and pd.notna(er[ic]) else float("nan")
            for av in sorted(inter):
                r = {
                    **base,
                    "artifact_type": k,
                    "artifact_value": str(av).strip(),
                    "edge_shared_count_channel": ec,
                    "edge_shared_idf_sum_channel": ei,
                }
                rows.append(r)
    if not rows:
        return pd.DataFrame(
            columns=[
                *[
                    c
                    for c in (
                        "shard_a",
                        "shard_b",
                        "link_label",
                        "semantic_aggregate",
                        "infra_aggregate",
                        "edge_taxonomy",
                    )
                    if c in edge_df.columns
                ],
                "artifact_type",
                "artifact_value",
                "edge_shared_count_channel",
                "edge_shared_idf_sum_channel",
            ]
        )
    return pd.DataFrame(rows)


def summarize_artifact_link_induction(
    long_df: pd.DataFrame,
    *,
    shard_freq_maps: dict[str, dict[str, int]] | None = None,
) -> pd.DataFrame:
    """Per (artifact_type, artifact_value): link counts, rates, shard frequency."""
    if long_df.empty:
        return pd.DataFrame(
            columns=[
                "artifact_type",
                "artifact_value",
                "n_same",
                "n_cross",
                "n_ambiguous",
                "n_confident_links",
                "n_total_rows",
                "n_distinct_shard_links",
                "n_shards_in_links",
                "frac_cross_among_confident",
                "frac_same_among_confident",
                "mean_semantic",
                "n_shards_containing",
            ]
        )

    def _agg(sub: pd.DataFrame) -> pd.Series:
        n_same = int((sub["link_label"] == "same").sum())
        n_cross = int((sub["link_label"] == "cross").sum())
        n_amb = int((sub["link_label"] == "ambiguous").sum())
        n_conf = n_same + n_cross
        n_edge = int(sub[["shard_a", "shard_b"]].drop_duplicates().shape[0])
        sem = (
            pd.to_numeric(sub["semantic_aggregate"], errors="coerce")
            if "semantic_aggregate" in sub.columns
            else pd.Series(dtype=float)
        )
        mean_sem = float(sem.mean()) if len(sem) else float("nan")
        shards_touched: set[str] = set()
        if "shard_a" in sub.columns:
            shards_touched |= set(sub["shard_a"].astype(str))
        if "shard_b" in sub.columns:
            shards_touched |= set(sub["shard_b"].astype(str))
        return pd.Series(
            {
                "n_same": n_same,
                "n_cross": n_cross,
                "n_ambiguous": n_amb,
                "n_confident_links": n_conf,
                "n_total_rows": int(len(sub)),
                "n_distinct_shard_links": n_edge,
                "n_shards_in_links": len(shards_touched),
                "frac_cross_among_confident": float(n_cross / n_conf) if n_conf else float("nan"),
                "frac_same_among_confident": float(n_same / n_conf) if n_conf else float("nan"),
                "mean_semantic": mean_sem,
            }
        )

    out = long_df.groupby(["artifact_type", "artifact_value"], sort=False).apply(_agg).reset_index()
    if shard_freq_maps:
        def _freq(r: pd.Series) -> int:
            m = shard_freq_maps.get(str(r["artifact_type"]), {})
            return int(m.get(str(r["artifact_value"]), 0))

        out["n_shards_containing"] = out.apply(_freq, axis=1)
    else:
        out["n_shards_containing"] = np.nan
    return out.sort_values(["n_cross", "frac_cross_among_confident"], ascending=[False, False])


def plot_label_short(s: str, max_len: int = 42) -> str:
    s = str(s)
    return s if len(s) <= max_len else s[: max_len - 3] + "..."


def artifact_filtering_candidates(
    summary_tbl: pd.DataFrame,
    *,
    min_conf_links: int = 5,
    high_shard_freq: int | None = None,
) -> pd.DataFrame:
    """
    Heuristic shortlist: broad shards + cross-heavy confident links (diagnostic only).
    """
    if summary_tbl.empty:
        return pd.DataFrame()
    s = summary_tbl.copy()
    if high_shard_freq is None and "n_shards_containing" in s.columns:
        vc = s["n_shards_containing"].dropna()
        high_shard_freq = int(max(10.0, float(vc.quantile(0.85)))) if len(vc) else 50
    elif high_shard_freq is None:
        high_shard_freq = 50
    s = s[s["n_confident_links"] >= min_conf_links].copy()
    if s.empty:
        return s
    sc = s["n_shards_containing"].fillna(0)
    fc = s["frac_cross_among_confident"].fillna(0)
    thr = int(high_shard_freq)
    s["_heuristic_score"] = (sc / max(1, thr)).clip(0, 3) + fc * 2.0
    s["high_shard_freq_heuristic"] = sc >= thr
    s = s.sort_values("_heuristic_score", ascending=False)
    return s.drop(columns=["_heuristic_score"]).head(25).reset_index(drop=True)
