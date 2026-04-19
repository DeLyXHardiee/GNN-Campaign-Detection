"""
Residual failure diagnostics for semantic shard graph baselines (GT evaluation only).

Used by ``semantic_shard_oracle_headroom_analysis.ipynb`` final section.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.utils.semantic_shard_oracle_headroom_helpers import parse_shard_set_cell


def _edge_key(a: str, b: str) -> tuple[str, str]:
    a, b = str(a), str(b)
    return (a, b) if a < b else (b, a)


class _UnionFind:
    def __init__(self, nodes: list[str]) -> None:
        self._p = {str(x): str(x) for x in nodes}

    def find(self, x: str) -> str:
        x = str(x)
        while self._p[x] != x:
            self._p[x] = self._p[self._p[x]]
            x = self._p[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self._p[rb] = ra

    def component_map(self) -> dict[str, str]:
        return {k: self.find(k) for k in self._p}


def _components_for_subgraph(nodes: list[str], edge_keys: set[tuple[str, str]]) -> dict[str, int]:
    uf = _UnionFind(nodes)
    for u, v in edge_keys:
        if u in uf._p and v in uf._p:
            uf.union(u, v)
    roots = uf.component_map()
    reps: dict[str, int] = {}
    cid = 0
    out: dict[str, int] = {}
    for n in nodes:
        r = roots[str(n)]
        if r not in reps:
            reps[r] = cid
            cid += 1
        out[str(n)] = reps[r]
    return out


def _shard_size_map(nodes_df: pd.DataFrame) -> dict[str, int]:
    if nodes_df is None or nodes_df.empty or "shard_id" not in nodes_df.columns:
        return {}
    col = "size" if "size" in nodes_df.columns else None
    if col is None:
        return {}
    return dict(zip(nodes_df["shard_id"].astype(str), nodes_df[col].astype(int), strict=False))


def _dominant_campaign(shard_summary: pd.DataFrame, sid: str) -> Any | None:
    m = shard_summary[shard_summary["shard_id"].astype(str) == str(sid)]
    if m.empty:
        return None
    c = m.iloc[0].get("dominant_campaign")
    if pd.isna(c) or c is None:
        return None
    if int(m.iloc[0].get("n_labeled", 0) or 0) < 1:
        return None
    return c


def label_edge_row(r: pd.Series, shard_summary: pd.DataFrame) -> str:
    ca = _dominant_campaign(shard_summary, str(r["shard_a"]))
    cb = _dominant_campaign(shard_summary, str(r["shard_b"]))
    if ca is None or cb is None:
        return "ambiguous"
    return "same" if ca == cb else "cross"


def load_step2_config(step2_dir: Path | str) -> dict[str, Any]:
    p = Path(step2_dir) / "semantic_shard_step2_graph_summary.json"
    if not p.is_file():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def load_centroid_matrix(
    step2_dir: Path | str,
    nodes_df: pd.DataFrame,
) -> tuple[np.ndarray | None, dict[str, int]]:
    """Return (matrix n_shards x d, shard_id -> row_index) or (None, {})."""
    p = Path(step2_dir) / "semantic_shard_step2_centroids.npy"
    if not p.is_file():
        return None, {}
    mat = np.load(p, mmap_mode="r")
    if mat.ndim != 2 or len(nodes_df) == 0:
        return np.asarray(mat, dtype=np.float64), {}
    ids = nodes_df["shard_id"].astype(str).tolist()
    if mat.shape[0] != len(ids):
        return np.asarray(mat, dtype=np.float64), {}
    idx = {s: i for i, s in enumerate(ids)}
    return np.asarray(mat, dtype=np.float64), idx


def active_edge_keys(edges_df: pd.DataFrame, *, min_edge_weight: float, weight_col: str = "edge_weight") -> set[tuple[str, str]]:
    if edges_df.empty or weight_col not in edges_df.columns:
        return set()
    out: set[tuple[str, str]] = set()
    for _, r in edges_df.iterrows():
        try:
            w = float(r[weight_col])
        except (TypeError, ValueError):
            continue
        if w >= float(min_edge_weight):
            out.add(_edge_key(str(r["shard_a"]), str(r["shard_b"])))
    return out


# --- 1. Campaign fracture inventory ---


def compute_campaign_fracture_table(
    assignments_df: pd.DataFrame,
    gt_label_map: dict[str, Any],
    edges_df: pd.DataFrame,
    nodes_df: pd.DataFrame,
    *,
    min_edge_weight: float = 0.0,
    weight_col: str = "edge_weight",
) -> pd.DataFrame:
    gt = {str(k): v for k, v in gt_label_map.items()}
    adf = assignments_df.copy()
    adf["external_id"] = adf["external_id"].astype(str)
    adf["shard_id"] = adf["shard_id"].astype(str)
    adf = adf[adf["external_id"].isin(gt)].copy()
    adf["campaign_id"] = adf["external_id"].map(gt)

    size_map = _shard_size_map(nodes_df)

    active_keys = active_edge_keys(edges_df, min_edge_weight=min_edge_weight, weight_col=weight_col)
    rows: list[dict[str, Any]] = []
    for cid, g in adf.groupby("campaign_id", sort=False):
        eids = g["external_id"].tolist()
        shards = sorted(set(g["shard_id"].astype(str)))
        n_gt = len(eids)
        n_sh = len(shards)
        noise_shards = [s for s in shards if int(size_map.get(s, 999999)) <= 1]
        n_noise_sh = len(noise_shards)
        emails_in_noise = int(g[g["shard_id"].isin(noise_shards)].shape[0])
        frac_noise_em = float(emails_in_noise / n_gt) if n_gt else float("nan")

        if n_sh <= 1:
            n_comp = 1
            largest = n_sh
            frac_largest = 1.0
        else:
            sub_edges = {k for k in active_keys if k[0] in shards and k[1] in shards}
            comp = _components_for_subgraph(shards, sub_edges)
            vc = Counter(comp.values())
            n_comp = len(vc)
            largest = max(vc.values()) if vc else 0
            frac_largest = float(largest / n_sh) if n_sh else float("nan")

        rows.append(
            {
                "campaign_id": cid,
                "n_gt_emails": n_gt,
                "n_shards_in_campaign": n_sh,
                "n_noise_shards_in_campaign": n_noise_sh,
                "n_graph_components_in_campaign": n_comp,
                "largest_component_size": int(largest),
                "largest_component_fraction": frac_largest,
                "n_emails_in_noise_shards": emails_in_noise,
                "fraction_emails_in_noise_shards": frac_noise_em,
            }
        )
    return pd.DataFrame(rows)


def campaign_fracture_summary_stats(fr: pd.DataFrame) -> pd.DataFrame:
    if fr.empty:
        return pd.DataFrame(
            [
                {
                    "n_gt_campaigns": 0,
                    "n_fractured_campaigns": 0,
                    "frac_fractured": float("nan"),
                    "n_campaigns_with_any_noise_shard": 0,
                    "n_fractured_campaigns_with_noise_shard": 0,
                    "mean_largest_component_fraction_fractured": float("nan"),
                    "median_largest_component_fraction_fractured": float("nan"),
                }
            ]
        )
    fractured = fr[fr["n_graph_components_in_campaign"] > 1]
    noise_any = fr[fr["n_noise_shards_in_campaign"] > 0]
    fc_noise = fractured[fractured["n_noise_shards_in_campaign"] > 0]
    return pd.DataFrame(
        [
            {
                "n_gt_campaigns": int(len(fr)),
                "n_fractured_campaigns": int(len(fractured)),
                "frac_fractured": float(len(fractured) / len(fr)),
                "n_campaigns_with_any_noise_shard": int(len(noise_any)),
                "n_fractured_campaigns_with_noise_shard": int(len(fc_noise)),
                "mean_largest_component_fraction_fractured": float(
                    fractured["largest_component_fraction"].mean()
                )
                if len(fractured)
                else float("nan"),
                "median_largest_component_fraction_fractured": float(
                    fractured["largest_component_fraction"].median()
                )
                if len(fractured)
                else float("nan"),
            }
        ]
    )


def plot_campaign_component_bar(fr: pd.DataFrame, *, title: str, ax: plt.Axes | None = None) -> plt.Figure:
    vc = fr["n_graph_components_in_campaign"].value_counts().sort_index()
    max_show = 12
    if len(vc) > max_show:
        head = vc.iloc[: max_show - 1]
        tail_sum = int(vc.iloc[max_show - 1 :].sum())
        cut = int(vc.index[max_show - 1])
        vc = pd.concat([head, pd.Series([tail_sum], index=[f">={cut}"])])
    fig = plt.figure(figsize=(7, 3.8)) if ax is None else ax.figure
    if ax is None:
        ax = fig.add_subplot(111)
    x = np.arange(len(vc))
    ax.bar(x, vc.values, color="#4c72b0")
    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in vc.index], rotation=0)
    ax.set_xlabel("Number of connected components (among campaign shards)")
    ax.set_ylabel("Number of GT campaigns")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_largest_component_hist(fractured: pd.DataFrame, *, title: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    vals = fractured["largest_component_fraction"].dropna().astype(float)
    if len(vals):
        ax.hist(vals, bins=min(25, max(5, len(vals) // 3)), color="#55a868", edgecolor="white")
    ax.set_xlabel("Largest component size (fraction of campaign shards)")
    ax.set_ylabel("Number of fractured campaigns")
    ax.set_title(title)
    fig.tight_layout()
    return fig


# --- 2. Noise contribution ---


def fractured_campaign_noise_summary(
    assignments_df: pd.DataFrame,
    gt_label_map: dict[str, Any],
    fracture_df: pd.DataFrame,
    nodes_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    fractured_ids = set(fracture_df.loc[fracture_df["n_graph_components_in_campaign"] > 1, "campaign_id"])
    gt = {str(k): v for k, v in gt_label_map.items()}
    adf = assignments_df.copy()
    adf["external_id"] = adf["external_id"].astype(str)
    adf["shard_id"] = adf["shard_id"].astype(str)
    adf = adf[adf["external_id"].isin(gt)].copy()
    adf["campaign_id"] = adf["external_id"].map(gt)
    adf = adf[adf["campaign_id"].isin(fractured_ids)].copy()

    size_map = _shard_size_map(nodes_df)
    adf["is_noise_shard"] = adf["shard_id"].map(lambda s: int(size_map.get(str(s), 999)) <= 1)

    n_fr_c = len(fractured_ids)
    n_fr_noise_c = int(
        fracture_df[
            (fracture_df["n_graph_components_in_campaign"] > 1)
            & (fracture_df["n_noise_shards_in_campaign"] > 0)
        ].shape[0]
    )
    n_em = int(len(adf))
    n_em_n = int(adf["is_noise_shard"].sum())
    summary = pd.DataFrame(
        [
            {
                "n_fractured_campaigns": n_fr_c,
                "n_fractured_campaigns_with_noise": n_fr_noise_c,
                "frac_fractured_campaigns_with_noise": float(n_fr_noise_c / n_fr_c) if n_fr_c else float("nan"),
                "n_fractured_emails": n_em,
                "n_fractured_emails_in_noise": n_em_n,
                "frac_fractured_emails_in_noise": float(n_em_n / n_em) if n_em else float("nan"),
            }
        ]
    )

    # Per-campaign bucket for stacked bar
    buckets = []
    for cid, g in adf.groupby("campaign_id"):
        off_noise = bool((g["is_noise_shard"]).any())
        all_noise = bool(g["is_noise_shard"].all()) if len(g) else False
        if not off_noise:
            buckets.append("no_noise_contribution")
        elif all_noise:
            buckets.append("all_emails_noise_shards")
        else:
            buckets.append("some_noise_contribution")
    bc = Counter(buckets)
    per = pd.DataFrame([{"bucket": k, "n_campaigns": v} for k, v in bc.items()])
    return summary, per


# --- 3. Missing-link opportunity ---


def classify_disconnected_same_campaign_pairs(
    fracture_df: pd.DataFrame,
    assignments_df: pd.DataFrame,
    gt_label_map: dict[str, Any],
    edges_df: pd.DataFrame,
    nodes_df: pd.DataFrame,
    *,
    active_min_edge_weight: float,
    near_miss_cos: float = 0.85,
    weight_col: str = "edge_weight",
    centroid_mat: np.ndarray | None = None,
    shard_to_idx: dict[str, int] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (pair_table, category_summary)."""
    gt = {str(k): v for k, v in gt_label_map.items()}
    adf = assignments_df.copy()
    adf["external_id"] = adf["external_id"].astype(str)
    adf["shard_id"] = adf["shard_id"].astype(str)
    adf = adf[adf["external_id"].isin(gt)].copy()
    adf["campaign_id"] = adf["external_id"].map(gt)

    active = active_edge_keys(edges_df, min_edge_weight=active_min_edge_weight, weight_col=weight_col)
    # full table lookup weight
    wmap: dict[tuple[str, str], float] = {}
    coss: dict[tuple[str, str], float] = {}
    if not edges_df.empty:
        for _, r in edges_df.iterrows():
            k = _edge_key(r["shard_a"], r["shard_b"])
            wmap[k] = float(r.get(weight_col, 0.0) or 0.0)
            if "centroid_cosine" in edges_df.columns:
                coss[k] = float(r["centroid_cosine"])

    size_map = _shard_size_map(nodes_df)
    idx = shard_to_idx or {}
    mat = centroid_mat

    def pair_cos(a: str, b: str) -> float:
        k = _edge_key(a, b)
        if k in coss:
            return coss[k]
        if mat is not None and a in idx and b in idx:
            va = mat[idx[a]]
            vb = mat[idx[b]]
            return float(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-12))
        return float("nan")

    def infra_overlap_count(a: str, b: str) -> int:
        ra = nodes_df[nodes_df["shard_id"].astype(str) == str(a)]
        rb = nodes_df[nodes_df["shard_id"].astype(str) == str(b)]
        if ra.empty or rb.empty:
            return 0
        ra, rb = ra.iloc[0], rb.iloc[0]
        total = 0
        overlap_cols = [str(c) for c in nodes_df.columns if str(c).endswith("_set")]
        for ch in overlap_cols:
            if ch not in nodes_df.columns:
                continue
            sa, sb = ra.get(ch), rb.get(ch)
            sa = parse_shard_set_cell(sa) if not isinstance(sa, set) else sa
            sb = parse_shard_set_cell(sb) if not isinstance(sb, set) else sb
            if sa & sb:
                total += 1
        return total

    pair_rows: list[dict[str, Any]] = []
    fractured = fracture_df[fracture_df["n_graph_components_in_campaign"] > 1]
    pair_seen: set[tuple[Any, str, str]] = set()

    for _, cr in fractured.iterrows():
        cid = cr["campaign_id"]
        g = adf[adf["campaign_id"] == cid]
        shards = sorted(set(g["shard_id"].astype(str)))
        if len(shards) < 2:
            continue
        sub_edges = {k for k in active if k[0] in shards and k[1] in shards}
        comp = _components_for_subgraph(shards, sub_edges)
        by_c: dict[int, list[str]] = {}
        for s in shards:
            by_c.setdefault(comp[s], []).append(s)
        comp_ids = list(by_c.keys())
        if len(comp_ids) < 2:
            continue
        for i in range(len(comp_ids)):
            for j in range(i + 1, len(comp_ids)):
                for sa in by_c[comp_ids[i]]:
                    for sb in by_c[comp_ids[j]]:
                        a, b = sa, sb
                        k = _edge_key(a, b)
                        ded = (cid, k[0], k[1])
                        if ded in pair_seen:
                            continue
                        pair_seen.add(ded)

                        w = wmap.get(k)
                        cat = "no_edge_no_direct_support"
                        if w is not None and w < float(active_min_edge_weight):
                            cat = "existing_edge_below_operating_threshold"
                        elif k not in wmap:
                            co = pair_cos(a, b)
                            if np.isfinite(co) and co >= float(near_miss_cos):
                                cat = "no_edge_but_high_semantic_candidate"
                            elif infra_overlap_count(a, b) > 0:
                                cat = "no_edge_but_some_direct_infra_support"
                            else:
                                cat = "no_edge_no_direct_support"
                        else:
                            # Candidate row exists with weight >= active — should not be disconnected
                            cat = "pair_anomaly_high_weight_but_disconnected"

                        pair_rows.append(
                            {
                                "campaign_id": cid,
                                "shard_a": a,
                                "shard_b": b,
                                "category": cat,
                                "edge_weight_if_any": w,
                                "centroid_cosine": pair_cos(a, b),
                            }
                        )

    pt = pd.DataFrame(pair_rows)
    if pt.empty:
        summ = pd.DataFrame(columns=["category", "n_same_campaign_disconnected_pairs", "fraction"])
    else:
        vc = pt["category"].value_counts()
        summ = (
            pd.DataFrame({"category": vc.index.astype(str), "n_same_campaign_disconnected_pairs": vc.values})
            .assign(fraction=lambda d: d["n_same_campaign_disconnected_pairs"] / d["n_same_campaign_disconnected_pairs"].sum())
        )
    return pt, summ


def plot_horizontal_category_bars(summ: pd.DataFrame, *, title: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 4.2))
    d = summ.sort_values("n_same_campaign_disconnected_pairs", ascending=True)
    y = np.arange(len(d))
    ax.barh(y, d["n_same_campaign_disconnected_pairs"].values, color="#8172b2")
    ax.set_yticks(y)
    ax.set_yticklabels(d["category"].astype(str))
    ax.set_xlabel("Count of same-campaign disconnected shard pairs")
    ax.set_title(title)
    fig.tight_layout()
    return fig


# --- 4. Cosine threshold sweep (new edges) ---


def cosine_threshold_new_edge_stats(
    edges_df: pd.DataFrame,
    centroid_mat: np.ndarray,
    shard_to_idx: dict[str, int],
    shard_summary: pd.DataFrame,
    *,
    thresholds: list[float],
    baseline_active_keys: set[tuple[str, str]],
    fracture_df: pd.DataFrame | None = None,
    assignments_df: pd.DataFrame | None = None,
    gt_label_map: dict[str, Any] | None = None,
    active_min_edge_weight: float = 0.0,
    weight_col: str = "edge_weight",
) -> pd.DataFrame:
    """For each τ: pairs with cos>=τ not in baseline_active_keys; label by GT."""
    ids = sorted(shard_to_idx.keys(), key=lambda s: shard_to_idx[s])
    n = len(ids)
    if n == 0 or centroid_mat.shape[0] < n:
        return pd.DataFrame()

    X = np.asarray(
        [centroid_mat[shard_to_idx[s]] for s in ids],
        dtype=np.float64,
    )
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    Xn = X / norms
    sim = Xn @ Xn.T

    # Precompute dominant campaign lookup once (avoid per-pair DataFrame scans).
    dom_lookup: dict[str, Any | None] = {}
    if not shard_summary.empty and "shard_id" in shard_summary.columns:
        tmp = shard_summary.copy()
        tmp["shard_id"] = tmp["shard_id"].astype(str)
        for _, r in tmp.iterrows():
            sid = str(r["shard_id"])
            c = r.get("dominant_campaign")
            n_lab = int(r.get("n_labeled", 0) or 0)
            if c is None or pd.isna(c) or n_lab < 1:
                dom_lookup[sid] = None
            else:
                dom_lookup[sid] = c

    # Baseline component map per fractured campaign (for optional reconnect count).
    reconnect_maps: dict[Any, dict[str, int]] = {}
    shard_to_campaign: dict[str, Any] = {}
    if fracture_df is not None and assignments_df is not None and gt_label_map is not None:
        gt = {str(k): v for k, v in gt_label_map.items()}
        adf = assignments_df.copy()
        adf["external_id"] = adf["external_id"].astype(str)
        adf["shard_id"] = adf["shard_id"].astype(str)
        adf = adf[adf["external_id"].isin(gt)].copy()
        adf["campaign_id"] = adf["external_id"].map(gt)
        # Stable map: shard -> campaign when labeled emails in shard are campaign-consistent.
        per_shard_campaigns = adf.groupby("shard_id")["campaign_id"].nunique()
        valid_shards = set(per_shard_campaigns[per_shard_campaigns == 1].index.astype(str))
        shard_to_campaign = (
            adf[adf["shard_id"].astype(str).isin(valid_shards)]
            .drop_duplicates(subset=["shard_id"])[["shard_id", "campaign_id"]]
            .set_index("shard_id")["campaign_id"]
            .to_dict()
        )
        active0 = active_edge_keys(
            edges_df, min_edge_weight=active_min_edge_weight, weight_col=weight_col
        )
        fr = fracture_df[fracture_df["n_graph_components_in_campaign"] > 1]
        for _, cr in fr.iterrows():
            cid = cr["campaign_id"]
            shards = sorted(set(adf.loc[adf["campaign_id"] == cid, "shard_id"].astype(str)))
            if len(shards) < 2:
                continue
            sub_e = {k for k in active0 if k[0] in shards and k[1] in shards}
            reconnect_maps[cid] = _components_for_subgraph(shards, sub_e)

    iu, ju = np.triu_indices(n, k=1)
    sim_u = sim[iu, ju]

    rows: list[dict[str, Any]] = []
    for tau in thresholds:
        new_same = new_cross = new_amb = 0
        bridge_campaigns: set[Any] = set()
        mask = sim_u >= float(tau)
        cand_i = iu[mask]
        cand_j = ju[mask]
        print(f"[cos-sweep] tau={tau:.3f} candidate_pairs={len(cand_i)}")
        for i, j in zip(cand_i.tolist(), cand_j.tolist(), strict=False):
            a = ids[i]
            b = ids[j]
            k = _edge_key(a, b)
            if k in baseline_active_keys:
                continue

            ca = dom_lookup.get(a)
            cb = dom_lookup.get(b)
            if ca is None or cb is None:
                lbl = "ambiguous"
            elif ca == cb:
                lbl = "same"
            else:
                lbl = "cross"

            if lbl == "same":
                new_same += 1
            elif lbl == "cross":
                new_cross += 1
            else:
                new_amb += 1

            if reconnect_maps and lbl == "same":
                cid_g = shard_to_campaign.get(a)
                if cid_g is None or cid_g != shard_to_campaign.get(b):
                    continue
                cmap = reconnect_maps.get(cid_g)
                if cmap is None:
                    continue
                if cmap.get(a) is not None and cmap.get(b) is not None and cmap[a] != cmap[b]:
                    bridge_campaigns.add(cid_g)

        tot_conf = new_same + new_cross
        rows.append(
            {
                "threshold": float(tau),
                "n_new_same": int(new_same),
                "n_new_cross": int(new_cross),
                "n_new_ambiguous": int(new_amb),
                "same_fraction_among_new_confident": float(new_same / tot_conf) if tot_conf else float("nan"),
                "cross_fraction_among_new_confident": float(new_cross / tot_conf) if tot_conf else float("nan"),
                "n_fractured_campaigns_with_new_same_bridge": int(len(bridge_campaigns)),
            }
        )
    return pd.DataFrame(rows)


# --- 5. Edge-weight retention sweep ---


def edge_weight_retention_table(
    edges_df: pd.DataFrame,
    shard_summary: pd.DataFrame,
    *,
    thresholds: list[float],
    weight_col: str = "edge_weight",
) -> pd.DataFrame:
    if edges_df.empty or weight_col not in edges_df.columns:
        return pd.DataFrame()
    labs = edges_df.assign(_lab=edges_df.apply(lambda r: label_edge_row(r, shard_summary), axis=1))
    n_same = int((labs["_lab"] == "same").sum())
    n_cross = int((labs["_lab"] == "cross").sum())
    rows: list[dict[str, Any]] = []
    for t in thresholds:
        m = labs[weight_col].astype(float) >= float(t)
        s_ret = int(((labs["_lab"] == "same") & m).sum())
        c_ret = int(((labs["_lab"] == "cross") & m).sum())
        rows.append(
            {
                "threshold": float(t),
                "frac_same_retained": float(s_ret / n_same) if n_same else float("nan"),
                "frac_cross_retained": float(c_ret / n_cross) if n_cross else float("nan"),
                "frac_same_removed": float(1.0 - s_ret / n_same) if n_same else float("nan"),
                "frac_cross_removed": float(1.0 - c_ret / n_cross) if n_cross else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def fractured_campaigns_at_edge_threshold(
    fracture_baseline: pd.DataFrame,
    assignments_df: pd.DataFrame,
    gt_label_map: dict[str, Any],
    edges_df: pd.DataFrame,
    *,
    thresholds: list[float],
    weight_col: str = "edge_weight",
) -> pd.DataFrame:
    """Optional: count fractured campaigns (n_comp>1) at each edge threshold."""
    gt = {str(k): v for k, v in gt_label_map.items()}
    adf = assignments_df.copy()
    adf["external_id"] = adf["external_id"].astype(str)
    adf["shard_id"] = adf["shard_id"].astype(str)
    adf = adf[adf["external_id"].isin(gt)].copy()
    adf["campaign_id"] = adf["external_id"].map(gt)

    rows: list[dict[str, Any]] = []
    for t in thresholds:
        keys = active_edge_keys(edges_df, min_edge_weight=t, weight_col=weight_col)
        n_frac = 0
        for cid, g in adf.groupby("campaign_id", sort=False):
            shards = sorted(set(g["shard_id"].astype(str)))
            if len(shards) < 2:
                continue
            sub = {k for k in keys if k[0] in shards and k[1] in shards}
            comp = _components_for_subgraph(shards, sub)
            if len(set(comp.values())) > 1:
                n_frac += 1
        rows.append({"threshold": float(t), "n_fractured_campaigns": int(n_frac)})
    return pd.DataFrame(rows)


# --- 6. Community attachment (off-main shards) ---


def off_main_shards_and_similarity(
    fracture_df: pd.DataFrame,
    assignments_df: pd.DataFrame,
    gt_label_map: dict[str, Any],
    edges_df: pd.DataFrame,
    nodes_df: pd.DataFrame,
    *,
    active_min_edge_weight: float,
    weight_col: str = "edge_weight",
    centroid_mat: np.ndarray | None = None,
    shard_to_idx: dict[str, int] | None = None,
) -> pd.DataFrame:
    gt = {str(k): v for k, v in gt_label_map.items()}
    adf = assignments_df.copy()
    adf["external_id"] = adf["external_id"].astype(str)
    adf["shard_id"] = adf["shard_id"].astype(str)
    adf = adf[adf["external_id"].isin(gt)].copy()
    adf["campaign_id"] = adf["external_id"].map(gt)

    active = active_edge_keys(edges_df, min_edge_weight=active_min_edge_weight, weight_col=weight_col)
    size_map = _shard_size_map(nodes_df)
    idx = shard_to_idx or {}

    def cos_ab(a: str, b: str) -> float:
        if centroid_mat is None or a not in idx or b not in idx:
            return float("nan")
        va = centroid_mat[idx[a]]
        vb = centroid_mat[idx[b]]
        return float(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-12))

    def n_edges_to_set(s: str, target_set: set[str]) -> int:
        c = 0
        for t in target_set:
            if s == t:
                continue
            k = _edge_key(s, t)
            if k in active:
                c += 1
        return c

    rows: list[dict[str, Any]] = []
    fractured = fracture_df[fracture_df["n_graph_components_in_campaign"] > 1]

    for _, cr in fractured.iterrows():
        cid = cr["campaign_id"]
        g = adf[adf["campaign_id"] == cid]
        shards = sorted(set(g["shard_id"].astype(str)))
        if len(shards) < 2:
            continue
        sub_edges = {k for k in active if k[0] in shards and k[1] in shards}
        comp = _components_for_subgraph(shards, sub_edges)
        vc = Counter(comp.values())
        main_c = max(vc, key=lambda k_: vc[k_])
        main_shards = {s for s, c in comp.items() if c == main_c}
        for s in shards:
            if s in main_shards:
                continue
            is_noise = int(size_map.get(s, 999)) <= 1
            cos_to_main = [cos_ab(s, t) for t in main_shards]
            max_cos = float(np.nanmax(cos_to_main)) if cos_to_main else float("nan")
            mean_cos = float(np.nanmean(cos_to_main)) if cos_to_main else float("nan")
            n_e = n_edges_to_set(s, main_shards)
            rows.append(
                {
                    "campaign_id": cid,
                    "shard_id": s,
                    "group": "noise_or_singleton" if is_noise else "normal_shard_off_component",
                    "max_cos_to_target_component": max_cos,
                    "mean_cos_to_target_component": mean_cos,
                    "n_active_edges_to_target_component": n_e,
                }
            )

    return pd.DataFrame(rows)


def community_attachment_summary(off_df: pd.DataFrame) -> pd.DataFrame:
    if off_df.empty:
        return pd.DataFrame(
            columns=[
                "group",
                "n_shards",
                "frac_sem_match_ge_0.90",
                "frac_sem_match_ge_0.85",
                "frac_with_any_direct_support",
                "frac_with_multi_edge_to_target",
            ]
        )

    def _frac(m: pd.Series) -> float:
        return float(m.mean()) if len(m) else float("nan")

    out_rows = []
    for grp, sub in off_df.groupby("group"):
        m90 = sub["max_cos_to_target_component"] >= 0.90
        m85 = sub["max_cos_to_target_component"] >= 0.85
        sup = sub["n_active_edges_to_target_component"] > 0
        mult = sub["n_active_edges_to_target_component"] > 1
        out_rows.append(
            {
                "group": str(grp),
                "n_shards": int(len(sub)),
                "frac_sem_match_ge_0.90": _frac(m90),
                "frac_sem_match_ge_0.85": _frac(m85),
                "frac_with_any_direct_support": _frac(sup),
                "frac_with_multi_edge_to_target": _frac(mult),
            }
        )
    return pd.DataFrame(out_rows)


def plot_max_cos_histogram(off_df: pd.DataFrame, *, title: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    for grp, color in (("noise_or_singleton", "#dd8452"), ("normal_shard_off_component", "#4c72b0")):
        sub = off_df[off_df["group"] == grp]
        if sub.empty:
            continue
        x = sub["max_cos_to_target_component"].dropna().astype(float)
        if len(x):
            ax.hist(x, bins=20, alpha=0.5, label=grp.replace("_", " "), color=color, density=False)
    ax.set_xlabel("Max centroid cosine to a shard in largest GT-connected component")
    ax.set_ylabel("Count of off-main shards")
    ax.legend()
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_attachment_stacked(off_df: pd.DataFrame, *, title: str) -> plt.Figure:
    def bucket(r: pd.Series) -> str:
        v = r["max_cos_to_target_component"]
        if not np.isfinite(v):
            return "no_clear_semantic_match"
        if v >= 0.90:
            return "strong_target_semantic_match"
        if v >= 0.72:
            return "weak_target_semantic_match"
        return "no_clear_semantic_match"

    if off_df.empty:
        fig, ax = plt.subplots(figsize=(7, 3.5))
        ax.set_title(title + " (no data)")
        return fig
    d = off_df.copy()
    d["bucket"] = d.apply(bucket, axis=1)
    ct = d.groupby(["group", "bucket"]).size().unstack(fill_value=0)
    ct = ct.reindex(columns=[c for c in ct.columns])
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    bottom = np.zeros(len(ct.index))
    colors = {"strong_target_semantic_match": "#55a868", "weak_target_semantic_match": "#ccb974", "no_clear_semantic_match": "#7f7f7f"}
    x = np.arange(len(ct.index))
    for col in ["strong_target_semantic_match", "weak_target_semantic_match", "no_clear_semantic_match"]:
        if col not in ct.columns:
            continue
        vals = ct[col].values.astype(float)
        ax.bar(x, vals, bottom=bottom, label=col.replace("_", " "), color=colors.get(col, "#333"))
        bottom = bottom + vals
    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in ct.index], rotation=15, ha="right")
    ax.set_ylabel("Number of off-main shards")
    ax.set_xlabel("Shard group")
    ax.legend()
    ax.set_title(title)
    fig.tight_layout()
    return fig
