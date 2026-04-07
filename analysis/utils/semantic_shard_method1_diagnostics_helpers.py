"""
Diagnostics helpers: Baseline vs Method 1 shard graphs and community detection.

Ground truth is used only for evaluation metrics and fragmentation tables, never for fitting.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from analysis.utils import completeness_fragmentation_helpers as cfh
from analysis.utils import semantic_shard_step3_helpers as s3
from analysis.utils.semantic_shard_edge_refinement_method1 import (
    Method1RefinementConfig,
    build_method1_edge_feature_frame,
    compute_method1_local_structure_features,
    compute_method1_view_scores,
    run_method1_edge_refinement_pipeline,
    save_method1_calibration_variant_bundle,
)


def check_community_detection_availability() -> dict[str, Any]:
    out: dict[str, Any] = {"louvain": False, "leiden": False, "notes": []}
    try:
        import networkx as nx  # noqa: F401

        out["louvain"] = True
    except Exception as e:  # pragma: no cover
        out["notes"].append(f"networkx: {e}")

    try:
        import igraph as ig  # noqa: F401
        import leidenalg as la  # noqa: F401

        out["leiden"] = True
    except Exception as e:
        out["notes"].append(f"igraph/leidenalg: {e}")
    return out


def _joint_best_row(sweep_df: pd.DataFrame) -> pd.Series:
    if sweep_df.empty:
        raise ValueError("empty sweep_df")
    return (
        sweep_df.sort_values(
            ["completeness", "v_measure", "homogeneity"],
            ascending=[False, False, False],
        )
        .iloc[0]
    )


def run_condition_sweep(
    *,
    condition_name: str,
    assignments_df: pd.DataFrame,
    shard_ids: list[str],
    edges_df: pd.DataFrame,
    gt_label_map: dict[str, Any],
    method: str,
    weight_col: str,
    resolution_values: list[float],
    min_edge_weight_values: list[float],
    seed: int,
) -> dict[str, Any]:
    sweep_df, email_by_key, info_by_key = s3.run_community_sweep(
        assignments_df=assignments_df,
        shard_ids=shard_ids,
        edges_df=edges_df,
        gt_label_map=gt_label_map,
        method=method,
        resolution_values=resolution_values,
        min_edge_weight_values=min_edge_weight_values,
        weight_col=weight_col,
        seed=seed,
    )
    best = _joint_best_row(sweep_df)
    key = str(best["setting_key"])
    shard_to_comm, info = s3.run_weighted_community_detection(
        shard_ids=shard_ids,
        edges_df=edges_df,
        method=str(best["method_requested"]),
        resolution=float(best["resolution"]),
        min_edge_weight=float(best["min_edge_weight"]),
        weight_col=str(best["weight_col"]),
        seed=seed,
    )
    email_pred_best = s3.map_shards_to_email_predictions(assignments_df, shard_to_comm)
    summary = {
        "condition": condition_name,
        "method_requested": str(best["method_requested"]),
        "method_used": str(best["method_used"]),
        "weight_col": str(best["weight_col"]),
        "best_setting_key": key,
        "best_resolution": float(best["resolution"]),
        "best_min_edge_weight": float(best["min_edge_weight"]),
        "homogeneity_at_best": float(best["homogeneity"]),
        "completeness_at_best": float(best["completeness"]),
        "v_measure_at_best": float(best["v_measure"]),
        "n_eval_at_best": float(best["n_eval"]),
        "coverage_gt_at_best": float(best["coverage_gt"]),
        "n_communities_at_best": float(best["n_communities"]),
        "max_homogeneity_sweep": float(sweep_df["homogeneity"].max()),
        "max_completeness_sweep": float(sweep_df["completeness"].max()),
        "max_v_measure_sweep": float(sweep_df["v_measure"].max()),
        "n_sweep_rows": int(len(sweep_df)),
    }
    return {
        "summary": summary,
        "sweep_df": sweep_df,
        "email_pred_best": email_pred_best,
        "shard_to_comm_best": shard_to_comm,
        "info_best": info,
        "email_preds_by_key": email_by_key,
    }


def graph_stats_at_thresholds(
    shard_ids: list[str],
    edges_df: pd.DataFrame,
    *,
    weight_col: str,
    min_edge_weight_values: list[float],
    label: str,
) -> pd.DataFrame:
    """Per-threshold connectivity and degree stats (undirected, edge-list)."""
    import networkx as nx

    sid_set = set(shard_ids)
    rows: list[dict[str, Any]] = []
    wcol = str(weight_col)
    if edges_df.empty or wcol not in edges_df.columns:
        for thr in min_edge_weight_values:
            rows.append(
                {
                    "graph_label": label,
                    "weight_col": wcol,
                    "min_edge_weight_threshold": float(thr),
                    "n_nodes": int(len(shard_ids)),
                    "n_edges_ge_threshold": 0,
                    "frac_edges_ge_threshold": 0.0,
                    "n_components": int(len(shard_ids)),
                    "giant_component_size": 1,
                    "mean_weighted_degree": 0.0,
                    "mean_unweighted_degree": 0.0,
                    "median_unweighted_degree": 0.0,
                    "p90_unweighted_degree": 0.0,
                }
            )
        return pd.DataFrame(rows)

    wseries = pd.to_numeric(edges_df[wcol], errors="coerce").fillna(0.0)
    n_e = int(len(edges_df))

    for thr in min_edge_weight_values:
        g = nx.Graph()
        g.add_nodes_from(shard_ids)
        thr_f = float(thr)
        kept = 0
        for _, r in edges_df.iterrows():
            if float(r[wcol]) >= thr_f:
                a, b = str(r["shard_a"]), str(r["shard_b"])
                if a in sid_set and b in sid_set:
                    g.add_edge(a, b, weight=float(r[wcol]))
                    kept += 1
        comps = list(nx.connected_components(g))
        sizes = [len(c) for c in comps] if comps else [0]
        giant = int(max(sizes)) if sizes else 0
        deg_sum = 0.0
        for u in g.nodes():
            deg_sum += sum(float(d.get("weight", 1.0)) for _, _, d in g.edges(u, data=True))
        mean_wdeg = float(deg_sum / max(1, g.number_of_nodes()))
        udegs = np.array([g.degree(u) for u in g.nodes()], dtype=np.float64)
        rows.append(
            {
                "graph_label": label,
                "weight_col": wcol,
                "min_edge_weight_threshold": thr_f,
                "n_nodes": int(len(shard_ids)),
                "n_edges_ge_threshold": int(kept),
                "frac_edges_ge_threshold": float(kept / max(1, n_e)),
                "n_components": int(nx.number_connected_components(g)),
                "giant_component_size": giant,
                "mean_weighted_degree": mean_wdeg,
                "mean_unweighted_degree": float(udegs.mean()) if len(udegs) else 0.0,
                "median_unweighted_degree": float(np.median(udegs)) if len(udegs) else 0.0,
                "p90_unweighted_degree": float(np.percentile(udegs, 90)) if len(udegs) else 0.0,
            }
        )
    return pd.DataFrame(rows)


def edge_weight_quantiles(series: pd.Series, qs: list[float]) -> dict[str, float]:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return {f"q{int(q*100)}": float("nan") for q in qs}
    out = {}
    for q in qs:
        out[f"q{int(q * 100)}"] = float(s.quantile(q))
    out["mean"] = float(s.mean())
    out["std"] = float(s.std())
    return out


def edge_change_analysis(refined_edges_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Requires edge_weight_orig, edge_trust, edge_weight_refined."""
    e = refined_edges_df.copy()
    for c in ("edge_weight_orig", "edge_trust", "edge_weight_refined"):
        if c not in e.columns:
            raise KeyError(c)
    e["_ratio"] = e["edge_weight_refined"] / np.maximum(e["edge_weight_orig"], 1e-12)
    e["_delta_w"] = e["edge_weight_refined"] - e["edge_weight_orig"]

    top_down = e.sort_values("_delta_w", ascending=True).head(50)
    least_changed = e.assign(_abs=np.abs(e["_delta_w"])).sort_values("_abs", ascending=True).head(50)

    qrows = []
    for col in ["edge_weight_orig", "edge_trust", "edge_weight_refined"]:
        qrows.append({"metric": col, **edge_weight_quantiles(e[col], [0.1, 0.25, 0.5, 0.75, 0.9])})
    qrows.append({"metric": "ratio_refined_over_orig", **edge_weight_quantiles(e["_ratio"], [0.1, 0.25, 0.5, 0.75, 0.9])})
    hist = pd.DataFrame(qrows)

    trust_thr = [0.2, 0.4, 0.6, 0.8]
    sup_rows = []
    n = len(e)
    for t in trust_thr:
        sup_rows.append(
            {
                "trust_lt_threshold": t,
                "n_edges": int((e["edge_trust"] < t).sum()),
                "frac_edges": float((e["edge_trust"] < t).mean()) if n else 0.0,
            }
        )
    suppression = pd.DataFrame(sup_rows)
    return {
        "top_downweighted": top_down,
        "least_changed": least_changed,
        "quantile_table": hist,
        "trust_suppression_counts": suppression,
    }


def attach_interpretive_columns_for_method1(
    refined_edges_df: pd.DataFrame,
    *,
    method1_cfg: Method1RefinementConfig | None = None,
) -> pd.DataFrame:
    """Recompute Method 1 semantic/infra/temporal views and local support (unsupervised)."""
    cfg = method1_cfg or Method1RefinementConfig()
    feat = build_method1_edge_feature_frame(refined_edges_df, weight_col="edge_weight_orig")
    views = compute_method1_view_scores(feat, cfg=cfg)
    local = compute_method1_local_structure_features(feat, cfg=cfg)
    out = refined_edges_df.copy().reset_index(drop=True)
    if len(out) != len(views):
        raise ValueError("refined edge row count mismatch after feature build")
    out["view_semantic"] = views["view_semantic"].to_numpy()
    out["view_infra"] = views["view_infra"].to_numpy()
    out["view_temporal"] = views["view_temporal"].to_numpy()
    out["local_support_score"] = local
    return out, {
        "infra_count_cols": feat["infra_count_cols"],
        "infra_idf_cols": feat["infra_idf_cols"],
        "infra_contrib_cols": feat["infra_contrib_cols"],
    }


def _qcut_labels(s: pd.Series, n_bins: int = 4) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    if s.notna().sum() < n_bins:
        return pd.Series(["all"] * len(s), index=s.index)
    try:
        return pd.qcut(s, q=n_bins, duplicates="drop")
    except ValueError:
        return pd.Series(["all"] * len(s), index=s.index)


def infra_diversity_rarity(
    refined_df: pd.DataFrame, feat_meta: dict[str, Any]
) -> tuple[pd.Series, pd.Series]:
    cnt_cols = feat_meta.get("infra_count_cols") or []
    idf_cols = feat_meta.get("infra_idf_cols") or []
    if cnt_cols:
        cnt_mat = refined_df[cnt_cols].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy()
        diversity = (cnt_mat > 0).sum(axis=1) / float(max(1, cnt_mat.shape[1]))
    else:
        diversity = pd.Series(0.5, index=refined_df.index)
    if idf_cols:
        idf_mat = refined_df[idf_cols].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy()
        rarity = idf_mat.sum(axis=1)
    else:
        rarity = pd.Series(0.0, index=refined_df.index)
    return pd.Series(diversity, index=refined_df.index), pd.Series(rarity, index=refined_df.index)


def suppression_profile_table(full_diag_df: pd.DataFrame, feat_meta: dict[str, Any]) -> pd.DataFrame:
    """Mean trust and mean shrink ratio by semantic/infra bins and diversity/rarity/local/temporal bins."""
    d = full_diag_df.copy()
    div, rar = infra_diversity_rarity(d, feat_meta)
    d["_div"] = div
    d["_rar"] = rar
    d["_ratio"] = d["edge_weight_refined"] / np.maximum(d["edge_weight_orig"], 1e-12)

    sem_bin = _qcut_labels(d["centroid_cosine"] if "centroid_cosine" in d.columns else d["view_semantic"])
    inf_bin = _qcut_labels(d["infra_score"] if "infra_score" in d.columns else d["view_infra"])
    d["_sem_bin"] = sem_bin.astype(str)
    d["_inf_bin"] = inf_bin.astype(str)
    d["_div_bin"] = _qcut_labels(d["_div"]).astype(str)
    d["_rar_bin"] = _qcut_labels(d["_rar"]).astype(str)
    d["_loc_bin"] = _qcut_labels(d["local_support_score"]).astype(str)
    d["_tmp_bin"] = _qcut_labels(
        d["temporal_score"] if "temporal_score" in d.columns else d["view_temporal"]
    ).astype(str)

    groups = [
        ("by_centroid_cosine_bin", "_sem_bin"),
        ("by_infra_score_bin", "_inf_bin"),
        ("by_sem_bin_x_inf_bin", ["_sem_bin", "_inf_bin"]),
        ("by_infra_diversity_bin", "_div_bin"),
        ("by_infra_rarity_sum_bin", "_rar_bin"),
        ("by_local_support_bin", "_loc_bin"),
        ("by_temporal_bin", "_tmp_bin"),
    ]
    rows: list[dict[str, Any]] = []
    for name, gdef in groups:
        if isinstance(gdef, list):
            gg = d.groupby(gdef, observed=False)
        else:
            gg = d.groupby(gdef, observed=False)
        for key, part in gg:
            names = gdef if isinstance(gdef, list) else [gdef]
            key_t = key if isinstance(key, tuple) else (key,)
            rec: dict[str, Any] = {"slice": name, "n_edges": len(part)}
            for i, (nm, kv) in enumerate(zip(names, key_t)):
                rec[f"bin_{i}_{nm.strip('_')}"] = str(kv)
            rec["mean_edge_trust"] = float(part["edge_trust"].mean())
            rec["mean_shrink_ratio"] = float(part["_ratio"].mean())
            rec["mean_edge_weight_orig"] = float(part["edge_weight_orig"].mean())
            rec["mean_edge_weight_refined"] = float(part["edge_weight_refined"].mean())
            rows.append(rec)
    return pd.DataFrame(rows)


def partition_diagnostics(
    shard_to_comm: dict[str, int],
    assignments_df: pd.DataFrame,
    *,
    condition_label: str,
) -> dict[str, Any]:
    vals = list(shard_to_comm.values())
    comm_sizes = Counter(vals)
    sizes = np.array(list(comm_sizes.values()))
    ep = s3.map_shards_to_email_predictions(assignments_df, shard_to_comm)
    email_comm_counts = ep.groupby("pred_community")["external_id"].nunique()

    top_comm = (
        pd.Series(comm_sizes)
        .sort_values(ascending=False)
        .head(15)
        .rename("n_shards")
        .reset_index()
        .rename(columns={"index": "community_id"})
    )
    return {
        "condition": condition_label,
        "n_communities": int(len(comm_sizes)),
        "n_singleton_shard_communities": int((sizes == 1).sum()),
        "mean_shards_per_community": float(sizes.mean()) if len(sizes) else 0.0,
        "median_shards_per_community": float(np.median(sizes)) if len(sizes) else 0.0,
        "max_shards_per_community": int(sizes.max()) if len(sizes) else 0,
        "mean_emails_per_community": float(email_comm_counts.mean()),
        "median_emails_per_community": float(email_comm_counts.median()),
        "top_communities_shard_count": top_comm,
    }


def prediction_changes(
    email_pred_a: pd.DataFrame,
    email_pred_b: pd.DataFrame,
    *,
    label_a: str,
    label_b: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    a = email_pred_a[["external_id", "pred_community"]].copy()
    b = email_pred_b[["external_id", "pred_community"]].copy()
    a.columns = ["external_id", f"pred_{label_a}"]
    b.columns = ["external_id", f"pred_{label_b}"]
    m = a.merge(b, on="external_id", how="inner")
    ch = m[m[f"pred_{label_a}"] != m[f"pred_{label_b}"]].copy()

    sa = email_pred_a[["shard_id", "pred_community"]].drop_duplicates()
    sb = email_pred_b[["shard_id", "pred_community"]].drop_duplicates()
    sa.columns = ["shard_id", f"pred_{label_a}"]
    sb.columns = ["shard_id", f"pred_{label_b}"]
    ms = sa.merge(sb, on="shard_id", how="inner")
    chs = ms[ms[f"pred_{label_a}"] != ms[f"pred_{label_b}"]].copy()
    return ch, chs


def pick_best_across_two_conditions(
    res_a: dict[str, Any],
    res_b: dict[str, Any],
    name_a: str,
    name_b: str,
) -> dict[str, Any]:
    """Choose condition with better joint objective on best row."""
    ra = res_a["summary"]
    rb = res_b["summary"]
    tuples = [
        (
            ra["completeness_at_best"],
            ra["v_measure_at_best"],
            ra["homogeneity_at_best"],
            name_a,
            res_a,
        ),
        (
            rb["completeness_at_best"],
            rb["v_measure_at_best"],
            rb["homogeneity_at_best"],
            name_b,
            res_b,
        ),
    ]
    tuples.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
    return {"winner_name": tuples[0][3], "winner_result": tuples[0][4], "ranked": tuples}


def baseline_communities_split_report(
    email_pred_baseline: pd.DataFrame,
    email_pred_method1: pd.DataFrame,
    *,
    min_baseline_community_emails: int = 30,
) -> pd.DataFrame:
    """Baseline communities with many emails that map to many Method 1 communities."""
    b = email_pred_baseline.copy()
    m = email_pred_method1.copy()
    mg = b.merge(
        m[["external_id", "pred_community"]].rename(columns={"pred_community": "pred_m1"}),
        on="external_id",
        how="inner",
    )
    rows = []
    for bc, g in mg.groupby("pred_community"):
        if len(g) < min_baseline_community_emails:
            continue
        n_m1 = g["pred_m1"].nunique()
        rows.append(
            {
                "baseline_community": int(bc),
                "n_emails": len(g),
                "n_method1_communities_spanning": int(n_m1),
                "split_indicator": float(n_m1 - 1),
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values(["n_method1_communities_spanning", "n_emails"], ascending=[False, False]).reset_index(
        drop=True
    )


def gt_fragmentation_side_by_side(
    campaign_to_members: dict[Any, list[str]],
    email_pred_baseline: pd.DataFrame,
    email_pred_method1: pd.DataFrame,
    *,
    baseline_label: str = "baseline",
    method1_label: str = "method1",
) -> pd.DataFrame:
    map_b = s3.proto_prediction_map_from_email_df(email_pred_baseline)
    map_m = s3.proto_prediction_map_from_email_df(email_pred_method1)
    fr_b = cfh.campaign_fragmentation_df(campaign_to_members, map_b)
    fr_m = cfh.campaign_fragmentation_df(campaign_to_members, map_m)
    fb = fr_b.rename(
        columns={
            "num_pred_clusters": f"num_pred_clusters_{baseline_label}",
            "dominant_fraction": f"dominant_fraction_{baseline_label}",
            "fragmentation_score": f"fragmentation_score_{baseline_label}",
        }
    )
    fm = fr_m[
        ["campaign_id", "num_pred_clusters", "dominant_fraction", "fragmentation_score"]
    ].rename(
        columns={
            "num_pred_clusters": f"num_pred_clusters_{method1_label}",
            "dominant_fraction": f"dominant_fraction_{method1_label}",
            "fragmentation_score": f"fragmentation_score_{method1_label}",
        }
    )
    out = fb.merge(fm, on="campaign_id", how="outer")
    out["delta_num_pred_clusters"] = (
        out[f"num_pred_clusters_{method1_label}"] - out[f"num_pred_clusters_{baseline_label}"]
    )
    out["delta_fragmentation_score"] = (
        out[f"fragmentation_score_{method1_label}"] - out[f"fragmentation_score_{baseline_label}"]
    )
    return out.sort_values("delta_num_pred_clusters", ascending=False).reset_index(drop=True)


def load_method1_config_json(method1_dir: Path) -> dict[str, Any] | None:
    p = Path(method1_dir) / "semantic_shard_method1_config.json"
    if not p.is_file():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Method 1 calibration experiments
# ---------------------------------------------------------------------------


def _calibration_error_row(spec: dict[str, Any], err: str) -> dict[str, Any]:
    rv = spec.get("resolution_values") or []
    mw = spec.get("min_edge_weight_values") or []
    return {
        "experiment_name": spec.get("experiment_name", ""),
        "community_method": spec.get("community_method", ""),
        "refinement_variant_name": spec.get("refinement_variant_name", ""),
        "blend_rule": spec.get("blend_rule", ""),
        "blend_floor": spec.get("blend_floor", ""),
        "trust_gamma": spec.get("trust_gamma", ""),
        "convex_alpha": spec.get("convex_alpha", ""),
        "use_perturbation_stability": spec.get("use_perturbation_stability", ""),
        "use_local_structure": spec.get("use_local_structure", ""),
        "edge_source_type": spec.get("edge_source_type", ""),
        "weight_column": spec.get("weight_column", ""),
        "threshold_grid_name": spec.get("threshold_grid_name", ""),
        "resolution_values": json.dumps(list(rv)),
        "min_edge_weight_values": json.dumps(list(mw)),
        "error": err,
    }


def trust_shrink_distribution_stats(refined_edges_df: pd.DataFrame) -> dict[str, Any]:
    """Trust and shrink ratio summaries for a Method 1 refined edge table."""
    req = ("edge_weight_orig", "edge_trust", "edge_weight_refined")
    for c in req:
        if c not in refined_edges_df.columns:
            nan = float("nan")
            return {
                "trust_mean": nan,
                "trust_median": nan,
                "trust_p90": nan,
                "trust_max": nan,
                "frac_trust_lt_0.2": nan,
                "frac_trust_lt_0.4": nan,
                "frac_trust_lt_0.6": nan,
                "shrink_mean": nan,
                "shrink_median": nan,
                "shrink_p90": nan,
                "frac_shrink_lt_0.25": nan,
                "frac_shrink_lt_0.5": nan,
                "trust_cal_mean": nan,
                "trust_cal_median": nan,
                "trust_cal_p90": nan,
                "trust_cal_max": nan,
            }

    t = pd.to_numeric(refined_edges_df["edge_trust"], errors="coerce")
    orig = pd.to_numeric(refined_edges_df["edge_weight_orig"], errors="coerce").fillna(0.0)
    ref = pd.to_numeric(refined_edges_df["edge_weight_refined"], errors="coerce")
    shrink = ref / np.maximum(orig, 1e-12)
    out = {
        "trust_mean": float(t.mean()),
        "trust_median": float(t.median()),
        "trust_p90": float(t.quantile(0.9)),
        "trust_max": float(t.max()),
        "frac_trust_lt_0.2": float((t < 0.2).mean()),
        "frac_trust_lt_0.4": float((t < 0.4).mean()),
        "frac_trust_lt_0.6": float((t < 0.6).mean()),
        "shrink_mean": float(shrink.mean()),
        "shrink_median": float(shrink.median()),
        "shrink_p90": float(shrink.quantile(0.9)),
        "frac_shrink_lt_0.25": float((shrink < 0.25).mean()),
        "frac_shrink_lt_0.5": float((shrink < 0.5).mean()),
    }
    if "edge_trust_calibrated" in refined_edges_df.columns:
        tc = pd.to_numeric(refined_edges_df["edge_trust_calibrated"], errors="coerce")
        out["trust_cal_mean"] = float(tc.mean())
        out["trust_cal_median"] = float(tc.median())
        out["trust_cal_p90"] = float(tc.quantile(0.9))
        out["trust_cal_max"] = float(tc.max())
    else:
        out["trust_cal_mean"] = float("nan")
        out["trust_cal_median"] = float("nan")
        out["trust_cal_p90"] = float("nan")
        out["trust_cal_max"] = float("nan")
    return out


def graph_stats_one_threshold(
    shard_ids: list[str],
    edges_df: pd.DataFrame,
    *,
    weight_col: str,
    threshold: float,
    label: str,
) -> dict[str, Any]:
    gdf = graph_stats_at_thresholds(
        shard_ids,
        edges_df,
        weight_col=weight_col,
        min_edge_weight_values=[float(threshold)],
        label=label,
    )
    return gdf.iloc[0].to_dict()


def method1_cfg_from_base_and_overrides(
    base: Method1RefinementConfig,
    overrides: dict[str, Any] | None,
) -> Method1RefinementConfig:
    d = base.to_dict()
    if overrides:
        for k, v in overrides.items():
            if k in Method1RefinementConfig.__dataclass_fields__:
                d[k] = v
    return Method1RefinementConfig.from_dict(d)


def ensure_refined_edges_variant(
    *,
    variant_cache_id: str,
    baseline_edges_df: pd.DataFrame,
    source_mode: str,
    method1_default_csv: Path | None,
    calibration_cache_dir: Path,
    base_method1_cfg: Method1RefinementConfig,
    method1_overrides: dict[str, Any] | None,
    force_recompute: bool,
    method1_calibration_runs_root: str | Path | None = None,
) -> tuple[pd.DataFrame, Path | None]:
    """
    Return refined edges for a calibration variant.

    ``source_mode``:
    - ``load_default``: read ``method1_default_csv`` if present (unless ``force_recompute``).
    - ``recompute``: run pipeline with ``base_method1_cfg`` + overrides; cache CSV under cache dir.
    - ``artifact_bundle``: read ``{method1_calibration_runs_root}/{variant_cache_id}/semantic_shard_step2_edges_refined.csv``;
      if missing (or ``force_recompute``), recompute and save the full bundle (CSV + JSON) under that directory.
    """
    calibration_cache_dir = Path(calibration_cache_dir)
    calibration_cache_dir.mkdir(parents=True, exist_ok=True)
    cache_p = calibration_cache_dir / f"method1_refined__{variant_cache_id}.csv"
    mode = str(source_mode).lower().strip()

    if mode == "artifact_bundle":
        if not method1_calibration_runs_root:
            raise ValueError("artifact_bundle mode requires method1_calibration_runs_root")
        bundle_dir = Path(method1_calibration_runs_root).expanduser().resolve() / str(variant_cache_id)
        edges_name = "semantic_shard_step2_edges_refined.csv"
        bundle_csv = bundle_dir / edges_name
        if bundle_csv.is_file() and not force_recompute:
            df = pd.read_csv(bundle_csv)
            df["shard_a"] = df["shard_a"].astype(str)
            df["shard_b"] = df["shard_b"].astype(str)
            return df, bundle_csv
        bundle_dir.mkdir(parents=True, exist_ok=True)
        cfg = method1_cfg_from_base_and_overrides(base_method1_cfg, method1_overrides or {})
        refined, fit_summary, _ = run_method1_edge_refinement_pipeline(
            baseline_edges_df, cfg=cfg, output_dir=None
        )
        save_method1_calibration_variant_bundle(
            refined, bundle_dir=bundle_dir, cfg=cfg, fit_summary=fit_summary
        )
        return refined, bundle_csv

    if mode == "load_default":
        p = method1_default_csv
        if p is not None and Path(p).is_file() and not force_recompute:
            df = pd.read_csv(p)
            df["shard_a"] = df["shard_a"].astype(str)
            df["shard_b"] = df["shard_b"].astype(str)
            return df, None
        cfg = method1_cfg_from_base_and_overrides(base_method1_cfg, method1_overrides or {})
        refined, _, _ = run_method1_edge_refinement_pipeline(
            baseline_edges_df, cfg=cfg, output_dir=None
        )
        refined.to_csv(cache_p, index=False)
        return refined, cache_p

    if mode != "recompute":
        raise ValueError(f"Unknown refined source_mode {source_mode!r}")

    if cache_p.is_file() and not force_recompute:
        df = pd.read_csv(cache_p)
        df["shard_a"] = df["shard_a"].astype(str)
        df["shard_b"] = df["shard_b"].astype(str)
        return df, cache_p

    cfg = method1_cfg_from_base_and_overrides(base_method1_cfg, method1_overrides or {})
    refined, _, _ = run_method1_edge_refinement_pipeline(baseline_edges_df, cfg=cfg, output_dir=None)
    refined.to_csv(cache_p, index=False)
    return refined, cache_p


def run_calibration_experiment_spec(
    spec: dict[str, Any],
    *,
    assignments_df: pd.DataFrame,
    shard_ids: list[str],
    baseline_edges_df: pd.DataFrame,
    gt_label_map: dict[str, Any],
    seed: int,
    method1_default_csv: Path | None,
    calibration_cache_dir: Path,
    base_method1_cfg: Method1RefinementConfig,
    force_recompute_refined: bool,
    skip_leiden_if_unavailable: bool,
    availability: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Execute one calibration row (one community method, one edge/threshold grid).

    ``spec`` keys: experiment_name, edge_source_type, community_method, weight_column,
    refinement_variant_name, use_perturbation_stability, use_local_structure, blend_rule,
    blend_floor, threshold_grid_name, resolution_values, min_edge_weight_values,
    refined_variant_cache_id, refined_source_mode, method1_overrides (optional).
    """
    avail = availability or check_community_detection_availability()
    method = str(spec["community_method"]).lower()
    if method == "leiden" and not avail.get("leiden") and skip_leiden_if_unavailable:
        return {
            "row": _calibration_error_row(spec, "leiden_unavailable"),
            "run_result": None,
            "error": "leiden_unavailable",
        }

    edge_src = str(spec["edge_source_type"]).lower()
    wcol = str(spec["weight_column"])

    if edge_src == "baseline":
        edges_df = baseline_edges_df
        trust_stats = {k: float("nan") for k in [
            "trust_mean", "trust_median", "trust_p90", "trust_max",
            "frac_trust_lt_0.2", "frac_trust_lt_0.4", "frac_trust_lt_0.6",
            "shrink_mean", "shrink_median", "shrink_p90",
            "frac_shrink_lt_0.25", "frac_shrink_lt_0.5",
            "trust_cal_mean", "trust_cal_median", "trust_cal_p90", "trust_cal_max",
        ]}
    else:
        refined, _ = ensure_refined_edges_variant(
            variant_cache_id=str(spec["refined_variant_cache_id"]),
            baseline_edges_df=baseline_edges_df,
            source_mode=str(spec["refined_source_mode"]),
            method1_default_csv=method1_default_csv,
            calibration_cache_dir=calibration_cache_dir,
            base_method1_cfg=base_method1_cfg,
            method1_overrides=spec.get("method1_overrides"),
            force_recompute=force_recompute_refined,
            method1_calibration_runs_root=spec.get("method1_calibration_runs_root"),
        )
        edges_df = refined
        if wcol not in edges_df.columns:
            return {
                "row": _calibration_error_row(spec, f"missing_column_{wcol}"),
                "run_result": None,
                "error": f"missing_column_{wcol}",
            }
        trust_stats = trust_shrink_distribution_stats(edges_df)

    try:
        res = run_condition_sweep(
            condition_name=str(spec["experiment_name"]),
            assignments_df=assignments_df,
            shard_ids=shard_ids,
            edges_df=edges_df,
            gt_label_map=gt_label_map,
            method=method,
            weight_col=wcol,
            resolution_values=list(spec["resolution_values"]),
            min_edge_weight_values=list(spec["min_edge_weight_values"]),
            seed=seed,
        )
    except Exception as e:
        return {
            "row": _calibration_error_row(spec, repr(e)),
            "run_result": None,
            "error": repr(e),
        }

    summ = res["summary"]
    thr = float(summ["best_min_edge_weight"])
    g = graph_stats_one_threshold(
        shard_ids,
        edges_df,
        weight_col=wcol,
        threshold=thr,
        label=str(spec["experiment_name"]),
    )
    part_full = partition_diagnostics(
        res["shard_to_comm_best"], assignments_df, condition_label=str(spec["experiment_name"])
    )
    top_comm = part_full.get("top_communities_shard_count")
    part = {k: v for k, v in part_full.items() if k != "top_communities_shard_count"}

    row: dict[str, Any] = {
        "experiment_name": spec["experiment_name"],
        "community_method": method,
        "refinement_variant_name": spec["refinement_variant_name"],
        "blend_rule": spec.get("blend_rule", "n/a"),
        "blend_floor": spec.get("blend_floor", ""),
        "trust_gamma": spec.get("trust_gamma", ""),
        "convex_alpha": spec.get("convex_alpha", ""),
        "use_perturbation_stability": spec.get("use_perturbation_stability", ""),
        "use_local_structure": spec.get("use_local_structure", ""),
        "edge_source_type": spec["edge_source_type"],
        "weight_column": wcol,
        "threshold_grid_name": spec["threshold_grid_name"],
        "resolution_values": json.dumps(list(spec["resolution_values"])),
        "min_edge_weight_values": json.dumps(list(spec["min_edge_weight_values"])),
        "best_homogeneity": summ["homogeneity_at_best"],
        "best_completeness": summ["completeness_at_best"],
        "best_v_measure": summ["v_measure_at_best"],
        "best_resolution": summ["best_resolution"],
        "best_min_edge_weight": summ["best_min_edge_weight"],
        "n_eval": summ["n_eval_at_best"],
        "coverage_gt": summ["coverage_gt_at_best"],
        "edges_surviving_at_best_threshold": g.get("n_edges_ge_threshold"),
        "connected_components_at_best_threshold": g.get("n_components"),
        "giant_component_size_at_best_threshold": g.get("giant_component_size"),
        "mean_weighted_degree_at_best_threshold": g.get("mean_weighted_degree"),
        "median_degree_at_best_threshold": g.get("median_unweighted_degree"),
        "p90_degree_at_best_threshold": g.get("p90_unweighted_degree"),
        "n_communities_at_best": part["n_communities"],
        "n_singleton_communities_at_best": part["n_singleton_shard_communities"],
        "mean_community_size_shards_at_best": part["mean_shards_per_community"],
        "median_community_size_shards_at_best": part["median_shards_per_community"],
        "largest_community_size_shards_at_best": part["max_shards_per_community"],
        **trust_stats,
        "error": "",
    }
    return {
        "row": row,
        "run_result": res,
        "graph_at_best": g,
        "partition_at_best": part,
        "top_communities": top_comm,
        "edges_df_used": edges_df,
    }


def pick_best_experiment_row(
    summary_df: pd.DataFrame,
    *,
    edge_source_type: str,
) -> pd.Series | None:
    sub = summary_df[summary_df["edge_source_type"].astype(str).str.lower() == edge_source_type.lower()]
    if "error" in sub.columns:
        sub = sub[(sub["error"].isna()) | (sub["error"].astype(str) == "")]
    sub = sub.dropna(subset=["best_completeness"])
    if sub.empty:
        return None
    sub = sub.sort_values(
        ["best_completeness", "best_v_measure", "best_homogeneity"],
        ascending=[False, False, False],
    )
    return sub.iloc[0]


def build_calibration_decision_markdown(
    summary_df: pd.DataFrame,
    *,
    best_baseline_experiment_name: str | None,
    best_method1_experiment_name: str | None,
) -> str:
    """Algorithmic answers for calibration decisions (data-grounded)."""
    df = summary_df.copy()
    if "error" in df.columns:
        df = df[df["error"].isna() | (df["error"].astype(str) == "")]
    lines: list[str] = []
    lines.append("# Method 1 calibration — decision summary\n")
    lines.append("Auto-generated from experiment summary CSV. Ground truth used only in metrics columns.\n")

    def _sub(variant_substr: str) -> pd.DataFrame:
        return df[df["refinement_variant_name"].astype(str).str.contains(variant_substr, regex=False)]

    # 1) Lower thresholds only: compare A (baseline grid) vs B (refined_low grid) same variant tag
    a = df[df["threshold_grid_name"].astype(str) == "baseline_tight"]
    b = df[df["threshold_grid_name"].astype(str) == "refined_low"]
    if not a.empty and not b.empty:
        ca = a["best_completeness"].mean()
        cb = b["best_completeness"].mean()
        lines.append("## 1. Does lowering thresholds alone recover completeness?\n")
        lines.append(
            f"- Mean **best_completeness** across runs with `baseline_tight` grid: **{ca:.4f}** (n={len(a)}).\n"
        )
        lines.append(
            f"- Mean **best_completeness** across runs with `refined_low` grid: **{cb:.4f}** (n={len(b)}).\n"
        )
        lines.append(f"- Delta (refined_low − baseline_tight): **{cb - ca:+.4f}**.\n")
        if cb > ca + 0.01:
            lines.append("- **Conclusion:** Lower refined thresholds **materially raise** reported completeness on average.\n")
        elif cb < ca - 0.01:
            lines.append("- **Conclusion:** Lower thresholds **did not** beat baseline-tight grid on average (check per-row names).\n")
        else:
            lines.append("- **Conclusion:** **Mixed / small** effect; inspect per-experiment rows.\n")
        a_only = df[df["refinement_variant_name"].astype(str) == "A_current_loaded"]
        b_only = df[df["refinement_variant_name"].astype(str) == "B_lower_threshold_only"]
        if not a_only.empty and not b_only.empty:
            ca2 = float(a_only["best_completeness"].mean())
            cb2 = float(b_only["best_completeness"].mean())
            lines.append(
                f"\n### Same refined edges, different grids (A vs B)\n"
                f"- **A_current_loaded** (baseline_tight on loaded refined): mean completeness **{ca2:.4f}** (n={len(a_only)}).\n"
                f"- **B_lower_threshold_only** (refined_low on same file): mean completeness **{cb2:.4f}** (n={len(b_only)}).\n"
                f"- Delta (B − A): **{cb2 - ca2:+.4f}**.\n"
            )
    else:
        lines.append("## 1. Does lowering thresholds alone recover completeness?\n")
        lines.append("- **Insufficient rows** (need both `baseline_tight` and `refined_low` threshold_grid_name).\n")

    # 2) Perturbation: compare variants containing `_no_perturb` vs `_current` or production, same threshold grid where possible
    lines.append("\n## 2. Does removing perturbation stability help?\n")
    poff = df[df["refinement_variant_name"].astype(str).str.contains("no_perturb")]
    pon = df[~df["refinement_variant_name"].astype(str).str.contains("no_perturb")]
    pon_m1 = pon[pon["edge_source_type"].astype(str).str.lower() == "method1"]
    if not poff.empty and not pon_m1.empty:
        lines.append(
            f"- Mean completeness **with** no_perturb in name: **{poff['best_completeness'].mean():.4f}** (n={len(poff)}).\n"
        )
        lines.append(
            f"- Mean completeness **other** Method 1 rows: **{pon_m1['best_completeness'].mean():.4f}** (n={len(pon_m1)}).\n"
        )
    else:
        lines.append("- **Not enough paired variants** to summarize perturbation ablation.\n")

    lines.append("\n## 3. Does removing local structure help?\n")
    loff = df[df["refinement_variant_name"].astype(str).str.contains("no_local")]
    if not loff.empty:
        other_m1 = df[
            (df["edge_source_type"].astype(str).str.lower() == "method1")
            & (~df["refinement_variant_name"].astype(str).str.contains("no_local"))
        ]
        lines.append(f"- Mean completeness **no_local** rows: **{loff['best_completeness'].mean():.4f}**.\n")
        if not other_m1.empty:
            lines.append(
                f"- Mean completeness **other** Method 1: **{other_m1['best_completeness'].mean():.4f}**.\n"
            )
    else:
        lines.append("- No `no_local` variant rows found.\n")

    lines.append("\n## 4. Does the softer blend rule help?\n")
    soft = df[df["blend_rule"].astype(str).str.lower() == "softened"]
    mult = df[
        (df["edge_source_type"].astype(str).str.lower() == "method1")
        & (df["blend_rule"].astype(str).str.lower() == "multiplicative")
    ]
    if not soft.empty and not mult.empty:
        lines.append(
            f"- Mean completeness **softened** blend: **{soft['best_completeness'].mean():.4f}** (n={len(soft)}).\n"
        )
        lines.append(
            f"- Mean completeness **multiplicative** Method 1: **{mult['best_completeness'].mean():.4f}** (n={len(mult)}).\n"
        )
    else:
        lines.append("- **Cannot compare** softened vs multiplicative (missing rows).\n")
    cvx = df[df["blend_rule"].astype(str).str.lower() == "convex"]
    if not cvx.empty:
        lines.append(
            f"- Mean completeness **convex** blend (norm orig + trust): **{cvx['best_completeness'].mean():.4f}** (n={len(cvx)}).\n"
        )

    lines.append("\n## 5. Strongest overall competitor to baseline\n")
    lines.append(f"- **Best baseline experiment (by completeness → V → homogeneity):** `{best_baseline_experiment_name}`.\n")
    lines.append(f"- **Best Method 1 calibration experiment:** `{best_method1_experiment_name}`.\n")
    bl = df[df["experiment_name"].astype(str) == str(best_baseline_experiment_name)]
    m1 = df[df["experiment_name"].astype(str) == str(best_method1_experiment_name)]
    if not bl.empty and not m1.empty:
        b0, m0 = bl.iloc[0], m1.iloc[0]
        lines.append(
            f"- Baseline best: completeness **{b0['best_completeness']:.4f}**, V **{b0['best_v_measure']:.4f}**, homogeneity **{b0['best_homogeneity']:.4f}**.\n"
        )
        lines.append(
            f"- Method 1 best: completeness **{m0['best_completeness']:.4f}**, V **{m0['best_v_measure']:.4f}**, homogeneity **{m0['best_homogeneity']:.4f}**.\n"
        )

    lines.append("\n## 6. Main failure mode (from aggregates)\n")
    if not df.empty:
        m1_only = df[df["edge_source_type"].astype(str).str.lower() == "method1"]
        if not m1_only.empty:
            low_trust = float(m1_only["trust_median"].median())
            low_shrink = float(m1_only["shrink_median"].median())
            sparse = float(m1_only["edges_surviving_at_best_threshold"].median())
            lines.append(
                f"- Median **trust_median** across Method 1 experiments: **{low_trust:.4f}**; **shrink_median**: **{low_shrink:.4f}**.\n"
            )
            lines.append(
                f"- Median **edges surviving** at chosen best threshold: **{sparse:.0f}** (compare to baseline runs).\n"
            )
        lines.append(
            "- Rank likely causes: **(a)** threshold mismatch if `refined_low` grid lifts completeness; **(b)** destructive blend if softened blend improves completeness vs multiplicative; **(c)** perturbation if no_perturb rows beat matched others; **(d)** local structure if no_local rows differ systematically; **(e)** combination if several changes help a little.\n"
        )

    lines.append("\n---\n*End.*\n")
    return "".join(lines)

