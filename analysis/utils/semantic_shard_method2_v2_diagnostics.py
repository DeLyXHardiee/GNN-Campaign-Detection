"""
Post-training diagnostics for Method 1 V2 edge plausibility vs baseline and teacher.

GT labels are used only inside Step 3 ``evaluate_external_metrics`` (same as existing notebooks).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from analysis.utils.raw_gnn_notebook import load_ground_truth_structures
from analysis.utils.semantic_shard_edge_teacher_score import TEACHER_WEIGHT_COL, build_teacher_scored_edges
from analysis.utils.semantic_shard_edge_plausibility_v2_gt_diagnostics import (
    attach_edge_taxonomy,
    build_same_cross_hsli_masks,
    compact_gaps_from_scores,
)
from analysis.utils.semantic_shard_edge_plausibility_v2_views import build_view_scores_df
from analysis.utils.semantic_shard_method2_v2_inference import (
    list_epoch_checkpoints,
    load_v2_training_config,
    score_edges_v2_checkpoint,
)
from analysis.utils.semantic_shard_step3_helpers import best_sweep_metric_row, run_community_sweep

SCORE_QUANTILE_LEVELS: tuple[float, ...] = (0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99)


def load_gt_label_map(gt_json_path: str | Path) -> dict[str, Any]:
    label_map, _eid_to_row, _campaign_to_members = load_ground_truth_structures(gt_json_path)
    return label_map


def merge_edge_score_frame(
    baseline_edges: pd.DataFrame,
    *,
    teacher_scored: pd.DataFrame | None = None,
    v2_scored: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Left-join optional teacher / V2 scored tables onto baseline edges on ``shard_a``, ``shard_b``.
    """
    base = baseline_edges.copy()
    base["shard_a"] = base["shard_a"].astype(str)
    base["shard_b"] = base["shard_b"].astype(str)
    keys = ["shard_a", "shard_b"]

    if teacher_scored is not None:
        t = teacher_scored.copy()
        t["shard_a"] = t["shard_a"].astype(str)
        t["shard_b"] = t["shard_b"].astype(str)
        new_t = [c for c in t.columns if c not in base.columns]
        base = base.merge(t[keys + new_t], on=keys, how="left")

    if v2_scored is not None:
        v = v2_scored.copy()
        v["shard_a"] = v["shard_a"].astype(str)
        v["shard_b"] = v["shard_b"].astype(str)
        new_v = [c for c in v.columns if c not in base.columns]
        base = base.merge(v[keys + new_v], on=keys, how="left")

    return base


def _finite_mask(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.isfinite(a) & np.isfinite(b)


def pearson_r(a: np.ndarray | pd.Series, b: np.ndarray | pd.Series) -> float:
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    m = _finite_mask(x, y)
    if m.sum() < 2:
        return float("nan")
    x = x[m]
    y = y[m]
    if np.std(x) < 1e-15 or np.std(y) < 1e-15:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman_r(a: np.ndarray | pd.Series, b: np.ndarray | pd.Series) -> float:
    x = pd.Series(a, dtype="float64")
    y = pd.Series(b, dtype="float64")
    m = x.notna() & y.notna()
    if m.sum() < 2:
        return float("nan")
    rx = x[m].rank(method="average")
    ry = y[m].rank(method="average")
    return pearson_r(rx.to_numpy(), ry.to_numpy())


def score_column_summary(series: pd.Series, name: str) -> dict[str, Any]:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        d = {f"{name}__{k}": float("nan") for k in ["min", "max", "mean", "std", "median"]}
        d[f"{name}__n"] = 0
        d[f"{name}__n_unique_approx"] = 0
        return d
    qd = {f"{name}__q_{int(q * 100):02d}": float(s.quantile(q)) for q in SCORE_QUANTILE_LEVELS}
    out: dict[str, Any] = {
        f"{name}__n": int(len(s)),
        f"{name}__min": float(s.min()),
        f"{name}__max": float(s.max()),
        f"{name}__mean": float(s.mean()),
        f"{name}__std": float(s.std(ddof=0)),
        f"{name}__median": float(s.median()),
        f"{name}__frac_mid_04_06": float(s.between(0.4, 0.6).mean()),
        f"{name}__frac_lt_01": float((s < 0.1).mean()),
        f"{name}__frac_gt_09": float((s > 0.9).mean()),
        f"{name}__n_unique_approx": int(s.nunique()),
    }
    out.update(qd)
    return out


def summaries_to_table(per_name: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for source, d in per_name.items():
        row: dict[str, Any] = {"score_source": source}
        for k, v in d.items():
            if "__" in k:
                _, metric = k.split("__", 1)
                row[metric] = v
            else:
                row[k] = v
        rows.append(row)
    return pd.DataFrame(rows)


def fraction_above_thresholds_table(
    series_by_name: dict[str, pd.Series],
    thresholds: tuple[float, ...] = (0.5, 0.6, 0.7, 0.8, 0.9),
) -> pd.DataFrame:
    rows = []
    for name, ser in series_by_name.items():
        s = pd.to_numeric(ser, errors="coerce").dropna()
        if s.empty:
            continue
        row: dict[str, Any] = {"score_source": name}
        for t in thresholds:
            row[f"frac_ge_{t:.2f}"] = float((s >= t).mean())
        rows.append(row)
    return pd.DataFrame(rows)


def save_disagreement_bucket_csvs(buckets: dict[str, pd.DataFrame], output_dir: str | Path) -> dict[str, str]:
    out = Path(output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    for k, df in buckets.items():
        safe = k.replace("/", "_")
        p = out / f"interesting_edges__{safe}.csv"
        df.to_csv(p, index=False)
        paths[k] = str(p)
    return paths


def pairwise_correlation_table(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    rows = []
    for i, ca in enumerate(cols):
        for cb in cols[i + 1 :]:
            a = pd.to_numeric(df[ca], errors="coerce")
            b = pd.to_numeric(df[cb], errors="coerce")
            rows.append(
                {
                    "col_a": ca,
                    "col_b": cb,
                    "pearson": pearson_r(a, b),
                    "spearman": spearman_r(a, b),
                    "n_pairwise": int((a.notna() & b.notna()).sum()),
                }
            )
    return pd.DataFrame(rows)


def rank_positions(descending_scores: np.ndarray) -> np.ndarray:
    """Lower rank = stronger edge (rank 0 = highest score)."""
    s = np.asarray(descending_scores, dtype=np.float64)
    order = np.argsort(-s, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.int64)
    ranks[order] = np.arange(len(s), dtype=np.int64)
    return ranks


def topk_overlap_scores(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    ks: Iterable[int] = (100, 500, 1000, 5000),
) -> pd.DataFrame:
    """Fraction of top-k by ``scores_a`` that also appear in top-k by ``scores_b`` (by edge index)."""
    sa = np.asarray(scores_a, dtype=np.float64)
    sb = np.asarray(scores_b, dtype=np.float64)
    n = len(sa)
    ra = rank_positions(sa)
    rb = rank_positions(sb)
    rows = []
    for k in ks:
        kk = min(int(k), n)
        if kk < 1:
            rows.append({"k": k, "overlap_count": 0, "overlap_fraction": float("nan")})
            continue
        top_a = set(np.where(ra < kk)[0].tolist())
        top_b = set(np.where(rb < kk)[0].tolist())
        inter = len(top_a & top_b)
        rows.append({"k": k, "overlap_count": inter, "overlap_fraction": float(inter) / float(kk)})
    return pd.DataFrame(rows)


def add_rank_columns(merged: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = merged.copy()
    for c in cols:
        if c not in out.columns:
            continue
        s = pd.to_numeric(out[c], errors="coerce").to_numpy(dtype=np.float64)
        out[f"rank_{c}"] = rank_positions(np.nan_to_num(s, nan=-np.inf))
    return out


def disagreement_edge_buckets(
    merged: pd.DataFrame,
    *,
    top_n: int = 500,
    baseline_col: str = "edge_weight",
    teacher_col: str = TEACHER_WEIGHT_COL,
    v2_col: str = "edge_plausibility",
    feature_cols: list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Tables of edges in the tails of score differences (interpretive inspection).

    Uses rank-based cutoffs on the marginal distributions (high = top ``top_n`` by rank).
    """
    m = merged.copy()
    for c in (baseline_col, teacher_col, v2_col):
        if c not in m.columns:
            m[c] = np.nan
    bw = pd.to_numeric(m[baseline_col], errors="coerce")
    tw = pd.to_numeric(m[teacher_col], errors="coerce")
    vw = pd.to_numeric(m[v2_col], errors="coerce")

    def _top_idx(s: pd.Series, n: int, largest: bool) -> set[int]:
        s = s.dropna()
        if s.empty:
            return set()
        n = min(n, len(s))
        if largest:
            return set(s.nlargest(n).index.tolist())
        return set(s.nsmallest(n).index.tolist())

    n = int(top_n)
    hi_b = _top_idx(bw, n, True)
    lo_b = _top_idx(bw, n, False)
    hi_t = _top_idx(tw, n, True)
    lo_t = _top_idx(tw, n, False)
    hi_v = _top_idx(vw, n, True)
    lo_v = _top_idx(vw, n, False)

    base_cols = ["shard_a", "shard_b", baseline_col, teacher_col, v2_col]
    extra = [c for c in (feature_cols or []) if c in m.columns]
    keep = list(dict.fromkeys(base_cols + extra))

    def _take(idxs: set[int], name: str) -> pd.DataFrame:
        if not idxs:
            return pd.DataFrame(columns=keep)
        return m.loc[sorted(idxs), keep].copy().assign(_bucket=name)

    out: dict[str, pd.DataFrame] = {}
    out["v2_high_baseline_low"] = _take(hi_v & lo_b, "v2_high_baseline_low")
    out["v2_low_baseline_high"] = _take(lo_v & hi_b, "v2_low_baseline_high")
    out["v2_high_teacher_low"] = _take(hi_v & lo_t, "v2_high_teacher_low")
    out["v2_low_teacher_high"] = _take(lo_v & hi_t, "v2_low_teacher_high")
    out["teacher_high_v2_low"] = _take(hi_t & lo_v, "teacher_high_v2_low")
    out["teacher_low_v2_high"] = _take(lo_t & hi_v, "teacher_low_v2_high")
    return out


def run_step3_sweep_df(
    *,
    assignments_df: pd.DataFrame,
    shard_ids: list[str],
    edges_df: pd.DataFrame,
    gt_label_map: dict[str, Any],
    method: str,
    resolution_values: list[float],
    min_edge_weight_values: list[float],
    weight_col: str,
    seed: int = 0,
) -> pd.DataFrame:
    sweep_df, _email_preds, _info = run_community_sweep(
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
    return sweep_df


def constrained_best_row(
    sweep_df: pd.DataFrame,
    *,
    maximize: str,
    subject_to: str,
    subject_min: float,
) -> pd.Series:
    if sweep_df.empty:
        return pd.Series(dtype=float)
    d = sweep_df.copy()
    d[maximize] = pd.to_numeric(d[maximize], errors="coerce")
    d[subject_to] = pd.to_numeric(d[subject_to], errors="coerce")
    d = d[np.isfinite(d[maximize]) & np.isfinite(d[subject_to])]
    d = d[d[subject_to] >= float(subject_min)]
    if d.empty:
        return pd.Series(dtype=float)
    return d.sort_values(maximize, ascending=False).iloc[0]


def fair_method_comparison_table(
    *,
    baseline_sweep: pd.DataFrame,
    teacher_sweep: pd.DataFrame,
    v2_sweep: pd.DataFrame,
    method_name_baseline: str = "baseline_edge_weight",
    method_name_teacher: str = "teacher_agreement",
    method_name_v2: str = "v2_plausibility",
) -> pd.DataFrame:
    """
    One row per method: best V-measure and constrained optima vs baseline thresholds.
    """
    b_best = best_sweep_metric_row(baseline_sweep, "v_measure")
    b_comp = float(b_best.get("completeness", float("nan"))) if not b_best.empty else float("nan")
    b_homo = float(b_best.get("homogeneity", float("nan"))) if not b_best.empty else float("nan")

    def _one(name: str, sdf: pd.DataFrame) -> dict[str, Any]:
        if sdf.empty:
            return {"method": name}
        br = best_sweep_metric_row(sdf, "v_measure")
        h_con = constrained_best_row(sdf, maximize="homogeneity", subject_to="completeness", subject_min=b_comp)
        c_hom = constrained_best_row(sdf, maximize="completeness", subject_to="homogeneity", subject_min=b_homo)
        out = {
            "method": name,
            "best_v_measure": float(br.get("v_measure", np.nan)),
            "best_homogeneity_at_best_v": float(br.get("homogeneity", np.nan)),
            "best_completeness_at_best_v": float(br.get("completeness", np.nan)),
            "min_edge_weight_at_best_v": float(br.get("min_edge_weight", np.nan)),
            "resolution_at_best_v": float(br.get("resolution", np.nan)),
            "n_edges_at_best_v": float(br.get("n_edges_after_threshold", np.nan)),
            "best_homogeneity_subject_completeness_ge_baseline_best": (
                float(h_con.get("homogeneity", np.nan)) if not h_con.empty else float("nan")
            ),
            "completeness_at_that_row": float(h_con.get("completeness", np.nan)) if not h_con.empty else float("nan"),
            "best_completeness_subject_homogeneity_ge_baseline_best": (
                float(c_hom.get("completeness", np.nan)) if not c_hom.empty else float("nan")
            ),
            "homogeneity_at_that_row": float(c_hom.get("homogeneity", np.nan)) if not c_hom.empty else float("nan"),
        }
        return out

    return pd.DataFrame(
        [
            _one(method_name_baseline, baseline_sweep),
            _one(method_name_teacher, teacher_sweep),
            _one(method_name_v2, v2_sweep),
        ]
    )


def top_n_sweep_results(sweep_df: pd.DataFrame, n: int = 25, by: str = "v_measure") -> pd.DataFrame:
    if sweep_df.empty:
        return sweep_df
    d = sweep_df.copy()
    d[by] = pd.to_numeric(d[by], errors="coerce")
    d = d[np.isfinite(d[by])]
    return d.sort_values(by, ascending=False).head(int(n)).reset_index(drop=True)


def nearest_edge_count_row(sweep_df: pd.DataFrame, target_n: float) -> pd.Series:
    if sweep_df.empty:
        return pd.Series(dtype=float)
    d = sweep_df.copy()
    d["n_edges_after_threshold"] = pd.to_numeric(d["n_edges_after_threshold"], errors="coerce")
    d = d[np.isfinite(d["n_edges_after_threshold"])]
    if d.empty:
        return pd.Series(dtype=float)
    d = d.assign(_dist=(d["n_edges_after_threshold"] - float(target_n)).abs())
    return d.sort_values("_dist", ascending=True).iloc[0]


def matched_edge_count_table(
    sweeps: dict[str, pd.DataFrame],
    target_edge_counts: list[float],
    *,
    pick_metric: str = "v_measure",
) -> pd.DataFrame:
    """
    For each method and target edge count, pick the sweep row whose graph size is closest.
    """
    rows = []
    for method, sdf in sweeps.items():
        for t in target_edge_counts:
            r = nearest_edge_count_row(sdf, t)
            if r.empty:
                rows.append({"method": method, "target_n_edges": t})
                continue
            rows.append(
                {
                    "method": method,
                    "target_n_edges": t,
                    "actual_n_edges": float(r.get("n_edges_after_threshold", np.nan)),
                    "min_edge_weight": float(r.get("min_edge_weight", np.nan)),
                    "resolution": float(r.get("resolution", np.nan)),
                    "v_measure": float(r.get("v_measure", np.nan)),
                    "homogeneity": float(r.get("homogeneity", np.nan)),
                    "completeness": float(r.get("completeness", np.nan)),
                    "n_communities": float(r.get("n_communities", np.nan)),
                    pick_metric: float(r.get(pick_metric, np.nan)),
                }
            )
    return pd.DataFrame(rows)


def load_v2_training_history(run_dir: str | Path) -> pd.DataFrame:
    p = Path(run_dir).expanduser().resolve() / "training_history.json"
    if not p.is_file():
        return pd.DataFrame()
    rows = json.loads(p.read_text(encoding="utf-8"))
    return pd.DataFrame(rows)


def v2_gt_same_cross_hsli_masks(
    baseline_edges: pd.DataFrame,
    assignments_df: pd.DataFrame,
    gt_label_map: dict[str, Any],
    v2_run_dir: str | Path,
    *,
    gt_min_dominant_fraction: float = 0.7,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Boolean masks over ``baseline_edges`` rows for same-campaign, cross-campaign (GT taxonomy),
    and the HS-LI view regime (semantic/infra thresholds from ``ranking_supervision_meta.json``).
    """
    run_dir = Path(v2_run_dir).expanduser().resolve()
    meta_th = load_v2_ranking_supervision_meta(run_dir) or {}
    thm = meta_th.get("thresholds") or {}
    tsem_diag = float(thm.get("thr_semantic_high", 0.65))
    tinf_diag = float(thm.get("thr_infra_false_bridge_max", 0.4))
    views_chk = build_view_scores_df(baseline_edges)
    sem_chk = pd.to_numeric(views_chk["view_semantic"], errors="coerce").fillna(0.0).to_numpy(
        dtype=np.float64
    )
    inf_chk = pd.to_numeric(views_chk["view_infra"], errors="coerce").fillna(0.0).to_numpy(
        dtype=np.float64
    )
    e_tax = attach_edge_taxonomy(
        baseline_edges.reset_index(drop=True),
        assignments_df,
        gt_label_map,
        min_dominant_fraction=gt_min_dominant_fraction,
    )
    return build_same_cross_hsli_masks(e_tax, sem_chk, inf_chk, tsem_diag, tinf_diag)


def _prefix_quantile_stats(prefix: str, arr: np.ndarray) -> dict[str, float]:
    x = arr[np.isfinite(arr)]
    row: dict[str, float] = {}
    if x.size == 0:
        row[f"{prefix}_mean"] = float("nan")
        row[f"{prefix}_median"] = float("nan")
        for q in (10, 25, 50, 75, 90):
            row[f"{prefix}_p{q}"] = float("nan")
        return row
    row[f"{prefix}_mean"] = float(np.mean(x))
    row[f"{prefix}_median"] = float(np.median(x))
    for q in (10, 25, 50, 75, 90):
        row[f"{prefix}_p{q}"] = float(np.quantile(x, q / 100.0))
    return row


def evaluate_v2_checkpoint_same_cross_score_stats(
    *,
    baseline_edges: pd.DataFrame,
    nodes_df: pd.DataFrame,
    v2_run_dir: str | Path,
    assignments_df: pd.DataFrame,
    gt_label_map: dict[str, Any],
    device: str = "cpu",
    epochs_filter: set[int] | None = None,
    gt_min_dominant_fraction: float = 0.7,
) -> pd.DataFrame:
    """
    For each ``checkpoints/epoch_XXXX.pt``, score all edges and summarize **same** vs **cross**
    campaign (GT taxonomy) score distributions, including the **HS-LI** subset.

    No Step 3 — cheap relative to ``evaluate_v2_checkpoint_epochs``. Intended for notebooks /
    separation diagnostics.
    """
    run_dir = Path(v2_run_dir).expanduser().resolve()
    cfg = load_v2_training_config(run_dir)
    cps = list_epoch_checkpoints(run_dir)
    if epochs_filter is not None:
        cps = [(e, p) for e, p in cps if e in epochs_filter]
    if not cps:
        return pd.DataFrame()

    same_m, cross_m, hsli_m = v2_gt_same_cross_hsli_masks(
        baseline_edges,
        assignments_df,
        gt_label_map,
        run_dir,
        gt_min_dominant_fraction=gt_min_dominant_fraction,
    )

    rows: list[dict[str, Any]] = []
    for ep, ck in cps:
        scores = score_edges_v2_checkpoint(
            baseline_edges,
            nodes_df,
            ck,
            run_dir=run_dir,
            cfg=cfg,
            device=device,
        )
        ms = scores[same_m]
        mc = scores[cross_m]
        row: dict[str, Any] = {
            "epoch": int(ep),
            "checkpoint": str(ck),
            "n_same": int(same_m.sum()),
            "n_cross": int(cross_m.sum()),
        }
        row.update(_prefix_quantile_stats("same", ms))
        row.update(_prefix_quantile_stats("cross", mc))
        row["mean_gap_same_minus_cross"] = (
            float(np.mean(ms) - np.mean(mc)) if ms.size and mc.size else float("nan")
        )
        row["median_gap_same_minus_cross"] = (
            float(np.median(ms) - np.median(mc)) if ms.size and mc.size else float("nan")
        )

        h_s = scores[same_m & hsli_m]
        h_c = scores[cross_m & hsli_m]
        row["n_hsli_same"] = int((same_m & hsli_m).sum())
        row["n_hsli_cross"] = int((cross_m & hsli_m).sum())
        row.update(_prefix_quantile_stats("hsli_same", h_s))
        row.update(_prefix_quantile_stats("hsli_cross", h_c))
        row["hsli_mean_gap_same_minus_cross"] = (
            float(np.mean(h_s) - np.mean(h_c)) if h_s.size and h_c.size else float("nan")
        )
        row["hsli_median_gap_same_minus_cross"] = (
            float(np.median(h_s) - np.median(h_c)) if h_s.size and h_c.size else float("nan")
        )
        rows.append(row)
    return pd.DataFrame(rows)


def score_v2_at_epoch_checkpoint(
    baseline_edges: pd.DataFrame,
    nodes_df: pd.DataFrame,
    v2_run_dir: str | Path,
    epoch_1based: int,
    *,
    device: str = "cpu",
) -> np.ndarray:
    """Full-graph scores from ``checkpoints/epoch_XXXX.pt`` for the given 1-based epoch index."""
    run_dir = Path(v2_run_dir).expanduser().resolve()
    cps = {e: p for e, p in list_epoch_checkpoints(run_dir)}
    if epoch_1based not in cps:
        raise FileNotFoundError(f"No epoch_{epoch_1based:04d}.pt under {run_dir / 'checkpoints'}")
    return score_edges_v2_checkpoint(
        baseline_edges,
        nodes_df,
        cps[epoch_1based],
        run_dir=run_dir,
        device=device,
    )


def evaluate_v2_checkpoint_epochs(
    *,
    baseline_edges: pd.DataFrame,
    nodes_df: pd.DataFrame,
    v2_run_dir: str | Path,
    assignments_df: pd.DataFrame,
    shard_ids: list[str],
    gt_label_map: dict[str, Any],
    method: str,
    resolution_values: list[float],
    min_edge_weight_values: list[float],
    seed: int = 0,
    device: str = "cpu",
    epochs_filter: set[int] | None = None,
    gt_score_separation_assignments_df: pd.DataFrame | None = None,
    gt_score_separation_label_map: dict[str, Any] | None = None,
    gt_min_dominant_fraction: float = 0.7,
) -> pd.DataFrame:
    """
    For each ``checkpoints/epoch_XXXX.pt``, score all edges and run the same Step 3 sweep.

    Requires training with ``save_every_epoch_checkpoint=True`` (or manually saved epoch checkpoints).

    Optional **same-vs-cross score separation** (diagnostics only): pass
    ``gt_score_separation_assignments_df`` and ``gt_score_separation_label_map`` (same schema as
    training GT diagnostics). Adds ``gt_mean_gap_same_minus_cross`` and
    ``gt_hsli_mean_gap_same_minus_cross`` columns when set.
    """
    run_dir = Path(v2_run_dir).expanduser().resolve()
    cfg = load_v2_training_config(run_dir)
    cps = list_epoch_checkpoints(run_dir)
    if epochs_filter is not None:
        cps = [(e, p) for e, p in cps if e in epochs_filter]

    gt_assign = gt_score_separation_assignments_df
    gt_lmap = gt_score_separation_label_map or gt_label_map
    sep_same = sep_cross = sep_hsli = None
    if gt_assign is not None and gt_lmap is not None:
        sep_same, sep_cross, sep_hsli = v2_gt_same_cross_hsli_masks(
            baseline_edges,
            gt_assign,
            gt_lmap,
            run_dir,
            gt_min_dominant_fraction=gt_min_dominant_fraction,
        )

    teacher_full = build_teacher_scored_edges(baseline_edges)
    te = teacher_full[TEACHER_WEIGHT_COL].to_numpy(dtype=np.float64)
    bw = pd.to_numeric(baseline_edges["edge_weight"], errors="coerce").to_numpy(dtype=np.float64)

    rows: list[dict[str, Any]] = []
    for ep, ck in cps:
        scores = score_edges_v2_checkpoint(
            baseline_edges,
            nodes_df,
            ck,
            run_dir=run_dir,
            cfg=cfg,
            device=device,
        )
        e_scored = baseline_edges.copy()
        e_scored["edge_plausibility"] = scores
        sweep = run_step3_sweep_df(
            assignments_df=assignments_df,
            shard_ids=shard_ids,
            edges_df=e_scored,
            gt_label_map=gt_label_map,
            method=method,
            resolution_values=resolution_values,
            min_edge_weight_values=min_edge_weight_values,
            weight_col="edge_plausibility",
            seed=seed,
        )
        best = best_sweep_metric_row(sweep, "v_measure")
        row_ck = {
            "epoch": int(ep),
            "checkpoint": str(ck),
            "best_v_measure": float(best.get("v_measure", np.nan)) if not best.empty else float("nan"),
            "best_homogeneity": float(best.get("homogeneity", np.nan)) if not best.empty else float("nan"),
            "best_completeness": float(best.get("completeness", np.nan)) if not best.empty else float("nan"),
            "min_edge_weight_at_best": float(best.get("min_edge_weight", np.nan)) if not best.empty else float("nan"),
            "resolution_at_best": float(best.get("resolution", np.nan)) if not best.empty else float("nan"),
            "n_edges_at_best": float(best.get("n_edges_after_threshold", np.nan)) if not best.empty else float("nan"),
            "score_mean": float(np.mean(scores)),
            "score_std": float(np.std(scores)),
            "score_q10": float(np.quantile(scores, 0.1)),
            "score_q90": float(np.quantile(scores, 0.9)),
            "pearson_vs_teacher": pearson_r(scores, te),
            "spearman_vs_teacher": spearman_r(scores, te),
            "pearson_vs_baseline_weight": pearson_r(scores, bw),
            "spearman_vs_baseline_weight": spearman_r(scores, bw),
        }
        if sep_same is not None:
            cg_ck = compact_gaps_from_scores(scores, sep_same, sep_cross, sep_hsli)  # type: ignore[arg-type]
            row_ck["gt_mean_gap_same_minus_cross"] = float(cg_ck["all_labeled_mean_gap_same_minus_cross"])
            row_ck["gt_hsli_mean_gap_same_minus_cross"] = float(cg_ck["hsli_mean_gap_same_minus_cross"])
        rows.append(row_ck)
    return pd.DataFrame(rows)


def load_v2_ranking_supervision_meta(run_dir: str | Path) -> dict[str, Any] | None:
    """Load ``ranking_supervision_meta.json`` from a V2 run directory, if present."""
    p = Path(run_dir).expanduser().resolve() / "ranking_supervision_meta.json"
    if not p.is_file():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def v2_ranking_supervision_fallback_per_epoch_df(history: pd.DataFrame) -> pd.DataFrame:
    """
    Per-epoch batch-mode counts from ``training_history.json`` (train batches only).

    Adds ``n_pair_batches``, fraction columns, and any missing pair-mode columns as 0.
    """
    if history.empty:
        return pd.DataFrame()
    h = history.copy()
    keys = [
        "pair_batches_buckets",
        "pair_batches_fallback_teacher",
        "pair_batches_fallback_partial",
        "pair_batches_legacy_teacher",
    ]
    for k in keys:
        if k not in h.columns:
            h[k] = 0.0
    h[keys] = h[keys].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    h["n_pair_batches"] = h[keys].sum(axis=1)
    tot = h["n_pair_batches"].replace(0.0, np.nan)
    h["frac_batches_bucket"] = h["pair_batches_buckets"] / tot
    h["frac_batches_fallback_teacher"] = h["pair_batches_fallback_teacher"] / tot
    h["frac_batches_fallback_partial"] = h["pair_batches_fallback_partial"] / tot
    h["frac_batches_legacy_teacher"] = h["pair_batches_legacy_teacher"] / tot
    h["frac_batches_any_teacher_fallback"] = (
        h["pair_batches_fallback_teacher"] + h["pair_batches_fallback_partial"] + h["pair_batches_legacy_teacher"]
    ) / tot
    return h


def v2_ranking_supervision_fallback_run_summary(history: pd.DataFrame) -> dict[str, Any]:
    """Aggregate pair-supervision batch counts over all epochs in ``history``."""
    fe = v2_ranking_supervision_fallback_per_epoch_df(history)
    if fe.empty:
        return {
            "n_epochs": 0,
            "total_bucket_batches": 0,
            "total_fallback_teacher_batches": 0,
            "total_fallback_partial_batches": 0,
            "total_legacy_teacher_batches": 0,
            "total_pair_batches": 0,
            "frac_all_batches_bucket": float("nan"),
            "frac_all_batches_any_teacher_fallback": float("nan"),
        }
    tb = float(fe["pair_batches_buckets"].sum())
    tft = float(fe["pair_batches_fallback_teacher"].sum())
    tfp = float(fe["pair_batches_fallback_partial"].sum())
    tlg = float(fe["pair_batches_legacy_teacher"].sum())
    tall = tb + tft + tfp + tlg
    return {
        "n_epochs": int(len(fe)),
        "total_bucket_batches": int(tb),
        "total_fallback_teacher_batches": int(tft),
        "total_fallback_partial_batches": int(tfp),
        "total_legacy_teacher_batches": int(tlg),
        "total_pair_batches": int(tall),
        "frac_all_batches_bucket": float(tb / tall) if tall > 0 else float("nan"),
        "frac_all_batches_any_teacher_fallback": float((tft + tfp + tlg) / tall) if tall > 0 else float("nan"),
    }


def v2_bucket_counts_table_from_meta(meta: dict[str, Any] | None) -> pd.DataFrame:
    if not meta or "bucket_counts_by_split" not in meta:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for split_name in ("full", "train", "val"):
        c = meta["bucket_counts_by_split"].get(split_name)
        if c is None:
            continue
        rows.append({"split": split_name, **dict(c)})
    return pd.DataFrame(rows)


def v2_positive_subpath_counts_table_from_meta(meta: dict[str, Any] | None) -> pd.DataFrame:
    if not meta or "positive_subpath_counts_by_split" not in meta:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for split_name in ("full", "train", "val"):
        c = meta["positive_subpath_counts_by_split"].get(split_name)
        if c is None:
            continue
        rows.append({"split": split_name, **dict(c)})
    return pd.DataFrame(rows)


def v2_hsli_regime_after_fit_table_from_meta(meta: dict[str, Any] | None) -> pd.DataFrame:
    """High-semantic / low-infra regime plausibility stats (post-training), by split."""
    if not meta or "high_sem_low_infra_regime_after_fit" not in meta:
        return pd.DataFrame()
    af = meta["high_sem_low_infra_regime_after_fit"]
    thr = af.get("thresholds_ref") or {}
    rows: list[dict[str, Any]] = []
    for split_name in ("full", "train", "val"):
        block = af.get(split_name)
        if block is None:
            continue
        rows.append(
            {
                "split": split_name,
                "thr_semantic_high": thr.get("thr_semantic_high"),
                "thr_infra_false_bridge_max": thr.get("thr_infra_false_bridge_max"),
                **dict(block),
            }
        )
    return pd.DataFrame(rows)


def v2_ranking_supervision_factual_note(
    meta: dict[str, Any] | None,
    run_summary: dict[str, Any],
) -> str:
    """Short interpretation from this run's artifacts only (no cross-run comparison)."""
    lines: list[str] = []
    f_any = run_summary.get("frac_all_batches_any_teacher_fallback")
    f_bkt = run_summary.get("frac_all_batches_bucket")
    if run_summary.get("total_pair_batches", 0) == 0:
        lines.append(
            "- **Fallback usage:** No pair-batch columns in ``training_history.json`` (older V2 run or empty history)."
        )
    elif f_any is None or (isinstance(f_any, float) and np.isnan(f_any)):
        lines.append("- **Fallback usage:** Could not compute fractions (zero pair batches).")
    elif f_any <= 0.02:
        lines.append(
            f"- **Fallback usage:** Teacher-style fallback batches are **rare** overall "
            f"(`frac_any_teacher_fallback` ≈ {f_any:.4f} over all train pair-batches)."
        )
    elif f_any <= 0.15:
        lines.append(
            f"- **Fallback usage:** Teacher-style fallback is **occasional** "
            f"(`frac_any_teacher_fallback` ≈ {f_any:.4f})."
        )
    else:
        lines.append(
            f"- **Fallback usage:** Teacher-style fallback is **frequent** "
            f"(`frac_any_teacher_fallback` ≈ {f_any:.4f}); check pool sizes and ``pair_batches_*`` breakdown."
        )
    if f_bkt is not None and not (isinstance(f_bkt, float) and np.isnan(f_bkt)):
        lines.append(
            f"- **Bucket-driven batches:** ≈ {f_bkt:.4f} of all train pair-batches used pure bucket sampling."
        )

    if not meta or "bucket_counts_by_split" not in meta:
        lines.append(
            "- **Train/val bucket health:** No ``ranking_supervision_meta.json`` (or missing split counts); re-train with current V2 code to record ``bucket_counts_by_split``."
        )
    else:
        bc = meta["bucket_counts_by_split"]
        tr = bc.get("train") or {}
        va = bc.get("val") or {}
        ntr = int(tr.get("n_edges_in_split", 0) or 0)
        nva = int(va.get("n_edges_in_split", 0) or 0)
        if ntr > 0:
            min_pool = min(
                int(tr.get("strong_pos", 0)),
                int(tr.get("strong_neg", 0)),
                int(tr.get("hard_neg", 0)),
            )
            hsli_tr = int(tr.get("hard_neg_hsli", 0))
            lines.append(
                f"- **Train split buckets:** n_edges={ntr}; min(strong_pos, strong_neg, hard_neg_union)={min_pool}; "
                f"hard_neg_hsli={hsli_tr}. Very small HS-LI pools push sampling toward other pair types or fallback."
            )
        if nva > 0:
            min_pool_v = min(
                int(va.get("strong_pos", 0)),
                int(va.get("strong_neg", 0)),
                int(va.get("hard_neg", 0)),
            )
            hsli_va = int(va.get("hard_neg_hsli", 0))
            lines.append(
                f"- **Val split buckets:** n_edges={nva}; min(strong_pos, strong_neg, hard_neg_union)={min_pool_v}; "
                f"hard_neg_hsli={hsli_va}."
            )

    sub = (meta or {}).get("positive_subpath_counts_by_split") or {}
    full_sp = (meta or {}).get("counts", {}).get("strong_pos")
    full_sub = sub.get("full") or {}
    b_pred = int(full_sub.get("backup_teacher_predicate_n", 0) or 0)
    b_in_sp = int(full_sub.get("backup_teacher_in_strong_pos", 0) or 0)
    sp_n = int(full_sp) if full_sp is not None else 0
    if sp_n > 0:
        lines.append(
            f"- **backup_teacher path (optional):** {b_pred} edges satisfy the backup predicate on the full graph; "
            f"{b_in_sp} land in **strong_pos** when backup is enabled. "
            f"Share of safe positives from backup ≈ {b_in_sp / max(sp_n, 1):.4f}."
        )
    elif meta:
        lines.append("- **backup_teacher path:** strong_pos count unavailable in meta.")

    st = (meta or {}).get("high_sem_low_infra_regime_static") or {}
    ne = int(st.get("n_edges", 0) or 0)
    if ne > 0:
        lines.append(
            f"- **HS–LI regime (bucket time):** n={ne} edges with high semantic + low infra (same infra cutoff as false-bridge core); "
            f"of these, {int(st.get('n_overlap_false_bridge_bucket', 0) or 0)} in false-bridge bucket, "
            f"{int(st.get('n_overlap_safe_pos_bucket', 0) or 0)} in safe bucket, "
            f"{int(st.get('n_regime_unassigned', 0) or 0)} unassigned."
        )
    hsli_fit = (meta or {}).get("high_sem_low_infra_regime_after_fit") or {}
    hf = hsli_fit.get("full") or {}
    if int(hf.get("n_edges", 0) or 0) > 0:
        m = float(hf.get("mean_edge_plausibility", float("nan")))
        lines.append(
            f"- **HS–LI regime (after fit, full graph):** n={int(hf['n_edges'])}; "
            f"mean edge_plausibility ≈ {m:.4f} (want this **low** if the model suppresses false bridges)."
        )

    return "\n".join(lines)


def save_v2_ranking_supervision_diagnostics_report(
    v2_run_dir: str | Path,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """
    Write ranking-supervision tables + JSON summary for a trained V2 run.

    Files (under ``output_dir``, default = ``v2_run_dir``):

    - ``v2_ranking_supervision_fallback_per_epoch.csv``
    - ``v2_ranking_supervision_bucket_counts_by_split.csv``
    - ``v2_ranking_supervision_positive_subpath_counts_by_split.csv``
    - ``v2_ranking_supervision_hsli_regime_after_fit.csv`` (V2.1 precision regime scores)
    - ``v2_ranking_supervision_report.json`` (run summary + factual note + meta snippets)
    """
    run_dir = Path(v2_run_dir).expanduser().resolve()
    out = Path(output_dir).expanduser().resolve() if output_dir is not None else run_dir
    out.mkdir(parents=True, exist_ok=True)

    hist = load_v2_training_history(run_dir)
    meta = load_v2_ranking_supervision_meta(run_dir)
    fe = v2_ranking_supervision_fallback_per_epoch_df(hist)
    summ = v2_ranking_supervision_fallback_run_summary(hist)
    note = v2_ranking_supervision_factual_note(meta, summ)
    btab = v2_bucket_counts_table_from_meta(meta)
    ptab = v2_positive_subpath_counts_table_from_meta(meta)
    hsli = v2_hsli_regime_after_fit_table_from_meta(meta)

    p_fe = out / "v2_ranking_supervision_fallback_per_epoch.csv"
    p_b = out / "v2_ranking_supervision_bucket_counts_by_split.csv"
    p_p = out / "v2_ranking_supervision_positive_subpath_counts_by_split.csv"
    p_hsli = out / "v2_ranking_supervision_hsli_regime_after_fit.csv"
    p_js = out / "v2_ranking_supervision_report.json"

    if not fe.empty:
        fe.to_csv(p_fe, index=False)
    if not btab.empty:
        btab.to_csv(p_b, index=False)
    if not ptab.empty:
        ptab.to_csv(p_p, index=False)
    if not hsli.empty:
        hsli.to_csv(p_hsli, index=False)

    payload: dict[str, Any] = {
        "v2_run_dir": str(run_dir),
        "fallback_run_summary": summ,
        "factual_note_markdown": note,
        "pools_train_sizes": (meta or {}).get("pools_train_sizes"),
        "pools_val_sizes": (meta or {}).get("pools_val_sizes"),
        "ranking_supervision_mode_config": (meta or {}).get("ranking_supervision_mode_config"),
    }
    if meta:
        payload["bucket_counts_by_split"] = meta.get("bucket_counts_by_split")
        payload["positive_subpath_counts_by_split"] = meta.get("positive_subpath_counts_by_split")
        for k in (
            "high_sem_low_infra_regime_static",
            "high_sem_low_infra_regime_after_fit",
            "precision_bucket_legend",
            "ranking_pair_legend",
            "mode",
        ):
            if k in meta:
                payload[k] = meta[k]
    p_js.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return {
        "paths": {
            "fallback_per_epoch_csv": str(p_fe) if not fe.empty else "",
            "bucket_counts_csv": str(p_b) if not btab.empty else "",
            "positive_subpath_csv": str(p_p) if not ptab.empty else "",
            "hsli_regime_after_fit_csv": str(p_hsli) if not hsli.empty else "",
            "report_json": str(p_js),
        },
        "fallback_per_epoch_df": fe,
        "bucket_counts_df": btab,
        "positive_subpath_df": ptab,
        "fallback_run_summary": summ,
        "factual_note": note,
    }


def sweep_long_for_plotting(
    sweeps: dict[str, pd.DataFrame],
    *,
    n_edges_col: str = "n_edges_after_threshold",
) -> pd.DataFrame:
    """Concatenate named sweeps with a ``method`` column for line/scatter plots."""
    parts = []
    for name, sdf in sweeps.items():
        if sdf.empty:
            continue
        x = sdf.copy()
        x["method"] = name
        parts.append(x)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)
