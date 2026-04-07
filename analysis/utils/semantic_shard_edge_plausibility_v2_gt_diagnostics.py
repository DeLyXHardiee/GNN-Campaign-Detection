"""
GT-only diagnostics for V2 edge scores (same- vs cross-campaign separation).

Uses labels for evaluation/reporting only — never for training supervision.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from analysis.utils.semantic_shard_oracle_headroom_helpers import (
    EDGE_TAXONOMY_CROSS,
    EDGE_TAXONOMY_SAME,
    build_shard_gt_summary,
    label_candidate_edges_taxonomy,
)


def _qtable(x: np.ndarray, qs: tuple[float, ...]) -> dict[str, float]:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {f"p{int(q*100)}": float("nan") for q in qs}
    return {f"p{int(q*100)}": float(np.quantile(x, q)) for q in qs}


def _class_score_summary(scores: np.ndarray) -> dict[str, Any]:
    s = scores[np.isfinite(scores)]
    n = int(s.size)
    if n == 0:
        out: dict[str, Any] = {"n": 0}
        out.update({f"p{q}": float("nan") for q in (10, 25, 50, 75, 90)})
        out["mean"] = float("nan")
        out["median"] = float("nan")
        return out
    qt = _qtable(s, (0.1, 0.25, 0.5, 0.75, 0.9))
    return {
        "n": n,
        "mean": float(np.mean(s)),
        "median": float(np.median(s)),
        **qt,
    }


def _threshold_grid(scores_labeled: np.ndarray) -> list[float]:
    """Mixed quantile-based and fixed thresholds on the labeled same+cross pool."""
    s = scores_labeled[np.isfinite(scores_labeled)]
    if s.size == 0:
        return [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
    qs = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
    t = [float(np.quantile(s, q)) for q in qs]
    merged = sorted({round(x, 6) for x in t + [0.02, 0.5, 0.98]})
    return merged


def _frac_above(scores: np.ndarray, t: float) -> float:
    s = scores[np.isfinite(scores)]
    if s.size == 0:
        return float("nan")
    return float((s >= t).mean())


def attach_edge_taxonomy(
    edges_df: pd.DataFrame,
    assignments_df: pd.DataFrame,
    gt_label_map: dict[str, Any],
    *,
    min_dominant_fraction: float = 0.7,
) -> pd.DataFrame:
    """Return ``edges_df`` copy with ``edge_taxonomy`` column (aligned row order)."""
    e = edges_df.reset_index(drop=True)
    shard_summary = build_shard_gt_summary(assignments_df, gt_label_map)
    tax = label_candidate_edges_taxonomy(
        e,
        shard_summary,
        min_dominant_fraction=min_dominant_fraction,
    )
    out = e.merge(
        tax[["shard_a", "shard_b", "edge_taxonomy"]],
        on=["shard_a", "shard_b"],
        how="left",
    )
    return out


def build_same_cross_hsli_masks(
    edges_with_tax: pd.DataFrame,
    sem: np.ndarray,
    infv: np.ndarray,
    thr_sem_high: float,
    thr_inf_false_max: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Boolean masks (n_edges,) for same-campaign, cross-campaign (labeled only), and HS-LI regime.

    HS-LI is semantic/infra filter only; combine with same/cross in summaries.
    """
    n = len(edges_with_tax)
    if len(sem) != n or len(infv) != n:
        raise ValueError("semantic / infra arrays must match edge row count")
    tax = edges_with_tax["edge_taxonomy"].astype(str).to_numpy()
    same = tax == EDGE_TAXONOMY_SAME
    cross = tax == EDGE_TAXONOMY_CROSS
    finite = np.isfinite(sem) & np.isfinite(infv)
    hsli = finite & (sem >= thr_sem_high) & (infv <= thr_inf_false_max)
    return same, cross, hsli


def compact_gaps_from_scores(
    scores: np.ndarray,
    same: np.ndarray,
    cross: np.ndarray,
    hsli: np.ndarray,
) -> dict[str, Any]:
    """Mean score gaps for logging (same − cross), all labeled and HS-LI subset."""
    out: dict[str, Any] = {}
    ms = scores[same]
    mc = scores[cross]
    out["all_labeled_mean_gap_same_minus_cross"] = (
        float(np.mean(ms) - np.mean(mc)) if ms.size and mc.size else float("nan")
    )
    mss = scores[same & hsli]
    msc = scores[cross & hsli]
    out["hsli_mean_gap_same_minus_cross"] = (
        float(np.mean(mss) - np.mean(msc)) if mss.size and msc.size else float("nan")
    )
    out["n_same"] = int(same.sum())
    out["n_cross"] = int(cross.sum())
    out["n_hsli_same"] = int((same & hsli).sum())
    out["n_hsli_cross"] = int((cross & hsli).sum())
    return out


def full_same_cross_separation_report(
    scores: np.ndarray,
    same: np.ndarray,
    cross: np.ndarray,
    hsli: np.ndarray,
) -> dict[str, Any]:
    """Quantiles, gaps, and threshold tables for all labeled same/cross and HS-LI slices."""
    ms = scores[same]
    mc = scores[cross]
    pool = scores[same | cross]

    rep: dict[str, Any] = {"same": _class_score_summary(ms), "cross": _class_score_summary(mc)}
    rep["mean_gap_same_minus_cross"] = (
        rep["same"]["mean"] - rep["cross"]["mean"]
        if rep["same"]["n"] and rep["cross"]["n"]
        else float("nan")
    )
    rep["median_gap_same_minus_cross"] = (
        rep["same"]["median"] - rep["cross"]["median"]
        if rep["same"]["n"] and rep["cross"]["n"]
        else float("nan")
    )

    thresh = _threshold_grid(pool)
    rows = []
    for t in thresh:
        rows.append(
            {
                "threshold": t,
                "frac_same_above": _frac_above(ms, t),
                "frac_cross_above": _frac_above(mc, t),
            }
        )
    rep["threshold_table"] = rows

    mss = scores[same & hsli]
    msc = scores[cross & hsli]
    pool_h = scores[(same | cross) & hsli]
    pool_hsli = pool_h if pool_h.size else np.concatenate([mss, msc]) if (mss.size or msc.size) else np.array([])
    rep["hsli"] = {
        "same": _class_score_summary(mss),
        "cross": _class_score_summary(msc),
        "mean_gap_same_minus_cross": (
            float(np.mean(mss) - np.mean(msc)) if mss.size and msc.size else float("nan")
        ),
        "median_gap_same_minus_cross": (
            float(np.median(mss) - np.median(msc)) if mss.size and msc.size else float("nan")
        ),
        "threshold_table": (
            [
                {
                    "threshold": t,
                    "frac_same_above": _frac_above(mss, t),
                    "frac_cross_above": _frac_above(msc, t),
                }
                for t in _threshold_grid(pool_hsli if pool_hsli.size else np.array([0.0]))
            ]
            if mss.size or msc.size
            else []
        ),
    }
    return rep


def write_gt_separation_artifacts(
    out_dir: Path,
    report: dict[str, Any],
) -> dict[str, str]:
    """Write JSON + threshold CSVs nested under ``out_dir``."""
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    p_json = out_dir / "v2_gt_score_separation.json"
    p_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    paths: dict[str, str] = {"v2_gt_score_separation_json": str(p_json)}

    main_th = pd.DataFrame(report.get("threshold_table", []))
    if not main_th.empty:
        p_th = out_dir / "v2_gt_score_separation_thresholds.csv"
        main_th.to_csv(p_th, index=False)
        paths["v2_gt_score_separation_thresholds_csv"] = str(p_th)

    hsli_th = pd.DataFrame((report.get("hsli") or {}).get("threshold_table", []))
    if not hsli_th.empty:
        p_h = out_dir / "v2_gt_score_separation_hsli_thresholds.csv"
        hsli_th.to_csv(p_h, index=False)
        paths["v2_gt_score_separation_hsli_thresholds_csv"] = str(p_h)

    return paths
