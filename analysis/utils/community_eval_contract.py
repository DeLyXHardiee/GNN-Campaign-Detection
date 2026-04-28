from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import completeness_score, homogeneity_score, v_measure_score


@dataclass(frozen=True)
class EvalMetrics:
    n_eval: float
    homogeneity: float
    completeness: float
    v_measure: float
    coverage_gt: float
    coverage_predictions: float


def evaluate_external_metrics(
    *,
    gt_label_map: dict[str, Any],
    pred_label_map: dict[str, int],
    n_predictions_total: int | None = None,
) -> dict[str, float]:
    gt = {str(k): v for k, v in gt_label_map.items()}
    pred = {str(k): int(v) for k, v in pred_label_map.items()}
    common = sorted(set(gt.keys()) & set(pred.keys()))
    n_pred = int(n_predictions_total if n_predictions_total is not None else len(pred))
    if not common:
        return {
            "n_eval": 0.0,
            "homogeneity": float("nan"),
            "completeness": float("nan"),
            "v_measure": float("nan"),
            "coverage_gt": 0.0,
            "coverage_predictions": 0.0,
        }
    y_true = [gt[e] for e in common]
    y_pred = [pred[e] for e in common]
    return {
        "n_eval": float(len(common)),
        "homogeneity": float(homogeneity_score(y_true, y_pred)),
        "completeness": float(completeness_score(y_true, y_pred)),
        "v_measure": float(v_measure_score(y_true, y_pred)),
        "coverage_gt": float(len(common) / max(1, len(gt))),
        "coverage_predictions": float(len(common) / max(1, n_pred)),
    }


def metric_sort_columns(primary: str) -> list[str]:
    p = str(primary).strip().lower().replace("-", "_")
    if p not in {"homogeneity", "completeness", "v_measure"}:
        p = "v_measure"
    all_m = ("homogeneity", "completeness", "v_measure")
    rest = [m for m in all_m if m != p]
    if p == "homogeneity":
        rest = ["v_measure", "completeness"]
    elif p == "completeness":
        rest = ["v_measure", "homogeneity"]
    else:
        rest = ["completeness", "homogeneity"]
    return [p, *rest]


def best_sweep_metric_row(sweep_df: pd.DataFrame, metric: str = "v_measure") -> pd.Series:
    if sweep_df.empty:
        return pd.Series(dtype=float)
    d = sweep_df.copy()
    metric = str(metric).strip().lower().replace("-", "_")
    if metric not in d.columns:
        metric = "v_measure"
    d = d[np.isfinite(pd.to_numeric(d[metric], errors="coerce"))]
    if d.empty:
        return pd.Series(dtype=float)
    tie_cols = [c for c in metric_sort_columns(metric) if c in d.columns]
    asc = [False] * len(tie_cols)
    return d.sort_values(tie_cols, ascending=asc).iloc[0]

