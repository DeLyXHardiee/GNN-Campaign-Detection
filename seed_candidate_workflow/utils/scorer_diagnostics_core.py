from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from seed_candidate_workflow.utils.scorer_diagnostics_contract import DiagnosticsResult


def quantiles_dict(x: np.ndarray, qs: tuple[float, ...]) -> dict[str, float]:
    if x.size == 0:
        return {f"q{int(q * 100)}": float("nan") for q in qs}
    return {f"q{int(q * 100)}": float(np.quantile(x, q)) for q in qs}


def safe_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    try:
        from sklearn.metrics import roc_auc_score
    except ImportError:
        return None
    if y_true.size < 2 or len(np.unique(y_true)) < 2:
        return None
    return float(roc_auc_score(y_true, y_score))


def basic_score_diagnostics(
    *,
    score_mode: str,
    graph_kind: str,
    scored_df: pd.DataFrame,
    score_col: str = "edge_weight",
) -> DiagnosticsResult:
    s = pd.to_numeric(scored_df.get(score_col), errors="coerce")
    finite = s[s.notna()]
    output_stats: dict[str, Any] = {
        "rows_total": int(len(scored_df)),
        "rows_with_finite_score": int(finite.shape[0]),
    }
    if not finite.empty:
        output_stats["score_summary"] = {
            "mean": float(finite.mean()),
            "median": float(finite.median()),
            "q10": float(finite.quantile(0.10)),
            "q90": float(finite.quantile(0.90)),
            "min": float(finite.min()),
            "max": float(finite.max()),
        }
    else:
        output_stats["score_summary"] = {
            "mean": None,
            "median": None,
            "q10": None,
            "q90": None,
            "min": None,
            "max": None,
        }

    prov_stats: dict[str, Any] = {}
    for col in ("from_seed", "from_semantic", "from_rare_artifact", "from_component", "from_2hop"):
        if col in scored_df.columns:
            v = scored_df[col].fillna(False).astype(bool)
            prov_stats[col] = {"count": int(v.sum()), "fraction": float(v.mean())}
    return DiagnosticsResult(
        scorer_name=score_mode,
        graph_kind=graph_kind,
        score_mode=score_mode,
        input_stats={"columns": list(scored_df.columns)},
        output_stats=output_stats,
        provenance_stats=prov_stats,
        scorer_specific={},
    )

