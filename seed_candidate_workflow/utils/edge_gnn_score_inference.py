"""Load Edge-GNN pair scores from ``edge_gnn_pair_scores.csv`` for community evaluation."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from seed_candidate_workflow.utils.anchor_candidate_eval_helpers import _pair
from seed_candidate_workflow.utils import graph_structure_helpers as gh


def load_edge_gnn_pair_scores(
    score_csv_path: str | Path,
    *,
    project_root: Path | None = None,
    on_duplicate: str = "error",
) -> tuple[dict[tuple[str, str], float], dict[str, Any]]:
    """
    Read Edge-GNN scores and return canonical ``(email_i, email_j) -> pu_score``.

    ``on_duplicate``:
      - ``error``: raise if the same pair has conflicting finite scores
      - ``keep_first``: keep first row (warn if scores differ)
    """
    root = project_root or gh.find_project_root()
    p = Path(str(score_csv_path)).expanduser()
    if not p.is_absolute():
        p = (root / p).resolve()
    else:
        p = p.resolve()
    if not p.is_file():
        raise FileNotFoundError(f"edge_gnn_pair_scores.csv not found: {p}")

    df = pd.read_csv(p, low_memory=False)
    for col in ("email_i", "email_j", "pu_score"):
        if col not in df.columns:
            raise ValueError(f"edge_gnn_pair_scores.csv missing required column: {col}")

    score_map: dict[tuple[str, str], float] = {}
    num_invalid = 0
    num_duplicate_rows = 0
    num_conflicting_duplicates = 0
    for _, row in df.iterrows():
        pk = _pair(str(row["email_i"]), str(row["email_j"]))
        raw = pd.to_numeric(row["pu_score"], errors="coerce")
        if pd.isna(raw) or not np.isfinite(float(raw)):
            num_invalid += 1
            continue
        val = float(raw)
        if pk in score_map:
            num_duplicate_rows += 1
            prev = score_map[pk]
            if abs(prev - val) > 1e-9:
                num_conflicting_duplicates += 1
                if on_duplicate == "error":
                    raise ValueError(
                        f"Conflicting pu_score for pair {pk}: {prev} vs {val} in {p}"
                    )
                warnings.warn(
                    f"Duplicate pair {pk} with different scores ({prev} vs {val}); keeping first.",
                    stacklevel=2,
                )
            continue
        score_map[pk] = val

    diag: dict[str, Any] = {
        "score_csv_path": str(p),
        "num_rows_read": int(len(df)),
        "num_scores": int(len(score_map)),
        "num_invalid_scores": int(num_invalid),
        "num_duplicate_pairs": int(num_duplicate_rows),
        "num_conflicting_duplicate_scores": int(num_conflicting_duplicates),
    }
    return score_map, diag


def resolve_edge_gnn_pair_scores_csv(
    scoring_cfg: dict[str, Any],
    *,
    project_root: Path | None = None,
) -> Path:
    """Resolve ``pair_scores_csv`` from ``edge_gnn_run`` block or explicit path."""
    root = project_root or gh.find_project_root()
    scfg = dict(scoring_cfg or {})
    edge_run = dict(scfg.get("edge_gnn_run") or {})
    raw = str(scfg.get("pair_scores_csv") or edge_run.get("pair_scores_csv") or "").strip()
    if raw:
        p = Path(raw).expanduser()
        if not p.is_absolute():
            p = (root / p).resolve()
        return p.resolve()

    run_dir_raw = str(edge_run.get("run_dir") or scfg.get("run_dir") or "").strip()
    if not run_dir_raw:
        raise ValueError(
            "edge_gnn scoring requires pair_scores_csv or edge_gnn_run.run_dir "
            "(default: <run_dir>/edge_gnn_pair_scores.csv)"
        )
    run_dir = Path(run_dir_raw).expanduser()
    if not run_dir.is_absolute():
        run_dir = (root / run_dir).resolve()
    return (run_dir / "edge_gnn_pair_scores.csv").resolve()


def scores_array_for_pair_dataframe(
    score_csv_path: str | Path,
    df_work: pd.DataFrame,
    *,
    project_root: Path | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Map ``edge_gnn_pair_scores.csv`` onto ``df_work`` rows via canonical ``(email_i, email_j)``.

    Returns ``(scores, diagnostics)`` with ``scores[i]`` aligned to ``df_work`` row index ``i``.
    """
    score_map, diag = load_edge_gnn_pair_scores(score_csv_path, project_root=project_root)
    scores = np.full(len(df_work), np.nan, dtype=np.float64)
    n_hit = 0
    for i, row in df_work.iterrows():
        pk = _pair(str(row["email_i"]), str(row["email_j"]))
        if pk in score_map:
            scores[int(i)] = score_map[pk]
            n_hit += 1
    diag = {
        **diag,
        "num_df_rows": int(len(df_work)),
        "num_rows_matched": int(n_hit),
        "frac_rows_matched": float(n_hit / max(len(df_work), 1)),
    }
    return scores, diag
