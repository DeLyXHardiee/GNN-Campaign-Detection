"""
Teacher-only edge scoring (no learning): three-view agreement for Method 1 V2 diagnostics.

Produces a scored edge bundle compatible with Step 3 via ``weight_col="edge_teacher_agreement"``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from analysis.utils.semantic_shard_edge_plausibility_v2_views import (
    VIEW_COLS,
    build_view_scores_df,
    compute_agreement_scalar,
)

TEACHER_WEIGHT_COL = "edge_teacher_agreement"
DEFAULT_TEACHER_OUTPUT_ROOT = "analysis/output/semantic_shard_edge_teacher"


def build_teacher_scored_edges(edges_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add ``view_semantic``, ``view_infra``, ``view_temporal``, and ``edge_teacher_agreement``.

    Agreement is the geometric mean of the three views (same as V2 training teacher).
    """
    edges_df = edges_df.copy()
    edges_df["shard_a"] = edges_df["shard_a"].astype(str)
    edges_df["shard_b"] = edges_df["shard_b"].astype(str)
    views = build_view_scores_df(edges_df)
    agree = compute_agreement_scalar(views)
    out = edges_df.reset_index(drop=True)
    for c in VIEW_COLS:
        out[c] = views[c].to_numpy(dtype=float)
    out[TEACHER_WEIGHT_COL] = agree
    return out


def save_teacher_bundle(
    edges_df: pd.DataFrame,
    *,
    run_id: str,
    output_root: str | Path = DEFAULT_TEACHER_OUTPUT_ROOT,
    extra_meta: dict[str, Any] | None = None,
) -> dict[str, str]:
    """
    Write scored CSV + small JSON metadata under ``output_root / run_id /``.

    Primary edge file name matches V2 convention for drop-in Step 3 use.
    """
    root = Path(output_root).expanduser().resolve()
    out_dir = root / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    scored = build_teacher_scored_edges(edges_df)
    p_edges = out_dir / "semantic_shard_step2_edges_scored.csv"
    scored.to_csv(p_edges, index=False)

    meta = {
        "run_id": run_id,
        "weight_col": TEACHER_WEIGHT_COL,
        "view_columns": list(VIEW_COLS),
        "n_edges": int(len(scored)),
        "description": "Unsupervised three-view geometric-mean teacher; no ML.",
    }
    if extra_meta:
        meta["extra"] = extra_meta
    p_meta = out_dir / "teacher_bundle_meta.json"
    p_meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return {
        "output_dir": str(out_dir),
        "scored_edges_csv": str(p_edges),
        "meta_json": str(p_meta),
    }
