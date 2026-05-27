#!/usr/bin/env python3
"""
Materialize timestamp-enabled pair_training_dataset.csv for _14_only_mlp ablation.

Reuses the same pair universe as baseline _13 (row keys email_i, email_j unchanged).
Fills time_gap_seconds_min from MISP event dates and stores log1p(seconds) for MLP input
(raw seconds kept in time_gap_seconds_raw for diagnostics).

Does not modify baseline bundle CSVs.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def _pair_time_gap_seconds(catalog: dict[str, dict[str, str]], email_i: str, email_j: str) -> float | None:
    from graph.common import to_unix_ts

    ti = catalog.get(str(email_i), {}) or {}
    tj = catalog.get(str(email_j), {}) or {}
    ts_i = to_unix_ts(ti.get("date_raw") or ti.get("timestamp_utc"))
    ts_j = to_unix_ts(tj.get("date_raw") or tj.get("timestamp_utc"))
    if ts_i is None or ts_j is None:
        return None
    return float(abs(int(ts_i) - int(ts_j)))


def materialize(
    *,
    source_pair_csv: Path,
    output_pair_csv: Path,
    output_summary_json: Path | None = None,
    feature_mode: str = "log1p_seconds",
    misp_json_path: Path | None = None,
) -> dict[str, Any]:
    from seed_candidate_workflow.utils import graph_structure_helpers as gh
    from seed_candidate_workflow.utils.pair_score_separation import _resolve_default_misp_json_path

    if feature_mode not in ("log1p_seconds", "raw_seconds"):
        raise ValueError(f"feature_mode must be log1p_seconds or raw_seconds, got {feature_mode!r}")

    df = pd.read_csv(source_pair_csv, low_memory=False)
    root = gh.find_project_root()
    if misp_json_path is not None:
        misp_path = Path(misp_json_path)
        if not misp_path.is_absolute():
            misp_path = (root / misp_path).resolve()
        else:
            misp_path = misp_path.resolve()
    else:
        misp_path = _resolve_default_misp_json_path(root)
    if misp_path is None:
        raise FileNotFoundError(
            "Could not resolve misp_json_path. Provide --misp-json-path or ensure pipeline_config.json has datasets/graph/preprocessing.misp_json_path."
        )
    from analysis.scripts.misp_email_text_catalog import load_misp_timestamps_by_external_id

    catalog = load_misp_timestamps_by_external_id(Path(misp_path), project_root=root)
    catalog_meta = {
        "status": "ok",
        "loader": "load_misp_timestamps_by_external_id",
        "misp_json_path": str(Path(misp_path).resolve()),
        "n_emails_with_timestamp": int(len(catalog)),
    }

    raw_gaps: list[float | None] = []
    for _, r in df.iterrows():
        raw_gaps.append(_pair_time_gap_seconds(catalog, str(r["email_i"]), str(r["email_j"])))

    raw_series = pd.Series(raw_gaps, dtype="float64")
    n_misp = int(raw_series.notna().sum())
    n_missing = int(raw_series.isna().sum())

    out = df.copy()
    out["time_gap_seconds_raw"] = raw_series
    if feature_mode == "log1p_seconds":
        out["time_gap_seconds_min"] = raw_series.apply(
            lambda x: float(math.log1p(x)) if pd.notna(x) and x >= 0 else np.nan
        )
    else:
        out["time_gap_seconds_min"] = raw_series

    # Training path uses fillna(0) for missing numerics; document that explicitly.
    filled = pd.to_numeric(out["time_gap_seconds_min"], errors="coerce").fillna(0.0)

    output_pair_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_pair_csv, index=False)

    feat = filled[raw_series.notna()]
    summary: dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_pair_csv": str(source_pair_csv.resolve()),
        "output_pair_csv": str(output_pair_csv.resolve()),
        "feature_mode": feature_mode,
        "temporal_feature": {
            "mlp_column": "time_gap_seconds_min",
            "transform": "log1p(max(raw_gap_seconds, 0))" if feature_mode == "log1p_seconds" else "raw_seconds",
            "raw_gap_source": "MISP date_raw / timestamp_utc via load_misp_text_catalog_for_pairs",
            "missing_raw_gap_filled_as": 0.0,
        },
        "pair_universe_unchanged": True,
        "n_rows": int(len(out)),
        "n_raw_gaps_from_misp": n_misp,
        "n_raw_gaps_missing": n_missing,
        "raw_gap_seconds": {
            "min": float(raw_series.min()) if n_misp else None,
            "max": float(raw_series.max()) if n_misp else None,
            "p50": float(raw_series.median()) if n_misp else None,
            "p95": float(raw_series.quantile(0.95)) if n_misp else None,
        },
        "mlp_input_time_gap_seconds_min": {
            "nonzero_after_fillna": int((filled > 0).sum()),
            "min": float(feat.min()) if len(feat) else None,
            "max": float(feat.max()) if len(feat) else None,
            "p50": float(feat.median()) if len(feat) else None,
            "p95": float(feat.quantile(0.95)) if len(feat) else None,
        },
        "misp_catalog_meta": catalog_meta,
    }
    if output_summary_json is not None:
        output_summary_json.parent.mkdir(parents=True, exist_ok=True)
        output_summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        summary["summary_json"] = str(output_summary_json.resolve())
    return summary


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--source-pair-csv",
        type=Path,
        default=_REPO
        / "seed_candidate_workflow/output/graph_bundles/main_gnn_pu_1_no_ts_dedup_task_identity_13/pair_training/main_gnn_pu_1_no_ts_dedup_task_identity_13/pair_training_dataset.csv",
    )
    p.add_argument(
        "--output-pair-csv",
        type=Path,
        default=_REPO
        / "seed_candidate_workflow/output/graph_bundles/14_only_mlp__with_timestamp__timestamp_ablation/pair_training/14_only_mlp__with_timestamp__timestamp_ablation/pair_training_dataset.csv",
    )
    p.add_argument("--feature-mode", choices=("log1p_seconds", "raw_seconds"), default="log1p_seconds")
    p.add_argument(
        "--misp-json-path",
        type=Path,
        default=None,
        help="Override the MISP JSON path used for timestamp lookup (date_raw/timestamp_utc).",
    )
    args = p.parse_args()

    out_summary = args.output_pair_csv.parent / "pair_training_dataset_timestamp_summary.json"
    summary = materialize(
        source_pair_csv=args.source_pair_csv.resolve(),
        output_pair_csv=args.output_pair_csv.resolve(),
        output_summary_json=out_summary,
        feature_mode=str(args.feature_mode),
        misp_json_path=args.misp_json_path.resolve() if args.misp_json_path is not None else None,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
