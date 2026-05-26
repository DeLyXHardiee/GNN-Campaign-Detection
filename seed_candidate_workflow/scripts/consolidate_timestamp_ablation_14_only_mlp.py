#!/usr/bin/env python3
"""Consolidate baseline vs timestamp-enabled _14_only_mlp ablation (community + training stability)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from seed_candidate_workflow.utils.timestamp_ablation_14_only_mlp import (  # noqa: E402
    community_sweep_csv,
    format_latex_comparison_table,
    load_manifest,
    pair_universe_stats,
    read_training_stability,
    scoring_run_dir,
    training_run_dir,
)


def _best_row(df: pd.DataFrame) -> pd.Series:
    d = df.copy()
    d["_v"] = pd.to_numeric(d["v_measure"], errors="coerce")
    return d.sort_values("_v", ascending=False).iloc[0]


def _community_best(repo: Path, scoring_run_id: str, gt_slug: str) -> dict[str, Any]:
    sweep_path = community_sweep_csv(repo, scoring_run_id, gt_slug=gt_slug)
    if not sweep_path.is_file():
        raise FileNotFoundError(f"Missing sweep: {sweep_path}")
    best = _best_row(pd.read_csv(sweep_path, low_memory=False))
    return {
        "scoring_run_id": scoring_run_id,
        "sweep_csv": str(sweep_path),
        "algorithm": str(best.get("method") or ""),
        "method": str(best.get("method") or ""),
        "threshold": float(best["min_edge_weight"]) if pd.notna(best.get("min_edge_weight")) else None,
        "resolution": float(best["resolution"]) if pd.notna(best.get("resolution")) else None,
        "homogeneity": float(best["homogeneity"]) if pd.notna(best.get("homogeneity")) else None,
        "completeness": float(best["completeness"]) if pd.notna(best.get("completeness")) else None,
        "v_measure": float(best["v_measure"]) if pd.notna(best.get("v_measure")) else None,
    }


def consolidate(*, manifest: dict[str, Any], repo: Path, out_dir: Path) -> dict[str, Any]:
    gt_slug = str(manifest.get("gt_slug") or "ground_truth")
    baseline = manifest["baseline"]
    ts = manifest["timestamp_enabled"]

    baseline_pairs = pair_universe_stats((repo / str(baseline["pair_dataset_csv"])).resolve())
    ts_pairs = pair_universe_stats((repo / str(ts["pair_dataset_csv"])).resolve())

    baseline_comm = _community_best(repo, str(baseline["scoring_run_id"]), gt_slug)
    ts_comm = _community_best(repo, str(ts["scoring_run_id"]), gt_slug)

    baseline_train = read_training_stability(training_run_dir(repo, str(baseline["run_id"])))
    ts_train = read_training_stability(training_run_dir(repo, str(ts["run_id"])))

    comparison_rows = [
        {
            "label": "no\\_timestamp (baseline)",
            "variant": "baseline",
            "run_id": str(baseline["run_id"]),
            **baseline_comm,
            **{f"train_{k}": v for k, v in baseline_train.items() if k.startswith("final_") or k == "stable_val_loss"},
        },
        {
            "label": "with\\_timestamp (ablation)",
            "variant": "timestamp_enabled",
            "run_id": str(ts["run_id"]),
            **ts_comm,
            **{f"train_{k}": v for k, v in ts_train.items() if k.startswith("final_") or k == "stable_val_loss"},
        },
    ]

    v_delta = None
    if baseline_comm.get("v_measure") is not None and ts_comm.get("v_measure") is not None:
        v_delta = float(ts_comm["v_measure"]) - float(baseline_comm["v_measure"])

    interpretation = (
        "Timestamp feature does not materially affect the learned pair scorer at community-detection level "
        "(|ΔV| < 0.01)."
        if v_delta is not None and abs(v_delta) < 0.01
        else (
            "Timestamp-enabled configuration changes community V-measure meaningfully; rerun full "
            "score-separation, threshold-stability, and prior-sensitivity analyses on the timestamp branch."
            if v_delta is not None
            else "Run community phase for timestamp branch before interpreting."
        )
    )

    report: dict[str, Any] = {
        "pair_universe_decision": manifest.get("pair_universe_decision"),
        "pair_universe_comparison": {
            "baseline": baseline_pairs,
            "timestamp_enabled": ts_pairs,
            "counts_match": baseline_pairs.get("n_pairs") == ts_pairs.get("n_pairs"),
        },
        "community_best": {
            "baseline": baseline_comm,
            "timestamp_enabled": ts_comm,
            "delta_v_measure": v_delta,
        },
        "training_stability": {
            "baseline": baseline_train,
            "timestamp_enabled": ts_train,
        },
        "interpretation": interpretation,
        "comparison_table_rows": comparison_rows,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    p_json = out_dir / "timestamp_ablation_14_only_mlp_comparison.json"
    p_csv = out_dir / "timestamp_ablation_14_only_mlp_comparison.csv"
    p_tex = out_dir / "timestamp_ablation_14_only_mlp_comparison.tex"

    pd.DataFrame(comparison_rows).to_csv(p_csv, index=False)
    p_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    p_tex.write_text(
        format_latex_comparison_table(
            [
                {
                    "label": "no timestamp",
                    "algorithm": baseline_comm["algorithm"],
                    "threshold": baseline_comm["threshold"],
                    "resolution": baseline_comm["resolution"],
                    "homogeneity": baseline_comm["homogeneity"],
                    "completeness": baseline_comm["completeness"],
                    "v_measure": baseline_comm["v_measure"],
                },
                {
                    "label": "with timestamp (log1p gap)",
                    "algorithm": ts_comm["algorithm"],
                    "threshold": ts_comm["threshold"],
                    "resolution": ts_comm["resolution"],
                    "homogeneity": ts_comm["homogeneity"],
                    "completeness": ts_comm["completeness"],
                    "v_measure": ts_comm["v_measure"],
                },
            ]
        ),
        encoding="utf-8",
    )

    manifest_out = {
        "outputs": {
            "comparison_json": str(p_json),
            "comparison_csv": str(p_csv),
            "latex_table": str(p_tex),
        },
        "scoring_run_dirs": {
            "baseline": str(scoring_run_dir(repo, str(baseline["scoring_run_id"]))),
            "timestamp_enabled": str(scoring_run_dir(repo, str(ts["scoring_run_id"]))),
        },
        "training_run_dirs": {
            "baseline": str(training_run_dir(repo, str(baseline["run_id"]))),
            "timestamp_enabled": str(training_run_dir(repo, str(ts["run_id"]))),
        },
    }
    p_manifest = out_dir / "timestamp_ablation_14_only_mlp_consolidation_manifest.json"
    p_manifest.write_text(json.dumps({**report, **manifest_out}, indent=2), encoding="utf-8")
    print(json.dumps({**manifest_out, "interpretation": interpretation}, indent=2))
    return report


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, default=None)
    args = p.parse_args()

    manifest = load_manifest(args.manifest)
    out_dir = args.out_dir
    if out_dir is None:
        out_dir = Path(
            str(manifest.get("consolidation_output_dir") or "seed_candidate_workflow/output/timestamp_ablation_14_only_mlp")
        )
    if not out_dir.is_absolute():
        out_dir = (_REPO / out_dir).resolve()

    consolidate(manifest=manifest, repo=_REPO, out_dir=out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
