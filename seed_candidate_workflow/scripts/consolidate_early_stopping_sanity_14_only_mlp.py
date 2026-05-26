#!/usr/bin/env python3
"""Consolidate early-stopping sanity vs fixed 30-epoch baseline _14_only_mlp."""
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

from seed_candidate_workflow.utils.early_stopping_sanity_14_only_mlp import (  # noqa: E402
    community_sweep_csv,
    format_latex_comparison_table,
    load_manifest,
    read_early_stopping_training_metrics,
    scoring_run_dir,
    training_run_dir,
)


def _best_row(df: pd.DataFrame) -> pd.Series:
    d = df.copy()
    d["_v"] = pd.to_numeric(d["v_measure"], errors="coerce")
    return d.sort_values("_v", ascending=False).iloc[0]


def _community_from_sweep(repo: Path, scoring_run_id: str, gt_slug: str) -> dict[str, Any]:
    sweep_path = community_sweep_csv(repo, scoring_run_id, gt_slug=gt_slug)
    if not sweep_path.is_file():
        raise FileNotFoundError(
            f"Missing sweep: {sweep_path}\n"
            "Run: python seed_candidate_workflow/scripts/run_early_stopping_sanity_14_only_mlp.py --phase community"
        )
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
    gt_slug = str(manifest["shared"].get("gt_slug") or "ground_truth")
    baseline = manifest["baseline"]
    es = manifest["early_stopping"]
    v_thresh = float(manifest.get("meaningful_v_delta_threshold") or 0.01)

    baseline_comm = dict(manifest["baseline"].get("community_best") or {})
    if baseline_comm.get("v_measure") is None:
        baseline_comm = _community_from_sweep(repo, str(baseline["scoring_run_id"]), gt_slug)

    es_comm = _community_from_sweep(repo, str(es["scoring_run_id"]), gt_slug)

    baseline_train = read_early_stopping_training_metrics(
        training_run_dir(repo, str(baseline["run_id"])), target_epochs=30
    )
    es_train = read_early_stopping_training_metrics(
        training_run_dir(repo, str(es["run_id"])),
        target_epochs=int(es.get("epochs") or 100),
    )

    v_delta = None
    if baseline_comm.get("v_measure") is not None and es_comm.get("v_measure") is not None:
        v_delta = float(es_comm["v_measure"]) - float(baseline_comm["v_measure"])

    interpretation = (
        "Fixed 30-epoch budget does not materially affect community detection (|ΔV| < "
        f"{v_thresh:.2f}); no need to rerun score-separation, prior sensitivity, or timestamp ablation."
        if v_delta is not None and abs(v_delta) < v_thresh
        else (
            "Early-stopped training changes community V-measure meaningfully; consider rerunning "
            "downstream analyses with the early-stopped checkpoint."
            if v_delta is not None
            else "Complete train and community phases before interpreting."
        )
    )

    report: dict[str, Any] = {
        "training_schedule": {
            "baseline": {"epochs": 30, "early_stopping_patience": 7},
            "early_stopping": {
                "epochs": int(es.get("epochs") or 100),
                "early_stopping_patience": int(es.get("early_stopping_patience") or 10),
            },
        },
        "community_best": {"baseline": baseline_comm, "early_stopping": es_comm, "delta_v_measure": v_delta},
        "training_stability": {"baseline": baseline_train, "early_stopping": es_train},
        "interpretation": interpretation,
    }

    table_rows = [
        {
            "label": "fixed 30 epochs (baseline)",
            "run_id": str(baseline["run_id"]),
            **{k: baseline_comm.get(k) for k in ("algorithm", "threshold", "resolution", "homogeneity", "completeness", "v_measure")},
        },
        {
            "label": "early stop (max 100, pat 10)",
            "run_id": str(es["run_id"]),
            **{k: es_comm.get(k) for k in ("algorithm", "threshold", "resolution", "homogeneity", "completeness", "v_measure")},
        },
    ]

    out_dir.mkdir(parents=True, exist_ok=True)
    p_json = out_dir / "early_stopping_sanity_14_only_mlp_comparison.json"
    p_csv = out_dir / "early_stopping_sanity_14_only_mlp_comparison.csv"
    p_tex = out_dir / "early_stopping_sanity_14_only_mlp_comparison.tex"
    pd.DataFrame(table_rows).to_csv(p_csv, index=False)
    p_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    p_tex.write_text(format_latex_comparison_table(table_rows), encoding="utf-8")

    paths = {
        "training_run_dir": str(training_run_dir(repo, str(es["run_id"]))),
        "community_sweep_csv": str(es_comm["sweep_csv"]),
        "comparison_json": str(p_json),
        "comparison_csv": str(p_csv),
        "latex_table": str(p_tex),
    }
    print(json.dumps({**paths, "interpretation": interpretation}, indent=2))
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
            str(manifest.get("consolidation_output_dir") or "seed_candidate_workflow/output/early_stopping_sanity_14_only_mlp")
        )
    if not out_dir.is_absolute():
        out_dir = (_REPO / out_dir).resolve()

    consolidate(manifest=manifest, repo=_REPO, out_dir=out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
