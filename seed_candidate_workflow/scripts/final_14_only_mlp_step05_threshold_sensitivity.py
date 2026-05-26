#!/usr/bin/env python3
"""Step 5: threshold sensitivity at best algorithm+resolution from step 4."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from seed_candidate_workflow.utils.final_14_only_mlp_thesis import (  # noqa: E402
    best_community_row,
    community_sweep_csv,
    load_manifest,
    repo_root,
    steps_dir,
)


def _build_thresh_exp(manifest: dict[str, Any], best: dict[str, Any]) -> dict[str, Any]:
    method = str(best.get("method") or best.get("algorithm") or "leiden").lower()
    if method not in ("louvain", "leiden"):
        method = "leiden"
    resolution = float(best["resolution"])
    scoring_run_id = str(manifest["threshold_stability_scoring_run_id"])
    return {
        "experiment": {
            "scoring_run_id": scoring_run_id,
            "graph_id": str(manifest["graph_id"]),
            "mode": "score_only",
            "graph_family": "seed_candidate",
        },
        "artifacts": {
            "graph_bundle_root": "seed_candidate_workflow/output/graph_bundles",
            "scoring_output_root": "seed_candidate_workflow/output/scoring_runs",
        },
        "selection": {
            "score_targets": ["seed_candidate"],
            "gt_set": "expanded_gt_only",
            "gt_sets_path": "seed_candidate_workflow/configs/experiments/gt_sets.json",
        },
        "scoring": {
            "score_mode": "seed_candidate_pu_v1",
            "params": {
                "pu": {
                    "pu_run": {
                        "run_dir": f"output/runs/{manifest['run_id']}",
                        "graph_pt": str(manifest["graph_pt"]),
                        "checkpoint": "best_model.pt",
                        "pair_dataset_csv": str(manifest["final_pair_dataset_csv"]),
                        "device": "cpu",
                        "no_to_undirected": False,
                    },
                    "seed_edge_weight": 1.0,
                    "weight_mode": "raw_score",
                    "export_non_seed_min_pu_score": 0.0,
                }
            },
        },
        "community": {
            "base_config": "seed_candidate_workflow/configs/anchor_community.default.json",
            "dedup_collapse_out_dir": "data/misp/misp_lake_dedup_task_identity",
            "sweep": {
                "methods": [method],
                "weight_thresholds": [round(x, 1) for x in [i / 10.0 for i in range(0, 10)]],
                "resolutions": [resolution],
                "use_edge_weights_in_partitioning": True,
                "sort_by": "v-measure",
            },
        },
        "_meta": {
            "fixed_method": method,
            "fixed_resolution": resolution,
            "source_best_from_step04": best,
        },
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, default=None)
    p.add_argument("--skip-existing", action="store_true")
    args = p.parse_args()

    repo = repo_root()
    manifest = load_manifest(args.manifest)
    gt_slug = str(manifest.get("gt_slug") or "ground_truth")

    step04_path = steps_dir(repo, manifest) / "step04_community_report.json"
    if step04_path.is_file():
        best = json.loads(step04_path.read_text(encoding="utf-8-sig"))["best_community"]
    else:
        sweep_main = community_sweep_csv(repo, str(manifest["scoring_run_id"]), gt_slug=gt_slug)
        best = best_community_row(sweep_main)

    exp_dict = _build_thresh_exp(manifest, best)
    out_dir = steps_dir(repo, manifest)
    exp_path = out_dir / "step05_threshold_sensitivity.experiment.json"
    exp_path.write_text(json.dumps(exp_dict, indent=2), encoding="utf-8")

    scoring_run_id = str(manifest["threshold_stability_scoring_run_id"])
    sweep_out = community_sweep_csv(repo, scoring_run_id, gt_slug=gt_slug)

    if not (args.skip_existing and sweep_out.is_file()):
        subprocess.run(
            [
                sys.executable,
                str(repo / "seed_candidate_workflow/pipelines/run_experiment.py"),
                "--config",
                str(exp_path),
            ],
            cwd=str(repo),
            check=True,
        )

    df_path = sweep_out
    import pandas as pd

    df = pd.read_csv(df_path, low_memory=False)
    report: dict[str, Any] = {
        "fixed_method": exp_dict["_meta"]["fixed_method"],
        "fixed_resolution": exp_dict["_meta"]["fixed_resolution"],
        "sweep_csv": str(df_path),
        "rows": df.to_dict(orient="records"),
    }
    p_json = out_dir / "step05_threshold_sensitivity_report.json"
    p_csv = out_dir / "step05_threshold_sensitivity_table.csv"
    df.to_csv(p_csv, index=False)
    p_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"sweep_csv": str(df_path), "report_json": str(p_json), "table_csv": str(p_csv)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
