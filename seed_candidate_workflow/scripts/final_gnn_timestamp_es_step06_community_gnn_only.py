#!/usr/bin/env python3
"""Step 6: PU scoring + community sweep for GNN-only pair scorer."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from seed_candidate_workflow.utils.final_gnn_timestamp_es_thesis import (  # noqa: E402
    community_sweep_in_run_dir,
    load_manifest,
    repo_root,
    resolve_best_community_from_sweep,
    resolve_repo_path,
    steps_dir,
    training_run_dir,
)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, default=None)
    p.add_argument("--skip-existing", action="store_true")
    args = p.parse_args()

    repo = repo_root()
    manifest = load_manifest(args.manifest)
    run_id = str(manifest["run_id_gnn_only"])
    run_dir = training_run_dir(repo, run_id)
    gt_slug = str(manifest.get("gt_slug") or "ground_truth")
    sweep = community_sweep_in_run_dir(run_dir, gt_slug=gt_slug)

    if not (args.skip_existing and sweep.is_file()):
        exp = resolve_repo_path(repo, str(manifest["community_experiment_gnn_only"]))
        subprocess.run(
            [sys.executable, str(repo / "seed_candidate_workflow/pipelines/run_experiment.py"), "--config", str(exp)],
            cwd=str(repo),
            check=True,
        )

    if not sweep.is_file():
        raise FileNotFoundError(f"Community sweep not found after experiment: {sweep}")

    best = resolve_best_community_from_sweep(sweep)
    report = {"run_id": run_id, "sweep_csv": str(sweep), "best_community": best}
    out = steps_dir(repo, manifest) / "step06_community_gnn_only_report.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
