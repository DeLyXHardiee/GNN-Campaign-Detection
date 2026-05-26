#!/usr/bin/env python3
"""Step 4: scored community detection sweep (expanded GT)."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

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


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, default=None)
    p.add_argument("--skip-existing", action="store_true")
    args = p.parse_args()

    repo = repo_root()
    manifest = load_manifest(args.manifest)
    scoring_run_id = str(manifest["scoring_run_id"])
    gt_slug = str(manifest.get("gt_slug") or "ground_truth")
    sweep = community_sweep_csv(repo, scoring_run_id, gt_slug=gt_slug)

    if not (args.skip_existing and sweep.is_file()):
        exp = (repo / str(manifest["community_experiment_config"])).resolve()
        subprocess.run(
            [sys.executable, str(repo / "seed_candidate_workflow/pipelines/run_experiment.py"), "--config", str(exp)],
            cwd=str(repo),
            check=True,
        )

    best = best_community_row(sweep)
    report = {"scoring_run_id": scoring_run_id, "sweep_csv": str(sweep), "best_community": best}
    out_dir = steps_dir(repo, manifest)
    p_json = out_dir / "step04_community_report.json"
    p_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
