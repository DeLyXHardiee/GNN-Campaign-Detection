#!/usr/bin/env python3
"""Run full thesis GNN pair-scoring pipeline (graph → train ×2 → community → diagnostics → consolidate)."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

STEPS = [
    ("01", "final_gnn_timestamp_es_step01_build_graph.py"),
    ("02", "final_gnn_timestamp_es_step02_verify_inputs.py"),
    ("03", "final_gnn_timestamp_es_step03_train_gnn_plus.py"),
    ("04", "final_gnn_timestamp_es_step04_train_gnn_only.py"),
    ("05", "final_gnn_timestamp_es_step05_community_gnn_plus.py"),
    ("06", "final_gnn_timestamp_es_step06_community_gnn_only.py"),
    ("07", "final_gnn_timestamp_es_step07_score_diagnostics.py"),
    ("08", "final_gnn_timestamp_es_step08_training_plots.py"),
    ("09", "final_gnn_timestamp_es_step09_consolidate.py"),
]

# Steps that implement --skip-existing (fast re-runs).
STEPS_WITH_SKIP_EXISTING = frozenset({"01", "02", "03", "04", "05", "06", "07", "08", "09"})


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, default=None)
    p.add_argument("--from-step", type=str, default="01", help="First step id, e.g. 03")
    p.add_argument("--to-step", type=str, default="09")
    p.add_argument("--skip-existing", action="store_true", help="Pass --skip-existing to each step where supported")
    p.add_argument("--skip-combined-kde", action="store_true")
    args = p.parse_args()

    manifest_arg = []
    if args.manifest:
        manifest_arg = ["--manifest", str(args.manifest)]

    started = False
    for step_id, script in STEPS:
        if not started:
            if step_id != args.from_step:
                continue
            started = True
        if step_id > args.to_step:
            break
        extra: list[str] = []
        if args.skip_existing and step_id in STEPS_WITH_SKIP_EXISTING:
            extra.append("--skip-existing")
        cmd = [sys.executable, str(_REPO / "seed_candidate_workflow/scripts" / script), *manifest_arg, *extra]
        if step_id == "09" and args.skip_combined_kde:
            cmd.append("--skip-combined-kde")
        print(f"\n=== Step {step_id}: {script} ===\n", flush=True)
        subprocess.run(cmd, cwd=str(_REPO), check=True)
    print("\nPipeline complete.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
