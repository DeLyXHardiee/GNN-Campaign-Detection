#!/usr/bin/env python3
"""Step 2: materialize final timestamp pair-training CSV (log1p gaps)."""
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
    load_manifest,
    pair_universe_stats,
    repo_root,
    resolve_repo_path,
    steps_dir,
)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, default=None)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    repo = repo_root()
    manifest = load_manifest(args.manifest)
    out_csv = resolve_repo_path(repo, str(manifest["final_pair_dataset_csv"]))
    summary_json = resolve_repo_path(repo, str(manifest["final_pair_materialize_summary_json"]))
    baseline_csv = resolve_repo_path(repo, str(manifest["baseline_pair_dataset_csv"]))

    if out_csv.is_file() and not args.force:
        print(f"[step02] exists: {out_csv}")
    else:
        cmd = [
            sys.executable,
            str(repo / "seed_candidate_workflow/scripts/materialize_timestamp_pair_training_dataset.py"),
            "--source-pair-csv",
            str(baseline_csv),
            "--output-pair-csv",
            str(out_csv),
        ]
        subprocess.run(cmd, cwd=str(repo), check=True)

    baseline_stats = pair_universe_stats(baseline_csv)
    final_stats = pair_universe_stats(out_csv)
    mat_summary = json.loads(summary_json.read_text(encoding="utf-8-sig")) if summary_json.is_file() else {}

    report = {
        "baseline_pair_universe": baseline_stats,
        "final_pair_universe": final_stats,
        "pair_counts_unchanged": baseline_stats.get("n_pairs") == final_stats.get("n_pairs"),
        "materialize_summary": mat_summary,
        "outputs": {
            "pair_csv": str(out_csv),
            "summary_json": str(summary_json),
        },
    }
    out_dir = steps_dir(repo, manifest)
    p_json = out_dir / "step02_materialize_report.json"
    p_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
