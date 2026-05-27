#!/usr/bin/env python3
"""Run thesis non-dedup vs post-dedup seed/candidate pair generation diagnostic."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from seed_candidate_workflow.utils.final_14_only_mlp_thesis import load_manifest, repo_root, resolve_repo_path
from seed_candidate_workflow.utils.thesis_nondedup_pair_generation_diagnostic import run_full_diagnostic


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Thesis output dir (default: .../graph_construction_diagnostics/nondedup_pair_generation/)",
    )
    p.add_argument("--post-dedup-pair-csv", type=Path, default=None, help="Reported post-dedup pair_training_dataset.csv")
    p.add_argument("--manifest", type=Path, default=None, help="final_14_only_mlp manifest for baseline pair CSV")
    p.add_argument(
        "--analyze-only",
        action="store_true",
        help="Skip generation; analyze existing diagnostic bundle pair CSV only",
    )
    p.add_argument("--force-hetero", action="store_true", help="Rebuild diagnostic hetero graph")
    args = p.parse_args()

    post_csv = args.post_dedup_pair_csv
    if post_csv is None and args.manifest:
        manifest = load_manifest(args.manifest)
        post_csv = resolve_repo_path(repo_root(), str(manifest.get("baseline_pair_dataset_csv") or ""))

    report = run_full_diagnostic(
        thesis_out_dir=args.out_dir,
        post_dedup_pair_csv=post_csv,
        run_generation=not args.analyze_only,
        force_hetero=args.force_hetero,
    )
    print(json.dumps(report.get("artifact_paths") or report, indent=2))
    out = args.out_dir
    if out is None:
        from seed_candidate_workflow.utils.thesis_nondedup_pair_generation_diagnostic import DiagnosticPaths

        out = DiagnosticPaths.resolve().thesis_out_dir
    print(f"\nWrote nondedup pair diagnostics to:\n  {out.resolve()}")


if __name__ == "__main__":
    main()
