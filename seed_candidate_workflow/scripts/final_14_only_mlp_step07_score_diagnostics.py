#!/usr/bin/env python3
"""Step 7: pair score separation + thesis statistics + KDE plots."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
for p in (_REPO, _REPO / "core", _REPO / "core" / "GNN"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from seed_candidate_workflow.utils.final_14_only_mlp_thesis import load_manifest, repo_root, resolve_repo_path, steps_dir, training_run_dir  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, default=None)
    p.add_argument("--skip-existing", action="store_true")
    args = p.parse_args()

    repo = repo_root()
    manifest = load_manifest(args.manifest)
    run_id = str(manifest["run_id"])
    run_dir = training_run_dir(repo, run_id)
    graph_pt = resolve_repo_path(repo, str(manifest["graph_pt"]))
    pair_csv = resolve_repo_path(repo, str(manifest["final_pair_dataset_csv"]))
    gt_path = resolve_repo_path(repo, str(manifest["gt_path"]))
    out_diag = run_dir / "pair_score_separation" / "thesis_score_diagnostics"

    # Thesis statistics + AUROC/AP
    subprocess.run(
        [
            sys.executable,
            str(repo / "seed_candidate_workflow/scripts/run_pair_score_thesis_diagnostics.py"),
            "--run-dir",
            str(run_dir),
            "--graph-pt",
            str(graph_pt),
            "--gt-path",
            str(gt_path),
            "--pair-csv",
            str(pair_csv),
            "--output-dir",
            str(out_diag),
        ],
        cwd=str(repo),
        check=True,
    )

    # Full score separation (optional rich outputs)
    sep_out = run_dir / "pair_score_separation"
    if not (args.skip_existing and (sep_out / "pair_score_separation_summary.json").is_file()):
        subprocess.run(
            [
                sys.executable,
                str(repo / "seed_candidate_workflow/scripts/run_pair_score_separation_analysis.py"),
                "--run-dir",
                str(run_dir),
                "--graph-pt",
                str(graph_pt),
                "--gt-path",
                str(gt_path),
                "--pair-csv",
                str(pair_csv),
            ],
            cwd=str(repo),
            check=True,
        )

    # KDE density plots
    kde_plots = sep_out / "plots"
    subprocess.run(
        [
            sys.executable,
            str(repo / "seed_candidate_workflow/scripts/run_pair_score_kde_density_plots.py"),
            "--run-dir",
            str(run_dir),
            "--graph-pt",
            str(graph_pt),
            "--gt-path",
            str(gt_path),
            "--pair-csv",
            str(pair_csv),
            "--output-dir",
            str(kde_plots),
        ],
        cwd=str(repo),
        check=True,
    )

    report = {
        "thesis_diagnostics_dir": str(out_diag),
        "score_separation_dir": str(sep_out),
        "kde_plots_dir": str(kde_plots),
    }
    out_dir = steps_dir(repo, manifest)
    p_json = out_dir / "step07_score_diagnostics_report.json"
    p_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
