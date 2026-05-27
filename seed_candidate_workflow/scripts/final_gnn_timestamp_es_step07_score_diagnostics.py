#!/usr/bin/env python3
"""Step 7: pair-score thesis diagnostics + KDE plots for both GNN thesis runs."""
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

from seed_candidate_workflow.utils.final_gnn_timestamp_es_thesis import (  # noqa: E402
    load_manifest,
    repo_root,
    resolve_repo_path,
    steps_dir,
    training_run_dir,
)


def _has_kde_plots(plots_dir: Path) -> bool:
    if not plots_dir.is_dir():
        return False
    return any(plots_dir.glob("score_density_kde_*.png"))


def _run_diagnostics_and_kde(
    *,
    repo: Path,
    run_id: str,
    graph_pt: Path,
    pair_csv: Path,
    gt_path: Path,
    out_diag: Path,
    skip_existing: bool = False,
) -> None:
    run_dir = training_run_dir(repo, run_id)
    out_diag.mkdir(parents=True, exist_ok=True)
    marker = out_diag / "thesis_pair_score_diagnostics.json"
    sep_out = run_dir / "pair_score_separation"
    sep_summary = sep_out / "pair_score_separation_summary.json"
    kde_dir = sep_out / "plots"

    if not (skip_existing and marker.is_file()):
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
    else:
        print(f"[step07] skip thesis diagnostics (exists): {marker}", flush=True)

    if not (skip_existing and sep_summary.is_file()):
        try:
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
        except subprocess.CalledProcessError as exc:
            print(
                f"[step07] WARNING: pair_score_separation_analysis failed for {run_id} "
                f"(exit {exc.returncode}); thesis AUROC/KDE still run.",
                flush=True,
            )
    else:
        print(f"[step07] skip separation analysis (exists): {sep_summary}", flush=True)

    if not (skip_existing and _has_kde_plots(kde_dir)):
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
                str(kde_dir),
            ],
            cwd=str(repo),
            check=True,
        )
    else:
        print(f"[step07] skip KDE plots (exist): {kde_dir}", flush=True)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, default=None)
    p.add_argument("--skip-existing", action="store_true")
    args = p.parse_args()

    repo = repo_root()
    manifest = load_manifest(args.manifest)
    graph_pt = resolve_repo_path(repo, str(manifest["graph_pt"]))
    pair_csv = resolve_repo_path(repo, str(manifest["final_pair_dataset_csv"]))
    gt_path = resolve_repo_path(repo, str(manifest["gt_path"]))

    reports: dict[str, str] = {}
    for key, run_id in (
        ("gnn_plus", str(manifest["run_id_gnn_plus"])),
        ("gnn_only", str(manifest["run_id_gnn_only"])),
    ):
        diag_out = training_run_dir(repo, run_id) / "pair_score_separation" / "thesis_score_diagnostics"
        marker = diag_out / "thesis_pair_score_diagnostics.json"
        _run_diagnostics_and_kde(
            repo=repo,
            run_id=run_id,
            graph_pt=graph_pt,
            pair_csv=pair_csv,
            gt_path=gt_path,
            out_diag=diag_out,
            skip_existing=bool(args.skip_existing),
        )
        reports[key] = str(marker)

    out = steps_dir(repo, manifest) / "step07_score_diagnostics_report.json"
    out.write_text(json.dumps(reports, indent=2), encoding="utf-8")
    print(json.dumps(reports, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
