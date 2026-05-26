#!/usr/bin/env python3
"""Step 10: consolidate thesis-facing outputs into one folder."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from seed_candidate_workflow.utils.final_14_only_mlp_thesis import (  # noqa: E402
    best_community_row,
    community_sweep_csv,
    copy_if_exists,
    format_latex_community_table,
    load_manifest,
    read_training_stability,
    repo_root,
    steps_dir,
    thesis_dir,
    training_run_dir,
)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, default=None)
    args = p.parse_args()

    repo = repo_root()
    manifest = load_manifest(args.manifest)
    tdir = thesis_dir(repo, manifest)
    sdir = steps_dir(repo, manifest)
    run_dir = training_run_dir(repo, str(manifest["run_id"]))
    gt_slug = str(manifest.get("gt_slug") or "ground_truth")

    sweep_main = community_sweep_csv(repo, str(manifest["scoring_run_id"]), gt_slug=gt_slug)
    best_final = best_community_row(sweep_main)
    baseline_best = dict(manifest.get("baseline_community_best") or {})
    v_delta = float(best_final["v_measure"]) - float(baseline_best.get("v_measure", 0))
    thresh = float(manifest.get("meaningful_v_delta_threshold") or 0.01)

    stability = read_training_stability(run_dir, target_epochs=int((manifest.get("training") or {}).get("epochs") or 100))

    interpretation = (
        f"Final canonical run differs materially from legacy no-timestamp 30-epoch baseline (|ΔV|={abs(v_delta):.4f} ≥ {thresh})."
        if abs(v_delta) >= thresh
        else (
            f"Final canonical run is aligned with legacy baseline at community level (|ΔV|={abs(v_delta):.4f} < {thresh})."
        )
    )

    copies: dict[str, str] = {}

    def _cp(src: Path, rel: str) -> None:
        dst = tdir / rel
        if copy_if_exists(src, dst):
            copies[rel] = str(dst)

    _cp(sweep_main, "community/final_community_sweep.csv")
    _cp(sdir / "step05_threshold_sensitivity_table.csv", "threshold_sensitivity/threshold_sensitivity_table.csv")
    _cp(sdir / "step05_threshold_sensitivity_report.json", "threshold_sensitivity/threshold_sensitivity_report.json")
    _cp(sdir / "step09_epoch_community_diagnostic.csv", "epoch_diagnostic/epoch_community_diagnostic.csv")
    _cp(sdir / "step09_epoch_community_diagnostic.json", "epoch_diagnostic/epoch_community_diagnostic.json")

    ps_dir = repo / str((manifest.get("prior_sensitivity") or {}).get("consolidation_output_dir") or "")
    _cp(ps_dir / "prior_sensitivity_best_by_pi.csv", "prior_sensitivity/prior_sensitivity_best_by_pi.csv")
    _cp(ps_dir / "prior_sensitivity_best_by_pi.tex", "prior_sensitivity/prior_sensitivity_best_by_pi.tex")

    thesis_diag = run_dir / "pair_score_separation" / "thesis_score_diagnostics"
    _cp(thesis_diag / "thesis_pair_score_statistics.csv", "pair_scores/thesis_pair_score_statistics.csv")
    _cp(thesis_diag / "thesis_pair_score_separation.csv", "pair_scores/thesis_pair_score_separation.csv")
    _cp(thesis_diag / "thesis_pair_score_diagnostics.json", "pair_scores/thesis_pair_score_diagnostics.json")
    _cp(thesis_diag / "thesis_pair_score_statistics.tex", "pair_scores/thesis_pair_score_statistics.tex")

    plots = run_dir / "pair_score_separation" / "plots"
    if plots.is_dir():
        kde_dst = tdir / "kde_plots"
        kde_dst.mkdir(parents=True, exist_ok=True)
        for png in plots.glob("score_density_kde_*.png"):
            shutil.copy2(png, kde_dst / png.name)
            copies[f"kde_plots/{png.name}"] = str(kde_dst / png.name)

    _cp(run_dir / "plots" / "loss_over_epochs_best_val_marked.png", "training/loss_over_epochs_best_val_marked.png")
    _cp(run_dir / "plots" / "loss_over_epochs.png", "training/loss_over_epochs.png")

    table_rows = [
        {"label": "legacy no-ts (30 ep)", **baseline_best},
        {"label": "final timestamp+ES", **best_final},
    ]
    tex_best = format_latex_community_table(
        table_rows,
        caption="Final canonical vs legacy \\texttt{14\\_only\\_mlp} (expanded GT).",
        label="tab:final-vs-legacy-community",
    )
    (tdir / "final_vs_legacy_community_best.tex").write_text(tex_best, encoding="utf-8")

    summary_md = "\n".join(
        [
            "# Final canonical `_14_only_mlp` thesis summary",
            "",
            "## Pair universe",
            "- Time gating **off** in candidate generation; pair **counts** match baseline `_13`.",
            "- Timestamp feature: `log1p(|ts_i - ts_j|)` in `time_gap_seconds_min`.",
            "",
            "## Training",
            f"- Best epoch: {stability.get('best_epoch')}; final epoch: {stability.get('final_epoch')}",
            f"- Early stopping triggered: {stability.get('early_stopping_triggered')}",
            f"- Best val loss: {stability.get('best_val_loss')}",
            "",
            "## Best community (expanded GT)",
            f"- **{best_final.get('algorithm')}** @ threshold {best_final.get('threshold')}, resolution {best_final.get('resolution')}",
            f"- H={best_final.get('homogeneity'):.3f}, C={best_final.get('completeness'):.3f}, V={best_final.get('v_measure'):.3f}",
            "",
            "## vs legacy baseline (Leiden 0.3 / 3.0, V=0.936)",
            f"- ΔV = {v_delta:+.4f}",
            "",
            f"**Interpretation:** {interpretation}",
            "",
            "## Raw paths",
            f"- Run: `{run_dir}`",
            f"- Community sweep: `{sweep_main}`",
            f"- Thesis bundle: `{tdir}`",
        ]
    )
    (tdir / "THESIS_SUMMARY.md").write_text(summary_md, encoding="utf-8")

    manifest_out: dict[str, Any] = {
        "thesis_output_dir": str(tdir),
        "raw_run_dir": str(run_dir),
        "community_sweep_csv": str(sweep_main),
        "best_final_community": best_final,
        "baseline_community": baseline_best,
        "delta_v_measure": v_delta,
        "training_stability": stability,
        "interpretation": interpretation,
        "copied_artifacts": copies,
    }
    (tdir / "paths_manifest.json").write_text(json.dumps(manifest_out, indent=2), encoding="utf-8")
    (tdir / "final_summary.json").write_text(json.dumps(manifest_out, indent=2), encoding="utf-8")
    print(json.dumps(manifest_out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
