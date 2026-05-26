#!/usr/bin/env python3
"""Step 8: train/validation loss plot with best-val epoch marker."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from seed_candidate_workflow.utils.final_14_only_mlp_thesis import (  # noqa: E402
    load_manifest,
    read_training_stability,
    repo_root,
    steps_dir,
    training_run_dir,
)
from seed_candidate_workflow.utils.plot_training_metrics import TRAIN_COLOR, VAL_COLOR  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, default=None)
    args = p.parse_args()

    repo = repo_root()
    manifest = load_manifest(args.manifest)
    run_dir = training_run_dir(repo, str(manifest["run_id"]))
    from seed_candidate_workflow.utils.plot_training_metrics import resolve_metrics_csv_path

    metrics_path = resolve_metrics_csv_path(run_dir)
    df = pd.read_csv(metrics_path)
    stability = read_training_stability(run_dir)

    epochs = df["epoch"].to_numpy()
    train = pd.to_numeric(df["train_loss"], errors="coerce")
    val = pd.to_numeric(df["val_loss"], errors="coerce")

    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    out_png = plots_dir / "loss_over_epochs_best_val_marked.png"

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(epochs, train, label="train loss", color=TRAIN_COLOR, linewidth=1.8)
    ax.plot(epochs, val, label="validation loss", color=VAL_COLOR, linewidth=1.8)
    best_ep = stability.get("best_epoch")
    if best_ep is not None:
        ax.axvline(int(best_ep), color="#2ca02c", linestyle="--", linewidth=1.2, label=f"best val (epoch {int(best_ep)})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Final MLP training (nnPU)")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)

    from seed_candidate_workflow.utils.plot_training_metrics import write_training_plots

    extra = write_training_plots(run_dir, metrics_csv="mlp/metrics.csv", plots_subdir="plots")
    extra_json = {k: (str(v) if isinstance(v, Path) else v) for k, v in extra.items()}

    report = {
        "plot_png": str(out_png),
        "metrics_csv": str(metrics_path),
        "training_stability": stability,
        "standard_plots": extra_json,
    }
    out_dir = steps_dir(repo, manifest)
    (out_dir / "step08_training_plot_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
