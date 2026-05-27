#!/usr/bin/env python3
"""Step 8: training/validation loss plots for both GNN thesis runs."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from seed_candidate_workflow.utils.final_gnn_timestamp_es_thesis import (  # noqa: E402
    copy_if_exists,
    load_manifest,
    read_training_stability,
    repo_root,
    steps_dir,
    thesis_dir,
    training_run_dir,
)
from seed_candidate_workflow.utils.plot_training_metrics import TRAIN_COLOR, VAL_COLOR  # noqa: E402


def _plot_loss(run_dir: Path, out_png: Path, title: str) -> dict[str, Any]:
    from seed_candidate_workflow.utils.plot_training_metrics import resolve_metrics_csv_path

    metrics_path = resolve_metrics_csv_path(run_dir)
    df = pd.read_csv(metrics_path)
    stability = read_training_stability(run_dir)
    epochs = df["epoch"].to_numpy()
    train = pd.to_numeric(df["train_loss"], errors="coerce")
    val = pd.to_numeric(df["val_loss"], errors="coerce")

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(epochs, train, label="train loss", color=TRAIN_COLOR, linewidth=1.8)
    ax.plot(epochs, val, label="validation loss", color=VAL_COLOR, linewidth=1.8)
    best_ep = stability.get("best_epoch")
    if best_ep is not None:
        ax.axvline(int(best_ep), color="#2ca02c", linestyle="--", linewidth=1.2, label=f"best val (epoch {int(best_ep)})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("nnPU loss")
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return stability


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, default=None)
    p.add_argument("--skip-existing", action="store_true")
    args = p.parse_args()

    repo = repo_root()
    manifest = load_manifest(args.manifest)
    step_out = steps_dir(repo, manifest) / "step08_training_plots_report.json"
    if args.skip_existing and step_out.is_file():
        print(f"[step08] skip (report exists): {step_out}")
        print(step_out.read_text(encoding="utf-8"))
        return 0

    tdir = thesis_dir(repo, manifest)
    train_out = tdir / "training"
    train_out.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {}
    for label, run_id_key, title in (
        ("gnn_plus", "run_id_gnn_plus", "GNN + explicit pair features (thesis ES100)"),
        ("gnn_only", "run_id_gnn_only", "GNN-only pair scorer (thesis ES100)"),
    ):
        run_id = str(manifest[run_id_key])
        run_dir = training_run_dir(repo, run_id)
        out_png = train_out / f"{run_id}__loss_over_epochs.png"
        stability = _plot_loss(run_dir, out_png, title)
        metrics = run_dir / "gnn" / "metrics.csv"
        copy_if_exists(metrics, train_out / f"{run_id}__metrics.csv")
        report[label] = {"run_id": run_id, "loss_plot": str(out_png), "training_stability": stability}

    step_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
