"""
Plot pair-training curves from a run's ``metrics.csv``.

Writes PNGs under ``<run_dir>/plots/``:

- ``loss_over_epochs.png`` — train vs validation loss
- ``accuracy_over_epochs.png`` — train vs validation placeholder accuracy (only if at
  least one series has finite values)

Example (from repository root)::

    python -m seed_candidate_workflow.utils.plot_training_metrics --run-dir output/runs/my_run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_metrics_csv(metrics_path: Path) -> pd.DataFrame:
    if not metrics_path.is_file():
        raise FileNotFoundError(f"metrics file not found: {metrics_path}")
    df = pd.read_csv(metrics_path, na_values=["", " "])
    if df.empty:
        raise ValueError(f"metrics CSV is empty: {metrics_path}")
    if "epoch" not in df.columns:
        raise ValueError(f"metrics CSV missing 'epoch' column: {metrics_path}")
    return df.sort_values("epoch").reset_index(drop=True)


def _finite_series(s: pd.Series) -> np.ndarray:
    v = pd.to_numeric(s, errors="coerce").to_numpy(dtype=np.float64)
    return v


def plot_loss_over_epochs(df: pd.DataFrame, out_path: Path, *, dpi: int = 120) -> None:
    epochs = df["epoch"].to_numpy()
    train = _finite_series(df["train_loss"])
    val = _finite_series(df["val_loss"])

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(epochs, train, label="train loss", color="#1f77b4", linewidth=1.8)
    ax.plot(epochs, val, label="validation loss", color="#ff7f0e", linewidth=1.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training and validation loss")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def plot_accuracy_over_epochs(df: pd.DataFrame, out_path: Path, *, dpi: int = 120) -> bool:
    if "train_placeholder_acc" not in df.columns or "val_placeholder_acc" not in df.columns:
        return False
    epochs = df["epoch"].to_numpy()
    train = _finite_series(df["train_placeholder_acc"])
    val = _finite_series(df["val_placeholder_acc"])
    if not (np.any(np.isfinite(train)) or np.any(np.isfinite(val))):
        return False

    fig, ax = plt.subplots(figsize=(8, 4.5))
    if np.any(np.isfinite(train)):
        ax.plot(epochs, train, label="train placeholder acc", color="#2ca02c", linewidth=1.8)
    if np.any(np.isfinite(val)):
        ax.plot(epochs, val, label="validation placeholder acc", color="#d62728", linewidth=1.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Placeholder accuracy (train / validation)")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    finite = np.concatenate([train[np.isfinite(train)], val[np.isfinite(val)]])
    top_y = float(np.max(finite)) if finite.size else 1.0
    ax.set_ylim(bottom=0.0, top=max(1.0, top_y * 1.05))
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return True


def write_training_plots(
    run_dir: Path | str,
    *,
    metrics_csv: str = "metrics.csv",
    plots_subdir: str = "plots",
    dpi: int = 120,
) -> dict[str, Any]:
    """
    Read ``metrics_csv`` inside ``run_dir`` and write plot PNGs under ``plots_subdir``.

    Returns a dict with paths written and whether accuracy was skipped.
    """
    run = Path(run_dir).expanduser().resolve()
    df = load_metrics_csv(run / metrics_csv)
    plots = run / plots_subdir
    loss_path = plots / "loss_over_epochs.png"
    acc_path = plots / "accuracy_over_epochs.png"

    plot_loss_over_epochs(df, loss_path, dpi=dpi)
    wrote_acc = plot_accuracy_over_epochs(df, acc_path, dpi=dpi)
    out: dict[str, Any] = {
        "run_dir": run,
        "loss_plot": loss_path,
        "accuracy_plot": acc_path if wrote_acc else None,
    }
    if not wrote_acc:
        out["accuracy_skipped_reason"] = (
            "no finite train_placeholder_acc / val_placeholder_acc in metrics"
        )
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Plot loss and accuracy vs epoch from a training run's metrics.csv.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Run directory containing metrics.csv (e.g. output/runs/my_experiment)",
    )
    parser.add_argument(
        "--metrics-csv",
        default="metrics.csv",
        help="Metrics filename inside run-dir (default: metrics.csv)",
    )
    parser.add_argument(
        "--plots-subdir",
        default="plots",
        help="Subfolder under run-dir for PNG output (default: plots)",
    )
    parser.add_argument("--dpi", type=int, default=120, help="PNG resolution (default: 120)")
    args = parser.parse_args(argv)

    repo = _repo_root()
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))

    result = write_training_plots(
        args.run_dir,
        metrics_csv=args.metrics_csv,
        plots_subdir=args.plots_subdir,
        dpi=args.dpi,
    )
    print(f"Wrote loss plot: {result['loss_plot']}")
    if result["accuracy_plot"] is not None:
        print(f"Wrote accuracy plot: {result['accuracy_plot']}")
    else:
        print(f"Skipped accuracy plot ({result.get('accuracy_skipped_reason', '')})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
