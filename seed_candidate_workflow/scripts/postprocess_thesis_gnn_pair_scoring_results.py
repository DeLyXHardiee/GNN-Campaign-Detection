#!/usr/bin/env python3
"""
Lightweight postprocessing for thesis_gnn_pair_scoring_results:
- generate training loss plots from metrics.csv if missing
- write a community-sweep best-config table (csv/json/tex)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


_REPO = Path(__file__).resolve().parents[2]


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _best_row_by_vmeasure(path: Path) -> dict[str, Any]:
    df = pd.read_csv(path, low_memory=False)
    df["v_measure"] = pd.to_numeric(df["v_measure"], errors="coerce")
    row = df.sort_values("v_measure", ascending=False).iloc[0].to_dict()
    return row


def _generate_loss_plot(metrics_csv: Path, out_png: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    df = pd.read_csv(metrics_csv, low_memory=False)
    df["epoch"] = pd.to_numeric(df["epoch"], errors="coerce")
    df["train_loss"] = pd.to_numeric(df["train_loss"], errors="coerce")
    df["val_loss"] = pd.to_numeric(df["val_loss"], errors="coerce")
    df = df.dropna(subset=["epoch"])

    _ensure_dir(out_png.parent)
    plt.figure(figsize=(7.2, 4.2), dpi=150)
    plt.plot(df["epoch"], df["train_loss"], label="train_loss", linewidth=1.5)
    plt.plot(df["epoch"], df["val_loss"], label="val_loss", linewidth=1.5)
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.25)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def main() -> int:
    root = _REPO / "seed_candidate_workflow/output/thesis_gnn_pair_scoring_results"
    tables_dir = _ensure_dir(root / "tables")
    training_dir = _ensure_dir(root / "training")

    run_13 = _REPO / "output/runs/main_gnn_pu_1_no_ts_dedup_task_identity_13"
    run_15 = _REPO / "output/runs/main_gnn_pu_1_no_ts_dedup_task_identity_15_gnn_only_scorer"
    run_mlp = _REPO / "output/runs/final_14_only_mlp__timestamp_feature__early_stopping"

    # Training plots from metrics.csv (appendix-friendly).
    _generate_loss_plot(
        run_13 / "gnn/metrics.csv",
        training_dir / "main_gnn_pu_1_no_ts_dedup_task_identity_13__loss_from_metrics.csv.png",
        title="GNN+explicit features: train/val loss (from metrics.csv)",
    )
    _generate_loss_plot(
        run_15 / "gnn/metrics.csv",
        training_dir / "main_gnn_pu_1_no_ts_dedup_task_identity_15_gnn_only_scorer__loss_from_metrics.csv.png",
        title="GNN-only scorer: train/val loss (from metrics.csv)",
    )

    # Community sweep best-config table (expanded GT sweeps).
    rows = []
    for run_id, sweep_csv in (
        (
            "final_14_only_mlp__timestamp_feature__early_stopping",
            run_mlp / "community/anchor_community_sweep__ground_truth.csv",
        ),
        (
            "main_gnn_pu_1_no_ts_dedup_task_identity_13",
            run_13 / "community/anchor_community_sweep__ground_truth.csv",
        ),
        (
            "main_gnn_pu_1_no_ts_dedup_task_identity_15_gnn_only_scorer",
            run_15 / "community/anchor_community_sweep__ground_truth.csv",
        ),
    ):
        best = _best_row_by_vmeasure(sweep_csv)
        rows.append(
            {
                "run_id": run_id,
                "sweep_csv": str(sweep_csv),
                "algorithm": best.get("method"),
                "threshold": best.get("min_edge_weight"),
                "resolution": best.get("resolution"),
                "homogeneity": best.get("homogeneity"),
                "completeness": best.get("completeness"),
                "v_measure": best.get("v_measure"),
                "n_communities": best.get("n_communities"),
                "retained_edges": best.get("n_edges_after_threshold", None),
            }
        )

    df = pd.DataFrame(rows)
    (tables_dir / "gnn_pair_scoring_community_sweep_best.csv").write_text(
        df.to_csv(index=False), encoding="utf-8"
    )
    (tables_dir / "gnn_pair_scoring_community_sweep_best.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )
    (tables_dir / "gnn_pair_scoring_community_sweep_best.tex").write_text(
        df.to_latex(index=False, float_format="%.3f"),
        encoding="utf-8",
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

