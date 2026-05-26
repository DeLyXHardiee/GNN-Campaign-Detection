#!/usr/bin/env python3
"""Compare epoch metrics for _14 MLP vs Edge-GNN (2-layer) vs Edge-GNN no-MP."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _metrics_path(run_dir: Path) -> Path | None:
    for rel in ("mlp/metrics.csv", "edge_gnn/metrics.csv", "metrics.csv"):
        p = run_dir / rel
        if p.is_file():
            return p
    return None


def _load(run_dir: Path, label: str) -> pd.DataFrame:
    p = _metrics_path(run_dir)
    if p is None:
        raise FileNotFoundError(f"No metrics.csv under {run_dir}")
    df = pd.read_csv(p)
    df["model"] = label
    return df


def _row_summary(df: pd.DataFrame, epoch: int) -> dict[str, object]:
    r = df[df["epoch"] == epoch]
    if r.empty:
        return {"epoch": epoch}
    r = r.iloc[0]
    return {
        "epoch": epoch,
        "train_loss": r.get("train_loss"),
        "val_loss": r.get("val_loss"),
        "train_p_pos": r.get("train_epoch_mean_pos_prob"),
        "train_p_unl": r.get("train_epoch_mean_unl_prob"),
        "train_sep": r.get("train_epoch_score_separation"),
        "val_p_pos": r.get("val_epoch_mean_pos_prob"),
        "val_p_unl": r.get("val_epoch_mean_unl_prob"),
        "val_sep": r.get("val_epoch_score_separation"),
        "train_r_p_pos": r.get("train_epoch_mean_r_p_pos"),
        "val_r_p_pos": r.get("val_epoch_mean_r_p_pos"),
        "val_r_u_neg": r.get("val_epoch_mean_r_u_neg"),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mlp-run", type=Path, required=True)
    ap.add_argument("--edge-gnn-run", type=Path, required=True)
    ap.add_argument("--edge-gnn-no-mp-run", type=Path, required=True)
    ap.add_argument("--epochs", type=int, nargs="+", default=[1, 5, 10, 30])
    args = ap.parse_args()

    runs = [
        ("_14_explicit_mlp", args.mlp_run.resolve()),
        ("_19_edge_gnn_2layer", args.edge_gnn_run.resolve()),
        ("_19_edge_gnn_no_mp", args.edge_gnn_no_mp_run.resolve()),
    ]

    rows: list[dict[str, object]] = []
    for label, path in runs:
        df = _load(path, label)
        for ep in args.epochs:
            row = _row_summary(df, ep)
            row["model"] = label
            row["run_dir"] = str(path)
            rows.append(row)

    out = pd.DataFrame(rows)
    cols = [
        "model",
        "epoch",
        "train_loss",
        "val_loss",
        "train_p_pos",
        "train_p_unl",
        "train_sep",
        "val_p_pos",
        "val_p_unl",
        "val_sep",
        "train_r_p_pos",
        "val_r_p_pos",
        "val_r_u_neg",
    ]
    print(out[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
