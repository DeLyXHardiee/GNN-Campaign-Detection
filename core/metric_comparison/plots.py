"""Matplotlib figures for featureset vs GNN metric comparison."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


def save_external_metrics_bar_chart(
    *,
    featureset_metrics: dict[str, Any] | None,
    gnn_metrics: dict[str, Any] | None,
    out_path: Path,
    dpi: int = 150,
) -> str | None:
    """
    Grouped bar chart: Homogeneity, Completeness, V-measure for each solution.
    Skips if both sides lack valid metrics.
    """
    names = ["homogeneity", "completeness", "v_measure"]
    labels = ["Homogeneity", "Completeness", "V-measure"]
    series: list[tuple[str, list[float]]] = []
    if featureset_metrics:
        series.append(
            (
                "Feature set",
                [float(featureset_metrics.get(n, 0.0)) for n in names],
            )
        )
    if gnn_metrics:
        series.append(
            ("GNN", [float(gnn_metrics.get(n, 0.0)) for n in names]),
        )
    if not series:
        return None

    x = range(len(names))
    width = 0.35 if len(series) == 2 else 0.5
    fig, ax = plt.subplots(figsize=(7, 4))
    offset = -width / 2 if len(series) == 2 else 0.0
    for i, (label, values) in enumerate(series):
        pos = [xi + offset + i * width for xi in x]
        ax.bar(pos, values, width=width, label=label)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("External clustering metrics vs ground truth")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return str(out_path)


def save_agreement_bar_chart(
    *,
    ari: float,
    ami: float,
    out_path: Path,
    dpi: int = 150,
) -> str:
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(["Adjusted Rand", "AMI"], [ari, ami], color=["#4c72b0", "#55a868"])
    ax.set_ylim(-1.05, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Agreement: feature set vs GNN partitions")
    ax.axhline(0.0, color="gray", linewidth=0.8)
    ax.grid(axis="y", alpha=0.3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return str(out_path)
