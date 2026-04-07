"""
Matplotlib plots for Method 1 V2 post-training diagnostics (static, notebook-friendly).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# matplotlib imported lazily in functions to avoid import cost in non-plot contexts


def _ensure_ax(ax):
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))
    return ax


def plot_score_histograms(
    data: dict[str, np.ndarray | pd.Series],
    *,
    bins: int = 60,
    alpha: float = 0.45,
    title: str = "Edge score histograms",
    ax=None,
):
    import matplotlib.pyplot as plt

    ax = _ensure_ax(ax)
    for label, s in data.items():
        x = pd.to_numeric(pd.Series(s), errors="coerce").dropna().to_numpy(dtype=np.float64)
        if len(x) == 0:
            continue
        ax.hist(x, bins=bins, alpha=alpha, label=label, density=True, histtype="stepfilled")
    ax.set_xlabel("Score")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend()
    return ax


def plot_score_cdf(
    data: dict[str, np.ndarray | pd.Series],
    *,
    title: str = "Edge score CDFs",
    ax=None,
):
    import matplotlib.pyplot as plt

    ax = _ensure_ax(ax)
    for label, s in data.items():
        x = np.sort(pd.to_numeric(pd.Series(s), errors="coerce").dropna().to_numpy(dtype=np.float64))
        if len(x) == 0:
            continue
        y = np.linspace(0.0, 1.0, len(x), endpoint=True)
        ax.plot(x, y, label=label, linewidth=1.5)
    ax.set_xlabel("Score")
    ax.set_ylabel("CDF")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.25)
    return ax


def plot_scatter_scores(
    x: np.ndarray | pd.Series,
    y: np.ndarray | pd.Series,
    *,
    xlab: str,
    ylab: str,
    title: str,
    hexbin: bool = False,
    max_points_scatter: int = 25_000,
    ax=None,
):
    import matplotlib.pyplot as plt

    ax = _ensure_ax(ax)
    xs = pd.to_numeric(pd.Series(x), errors="coerce")
    ys = pd.to_numeric(pd.Series(y), errors="coerce")
    m = xs.notna() & ys.notna()
    xa = xs[m].to_numpy(dtype=np.float64)
    ya = ys[m].to_numpy(dtype=np.float64)
    if len(xa) == 0:
        ax.set_title(title + " (no data)")
        return ax
    if hexbin or len(xa) > max_points_scatter:
        hb = ax.hexbin(xa, ya, gridsize=80, mincnt=1, cmap="viridis")
        plt.colorbar(hb, ax=ax, label="count")
    else:
        ax.scatter(xa, ya, s=2, alpha=0.25)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_title(title)
    return ax


def plot_topk_overlap_bars(overlap_df: pd.DataFrame, *, title: str = "Top-k rank overlap", ax=None):
    import matplotlib.pyplot as plt

    ax = _ensure_ax(ax)
    if overlap_df.empty:
        ax.set_title(title + " (empty)")
        return ax
    d = overlap_df.copy()
    x = d["k"].astype(str)
    h = d["overlap_fraction"].to_numpy(dtype=float)
    ax.bar(x, h, color="steelblue", edgecolor="black", linewidth=0.5)
    ax.set_xlabel("k (top edges)")
    ax.set_ylabel("Overlap fraction")
    ax.set_title(title)
    ax.set_ylim(0.0, 1.05)
    return ax


def plot_epoch_curves(epoch_df: pd.DataFrame, *, ycols: list[str], title: str, ax=None):
    import matplotlib.pyplot as plt

    ax = _ensure_ax(ax)
    if epoch_df.empty or "epoch" not in epoch_df.columns:
        ax.set_title(title + " (no data)")
        return ax
    ep = epoch_df["epoch"].to_numpy()
    for c in ycols:
        if c in epoch_df.columns:
            ax.plot(ep, pd.to_numeric(epoch_df[c], errors="coerce"), label=c, marker="o", markersize=3)
    ax.set_xlabel("Epoch (checkpoint)")
    ax.set_ylabel("Value")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    return ax


def plot_metrics_vs_n_edges(
    long_sweep: pd.DataFrame,
    *,
    methods: list[str] | None = None,
    metric: str = "v_measure",
    title: str | None = None,
    ax=None,
):
    import matplotlib.pyplot as plt

    ax = _ensure_ax(ax)
    if long_sweep.empty:
        ax.set_title((title or metric) + " (no data)")
        return ax
    d = long_sweep.copy()
    if methods is not None:
        d = d[d["method"].isin(methods)]
    for name, g in d.groupby("method"):
        g = g.sort_values("n_edges_after_threshold")
        ax.plot(
            g["n_edges_after_threshold"],
            pd.to_numeric(g[metric], errors="coerce"),
            marker="o",
            markersize=2,
            linewidth=1,
            label=name,
            alpha=0.85,
        )
    ax.set_xlabel("Edges after threshold (graph size)")
    ax.set_ylabel(metric)
    ax.set_title(title or f"{metric} vs graph size")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    return ax


def plot_training_history_losses(hist_df: pd.DataFrame, *, ax=None):
    import matplotlib.pyplot as plt

    ax = _ensure_ax(ax)
    if hist_df.empty:
        ax.set_title("Training history (empty)")
        return ax
    d = hist_df.copy()
    if "epoch" not in d.columns and d.index.name == "epoch":
        d = d.reset_index()
    if "epoch" in d.columns:
        xvals = pd.to_numeric(d["epoch"], errors="coerce").to_numpy(dtype=float)
    else:
        xvals = np.arange(len(d), dtype=float)
    pairs = [
        ("train_loss_ranking", "train ranking loss"),
        ("train_loss_stability", "train stability loss"),
        ("train_loss_agreement_aux", "train agreement aux loss"),
        ("train_loss_total", "train total loss"),
    ]
    for col, lab in pairs:
        if col in d.columns:
            y = pd.to_numeric(d[col], errors="coerce").to_numpy(dtype=float)
            ax.plot(xvals, y, label=lab)
    if "val_loss_total" in d.columns:
        yv = pd.to_numeric(d["val_loss_total"], errors="coerce").to_numpy(dtype=float)
        ax.plot(xvals, yv, label="val total loss", linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("V2 training losses")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    return ax


def plot_training_history_multipanel(
    hist_df: pd.DataFrame, *, figsize: tuple[float, float] = (11, 9)
):
    """
    Multi-panel view: component losses (incl. hub / anti-compress), totals, score stds,
    and optional per-epoch GT separation gaps if present in ``training_history.json``.
    """
    import matplotlib.pyplot as plt

    if hist_df.empty:
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_title("Training history (empty)")
        return fig, np.array([[ax]])
    d = hist_df.copy()
    if "epoch" not in d.columns and d.index.name == "epoch":
        d = d.reset_index()
    if "epoch" in d.columns:
        xvals = pd.to_numeric(d["epoch"], errors="coerce").to_numpy(dtype=float)
    else:
        xvals = np.arange(len(d), dtype=float)

    fig, axes = plt.subplots(2, 2, figsize=figsize, constrained_layout=True)
    ax = axes[0, 0]
    comp = [
        ("train_loss_ranking", "ranking"),
        ("train_loss_stability", "stability"),
        ("train_loss_hub", "hub"),
        ("train_loss_anti_compress", "anti-compress"),
        ("train_loss_agreement_aux", "agreement aux"),
    ]
    for col, lab in comp:
        if col in d.columns:
            ax.plot(xvals, pd.to_numeric(d[col], errors="coerce"), label=lab)
    ax.set_title("Train component losses")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.25)

    ax = axes[0, 1]
    if "train_loss_total" in d.columns:
        ax.plot(xvals, pd.to_numeric(d["train_loss_total"], errors="coerce"), label="train total")
    if "val_loss_total" in d.columns:
        ax.plot(
            xvals,
            pd.to_numeric(d["val_loss_total"], errors="coerce"),
            linestyle="--",
            label="val total",
        )
    ax.set_title("Total loss")
    ax.set_xlabel("Epoch")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)

    ax = axes[1, 0]
    std_cols = [
        ("train_mean_batch_score_std", "train batch score std"),
        ("val_mean_batch_score_std", "val batch score std"),
        ("score_std_full_graph", "full-graph std"),
    ]
    any_std = False
    for col, lab in std_cols:
        if col in d.columns:
            ax.plot(xvals, pd.to_numeric(d[col], errors="coerce"), label=lab)
            any_std = True
    ax.set_title("Score std" if any_std else "Score std (no columns)")
    ax.set_xlabel("Epoch")
    if any_std:
        ax.legend(fontsize=7)
    ax.grid(True, alpha=0.25)

    ax = axes[1, 1]
    gap_cols = [
        ("diag_gt_mean_gap_same_minus_cross", "GT gap (train log)"),
        ("diag_hsli_mean_gap_same_minus_cross", "HS-LI gap (train log)"),
    ]
    any_g = False
    for col, lab in gap_cols:
        if col in d.columns:
            ax.plot(
                xvals,
                pd.to_numeric(d[col], errors="coerce"),
                label=lab,
                marker="o",
                markersize=2,
            )
            any_g = True
    if not any_g:
        ax.text(
            0.5,
            0.5,
            "No diag_gt / diag_hsli gap columns\n(enable GT separation logging in training)",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=9,
        )
        ax.set_axis_off()
    else:
        ax.set_title("Training-time GT separation (diagnostics)")
        ax.set_xlabel("Epoch")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.25)
    return fig, axes


def plot_same_cross_score_evolution(
    stats_df: pd.DataFrame,
    *,
    title: str = "V2 plausibility: same vs cross (GT taxonomy)",
    figsize: tuple[float, float] = (10, 7),
):
    """Mean score vs epoch with p25–p75 band for same vs cross (all labeled and HS-LI rows)."""
    import matplotlib.pyplot as plt

    if stats_df.empty or "epoch" not in stats_df.columns:
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_title(title + " (no data)")
        return fig, np.array([ax])
    d = stats_df.sort_values("epoch")
    ep = pd.to_numeric(d["epoch"], errors="coerce").to_numpy(dtype=float)
    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)
    ax0 = axes[0]
    for name, color, lab in [
        ("same", "C0", "same-campaign"),
        ("cross", "C1", "cross-campaign"),
    ]:
        mean_c = f"{name}_mean"
        if mean_c not in d.columns:
            continue
        m = pd.to_numeric(d[mean_c], errors="coerce").to_numpy(dtype=float)
        lo = (
            pd.to_numeric(d[f"{name}_p25"], errors="coerce").to_numpy(dtype=float)
            if f"{name}_p25" in d.columns
            else m
        )
        hi = (
            pd.to_numeric(d[f"{name}_p75"], errors="coerce").to_numpy(dtype=float)
            if f"{name}_p75" in d.columns
            else m
        )
        ax0.fill_between(ep, lo, hi, alpha=0.22, color=color)
        ax0.plot(ep, m, marker="o", markersize=3, label=f"{lab} mean (p25–p75)", color=color)
    ax0.set_ylabel("V2 score")
    ax0.legend(fontsize=8)
    ax0.grid(True, alpha=0.25)
    ax0.set_title(title + " — all labeled")

    ax1 = axes[1]
    for name, color, lab in [
        ("hsli_same", "C2", "HS-LI same"),
        ("hsli_cross", "C3", "HS-LI cross"),
    ]:
        mean_c = f"{name}_mean"
        if mean_c not in d.columns:
            continue
        m = pd.to_numeric(d[mean_c], errors="coerce").to_numpy(dtype=float)
        lo = (
            pd.to_numeric(d[f"{name}_p25"], errors="coerce").to_numpy(dtype=float)
            if f"{name}_p25" in d.columns
            else m
        )
        hi = (
            pd.to_numeric(d[f"{name}_p75"], errors="coerce").to_numpy(dtype=float)
            if f"{name}_p75" in d.columns
            else m
        )
        ax1.fill_between(ep, lo, hi, alpha=0.22, color=color)
        ax1.plot(ep, m, marker="o", markersize=3, label=f"{lab} mean (p25–p75)", color=color)
    ax1.set_xlabel("Checkpoint epoch")
    ax1.set_ylabel("V2 score")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.25)
    ax1.set_title("HS-LI subset (high semantic / low infra-false)")
    fig.tight_layout()
    return fig, axes


def plot_separation_gaps_vs_epoch(
    df: pd.DataFrame,
    *,
    title: str = "Separation: mean(same) − mean(cross)",
    ax=None,
):
    """Plots whichever gap columns exist (score-stats and/or per-epoch Step-3 table)."""
    import matplotlib.pyplot as plt

    ax = _ensure_ax(ax)
    if df.empty or "epoch" not in df.columns:
        ax.set_title(title + " (no data)")
        return ax
    ep = pd.to_numeric(df["epoch"], errors="coerce").to_numpy(dtype=float)
    pairs = [
        ("mean_gap_same_minus_cross", "all labeled (score-stats table)"),
        ("hsli_mean_gap_same_minus_cross", "HS-LI (score-stats table)"),
        ("gt_mean_gap_same_minus_cross", "compact gap (epoch eval row)"),
        ("gt_hsli_mean_gap_same_minus_cross", "HS-LI compact (epoch eval row)"),
    ]
    plotted = False
    for col, lab in pairs:
        if col in df.columns:
            y = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
            ax.plot(ep, y, marker="o", markersize=3, label=lab)
            plotted = True
    if not plotted:
        ax.set_title(title + " (no gap columns)")
        return ax
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Gap")
    ax.set_title(title)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.25)
    return ax


def plot_same_cross_histogram_snapshots(
    snapshots: dict[str, tuple[np.ndarray, np.ndarray]],
    *,
    bins: int = 50,
    fig_width_per_col: float = 4.0,
    row_height: float = 3.2,
):
    """
    ``snapshots``: panel title -> (same-campaign scores, cross-campaign scores), e.g. three epochs.
    """
    import matplotlib.pyplot as plt

    if not snapshots:
        fig, ax = plt.subplots()
        ax.set_title("Same/cross histograms (empty)")
        return fig, np.array([ax])
    n = len(snapshots)
    fig, axes = plt.subplots(
        1,
        n,
        figsize=(fig_width_per_col * n, row_height),
        squeeze=False,
    )
    axes_flat = axes.ravel()
    for ax, (lab, pair) in zip(axes_flat, snapshots.items()):
        s_same, s_cross = pair
        for arr, name, color in [
            (s_same, "same", "C0"),
            (s_cross, "cross", "C1"),
        ]:
            x = np.asarray(arr, dtype=np.float64)
            x = x[np.isfinite(x)]
            if x.size == 0:
                continue
            ax.hist(x, bins=bins, alpha=0.38, label=name, density=True, color=color)
        ax.set_title(lab)
        ax.set_xlabel("V2 score")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig, axes_flat


def save_figure(path: str | Path, fig=None, dpi: int = 120) -> None:
    import matplotlib.pyplot as plt

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    (fig if fig is not None else plt.gcf()).savefig(path, dpi=dpi, bbox_inches="tight")


def plot_sweep_heatmap(
    sweep_df: pd.DataFrame,
    *,
    row_col: str = "min_edge_weight",
    col_col: str = "resolution",
    value_col: str = "v_measure",
    title: str | None = None,
    ax=None,
):
    import matplotlib.pyplot as plt

    ax = _ensure_ax(ax)
    if sweep_df.empty:
        ax.set_title((title or value_col) + " (empty)")
        return ax
    d = sweep_df.copy()
    d[row_col] = pd.to_numeric(d[row_col], errors="coerce")
    d[col_col] = pd.to_numeric(d[col_col], errors="coerce")
    d[value_col] = pd.to_numeric(d[value_col], errors="coerce")
    piv = d.pivot_table(index=row_col, columns=col_col, values=value_col, aggfunc="mean")
    im = ax.imshow(piv.to_numpy(dtype=float), aspect="auto", origin="lower", cmap="viridis")
    ax.set_xticks(np.arange(piv.shape[1]))
    ax.set_xticklabels([f"{x:.3g}" for x in piv.columns], rotation=45, ha="right", fontsize=7)
    ax.set_yticks(np.arange(piv.shape[0]))
    ax.set_yticklabels([f"{y:.3g}" for y in piv.index], fontsize=7)
    ax.set_xlabel(col_col)
    ax.set_ylabel(row_col)
    ax.set_title(title or f"{value_col} heatmap ({row_col} × {col_col})")
    plt.colorbar(im, ax=ax, label=value_col)
    return ax


def write_diagnostics_report_md(
    out_path: str | Path,
    sections: list[tuple[str, str]],
    *,
    title: str = "Method 2 / V2 diagnostics summary",
) -> None:
    """Write a small markdown report from (heading, markdown_body) pairs."""
    lines = [f"# {title}", ""]
    for h, body in sections:
        lines.append(f"## {h}")
        lines.append("")
        lines.append(body.strip())
        lines.append("")
    Path(out_path).write_text("\n".join(lines), encoding="utf-8")
