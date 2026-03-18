from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd


def load_dbscan_sweep_csvs(db_scan_dir: str | Path, *, model_file: str | None = None) -> pd.DataFrame:
    """
    Load and concatenate DBSCAN sweep CSVs under `db_scan_dir`.
    Optionally filter by `model_file` if that column exists.
    """
    db_scan_dir = Path(db_scan_dir)
    csv_files = sorted(db_scan_dir.glob("clustering_results_eps_*.csv"))
    if not csv_files:
        return pd.DataFrame()

    dfs = []
    for f in csv_files:
        df = pd.read_csv(f)

        # Some older CSVs don't have an explicit epsilon column; derive from filename.
        if "epsilon" not in df.columns:
            eps_str = f.stem.split("clustering_results_eps_")[-1].replace("_", ".")
            try:
                df["epsilon"] = float(eps_str)
            except ValueError:
                pass

        if model_file is not None and "model_file" in df.columns:
            df = df[df["model_file"] == model_file]
        dfs.append(df)

    out = pd.concat(dfs, ignore_index=True)
    if "epsilon" in out.columns:
        out = out.sort_values("epsilon")
    return out


def _extract_epoch(name: object) -> int | None:
    if not isinstance(name, str):
        return None
    m = re.search(r"(?:epoch[_-]?)(\d+)", name)
    if m:
        return int(m.group(1))
    m2 = re.search(r"(\d+)", name)
    return int(m2.group(1)) if m2 else None


def load_dbscan_results_for_epsilon(db_scan_dir: str | Path, epsilon: float) -> pd.DataFrame:
    """
    Load a single DBSCAN result CSV for a fixed epsilon.
    Also derives an `epoch` column from `model_file` if missing.
    """
    db_scan_dir = Path(db_scan_dir)
    csv_path = db_scan_dir / f"clustering_results_eps_{str(epsilon).replace('.', '_')}.csv"
    if not csv_path.exists():
        return pd.DataFrame()

    df = pd.read_csv(csv_path)
    if "epsilon" in df.columns:
        df = df[df["epsilon"] == epsilon]
    if "model_file" in df.columns:
        df = df[df["model_file"].notna()]

    if "epoch" not in df.columns and "model_file" in df.columns:
        df["epoch"] = df["model_file"].apply(_extract_epoch)

    if "epoch" in df.columns and df["epoch"].isna().any():
        df.loc[df["epoch"].isna(), "epoch"] = range(len(df[df["epoch"].isna()]))
    if "epoch" in df.columns:
        df = df.sort_values("epoch")
    return df


def plot_coverage_and_noise_fraction(
    df: pd.DataFrame,
    *,
    x: str,
    total_items: int | None = None,
    title: str | None = None,
    figsize: tuple[int, int] = (6, 3),
):
    fig, ax = plt.subplots(figsize=figsize)
    if "coverage" in df.columns:
        ax.plot(df[x], df["coverage"], linewidth=2, label="coverage")

    if total_items is not None and "n_noise" in df.columns:
        noise_frac = df["n_noise"] / max(1, total_items)
        ax.plot(df[x], noise_frac, linewidth=2, label="noise fraction")

    ax.set_xlabel(x)
    ax.set_ylabel("coverage / noise fraction")
    if title:
        ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    return fig, ax


def plot_metric_lines(
    df: pd.DataFrame,
    *,
    x: str,
    metrics: Iterable[str],
    title: str | None = None,
    ylabel: str = "score",
    figsize: tuple[int, int] = (6, 3),
):
    fig, ax = plt.subplots(figsize=figsize)
    for m in metrics:
        if m in df.columns:
            ax.plot(df[x], df[m], linewidth=2, label=m)
    ax.set_xlabel(x)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    return fig, ax


def plot_n_clusters(df: pd.DataFrame, *, x: str = "epsilon", title: str | None = None):
    fig, ax = plt.subplots(figsize=(6, 3))
    if "n_clusters" in df.columns:
        ax.plot(df[x], df["n_clusters"], linewidth=2)
    ax.set_xlabel(x)
    ax.set_ylabel("num clusters")
    if title:
        ax.set_title(title)
    ax.grid(True, alpha=0.3)
    return fig, ax


def plot_dbscan_epsilon_sweep(metrics_df: pd.DataFrame, *, total_emails: int | None = None, title_prefix: str = ""):
    """
    Create the standard set of DBSCAN epsilon sweep plots.
    Returns a list of (fig, ax).
    """
    if metrics_df.empty:
        return []

    eps = metrics_df["epsilon"] if "epsilon" in metrics_df.columns else None
    prefix = f"{title_prefix} " if title_prefix else ""
    figs = []

    figs.append(
        plot_coverage_and_noise_fraction(
            metrics_df,
            x="epsilon",
            total_items=total_emails,
            title=f"{prefix}Coverage vs epsilon",
        )
    )
    figs.append(
        plot_metric_lines(
            metrics_df,
            x="epsilon",
            metrics=("homogeneity", "completeness", "v_measure", "silhouette"),
            title=f"{prefix}Homogeneity / Completeness / V-measure / Silhouette vs epsilon",
        )
    )
    figs.append(plot_n_clusters(metrics_df, x="epsilon", title=f"{prefix}Number of clusters vs epsilon"))
    return figs


def plot_meanshift_quantile_sweep(ms_df: pd.DataFrame, *, title: str = "MeanShift metrics vs quantile"):
    if ms_df.empty:
        return None
    if "quantile" in ms_df.columns:
        ms_df = ms_df.sort_values("quantile")

    fig, ax = plt.subplots(figsize=(6, 3))
    for col in ("homogeneity", "completeness", "v_measure"):
        if col in ms_df.columns:
            ax.plot(ms_df["quantile"], ms_df[col], linewidth=2, label=col)
    ax.set_xlabel("quantile")
    ax.set_ylabel("score")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    return fig, ax

