from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def save_dataframe(df: pd.DataFrame, path: str | Path) -> str:
    """
    Save `df` as a CSV to `path` and return the string path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return str(path)


def load_dbscan_sweep_csvs(db_scan_dir: str | Path, *, model_file: str | None = None) -> pd.DataFrame:
    """
    Load and concatenate DBSCAN sweep CSVs under `db_scan_dir`.

    Current pipeline output is typically:
      - `**/*_dbscan_sweep.csv` (one CSV per model/checkpoint)
      - includes per-row `epsilon` and a `model` column

    For backwards compatibility, older outputs like `clustering_results_eps_*.csv`
    are also supported.
    """
    db_scan_dir = Path(db_scan_dir)
    # Current naming: `<model_stem>_dbscan_sweep.csv`
    csv_files = sorted(db_scan_dir.rglob("*_dbscan_sweep.csv"))

    # Back-compat: older per-epsilon CSVs.
    if not csv_files:
        csv_files = sorted(db_scan_dir.rglob("clustering_results_eps_*.csv"))

    if not csv_files:
        return pd.DataFrame()

    dfs = []
    for f in csv_files:
        df = pd.read_csv(f)

        # Some older CSVs don't have an explicit epsilon column; derive from filename.
        if "epsilon" not in df.columns and "clustering_results_eps_" in f.stem:
            eps_str = f.stem.split("clustering_results_eps_")[-1].replace("_", ".")
            try:
                df["epsilon"] = float(eps_str)
            except ValueError:
                pass

        if model_file is not None:
            if "model_file" in df.columns:
                df = df[df["model_file"] == model_file]
            elif "model" in df.columns:
                # In current pipeline outputs, `model` is typically the checkpoint stem
                # (e.g. `best_model` from `best_model.pt`).
                model_key = (
                    Path(model_file).stem if "/" in model_file or "." in model_file else model_file
                )
                df = df[df["model"] == model_key]

        dfs.append(df)

    out = pd.concat(dfs, ignore_index=True)
    if "epsilon" in out.columns:
        out = out.sort_values("epsilon")
    return out


def load_meanshift_sweep_csvs(
    meanshift_dir: str | Path, *, model_file: str | None = None
) -> pd.DataFrame:
    """
    Load and concatenate MeanShift sweep CSVs under `meanshift_dir`.

    Current pipeline output is typically:
      - `**/*_meanshift_sweep.csv` (one CSV per model/checkpoint)
      - includes per-row `quantile` and a `model` column

    Returns an empty DataFrame if no files are found.
    """
    meanshift_dir = Path(meanshift_dir)
    # Current naming: `<model_stem>_meanshift_sweep.csv`
    csv_files = sorted(meanshift_dir.rglob("*_meanshift_sweep.csv"))

    # Optional back-compat if you ever generate "clustering_results_quantile_*.csv".
    if not csv_files:
        csv_files = sorted(meanshift_dir.rglob("clustering_results_quantile_*.csv"))

    if not csv_files:
        return pd.DataFrame()

    dfs = []
    for f in csv_files:
        df = pd.read_csv(f)

        # Some older CSVs might not have `quantile`; derive from filename.
        if "quantile" not in df.columns and "clustering_results_quantile_" in f.stem:
            q_str = f.stem.split("clustering_results_quantile_")[-1].replace("_", ".")
            try:
                df["quantile"] = float(q_str)
            except ValueError:
                pass

        if model_file is not None:
            if "model_file" in df.columns:
                df = df[df["model_file"] == model_file]
            elif "model" in df.columns:
                model_key = (
                    Path(model_file).stem if "/" in model_file or "." in model_file else model_file
                )
                df = df[df["model"] == model_key]

        dfs.append(df)

    out = pd.concat(dfs, ignore_index=True)
    if "quantile" in out.columns:
        out = out.sort_values("quantile")
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
    Load DBSCAN sweep results for a fixed epsilon.

    Current pipeline outputs one `dbscan_sweep.csv` per model with `epsilon`
    stored as a column, so this filters the concatenated sweep rows.

    Also derives an `epoch` column from `model` if missing (or from `model_file`
    for backwards compatibility).
    """
    metrics_df = load_dbscan_sweep_csvs(db_scan_dir)
    if metrics_df.empty:
        return metrics_df

    if "epsilon" not in metrics_df.columns:
        return pd.DataFrame()

    eps_col = metrics_df["epsilon"]
    try:
        eps_numeric = pd.to_numeric(eps_col, errors="coerce")
        # Use tolerance for float comparisons.
        mask = eps_numeric.notna() & np.isclose(eps_numeric.to_numpy(), float(epsilon), rtol=1e-6, atol=1e-12)
        df = metrics_df[mask].copy()
    except Exception:
        df = metrics_df[metrics_df["epsilon"] == epsilon].copy()

    if "model" in df.columns:
        df = df[df["model"].notna()]
    elif "model_file" in df.columns:
        df = df[df["model_file"].notna()]

    if "epoch" not in df.columns:
        if "model" in df.columns:
            df["epoch"] = df["model"].apply(_extract_epoch)
        elif "model_file" in df.columns:
            df["epoch"] = df["model_file"].apply(_extract_epoch)

    if "epoch" in df.columns and df["epoch"].isna().any():
        df.loc[df["epoch"].isna(), "epoch"] = range(len(df[df["epoch"].isna()]))

    if "epoch" in df.columns:
        df = df.sort_values("epoch")
    return df


def load_meanshift_results_for_quantile(
    meanshift_dir: str | Path, quantile: float
) -> pd.DataFrame:
    """
    Load MeanShift sweep results for a fixed quantile.

    Mirrors `load_dbscan_results_for_epsilon`, but filters on `quantile` and
    derives `epoch` from the model identifier when needed.
    """
    metrics_df = load_meanshift_sweep_csvs(meanshift_dir)
    if metrics_df.empty:
        return metrics_df

    if "quantile" not in metrics_df.columns:
        return pd.DataFrame()

    q_col = metrics_df["quantile"]
    try:
        q_numeric = pd.to_numeric(q_col, errors="coerce")
        mask = q_numeric.notna() & np.isclose(q_numeric.to_numpy(), float(quantile), rtol=1e-6, atol=1e-12)
        df = metrics_df[mask].copy()
    except Exception:
        df = metrics_df[metrics_df["quantile"] == quantile].copy()

    if "model" in df.columns:
        df = df[df["model"].notna()]
    elif "model_file" in df.columns:
        df = df[df["model_file"].notna()]

    if "epoch" not in df.columns:
        if "model" in df.columns:
            df["epoch"] = df["model"].apply(_extract_epoch)
        elif "model_file" in df.columns:
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
    if "coverage_ground_truth" in df.columns:
        ax.plot(
            df[x],
            df["coverage_ground_truth"],
            linewidth=2,
            label="ground-truth coverage",
        )

    if "n_noise" in df.columns and "n_embeddings" in df.columns:
        noise_frac = df["n_noise"] / df["n_embeddings"].clip(lower=1)
        ax.plot(df[x], noise_frac, linewidth=2, label="noise fraction")

    ax.set_xlabel(x)
    ax.set_ylabel("ground-truth coverage / noise fraction")
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


def plot_dbscan_epsilon_sweep(
    metrics_df: pd.DataFrame,
    *,
    total_emails: int | None = None,
    title_prefix: str = "",
):
    """
    Existing convenience wrapper: returns (ground-truth coverage, combined metrics, n_clusters).
    """
    if metrics_df.empty:
        return []

    eps = metrics_df["epsilon"] if "epsilon" in metrics_df.columns else None
    _ = eps
    prefix = f"{title_prefix} " if title_prefix else ""
    figs = []

    figs.append(
        plot_coverage_and_noise_fraction(
            metrics_df,
            x="epsilon",
            total_items=total_emails,
            title=f"{prefix}Ground-truth coverage vs epsilon",
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


def plot_dbscan_scores_vs_epsilon(
    metrics_df: pd.DataFrame,
    *,
    title_prefix: str = "",
):
    """
    homogeneity/completeness/v_measure vs epsilon (silhouette excluded).
    """
    if metrics_df.empty or "epsilon" not in metrics_df.columns:
        return None
    prefix = f"{title_prefix} " if title_prefix else ""
    return plot_metric_lines(
        metrics_df,
        x="epsilon",
        metrics=("homogeneity", "completeness", "v_measure"),
        title=f"{prefix}Homogeneity / Completeness / V-measure vs epsilon",
        ylabel="score",
    )


def plot_dbscan_silhouette_vs_epsilon(
    metrics_df: pd.DataFrame,
    *,
    title_prefix: str = "",
):
    """
    silhouette vs epsilon.
    """
    if metrics_df.empty or "epsilon" not in metrics_df.columns:
        return None
    prefix = f"{title_prefix} " if title_prefix else ""
    return plot_metric_lines(
        metrics_df,
        x="epsilon",
        metrics=("silhouette",),
        title=f"{prefix}Silhouette vs epsilon",
        ylabel="silhouette",
    )


def plot_meanshift_quantile_sweep(ms_df: pd.DataFrame, *, title: str = "MeanShift metrics vs quantile"):
    """
    Backwards-compatible function: homogeneity/completeness/v_measure vs quantile.
    (kept from the original messy_plot_code.py).
    """
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


def plot_meanshift_quantile_sweep_all(
    ms_df: pd.DataFrame,
    *,
    total_emails: int | None = None,
    title_prefix: str = "",
):
    """
    MeanShift quantile sweep plots matching your requested set:
      1) coverage vs quantile
      2) homogeneity/completeness/v_measure vs quantile (in one plot)
      3) silhouette vs quantile
      4) number of clusters vs quantile
    Returns a list of (fig, ax) in the same order.
    """
    if ms_df.empty or "quantile" not in ms_df.columns:
        return []

    prefix = f"{title_prefix} " if title_prefix else ""
    if "quantile" in ms_df.columns:
        ms_df = ms_df.sort_values("quantile")

    figs = []
    figs.append(
        plot_coverage_and_noise_fraction(
            ms_df,
            x="quantile",
            total_items=total_emails,
            title=f"{prefix}Ground-truth coverage vs quantile",
        )
    )
    figs.append(
        plot_metric_lines(
            ms_df,
            x="quantile",
            metrics=("homogeneity", "completeness", "v_measure"),
            title=f"{prefix}Homogeneity / Completeness / V-measure vs quantile",
        )
    )
    figs.append(
        plot_metric_lines(
            ms_df,
            x="quantile",
            metrics=("silhouette",),
            title=f"{prefix}Silhouette vs quantile",
            ylabel="silhouette",
        )
    )
    figs.append(plot_n_clusters(ms_df, x="quantile", title=f"{prefix}Number of clusters vs quantile"))
    return figs


def plot_meanshift_metrics_vs_epoch_at_quantile(
    df: pd.DataFrame,
    *,
    quantile: float,
    model_name: str = "",
    total_emails: int | None = None,
):
    """
    MeanShift "locked quantile" plots across epochs (when multiple checkpoints/epochs exist).

    Returns:
      list[(fig, ax)] in a stable order:
        1) homogeneity/completeness/v_measure/silhouette vs epoch
        2) coverage and noise fraction vs epoch (noise fraction if total_emails is provided)
    """
    if df.empty:
        return []

    df = df.copy()

    if "quantile" in df.columns:
        df = df[df["quantile"] == quantile]

    if "epoch" not in df.columns:
        if "model" in df.columns:
            df["epoch"] = df["model"].apply(_extract_epoch)
        elif "model_file" in df.columns:
            df["epoch"] = df["model_file"].apply(_extract_epoch)

    if "epoch" in df.columns:
        df = df.sort_values("epoch")

    plots: list[tuple[plt.Figure, plt.Axes]] = []

    fig1, ax1 = plot_metric_lines(
        df,
        x="epoch",
        metrics=("homogeneity", "completeness", "v_measure", "silhouette"),
        title=f"Metrics vs epoch at quantile={quantile} ({model_name})",
        ylabel="score",
        figsize=(6, 3),
    )
    plots.append((fig1, ax1))

    fig2, ax2 = plt.subplots(figsize=(6, 3))
    if "coverage_ground_truth" in df.columns:
        ax2.plot(
            df["epoch"],
            df["coverage_ground_truth"],
            linewidth=2,
            label="ground-truth coverage",
        )
    if "n_noise" in df.columns and "n_embeddings" in df.columns:
        noise_frac = df["n_noise"] / df["n_embeddings"].clip(lower=1)
        ax2.plot(df["epoch"], noise_frac, linewidth=2, label="noise fraction")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("ground-truth coverage / noise fraction")
    title2 = f"Ground-truth coverage & noise vs epoch at quantile={quantile} ({model_name})"
    ax2.set_title(title2)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    plots.append((fig2, ax2))

    return plots


def make_best_model_dbscan_epsilon_metrics_csv(
    db_scan_dir: str | Path,
    run_name: str,
    *,
    model_file: str = "best_model.pt",
):
    """
    Load DBSCAN epsilon sweep CSVs, filter to `model_file` rows (when present),
    and save a single CSV aggregating best-model metrics across epsilon values.
    """
    metrics_df = load_dbscan_sweep_csvs(db_scan_dir, model_file=model_file)
    if metrics_df.empty:
        return metrics_df, None

    out_path = Path(db_scan_dir) / f"best_model_{run_name}_epsilon_metrics.csv"
    saved_path = save_dataframe(metrics_df, out_path)
    return metrics_df, saved_path


def plot_dbscan_metrics_vs_epoch_at_epsilon(
    df: pd.DataFrame,
    *,
    epsilon: float,
    model_name: str = "",
    total_emails: int | None = None,
):
    """
    Plot DBSCAN metrics vs `epoch` for a fixed epsilon.

    Returns:
      list[(fig, ax)] in a stable order:
        1) homogeneity/completeness/v_measure/silhouette vs epoch
        2) coverage and noise fraction vs epoch (noise fraction if total_emails is provided)
    """
    if df.empty:
        return []

    df = df.copy()

    if "epsilon" in df.columns:
        df = df[df["epsilon"] == epsilon]

    if "epoch" not in df.columns:
        if "model" in df.columns:
            df["epoch"] = df["model"].apply(_extract_epoch)
        elif "model_file" in df.columns:
            df["epoch"] = df["model_file"].apply(_extract_epoch)

    if "epoch" in df.columns:
        df = df.sort_values("epoch")

    plots: list[tuple[plt.Figure, plt.Axes]] = []

    fig1, ax1 = plot_metric_lines(
        df,
        x="epoch",
        metrics=("homogeneity", "completeness", "v_measure", "silhouette"),
        title=f"Metrics vs epoch at epsilon={epsilon} ({model_name})",
        ylabel="score",
        figsize=(6, 3),
    )
    plots.append((fig1, ax1))

    fig2, ax2 = plt.subplots(figsize=(6, 3))
    if "coverage_ground_truth" in df.columns:
        ax2.plot(
            df["epoch"],
            df["coverage_ground_truth"],
            linewidth=2,
            label="ground-truth coverage",
        )
    if "n_noise" in df.columns and "n_embeddings" in df.columns:
        noise_frac = df["n_noise"] / df["n_embeddings"].clip(lower=1)
        ax2.plot(df["epoch"], noise_frac, linewidth=2, label="noise fraction")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("ground-truth coverage / noise fraction")
    title2 = f"Ground-truth coverage & noise vs epoch at epsilon={epsilon} ({model_name})"
    ax2.set_title(title2)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    plots.append((fig2, ax2))

    return plots


__all__ = [
    "save_dataframe",
    "load_dbscan_sweep_csvs",
    "load_meanshift_sweep_csvs",
    "load_dbscan_results_for_epsilon",
    "load_meanshift_results_for_quantile",
    "plot_coverage_and_noise_fraction",
    "plot_metric_lines",
    "plot_n_clusters",
    "plot_dbscan_epsilon_sweep",
    "plot_dbscan_scores_vs_epsilon",
    "plot_dbscan_silhouette_vs_epsilon",
    "plot_meanshift_quantile_sweep",
    "plot_meanshift_quantile_sweep_all",
    "plot_meanshift_metrics_vs_epoch_at_quantile",
    "plot_dbscan_metrics_vs_epoch_at_epsilon",
    "make_best_model_dbscan_epsilon_metrics_csv",
]

