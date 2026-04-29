from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from config.pipeline_config import (
    GnnPathLayout,
    gnn_path_layout_from_pipeline,
    load_pipeline_config,
    resolve_project_path,
)
from core.clustering.clusteringMetrics import extract_ground_truth_labels
from src.plots.clustering_plot_utils import (
    load_dbscan_results_for_epsilon,
    load_dbscan_sweep_csvs,
    load_meanshift_results_for_quantile,
    load_meanshift_sweep_csvs,
    plot_coverage_and_noise_fraction,
    plot_dbscan_metrics_vs_epoch_at_epsilon,
    plot_dbscan_scores_vs_epsilon,
    plot_dbscan_silhouette_vs_epsilon,
    plot_meanshift_metrics_vs_epoch_at_quantile,
    plot_meanshift_quantile_sweep_all,
    plot_n_clusters,
)


def _save_fig(fig, path: Path, *, dpi: int = 150) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def run_clustering_plot_stage(
    *,
    output_dir: str | Path,
    total_emails: int | None = None,
    dpi: int = 150,
    path_layout: GnnPathLayout | None = None,
) -> dict[str, Any]:
    """
    Read clustering sweep CSVs and generate plots for seed_candidate_workflow.

    Expected clustering output (written by `run_clustering_stage`); subdirs match
    ``pipeline_config.json`` ``gnn.clustering_subdir`` and ``gnn.clustering_plots_subdir``.
    """
    cfg = load_pipeline_config()
    layout = path_layout or gnn_path_layout_from_pipeline(cfg)
    output_dir_p = Path(output_dir)
    clustering_out = output_dir_p / layout.clustering_subdir
    plots_out = clustering_out / layout.clustering_plots_subdir
    plots_out.mkdir(parents=True, exist_ok=True)

    training_cfg = cfg.get("training", {})
    model_save_name = training_cfg.get("model_save_name", "best_model.pt")

    # cluster_stage writes the chosen best locked parameters here.
    stage_result_path = clustering_out / layout.stage_result_json
    best_locked_params: dict[str, dict[str, Any]] = {}
    if stage_result_path.exists():
        try:
            stage_result = json.loads(stage_result_path.read_text(encoding="utf-8"))
            best_locked_params = stage_result.get("best_locked_params", {}) or {}
        except Exception:
            best_locked_params = {}

    if total_emails is None:
        gt_rel = cfg.get("datasets", {}).get("ground_truth_json")
        gt_path = resolve_project_path(gt_rel) if gt_rel else None
        if gt_path:
            ground_truth_labels = extract_ground_truth_labels(gt_path)
            total_emails = len(ground_truth_labels)

    saved: list[str] = []

    # ---- DBSCAN plots (vs epsilon) ----
    dbscan_dir = clustering_out / "dbscan"
    if dbscan_dir.exists():
        dbscan_df = load_dbscan_sweep_csvs(dbscan_dir, model_file=model_save_name)

        if dbscan_df.empty:
            pass
        else:
            # coverage vs epsilon
            fig, _ax = plot_coverage_and_noise_fraction(
                dbscan_df,
                x="epsilon",
                total_items=total_emails,
                title="DBSCAN: ground-truth coverage vs epsilon",
            )
            saved.append(
                _save_fig(fig, plots_out / "dbscan_coverage_vs_epsilon.png", dpi=dpi)
            )

            # score (homogeneity / completeness / v-measure) vs epsilon
            res_scores = plot_dbscan_scores_vs_epsilon(
                dbscan_df,
                title_prefix="DBSCAN: ",
            )
            if res_scores is not None:
                fig, _ax = res_scores
                saved.append(_save_fig(fig, plots_out / "dbscan_scores_vs_epsilon.png", dpi=dpi))

            # silhouette vs epsilon
            res_sil = plot_dbscan_silhouette_vs_epsilon(
                dbscan_df,
                title_prefix="DBSCAN: ",
            )
            if res_sil is not None:
                fig, _ax = res_sil
                saved.append(
                    _save_fig(fig, plots_out / "dbscan_silhouette_vs_epsilon.png", dpi=dpi)
                )

            # n_clusters vs epsilon
            fig, _ax = plot_n_clusters(
                dbscan_df,
                x="epsilon",
                title="DBSCAN: num clusters vs epsilon",
            )
            saved.append(_save_fig(fig, plots_out / "dbscan_n_clusters_vs_epsilon.png", dpi=dpi))

            # score vs epoch at the selected best locked epsilon
            best_eps = None
            if "dbscan" in best_locked_params and "epsilon" in best_locked_params["dbscan"]:
                best_eps = float(best_locked_params["dbscan"]["epsilon"])
            if best_eps is not None:
                df_eps = load_dbscan_results_for_epsilon(dbscan_dir, epsilon=best_eps)
                if not df_eps.empty and "model" in df_eps.columns:
                    # Keep only epoch checkpoints (exclude `best_model` row).
                    df_eps = df_eps[df_eps["model"].astype(str).str.contains("model_epoch_")].copy()
                plots = plot_dbscan_metrics_vs_epoch_at_epsilon(
                    df_eps,
                    epsilon=float(best_eps),
                    model_name=f"dbscan@eps={best_eps}",
                    total_emails=total_emails,
                )
                for idx, (fig_i, _ax_i) in enumerate(plots):
                    suffix = "metrics" if idx == 0 else "coverage_noise"
                    fname = f"dbscan_{suffix}_vs_epoch_at_epsilon_{str(best_eps).replace('.','_')}.png"
                    saved.append(_save_fig(fig_i, plots_out / fname, dpi=dpi))

    # ---- MeanShift plots (vs quantile) ----
    meanshift_dir = clustering_out / "meanshift"
    if meanshift_dir.exists():
        ms_df = load_meanshift_sweep_csvs(meanshift_dir, model_file=model_save_name)
        figs = plot_meanshift_quantile_sweep_all(ms_df, total_emails=total_emails, title_prefix="MeanShift: ")
        for i, (fig, _ax) in enumerate(figs or []):
            fname_map = {
                0: "meanshift_coverage_vs_quantile.png",
                1: "meanshift_scores_vs_quantile.png",
                2: "meanshift_silhouette_vs_quantile.png",
                3: "meanshift_n_clusters_vs_quantile.png",
            }
            saved.append(_save_fig(fig, plots_out / fname_map.get(i, f"meanshift_plot_{i}.png"), dpi=dpi))

        # score vs epoch at the selected best locked quantile
        best_q = None
        if "meanshift" in best_locked_params and "quantile" in best_locked_params["meanshift"]:
            best_q = float(best_locked_params["meanshift"]["quantile"])
        if best_q is not None:
            df_q = load_meanshift_results_for_quantile(meanshift_dir, quantile=best_q)
            if not df_q.empty and "model" in df_q.columns:
                df_q = df_q[df_q["model"].astype(str).str.contains("model_epoch_")].copy()
            plots = plot_meanshift_metrics_vs_epoch_at_quantile(
                df_q,
                quantile=float(best_q),
                model_name=f"meanshift@q={best_q}",
                total_emails=total_emails,
            )
            for idx, (fig_i, _ax_i) in enumerate(plots):
                suffix = "metrics" if idx == 0 else "coverage_noise"
                fname = f"meanshift_{suffix}_vs_epoch_at_quantile_{str(best_q).replace('.','_')}.png"
                saved.append(_save_fig(fig_i, plots_out / fname, dpi=dpi))

    result = {"plots_dir": str(plots_out), "saved_plots": saved}
    (plots_out / layout.stage_result_json).write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    return result


__all__ = ["run_clustering_plot_stage"]

