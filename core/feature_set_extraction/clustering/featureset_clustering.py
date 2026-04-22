"""
Feature-set clustering pipeline.

Coordinates grid-search clustering over all feature-set JSON files produced by
`feature_set_extraction.feature_set_extraction`.  Embeddings are built once per
(feature set, ``n_components``) and reused across DBSCAN / Mean Shift / HDBSCAN
parameter sweeps.  Delegates clustering and metric computation to
`clustering.clusteringMetrics` (shared module) and feature-vector preprocessing to
`feature_set_extraction.cluster_comparison.clusteringCommonFunctions`.
"""

import json
import os
from datetime import datetime
from pathlib import Path

from clustering.clusteringMetrics import (
    extract_ground_truth_labels,
    fit_predict_labels,
    run_db_scan_analysis,
    run_hdbscan_analysis,
    run_meanshift_analysis,
)
try:
    from core.visualization.campaign_utils import build_campaign_artifact_payload
except ModuleNotFoundError:
    from visualization.campaign_utils import build_campaign_artifact_payload
from config.pipeline_config import (
    FEATURESET_CLUSTERING_CONFIG,
    PIPELINE_CONFIG,
    output_runs_parent_from_pipeline,
)
from config.run_output_paths import resolve_session_run_output_dir
from feature_set_extraction.clustering.featureset_embeddings import (
    get_embedding_map,
    warm_embedding_cache,
)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _results_dir(run_output_dir: Path) -> Path:
    d = run_output_dir / "featureset_clustering" / "results"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _write_run_header(score_f, algorithm: str, params: str) -> None:
    score_f.write("\n" + "=" * 80 + "\n")
    score_f.write(f"{algorithm} Run - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    score_f.write(f"Parameters: {params}\n")
    score_f.write("=" * 80 + "\n\n")


def _featureset_selection_key(
    metrics: dict,
    min_cov_gt: float,
    min_cov_all: float,
) -> tuple[bool, float]:
    """Prefer rows meeting coverage thresholds, then higher v_measure."""
    strict = (
        float(metrics.get("coverage_ground_truth", 0.0)) >= min_cov_gt
        and float(metrics.get("coverage_all", 0.0)) >= min_cov_all
    )
    vm = float(metrics.get("v_measure", 0.0))
    return (strict, vm)


def _update_featureset_best(
    best: dict,
    *,
    algorithm: str,
    fs_name: str,
    n_components: int,
    metrics: dict,
    **extra: object,
) -> None:
    if metrics.get("clustering_error"):
        return
    key = _featureset_selection_key(
        metrics,
        FEATURESET_CLUSTERING_CONFIG.min_coverage_ground_truth,
        FEATURESET_CLUSTERING_CONFIG.min_coverage_all,
    )
    cur_key = best.get("selection_key")
    if cur_key is None or key > cur_key:
        best.clear()
        best.update(
            algorithm=algorithm,
            fs_name=fs_name,
            n_components=n_components,
            selection_key=key,
            v_measure=float(metrics.get("v_measure", 0.0)),
            **extra,
        )


def _write_featureset_campaigns_json(
    *,
    best: dict,
    run_output_path: Path,
) -> str | None:
    """Rebuild embeddings for the winning config and write ``campaigns_featureset.json``."""
    if not best.get("algorithm"):
        print("Skipping campaigns_featureset.json: no clustering result recorded.")
        return None

    min_samples = FEATURESET_CLUSTERING_CONFIG.min_samples
    n_samples = FEATURESET_CLUSTERING_CONFIG.n_samples
    hdbscan_min_samples = FEATURESET_CLUSTERING_CONFIG.hdbscan_min_samples
    hdbscan_metric = FEATURESET_CLUSTERING_CONFIG.hdbscan_metric

    embedding_map, skip_msg = get_embedding_map(str(best["fs_name"]), int(best["n_components"]))
    if embedding_map is None:
        print(f"Skipping campaigns_featureset.json: {skip_msg}")
        return None
    algo = str(best["algorithm"])
    if algo == "dbscan":
        sorted_ids, labels = fit_predict_labels(
            embedding_map,
            "dbscan",
            epsilon=float(best["epsilon"]),
            min_samples=int(min_samples),
        )
        params = {"epsilon": float(best["epsilon"]), "min_samples": int(min_samples)}
    elif algo == "meanshift":
        sorted_ids, labels = fit_predict_labels(
            embedding_map,
            "meanshift",
            quantile=float(best["quantile"]),
            n_samples=n_samples,
        )
        params = {"quantile": float(best["quantile"]), "n_samples": n_samples}
    elif algo == "hdbscan":
        sorted_ids, labels = fit_predict_labels(
            embedding_map,
            "hdbscan",
            min_cluster_size=int(best["min_cluster_size"]),
            hdbscan_min_samples=hdbscan_min_samples,
            hdbscan_metric=hdbscan_metric,
        )
        params = {
            "min_cluster_size": int(best["min_cluster_size"]),
            "min_samples": hdbscan_min_samples,
            "metric": hdbscan_metric,
        }
    else:
        return None

    payload = build_campaign_artifact_payload(
        solution="featureset",
        algorithm=algo,
        sorted_ids=sorted_ids,
        labels=labels,
        params=params,
        metrics={"v_measure": float(best.get("v_measure", 0.0))},
        feature_set=str(best["fs_name"]),
        n_components=int(best["n_components"]),
    )
    out_dir = run_output_path / "featureset_clustering"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "campaigns_featureset.json"
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote featureset campaign assignments: {out_path}")
    return str(out_path)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_featureset_clustering(
    run_output_dir: str | os.PathLike | None = None,
) -> None:
    """
    Run DBSCAN, Mean Shift, and (optionally) HDBSCAN grid searches over all feature
    sets, computing clustering metrics via the shared `clustering.clusteringMetrics`
    helpers (including `run_hdbscan_analysis`, aligned with the GNN clustering stage).

    Preprocessed embeddings are computed once per (feature set, ``n_components``) and
    reused for every clustering hyperparameter combination.

    Score files are written under
    ``<run_output_dir>/featureset_clustering/results/`` (see ``output_runs_root`` /
    :func:`config.run_output_paths.resolve_session_run_output_dir`). When
    ``run_output_dir`` is omitted, the session run directory is used (same folder as
    GNN train/eval/clustering when run in one process).

    Per-invocation score files (overwrite each run):
      dbscan_scores.txt, meanshift_scores.txt, hdbscan_scores.txt
    """

    ground_truth_json = FEATURESET_CLUSTERING_CONFIG.ground_truth_json
    if not ground_truth_json:
        raise ValueError("pipeline_config datasets.ground_truth_json is required for featureset clustering.")
    eps_values = FEATURESET_CLUSTERING_CONFIG.eps_values
    min_samples = FEATURESET_CLUSTERING_CONFIG.min_samples
    quantile_values = FEATURESET_CLUSTERING_CONFIG.quantile_values
    n_samples = FEATURESET_CLUSTERING_CONFIG.n_samples
    hdbscan_enabled = FEATURESET_CLUSTERING_CONFIG.hdbscan_enabled
    min_cluster_size_values = FEATURESET_CLUSTERING_CONFIG.min_cluster_size_values
    hdbscan_min_samples = FEATURESET_CLUSTERING_CONFIG.hdbscan_min_samples
    feature_sets = FEATURESET_CLUSTERING_CONFIG.feature_sets
    hdbscan_metric = FEATURESET_CLUSTERING_CONFIG.hdbscan_metric
    n_components_values = FEATURESET_CLUSTERING_CONFIG.n_components_values
    max_tfidf_features = FEATURESET_CLUSTERING_CONFIG.max_tfidf_features
    remove_outliers = FEATURESET_CLUSTERING_CONFIG.remove_outliers
    outlier_contamination = FEATURESET_CLUSTERING_CONFIG.outlier_contamination

    if not Path(ground_truth_json).exists():
        raise FileNotFoundError(
            f"Ground truth JSON not found: {ground_truth_json}"
        )
    ground_truth_labels = extract_ground_truth_labels(ground_truth_json)
    print(
        f"Ground truth loaded: {len(ground_truth_labels)} emails in "
        f"{len(set(ground_truth_labels.values()))} clusters"
    )

    if run_output_dir is None:
        cfg_fc = PIPELINE_CONFIG
        run_output_path = resolve_session_run_output_dir(
            cfg_fc,
            runs_root=output_runs_parent_from_pipeline(cfg_fc),
        )
    else:
        run_output_path = Path(run_output_dir)
    print(f"Run output directory: {run_output_path.resolve()}")
    results_dir = _results_dir(run_output_path)

    best_run: dict = {}

    warm_embedding_cache(feature_sets, n_components_values)

    # ------------------------------------------------------------------ DBSCAN
    print(f"\n{'='*80}")
    print("DBSCAN Parameter Grid Search")
    print(f"Testing {len(eps_values)} eps values: {eps_values}")
    print(f"Testing {len(n_components_values)} SVD components: {n_components_values}")
    print(f"Total configurations: {len(eps_values) * len(n_components_values)}")
    print(f"{'='*80}\n")

    dbscan_scores_path = results_dir / "dbscan_scores.txt"
    with open(dbscan_scores_path, "w", encoding="utf-8") as score_f:
        _write_run_header(
            score_f,
            "DBSCAN",
            f"eps_values={eps_values}, min_samples={min_samples}, "
            f"max_tfidf_features={max_tfidf_features}, "
            f"n_components_values={n_components_values}, "
            f"remove_outliers={remove_outliers}, "
            f"outlier_contamination={outlier_contamination}",
        )

        for fs_name in feature_sets:
            for eps in eps_values:
                for n_components in n_components_values:
                    print(f"\n{'='*80}")
                    print(
                        f"{fs_name} | eps={eps}, n_components={n_components}, "
                        f"min_samples={min_samples}, "
                        f"remove_outliers={remove_outliers}"
                    )
                    print(f"{'='*80}")

                    embedding_map, skip_msg = get_embedding_map(fs_name, n_components)
                    if embedding_map is None:
                        print(skip_msg)
                        score_f.write(skip_msg + "\n")
                        continue

                    metrics = run_db_scan_analysis(
                        id_to_embedding_map=embedding_map,
                        ground_truth_labels=ground_truth_labels,
                        epsilon=eps,
                        min_samples=min_samples,
                    )
                    _update_featureset_best(
                        best_run,
                        algorithm="dbscan",
                        fs_name=fs_name,
                        n_components=n_components,
                        metrics=metrics,
                        epsilon=eps,
                    )
                    metric_text = (
                        f"{fs_name} | eps={eps} | n_components={n_components} | "
                        f"silhouette={metrics['silhouette']:.4f}, "
                        f"H={metrics['homogeneity']:.4f}, "
                        f"C={metrics['completeness']:.4f}, "
                        f"V={metrics['v_measure']:.4f}, "
                        f"clusters={metrics['n_clusters']}, "
                        f"noise={metrics['n_noise']}, "
                        f"coverage_ground_truth={metrics['coverage_ground_truth']:.4f}, "
                        f"n={metrics['n_samples']}"
                    )
                    print(metric_text)
                    score_f.write(metric_text + "\n")

    print(f"\n{'='*80}")
    print("DBSCAN grid search complete!")
    print(f"Results saved to: {dbscan_scores_path}")
    print(f"{'='*80}\n")

    # --------------------------------------------------------------- Mean Shift
    print(f"\n{'='*80}")
    print("Mean Shift Parameter Grid Search")
    print(f"Testing {len(quantile_values)} quantile values: {quantile_values}")
    print(f"Testing {len(n_components_values)} SVD components: {n_components_values}")
    print(f"Total configurations: {len(quantile_values) * len(n_components_values)}")
    print(f"{'='*80}\n")

    meanshift_scores_path = results_dir / "meanshift_scores.txt"
    with open(meanshift_scores_path, "w", encoding="utf-8") as score_f:
        _write_run_header(
            score_f,
            "Mean Shift",
            f"quantile_values={quantile_values}, n_samples={n_samples}, "
            f"max_tfidf_features={max_tfidf_features}, "
            f"n_components_values={n_components_values}, "
            f"remove_outliers={remove_outliers}, "
            f"outlier_contamination={outlier_contamination}",
        )

        for fs_name in feature_sets:
            for quantile in quantile_values:
                for n_components in n_components_values:
                    print(f"\n{'='*80}")
                    print(
                        f"{fs_name} | quantile={quantile}, n_components={n_components}, "
                        f"n_samples={n_samples}, "
                        f"remove_outliers={remove_outliers}"
                    )
                    print(f"{'='*80}")

                    embedding_map, skip_msg = get_embedding_map(fs_name, n_components)
                    if embedding_map is None:
                        print(skip_msg)
                        score_f.write(skip_msg + "\n")
                        continue

                    metrics = run_meanshift_analysis(
                        id_to_embedding_map=embedding_map,
                        ground_truth_labels=ground_truth_labels,
                        quantile=quantile,
                        n_samples=n_samples,
                    )
                    _update_featureset_best(
                        best_run,
                        algorithm="meanshift",
                        fs_name=fs_name,
                        n_components=n_components,
                        metrics=metrics,
                        quantile=quantile,
                    )
                    metric_text = (
                        f"{fs_name} | quantile={quantile} | n_components={n_components} | "
                        f"silhouette={metrics['silhouette']:.4f}, "
                        f"H={metrics['homogeneity']:.4f}, "
                        f"C={metrics['completeness']:.4f}, "
                        f"V={metrics['v_measure']:.4f}, "
                        f"clusters={metrics['n_clusters']}, "
                        f"noise={metrics['n_noise']}, "
                        f"coverage_ground_truth={metrics['coverage_ground_truth']:.4f}, "
                        f"n={metrics['n_samples']}, "
                        f"bandwidth={metrics.get('bandwidth')}"
                    )
                    print(metric_text)
                    score_f.write(metric_text + "\n")

    print(f"\n{'='*80}")
    print("Mean Shift grid search complete!")
    print(f"Results saved to: {meanshift_scores_path}")
    print(f"{'='*80}\n")

    # --------------------------------------------------------------- HDBSCAN
    hdbscan_scores_path = results_dir / "hdbscan_scores.txt"
    if hdbscan_enabled:
        print(f"\n{'='*80}")
        print("HDBSCAN Parameter Grid Search")
        print(
            f"Testing {len(min_cluster_size_values)} min_cluster_size values: "
            f"{min_cluster_size_values}"
        )
        print(f"Testing {len(n_components_values)} SVD components: {n_components_values}")
        print(
            f"Total configurations: "
            f"{len(min_cluster_size_values) * len(n_components_values)}"
        )
        print(f"{'='*80}\n")

        with open(hdbscan_scores_path, "w", encoding="utf-8") as score_f:
            _write_run_header(
                score_f,
                "HDBSCAN",
                f"min_cluster_size_values={min_cluster_size_values}, "
                f"hdbscan_min_samples={hdbscan_min_samples}, "
                f"metric={hdbscan_metric}, "
                f"max_tfidf_features={max_tfidf_features}, "
                f"n_components_values={n_components_values}, "
                f"remove_outliers={remove_outliers}, "
                f"outlier_contamination={outlier_contamination}",
            )

            for fs_name in feature_sets:
                for min_cluster_size in min_cluster_size_values:
                    for n_components in n_components_values:
                        print(f"\n{'='*80}")
                        print(
                            f"{fs_name} | min_cluster_size={min_cluster_size}, "
                            f"n_components={n_components}, "
                            f"hdbscan_min_samples={hdbscan_min_samples}, "
                            f"remove_outliers={remove_outliers}"
                        )
                        print(f"{'='*80}")

                        embedding_map, skip_msg = get_embedding_map(fs_name, n_components)
                        if embedding_map is None:
                            print(skip_msg)
                            score_f.write(skip_msg + "\n")
                            continue

                        metrics = run_hdbscan_analysis(
                            id_to_embedding_map=embedding_map,
                            ground_truth_labels=ground_truth_labels,
                            min_cluster_size=min_cluster_size,
                            min_samples=hdbscan_min_samples,
                            metric=hdbscan_metric,
                        )
                        _update_featureset_best(
                            best_run,
                            algorithm="hdbscan",
                            fs_name=fs_name,
                            n_components=n_components,
                            metrics=metrics,
                            min_cluster_size=min_cluster_size,
                        )
                        ms_note = metrics.get("min_samples")
                        metric_text = (
                            f"{fs_name} | min_cluster_size={min_cluster_size} | "
                            f"n_components={n_components} | "
                            f"silhouette={metrics['silhouette']:.4f}, "
                            f"H={metrics['homogeneity']:.4f}, "
                            f"C={metrics['completeness']:.4f}, "
                            f"V={metrics['v_measure']:.4f}, "
                            f"clusters={metrics['n_clusters']}, "
                            f"noise={metrics['n_noise']}, "
                            f"coverage_ground_truth={metrics['coverage_ground_truth']:.4f}, "
                            f"n={metrics['n_samples']}, "
                            f"hdbscan_min_samples={ms_note}"
                        )
                        print(metric_text)
                        score_f.write(metric_text + "\n")

        print(f"\n{'='*80}")
        print("HDBSCAN grid search complete!")
        print(f"Results saved to: {hdbscan_scores_path}")
        print(f"{'='*80}\n")

    _write_featureset_campaigns_json(
        best=best_run,
        run_output_path=run_output_path,
    )

    print(f"\n{'='*80}")
    print("All grid searches complete!")
    print("Review homogeneity scores to find optimal parameters:")
    print(f"  - DBSCAN:     {dbscan_scores_path}")
    print(f"  - Mean Shift: {meanshift_scores_path}")
    if hdbscan_enabled:
        print(f"  - HDBSCAN:    {hdbscan_scores_path}")
    print(f"{'='*80}")
