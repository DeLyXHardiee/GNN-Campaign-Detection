"""
Feature-set clustering pipeline.

Coordinates grid-search clustering over all feature-set JSON files produced by
`feature_set_extraction.feature_set_extraction`.  Delegates clustering and metric
computation to `clustering.clusteringMetrics` (shared module) and feature-vector
preprocessing to `feature_set_extraction.cluster_comparison.clusteringCommonFunctions`.
"""

import json
import os
from datetime import datetime
from pathlib import Path

import config.blas_env  # noqa: F401 — before NumPy

import numpy as np

from clustering.clusteringMetrics import (
    extract_ground_truth_labels,
    run_db_scan_analysis,
    run_hdbscan_analysis,
    run_meanshift_analysis,
)
from config.pipeline_config import load_pipeline_config, output_runs_parent_from_pipeline
from config.run_output_paths import resolve_session_run_output_dir

from feature_set_extraction.cluster_comparison.clusteringCommonFunctions import (
    preprocess_for_clustering,
    record_cluster_id,
    remove_outliers_from_matrix,
)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

FEATURE_SETS = ["FS1", "FS2", "FS3", "FS4", "FS5", "FS6", "FS7"]

_PACKAGE_DIR = Path(__file__).resolve().parent.parent  # core/feature_set_extraction/


def _featuresets_dir() -> Path:
    return _PACKAGE_DIR / "output" / "featuresets"


def _results_dir(run_output_dir: Path) -> Path:
    d = run_output_dir / "featureset_clustering" / "results"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _load_records(dataset_base: str, fs_name: str):
    """Return (records, path) or (None, path) when the file is missing."""
    path = _featuresets_dir() / f"{dataset_base}-{fs_name}.json"
    if not path.exists():
        return None, path
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f), path


def _build_embedding_map(
    records,
    max_tfidf_features,
    n_components,
    remove_outliers: bool,
    outlier_contamination: float,
    embeddings_output_dir: str | os.PathLike | None = None,
) -> dict[str, np.ndarray]:
    """Preprocess records into an external_id -> embedding dict."""
    idxs = [record_cluster_id(r) for r in records]
    X, _ = preprocess_for_clustering(
        records,
        max_tfidf_features,
        n_components=n_components,
        embeddings_output_dir=embeddings_output_dir,
    )
    if remove_outliers:
        X, keep_mask, removed = remove_outliers_from_matrix(
            X, contamination=outlier_contamination
        )
        idxs = [idx for idx, keep in zip(idxs, keep_mask) if keep]
        print(
            f"  Removed {removed} outliers "
            f"(contamination={outlier_contamination})"
        )
    return {eid: np.asarray(vec, dtype=np.float64) for eid, vec in zip(idxs, X)}


def _write_run_header(score_f, algorithm: str, params: str) -> None:
    score_f.write("\n" + "=" * 80 + "\n")
    score_f.write(f"{algorithm} Run - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    score_f.write(f"Parameters: {params}\n")
    score_f.write("=" * 80 + "\n\n")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_featureset_clustering(
    dataset_base: str = "synthetic_email_dataset_50",
    ground_truth_json: str | None = None,
    # DBSCAN
    eps_values: list[float] | None = None,
    min_samples: int = 5,
    # Mean Shift
    quantile_values: list[float] | None = None,
    n_samples: int = 500,
    # HDBSCAN
    hdbscan_enabled: bool = True,
    min_cluster_size_values: list[int] | None = None,
    hdbscan_min_samples: int | None = None,
    # shared
    n_components_values: list[int] | None = None,
    max_tfidf_features: int | None = None,
    remove_outliers: bool = True,
    outlier_contamination: float = 0.05,
    embeddings_output_dir: str | os.PathLike | None = None,
    run_output_dir: str | os.PathLike | None = None,
) -> None:
    """
    Run DBSCAN, Mean Shift, and (optionally) HDBSCAN grid searches over all feature
    sets, computing clustering metrics via the shared `clustering.clusteringMetrics`
    helpers (including `run_hdbscan_analysis`, aligned with the GNN clustering stage).

    Score files are written under
    ``<run_output_dir>/featureset_clustering/results/`` (see ``output_runs_root`` /
    :func:`config.run_output_paths.resolve_session_run_output_dir`). When
    ``run_output_dir`` is omitted, the session run directory is used (same folder as
    GNN train/eval/clustering when run in one process).

    Per-invocation score files (overwrite each run):
      dbscan_scores.txt, meanshift_scores.txt, hdbscan_scores.txt
    """
    eps_values = eps_values or [1, 1.5, 2]
    quantile_values = quantile_values or [0.25]
    n_components_values = n_components_values or [1000]
    min_cluster_size_values = min_cluster_size_values or [2]

    if not ground_truth_json or not Path(ground_truth_json).exists():
        raise FileNotFoundError(
            f"Ground truth JSON not found: {ground_truth_json}"
        )
    ground_truth_labels = extract_ground_truth_labels(ground_truth_json)
    print(
        f"Ground truth loaded: {len(ground_truth_labels)} emails in "
        f"{len(set(ground_truth_labels.values()))} clusters"
    )

    if run_output_dir is None:
        cfg_fc = load_pipeline_config()
        run_output_path = resolve_session_run_output_dir(
            cfg_fc,
            runs_root=output_runs_parent_from_pipeline(cfg_fc),
        )
    else:
        run_output_path = Path(run_output_dir)
    print(f"Run output directory: {run_output_path.resolve()}")
    results_dir = _results_dir(run_output_path)

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

        for eps in eps_values:
            for n_components in n_components_values:
                print(f"\n{'='*80}")
                print(
                    f"eps={eps}, n_components={n_components}, "
                    f"min_samples={min_samples}, "
                    f"remove_outliers={remove_outliers}"
                )
                print(f"{'='*80}")

                for fs_name in FEATURE_SETS:
                    records, fs_path = _load_records(dataset_base, fs_name)
                    if records is None:
                        msg = f"{fs_name}: SKIPPED (file not found: {fs_path})"
                        print(msg)
                        score_f.write(msg + "\n")
                        continue

                    embedding_map = _build_embedding_map(
                        records,
                        max_tfidf_features,
                        n_components,
                        remove_outliers,
                        outlier_contamination,
                        embeddings_output_dir=embeddings_output_dir,
                    )
                    if not embedding_map:
                        msg = f"{fs_name}: SKIPPED (empty embedding map)"
                        print(msg)
                        score_f.write(msg + "\n")
                        continue

                    metrics = run_db_scan_analysis(
                        id_to_embedding_map=embedding_map,
                        ground_truth_labels=ground_truth_labels,
                        epsilon=eps,
                        min_samples=min_samples,
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

        for quantile in quantile_values:
            for n_components in n_components_values:
                print(f"\n{'='*80}")
                print(
                    f"quantile={quantile}, n_components={n_components}, "
                    f"n_samples={n_samples}, "
                    f"remove_outliers={remove_outliers}"
                )
                print(f"{'='*80}")

                for fs_name in FEATURE_SETS:
                    records, fs_path = _load_records(dataset_base, fs_name)
                    if records is None:
                        msg = f"{fs_name}: SKIPPED (file not found: {fs_path})"
                        print(msg)
                        score_f.write(msg + "\n")
                        continue

                    embedding_map = _build_embedding_map(
                        records,
                        max_tfidf_features,
                        n_components,
                        remove_outliers,
                        outlier_contamination,
                        embeddings_output_dir=embeddings_output_dir,
                    )
                    if not embedding_map:
                        msg = f"{fs_name}: SKIPPED (empty embedding map)"
                        print(msg)
                        score_f.write(msg + "\n")
                        continue

                    metrics = run_meanshift_analysis(
                        id_to_embedding_map=embedding_map,
                        ground_truth_labels=ground_truth_labels,
                        quantile=quantile,
                        n_samples=n_samples,
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
                f"max_tfidf_features={max_tfidf_features}, "
                f"n_components_values={n_components_values}, "
                f"remove_outliers={remove_outliers}, "
                f"outlier_contamination={outlier_contamination}",
            )

            for min_cluster_size in min_cluster_size_values:
                for n_components in n_components_values:
                    print(f"\n{'='*80}")
                    print(
                        f"min_cluster_size={min_cluster_size}, "
                        f"n_components={n_components}, "
                        f"hdbscan_min_samples={hdbscan_min_samples}, "
                        f"remove_outliers={remove_outliers}"
                    )
                    print(f"{'='*80}")

                    for fs_name in FEATURE_SETS:
                        records, fs_path = _load_records(dataset_base, fs_name)
                        if records is None:
                            msg = f"{fs_name}: SKIPPED (file not found: {fs_path})"
                            print(msg)
                            score_f.write(msg + "\n")
                            continue

                        embedding_map = _build_embedding_map(
                            records,
                            max_tfidf_features,
                            n_components,
                            remove_outliers,
                            outlier_contamination,
                            embeddings_output_dir=embeddings_output_dir,
                        )
                        if not embedding_map:
                            msg = f"{fs_name}: SKIPPED (empty embedding map)"
                            print(msg)
                            score_f.write(msg + "\n")
                            continue

                        metrics = run_hdbscan_analysis(
                            id_to_embedding_map=embedding_map,
                            ground_truth_labels=ground_truth_labels,
                            min_cluster_size=min_cluster_size,
                            min_samples=hdbscan_min_samples,
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

    print(f"\n{'='*80}")
    print("All grid searches complete!")
    print("Review homogeneity scores to find optimal parameters:")
    print(f"  - DBSCAN:     {dbscan_scores_path}")
    print(f"  - Mean Shift: {meanshift_scores_path}")
    if hdbscan_enabled:
        print(f"  - HDBSCAN:    {hdbscan_scores_path}")
    print(f"{'='*80}")
