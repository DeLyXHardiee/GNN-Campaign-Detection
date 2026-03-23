"""
Feature-set clustering pipeline.

Coordinates grid-search clustering over all feature-set JSON files produced by
`feature_set_extraction.feature_set_extraction`.  Delegates clustering and metric
computation to `clustering.clusteringMetrics` (shared module) and feature-vector
preprocessing to `feature_set_extraction.cluster_comparison.clusteringCommonFunctions`.
"""

import json
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

from clustering.clusteringMetrics import (
    extract_ground_truth_labels,
    run_db_scan_analysis,
    run_meanshift_analysis,
)
from feature_set_extraction.cluster_comparison.clusteringCommonFunctions import (
    preprocess_for_clustering,
    record_cluster_id,
    remove_outliers_from_matrix,
    save_clusters_to_json,
)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

FEATURE_SETS = ["FS1", "FS2", "FS3", "FS4", "FS5", "FS6", "FS7"]

_PACKAGE_DIR = Path(__file__).resolve().parent.parent  # core/feature_set_extraction/


def _featuresets_dir() -> Path:
    return _PACKAGE_DIR / "output" / "featuresets"


def _results_dir() -> Path:
    d = _PACKAGE_DIR / "output" / "fsclusters" / "results"
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
) -> dict[str, np.ndarray]:
    """Preprocess records into an external_id -> embedding dict."""
    idxs = [record_cluster_id(r) for r in records]
    X, _ = preprocess_for_clustering(
        records,
        max_tfidf_features,
        n_components=n_components,
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
    # shared
    n_components_values: list[int] | None = None,
    max_tfidf_features: int | None = None,
    remove_outliers: bool = True,
    outlier_contamination: float = 0.05,
) -> None:
    """
    Run DBSCAN and Mean Shift grid searches over all feature sets, computing
    clustering metrics via the shared `clustering.clusteringMetrics` helpers.

    Results are appended to:
      core/feature_set_extraction/output/fsclusters/results/dbscan_scores.txt
      core/feature_set_extraction/output/fsclusters/results/meanshift_scores.txt
    """
    eps_values = eps_values or [1, 1.5, 2]
    quantile_values = quantile_values or [0.25]
    n_components_values = n_components_values or [1000]

    if not ground_truth_json or not Path(ground_truth_json).exists():
        raise FileNotFoundError(
            f"Ground truth JSON not found: {ground_truth_json}"
        )
    ground_truth_labels = extract_ground_truth_labels(ground_truth_json)
    print(
        f"Ground truth loaded: {len(ground_truth_labels)} emails in "
        f"{len(set(ground_truth_labels.values()))} clusters"
    )

    results_dir = _results_dir()

    # ------------------------------------------------------------------ DBSCAN
    print(f"\n{'='*80}")
    print("DBSCAN Parameter Grid Search")
    print(f"Testing {len(eps_values)} eps values: {eps_values}")
    print(f"Testing {len(n_components_values)} SVD components: {n_components_values}")
    print(f"Total configurations: {len(eps_values) * len(n_components_values)}")
    print(f"{'='*80}\n")

    dbscan_scores_path = results_dir / "dbscan_scores.txt"
    with open(dbscan_scores_path, "a", encoding="utf-8") as score_f:
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
                        records, max_tfidf_features, n_components,
                        remove_outliers, outlier_contamination,
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
                        f"coverage={metrics['coverage']:.4f}, "
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
    with open(meanshift_scores_path, "a", encoding="utf-8") as score_f:
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
                        records, max_tfidf_features, n_components,
                        remove_outliers, outlier_contamination,
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
                        f"coverage={metrics['coverage']:.4f}, "
                        f"n={metrics['n_samples']}, "
                        f"bandwidth={metrics.get('bandwidth')}"
                    )
                    print(metric_text)
                    score_f.write(metric_text + "\n")

    print(f"\n{'='*80}")
    print("Mean Shift grid search complete!")
    print(f"Results saved to: {meanshift_scores_path}")
    print(f"{'='*80}\n")

    print(f"\n{'='*80}")
    print("All grid searches complete!")
    print(f"Review homogeneity scores to find optimal parameters:")
    print(f"  - DBSCAN:     {dbscan_scores_path}")
    print(f"  - Mean Shift: {meanshift_scores_path}")
    print(f"{'='*80}")
