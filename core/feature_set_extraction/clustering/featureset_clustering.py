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

import config.blas_env  # noqa: F401 — before NumPy

import numpy as np

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
#FEATURE_SETS = ["FS1", "FS2", "FS3"]#, "FS4", "FS5", "FS6", "FS7"]

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


def _build_featureset_embedding_cache(
    *,
    dataset_base: str,
    n_components_values: list[int],
    max_tfidf_features: int | None,
    remove_outliers: bool,
    outlier_contamination: float,
    embeddings_output_dir: str | os.PathLike | None,
) -> tuple[
    dict[tuple[str, int], dict[str, np.ndarray]],
    dict[tuple[str, int], str],
]:
    """
    Build each (feature set name, n_components) embedding map once.

    Returns ``(cache, skip_messages)``. Keys present in ``skip_messages`` failed
    preprocessing or had no data; successful maps live only in ``cache``.
    """
    cache: dict[tuple[str, int], dict[str, np.ndarray]] = {}
    skip_messages: dict[tuple[str, int], str] = {}

    total = len(n_components_values) * len(FEATURE_SETS)
    done = 0
    print(
        f"\nBuilding embedding cache: {len(n_components_values)} n_components × "
        f"{len(FEATURE_SETS)} feature sets = {total} combinations "
        f"(shared across all clustering sweeps)\n"
    )

    for n_components in n_components_values:
        for fs_name in FEATURE_SETS:
            done += 1
            key = (fs_name, n_components)
            records, fs_path = _load_records(dataset_base, fs_name)
            if records is None:
                skip_messages[key] = (
                    f"{fs_name}: SKIPPED (file not found: {fs_path})"
                )
                print(f"  [{done}/{total}] {skip_messages[key]}")
                continue

            print(
                f"  [{done}/{total}] {fs_name}, n_components={n_components} "
                f"(preprocess + optional outlier removal)…"
            )
            embedding_map = _build_embedding_map(
                records,
                max_tfidf_features,
                n_components,
                remove_outliers,
                outlier_contamination,
                embeddings_output_dir=embeddings_output_dir,
            )
            if not embedding_map:
                skip_messages[key] = f"{fs_name}: SKIPPED (empty embedding map)"
                print(f"  [{done}/{total}] {skip_messages[key]}")
                continue

            cache[key] = embedding_map
            print(
                f"  [{done}/{total}] {fs_name}, n_components={n_components} "
                f"→ cached {len(embedding_map)} embeddings"
            )

    n_ok = len(cache)
    n_skip = len(skip_messages)
    print(
        f"\nEmbedding cache ready: {n_ok} usable, {n_skip} skipped "
        f"(total keys {n_ok + n_skip}).\n"
    )
    return cache, skip_messages


def _embedding_for_sweep(
    cache: dict[tuple[str, int], dict[str, np.ndarray]],
    skip_messages: dict[tuple[str, int], str],
    fs_name: str,
    n_components: int,
) -> tuple[dict[str, np.ndarray] | None, str | None]:
    """Return (embedding_map, None) or (None, message) for score file / print."""
    key = (fs_name, n_components)
    if key in cache:
        return cache[key], None
    if key in skip_messages:
        return None, skip_messages[key]
    return None, f"{fs_name}: SKIPPED (no cache entry for n_components={n_components})"


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
    min_cov_gt: float,
    min_cov_all: float,
    **extra: object,
) -> None:
    if metrics.get("clustering_error"):
        return
    key = _featureset_selection_key(metrics, min_cov_gt, min_cov_all)
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
    dataset_base: str,
    max_tfidf_features: int | None,
    remove_outliers: bool,
    outlier_contamination: float,
    embeddings_output_dir: str | os.PathLike | None,
    min_samples: int,
    n_samples: int,
    hdbscan_min_samples: int | None,
) -> str | None:
    """Rebuild embeddings for the winning config and write ``campaigns_featureset.json``."""
    if not best.get("algorithm"):
        print("Skipping campaigns_featureset.json: no clustering result recorded.")
        return None

    records, _ = _load_records(dataset_base, str(best["fs_name"]))
    if records is None:
        print("Skipping campaigns_featureset.json: feature set records missing.")
        return None

    embedding_map = _build_embedding_map(
        records,
        max_tfidf_features,
        int(best["n_components"]),
        remove_outliers,
        outlier_contamination,
        embeddings_output_dir=embeddings_output_dir,
    )
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
        )
        params = {
            "min_cluster_size": int(best["min_cluster_size"]),
            "min_samples": hdbscan_min_samples,
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
    min_coverage_ground_truth: float = 0.5,
    min_coverage_all: float = 0.5,
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

    best_run: dict = {}

    embedding_cache, embedding_skip_messages = _build_featureset_embedding_cache(
        dataset_base=dataset_base,
        n_components_values=n_components_values,
        max_tfidf_features=max_tfidf_features,
        remove_outliers=remove_outliers,
        outlier_contamination=outlier_contamination,
        embeddings_output_dir=embeddings_output_dir,
    )

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
                    embedding_map, skip_msg = _embedding_for_sweep(
                        embedding_cache,
                        embedding_skip_messages,
                        fs_name,
                        n_components,
                    )
                    if embedding_map is None:
                        msg = skip_msg or (
                            f"{fs_name}: SKIPPED (no cache entry for "
                            f"n_components={n_components})"
                        )
                        print(msg)
                        score_f.write(msg + "\n")
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
                        min_cov_gt=min_coverage_ground_truth,
                        min_cov_all=min_coverage_all,
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
                    embedding_map, skip_msg = _embedding_for_sweep(
                        embedding_cache,
                        embedding_skip_messages,
                        fs_name,
                        n_components,
                    )
                    if embedding_map is None:
                        msg = skip_msg or (
                            f"{fs_name}: SKIPPED (no cache entry for "
                            f"n_components={n_components})"
                        )
                        print(msg)
                        score_f.write(msg + "\n")
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
                        min_cov_gt=min_coverage_ground_truth,
                        min_cov_all=min_coverage_all,
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
                        embedding_map, skip_msg = _embedding_for_sweep(
                            embedding_cache,
                            embedding_skip_messages,
                            fs_name,
                            n_components,
                        )
                        if embedding_map is None:
                            msg = skip_msg or (
                                f"{fs_name}: SKIPPED (no cache entry for "
                                f"n_components={n_components})"
                            )
                            print(msg)
                            score_f.write(msg + "\n")
                            continue

                        metrics = run_hdbscan_analysis(
                            id_to_embedding_map=embedding_map,
                            ground_truth_labels=ground_truth_labels,
                            min_cluster_size=min_cluster_size,
                            min_samples=hdbscan_min_samples,
                        )
                        _update_featureset_best(
                            best_run,
                            algorithm="hdbscan",
                            fs_name=fs_name,
                            n_components=n_components,
                            metrics=metrics,
                            min_cov_gt=min_coverage_ground_truth,
                            min_cov_all=min_coverage_all,
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
        dataset_base=dataset_base,
        max_tfidf_features=max_tfidf_features,
        remove_outliers=remove_outliers,
        outlier_contamination=outlier_contamination,
        embeddings_output_dir=embeddings_output_dir,
        min_samples=min_samples,
        n_samples=n_samples,
        hdbscan_min_samples=hdbscan_min_samples,
    )

    print(f"\n{'='*80}")
    print("All grid searches complete!")
    print("Review homogeneity scores to find optimal parameters:")
    print(f"  - DBSCAN:     {dbscan_scores_path}")
    print(f"  - Mean Shift: {meanshift_scores_path}")
    if hdbscan_enabled:
        print(f"  - HDBSCAN:    {hdbscan_scores_path}")
    print(f"{'='*80}")
