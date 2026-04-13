import json
from typing import Any, Callable

import numpy as np
from sklearn.cluster import DBSCAN, MeanShift, estimate_bandwidth
from sklearn.metrics import (
    calinski_harabasz_score,
    completeness_score,
    davies_bouldin_score,
    homogeneity_score,
    silhouette_score,
    v_measure_score,
)


def extract_ground_truth_labels(path: str) -> dict[str, Any]:
    """
    Extract ground truth labels from a JSON file with format:
    {"clusters": {"label_store_1/49": [{"external_id": str, ...}, ...], ...}}

    Cluster keys are normalized by stripping the "label_store_*/" prefix.
    Returns dict mapping external_id -> cluster_id (int when possible).
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    label_map: dict[str, Any] = {}
    duplicate_external_ids: set[str] = set()
    clusters = data.get("clusters", {})
    for raw_key, emails in clusters.items():
        # Remove "label_store_*/" prefix; use the part after the last "/" as cluster id
        cluster_id_str = raw_key.split("/")[-1] if "/" in raw_key else raw_key
        try:
            campaign_id: Any = int(cluster_id_str)
        except ValueError:
            campaign_id = cluster_id_str
        for email in emails:
            email_id = email.get("external_id")
            if email_id is None:
                continue
            email_id_str = str(email_id)
            if email_id_str in label_map:
                duplicate_external_ids.add(email_id_str)
                continue
            label_map[email_id_str] = campaign_id
    if duplicate_external_ids:
        sample = sorted(duplicate_external_ids)[:10]
        suffix = "..." if len(duplicate_external_ids) > 10 else ""
        raise ValueError(
            "Duplicate `external_id` values found in ground truth JSON. "
            f"Examples: {sample}{suffix}"
        )
    return label_map


def compute_internal_metrics(
    id_to_embedding_map: dict[str, np.ndarray], labels: list[int] | np.ndarray
) -> dict[str, Any]:
    """
    Compute clustering internal metrics (silhouette, DB index, CH index).

    Assumes ``labels[i]`` corresponds to the i-th id in ``sorted(id_to_embedding_map.keys())``.
    """
    sorted_ids = sorted(id_to_embedding_map.keys(), key=str)
    labels = np.asarray(labels)
    if len(labels) != len(sorted_ids):
        raise ValueError(
            f"len(labels)={len(labels)} must equal len(id_to_embedding_map)={len(sorted_ids)}"
        )

    embeddings = np.stack(
        [np.asarray(id_to_embedding_map[k], dtype=np.float64) for k in sorted_ids]
    )

    mask = labels != -1
    usable = labels[mask]
    uniq = np.unique(usable) if usable.size else []
    if mask.sum() > 1 and len(uniq) > 1:
        silhouette = float(silhouette_score(embeddings[mask], usable))
        db_index = float(davies_bouldin_score(embeddings[mask], usable))
        ch_index = float(calinski_harabasz_score(embeddings[mask], usable))
    else:
        silhouette, db_index, ch_index = -1.0, float("inf"), 0.0

    return {"silhouette": silhouette, "db_index": db_index, "ch_index": ch_index}


def compute_external_metrics(true_labels: list[Any], predicted_labels: list[int]) -> dict[str, Any]:
    """
    Compute clustering external metrics (homogeneity, completeness, v-measure).

    Expects aligned label vectors (same length, same order).
    """
    n_samples = len(true_labels)
    if n_samples < 2:
        return {
            "homogeneity": 0.0,
            "completeness": 0.0,
            "v_measure": 0.0,
            "n_samples": int(n_samples),
        }

    homogeneity = homogeneity_score(true_labels, predicted_labels)
    completeness = completeness_score(true_labels, predicted_labels)
    v_measure = v_measure_score(true_labels, predicted_labels)

    return {
        "homogeneity": float(homogeneity),
        "completeness": float(completeness),
        "v_measure": float(v_measure),
        "n_samples": int(n_samples),
    }


def _aligned_true_predicted_labels(
    sorted_ids: list[str],
    labels: list[int] | np.ndarray,
    ground_truth_labels: dict[str, Any],
) -> tuple[list[Any], list[int]]:
    """
    Align predicted cluster labels with ground-truth labels (by external_id).

    Skips:
    - predicted noise points where predicted label == -1
    - ids not present in ground_truth_labels
    """
    gt_get: Callable[[str], Any] = ground_truth_labels.get
    true_labels: list[Any] = []
    predicted_labels: list[int] = []
    for eid, lab in zip(sorted_ids, labels):
        if lab == -1:
            continue
        true = gt_get(eid)
        if true is None:
            continue
        true_labels.append(true)
        predicted_labels.append(int(lab))
    return true_labels, predicted_labels


# Backwards-compatible alias for a historical misspelling.
def alligned_true_predictived_labels(
    sorted_ids: list[str],
    labels: list[int] | np.ndarray,
    ground_truth_labels: dict[str, Any],
) -> tuple[list[Any], list[int]]:
    return _aligned_true_predicted_labels(
        sorted_ids=sorted_ids,
        labels=labels,
        ground_truth_labels=ground_truth_labels,
    )


def compute_all_metrics(
    id_to_embedding_map: dict[str, np.ndarray],
    labels: list[int] | np.ndarray,
    ground_truth_labels: dict[str, Any],
) -> dict[str, Any]:
    """
    Compute both internal + external clustering metrics and return one dict.
    """
    sorted_ids = sorted(id_to_embedding_map.keys(), key=str)
    labels = np.asarray(labels)
    if len(labels) != len(sorted_ids):
        raise ValueError(
            f"len(labels)={len(labels)} must equal len(id_to_embedding_map)={len(sorted_ids)}"
        )

    internal = compute_internal_metrics(id_to_embedding_map, labels)

    true_labels, predicted_labels = _aligned_true_predicted_labels(
        sorted_ids=sorted_ids,
        labels=labels,
        ground_truth_labels=ground_truth_labels,
    )
    external = compute_external_metrics(true_labels, predicted_labels)

    n_clusters = int(len(set(labels)) - (1 if -1 in labels else 0))
    n_noise = int((labels == -1).sum())
    n_embeddings = int(len(sorted_ids))
    n_non_noise = int(n_embeddings - n_noise)

    # Two different "coverage" definitions:
    # 1) Ground-truth coverage: among all ground-truth-labeled items, how many were predicted as non-noise.
    #coverage_ground_truth = external["n_samples"] / max(1, len(ground_truth_labels))
    coverage_ground_truth = len(true_labels) / max(1, len(ground_truth_labels))
    # 2) All-items coverage: among all embeddings, how many were predicted as non-noise (regardless of ground-truth presence).
    coverage_all = n_non_noise / max(1, n_embeddings)

    return {
        **internal,
        **external,
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "n_non_noise": n_non_noise,
        "n_embeddings": n_embeddings,
        "coverage_ground_truth": coverage_ground_truth,
        "coverage_all": coverage_all,
    }


def _emb_matrix_from_id_to_embedding(
    id_to_embedding_map: dict[str, np.ndarray],
) -> tuple[list[str], np.ndarray]:
    sorted_ids = sorted(id_to_embedding_map.keys(), key=str)
    embeddings = np.stack(
        [np.asarray(id_to_embedding_map[k], dtype=np.float64) for k in sorted_ids]
    )
    return sorted_ids, embeddings


def run_db_scan_analysis(
    id_to_embedding_map: dict[str, np.ndarray],
    ground_truth_labels: dict[str, Any],
    epsilon: float,
    min_samples: int = 5,
) -> dict[str, Any]:
    sorted_ids, embeddings = _emb_matrix_from_id_to_embedding(id_to_embedding_map)
    clusterer = DBSCAN(eps=epsilon, min_samples=min_samples, metric="euclidean")
    labels = clusterer.fit_predict(embeddings)
    metrics = compute_all_metrics(id_to_embedding_map, labels, ground_truth_labels)
    metrics["clustering_type"] = "dbscan"
    metrics["epsilon"] = float(epsilon)
    metrics["min_samples"] = int(min_samples)
    metrics["n_embeddings"] = int(len(sorted_ids))
    return metrics


def run_meanshift_analysis(
    id_to_embedding_map: dict[str, np.ndarray],
    ground_truth_labels: dict[str, Any],
    quantile: float,
    n_samples: int | None = None,
) -> dict[str, Any]:
    sorted_ids, embeddings = _emb_matrix_from_id_to_embedding(id_to_embedding_map)
    n_embeddings = int(len(sorted_ids))
    bw = quantile#estimate_bandwidth(embeddings, quantile=float(quantile), n_samples=n_samples)
    clusterer = MeanShift(bandwidth=bw, bin_seeding=True)
    try:
        labels = clusterer.fit_predict(embeddings)
    except ValueError as exc:
        # sklearn MeanShift (bin_seeding): no grid seed has any point within bandwidth.
        if "No point was within" not in str(exc):
            raise
        metrics: dict[str, Any] = {
            "silhouette": -1.0,
            "db_index": float("inf"),
            "ch_index": 0.0,
            "homogeneity": 0.0,
            "completeness": 0.0,
            "v_measure": 0.0,
            "n_clusters": 0,
            "n_noise": n_embeddings,
            "n_non_noise": 0,
            "n_embeddings": n_embeddings,
            "coverage_ground_truth": 0.0,
            "coverage_all": 0.0,
            "clustering_type": "meanshift",
            "quantile": float(quantile),
            "bandwidth": float(bw) if bw is not None else None,
            "clustering_error": str(exc),
        }
        metrics["n_samples"] = None if n_samples is None else int(n_samples)
        return metrics

    metrics = compute_all_metrics(id_to_embedding_map, labels, ground_truth_labels)
    metrics["clustering_type"] = "meanshift"
    metrics["quantile"] = float(quantile)
    metrics["n_samples"] = None if n_samples is None else int(n_samples)
    metrics["bandwidth"] = float(bw) if bw is not None else None
    metrics["n_embeddings"] = n_embeddings
    return metrics


def fit_predict_labels(
    id_to_embedding_map: dict[str, np.ndarray],
    algorithm: str,
    *,
    epsilon: float | None = None,
    min_samples: int = 5,
    quantile: float | None = None,
    n_samples: int | None = None,
    min_cluster_size: int | None = None,
    hdbscan_min_samples: int | None = None,
) -> tuple[list[str], np.ndarray]:
    """
    Run clustering without ground-truth metrics; return (sorted_external_ids, label_array).

    ``algorithm`` is one of: ``dbscan``, ``meanshift``, ``hdbscan``.
    On failure (e.g. MeanShift bin seeding), returns all labels -1.
    """
    algo = str(algorithm).lower().strip()
    sorted_ids, embeddings = _emb_matrix_from_id_to_embedding(id_to_embedding_map)
    n_embeddings = int(len(sorted_ids))

    if algo == "dbscan":
        if epsilon is None:
            raise ValueError("fit_predict_labels(dbscan) requires epsilon=")
        clusterer = DBSCAN(eps=float(epsilon), min_samples=int(min_samples), metric="euclidean")
        labels = clusterer.fit_predict(embeddings)
        return sorted_ids, np.asarray(labels)

    if algo == "meanshift":
        if quantile is None:
            raise ValueError("fit_predict_labels(meanshift) requires quantile=")
        bw = estimate_bandwidth(embeddings, quantile=float(quantile), n_samples=n_samples)
        clusterer = MeanShift(bandwidth=bw, bin_seeding=True)
        try:
            labels = clusterer.fit_predict(embeddings)
        except ValueError as exc:
            if "No point was within" not in str(exc):
                raise
            return sorted_ids, np.full(n_embeddings, -1, dtype=np.int64)
        return sorted_ids, np.asarray(labels)

    if algo == "hdbscan":
        if min_cluster_size is None:
            raise ValueError("fit_predict_labels(hdbscan) requires min_cluster_size=")
        import hdbscan  # type: ignore

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=int(min_cluster_size),
            min_samples=None if hdbscan_min_samples is None else int(hdbscan_min_samples),
        )
        labels = clusterer.fit_predict(embeddings)
        return sorted_ids, np.asarray(labels)

    raise ValueError(f"Unknown algorithm={algorithm!r}; expected dbscan, meanshift, hdbscan.")


def run_hdbscan_analysis(
    id_to_embedding_map: dict[str, np.ndarray],
    ground_truth_labels: dict[str, Any],
    min_cluster_size: int,
    min_samples: int | None = None,
) -> dict[str, Any]:
    # Lazy import: `hdbscan` is a native extension and can crash some environments
    # during import; we only need it for the HDBSCAN analysis path.
    import hdbscan  # type: ignore

    sorted_ids, embeddings = _emb_matrix_from_id_to_embedding(id_to_embedding_map)
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=int(min_cluster_size),
        min_samples=None if min_samples is None else int(min_samples),
    )
    labels = clusterer.fit_predict(embeddings)
    metrics = compute_all_metrics(id_to_embedding_map, labels, ground_truth_labels)
    metrics["clustering_type"] = "hdbscan"
    metrics["min_cluster_size"] = int(min_cluster_size)
    metrics["min_samples"] = None if min_samples is None else int(min_samples)
    metrics["n_embeddings"] = int(len(sorted_ids))
    return metrics


__all__ = [
    "extract_ground_truth_labels",
    "compute_internal_metrics",
    "compute_external_metrics",
    "alligned_true_predictived_labels",
    "compute_all_metrics",
    "_emb_matrix_from_id_to_embedding",
    "fit_predict_labels",
    "run_db_scan_analysis",
    "run_meanshift_analysis",
    "run_hdbscan_analysis",
]
