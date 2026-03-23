import json
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import hdbscan
from sklearn.cluster import DBSCAN, MeanShift, estimate_bandwidth
from sklearn.metrics import (
    calinski_harabasz_score,
    completeness_score,
    davies_bouldin_score,
    homogeneity_score,
    silhouette_score,
    v_measure_score,
)

from src.model_io import load_model_checkpoint

@torch.no_grad()
def extract_email_embeddings(model, data, device, external_ids):
    """
    Run the model and return email-node embeddings keyed by email ``external_id``.

    PyG orders email nodes 0 .. n-1; row i of h['email'] is the embedding for
    that node. Pass ``external_ids`` from the graph metadata (e.g.
    metadata["email_attrs"]["external_id"]) when loading the graph; the graph
    itself does not store external_id (PyG loaders require tensor-only node stores).
    """
    model.eval()
    graph = data.to(device)
    x_dict = graph.x_dict
    edge_index_dict = graph.edge_index_dict
    h = model(x_dict, edge_index_dict)
    email_vecs = h["email"].cpu().numpy()

    external_ids = list(external_ids)

    if len(external_ids) != len(email_vecs):
        raise ValueError(
            f"Email external_id length ({len(external_ids)}) does not match number of email embeddings ({len(email_vecs)})."
        )

    # external_id must be unique; otherwise dict keys collide.
    if len(set(map(str, external_ids))) != len(external_ids):
        raise ValueError("Duplicate `external_id` values found on email nodes; cannot build a unique id->embedding map.")

    return {
        str(eid.item() if isinstance(eid, np.generic) else eid): email_vecs[i].copy()
        for i, eid in enumerate(external_ids)
    }

def extract_ground_truth_labels(path):
    """
    Extract ground truth labels from a JSON file with format:
    {"clusters": {"label_store_1/49": [{"external_id": str, ...}, ...], ...}}
    Cluster keys are normalized by stripping the "label_store_*/" prefix.
    Returns dict mapping external_id -> cluster_id (int when possible).
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    label_map = {}
    duplicate_external_ids = set()
    clusters = data.get("clusters", {})
    for raw_key, emails in clusters.items():
        # Remove "label_store_*/" prefix; use the part after the last "/" as cluster id
        cluster_id_str = raw_key.split("/")[-1] if "/" in raw_key else raw_key
        try:
            campaign_id = int(cluster_id_str)
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

def compute_internal_metrics(id_to_embedding_map, labels):
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


def compute_external_metrics(true_labels, predicted_labels):
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


def _aligned_true_predicted_labels(sorted_ids, labels, ground_truth_labels):
    """
    Align predicted cluster labels with ground-truth labels (by external_id).

    Skips:
    - predicted noise points where predicted label == -1
    - ids not present in ground_truth_labels
    """
    gt_get = ground_truth_labels.get
    true_labels = []
    predicted_labels = []
    for eid, lab in zip(sorted_ids, labels):
        if lab == -1:
            continue
        true = gt_get(eid)
        if true is None:
            continue
        true_labels.append(true)
        predicted_labels.append(int(lab))
    return true_labels, predicted_labels


def compute_all_metrics(id_to_embedding_map, labels, ground_truth_labels):
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

    return {
        **internal,
        **external,
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "coverage": external["n_samples"] / max(1, len(ground_truth_labels)),
        "n_embeddings": int(len(sorted_ids)),
    }

def _emb_matrix_from_id_to_embedding(id_to_embedding_map):
    sorted_ids = sorted(id_to_embedding_map.keys(), key=str)
    embeddings = np.stack(
        [np.asarray(id_to_embedding_map[k], dtype=np.float64) for k in sorted_ids]
    )
    return sorted_ids, embeddings


def run_db_scan_analysis(id_to_embedding_map, ground_truth_labels, epsilon, min_samples=5):
    sorted_ids, embeddings = _emb_matrix_from_id_to_embedding(id_to_embedding_map)
    clusterer = DBSCAN(eps=epsilon, min_samples=min_samples, metric="euclidean")
    labels = clusterer.fit_predict(embeddings)
    metrics = compute_all_metrics(id_to_embedding_map, labels, ground_truth_labels)
    metrics["clustering_type"] = "dbscan"
    metrics["epsilon"] = float(epsilon)
    metrics["min_samples"] = int(min_samples)
    metrics["n_embeddings"] = int(len(sorted_ids))
    return metrics


def _meanshift_seeding_failure_metrics(
    *,
    quantile,
    n_samples,
    bandwidth,
    n_embeddings,
    error_message: str,
):
    """Row shape aligned with successful MeanShift sweep rows when sklearn aborts seeding."""
    return {
        "silhouette": float("nan"),
        "db_index": float("nan"),
        "ch_index": float("nan"),
        "homogeneity": float("nan"),
        "completeness": float("nan"),
        "v_measure": float("nan"),
        "n_clusters": -1,
        "n_noise": -1,
        "coverage": 0.0,
        "n_embeddings": int(n_embeddings),
        "clustering_type": "meanshift",
        "quantile": float(quantile),
        "n_samples": None if n_samples is None else int(n_samples),
        "bandwidth": float(bandwidth) if bandwidth is not None else None,
        "clustering_error": error_message,
    }


def run_meanshift_analysis(id_to_embedding_map, ground_truth_labels, quantile, n_samples=None):
    sorted_ids, embeddings = _emb_matrix_from_id_to_embedding(id_to_embedding_map)
    n_emb = int(len(sorted_ids))
    bw = estimate_bandwidth(embeddings, quantile=float(quantile), n_samples=n_samples)
    clusterer = MeanShift(bandwidth=bw, bin_seeding=True)
    try:
        labels = clusterer.fit_predict(embeddings)
    except ValueError as e:
        msg = str(e)
        if "No point was within bandwidth" in msg and "seed" in msg:
            return _meanshift_seeding_failure_metrics(
                quantile=quantile,
                n_samples=n_samples,
                bandwidth=bw,
                n_embeddings=n_emb,
                error_message=msg,
            )
        raise
    metrics = compute_all_metrics(id_to_embedding_map, labels, ground_truth_labels)
    metrics["clustering_type"] = "meanshift"
    metrics["quantile"] = float(quantile)
    metrics["n_samples"] = None if n_samples is None else int(n_samples)
    metrics["bandwidth"] = float(bw) if bw is not None else None
    metrics["n_embeddings"] = n_emb
    return metrics


def run_hdbscan_analysis(id_to_embedding_map, ground_truth_labels, min_cluster_size, min_samples=None):
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


def _collect_clustering_sweep_metrics(id_to_embedding_map, ground_truth_labels, clustering_config):
    algo = str(clustering_config["cluster_algorithm"]).lower()

    if algo == "dbscan":
        eps_values = clustering_config["epsilon_values"]
        min_samples = clustering_config.get("min_samples", 5)
        return [
            run_db_scan_analysis(
                id_to_embedding_map, ground_truth_labels, epsilon=eps, min_samples=min_samples
            )
            for eps in eps_values
        ]

    if algo == "meanshift":
        quantile_values = clustering_config["quantile_values"]
        n_samples = clustering_config.get("n_samples")
        return [
            run_meanshift_analysis(
                id_to_embedding_map, ground_truth_labels, quantile=q, n_samples=n_samples
            )
            for q in quantile_values
        ]

    if algo == "hdbscan":
        mcs_values = clustering_config["min_cluster_size_values"]
        min_samples = clustering_config.get("min_samples")
        return [
            run_hdbscan_analysis(
                id_to_embedding_map,
                ground_truth_labels,
                min_cluster_size=mcs,
                min_samples=min_samples,
            )
            for mcs in mcs_values
        ]

    raise ValueError(f"Unknown cluster_algorithm={algo!r}. Expected: dbscan, meanshift, hdbscan.")


def save_metrics_csv(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("No rows to save.")

    all_keys = set().union(*(r.keys() for r in rows))
    preferred_order = [
        # Identifiers / algorithm
        "model",
        "clustering_type",
        # DBSCAN
        "epsilon",
        "min_samples",
        # MeanShift
        "quantile",
        "n_samples",
        "bandwidth",
        "clustering_error",
        # HDBSCAN
        "min_cluster_size",
        # Internal metrics
        "silhouette",
        "db_index",
        "ch_index",
        # External metrics
        "homogeneity",
        "completeness",
        "v_measure",
        # Coverage / size
        "n_embeddings",
        "n_clusters",
        "n_noise",
        "coverage",
        # External alignment counts
        "n_samples_external",
    ]

    remaining = sorted(k for k in all_keys if k not in preferred_order)
    fieldnames = [k for k in preferred_order if k in all_keys] + remaining
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    return str(path)


def sweep_clustering_for_one_model(
    model,
    data,
    device,
    ground_truth_labels,
    clustering_config,
    output_dir,
    model_column_name="model",
    *,
    email_external_ids,
):
    """
    Run clustering sweep for one model. Writes ``<output_dir>/<model_column_name>_<algo>_sweep.csv``
    (e.g. best_model_dbscan_sweep.csv). Use training.model_save_name stem for consistency.
    email_external_ids: list of external_id per email node from metadata (email_attrs.external_id).
    """
    id_to_emb = extract_email_embeddings(model, data, device, external_ids=email_external_ids)
    cfg = dict(clustering_config)

    rows = _collect_clustering_sweep_metrics(id_to_emb, ground_truth_labels, cfg)
    for r in rows:
        r["model"] = model_column_name

    algo = str(cfg["cluster_algorithm"]).lower()
    output_dir = Path(output_dir)
    csv_path = output_dir / f"{model_column_name}_{algo}_sweep.csv"
    save_metrics_csv(rows, csv_path)
    return {"csv_path": str(csv_path), "rows": rows}


def sweep_clustering_for_many_models(
    data,
    device,
    checkpoints,
    ground_truth_labels,
    clustering_config,
    output_dir,
    *,
    email_external_ids,
):
    """
    Run the same clustering sweep across multiple checkpoint files.

    `checkpoints` should be an iterable of full paths to `.pt` files.
    email_external_ids: list of external_id per email node (e.g. from metadata email_attrs.external_id).
    """
    output_dir = Path(output_dir)
    results = []
    for ckpt in sorted(checkpoints, key=lambda p: str(p)):
        ckpt_path = Path(ckpt).expanduser()
        model, predictor, _checkpoint = load_model_checkpoint(
            device=device,
            metadata=data.metadata(),
            filename=str(ckpt_path),
        )
        _ = predictor
        model_column_name = ckpt_path.stem
        results.append(
            sweep_clustering_for_one_model(
                model=model,
                data=data,
                device=device,
                ground_truth_labels=ground_truth_labels,
                clustering_config=clustering_config,
                output_dir=output_dir,
                model_column_name=model_column_name,
                email_external_ids=email_external_ids,
            )
        )
    return results


def run_locked_param_across_checkpoints(
    data,
    device,
    checkpoints,
    ground_truth_labels,
    clustering_config,
    locked_param_value,
    output_dir,
    *,
    email_external_ids,
):
    """
    Keep one clustering parameter fixed and run it across multiple checkpoints.
    email_external_ids: list of external_id per email node (e.g. from metadata).
    """
    algo = str(clustering_config["cluster_algorithm"]).lower()
    locked_cfg = dict(clustering_config)

    if algo == "dbscan":
        locked_cfg["epsilon_values"] = [locked_param_value]
    elif algo == "meanshift":
        locked_cfg["quantile_values"] = [locked_param_value]
    elif algo == "hdbscan":
        locked_cfg["min_cluster_size_values"] = [locked_param_value]
    else:
        raise ValueError(f"Unknown cluster_algorithm={algo!r}")

    return sweep_clustering_for_many_models(
        data=data,
        device=device,
        checkpoints=checkpoints,
        ground_truth_labels=ground_truth_labels,
        clustering_config=locked_cfg,
        output_dir=output_dir,
        email_external_ids=email_external_ids,
    )
