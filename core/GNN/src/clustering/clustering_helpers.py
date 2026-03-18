import json
from collections import defaultdict

import numpy as np
import torch
from sklearn.cluster import DBSCAN
from sklearn.metrics import (
    calinski_harabasz_score,
    completeness_score,
    davies_bouldin_score,
    homogeneity_score,
    silhouette_score,
    v_measure_score,
)

@torch.no_grad()
def extract_email_embeddings(model, data, device):
    """
    Run the model and return email-node embeddings keyed by graph-local index.

    PyG orders email nodes 0 .. n-1; row i of h['email'] is the embedding for
    that node. Keys in the returned dict are those indices; each value is a
    1-D copy of that row (so mutating it does not affect the original stack).
    """
    model.eval()
    x_dict = data.to(device).x_dict
    edge_index_dict = data.to(device).edge_index_dict
    h = model(x_dict, edge_index_dict)
    email_vecs = h['email'].cpu().numpy()
    return {i: email_vecs[i].copy() for i in range(len(email_vecs))}

def extract_ground_truth_labels(path_or_data):
    """
    Extract ground truth labels from a JSON file or loaded dict with format:
    {"clusters": {"label_store_1/49": [{"record_id": str, ...}, ...], ...}}
    Cluster keys are normalized by stripping the "label_store_*/" prefix.
    Returns dict mapping record_id -> cluster_id (int when possible).
    """
    with open(path_or_data, "r", encoding="utf-8") as f:
        data = json.load(f)
    label_map = {}
    clusters = data.get("clusters", {})
    for raw_key, records in clusters.items():
        # Remove "label_store_*/" prefix; use the part after the last "/" as cluster id
        cluster_id_str = raw_key.split("/")[-1] if "/" in raw_key else raw_key
        try:
            campaign_id = int(cluster_id_str)
        except ValueError:
            campaign_id = cluster_id_str
        for rec in records:
            record_id = rec.get("external_id")
            if record_id is None:
                continue
            try:
                rid = int(record_id)
            except (ValueError, TypeError):
                rid = record_id
            label_map[rid] = campaign_id
    return label_map

def compute_internal_metrics(id_to_embedding_map, labels):
    """
    Compute clustering internal metrics (silhouette, DB index, CH index).

    Assumes ``labels[i]`` corresponds to the i-th id in ``sorted(id_to_embedding_map.keys())``.
    """
    sorted_ids = sorted(id_to_embedding_map.keys())
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


def compute_all_metrics(id_to_embedding_map, labels, ground_truth_labels):
    """
    Compute both internal + external clustering metrics and return one dict.
    """
    sorted_ids = sorted(id_to_embedding_map.keys())
    labels = np.asarray(labels)
    if len(labels) != len(sorted_ids):
        raise ValueError(
            f"len(labels)={len(labels)} must equal len(id_to_embedding_map)={len(sorted_ids)}"
        )

    internal = compute_internal_metrics(id_to_embedding_map, labels)

    # External metrics only need aligned (true, predicted) labels for ids that
    # exist in ground truth. No need to construct a clusters dict.
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

def run_clustering_analysis_for_embeddings(
    id_to_embedding_map,
    ground_truth_labels,
    epsilon,
    min_samples=5,
):

    sorted_ids = sorted(id_to_embedding_map.keys())
    embeddings = np.stack(
        [np.asarray(id_to_embedding_map[i], dtype=np.float64) for i in sorted_ids]
    )

    clusterer = DBSCAN(eps=epsilon, min_samples=min_samples, metric="euclidean")
    labels = clusterer.fit_predict(embeddings)

    metrics = compute_all_metrics(id_to_embedding_map, labels, ground_truth_labels)
    metrics["epsilon"] = float(epsilon)
    metrics["min_samples"] = int(min_samples)
    return metrics



def run_clustering_analysis_across_models(data, device, run_dir, epsilon_values, ground_truth_labels):
    from src.model_io import load_full_run

    run_path = Path("../models") / run_dir
    model_files = sorted(run_path.glob("*.pt"))

    results = []
    for eps in epsilon_values:
        print(f"### Clustering with epsilon={eps} ###")
        for model_file in model_files:
            print(f"Evaluating {model_file.name}...")
            model, predictor, checkpoint = load_model_checkpoint(device=device, metadata=data.metadata(), filename=model_file)

            id_to_emb = extract_email_embeddings(model, data, device)
            sorted_ids = sorted(id_to_emb.keys())
            embeddings = np.stack([id_to_emb[i] for i in sorted_ids])

            clusterer = DBSCAN(eps=eps, min_samples=5, metric="euclidean")
            labels = clusterer.fit_predict(embeddings)

            metrics = compute_all_metrics(id_to_emb, labels, ground_truth_labels)

            results.append({
                "model_file": model_file.name,
                "epsilon": eps,
                "silhouette": metrics["silhouette"],
                "db_index": metrics["db_index"],
                "ch_index": metrics["ch_index"],
                "homogeneity": metrics["homogeneity"],
                "n_samples": metrics["n_samples"],
                "completeness": metrics["completeness"],
                "v_measure": metrics["v_measure"],
                "n_clusters": metrics["n_clusters"],
                "coverage": metrics["coverage"],
                "n_noise": metrics["n_noise"],
            })

    return results
