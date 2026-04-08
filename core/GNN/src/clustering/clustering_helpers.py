import csv
import sys
from pathlib import Path

import numpy as np
import torch

from src.model_io import load_model_checkpoint

# General clustering/metric utilities live in core/clustering/clusteringMetrics.py.
# GNN stage code imports these for ground-truth alignment + metric computation.
_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from core.clustering.clusteringMetrics import (  # noqa: E402
    extract_ground_truth_labels,
    compute_all_metrics,
    compute_external_metrics,
    compute_internal_metrics,
    alligned_true_predictived_labels,
    _emb_matrix_from_id_to_embedding,
    run_db_scan_analysis,
    run_hdbscan_analysis,
    run_meanshift_analysis,
)

@torch.no_grad()
def extract_email_embeddings(model, data, device, external_ids, graph_meta=None):
    """
    Run the model and return per-email embeddings keyed by ``external_id``.

    For standard graphs, row i of ``h['email']`` matches email i.

    For ``email_cluster`` supernode graphs, cluster embeddings are broadcast to each
    member email using ``graph_meta['email_attrs']['email_cluster_index']`` (same
    order as ``external_id``).
    """
    model.eval()
    graph = data.to(device)
    x_dict = graph.x_dict
    edge_index_dict = graph.edge_index_dict
    h = model(x_dict, edge_index_dict)

    meta = graph_meta or {}
    primary = meta.get("primary_ntype")
    if primary is None:
        primary = "email_cluster" if "email_cluster" in h else "email"

    external_ids = list(external_ids)

    if primary == "email_cluster":
        if "email_cluster" not in h:
            raise KeyError("Graph metadata indicates primary_ntype=email_cluster but model output has no 'email_cluster'.")
        cluster_vecs = h["email_cluster"].cpu().numpy()
        ea = meta.get("email_attrs") or {}
        idx_list = ea.get("email_cluster_index")
        if not idx_list or len(idx_list) != len(external_ids):
            raise ValueError(
                "email_cluster supernode graph requires metadata email_attrs.email_cluster_index "
                f"with same length as external_id ({len(external_ids)} vs {len(idx_list or [])})."
            )
        email_vecs = np.stack([cluster_vecs[int(idx_list[i])] for i in range(len(external_ids))], axis=0)
    else:
        if "email" not in h:
            raise KeyError("Model output has no 'email' node type.")
        email_vecs = h["email"].cpu().numpy()
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
        "n_non_noise",
        "coverage_ground_truth",
        "coverage_all",
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
    graph_meta=None,
):
    """
    Run clustering sweep for one model. Writes ``<output_dir>/<model_column_name>_<algo>_sweep.csv``
    (e.g. best_model_dbscan_sweep.csv). Use training.model_save_name stem for consistency.
    email_external_ids: list of external_id per email node from metadata (email_attrs.external_id).
    graph_meta: optional companion .meta.json dict (primary_ntype, email_attrs.email_cluster_index).
    """
    id_to_emb = extract_email_embeddings(
        model, data, device, external_ids=email_external_ids, graph_meta=graph_meta
    )
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
    graph_meta=None,
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
                graph_meta=graph_meta,
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
    graph_meta=None,
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
        graph_meta=graph_meta,
    )
