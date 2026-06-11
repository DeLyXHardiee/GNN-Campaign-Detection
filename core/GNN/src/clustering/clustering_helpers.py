import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

from src.model_io import load_model_checkpoint

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


def _l2_normalize_vector(vec: np.ndarray) -> np.ndarray:
    """Unit L2 norm; if norm is zero, return the vector unchanged."""
    v = np.asarray(vec, dtype=np.float64).reshape(-1)
    n = float(np.linalg.norm(v))
    if n <= 0.0:
        return v.copy()
    return (v / n).astype(np.float64)


def build_hybrid_embedding_map(
    id_to_gnn: dict[str, np.ndarray],
    id_to_raw: dict[str, np.ndarray],
    *,
    weight_raw: float,
    weight_gnn: float,
) -> dict[str, np.ndarray]:
    """
    Per email: L2-normalize raw and GNN vectors, scale by weights, concatenate,
    then L2-normalize the concatenated vector.
    """
    wr = float(weight_raw)
    wg = float(weight_gnn)
    if id_to_gnn.keys() != id_to_raw.keys():
        missing_g = set(id_to_raw) - set(id_to_gnn)
        missing_r = set(id_to_gnn) - set(id_to_raw)
        raise ValueError(
            "Hybrid embeddings require identical key sets for GNN and raw maps; "
            f"missing_from_gnn={len(missing_g)} missing_from_raw={len(missing_r)}"
        )
    out: dict[str, np.ndarray] = {}
    for eid in id_to_gnn:
        r_hat = _l2_normalize_vector(id_to_raw[eid])
        g_hat = _l2_normalize_vector(id_to_gnn[eid])
        concat = np.concatenate([wr * r_hat, wg * g_hat], axis=0)
        out[eid] = _l2_normalize_vector(concat)
    return out

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

    if len(set(map(str, external_ids))) != len(external_ids):
        raise ValueError("Duplicate `external_id` values found on email nodes; cannot build a unique id->embedding map.")

    return {
        str(eid.item() if isinstance(eid, np.generic) else eid): email_vecs[i].copy()
        for i, eid in enumerate(external_ids)
    }


def extract_raw_email_embeddings(data, external_ids):
    """
    Return raw graph email-node features keyed by email ``external_id``.

    This is the pre-training baseline representation directly from the graph
    (`data["email"].x`) with no encoder forward pass.
    """
    if "email" not in data.node_types:
        raise ValueError("Node type 'email' not found in graph.")
    email_x = data["email"].x
    if email_x is None:
        raise ValueError("data['email'].x is missing; cannot build raw embedding baseline.")
    if not isinstance(email_x, torch.Tensor):
        raise ValueError(f"Expected tensor data['email'].x, got {type(email_x).__name__}.")

    email_vecs = email_x.detach().cpu().numpy()
    external_ids = list(external_ids)
    if len(external_ids) != len(email_vecs):
        raise ValueError(
            f"Email external_id length ({len(external_ids)}) does not match number of raw email rows ({len(email_vecs)})."
        )
    if len(set(map(str, external_ids))) != len(external_ids):
        raise ValueError("Duplicate `external_id` values found on email nodes; cannot build a unique id->embedding map.")

    return {
        str(eid.item() if isinstance(eid, np.generic) else eid): email_vecs[i].copy()
        for i, eid in enumerate(external_ids)
    }


def load_transformer_subject_body_embeddings_from_cache(
    *,
    embeddings_json_path: str | Path,
) -> dict[str, np.ndarray]:
    """
    Load untouched transformer subject+body embeddings from embeddings cache JSON.

    Expected cache format is utils.embeddings/embedder.py output:
      {
        "by_key": {
          "<external_id>": {"subj": [...], "body": [...], ...},
          ...
        },
        ...
      }

    Returns id->concat(subject, body) without any projection/reduction.
    """
    p = Path(embeddings_json_path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"Transformer embeddings cache not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    by_key = payload.get("by_key")
    if not isinstance(by_key, dict):
        raise ValueError(
            f"Invalid transformer embeddings cache format at {p}: missing dict `by_key`."
        )

    id_to_emb: dict[str, np.ndarray] = {}
    for k, v in by_key.items():
        if not isinstance(v, dict):
            continue
        subj = np.asarray(v.get("subj") or [], dtype=np.float32).reshape(-1)
        body = np.asarray(v.get("body") or [], dtype=np.float32).reshape(-1)
        if subj.size == 0 and body.size == 0:
            continue
        eid = str(v.get("external_id") or k)
        id_to_emb[eid] = np.concatenate([subj, body], axis=0)

    if not id_to_emb:
        raise ValueError(
            f"No subject/body vectors found in transformer embeddings cache: {p}"
        )
    if len(set(id_to_emb.keys())) != len(id_to_emb):
        raise ValueError("Duplicate external_id entries in transformer embeddings cache.")
    return id_to_emb


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
        metric = clustering_config.get("metric", "cosine")
        return [
            run_hdbscan_analysis(
                id_to_embedding_map,
                ground_truth_labels,
                min_cluster_size=mcs,
                min_samples=min_samples,
                metric=str(metric),
            )
            for mcs in mcs_values
        ]

    raise ValueError(f"Unknown cluster_algorithm={algo!r}. Expected: dbscan, meanshift, hdbscan.")


def _restrict_embeddings_to_ground_truth(
    id_to_embedding_map: dict[str, np.ndarray],
    ground_truth_labels: dict[str, object],
) -> dict[str, np.ndarray]:
    """Keep only embeddings whose external_id appears in ground_truth_labels."""
    gt_ids = set(map(str, ground_truth_labels.keys()))
    filtered = {eid: vec for eid, vec in id_to_embedding_map.items() if str(eid) in gt_ids}
    if not filtered:
        raise ValueError(
            "cluster_only_ground_truth=True but no overlap between embedding ids and ground truth ids."
        )
    return filtered


def save_metrics_csv(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("No rows to save.")

    all_keys = set().union(*(r.keys() for r in rows))
    preferred_order = [
        "model",
        "embedding_mode",
        "clustering_type",
        "epsilon",
        "min_samples",
        "quantile",
        "n_samples",
        "bandwidth",
        "clustering_error",
        "min_cluster_size",
        "metric",
        "silhouette",
        "db_index",
        "ch_index",
        "homogeneity",
        "completeness",
        "v_measure",
        "n_embeddings",
        "n_clusters",
        "n_noise",
        "n_non_noise",
        "coverage_ground_truth",
        "coverage_all",
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
    cluster_only_ground_truth: bool = False,
    use_hybrid_embeddings: bool = False,
    hybrid_raw_weight: float = 1.0,
    hybrid_gnn_weight: float = 1.0,
):
    """
    Run clustering sweep for one model. Writes ``<output_dir>/<model_column_name>_<algo>_sweep.csv``
    (e.g. best_model_dbscan_sweep.csv). Use training.model_save_name stem for consistency.
    email_external_ids: list of external_id per email node from metadata (email_attrs.external_id).

    When ``use_hybrid_embeddings`` is True, each embedding is built from L2-normalized raw
    and GNN vectors with configurable weights, concatenated, then L2-normalized again.
    """
    id_to_emb = extract_email_embeddings(model, data, device, external_ids=email_external_ids)
    return sweep_clustering_for_embedding_map(
        id_to_embedding_map=id_to_emb,
        ground_truth_labels=ground_truth_labels,
        clustering_config=clustering_config,
        output_dir=output_dir,
        model_column_name=model_column_name,
    )


def sweep_clustering_for_embedding_map(
    *,
    id_to_embedding_map,
    ground_truth_labels,
    clustering_config,
    output_dir,
    model_column_name="model",
):
    """
    Run clustering sweep for a precomputed id->embedding mapping.

    Writes ``<output_dir>/<model_column_name>_<algo>_sweep.csv``.
    """
    cfg = dict(clustering_config)
    rows = _collect_clustering_sweep_metrics(id_to_embedding_map, ground_truth_labels, cfg)
    for r in rows:
        r["model"] = model_column_name
        r["embedding_mode"] = "raw"

    algo = str(cfg["cluster_algorithm"]).lower()
    output_dir = Path(output_dir)
    csv_path = output_dir / f"{model_column_name}_{algo}_sweep.csv"
    save_metrics_csv(rows, csv_path)
    return {"csv_path": str(csv_path), "rows": rows}


def sweep_clustering_for_transformer_text_embeddings(
    *,
    ground_truth_labels,
    clustering_config,
    output_dir,
    embeddings_json_path: str | Path,
    model_column_name: str = "transformer_text_embeddings",
    cluster_only_ground_truth: bool = False,
):
    """
    Run clustering sweep on untouched transformer subject+body embeddings baseline.
    Embeddings are loaded from the cache JSON produced by utils.embeddings.
    """
    id_to_emb = load_transformer_subject_body_embeddings_from_cache(
        embeddings_json_path=embeddings_json_path
    )
    if cluster_only_ground_truth:
        id_to_emb = _restrict_embeddings_to_ground_truth(id_to_emb, ground_truth_labels)
    cfg = dict(clustering_config)
    rows = _collect_clustering_sweep_metrics(id_to_emb, ground_truth_labels, cfg)
    for r in rows:
        r["model"] = model_column_name
        r["embedding_mode"] = "transformer_subject_body_raw"

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
    cluster_only_ground_truth: bool = False,
    use_hybrid_embeddings: bool = False,
    hybrid_raw_weight: float = 1.0,
    hybrid_gnn_weight: float = 1.0,
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
                cluster_only_ground_truth=cluster_only_ground_truth,
                use_hybrid_embeddings=use_hybrid_embeddings,
                hybrid_raw_weight=hybrid_raw_weight,
                hybrid_gnn_weight=hybrid_gnn_weight,
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
    cluster_only_ground_truth: bool = False,
    use_hybrid_embeddings: bool = False,
    hybrid_raw_weight: float = 1.0,
    hybrid_gnn_weight: float = 1.0,
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
        cluster_only_ground_truth=cluster_only_ground_truth,
        use_hybrid_embeddings=use_hybrid_embeddings,
        hybrid_raw_weight=hybrid_raw_weight,
        hybrid_gnn_weight=hybrid_gnn_weight,
    )
