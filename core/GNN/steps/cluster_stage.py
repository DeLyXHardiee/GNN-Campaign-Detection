from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch

from config.pipeline_config import (
    GnnPathLayout,
    gnn_path_layout_from_pipeline,
    load_pipeline_config,
)
try:
    from core.clustering.clusteringMetrics import fit_predict_labels
    from core.visualization.campaign_utils import build_campaign_artifact_payload
except ModuleNotFoundError:
    from clustering.clusteringMetrics import fit_predict_labels
    from visualization.campaign_utils import build_campaign_artifact_payload
from src.clustering.clustering_helpers import (
    extract_email_embeddings,
    extract_ground_truth_labels,
    run_locked_param_across_checkpoints,
    sweep_clustering_for_one_model,
)
from src.load_graph_data import load_hetero_pt
from src.model_io import load_model_checkpoint, select_device


def run_clustering_stage(
    *,
    graph_path: str | Path,
    ground_truth_path: str | Path,
    checkpoint_path: str | Path,
    output_dir: str | Path,
    clustering_cfg: dict[str, Any],
    min_coverage_ground_truth: float = 0.5,
    min_coverage_all: float | None = None,
    model_save_name: str,
    device_pref: str | None,
    to_undirected: bool,
    path_layout: GnnPathLayout | None = None,
) -> dict[str, Any]:
    graph_path = str(graph_path)
    ground_truth_path = str(ground_truth_path)
    checkpoint_path = str(checkpoint_path)

    if not graph_path:
        raise ValueError("GRAPH_PATH is empty in core/GNN/run_pipeline.py.")
    if not ground_truth_path:
        raise ValueError("GROUND_TRUTH_PATH is empty in core/GNN/run_pipeline.py (required for clustering).")
    if not checkpoint_path:
        raise ValueError("CHECKPOINT_PATH is empty in core/GNN/run_pipeline.py (required for clustering).")

    layout = path_layout or gnn_path_layout_from_pipeline(load_pipeline_config())

    output_dir = Path(output_dir)
    clustering_out = output_dir / layout.clustering_subdir
    clustering_out.mkdir(parents=True, exist_ok=True)

    pref = torch.device(device_pref) if isinstance(device_pref, str) and device_pref else None
    device = select_device(pref)

    data = load_hetero_pt(
        path=str(Path(graph_path).expanduser()),
        to_undirected=bool(to_undirected),
    )
    # Load metadata for email external_id (graph does not store it; PyG loaders require tensor-only node stores)
    meta_path = Path(graph_path).expanduser().with_suffix(".meta.json")
    if not meta_path.exists():
        raise FileNotFoundError(
            f"Graph metadata not found: {meta_path}. "
            "Clustering requires email_attrs.external_id from the companion .meta.json."
        )
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    email_external_ids = meta.get("email_attrs", {}).get("external_id")
    if not email_external_ids:
        raise ValueError(
            f"Metadata at {meta_path} has no email_attrs.external_id. "
            "Clustering requires external_id per email node."
        )
    ground_truth = extract_ground_truth_labels(ground_truth_path)

    model, predictor, checkpoint = load_model_checkpoint(
        device=device, metadata=data.metadata(), filename=checkpoint_path
    )
    _ = predictor, checkpoint

    # clustering_cfg is a dict: algo_name -> { "enabled": bool, ...params }.
    # Model name comes from training.model_save_name (stem) so it stays consistent when not running training.
    outputs: dict[str, dict[str, str]] = {}
    model_stem = Path(model_save_name).stem

    # Default `min_coverage_all` to the same threshold as ground truth coverage.
    if min_coverage_all is None:
        min_coverage_all = float(min_coverage_ground_truth)

    # Select a single locked parameter from the best-model sweep.
    # Criterion: maximize v_measure, with minimum thresholds on BOTH:
    # - coverage_ground_truth
    # - coverage_all
    best_locked_params: dict[str, dict[str, Any]] = {}

    for algo_name, algo_cfg in clustering_cfg.items():
        if not algo_cfg.get("enabled", False):
            continue
        algo_name = str(algo_name).lower().strip()
        cfg_for_sweep = {k: v for k, v in algo_cfg.items() if k != "enabled"}
        cfg_for_sweep["cluster_algorithm"] = algo_name

        algo_out = clustering_out / algo_name
        sweep_res = sweep_clustering_for_one_model(
            model=model,
            data=data,
            device=device,
            ground_truth_labels=ground_truth,
            clustering_config=cfg_for_sweep,
            output_dir=algo_out,
            model_column_name=model_stem,
            email_external_ids=email_external_ids,
        )
        algo_entry: dict[str, Any] = {
            "csv_path": str(sweep_res["csv_path"]),
            "output_dir": str(algo_out),
        }

        # Choose best epsilon / quantile / min_cluster_size from the best-model sweep.
        rows = sweep_res.get("rows") or []
        if algo_name == "dbscan":
            param_key = "epsilon"
        elif algo_name == "meanshift":
            param_key = "quantile"
        elif algo_name == "hdbscan":
            param_key = "min_cluster_size"
        else:
            param_key = None

        if param_key is not None and rows:
            candidates = [
                r
                for r in rows
                if float(r.get("coverage_ground_truth", 0.0)) >= min_coverage_ground_truth
                and float(r.get("coverage_all", 0.0)) >= min_coverage_all
            ]
            pool = candidates if candidates else list(rows)
            best_row = max(pool, key=lambda r: float(r.get("v_measure", 0.0)))
            best_locked_params[algo_name] = {
                param_key: float(best_row.get(param_key)),
                "v_measure": float(best_row.get("v_measure", 0.0)),
                "coverage_ground_truth": float(best_row.get("coverage_ground_truth", 0.0)),
                "coverage_all": float(best_row.get("coverage_all", 0.0)),
                "min_coverage_ground_truth": float(min_coverage_ground_truth),
                "min_coverage_all": float(min_coverage_all),
            }

        clustering_errors = [
            str(r["clustering_error"])
            for r in sweep_res["rows"]
            if r.get("clustering_error")
        ]
        if clustering_errors:
            algo_entry["clustering_errors"] = clustering_errors
        outputs[algo_name] = algo_entry

    # Run locked-parameter clustering across epoch checkpoints so we can plot metrics vs epoch
    # without doing a full epsilon/quantile sweep for every checkpoint.
    models_dir = Path(checkpoint_path).parent
    epoch_ckpts: list[Path] = []
    for p in models_dir.glob("model_epoch_*.pt"):
        m = re.search(r"(\d+)", p.stem)
        if not m:
            continue
        epoch_ckpts.append(p)
    def _epoch_num(p: Path) -> int:
        em = re.search(r"(\d+)", p.stem)
        return int(em.group(1)) if em else 0

    epoch_ckpts = sorted(epoch_ckpts, key=_epoch_num)

    if epoch_ckpts and best_locked_params:
        for algo_name, best in best_locked_params.items():
            algo_cfg = clustering_cfg.get(algo_name) if isinstance(clustering_cfg, dict) else None
            if not isinstance(algo_cfg, dict) or not algo_cfg.get("enabled", False):
                continue

            if algo_name == "dbscan":
                locked_param_value = best["epsilon"]
            elif algo_name == "meanshift":
                locked_param_value = best["quantile"]
            elif algo_name == "hdbscan":
                locked_param_value = best["min_cluster_size"]
            else:
                continue

            cfg_for_sweep = {k: v for k, v in algo_cfg.items() if k != "enabled"}
            cfg_for_sweep["cluster_algorithm"] = algo_name

            algo_out = clustering_out / algo_name
            run_locked_param_across_checkpoints(
                data=data,
                device=device,
                checkpoints=[str(p) for p in epoch_ckpts],
                ground_truth_labels=ground_truth,
                clustering_config=cfg_for_sweep,
                locked_param_value=locked_param_value,
                output_dir=algo_out,
                email_external_ids=email_external_ids,
            )

    campaigns_gnn_path: str | None = None
    if best_locked_params:
        best_algo_name, best_info = max(
            best_locked_params.items(),
            key=lambda kv: float(kv[1].get("v_measure", 0.0)),
        )
        algo_cfg_best = (
            clustering_cfg.get(best_algo_name) if isinstance(clustering_cfg, dict) else None
        )
        if not isinstance(algo_cfg_best, dict):
            algo_cfg_best = {}

        id_to_emb = extract_email_embeddings(
            model, data, device, external_ids=email_external_ids
        )

        if best_algo_name == "dbscan":
            sorted_ids, labels = fit_predict_labels(
                id_to_emb,
                "dbscan",
                epsilon=float(best_info["epsilon"]),
                min_samples=int(algo_cfg_best.get("min_samples", 5)),
            )
            params_out: dict[str, Any] = {
                "epsilon": float(best_info["epsilon"]),
                "min_samples": int(algo_cfg_best.get("min_samples", 5)),
            }
        elif best_algo_name == "meanshift":
            sorted_ids, labels = fit_predict_labels(
                id_to_emb,
                "meanshift",
                quantile=float(best_info["quantile"]),
                n_samples=algo_cfg_best.get("n_samples"),
            )
            params_out = {
                "quantile": float(best_info["quantile"]),
                "n_samples": algo_cfg_best.get("n_samples"),
            }
        elif best_algo_name == "hdbscan":
            sorted_ids, labels = fit_predict_labels(
                id_to_emb,
                "hdbscan",
                min_cluster_size=int(best_info["min_cluster_size"]),
                hdbscan_min_samples=algo_cfg_best.get("min_samples"),
            )
            params_out = {
                "min_cluster_size": int(best_info["min_cluster_size"]),
                "min_samples": algo_cfg_best.get("min_samples"),
            }
        else:
            sorted_ids, labels = [], np.array([], dtype=np.int64)
            params_out = {}

        payload = build_campaign_artifact_payload(
            solution="gnn",
            algorithm=best_algo_name,
            sorted_ids=sorted_ids,
            labels=labels,
            params=params_out,
            metrics={
                "v_measure": float(best_info.get("v_measure", 0.0)),
                "coverage_ground_truth": float(best_info.get("coverage_ground_truth", 0.0)),
                "coverage_all": float(best_info.get("coverage_all", 0.0)),
            },
            model_name=model_stem,
        )
        out_p = clustering_out / "campaigns_gnn.json"
        out_p.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        campaigns_gnn_path = str(out_p)

    result = {
        "output_dir": str(clustering_out),
        "model_column_name": model_stem,
        "algorithms": outputs,
        "best_locked_params": best_locked_params,
        "locked_param_min_coverage_ground_truth": float(min_coverage_ground_truth),
        "locked_param_min_coverage_all": float(min_coverage_all),
        "locked_param_epoch_checkpoints": [str(p) for p in epoch_ckpts],
        "campaigns_gnn_path": campaigns_gnn_path,
    }
    (clustering_out / layout.stage_result_json).write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    return result

