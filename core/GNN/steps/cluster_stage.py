from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from src.clustering.clustering_helpers import extract_ground_truth_labels, sweep_clustering_for_one_model
from src.load_graph_data import load_hetero_pt
from src.model_io import load_model_checkpoint, select_device


def run_clustering_stage(
    *,
    graph_path: str | Path,
    ground_truth_path: str | Path,
    checkpoint_path: str | Path,
    output_dir: str | Path,
    clustering_cfg: dict[str, Any],
    model_save_name: str,
    device_pref: str | None,
    to_undirected: bool,
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

    output_dir = Path(output_dir)
    clustering_out = output_dir / "clustering"
    clustering_out.mkdir(parents=True, exist_ok=True)

    pref = torch.device(device_pref) if isinstance(device_pref, str) and device_pref else None
    device = select_device(pref)

    data = load_hetero_pt(
        path=str(Path(graph_path).expanduser()),
        to_undirected=bool(to_undirected),
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
        )
        outputs[algo_name] = {
            "csv_path": str(sweep_res["csv_path"]),
            "output_dir": str(algo_out),
        }

    result = {
        "output_dir": str(clustering_out),
        "model_column_name": model_stem,
        "algorithms": outputs,
    }
    (clustering_out / "stage_result.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    return result

