from __future__ import annotations

import json
from pathlib import Path

import torch

from src.clustering.clustering_helpers import extract_ground_truth_labels, sweep_clustering_for_one_model
from src.load_graph_data import load_hetero_pt
from src.model_io import load_model_checkpoint, select_device


def run_clustering_stage(*, config_path: str | Path) -> dict:
    cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))

    graph_path = cfg["graph_path"]
    ground_truth_path = cfg["ground_truth_path"]
    if not graph_path:
        raise ValueError("cfg.graph_path is empty. Fill it in core/GNN/gnn_stage_pipeline_config.json.")
    if not ground_truth_path:
        raise ValueError("cfg.ground_truth_path is empty. Fill it in core/GNN/gnn_stage_pipeline_config.json.")

    output_dir = Path(cfg["output_dir"])
    clustering_out = output_dir / "clustering"
    clustering_out.mkdir(parents=True, exist_ok=True)

    checkpoint_path = cfg.get("checkpoint_path") or ""
    if not checkpoint_path:
        training_cfg = cfg["training"]
        checkpoint_path = str(output_dir / "training" / training_cfg["model_save_name"])
    if not checkpoint_path:
        raise ValueError("checkpoint_path is empty and could not be derived from output_dir/training.")

    device_pref = cfg.get("device")
    pref = torch.device(device_pref) if isinstance(device_pref, str) and device_pref else None
    device = select_device(pref)

    data = load_hetero_pt(
        path=str(Path(graph_path).expanduser()),
        to_undirected=bool(cfg["to_undirected"]),
    )
    ground_truth = extract_ground_truth_labels(ground_truth_path)

    model, predictor, checkpoint = load_model_checkpoint(
        device=device, metadata=data.metadata(), filename=checkpoint_path
    )
    _ = predictor, checkpoint

    clustering_cfg = cfg["clustering"]
    model_name = clustering_cfg["model_name"]

    sweep_clustering_for_one_model(
        model=model,
        data=data,
        device=device,
        ground_truth_labels=ground_truth,
        clustering_config=clustering_cfg["config"],
        output_dir=clustering_out,
        model_name=model_name,
    )

    algo = str(clustering_cfg["config"]["cluster_algorithm"]).lower()
    csv_path = clustering_out / model_name / f"{algo}_sweep.csv"
    return {"csv_path": str(csv_path), "output_dir": str(clustering_out)}

