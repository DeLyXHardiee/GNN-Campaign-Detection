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
    model_name: str,
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

    sweep_clustering_for_one_model(
        model=model,
        data=data,
        device=device,
        ground_truth_labels=ground_truth,
        clustering_config=clustering_cfg,
        output_dir=clustering_out,
        model_name=model_name,
        model_column_name=Path(checkpoint_path).stem,
    )

    algo = str(clustering_cfg["cluster_algorithm"]).lower()
    model_dir = clustering_out / model_name
    csv_path = model_dir / f"{algo}_sweep.csv"

    result = {"csv_path": str(csv_path), "output_dir": str(clustering_out), "model_column_name": Path(checkpoint_path).stem}
    (model_dir / "stage_result.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    return result

