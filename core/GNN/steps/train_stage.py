from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from config.pipeline_config import GnnPathLayout, gnn_path_layout_from_pipeline, load_pipeline_config
from src.load_graph_data import load_hetero_pt
from src.train import run_training
from src.model_io import select_device

from steps.pipeline_paths import run_dir_for


def run_train_stage(
    *,
    graph_path: str | Path,
    runs_parent: str | Path,
    run_id: str,
    training_cfg: dict[str, Any],
    device_pref: str | None,
    to_undirected: bool,
    path_layout: GnnPathLayout | None = None,
) -> dict[str, Any]:
    """
    Train into ``<runs_parent>/<run_id>/`` (``run_id`` from config). Subpaths for
    checkpoints and artifacts come from ``pipeline_config.json`` ``gnn`` (via ``path_layout``).
    """
    graph_path = str(graph_path)
    if not graph_path:
        raise ValueError("GRAPH_PATH is empty in run_pipeline.py.")

    layout = path_layout or gnn_path_layout_from_pipeline(load_pipeline_config())

    pref = torch.device(device_pref) if isinstance(device_pref, str) and device_pref else None
    device = select_device(pref)

    data = load_hetero_pt(
        path=str(Path(graph_path).expanduser()),
        to_undirected=bool(to_undirected),
    )

    run_dir = run_dir_for(runs_parent, run_id)

    run_training(
        DEVICE=device,
        TORCH_SEED=int(training_cfg["torch_seed"]),
        data=data,
        primary_ntype=training_cfg["primary_ntype"],
        hidden=int(training_cfg["hidden"]),
        out_dim=int(training_cfg["out_dim"]),
        layers=int(training_cfg["layers"]),
        dropout=float(training_cfg["dropout"]),
        neg_ratio=float(training_cfg["neg_ratio"]),
        batch_size=int(training_cfg["batch_size"]),
        fanout=training_cfg["fanout"],
        val_ratio=float(training_cfg["val_ratio"]),
        test_ratio=float(training_cfg["test_ratio"]),
        epochs=int(training_cfg["epochs"]),
        lr=float(training_cfg["lr"]),
        wd=float(training_cfg["wd"]),
        score_head=training_cfg["score_head"],
        early_stopping_patience=int(training_cfg["early_stopping_patience"]),
        lr_reduce_patience=int(training_cfg["lr_reduce_patience"]),
        lr_reduce_factor=float(training_cfg["lr_reduce_factor"]),
        lr_reduce_min=float(training_cfg["lr_reduce_min"]),
        supervised_edge_types=training_cfg["supervised_edge_types"],
        model_save_name=training_cfg["model_save_name"],
        contrastive_edges=training_cfg["contrastive_edges"],
        contrastive_weight=float(training_cfg["contrastive_weight"]),
        run_dir=str(run_dir),
        runs_parent=runs_parent,
        models_subdir=layout.models_subdir,
        metrics_csv=layout.metrics_csv,
        training_config_json=layout.training_config_json,
    )

    model_save = training_cfg["model_save_name"]
    best_ckpt = run_dir / layout.models_subdir / model_save
    result = {
        "run_dir": str(run_dir),
        "models_dir": str(run_dir / layout.models_subdir),
        "best_checkpoint_path": str(best_ckpt),
        "metrics_csv_path": str(run_dir / layout.metrics_csv),
        "training_config_path": str(run_dir / layout.training_config_json),
    }
    (run_dir / layout.stage_result_json).write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    return result
