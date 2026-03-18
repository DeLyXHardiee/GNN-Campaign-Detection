from __future__ import annotations

import json
from pathlib import Path

import torch

from src.load_graph_data import load_hetero_pt
from src.train import run_training
from src.model_io import select_device


def run_train_stage(*, config_path: str | Path) -> dict:
    cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))

    graph_path = cfg["graph_path"]
    if not graph_path:
        raise ValueError("cfg.graph_path is empty. Fill it in core/GNN/gnn_stage_pipeline_config.json.")

    output_dir = Path(cfg["output_dir"])
    train_out = output_dir / "training"
    train_out.mkdir(parents=True, exist_ok=True)

    device_pref = cfg.get("device")
    pref = torch.device(device_pref) if isinstance(device_pref, str) and device_pref else None
    device = select_device(pref)

    data = load_hetero_pt(
        path=str(Path(graph_path).expanduser()),
        to_undirected=bool(cfg["to_undirected"]),
    )

    training_cfg = cfg["training"]
    model, predictor, loaders, splits = run_training(
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
        run_dir=train_out,
    )

    best_ckpt = train_out / training_cfg["model_save_name"]
    return {
        "train_dir": str(train_out),
        "best_checkpoint_path": str(best_ckpt),
        "model": model,
        "predictor": predictor,
        "loaders": loaders,
        "splits": splits,
    }

