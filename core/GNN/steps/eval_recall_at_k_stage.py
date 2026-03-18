from __future__ import annotations

import json
from pathlib import Path

import torch

from src.eval_link.recall_at_k import run_recall_at_k_analysis
from src.load_graph_data import load_hetero_pt
from src.model_io import load_full_run, select_device


def run_recall_at_k_stage(*, config_path: str | Path) -> dict:
    cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))

    graph_path = cfg["graph_path"]
    if not graph_path:
        raise ValueError("cfg.graph_path is empty. Fill it in core/GNN/gnn_stage_pipeline_config.json.")

    output_dir = Path(cfg["output_dir"])
    out = output_dir / "eval_recall_at_k"
    out.mkdir(parents=True, exist_ok=True)

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

    model, predictor, loaders, splits, _checkpoint = load_full_run(
        data=data,
        device=device,
        filename=checkpoint_path,
    )

    recall_cfg = cfg["evaluation"]["recall_at_k"]
    res = run_recall_at_k_analysis(
        device=device,
        model=model,
        splits=splits,
        output_dir=out,
        K_list=recall_cfg["K_list"],
        use_dot=bool(recall_cfg["use_dot"]),
    )

    (out / "eval_config.json").write_text(
        json.dumps(
            {"checkpoint_path": checkpoint_path, "graph_path": graph_path, **recall_cfg},
            indent=2,
        ),
        encoding="utf-8",
    )
    return res | {"output_dir": str(out)}

