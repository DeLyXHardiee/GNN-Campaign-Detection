from __future__ import annotations

import json
from pathlib import Path

import torch

from src.eval_link.auroc_ap import run_auroc_ap_analysis
from src.load_graph_data import load_hetero_pt
from src.model_io import load_full_run, select_device


def run_auroc_ap_stage(*, config_path: str | Path) -> dict:
    cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))

    graph_path = cfg["graph_path"]
    if not graph_path:
        raise ValueError("cfg.graph_path is empty. Fill it in core/GNN/gnn_stage_pipeline_config.json.")

    output_dir = Path(cfg["output_dir"])
    out = output_dir / "eval_auroc_ap"
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

    res = run_auroc_ap_analysis(
        device=device,
        model=model,
        predictor=predictor,
        loaders_test=loaders["test"],
        output_dir=out,
    )

    # Keep a minimal record of what we ran.
    (out / "eval_config.json").write_text(
        json.dumps(
            {
                "checkpoint_path": checkpoint_path,
                "graph_path": graph_path,
                **cfg["evaluation"]["auroc_ap"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return res | {"output_dir": str(out)}

