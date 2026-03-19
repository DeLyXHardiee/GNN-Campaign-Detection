from __future__ import annotations

from pathlib import Path
from typing import Any, Tuple

import torch

from src.load_graph_data import load_hetero_pt
from src.model_io import load_full_run, select_device


def load_graph_and_run(
    *,
    graph_path: str | Path,
    checkpoint_path: str | Path,
    device_pref: str | None,
    to_undirected: bool,
) -> Tuple[Any, Any, Any, dict, dict, dict]:
    """
    Load graph from disk and rebuild a model/predictor + splits from a checkpoint.
    """
    graph_path = str(graph_path)
    checkpoint_path = str(checkpoint_path)

    if not graph_path:
        raise ValueError("GRAPH_PATH is empty in run_pipeline.py.")
    if not checkpoint_path:
        raise ValueError("CHECKPOINT_PATH is empty in run_pipeline.py.")

    pref = torch.device(device_pref) if isinstance(device_pref, str) and device_pref else None
    device = select_device(pref)

    data = load_hetero_pt(
        path=str(Path(graph_path).expanduser()),
        to_undirected=bool(to_undirected),
    )

    model, predictor, loaders, splits, checkpoint = load_full_run(
        data=data,
        device=device,
        filename=checkpoint_path,
    )

    return device, model, predictor, loaders, splits, checkpoint

