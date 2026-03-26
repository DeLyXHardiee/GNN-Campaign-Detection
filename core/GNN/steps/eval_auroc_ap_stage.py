from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from config.pipeline_config import GnnPathLayout, gnn_path_layout_from_pipeline, load_pipeline_config
from steps.eval_stage_utils import load_graph_and_run
from src.eval_link.auroc_ap import run_auroc_ap_analysis


def run_auroc_ap_stage(
    *,
    graph_path: str | Path,
    checkpoint_path: str | Path,
    output_dir: str | Path,
    evaluation_cfg: dict[str, Any],
    device_pref: str | None,
    to_undirected: bool,
    path_layout: GnnPathLayout | None = None,
) -> dict[str, Any]:
    graph_path = str(graph_path)
    checkpoint_path = str(checkpoint_path)
    if not graph_path:
        raise ValueError("GRAPH_PATH is empty in core/GNN/run_pipeline.py.")
    if not checkpoint_path:
        raise ValueError("CHECKPOINT_PATH is empty in core/GNN/run_pipeline.py (required for eval).")

    layout = path_layout or gnn_path_layout_from_pipeline(load_pipeline_config())

    output_dir = Path(output_dir)
    out = output_dir / layout.eval_auroc_ap_subdir
    out.mkdir(parents=True, exist_ok=True)

    device, model, predictor, loaders, splits, checkpoint = load_graph_and_run(
        graph_path=graph_path,
        checkpoint_path=checkpoint_path,
        device_pref=device_pref,
        to_undirected=to_undirected,
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
                **evaluation_cfg,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    result = res | {"output_dir": str(out)}
    (out / layout.stage_result_json).write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    return result

