from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.eval_link.recall_at_k import run_recall_at_k_analysis
from steps.eval_stage_utils import load_graph_and_run


def run_recall_at_k_stage(
    *,
    graph_path: str | Path,
    checkpoint_path: str | Path,
    output_dir: str | Path,
    evaluation_cfg: dict[str, Any],
    device_pref: str | None,
    to_undirected: bool,
) -> dict[str, Any]:
    graph_path = str(graph_path)
    checkpoint_path = str(checkpoint_path)
    if not graph_path:
        raise ValueError("GRAPH_PATH is empty in core/GNN/run_pipeline.py.")
    if not checkpoint_path:
        raise ValueError("CHECKPOINT_PATH is empty in core/GNN/run_pipeline.py (required for eval).")

    output_dir = Path(output_dir)
    out = output_dir / "eval_recall_at_k"
    out.mkdir(parents=True, exist_ok=True)

    device, model, predictor, loaders, splits, _checkpoint = load_graph_and_run(
        graph_path=graph_path,
        checkpoint_path=checkpoint_path,
        device_pref=device_pref,
        to_undirected=to_undirected,
    )

    res = run_recall_at_k_analysis(
        device=device,
        model=model,
        splits=splits,
        output_dir=out,
        K_list=evaluation_cfg["K_list"],
        use_dot=bool(evaluation_cfg["use_dot"]),
    )

    (out / "eval_config.json").write_text(
        json.dumps(
            {"checkpoint_path": checkpoint_path, "graph_path": graph_path, **evaluation_cfg},
            indent=2,
        ),
        encoding="utf-8",
    )
    result = res | {"output_dir": str(out)}
    (out / "stage_result.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    return result

