from __future__ import annotations

from pathlib import Path
from typing import Any

from config.pipeline_config import default_hetero_graph_pt_path, resolve_project_path
from steps.pipeline_paths import run_dir_for, sanitize_run_id


def load_gnn_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    """
    Extract the GNN-relevant keys from the unified `pipeline_config.json`.
    """
    device_pref = cfg["device"]
    to_undirected = bool(cfg["to_undirected"])
    run_id = cfg["run_id"]
    training_cfg = cfg["training"]
    evaluation_auroc_cfg = cfg["evaluation"].get("auroc_ap", {})
    recall_cfg = cfg["evaluation"]["recall_at_k"]
    gnn_clustering_cfg = cfg["gnn_clustering"]["config"]
    gnn_clustering_selection_cfg = cfg["gnn_clustering"].get("selection", {})

    return {
        "device_pref": device_pref,
        "to_undirected": to_undirected,
        "run_id": run_id,
        "training_cfg": training_cfg,
        "evaluation_auroc_cfg": evaluation_auroc_cfg,
        "recall_cfg": recall_cfg,
        "gnn_clustering_cfg": gnn_clustering_cfg,
        "gnn_clustering_selection_cfg": gnn_clustering_selection_cfg,
    }


def resolve_gnn_paths(
    *,
    cfg: dict[str, Any],
    run_dir: str | Path | None,
    runs_parent: str | Path,
    checkpoint_path: str | Path | None,
    graph_path: str | Path | None,
    ground_truth_path: str | Path | None,
    require_ground_truth: bool,
) -> tuple[str, str, str, str]:
    """
    Resolve concrete paths for the GNN stages from the unified config.

    `ground_truth_path` is only required by the clustering stage; callers can set
    `require_ground_truth=False` so training/eval don't fail if that config key is absent.
    """
    g = load_gnn_cfg(cfg)
    runs_parent = str(runs_parent)

    # run_dir
    if run_dir is None or str(run_dir).strip() == "":
        run_dir_path = run_dir_for(runs_parent, sanitize_run_id(str(g["run_id"]))).resolve()
        run_dir_str = str(run_dir_path)
    else:
        run_dir_str = str(Path(run_dir).resolve())

    # checkpoint_path
    if checkpoint_path is None or str(checkpoint_path).strip() == "":
        checkpoint_path_str = str(
            Path(run_dir_str) / "models" / str(g["training_cfg"]["model_save_name"])
        )
    else:
        checkpoint_path_str = str(Path(checkpoint_path).resolve())

    # graph_path
    if graph_path is None or str(graph_path).strip() == "":
        graph_path_str = default_hetero_graph_pt_path()
    else:
        graph_path_str = str(Path(graph_path).resolve())

    # ground_truth_path (clustering only)
    if not require_ground_truth:
        ground_truth_path_str = ""
    elif ground_truth_path is None or str(ground_truth_path).strip() == "":
        gt = cfg.get("datasets", {}).get("ground_truth_json")
        ground_truth_path_str = resolve_project_path(gt)
        if not ground_truth_path_str:
            raise ValueError(
                "pipeline_config.json: set datasets.ground_truth_json for clustering."
            )
    else:
        ground_truth_path_str = str(Path(ground_truth_path).resolve())

    return run_dir_str, checkpoint_path_str, graph_path_str, ground_truth_path_str

