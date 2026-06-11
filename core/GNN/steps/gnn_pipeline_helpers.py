from __future__ import annotations

from pathlib import Path
from typing import Any

from config.pipeline_config import (
    GnnPathLayout,
    default_hetero_graph_pt_path,
    gnn_path_layout_for_pair_backend,
    gnn_path_layout_from_pipeline,
    pair_training_enabled_backend_slugs,
    resolve_project_path,
)
from config.run_output_paths import resolve_session_run_output_dir


def _effective_runs_parent(
    runs_parent: str | Path | None,
    layout: GnnPathLayout,
    *,
    project_root: Path | None = None,
) -> str:
    s = "" if runs_parent is None else str(runs_parent).strip()
    if not s:
        return layout.runs_parent
    resolved = resolve_project_path(s, project_root=project_root)
    if resolved:
        return resolved
    return str(Path(s).expanduser().resolve())


def load_gnn_cfg(cfg: dict[str, Any], *, project_root: Path | None = None) -> dict[str, Any]:
    """
    Extract the GNN-relevant keys from the unified `pipeline_config.json`.
    """
    device_pref = cfg["device"]
    to_undirected = bool(cfg["to_undirected"])
    run_id = cfg["run_id"]
    training_cfg = cfg["training"]
    evaluation_auroc_cfg = cfg["evaluation"].get("auroc_ap", {})
    recall_cfg = cfg["evaluation"]["recall_at_k"]
    gnn_cluster_root = cfg["gnn_clustering"]
    gnn_clustering_cfg = gnn_cluster_root["config"]
    gnn_clustering_selection_cfg = gnn_cluster_root.get("selection", {})
    gnn_clustering_baselines_cfg = gnn_cluster_root.get("baselines", {})
    path_layout = gnn_path_layout_from_pipeline(cfg, project_root=project_root)

    return {
        "device_pref": device_pref,
        "to_undirected": to_undirected,
        "run_id": run_id,
        "training_cfg": training_cfg,
        "evaluation_auroc_cfg": evaluation_auroc_cfg,
        "recall_cfg": recall_cfg,
        "gnn_clustering_cfg": gnn_clustering_cfg,
        "gnn_clustering_selection_cfg": gnn_clustering_selection_cfg,
        "gnn_clustering_baselines_cfg": gnn_clustering_baselines_cfg,
        "path_layout": path_layout,
    }


def resolve_gnn_paths(
    *,
    cfg: dict[str, Any],
    run_dir: str | Path | None,
    runs_parent: str | Path | None,
    checkpoint_path: str | Path | None,
    graph_path: str | Path | None,
    ground_truth_path: str | Path | None,
    require_ground_truth: bool,
    project_root: Path | None = None,
) -> tuple[str, str, str, str]:
    """
    Resolve concrete paths for the GNN stages from the unified config.

    `ground_truth_path` is only required by the clustering stage; callers can set
    `require_ground_truth=False` so training/eval don't fail if that config key is absent.
    """
    g = load_gnn_cfg(cfg, project_root=project_root)
    layout: GnnPathLayout = g["path_layout"]
    runs_parent_eff = _effective_runs_parent(runs_parent, layout, project_root=project_root)

    if run_dir is None or str(run_dir).strip() == "":
        run_dir_path = resolve_session_run_output_dir(
            cfg,
            project_root=project_root,
            runs_root=runs_parent_eff,
        ).resolve()
        run_dir_str = str(run_dir_path)
    else:
        run_dir_str = str(Path(run_dir).resolve())

    if checkpoint_path is None or str(checkpoint_path).strip() == "":
        objective = str(g["training_cfg"].get("training_objective", "link_prediction")).lower().strip()
        if objective == "pair_supervision":
            be = pair_training_enabled_backend_slugs(cfg)
            if not be:
                be = ["gnn"]
            sub_layout = gnn_path_layout_for_pair_backend(layout, be[0])
            checkpoint_path_str = str(
                Path(run_dir_str) / sub_layout.models_subdir / str(g["training_cfg"]["model_save_name"])
            )
        else:
            checkpoint_path_str = str(
                Path(run_dir_str) / layout.models_subdir / str(g["training_cfg"]["model_save_name"])
            )
    else:
        checkpoint_path_str = str(Path(checkpoint_path).resolve())

    if graph_path is None or str(graph_path).strip() == "":
        graph_path_str = default_hetero_graph_pt_path()
    else:
        graph_path_str = str(Path(graph_path).resolve())

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

