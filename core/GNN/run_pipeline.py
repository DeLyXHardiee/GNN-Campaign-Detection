"""
Manual toggle pipeline for the GNN project.

All filesystem paths come from ``pipeline_config.json`` at the repo root
(``gnn`` block, ``graph`` / ``datasets`` for the hetero graph, etc.). Optional
overrides below only apply when non-empty.

One experiment folder per resolved run (see ``output_runs_root`` / session allocation in ``config.run_output_paths``):

  <output_runs_root>/<run_id or ``run_id (1)`` …>/
    <gnn.models_subdir>/
    <gnn.training_config_json>, <gnn.metrics_csv>, <gnn.run_stage_result_json>
    <gnn.eval_auroc_ap_subdir>/, <gnn.eval_recall_at_k_subdir>/
    <gnn.clustering_subdir>/ ...
"""

from __future__ import annotations

import sys
from pathlib import Path

# Resolve imports: ``config`` lives under ``core/``, steps under ``core/GNN/``.
_CORE_ROOT = Path(__file__).resolve().parents[1]
_GNN_ROOT = Path(__file__).resolve().parent
for _p in (_CORE_ROOT, _GNN_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from config.pipeline_config import (  # noqa: E402
    load_pipeline_config,
    output_runs_parent_from_pipeline,
    resolve_project_path,
)
from config.run_output_paths import resolve_session_run_output_dir  # noqa: E402
from steps.cluster_stage import run_clustering_stage  # noqa: E402
from steps.clustering_plot_stage import run_clustering_plot_stage  # noqa: E402
from steps.eval_auroc_ap_stage import run_auroc_ap_stage  # noqa: E402
from steps.eval_recall_at_k_stage import run_recall_at_k_stage  # noqa: E402
from steps.gnn_pipeline_helpers import load_gnn_cfg, resolve_gnn_paths  # noqa: E402
from steps.train_stage import run_train_stage  # noqa: E402

# Optional overrides (leave empty to use pipeline_config.json only).
RUN_DIR_OVERRIDE = ""
CHECKPOINT_PATH_OVERRIDE = ""
GRAPH_PATH_OVERRIDE = ""
GROUND_TRUTH_PATH_OVERRIDE = ""
RUNS_PARENT_OVERRIDE = ""


def main() -> None:
    cfg = load_pipeline_config()
    g = load_gnn_cfg(cfg)

    run_dir_arg = RUN_DIR_OVERRIDE.strip() or None
    checkpoint_arg = CHECKPOINT_PATH_OVERRIDE.strip() or None
    graph_arg = GRAPH_PATH_OVERRIDE.strip() or None
    gt_arg = GROUND_TRUTH_PATH_OVERRIDE.strip() or None
    runs_parent_arg = RUNS_PARENT_OVERRIDE.strip() or None

    training_cfg = cfg["training"]
    evaluation_auroc_cfg = cfg["evaluation"].get("auroc_ap", {})
    recall_cfg = cfg["evaluation"]["recall_at_k"]

    clustering_root = cfg["gnn_clustering"]
    clustering_cfg = clustering_root["config"]
    clustering_selection_cfg = clustering_root.get("selection", {})

    run_dir = run_dir_arg or ""
    if not run_dir:
        run_dir = str(
            resolve_session_run_output_dir(
                cfg,
                runs_root=output_runs_parent_from_pipeline(cfg),
            ).resolve()
        )

    checkpoint_path = (checkpoint_arg or "").strip()
    if run_dir and not checkpoint_path:
        checkpoint_path = str(Path(run_dir) / "models" / training_cfg["model_save_name"])

    '''
    # Uncomment to train into <RUNS_PARENT>/<run_id>/.
    run_train_stage(
        graph_path=GRAPH_PATH,
        runs_parent=RUNS_PARENT,
        run_id=cfg["run_id"],
        training_cfg=training_cfg,
        device_pref=device_pref,
        to_undirected=to_undirected,
    )
    gt_raw = (gt_arg or "").strip()
    if gt_raw:
        ground_truth_path_str = resolve_project_path(gt_raw) or str(Path(gt_raw).expanduser().resolve())
    else:
        ground_truth_path_str = resolve_project_path(cfg.get("datasets", {}).get("ground_truth_json")) or ""
    if not ground_truth_path_str:
        raise ValueError(
            "Set datasets.ground_truth_json in pipeline_config.json or GROUND_TRUTH_PATH_OVERRIDE for clustering."
        )

    training_cfg = g["training_cfg"]
    device_pref = g["device_pref"]
    to_undirected = g["to_undirected"]
    clustering_cfg = g["gnn_clustering_cfg"]
    layout = g["path_layout"]

    runs_parent_effective = layout.runs_parent
    if run_dir_arg:
        runs_parent_effective = str(Path(run_dir_str).resolve().parent)

    # Uncomment to train / eval (same ``run_dir_str`` / checkpoint as clustering).
    # run_train_stage(
    #     graph_path=graph_path_str,
    #     runs_parent=runs_parent_effective,
    #     run_id=str(g["run_id"]),
    #     training_cfg=training_cfg,
    #     path_layout=layout,
    #     device_pref=device_pref,
    #     to_undirected=to_undirected,
    # )
    # run_auroc_ap_stage(
    #     graph_path=graph_path_str,
    #     checkpoint_path=checkpoint_path_str,
    #     output_dir=run_dir_str,
    #     evaluation_cfg=g["evaluation_auroc_cfg"],
    #     path_layout=layout,
    #     device_pref=device_pref,
    #     to_undirected=to_undirected,
    # )
    # run_recall_at_k_stage(
    #     graph_path=graph_path_str,
    #     checkpoint_path=checkpoint_path_str,
    #     output_dir=run_dir_str,
    #     evaluation_cfg=g["recall_cfg"],
    #     path_layout=layout,
    #     device_pref=device_pref,
    #     to_undirected=to_undirected,
    # )

    run_clustering_stage(
        graph_path=graph_path_str,
        ground_truth_path=ground_truth_path_str,
        checkpoint_path=checkpoint_path_str,
        output_dir=run_dir_str,
        clustering_cfg=clustering_cfg,
        min_coverage_ground_truth=float(
            clustering_selection_cfg.get("min_coverage_ground_truth", 0.5)
        ),
        min_coverage_all=float(
            clustering_selection_cfg.get(
                "min_coverage_all",
                clustering_selection_cfg.get("min_coverage_ground_truth", 0.5),
            )
        ),
        model_save_name=training_cfg["model_save_name"],
        path_layout=layout,
        device_pref=device_pref,
        to_undirected=to_undirected,
    )

    # run_clustering_plot_stage(output_dir=run_dir_str, path_layout=layout)

    print("Done. run_dir:", run_dir_str)
    print("Graph:", graph_path_str)
    print("Ground truth:", ground_truth_path_str)


if __name__ == "__main__":
    main()
