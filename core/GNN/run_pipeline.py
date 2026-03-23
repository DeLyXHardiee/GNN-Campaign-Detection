"""
Manual toggle pipeline for the GNN project.

One experiment folder per ``run_id`` in ``pipeline_config.json``:

  <RUNS_PARENT>/<run_id>/
    models/
    training_config.json
    metrics.csv
    stage_result.json
    eval_auroc_ap/
    eval_recall_at_k/
    clustering/

Train, eval, and clustering all use the same path — no timestamps or sidecar files.
Keep ``run_id`` and ``training.model_save_name`` consistent with the run/checkpoint you use,
especially when you skip training and only run eval or clustering.

Optional ``RUN_DIR`` below overrides ``<RUNS_PARENT>/<run_id>`` (full path).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from steps.cluster_stage import run_clustering_stage
from steps.eval_auroc_ap_stage import run_auroc_ap_stage
from steps.eval_recall_at_k_stage import run_recall_at_k_stage
from steps.pipeline_paths import run_dir_for, sanitize_run_id
from steps.train_stage import run_train_stage


# pipeline_config.json lives at repo root (two levels above this file).
CONFIG_PATH = Path(__file__).resolve().parents[2] / "pipeline_config.json"

# Paths you fill in.
GRAPH_PATH = "../graph/output/incidents-20260211-misp_hetero.pt"
GROUND_TRUTH_PATH = "../../data/groundtruth/ground_truth.json"

# Parent directory where training creates run_<timestamp>/ (see docstring).
RUNS_PARENT = ""

# Full path override. Empty → <RUNS_PARENT>/<run_id> from config.
RUN_DIR = ""

# Optional override; if empty, uses RUN_DIR/models/<model_save_name from config>.
CHECKPOINT_PATH = ""


def main() -> None:
    cfg: dict[str, Any] = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

    device_pref = cfg["device"]
    to_undirected = bool(cfg["to_undirected"])

    training_cfg = cfg["training"]
    evaluation_auroc_cfg = cfg["evaluation"].get("auroc_ap", {})
    recall_cfg = cfg["evaluation"]["recall_at_k"]

    clustering_root = cfg["gnn_clustering"]
    clustering_cfg = clustering_root["config"]

    run_dir = RUN_DIR.strip()
    if not run_dir:
        run_dir = str(run_dir_for(RUNS_PARENT, sanitize_run_id(cfg["run_id"])).resolve())

    checkpoint_path = CHECKPOINT_PATH.strip()
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

    # Uncomment eval/clustering (same run_id / RUN_DIR as training).
    run_auroc_ap_stage(
        graph_path=GRAPH_PATH,
        checkpoint_path=checkpoint_path,
        output_dir=run_dir,
        evaluation_cfg=evaluation_auroc_cfg,
        device_pref=device_pref,
        to_undirected=to_undirected,
    )

    run_recall_at_k_stage(
        graph_path=GRAPH_PATH,
        checkpoint_path=checkpoint_path,
        output_dir=run_dir,
        evaluation_cfg=recall_cfg,
        device_pref=device_pref,
        to_undirected=to_undirected,
    )
    '''
    
    run_clustering_stage(
        graph_path=GRAPH_PATH,
        ground_truth_path=GROUND_TRUTH_PATH,
        checkpoint_path=checkpoint_path,
        output_dir=run_dir,
        clustering_cfg=clustering_cfg,
        model_save_name=training_cfg["model_save_name"],
        device_pref=device_pref,
        to_undirected=to_undirected,
    )

    print("Done. run_dir:", run_dir)


if __name__ == "__main__":
    main()
