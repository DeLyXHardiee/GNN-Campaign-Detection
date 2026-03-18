"""
Manual toggle pipeline for the GNN project.

Run from `core/GNN` and toggle stages by commenting/uncommenting the calls below.
"""

from __future__ import annotations

from pathlib import Path

from steps.cluster_stage import run_clustering_stage
from steps.eval_auroc_ap_stage import run_auroc_ap_stage
from steps.eval_recall_at_k_stage import run_recall_at_k_stage
from steps.train_stage import run_train_stage


CONFIG_PATH = Path(__file__).with_name("gnn_stage_pipeline_config.json")


def main() -> None:
    # Uncomment to train.
    # run_train_stage(config_path=CONFIG_PATH)

    # Uncomment to run AUROC/AP eval.
    # run_auroc_ap_stage(config_path=CONFIG_PATH)

    # Uncomment to run Recall@K eval.
    # run_recall_at_k_stage(config_path=CONFIG_PATH)

    # Uncomment to run clustering sweep.
    # run_clustering_stage(config_path=CONFIG_PATH)

    print("Done. Enabled stages write artifacts under cfg.output_dir.")


if __name__ == "__main__":
    main()

