#!/usr/bin/env python3
"""
Optional CLI wrapper around run_train_stage. Prefer the standard path:

  1. Merge pipeline_fragment.dedup_task_identity_2.json (or _v1) into pipeline_config.json
  2. python core/main.py   # run_gnn() reads run_id + pair_training.pair_dataset_csv

This script exists for ad-hoc overrides only.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_GNN = _REPO / "core" / "GNN"
_CORE = _REPO / "core"
for p in (_GNN, _CORE, _REPO):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from config.pipeline_config import load_pipeline_config  # noqa: E402
from steps.train_stage import run_train_stage  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-id", required=True, help="Logical run id (output/runs/<run-id>/)")
    ap.add_argument(
        "--pair-dataset-csv",
        required=True,
        help="pair_training_dataset.csv from graph bundle",
    )
    ap.add_argument(
        "--graph-pt",
        default="core/graph/output/main_gnn_pu_1_no_ts_dedup_task_identity_hetero.pt",
        help="Hetero graph used for email embeddings",
    )
    ap.add_argument(
        "--runs-parent",
        default="output/runs",
        help="Parent directory for training runs",
    )
    ap.add_argument("--device", default=None, help="Override pipeline_config device")
    args = ap.parse_args()

    cfg = load_pipeline_config(project_root=_REPO)
    training_cfg = {**dict(cfg.get("pair_training") or {}), **dict(cfg.get("training") or {})}
    training_cfg["training_objective"] = "pair_supervision"
    training_cfg["pair_dataset_csv"] = str(args.pair_dataset_csv)

    device = args.device or str(cfg.get("device") or "cpu")
    to_undirected = bool(cfg.get("to_undirected", True))

    graph_pt = Path(args.graph_pt)
    if not graph_pt.is_absolute():
        graph_pt = (_REPO / graph_pt).resolve()

    result = run_train_stage(
        graph_path=str(graph_pt),
        runs_parent=str(args.runs_parent),
        run_id=str(args.run_id),
        training_cfg=training_cfg,
        device_pref=device,
        to_undirected=to_undirected,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
