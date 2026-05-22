#!/usr/bin/env python3
"""
Write a scored seed-candidate PairGraph CSV (email_i, email_j, edge_weight, …) for gap analysis.

The community sweep consumes PU-scored edges in memory; this reproduces that scoring so
``run_dedup_vs_expanded_gap_analysis.py`` can read weights from disk.

Example (repo root, same paths as exp50 / graph_id _10):

  python analysis/scripts/materialize_pu_scored_edges_for_gap.py ^
    --unscored-csv seed_candidate_workflow/output/graph_bundles/main_gnn_pu_1_no_ts_dedup_task_identity_10/seed_candidate/main_gnn_pu_1_no_ts_dedup_task_identity_10/seed_candidate_pairgraph_unscored.csv ^
    --out-csv output/analysis/materialized_pu_scored_edges_main_gnn_pu_1_no_ts_dedup_task_identity_10.csv ^
    --pu-run-dir output/runs/main_gnn_pu_1_no_ts_dedup_task_identity_10 ^
    --graph-pt core/graph/output/main_gnn_pu_1_no_ts_dedup_task_identity_6_hetero.pt ^
    --pair-dataset-csv seed_candidate_workflow/output/graph_bundles/main_gnn_pu_1_no_ts_dedup_task_identity_10/pair_training/main_gnn_pu_1_no_ts_dedup_task_identity_10/pair_training_dataset.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from seed_candidate_workflow.utils import graph_structure_helpers as gh
from seed_candidate_workflow.utils.graph_scorer_registry import apply_scorer


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--unscored-csv", type=Path, required=True)
    ap.add_argument("--out-csv", type=Path, required=True)
    ap.add_argument("--pu-run-dir", type=Path, required=True, help="GNN run dir (contains training_config.json / checkpoint layout)")
    ap.add_argument("--graph-pt", type=Path, required=True)
    ap.add_argument("--pair-dataset-csv", type=Path, required=True)
    ap.add_argument("--checkpoint", type=str, default="best_model.pt")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--no-to-undirected", action="store_true")
    args = ap.parse_args()

    root = gh.find_project_root()
    unscored = args.unscored_csv.expanduser()
    if not unscored.is_absolute():
        unscored = (root / unscored).resolve()
    out = args.out_csv.expanduser()
    if not out.is_absolute():
        out = (root / out).resolve()
    pu_run_dir = args.pu_run_dir.expanduser()
    if not pu_run_dir.is_absolute():
        pu_run_dir = (root / pu_run_dir).resolve()
    graph_pt = args.graph_pt.expanduser()
    if not graph_pt.is_absolute():
        graph_pt = (root / graph_pt).resolve()
    pair_csv = args.pair_dataset_csv.expanduser()
    if not pair_csv.is_absolute():
        pair_csv = (root / pair_csv).resolve()

    df = pd.read_csv(unscored, low_memory=False)
    score_params = {
        "pu_run": {
            "run_dir": str(pu_run_dir),
            "graph_pt": str(graph_pt),
            "checkpoint": str(args.checkpoint),
            "device": str(args.device),
            "no_to_undirected": bool(args.no_to_undirected),
            "pair_dataset_csv": str(pair_csv),
        },
        "seed_edge_weight": 1.0,
        "weight_mode": "raw_score",
        "export_non_seed_min_pu_score": 0.0,
    }
    sr = apply_scorer(
        score_mode="seed_candidate_pu_v1",
        graph_kind="seed_candidate",
        score_params=score_params,
        payload={"candidate_union_df": df},
        diagnostics_cfg={},
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    sr.scored_all.to_csv(out, index=False)
    print(str(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
