#!/usr/bin/env python3
"""
Thesis-ready learned pair score statistics (expanded GT, same vs cross campaign).

Example (_14_only_mlp, same inputs as community exp62):

  python seed_candidate_workflow/scripts/run_pair_score_thesis_diagnostics.py ^
    --run-dir output/runs/main_gnn_pu_1_no_ts_dedup_task_identity_14_only_mlp ^
    --graph-pt core/graph/output/main_gnn_pu_1_no_ts_dedup_task_identity_12_hetero.pt ^
    --gt-path data/groundtruth/ground_truth.json ^
    --pair-csv seed_candidate_workflow/output/graph_bundles/main_gnn_pu_1_no_ts_dedup_task_identity_13/pair_training/main_gnn_pu_1_no_ts_dedup_task_identity_13/pair_training_dataset.csv ^
    --scoring-run-id main_gnn_pu_1_no_ts_dedup_task_identity_14_only_mlp__expanded_full_gt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
if str(_REPO / "core") not in sys.path:
    sys.path.insert(0, str(_REPO / "core"))
if str(_REPO / "core" / "GNN") not in sys.path:
    sys.path.insert(0, str(_REPO / "core" / "GNN"))

from seed_candidate_workflow.utils.pair_model_inference import (  # noqa: E402
    resolve_pair_dataset_csv_path,
    resolve_pair_supervision_run_artifacts,
)
from seed_candidate_workflow.utils.pair_score_thesis_diagnostics import (  # noqa: E402
    run_thesis_pair_score_diagnostics,
)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Thesis pair score statistics vs expanded GT.")
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--graph-pt", type=Path, required=True)
    p.add_argument(
        "--gt-path",
        type=Path,
        default=_REPO / "data/groundtruth/ground_truth.json",
        help="Expanded GT (default: ground_truth.json).",
    )
    p.add_argument("--pair-csv", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--scoring-run-id", type=str, default=None)
    p.add_argument(
        "--scored-pairs-csv",
        type=Path,
        default=None,
        help="Optional pre-scored pairs with pu_score (else MLP inference).",
    )
    p.add_argument("--checkpoint", type=str, default="best_model.pt")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--no-to-undirected", action="store_true")
    args = p.parse_args(argv)

    run_dir = args.run_dir.resolve()
    if args.scored_pairs_csv is None:
        edge_scores_csv = (run_dir / "edge_gnn_pair_scores.csv").resolve()
        if not edge_scores_csv.is_file():
            resolve_pair_supervision_run_artifacts(
                run_dir,
                checkpoint_name=str(args.checkpoint),
                project_root=_REPO,
            )

    pair_csv = (
        args.pair_csv.resolve()
        if args.pair_csv is not None
        else resolve_pair_dataset_csv_path(run_dir, project_root=_REPO)
    )

    out = run_thesis_pair_score_diagnostics(
        run_dir=run_dir,
        graph_pt=args.graph_pt.resolve(),
        pair_csv=pair_csv,
        gt_path=args.gt_path.resolve(),
        output_dir=args.output_dir,
        checkpoint_name=str(args.checkpoint),
        device=str(args.device),
        to_undirected=not bool(args.no_to_undirected),
        scoring_run_id=args.scoring_run_id,
        scored_pairs_csv=args.scored_pairs_csv,
    )
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
