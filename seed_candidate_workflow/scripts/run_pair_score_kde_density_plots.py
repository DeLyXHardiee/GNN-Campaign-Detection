#!/usr/bin/env python3
"""
Write same-vs-cross campaign KDE density overlays (thesis figures) without full pair score separation.

Example (_14_only_mlp, dedup GT):

  python seed_candidate_workflow/scripts/run_pair_score_kde_density_plots.py ^
    --run-dir output/runs/main_gnn_pu_1_no_ts_dedup_task_identity_14_only_mlp ^
    --graph-pt core/graph/output/main_gnn_pu_1_no_ts_dedup_task_identity_12_hetero.pt ^
    --gt-path data/groundtruth/ground_truth.dedup_task_identity.json ^
    --pair-csv seed_candidate_workflow/output/graph_bundles/main_gnn_pu_1_no_ts_dedup_task_identity_13/pair_training/main_gnn_pu_1_no_ts_dedup_task_identity_13/pair_training_dataset.csv
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

from seed_candidate_workflow.utils.pair_score_separation import (  # noqa: E402
    _gt_json_paths_from_dir,
    run_pair_score_kde_density_plots,
)
from seed_candidate_workflow.utils.pair_score_thesis_diagnostics import (  # noqa: E402
    run_thesis_pair_score_diagnostics,
)
from seed_candidate_workflow.utils.pair_model_inference import (  # noqa: E402
    resolve_pair_dataset_csv_path,
    resolve_pair_supervision_run_artifacts,
)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description="Same vs cross campaign KDE density plots for learned pair scores."
    )
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--graph-pt", type=Path, required=True)
    p.add_argument("--pair-csv", type=Path, default=None)
    p.add_argument("--gt-dir", type=Path, default=None)
    p.add_argument("--gt-path", type=Path, action="append", default=None)
    p.add_argument("--gt-include-report-json", action="store_true")
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--checkpoint", type=str, default="best_model.pt")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--no-to-undirected", action="store_true")
    p.add_argument(
        "--emit-thesis-diagnostics",
        action="store_true",
        help="Also write thesis_pair_score_* under pair_score_separation/thesis_score_diagnostics/ "
        "(uses first --gt-path only).",
    )
    args = p.parse_args(argv)

    if bool(args.gt_dir) == bool(args.gt_path):
        raise SystemExit("Provide exactly one of: --gt-dir, or one or more --gt-path.")

    run_dir = args.run_dir.resolve()
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
    if args.gt_dir is not None:
        gt_paths = _gt_json_paths_from_dir(
            args.gt_dir, include_report_json=bool(args.gt_include_report_json)
        )
    else:
        gt_paths = [Path(p).resolve() for p in (args.gt_path or [])]

    out = run_pair_score_kde_density_plots(
        run_dir=run_dir,
        graph_pt=args.graph_pt.resolve(),
        pair_csv=pair_csv,
        gt_paths=gt_paths,
        output_dir=args.output_dir,
        checkpoint_name=str(args.checkpoint),
        device=str(args.device),
        to_undirected=not bool(args.no_to_undirected),
    )
    if args.emit_thesis_diagnostics and gt_paths:
        thesis_out = run_thesis_pair_score_diagnostics(
            run_dir=run_dir,
            graph_pt=args.graph_pt.resolve(),
            pair_csv=pair_csv,
            gt_path=gt_paths[0],
            output_dir=(args.output_dir or (run_dir / "pair_score_separation")).resolve()
            / "thesis_score_diagnostics",
            checkpoint_name=str(args.checkpoint),
            device=str(args.device),
            to_undirected=not bool(args.no_to_undirected),
        )
        out["thesis_diagnostics"] = {
            "output_paths": thesis_out.get("output_paths"),
            "gt_path": thesis_out.get("gt_path"),
        }

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
