#!/usr/bin/env python3
"""
CLI entry for post-training PU pair score separation (same vs cross campaign on GT-covered pairs).

Run from repository root, for example:

  python seed_candidate_workflow/scripts/run_pair_score_separation_analysis.py ^
    --run-dir output/runs/main_gnn_pu_1_no_ts_dedup_task_identity_11 ^
    --graph-pt core/graph/output/main_gnn_pu_1_no_ts_dedup_task_identity_6_hetero.pt ^
    --gt-path data/groundtruth/ground_truth.dedup_task_identity.json

  # Multiple GT JSONs (repeat --gt-path):
  python seed_candidate_workflow/scripts/run_pair_score_separation_analysis.py ... ^
    --gt-path data/groundtruth/ground_truth_merged.json ^
    --gt-path data/groundtruth/ground_truth.json
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
if str(_REPO / "core") not in sys.path:
    sys.path.insert(0, str(_REPO / "core"))
if str(_REPO / "core" / "GNN") not in sys.path:
    sys.path.insert(0, str(_REPO / "core" / "GNN"))

from seed_candidate_workflow.utils.pair_score_separation import main  # noqa: E402


if __name__ == "__main__":
    main()
