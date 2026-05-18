#!/usr/bin/env python3
"""
CLI: low-band candidate feature discovery (same vs cross unlabeled, score <= band).

Example (run 5, dedup GT):

  python seed_candidate_workflow/scripts/run_pair_low_band_feature_discovery.py ^
    --run-dir output/runs/main_gnn_pu_1_no_ts_dedup_task_identity_5 ^
    --graph-pt core/graph/output/main_gnn_pu_1_no_ts_dedup_task_identity_5_hetero.pt ^
    --gt-path data/groundtruth/ground_truth.dedup_task_identity.json
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

from seed_candidate_workflow.utils.pair_low_band_feature_discovery import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
