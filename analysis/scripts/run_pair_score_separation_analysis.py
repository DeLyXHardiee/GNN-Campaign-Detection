#!/usr/bin/env python3
"""
CLI entry for post-training PU pair score separation (same vs cross campaign on GT-covered pairs).

Run from repository root, for example:

  python analysis/scripts/run_pair_score_separation_analysis.py ^
    --run-dir core/GNN/outputs/pair_pu_001 ^
    --graph-pt core/graph/output/incidents-lake-misp-large_hetero.pt ^
    --gt-dir data/groundtruth
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

from analysis.utils.pair_score_separation import main  # noqa: E402


if __name__ == "__main__":
    main()
