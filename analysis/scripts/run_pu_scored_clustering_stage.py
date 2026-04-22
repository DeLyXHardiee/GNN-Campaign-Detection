#!/usr/bin/env python3
"""
Run PU-scored anchor clustering: score pair universe, build edges, community sweep.

  python analysis/scripts/run_pu_scored_clustering_stage.py ^
    --config analysis/configs/anchor_pu_scored_clustering.default.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from analysis.utils.anchor_pu_scored_clustering_helpers import run_anchor_pu_scored_clustering_stage


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--config",
        type=Path,
        default=_REPO / "analysis" / "configs" / "anchor_pu_scored_clustering.default.json",
        help="JSON config for PU scored clustering stage.",
    )
    args = p.parse_args()
    cfg_path = args.config.expanduser().resolve()
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    cfg["_pipeline_config_path"] = str(cfg_path)
    out = run_anchor_pu_scored_clustering_stage(cfg)
    print(json.dumps({k: out[k] for k in sorted(out)}, indent=2))


if __name__ == "__main__":
    main()
