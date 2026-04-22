#!/usr/bin/env python3
"""
Threshold-conditioned retention analysis on PU-scored candidate pairs.

Example:
  python analysis/scripts/run_pu_threshold_retention_analysis.py ^
    --scored-csv analysis/output/anchor_candidates/deafult_anchor_seeds/candidate_generation_20260419T115430Z/pu_scored_candidate_edges_all.csv ^
    --gt-dir data/groundtruth
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from analysis.utils.pu_threshold_retention_analysis import (  # noqa: E402
    DEFAULT_THRESHOLDS,
    run_pu_threshold_retention_analysis,
)


def _gt_paths_from_dir(gt_dir: Path, *, include_report_json: bool) -> list[Path]:
    d = gt_dir.expanduser().resolve()
    if not d.is_dir():
        raise SystemExit(f"--gt-dir is not a directory: {d}")
    paths = sorted(d.glob("*.json"))
    if not include_report_json:
        paths = [p for p in paths if "report" not in p.name.lower()]
    return paths


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scored-csv", type=Path, required=True, help="pu_scored_candidate_edges_all.csv")
    p.add_argument("--gt", type=Path, action="append", default=[], help="Ground truth JSON (repeat)")
    p.add_argument("--gt-dir", type=Path, default=None, help="Use every *.json in this dir")
    p.add_argument(
        "--gt-include-report-json",
        action="store_true",
        help="With --gt-dir, include files with 'report' in filename",
    )
    p.add_argument(
        "--thresholds",
        type=str,
        default=",".join(str(x) for x in DEFAULT_THRESHOLDS),
        help="Comma-separated threshold list, e.g. 0,0.05,0.1,0.2",
    )
    p.add_argument("--output-dir", type=Path, default=None, help="Default: <scored-csv-dir>/pu_threshold_retention")
    p.add_argument("--no-keep-seeds-always", action="store_true", help="Do not force seed edges kept")
    p.add_argument("--no-plots", action="store_true")
    args = p.parse_args()

    gt_paths: list[Path] = [Path(x).expanduser().resolve() for x in args.gt]
    if args.gt_dir is not None:
        gt_paths.extend(
            _gt_paths_from_dir(args.gt_dir, include_report_json=bool(args.gt_include_report_json))
        )
    # dedup order-preserving
    seen: set[Path] = set()
    uniq: list[Path] = []
    for pth in gt_paths:
        if pth not in seen:
            seen.add(pth)
            uniq.append(pth)
    gt_paths = uniq
    if not gt_paths:
        raise SystemExit("Provide at least one --gt or --gt-dir")

    thresholds = [float(x.strip()) for x in str(args.thresholds).split(",") if x.strip()]
    out = run_pu_threshold_retention_analysis(
        scored_pairs_csv=args.scored_csv.expanduser().resolve(),
        gt_paths=gt_paths,
        thresholds=thresholds,
        output_dir=args.output_dir,
        keep_seeds_always=not bool(args.no_keep_seeds_always),
        make_plots=not bool(args.no_plots),
    )
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

