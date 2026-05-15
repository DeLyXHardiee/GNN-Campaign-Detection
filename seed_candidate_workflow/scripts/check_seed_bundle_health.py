#!/usr/bin/env python3
"""
Guardrail: fail if seed union components collapsed or pair_training is dominated by same_seed_component rows.

Example::

  python seed_candidate_workflow/scripts/check_seed_bundle_health.py ^
    --anchor-seed-summary seed_candidate_workflow/output/graph_bundles/my_graph/seed/my_graph/seed_generation_XXX/anchor_seed_summary.json ^
    --pair-training-summary seed_candidate_workflow/output/graph_bundles/my_graph/pair_training/my_graph/pair_training_dataset_summary.json

See docs/experiments/seed_union_acceptance_thresholds.md for default thresholds rationale.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from seed_candidate_workflow.utils.seed_bundle_health import run_health_checks


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--anchor-seed-summary",
        type=Path,
        default=None,
        help="anchor_seed_summary.json from latest seed_generation_*",
    )
    p.add_argument(
        "--pair-training-summary",
        type=Path,
        default=None,
        help="pair_training_dataset_summary.json",
    )
    p.add_argument("--max-union-largest-component", type=int, default=400)
    p.add_argument("--min-union-components", type=int, default=450)
    p.add_argument("--max-same-seed-component-fraction", type=float, default=0.48)
    args = p.parse_args(argv)

    if args.anchor_seed_summary is None and args.pair_training_summary is None:
        p.error("provide at least one of --anchor-seed-summary or --pair-training-summary")

    errs = run_health_checks(
        anchor_seed_summary=args.anchor_seed_summary,
        pair_training_summary=args.pair_training_summary,
        max_union_largest_component=args.max_union_largest_component,
        min_union_components=args.min_union_components,
        max_same_seed_component_fraction=args.max_same_seed_component_fraction,
    )

    if not errs:
        print("OK: seed bundle health checks passed.")
        return 0

    print("FAILED:", file=sys.stderr)
    for e in errs:
        print(f"  - {e}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
