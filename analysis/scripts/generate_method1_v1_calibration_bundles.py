"""
Generate all Method 1 Version 1 calibration refined-edge bundles (unsupervised).

Usage (from repo root)::

    python analysis/scripts/generate_method1_v1_calibration_bundles.py

Requires Step 2 graph artifacts under analysis/output/semantic_shard_step2_graph/.
Writes to analysis/output/semantic_shard_method1_diagnostics/method1_v1_calibration_runs/<variant_id>/.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.utils import semantic_shard_step3_helpers as s3
from analysis.utils.method1_v1_calibration_variants import generate_all_v1_calibration_bundles
from analysis.utils.semantic_shard_edge_refinement_method1 import Method1RefinementConfig
from analysis.utils.semantic_shard_method1_diagnostics_helpers import load_method1_config_json


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--step2-dir",
        type=Path,
        default=PROJECT_ROOT / "analysis" / "output" / "semantic_shard_step2_graph",
        help="Directory with Step 2 shard graph CSVs",
    )
    p.add_argument(
        "--method1-dir",
        type=Path,
        default=PROJECT_ROOT / "analysis" / "output" / "semantic_shard_method1",
        help="Optional: load semantic_shard_method1_config.json defaults from here",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT
        / "analysis"
        / "output"
        / "semantic_shard_method1_diagnostics"
        / "method1_v1_calibration_runs",
        help="Root directory for per-variant subfolders",
    )
    p.add_argument("--force", action="store_true", help="Overwrite existing bundle CSVs")
    args = p.parse_args()

    _, baseline_edges_df, _ = s3.load_step2_artifacts(args.step2_dir)
    cfg_json = load_method1_config_json(args.method1_dir)
    base_cfg = Method1RefinementConfig.from_dict(cfg_json) if cfg_json else Method1RefinementConfig()
    done = generate_all_v1_calibration_bundles(
        baseline_edges_df,
        base_cfg=base_cfg,
        runs_root=args.out_dir,
        force=args.force,
    )
    print(f"Wrote {len(done)} variant bundles under {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
