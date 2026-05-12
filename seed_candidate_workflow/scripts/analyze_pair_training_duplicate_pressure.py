#!/usr/bin/env python3
"""
Pair-level duplicate / easy-edge pressure analysis.

Runbook
-------
1) Build duplicate membership from the same MISP JSON slice you care about::

     python data/misp/analyze_misp_duplicate_emails.py \\
       --input-json data/misp/incidents-lake-misp.json \\
       --out-dir data/misp/duplicate_email_analysis

   This writes ``email_duplicate_cluster.parquet`` and ``misp_loaded_external_ids.parquet``.

2) Run this script on ``pair_training_dataset.csv``::

     python seed_candidate_workflow/scripts/analyze_pair_training_duplicate_pressure.py \\
       --pair-csv seed_candidate_workflow/output/graph_bundles/<graph_id>/pair_training/<graph_id>/pair_training_dataset.csv \\
       --email-cluster-parquet data/misp/duplicate_email_analysis/email_duplicate_cluster.parquet \\
       --graph-meta-json <path/to/graph.meta.json> \\
       --misp-loaded-ids-parquet data/misp/duplicate_email_analysis/misp_loaded_external_ids.parquet

   Optional: mirror training splits using the same ratios/seed as a finished run::

     ... --apply-split --training-config-json output/runs/<run_id>/training_config.json

   Optional: write a wide debug parquet of labeled pair rows::

     ... --write-augmented-parquet
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from seed_candidate_workflow.utils.pair_duplicate_pressure_analysis import run_pair_duplicate_pressure


def _default_out_dir(pair_csv: Path) -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return pair_csv.resolve().parent / f"pair_duplicate_pressure_{ts}"


def _load_split_from_training_config(path: Path) -> tuple[float, float, int]:
    cfg = json.loads(path.read_text(encoding="utf-8"))
    return (
        float(cfg.get("pair_val_ratio", 0.1)),
        float(cfg.get("pair_test_ratio", 0.1)),
        int(cfg.get("pair_split_seed", cfg.get("torch_seed", 42))),
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pair-csv", type=Path, required=True)
    p.add_argument("--email-cluster-parquet", type=Path, required=True)
    p.add_argument(
        "--misp-loaded-ids-parquet",
        type=Path,
        default=None,
        help="Default: <parent of email-cluster-parquet>/misp_loaded_external_ids.parquet if that file exists.",
    )
    p.add_argument("--graph-meta-json", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument(
        "--include-unmapped-graph-rows",
        action="store_true",
        help="Do not filter to rows with both graph_email_idx_* present (not the training contract).",
    )
    p.add_argument("--apply-split", action="store_true", help="Emit split_projection using pair_train split logic.")
    p.add_argument("--pair-val-ratio", type=float, default=0.1)
    p.add_argument("--pair-test-ratio", type=float, default=0.1)
    p.add_argument("--pair-split-seed", type=int, default=42)
    p.add_argument(
        "--training-config-json",
        type=Path,
        default=None,
        help="When set with --apply-split, overrides split ratios/seed from this file (pair_* keys).",
    )
    p.add_argument("--write-augmented-parquet", action="store_true")
    args = p.parse_args()

    pair_csv = args.pair_csv
    mem = args.email_cluster_parquet
    mip = args.misp_loaded_ids_parquet
    if mip is None:
        cand = mem.resolve().parent / "misp_loaded_external_ids.parquet"
        mip = cand if cand.is_file() else None

    out_dir = args.out_dir or _default_out_dir(pair_csv)

    vr, tr, seed = float(args.pair_val_ratio), float(args.pair_test_ratio), int(args.pair_split_seed)
    if args.training_config_json is not None:
        vr, tr, seed = _load_split_from_training_config(args.training_config_json.expanduser().resolve())

    summary = run_pair_duplicate_pressure(
        pair_csv=pair_csv,
        membership_parquet=mem,
        out_dir=out_dir,
        graph_meta_json=args.graph_meta_json,
        misp_loaded_ids_parquet=mip,
        training_rows_only=not bool(args.include_unmapped_graph_rows),
        apply_split=bool(args.apply_split),
        pair_val_ratio=vr,
        pair_test_ratio=tr,
        pair_split_seed=seed,
        write_augmented_parquet=bool(args.write_augmented_parquet),
    )
    out = {"out_dir": str(out_dir), **summary.get("artifacts", {})}
    if "split_summary_json" in summary:
        out["pair_duplicate_pressure_split_summary_json"] = summary["split_summary_json"]
    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
