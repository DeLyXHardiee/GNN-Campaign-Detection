#!/usr/bin/env python3
"""Step 2: verify final pair universe, graph, and no time-gating in candidate generation."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from seed_candidate_workflow.utils.final_14_only_mlp_thesis import verify_time_gating_disabled  # noqa: E402
from seed_candidate_workflow.utils.final_gnn_timestamp_es_thesis import (  # noqa: E402
    load_manifest,
    repo_root,
    resolve_repo_path,
    steps_dir,
)
from seed_candidate_workflow.utils.timestamp_ablation_14_only_mlp import pair_universe_stats  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, default=None)
    p.add_argument("--skip-existing", action="store_true")
    args = p.parse_args()

    repo = repo_root()
    manifest = load_manifest(args.manifest)
    out = steps_dir(repo, manifest) / "step02_verify_inputs_report.json"
    if args.skip_existing and out.is_file():
        print(f"[step02] skip (report exists): {out}")
        print(out.read_text(encoding="utf-8"))
        return 0
    pair_csv = resolve_repo_path(repo, str(manifest["final_pair_dataset_csv"]))
    graph_pt = resolve_repo_path(repo, str(manifest["graph_pt"]))
    cand_cfg = resolve_repo_path(repo, str(manifest["candidate_generation_config"]))

    if not pair_csv.is_file():
        raise FileNotFoundError(
            f"Final pair dataset missing: {pair_csv}. Run final_14_only_mlp_step02_materialize first."
        )
    if not graph_pt.is_file():
        raise FileNotFoundError(f"Timestamp heterograph missing: {graph_pt}. Run step01 first.")

    stats = pair_universe_stats(pair_csv)
    tg = verify_time_gating_disabled(cand_cfg)
    df = pd.read_csv(pair_csv, low_memory=False, nrows=5)
    has_ts_col = "time_gap_seconds_min" in df.columns

    report = {
        "pair_csv": str(pair_csv),
        "pair_universe_stats": stats,
        "graph_pt": str(graph_pt),
        "time_gating_check": tg,
        "pair_csv_has_time_gap_column": bool(has_ts_col),
        "note": (
            "GNN training uses heterograph email timestamp node features; pair CSV is the final "
            "timestamp-materialized universe (same keys as _13, log1p time_gap in explicit columns for GNN+features)."
        ),
    }
    if not tg.get("all_gating_disabled", False):
        raise RuntimeError(f"Time gating not disabled in candidate config: {cand_cfg}")

    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
