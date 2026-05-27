#!/usr/bin/env python3
"""Run thesis graph-construction diagnostics (dedup, relation channels, config audit)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from seed_candidate_workflow.utils.final_14_only_mlp_thesis import load_manifest, repo_root, resolve_repo_path
from seed_candidate_workflow.utils.thesis_graph_construction_diagnostics import run_all_diagnostics


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output root (default: thesis bundle graph_construction_diagnostics/)",
    )
    p.add_argument("--manifest", type=Path, default=None, help="final_14_only_mlp manifest.json")
    args = p.parse_args()

    repo = repo_root()
    manifest = load_manifest(args.manifest) if args.manifest else load_manifest()
    out_dir = args.out_dir or resolve_repo_path(
        repo,
        str(manifest.get("thesis_output_dir") or "seed_candidate_workflow/output/final_14_only_mlp_timestamp_es_thesis"),
    ) / "graph_construction_diagnostics"

    pair_csv = resolve_repo_path(repo, str(manifest.get("baseline_pair_dataset_csv") or ""))
    if not pair_csv.is_file():
        pair_csv = None

    paths = run_all_diagnostics(
        out_dir=out_dir,
        pair_training_csv=pair_csv,
    )
    print(json.dumps(paths, indent=2))
    print(f"\nWrote graph-construction diagnostics to:\n  {out_dir.resolve()}")


if __name__ == "__main__":
    main()
