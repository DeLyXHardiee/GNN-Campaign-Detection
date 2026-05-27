#!/usr/bin/env python3
"""Step 1: build timestamp-enabled heterograph for thesis GNN runs."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from seed_candidate_workflow.utils.final_gnn_timestamp_es_thesis import (  # noqa: E402
    load_manifest,
    repo_root,
    resolve_repo_path,
    steps_dir,
    thesis_dir,
    write_graph_timestamp_summary,
)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, default=None)
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--rebuild", action="store_true")
    args = p.parse_args()

    repo = repo_root()
    manifest = load_manifest(args.manifest)
    graph_pt = resolve_repo_path(repo, str(manifest["graph_pt"]))
    meta_json = resolve_repo_path(repo, str(manifest["graph_meta_json"]))
    tdir = thesis_dir(repo, manifest)

    if args.skip_existing and graph_pt.is_file() and meta_json.is_file() and not args.rebuild:
        print(f"[step01] skip (graph exists): {graph_pt}")
    else:
        frag = resolve_repo_path(repo, str(manifest["pipeline_fragment_graph"]))
        subprocess.run(
            [sys.executable, str(repo / "seed_candidate_workflow/scripts/merge_pipeline_fragment.py"), str(frag)],
            cwd=str(repo),
            check=True,
        )
        subprocess.run([sys.executable, str(repo / "core/main.py"), "graph"], cwd=str(repo), check=True)

    summary = write_graph_timestamp_summary(
        graph_pt=graph_pt,
        meta_json=meta_json,
        out_path=tdir / "graph_timestamp_summary.json",
        manifest=manifest,
    )
    report = {"graph_pt": str(graph_pt), "graph_summary": summary}
    out = steps_dir(repo, manifest) / "step01_build_graph_report.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
