"""
Dedup-GT vs expanded-GT evaluation gap analysis (read-only on existing artifacts).

Example:

  python analysis/scripts/run_dedup_vs_expanded_gap_analysis.py \\
    --config analysis/configs/dedup_vs_expanded_gap.default.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from seed_candidate_workflow.utils import graph_structure_helpers as gh
from analysis.utils.dedup_vs_expanded_gap_analysis import (
    build_config_from_cli_and_json,
    run_dedup_vs_expanded_gap_analysis,
    write_dedup_vs_expanded_gap_outputs,
)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=None, help="JSON config (see analysis/configs/)")
    ap.add_argument("--out-dir", type=str, default=None, help="Output directory for CSV/JSON/HTML")
    ap.add_argument("--write-html", action="store_true", help="Write dedup_vs_expanded_gap_lossy_campaigns.html")
    ap.add_argument("--gt-dedup-json", type=str, default=None)
    ap.add_argument("--gt-expanded-json", type=str, default=None)
    ap.add_argument("--community-multi-gt-summary-json", type=str, default=None)
    ap.add_argument("--dedup-best-json", type=str, default=None)
    ap.add_argument("--expanded-best-json", type=str, default=None)
    ap.add_argument("--anchor-run-dir", type=str, default=None)
    ap.add_argument("--scored-edges-csv", type=str, default=None)
    ap.add_argument("--dedup-collapse-out-dir", type=str, default=None)
    ap.add_argument("--external-id-map-csv", type=str, default=None)
    ap.add_argument("--candidate-union-csv", type=str, default=None)
    ap.add_argument("--pair-training-csv", type=str, default=None)
    args = ap.parse_args()

    project_root = gh.find_project_root()
    cfg_json: dict[str, Any] = {}
    if args.config is not None:
        cfg_json = _load_json(args.config)

    overrides = {
        "gt_dedup_json": args.gt_dedup_json,
        "gt_expanded_json": args.gt_expanded_json,
        "community_multi_gt_summary_json": args.community_multi_gt_summary_json,
        "dedup_best_json": args.dedup_best_json,
        "expanded_best_json": args.expanded_best_json,
        "anchor_run_dir": args.anchor_run_dir,
        "scored_edges_csv": args.scored_edges_csv,
        "dedup_collapse_out_dir": args.dedup_collapse_out_dir,
        "external_id_map_csv": args.external_id_map_csv,
        "candidate_union_csv": args.candidate_union_csv,
        "pair_training_csv": args.pair_training_csv,
        "out_dir": args.out_dir,
    }
    out_raw = overrides.pop("out_dir", None)
    out_dir = Path(str(out_raw or cfg_json.get("out_dir") or "output/analysis/dedup_vs_expanded_gap")).expanduser()
    if not out_dir.is_absolute():
        out_dir = (project_root / out_dir).resolve()

    cfg = build_config_from_cli_and_json(
        project_root=project_root,
        config_json=cfg_json,
        overrides=overrides,
    )
    result = run_dedup_vs_expanded_gap_analysis(cfg)
    paths = write_dedup_vs_expanded_gap_outputs(result, out_dir, write_html=bool(args.write_html))
    print(json.dumps({"output_paths": paths}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
