"""CLI: build email-email pair training dataset from seed + candidate union CSVs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from seed_candidate_workflow.utils.pair_training_dataset_helpers import build_pair_training_dataset


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--seed-edges-all",
        type=Path,
        required=True,
        help="Path to seed_edges_all.csv (email_i, email_j).",
    )
    p.add_argument(
        "--candidate-union",
        type=Path,
        required=True,
        help="Path to candidate_union.csv.",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Directory to write pair_training_dataset*.csv/json(+parquet).",
    )
    p.add_argument(
        "--graph-meta-json",
        type=Path,
        default=None,
        help="Path to hetero graph .meta.json for external_id -> email row index. "
        "If omitted, tries resolve_graph_analysis_paths() from repo root.",
    )
    p.add_argument(
        "--graph-id",
        type=str,
        dest="graph_id",
        default=None,
        help="Optional graph id recorded in pair_training_dataset_summary.json metadata.",
    )
    p.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="Project root when resolving default graph meta (optional).",
    )
    p.add_argument(
        "--pipeline-json",
        type=Path,
        default=None,
        help="Optional pipeline_config.json; uses pair_training.reliable_negative_pool when present.",
    )
    p.add_argument("--no-parquet", action="store_true", help="Skip writing Parquet.")
    p.add_argument("--no-rejects", action="store_true", help="Skip writing rejects CSV.")
    args = p.parse_args()

    rn_pool = None
    if args.pipeline_json is not None:
        ppath = args.pipeline_json.expanduser().resolve()
        cfg = json.loads(ppath.read_text(encoding="utf-8"))
        rn_pool = (cfg.get("pair_training") or {}).get("reliable_negative_pool")

    out = build_pair_training_dataset(
        seed_edges_all_csv=args.seed_edges_all,
        candidate_union_csv=args.candidate_union,
        output_dir=args.out_dir,
        graph_meta_json=args.graph_meta_json,
        graph_id=args.graph_id,
        write_parquet=not args.no_parquet,
        write_rejects_csv=not args.no_rejects,
        project_root=args.project_root.expanduser().resolve() if args.project_root else None,
        reliable_negative_pool=rn_pool,
    )
    print(json.dumps({k: v for k, v in out.items() if k != "summary"}, indent=2))
    pc = (out.get("summary") or {}).get("pair_counts") or {}
    mq = (out.get("summary") or {}).get("mapping_quality") or {}
    print(
        f"pairs: total={pc.get('n_unique_pairs_final')} "
        f"positive={pc.get('n_positive_pairs')} unlabeled={pc.get('n_unlabeled_pairs')} "
        f"reliable_negative={pc.get('n_reliable_negative_pairs', 0)} | "
        f"mapped_both={mq.get('n_rows_both_graph_indices_present')} "
        f"missing_any={mq.get('n_rows_missing_either_graph_index')}"
    )


if __name__ == "__main__":
    main()
