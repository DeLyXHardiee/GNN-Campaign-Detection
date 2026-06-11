"""
Bridge-candidate retrieval + scoring experiment for missing (non-edge) pairs.

Example (from repo root):

  python seed_candidate_workflow/scripts/run_bridge_candidate_experiment.py ^
    --run-dir output/runs/main_gnn_pu_1_no_ts_dedup_task_identity_13 ^
    --graph-pt core/graph/output/main_gnn_pu_1_no_ts_dedup_task_identity_12_hetero.pt ^
    --gt-path data/groundtruth/ground_truth.dedup_task_identity.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
if str(_REPO / "core") not in sys.path:
    sys.path.insert(0, str(_REPO / "core"))
if str(_REPO / "core" / "GNN") not in sys.path:
    sys.path.insert(0, str(_REPO / "core" / "GNN"))

from seed_candidate_workflow.utils.bridge_candidate_experiment import (  # noqa: E402
    BridgeCandidateConfig,
    run_bridge_candidate_experiment,
)


def _parse_thresholds(raw: str | None) -> tuple[float, ...]:
    if not raw:
        return (0.8, 0.9)
    parts = [p.strip() for p in str(raw).split(",") if p.strip()]
    return tuple(float(p) for p in parts)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description="Retrieve broad missing-pair bridge candidates and score with trained PU pair model."
    )
    p.add_argument("--run-dir", type=Path, required=True, help="PU pair supervision run directory")
    p.add_argument("--graph-pt", type=Path, required=True, help="Hetero graph .pt used for training")
    p.add_argument("--gt-path", type=Path, default=None, help="Optional GT JSON for diagnostic-only eval")
    p.add_argument("--pair-csv", type=Path, default=None, help="Override pair_training_dataset.csv")
    p.add_argument("--candidate-union-csv", type=Path, default=None)
    p.add_argument("--seed-edges-csv", type=Path, default=None)
    p.add_argument("--graph-meta-json", type=Path, default=None)
    p.add_argument("--embeddings-json", type=Path, default=None)
    p.add_argument("--semantic-top-k-missing", type=int, default=50)
    p.add_argument("--body-only-top-k-missing", type=int, default=50)
    p.add_argument("--path-top-k-missing", type=int, default=50)
    p.add_argument("--max-bridge-candidates", type=int, default=500_000)
    p.add_argument("--bridge-score-thresholds", type=str, default="0.8,0.9")
    p.add_argument("--score-batch-size", type=int, default=256)
    p.add_argument("--checkpoint", type=str, default="best_model.pt")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--skip-gt-bridge-diagnostics", action="store_true")
    p.add_argument("--no-sender-localpart", action="store_true")
    p.add_argument("--no-html-fp", action="store_true")
    p.add_argument("--no-registrable-domain", action="store_true")
    p.add_argument("--no-to-undirected", action="store_true")
    args = p.parse_args(argv)

    cfg = BridgeCandidateConfig(
        run_dir=args.run_dir.resolve(),
        graph_pt=args.graph_pt.resolve(),
        gt_path=args.gt_path.resolve() if args.gt_path else None,
        pair_csv=args.pair_csv.resolve() if args.pair_csv else None,
        candidate_union_csv=args.candidate_union_csv.resolve() if args.candidate_union_csv else None,
        seed_edges_csv=args.seed_edges_csv.resolve() if args.seed_edges_csv else None,
        graph_meta_json=args.graph_meta_json.resolve() if args.graph_meta_json else None,
        embeddings_json=args.embeddings_json.resolve() if args.embeddings_json else None,
        semantic_top_k_missing=int(args.semantic_top_k_missing),
        body_only_top_k_missing=int(args.body_only_top_k_missing),
        path_top_k_missing=int(args.path_top_k_missing),
        max_bridge_candidates=int(args.max_bridge_candidates),
        score_thresholds=_parse_thresholds(args.bridge_score_thresholds),
        score_batch_size=int(args.score_batch_size),
        skip_gt_diagnostics=bool(args.skip_gt_bridge_diagnostics),
        enable_sender_localpart=not bool(args.no_sender_localpart),
        enable_html_fp=not bool(args.no_html_fp),
        enable_registrable_domain=not bool(args.no_registrable_domain),
        device=str(args.device),
        checkpoint_name=str(args.checkpoint),
        to_undirected=not bool(args.no_to_undirected),
    )
    out = run_bridge_candidate_experiment(cfg)
    print(json.dumps({k: out[k] for k in ("output_dir", "summary_path", "main_csv", "n_bridge_candidates")}, indent=2))
    print(json.dumps(out.get("export_paths") or {}, indent=2))


if __name__ == "__main__":
    main()
