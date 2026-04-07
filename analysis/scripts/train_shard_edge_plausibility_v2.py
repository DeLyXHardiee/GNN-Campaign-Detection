"""CLI: train Method 1 V2 edge plausibility MLP from Step 2 artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.utils.semantic_shard_edge_plausibility_v2_config import EdgePlausibilityV2Config
from analysis.utils.semantic_shard_edge_plausibility_v2_train import train_and_score_edge_plausibility
from analysis.utils.semantic_shard_step3_helpers import load_step2_artifacts


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--step2-dir",
        type=Path,
        default=PROJECT_ROOT / "analysis" / "output" / "semantic_shard_step2_graph",
    )
    p.add_argument("--run-id", type=str, default="v2_cli_run")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default=None)
    p.add_argument(
        "--save-every-epoch-checkpoint",
        action="store_true",
        help="Write checkpoints/epoch_XXXX.pt each epoch for diagnostics (larger disk use).",
    )
    p.add_argument(
        "--no-gt-separation",
        action="store_true",
        help="Disable GT-only separation logging (no diag_* in training_history / no v2_gt_score_separation files).",
    )
    p.add_argument(
        "--gt-json",
        type=Path,
        default=None,
        help="Ground-truth JSON for separation diagnostics (default: pipeline_config datasets.ground_truth_json, else data/groundtruth/ground_truth.json; never dedup).",
    )
    p.add_argument(
        "--assignments-csv",
        type=Path,
        default=None,
        help="Step1 semantic_shard_step1_assignments.csv for separation diagnostics (default: search semantic_shard_step1/ and semantic_shard_step1_graph/ under analysis/output, plus sibling of --step2-dir).",
    )
    args = p.parse_args()

    nodes, edges, _ = load_step2_artifacts(args.step2_dir)
    cfg = EdgePlausibilityV2Config(
        random_seed=args.seed,
        run_id=args.run_id,
        epochs=args.epochs,
        batch_size=args.batch_size,
        output_root=str(PROJECT_ROOT / "analysis" / "output" / "semantic_shard_edge_v2"),
        save_every_epoch_checkpoint=args.save_every_epoch_checkpoint,
        log_gt_separation=not args.no_gt_separation,
        gt_separation_gt_json=str(args.gt_json) if args.gt_json is not None else None,
        gt_separation_assignments_csv=str(args.assignments_csv)
        if args.assignments_csv is not None
        else None,
        gt_separation_step2_dir=str(args.step2_dir.expanduser().resolve()),
    )
    out = train_and_score_edge_plausibility(edges, nodes, cfg, device=args.device)
    print("Wrote:", out["paths"])


if __name__ == "__main__":
    main()
