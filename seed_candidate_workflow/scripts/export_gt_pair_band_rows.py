"""
Export GT-stratified pair rows by score band for manual review.

Default band ``cross_high`` matches pair score separation high band:
score > high_min and score <= 1.0, both endpoints in GT, different campaigns.

Example (dedup_strict run, inspect false-positive-ish cross edges):

  python seed_candidate_workflow/scripts/export_gt_pair_band_rows.py ^
    --run-dir output/runs/main_gnn_pu_1_no_ts_dedup_strict ^
    --graph-pt core/graph/output/main_gnn_pu_1_no_ts_dedup_strict_hetero.pt ^
    --gt-path data/groundtruth/ground_truth.dedup_strict.json ^
    --out-csv output/runs/main_gnn_pu_1_no_ts_dedup_strict/cross_campaign_high_scores.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_PROJECT_ROOT / "core" / "GNN") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "core" / "GNN"))

from seed_candidate_workflow.utils.pair_model_inference import (
    load_pair_supervision_for_inference,
    score_pair_rows,
)
from seed_candidate_workflow.utils.raw_gnn_notebook import load_ground_truth_structures
from src.pair_train import load_pair_training_dataframe


def _resolve_pair_csv(run_dir: Path, pair_csv: Path | None) -> Path:
    if pair_csv is not None:
        return pair_csv.expanduser().resolve()
    cfg_path = run_dir / "training_config.json"
    if not cfg_path.is_file():
        raise SystemExit(f"Missing {cfg_path}; pass --pair-csv")
    tc = json.loads(cfg_path.read_text(encoding="utf-8"))
    raw = tc.get("pair_dataset_csv")
    if not raw:
        raise SystemExit("pair_dataset_csv missing in training_config.json; pass --pair-csv")
    p = Path(str(raw))
    if not p.is_absolute():
        p = (_PROJECT_ROOT / p).resolve()
    return p.resolve()


def _band_mask(
    *,
    band: str,
    same_mask: np.ndarray,
    cross_mask: np.ndarray,
    eval_mask: np.ndarray,
    scores: np.ndarray,
    low_max: float,
    high_min: float,
) -> np.ndarray:
    s = scores
    finite = np.isfinite(s)
    low = finite & (s >= 0.0) & (s <= float(low_max))
    high = finite & (s > float(high_min)) & (s <= 1.0)
    b = str(band).strip().lower()
    if b == "cross_high":
        return eval_mask & cross_mask & high
    if b == "cross_low":
        return eval_mask & cross_mask & low
    if b == "same_low":
        return eval_mask & same_mask & low
    if b == "same_high":
        return eval_mask & same_mask & high
    raise SystemExit(f"Unknown --band {band!r}; use cross_high|cross_low|same_low|same_high")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--graph-pt", type=Path, required=True)
    p.add_argument("--pair-csv", type=Path, default=None)
    p.add_argument("--gt-path", type=Path, required=True)
    p.add_argument(
        "--band",
        type=str,
        default="cross_high",
        help="cross_high|cross_low|same_low|same_high (GT-covered pairs only)",
    )
    p.add_argument("--high-min", type=float, default=0.8)
    p.add_argument("--low-max", type=float, default=0.4)
    p.add_argument("--checkpoint", type=str, default="best_model.pt")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--no-to-undirected", action="store_true")
    p.add_argument("--out-csv", type=Path, required=True)
    p.add_argument("--max-rows", type=int, default=0, help="0 = no limit")
    args = p.parse_args()

    run_dir = args.run_dir.expanduser().resolve()
    graph_pt = args.graph_pt.expanduser().resolve()
    gt_path = args.gt_path.expanduser().resolve()
    pair_csv = _resolve_pair_csv(run_dir, args.pair_csv)

    df, _stats = load_pair_training_dataframe(pair_csv)
    df_work = df.reset_index(drop=True)
    df_work["_row"] = np.arange(len(df_work), dtype=np.int64)

    bundle = load_pair_supervision_for_inference(
        run_dir=run_dir,
        graph_pt=graph_pt,
        checkpoint_name=str(args.checkpoint),
        device=str(args.device),
        to_undirected=not bool(args.no_to_undirected),
    )
    scores = score_pair_rows(
        model=bundle["model"],
        pair_scorer=bundle["pair_scorer"],
        data_cpu=bundle["data_cpu"],
        df_work=df_work,
        device=bundle["device"],
        fanout=bundle["fanout"],
        pair_batch_size=bundle["pair_batch_size"],
        max_unique_emails=bundle["max_unique_emails"],
        pair_feature_columns=bundle.get("pair_feature_columns"),
    )

    label_map, _eid_row, _camp = load_ground_truth_structures(gt_path)
    label_map = {str(k): v for k, v in label_map.items()}

    ei = df_work["email_i"].astype(str).values
    ej = df_work["email_j"].astype(str).values
    n = len(df_work)
    camp_i = np.array([label_map.get(str(ei[k])) for k in range(n)], dtype=object)
    camp_j = np.array([label_map.get(str(ej[k])) for k in range(n)], dtype=object)
    both = np.array(
        [camp_i[k] is not None and camp_j[k] is not None for k in range(n)],
        dtype=bool,
    )
    same_mask = both & (camp_i == camp_j)
    cross_mask = both & (camp_i != camp_j)
    eval_mask = both & np.isfinite(scores)

    mask = _band_mask(
        band=str(args.band),
        same_mask=same_mask,
        cross_mask=cross_mask,
        eval_mask=eval_mask,
        scores=scores,
        low_max=float(args.low_max),
        high_min=float(args.high_min),
    )
    sub = df_work.loc[mask].copy()
    sub["pu_score"] = scores[mask]
    sub["gt_campaign_i"] = [camp_i[i] for i in sub.index]
    sub["gt_campaign_j"] = [camp_j[i] for i in sub.index]
    sub = sub.sort_values("pu_score", ascending=False).reset_index(drop=True)
    if int(args.max_rows) > 0:
        sub = sub.head(int(args.max_rows)).copy()

    out = args.out_csv.expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out, index=False)
    print(
        json.dumps(
            {
                "wrote": str(out),
                "n_rows": int(len(sub)),
                "band": str(args.band),
                "high_min": float(args.high_min),
                "low_max": float(args.low_max),
                "pair_csv": str(pair_csv),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
