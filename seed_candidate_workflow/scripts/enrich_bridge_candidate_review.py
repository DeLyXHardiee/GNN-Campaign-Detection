"""
Re-run bridge explainability (enrichment + HTML + band/trust summaries) from an existing
bridge_candidate_scores.csv without re-scoring or re-retrieval.

Example (from repo root):

  python seed_candidate_workflow/scripts/enrich_bridge_candidate_review.py ^
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

import pandas as pd

from seed_candidate_workflow.utils.bridge_candidate_experiment import load_connected_pair_keys  # noqa: E402
from seed_candidate_workflow.utils.bridge_candidate_review import (  # noqa: E402
    _attach_gt_campaign_columns,
    enrich_bridge_dataframe_for_review,
    export_bridge_review_artifacts,
)


def main() -> None:
    p = argparse.ArgumentParser(description="Enrich existing bridge candidate CSV for review.")
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--graph-pt", type=Path, required=True)
    p.add_argument("--scores-csv", type=Path, default=None)
    p.add_argument("--gt-path", type=Path, default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--gnn-latent-max-rows", type=int, default=500)
    args = p.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    run_dir = Path(args.run_dir).resolve()
    out_root = run_dir / "bridge_candidate_experiment"
    scores_csv = (
        Path(args.scores_csv).resolve()
        if args.scores_csv
        else out_root / "bridge_candidate_scores.csv"
    )
    df = pd.read_csv(scores_csv, low_memory=False)

    graph_pt = Path(args.graph_pt).resolve()
    meta_json = graph_pt.with_suffix(".meta.json")
    cand_hint = (
        project_root
        / "seed_candidate_workflow"
        / "output"
        / "graph_bundles"
        / run_dir.name
        / "candidate"
        / run_dir.name
        / "candidate_union.csv"
    )
    cand_csv = cand_hint if cand_hint.is_file() else None
    connected = load_connected_pair_keys(candidate_union_csv=cand_csv, seed_edges_csv=None)

    try:
        from seed_candidate_workflow.utils.pair_score_separation import (
            _load_email_text_catalog,
            _resolve_default_misp_json_path,
        )

        misp_path = _resolve_default_misp_json_path(project_root)
    except Exception:
        misp_path = None

    df, review_meta = enrich_bridge_dataframe_for_review(
        df,
        project_root=project_root,
        run_dir=run_dir,
        graph_pt=graph_pt,
        connected=connected,
        candidate_union_csv=cand_csv,
        pair_csv=None,
        compute_gnn_latent_max_rows=int(args.gnn_latent_max_rows),
        device=str(args.device),
        misp_json_path=misp_path,
    )

    label_map = None
    if args.gt_path and Path(args.gt_path).is_file():
        from seed_candidate_workflow.utils.raw_gnn_notebook import load_ground_truth_structures

        label_map, _, _ = load_ground_truth_structures(Path(args.gt_path))
        label_map = {str(k): v for k, v in label_map.items()}
        df = _attach_gt_campaign_columns(df, label_map=label_map)
        from seed_candidate_workflow.utils.bridge_candidate_experiment import _gt_labels_for_pairs

        df["gt_relation"] = _gt_labels_for_pairs(df, label_map)

    try:
        from seed_candidate_workflow.utils.pair_score_separation import _load_email_text_catalog

        catalog, _ = _load_email_text_catalog(project_root=project_root, misp_json_path=misp_path)
    except Exception:
        catalog = {}

    review_export = export_bridge_review_artifacts(
        df,
        out_root=out_root,
        email_catalog=catalog,
        label_map=label_map,
        review_meta=review_meta,
    )

    summary_path = out_root / "bridge_candidate_summary.json"
    summary: dict = {}
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["bridge_review_meta"] = review_meta
    summary["bridge_feature_population_diagnostics"] = review_export.get("bridge_feature_population_diagnostics")
    summary["bridge_band_analysis"] = review_export.get("bridge_band_analysis")
    summary["bridge_suspicious_high_score_analysis"] = review_export.get("bridge_suspicious_high_score_analysis")
    summary["bridge_trustworthiness_recommendation"] = review_export.get("bridge_trustworthiness_recommendation")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    if not catalog:
        print("WARNING: email catalog empty — HTML review files were not written.")
    else:
        for k, p in (review_export.get("export_paths") or {}).items():
            if k.endswith("_html"):
                print(f"Wrote {p}")
    print(f"Wrote enriched artifacts under {out_root}")


if __name__ == "__main__":
    main()
