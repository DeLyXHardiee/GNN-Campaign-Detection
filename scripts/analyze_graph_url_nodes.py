#!/usr/bin/env python3
"""
Rank URL nodes by email-link degree and count same/cross-campaign pairs per URL.

Writes two CSVs (all URL rows + global totals) using ground truth from pipeline_config.

Example (repo root):

  python scripts/analyze_graph_url_nodes.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CORE = REPO / "core"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

from graph.utils.url_analysis import (  # noqa: E402
    collect_url_campaign_pair_rows,
    collect_url_node_rows,
    write_url_campaign_pair_rows_csv,
    write_url_campaign_pair_totals_csv,
    write_url_node_rows_csv,
)


def _default_paths() -> tuple[Path, Path, Path, Path]:
    stem = "main_gnn_pu_1_no_ts_dedup_task_identity"
    graph_pt = REPO / "core" / "graph" / "output" / f"{stem}_hetero.pt"
    meta_json = REPO / "core" / "graph" / "output" / f"{stem}_hetero.meta.json"
    out_csv = REPO / "output" / "analysis" / f"url_nodes_{stem}.csv"
    gt_json = REPO / "data" / "groundtruth" / "ground_truth.dedup_task_identity.json"
    cfg = REPO / "pipeline_config.json"
    if cfg.is_file():
        try:
            pc = json.loads(cfg.read_text(encoding="utf-8"))
            g = pc.get("graph") or {}
            ds = pc.get("datasets") or {}
            stem_cfg = str(g.get("hetero_graph_stem") or stem).strip() or stem
            override = str(g.get("graph_pt_path_override") or "").strip()
            out_dir = Path(str(g.get("output_dir") or "core/graph/output"))
            if not out_dir.is_absolute():
                out_dir = REPO / out_dir
            graph_pt = (
                (REPO / override).resolve()
                if override
                else (out_dir / f"{stem_cfg}_hetero.pt").resolve()
            )
            meta_json = graph_pt.with_suffix(".meta.json")
            out_csv = REPO / "output" / "analysis" / f"url_nodes_{stem_cfg}.csv"
            gt_raw = str(ds.get("ground_truth_json") or "").strip()
            if gt_raw:
                gt_p = Path(gt_raw)
                gt_json = gt_p if gt_p.is_absolute() else (REPO / gt_p).resolve()
        except Exception:
            pass
    return graph_pt, meta_json, out_csv, gt_json


def main() -> int:
    default_graph, default_meta, default_csv, default_gt = _default_paths()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--graph-pt", type=Path, default=default_graph)
    ap.add_argument("--meta-json", type=Path, default=default_meta)
    ap.add_argument("--out-csv", type=Path, default=default_csv)
    ap.add_argument(
        "--gt-json",
        type=Path,
        default=default_gt,
        help="Ground truth for same/cross campaign pair labels",
    )
    ap.add_argument(
        "--out-campaign-csv",
        type=Path,
        default=None,
        help="Per-URL campaign pair CSV (default: <out-csv stem>_campaign_pairs.csv)",
    )
    ap.add_argument(
        "--out-campaign-totals-csv",
        type=Path,
        default=None,
        help="One-row global totals CSV (default: <out-csv stem>_campaign_pair_totals.csv)",
    )
    ap.add_argument(
        "--skip-campaign-pairs",
        action="store_true",
        help="Only write degree CSV, not campaign pair analysis",
    )
    args = ap.parse_args()

    graph_pt = args.graph_pt.expanduser().resolve()
    meta_json = args.meta_json.expanduser().resolve()
    out_csv = args.out_csv.expanduser().resolve()
    gt_json = args.gt_json.expanduser().resolve()

    if not graph_pt.is_file():
        print(f"ERROR: graph not found: {graph_pt}", file=sys.stderr)
        return 1
    if not meta_json.is_file():
        print(f"ERROR: metadata not found: {meta_json}", file=sys.stderr)
        return 1

    rows, summary = collect_url_node_rows(str(graph_pt), str(meta_json))
    csv_path = write_url_node_rows_csv(rows, str(out_csv))
    print(f"Wrote {len(rows):,} URL degree rows to: {csv_path}")
    print(
        f"(email->url edges: {summary['n_email_has_url_edges']:,}, "
        f"max degree: {summary['max_email_edge_degree']:,})"
    )

    if not args.skip_campaign_pairs:
        if not gt_json.is_file():
            print(f"ERROR: ground truth not found: {gt_json}", file=sys.stderr)
            return 1
        camp_csv = args.out_campaign_csv
        if camp_csv is None:
            camp_csv = out_csv.with_name(f"{out_csv.stem}_campaign_pairs.csv")
        else:
            camp_csv = camp_csv.expanduser().resolve()
        totals_csv = args.out_campaign_totals_csv
        if totals_csv is None:
            totals_csv = out_csv.with_name(f"{out_csv.stem}_campaign_pair_totals.csv")
        else:
            totals_csv = totals_csv.expanduser().resolve()

        camp_rows, camp_summary = collect_url_campaign_pair_rows(
            str(graph_pt), str(meta_json), str(gt_json)
        )
        camp_path = write_url_campaign_pair_rows_csv(camp_rows, str(camp_csv))
        totals_path = write_url_campaign_pair_totals_csv(camp_summary, str(totals_csv))
        print(f"Wrote {len(camp_rows):,} URL campaign-pair rows to: {camp_path}")
        print(f"Wrote global totals to: {totals_path}")
        print(
            f"URL-induced email pairs: {camp_summary.get('n_email_pairs_total', 0):,} | "
            f"same-campaign: {camp_summary.get('n_same_campaign_pairs', 0):,} | "
            f"cross-campaign: {camp_summary.get('n_cross_campaign_pairs', 0):,} | "
            f"unlabeled: {camp_summary.get('n_unlabeled_pairs', 0):,}"
        )
        if camp_summary.get("frac_same_among_gt_pairs") is not None:
            print(
                f"(among GT-labeled pairs only: "
                f"same={100 * float(camp_summary['frac_same_among_gt_pairs']):.1f}%, "
                f"cross={100 * float(camp_summary['frac_cross_among_gt_pairs']):.1f}%)"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
