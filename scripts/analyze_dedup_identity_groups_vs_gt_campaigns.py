#!/usr/bin/env python3
"""
Check whether deduplicated identity collapse groups span multiple GT campaigns.

Uses full-lake ground truth (``ground_truth.json``) to label every member
``external_id`` in each collapse cluster — not the rep-remapped dedup GT file.

Example (repo root):

  python scripts/analyze_dedup_identity_groups_vs_gt_campaigns.py
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
CORE = REPO / "core"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

from graph.utils.url_analysis import load_gt_label_map  # noqa: E402

DEFAULT_COLLAPSE_DIR = REPO / "data" / "misp" / "misp_lake_dedup_task_identity"
DEFAULT_GT = REPO / "data" / "groundtruth" / "ground_truth.json"
DEFAULT_OUT_DIR = REPO / "output" / "analysis" / "dedup_task_identity_gt_campaigns"


def _campaigns_for_members(
    member_ids: list[str],
    label_map: dict[str, Any],
) -> tuple[list[Any], list[str], Counter[Any]]:
    """Return (labeled_campaigns, unlabeled_ids, campaign_counts)."""
    labeled: list[Any] = []
    unlabeled: list[str] = []
    counts: Counter[Any] = Counter()
    for eid in member_ids:
        cid = label_map.get(eid)
        if cid is None:
            unlabeled.append(eid)
            continue
        labeled.append(cid)
        counts[cid] += 1
    return labeled, unlabeled, counts


def analyze_clusters(
    collapsed_clusters: list[dict[str, Any]],
    label_map: dict[str, Any],
    *,
    min_group_size: int = 2,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    n_clusters = 0
    n_with_any_gt = 0
    n_with_multi_campaign = 0
    n_member_rows = 0
    n_member_labeled = 0

    for cluster in collapsed_clusters:
        gs = int(cluster.get("group_size") or 0)
        if gs < min_group_size:
            continue
        members = list(cluster.get("member_external_ids") or [])
        if not members:
            continue

        n_clusters += 1
        n_member_rows += len(members)
        labeled, unlabeled, counts = _campaigns_for_members(members, label_map)
        n_member_labeled += len(labeled)
        distinct = sorted(counts.keys(), key=lambda x: (str(type(x)), str(x)))
        n_distinct = len(distinct)
        crosses = n_distinct > 1

        if labeled:
            n_with_any_gt += 1
        if crosses:
            n_with_multi_campaign += 1

        dominant: Any = ""
        purity = ""
        if counts:
            dominant, dom_n = counts.most_common(1)[0]
            purity = float(dom_n) / max(len(labeled), 1)

        rows.append(
            {
                "cluster_id": str(cluster.get("cluster_id") or ""),
                "signature_type": str(cluster.get("signature_type") or ""),
                "representative_external_id": str(
                    cluster.get("representative_external_id") or ""
                ),
                "group_size": gs,
                "n_members_with_gt": len(labeled),
                "n_members_unlabeled": len(unlabeled),
                "n_distinct_gt_campaigns": n_distinct,
                "crosses_gt_campaigns": int(crosses),
                "gt_campaign_ids": ";".join(str(c) for c in distinct),
                "dominant_gt_campaign": str(dominant) if dominant != "" else "",
                "gt_purity_among_labeled": f"{purity:.6f}" if purity != "" else "",
                "gt_campaign_member_counts": ";".join(
                    f"{c}:{counts[c]}" for c in distinct
                ),
            }
        )

    rows.sort(
        key=lambda r: (
            -int(r["crosses_gt_campaigns"]),
            -int(r["n_distinct_gt_campaigns"]),
            -int(r["group_size"]),
            str(r["cluster_id"]),
        )
    )

    summary = {
        "ground_truth_file": "",
        "collapse_clusters_file": "",
        "min_group_size": min_group_size,
        "n_collapse_clusters_size_ge_min": n_clusters,
        "n_collapse_clusters_with_any_gt_member": n_with_any_gt,
        "n_collapse_clusters_crossing_gt_campaigns": n_with_multi_campaign,
        "n_member_rows_in_clusters": n_member_rows,
        "n_member_rows_with_gt_label": n_member_labeled,
        "gt_coverage_fraction_in_clusters": (
            float(n_member_labeled) / max(n_member_rows, 1)
        ),
        "fraction_gt_clusters_that_cross_campaigns": (
            float(n_with_multi_campaign) / max(n_with_any_gt, 1)
        ),
    }
    return rows, summary


def write_cluster_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "cluster_id",
        "signature_type",
        "representative_external_id",
        "group_size",
        "n_members_with_gt",
        "n_members_unlabeled",
        "n_distinct_gt_campaigns",
        "crosses_gt_campaigns",
        "gt_campaign_ids",
        "dominant_gt_campaign",
        "gt_purity_among_labeled",
        "gt_campaign_member_counts",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--collapse-clusters-json",
        type=Path,
        default=DEFAULT_COLLAPSE_DIR / "collapsed_clusters.json",
        help="collapsed_clusters.json from dedup collapse sidecar",
    )
    ap.add_argument(
        "--gt-json",
        type=Path,
        default=DEFAULT_GT,
        help="Full-lake ground truth (not dedup-remapped)",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Directory for CSV + summary JSON",
    )
    ap.add_argument(
        "--min-group-size",
        type=int,
        default=2,
        help="Only analyze collapse clusters with at least this many members",
    )
    args = ap.parse_args()

    clusters_path = args.collapse_clusters_json.expanduser().resolve()
    gt_path = args.gt_json.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()

    if not clusters_path.is_file():
        print(f"ERROR: collapse clusters not found: {clusters_path}", file=sys.stderr)
        return 1
    if not gt_path.is_file():
        print(f"ERROR: ground truth not found: {gt_path}", file=sys.stderr)
        return 1

    collapsed = json.loads(clusters_path.read_text(encoding="utf-8"))
    if not isinstance(collapsed, list):
        print("ERROR: collapsed_clusters.json must be a JSON array", file=sys.stderr)
        return 1

    label_map = load_gt_label_map(gt_path)
    rows, summary = analyze_clusters(
        collapsed,
        label_map,
        min_group_size=max(1, int(args.min_group_size)),
    )
    summary["ground_truth_file"] = str(gt_path)
    summary["collapse_clusters_file"] = str(clusters_path)
    summary["n_gt_labeled_emails_in_full_gt"] = len(label_map)

    all_csv = out_dir / "dedup_identity_clusters_gt_campaigns.csv"
    write_cluster_csv(all_csv, rows)

    cross_rows = [r for r in rows if int(r["crosses_gt_campaigns"])]
    cross_csv = out_dir / "dedup_identity_clusters_cross_gt_campaigns.csv"
    if cross_rows:
        write_cluster_csv(cross_csv, cross_rows)

    summary_path = out_dir / "dedup_identity_clusters_gt_campaigns_summary.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2))
    print(f"\nWrote {all_csv} ({len(rows)} cluster rows)")
    if cross_rows:
        print(f"Wrote {cross_csv} ({len(cross_rows)} cross-campaign clusters)")
    else:
        print("No clusters cross GT campaigns (among labeled members).")
    print(f"Wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
