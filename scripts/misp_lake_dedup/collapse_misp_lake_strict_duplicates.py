"""
Collapse duplicate emails in a MISP lake JSON using a configurable deterministic signature.

Produces a reduced JSON array (one representative raw event per signature cluster),
sidecar mapping/analytics, optional remapped ground_truth.json, and GT purity diagnostics.

Signatures (see misp_email_identity):
  - strict_full_email
  - strict_task_message_identity  (task-grounded; ignores delivery-instance noise in body)
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_PKG_DIR = Path(__file__).resolve().parent
if str(_PKG_DIR) not in sys.path:
    sys.path.insert(0, str(_PKG_DIR))
import misp_email_identity as mei  # noqa: E402

try:
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None  # type: ignore[assignment]

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "misp" / "incidents-lake-misp.json"


def run_collapse(
    *,
    input_json: Path,
    out_json: Path,
    out_dir: Path,
    collapse_signature_type: str,
    max_events: int | None,
    top_k: int,
    ground_truth_in: Path | None,
    ground_truth_out: Path | None,
    compare_signature_types: bool = True,
) -> dict[str, Any]:
    input_json = input_json.expanduser().resolve()
    out_json = out_json.expanduser().resolve()
    out_dir = out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if not input_json.is_file():
        raise FileNotFoundError(f"Input JSON not found: {input_json}")

    sig_type = mei.resolve_collapse_signature_type(collapse_signature_type)
    sig_fn = mei.resolve_signature_fn(sig_type)

    raw_events, records = mei._load_raw_events_and_records(input_json, max_events=max_events)
    if len(raw_events) != len(records):
        raise RuntimeError("internal: raw_events and records length mismatch")

    n_in = len(records)
    pre_summary, _ = mei.analyze_signature_duplicate_burden(
        records, sig_fn, signature_name=sig_type, top_k=max(1, top_k)
    )

    signature_comparison: dict[str, Any] | None = None
    if compare_signature_types:
        comparison_rows: list[dict[str, Any]] = []
        for other in mei.COLLAPSE_SIGNATURE_CHOICES:
            other_fn = mei.resolve_signature_fn(other)
            summ, _ = mei.analyze_signature_duplicate_burden(
                records, other_fn, signature_name=other, top_k=max(1, top_k)
            )
            comparison_rows.append(
                {
                    "signature_type": other,
                    "n_emails_total": summ["n_emails_total"],
                    "n_events_out_if_collapsed": int(
                        summ["n_groups_total"]
                    ),
                    "n_events_removed_if_collapsed": int(
                        summ["n_emails_total"] - summ["n_groups_total"]
                    ),
                    "reduction_ratio": float(
                        (summ["n_emails_total"] - summ["n_groups_total"]) / max(summ["n_emails_total"], 1)
                    ),
                    "n_duplicate_groups_size_ge_2": summ["n_duplicate_groups_size_ge_2"],
                    "n_emails_in_duplicate_groups": summ["n_emails_in_duplicate_groups"],
                    "estimated_easy_edges_removed_if_collapsed": summ[
                        "estimated_easy_edges_from_duplicate_groups"
                    ],
                    "max_group_size": summ["max_group_size"],
                }
            )
        signature_comparison = {
            "signatures_compared": list(mei.COLLAPSE_SIGNATURE_CHOICES),
            "active_collapse_signature": sig_type,
            "by_signature": comparison_rows,
        }

    sig_to_raw: dict[str, list[dict[str, Any]]] = defaultdict(list)
    sig_to_rec: dict[str, list[mei.EmailRecord]] = defaultdict(list)
    for raw, rec in zip(raw_events, records, strict=True):
        sig = sig_fn(rec)
        sig_to_raw[sig].append(raw)
        sig_to_rec[sig].append(rec)

    output_raw: list[dict[str, Any]] = []
    collapsed_clusters: list[dict[str, Any]] = []
    id_to_rep: dict[str, str] = {}
    map_rows: list[dict[str, Any]] = []
    n_dup_clusters = 0

    for sig, members_rec in sorted(sig_to_rec.items(), key=lambda kv: mei._cluster_id_full(kv[0])):
        members_raw = sig_to_raw[sig]
        gs = len(members_rec)
        cid = mei._cluster_id_full(sig)
        pairs = list(zip(members_raw, members_rec, strict=True))
        pairs.sort(key=lambda p: p[1].external_id)
        raw_rep, rec_rep = pairs[0]
        rep_id = rec_rep.external_id
        ext_ids = sorted(r.external_id for r in members_rec)

        if gs >= 2:
            n_dup_clusters += 1
            collapsed_clusters.append(
                {
                    "cluster_id": cid,
                    "signature_hash12": mei._sha12(sig),
                    "signature_type": sig_type,
                    "representative_external_id": rep_id,
                    "member_external_ids": ext_ids,
                    "group_size": gs,
                }
            )

        output_raw.append(raw_rep)
        for eid in ext_ids:
            id_to_rep[eid] = rep_id
            map_rows.append(
                {
                    "external_id": eid,
                    "representative_external_id": rep_id,
                    "cluster_id": cid,
                    "signature_type": sig_type,
                    "is_representative": bool(eid == rep_id),
                    "group_size": gs,
                }
            )

    out_triples: list[tuple[str, dict[str, Any], mei.EmailRecord]] = []
    for i, raw in enumerate(output_raw):
        r = mei._extract_email_record(raw, i)
        out_triples.append((r.external_id, raw, r))
    out_triples.sort(key=lambda t: t[0])
    output_sorted = [t[1] for t in out_triples]

    post_records = [mei._extract_email_record(raw, i) for i, raw in enumerate(output_sorted)]
    post_summary, _ = mei.analyze_signature_duplicate_burden(
        post_records, sig_fn, signature_name=sig_type, top_k=max(1, top_k)
    )
    n_out = len(output_sorted)
    n_removed = n_in - n_out
    reduction_ratio = float(n_removed / max(n_in, 1))

    for r in records:
        if r.external_id not in id_to_rep:
            raise RuntimeError(f"external_id missing from collapse map: {r.external_id!r}")
    if len(map_rows) != n_in:
        raise RuntimeError(f"map_rows length {len(map_rows)} != n_in {n_in}")

    collapsed_clusters.sort(
        key=lambda c: (-int(c.get("group_size", 0)), str(c.get("representative_external_id", "")))
    )

    pre_pairs = pre_summary["all_possible_pairs_n_choose_2"]
    post_pairs = post_summary["all_possible_pairs_n_choose_2"]
    easy_pre = pre_summary["estimated_easy_edges_from_duplicate_groups"]

    gt_quality: dict[str, Any] | None = None
    if ground_truth_in is not None and ground_truth_in.is_file():
        gt_quality = mei.evaluate_collapse_clusters_against_ground_truth(
            collapsed_clusters,
            ground_truth_path=ground_truth_in,
            signature_type=sig_type,
        )

    collapse_summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "collapse_signature_type": sig_type,
        "collapse_signature_description": mei.SIGNATURE_DESCRIPTIONS.get(sig_type, ""),
        "input_json": str(input_json),
        "out_json": str(out_json),
        "out_dir": str(out_dir),
        "max_events_arg": max_events,
        "representative_selection_rule": "lexicographically_smallest_external_id",
        "n_events_in": n_in,
        "n_events_out": n_out,
        "n_events_removed": n_removed,
        "reduction_ratio": reduction_ratio,
        "n_duplicate_clusters_merged": n_dup_clusters,
        "pre_collapse_duplicate_analysis": pre_summary,
        "post_collapse_duplicate_analysis": post_summary,
        "signature_comparison_before_collapse": signature_comparison,
        "ground_truth_cluster_quality": gt_quality,
        "delta": {
            "all_possible_pairs_n_choose_2_before": int(pre_pairs),
            "all_possible_pairs_n_choose_2_after": int(post_pairs),
            "all_possible_pairs_removed": int(pre_pairs - post_pairs),
            "estimated_intra_duplicate_easy_edges_removed": int(easy_pre),
            "estimated_easy_edges_from_duplicate_groups_after": int(
                post_summary["estimated_easy_edges_from_duplicate_groups"]
            ),
        },
        "notes": [
            mei.SIGNATURE_DESCRIPTIONS.get(sig_type, ""),
            "estimated_intra_duplicate_easy_edges_removed equals pre-collapse easy-edge mass for this signature.",
            "Rebuild graph / seed / candidate / pair_training from out_json for downstream experiments.",
        ],
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps(output_sorted, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    manifest = {
        "created_at_utc": collapse_summary["created_at_utc"],
        "collapse_signature_type": sig_type,
        "input_json": str(input_json),
        "out_json": str(out_json),
        "n_events_in": n_in,
        "n_events_out": n_out,
        "n_events_removed": n_removed,
        "reduction_ratio": reduction_ratio,
        "n_duplicate_clusters_merged": n_dup_clusters,
        "representative_selection_rule": collapse_summary["representative_selection_rule"],
        "artifact_paths": {
            "collapse_summary_json": str(out_dir / "collapse_summary.json"),
            "external_id_map_parquet": str(out_dir / "external_id_map.parquet"),
            "external_id_map_csv": str(out_dir / "external_id_map.csv"),
            "collapsed_clusters_json": str(out_dir / "collapsed_clusters.json"),
        },
    }
    (out_dir / "collapse_summary.json").write_text(
        json.dumps(collapse_summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out_dir / "collapsed_clusters.json").write_text(
        json.dumps(collapsed_clusters, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    p_parquet = out_dir / "external_id_map.parquet"
    p_csv = out_dir / "external_id_map.csv"
    parquet_note: str | None = None
    if map_rows:
        if pd is not None:
            df_map = pd.DataFrame(map_rows)
            try:
                df_map.to_parquet(p_parquet, index=False)
            except Exception as exc:  # pragma: no cover
                parquet_note = f"{type(exc).__name__}: {exc}"
            df_map.to_csv(p_csv, index=False)
        else:
            with p_csv.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(
                    f,
                    fieldnames=[
                        "external_id",
                        "representative_external_id",
                        "cluster_id",
                        "signature_type",
                        "is_representative",
                        "group_size",
                    ],
                )
                w.writeheader()
                w.writerows(map_rows)

    if parquet_note:
        manifest["parquet_write_note"] = parquet_note
    (out_dir / "collapse_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    if ground_truth_in is not None and ground_truth_out is not None:
        remap_ground_truth_json(ground_truth_in, ground_truth_out, id_to_rep)

    return {
        "collapse_summary": collapse_summary,
        "manifest": manifest,
        "out_json": str(out_json),
        "out_dir": str(out_dir),
    }


def remap_ground_truth_json(
    ground_truth_in: Path,
    ground_truth_out: Path,
    id_to_rep: dict[str, str],
) -> None:
    """Remap every external_id in clusters.* to its representative; dedupe within each cluster list."""
    ground_truth_in = ground_truth_in.expanduser().resolve()
    ground_truth_out = ground_truth_out.expanduser().resolve()
    data = json.loads(ground_truth_in.read_text(encoding="utf-8"))
    clusters = data.get("clusters")
    if not isinstance(clusters, dict):
        raise TypeError("ground_truth.json: expected top-level 'clusters' object")

    for _cid, rows in clusters.items():
        if not isinstance(rows, list):
            continue
        new_rows: list[dict[str, str]] = []
        seen: set[str] = set()
        for row in rows:
            if not isinstance(row, dict):
                continue
            eid = str(row.get("external_id", "")).strip()
            if not eid:
                continue
            rep = id_to_rep.get(eid, eid)
            if rep in seen:
                continue
            seen.add(rep)
            new_rows.append({"external_id": rep})
        clusters[_cid] = new_rows

    ground_truth_out.parent.mkdir(parents=True, exist_ok=True)
    ground_truth_out.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-json", type=Path, default=DEFAULT_INPUT)
    ap.add_argument(
        "--collapse-signature-type",
        type=str,
        default=mei.SIGNATURE_STRICT_TASK_MESSAGE,
        choices=list(mei.COLLAPSE_SIGNATURE_CHOICES),
        help="Deterministic signature used to merge duplicate emails.",
    )
    ap.add_argument(
        "--out-json",
        type=Path,
        default=None,
        help="Collapsed MISP JSON (default: data/misp/incidents-lake-misp.<stem>.json)",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Sidecar directory (default: data/misp/misp_lake_<stem>/)",
    )
    ap.add_argument("--max-events", type=int, default=0, help="0 = all events")
    ap.add_argument("--top-k", type=int, default=25, help="Top duplicate groups in summaries")
    ap.add_argument(
        "--ground-truth-in",
        type=Path,
        default=PROJECT_ROOT / "data" / "groundtruth" / "ground_truth.json",
        help="Source ground_truth.json for remap + GT cluster-quality diagnostics",
    )
    ap.add_argument(
        "--ground-truth-out",
        type=Path,
        default=None,
        help="Remapped GT path (default: ground_truth.<dedup_stem>.json)",
    )
    ap.add_argument(
        "--no-signature-comparison",
        action="store_true",
        help="Skip pre-collapse burden comparison across collapse signatures",
    )
    args = ap.parse_args()

    sig_type = mei.resolve_collapse_signature_type(args.collapse_signature_type)
    defaults = mei.default_collapse_paths(sig_type, project_root=PROJECT_ROOT)
    out_json = args.out_json.expanduser().resolve() if args.out_json else defaults["out_json"]
    out_dir = args.out_dir.expanduser().resolve() if args.out_dir else defaults["out_dir"]

    gt_in = args.ground_truth_in.expanduser().resolve() if args.ground_truth_in else None
    gt_out = (
        args.ground_truth_out.expanduser().resolve()
        if args.ground_truth_out
        else defaults["ground_truth_out_default"]
    )
    if gt_in is None:
        gt_out = None

    max_events = int(args.max_events) if int(args.max_events) > 0 else None
    result = run_collapse(
        input_json=args.input_json,
        out_json=out_json,
        out_dir=out_dir,
        collapse_signature_type=sig_type,
        max_events=max_events,
        top_k=int(args.top_k),
        ground_truth_in=gt_in,
        ground_truth_out=gt_out,
        compare_signature_types=not bool(args.no_signature_comparison),
    )
    print(
        json.dumps(
            {
                "collapse_signature_type": sig_type,
                "out_json": result["out_json"],
                "out_dir": result["out_dir"],
                "ground_truth_out": str(gt_out) if gt_out else None,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
