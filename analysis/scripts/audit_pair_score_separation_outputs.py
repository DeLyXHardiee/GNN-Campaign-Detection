#!/usr/bin/env python3
"""
Read-only audit of pair_score_separation output layout and feature visibility.

Writes:
  output/analysis/pair_score_separation_output_audit_summary.json
  output/analysis/pair_score_separation_output_audit_report.md
  output/analysis/pair_score_separation_output_inventory.csv

Does not modify pair_score_separation pipeline behavior.
"""
from __future__ import annotations

import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[2]
OUT_ANALYSIS = _REPO / "output" / "analysis"


def _artifact_catalog_from_code() -> list[dict[str, Any]]:
    """Static catalog derived from pair_score_separation.py + pair_same_unlabeled_rescued_collapsed.py."""
    creator = "seed_candidate_workflow.utils.pair_score_separation:run_pair_score_separation_analysis"
    rescued = "seed_candidate_workflow.utils.pair_same_unlabeled_rescued_collapsed:run_same_unlabeled_rescued_vs_collapsed_analysis"
    low_disc = "seed_candidate_workflow.utils.pair_low_band_feature_discovery:run_low_band_feature_discovery"

    def row(
        path: str,
        category: str,
        purpose: str,
        *,
        creator_fn: str = creator,
        recommendation: str = "unclear",
        notes: str = "",
    ) -> dict[str, Any]:
        return {
            "path_pattern": path,
            "category": category,
            "creator": creator_fn,
            "purpose": purpose,
            "recommendation": recommendation,
            "notes": notes,
        }

    items: list[dict[str, Any]] = []
    items.append(
        row(
            "pair_score_separation_summary.json",
            "core_json",
            "Front-page run manifest: per-GT score stats, plot paths, band diagnostics, export pointers.",
            recommendation="keep_as_primary",
        )
    )
    items.append(
        row(
            "pair_score_separation_summary.json → per_gt[].band_diagnostics",
            "core_json",
            "Per-GT band counts (low/high same/cross unlabeled).",
            recommendation="keep_as_primary",
            notes="Embedded in main summary; avoid duplicating in separate files long-term.",
        )
    )
    items.append(
        row(
            "pair_low_band_separator_summary.json",
            "core_json",
            "Marginal separator stats for low band (same vs cross unlabeled).",
            recommendation="merge_into_main_summary",
            notes="Heavy overlap with pair_score_separation_summary per_gt band_diagnostics + separator tables.",
        )
    )
    items.append(
        row(
            "pair_low_band_joint_separator_summary.json",
            "core_json",
            "Joint boolean rules in low band (same vs cross unlabeled).",
            recommendation="keep_but_move_to_debug",
            notes="Essential for tuning but verbose; not front-page.",
        )
    )
    items.append(
        row(
            "pair_high_band_separator_summary.json",
            "core_json",
            "Marginal separator stats for high-score unlabeled band.",
            recommendation="merge_into_main_summary",
        )
    )
    items.append(
        row(
            "pair_high_band_joint_separator_summary.json",
            "core_json",
            "Joint rules for high-score false-positive unlabeled pairs.",
            recommendation="keep_but_move_to_debug",
        )
    )
    items.append(
        row(
            "pair_high_band_false_positive_summary.json",
            "core_json",
            "High-band FP cohort stats + manual_review_export metadata.",
            recommendation="keep_but_move_to_debug",
        )
    )
    items.append(
        row(
            "pair_low_band_twohop_channel_summary.json",
            "core_json",
            "2-hop channel attribution in low band.",
            recommendation="keep_but_move_to_debug",
            notes="Specialized frontier; referenced from low_joint summary.",
        )
    )
    items.append(
        row(
            f"pair_same_unlabeled_rescued_vs_collapsed_summary.json",
            "core_json",
            "Rescued vs collapsed marginal + recommendations.",
            creator_fn=rescued,
            recommendation="keep_as_primary",
        )
    )
    items.append(
        row(
            "pair_same_unlabeled_rescued_vs_collapsed_joint_summary.json",
            "core_json",
            "Rescued vs collapsed joint rule rankings.",
            creator_fn=rescued,
            recommendation="keep_but_move_to_debug",
        )
    )
    items.append(
        row(
            "pair_low_band_feature_discovery_summary.json",
            "core_json",
            "Extended low-band feature discovery + positive alignment (body/path/sender).",
            creator_fn=low_disc,
            recommendation="keep_but_move_to_debug",
            notes="Separate entrypoint; same output folder — easy to confuse with main run.",
        )
    )

    for stem in [
        "pair_low_band_separator_table.csv",
        "pair_low_band_joint_separator_table.csv",
        "pair_high_band_separator_table.csv",
        "pair_high_band_joint_separator_table.csv",
        "pair_score_band_diagnostics.csv",
        "pair_low_band_twohop_channel_summary.csv",
        "pair_same_unlabeled_rescued_vs_collapsed_table.csv",
        "pair_same_unlabeled_rescued_vs_collapsed_pairs_for_review.csv",
    ]:
        items.append(
            row(
                stem,
                "debug_csv",
                "Machine-readable separator / diagnostic table.",
                recommendation="keep_but_move_to_debug",
            )
        )

    for stem in [
        "pair_low_band_unlabeled_pairs.csv",
        "pair_high_band_false_positive_pairs.csv",
        "pair_high_band_true_positive_pairs.csv",
    ]:
        items.append(
            row(
                stem,
                "debug_csv",
                "Full pair listing for band (large).",
                recommendation="keep_but_move_to_debug",
                notes="Rarely opened manually; feeds review exports.",
            )
        )

    for stem in [
        "pair_low_band_unlabeled_pairs_for_review.csv",
        "pair_low_band_same_campaign_unlabeled_pairs_for_review.csv",
        "pair_low_band_cross_campaign_unlabeled_pairs_for_review.csv",
        "pair_high_band_false_positive_pairs_for_review.csv",
        "pair_cross_campaign_positive_pairs_for_review.csv",
    ]:
        items.append(
            row(
                stem,
                "review_csv",
                "Human review export with email text previews + inspection columns.",
                recommendation="keep_but_move_to_debug",
                notes="JSONL mirrors CSV; pick one format in cleanup.",
            )
        )

    for stem in [
        "pair_low_band_unlabeled_pairs_for_review.html",
        "pair_low_band_same_campaign_unlabeled_pairs_for_review.html",
        "pair_low_band_cross_campaign_unlabeled_pairs_for_review.html",
        "pair_high_band_false_positive_pairs_for_review.html",
        "pair_cross_campaign_positive_pairs_for_review.html",
    ]:
        items.append(
            row(
                stem,
                "review_html",
                "Sidebar TOC + pair cards; uses _write_pairs_for_review_html template.",
                recommendation="keep_as_primary",
                notes="Readable layout; missing body_only/path Jaccard in card header.",
            )
        )

    for stem in [
        "pair_same_unlabeled_rescued_for_review.html",
        "pair_same_unlabeled_collapsed_for_review.html",
        "pair_same_unlabeled_mid_for_review.html",
        "pair_same_unlabeled_rescued_vs_collapsed.html",
    ]:
        items.append(
            row(
                stem,
                "review_html",
                "Rescued/collapsed bucket review.",
                creator_fn=rescued,
                recommendation="keep_as_primary" if "rescued_vs_collapsed" not in stem else "keep_but_move_to_debug",
                notes=(
                    "Per-bucket HTML readable; combined rescued_vs_collapsed.html is huge (10k+ pairs) "
                    "and loads all cards at once — primary width/readability pain."
                ),
            )
        )

    items.append(
        row(
            "plots/score_distribution_*.png",
            "plots",
            "Per-GT and global score histograms (same/cross/pos/unl/cross_component variants).",
            recommendation="keep_as_primary",
            notes="~13 PNGs per GT file when multiple --gt-path; consider plots/<gt_slug>/ subfolder.",
        )
    )

    return items


def _html_feature_matrix() -> dict[str, Any]:
    low_high_template = "_write_pairs_for_review_html / _pair_card_html"
    rescued_template = "_write_rescued_collapsed_review_html (custom meta-grid)"

    features = [
        "body_token_jaccard",
        "body_char4gram_jaccard",
        "body_only_token_jaccard",
        "body_only_char4gram_jaccard",
        "path_token_jaccard_combined",
        "url_path_token_jaccard",
        "stem_path_token_jaccard",
        "sender_localpart_norm_jaccard",
        "shared_sender_count",
        "shared_stem_count",
        "shared_url_count",
        "n_shared_core_channels",
        "twohop_via_*",
        "from_2hop",
        "source_count",
    ]

    def col(html_type: str, feat: str) -> str:
        matrix = {
            (low_high_template, "body_token_jaccard"): "missing",
            (low_high_template, "body_char4gram_jaccard"): "missing",
            (low_high_template, "body_only_token_jaccard"): "missing",
            (low_high_template, "body_only_char4gram_jaccard"): "missing",
            (low_high_template, "path_token_jaccard_combined"): "missing",
            (low_high_template, "url_path_token_jaccard"): "missing",
            (low_high_template, "stem_path_token_jaccard"): "missing",
            (low_high_template, "sender_localpart_norm_jaccard"): "missing",
            (low_high_template, "shared_sender_count"): "partial",
            (low_high_template, "shared_stem_count"): "partial",
            (low_high_template, "shared_url_count"): "partial",
            (low_high_template, "n_shared_core_channels"): "shown",
            (low_high_template, "twohop_via_*"): "partial",
            (low_high_template, "from_2hop"): "shown",
            (low_high_template, "source_count"): "missing",
            (rescued_template, "body_token_jaccard"): "shown",
            (rescued_template, "body_char4gram_jaccard"): "shown",
            (rescued_template, "body_only_token_jaccard"): "shown",
            (rescued_template, "body_only_char4gram_jaccard"): "shown",
            (rescued_template, "path_token_jaccard_combined"): "shown",
            (rescued_template, "url_path_token_jaccard"): "shown",
            (rescued_template, "stem_path_token_jaccard"): "shown",
            (rescued_template, "sender_localpart_norm_jaccard"): "shown",
            (rescued_template, "shared_sender_count"): "partial",
            (rescued_template, "source_count"): "shown",
        }
        for k, v in list(matrix.items()):
            if k[0] == html_type and (feat == k[1] or (feat.endswith("*") and "twohop" in k[1])):
                return v
        if html_type == rescued_template and feat.startswith("body"):
            return "shown"
        if html_type == rescued_template and "path" in feat:
            return "shown"
        if html_type == low_high_template and feat.startswith("shared_"):
            return "partial"
        return matrix.get((html_type, feat), "missing")

    pages = [
        {
            "html_artifact": "pair_low_band_*_for_review.html",
            "template": low_high_template,
            "readable": "good",
            "width_issue": "low",
            "features_shown": ["score", "semantic_cosine", "time_gap", "provenance", "n_shared_core_channels", "admitting_evidence_lists"],
            "features_missing_in_template": [
                "body_*",
                "body_only_*",
                "path_token_jaccard_combined",
                "url_path_token_jaccard",
                "stem_path_token_jaccard",
                "sender_localpart_norm_jaccard",
                "source_count",
                "explicit shared_* counts",
            ],
            "data_available_in_csv": "yes — if pair_training_dataset has columns or MISP backfill in rescued path only",
            "root_cause": "_INSPECTION_FEATURE_COLS omits Jaccard columns; _pair_card_html meta-grid is short.",
        },
        {
            "html_artifact": "pair_high_band_false_positive_pairs_for_review.html",
            "template": low_high_template,
            "readable": "good",
            "width_issue": "low",
            "features_missing_in_template": ["body_only_*", "path_*", "body_token_jaccard"],
            "data_available_in_csv": "partial — inspection frame lacks path/body jaccard unless merged from df_eval",
        },
        {
            "html_artifact": "pair_same_unlabeled_rescued_vs_collapsed.html",
            "template": rescued_template,
            "readable": "poor",
            "width_issue": "high",
            "features_shown": [
                "score",
                "semantic",
                "body_token_jaccard",
                "body_char4gram_jaccard",
                "body_only_token_jaccard",
                "body_only_char4gram_jaccard",
                "sender_localpart_norm_jaccard",
                "path_token_jaccard_combined",
                "url_path_token_jaccard",
                "stem_path_token_jaccard",
                "provenance",
                "source_count",
            ],
            "features_missing_in_template": ["twohop channel badges in compact form only"],
            "data_available_in_csv": "yes — pair_same_unlabeled_rescued_vs_collapsed_pairs_for_review.csv",
            "root_cause": (
                "Custom meta-grid packs 10+ metrics on one flex row per card; no sidebar TOC; "
                "single HTML contains all buckets → very large DOM (10k+ pairs on full runs)."
            ),
        },
    ]

    return {"features_checked": features, "pages": pages, "templates": [low_high_template, rescued_template]}


def _scan_run_dir(run_dir: Path) -> dict[str, Any]:
    pss = run_dir / "pair_score_separation"
    if not pss.is_dir():
        return {"exists": False, "path": str(pss)}
    files = sorted(pss.rglob("*"))
    rows = []
    for f in files:
        if f.is_file():
            rel = str(f.relative_to(pss)).replace("\\", "/")
            rows.append({"relative_path": rel, "size_bytes": f.stat().st_size})
    htmls = [r for r in rows if r["relative_path"].endswith(".html")]
    jsons = [r for r in rows if r["relative_path"].endswith(".json")]
    csvs = [r for r in rows if r["relative_path"].endswith(".csv")]
    pngs = [r for r in rows if r["relative_path"].endswith(".png")]
    total_bytes = sum(r["size_bytes"] for r in rows)
    return {
        "exists": True,
        "path": str(pss),
        "n_files": len(rows),
        "total_size_mb": round(total_bytes / (1024 * 1024), 2),
        "n_html": len(htmls),
        "n_json": len(jsons),
        "n_csv": len(csvs),
        "n_png": len(pngs),
        "largest_html": max(htmls, key=lambda x: x["size_bytes"], default=None),
        "files": rows,
    }


def main() -> int:
    OUT_ANALYSIS.mkdir(parents=True, exist_ok=True)
    catalog = _artifact_catalog_from_code()
    example_run = _REPO / "output" / "runs" / "main_gnn_pu_1_no_ts_dedup_task_identity_10"
    scanned = _scan_run_dir(example_run)

    summary: dict[str, Any] = {
        "schema": "pair_score_separation_output_audit_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "inspected_modules": [
            "seed_candidate_workflow/utils/pair_score_separation.py",
            "seed_candidate_workflow/utils/pair_same_unlabeled_rescued_collapsed.py",
            "seed_candidate_workflow/utils/pair_low_band_feature_discovery.py",
            "seed_candidate_workflow/utils/scorer_diagnostics_rules.py",
            "seed_candidate_workflow/scripts/run_pair_score_separation_analysis.py",
        ],
        "example_run_scanned": str(example_run),
        "example_run_inventory": scanned,
        "primary_outputs_recommended": [
            "pair_score_separation_summary.json",
            "plots/score_distribution_all_scored_pairs.png",
            "plots/score_distribution_same_campaign_<gt>.png",
            "plots/score_distribution_cross_campaign_<gt>.png",
            "pair_low_band_unlabeled_pairs_for_review.html",
            "pair_same_unlabeled_rescued_vs_collapsed_summary.json",
            "pair_same_unlabeled_rescued_for_review.html",
            "pair_same_unlabeled_collapsed_for_review.html",
        ],
        "biggest_problems": [
            "Flat output directory (~45+ files per run) with no subfolders.",
            "Six+ overlapping JSON summaries (low/high marginal + joint + FP + twohop + rescued).",
            "Separator FEATURE_KEYS_DEFAULT omits body_only and path Jaccard — JSON summaries under-report new features.",
            "Low/high band HTML uses sparse meta-grid; rescued HTML shows Jaccards but is unreadable at scale.",
            "pair_same_unlabeled_rescued_vs_collapsed.html can exceed 40MB (all pairs, no TOC).",
            "Duplicate CSV+JSONL+HTML triple exports for each review cohort.",
            "13 plots per GT when multiple --gt-path passed — filenames help but clutter top level plots/.",
        ],
        "cleanup_first_priorities": [
            "1. Reorganize output into review_html/, core_json/, plots/, debug_csv/ (no behavior change).",
            "2. Unify HTML pair card: add body_only + path Jaccard block to _pair_card_html (reuse rescued template fields).",
            "3. Split rescued_vs_collapsed combined HTML or add sidebar TOC; cap pairs per page.",
            "4. Collapse JSON summaries into pair_score_separation_summary.json + optional debug bundle.",
            "5. Extend FEATURE_KEYS_DEFAULT / band separator stats for body_only_* and path_token_jaccard_combined.",
        ],
        "proposed_folder_layout": {
            "pair_score_separation/": {
                "README.txt": "index of artifacts",
                "core_json/": ["pair_score_separation_summary.json", "pair_same_unlabeled_rescued_vs_collapsed_summary.json"],
                "review_html/": ["*_for_review.html"],
                "plots/": ["score_distribution_*.png"],
                "debug_csv/": ["*_table.csv", "*_pairs.csv", "*.jsonl"],
                "debug_json/": ["pair_*_separator_summary.json", "pair_high_band_false_positive_summary.json"],
            }
        },
        "summary_json_overlap": {
            "pair_score_separation_summary.json": {
                "role": "canonical run index",
                "overlaps_with": ["per_gt plots paths", "band_diagnostics"],
            },
            "pair_low_band_separator_summary.json": {
                "role": "low marginal separators",
                "overlaps_with": ["pair_score_separation_summary per_gt", "pair_low_band_joint_separator_summary.json"],
                "stale_risk": "FEATURE_KEYS_DEFAULT lacks body/path jaccard",
            },
            "pair_high_band_separator_summary.json": {
                "role": "high marginal separators",
                "overlaps_with": ["pair_high_band_joint", "pair_high_band_false_positive_summary.json"],
            },
            "pair_same_unlabeled_rescued_vs_collapsed_summary.json": {
                "role": "rescued vs collapsed — keep separate",
                "overlaps_with": ["joint summary only for rule tables"],
            },
        },
        "html_feature_audit": _html_feature_matrix(),
        "artifact_catalog": catalog,
        "plot_recommendations": {
            "keep": [
                "score_distribution_all_scored_pairs.png",
                "score_distribution_same_campaign_<gt>.png",
                "score_distribution_cross_campaign_<gt>.png",
            ],
            "optional_debug": [
                "score_distribution_cross_component_*",
                "positive_only / unlabeled_only variants",
            ],
            "defer": [
                "html_fp bucket plots (not produced today)",
                "source_count bucket plots",
                "body_only jaccard bucket plots",
            ],
            "rationale": "Histograms already answer same vs cross separation; feature-specific plots only after HTML/summary show body/path lift.",
        },
    }

    inv_path = OUT_ANALYSIS / "pair_score_separation_output_inventory.csv"
    with inv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["path_pattern", "category", "creator", "purpose", "recommendation", "notes"],
        )
        w.writeheader()
        for item in catalog:
            w.writerow({k: item.get(k, "") for k in w.fieldnames})

    json_path = OUT_ANALYSIS / "pair_score_separation_output_audit_summary.json"
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    md = _render_markdown(summary, scanned)
    md_path = OUT_ANALYSIS / "pair_score_separation_output_audit_report.md"
    md_path.write_text(md, encoding="utf-8")

    print(json.dumps({"written": [str(json_path), str(md_path), str(inv_path)]}, indent=2))
    return 0


def _render_markdown(summary: dict[str, Any], scanned: dict[str, Any]) -> str:
    lines = [
        "# Pair score separation — output audit",
        "",
        f"Generated: `{summary['created_at_utc']}` (read-only audit; no pipeline changes).",
        "",
        "## 1. Scope",
        "",
        "This audit inventories artifacts under `<run_dir>/pair_score_separation/` produced by:",
        "",
        "- `python seed_candidate_workflow/scripts/run_pair_score_separation_analysis.py`",
        "- Rescued-vs-collapsed block inside the same run (`enable_rescued_collapsed_analysis`)",
        "- Optional separate: `pair_low_band_feature_discovery` (writes into the same folder)",
        "",
        "## 2. Example run on disk",
        "",
    ]
    if scanned.get("exists"):
        lines.extend(
            [
                f"Scanned: `{scanned['path']}`",
                f"- **{scanned['n_files']}** files, **{scanned['total_size_mb']} MB** total",
                f"- HTML: {scanned['n_html']}, JSON: {scanned['n_json']}, CSV: {scanned['n_csv']}, PNG: {scanned['n_png']}",
            ]
        )
        if scanned.get("largest_html"):
            lh = scanned["largest_html"]
            lines.append(
                f"- Largest HTML: `{lh['relative_path']}` ({round(lh['size_bytes'] / (1024 * 1024), 1)} MB)"
            )
    else:
        lines.append("No example `pair_score_separation` folder found on disk for _10/_11; catalog is code-derived.")

    lines.extend(
        [
            "",
            "## 3. Primary outputs (recommended front page)",
            "",
        ]
    )
    for p in summary["primary_outputs_recommended"]:
        lines.append(f"- `{p}`")

    lines.extend(
        [
            "",
            "## 4. Biggest problems",
            "",
        ]
    )
    for p in summary["biggest_problems"]:
        lines.append(f"- {p}")

    lines.extend(
        [
            "",
            "## 5. HTML feature visibility",
            "",
            "| Page family | Readable? | Width issue | Body/path Jaccard in HTML |",
            "|-------------|-----------|-------------|---------------------------|",
        ]
    )
    for page in summary["html_feature_audit"]["pages"]:
        missing = ", ".join(page.get("features_missing_in_template", [])[:4])
        lines.append(
            f"| `{page['html_artifact']}` | {page.get('readable', '?')} | {page.get('width_issue', '?')} | missing: {missing or 'see report'} |"
        )

    lines.extend(
        [
            "",
            "**Key finding:** Low/high band review HTML (`_write_pairs_for_review_html`) does **not** render "
            "`body_only_*`, `path_token_jaccard_combined`, or `url_path_token_jaccard` in the card header, even when "
            "columns exist on the pair-training table. Rescued-vs-collapsed HTML **does** show them, but packs "
            "10+ metrics into one `meta-grid` row and often ships one giant file for all buckets.",
            "",
            "## 6. JSON summary overlap",
            "",
        ]
    )
    for name, meta in summary["summary_json_overlap"].items():
        lines.append(f"### `{name}`")
        lines.append(f"- Role: {meta.get('role', '')}")
        if meta.get("overlaps_with"):
            lines.append(f"- Overlaps: {', '.join(meta['overlaps_with'])}")
        if meta.get("stale_risk"):
            lines.append(f"- Stale risk: {meta['stale_risk']}")
        lines.append("")

    lines.extend(
        [
            "## 7. Proposed folder layout (cleanup phase)",
            "",
            "```",
            "pair_score_separation/",
            "  core_json/          # 1–2 summaries",
            "  review_html/        # human review pages",
            "  plots/",
            "  debug_csv/          # tables + full pair dumps",
            "  debug_json/         # separator / joint summaries",
            "```",
            "",
            "## 8. Cleanup order (recommended)",
            "",
        ]
    )
    for step in summary["cleanup_first_priorities"]:
        lines.append(f"- {step}" if not str(step).strip().startswith("-") else str(step))

    lines.extend(
        [
            "",
            "## 9. Full artifact catalog",
            "",
            "See `pair_score_separation_output_inventory.csv` and "
            "`pair_score_separation_output_audit_summary.json` for per-file recommendations "
            "(keep_as_primary / debug / merge / remove).",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
