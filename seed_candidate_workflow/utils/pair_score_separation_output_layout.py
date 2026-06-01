"""Subfolder layout, export flags, and artifact manifest for ``pair_score_separation``."""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

LAYOUT_VERSION = "pair_score_separation_layout_v2"

SUBDIRS = (
    "core_json",
    "review_html",
    "plots",
    "debug_json",
    "debug_csv",
)

_MANIFEST_NAME = "pair_score_separation_manifest.json"

_PRIMARY_JSON_RE = re.compile(
    r"^(pair_score_separation_summary|pair_score_separation_manifest|"
    r"pair_mid_band_frontier_summary(?:__[^/]+)?|"
    r"pair_frontier_analysis_summary(?:__[^/]+)?|"
    r"pair_high_cross_unlabeled_analysis_summary(?:__[^/]+)?|"
    r"pair_same_unlabeled_rescued_vs_collapsed(?:__[^/]+)?_summary)\.json$"
)


@dataclass(frozen=True)
class ExportFlags:
    """Controls verbose/debug artifact generation (defaults: primary-only)."""

    emit_debug_json: bool = False
    emit_debug_csv: bool = False
    emit_debug_html: bool = False
    emit_review_jsonl: bool = False

    @classmethod
    def from_cli(
        cls,
        *,
        emit_debug_json: bool = False,
        emit_debug_csv: bool = False,
        emit_debug_html: bool = False,
        emit_review_jsonl: bool = False,
    ) -> ExportFlags:
        return cls(
            emit_debug_json=bool(emit_debug_json),
            emit_debug_csv=bool(emit_debug_csv),
            emit_debug_html=bool(emit_debug_html),
            emit_review_jsonl=bool(emit_review_jsonl),
        )


def path_for_write(path: Path) -> str:
    """Create parent dirs and return a path string safe for open/save on Windows (MAX_PATH)."""
    p = Path(path).resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    if os.name != "nt":
        return str(p)
    s = str(p)
    if s.startswith("\\\\?\\"):
        return s
    if s.startswith("\\\\"):
        return "\\\\?\\UNC\\" + s[2:]
    return "\\\\?\\" + s


def ensure_pair_score_separation_layout(root: Path) -> dict[str, Path]:
    """Create standard subdirs under ``root`` and return path handles."""
    root = Path(root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    out: dict[str, Path] = {"root": root}
    for name in SUBDIRS:
        p = root / name
        p.mkdir(parents=True, exist_ok=True)
        out[name] = p
    return out


def rel_to_root(layout: dict[str, Path], path: Path) -> str:
    return str(path.resolve().relative_to(layout["root"].resolve()).as_posix())


def _is_primary_json_filename(name: str) -> bool:
    return bool(_PRIMARY_JSON_RE.match(name))


def _classify_artifact(rel_path: str) -> tuple[str, bool]:
    """Return (category, is_primary)."""
    parts = rel_path.replace("\\", "/").split("/")
    if len(parts) < 2:
        return ("other", False)
    folder, name = parts[0], parts[-1]
    if folder == "core_json":
        return ("primary_json", _is_primary_json_filename(name))
    if folder == "review_html":
        primary = name in {
            "pair_low_band_unlabeled_pairs_for_review.html",
            "pair_mid_band_same_unlabeled_for_review.html",
            "pair_mid_band_cross_unlabeled_for_review.html",
            "pair_high_band_same_unlabeled_for_review.html",
            "pair_low_band_same_unlabeled_for_review.html",
        } or (
            name.startswith("pair_same_unlabeled_rescued_for_review")
            and name.endswith(".html")
        ) or (
            name.startswith("pair_same_unlabeled_collapsed_for_review")
            and name.endswith(".html")
        ) or name in (
            "pair_high_cross_unlabeled_for_review.html",
            "pair_high_same_unlabeled_for_review.html",
        )
        return ("review_html", primary)
    if folder == "plots":
        primary = name == "score_distribution_all_scored_pairs.png" or (
            name.startswith("score_distribution_same_campaign_")
            and name.endswith(".png")
            and "cross_component" not in name
            and "positive" not in name
            and "unlabeled" not in name
        ) or (
            name.startswith("score_distribution_cross_campaign_")
            and name.endswith(".png")
            and "cross_component" not in name
            and "positive" not in name
            and "unlabeled" not in name
        )
        return ("plot", primary)
    if folder == "debug_json":
        return ("debug_json", False)
    if folder == "debug_csv":
        primary = name == "detail_pair_score_band_diagnostics.csv"
        return ("debug_csv", primary)
    return ("other", False)


def scan_generated_artifacts(layout: dict[str, Path]) -> list[dict[str, Any]]:
    """List files under the layout tree with category and size."""
    entries: list[dict[str, Any]] = []
    root = layout["root"]
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = rel_to_root(layout, path)
        category, is_primary = _classify_artifact(rel)
        try:
            size_bytes = int(path.stat().st_size)
        except OSError:
            size_bytes = None
        entries.append(
            {
                "path": rel,
                "category": category,
                "tier": "primary" if is_primary else "secondary",
                "size_bytes": size_bytes,
            }
        )
    return entries


def build_artifact_manifest(
    *,
    layout: dict[str, Path],
    export_flags: ExportFlags,
    run_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    artifacts = scan_generated_artifacts(layout)
    primary_jsons = sorted(
        a["path"] for a in artifacts if a["category"] == "primary_json" and a["tier"] == "primary"
    )
    debug_jsons = sorted(a["path"] for a in artifacts if a["category"] == "debug_json")
    return {
        "manifest_version": LAYOUT_VERSION,
        "export_flags": asdict(export_flags),
        "artifacts": artifacts,
        "primary_jsons": primary_jsons,
        "debug_jsons": debug_jsons,
        "counts": {
            "n_artifacts": len(artifacts),
            "n_primary": sum(1 for a in artifacts if a["tier"] == "primary"),
            "n_secondary": sum(1 for a in artifacts if a["tier"] == "secondary"),
        },
        "run_meta": run_meta or {},
    }


def write_artifact_manifest(
    *,
    layout: dict[str, Path],
    export_flags: ExportFlags,
    run_meta: dict[str, Any] | None = None,
) -> Path:
    manifest_path = layout["core_json"] / _MANIFEST_NAME
    payload = build_artifact_manifest(
        layout=layout, export_flags=export_flags, run_meta=run_meta
    )
    manifest_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return manifest_path


def recommended_reading_order(
    *,
    layout: dict[str, Path],
    has_rescued_collapsed: bool = True,
    has_mid_band: bool = True,
    has_high_cross: bool = False,
    gnn_only_scorer_hint: bool = False,
) -> list[dict[str, str]]:
    steps: list[dict[str, str]] = [
        {
            "step": 1,
            "label": "Main summary",
            "path": rel_to_root(layout, layout["core_json"] / "pair_score_separation_summary.json"),
        },
        {
            "step": 2,
            "label": "Artifact manifest (what was generated)",
            "path": rel_to_root(layout, layout["core_json"] / _MANIFEST_NAME),
        },
        {
            "step": 3,
            "label": "Low-band manual review HTML",
            "path": rel_to_root(
                layout, layout["review_html"] / "pair_low_band_unlabeled_pairs_for_review.html"
            ),
        },
    ]
    n = 4
    if has_rescued_collapsed:
        steps.extend(
            [
                {
                    "step": n,
                    "label": "Rescued vs collapsed summary",
                    "path": rel_to_root(
                        layout,
                        layout["core_json"] / "pair_same_unlabeled_rescued_vs_collapsed_summary.json",
                    ),
                },
                {
                    "step": n + 1,
                    "label": "Rescued same-campaign unlabeled review HTML",
                    "path": rel_to_root(
                        layout,
                        layout["review_html"] / "pair_same_unlabeled_rescued_for_review.html",
                    ),
                },
                {
                    "step": n + 2,
                    "label": "Collapsed same-campaign unlabeled review HTML",
                    "path": rel_to_root(
                        layout,
                        layout["review_html"] / "pair_same_unlabeled_collapsed_for_review.html",
                    ),
                },
            ]
        )
        n += 3
    if has_mid_band:
        steps.extend(
            [
                {
                    "step": n,
                    "label": "Frontier analysis summary (low/mid/high unlabeled)",
                    "path": rel_to_root(
                        layout, layout["core_json"] / "pair_frontier_analysis_summary.json"
                    ),
                },
                {
                    "step": n + 1,
                    "label": "Low same unlabeled review HTML",
                    "path": rel_to_root(
                        layout,
                        layout["review_html"] / "pair_low_band_same_unlabeled_for_review.html",
                    ),
                },
                {
                    "step": n + 2,
                    "label": "Mid same unlabeled review HTML",
                    "path": rel_to_root(
                        layout,
                        layout["review_html"] / "pair_mid_band_same_unlabeled_for_review.html",
                    ),
                },
                {
                    "step": n + 3,
                    "label": "High same unlabeled review HTML",
                    "path": rel_to_root(
                        layout,
                        layout["review_html"] / "pair_high_band_same_unlabeled_for_review.html",
                    ),
                },
                {
                    "step": n + 4,
                    "label": "Mid cross unlabeled review HTML",
                    "path": rel_to_root(
                        layout,
                        layout["review_html"] / "pair_mid_band_cross_unlabeled_for_review.html",
                    ),
                },
            ]
        )
        n += 5
    if has_high_cross:
        hc_steps = [
            {
                "step": n,
                "label": "High-cross unlabeled analysis summary (GNN-only FP diagnosis)",
                "path": rel_to_root(
                    layout,
                    layout["core_json"] / "pair_high_cross_unlabeled_analysis_summary.json",
                ),
            },
            {
                "step": n + 1,
                "label": "High-score cross-campaign unlabeled review HTML",
                "path": rel_to_root(
                    layout,
                    layout["review_html"] / "pair_high_cross_unlabeled_for_review.html",
                ),
            },
            {
                "step": n + 2,
                "label": "High-score same-campaign unlabeled review HTML (comparison cohort)",
                "path": rel_to_root(
                    layout,
                    layout["review_html"] / "pair_high_same_unlabeled_for_review.html",
                ),
            },
        ]
        if gnn_only_scorer_hint:
            hc_steps[0]["label"] = (
                "★ GNN-only ablation: high-cross false-positive diagnosis — start here"
            )
        steps.extend(hc_steps)
        n += 3
    steps.append(
        {
            "step": n,
            "label": "All-scored-pairs score histogram",
            "path": rel_to_root(layout, layout["plots"] / "score_distribution_all_scored_pairs.png"),
        }
    )
    steps.append(
        {
            "step": n + 1,
            "label": "Debug JSON/CSV (optional — enable --emit-debug-* flags)",
            "path": "debug_json/ and debug_csv/",
        }
    )
    return steps


def build_primary_outputs(
    *,
    layout: dict[str, Path],
    rescued_suffix: str = "",
    per_gt: list[dict[str, Any]] | None = None,
    has_mid_band: bool = True,
    has_high_cross: bool = False,
    gnn_only_scorer_hint: bool = False,
) -> dict[str, str]:
    """Minimal navigation hub — primary artifacts only."""
    rc_name = f"pair_same_unlabeled_rescued_vs_collapsed{rescued_suffix}_summary.json"
    out: dict[str, str] = {
        "main_summary_json": rel_to_root(
            layout, layout["core_json"] / "pair_score_separation_summary.json"
        ),
        "artifact_manifest_json": rel_to_root(layout, layout["core_json"] / _MANIFEST_NAME),
        "low_band_review_html": rel_to_root(
            layout, layout["review_html"] / "pair_low_band_unlabeled_pairs_for_review.html"
        ),
        "rescued_vs_collapsed_summary_json": rel_to_root(layout, layout["core_json"] / rc_name),
        "rescued_review_html": rel_to_root(
            layout,
            layout["review_html"] / f"pair_same_unlabeled_rescued_for_review{rescued_suffix}.html",
        ),
        "collapsed_review_html": rel_to_root(
            layout,
            layout["review_html"] / f"pair_same_unlabeled_collapsed_for_review{rescued_suffix}.html",
        ),
        "plot_all_scored_pairs": rel_to_root(
            layout, layout["plots"] / "score_distribution_all_scored_pairs.png"
        ),
    }
    if has_mid_band:
        out["frontier_analysis_summary_json"] = rel_to_root(
            layout, layout["core_json"] / "pair_frontier_analysis_summary.json"
        )
        out["frontier_analysis_joint_summary_json"] = rel_to_root(
            layout, layout["core_json"] / "pair_frontier_analysis_joint_summary.json"
        )
        out["frontier_low_same_review_html"] = rel_to_root(
            layout, layout["review_html"] / "pair_low_band_same_unlabeled_for_review.html"
        )
        out["frontier_mid_same_review_html"] = rel_to_root(
            layout, layout["review_html"] / "pair_mid_band_same_unlabeled_for_review.html"
        )
        out["frontier_high_same_review_html"] = rel_to_root(
            layout, layout["review_html"] / "pair_high_band_same_unlabeled_for_review.html"
        )
        out["frontier_mid_cross_review_html"] = rel_to_root(
            layout, layout["review_html"] / "pair_mid_band_cross_unlabeled_for_review.html"
        )
        out["mid_band_frontier_summary_json"] = rel_to_root(
            layout, layout["core_json"] / "pair_mid_band_frontier_summary.json"
        )
        out["mid_band_joint_summary_json"] = rel_to_root(
            layout, layout["core_json"] / "pair_mid_band_frontier_joint_summary.json"
        )
    if has_high_cross:
        out["high_cross_unlabeled_analysis_summary_json"] = rel_to_root(
            layout, layout["core_json"] / "pair_high_cross_unlabeled_analysis_summary.json"
        )
        out["high_cross_unlabeled_for_review_html"] = rel_to_root(
            layout, layout["review_html"] / "pair_high_cross_unlabeled_for_review.html"
        )
        out["high_same_unlabeled_for_review_html"] = rel_to_root(
            layout, layout["review_html"] / "pair_high_same_unlabeled_for_review.html"
        )
        if gnn_only_scorer_hint:
            out["gnn_only_high_cross_start_here"] = out["high_cross_unlabeled_analysis_summary_json"]
    if per_gt and len(per_gt) == 1:
        gt0 = per_gt[0]
        if gt0.get("plot_same_campaign"):
            out["plot_same_campaign_gt"] = str(gt0["plot_same_campaign"])
        if gt0.get("plot_cross_campaign"):
            out["plot_cross_campaign_gt"] = str(gt0["plot_cross_campaign"])
    return out


def build_navigation_index(
    *,
    layout: dict[str, Path],
    export_flags: ExportFlags,
    primary_outputs: dict[str, str],
    has_rescued_collapsed: bool = True,
    has_mid_band: bool = True,
    has_high_cross: bool = False,
    gnn_only_scorer_hint: bool = False,
) -> dict[str, Any]:
    manifest = build_artifact_manifest(layout=layout, export_flags=export_flags)
    return {
        "layout_version": LAYOUT_VERSION,
        "export_flags": asdict(export_flags),
        "primary_jsons": manifest["primary_jsons"],
        "debug_jsons": manifest["debug_jsons"],
        "primary_outputs": primary_outputs,
        "recommended_reading_order": recommended_reading_order(
            layout=layout,
            has_rescued_collapsed=has_rescued_collapsed,
            has_mid_band=has_mid_band,
            has_high_cross=has_high_cross,
            gnn_only_scorer_hint=gnn_only_scorer_hint,
        ),
    }


def primary_outputs_block(
    *,
    layout: dict[str, Path],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    block = build_primary_outputs(layout=layout)
    if extra:
        block.update({k: v for k, v in extra.items() if k in block})
    return block
