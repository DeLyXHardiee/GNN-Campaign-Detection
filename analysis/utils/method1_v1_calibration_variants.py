"""
Final Method 1 Version 1 calibration variant definitions (unsupervised).

Defines a focused set of blend + trust-gamma combinations, stable directory names,
and helpers to generate artifacts and build diagnostics experiment specs.

Convex combination rule (``blend_rule == "convex"``)::

    edge_weight_refined = alpha * norm(edge_weight_orig) + (1 - alpha) * edge_trust_calibrated

where ``norm`` is **percentile min–max** on the Step-2 edge list (unsupervised):

- ``p_lo = 5th percentile``, ``p_hi = 95th percentile`` of ``edge_weight_orig``
- ``norm_w = clip((w - p_lo) / max(p_hi - p_lo, 1e-12), 0, 1)``

Both summands lie in ``[0, 1]``, so ``edge_weight_refined`` lies in ``[0, 1]``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from analysis.utils.semantic_shard_edge_refinement_method1 import (
    Method1RefinementConfig,
    run_method1_edge_refinement_pipeline,
    save_method1_calibration_variant_bundle,
)


# Fair low grid for refined weights (optional high tail for sanity)
THRESH_V1_REFINED_LOW: list[float] = [
    0.02,
    0.05,
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.40,
    0.50,
    0.55,
    0.60,
]

STANDARD_ARTIFACT_FILENAMES = {
    "edges": "semantic_shard_step2_edges_refined.csv",
    "config": "semantic_shard_method1_config.json",
    "summary": "semantic_shard_method1_fit_summary.json",
}


def variant_id_softened(blend_floor: float, trust_gamma: float) -> str:
    return f"m1_soft_floor{int(round(float(blend_floor) * 100)):03d}_gamma{int(round(float(trust_gamma) * 100)):03d}"


def variant_id_multiplicative(trust_gamma: float) -> str:
    return f"m1_mult_gamma{int(round(float(trust_gamma) * 100)):03d}"


def variant_id_convex(convex_alpha: float, trust_gamma: float) -> str:
    return f"m1_convex_a{int(round(float(convex_alpha) * 100)):03d}_gamma{int(round(float(trust_gamma) * 100)):03d}"


def v1_final_variant_definitions() -> list[dict[str, Any]]:
    """
    Focused V1 calibration variants (perturbation + local structure ON in base cfg).

    Each entry: variant_id, refinement_variant_name (human), method1_overrides (for cfg merge).
    """
    rows: list[dict[str, Any]] = []

    def add(vid: str, label: str, overrides: dict[str, Any]) -> None:
        rows.append(
            {
                "variant_id": vid,
                "refinement_variant_name": label,
                "method1_overrides": overrides,
            }
        )

    # Group 1 — softened multiplicative
    add(variant_id_softened(0.2, 1.0), "V1_soft_f0.2_g1.0", {"blend_rule": "softened", "blend_floor": 0.2, "trust_gamma": 1.0})
    add(variant_id_softened(0.4, 1.0), "V1_soft_f0.4_g1.0", {"blend_rule": "softened", "blend_floor": 0.4, "trust_gamma": 1.0})
    add(variant_id_softened(0.6, 1.0), "V1_soft_f0.6_g1.0", {"blend_rule": "softened", "blend_floor": 0.6, "trust_gamma": 1.0})
    add(variant_id_softened(0.4, 0.75), "V1_soft_f0.4_g0.75", {"blend_rule": "softened", "blend_floor": 0.4, "trust_gamma": 0.75})
    add(variant_id_softened(0.4, 0.5), "V1_soft_f0.4_g0.5", {"blend_rule": "softened", "blend_floor": 0.4, "trust_gamma": 0.5})

    # Group 2 — convex combination
    add(variant_id_convex(0.25, 1.0), "V1_convex_a0.25_g1.0", {"blend_rule": "convex", "convex_alpha": 0.25, "trust_gamma": 1.0})
    add(variant_id_convex(0.5, 1.0), "V1_convex_a0.5_g1.0", {"blend_rule": "convex", "convex_alpha": 0.5, "trust_gamma": 1.0})
    add(variant_id_convex(0.75, 1.0), "V1_convex_a0.75_g1.0", {"blend_rule": "convex", "convex_alpha": 0.75, "trust_gamma": 1.0})
    add(variant_id_convex(0.5, 0.75), "V1_convex_a0.5_g0.75", {"blend_rule": "convex", "convex_alpha": 0.5, "trust_gamma": 0.75})
    add(variant_id_convex(0.5, 0.5), "V1_convex_a0.5_g0.5", {"blend_rule": "convex", "convex_alpha": 0.5, "trust_gamma": 0.5})

    # Group 3 — multiplicative reference
    add(variant_id_multiplicative(1.0), "V1_mult_g1.0", {"blend_rule": "multiplicative", "trust_gamma": 1.0})
    add(variant_id_multiplicative(0.75), "V1_mult_g0.75", {"blend_rule": "multiplicative", "trust_gamma": 0.75})
    add(variant_id_multiplicative(0.5), "V1_mult_g0.5", {"blend_rule": "multiplicative", "trust_gamma": 0.5})

    return rows


def discover_calibration_variant_bundles(runs_root: Path) -> list[dict[str, Any]]:
    """
    Scan ``runs_root`` for subdirectories containing a refined edge CSV.

    Returns sorted list of dicts with paths (only entries with edges file).
    """
    runs_root = Path(runs_root)
    if not runs_root.is_dir():
        return []
    out: list[dict[str, Any]] = []
    edges_name = STANDARD_ARTIFACT_FILENAMES["edges"]
    for d in sorted(runs_root.iterdir()):
        if not d.is_dir():
            continue
        p_edges = d / edges_name
        if not p_edges.is_file():
            continue
        p_cfg = d / STANDARD_ARTIFACT_FILENAMES["config"]
        p_sum = d / STANDARD_ARTIFACT_FILENAMES["summary"]
        out.append(
            {
                "variant_id": d.name,
                "bundle_dir": d.resolve(),
                "edges_csv": p_edges.resolve(),
                "config_json": p_cfg.resolve() if p_cfg.is_file() else None,
                "fit_summary_json": p_sum.resolve() if p_sum.is_file() else None,
            }
        )
    return out


def write_manifest(runs_root: Path, bundles: list[dict[str, Any]] | None = None) -> Path:
    """Write ``manifest.json`` listing variant_ids under ``runs_root``."""
    runs_root = Path(runs_root)
    runs_root.mkdir(parents=True, exist_ok=True)
    if bundles is None:
        bundles = discover_calibration_variant_bundles(runs_root)
    manifest = {
        "variants": [{"variant_id": b["variant_id"]} for b in bundles],
        "threshold_grid_v1_refined_low": THRESH_V1_REFINED_LOW,
    }
    p = runs_root / "manifest.json"
    p.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return p


def generate_v1_calibration_bundle(
    baseline_edges_df: pd.DataFrame,
    *,
    base_cfg: Method1RefinementConfig,
    variant_id: str,
    method1_overrides: dict[str, Any],
    runs_root: Path,
    force: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any], Path]:
    """
    Run Method 1 and save CSV + config + fit summary under ``runs_root / variant_id /``.

    Returns ``(refined_df, fit_summary, bundle_dir)``.
    """
    runs_root = Path(runs_root)
    bundle_dir = runs_root / variant_id
    edges_path = bundle_dir / STANDARD_ARTIFACT_FILENAMES["edges"]
    if edges_path.is_file() and not force:
        df = pd.read_csv(edges_path)
        df["shard_a"] = df["shard_a"].astype(str)
        df["shard_b"] = df["shard_b"].astype(str)
        summary_path = bundle_dir / STANDARD_ARTIFACT_FILENAMES["summary"]
        fit_summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {}
        return df, fit_summary, bundle_dir

    d = base_cfg.to_dict()
    for k, v in (method1_overrides or {}).items():
        if k in Method1RefinementConfig.__dataclass_fields__:
            d[k] = v
    cfg = Method1RefinementConfig.from_dict(d)
    refined, fit_summary, _ = run_method1_edge_refinement_pipeline(
        baseline_edges_df, cfg=cfg, output_dir=None
    )
    save_method1_calibration_variant_bundle(refined, bundle_dir=bundle_dir, cfg=cfg, fit_summary=fit_summary)
    return refined, fit_summary, bundle_dir


def generate_all_v1_calibration_bundles(
    baseline_edges_df: pd.DataFrame,
    *,
    base_cfg: Method1RefinementConfig,
    runs_root: Path,
    force: bool = False,
) -> list[dict[str, Any]]:
    """Generate every variant from :func:`v1_final_variant_definitions`."""
    runs_root = Path(runs_root)
    runs_root.mkdir(parents=True, exist_ok=True)
    done: list[dict[str, Any]] = []
    for spec in v1_final_variant_definitions():
        _, _, bundle_dir = generate_v1_calibration_bundle(
            baseline_edges_df,
            base_cfg=base_cfg,
            variant_id=spec["variant_id"],
            method1_overrides=spec["method1_overrides"],
            runs_root=runs_root,
            force=force,
        )
        done.append({"variant_id": spec["variant_id"], "bundle_dir": str(bundle_dir)})
    write_manifest(runs_root)
    return done


def experiment_specs_for_v1_variants(
    *,
    resolution_values: list[float],
    min_edge_weight_values: list[float] | None = None,
    runs_root: Path,
) -> list[dict[str, Any]]:
    """
    Build calibration experiment spec dicts (Louvain + Leiden each) for diagnostics.

    ``runs_root`` should be the directory containing per-variant subfolders (e.g. ``DIAG_OUT_DIR / "method1_v1_calibration_runs"``).
    """
    thr = min_edge_weight_values if min_edge_weight_values is not None else THRESH_V1_REFINED_LOW
    runs_root = Path(runs_root)
    specs: list[dict[str, Any]] = []
    for v in v1_final_variant_definitions():
        vid = v["variant_id"]
        ov = v["method1_overrides"]
        blend = str(ov.get("blend_rule", "multiplicative"))
        floor = float(ov.get("blend_floor", 0.0))
        gamma = float(ov.get("trust_gamma", 1.0))
        c_alpha = float(ov["convex_alpha"]) if blend == "convex" else math.nan
        for cm in ("louvain", "leiden"):
            specs.append(
                {
                    "experiment_name": f"V1_{vid}_{cm}",
                    "edge_source_type": "method1",
                    "community_method": cm,
                    "weight_column": "edge_weight_refined",
                    "refinement_variant_name": v["refinement_variant_name"],
                    "use_perturbation_stability": True,
                    "use_local_structure": True,
                    "blend_rule": blend,
                    "blend_floor": floor if blend == "softened" else "",
                    "trust_gamma": gamma,
                    "convex_alpha": c_alpha if blend == "convex" else "",
                    "threshold_grid_name": "v1_refined_low",
                    "resolution_values": resolution_values,
                    "min_edge_weight_values": list(thr),
                    "refined_variant_cache_id": vid,
                    "refined_source_mode": "artifact_bundle",
                    "method1_calibration_runs_root": str(runs_root.resolve()),
                    "method1_overrides": ov,
                }
            )
    return specs
