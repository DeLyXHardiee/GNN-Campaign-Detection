"""Health checks for seed union and pair_training_dataset summaries (guardrails)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def run_health_checks(
    *,
    anchor_seed_summary: Path | None = None,
    pair_training_summary: Path | None = None,
    max_union_largest_component: int = 400,
    min_union_components: int = 450,
    max_same_seed_component_fraction: float = 0.48,
) -> list[str]:
    """Return a list of human-readable failure messages (empty => OK)."""
    errs: list[str] = []

    if anchor_seed_summary is not None:
        p = anchor_seed_summary.expanduser().resolve()
        if not p.is_file():
            errs.append(f"anchor_seed_summary not found: {p}")
        else:
            d = load_json(p)
            uni = (d.get("union_edges") or {}).get("metrics") or {}
            top = uni.get("component_size_distribution_top50") or []
            n_comp = int(uni.get("n_components") or 0)
            largest = int(top[0]) if top else 0
            if largest > int(max_union_largest_component):
                errs.append(
                    f"union largest component {largest} > max {max_union_largest_component}"
                )
            if n_comp < int(min_union_components):
                errs.append(
                    f"union n_components {n_comp} < min {min_union_components}"
                )

    if pair_training_summary is not None:
        p = pair_training_summary.expanduser().resolve()
        if not p.is_file():
            errs.append(f"pair_training_dataset_summary not found: {p}")
        else:
            d = load_json(p)
            cc = d.get("component_context") or {}
            n_tot = int((d.get("pair_counts") or {}).get("n_unique_pairs_final") or 0)
            same = int(cc.get("n_pairs_same_seed_component") or 0)
            if n_tot > 0:
                frac = same / n_tot
                cap = float(max_same_seed_component_fraction)
                if frac > cap:
                    errs.append(
                        f"same_seed_component fraction {frac:.4f} > max {cap:.4f}"
                    )

    return errs
