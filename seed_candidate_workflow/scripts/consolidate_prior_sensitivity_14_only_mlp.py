#!/usr/bin/env python3
"""Consolidate best community row per nnPU prior for _14_only_mlp prior-sensitivity runs."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from seed_candidate_workflow.utils.prior_sensitivity_14_only_mlp import (  # noqa: E402
    community_sweep_csv,
    format_latex_prior_table,
    load_manifest,
    prior_entries,
    scoring_run_dir,
)


def _best_row(df: pd.DataFrame) -> pd.Series:
    d = df.copy()
    d["_v"] = pd.to_numeric(d["v_measure"], errors="coerce")
    d = d.sort_values("_v", ascending=False)
    return d.iloc[0]


def consolidate(
    *,
    manifest: dict[str, Any],
    repo: Path,
    gt_slug: str,
    out_dir: Path,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    inputs: list[dict[str, str]] = []

    for entry in prior_entries(manifest):
        pi = float(entry["pi"])
        scoring_run_id = str(entry["scoring_run_id"])
        run_id = str(entry["run_id"])
        sweep_path = community_sweep_csv(repo, scoring_run_id, gt_slug=gt_slug)
        if not sweep_path.is_file():
            raise FileNotFoundError(
                f"Missing community sweep for pi={pi}: {sweep_path}\n"
                f"Run: python seed_candidate_workflow/scripts/run_prior_sensitivity_14_only_mlp.py --phase community"
            )
        df = pd.read_csv(sweep_path, low_memory=False)
        best = _best_row(df)
        row = {
            "pi": pi,
            "pi_slug": str(entry.get("pi_slug") or ""),
            "run_id": run_id,
            "scoring_run_id": scoring_run_id,
            "algorithm": str(best.get("method") or ""),
            "method": str(best.get("method") or ""),
            "threshold": float(best["min_edge_weight"]) if pd.notna(best.get("min_edge_weight")) else None,
            "min_edge_weight": float(best["min_edge_weight"]) if pd.notna(best.get("min_edge_weight")) else None,
            "resolution": float(best["resolution"]) if pd.notna(best.get("resolution")) else None,
            "homogeneity": float(best["homogeneity"]) if pd.notna(best.get("homogeneity")) else None,
            "completeness": float(best["completeness"]) if pd.notna(best.get("completeness")) else None,
            "v_measure": float(best["v_measure"]) if pd.notna(best.get("v_measure")) else None,
            "n_edges_after_threshold": float(best["n_edges_after_threshold"])
            if pd.notna(best.get("n_edges_after_threshold"))
            else None,
            "n_communities": float(best["n_communities"]) if pd.notna(best.get("n_communities")) else None,
            "n_eval": float(best["n_eval"]) if pd.notna(best.get("n_eval")) else None,
            "coverage_gt": float(best["coverage_gt"]) if pd.notna(best.get("coverage_gt")) else None,
            "sweep_csv": str(sweep_path),
            "training_run_dir": str(repo / "output/runs" / run_id),
        }
        rows.append(row)
        inputs.append({"pi": str(pi), "sweep_csv": str(sweep_path)})

    out_dir.mkdir(parents=True, exist_ok=True)
    best_df = pd.DataFrame(rows).sort_values("pi")
    p_csv = out_dir / "prior_sensitivity_14_only_mlp_best_by_pi.csv"
    p_json = out_dir / "prior_sensitivity_14_only_mlp_best_by_pi.json"
    best_df.to_csv(p_csv, index=False)
    p_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    latex = format_latex_prior_table(rows)
    p_tex = out_dir / "prior_sensitivity_14_only_mlp_best_by_pi.tex"
    p_tex.write_text(latex, encoding="utf-8")

    manifest_out = {
        "gt_slug": gt_slug,
        "n_priors": len(rows),
        "inputs": inputs,
        "outputs": {
            "best_by_pi_csv": str(p_csv),
            "best_by_pi_json": str(p_json),
            "latex_table": str(p_tex),
        },
        "per_prior_scoring_run_dirs": {
            str(r["pi"]): str(scoring_run_dir(repo, str(r["scoring_run_id"]))) for r in rows
        },
    }
    p_manifest = out_dir / "prior_sensitivity_14_only_mlp_consolidation_manifest.json"
    p_manifest.write_text(json.dumps(manifest_out, indent=2), encoding="utf-8")
    manifest_out["manifest_json"] = str(p_manifest)
    return manifest_out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--gt-slug", type=str, default="ground_truth")
    args = p.parse_args()

    manifest = load_manifest(args.manifest)
    out_dir = args.out_dir
    if out_dir is None:
        out_dir = Path(str(manifest.get("consolidation_output_dir") or "seed_candidate_workflow/output/prior_sensitivity_14_only_mlp"))
    if not out_dir.is_absolute():
        out_dir = (_REPO / out_dir).resolve()

    out = consolidate(manifest=manifest, repo=_REPO, gt_slug=str(args.gt_slug).strip(), out_dir=out_dir)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
