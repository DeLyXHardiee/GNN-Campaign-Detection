"""
Merge thesis anchor/candidate graph community sweeps into one CSV + JSON for results/appendix.

Reads per-target sweep CSVs under scoring_runs (expanded GT slug ``ground_truth`` by default).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from seed_candidate_workflow.utils import graph_structure_helpers as gh


BASELINE_SPECS: list[dict[str, str]] = [
    {
        "baseline": "expanded_anchor_graph_community_detection",
        "scoring_run_id": "thesis_expanded_anchor_graph_community_detection__14_only_mlp_pair_universe__expanded_gt",
        "target": "anchor",
    },
    {
        "baseline": "seed_candidate_graph_community_detection",
        "scoring_run_id": "thesis_candidate_graph_community_detection__14_only_mlp_pair_universe__expanded_gt",
        "target": "seed_candidate",
    },
]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _load_sweep_csv(path: Path, *, baseline: str, target: str, scoring_run_id: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df.insert(0, "baseline", baseline)
    df.insert(1, "graph_target", target)
    df.insert(2, "scoring_run_id", scoring_run_id)
    df["algorithm"] = df["method"].astype(str)
    df["threshold"] = pd.to_numeric(df["min_edge_weight"], errors="coerce")
    df["resolution"] = pd.to_numeric(df["resolution"], errors="coerce")
    if "n_graph_nodes" not in df.columns:
        df["n_graph_nodes"] = pd.NA
    df["edges_used"] = pd.to_numeric(df.get("n_edges_after_threshold"), errors="coerce")
    return df


def _best_from_sweep(df: pd.DataFrame) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=object)
    d = df.copy()
    d["_v"] = pd.to_numeric(d["v_measure"], errors="coerce")
    d = d.sort_values("_v", ascending=False)
    return d.iloc[0].drop(labels=["_v"], errors="ignore")


def consolidate(
    *,
    project_root: Path,
    scoring_output_root: Path,
    gt_slug: str,
    out_dir: Path,
) -> dict[str, Any]:
    all_rows: list[pd.DataFrame] = []
    best_rows: list[dict[str, Any]] = []
    inputs: list[dict[str, str]] = []

    for spec in BASELINE_SPECS:
        comm_dir = (
            scoring_output_root
            / spec["scoring_run_id"]
            / spec["target"]
            / "community"
        )
        sweep_csv = comm_dir / f"anchor_community_sweep__{gt_slug}.csv"
        best_json = comm_dir / f"anchor_community_best__{gt_slug}.json"
        multi_json = comm_dir / "anchor_community_multi_gt_summary.json"
        if not sweep_csv.is_file():
            raise FileNotFoundError(f"Missing sweep CSV for {spec['baseline']}: {sweep_csv}")
        df = _load_sweep_csv(
            sweep_csv,
            baseline=spec["baseline"],
            target=spec["target"],
            scoring_run_id=spec["scoring_run_id"],
        )
        if multi_json.is_file():
            summary = _read_json(multi_json)
            n_nodes = summary.get("n_graph_nodes")
            if n_nodes is not None:
                df["n_graph_nodes"] = int(n_nodes)
        if best_json.is_file():
            best_payload = _read_json(best_json)
            br = dict(best_payload.get("best_row") or {})
            br["baseline"] = spec["baseline"]
            br["graph_target"] = spec["target"]
            br["scoring_run_id"] = spec["scoring_run_id"]
            best_rows.append(br)
        else:
            best_rows.append(_best_from_sweep(df).to_dict())
        all_rows.append(df)
        inputs.append(
            {
                "baseline": spec["baseline"],
                "sweep_csv": str(sweep_csv),
                "best_json": str(best_json) if best_json.is_file() else "",
                "multi_gt_summary_json": str(multi_json) if multi_json.is_file() else "",
            }
        )

    full = pd.concat(all_rows, ignore_index=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    p_full_csv = out_dir / "thesis_baseline_community_sweep_all_configs.csv"
    p_full_json = out_dir / "thesis_baseline_community_sweep_all_configs.json"
    full.to_csv(p_full_csv, index=False)
    p_full_json.write_text(
        json.dumps(full.to_dict(orient="records"), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    best_df = pd.DataFrame(best_rows)
    p_best_csv = out_dir / "thesis_baseline_community_best_by_v_measure.csv"
    p_best_json = out_dir / "thesis_baseline_community_best_by_v_measure.json"
    best_df.to_csv(p_best_csv, index=False)
    p_best_json.write_text(
        json.dumps(best_rows, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    main_cols = [
        "baseline",
        "algorithm",
        "method",
        "threshold",
        "min_edge_weight",
        "resolution",
        "homogeneity",
        "completeness",
        "v_measure",
        "n_graph_nodes",
        "edges_used",
        "n_edges_after_threshold",
        "n_communities",
        "n_eval",
        "coverage_gt",
    ]
    best_tbl = pd.DataFrame(best_rows)
    if "method" in best_tbl.columns and "algorithm" not in best_tbl.columns:
        best_tbl["algorithm"] = best_tbl["method"]
    if "min_edge_weight" in best_tbl.columns and "threshold" not in best_tbl.columns:
        best_tbl["threshold"] = best_tbl["min_edge_weight"]
    if "n_edges_after_threshold" in best_tbl.columns and "edges_used" not in best_tbl.columns:
        best_tbl["edges_used"] = best_tbl["n_edges_after_threshold"]
    present = [c for c in main_cols if c in best_tbl.columns]
    main = best_tbl[present].copy()
    p_main_csv = out_dir / "thesis_baseline_community_main_table.csv"
    p_main_json = out_dir / "thesis_baseline_community_main_table.json"
    main.to_csv(p_main_csv, index=False)
    p_main_json.write_text(
        json.dumps(main.to_dict(orient="records"), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    manifest = {
        "gt_slug": gt_slug,
        "n_sweep_rows_total": int(len(full)),
        "n_configs_per_baseline": int(len(full) / max(1, len(BASELINE_SPECS))),
        "inputs": inputs,
        "outputs": {
            "full_sweep_csv": str(p_full_csv),
            "full_sweep_json": str(p_full_json),
            "best_by_baseline_csv": str(p_best_csv),
            "best_by_baseline_json": str(p_best_json),
            "main_table_csv": str(p_main_csv),
            "main_table_json": str(p_main_json),
        },
    }
    p_manifest = out_dir / "thesis_baseline_community_consolidation_manifest.json"
    p_manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    manifest["manifest_json"] = str(p_manifest)
    return manifest


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--scoring-output-root",
        type=Path,
        default=Path("seed_candidate_workflow/output/scoring_runs"),
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("seed_candidate_workflow/output/thesis_baseline_community_detection"),
    )
    p.add_argument(
        "--gt-slug",
        type=str,
        default="ground_truth",
        help="GT file stem for expanded eval (default: ground_truth.json).",
    )
    args = p.parse_args()
    project_root = gh.find_project_root()
    scoring_root = args.scoring_output_root
    if not scoring_root.is_absolute():
        scoring_root = (project_root / scoring_root).resolve()
    out_dir = args.out_dir
    if not out_dir.is_absolute():
        out_dir = (project_root / out_dir).resolve()
    manifest = consolidate(
        project_root=project_root,
        scoring_output_root=scoring_root,
        gt_slug=str(args.gt_slug).strip(),
        out_dir=out_dir,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
