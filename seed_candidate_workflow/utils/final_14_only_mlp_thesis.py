"""Canonical thesis final _14_only_mlp run (timestamp log1p + early stopping)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def load_manifest(manifest_path: Path | None = None) -> dict[str, Any]:
    repo = Path(__file__).resolve().parents[2]
    path = manifest_path or (
        repo / "seed_candidate_workflow/configs/final_14_only_mlp/final_14_only_mlp.manifest.json"
    )
    return json.loads(Path(path).resolve().read_text(encoding="utf-8-sig"))


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_repo_path(repo: Path, rel: str) -> Path:
    p = Path(rel)
    return p if p.is_absolute() else (repo / p).resolve()


def training_run_dir(repo: Path, run_id: str) -> Path:
    return (repo / "output/runs" / run_id).resolve()


def scoring_run_dir(repo: Path, scoring_run_id: str) -> Path:
    return (repo / "seed_candidate_workflow/output/scoring_runs" / scoring_run_id).resolve()


def community_sweep_csv(repo: Path, scoring_run_id: str, *, gt_slug: str = "ground_truth") -> Path:
    return (
        scoring_run_dir(repo, scoring_run_id)
        / "seed_candidate"
        / "community"
        / f"anchor_community_sweep__{gt_slug}.csv"
    )


def steps_dir(repo: Path, manifest: dict[str, Any]) -> Path:
    d = resolve_repo_path(repo, str(manifest.get("steps_output_dir") or "seed_candidate_workflow/output/final_14_only_mlp_timestamp_es_steps"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def thesis_dir(repo: Path, manifest: dict[str, Any]) -> Path:
    d = resolve_repo_path(repo, str(manifest.get("thesis_output_dir") or "seed_candidate_workflow/output/final_14_only_mlp_timestamp_es_thesis"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def build_final_training_cfg(
    *,
    pair_dataset_csv: str,
    reference_training_config: Path,
    project_root: Path | None = None,
    pi: float = 0.1,
    epochs: int = 100,
    early_stopping_patience: int = 10,
    save_best_val_checkpoint_history: bool = True,
) -> dict[str, Any]:
    from seed_candidate_workflow.utils.prior_sensitivity_14_only_mlp import build_14_only_mlp_training_cfg

    cfg = build_14_only_mlp_training_cfg(
        pi=pi,
        pair_dataset_csv=pair_dataset_csv,
        reference_training_config=reference_training_config,
        project_root=project_root,
    )
    cfg["epochs"] = int(epochs)
    cfg["early_stopping_patience"] = int(early_stopping_patience)
    cfg["save_best_val_checkpoint_history"] = bool(save_best_val_checkpoint_history)
    return cfg


def read_training_stability(run_dir: Path, *, target_epochs: int | None = None) -> dict[str, Any]:
    from seed_candidate_workflow.utils.early_stopping_sanity_14_only_mlp import read_early_stopping_training_metrics

    return read_early_stopping_training_metrics(run_dir, target_epochs=target_epochs)


def pair_universe_stats(pair_csv: Path) -> dict[str, Any]:
    from seed_candidate_workflow.utils.timestamp_ablation_14_only_mlp import pair_universe_stats as _pus

    return _pus(pair_csv)


def verify_time_gating_disabled(candidate_config_path: Path) -> dict[str, Any]:
    cfg = json.loads(candidate_config_path.read_text(encoding="utf-8-sig"))
    gens = (cfg.get("candidates") or {}).get("generators") or []
    rows: list[dict[str, Any]] = []
    for g in gens:
        if not isinstance(g, dict):
            continue
        name = str(g.get("name") or "")
        enabled = bool(g.get("enabled", True))
        gc = g.get("config") or {}
        tg = gc.get("time_gating_enabled")
        rows.append(
            {
                "name": name,
                "enabled": enabled,
                "time_gating_enabled": tg,
                "max_time_gap_seconds": gc.get("max_time_gap_seconds"),
            }
        )
    gatable = [r for r in rows if r.get("time_gating_enabled") is not None]
    enabled_gatable = [r for r in gatable if r["enabled"]]
    return {
        "candidate_config": str(candidate_config_path),
        "n_generators": len(rows),
        "gatable_generators": gatable,
        "enabled_with_time_gating_flag": enabled_gatable,
        "all_gating_disabled": all(not bool(r.get("time_gating_enabled")) for r in enabled_gatable),
    }


def best_community_row(sweep_csv: Path) -> dict[str, Any]:
    df = pd.read_csv(sweep_csv, low_memory=False)
    df["_v"] = pd.to_numeric(df["v_measure"], errors="coerce")
    best = df.sort_values("_v", ascending=False).iloc[0]
    return _community_row_from_sweep_series(best, source=str(sweep_csv))


def _community_row_from_sweep_series(best: pd.Series, *, source: str) -> dict[str, Any]:
    return {
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
        "source": source,
    }


def _community_row_from_best_json(data: dict[str, Any], *, source: Path) -> dict[str, Any]:
    row = data.get("best_row") if isinstance(data.get("best_row"), dict) else data
    return {
        "algorithm": str(row.get("method") or ""),
        "method": str(row.get("method") or ""),
        "threshold": float(row["min_edge_weight"]) if row.get("min_edge_weight") is not None else None,
        "min_edge_weight": float(row["min_edge_weight"]) if row.get("min_edge_weight") is not None else None,
        "resolution": float(row["resolution"]) if row.get("resolution") is not None else None,
        "homogeneity": float(row["homogeneity"]) if row.get("homogeneity") is not None else None,
        "completeness": float(row["completeness"]) if row.get("completeness") is not None else None,
        "v_measure": float(row["v_measure"]) if row.get("v_measure") is not None else None,
        "n_edges_after_threshold": float(row["n_edges_after_threshold"])
        if row.get("n_edges_after_threshold") is not None
        else None,
        "n_communities": float(row["n_communities"]) if row.get("n_communities") is not None else None,
        "source": str(source),
    }


def resolve_best_community_settings(
    repo: Path,
    manifest: dict[str, Any],
    *,
    gt_slug: str | None = None,
) -> dict[str, Any]:
    """
    Canonical best community hyperparameters for downstream steps (5, 9, 10).

    Priority:
      1. ``output/runs/<run_id>/community/anchor_community_best__<gt_slug>.json``
      2. Same under scoring run (if present)
      3. Full sweep CSV from step 4 scoring run
    """
    slug = gt_slug or str(manifest.get("gt_slug") or "ground_truth")
    run_id = str(manifest["run_id"])

    explicit = manifest.get("community_best_json")
    if explicit:
        p = resolve_repo_path(repo, str(explicit))
        if p.is_file():
            return _community_row_from_best_json(
                json.loads(p.read_text(encoding="utf-8-sig")), source=p
            )

    run_best = training_run_dir(repo, run_id) / "community" / f"anchor_community_best__{slug}.json"
    if run_best.is_file():
        return _community_row_from_best_json(json.loads(run_best.read_text(encoding="utf-8-sig")), source=run_best)

    scoring_best = (
        scoring_run_dir(repo, str(manifest["scoring_run_id"]))
        / "seed_candidate"
        / "community"
        / f"anchor_community_best__{slug}.json"
    )
    if scoring_best.is_file():
        return _community_row_from_best_json(json.loads(scoring_best.read_text(encoding="utf-8-sig")), source=scoring_best)

    sweep = community_sweep_csv(repo, str(manifest["scoring_run_id"]), gt_slug=slug)
    if sweep.is_file():
        return best_community_row(sweep)

    raise FileNotFoundError(
        "Could not resolve best community settings. Run step04 or ensure "
        f"{run_best} exists."
    )


def format_latex_community_table(rows: list[dict[str, Any]], *, caption: str, label: str) -> str:
    lines = [
        "% Requires \\usepackage{booktabs}",
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\small",
        r"\begin{tabular}{l l r r r r r}",
        r"\toprule",
        r"Run & Algorithm & Threshold & Resolution & $H$ & $C$ & $V$ \\",
        r"\midrule",
    ]
    for row in rows:
        label_s = str(row.get("label") or "---")
        algo = str(row.get("algorithm") or row.get("method") or "---")
        thr = row.get("threshold")
        thr_s = f"{float(thr):.1f}" if thr is not None else "---"
        res = row.get("resolution")
        res_s = f"{float(res):.1f}" if res is not None else "---"

        def _f(k: str) -> str:
            v = row.get(k)
            return f"{float(v):.3f}" if v is not None else "---"

        lines.append(
            f"{label_s} & {algo} & {thr_s} & {res_s} & {_f('homogeneity')} & {_f('completeness')} & {_f('v_measure')} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    return "\n".join(lines)


def copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.is_file():
        return False
    src_resolved = src.resolve()
    dst_resolved = dst.resolve()
    if src_resolved == dst_resolved:
        return True
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_resolved, dst_resolved)
    return True
