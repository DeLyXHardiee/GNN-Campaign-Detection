"""Timestamp ablation for _14_only_mlp explicit pair scorer (no-ts vs MISP time_gap)."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def load_manifest(manifest_path: Path | None = None) -> dict[str, Any]:
    repo = Path(__file__).resolve().parents[2]
    path = manifest_path or (
        repo / "seed_candidate_workflow/configs/timestamp_ablation/timestamp_ablation_14_only_mlp.manifest.json"
    )
    return json.loads(Path(path).resolve().read_text(encoding="utf-8-sig"))


def build_14_only_mlp_training_cfg(
    *,
    pair_dataset_csv: str,
    reference_training_config: Path,
    project_root: Path | None = None,
    pi: float = 0.1,
) -> dict[str, Any]:
    """Same architecture/split/optimizer as baseline _14_only_mlp; timestamp pair CSV only."""
    from seed_candidate_workflow.utils.prior_sensitivity_14_only_mlp import build_14_only_mlp_training_cfg

    return build_14_only_mlp_training_cfg(
        pi=pi,
        pair_dataset_csv=pair_dataset_csv,
        reference_training_config=reference_training_config,
        project_root=project_root,
    )


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


def read_training_stability(run_dir: Path) -> dict[str, Any]:
    """Final train/val loss and NaN flags from mlp/metrics.csv."""
    metrics_path = run_dir / "mlp" / "metrics.csv"
    out: dict[str, Any] = {"metrics_csv": str(metrics_path), "found": metrics_path.is_file()}
    if not metrics_path.is_file():
        return out

    df = pd.read_csv(metrics_path, low_memory=False)
    if df.empty:
        out["empty"] = True
        return out

    for col in ("train_loss", "val_loss"):
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce")
            out[f"final_{col}"] = float(s.iloc[-1]) if s.notna().any() else None
            out[f"any_nan_{col}"] = bool(s.isna().any())
            out[f"max_{col}"] = float(s.max()) if s.notna().any() else None

    out["n_epochs_logged"] = int(len(df))
    if "val_loss" in df.columns:
        v = pd.to_numeric(df["val_loss"], errors="coerce")
        out["best_val_loss"] = float(v.min()) if v.notna().any() else None
        out["stable_val_loss"] = bool(
            v.notna().all() and (v.max() < 1e6) and not bool(np.isinf(v).any())
        )
    return out


def pair_universe_stats(pair_csv: Path) -> dict[str, Any]:
    df = pd.read_csv(pair_csv, low_memory=False)
    seed = df["is_seed_pair"].fillna(False).astype(bool) if "is_seed_pair" in df.columns else pd.Series(False, index=df.index)
    cand = (
        df["is_candidate_pair"].fillna(False).astype(bool)
        if "is_candidate_pair" in df.columns
        else pd.Series(True, index=df.index)
    )
    non_seed_cand = cand & ~seed
    emails = set()
    if "email_i" in df.columns and "email_j" in df.columns:
        emails |= set(df["email_i"].astype(str))
        emails |= set(df["email_j"].astype(str))

    tg = pd.to_numeric(df.get("time_gap_seconds_min"), errors="coerce")
    stats: dict[str, Any] = {
        "pair_csv": str(pair_csv),
        "n_pairs": int(len(df)),
        "n_seed_positive_pairs": int(seed.sum()),
        "n_non_seed_candidate_pairs": int(non_seed_cand.sum()),
        "n_unique_emails_incident": int(len(emails)),
    }
    if tg.notna().any():
        stats["time_gap_seconds_min"] = {
            "non_null_count": int(tg.notna().sum()),
            "min": float(tg.min()),
            "max": float(tg.max()),
            "mean": float(tg.mean()),
            "p50": float(tg.median()),
            "p95": float(tg.quantile(0.95)),
        }
    return stats


def format_latex_comparison_table(rows: list[dict[str, Any]], *, caption: str | None = None) -> str:
    cap = caption or (
        "Timestamp ablation for \\texttt{14\\_only\\_mlp} explicit MLP "
        "(expanded ground truth; best community row by $V$)."
    )
    lines = [
        "% Requires \\usepackage{booktabs}",
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{{cap}}}",
        r"\label{tab:14-only-mlp-timestamp-ablation}",
        r"\small",
        r"\begin{tabular}{l l r r r r r r}",
        r"\toprule",
        r"Run & Algorithm & Threshold & Resolution & $H$ & $C$ & $V$ \\",
        r"\midrule",
    ]
    for row in rows:
        label = str(row.get("label") or row.get("run_id") or "---")
        algo = str(row.get("algorithm") or row.get("method") or "---")
        thr = row.get("threshold")
        thr_s = f"{float(thr):.1f}" if thr is not None else "---"
        res = row.get("resolution")
        res_s = f"{float(res):.1f}" if res is not None else "---"

        def _f(key: str) -> str:
            v = row.get(key)
            return f"{float(v):.3f}" if v is not None else "---"

        lines.append(
            f"{label} & {algo} & {thr_s} & {res_s} & {_f('homogeneity')} & {_f('completeness')} & {_f('v_measure')} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    return "\n".join(lines)
