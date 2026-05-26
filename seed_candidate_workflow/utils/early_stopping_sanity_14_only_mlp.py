"""Early-stopping sanity run for _14_only_mlp (epochs=100, patience=10 on val nnPU loss)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def load_manifest(manifest_path: Path | None = None) -> dict[str, Any]:
    repo = Path(__file__).resolve().parents[2]
    path = manifest_path or (
        repo
        / "seed_candidate_workflow/configs/early_stopping_sanity/early_stopping_sanity_14_only_mlp.manifest.json"
    )
    return json.loads(Path(path).resolve().read_text(encoding="utf-8-sig"))


def build_early_stopping_training_cfg(
    *,
    pair_dataset_csv: str,
    reference_training_config: Path,
    project_root: Path | None = None,
    pi: float = 0.1,
    epochs: int = 100,
    early_stopping_patience: int = 10,
) -> dict[str, Any]:
    """Same architecture/split/data as baseline _14_only_mlp; only epoch budget + ES patience change."""
    from seed_candidate_workflow.utils.prior_sensitivity_14_only_mlp import build_14_only_mlp_training_cfg

    cfg = build_14_only_mlp_training_cfg(
        pi=pi,
        pair_dataset_csv=pair_dataset_csv,
        reference_training_config=reference_training_config,
        project_root=project_root,
    )
    cfg["epochs"] = int(epochs)
    cfg["early_stopping_patience"] = int(early_stopping_patience)
    return cfg


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


def read_early_stopping_training_metrics(
    run_dir: Path,
    *,
    target_epochs: int | None = None,
) -> dict[str, Any]:
    """Best/final epoch, val loss, early-stop flag from mlp/metrics.csv (+ training_config epochs)."""
    run_dir = Path(run_dir).resolve()
    metrics_path = run_dir / "mlp" / "metrics.csv"
    cfg_path = run_dir / "mlp" / "training_config.json"
    out: dict[str, Any] = {"metrics_csv": str(metrics_path), "found": metrics_path.is_file()}

    if cfg_path.is_file():
        cfg = json.loads(cfg_path.read_text(encoding="utf-8-sig"))
        out["target_epochs"] = int(cfg.get("epochs") or target_epochs or 0)
        out["early_stopping_patience"] = int(cfg.get("early_stopping_patience") or 0)
    elif target_epochs is not None:
        out["target_epochs"] = int(target_epochs)

    if not metrics_path.is_file():
        return out

    df = pd.read_csv(metrics_path, low_memory=False)
    if df.empty or "epoch" not in df.columns:
        out["empty"] = True
        return out

    out["n_epochs_logged"] = int(len(df))
    out["final_epoch"] = int(pd.to_numeric(df["epoch"], errors="coerce").iloc[-1])

    if "train_loss" in df.columns:
        tr = pd.to_numeric(df["train_loss"], errors="coerce")
        out["final_train_loss"] = float(tr.iloc[-1]) if tr.notna().any() else None
        out["any_nan_train_loss"] = bool(tr.isna().any())
        out["max_train_loss"] = float(tr.max()) if tr.notna().any() else None

    if "val_loss" in df.columns:
        va = pd.to_numeric(df["val_loss"], errors="coerce")
        out["final_val_loss"] = float(va.iloc[-1]) if va.notna().any() else None
        out["any_nan_val_loss"] = bool(va.isna().any())
        out["max_val_loss"] = float(va.max()) if va.notna().any() else None
        out["best_val_loss"] = float(va.min()) if va.notna().any() else None
        best_idx = va.idxmin()
        out["best_epoch"] = int(pd.to_numeric(df.loc[best_idx, "epoch"], errors="coerce"))
        out["stable_val_loss"] = bool(
            va.notna().all() and (va.max() < 1e6) and not bool(np.isinf(va).any())
        )

    tgt = out.get("target_epochs")
    if tgt is not None and out.get("final_epoch") is not None:
        out["early_stopping_triggered"] = int(out["final_epoch"]) < int(tgt)
    return out


def format_latex_comparison_table(rows: list[dict[str, Any]], *, caption: str | None = None) -> str:
    cap = caption or (
        "Early-stopping sanity vs fixed 30-epoch \\texttt{14\\_only\\_mlp} "
        "(expanded ground truth; best community row by $V$)."
    )
    lines = [
        "% Requires \\usepackage{booktabs}",
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{{cap}}}",
        r"\label{tab:14-only-mlp-early-stopping-sanity}",
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
