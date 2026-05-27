#!/usr/bin/env python3
"""
Step 9: post-hoc community recovery across sampled best-validation checkpoints.

Uses only saved best-validation checkpoints, but subsamples them by epoch. For each
selected checkpoint, community method and resolution are fixed to the final selected
run's best values, while edge threshold is swept from 0.0 to 0.9. The best threshold
per selected checkpoint is reported.

NOT used for model selection (selection remains lowest validation nnPU on best_model.pt).
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from seed_candidate_workflow.utils.final_14_only_mlp_thesis import (  # noqa: E402
    community_sweep_csv,
    load_manifest,
    repo_root,
    resolve_best_community_settings,
    steps_dir,
    training_run_dir,
)


def _load_checkpoint_manifest(hist_dir: Path) -> dict[int, dict[str, Any]]:
    """Return metadata keyed by epoch for best-val checkpoint history."""
    out: dict[int, dict[str, Any]] = {}
    manifest_line = hist_dir / "best_val_epochs_manifest.jsonl"
    if not manifest_line.is_file():
        return out
    for line in manifest_line.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        try:
            epoch = int(rec.get("epoch"))
        except Exception:
            continue
        out[epoch] = rec
    return out


def _checkpoint_epoch(path: Path) -> int:
    return int(path.stem.split("_")[-1])


def _select_checkpoint_epochs(checkpoints: list[Path]) -> list[Path]:
    """
    Subsample saved best-validation checkpoints.

    Policy:
      - include all saved best checkpoints with epoch <= 10
      - after epoch 10, require >=5 epochs since previous selected checkpoint
      - after epoch 50, require >=10 epochs since previous selected checkpoint
      - always include the final selected best-validation checkpoint
    """
    ordered = sorted(checkpoints, key=_checkpoint_epoch)
    selected: list[Path] = []
    last_selected_epoch: int | None = None
    for ckpt in ordered:
        epoch = _checkpoint_epoch(ckpt)
        if epoch <= 10:
            selected.append(ckpt)
            last_selected_epoch = epoch
            continue
        min_gap = 10 if epoch > 50 else 5
        if last_selected_epoch is None or epoch - last_selected_epoch >= min_gap:
            selected.append(ckpt)
            last_selected_epoch = epoch

    if ordered and (not selected or selected[-1] != ordered[-1]):
        selected.append(ordered[-1])

    return selected


def _sweep_has_expected_thresholds(path: Path, expected: list[float]) -> bool:
    if not path.is_file():
        return False
    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception:
        return False
    if "min_edge_weight" not in df.columns:
        return False
    got = sorted(round(float(x), 1) for x in pd.to_numeric(df["min_edge_weight"], errors="coerce").dropna())
    return got == sorted(round(float(x), 1) for x in expected)


def _best_threshold_row(sweep_csv: Path) -> dict[str, Any]:
    df = pd.read_csv(sweep_csv, low_memory=False)
    df["_v"] = pd.to_numeric(df["v_measure"], errors="coerce")
    best = df.sort_values("_v", ascending=False).iloc[0]
    return {
        "best_threshold": float(best["min_edge_weight"]),
        "homogeneity": float(best["homogeneity"]),
        "completeness": float(best["completeness"]),
        "v_measure": float(best["v_measure"]),
        "n_edges_after_threshold": float(best["n_edges_after_threshold"]),
        "n_communities": float(best["n_communities"]),
    }


def _write_v_measure_plot(df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(
        pd.to_numeric(df["epoch"], errors="coerce"),
        pd.to_numeric(df["v_measure"], errors="coerce"),
        marker="o",
        linewidth=1.8,
        color="#1f77b4",
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("V-measure")
    ax.set_title("Epoch diagnostic (best-val checkpoints only)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, default=None)
    p.add_argument("--skip-existing", action="store_true")
    args = p.parse_args()

    repo = repo_root()
    manifest = load_manifest(args.manifest)
    run_id = str(manifest["run_id"])
    main_run = training_run_dir(repo, run_id)
    hist_dir = main_run / "mlp" / "models" / "best_val_epochs"
    if not hist_dir.is_dir():
        raise FileNotFoundError(
            f"No best_val_epochs directory at {hist_dir}. Re-run step03 with save_best_val_checkpoint_history=true."
        )
    all_ckpts = sorted(hist_dir.glob("epoch_*.pt"), key=_checkpoint_epoch)
    if not all_ckpts:
        raise FileNotFoundError(f"No epoch_*.pt checkpoints found in {hist_dir}")
    selected_ckpts = _select_checkpoint_epochs(all_ckpts)
    ckpt_meta = _load_checkpoint_manifest(hist_dir)

    gt_slug = str(manifest.get("gt_slug") or "ground_truth")
    fixed = resolve_best_community_settings(repo, manifest, gt_slug=gt_slug)
    print(
        f"[step09] fixed community from {fixed.get('source')}: "
        f"{fixed.get('method')} resolution={fixed.get('resolution')} "
        f"(sweep thresholds 0.0-0.9 per selected checkpoint)"
    )

    method = str(fixed.get("method") or fixed.get("algorithm") or "louvain").lower()
    resolution = float(fixed["resolution"])
    thresholds = [round(i / 10.0, 1) for i in range(0, 10)]

    rows: list[dict[str, Any]] = []
    out_dir = steps_dir(repo, manifest)
    diag_root = out_dir / "epoch_community_diagnostic_runs"
    diag_root.mkdir(parents=True, exist_ok=True)

    for ckpt in selected_ckpts:
        epoch = _checkpoint_epoch(ckpt)
        diag_run_id = f"{run_id}__epoch_diag__epoch_{epoch:03d}"
        diag_run = repo / "output/runs" / diag_run_id
        diag_models = diag_run / "mlp" / "models"
        diag_models.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ckpt, diag_models / "best_model.pt")
        # training_config for scorer metadata
        tc_src = main_run / "mlp" / "training_config.json"
        if tc_src.is_file():
            shutil.copy2(tc_src, diag_run / "mlp" / "training_config.json")

        scoring_run_id = f"final_14_only_mlp__epoch_diag__e{epoch:03d}__expanded_gt"
        sweep_csv = community_sweep_csv(repo, scoring_run_id, gt_slug=gt_slug)

        if not (args.skip_existing and _sweep_has_expected_thresholds(sweep_csv, thresholds)):
            exp = {
                "experiment": {
                    "scoring_run_id": scoring_run_id,
                    "graph_id": str(manifest["graph_id"]),
                    "mode": "score_only",
                    "graph_family": "seed_candidate",
                },
                "artifacts": {
                    "graph_bundle_root": "seed_candidate_workflow/output/graph_bundles",
                    "scoring_output_root": "seed_candidate_workflow/output/scoring_runs",
                },
                "selection": {
                    "score_targets": ["seed_candidate"],
                    "gt_set": "expanded_gt_only",
                    "gt_sets_path": "seed_candidate_workflow/configs/experiments/gt_sets.json",
                },
                "scoring": {
                    "score_mode": "seed_candidate_pu_v1",
                    "params": {
                        "pu": {
                            "pu_run": {
                                "run_dir": f"output/runs/{diag_run_id}",
                                "graph_pt": str(manifest["graph_pt"]),
                                "checkpoint": "best_model.pt",
                                "pair_dataset_csv": str(manifest["final_pair_dataset_csv"]),
                                "device": "cpu",
                                "no_to_undirected": False,
                            },
                            "seed_edge_weight": 1.0,
                            "weight_mode": "raw_score",
                            "export_non_seed_min_pu_score": 0.0,
                        }
                    },
                },
                "community": {
                    "base_config": "seed_candidate_workflow/configs/anchor_community.default.json",
                    "dedup_collapse_out_dir": "data/misp/misp_lake_dedup_task_identity",
                    "sweep": {
                        "methods": [method],
                        "weight_thresholds": thresholds,
                        "resolutions": [resolution],
                        "use_edge_weights_in_partitioning": True,
                        "sort_by": "v-measure",
                    },
                },
            }
            exp_path = diag_root / f"epoch_{epoch:03d}.experiment.json"
            exp_path.write_text(json.dumps(exp, indent=2), encoding="utf-8")
            subprocess.run(
                [
                    sys.executable,
                    str(repo / "seed_candidate_workflow/pipelines/run_experiment.py"),
                    "--config",
                    str(exp_path),
                ],
                cwd=str(repo),
                check=True,
            )

        best_row = _best_threshold_row(sweep_csv)
        val_loss = ckpt_meta.get(epoch, {}).get("val_loss")
        rows.append(
            {
                "epoch": epoch,
                "checkpoint": str(ckpt),
                "val_loss_at_save": val_loss,
                "fixed_method": method,
                "fixed_resolution": resolution,
                "thresholds_swept": "|".join(f"{x:.1f}" for x in thresholds),
                "best_threshold": best_row["best_threshold"],
                "homogeneity": best_row["homogeneity"],
                "completeness": best_row["completeness"],
                "v_measure": best_row["v_measure"],
                "n_edges_after_threshold": best_row["n_edges_after_threshold"],
                "n_communities": best_row["n_communities"],
                "sweep_csv": str(sweep_csv),
                "diagnostic_only_not_for_model_selection": True,
            }
        )

    df = pd.DataFrame(rows).sort_values("epoch")
    p_csv = out_dir / "step09_epoch_community_diagnostic.csv"
    p_json = out_dir / "step09_epoch_community_diagnostic.json"
    p_tex = out_dir / "step09_epoch_community_diagnostic.tex"
    p_png = out_dir / "step09_epoch_community_diagnostic_v_measure.png"
    df.to_csv(p_csv, index=False)
    _write_v_measure_plot(df, p_png)
    report = {
        "disclaimer": (
            "Post-hoc diagnostic only. Final model selected by validation nnPU "
            "(best_model.pt), not by these H/C/V values."
        ),
        "checkpoint_selection_policy": {
            "n_saved_best_validation_checkpoints": len(all_ckpts),
            "n_selected_checkpoints": len(selected_ckpts),
            "include_all_epochs_leq": 10,
            "min_gap_after_epoch_10": 5,
            "min_gap_after_epoch_50": 10,
            "always_include_final_saved_best_validation_checkpoint": True,
            "all_saved_epochs": [_checkpoint_epoch(p) for p in all_ckpts],
            "selected_epochs": [_checkpoint_epoch(p) for p in selected_ckpts],
        },
        "fixed_community_setting": {
            **fixed,
            "threshold_policy": "sweep 0.0 to 0.9 per selected checkpoint",
            "thresholds": thresholds,
        },
        "outputs": {
            "csv": str(p_csv),
            "json": str(p_json),
            "latex": str(p_tex),
            "v_measure_plot": str(p_png),
        },
        "rows": rows,
    }
    p_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    tex = [
        "% Requires \\usepackage{booktabs}",
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Epoch diagnostic at fixed community settings (not used for model selection).}",
        r"\label{tab:final-epoch-community-diagnostic}",
        r"\small",
        r"\begin{tabular}{r r r r r r r}",
        r"\toprule",
        r"Epoch & Val loss & Threshold & $H$ & $C$ & $V$ & Communities \\",
        r"\midrule",
    ]
    for r in rows:
        tex.append(
            f"{int(r['epoch'])} & {float(r['val_loss_at_save'] or 0):.4f} & "
            f"{r['best_threshold']:.1f} & {r['homogeneity']:.3f} & "
            f"{r['completeness']:.3f} & {r['v_measure']:.3f} & "
            f"{int(r['n_communities'])} \\\\"
        )
    tex.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    p_tex.write_text("\n".join(tex), encoding="utf-8")
    print(json.dumps(report["outputs"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
