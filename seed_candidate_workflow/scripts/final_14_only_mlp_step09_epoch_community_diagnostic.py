#!/usr/bin/env python3
"""
Step 9: post-hoc community H/C/V at fixed settings for each best-val checkpoint epoch.

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

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from seed_candidate_workflow.utils.final_14_only_mlp_thesis import (  # noqa: E402
    best_community_row,
    community_sweep_csv,
    load_manifest,
    repo_root,
    steps_dir,
    training_run_dir,
)


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

    step04 = steps_dir(repo, manifest) / "step04_community_report.json"
    if step04.is_file():
        fixed = json.loads(step04.read_text(encoding="utf-8-sig"))["best_community"]
    else:
        fixed = best_community_row(community_sweep_csv(repo, str(manifest["scoring_run_id"])))

    method = str(fixed.get("method") or "leiden").lower()
    resolution = float(fixed["resolution"])
    threshold = float(fixed["threshold"])
    gt_slug = str(manifest.get("gt_slug") or "ground_truth")

    rows: list[dict[str, Any]] = []
    out_dir = steps_dir(repo, manifest)
    diag_root = out_dir / "epoch_community_diagnostic_runs"
    diag_root.mkdir(parents=True, exist_ok=True)

    for ckpt in sorted(hist_dir.glob("epoch_*.pt")):
        epoch = int(ckpt.stem.split("_")[-1])
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

        if not (args.skip_existing and sweep_csv.is_file()):
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
                        "weight_thresholds": [threshold],
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

        sweep_df = __import__("pandas").read_csv(sweep_csv, low_memory=False)
        r0 = sweep_df.iloc[0]
        manifest_line = hist_dir / "best_val_epochs_manifest.jsonl"
        val_loss = None
        if manifest_line.is_file():
            for line in manifest_line.read_text(encoding="utf-8").splitlines():
                rec = json.loads(line)
                if int(rec.get("epoch", -1)) == epoch:
                    val_loss = rec.get("val_loss")
        rows.append(
            {
                "epoch": epoch,
                "checkpoint": str(ckpt),
                "val_loss_at_save": val_loss,
                "fixed_method": method,
                "fixed_threshold": threshold,
                "fixed_resolution": resolution,
                "homogeneity": float(r0["homogeneity"]),
                "completeness": float(r0["completeness"]),
                "v_measure": float(r0["v_measure"]),
                "n_edges_after_threshold": float(r0["n_edges_after_threshold"]),
                "n_communities": float(r0["n_communities"]),
                "diagnostic_only_not_for_model_selection": True,
            }
        )

    import pandas as pd

    df = pd.DataFrame(rows).sort_values("epoch")
    p_csv = out_dir / "step09_epoch_community_diagnostic.csv"
    p_json = out_dir / "step09_epoch_community_diagnostic.json"
    df.to_csv(p_csv, index=False)
    report = {
        "disclaimer": "Post-hoc diagnostic only. Final model selected by validation nnPU (best_model.pt), not by these H/C/V values.",
        "fixed_community_setting_from_step04": fixed,
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
        r"\begin{tabular}{r r r r r r}",
        r"\toprule",
        r"Epoch & Val loss & $H$ & $C$ & $V$ & Communities \\",
        r"\midrule",
    ]
    for r in rows:
        tex.append(
            f"{int(r['epoch'])} & {float(r['val_loss_at_save'] or 0):.4f} & "
            f"{r['homogeneity']:.3f} & {r['completeness']:.3f} & {r['v_measure']:.3f} & {int(r['n_communities'])} \\\\"
        )
    tex.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    (out_dir / "step09_epoch_community_diagnostic.tex").write_text("\n".join(tex), encoding="utf-8")
    print(json.dumps({"csv": str(p_csv), "json": str(p_json)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
