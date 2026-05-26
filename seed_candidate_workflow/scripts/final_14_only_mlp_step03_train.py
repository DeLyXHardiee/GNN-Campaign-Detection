#!/usr/bin/env python3
"""Step 3: train final MLP (epochs=100, ES patience=10, best-val checkpoint history)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_GNN = _REPO / "core" / "GNN"
_CORE = _REPO / "core"
for p in (_GNN, _CORE, _REPO):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from seed_candidate_workflow.utils.final_14_only_mlp_thesis import (  # noqa: E402
    build_final_training_cfg,
    load_manifest,
    read_training_stability,
    repo_root,
    resolve_repo_path,
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
    run_dir = training_run_dir(repo, run_id)
    ckpt = run_dir / "mlp" / "models" / "best_model.pt"

    if args.skip_existing and ckpt.is_file():
        print(f"[step03] skip (checkpoint exists): {ckpt}")
    else:
        from steps.train_stage import run_train_stage

        ref_cfg = training_run_dir(repo, str(manifest["baseline_run_id"])) / "mlp" / "training_config.json"
        if not ref_cfg.is_file():
            raise FileNotFoundError(f"Baseline training_config missing: {ref_cfg}")

        pair_csv = resolve_repo_path(repo, str(manifest["final_pair_dataset_csv"]))
        tr = manifest.get("training") or {}
        training_cfg = build_final_training_cfg(
            pair_dataset_csv=str(pair_csv),
            reference_training_config=ref_cfg,
            project_root=repo,
            pi=float(manifest.get("nnpu_pi") or 0.1),
            epochs=int(tr.get("epochs") or 100),
            early_stopping_patience=int(tr.get("early_stopping_patience") or 10),
            save_best_val_checkpoint_history=bool(tr.get("save_best_val_checkpoint_history", True)),
        )
        print(f"[step03] training run_id={run_id}")
        run_train_stage(
            graph_path=str(resolve_repo_path(repo, str(manifest["graph_pt"]))),
            runs_parent=str((repo / "output/runs").resolve()),
            run_id=run_id,
            training_cfg=training_cfg,
            device_pref="cpu",
            to_undirected=True,
            pair_training_overrides={"pair_training_backends_override": ["mlp"]},
        )

    stability = read_training_stability(run_dir, target_epochs=int((manifest.get("training") or {}).get("epochs") or 100))
    report = {
        "run_dir": str(run_dir),
        "checkpoint": str(ckpt),
        "metrics_csv": str(run_dir / "mlp" / "metrics.csv"),
        "best_val_epochs_dir": str(run_dir / "mlp" / "models" / "best_val_epochs"),
        "training_stability": stability,
    }
    out_dir = steps_dir(repo, manifest)
    p_json = out_dir / "step03_train_report.json"
    p_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
