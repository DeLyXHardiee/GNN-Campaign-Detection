#!/usr/bin/env python3
"""Step 3: train thesis GNN + explicit pair features (ES100)."""
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

from seed_candidate_workflow.utils.final_gnn_timestamp_es_thesis import (  # noqa: E402
    build_gnn_training_cfg,
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
    run_id = str(manifest["run_id_gnn_plus"])
    run_dir = training_run_dir(repo, run_id)
    ckpt = run_dir / "gnn" / "models" / "best_model.pt"

    if args.skip_existing and ckpt.is_file():
        print(f"[step03] skip (checkpoint exists): {ckpt}")
    else:
        from steps.train_stage import run_train_stage

        ref = resolve_repo_path(repo, str(manifest["reference_gnn_plus_training_config"]))
        pair_csv = resolve_repo_path(repo, str(manifest["final_pair_dataset_csv"]))
        tr = manifest.get("training") or {}
        training_cfg = build_gnn_training_cfg(
            pair_dataset_csv=str(pair_csv),
            reference_training_config=ref,
            gnn_only=False,
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
            pair_training_overrides={"pair_training_backends_override": ["gnn"]},
        )

    stability = read_training_stability(run_dir, target_epochs=int((manifest.get("training") or {}).get("epochs") or 100))
    report = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "checkpoint": str(ckpt),
        "metrics_csv": str(run_dir / "gnn" / "metrics.csv"),
        "training_stability": stability,
    }
    out = steps_dir(repo, manifest) / "step03_train_gnn_plus_report.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
