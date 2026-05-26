#!/usr/bin/env python3
"""
Early-stopping sanity for _14_only_mlp: epochs=100, patience=10 on validation nnPU loss.

Does not modify output/runs/main_gnn_pu_1_no_ts_dedup_task_identity_14_only_mlp.

  python seed_candidate_workflow/scripts/run_early_stopping_sanity_14_only_mlp.py --phase all
  python seed_candidate_workflow/scripts/run_early_stopping_sanity_14_only_mlp.py --phase train
  python seed_candidate_workflow/scripts/run_early_stopping_sanity_14_only_mlp.py --phase community
  python seed_candidate_workflow/scripts/run_early_stopping_sanity_14_only_mlp.py --phase consolidate
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[2]
_GNN = _REPO / "core" / "GNN"
_CORE = _REPO / "core"
for p in (_GNN, _CORE, _REPO):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from seed_candidate_workflow.utils.early_stopping_sanity_14_only_mlp import (  # noqa: E402
    build_early_stopping_training_cfg,
    community_sweep_csv,
    load_manifest,
    training_run_dir,
)


def _train(manifest: dict[str, Any], *, skip_existing: bool) -> Path:
    from steps.train_stage import run_train_stage

    es = manifest["early_stopping"]
    shared = manifest["shared"]
    run_id = str(es["run_id"])
    run_dir = training_run_dir(_REPO, run_id)
    ckpt = run_dir / "mlp" / "models" / "best_model.pt"
    if skip_existing and ckpt.is_file():
        print(f"[train] skip (checkpoint exists): {ckpt}")
        return ckpt

    baseline_run_id = str(manifest["baseline"]["run_id"])
    ref_cfg = training_run_dir(_REPO, baseline_run_id) / "mlp" / "training_config.json"
    if not ref_cfg.is_file():
        raise FileNotFoundError(f"Baseline training_config.json not found: {ref_cfg}")

    pair_csv = (_REPO / str(shared["pair_dataset_csv"])).resolve()
    graph_pt = (_REPO / str(shared["graph_pt"])).resolve()
    pi = float(shared.get("nnpu_pi") or 0.1)
    training_cfg = build_early_stopping_training_cfg(
        pair_dataset_csv=str(pair_csv),
        reference_training_config=ref_cfg,
        project_root=_REPO,
        pi=pi,
        epochs=int(es.get("epochs") or 100),
        early_stopping_patience=int(es.get("early_stopping_patience") or 10),
    )
    print(
        f"[train] run_id={run_id} epochs={training_cfg['epochs']} "
        f"early_stopping_patience={training_cfg['early_stopping_patience']} pi={pi}"
    )
    run_train_stage(
        graph_path=str(graph_pt),
        runs_parent=str((_REPO / "output/runs").resolve()),
        run_id=run_id,
        training_cfg=training_cfg,
        device_pref="cpu",
        to_undirected=True,
        pair_training_overrides={"pair_training_backends_override": ["mlp"]},
    )
    if not ckpt.is_file():
        raise FileNotFoundError(f"Training finished but checkpoint missing: {ckpt}")
    return ckpt


def _community(manifest: dict[str, Any], *, skip_existing: bool) -> Path:
    es = manifest["early_stopping"]
    scoring_run_id = str(es["scoring_run_id"])
    gt_slug = str(manifest.get("gt_slug") or manifest["shared"].get("gt_slug") or "ground_truth")
    sweep = community_sweep_csv(_REPO, scoring_run_id, gt_slug=gt_slug)
    if skip_existing and sweep.is_file():
        print(f"[community] skip (sweep exists): {sweep}")
        return sweep

    exp_path = (_REPO / str(es["experiment_config"])).resolve()
    cmd = [
        sys.executable,
        str(_REPO / "seed_candidate_workflow/pipelines/run_experiment.py"),
        "--config",
        str(exp_path),
    ]
    print(f"[community] config={exp_path.name}")
    subprocess.run(cmd, cwd=str(_REPO), check=True)
    if not sweep.is_file():
        raise FileNotFoundError(f"Community sweep missing: {sweep}")
    return sweep


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, default=None)
    p.add_argument(
        "--phase",
        choices=("train", "community", "all", "consolidate"),
        default="all",
    )
    p.add_argument("--skip-existing", action="store_true")
    args = p.parse_args()

    manifest = load_manifest(args.manifest)

    if args.phase in ("train", "all"):
        _train(manifest, skip_existing=bool(args.skip_existing))
    if args.phase in ("community", "all"):
        _community(manifest, skip_existing=bool(args.skip_existing))
    if args.phase in ("consolidate", "all"):
        cmd = [
            sys.executable,
            str(_REPO / "seed_candidate_workflow/scripts/consolidate_early_stopping_sanity_14_only_mlp.py"),
        ]
        if args.manifest is not None:
            cmd.extend(["--manifest", str(Path(args.manifest).resolve())])
        subprocess.run(cmd, cwd=str(_REPO), check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
