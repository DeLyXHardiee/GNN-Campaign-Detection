#!/usr/bin/env python3
"""
Timestamp ablation for _14_only_mlp: materialize MISP time_gap pair CSV, train, score+community.

Does not modify output/runs/main_gnn_pu_1_no_ts_dedup_task_identity_14_only_mlp.

  python seed_candidate_workflow/scripts/run_timestamp_ablation_14_only_mlp.py --phase all
  python seed_candidate_workflow/scripts/run_timestamp_ablation_14_only_mlp.py --phase materialize
  python seed_candidate_workflow/scripts/run_timestamp_ablation_14_only_mlp.py --phase train
  python seed_candidate_workflow/scripts/run_timestamp_ablation_14_only_mlp.py --phase community
  python seed_candidate_workflow/scripts/run_timestamp_ablation_14_only_mlp.py --phase consolidate
"""
from __future__ import annotations

import argparse
import json
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

from seed_candidate_workflow.utils.timestamp_ablation_14_only_mlp import (  # noqa: E402
    build_14_only_mlp_training_cfg,
    community_sweep_csv,
    load_manifest,
    training_run_dir,
)


def _materialize(manifest: dict[str, Any]) -> Path:
    ts = manifest["timestamp_enabled"]
    out_csv = (_REPO / str(ts["pair_dataset_csv"])).resolve()
    if out_csv.is_file():
        print(f"[materialize] exists: {out_csv}")
        return out_csv

    cmd = [
        sys.executable,
        str(_REPO / "seed_candidate_workflow/scripts/materialize_timestamp_pair_training_dataset.py"),
        "--source-pair-csv",
        str((_REPO / str(manifest["baseline"]["pair_dataset_csv"])).resolve()),
        "--output-pair-csv",
        str(out_csv),
    ]
    print("[materialize] writing timestamp pair CSV")
    subprocess.run(cmd, cwd=str(_REPO), check=True)
    if not out_csv.is_file():
        raise FileNotFoundError(f"Materialize failed: {out_csv}")
    return out_csv


def _train(manifest: dict[str, Any], *, skip_existing: bool) -> Path:
    from steps.train_stage import run_train_stage

    ts = manifest["timestamp_enabled"]
    run_id = str(ts["run_id"])
    run_dir = training_run_dir(_REPO, run_id)
    ckpt = run_dir / "mlp" / "models" / "best_model.pt"
    if skip_existing and ckpt.is_file():
        print(f"[train] skip (checkpoint exists): {ckpt}")
        return ckpt

    baseline_run_id = str(manifest["baseline"]["run_id"])
    ref_cfg = training_run_dir(_REPO, baseline_run_id) / "mlp" / "training_config.json"
    if not ref_cfg.is_file():
        raise FileNotFoundError(f"Baseline training_config.json not found: {ref_cfg}")

    pair_csv = (_REPO / str(ts["pair_dataset_csv"])).resolve()
    if not pair_csv.is_file():
        raise FileNotFoundError(f"Pair CSV missing; run --phase materialize first: {pair_csv}")

    graph_pt = (_REPO / str(ts["graph_pt"])).resolve()
    pi = float(manifest.get("nnpu_pi") or 0.1)
    training_cfg = build_14_only_mlp_training_cfg(
        pair_dataset_csv=str(pair_csv),
        reference_training_config=ref_cfg,
        project_root=_REPO,
        pi=pi,
    )
    print(f"[train] run_id={run_id} pi={pi}")
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
    ts = manifest["timestamp_enabled"]
    scoring_run_id = str(ts["scoring_run_id"])
    gt_slug = str(manifest.get("gt_slug") or "ground_truth")
    sweep = community_sweep_csv(_REPO, scoring_run_id, gt_slug=gt_slug)
    if skip_existing and sweep.is_file():
        print(f"[community] skip (sweep exists): {sweep}")
        return sweep

    exp_path = (_REPO / str(ts["experiment_config"])).resolve()
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
        choices=("materialize", "train", "community", "all", "consolidate"),
        default="all",
    )
    p.add_argument("--skip-existing", action="store_true")
    args = p.parse_args()

    manifest = load_manifest(args.manifest)

    if args.phase in ("materialize", "all"):
        _materialize(manifest)
    if args.phase in ("train", "all"):
        _train(manifest, skip_existing=bool(args.skip_existing))
    if args.phase in ("community", "all"):
        _community(manifest, skip_existing=bool(args.skip_existing))
    if args.phase in ("consolidate", "all"):
        cmd = [
            sys.executable,
            str(_REPO / "seed_candidate_workflow/scripts/consolidate_timestamp_ablation_14_only_mlp.py"),
        ]
        if args.manifest is not None:
            cmd.extend(["--manifest", str(Path(args.manifest).resolve())])
        subprocess.run(cmd, cwd=str(_REPO), check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
