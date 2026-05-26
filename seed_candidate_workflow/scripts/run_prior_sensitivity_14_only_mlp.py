#!/usr/bin/env python3
"""
Train + score + community sweep for _14_only_mlp nnPU prior sensitivity (pi in {0.05, 0.10, 0.20, 0.30}).

Does not modify output/runs/main_gnn_pu_1_no_ts_dedup_task_identity_14_only_mlp.

  python seed_candidate_workflow/scripts/run_prior_sensitivity_14_only_mlp.py --phase all
  python seed_candidate_workflow/scripts/run_prior_sensitivity_14_only_mlp.py --phase train --pi 0.05
  python seed_candidate_workflow/scripts/run_prior_sensitivity_14_only_mlp.py --phase community
  python seed_candidate_workflow/scripts/run_prior_sensitivity_14_only_mlp.py --phase consolidate
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

from seed_candidate_workflow.utils.prior_sensitivity_14_only_mlp import (  # noqa: E402
    build_14_only_mlp_training_cfg,
    community_sweep_csv,
    load_manifest,
    prior_entries,
    training_run_dir,
)


def _filter_priors(entries: list[dict[str, Any]], pi_filter: float | None) -> list[dict[str, Any]]:
    if pi_filter is None:
        return entries
    return [e for e in entries if abs(float(e["pi"]) - float(pi_filter)) < 1e-9]


def _train_one(
    *,
    repo: Path,
    entry: dict[str, Any],
    manifest: dict[str, Any],
    graph_pt: Path,
    skip_existing: bool,
) -> Path:
    from steps.train_stage import run_train_stage

    run_id = str(entry["run_id"])
    pi = float(entry["pi"])
    run_dir = training_run_dir(repo, run_id)
    ckpt = run_dir / "mlp" / "models" / "best_model.pt"
    if skip_existing and ckpt.is_file():
        print(f"[train] skip pi={pi} (checkpoint exists): {ckpt}")
        return ckpt

    ref_cfg = repo / str(manifest["reference_baseline_run_dir"]) / "mlp" / "training_config.json"
    if not ref_cfg.is_file():
        raise FileNotFoundError(f"Baseline training_config.json not found: {ref_cfg}")

    pair_csv = repo / str(manifest["pair_dataset_csv"])
    training_cfg = build_14_only_mlp_training_cfg(
        pi=pi,
        pair_dataset_csv=str(pair_csv.resolve()),
        reference_training_config=ref_cfg,
        project_root=repo,
    )
    print(f"[train] pi={pi} -> {run_dir}")
    run_train_stage(
        graph_path=str(graph_pt.resolve()),
        runs_parent=str((repo / "output/runs").resolve()),
        run_id=run_id,
        training_cfg=training_cfg,
        device_pref="cpu",
        to_undirected=True,
        pair_training_overrides={"pair_training_backends_override": ["mlp"]},
    )
    if not ckpt.is_file():
        raise FileNotFoundError(f"Training finished but checkpoint missing: {ckpt}")
    return ckpt


def _community_one(*, repo: Path, entry: dict[str, Any], skip_existing: bool) -> Path:
    exp_rel = str(entry["experiment_config"])
    exp_path = (repo / exp_rel).resolve()
    if not exp_path.is_file():
        raise FileNotFoundError(f"Experiment config missing: {exp_path}")

    sweep = community_sweep_csv(repo, str(entry["scoring_run_id"]))
    if skip_existing and sweep.is_file():
        print(f"[community] skip pi={entry['pi']} (sweep exists): {sweep}")
        return sweep

    cmd = [
        sys.executable,
        str(repo / "seed_candidate_workflow/pipelines/run_experiment.py"),
        "--config",
        str(exp_path),
    ]
    print(f"[community] pi={entry['pi']} config={exp_path.name}")
    subprocess.run(cmd, cwd=str(repo), check=True)
    if not sweep.is_file():
        raise FileNotFoundError(f"Community sweep missing after run: {sweep}")
    return sweep


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, default=None)
    p.add_argument(
        "--phase",
        choices=("train", "community", "all", "consolidate"),
        default="all",
    )
    p.add_argument("--pi", type=float, default=None, help="Run only this prior value (e.g. 0.05).")
    p.add_argument("--skip-existing", action="store_true")
    args = p.parse_args()

    manifest = load_manifest(args.manifest)
    entries = _filter_priors(prior_entries(manifest), args.pi)
    graph_pt = _REPO / str(manifest["graph_pt"])

    if args.phase in ("train", "all"):
        for entry in entries:
            _train_one(
                repo=_REPO,
                entry=entry,
                manifest=manifest,
                graph_pt=graph_pt,
                skip_existing=bool(args.skip_existing),
            )

    if args.phase in ("community", "all"):
        for entry in entries:
            _community_one(repo=_REPO, entry=entry, skip_existing=bool(args.skip_existing))

    if args.phase in ("consolidate", "all"):
        cmd = [
            sys.executable,
            str(_REPO / "seed_candidate_workflow/scripts/consolidate_prior_sensitivity_14_only_mlp.py"),
        ]
        if args.manifest is not None:
            cmd.extend(["--manifest", str(Path(args.manifest).resolve())])
        subprocess.run(cmd, cwd=str(_REPO), check=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
