#!/usr/bin/env python3
"""Step 6: prior sensitivity (pi in 0.05, 0.10, 0.20, 0.30) on final timestamp+ES setup."""
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

from seed_candidate_workflow.utils.final_14_only_mlp_thesis import (  # noqa: E402
    build_final_training_cfg,
    community_sweep_csv,
    load_manifest,
    repo_root,
    steps_dir,
    training_run_dir,
)


def _load_prior_manifest(repo: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    ps = manifest.get("prior_sensitivity") or {}
    path = repo / str(ps.get("manifest") or "seed_candidate_workflow/configs/final_14_only_mlp/final_14_only_mlp.prior_sensitivity.manifest.json")
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _train_one(repo: Path, entry: dict[str, Any], ps_manifest: dict[str, Any], *, skip_existing: bool) -> None:
    from steps.train_stage import run_train_stage

    run_id = str(entry["run_id"])
    ckpt = training_run_dir(repo, run_id) / "mlp" / "models" / "best_model.pt"
    if skip_existing and ckpt.is_file():
        print(f"[step06 train] skip pi={entry['pi']}: {ckpt}")
        return

    ref_run = str(ps_manifest.get("reference_training_config_run_id") or manifest_run_id_fallback(repo))
    ref_cfg = training_run_dir(repo, ref_run) / "mlp" / "training_config.json"
    tr = ps_manifest.get("training") or {}
    training_cfg = build_final_training_cfg(
        pair_dataset_csv=str((repo / str(ps_manifest["pair_dataset_csv"])).resolve()),
        reference_training_config=ref_cfg,
        project_root=repo,
        pi=float(entry["pi"]),
        epochs=int(tr.get("epochs") or 100),
        early_stopping_patience=int(tr.get("early_stopping_patience") or 10),
        save_best_val_checkpoint_history=bool(tr.get("save_best_val_checkpoint_history", False)),
    )
    run_train_stage(
        graph_path=str((repo / str(ps_manifest["graph_pt"])).resolve()),
        runs_parent=str((repo / "output/runs").resolve()),
        run_id=run_id,
        training_cfg=training_cfg,
        device_pref="cpu",
        to_undirected=True,
        pair_training_overrides={"pair_training_backends_override": ["mlp"]},
    )


def manifest_run_id_fallback(repo: Path) -> str:
    m = json.loads(
        (repo / "seed_candidate_workflow/configs/final_14_only_mlp/final_14_only_mlp.manifest.json").read_text(
            encoding="utf-8-sig"
        )
    )
    return str(m["run_id"])


def _community_one(repo: Path, entry: dict[str, Any], gt_slug: str, *, skip_existing: bool) -> None:
    sweep = community_sweep_csv(repo, str(entry["scoring_run_id"]), gt_slug=gt_slug)
    if skip_existing and sweep.is_file():
        print(f"[step06 community] skip pi={entry['pi']}")
        return
    exp = (repo / str(entry["experiment_config"])).resolve()
    subprocess.run(
        [sys.executable, str(repo / "seed_candidate_workflow/pipelines/run_experiment.py"), "--config", str(exp)],
        cwd=str(repo),
        check=True,
    )


def _consolidate_prior(repo: Path, ps_manifest: dict[str, Any], out_dir: Path, gt_slug: str) -> None:
    import pandas as pd

    from seed_candidate_workflow.utils.early_stopping_sanity_14_only_mlp import format_latex_comparison_table

    rows: list[dict[str, Any]] = []
    for entry in ps_manifest.get("priors") or []:
        sweep = community_sweep_csv(repo, str(entry["scoring_run_id"]), gt_slug=gt_slug)
        df = pd.read_csv(sweep, low_memory=False)
        df["_v"] = pd.to_numeric(df["v_measure"], errors="coerce")
        best = df.sort_values("_v", ascending=False).iloc[0]
        rows.append(
            {
                "pi": float(entry["pi"]),
                "run_id": str(entry["run_id"]),
                "scoring_run_id": str(entry["scoring_run_id"]),
                "algorithm": str(best.get("method") or ""),
                "threshold": float(best["min_edge_weight"]) if pd.notna(best.get("min_edge_weight")) else None,
                "resolution": float(best["resolution"]) if pd.notna(best.get("resolution")) else None,
                "homogeneity": float(best["homogeneity"]) if pd.notna(best.get("homogeneity")) else None,
                "completeness": float(best["completeness"]) if pd.notna(best.get("completeness")) else None,
                "v_measure": float(best["v_measure"]) if pd.notna(best.get("v_measure")) else None,
                "sweep_csv": str(sweep),
            }
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_dir / "prior_sensitivity_best_by_pi.csv", index=False)
    (out_dir / "prior_sensitivity_best_by_pi.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    tex_lines = [
        "% Requires \\usepackage{booktabs}",
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Prior sensitivity (final timestamp + early stopping MLP).}",
        r"\label{tab:final-prior-sensitivity}",
        r"\small",
        r"\begin{tabular}{l l r r r r r r}",
        r"\toprule",
        r"$\pi$ & Algorithm & Threshold & Resolution & $H$ & $C$ & $V$ \\",
        r"\midrule",
    ]
    for r in rows:
        tex_lines.append(
            f"{r['pi']:.2f} & {r['algorithm']} & {r['threshold']:.1f} & {r['resolution']:.1f} & "
            f"{r['homogeneity']:.3f} & {r['completeness']:.3f} & {r['v_measure']:.3f} \\\\"
        )
    tex_lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    (out_dir / "prior_sensitivity_best_by_pi.tex").write_text("\n".join(tex_lines), encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, default=None)
    p.add_argument("--phase", choices=("train", "community", "all", "consolidate"), default="all")
    p.add_argument("--pi", type=float, default=None)
    p.add_argument("--skip-existing", action="store_true")
    args = p.parse_args()

    repo = repo_root()
    manifest = load_manifest(args.manifest)
    ps_manifest = _load_prior_manifest(repo, manifest)
    gt_slug = str(manifest.get("gt_slug") or "ground_truth")
    entries = [dict(e) for e in ps_manifest.get("priors") or []]
    if args.pi is not None:
        entries = [e for e in entries if abs(float(e["pi"]) - float(args.pi)) < 1e-9]

    if args.phase in ("train", "all"):
        for e in entries:
            _train_one(repo, e, ps_manifest, skip_existing=bool(args.skip_existing))
    if args.phase in ("community", "all"):
        for e in entries:
            _community_one(repo, e, gt_slug, skip_existing=bool(args.skip_existing))
    if args.phase in ("consolidate", "all"):
        out = repo / str((manifest.get("prior_sensitivity") or {}).get("consolidation_output_dir") or "")
        _consolidate_prior(repo, ps_manifest, out.resolve(), gt_slug)
        print(f"[step06] consolidated -> {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
