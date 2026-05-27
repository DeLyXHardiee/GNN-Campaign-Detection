#!/usr/bin/env python3
"""Step 9: consolidate thesis GNN pair-scoring outputs into final_gnn_pair_scoring_timestamp_es_thesis/."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from seed_candidate_workflow.utils.final_14_only_mlp_thesis import load_manifest as load_mlp_manifest  # noqa: E402
from seed_candidate_workflow.utils.final_gnn_timestamp_es_thesis import (  # noqa: E402
    community_sweep_in_run_dir,
    copy_if_exists,
    load_manifest,
    read_training_stability,
    repo_root,
    resolve_best_community_from_sweep,
    resolve_repo_path,
    steps_dir,
    thesis_dir,
    training_run_dir,
)
from seed_candidate_workflow.utils.pair_score_thesis_diagnostics import (  # noqa: E402
    SLICE_ALL,
    SLICE_NON_SEED,
    SLICE_SEED,
)


def _load_diag(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _slice_row(diag: dict[str, Any], slice_id: str) -> dict[str, Any]:
    for sl in diag.get("slices", []) or []:
        if sl.get("slice_id") == slice_id:
            return sl
    raise KeyError(f"missing slice {slice_id}")


def _nonseed_summary(diag: dict[str, Any], label: str) -> dict[str, Any]:
    sl = _slice_row(diag, SLICE_NON_SEED)
    return {
        "method": label,
        "non_seed_auroc": sl.get("auroc_same_vs_cross"),
        "non_seed_ap": sl.get("average_precision_same_vs_cross"),
        "average_precision_class_imbalance_sensitive": True,
        "same_median": (sl.get("same_campaign") or {}).get("median"),
        "cross_median": (sl.get("cross_campaign") or {}).get("median"),
        "same_n": sl.get("n_same_campaign"),
        "cross_n": sl.get("n_cross_campaign"),
    }


def _try_combined_nonseed_kde(repo: Path, manifest: dict[str, Any], plots_dir: Path) -> str | None:
    try:
        from seed_candidate_workflow.scripts.gather_thesis_gnn_pair_scoring_results import (  # noqa: E402
            _kde_plot_same_vs_cross,
            _score_nonseed_pairs,
        )
    except Exception:
        return None

    graph_pt = resolve_repo_path(repo, str(manifest["graph_pt"]))
    pair_csv = resolve_repo_path(repo, str(manifest["final_pair_dataset_csv"]))
    gt_path = resolve_repo_path(repo, str(manifest["gt_path"]))
    mlp_manifest = load_mlp_manifest(resolve_repo_path(repo, str(manifest["final_mlp_manifest"])))
    runs = [
        ("Explicit-feature MLP", training_run_dir(repo, str(mlp_manifest["run_id"]))),
        ("GNN + explicit pair features", training_run_dir(repo, str(manifest["run_id_gnn_plus"]))),
        ("GNN-only scorer", training_run_dir(repo, str(manifest["run_id_gnn_only"]))),
    ]
    same_plus, cross_plus, _ = _score_nonseed_pairs(
        run_dir=runs[1][1], graph_pt=graph_pt, pair_csv=pair_csv, gt_path=gt_path
    )
    same_only, cross_only, _ = _score_nonseed_pairs(
        run_dir=runs[2][1], graph_pt=graph_pt, pair_csv=pair_csv, gt_path=gt_path
    )
    out = plots_dir / "kde_nonseed_combined_gnn_variants.png"
    import matplotlib.pyplot as plt

    try:
        import seaborn as sns
    except Exception:
        return None
    plt.figure(figsize=(8, 4.5), dpi=150)
    sns.kdeplot(same_plus, label=f"GNN+features same (n={len(same_plus):,})", bw_adjust=1.1)
    sns.kdeplot(cross_plus, label=f"GNN+features cross (n={len(cross_plus):,})", bw_adjust=1.1)
    sns.kdeplot(same_only, label=f"GNN-only same (n={len(same_only):,})", linestyle="--", bw_adjust=1.1)
    sns.kdeplot(cross_only, label=f"GNN-only cross (n={len(cross_only):,})", linestyle="--", bw_adjust=1.1)
    plt.title("Non-seed candidates: thesis GNN variants")
    plt.xlabel("Pair score (pu_score)")
    plt.ylabel("Density")
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(out)
    plt.close()
    return str(out)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, default=None)
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--skip-combined-kde", action="store_true")
    args = p.parse_args()

    repo = repo_root()
    manifest = load_manifest(args.manifest)
    tdir = thesis_dir(repo, manifest)
    paths_manifest = tdir / "paths_manifest.json"
    if args.skip_existing and paths_manifest.is_file():
        print(f"[step09] skip (thesis bundle exists): {paths_manifest}")
        print(paths_manifest.read_text(encoding="utf-8"))
        return 0
    gt_slug = str(manifest.get("gt_slug") or "ground_truth")

    comm_dir = tdir / "community"
    diag_dir = tdir / "score_diagnostics"
    plots_dir = tdir / "plots"
    tables_dir = tdir / "comparison_tables"
    train_dir = tdir / "training"
    for d in (comm_dir, diag_dir, plots_dir, tables_dir, train_dir):
        d.mkdir(parents=True, exist_ok=True)

    run_plus = training_run_dir(repo, str(manifest["run_id_gnn_plus"]))
    run_only = training_run_dir(repo, str(manifest["run_id_gnn_only"]))
    run_mlp = training_run_dir(repo, str(manifest["final_mlp_run_id"]))

    sweep_plus = community_sweep_in_run_dir(run_plus, gt_slug=gt_slug)
    sweep_only = community_sweep_in_run_dir(run_only, gt_slug=gt_slug)
    sweep_mlp = run_mlp / "community" / f"anchor_community_sweep__{gt_slug}.csv"

    best_plus = resolve_best_community_from_sweep(sweep_plus)
    best_only = resolve_best_community_from_sweep(sweep_only)
    best_mlp = resolve_best_community_from_sweep(sweep_mlp) if sweep_mlp.is_file() else {}

    copy_if_exists(sweep_plus, comm_dir / "gnn_plus_anchor_community_sweep.csv")
    copy_if_exists(sweep_only, comm_dir / "gnn_only_anchor_community_sweep.csv")
    (comm_dir / "best_configs.json").write_text(
        json.dumps({"gnn_plus": best_plus, "gnn_only": best_only, "final_mlp": best_mlp}, indent=2),
        encoding="utf-8",
    )

    diag_plus_path = run_plus / "pair_score_separation/thesis_score_diagnostics/thesis_pair_score_diagnostics.json"
    diag_only_path = run_only / "pair_score_separation/thesis_score_diagnostics/thesis_pair_score_diagnostics.json"
    diag_mlp_path = run_mlp / "pair_score_separation/thesis_score_diagnostics/thesis_pair_score_diagnostics.json"

    copy_if_exists(diag_plus_path, diag_dir / "gnn_plus_thesis_pair_score_diagnostics.json")
    copy_if_exists(diag_only_path, diag_dir / "gnn_only_thesis_pair_score_diagnostics.json")
    copy_if_exists(diag_mlp_path, diag_dir / "final_mlp_thesis_pair_score_diagnostics.json")

    diag_plus = _load_diag(diag_plus_path) if diag_plus_path.is_file() else {"slices": []}
    diag_only = _load_diag(diag_only_path) if diag_only_path.is_file() else {"slices": []}
    diag_mlp = _load_diag(diag_mlp_path) if diag_mlp_path.is_file() else {"slices": []}

    ranking_rows = [
        _nonseed_summary(diag_mlp, "Explicit-feature MLP scorer"),
        _nonseed_summary(diag_plus, "GNN + explicit pair features"),
        _nonseed_summary(diag_only, "GNN-only scorer"),
    ]
    pd.DataFrame(ranking_rows).to_csv(tables_dir / "non_seed_ranking_comparison.csv", index=False)
    (tables_dir / "non_seed_ranking_comparison.json").write_text(json.dumps(ranking_rows, indent=2), encoding="utf-8")

    main_rows = [
        {
            "method": "Unscored seed/candidate graph",
            "uses_gnn": False,
            "uses_explicit_pair_features": False,
            "algorithm": "",
            "threshold": "",
            "resolution": "",
            "homogeneity": 0.90,
            "completeness": 0.91,
            "v_measure": 0.90,
            "communities": "",
        },
        {
            "method": "Explicit-feature MLP scorer",
            "uses_gnn": False,
            "uses_explicit_pair_features": True,
            "algorithm": best_mlp.get("algorithm"),
            "threshold": best_mlp.get("threshold"),
            "resolution": best_mlp.get("resolution"),
            "homogeneity": best_mlp.get("homogeneity"),
            "completeness": best_mlp.get("completeness"),
            "v_measure": best_mlp.get("v_measure"),
            "communities": best_mlp.get("n_communities"),
        },
        {
            "method": "GNN + explicit pair features",
            "uses_gnn": True,
            "uses_explicit_pair_features": True,
            "algorithm": best_plus.get("algorithm"),
            "threshold": best_plus.get("threshold"),
            "resolution": best_plus.get("resolution"),
            "homogeneity": best_plus.get("homogeneity"),
            "completeness": best_plus.get("completeness"),
            "v_measure": best_plus.get("v_measure"),
            "communities": best_plus.get("n_communities"),
        },
        {
            "method": "GNN-only scorer",
            "uses_gnn": True,
            "uses_explicit_pair_features": False,
            "algorithm": best_only.get("algorithm"),
            "threshold": best_only.get("threshold"),
            "resolution": best_only.get("resolution"),
            "homogeneity": best_only.get("homogeneity"),
            "completeness": best_only.get("completeness"),
            "v_measure": best_only.get("v_measure"),
            "communities": best_only.get("n_communities"),
        },
    ]
    pd.DataFrame(main_rows).to_csv(tables_dir / "main_comparison.csv", index=False)
    (tables_dir / "main_comparison.json").write_text(json.dumps(main_rows, indent=2), encoding="utf-8")

    gnn_best_rows = [
        {"variant": "GNN + explicit pair features", **best_plus},
        {"variant": "GNN-only scorer", **best_only},
    ]
    pd.DataFrame(gnn_best_rows).to_csv(tables_dir / "gnn_best_configs.csv", index=False)
    (tables_dir / "gnn_best_configs.json").write_text(json.dumps(gnn_best_rows, indent=2), encoding="utf-8")

    # Copy KDE / loss plots
    for run_id, prefix in (
        (str(manifest["run_id_gnn_plus"]), "gnn_plus"),
        (str(manifest["run_id_gnn_only"]), "gnn_only"),
    ):
        sep_plots = training_run_dir(repo, run_id) / "pair_score_separation" / "plots"
        if sep_plots.is_dir():
            for png in sep_plots.glob("*.png"):
                copy_if_exists(png, plots_dir / f"{prefix}__{png.name}")
        copy_if_exists(train_dir / f"{run_id}__loss_over_epochs.png", plots_dir / f"{prefix}__loss.png")

    combined_kde = None
    if not args.skip_combined_kde:
        combined_kde = _try_combined_nonseed_kde(repo, manifest, plots_dir)

    stab_plus = read_training_stability(run_plus, target_epochs=100)
    stab_only = read_training_stability(run_only, target_epochs=100)

    old_13 = training_run_dir(repo, str(manifest["reference_gnn_plus_run_id"]))
    old_15 = training_run_dir(repo, str(manifest["reference_gnn_only_run_id"]))
    old_plus_stab = read_training_stability(old_13, target_epochs=30) if (old_13 / "gnn/metrics.csv").is_file() else {}
    old_only_stab = read_training_stability(old_15, target_epochs=30) if (old_15 / "gnn/metrics.csv").is_file() else {}

    summary_md = tdir / "gnn_pair_scoring_summary.md"
    lines = [
        "# Thesis GNN pair scoring (timestamp heterograph + ES100)",
        "",
        "## Runs",
        f"- GNN + explicit pair features: `{manifest['run_id_gnn_plus']}`",
        f"- GNN-only scorer: `{manifest['run_id_gnn_only']}`",
        f"- Heterograph: `{manifest['graph_pt']}`",
        f"- Pair universe: `{manifest['final_pair_dataset_csv']}`",
        "",
        "## Training (best validation nnPU loss)",
        f"- GNN+features best epoch: {stab_plus.get('best_epoch')} (val_loss={stab_plus.get('best_val_loss')})",
        f"- GNN-only best epoch: {stab_only.get('best_epoch')} (val_loss={stab_only.get('best_val_loss')})",
        "",
        "## Community (expanded GT, best V)",
        f"- GNN+features: {best_plus}",
        f"- GNN-only: {best_only}",
        f"- Final MLP (reference): {best_mlp}",
        "",
        "## Legacy 30-epoch no-ts reference",
        f"- Old GNN+features (_13): best epoch {old_plus_stab.get('best_epoch')}, V from prior sweep ~0.913",
        f"- Old GNN-only (_15): best epoch {old_only_stab.get('best_epoch')}",
        "",
        "## Notes",
        "- Diagnostics only; thesis interpretation is written separately.",
        "- AP is class-imbalance sensitive.",
        "",
    ]
    if combined_kde:
        lines.append(f"- Combined GNN non-seed KDE: `{combined_kde}`")
    summary_md.write_text("\n".join(lines), encoding="utf-8")

    paths = {
        "thesis_output_dir": str(tdir),
        "graph_timestamp_summary": str(tdir / "graph_timestamp_summary.json"),
        "gnn_pair_scoring_summary": str(summary_md),
        "main_comparison_csv": str(tables_dir / "main_comparison.csv"),
        "non_seed_ranking_csv": str(tables_dir / "non_seed_ranking_comparison.csv"),
        "community_dir": str(comm_dir),
        "score_diagnostics_dir": str(diag_dir),
        "plots_dir": str(plots_dir),
    }
    (tdir / "paths_manifest.json").write_text(json.dumps(paths, indent=2), encoding="utf-8")

    report = {
        "paths": paths,
        "training_stability": {"gnn_plus": stab_plus, "gnn_only": stab_only},
        "best_community": {"gnn_plus": best_plus, "gnn_only": best_only, "final_mlp": best_mlp},
    }
    (steps_dir(repo, manifest) / "step09_consolidate_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
