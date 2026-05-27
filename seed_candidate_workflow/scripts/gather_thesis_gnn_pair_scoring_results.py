#!/usr/bin/env python3
"""
Gather thesis-facing metrics/plots for GNN pair scoring runs (no retraining).

Outputs a consolidated folder under:
  seed_candidate_workflow/output/thesis_gnn_pair_scoring_results/
"""

from __future__ import annotations

import json
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
if str(_REPO / "core") not in sys.path:
    sys.path.insert(0, str(_REPO / "core"))
if str(_REPO / "core" / "GNN") not in sys.path:
    sys.path.insert(0, str(_REPO / "core" / "GNN"))


from seed_candidate_workflow.utils.pair_score_thesis_diagnostics import (  # noqa: E402
    SLICE_ALL,
    SLICE_NON_SEED,
    SLICE_SEED,
    run_thesis_pair_score_diagnostics,
)
from seed_candidate_workflow.utils.raw_gnn_notebook import (  # noqa: E402
    load_ground_truth_structures,
)
from seed_candidate_workflow.utils.pair_model_inference import (  # noqa: E402
    load_pair_supervision_for_inference,
    resolve_pair_dataset_csv_path,
    score_pair_rows,
)
from src.pair_train import load_pair_training_dataframe  # noqa: E402


@dataclass(frozen=True)
class CommunityBestConfig:
    run_id: str
    sweep_csv: str
    algorithm: str
    threshold: float
    resolution: float
    homogeneity: float
    completeness: float
    v_measure: float
    n_communities: int
    retained_edges: int | None


def _read_csv_best_by_vmeasure(path: Path) -> dict[str, Any]:
    df = pd.read_csv(path, low_memory=False)
    if "v_measure" not in df.columns:
        raise ValueError(f"community sweep missing v_measure column: {path}")
    # Ensure numeric compare.
    df["v_measure"] = pd.to_numeric(df["v_measure"], errors="coerce")
    best = df.sort_values("v_measure", ascending=False).iloc[0].to_dict()
    return best


def _best_config_from_sweep(*, run_id: str, sweep_csv: Path) -> CommunityBestConfig:
    best = _read_csv_best_by_vmeasure(sweep_csv)
    retained_edges = None
    for k in ("n_edges_after_threshold", "retained_edges", "n_edges_retained"):
        if k in best and pd.notna(best[k]):
            retained_edges = int(float(best[k]))
            break
    return CommunityBestConfig(
        run_id=run_id,
        sweep_csv=str(sweep_csv),
        algorithm=str(best.get("method", "")),
        threshold=float(best.get("min_edge_weight", np.nan)),
        resolution=float(best.get("resolution", np.nan)),
        homogeneity=float(best.get("homogeneity", np.nan)),
        completeness=float(best.get("completeness", np.nan)),
        v_measure=float(best.get("v_measure", np.nan)),
        n_communities=int(float(best.get("n_communities", np.nan))),
        retained_edges=retained_edges,
    )


def _load_thesis_diag_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _slice_row(diag: dict[str, Any], slice_id: str) -> dict[str, Any]:
    for sl in diag.get("slices", []) or []:
        if sl.get("slice_id") == slice_id:
            return sl
    raise KeyError(f"slice_id not found: {slice_id} in {diag.get('run_dir')}")


def _extract_nonseed_summary(diag: dict[str, Any], scorer_label: str) -> dict[str, Any]:
    sl = _slice_row(diag, SLICE_NON_SEED)
    return {
        "scorer": scorer_label,
        "non_seed_auroc": sl.get("auroc_same_vs_cross"),
        "non_seed_ap": sl.get("average_precision_same_vs_cross"),
        "average_precision_class_imbalance_sensitive": True,
        "same_median": (sl.get("same_campaign") or {}).get("median"),
        "cross_median": (sl.get("cross_campaign") or {}).get("median"),
        "cross_n": sl.get("n_cross_campaign"),
        "same_n": sl.get("n_same_campaign"),
    }


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.is_file():
        return False
    _ensure_dir(dst.parent)
    shutil.copy2(src, dst)
    return True


def _score_nonseed_pairs(
    *,
    run_dir: Path,
    graph_pt: Path,
    pair_csv: Path,
    gt_path: Path,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """
    Score pairs using the run's checkpoint and return (same_scores, cross_scores) for
    non-seed candidate pairs covered by expanded GT.
    """
    df, _ = load_pair_training_dataframe(pair_csv)
    df_work = df.reset_index(drop=True).copy()

    label_map, _eid_to_row, _campaign_to_members = load_ground_truth_structures(gt_path)

    # Seed mask
    if "is_seed_pair" in df_work.columns:
        seed = df_work["is_seed_pair"].fillna(False).astype(bool).to_numpy()
    elif "from_seed" in df_work.columns:
        seed = df_work["from_seed"].fillna(False).astype(bool).to_numpy()
    elif "pair_status" in df_work.columns:
        seed = df_work["pair_status"].astype(str).str.lower().eq("positive").to_numpy()
    else:
        seed = np.zeros(len(df_work), dtype=bool)

    ei = df_work["email_i"].astype(str).to_numpy()
    ej = df_work["email_j"].astype(str).to_numpy()
    camp_i = np.array([label_map.get(str(x)) for x in ei], dtype=object)
    camp_j = np.array([label_map.get(str(x)) for x in ej], dtype=object)
    both = (camp_i != None) & (camp_j != None)  # noqa: E711
    same = both & (camp_i == camp_j)
    cross = both & (camp_i != camp_j)

    bundle = load_pair_supervision_for_inference(
        run_dir=run_dir,
        graph_pt=graph_pt,
        checkpoint_name="best_model.pt",
        device=device,
        to_undirected=True,
    )
    df_work["_row"] = np.arange(len(df_work), dtype=np.int64)
    scores = score_pair_rows(
        model=bundle["model"],
        pair_scorer=bundle["pair_scorer"],
        data_cpu=bundle["data_cpu"],
        df_work=df_work,
        device=bundle["device"],
        fanout=bundle["fanout"],
        pair_batch_size=bundle["pair_batch_size"],
        max_unique_emails=bundle["max_unique_emails"],
        pair_feature_columns=bundle.get("pair_feature_columns"),
    )
    scores = np.asarray(scores, dtype=float)
    finite = np.isfinite(scores)

    nonseed = ~seed
    mask = nonseed & both & finite
    same_s = scores[mask & same]
    cross_s = scores[mask & cross]

    meta = {
        "n_pairs_total": int(len(df_work)),
        "n_nonseed_gt_covered_scored": int(mask.sum()),
        "n_nonseed_same": int((mask & same).sum()),
        "n_nonseed_cross": int((mask & cross).sum()),
        "checkpoint_path": str(bundle["checkpoint_path"]),
    }
    return same_s, cross_s, meta


def _kde_plot_same_vs_cross(
    *,
    same_scores: np.ndarray,
    cross_scores: np.ndarray,
    title: str,
    out_png: Path,
) -> None:
    import matplotlib.pyplot as plt

    try:
        import seaborn as sns
    except Exception:
        sns = None

    _ensure_dir(out_png.parent)
    plt.figure(figsize=(7.2, 4.2), dpi=150)

    if sns is not None:
        sns.kdeplot(same_scores, label=f"Same (n={len(same_scores):,})", fill=False, bw_adjust=1.1)
        sns.kdeplot(cross_scores, label=f"Cross (n={len(cross_scores):,})", fill=False, bw_adjust=1.1)
    else:
        # Fallback: histogram density.
        plt.hist(same_scores, bins=60, density=True, alpha=0.3, label=f"Same (n={len(same_scores):,})")
        plt.hist(cross_scores, bins=60, density=True, alpha=0.3, label=f"Cross (n={len(cross_scores):,})")

    plt.title(title)
    plt.xlabel("Pair score (pu_score)")
    plt.ylabel("Density")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def main() -> int:
    output_root = _REPO / "seed_candidate_workflow" / "output" / "thesis_gnn_pair_scoring_results"
    plots_dir = _ensure_dir(output_root / "plots")
    tables_dir = _ensure_dir(output_root / "tables")
    diagnostics_dir = _ensure_dir(output_root / "pair_score_diagnostics")
    training_dir = _ensure_dir(output_root / "training")

    graph_pt = _REPO / "core/graph/output/main_gnn_pu_1_no_ts_dedup_task_identity_12_hetero.pt"
    gt_path = _REPO / "data/groundtruth/ground_truth.json"

    run_13 = _REPO / "output/runs/main_gnn_pu_1_no_ts_dedup_task_identity_13"
    run_15 = _REPO / "output/runs/main_gnn_pu_1_no_ts_dedup_task_identity_15_gnn_only_scorer"
    run_mlp = _REPO / "output/runs/final_14_only_mlp__timestamp_feature__early_stopping"

    # 1) Community sweeps (expanded GT)
    best_13 = _best_config_from_sweep(
        run_id="main_gnn_pu_1_no_ts_dedup_task_identity_13",
        sweep_csv=run_13 / "community/anchor_community_sweep__ground_truth.csv",
    )
    best_15 = _best_config_from_sweep(
        run_id="main_gnn_pu_1_no_ts_dedup_task_identity_15_gnn_only_scorer",
        sweep_csv=run_15 / "community/anchor_community_sweep__ground_truth.csv",
    )
    best_mlp = _best_config_from_sweep(
        run_id="final_14_only_mlp__timestamp_feature__early_stopping",
        sweep_csv=run_mlp / "community/anchor_community_sweep__ground_truth.csv",
    )

    # 2) Pair score thesis diagnostics
    # MLP already computed in-run; reuse it.
    mlp_diag_path = (
        run_mlp
        / "pair_score_separation/thesis_score_diagnostics/thesis_pair_score_diagnostics.json"
    )
    mlp_diag = _load_thesis_diag_json(mlp_diag_path)

    # GNN variants: compute into consolidated folder (no writes into run dirs).
    pair_csv_13 = resolve_pair_dataset_csv_path(run_13, project_root=_REPO)
    pair_csv_15 = resolve_pair_dataset_csv_path(run_15, project_root=_REPO)

    diag_13 = run_thesis_pair_score_diagnostics(
        run_dir=run_13,
        graph_pt=graph_pt,
        pair_csv=pair_csv_13,
        gt_path=gt_path,
        output_dir=diagnostics_dir / "gnn_plus_explicit_features",
        checkpoint_name="best_model.pt",
        device="cpu",
        to_undirected=True,
        scoring_run_id="main_gnn_pu_1_no_ts_dedup_task_identity_13__expanded_full_gt",
    )
    diag_15 = run_thesis_pair_score_diagnostics(
        run_dir=run_15,
        graph_pt=graph_pt,
        pair_csv=pair_csv_15,
        gt_path=gt_path,
        output_dir=diagnostics_dir / "gnn_only_scorer",
        checkpoint_name="best_model.pt",
        device="cpu",
        to_undirected=True,
        scoring_run_id="main_gnn_pu_1_no_ts_dedup_task_identity_15_gnn_only_scorer__expanded_full_gt",
    )

    # 3) Non-seed summary table (ranking diagnostics main table)
    ranking_rows = [
        _extract_nonseed_summary(mlp_diag, "Explicit-feature MLP scorer"),
        _extract_nonseed_summary(diag_13, "GNN + explicit pair features"),
        _extract_nonseed_summary(diag_15, "GNN-only scorer"),
    ]
    ranking_df = pd.DataFrame(ranking_rows)
    ranking_csv = tables_dir / "gnn_pair_score_ranking_nonseed.csv"
    ranking_json = tables_dir / "gnn_pair_score_ranking_nonseed.json"
    ranking_df.to_csv(ranking_csv, index=False)
    ranking_json.write_text(json.dumps(ranking_rows, indent=2), encoding="utf-8")

    # Minimal LaTeX for the main ranking table.
    ranking_tex = tables_dir / "gnn_pair_score_ranking_nonseed.tex"
    ranking_tex.write_text(
        ranking_df.to_latex(index=False, float_format="%.3f"),
        encoding="utf-8",
    )

    # 4) Main comparison table (community quality)
    main_rows = [
        {
            "method": "Unscored seed/candidate graph",
            "uses_gnn_message_passing": False,
            "uses_explicit_pair_features": False,
            "homogeneity": 0.90,
            "completeness": 0.91,
            "v_measure": 0.90,
        },
        {
            "method": "Explicit-feature MLP scorer",
            "uses_gnn_message_passing": False,
            "uses_explicit_pair_features": True,
            "homogeneity": float(best_mlp.homogeneity),
            "completeness": float(best_mlp.completeness),
            "v_measure": float(best_mlp.v_measure),
        },
        {
            "method": "GNN + explicit pair features",
            "uses_gnn_message_passing": True,
            "uses_explicit_pair_features": True,
            "homogeneity": float(best_13.homogeneity),
            "completeness": float(best_13.completeness),
            "v_measure": float(best_13.v_measure),
        },
        {
            "method": "GNN-only scorer",
            "uses_gnn_message_passing": True,
            "uses_explicit_pair_features": False,
            "homogeneity": float(best_15.homogeneity),
            "completeness": float(best_15.completeness),
            "v_measure": float(best_15.v_measure),
        },
    ]
    main_df = pd.DataFrame(main_rows)
    (tables_dir / "gnn_pair_scoring_main_comparison.csv").write_text(
        main_df.to_csv(index=False), encoding="utf-8"
    )
    (tables_dir / "gnn_pair_scoring_main_comparison.json").write_text(
        json.dumps(main_rows, indent=2), encoding="utf-8"
    )
    (tables_dir / "gnn_pair_scoring_main_comparison.tex").write_text(
        main_df.to_latex(index=False, float_format="%.3f"),
        encoding="utf-8",
    )

    # 5) Best-configs table (GNN variants)
    best_cfg_rows = [
        {
            "variant": "GNN + explicit pair features",
            **{
                "algorithm": best_13.algorithm,
                "threshold": best_13.threshold,
                "resolution": best_13.resolution,
                "homogeneity": best_13.homogeneity,
                "completeness": best_13.completeness,
                "v_measure": best_13.v_measure,
                "communities": best_13.n_communities,
            },
        },
        {
            "variant": "GNN-only scorer",
            **{
                "algorithm": best_15.algorithm,
                "threshold": best_15.threshold,
                "resolution": best_15.resolution,
                "homogeneity": best_15.homogeneity,
                "completeness": best_15.completeness,
                "v_measure": best_15.v_measure,
                "communities": best_15.n_communities,
            },
        },
    ]
    best_cfg_df = pd.DataFrame(best_cfg_rows)
    (tables_dir / "gnn_pair_scoring_best_configs.csv").write_text(
        best_cfg_df.to_csv(index=False), encoding="utf-8"
    )
    (tables_dir / "gnn_pair_scoring_best_configs.json").write_text(
        json.dumps(best_cfg_rows, indent=2), encoding="utf-8"
    )
    (tables_dir / "gnn_pair_scoring_best_configs.tex").write_text(
        best_cfg_df.to_latex(index=False, float_format="%.3f"),
        encoding="utf-8",
    )

    # 6) KDE plots (non-seed only, expanded GT)
    kde_meta: dict[str, Any] = {}
    same_13, cross_13, meta_13 = _score_nonseed_pairs(
        run_dir=run_13, graph_pt=graph_pt, pair_csv=pair_csv_13, gt_path=gt_path
    )
    kde_meta["gnn_plus_explicit_features"] = meta_13
    _kde_plot_same_vs_cross(
        same_scores=same_13,
        cross_scores=cross_13,
        title="Non-seed candidate pairs: same vs cross campaign (GNN + explicit features)",
        out_png=plots_dir / "kde_nonseed_same_vs_cross__gnn_plus_explicit_features.png",
    )

    same_15, cross_15, meta_15 = _score_nonseed_pairs(
        run_dir=run_15, graph_pt=graph_pt, pair_csv=pair_csv_15, gt_path=gt_path
    )
    kde_meta["gnn_only_scorer"] = meta_15
    _kde_plot_same_vs_cross(
        same_scores=same_15,
        cross_scores=cross_15,
        title="Non-seed candidate pairs: same vs cross campaign (GNN-only scorer)",
        out_png=plots_dir / "kde_nonseed_same_vs_cross__gnn_only_scorer.png",
    )

    (output_root / "kde_generation_meta.json").write_text(
        json.dumps(kde_meta, indent=2), encoding="utf-8"
    )

    # 7) Training summaries: copy existing plots + parse metrics.csv best epoch
    training_rows: list[dict[str, Any]] = []
    for run_dir, label in (
        (run_13, "GNN + explicit pair features"),
        (run_15, "GNN-only scorer"),
    ):
        metrics_path = run_dir / "gnn/metrics.csv"
        dfm = pd.read_csv(metrics_path)
        dfm["val_loss"] = pd.to_numeric(dfm["val_loss"], errors="coerce")
        best_i = int(dfm["val_loss"].idxmin())
        best_row = dfm.iloc[best_i].to_dict()
        last_row = dfm.iloc[-1].to_dict()

        copied = {
            "training_metrics_png": _copy_if_exists(
                run_dir / "gnn/training_metrics.png",
                training_dir / f"{run_dir.name}__training_metrics.png",
            ),
            "loss_over_epochs_png": _copy_if_exists(
                run_dir / "gnn/loss_over_epochs.png",
                training_dir / f"{run_dir.name}__loss_over_epochs.png",
            ),
        }

        training_rows.append(
            {
                "run_dir": str(run_dir),
                "variant": label,
                "best_epoch_by_val_loss": int(best_row.get("epoch")),
                "best_val_loss": float(best_row.get("val_loss")),
                "best_train_loss": float(best_row.get("train_loss")),
                "final_epoch": int(last_row.get("epoch")),
                "final_val_loss": float(last_row.get("val_loss")),
                "final_train_loss": float(last_row.get("train_loss")),
                "plots_copied": copied,
            }
        )

    (tables_dir / "gnn_training_summaries.json").write_text(
        json.dumps(training_rows, indent=2), encoding="utf-8"
    )

    # 8) Summary markdown + manifest
    manifest = {
        "output_root": str(output_root),
        "inputs": {
            "expanded_gt": str(gt_path),
            "graph_pt": str(graph_pt),
            "run_dirs": {
                "gnn_plus_explicit_features": str(run_13),
                "gnn_only_scorer": str(run_15),
                "explicit_feature_mlp": str(run_mlp),
            },
            "pair_csv": {
                "run_13": str(pair_csv_13),
                "run_15": str(pair_csv_15),
                "mlp": str(mlp_diag.get("pair_csv")),
            },
        },
        "community_best": {
            "run_13": asdict(best_13),
            "run_15": asdict(best_15),
            "mlp": asdict(best_mlp),
        },
        "outputs": {
            "tables_dir": str(tables_dir),
            "plots_dir": str(plots_dir),
            "diagnostics_dir": str(diagnostics_dir),
            "training_dir": str(training_dir),
        },
    }
    (output_root / "paths_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    summary_md = output_root / "gnn_pair_scoring_summary.md"
    summary_lines = [
        "# GNN pair scoring (thesis-facing summary)",
        "",
        "## Community detection (best sweep config, expanded GT)",
        "",
        f"- **Explicit-feature MLP scorer**: H={best_mlp.homogeneity:.3f}, C={best_mlp.completeness:.3f}, V={best_mlp.v_measure:.3f}",
        f"- **GNN + explicit pair features**: H={best_13.homogeneity:.3f}, C={best_13.completeness:.3f}, V={best_13.v_measure:.3f}",
        f"- **GNN-only scorer**: H={best_15.homogeneity:.3f}, C={best_15.completeness:.3f}, V={best_15.v_measure:.3f}",
        "",
        "## Pair-score ranking (expanded GT; same-campaign is positive class)",
        "",
        f"- **MLP non-seed AUROC/AP**: {ranking_rows[0]['non_seed_auroc']:.3f} / {ranking_rows[0]['non_seed_ap']:.3f} (AP is class-imbalance-sensitive)",
        f"- **GNN+feat non-seed AUROC/AP**: {ranking_rows[1]['non_seed_auroc']:.3f} / {ranking_rows[1]['non_seed_ap']:.3f} (AP is class-imbalance-sensitive)",
        f"- **GNN-only non-seed AUROC/AP**: {ranking_rows[2]['non_seed_auroc']:.3f} / {ranking_rows[2]['non_seed_ap']:.3f} (AP is class-imbalance-sensitive)",
        "",
        "## Answers to the four thesis questions",
        "",
        "1. **Does GNN message passing improve over the explicit-feature MLP scorer?**",
        "   - Compare `v_measure` in `tables/gnn_pair_scoring_main_comparison.*` and non-seed AUROC/AP in `tables/gnn_pair_score_ranking_nonseed.*`.",
        "2. **Does adding explicit pair features to the GNN variant help?**",
        "   - Compare `_13` vs `_15` in the two tables above.",
        "3. **Does GNN-only pair scoring outperform the unscored seed/candidate graph?**",
        "   - Compare V-measure vs the baseline row (H=0.90, C=0.91, V=0.90) in the main comparison table.",
        "4. **Are GNN pair scores less separable than explicit MLP scores on non-seed candidate pairs?**",
        "   - Compare non-seed AUROC/AP and the KDE plots in `plots/`.",
        "",
    ]
    summary_md.write_text("\n".join(summary_lines), encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

