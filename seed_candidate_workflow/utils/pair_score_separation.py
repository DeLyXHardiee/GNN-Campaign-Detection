"""
Post-training score separation: same-campaign vs cross-campaign on GT-covered candidate pairs.

Loads a pair-supervision checkpoint, scores rows from pair_training_dataset.csv,
labels pairs using ground-truth JSON (email external_id -> campaign), and writes
plots + pair_score_separation_summary.json.

Also writes ``score_distribution_all_scored_pairs.png``: a histogram of every finite
model score in the pair table (no ground-truth filter), for comparison with GT-only
same/cross plots.
"""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# GNN code uses package ``src`` under core/GNN/src; mirror core/main.py and the
# seed_candidate_workflow wrapper script so ``python -m`` from repo root works.
_GNN_ROOT = Path(__file__).resolve().parents[2] / "core" / "GNN"
if str(_GNN_ROOT) not in sys.path:
    sys.path.insert(0, str(_GNN_ROOT))

from seed_candidate_workflow.utils.anchor_graph_helpers import load_anchor_graph_artifacts
from seed_candidate_workflow.utils.pair_model_inference import (
    load_pair_supervision_for_inference as _load_pair_supervision_for_inference,
    score_pair_rows as _score_pair_rows,
)
from seed_candidate_workflow.utils.raw_gnn_notebook import load_ground_truth_structures
from seed_candidate_workflow.utils.scorer_diagnostics_core import (
    quantiles_dict as _quantiles_dict_core,
    safe_auroc as _safe_auroc_core,
)
from seed_candidate_workflow.utils.pair_inspection_admitting_evidence import (
    enrich_inspection_with_admitting_evidence,
    load_admitting_evidence_index,
    resolve_candidate_generation_dir,
    resolve_seed_generation_dir,
)
from seed_candidate_workflow.utils.pair_score_separation_output_layout import (
    ExportFlags,
    build_navigation_index,
    build_primary_outputs,
    ensure_pair_score_separation_layout,
    path_for_write,
    rel_to_root,
    write_artifact_manifest,
)
from seed_candidate_workflow.utils.pair_low_band_twohop_channel import (
    build_channel_summary_table,
    build_twohop_channel_recommendations,
    extend_bool_terms_for_low_band_channels,
    low_band_twohop_joint_rule_names,
)
from seed_candidate_workflow.utils.scorer_diagnostics_rules import (
    BINARY_CONDITION_RULES_DEFAULT,
    CANDIDATE_RULES_DEFAULT,
    FEATURE_KEYS_DEFAULT,
    PROVENANCE_KEYS_DEFAULT,
    SEMANTIC_BUCKET_RULES_DEFAULT,
    SHARED_EVIDENCE_KEYS_DEFAULT,
)


def load_pair_supervision_for_inference(
    *,
    run_dir: Path,
    graph_pt: Path,
    checkpoint_name: str = "best_model.pt",
    device: str = "cpu",
    to_undirected: bool = True,
) -> dict[str, Any]:
    return _load_pair_supervision_for_inference(
        run_dir=run_dir,
        graph_pt=graph_pt,
        checkpoint_name=checkpoint_name,
        device=device,
        to_undirected=to_undirected,
    )


def _sanitize_filename_stem(name: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9._-]+", "_", name).strip("_")
    return s[:120] if s else "gt"


def _quantiles_dict(x: np.ndarray, qs: tuple[float, ...]) -> dict[str, float]:
    return _quantiles_dict_core(x, qs)


def _safe_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    return _safe_auroc_core(y_true, y_score)


def score_pair_rows(
    *,
    model: Any,
    pair_scorer: Any,
    data_cpu: Any,
    df_work: pd.DataFrame,
    device: Any,
    fanout: list[int],
    pair_batch_size: int,
    max_unique_emails: int,
    with_logits: bool = False,
    pair_feature_columns: list[str] | None = None,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    return _score_pair_rows(
        model=model,
        pair_scorer=pair_scorer,
        data_cpu=data_cpu,
        df_work=df_work,
        device=device,
        fanout=fanout,
        pair_batch_size=pair_batch_size,
        max_unique_emails=max_unique_emails,
        with_logits=with_logits,
        pair_feature_columns=pair_feature_columns,
    )


def _bin_edges_for_scores(
    same_scores: np.ndarray,
    cross_scores: np.ndarray,
    *,
    n_bins: int = 36,
) -> np.ndarray | None:
    """Shared bin edges for comparable same vs cross histograms (finite values only)."""
    same_scores = same_scores[np.isfinite(same_scores)]
    cross_scores = cross_scores[np.isfinite(cross_scores)]
    parts = [s for s in (same_scores, cross_scores) if s.size > 0]
    if not parts:
        return None
    all_s = np.concatenate(parts)
    lo, hi = float(np.min(all_s)), float(np.max(all_s))
    if hi <= lo:
        hi = lo + 1e-6
    return np.linspace(lo, hi, int(n_bins))


def _plot_score_histogram_counts(
    scores: np.ndarray,
    *,
    title: str,
    out_path: Path,
    bins: np.ndarray | None,
    cohort_label: str,
    color: str,
    xlabel: str = "Model score (sigmoid probability)",
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scores = scores[np.isfinite(scores)]
    save_path = path_for_write(out_path)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if scores.size == 0:
        ax.text(0.5, 0.5, f"No scored pairs ({cohort_label})", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Number of pairs")
        fig.tight_layout()
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        return

    if bins is None or bins.size < 2:
        lo, hi = float(np.min(scores)), float(np.max(scores))
        if hi <= lo:
            hi = lo + 1e-6
        bins = np.linspace(lo, hi, 36)

    ax.hist(
        scores,
        bins=bins,
        density=False,
        color=color,
        edgecolor="black",
        linewidth=0.25,
        alpha=0.85,
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Number of pairs")
    ax.set_title(f"{title} (n={scores.size})")
    fig.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _write_split_same_cross_histograms(
    *,
    same_scores: np.ndarray,
    cross_scores: np.ndarray,
    title_base: str,
    out_same: Path,
    out_cross: Path,
    xlabel: str = "Model score (sigmoid probability)",
) -> None:
    bins = _bin_edges_for_scores(same_scores, cross_scores)
    _plot_score_histogram_counts(
        same_scores,
        title=f"{title_base} — same_campaign",
        out_path=out_same,
        bins=bins,
        cohort_label="same_campaign",
        color="tab:orange",
        xlabel=xlabel,
    )
    _plot_score_histogram_counts(
        cross_scores,
        title=f"{title_base} — cross_campaign",
        out_path=out_cross,
        bins=bins,
        cohort_label="cross_campaign",
        color="tab:blue",
        xlabel=xlabel,
    )


def _kde_density_normalized(scores: np.ndarray, x_grid: np.ndarray) -> np.ndarray:
    """Gaussian KDE on ``x_grid``, rescaled so ∫ density dx = 1 over the grid."""
    s = np.clip(np.asarray(scores, dtype=float)[np.isfinite(scores)], 0.0, 1.0)
    y = np.zeros_like(x_grid, dtype=float)
    if s.size == 0:
        return y
    if s.size == 1:
        idx = int(np.argmin(np.abs(x_grid - float(s[0]))))
        y[idx] = 1.0
    else:
        from scipy.stats import gaussian_kde

        kde = gaussian_kde(s)
        y = np.maximum(kde(x_grid), 0.0)
    if hasattr(np, "trapezoid"):
        area = float(np.trapezoid(y, x_grid))
    else:
        area = float(np.trapz(y, x_grid))
    if area > 0.0:
        y = y / area
    return y


def _write_same_cross_kde_overlay(
    *,
    same_scores: np.ndarray,
    cross_scores: np.ndarray,
    title: str,
    out_path: Path,
    xlabel: str = "Learned pair score",
) -> None:
    """One figure: same vs cross campaign KDE curves, each normalized to unit area."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    color_same = "#e67e22"
    color_cross = "#1f77b4"
    fill_alpha = 0.42

    x_grid = np.linspace(0.0, 1.0, 256)
    same_c = np.asarray(same_scores, dtype=float)[np.isfinite(same_scores)]
    cross_c = np.asarray(cross_scores, dtype=float)[np.isfinite(cross_scores)]
    y_same = _kde_density_normalized(same_c, x_grid)
    y_cross = _kde_density_normalized(cross_c, x_grid)

    save_path = path_for_write(out_path)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if same_c.size == 0 and cross_c.size == 0:
        ax.text(0.5, 0.5, "No scored pairs in cohort", ha="center", va="center", transform=ax.transAxes)
    else:
        # Cross first, then same — overlapping regions blend via alpha.
        if cross_c.size > 0:
            ax.fill_between(
                x_grid,
                y_cross,
                0.0,
                color=color_cross,
                alpha=fill_alpha,
                linewidth=0.0,
                zorder=1,
            )
            ax.plot(
                x_grid,
                y_cross,
                color=color_cross,
                linewidth=2.0,
                label=f"Cross campaign ($n$={int(cross_c.size):,})",
                zorder=3,
            )
        if same_c.size > 0:
            ax.fill_between(
                x_grid,
                y_same,
                0.0,
                color=color_same,
                alpha=fill_alpha,
                linewidth=0.0,
                zorder=2,
            )
            ax.plot(
                x_grid,
                y_same,
                color=color_same,
                linewidth=2.0,
                label=f"Same campaign ($n$={int(same_c.size):,})",
                zorder=4,
            )
        ax.legend(loc="upper right", frameon=True, fontsize=9)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(bottom=0.0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Normalized density")
    ax.set_title(title)
    ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.5, zorder=0)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _write_same_cross_kde_plots_for_gt(
    *,
    gt_path: Path,
    plots_dir: Path,
    df_work: pd.DataFrame,
    scores: np.ndarray,
    scored_mask: np.ndarray,
) -> dict[str, Path]:
    """Write three KDE overlays (all GT pairs, positives only, unlabeled only)."""
    label_map, _eid_row, _camp = load_ground_truth_structures(gt_path)
    label_map = {str(k): v for k, v in label_map.items()}
    ei = df_work["email_i"].astype(str).values
    ej = df_work["email_j"].astype(str).values
    n = len(df_work)
    scored = scored_mask
    camp_i = np.array([label_map.get(str(ei[k])) for k in range(n)], dtype=object)
    camp_j = np.array([label_map.get(str(ej[k])) for k in range(n)], dtype=object)
    both = np.array(
        [camp_i[k] is not None and camp_j[k] is not None for k in range(n)],
        dtype=bool,
    )
    same_mask = both & (camp_i == camp_j)
    cross_mask = both & (camp_i != camp_j)
    pos_mask = (
        df_work["pair_status"].astype(str).str.lower().eq("positive").to_numpy()
        if "pair_status" in df_work.columns
        else np.zeros(n, dtype=bool)
    )
    unl_mask = (
        df_work["pair_status"].astype(str).str.lower().eq("unlabeled").to_numpy()
        if "pair_status" in df_work.columns
        else np.zeros(n, dtype=bool)
    )
    stem = _sanitize_filename_stem(gt_path.stem)
    out_all = plots_dir / f"score_density_kde_same_vs_cross_{stem}.png"
    out_pos = plots_dir / f"score_density_kde_same_vs_cross_positive_{stem}.png"
    out_unl = plots_dir / f"score_density_kde_same_vs_cross_unlabeled_{stem}.png"
    _write_same_cross_kde_overlay(
        same_scores=scores[same_mask & scored],
        cross_scores=scores[cross_mask & scored],
        title="Pair score density — positives and unlabeled",
        out_path=out_all,
    )
    _write_same_cross_kde_overlay(
        same_scores=scores[same_mask & pos_mask & scored],
        cross_scores=scores[cross_mask & pos_mask & scored],
        title="Pair score density — positives only",
        out_path=out_pos,
    )
    _write_same_cross_kde_overlay(
        same_scores=scores[same_mask & unl_mask & scored],
        cross_scores=scores[cross_mask & unl_mask & scored],
        title="Pair score density — unlabeled only",
        out_path=out_unl,
    )
    return {
        "plot_kde_same_vs_cross_all_gt": out_all,
        "plot_kde_same_vs_cross_positive_only": out_pos,
        "plot_kde_same_vs_cross_unlabeled_only": out_unl,
    }


def _summarize_one_gt(
    *,
    gt_path: Path,
    gt_label_map: dict[str, Any],
    email_i: pd.Series,
    email_j: pd.Series,
    scores: np.ndarray,
    cross_component_mask: np.ndarray | None,
) -> dict[str, Any]:
    ei = email_i.astype(str).values
    ej = email_j.astype(str).values
    n = len(ei)
    camp_i = np.array([gt_label_map.get(str(a)) for a in ei], dtype=object)
    camp_j = np.array([gt_label_map.get(str(b)) for b in ej], dtype=object)
    both = np.array(
        [camp_i[k] is not None and camp_j[k] is not None for k in range(n)],
        dtype=bool,
    )
    same_mask = np.zeros(n, dtype=bool)
    cross_mask = np.zeros(n, dtype=bool)
    for k in range(n):
        if not both[k]:
            continue
        if camp_i[k] == camp_j[k]:
            same_mask[k] = True
        else:
            cross_mask[k] = True
    scored = np.isfinite(scores)
    same_s = scores[same_mask & scored]
    cross_s = scores[cross_mask & scored]

    qs = (0.1, 0.25, 0.5, 0.75, 0.9)
    mask_eval = both & scored
    y_bin = same_mask[mask_eval].astype(np.int32)
    s_eval = scores[mask_eval]
    auroc = _safe_auroc(y_bin, s_eval)

    out: dict[str, Any] = {
        "gt_path": str(gt_path.resolve()),
        "n_gt_covered_candidate_pairs": int(mask_eval.sum()),
        "n_same_campaign_pairs": int(same_s.size),
        "n_cross_campaign_pairs": int(cross_s.size),
        "mean_score_same_campaign": float(np.mean(same_s)) if same_s.size else None,
        "mean_score_cross_campaign": float(np.mean(cross_s)) if cross_s.size else None,
        "median_score_same_campaign": float(np.median(same_s)) if same_s.size else None,
        "median_score_cross_campaign": float(np.median(cross_s)) if cross_s.size else None,
        "quantiles_same_campaign": _quantiles_dict(same_s, qs),
        "quantiles_cross_campaign": _quantiles_dict(cross_s, qs),
        "difference_in_means": float(np.mean(same_s) - np.mean(cross_s))
        if same_s.size and cross_s.size
        else None,
        "difference_in_medians": float(np.median(same_s) - np.median(cross_s))
        if same_s.size and cross_s.size
        else None,
        "auroc_same_vs_cross_on_gt_covered_pairs": auroc,
    }

    if cross_component_mask is not None and cross_component_mask.any():
        m_cc = both & cross_component_mask & scored
        same_cc = scores[same_mask & cross_component_mask & scored]
        cross_cc = scores[cross_mask & cross_component_mask & scored]
        out["cross_component_subset"] = {
            "n_rows_cross_seed_component_flag": int(cross_component_mask.sum()),
            "n_gt_covered_in_subset": int(m_cc.sum()),
            "n_same_campaign_in_subset": int(same_cc.size),
            "n_cross_campaign_in_subset": int(cross_cc.size),
            "mean_score_same_campaign": float(np.mean(same_cc)) if same_cc.size else None,
            "mean_score_cross_campaign": float(np.mean(cross_cc)) if cross_cc.size else None,
            "difference_in_means": float(np.mean(same_cc) - np.mean(cross_cc))
            if same_cc.size and cross_cc.size
            else None,
            "auroc_same_vs_cross": _safe_auroc(
                same_mask[m_cc].astype(np.int32), scores[m_cc]
            ),
        }
    return out


def _infer_graph_id_from_pair_csv(pair_csv: Path) -> str | None:
    parts = [p.lower() for p in pair_csv.parts]
    try:
        i = parts.index("graph_bundles")
        if i + 1 < len(pair_csv.parts):
            v = str(pair_csv.parts[i + 1]).strip()
            if v:
                return v
    except ValueError:
        pass
    try:
        i = parts.index("anchor_candidates")
    except ValueError:
        return None
    if i + 1 >= len(pair_csv.parts):
        return None
    return str(pair_csv.parts[i + 1]).strip() or None


def _load_anchor_nodes_by_email(
    *,
    pair_csv: Path,
    project_root: Path,
    explicit_anchor_run_dir: Path | None = None,
) -> tuple[dict[str, dict[str, set[str]]], dict[str, Any]]:
    if explicit_anchor_run_dir is not None:
        run_dir = explicit_anchor_run_dir.resolve()
    else:
        run_id = _infer_graph_id_from_pair_csv(pair_csv)
        if not run_id:
            return {}, {"status": "skipped", "reason": "could_not_infer_graph_id_from_pair_csv"}
        run_dir = (
            project_root / "seed_candidate_workflow" / "output" / "graph_bundles" / run_id / "anchor" / run_id
        ).resolve()
    if not run_dir.is_dir():
        return {}, {"status": "skipped", "reason": f"anchor_run_dir_not_found:{run_dir}"}

    nodes_df, _edges, _cand, _summary, _g = load_anchor_graph_artifacts(
        run_dir, load_graph_pickle=False
    )
    shared_cols = [
        "url_set",
        "sender_set",
        "attachment_set",
        "sender_email_domain_set",
        "domain_set",
        "stem_set",
        "html_structure_fingerprint_set",
        "received_host_set",
    ]
    keep = [c for c in shared_cols if c in nodes_df.columns]
    if "external_id" not in nodes_df.columns or not keep:
        return {}, {"status": "skipped", "reason": "anchor_nodes_missing_external_or_shared_cols"}

    out: dict[str, dict[str, set[str]]] = {}
    for _, r in nodes_df[["external_id", *keep]].iterrows():
        eid = str(r["external_id"])
        row: dict[str, set[str]] = {}
        for c in keep:
            v = r[c]
            if isinstance(v, set):
                row[c] = {str(x) for x in v if str(x).strip()}
            elif isinstance(v, (list, tuple)):
                row[c] = {str(x) for x in v if str(x).strip()}
            else:
                row[c] = set()
        out[eid] = row
    return out, {"status": "ok", "anchor_run_dir": str(run_dir), "shared_columns": keep}


def _safe_float_stats(x: pd.Series) -> dict[str, float | None]:
    s = pd.to_numeric(x, errors="coerce")
    s = s[s.notna()]
    if s.empty:
        return {"mean": None, "median": None, "q25": None, "q75": None, "n_non_null": 0, "n_missing": int(len(x))}
    return {
        "mean": float(s.mean()),
        "median": float(s.median()),
        "q25": float(s.quantile(0.25)),
        "q75": float(s.quantile(0.75)),
        "n_non_null": int(s.shape[0]),
        "n_missing": int(len(x) - s.shape[0]),
    }


def _summarize_group(
    *,
    gdf: pd.DataFrame,
    n_total_eval: int,
    nodes_by_email: dict[str, dict[str, set[str]]],
) -> dict[str, Any]:
    n_edges = int(len(gdf))
    out: dict[str, Any] = {
        "n_edges": n_edges,
        "fraction_of_gt_covered_candidate_pairs": (float(n_edges / n_total_eval) if n_total_eval > 0 else None),
    }

    if n_edges == 0:
        out["provenance"] = {}
        out["feature_summaries"] = {}
        out["shared_evidence"] = {}
        out["score_summary"] = {"mean": None, "median": None, "q10": None, "q25": None, "q50": None, "q75": None, "q90": None}
        return out

    src = pd.to_numeric(gdf.get("source_count"), errors="coerce")
    prov_counts = {
        "from_semantic": int(gdf.get("from_semantic", False).fillna(False).astype(bool).sum()),
        "from_rare_artifact": int(gdf.get("from_rare_artifact", False).fillna(False).astype(bool).sum()),
        "from_2hop": int(gdf.get("from_2hop", False).fillna(False).astype(bool).sum()),
        "from_component": int(gdf.get("from_component", False).fillna(False).astype(bool).sum()),
        "source_count_eq_1": int(src.eq(1).sum()),
        "source_count_eq_2": int(src.eq(2).sum()),
        "source_count_ge_3": int(src.ge(3).sum()),
        "same_seed_component_flag": int(gdf.get("same_seed_component_flag", False).fillna(False).astype(bool).sum()),
        "cross_seed_component_flag": int(gdf.get("cross_seed_component_flag", False).fillna(False).astype(bool).sum()),
    }
    out["provenance"] = {
        k: {"count": v, "fraction": float(v / n_edges)} for k, v in prov_counts.items()
    }

    feat_cols = [
        "semantic_cosine_max",
        "rare_artifact_rarity_max",
        "twohop_rarity_max",
        "component_cosine_max",
        "time_gap_seconds_min",
    ]
    out["feature_summaries"] = {c: _safe_float_stats(gdf[c]) if c in gdf.columns else {"mean": None, "median": None, "q25": None, "q75": None, "n_non_null": 0, "n_missing": n_edges} for c in feat_cols}

    score_s = pd.to_numeric(gdf["score"], errors="coerce")
    score_s = score_s[score_s.notna()]
    out["score_summary"] = {
        "mean": float(score_s.mean()) if not score_s.empty else None,
        "median": float(score_s.median()) if not score_s.empty else None,
        "q10": float(score_s.quantile(0.10)) if not score_s.empty else None,
        "q25": float(score_s.quantile(0.25)) if not score_s.empty else None,
        "q50": float(score_s.quantile(0.50)) if not score_s.empty else None,
        "q75": float(score_s.quantile(0.75)) if not score_s.empty else None,
        "q90": float(score_s.quantile(0.90)) if not score_s.empty else None,
    }

    shared_defs = [
        ("url_set", "shared_url"),
        ("sender_set", "shared_sender"),
        ("attachment_set", "shared_attachment"),
        ("sender_email_domain_set", "shared_sender_domain"),
        ("domain_set", "shared_domain"),
        ("stem_set", "shared_stem"),
    ]
    shared_counts = {label: [] for _col, label in shared_defs}
    missing_pair = 0
    for _, r in gdf.iterrows():
        a = str(r["email_i"])
        b = str(r["email_j"])
        na = nodes_by_email.get(a)
        nb = nodes_by_email.get(b)
        if na is None or nb is None:
            missing_pair += 1
            for _col, label in shared_defs:
                shared_counts[label].append(0)
            continue
        for col, label in shared_defs:
            sa = na.get(col) or set()
            sb = nb.get(col) or set()
            shared_counts[label].append(int(len(sa & sb)))
    out["shared_evidence"] = {}
    for _col, label in shared_defs:
        arr = np.array(shared_counts[label], dtype=np.int64)
        out["shared_evidence"][label] = {
            "fraction_edges_with_at_least_1": float((arr >= 1).mean()) if arr.size else None,
            "mean_shared_count_per_edge": float(arr.mean()) if arr.size else None,
        }
    out["shared_evidence"]["n_pairs_missing_anchor_node_context"] = int(missing_pair)
    return out


def _compute_band_diagnostics_for_gt(
    *,
    df_work: pd.DataFrame,
    scores: np.ndarray,
    same_mask: np.ndarray,
    cross_mask: np.ndarray,
    eval_mask: np.ndarray,
    nodes_by_email: dict[str, dict[str, set[str]]],
    low_max: float,
    high_min: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    df_eval = df_work.loc[eval_mask].copy()
    df_eval["score"] = scores[eval_mask]
    same_eval = same_mask[eval_mask]
    cross_eval = cross_mask[eval_mask]
    low = df_eval["score"].ge(0.0) & df_eval["score"].le(float(low_max))
    high = df_eval["score"].gt(float(high_min)) & df_eval["score"].le(1.0)
    groups = {
        "same_campaign_low_score": same_eval & low.to_numpy(dtype=bool, copy=False),
        "same_campaign_high_score": same_eval & high.to_numpy(dtype=bool, copy=False),
        "cross_campaign_low_score": cross_eval & low.to_numpy(dtype=bool, copy=False),
        "cross_campaign_high_score": cross_eval & high.to_numpy(dtype=bool, copy=False),
    }
    n_total_eval = int(len(df_eval))
    diag: dict[str, Any] = {
        "band_definitions": {
            "low": {"min_inclusive": 0.0, "max_inclusive": float(low_max)},
            "high": {"min_exclusive": float(high_min), "max_inclusive": 1.0},
        },
        "n_gt_covered_candidate_pairs_with_finite_scores": n_total_eval,
        "groups": {},
    }
    csv_rows: list[dict[str, Any]] = []
    for gname, gmask in groups.items():
        gdf = df_eval.loc[gmask].copy()
        gs = _summarize_group(gdf=gdf, n_total_eval=n_total_eval, nodes_by_email=nodes_by_email)
        diag["groups"][gname] = gs
        row = {
            "group_name": gname,
            "n_edges": gs["n_edges"],
            "fraction_of_gt_covered_candidate_pairs": gs["fraction_of_gt_covered_candidate_pairs"],
            "score_mean": gs["score_summary"]["mean"],
            "score_median": gs["score_summary"]["median"],
            "score_q10": gs["score_summary"]["q10"],
            "score_q25": gs["score_summary"]["q25"],
            "score_q50": gs["score_summary"]["q50"],
            "score_q75": gs["score_summary"]["q75"],
            "score_q90": gs["score_summary"]["q90"],
        }
        for key, val in gs["provenance"].items():
            row[f"prov_frac_{key}"] = val.get("fraction")
        for feat, val in gs["feature_summaries"].items():
            row[f"{feat}_mean"] = val.get("mean")
            row[f"{feat}_median"] = val.get("median")
            row[f"{feat}_q25"] = val.get("q25")
            row[f"{feat}_q75"] = val.get("q75")
        for key, val in gs["shared_evidence"].items():
            if isinstance(val, dict):
                row[f"{key}_fraction_ge1"] = val.get("fraction_edges_with_at_least_1")
                row[f"{key}_mean_count"] = val.get("mean_shared_count_per_edge")
            else:
                row[key] = val
        csv_rows.append(row)
    return diag, csv_rows


def _safe_float(v: Any) -> float | None:
    if v is None:
        return None
    x = pd.to_numeric(v, errors="coerce")
    if pd.isna(x):
        return None
    return float(x)


def _compare_fraction_metric(*, same_v: Any, cross_v: Any) -> dict[str, Any]:
    s = _safe_float(same_v)
    c = _safe_float(cross_v)
    diff = (s - c) if (s is not None and c is not None) else None
    enrich: float | None
    if s is None or c is None:
        enrich = None
    elif c == 0.0:
        enrich = None
    else:
        enrich = float(s / c)
    return {
        "same_value": s,
        "cross_value": c,
        "same_low_value": s,
        "cross_low_value": c,
        "difference_same_minus_cross": diff,
        "abs_difference": (abs(diff) if diff is not None else None),
        "enrichment_same_over_cross": enrich,
    }


def _build_band_separator_for_gt(
    *,
    gt_path: Path,
    band_diag: dict[str, Any],
    band_kind: str,
    same_group_key: str,
    cross_group_key: str,
    thresholds_key: str,
    same_value_col: str,
    cross_value_col: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    groups = band_diag.get("groups") or {}
    same = groups.get(same_group_key) or {}
    cross = groups.get(cross_group_key) or {}
    n_total_eval = int(band_diag.get("n_gt_covered_candidate_pairs_with_finite_scores") or 0)
    n_same = int(same.get("n_edges") or 0)
    n_cross = int(cross.get("n_edges") or 0)
    n_band = int(n_same + n_cross)

    out: dict[str, Any] = {
        "gt_path": str(gt_path.resolve()),
        "band_kind": band_kind,
        f"{band_kind}_band_thresholds": band_diag.get("band_definitions", {}).get(thresholds_key, {}),
        "counts": {
            f"n_same_campaign_{band_kind}_score": n_same,
            f"n_cross_campaign_{band_kind}_score": n_cross,
            f"n_total_{band_kind}_band_gt_covered_pairs": n_band,
            "n_total_gt_covered_candidate_pairs_with_finite_scores": n_total_eval,
            f"fraction_gt_covered_pairs_that_are_{band_kind}_band_same": (
                float(n_same / n_total_eval) if n_total_eval > 0 else None
            ),
            f"fraction_gt_covered_pairs_that_are_{band_kind}_band_cross": (
                float(n_cross / n_total_eval) if n_total_eval > 0 else None
            ),
        },
    }

    rows: list[dict[str, Any]] = []

    prov_keys = list(PROVENANCE_KEYS_DEFAULT)
    prov_out: dict[str, Any] = {}
    for k in prov_keys:
        same_v = ((same.get("provenance") or {}).get(k) or {}).get("fraction")
        cross_v = ((cross.get("provenance") or {}).get(k) or {}).get("fraction")
        cmp = _compare_fraction_metric(same_v=same_v, cross_v=cross_v)
        prov_out[k] = cmp
        rows.append(
            {
                "gt_path": str(gt_path.resolve()),
                "band_kind": band_kind,
                "metric_group": "provenance",
                "metric_name": k,
                same_value_col: cmp["same_value"],
                cross_value_col: cmp["cross_value"],
                "difference": cmp["difference_same_minus_cross"],
                "enrichment": cmp["enrichment_same_over_cross"],
                "abs_difference": cmp["abs_difference"],
            }
        )
    out["provenance_comparison"] = prov_out

    feature_keys = list(FEATURE_KEYS_DEFAULT)
    feat_out: dict[str, Any] = {}
    for k in feature_keys:
        ssum = (same.get("feature_summaries") or {}).get(k) or {}
        csum = (cross.get("feature_summaries") or {}).get(k) or {}
        ms = _safe_float(ssum.get("mean"))
        mc = _safe_float(csum.get("mean"))
        med_s = _safe_float(ssum.get("median"))
        med_c = _safe_float(csum.get("median"))
        feat_out[k] = {
            f"mean_same_{band_kind}": ms,
            f"mean_cross_{band_kind}": mc,
            f"median_same_{band_kind}": med_s,
            f"median_cross_{band_kind}": med_c,
            "difference_in_means_same_minus_cross": (ms - mc) if (ms is not None and mc is not None) else None,
            "difference_in_medians_same_minus_cross": (
                (med_s - med_c) if (med_s is not None and med_c is not None) else None
            ),
            f"n_missing_same_{band_kind}": int(ssum.get("n_missing") or 0),
            f"n_missing_cross_{band_kind}": int(csum.get("n_missing") or 0),
            f"n_non_null_same_{band_kind}": int(ssum.get("n_non_null") or 0),
            f"n_non_null_cross_{band_kind}": int(csum.get("n_non_null") or 0),
        }
        rows.append(
            {
                "gt_path": str(gt_path.resolve()),
                "band_kind": band_kind,
                "metric_group": "feature_mean",
                "metric_name": k,
                same_value_col: ms,
                cross_value_col: mc,
                "difference": feat_out[k]["difference_in_means_same_minus_cross"],
                "enrichment": None,
                "abs_difference": (
                    abs(feat_out[k]["difference_in_means_same_minus_cross"])
                    if feat_out[k]["difference_in_means_same_minus_cross"] is not None
                    else None
                ),
            }
        )
    out["feature_comparison"] = feat_out

    shared_keys = list(SHARED_EVIDENCE_KEYS_DEFAULT)
    shared_out: dict[str, Any] = {}
    for k in shared_keys:
        same_v = ((same.get("shared_evidence") or {}).get(k) or {}).get("fraction_edges_with_at_least_1")
        cross_v = ((cross.get("shared_evidence") or {}).get(k) or {}).get("fraction_edges_with_at_least_1")
        cmp = _compare_fraction_metric(same_v=same_v, cross_v=cross_v)
        shared_out[k] = cmp
        rows.append(
            {
                "gt_path": str(gt_path.resolve()),
                "band_kind": band_kind,
                "metric_group": "shared_evidence",
                "metric_name": k,
                same_value_col: cmp["same_value"],
                cross_value_col: cmp["cross_value"],
                "difference": cmp["difference_same_minus_cross"],
                "enrichment": cmp["enrichment_same_over_cross"],
                "abs_difference": cmp["abs_difference"],
            }
        )
    out["shared_evidence_comparison"] = shared_out

    ranked = [r for r in rows if r.get("abs_difference") is not None]
    ranked.sort(key=lambda r: float(r["abs_difference"]), reverse=True)
    top = ranked[:10]
    out["ranked_separators_top10"] = [
        {
            "rank": i + 1,
            "metric_group": r["metric_group"],
            "metric_name": r["metric_name"],
            "same_value": r[same_value_col],
            "cross_value": r[cross_value_col],
            "difference_same_minus_cross": r["difference"],
            "abs_difference": r["abs_difference"],
            "enrichment_same_over_cross": r["enrichment"],
            "favors": (
                "same_campaign"
                if (r["difference"] is not None and float(r["difference"]) > 0)
                else "cross_campaign"
                if (r["difference"] is not None and float(r["difference"]) < 0)
                else "tie"
            ),
        }
        for i, r in enumerate(top)
    ]
    out["ranked_separators_favoring_same_top10"] = [
        r for r in out["ranked_separators_top10"] if r.get("favors") == "same_campaign"
    ][:10]
    out["ranked_separators_favoring_cross_top10"] = [
        r for r in sorted(
            out["ranked_separators_top10"],
            key=lambda r: abs(float(r.get("difference") or 0.0)),
            reverse=True,
        )
        if r.get("favors") == "cross_campaign"
    ][:10]

    return out, rows


def _build_low_band_separator_for_gt(
    *,
    gt_path: Path,
    band_diag: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    out, rows = _build_band_separator_for_gt(
        gt_path=gt_path,
        band_diag=band_diag,
        band_kind="low",
        same_group_key="same_campaign_low_score",
        cross_group_key="cross_campaign_low_score",
        thresholds_key="low",
        same_value_col="same_low_value",
        cross_value_col="cross_low_value",
    )
    out["low_band_thresholds"] = out.get("low_band_thresholds")
    return out, rows


def _build_high_band_separator_for_gt(
    *,
    gt_path: Path,
    band_diag: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    return _build_band_separator_for_gt(
        gt_path=gt_path,
        band_diag=band_diag,
        band_kind="high",
        same_group_key="same_campaign_high_score",
        cross_group_key="cross_campaign_high_score",
        thresholds_key="high",
        same_value_col="same_high_value",
        cross_value_col="cross_high_value",
    )


def _safe_enrichment(same_v: float | None, cross_v: float | None) -> float | None:
    if same_v is None or cross_v is None or cross_v == 0.0:
        return None
    return float(same_v / cross_v)


def _cmp_from_masks(
    *,
    cond_same: np.ndarray,
    base_same: np.ndarray,
    cond_cross: np.ndarray,
    base_cross: np.ndarray,
    value_key_same: str = "same_value",
    value_key_cross: str = "cross_value",
    count_key_same: str = "n_same",
    count_key_cross: str = "n_cross",
) -> dict[str, Any]:
    ns = int(base_same.sum())
    nc = int(base_cross.sum())
    fs = float((cond_same & base_same).sum() / ns) if ns > 0 else None
    fc = float((cond_cross & base_cross).sum() / nc) if nc > 0 else None
    diff = (fs - fc) if (fs is not None and fc is not None) else None
    out = {
        value_key_same: fs,
        value_key_cross: fc,
        "difference_same_minus_cross": diff,
        "abs_difference": (abs(diff) if diff is not None else None),
        "enrichment_same_over_cross": _safe_enrichment(fs, fc),
        count_key_same: ns,
        count_key_cross: nc,
    }
    return out


HIGH_BAND_EXTRA_JOINT_RULES: tuple[str, ...] = (
    "semantic_ge_0_90",
    "semantic_ge_0_90_AND_shared_sender",
    "semantic_ge_0_90_AND_shared_stem",
    "semantic_ge_0_90_AND_shared_sender_domain",
    "n_shared_core_channels_ge_1",
    "semantic_ge_0_90_AND_n_shared_core_channels_ge_1",
)


def _joint_row(
    *,
    gt_path: Path,
    band_kind: str,
    analysis_section: str,
    condition_name: str,
    cmp: dict[str, Any],
    value_key_same: str,
    value_key_cross: str,
    focus: str,
) -> dict[str, Any]:
    return {
        "gt_path": str(gt_path.resolve()),
        "band_kind": band_kind,
        "focus": focus,
        "analysis_section": analysis_section,
        "condition_name": condition_name,
        value_key_same: cmp.get(value_key_same, cmp.get("same_low_value")),
        value_key_cross: cmp.get(value_key_cross, cmp.get("cross_low_value")),
        "difference": cmp["difference_same_minus_cross"],
        "enrichment": cmp["enrichment_same_over_cross"],
        "abs_difference": cmp["abs_difference"],
        "favors": (
            "same_campaign"
            if (cmp["difference_same_minus_cross"] is not None and float(cmp["difference_same_minus_cross"]) > 0)
            else "cross_campaign"
            if (cmp["difference_same_minus_cross"] is not None and float(cmp["difference_same_minus_cross"]) < 0)
            else "tie"
        ),
    }


def _generate_high_band_recommendations(
    *,
    marginal: dict[str, Any] | None,
    joint: dict[str, Any],
) -> dict[str, Any]:
    """Readable recommendations for high-score unlabeled same vs cross separation."""

    def _fmt_sep(r: dict[str, Any]) -> str:
        sv = r.get("same_high_value", r.get("same_value"))
        cv = r.get("cross_high_value", r.get("cross_value"))
        name = r.get("condition_name") or r.get("metric_name") or "?"
        sec = r.get("analysis_section") or r.get("metric_group") or ""
        diff = r.get("difference_same_minus_cross", r.get("difference"))
        if sv is None or cv is None:
            return f"{sec}:{name}"
        return f"{sec}:{name} (same={sv:.3f}, cross={cv:.3f}, Δ={diff:+.3f})"

    good_same: list[str] = []
    for r in joint.get("ranked_joint_separators_favoring_same_top10") or []:
        good_same.append(_fmt_sep(r))
    if marginal:
        for r in marginal.get("ranked_separators_favoring_same_top10") or []:
            good_same.append(_fmt_sep(r))

    dangerous_cross: list[str] = []
    for r in joint.get("ranked_joint_separators_favoring_cross_top10") or []:
        dangerous_cross.append(_fmt_sep(r))
    if marginal:
        for r in marginal.get("ranked_separators_favoring_cross_top10") or []:
            dangerous_cross.append(_fmt_sep(r))

    next_steps: list[str] = []
    cross_fp_signals = {s.split("(")[0].split(":")[-1] for s in dangerous_cross[:5]}
    if any("from_2hop" in s or "2hop" in s for s in dangerous_cross):
        next_steps.append(
            "High-score cross-campaign false positives are enriched for 2-hop provenance; "
            "tighten upstream 2-hop candidate generation or down-rank 2hop-only high scores."
        )
    if any("from_component" in s or "component" in s for s in dangerous_cross):
        next_steps.append(
            "Component expansion appears in the dangerous high-score cross regime; "
            "review component_expansion_v1 gates or require stronger shared-channel support."
        )
    if any("shared_sender" in s and "NOT" in s for s in dangerous_cross):
        next_steps.append(
            "Cross high-score pairs often lack shared sender; add sender/shared-channel requirements "
            "for high-confidence promotion or ranking penalties when sender is absent."
        )
    if any("semantic_ge_0_90" in s for s in good_same):
        next_steps.append(
            "True high-score same-campaign unlabeled pairs are often semantic-backed (≥0.90); "
            "preserve semantic+support combinations in candidate/seed unions."
        )
    if any("shared_stem" in s or "shared_sender" in s for s in good_same):
        next_steps.append(
            "Shared sender/stem combinations characterize good high-score same-campaign unlabeled pairs; "
            "use these as positive ranking features or required support for high-band acceptance."
        )
    if not next_steps:
        next_steps.append(
            "Inspect ranked_joint_separators tables for the largest |Δ| conditions; "
            "prioritize upstream filtering when cross-favoring rules overlap provenance families."
        )
    if cross_fp_signals and good_same:
        next_steps.append(
            "Consider a ranking loss or score calibration focused on high-band cross-campaign "
            "false positives vs high-band same-campaign unlabeled pairs using the top separators above."
        )

    return {
        "what_defines_good_high_score_same_campaign_unlabeled": good_same[:12],
        "what_defines_dangerous_high_score_cross_campaign_unlabeled": dangerous_cross[:12],
        "implied_next_steps": next_steps,
    }


def _numeric_series_for_eval(df_eval: pd.DataFrame, col: str) -> pd.Series:
    """Per-row numeric series; missing columns become NaN (not a scalar)."""
    if col not in df_eval.columns:
        return pd.Series(np.nan, index=df_eval.index, dtype=float)
    return pd.to_numeric(df_eval[col], errors="coerce")


def _build_band_joint_separator_for_gt(
    *,
    gt_path: Path,
    df_eval: pd.DataFrame,
    same_band_mask_eval: np.ndarray,
    cross_band_mask_eval: np.ndarray,
    band_kind: str,
    band_thresholds: dict[str, float],
    nodes_by_email: dict[str, dict[str, set[str]]],
    value_key_same: str,
    value_key_cross: str,
    focus: str = "all_pairs_in_band",
    extra_joint_rules: tuple[str, ...] = (),
    marginal_sep: dict[str, Any] | None = None,
    include_recommendations: bool = False,
    twohop_channel_analysis: bool = False,
    evidence_index: dict[tuple[str, str], list[dict[str, Any]]] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """
    Joint separator analysis between same- and cross-campaign pairs within a score band.
    """
    out_rows: list[dict[str, Any]] = []
    n_eval = int(len(df_eval))
    count_same_key = f"n_same_campaign_{band_kind}_score"
    count_cross_key = f"n_cross_campaign_{band_kind}_score"
    count_total_key = f"n_total_{band_kind}_band_gt_covered_pairs"
    if n_eval == 0:
        empty: dict[str, Any] = {
            "gt_path": str(gt_path.resolve()),
            "band_kind": band_kind,
            "focus": focus,
            f"{band_kind}_band_thresholds": band_thresholds,
            "counts": {
                count_same_key: 0,
                count_cross_key: 0,
                count_total_key: 0,
            },
            "binary_joint_comparisons": {},
            "semantic_bucket_analysis": {},
            "candidate_rule_analysis": {},
            "ranked_joint_separators_top15": [],
            "ranked_joint_separators_favoring_same_top10": [],
            "ranked_joint_separators_favoring_cross_top10": [],
        }
        if include_recommendations:
            empty["recommendations"] = _generate_high_band_recommendations(
                marginal=marginal_sep,
                joint=empty,
            )
        return empty, out_rows

    fs = df_eval.get("from_semantic", False).fillna(False).astype(bool).to_numpy()
    f2 = df_eval.get("from_2hop", False).fillna(False).astype(bool).to_numpy()
    fcomp = df_eval.get("from_component", False).fillna(False).astype(bool).to_numpy()
    fra = df_eval.get("from_rare_artifact", False).fillna(False).astype(bool).to_numpy()
    sem = _numeric_series_for_eval(df_eval, "semantic_cosine_max")
    same_seed = (
        df_eval.get("same_seed_component_flag", False).fillna(False).astype(bool).to_numpy()
        if "same_seed_component_flag" in df_eval.columns
        else np.zeros(n_eval, dtype=bool)
    )
    cross_seed = (
        df_eval.get("cross_seed_component_flag", False).fillna(False).astype(bool).to_numpy()
        if "cross_seed_component_flag" in df_eval.columns
        else np.zeros(n_eval, dtype=bool)
    )

    n = len(df_eval)
    has_shared_sender = np.zeros(n, dtype=bool)
    has_shared_stem = np.zeros(n, dtype=bool)
    has_shared_sender_domain = np.zeros(n, dtype=bool)
    has_shared_url = np.zeros(n, dtype=bool)
    has_shared_attachment = np.zeros(n, dtype=bool)
    has_shared_domain = np.zeros(n, dtype=bool)
    has_shared_received_host = np.zeros(n, dtype=bool)
    has_shared_html_fp = np.zeros(n, dtype=bool)
    for i, r in enumerate(df_eval.itertuples(index=False)):
        a = str(getattr(r, "email_i"))
        b = str(getattr(r, "email_j"))
        na = nodes_by_email.get(a)
        nb = nodes_by_email.get(b)
        if na is None or nb is None:
            continue
        has_shared_sender[i] = bool((na.get("sender_set") or set()) & (nb.get("sender_set") or set()))
        has_shared_stem[i] = bool((na.get("stem_set") or set()) & (nb.get("stem_set") or set()))
        has_shared_sender_domain[i] = bool(
            (na.get("sender_email_domain_set") or set()) & (nb.get("sender_email_domain_set") or set())
        )
        has_shared_url[i] = bool((na.get("url_set") or set()) & (nb.get("url_set") or set()))
        has_shared_attachment[i] = bool((na.get("attachment_set") or set()) & (nb.get("attachment_set") or set()))
        has_shared_domain[i] = bool((na.get("domain_set") or set()) & (nb.get("domain_set") or set()))
        has_shared_received_host[i] = bool(
            (na.get("received_host_set") or set()) & (nb.get("received_host_set") or set())
        )
        has_shared_html_fp[i] = bool(
            (na.get("html_structure_fingerprint_set") or set())
            & (nb.get("html_structure_fingerprint_set") or set())
        )

    n_core = (
        has_shared_sender.astype(np.int8)
        + has_shared_stem.astype(np.int8)
        + has_shared_sender_domain.astype(np.int8)
        + has_shared_url.astype(np.int8)
        + has_shared_attachment.astype(np.int8)
        + has_shared_domain.astype(np.int8)
        + has_shared_received_host.astype(np.int8)
        + has_shared_html_fp.astype(np.int8)
    )

    same_band = same_band_mask_eval.astype(bool)
    cross_band = cross_band_mask_eval.astype(bool)
    n_same = int(same_band.sum())
    n_cross = int(cross_band.sum())
    n_band = int(n_same + n_cross)

    sem_ge_90 = sem.ge(0.90).fillna(False).to_numpy()
    n_core_ge_1 = n_core >= 1

    bool_terms: dict[str, np.ndarray] = {
        "from_semantic": fs,
        "from_2hop": f2,
        "from_component": fcomp,
        "from_rare_artifact": fra,
        "same_seed_component_flag": same_seed,
        "cross_seed_component_flag": cross_seed,
        "shared_sender": has_shared_sender,
        "shared_stem": has_shared_stem,
        "shared_sender_domain": has_shared_sender_domain,
        "shared_url": has_shared_url,
        "shared_attachment": has_shared_attachment,
        "shared_domain": has_shared_domain,
        "shared_received_host": has_shared_received_host,
        "shared_html_fp": has_shared_html_fp,
        "semantic_ge_0_90": sem_ge_90,
        "n_shared_core_channels_ge_1": n_core_ge_1,
    }
    sc = _numeric_series_for_eval(df_eval, "source_count")
    bool_terms["source_count_eq_1"] = sc.eq(1).fillna(False).to_numpy(dtype=bool)
    bool_terms["source_count_ge_2"] = sc.ge(2).fillna(False).to_numpy(dtype=bool)

    if twohop_channel_analysis:
        body_only_tok = _numeric_series_for_eval(df_eval, "body_only_token_jaccard")
        body_only_c4 = _numeric_series_for_eval(df_eval, "body_only_char4gram_jaccard")
        path_comb = _numeric_series_for_eval(df_eval, "path_token_jaccard_combined")
        url_path = _numeric_series_for_eval(df_eval, "url_path_token_jaccard")
        bool_terms["body_only_token_jaccard_ge_0_25"] = body_only_tok.ge(0.25).fillna(False).to_numpy()
        bool_terms["body_only_char4gram_jaccard_ge_0_25"] = body_only_c4.ge(0.25).fillna(False).to_numpy()
        bool_terms["path_token_jaccard_combined_ge_0_25"] = path_comb.ge(0.25).fillna(False).to_numpy()
        bool_terms["url_path_token_jaccard_ge_0_25"] = url_path.ge(0.25).fillna(False).to_numpy()
        bool_terms = extend_bool_terms_for_low_band_channels(
            bool_terms,
            df_eval,
            nodes_by_email=nodes_by_email,
            evidence_index=evidence_index,
        )
        extra_joint_rules = tuple(extra_joint_rules) + low_band_twohop_joint_rule_names()

    def _eval_rule(expr: str) -> np.ndarray:
        toks = expr.split("_AND_")
        if not toks:
            return np.zeros(n, dtype=bool)
        out_mask = np.ones(n, dtype=bool)
        for tok in toks:
            neg = tok.startswith("NOT_")
            key = tok[4:] if neg else tok
            base = bool_terms.get(key)
            if base is None:
                return np.zeros(n, dtype=bool)
            out_mask = out_mask & (~base if neg else base)
        return out_mask

    rule_names = list(BINARY_CONDITION_RULES_DEFAULT) + list(extra_joint_rules)
    condition_defs: list[tuple[str, np.ndarray]] = [(name, _eval_rule(name)) for name in rule_names]

    def _append_cmp(section: str, name: str, cmp: dict[str, Any]) -> None:
        out_rows.append(
            _joint_row(
                gt_path=gt_path,
                band_kind=band_kind,
                analysis_section=section,
                condition_name=name,
                cmp=cmp,
                value_key_same=value_key_same,
                value_key_cross=value_key_cross,
                focus=focus,
            )
        )

    bin_out: dict[str, Any] = {}
    for name, cond in condition_defs:
        cmp = _cmp_from_masks(
            cond_same=cond,
            base_same=same_band,
            cond_cross=cond,
            base_cross=cross_band,
            value_key_same=value_key_same,
            value_key_cross=value_key_cross,
        )
        bin_out[name] = cmp
        _append_cmp("binary_joint_comparisons", name, cmp)

    # Bucketed semantic analysis
    bucket_defs: list[tuple[str, np.ndarray]] = []
    for bname, low, high in SEMANTIC_BUCKET_RULES_DEFAULT:
        mask = np.ones(n, dtype=bool)
        if low is not None:
            mask = mask & sem.ge(float(low)).fillna(False).to_numpy()
        if high is not None:
            mask = mask & sem.lt(float(high)).fillna(False).to_numpy()
        bucket_defs.append((bname, mask))
    sem_out: dict[str, Any] = {}
    for bname, bmask in bucket_defs:
        cmp_base = _cmp_from_masks(
            cond_same=bmask,
            base_same=same_band,
            cond_cross=bmask,
            base_cross=cross_band,
            value_key_same=value_key_same,
            value_key_cross=value_key_cross,
        )
        sem_out[bname] = {"bucket": cmp_base}
        _append_cmp("semantic_bucket_analysis", bname, cmp_base)
        crossed = [
            (f"{bname}_AND_shared_sender", bmask & has_shared_sender),
            (f"{bname}_AND_NOT_shared_sender", bmask & ~has_shared_sender),
            (f"{bname}_AND_from_2hop", bmask & f2),
            (f"{bname}_AND_NOT_from_2hop", bmask & ~f2),
        ]
        for cname, cmask in crossed:
            cmp = _cmp_from_masks(
                cond_same=cmask,
                base_same=same_band,
                cond_cross=cmask,
                base_cross=cross_band,
                value_key_same=value_key_same,
                value_key_cross=value_key_cross,
            )
            sem_out[bname][cname] = cmp
            _append_cmp("semantic_bucket_analysis", cname, cmp)

    # Candidate rule templates
    rule_defs: list[tuple[str, np.ndarray]] = []
    for rname in CANDIDATE_RULES_DEFAULT:
        expr = rname.split("__", 1)[1] if "__" in rname else rname
        if "semantic_ge_0_93" in expr:
            dynamic = sem.ge(0.93).fillna(False).to_numpy()
            expr = expr.replace("semantic_ge_0_93", "dynamic_semantic_ge_0_93")
            bool_terms["dynamic_semantic_ge_0_93"] = dynamic
        rule_defs.append((rname, _eval_rule(expr)))
    rule_out: dict[str, Any] = {}
    for rname, rmask in rule_defs:
        cmp = _cmp_from_masks(
            cond_same=rmask,
            base_same=same_band,
            cond_cross=rmask,
            base_cross=cross_band,
            value_key_same=value_key_same,
            value_key_cross=value_key_cross,
        )
        rule_out[rname] = cmp
        _append_cmp("candidate_rule_analysis", rname, cmp)

    ranked = [r for r in out_rows if r.get("abs_difference") is not None]
    ranked.sort(key=lambda r: float(r["abs_difference"]), reverse=True)
    ranked_top = ranked[:15]

    def _ranked_entry(r: dict[str, Any], rank: int) -> dict[str, Any]:
        return {
            "rank": rank,
            "analysis_section": r["analysis_section"],
            "condition_name": r["condition_name"],
            "focus": r.get("focus"),
            value_key_same: r.get(value_key_same),
            value_key_cross: r.get(value_key_cross),
            "difference_same_minus_cross": r["difference"],
            "abs_difference": r["abs_difference"],
            "enrichment_same_over_cross": r["enrichment"],
            "favors": r.get("favors"),
        }

    ranked_entries = [_ranked_entry(r, i + 1) for i, r in enumerate(ranked_top)]
    favor_same = [e for e in ranked_entries if e.get("favors") == "same_campaign"][:10]
    favor_cross = [e for e in ranked_entries if e.get("favors") == "cross_campaign"][:10]

    out: dict[str, Any] = {
        "gt_path": str(gt_path.resolve()),
        "band_kind": band_kind,
        "focus": focus,
        f"{band_kind}_band_thresholds": band_thresholds,
        "counts": {
            count_same_key: n_same,
            count_cross_key: n_cross,
            count_total_key: n_band,
        },
        "binary_joint_comparisons": bin_out,
        "semantic_bucket_analysis": sem_out,
        "candidate_rule_analysis": rule_out,
        "ranked_joint_separators_top15": ranked_entries,
        "ranked_joint_separators_favoring_same_top10": favor_same,
        "ranked_joint_separators_favoring_cross_top10": favor_cross,
    }
    # Backward-compatible keys for low-band consumers.
    if band_kind == "low":
        out["low_band_thresholds"] = band_thresholds
        out["counts"]["n_same_campaign_low_score"] = n_same
        out["counts"]["n_cross_campaign_low_score"] = n_cross
        out["counts"]["n_total_low_band_gt_covered_pairs"] = n_band
        for e in ranked_entries:
            e["same_low_value"] = e.get(value_key_same)
            e["cross_low_value"] = e.get(value_key_cross)
        if twohop_channel_analysis:
            out["twohop_channel_joint_comparisons"] = {
                k: v for k, v in bin_out.items() if str(k).startswith("twohop_via_")
            }
            out["twohop_channel_joint_ranked_top15"] = [
                e for e in ranked_entries if str(e.get("condition_name") or "").startswith("twohop_via_")
            ][:15]

    if include_recommendations:
        out["recommendations"] = _generate_high_band_recommendations(
            marginal=marginal_sep,
            joint=out,
        )
    return out, out_rows


def _build_low_band_joint_separator_for_gt(
    *,
    gt_path: Path,
    df_eval: pd.DataFrame,
    same_low_mask_eval: np.ndarray,
    cross_low_mask_eval: np.ndarray,
    low_max: float,
    nodes_by_email: dict[str, dict[str, set[str]]],
    evidence_index: dict[tuple[str, str], list[dict[str, Any]]] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    return _build_band_joint_separator_for_gt(
        gt_path=gt_path,
        df_eval=df_eval,
        same_band_mask_eval=same_low_mask_eval,
        cross_band_mask_eval=cross_low_mask_eval,
        band_kind="low",
        band_thresholds={"min_inclusive": 0.0, "max_inclusive": float(low_max)},
        nodes_by_email=nodes_by_email,
        value_key_same="same_low_value",
        value_key_cross="cross_low_value",
        focus="unlabeled_only",
        twohop_channel_analysis=True,
        evidence_index=evidence_index,
    )


def _build_high_band_joint_separator_for_gt(
    *,
    gt_path: Path,
    df_eval: pd.DataFrame,
    same_high_unl_mask_eval: np.ndarray,
    cross_high_unl_mask_eval: np.ndarray,
    high_min: float,
    nodes_by_email: dict[str, dict[str, set[str]]],
    marginal_sep: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    return _build_band_joint_separator_for_gt(
        gt_path=gt_path,
        df_eval=df_eval,
        same_band_mask_eval=same_high_unl_mask_eval,
        cross_band_mask_eval=cross_high_unl_mask_eval,
        band_kind="high",
        band_thresholds={"min_exclusive": float(high_min), "max_inclusive": 1.0},
        nodes_by_email=nodes_by_email,
        value_key_same="same_high_value",
        value_key_cross="cross_high_value",
        focus="unlabeled_only",
        extra_joint_rules=HIGH_BAND_EXTRA_JOINT_RULES,
        marginal_sep=marginal_sep,
        include_recommendations=True,
    )


_PAIR_SHARED_CHANNEL_DEFS: tuple[tuple[str, str], ...] = (
    ("sender_set", "sender"),
    ("sender_email_domain_set", "sender_domain"),
    ("url_set", "url"),
    ("stem_set", "stem"),
    ("domain_set", "domain"),
    ("attachment_set", "attachment"),
    ("html_structure_fingerprint_set", "html_fp"),
    ("received_host_set", "received_host"),
)

_INSPECTION_PROVENANCE_COLS: tuple[str, ...] = (
    "from_seed",
    "from_semantic",
    "from_2hop",
    "from_component",
    "from_rare_artifact",
    "same_seed_component_flag",
    "cross_seed_component_flag",
)

_INSPECTION_FEATURE_COLS: tuple[str, ...] = (
    "semantic_cosine_max",
    "component_cosine_max",
    "twohop_rarity_max",
    "rare_artifact_rarity_max",
    "time_gap_seconds_min",
    "source_count",
    "body_token_jaccard",
    "body_char4gram_jaccard",
    "body_only_token_jaccard",
    "body_only_char4gram_jaccard",
    "path_token_jaccard_combined",
    "url_path_token_jaccard",
    "stem_path_token_jaccard",
    "sender_localpart_norm_jaccard",
)

_PAIR_CARD_METRIC_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "Body similarity",
        (
            "body_token_jaccard",
            "body_char4gram_jaccard",
            "body_only_token_jaccard",
            "body_only_char4gram_jaccard",
        ),
    ),
    (
        "Path similarity",
        (
            "path_token_jaccard_combined",
            "url_path_token_jaccard",
            "stem_path_token_jaccard",
        ),
    ),
    (
        "Sender / semantic / support",
        (
            "sender_localpart_norm_jaccard",
            "semantic_cosine",
            "semantic_cosine_max",
            "source_count",
            "n_shared_core_channels",
        ),
    ),
    (
        "Shared artifact counts",
        (
            "shared_sender_count",
            "shared_stem_count",
            "shared_url_count",
            "shared_attachment_count",
            "shared_sender_domain_count",
            "shared_domain_count",
        ),
    ),
    (
        "Latent diagnostics",
        ("gnn_encoder_cosine", "gnn_encoder_l2", "static_subj_body_cosine"),
    ),
    (
        "Seed / component context",
        ("same_seed_component_flag", "cross_seed_component_flag"),
    ),
)

_TWOHOP_VIA_COL_PREFIX = "twohop_via_"


def _format_shared_values(values: set[str], *, max_items: int = 40) -> str:
    items = sorted({str(v).strip() for v in values if str(v).strip()})
    if not items:
        return ""
    if len(items) <= max_items:
        return "|".join(items)
    head = "|".join(items[:max_items])
    return f"{head}|...(+{len(items) - max_items}_more)"


def _pair_shared_evidence_detail(
    email_i: str,
    email_j: str,
    nodes_by_email: dict[str, dict[str, set[str]]],
) -> dict[str, Any]:
    na = nodes_by_email.get(email_i)
    nb = nodes_by_email.get(email_j)
    out: dict[str, Any] = {"anchor_context_missing": bool(na is None or nb is None)}
    n_core = 0
    for anchor_col, short in _PAIR_SHARED_CHANNEL_DEFS:
        bool_col = f"has_shared_{short}"
        count_col = f"shared_{short}_count"
        values_col = f"shared_{short}_values"
        if na is None or nb is None:
            out[bool_col] = False
            out[count_col] = 0
            out[values_col] = ""
            continue
        inter = (na.get(anchor_col) or set()) & (nb.get(anchor_col) or set())
        shared = len(inter) > 0
        out[bool_col] = shared
        out[count_col] = int(len(inter))
        out[values_col] = _format_shared_values(inter)
        if shared and short in {
            "sender",
            "sender_domain",
            "url",
            "stem",
            "domain",
            "attachment",
            "html_fp",
            "received_host",
        }:
            n_core += 1
    out["n_shared_core_channels"] = int(n_core)
    return out


def _provenance_combo_from_row(row: pd.Series) -> str:
    parts: list[str] = []
    for col in _INSPECTION_PROVENANCE_COLS:
        if not col.startswith("from_"):
            continue
        if bool(row.get(col, False)):
            parts.append(col.replace("from_", ""))
    return "+".join(parts) if parts else "none"


def _build_high_band_inspection_dataframe(
    *,
    df_eval: pd.DataFrame,
    row_mask: np.ndarray,
    gt_path: Path,
    label_map: dict[str, Any],
    gt_relation: str,
    nodes_by_email: dict[str, dict[str, set[str]]],
    cohort: str,
) -> pd.DataFrame:
    """One row per GT-covered pair in the high-score unlabeled cohort."""
    mask = np.asarray(row_mask, dtype=bool)
    if mask.size != len(df_eval):
        raise ValueError("row_mask length must match df_eval")
    sub = df_eval.loc[mask].copy()
    if sub.empty:
        return pd.DataFrame()

    rows_out: list[dict[str, Any]] = []
    for _, r in sub.iterrows():
        ei = str(r["email_i"])
        ej = str(r["email_j"])
        rec: dict[str, Any] = {
            "gt_path": str(gt_path.resolve()),
            "gt_name": gt_path.name,
            "cohort": cohort,
            "email_i": ei,
            "email_j": ej,
            "score": float(r["score"]) if pd.notna(r["score"]) else None,
            "gt_relation": gt_relation,
            "gt_campaign_i": label_map.get(ei),
            "gt_campaign_j": label_map.get(ej),
        }
        if "pair_status" in sub.columns:
            rec["pair_status"] = str(r.get("pair_status") or "")
        for col in _INSPECTION_PROVENANCE_COLS:
            if col in sub.columns:
                rec[col] = bool(r.get(col, False))
        for col in _INSPECTION_FEATURE_COLS:
            if col in sub.columns:
                v = pd.to_numeric(r.get(col), errors="coerce")
                rec[col] = (float(v) if pd.notna(v) else None)
        rec.update(_pair_shared_evidence_detail(ei, ej, nodes_by_email))
        if "n_shared_core_channels" in sub.columns:
            v = pd.to_numeric(r.get("n_shared_core_channels"), errors="coerce")
            if pd.notna(v):
                rec["n_shared_core_channels_dataset"] = int(v)
        rec["provenance_combo"] = _provenance_combo_from_row(r)
        rows_out.append(rec)

    df_out = pd.DataFrame(rows_out)
    preferred = [
        "gt_path",
        "gt_name",
        "cohort",
        "email_i",
        "email_j",
        "score",
        "gt_relation",
        "gt_campaign_i",
        "gt_campaign_j",
        "pair_status",
        *_INSPECTION_PROVENANCE_COLS,
        *_INSPECTION_FEATURE_COLS,
        "n_shared_core_channels",
        "n_shared_core_channels_dataset",
        "anchor_context_missing",
    ]
    for _anchor_col, short in _PAIR_SHARED_CHANNEL_DEFS:
        preferred.extend(
            [f"has_shared_{short}", f"shared_{short}_count", f"shared_{short}_values"]
        )
    preferred.append("provenance_combo")
    ordered = [c for c in preferred if c in df_out.columns]
    rest = [c for c in df_out.columns if c not in ordered]
    return df_out[ordered + rest].sort_values("score", ascending=False, na_position="last")


def _resolve_default_misp_json_path(project_root: Path) -> Path | None:
    cfg_path = project_root / "pipeline_config.json"
    if cfg_path.is_file():
        with open(cfg_path, encoding="utf-8") as f:
            cfg = json.load(f)
        for block_key in ("datasets", "graph", "preprocessing"):
            block = cfg.get(block_key) or {}
            raw = block.get("misp_json_path")
            if not raw:
                continue
            p = Path(str(raw))
            if not p.is_absolute():
                p = project_root / p
            if p.is_file():
                return p.resolve()
    fallback = project_root / "data/misp/incidents-lake-misp.dedup_task_identity.json"
    return fallback.resolve() if fallback.is_file() else None


def _load_email_text_catalog(
    *,
    project_root: Path,
    misp_json_path: Path | None,
    misp_translated_json_path: Path | None,
) -> tuple[dict[str, dict[str, str]], dict[str, Any]]:
    """Load subject/body by external_id from MISP JSON (+ optional translated sidecar)."""
    analysis_scripts = (project_root / "analysis" / "scripts").resolve()
    if str(analysis_scripts) not in sys.path:
        sys.path.insert(0, str(analysis_scripts))
    from misp_email_text_catalog import (
        load_misp_subject_body_by_external_id,
        load_translated_email_text_by_external_id,
    )

    meta: dict[str, Any] = {"status": "skipped"}
    if misp_json_path is None or not misp_json_path.is_file():
        meta["reason"] = f"misp_json_not_found:{misp_json_path}"
        return {}, meta

    catalog = load_misp_subject_body_by_external_id(misp_json_path, project_root=project_root)
    meta = {
        "status": "ok",
        "misp_json_path": str(misp_json_path.resolve()),
        "n_emails_with_text": int(len(catalog)),
    }
    if misp_translated_json_path is not None and misp_translated_json_path.is_file():
        translated = load_translated_email_text_by_external_id(misp_translated_json_path)
        meta["misp_translated_json_path"] = str(misp_translated_json_path.resolve())
        meta["n_translated_emails"] = int(len(translated))
        for eid, tr in translated.items():
            base = catalog.get(eid) or {"subject": "", "body": ""}
            catalog[eid] = {
                "subject": str(tr.get("subject") or "").strip() or str(base.get("subject") or ""),
                "body": str(tr.get("body") or "").strip() or str(base.get("body") or ""),
            }
    return catalog, meta


def _text_preview(text: str, *, max_chars: int) -> str:
    s = " ".join(str(text or "").replace("\r\n", "\n").replace("\r", "\n").split())
    if max_chars <= 0 or len(s) <= max_chars:
        return s
    return s[: max_chars - 3] + "..."


def _text_as_line_list(text: str, wrap_width: int) -> list[str]:
    import textwrap

    raw = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    if wrap_width <= 0:
        return raw.splitlines() or [""]
    lines: list[str] = []
    for para in raw.split("\n"):
        if not para.strip():
            lines.append("")
        else:
            lines.extend(textwrap.wrap(para, width=wrap_width) or [""])
    return lines if lines else [""]


def _classify_high_score_fp_regime(row: pd.Series) -> str:
    domain_vals = str(row.get("shared_domain_values") or "")
    if bool(row.get("has_shared_domain")) and "hxxps:" in domain_vals:
        return "shared_domain_hxxps"
    n_core = int(row.get("n_shared_core_channels") or 0)
    if n_core == 0 and bool(row.get("from_semantic")):
        return "semantic_only"
    if n_core == 0:
        return "no_shared_core_artifacts"
    return "other_shared_artifacts"


def _classify_low_band_review_regime(row: pd.Series) -> str:
    """
    Sidebar filter key for low-band manual review HTML.

    Primary split: same-campaign vs cross-campaign unlabeled low-score pairs.
    Sub-suffix marks exact-zero scores (common collapse mode).
    """
    rel = str(row.get("gt_relation") or "")
    if rel == "same_campaign":
        base = "same_campaign_low"
    elif rel == "cross_campaign":
        base = "cross_campaign_low"
    else:
        base = "unknown_low"
    score = pd.to_numeric(row.get("score"), errors="coerce")
    if pd.notna(score) and float(score) <= 0.0:
        return f"{base}__score_zero"
    return base


def _low_band_review_prompt(row: pd.Series) -> str:
    rel = str(row.get("gt_relation") or "")
    if rel == "same_campaign":
        return (
            "GT same campaign but model score is in the low band. "
            "What signal could pull this pair up without lifting cross-campaign low pairs?"
        )
    if rel == "cross_campaign":
        return (
            "GT cross campaign with low model score (desired). "
            "Confirm this pair should stay near the bottom."
        )
    return "Low-score unlabeled pair — inspect whether the score matches GT."


def _shared_artifacts_brief(row: pd.Series) -> str:
    """Legacy brief; prefer shared_evidence_brief when admitting evidence is attached."""
    brief = str(row.get("shared_evidence_brief") or "").strip()
    if brief:
        return brief
    parts: list[str] = []
    for _anchor_col, short in _PAIR_SHARED_CHANNEL_DEFS:
        if bool(row.get(f"has_shared_{short}")):
            vals = str(row.get(f"shared_{short}_values") or "").strip()
            parts.append(f"{short}={vals}" if vals else short)
    return "; ".join(parts) if parts else "none"


def _warning_badges_html(flags_raw: Any) -> str:
    flags = str(flags_raw or "").strip()
    if not flags:
        return ""
    badges = [
        f'<span class="badge warn-flag">{html.escape(w.strip())}</span>'
        for w in flags.split("|")
        if w.strip()
    ]
    return " ".join(badges)


def _evidence_list_html(block_label: str, lines_raw: str) -> str:
    lines = [ln.strip() for ln in str(lines_raw or "").split("\n") if ln.strip()]
    if not lines:
        return ""
    items = "".join(f"<li>{html.escape(ln)}</li>" for ln in lines)
    return (
        '<div class="evidence-block">'
        f'<div class="evidence-label">{html.escape(block_label)}</div>'
        f'<ul class="evidence-list">{items}</ul>'
        "</div>"
    )


def _admitting_evidence_section_html(row: pd.Series) -> str:
    direct_blk = _evidence_list_html("Direct shared artifacts", str(row.get("direct_shared_evidence_lines") or ""))
    admit_blk = _evidence_list_html(
        "Candidate-family admitting evidence", str(row.get("admitting_evidence_lines") or "")
    )
    if not direct_blk and not admit_blk:
        prov = html.escape(str(row.get("provenance_combo") or "none"))
        return (
            '<section class="admitting-evidence">'
            "<h4>Shared evidence / admitting evidence</h4>"
            f'<p class="evidence-empty">No direct overlap or source rows loaded (provenance: {prov}).</p>'
            "</section>"
        )
    return (
        '<section class="admitting-evidence">'
        "<h4>Shared evidence / admitting evidence</h4>"
        f"{direct_blk}{admit_blk}"
        "</section>"
    )


def _resolve_embeddings_json_for_review(project_root: Path) -> Path | None:
    p = (project_root / "core" / "utils" / "embeddings" / "output" / "embeddings.json").resolve()
    return p if p.is_file() else None


def _load_embeddings_by_external_id_review(path: Path) -> dict[str, np.ndarray]:
    """subject+body concat vectors (same layout as semantic_cosine_max in pipeline)."""
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    by_key = payload.get("by_key")
    if not isinstance(by_key, dict):
        return {}
    id_to_emb: dict[str, np.ndarray] = {}
    for k, v in by_key.items():
        if not isinstance(v, dict):
            continue
        subj = np.asarray(v.get("subj") or [], dtype=np.float64).reshape(-1)
        body = np.asarray(v.get("body") or [], dtype=np.float64).reshape(-1)
        if subj.size == 0 and body.size == 0:
            continue
        eid = str(v.get("external_id") or k).strip()
        if not eid:
            continue
        id_to_emb[eid] = np.concatenate([subj, body], axis=0)
    return id_to_emb


def _cosine_normalized(a: np.ndarray, b: np.ndarray) -> float | None:
    if a.shape != b.shape or a.size == 0:
        return None
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na <= 0.0 or nb <= 0.0:
        return None
    return float(np.dot(a, b) / (na * nb))


def _inject_semantic_cosine_for_manual_review(df: pd.DataFrame) -> pd.DataFrame:
    """
    Populate semantic_cosine_for_display: pair-training semantic_cosine_max when present,
    else cosine from embeddings cache (same subj+body vectors used to build reciprocal edges).
    """
    if df.empty:
        return df
    out = df.copy()
    project_root = Path(__file__).resolve().parents[2]
    tbl = (
        pd.to_numeric(out["semantic_cosine_max"], errors="coerce")
        if "semantic_cosine_max" in out.columns
        else pd.Series(np.nan, index=out.index, dtype=np.float64)
    )

    emb_path = _resolve_embeddings_json_for_review(project_root)
    id_to_emb: dict[str, np.ndarray] = {}
    if emb_path is not None:
        try:
            id_to_emb = _load_embeddings_by_external_id_review(emb_path)
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            id_to_emb = {}

    emb_list: list[float | None] = []
    if id_to_emb:
        for _, r in out.iterrows():
            vi = id_to_emb.get(str(r["email_i"]))
            vj = id_to_emb.get(str(r["email_j"]))
            if vi is None or vj is None:
                emb_list.append(None)
            else:
                emb_list.append(_cosine_normalized(vi, vj))
    else:
        emb_list = [None] * len(out)

    disp = tbl.copy().astype(np.float64)
    src_raw: list[str] = []
    for i in range(len(out)):
        tval = tbl.iloc[i]
        emb_c = emb_list[i]
        if pd.notna(tval):
            src_raw.append("pair_training_csv")
            disp.iloc[i] = float(tval)
        elif emb_c is not None and np.isfinite(emb_c):
            src_raw.append("embedding_cache")
            disp.iloc[i] = float(emb_c)
        else:
            src_raw.append("missing")
            disp.iloc[i] = np.nan

    out["embedding_cosine_subj_body_cache"] = emb_list
    out["semantic_cosine_for_display"] = disp
    out["semantic_cosine_source"] = src_raw
    # Review HTML/CSV metric chips read ``semantic_cosine``; pair CSV often leaves
    # ``semantic_cosine_max`` null for 2-hop-only rows while the embed cache has values.
    out["semantic_cosine"] = disp
    meta_path = emb_path.resolve() if emb_path is not None else None
    out.attrs["semantic_cosine_injection"] = {
        "embeddings_json": str(meta_path) if meta_path else None,
        "n_rows_from_pair_csv": int((tbl.notna()).sum()),
        "n_rows_from_embedding_cache": int(sum(1 for s in src_raw if s == "embedding_cache")),
        "n_rows_missing": int(sum(1 for s in src_raw if s == "missing")),
    }
    return out


def _enrich_pairs_with_email_text(
    df_pairs: pd.DataFrame,
    *,
    email_text_by_eid: dict[str, dict[str, str]],
    preview_chars: int,
    regime_fn: Any | None = None,
    review_prompt_fn: Any | None = None,
) -> pd.DataFrame:
    if df_pairs.empty:
        return df_pairs.copy()

    classify_regime = regime_fn or _classify_high_score_fp_regime
    out = df_pairs.copy()
    regimes: list[str] = []
    briefs: list[str] = []
    review_notes: list[str] = []
    subj_i: list[str] = []
    subj_j: list[str] = []
    body_prev_i: list[str] = []
    body_prev_j: list[str] = []
    ts_i: list[str] = []
    ts_j: list[str] = []
    miss_i: list[bool] = []
    miss_j: list[bool] = []

    for _, r in out.iterrows():
        regimes.append(classify_regime(r))
        briefs.append(_shared_artifacts_brief(r))
        if review_prompt_fn is not None:
            review_notes.append(str(review_prompt_fn(r)))
        ei = str(r["email_i"])
        ej = str(r["email_j"])
        ti = email_text_by_eid.get(ei) or {}
        tj = email_text_by_eid.get(ej) or {}
        si = str(ti.get("subject") or "")
        sj = str(tj.get("subject") or "")
        bi = str(ti.get("body") or "")
        bj = str(tj.get("body") or "")
        subj_i.append(si)
        subj_j.append(sj)
        body_prev_i.append(_text_preview(bi, max_chars=preview_chars))
        body_prev_j.append(_text_preview(bj, max_chars=preview_chars))
        ts_i.append(str(ti.get("timestamp_utc") or ""))
        ts_j.append(str(tj.get("timestamp_utc") or ""))
        miss_i.append(ei not in email_text_by_eid)
        miss_j.append(ej not in email_text_by_eid)

    out["fp_regime"] = regimes
    out["shared_artifacts_brief"] = briefs
    if review_prompt_fn is not None:
        out["gt_review_note"] = review_notes
    out["email_i_subject"] = subj_i
    out["email_j_subject"] = subj_j
    out["email_i_body_preview"] = body_prev_i
    out["email_j_body_preview"] = body_prev_j
    out["email_i_timestamp_utc"] = ts_i
    out["email_j_timestamp_utc"] = ts_j
    out["email_i_text_missing"] = miss_i
    out["email_j_text_missing"] = miss_j

    out = _inject_semantic_cosine_for_manual_review(out)

    front = [
        "fp_regime",
        "shared_artifacts_brief",
        "email_i_subject",
        "email_j_subject",
        "email_i_body_preview",
        "email_j_body_preview",
        "email_i_timestamp_utc",
        "email_j_timestamp_utc",
        "email_i_text_missing",
        "email_j_text_missing",
    ]
    rest = [c for c in out.columns if c not in front]
    insert_at = rest.index("score") + 1 if "score" in rest else 0
    ordered = rest[:insert_at] + front + rest[insert_at:]
    return out[ordered]


def _write_pairs_for_review_jsonl(
    df_pairs: pd.DataFrame,
    *,
    out_path: Path,
    email_text_by_eid: dict[str, dict[str, str]],
    wrap_width: int,
) -> None:
    with out_path.open("w", encoding="utf-8") as f:
        for pair_idx, (_, r) in enumerate(df_pairs.iterrows()):
            ei = str(r["email_i"])
            ej = str(r["email_j"])
            ti = email_text_by_eid.get(ei) or {"subject": "", "body": ""}
            tj = email_text_by_eid.get(ej) or {"subject": "", "body": ""}
            rec: dict[str, Any] = {
                "pair_index": int(pair_idx),
                "score": r.get("score"),
                "gt_relation": r.get("gt_relation"),
                "gt_campaign_i": r.get("gt_campaign_i"),
                "gt_campaign_j": r.get("gt_campaign_j"),
                "fp_regime": r.get("fp_regime"),
                "provenance_combo": r.get("provenance_combo"),
                "semantic_cosine_max": r.get("semantic_cosine_max"),
                "semantic_cosine_for_display": r.get("semantic_cosine_for_display"),
                "semantic_cosine_source": r.get("semantic_cosine_source"),
                "embedding_cosine_subj_body_cache": r.get("embedding_cosine_subj_body_cache"),
                "shared_artifacts_brief": r.get("shared_artifacts_brief"),
                "shared_evidence_brief": r.get("shared_evidence_brief"),
                "direct_shared_evidence_lines": r.get("direct_shared_evidence_lines"),
                "admitting_evidence_lines": r.get("admitting_evidence_lines"),
                "admitting_evidence_json": r.get("admitting_evidence_json"),
                "inspection_warning_flags": r.get("inspection_warning_flags"),
                "from_semantic": r.get("from_semantic"),
                "from_2hop": r.get("from_2hop"),
                "from_rare_artifact": r.get("from_rare_artifact"),
                "from_component": r.get("from_component"),
                "from_seed": r.get("from_seed"),
                "source_count": r.get("source_count"),
                "twohop_rarity_max": r.get("twohop_rarity_max"),
                "rare_artifact_rarity_max": r.get("rare_artifact_rarity_max"),
                "component_cosine_max": r.get("component_cosine_max"),
                "n_shared_core_channels": r.get("n_shared_core_channels"),
                "email_i": {
                    "external_id": ei,
                    "timestamp_utc": str(ti.get("timestamp_utc") or ""),
                    "subject_lines": _text_as_line_list(str(ti.get("subject") or ""), wrap_width),
                    "body_lines": _text_as_line_list(str(ti.get("body") or ""), wrap_width),
                },
                "email_j": {
                    "external_id": ej,
                    "timestamp_utc": str(tj.get("timestamp_utc") or ""),
                    "subject_lines": _text_as_line_list(str(tj.get("subject") or ""), wrap_width),
                    "body_lines": _text_as_line_list(str(tj.get("body") or ""), wrap_width),
                },
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _format_email_body_html(body: str) -> str:
    """Render body with visible section breaks for duplicated MIME parts."""
    raw = str(body or "")
    if not raw.strip():
        return '<p class="empty">(no body text)</p>'
    parts = re.split(r"\s*---\s*mail_boundary\s*---\s*", raw, flags=re.IGNORECASE)
    chunks: list[str] = []
    for i, part in enumerate(parts):
        escaped = html.escape(part.strip())
        if not escaped:
            continue
        if i > 0:
            chunks.append('<div class="mail-boundary">— mail part boundary —</div>')
        chunks.append(f'<pre class="email-body">{escaped}</pre>')
    return "\n".join(chunks) if chunks else '<p class="empty">(no body text)</p>'


def _email_pane_html(
    *,
    label: str,
    external_id: str,
    campaign: Any,
    email_text_by_eid: dict[str, dict[str, str]],
) -> str:
    rec = email_text_by_eid.get(external_id) or {}
    subject = html.escape(str(rec.get("subject") or "(no subject)"))
    body_html = _format_email_body_html(str(rec.get("body") or ""))
    eid_full = html.escape(external_id)
    eid_short = html.escape(external_id if len(external_id) <= 72 else external_id[:69] + "...")
    camp = html.escape(str(campaign) if campaign is not None else "?")
    missing = external_id not in email_text_by_eid
    missing_badge = '<span class="badge warn">text missing</span>' if missing else ""
    ts = html.escape(str(rec.get("timestamp_utc") or "").strip() or "(timestamp unknown)")
    return f"""
    <article class="email-pane">
      <header class="email-pane-header">
        <h3>{html.escape(label)} <span class="campaign">GT campaign {camp}</span> {missing_badge}</h3>
        <p class="timestamp"><strong>Sent</strong> {ts}</p>
        <p class="eid" title="{eid_full}">{eid_short}</p>
      </header>
      <div class="subject-block">
        <div class="label">Subject</div>
        <div class="subject">{subject}</div>
      </div>
      <div class="body-block">
        <div class="label">Body</div>
        {body_html}
      </div>
    </article>
    """


def _format_pair_metric_value(row: pd.Series, col: str, *, prec: int = 3) -> str:
    if col == "semantic_cosine" and col not in row.index:
        for fallback in ("semantic_cosine_for_display", "semantic_cosine_max"):
            if fallback in row.index:
                col = fallback
                break
    if col not in row.index:
        return "—"
    val = row.get(col)
    if col.endswith("_flag") or isinstance(val, (bool, np.bool_)):
        if pd.isna(val):
            return "—"
        return "yes" if bool(val) else "no"
    num = pd.to_numeric(val, errors="coerce")
    if pd.isna(num):
        return "—"
    return f"{float(num):.{prec}f}"


def _twohop_attribution_summary(row: pd.Series) -> str:
    parts: list[str] = []
    for col in sorted(row.index.astype(str)):
        if not col.startswith(_TWOHOP_VIA_COL_PREFIX):
            continue
        v = row.get(col)
        if pd.isna(v):
            continue
        if bool(v):
            parts.append(col.replace(_TWOHOP_VIA_COL_PREFIX, ""))
    if not parts:
        return "none"
    return ", ".join(parts[:8]) + ("…" if len(parts) > 8 else "")


def _pair_metric_groups_html(row: pd.Series) -> str:
    blocks: list[str] = []
    for title, cols in _PAIR_CARD_METRIC_GROUPS:
        chips = "".join(
            f'<span class="metric-chip"><strong>{html.escape(c)}</strong> '
            f"{html.escape(_format_pair_metric_value(row, c))}</span>"
            for c in cols
        )
        blocks.append(
            f'<div class="metric-group"><div class="metric-group-title">{html.escape(title)}</div>'
            f'<div class="metric-chips">{chips}</div></div>'
        )
    twohop = _twohop_attribution_summary(row)
    blocks.append(
        '<div class="metric-group"><div class="metric-group-title">2-hop attribution</div>'
        f'<div class="metric-chips"><span class="metric-chip">{html.escape(twohop)}</span></div></div>'
    )
    return f'<div class="metric-groups">{"".join(blocks)}</div>'


def _pair_card_html(
    *,
    pair_idx: int,
    row: pd.Series,
    email_text_by_eid: dict[str, dict[str, str]],
    review_prompt: str = "Do these look like the same phishing campaign?",
    gt_note: str = "labeled cross",
    filter_column: str = "fp_regime",
) -> str:
    row_prompt = str(row.get("gt_review_note") or "").strip()
    if row_prompt:
        review_prompt = row_prompt
    rel = str(row.get("gt_relation") or "").strip()
    if rel:
        gt_note = rel.replace("_", " ")
    regime_raw = str(row.get("fp_regime") or "")
    regime = html.escape(regime_raw)
    score = row.get("score")
    score_s = f"{float(score):.4f}" if score is not None and pd.notna(score) else "?"
    sem = pd.to_numeric(row.get("semantic_cosine_for_display"), errors="coerce")
    if pd.isna(sem):
        sem = pd.to_numeric(row.get("semantic_cosine_max"), errors="coerce")
    sem_s = f"{float(sem):.4f}" if pd.notna(sem) else "?"
    sem_src = str(row.get("semantic_cosine_source") or "")
    sem_src_html = ""
    if sem_src == "embedding_cache":
        sem_src_html = ' <span class="cosine-src">(subj+body embed cache)</span>'
    elif sem_src == "pair_training_csv":
        sem_src_html = ' <span class="cosine-src">(pair CSV)</span>'
    ci = html.escape(str(row.get("gt_campaign_i")))
    cj = html.escape(str(row.get("gt_campaign_j")))
    prov = html.escape(str(row.get("provenance_combo") or ""))
    shared = html.escape(str(row.get("shared_artifacts_brief") or "none"))
    pair_status = html.escape(str(row.get("pair_status") or ""))
    what_high = str(row.get("what_made_it_high") or "").strip()
    likely_tags = str(row.get("likely_explanation_tags") or "").strip()
    failure_mode_html = ""
    if what_high or likely_tags:
        failure_mode_html = (
            '<div class="meta-grid meta-grid-failure">'
            + (f'<span><strong>What made it high</strong> <code>{html.escape(what_high)}</code></span>' if what_high else "")
            + (
                f'<span><strong>Likely tags</strong> <code>{html.escape(likely_tags)}</code></span>'
                if likely_tags
                else ""
            )
            + "</div>"
        )
    tgap = row.get("time_gap_seconds_min")
    tgap_s = (
        f"{float(tgap):.0f}s"
        if tgap is not None and pd.notna(tgap)
        else "?"
    )
    regime_class = re.sub(r"[^a-z0-9_]+", "_", regime_raw.lower()) or "unknown"
    filter_value = str(row.get(filter_column) or regime_raw or "unknown")
    pane_i = _email_pane_html(
        label="Email A",
        external_id=str(row["email_i"]),
        campaign=row.get("gt_campaign_i"),
        email_text_by_eid=email_text_by_eid,
    )
    pane_j = _email_pane_html(
        label="Email B",
        external_id=str(row["email_j"]),
        campaign=row.get("gt_campaign_j"),
        email_text_by_eid=email_text_by_eid,
    )
    warn_html = _warning_badges_html(row.get("inspection_warning_flags"))
    ch_badges = str(row.get("twohop_channel_badges") or "").strip()
    if ch_badges:
        ch_html = " ".join(
            f'<span class="badge channel-badge">{html.escape(b.strip())}</span>'
            for b in ch_badges.split("|")
            if b.strip()
        )
        warn_html = f"{warn_html} {ch_html}".strip()
    evidence_html = _admitting_evidence_section_html(row)
    metric_groups_html = _pair_metric_groups_html(row)
    gt_rel_attr = html.escape(str(row.get("gt_relation") or ""))
    bucket_attr = ""
    if "same_unlabeled_bucket" in row.index and pd.notna(row.get("same_unlabeled_bucket")):
        bucket_attr = f' data-bucket="{html.escape(str(row.get("same_unlabeled_bucket")))}"'
    return f"""
    <section class="pair-card regime-{regime_class}" id="pair-{pair_idx}"
      data-filter-value="{html.escape(filter_value)}" data-regime="{html.escape(regime_raw or 'unknown')}"{bucket_attr}
      data-gt-relation="{gt_rel_attr}">
      <header class="pair-header">
        <h2>Pair {pair_idx + 1}</h2>
        <div class="warning-badges">{warn_html}</div>
        <div class="meta-grid meta-grid-core">
          <span><strong>Model score</strong> {score_s}</span>
          <span><strong>Semantic cosine</strong> {sem_s}{sem_src_html}</span>
          <span><strong>GT</strong> campaign {ci} vs {cj} ({html.escape(gt_note)})</span>
          <span><strong>Pair status</strong> {pair_status}</span>
          <span><strong>Time gap</strong> {html.escape(tgap_s)}</span>
          <span><strong>Regime</strong> <code>{regime}</code></span>
          <span><strong>Provenance</strong> {prov}</span>
          <span><strong>Evidence</strong> {shared}</span>
        </div>
        {failure_mode_html}
        {metric_groups_html}
        <p class="review-prompt">{html.escape(review_prompt)}</p>
        {evidence_html}
      </header>
      <div class="pair-columns">
        {pane_i}
        {pane_j}
      </div>
    </section>
    """


def _write_pairs_for_review_html(
    df_pairs: pd.DataFrame,
    *,
    out_path: Path,
    email_text_by_eid: dict[str, dict[str, str]],
    title: str,
    subtitle: str,
    review_prompt: str = "Do these look like the same phishing campaign?",
    gt_note: str = "GT cross-campaign",
    filter_column: str = "fp_regime",
    page_banner_html: str | None = None,
    max_main_width: str = "56rem",
) -> None:
    if df_pairs.empty:
        out_path.write_text(
            "<!DOCTYPE html>\n<html lang=\"en\"><head><meta charset=\"utf-8\"/>"
            f"<title>{html.escape(title)}</title></head>"
            f"<body><h1>{html.escape(title)}</h1>"
            f"<p>{html.escape(subtitle)}</p>"
            "<p><strong>No pairs in this cohort for the current GT file(s) and filters.</strong></p>"
            "</body></html>",
            encoding="utf-8",
        )
        return

    toc_items: list[str] = []
    cards: list[str] = []
    filter_values = sorted(
        {
            str(r.get(filter_column) or r.get("fp_regime") or "unknown")
            for _, r in df_pairs.iterrows()
        }
    )

    for pair_idx, (_, row) in enumerate(df_pairs.iterrows()):
        regime = str(row.get("fp_regime") or "")
        filt_val = str(row.get(filter_column) or regime or "unknown")
        score = row.get("score")
        score_s = f"{float(score):.3f}" if score is not None and pd.notna(score) else "?"
        ci = row.get("gt_campaign_i")
        cj = row.get("gt_campaign_j")
        regime_class = re.sub(r"[^a-z0-9_]+", "_", regime.lower()) or "unknown"
        toc_items.append(
            f'<a class="toc-item regime-{regime_class}" href="#pair-{pair_idx}" '
            f'data-filter-value="{html.escape(filt_val)}">'
            f'<span class="toc-num">{pair_idx + 1}</span>'
            f'<span class="toc-main">score {score_s} · camp {ci} vs {cj}</span>'
            f'<span class="toc-sub">{html.escape(filt_val)}</span></a>'
        )
        cards.append(
            _pair_card_html(
                pair_idx=pair_idx,
                row=row,
                email_text_by_eid=email_text_by_eid,
                review_prompt=review_prompt,
                gt_note=gt_note,
                filter_column=filter_column,
            )
        )

    value_filters = "".join(
        f'<button type="button" class="filter-btn" data-filter="{html.escape(r)}">{html.escape(r)}</button>'
        for r in filter_values
    )
    banner_block = page_banner_html or ""

    doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{html.escape(title)}</title>
  <style>
    :root {{
      --bg: #0f1419;
      --panel: #1a2332;
      --panel2: #243044;
      --text: #e7ecf3;
      --muted: #9aa8bc;
      --accent: #6cb6ff;
      --warn: #f0a020;
      --border: #334155;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Segoe UI", system-ui, sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.45;
    }}
    .layout {{
      display: grid;
      grid-template-columns: 280px 1fr;
      min-height: 100vh;
    }}
    .sidebar {{
      position: sticky;
      top: 0;
      height: 100vh;
      overflow-y: auto;
      padding: 1rem;
      background: var(--panel);
      border-right: 1px solid var(--border);
    }}
    .sidebar h1 {{
      font-size: 1rem;
      margin: 0 0 0.25rem;
    }}
    .sidebar .subtitle {{
      font-size: 0.8rem;
      color: var(--muted);
      margin: 0 0 1rem;
    }}
    .filters {{
      display: flex;
      flex-wrap: wrap;
      gap: 0.35rem;
      margin-bottom: 0.75rem;
    }}
    .filter-btn {{
      font-size: 0.72rem;
      padding: 0.25rem 0.5rem;
      border-radius: 999px;
      border: 1px solid var(--border);
      background: var(--panel2);
      color: var(--text);
      cursor: pointer;
    }}
    .filter-btn.active {{
      background: var(--accent);
      color: #0b1220;
      border-color: var(--accent);
    }}
    .toc {{
      display: flex;
      flex-direction: column;
      gap: 0.35rem;
    }}
    .toc-item {{
      display: grid;
      grid-template-columns: 2rem 1fr;
      grid-template-rows: auto auto;
      gap: 0 0.5rem;
      padding: 0.45rem 0.5rem;
      border-radius: 6px;
      text-decoration: none;
      color: var(--text);
      border: 1px solid transparent;
      font-size: 0.78rem;
    }}
    .toc-item:hover {{ border-color: var(--border); background: var(--panel2); }}
    .toc-num {{ grid-row: 1 / 3; font-weight: 700; color: var(--accent); align-self: center; }}
    .toc-main {{ font-weight: 600; }}
    .toc-sub {{ color: var(--muted); grid-column: 2; }}
    .main {{
      padding: 1.25rem 1.5rem 3rem;
      max-width: {max_main_width};
    }}
    .page-banner {{
      margin: 0 0 1rem;
      padding: 0.65rem 0.85rem;
      background: #1e2a3d;
      border: 1px solid var(--border);
      border-radius: 6px;
      font-size: 0.85rem;
      color: var(--muted);
      max-width: {max_main_width};
    }}
    .page-banner strong {{ color: var(--text); }}
    .metric-groups {{
      display: grid;
      gap: 0.55rem;
      margin: 0.5rem 0 0.75rem;
      max-width: {max_main_width};
    }}
    .metric-group-title {{
      font-size: 0.72rem;
      text-transform: uppercase;
      letter-spacing: 0.04em;
      color: var(--muted);
      margin-bottom: 0.25rem;
    }}
    .metric-chips {{
      display: flex;
      flex-wrap: wrap;
      gap: 0.35rem 0.65rem;
    }}
    .metric-chip {{
      font-size: 0.78rem;
      padding: 0.2rem 0.5rem;
      background: var(--panel2);
      border: 1px solid var(--border);
      border-radius: 4px;
      color: var(--muted);
    }}
    .metric-chip strong {{
      color: var(--text);
      font-weight: 600;
      margin-right: 0.25rem;
    }}
    .meta-grid-core {{
      max-width: {max_main_width};
    }}
    .pair-card {{
      margin-bottom: 2.5rem;
      padding-bottom: 2rem;
      border-bottom: 2px solid var(--border);
    }}
    .pair-card.hidden {{ display: none; }}
    .pair-header h2 {{ margin: 0 0 0.5rem; font-size: 1.25rem; }}
    .meta-grid {{
      display: flex;
      flex-wrap: wrap;
      gap: 0.5rem 1.25rem;
      font-size: 0.85rem;
      color: var(--muted);
      margin-bottom: 0.5rem;
    }}
    .meta-grid strong {{ color: var(--text); }}
    .review-prompt {{
      margin: 0.5rem 0 1rem;
      padding: 0.5rem 0.75rem;
      background: #2a2218;
      border-left: 3px solid var(--warn);
      font-size: 0.9rem;
    }}
    .warning-badges {{
      display: flex;
      flex-wrap: wrap;
      gap: 0.35rem;
      margin-bottom: 0.5rem;
    }}
    .badge.warn-flag {{
      background: #3d2a14;
      color: #f5c878;
      border: 1px solid #8a5a18;
      font-size: 0.72rem;
      padding: 0.15rem 0.45rem;
      border-radius: 4px;
    }}
    .badge.channel-badge {{
      background: #1e2a3d;
      color: #9fd4ff;
      border: 1px solid #3a6ea8;
      font-size: 0.72rem;
      padding: 0.15rem 0.45rem;
      border-radius: 4px;
      margin-right: 0.2rem;
    }}
    .admitting-evidence {{
      margin: 0.75rem 0 1rem;
      padding: 0.65rem 0.85rem;
      background: #141c28;
      border: 1px solid var(--border);
      border-radius: 6px;
      font-size: 0.85rem;
    }}
    .admitting-evidence h4 {{
      margin: 0 0 0.5rem;
      font-size: 0.9rem;
      color: var(--accent);
    }}
    .evidence-block {{
      margin-bottom: 0.5rem;
    }}
    .evidence-label {{
      font-weight: 600;
      color: var(--muted);
      margin-bottom: 0.25rem;
    }}
    .evidence-list {{
      margin: 0;
      padding-left: 1.25rem;
      color: var(--text);
    }}
    .evidence-list li {{
      margin-bottom: 0.2rem;
      word-break: break-word;
    }}
    .evidence-empty {{
      color: var(--muted);
      margin: 0;
    }}
    .pair-columns {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 1rem;
      align-items: start;
    }}
    @media (max-width: 1100px) {{
      .layout {{ grid-template-columns: 1fr; }}
      .sidebar {{ position: relative; height: auto; max-height: 40vh; }}
      .pair-columns {{ grid-template-columns: 1fr; }}
    }}
    .email-pane {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 8px;
      overflow: hidden;
    }}
    .email-pane-header {{
      padding: 0.65rem 0.85rem;
      background: var(--panel2);
      border-bottom: 1px solid var(--border);
    }}
    .email-pane-header h3 {{
      margin: 0;
      font-size: 0.95rem;
    }}
    .campaign {{ color: var(--accent); font-weight: 600; }}
    .timestamp {{ margin: 0.2rem 0 0; font-size: 0.8rem; color: var(--accent); }}
    .eid {{ margin: 0.35rem 0 0; font-size: 0.7rem; color: var(--muted); word-break: break-all; }}
    .badge {{
      font-size: 0.65rem;
      padding: 0.1rem 0.35rem;
      border-radius: 4px;
      vertical-align: middle;
    }}
    .badge.warn {{ background: #4a3520; color: var(--warn); }}
    .subject-block, .body-block {{ padding: 0.75rem 0.85rem; }}
    .label {{
      font-size: 0.7rem;
      text-transform: uppercase;
      letter-spacing: 0.04em;
      color: var(--muted);
      margin-bottom: 0.35rem;
    }}
    .subject {{
      font-size: 1rem;
      font-weight: 600;
      white-space: pre-wrap;
    }}
    .email-body {{
      margin: 0;
      white-space: pre-wrap;
      word-break: break-word;
      font-family: "Cascadia Code", "Consolas", monospace;
      font-size: 0.78rem;
      line-height: 1.5;
      max-height: 520px;
      overflow: auto;
      background: #0d1117;
      padding: 0.65rem;
      border-radius: 4px;
      border: 1px solid var(--border);
    }}
    .mail-boundary {{
      margin: 0.75rem 0;
      padding: 0.35rem 0.5rem;
      text-align: center;
      font-size: 0.72rem;
      color: var(--warn);
      border: 1px dashed var(--warn);
      border-radius: 4px;
    }}
    .empty {{ color: var(--muted); font-style: italic; }}
    .cosine-src {{ font-size: 0.75rem; color: var(--muted); font-weight: normal; }}
  </style>
</head>
<body>
  <div class="layout">
    <aside class="sidebar">
      <h1>{html.escape(title)}</h1>
      <p class="subtitle">{html.escape(subtitle)}</p>
      <div class="filters">
        <button type="button" class="filter-btn active" data-filter="all">all</button>
        {value_filters}
      </div>
      <nav class="toc" id="toc">
        {"".join(toc_items)}
      </nav>
    </aside>
    <main class="main" id="main">
      {banner_block}
      {"".join(cards)}
    </main>
  </div>
  <script>
    const filterBtns = document.querySelectorAll('.filter-btn');
    const cards = document.querySelectorAll('.pair-card');
    const tocItems = document.querySelectorAll('.toc-item');
    filterBtns.forEach(btn => {{
      btn.addEventListener('click', () => {{
        filterBtns.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        const f = btn.dataset.filter;
        cards.forEach(card => {{
          const show = f === 'all' || card.dataset.filterValue === f;
          card.classList.toggle('hidden', !show);
        }});
        tocItems.forEach(link => {{
          const show = f === 'all' || link.dataset.filterValue === f;
          link.style.display = show ? '' : 'none';
        }});
      }});
    }});
  </script>
</body>
</html>
"""
    out_path.write_text(doc, encoding="utf-8")


def _export_high_band_pairs_for_manual_review(
    *,
    df_false_positive: pd.DataFrame,
    layout: dict[str, Path],
    email_text_by_eid: dict[str, dict[str, str]],
    email_text_meta: dict[str, Any],
    preview_chars: int,
    wrap_width: int,
    export_flags: ExportFlags,
) -> dict[str, Any]:
    """Optional debug CSV/HTML for high-score cross-campaign false positives."""
    paths: dict[str, str] = {}
    notes: list[str] = []

    if df_false_positive.empty:
        notes.append("no_false_positive_pairs_to_export")
        return {"paths": paths, "notes": notes, "email_text_catalog": email_text_meta}
    if not export_flags.emit_debug_csv and not export_flags.emit_debug_html:
        notes.append("skipped_high_band_review_exports_emit_debug_csv_or_html_false")
        return {"paths": paths, "notes": notes, "email_text_catalog": email_text_meta}

    df_review = _enrich_pairs_with_email_text(
        df_false_positive,
        email_text_by_eid=email_text_by_eid,
        preview_chars=preview_chars,
    )
    debug_csv = layout["debug_csv"]
    if export_flags.emit_debug_csv:
        csv_path = debug_csv / "debug_pair_high_band_false_positive_pairs_for_review.csv"
        df_review.to_csv(csv_path, index=False)
        paths["false_positive_pairs_for_review_csv"] = str(csv_path)
        if export_flags.emit_review_jsonl:
            jsonl_path = debug_csv / "debug_pair_high_band_false_positive_pairs_for_review.jsonl"
            _write_pairs_for_review_jsonl(
                df_review,
                out_path=jsonl_path,
                email_text_by_eid=email_text_by_eid,
                wrap_width=wrap_width,
            )
            paths["false_positive_pairs_for_review_jsonl"] = str(jsonl_path)

    if export_flags.emit_debug_html:
        html_path = debug_csv / "debug_pair_high_band_false_positive_pairs_for_review.html"
        _write_pairs_for_review_html(
            df_review,
            out_path=html_path,
            email_text_by_eid=email_text_by_eid,
            title="High-score cross-campaign pairs (manual review)",
            subtitle="Side-by-side subject/body for GT eyeballing. Use filters in the sidebar.",
        )
        paths["false_positive_pairs_for_review_html"] = str(html_path)

    df_sem = df_review[df_review["fp_regime"].isin({"semantic_only", "no_shared_core_artifacts"})]
    if not df_sem.empty and export_flags.emit_debug_csv:
        sem_csv = debug_csv / "debug_pair_high_band_semantic_only_false_positive_pairs_for_review.csv"
        df_sem.to_csv(sem_csv, index=False)
        paths["semantic_only_false_positive_pairs_for_review_csv"] = str(sem_csv)
        if export_flags.emit_review_jsonl:
            sem_jsonl = debug_csv / "debug_pair_high_band_semantic_only_false_positive_pairs_for_review.jsonl"
            _write_pairs_for_review_jsonl(
                df_sem,
                out_path=sem_jsonl,
                email_text_by_eid=email_text_by_eid,
                wrap_width=wrap_width,
            )
            paths["semantic_only_false_positive_pairs_for_review_jsonl"] = str(sem_jsonl)
    if not df_sem.empty and export_flags.emit_debug_html:
        sem_html = debug_csv / "debug_pair_high_band_semantic_only_false_positive_pairs_for_review.html"
        _write_pairs_for_review_html(
            df_sem,
            out_path=sem_html,
            email_text_by_eid=email_text_by_eid,
            title="Semantic-only high-score false positives",
            subtitle="No shared sender/URL/stem/domain artifacts — compare content for GT errors.",
        )
        paths["semantic_only_false_positive_pairs_for_review_html"] = str(sem_html)

    if not email_text_by_eid:
        notes.append("email_text_catalog_empty_subjects_and_bodies_will_be_blank")

    regime_counts = df_review["fp_regime"].value_counts().to_dict()
    return {
        "paths": paths,
        "notes": notes,
        "email_text_catalog": email_text_meta,
        "fp_regime_counts": {str(k): int(v) for k, v in regime_counts.items()},
        "n_pairs_with_both_email_texts": int(
            ((~df_review["email_i_text_missing"]) & (~df_review["email_j_text_missing"])).sum()
        ),
    }


def _export_low_band_pairs_for_manual_review(
    *,
    df_low_band: pd.DataFrame,
    layout: dict[str, Path],
    email_text_by_eid: dict[str, dict[str, str]],
    email_text_meta: dict[str, Any],
    preview_chars: int,
    wrap_width: int,
    low_score_max: float,
    export_flags: ExportFlags,
) -> dict[str, Any]:
    """Primary low-band HTML; optional debug CSV/JSONL/split HTML."""
    paths: dict[str, str] = {}
    notes: list[str] = []

    review_html = layout["review_html"]
    debug_csv = layout["debug_csv"]
    if df_low_band.empty:
        notes.append("no_low_band_unlabeled_pairs_to_export")
        html_path = review_html / "pair_low_band_unlabeled_pairs_for_review.html"
        _write_pairs_for_review_html(
            df_low_band,
            out_path=html_path,
            email_text_by_eid=email_text_by_eid,
            title="Low-score unlabeled pairs (manual review)",
            subtitle=(
                f"No GT-covered unlabeled pairs in score band [0, {float(low_score_max)}] "
                "for the current GT file(s)."
            ),
        )
        paths["low_band_unlabeled_pairs_for_review_html"] = str(html_path)
        return {"paths": paths, "notes": notes, "email_text_catalog": email_text_meta}

    df_review = _enrich_pairs_with_email_text(
        df_low_band,
        email_text_by_eid=email_text_by_eid,
        preview_chars=preview_chars,
        regime_fn=_classify_low_band_review_regime,
        review_prompt_fn=_low_band_review_prompt,
    )
    df_review = df_review.sort_values(
        ["gt_relation", "score", "email_i", "email_j"],
        ascending=[True, True, True, True],
        na_position="last",
    ).reset_index(drop=True)

    stem = "pair_low_band_unlabeled_pairs_for_review"
    if export_flags.emit_debug_csv:
        csv_path = debug_csv / f"debug_{stem}.csv"
        df_review.to_csv(csv_path, index=False)
        paths["low_band_unlabeled_pairs_for_review_csv"] = str(csv_path)
        if export_flags.emit_review_jsonl:
            jsonl_path = debug_csv / f"debug_{stem}.jsonl"
            _write_pairs_for_review_jsonl(
                df_review,
                out_path=jsonl_path,
                email_text_by_eid=email_text_by_eid,
                wrap_width=wrap_width,
            )
            paths["low_band_unlabeled_pairs_for_review_jsonl"] = str(jsonl_path)

    html_path = review_html / f"{stem}.html"
    _write_pairs_for_review_html(
        df_review,
        out_path=html_path,
        email_text_by_eid=email_text_by_eid,
        title="Low-score unlabeled pairs (manual review)",
        subtitle=(
            f"GT-covered unlabeled pairs with model score in [0, {float(low_score_max)}]. "
            "Filter by same-campaign vs cross-campaign; inspect score-zero collapses."
        ),
        review_prompt=(
            "Compare low-score same-campaign pairs (should often be rescuable) "
            "vs low-score cross-campaign pairs (should stay low)."
        ),
        gt_note="low-band unlabeled",
    )
    paths["low_band_unlabeled_pairs_for_review_html"] = str(html_path)

    if export_flags.emit_debug_csv or export_flags.emit_debug_html:
        df_same = df_review[df_review["gt_relation"].astype(str) == "same_campaign"]
        if not df_same.empty and export_flags.emit_debug_csv:
            same_csv = debug_csv / "debug_pair_low_band_same_campaign_unlabeled_pairs_for_review.csv"
            df_same.to_csv(same_csv, index=False)
            paths["low_band_same_campaign_unlabeled_pairs_for_review_csv"] = str(same_csv)
        if not df_same.empty and export_flags.emit_debug_html:
            same_html = debug_csv / "debug_pair_low_band_same_campaign_unlabeled_pairs_for_review.html"
            _write_pairs_for_review_html(
                df_same,
                out_path=same_html,
                email_text_by_eid=email_text_by_eid,
                title="Low-score same-campaign unlabeled pairs",
                subtitle=f"Score band [0, {float(low_score_max)}] — pairs that should move up.",
                review_prompt=(
                    "Same GT campaign but low model score. What evidence is missing from the scorer?"
                ),
                gt_note="same_campaign unlabeled",
            )
            paths["low_band_same_campaign_unlabeled_pairs_for_review_html"] = str(same_html)

        df_cross = df_review[df_review["gt_relation"].astype(str) == "cross_campaign"]
        if not df_cross.empty and export_flags.emit_debug_csv:
            cross_csv = debug_csv / "debug_pair_low_band_cross_campaign_unlabeled_pairs_for_review.csv"
            df_cross.to_csv(cross_csv, index=False)
            paths["low_band_cross_campaign_unlabeled_pairs_for_review_csv"] = str(cross_csv)
        if not df_cross.empty and export_flags.emit_debug_html:
            cross_html = debug_csv / "debug_pair_low_band_cross_campaign_unlabeled_pairs_for_review.html"
            _write_pairs_for_review_html(
                df_cross,
                out_path=cross_html,
                email_text_by_eid=email_text_by_eid,
                title="Low-score cross-campaign unlabeled pairs",
                subtitle=f"Score band [0, {float(low_score_max)}] — pairs that should stay low.",
                review_prompt="Cross GT campaigns with low score — confirm these should remain separated.",
                gt_note="cross_campaign unlabeled",
            )
            paths["low_band_cross_campaign_unlabeled_pairs_for_review_html"] = str(cross_html)

    if not email_text_by_eid:
        notes.append("email_text_catalog_empty_subjects_and_bodies_will_be_blank")

    regime_counts = df_review["fp_regime"].value_counts().to_dict()
    return {
        "paths": paths,
        "notes": notes,
        "email_text_catalog": email_text_meta,
        "fp_regime_counts": {str(k): int(v) for k, v in regime_counts.items()},
        "n_pairs": int(len(df_review)),
        "n_same_campaign": int((df_review["gt_relation"].astype(str) == "same_campaign").sum()),
        "n_cross_campaign": int((df_review["gt_relation"].astype(str) == "cross_campaign").sum()),
        "n_pairs_with_both_email_texts": int(
            ((~df_review["email_i_text_missing"]) & (~df_review["email_j_text_missing"])).sum()
        ),
    }


def _export_cross_campaign_positive_pairs_for_manual_review(
    *,
    df_cross_positive: pd.DataFrame,
    layout: dict[str, Path],
    email_text_by_eid: dict[str, dict[str, str]],
    email_text_meta: dict[str, Any],
    preview_chars: int,
    wrap_width: int,
    export_flags: ExportFlags,
) -> dict[str, Any]:
    """Optional debug export for GT cross-campaign pairs with training label positive."""
    paths: dict[str, str] = {}
    debug_csv = layout["debug_csv"]
    if df_cross_positive.empty:
        if not export_flags.emit_debug_html:
            return {
                "paths": paths,
                "n_pairs": 0,
                "email_text_catalog": email_text_meta,
                "note": "empty_cohort_no_gt_cross_campaign_pairs_with_positive_label",
            }
        html_path = debug_csv / "debug_pair_cross_campaign_positive_pairs_for_review.html"
        _write_pairs_for_review_html(
            df_cross_positive,
            out_path=html_path,
            email_text_by_eid=email_text_by_eid,
            title="Cross-campaign pairs labeled positive (manual review)",
            subtitle="No GT cross-campaign pairs with pair_status=positive in this run.",
            gt_note="GT cross-campaign, pair_status positive",
        )
        paths["cross_campaign_positive_pairs_for_review_html"] = str(html_path)
        return {
            "paths": paths,
            "n_pairs": 0,
            "email_text_catalog": email_text_meta,
            "note": "empty_cohort_no_gt_cross_campaign_pairs_with_positive_label",
        }

    if not export_flags.emit_debug_csv and not export_flags.emit_debug_html:
        return {
            "paths": paths,
            "n_pairs": int(len(df_cross_positive)),
            "email_text_catalog": email_text_meta,
            "note": "skipped_cross_positive_review_exports_emit_debug_csv_or_html_false",
        }

    df_review = _enrich_pairs_with_email_text(
        df_cross_positive,
        email_text_by_eid=email_text_by_eid,
        preview_chars=preview_chars,
    )
    stem = "pair_cross_campaign_positive_pairs_for_review"
    if export_flags.emit_debug_csv:
        csv_path = debug_csv / f"debug_{stem}.csv"
        df_review.to_csv(csv_path, index=False)
        paths["cross_campaign_positive_pairs_for_review_csv"] = str(csv_path)
        if export_flags.emit_review_jsonl:
            jsonl_path = debug_csv / f"debug_{stem}.jsonl"
            _write_pairs_for_review_jsonl(
                df_review,
                out_path=jsonl_path,
                email_text_by_eid=email_text_by_eid,
                wrap_width=wrap_width,
            )
            paths["cross_campaign_positive_pairs_for_review_jsonl"] = str(jsonl_path)

    if export_flags.emit_debug_html:
        html_path = debug_csv / f"debug_{stem}.html"
        _write_pairs_for_review_html(
            df_review,
            out_path=html_path,
            email_text_by_eid=email_text_by_eid,
            title="Cross-campaign pairs labeled positive (manual review)",
            subtitle=(
                "Training/seeds marked positive but GT says different campaigns. "
                "Compare side-by-side with timestamps."
            ),
            review_prompt=(
                "Training label is positive, but GT assigns different campaigns. "
                "Should these be the same campaign?"
            ),
            gt_note="GT cross-campaign, pair_status positive",
        )
        paths["cross_campaign_positive_pairs_for_review_html"] = str(html_path)

        df_sem = df_review[df_review["fp_regime"].isin({"semantic_only", "no_shared_core_artifacts"})]
        if not df_sem.empty:
            sem_stem = "pair_cross_campaign_positive_semantic_only_pairs_for_review"
            sem_html = debug_csv / f"debug_{sem_stem}.html"
            _write_pairs_for_review_html(
                df_sem,
                out_path=sem_html,
                email_text_by_eid=email_text_by_eid,
                title="Cross-campaign positives — semantic-only (no shared artifacts)",
                subtitle="Positive-labeled cross-GT pairs with no shared structural artifacts.",
                review_prompt=(
                    "Positive label + different GT campaigns + no shared artifacts. "
                    "Same campaign or GT error?"
                ),
                gt_note="GT cross-campaign, pair_status positive",
            )
            paths["cross_campaign_positive_semantic_only_pairs_for_review_html"] = str(sem_html)

    return {
        "paths": paths,
        "n_pairs": int(len(df_review)),
        "email_text_catalog": email_text_meta,
    }


def _build_high_band_artifact_summary(
    df_pairs: pd.DataFrame,
    *,
    cohort_label: str,
) -> pd.DataFrame:
    """Aggregate high-band pairs by concrete shared artifact value."""
    if df_pairs.empty:
        return pd.DataFrame(
            columns=[
                "cohort",
                "artifact_type",
                "artifact_value",
                "n_pairs",
                "mean_score",
                "max_score",
                "n_distinct_emails",
            ]
        )

    long_rows: list[dict[str, Any]] = []
    for _, r in df_pairs.iterrows():
        score = r.get("score")
        for _anchor_col, short in _PAIR_SHARED_CHANNEL_DEFS:
            values_col = f"shared_{short}_values"
            raw = r.get(values_col)
            if raw is None or (isinstance(raw, float) and pd.isna(raw)):
                continue
            text = str(raw).strip()
            if not text:
                continue
            for token in text.split("|"):
                token = token.strip()
                if not token or token.startswith("...("):
                    continue
                long_rows.append(
                    {
                        "cohort": cohort_label,
                        "artifact_type": short,
                        "artifact_value": token,
                        "score": score,
                        "email_i": r.get("email_i"),
                        "email_j": r.get("email_j"),
                    }
                )

    if not long_rows:
        return pd.DataFrame(
            columns=[
                "cohort",
                "artifact_type",
                "artifact_value",
                "n_pairs",
                "mean_score",
                "max_score",
                "n_distinct_emails",
            ]
        )

    long_df = pd.DataFrame(long_rows)

    def _agg_group(g: pd.DataFrame) -> pd.Series:
        emails: set[str] = set()
        emails.update(g["email_i"].astype(str).tolist())
        emails.update(g["email_j"].astype(str).tolist())
        scores = pd.to_numeric(g["score"], errors="coerce")
        return pd.Series(
            {
                "n_pairs": int(len(g)),
                "mean_score": float(scores.mean()) if scores.notna().any() else None,
                "max_score": float(scores.max()) if scores.notna().any() else None,
                "n_distinct_emails": int(len(emails)),
            }
        )

    agg_rows: list[dict[str, Any]] = []
    for (cohort, art_type, art_val), g in long_df.groupby(
        ["cohort", "artifact_type", "artifact_value"], sort=False
    ):
        stats = _agg_group(g)
        agg_rows.append(
            {
                "cohort": cohort,
                "artifact_type": art_type,
                "artifact_value": art_val,
                **stats.to_dict(),
            }
        )
    summary = pd.DataFrame(agg_rows)
    return summary.sort_values(
        ["n_pairs", "mean_score"],
        ascending=[False, False],
        na_position="last",
    )


def _build_high_band_false_positive_json_summary(
    *,
    df_false_positive: pd.DataFrame,
    df_true_positive: pd.DataFrame,
    artifact_summary: pd.DataFrame,
    high_score_min: float,
) -> dict[str, Any]:
    from collections import Counter

    n_fp = int(len(df_false_positive))
    out: dict[str, Any] = {
        "high_score_band": {"min_exclusive": float(high_score_min), "max_inclusive": 1.0},
        "cohort_definition": "GT-covered cross_campaign unlabeled pairs in high-score band",
        "n_false_positive_pairs_exported": n_fp,
        "n_true_positive_pairs_exported": int(len(df_true_positive)),
        "anchor_context_missing_fraction": (
            float(df_false_positive["anchor_context_missing"].mean())
            if n_fp > 0 and "anchor_context_missing" in df_false_positive.columns
            else None
        ),
    }

    if n_fp == 0:
        out["top_shared_artifacts_by_frequency"] = []
        out["top_shared_artifacts_by_mean_score"] = []
        out["top_provenance_combinations"] = []
        out["top_repeated_email_ids"] = []
        return out

    fp_art = artifact_summary
    if not fp_art.empty:
        by_freq = fp_art.sort_values("n_pairs", ascending=False).head(25)
        out["top_shared_artifacts_by_frequency"] = [
            {
                "artifact_type": r["artifact_type"],
                "artifact_value": r["artifact_value"],
                "n_pairs": int(r["n_pairs"]),
                "mean_score": r["mean_score"],
                "n_distinct_emails": int(r["n_distinct_emails"]),
            }
            for _, r in by_freq.iterrows()
        ]
        by_score = fp_art[fp_art["n_pairs"] >= 2].sort_values(
            ["mean_score", "n_pairs"], ascending=[False, False]
        ).head(25)
        out["top_shared_artifacts_by_mean_score"] = [
            {
                "artifact_type": r["artifact_type"],
                "artifact_value": r["artifact_value"],
                "n_pairs": int(r["n_pairs"]),
                "mean_score": r["mean_score"],
                "max_score": r.get("max_score"),
            }
            for _, r in by_score.iterrows()
        ]
    else:
        out["top_shared_artifacts_by_frequency"] = []
        out["top_shared_artifacts_by_mean_score"] = []

    if "provenance_combo" in df_false_positive.columns:
        prov = (
            df_false_positive["provenance_combo"]
            .astype(str)
            .value_counts()
            .head(15)
            .reset_index()
        )
        prov.columns = ["provenance_combo", "n_pairs"]
        out["top_provenance_combinations"] = prov.to_dict(orient="records")
    else:
        out["top_provenance_combinations"] = []

    email_counts: Counter[str] = Counter()
    for _, r in df_false_positive.iterrows():
        email_counts[str(r["email_i"])] += 1
        email_counts[str(r["email_j"])] += 1
    out["top_repeated_email_ids"] = [
        {"email_id": eid, "n_high_score_false_positive_pairs": int(cnt)}
        for eid, cnt in email_counts.most_common(25)
    ]

    if not df_true_positive.empty and "provenance_combo" in df_true_positive.columns:
        out["true_positive_provenance_combinations_top10"] = (
            df_true_positive["provenance_combo"].astype(str).value_counts().head(10).to_dict()
        )

    return out


def run_pair_score_kde_density_plots(
    *,
    run_dir: Path,
    graph_pt: Path,
    pair_csv: Path,
    gt_paths: list[Path],
    output_dir: Path | None = None,
    checkpoint_name: str = "best_model.pt",
    device: str = "cpu",
    to_undirected: bool = True,
) -> dict[str, Any]:
    """Score pairs and write same-vs-cross KDE overlays only (fast path for thesis figures)."""
    from src.pair_train import load_pair_training_dataframe

    project_root = Path(__file__).resolve().parents[2]
    run_dir = Path(run_dir).resolve()
    graph_pt = Path(graph_pt).resolve()
    pair_csv = Path(pair_csv).resolve()
    out_dir = (output_dir or (run_dir / "pair_score_separation")).resolve()
    layout = ensure_pair_score_separation_layout(out_dir)
    plots_dir = layout["plots"]

    df, _stats = load_pair_training_dataframe(pair_csv)
    df_work = df.reset_index(drop=True)
    df_work["_row"] = np.arange(len(df_work), dtype=np.int64)

    edge_scores_csv = (run_dir / "edge_gnn_pair_scores.csv").resolve()
    if edge_scores_csv.is_file():
        from seed_candidate_workflow.utils.edge_gnn_score_inference import (
            scores_array_for_pair_dataframe,
        )

        scores, _edge_score_diag = scores_array_for_pair_dataframe(
            edge_scores_csv, df_work, project_root=project_root
        )
        bundle = {"checkpoint_path": str(edge_scores_csv)}
    else:
        bundle = load_pair_supervision_for_inference(
            run_dir=run_dir,
            graph_pt=graph_pt,
            checkpoint_name=checkpoint_name,
            device=device,
            to_undirected=to_undirected,
        )
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
    scored_mask = np.isfinite(scores)

    per_gt: list[dict[str, Any]] = []
    for gt_path in gt_paths:
        gt_path = Path(gt_path).resolve()
        kde_plots = _write_same_cross_kde_plots_for_gt(
            gt_path=gt_path,
            plots_dir=plots_dir,
            df_work=df_work,
            scores=scores,
            scored_mask=scored_mask,
        )
        per_gt.append(
            {
                "gt_path": str(gt_path),
                **{k: rel_to_root(layout, v) for k, v in kde_plots.items()},
            }
        )

    payload = {
        "run_dir": str(run_dir),
        "graph_pt": str(graph_pt),
        "pair_csv": str(pair_csv),
        "checkpoint": str(bundle["checkpoint_path"]),
        "n_pair_rows_scored": int(len(df_work)),
        "n_finite_scores": int(scored_mask.sum()),
        "per_gt": per_gt,
        "output_layout": {k: rel_to_root(layout, v) for k, v in layout.items() if k != "root"},
    }
    summary_path = layout["core_json"] / "pair_score_kde_density_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    payload["summary_path"] = str(summary_path)
    return payload


def run_pair_score_separation_analysis(
    *,
    run_dir: Path,
    graph_pt: Path,
    pair_csv: Path,
    gt_paths: list[Path],
    output_dir: Path | None,
    checkpoint_name: str = "best_model.pt",
    device: str = "cpu",
    to_undirected: bool = True,
    low_score_max: float = 0.15,
    mid_score_min: float = 0.15,
    mid_score_max: float = 0.50,
    high_score_min: float = 0.8,
    anchor_run_dir: Path | None = None,
    misp_json_path: Path | None = None,
    misp_translated_json_path: Path | None = None,
    skip_email_text_export: bool = False,
    email_text_preview_chars: int = 500,
    email_text_wrap_width: int = 88,
    enable_rescued_collapsed_analysis: bool = True,
    enable_mid_band_frontier_analysis: bool = True,
    enable_frontier_analysis: bool | None = None,
    enable_high_cross_unlabeled_analysis: bool = True,
    high_cross_score_min: float = 0.80,
    mid_cross_score_min: float = 0.70,
    rescued_score_min: float = 0.80,
    collapsed_score_max: float = 0.10,
    community_cut_score_min: float = 0.30,
    community_cut_score_max: float = 0.50,
    export_flags: ExportFlags | None = None,
) -> dict[str, Any]:
    from src.pair_train import load_pair_training_dataframe

    project_root = Path(__file__).resolve().parents[2]
    run_dir = Path(run_dir).resolve()
    graph_pt = Path(graph_pt).resolve()
    pair_csv = Path(pair_csv).resolve()
    out_dir = (output_dir or (run_dir / "pair_score_separation")).resolve()
    layout = ensure_pair_score_separation_layout(out_dir)
    flags = export_flags or ExportFlags()
    plots_dir = layout["plots"]

    df, _stats = load_pair_training_dataframe(pair_csv)
    df_work = df.reset_index(drop=True)
    df_work["_row"] = np.arange(len(df_work), dtype=np.int64)

    cross_comp = None
    if "cross_seed_component_flag" in df_work.columns:
        cross_comp = df_work["cross_seed_component_flag"].fillna(False).astype(bool).to_numpy()

    edge_scores_csv = (run_dir / "edge_gnn_pair_scores.csv").resolve()
    score_source = "inference"
    edge_score_diag: dict[str, Any] = {}
    if edge_scores_csv.is_file():
        from seed_candidate_workflow.utils.edge_gnn_score_inference import (
            scores_array_for_pair_dataframe,
        )

        scores, edge_score_diag = scores_array_for_pair_dataframe(
            edge_scores_csv, df_work, project_root=project_root
        )
        score_source = "edge_gnn_pair_scores_csv"
        bundle = {
            "pair_encoder_backend": "edge_gnn",
            "checkpoint_path": str(edge_scores_csv),
            "training_config_path": str(run_dir / "edge_gnn" / "training_config.json"),
        }
    else:
        bundle = load_pair_supervision_for_inference(
            run_dir=run_dir,
            graph_pt=graph_pt,
            checkpoint_name=checkpoint_name,
            device=device,
            to_undirected=to_undirected,
        )
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
    scored_mask = np.isfinite(scores)
    plot_all_scored = plots_dir / "score_distribution_all_scored_pairs.png"
    _plot_score_histogram_counts(
        scores[scored_mask],
        title="Score distribution — all scored pairs (no GT filter)",
        out_path=plot_all_scored,
        bins=None,
        cohort_label="all_scored",
        color="tab:purple",
    )
    nodes_by_email, shared_ctx = _load_anchor_nodes_by_email(
        pair_csv=pair_csv,
        project_root=project_root,
        explicit_anchor_run_dir=anchor_run_dir,
    )
    cand_gen_dir = resolve_candidate_generation_dir(pair_training_csv=pair_csv)
    seed_gen_dir = resolve_seed_generation_dir(pair_training_csv=pair_csv)
    admitting_evidence_index, admitting_evidence_meta = load_admitting_evidence_index(
        candidate_generation_dir=cand_gen_dir,
        seed_generation_dir=seed_gen_dir,
    )

    per_gt: list[dict[str, Any]] = []
    band_rows: list[dict[str, Any]] = []
    low_sep_rows: list[dict[str, Any]] = []
    low_sep_per_gt: list[dict[str, Any]] = []
    low_joint_rows: list[dict[str, Any]] = []
    low_joint_per_gt: list[dict[str, Any]] = []
    high_sep_rows: list[dict[str, Any]] = []
    high_sep_per_gt: list[dict[str, Any]] = []
    high_joint_rows: list[dict[str, Any]] = []
    high_joint_per_gt: list[dict[str, Any]] = []
    high_fp_inspection_frames: list[pd.DataFrame] = []
    high_tp_inspection_frames: list[pd.DataFrame] = []
    cross_pos_inspection_frames: list[pd.DataFrame] = []
    low_band_inspection_frames: list[pd.DataFrame] = []
    for gt_path in gt_paths:
        gt_path = Path(gt_path).resolve()
        label_map, _eid_row, _camp = load_ground_truth_structures(gt_path)
        label_map = {str(k): v for k, v in label_map.items()}

        ei = df_work["email_i"].astype(str).values
        ej = df_work["email_j"].astype(str).values
        n = len(df_work)
        scored = scored_mask
        camp_i = np.array([label_map.get(str(ei[k])) for k in range(n)], dtype=object)
        camp_j = np.array([label_map.get(str(ej[k])) for k in range(n)], dtype=object)
        both = np.array(
            [camp_i[k] is not None and camp_j[k] is not None for k in range(n)],
            dtype=bool,
        )
        same_mask = both & (camp_i == camp_j)
        cross_mask = both & (camp_i != camp_j)
        same_s = scores[same_mask & scored]
        cross_s = scores[cross_mask & scored]

        pos_mask = (
            df_work["pair_status"].astype(str).str.lower().eq("positive").to_numpy()
            if "pair_status" in df_work.columns
            else np.zeros(n, dtype=bool)
        )
        unl_mask = (
            df_work["pair_status"].astype(str).str.lower().eq("unlabeled").to_numpy()
            if "pair_status" in df_work.columns
            else np.zeros(n, dtype=bool)
        )


        stem = _sanitize_filename_stem(gt_path.stem)
        title = f"Score distribution (GT: {gt_path.name})"
        plot_same = plots_dir / f"score_distribution_same_campaign_{stem}.png"
        plot_cross = plots_dir / f"score_distribution_cross_campaign_{stem}.png"
        _write_split_same_cross_histograms(
            same_scores=same_s,
            cross_scores=cross_s,
            title_base=title,
            out_same=plot_same,
            out_cross=plot_cross,
        )

        plot_same_pos = plots_dir / f"score_distribution_same_campaign_positive_{stem}.png"
        plot_cross_pos = plots_dir / f"score_distribution_cross_campaign_positive_{stem}.png"
        _write_split_same_cross_histograms(
            same_scores=scores[same_mask & pos_mask & scored],
            cross_scores=scores[cross_mask & pos_mask & scored],
            title_base=f"{title} — positives only",
            out_same=plot_same_pos,
            out_cross=plot_cross_pos,
        )
        plot_same_unl = plots_dir / f"score_distribution_same_campaign_unlabeled_{stem}.png"
        plot_cross_unl = plots_dir / f"score_distribution_cross_campaign_unlabeled_{stem}.png"
        _write_split_same_cross_histograms(
            same_scores=scores[same_mask & unl_mask & scored],
            cross_scores=scores[cross_mask & unl_mask & scored],
            title_base=f"{title} — unlabeled only",
            out_same=plot_same_unl,
            out_cross=plot_cross_unl,
        )
        kde_plots = _write_same_cross_kde_plots_for_gt(
            gt_path=gt_path,
            plots_dir=plots_dir,
            df_work=df_work,
            scores=scores,
            scored_mask=scored_mask,
        )

        cc_plot_same: Path | None = None
        cc_plot_cross: Path | None = None
        cc_plot_same_pos: Path | None = None
        cc_plot_cross_pos: Path | None = None
        cc_plot_same_unl: Path | None = None
        cc_plot_cross_unl: Path | None = None

        if cross_comp is not None:
            cc_mask = cross_comp.astype(bool)
            s_cc = scores[same_mask & cc_mask & scored]
            c_cc = scores[cross_mask & cc_mask & scored]
            cc_plot_same = plots_dir / f"score_distribution_cross_component_same_campaign_{stem}.png"
            cc_plot_cross = plots_dir / f"score_distribution_cross_component_cross_campaign_{stem}.png"
            _write_split_same_cross_histograms(
                same_scores=s_cc,
                cross_scores=c_cc,
                title_base=f"{title} — cross_seed_component_flag only",
                out_same=cc_plot_same,
                out_cross=cc_plot_cross,
            )

            cc_plot_same_pos = (
                plots_dir
                / f"score_distribution_cross_component_same_campaign_positive_{stem}.png"
            )
            cc_plot_cross_pos = (
                plots_dir
                / f"score_distribution_cross_component_cross_campaign_positive_{stem}.png"
            )
            _write_split_same_cross_histograms(
                same_scores=scores[same_mask & pos_mask & cc_mask & scored],
                cross_scores=scores[cross_mask & pos_mask & cc_mask & scored],
                title_base=f"{title} — positives only — cross_seed_component_flag only",
                out_same=cc_plot_same_pos,
                out_cross=cc_plot_cross_pos,
            )
            cc_plot_same_unl = (
                plots_dir
                / f"score_distribution_cross_component_same_campaign_unlabeled_{stem}.png"
            )
            cc_plot_cross_unl = (
                plots_dir
                / f"score_distribution_cross_component_cross_campaign_unlabeled_{stem}.png"
            )
            _write_split_same_cross_histograms(
                same_scores=scores[same_mask & unl_mask & cc_mask & scored],
                cross_scores=scores[cross_mask & unl_mask & cc_mask & scored],
                title_base=f"{title} — unlabeled only — cross_seed_component_flag only",
                out_same=cc_plot_same_unl,
                out_cross=cc_plot_cross_unl,
            )


        summary = _summarize_one_gt(
            gt_path=gt_path,
            gt_label_map=label_map,
            email_i=df_work["email_i"],
            email_j=df_work["email_j"],
            scores=scores,
            cross_component_mask=cross_comp,
        )
        band_diag, csv_rows = _compute_band_diagnostics_for_gt(
            df_work=df_work,
            scores=scores,
            same_mask=same_mask,
            cross_mask=cross_mask,
            eval_mask=(both & scored),
            nodes_by_email=nodes_by_email,
            low_max=low_score_max,
            high_min=high_score_min,
        )
        summary["band_diagnostics"] = band_diag
        summary["plot_same_campaign"] = rel_to_root(layout, plot_same)
        summary["plot_cross_campaign"] = rel_to_root(layout, plot_cross)

        summary["plot_same_campaign_positive_only"] = rel_to_root(layout, plot_same_pos)
        summary["plot_cross_campaign_positive_only"] = rel_to_root(layout, plot_cross_pos)
        summary["plot_same_campaign_unlabeled_only"] = rel_to_root(layout, plot_same_unl)
        summary["plot_cross_campaign_unlabeled_only"] = rel_to_root(layout, plot_cross_unl)
        summary["plot_kde_same_vs_cross_all_gt"] = rel_to_root(
            layout, kde_plots["plot_kde_same_vs_cross_all_gt"]
        )
        summary["plot_kde_same_vs_cross_positive_only"] = rel_to_root(
            layout, kde_plots["plot_kde_same_vs_cross_positive_only"]
        )
        summary["plot_kde_same_vs_cross_unlabeled_only"] = rel_to_root(
            layout, kde_plots["plot_kde_same_vs_cross_unlabeled_only"]
        )
        if cc_plot_same is not None and cc_plot_cross is not None:
            summary["plot_cross_component_same_campaign"] = rel_to_root(layout, cc_plot_same)
            summary["plot_cross_component_cross_campaign"] = rel_to_root(layout, cc_plot_cross)
        if cc_plot_same_pos is not None and cc_plot_cross_pos is not None:
            summary["plot_cross_component_same_campaign_positive_only"] = rel_to_root(
                layout, cc_plot_same_pos
            )
            summary["plot_cross_component_cross_campaign_positive_only"] = rel_to_root(
                layout, cc_plot_cross_pos
            )
        if cc_plot_same_unl is not None and cc_plot_cross_unl is not None:
            summary["plot_cross_component_same_campaign_unlabeled_only"] = rel_to_root(
                layout, cc_plot_same_unl
            )
            summary["plot_cross_component_cross_campaign_unlabeled_only"] = rel_to_root(
                layout, cc_plot_cross_unl
            )
        per_gt.append(summary)
        low_sep, low_rows = _build_low_band_separator_for_gt(
            gt_path=gt_path,
            band_diag=band_diag,
        )
        low_sep_per_gt.append(low_sep)
        low_sep_rows.extend(low_rows)
        # Joint-condition separators in low band (same vs cross)
        df_eval = df_work.loc[(both & scored)].copy()
        df_eval["score"] = scores[(both & scored)]
        same_eval = same_mask[(both & scored)]
        cross_eval = cross_mask[(both & scored)]
        low = df_eval["score"].ge(0.0) & df_eval["score"].le(float(low_score_max))
        mid = df_eval["score"].gt(float(mid_score_min)) & df_eval["score"].le(float(mid_score_max))
        same_low_eval = same_eval & low.to_numpy(dtype=bool, copy=False)
        cross_low_eval = cross_eval & low.to_numpy(dtype=bool, copy=False)
        unl_eval = unl_mask[(both & scored)]
        same_low_unl_eval = same_low_eval & unl_eval
        cross_low_unl_eval = cross_low_eval & unl_eval
        low_joint, low_joint_table = _build_low_band_joint_separator_for_gt(
            gt_path=gt_path,
            df_eval=df_eval,
            same_low_mask_eval=same_low_unl_eval,
            cross_low_mask_eval=cross_low_unl_eval,
            low_max=low_score_max,
            nodes_by_email=nodes_by_email,
            evidence_index=admitting_evidence_index,
        )
        low_joint_per_gt.append(low_joint)
        low_joint_rows.extend(low_joint_table)

        high_sep, high_rows = _build_high_band_separator_for_gt(
            gt_path=gt_path,
            band_diag=band_diag,
        )
        high = df_eval["score"].gt(float(high_score_min)) & df_eval["score"].le(1.0)
        same_high_unl_eval = same_eval & high.to_numpy(dtype=bool, copy=False) & unl_mask[(both & scored)]
        cross_high_unl_eval = cross_eval & high.to_numpy(dtype=bool, copy=False) & unl_mask[(both & scored)]
        high_joint, high_joint_table = _build_high_band_joint_separator_for_gt(
            gt_path=gt_path,
            df_eval=df_eval,
            same_high_unl_mask_eval=same_high_unl_eval,
            cross_high_unl_mask_eval=cross_high_unl_eval,
            high_min=high_score_min,
            nodes_by_email=nodes_by_email,
            marginal_sep=high_sep,
        )
        high_sep["recommendations"] = high_joint.get("recommendations") or _generate_high_band_recommendations(
            marginal=high_sep,
            joint=high_joint,
        )
        high_sep["unlabeled_high_band_counts"] = {
            "n_same_campaign_high_score_unlabeled": int(same_high_unl_eval.sum()),
            "n_cross_campaign_high_score_unlabeled": int(cross_high_unl_eval.sum()),
        }
        high_sep_per_gt.append(high_sep)
        high_sep_rows.extend(high_rows)
        high_joint_per_gt.append(high_joint)
        high_joint_rows.extend(high_joint_table)

        df_fp = _build_high_band_inspection_dataframe(
            df_eval=df_eval,
            row_mask=cross_high_unl_eval,
            gt_path=gt_path,
            label_map=label_map,
            gt_relation="cross_campaign",
            nodes_by_email=nodes_by_email,
            cohort="high_score_false_positive_unlabeled",
        )
        df_tp = _build_high_band_inspection_dataframe(
            df_eval=df_eval,
            row_mask=same_high_unl_eval,
            gt_path=gt_path,
            label_map=label_map,
            gt_relation="same_campaign",
            nodes_by_email=nodes_by_email,
            cohort="high_score_true_positive_unlabeled",
        )
        if not df_fp.empty:
            high_fp_inspection_frames.append(df_fp)
        if not df_tp.empty:
            high_tp_inspection_frames.append(df_tp)

        cross_pos_eval = cross_eval & pos_mask[(both & scored)]
        df_cross_pos = _build_high_band_inspection_dataframe(
            df_eval=df_eval,
            row_mask=cross_pos_eval,
            gt_path=gt_path,
            label_map=label_map,
            gt_relation="cross_campaign",
            nodes_by_email=nodes_by_email,
            cohort="cross_campaign_training_positive",
        )
        if not df_cross_pos.empty:
            cross_pos_inspection_frames.append(df_cross_pos)

        df_same_low = _build_high_band_inspection_dataframe(
            df_eval=df_eval,
            row_mask=same_low_unl_eval,
            gt_path=gt_path,
            label_map=label_map,
            gt_relation="same_campaign",
            nodes_by_email=nodes_by_email,
            cohort="low_score_same_campaign_unlabeled",
        )
        df_cross_low = _build_high_band_inspection_dataframe(
            df_eval=df_eval,
            row_mask=cross_low_unl_eval,
            gt_path=gt_path,
            label_map=label_map,
            gt_relation="cross_campaign",
            nodes_by_email=nodes_by_email,
            cohort="low_score_cross_campaign_unlabeled",
        )
        if not df_same_low.empty:
            low_band_inspection_frames.append(df_same_low)
        if not df_cross_low.empty:
            low_band_inspection_frames.append(df_cross_low)

        for row in csv_rows:
            band_rows.append(
                {"gt_path": str(gt_path.resolve()), "gt_name": gt_path.name, **row}
            )

    payload = {
        "run_dir": str(run_dir),
        "graph_pt": str(graph_pt),
        "pair_csv": str(pair_csv),
        "checkpoint": str(bundle["checkpoint_path"]),
        "device": device,
        "band_config": {
            "low_score_band": [0.0, float(low_score_max)],
            "mid_score_band": [float(mid_score_min), float(mid_score_max)],
            "high_score_band": [float(high_score_min), 1.0],
            "community_cut_zone": [float(community_cut_score_min), float(community_cut_score_max)],
        },
        "shared_evidence_context": shared_ctx,
        "admitting_evidence_catalog": admitting_evidence_meta,
        "per_gt": per_gt,
        "n_pair_rows_scored": int(len(df_work)),
        "n_finite_scores": int(scored_mask.sum()),
        "plot_all_scored_pairs": rel_to_root(layout, plot_all_scored),
        "output_layout": {k: rel_to_root(layout, v) for k, v in layout.items() if k != "root"},
    }
    core_json = layout["core_json"]
    debug_json = layout["debug_json"]
    debug_csv = layout["debug_csv"]
    summary_path = core_json / "pair_score_separation_summary.json"

    band_csv_path = debug_csv / "detail_pair_score_band_diagnostics.csv"
    pd.DataFrame(band_rows).to_csv(band_csv_path, index=False)

    low_sep_summary_path: Path | None = None
    low_joint_summary_path: Path | None = None
    low_sep_csv_path: Path | None = None
    low_joint_csv_path: Path | None = None
    high_sep_summary_path: Path | None = None
    high_joint_summary_path: Path | None = None
    high_sep_csv_path: Path | None = None
    high_joint_csv_path: Path | None = None
    twohop_channel_summary_path: Path | None = None
    twohop_channel_csv_path: Path | None = None
    high_fp_json_path: Path | None = None

    low_joint_payload = {
        "run_dir": str(run_dir),
        "graph_pt": str(graph_pt),
        "pair_csv": str(pair_csv),
        "checkpoint": str(bundle["checkpoint_path"]),
        "band_config": {
            "low_score_band": [0.0, float(low_score_max)],
            "high_score_band": [float(high_score_min), 1.0],
        },
        "per_gt": low_joint_per_gt,
    }
    if flags.emit_debug_json:
        low_sep_summary_path = debug_json / "detail_pair_low_band_separator_summary.json"
        with open(low_sep_summary_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "run_dir": str(run_dir),
                    "graph_pt": str(graph_pt),
                    "pair_csv": str(pair_csv),
                    "checkpoint": str(bundle["checkpoint_path"]),
                    "band_config": low_joint_payload["band_config"],
                    "per_gt": low_sep_per_gt,
                },
                f,
                indent=2,
                default=str,
            )
        low_joint_summary_path = debug_json / "detail_pair_low_band_joint_separator_summary.json"
        with open(low_joint_summary_path, "w", encoding="utf-8") as f:
            json.dump(low_joint_payload, f, indent=2, default=str)
        high_sep_summary_path = debug_json / "detail_pair_high_band_separator_summary.json"
        with open(high_sep_summary_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "run_dir": str(run_dir),
                    "graph_pt": str(graph_pt),
                    "pair_csv": str(pair_csv),
                    "checkpoint": str(bundle["checkpoint_path"]),
                    "band_config": low_joint_payload["band_config"],
                    "analysis_focus": (
                        "Marginal separators for all GT-covered pairs in the high-score band; "
                        "joint/recommendations use unlabeled-only high-band pairs."
                    ),
                    "per_gt": high_sep_per_gt,
                },
                f,
                indent=2,
                default=str,
            )
        high_joint_summary_path = debug_json / "detail_pair_high_band_joint_separator_summary.json"
        with open(high_joint_summary_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "run_dir": str(run_dir),
                    "graph_pt": str(graph_pt),
                    "pair_csv": str(pair_csv),
                    "checkpoint": str(bundle["checkpoint_path"]),
                    "band_config": low_joint_payload["band_config"],
                    "analysis_focus": (
                        "Joint separators for unlabeled pairs in the high-score band (same vs cross)."
                    ),
                    "per_gt": high_joint_per_gt,
                },
                f,
                indent=2,
                default=str,
            )
    if flags.emit_debug_csv:
        low_sep_csv_path = debug_csv / "detail_pair_low_band_separator_table.csv"
        pd.DataFrame(low_sep_rows).to_csv(low_sep_csv_path, index=False)
        low_joint_csv_path = debug_csv / "detail_pair_low_band_joint_separator_table.csv"
        pd.DataFrame(low_joint_rows).to_csv(low_joint_csv_path, index=False)
        high_sep_csv_path = debug_csv / "detail_pair_high_band_separator_table.csv"
        pd.DataFrame(high_sep_rows).to_csv(high_sep_csv_path, index=False)
        high_joint_csv_path = debug_csv / "detail_pair_high_band_joint_separator_table.csv"
        pd.DataFrame(high_joint_rows).to_csv(high_joint_csv_path, index=False)

    df_high_fp = (
        pd.concat(high_fp_inspection_frames, ignore_index=True)
        if high_fp_inspection_frames
        else pd.DataFrame()
    )
    df_high_tp = (
        pd.concat(high_tp_inspection_frames, ignore_index=True)
        if high_tp_inspection_frames
        else pd.DataFrame()
    )
    df_cross_pos_all = (
        pd.concat(cross_pos_inspection_frames, ignore_index=True)
        if cross_pos_inspection_frames
        else pd.DataFrame()
    )
    df_low_band_all = (
        pd.concat(low_band_inspection_frames, ignore_index=True)
        if low_band_inspection_frames
        else pd.DataFrame()
    )

    def _enrich_inspection_for_review(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        return enrich_inspection_with_admitting_evidence(
            df, evidence_index=admitting_evidence_index
        )

    df_high_fp = _enrich_inspection_for_review(df_high_fp)
    df_high_tp = _enrich_inspection_for_review(df_high_tp)
    df_cross_pos_all = _enrich_inspection_for_review(df_cross_pos_all)
    df_low_band_all = _enrich_inspection_for_review(df_low_band_all)

    twohop_channel_rows, twohop_channel_meta = build_channel_summary_table(df_low_band_all)
    twohop_channel_recs = build_twohop_channel_recommendations(
        twohop_channel_rows,
        joint_payload=low_joint_per_gt[0] if low_joint_per_gt else None,
    )
    if flags.emit_debug_json:
        twohop_channel_summary_path = debug_json / "detail_pair_low_band_twohop_channel_summary.json"
        twohop_channel_payload = {
            "run_dir": str(run_dir),
            "pair_csv": str(pair_csv),
            "band_config": {"low_score_band": [0.0, float(low_score_max)]},
            "meta": twohop_channel_meta,
            "per_channel": twohop_channel_rows,
            "twohop_channel_recommendations": twohop_channel_recs,
            "admitting_evidence_catalog": admitting_evidence_meta,
        }
        with open(twohop_channel_summary_path, "w", encoding="utf-8") as f:
            json.dump(twohop_channel_payload, f, indent=2, default=str)
        if low_joint_summary_path is not None:
            low_joint_payload["twohop_channel_summary_path"] = rel_to_root(
                layout, twohop_channel_summary_path
            )
            low_joint_payload["twohop_channel_recommendations"] = twohop_channel_recs
            low_joint_payload["twohop_channel_per_channel"] = twohop_channel_rows
            with open(low_joint_summary_path, "w", encoding="utf-8") as f:
                json.dump(low_joint_payload, f, indent=2, default=str)
    if flags.emit_debug_csv:
        twohop_channel_csv_path = debug_csv / "detail_pair_low_band_twohop_channel_summary.csv"
        pd.DataFrame(twohop_channel_rows).to_csv(twohop_channel_csv_path, index=False)

    payload["twohop_channel_recommendations"] = twohop_channel_recs

    low_band_pairs_path: Path | None = None
    high_fp_pairs_path: Path | None = None
    high_tp_pairs_path: Path | None = None
    high_fp_artifact_path: Path | None = None
    if flags.emit_debug_csv:
        low_band_pairs_path = debug_csv / "debug_pair_low_band_unlabeled_pairs.csv"
        df_low_band_all.to_csv(low_band_pairs_path, index=False)
        high_fp_pairs_path = debug_csv / "debug_pair_high_band_false_positive_pairs.csv"
        df_high_fp.to_csv(high_fp_pairs_path, index=False)
        high_tp_pairs_path = debug_csv / "debug_pair_high_band_true_positive_pairs.csv"
        df_high_tp.to_csv(high_tp_pairs_path, index=False)
        high_fp_artifact_summary = _build_high_band_artifact_summary(
            df_high_fp,
            cohort_label="high_score_false_positive_unlabeled",
        )
        high_fp_artifact_path = debug_csv / "debug_pair_high_band_false_positive_artifact_summary.csv"
        high_fp_artifact_summary.to_csv(high_fp_artifact_path, index=False)
    else:
        high_fp_artifact_summary = _build_high_band_artifact_summary(
            df_high_fp,
            cohort_label="high_score_false_positive_unlabeled",
        )

    high_fp_json = _build_high_band_false_positive_json_summary(
        df_false_positive=df_high_fp,
        df_true_positive=df_high_tp,
        artifact_summary=high_fp_artifact_summary,
        high_score_min=high_score_min,
    )
    if flags.emit_debug_json:
        high_fp_json_path = debug_json / "detail_pair_high_band_false_positive_summary.json"
        with open(high_fp_json_path, "w", encoding="utf-8") as f:
            json.dump(high_fp_json, f, indent=2, default=str)
    else:
        payload["high_band_false_positive_digest"] = {
            "n_false_positive": int(len(df_high_fp)),
            "n_true_positive": int(len(df_high_tp)),
            "fp_regime_counts": high_fp_json.get("fp_regime_counts"),
            "note": "Full detail JSON at debug_json/detail_pair_high_band_false_positive_summary.json "
            "(pass --emit-debug-json).",
        }

    manual_review_export: dict[str, Any] = {"skipped": True}
    cross_positive_review_export: dict[str, Any] = {"skipped": True}
    low_band_review_export: dict[str, Any] = {"skipped": True}
    review_paths: dict[str, str] = {}
    cross_positive_review_paths: dict[str, str] = {}
    low_band_review_paths: dict[str, str] = {}
    if not skip_email_text_export:
        resolved_misp = misp_json_path
        if resolved_misp is not None and not resolved_misp.is_absolute():
            resolved_misp = (project_root / resolved_misp).resolve()
        if resolved_misp is None or not resolved_misp.is_file():
            resolved_misp = _resolve_default_misp_json_path(project_root)
        resolved_tr: Path | None = None
        if misp_translated_json_path is not None:
            resolved_tr = misp_translated_json_path
            if not resolved_tr.is_absolute():
                resolved_tr = (project_root / resolved_tr).resolve()
        email_catalog, email_catalog_meta = _load_email_text_catalog(
            project_root=project_root,
            misp_json_path=resolved_misp,
            misp_translated_json_path=resolved_tr,
        )
        manual_review_export = _export_high_band_pairs_for_manual_review(
            df_false_positive=df_high_fp,
            layout=layout,
            email_text_by_eid=email_catalog,
            email_text_meta=email_catalog_meta,
            preview_chars=int(email_text_preview_chars),
            wrap_width=int(email_text_wrap_width),
            export_flags=flags,
        )
        review_paths = manual_review_export.get("paths") or {}
        if high_fp_json_path is not None:
            high_fp_json["manual_review_export"] = manual_review_export
            with open(high_fp_json_path, "w", encoding="utf-8") as f:
                json.dump(high_fp_json, f, indent=2, default=str)

        cross_positive_review_export = _export_cross_campaign_positive_pairs_for_manual_review(
            df_cross_positive=df_cross_pos_all,
            layout=layout,
            email_text_by_eid=email_catalog,
            email_text_meta=email_catalog_meta,
            preview_chars=int(email_text_preview_chars),
            wrap_width=int(email_text_wrap_width),
            export_flags=flags,
        )
        cross_positive_review_paths = cross_positive_review_export.get("paths") or {}
        if high_fp_json_path is not None:
            high_fp_json["cross_campaign_positive_manual_review_export"] = cross_positive_review_export
            with open(high_fp_json_path, "w", encoding="utf-8") as f:
                json.dump(high_fp_json, f, indent=2, default=str)

        low_band_review_export = _export_low_band_pairs_for_manual_review(
            df_low_band=df_low_band_all,
            layout=layout,
            email_text_by_eid=email_catalog,
            email_text_meta=email_catalog_meta,
            preview_chars=int(email_text_preview_chars),
            wrap_width=int(email_text_wrap_width),
            low_score_max=float(low_score_max),
            export_flags=flags,
        )
        low_band_review_paths = low_band_review_export.get("paths") or {}
        payload["low_band_unlabeled_manual_review_export"] = {
            "status": "ok",
            "n_pairs": low_band_review_export.get("n_pairs"),
            "primary_html": low_band_review_paths.get("low_band_unlabeled_pairs_for_review_html"),
        }
    else:
        cross_positive_review_export = {"skipped": True, "reason": "skip_email_text_export"}
        cross_positive_review_paths = {}
        low_band_review_export = {"skipped": True, "reason": "skip_email_text_export"}
        low_band_review_paths = {}
        email_catalog = {}

    run_frontier = (
        enable_frontier_analysis
        if enable_frontier_analysis is not None
        else enable_mid_band_frontier_analysis
    )
    mid_band_paths: dict[str, str] = {}
    frontier_paths: dict[str, str] = {}
    if run_frontier:
        from seed_candidate_workflow.utils.pair_mid_band_frontier import (
            MidBandThresholds,
            run_mid_band_frontier_analysis,
        )

        th = MidBandThresholds(
            low_score_max=float(low_score_max),
            mid_score_min=float(mid_score_min),
            mid_score_max=float(mid_score_max),
            high_score_min=float(high_score_min),
            rescued_score_min=float(rescued_score_min),
            community_cut_score_min=float(community_cut_score_min),
            community_cut_score_max=float(community_cut_score_max),
        )
        per_gt_mid: list[dict[str, Any]] = []
        for gt_path in gt_paths:
            gt_path = Path(gt_path).resolve()
            label_map, _eid_row, _camp = load_ground_truth_structures(gt_path)
            label_map = {str(k): v for k, v in label_map.items()}
            suffix = f"__{_sanitize_filename_stem(gt_path.stem)}" if len(gt_paths) > 1 else ""
            mid_out = run_mid_band_frontier_analysis(
                df_work=df_work,
                scores=scores,
                gt_path=gt_path,
                label_map=label_map,
                layout=layout,
                nodes_by_email=nodes_by_email,
                evidence_index=admitting_evidence_index,
                email_text_by_eid=email_catalog if email_catalog else None,
                thresholds=th,
                export_flags=flags,
                filename_suffix=suffix,
            )
            per_gt_mid.append({"gt_path": str(gt_path), "gt_name": gt_path.name, **mid_out})
            if not mid_band_paths:
                mid_band_paths = {
                    k: v for k, v in (mid_out.get("paths") or {}).items() if isinstance(v, str)
                }
                mid_band_paths["summary_path"] = mid_out.get("summary_path")
                mid_band_paths["joint_summary_path"] = mid_out.get("joint_summary_path")
                mid_band_paths["table_path"] = mid_out.get("table_path")
            if not frontier_paths:
                frontier_paths = {
                    k: v for k, v in (mid_out.get("paths") or {}).items() if isinstance(v, str)
                }
                frontier_paths["summary_path"] = mid_out.get("frontier_summary_path") or mid_out.get(
                    "summary_path"
                )
                frontier_paths["joint_summary_path"] = mid_out.get(
                    "frontier_joint_summary_path"
                ) or mid_out.get("joint_summary_path")
                frontier_paths["table_path"] = mid_out.get("table_path")
        payload["mid_band_frontier_analysis"] = {
            "skipped": False,
            "thresholds": {
                "low_score_max": th.low_score_max,
                "mid_score_min_exclusive": th.mid_score_min,
                "mid_score_max_inclusive": th.mid_score_max,
                "high_score_min_exclusive": th.high_score_min,
                "community_cut_score_range": [
                    th.community_cut_score_min,
                    th.community_cut_score_max,
                ],
            },
            "per_gt": per_gt_mid,
        }
        payload["pair_frontier_analysis"] = payload["mid_band_frontier_analysis"]
        if per_gt_mid:
            payload["mid_band_frontier_digest"] = per_gt_mid[0].get("digest")
            payload["frontier_analysis_digest"] = per_gt_mid[0].get("digest")
    else:
        payload["mid_band_frontier_analysis"] = {"skipped": True, "reason": "disabled"}
        payload["pair_frontier_analysis"] = {"skipped": True, "reason": "disabled"}

    high_cross_paths: dict[str, str] = {}
    per_gt_high_cross: list[dict[str, Any]] = []
    run_encoder_hint = str(bundle.get("pair_encoder_backend") or "")
    try:
        tc = bundle.get("train_cfg") or {}
        if tc.get("pair_scorer_use_explicit_features") is False:
            run_encoder_hint = run_encoder_hint or "gnn_embedding_only_scorer"
        if tc.get("pair_scorer_use_embedding_features") is False:
            run_encoder_hint = run_encoder_hint or "explicit_only_scorer"
    except Exception:
        pass

    if enable_high_cross_unlabeled_analysis:
        from seed_candidate_workflow.utils.pair_high_cross_unlabeled_analysis import (
            HighCrossThresholds,
            run_pair_high_cross_unlabeled_analysis,
        )

        hth = HighCrossThresholds(
            high_cross_score_min=float(high_cross_score_min),
            mid_cross_score_min=float(mid_cross_score_min),
        )
        for gt_path in gt_paths:
            gt_path = Path(gt_path).resolve()
            label_map, _eid_row, _camp = load_ground_truth_structures(gt_path)
            label_map = {str(k): v for k, v in label_map.items()}
            suffix = f"__{_sanitize_filename_stem(gt_path.stem)}" if len(gt_paths) > 1 else ""
            hc_out = run_pair_high_cross_unlabeled_analysis(
                df_work=df_work,
                scores=scores,
                gt_path=gt_path,
                label_map=label_map,
                layout=layout,
                nodes_by_email=nodes_by_email,
                evidence_index=admitting_evidence_index,
                email_text_by_eid=email_catalog if email_catalog else None,
                thresholds=hth,
                export_flags=flags,
                filename_suffix=suffix,
                project_root=project_root,
                run_dir=run_dir,
                graph_pt=graph_pt,
                inference_bundle=bundle,
                run_encoder_hint=run_encoder_hint or None,
            )
            per_gt_high_cross.append({"gt_path": str(gt_path), "gt_name": gt_path.name, **hc_out})
            if not high_cross_paths:
                high_cross_paths = {k: v for k, v in (hc_out.get("paths") or {}).items() if isinstance(v, str)}
        payload["pair_high_cross_unlabeled_analysis"] = {
            "skipped": False,
            "thresholds": {
                "high_cross_score_min": hth.high_cross_score_min,
                "mid_cross_score_min": hth.mid_cross_score_min,
            },
            "per_gt": per_gt_high_cross,
            "run_encoder_hint": run_encoder_hint,
        }
        if per_gt_high_cross:
            payload["high_cross_unlabeled_digest"] = per_gt_high_cross[0].get("digest")
    else:
        payload["pair_high_cross_unlabeled_analysis"] = {"skipped": True, "reason": "disabled"}
        payload["high_cross_unlabeled_digest"] = None

    rescued_collapsed_exports: dict[str, Any] = {"skipped": True}
    rescued_collapsed_paths: dict[str, str] = {}
    per_gt_rescued: list[dict[str, Any]] = []
    if enable_rescued_collapsed_analysis:
        from seed_candidate_workflow.utils.pair_same_unlabeled_rescued_collapsed import (
            run_same_unlabeled_rescued_vs_collapsed_analysis,
        )

        unl_mask_global = (
            df_work["pair_status"].astype(str).str.lower().eq("unlabeled").to_numpy()
            if "pair_status" in df_work.columns
            else np.zeros(len(df_work), dtype=bool)
        )
        ei_all = df_work["email_i"].astype(str).values
        ej_all = df_work["email_j"].astype(str).values
        for gt_path in gt_paths:
            gt_path = Path(gt_path).resolve()
            label_map, _eid_row, _camp = load_ground_truth_structures(gt_path)
            label_map = {str(k): v for k, v in label_map.items()}
            n = len(df_work)
            camp_i = np.array([label_map.get(str(ei_all[k])) for k in range(n)], dtype=object)
            camp_j = np.array([label_map.get(str(ej_all[k])) for k in range(n)], dtype=object)
            both = np.array(
                [camp_i[k] is not None and camp_j[k] is not None for k in range(n)],
                dtype=bool,
            )
            cross_unl = both & (camp_i != camp_j) & unl_mask_global & scored_mask
            stem = _sanitize_filename_stem(gt_path.stem)
            suffix = f"__{stem}" if len(gt_paths) > 1 else ""
            paths = run_same_unlabeled_rescued_vs_collapsed_analysis(
                df_work=df_work,
                scores=scores,
                gt_path=gt_path,
                label_map=label_map,
                out_dir=out_dir,
                nodes_by_email=nodes_by_email,
                evidence_index=admitting_evidence_index,
                email_text_by_eid=email_catalog,
                rescued_score_min=float(rescued_score_min),
                collapsed_score_max=float(collapsed_score_max),
                email_text_preview_chars=int(email_text_preview_chars),
                email_text_wrap_width=int(email_text_wrap_width),
                cross_unlabeled_mask=cross_unl,
                filename_suffix=suffix,
                export_flags=flags,
            )
            per_gt_rescued.append(
                {
                    "gt_path": str(gt_path),
                    "gt_name": gt_path.name,
                    "filename_suffix": suffix,
                    **paths,
                }
            )
            if not rescued_collapsed_paths:
                rescued_collapsed_paths = {
                    k: v for k, v in paths.items() if isinstance(v, str)
                }
        rescued_collapsed_exports = {
            "skipped": False,
            "thresholds": {
                "rescued_same_unlabeled_min_score": float(rescued_score_min),
                "collapsed_same_unlabeled_max_score": float(collapsed_score_max),
            },
            "per_gt": per_gt_rescued,
        }
        payload["rescued_vs_collapsed_same_unlabeled_analysis"] = {
            "skipped": False,
            "thresholds": rescued_collapsed_exports["thresholds"],
            "per_gt": [
                {
                    "gt_path": e.get("gt_path"),
                    "gt_name": e.get("gt_name"),
                    "summary_path": e.get("summary_path"),
                    "review_html_rescued": e.get("review_html_rescued"),
                    "review_html_collapsed": e.get("review_html_collapsed"),
                }
                for e in per_gt_rescued
            ],
        }

    rescued_suffix = ""
    if per_gt_rescued:
        rescued_suffix = str(per_gt_rescued[0].get("filename_suffix") or "")
    has_high_cross = bool(enable_high_cross_unlabeled_analysis and per_gt_high_cross)
    primary_outputs = build_primary_outputs(
        layout=layout,
        rescued_suffix=rescued_suffix,
        per_gt=per_gt,
        has_mid_band=run_frontier,
        has_high_cross=has_high_cross,
        gnn_only_scorer_hint="embedding_only" in (run_encoder_hint or "").lower()
        or "gnn_only" in str(run_dir.name).lower(),
    )
    nav = build_navigation_index(
        layout=layout,
        export_flags=flags,
        primary_outputs=primary_outputs,
        has_rescued_collapsed=enable_rescued_collapsed_analysis,
        has_mid_band=run_frontier,
        has_high_cross=has_high_cross,
        gnn_only_scorer_hint="embedding_only" in (run_encoder_hint or "").lower()
        or "gnn_only" in str(run_dir.name).lower(),
    )
    payload.update(nav)
    manifest_path = write_artifact_manifest(
        layout=layout,
        export_flags=flags,
        run_meta={
            "run_dir": str(run_dir),
            "pair_csv": str(pair_csv),
            "n_gt_files": len(gt_paths),
        },
    )
    payload["artifact_manifest_json"] = rel_to_root(layout, manifest_path)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)

    def _opt(p: Path | None) -> str | None:
        return str(p) if p is not None else None

    return {
        "output_dir": str(out_dir),
        "summary_path": str(summary_path),
        "manifest_path": str(manifest_path),
        "primary_outputs": primary_outputs,
        "band_csv_path": str(band_csv_path),
        "low_separator_summary_path": _opt(low_sep_summary_path),
        "low_separator_csv_path": _opt(low_sep_csv_path),
        "low_joint_separator_summary_path": _opt(low_joint_summary_path),
        "low_joint_separator_csv_path": _opt(low_joint_csv_path),
        "high_separator_summary_path": _opt(high_sep_summary_path),
        "high_separator_csv_path": _opt(high_sep_csv_path),
        "high_joint_separator_summary_path": _opt(high_joint_summary_path),
        "high_joint_separator_csv_path": _opt(high_joint_csv_path),
        "high_false_positive_pairs_csv_path": _opt(high_fp_pairs_path),
        "high_true_positive_pairs_csv_path": _opt(high_tp_pairs_path),
        "high_false_positive_artifact_summary_csv_path": _opt(high_fp_artifact_path),
        "high_false_positive_summary_json_path": _opt(high_fp_json_path),
        "high_false_positive_pairs_for_review_csv_path": review_paths.get(
            "false_positive_pairs_for_review_csv"
        ),
        "high_false_positive_pairs_for_review_jsonl_path": review_paths.get(
            "false_positive_pairs_for_review_jsonl"
        ),
        "high_semantic_only_false_positive_pairs_for_review_csv_path": review_paths.get(
            "semantic_only_false_positive_pairs_for_review_csv"
        ),
        "high_semantic_only_false_positive_pairs_for_review_jsonl_path": review_paths.get(
            "semantic_only_false_positive_pairs_for_review_jsonl"
        ),
        "high_false_positive_pairs_for_review_html_path": review_paths.get(
            "false_positive_pairs_for_review_html"
        ),
        "high_semantic_only_false_positive_pairs_for_review_html_path": review_paths.get(
            "semantic_only_false_positive_pairs_for_review_html"
        ),
        "cross_campaign_positive_pairs_for_review_html_path": cross_positive_review_paths.get(
            "cross_campaign_positive_pairs_for_review_html"
        ),
        "cross_campaign_positive_semantic_only_pairs_for_review_html_path": (
            cross_positive_review_paths.get("cross_campaign_positive_semantic_only_pairs_for_review_html")
        ),
        "low_band_unlabeled_pairs_csv_path": str(low_band_pairs_path),
        "low_band_unlabeled_pairs_for_review_csv_path": low_band_review_paths.get(
            "low_band_unlabeled_pairs_for_review_csv"
        ),
        "low_band_unlabeled_pairs_for_review_jsonl_path": low_band_review_paths.get(
            "low_band_unlabeled_pairs_for_review_jsonl"
        ),
        "low_band_unlabeled_pairs_for_review_html_path": low_band_review_paths.get(
            "low_band_unlabeled_pairs_for_review_html"
        ),
        "low_band_same_campaign_unlabeled_pairs_for_review_html_path": low_band_review_paths.get(
            "low_band_same_campaign_unlabeled_pairs_for_review_html"
        ),
        "low_band_cross_campaign_unlabeled_pairs_for_review_html_path": low_band_review_paths.get(
            "low_band_cross_campaign_unlabeled_pairs_for_review_html"
        ),
        "mid_band_frontier_summary_json_path": mid_band_paths.get("summary_path"),
        "mid_band_frontier_joint_summary_json_path": mid_band_paths.get("joint_summary_path"),
        "mid_band_frontier_table_csv_path": mid_band_paths.get("table_path"),
        "mid_band_same_unlabeled_for_review_html_path": mid_band_paths.get("mid_same_review_html"),
        "mid_band_cross_unlabeled_for_review_html_path": mid_band_paths.get("mid_cross_review_html"),
        "mid_band_same_vs_cross_for_review_html_path": mid_band_paths.get("mid_same_vs_cross_review_html"),
        "mid_band_same_vs_rescued_for_review_html_path": mid_band_paths.get("mid_same_vs_rescued_review_html"),
        "frontier_analysis_summary_json_path": frontier_paths.get("summary_path"),
        "frontier_analysis_joint_summary_json_path": frontier_paths.get("joint_summary_path"),
        "frontier_analysis_table_csv_path": frontier_paths.get("table_path"),
        "frontier_high_same_unlabeled_for_review_html_path": frontier_paths.get("high_same_review_html"),
        "frontier_low_same_unlabeled_for_review_html_path": frontier_paths.get("low_same_review_html"),
        "low_band_twohop_channel_summary_json_path": str(twohop_channel_summary_path),
        "low_band_twohop_channel_summary_csv_path": str(twohop_channel_csv_path),
        "rescued_collapsed_summary_path": rescued_collapsed_paths.get("summary_path"),
        "rescued_collapsed_table_path": rescued_collapsed_paths.get("table_path"),
        "rescued_collapsed_joint_path": rescued_collapsed_paths.get("joint_path"),
        "rescued_collapsed_review_html_path": rescued_collapsed_paths.get("review_html"),
        "high_cross_unlabeled_analysis_summary_json_path": high_cross_paths.get("summary_path"),
        "high_cross_unlabeled_analysis_joint_summary_json_path": high_cross_paths.get(
            "joint_summary_path"
        ),
        "high_cross_unlabeled_analysis_table_csv_path": high_cross_paths.get("table_path"),
        "high_cross_unlabeled_for_review_html_path": high_cross_paths.get(
            "high_cross_unlabeled_for_review_html"
        ),
        "high_same_unlabeled_for_review_html_path": high_cross_paths.get(
            "high_same_unlabeled_for_review_html"
        ),
        "payload": payload,
    }


def _gt_json_paths_from_dir(gt_dir: Path, *, include_report_json: bool) -> list[Path]:
    d = gt_dir.resolve()
    if not d.is_dir():
        raise SystemExit(f"--gt-dir is not a directory: {d}")
    paths = sorted(d.glob("*.json"))
    if not include_report_json:
        paths = [p for p in paths if "report" not in p.name.lower()]
    return paths


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="PU pair model score separation vs GT campaigns.")
    p.add_argument("--run-dir", type=Path, required=True, help="Pair supervision run directory (e.g. core/GNN/outputs/pair_pu_001)")
    p.add_argument("--graph-pt", type=Path, required=True, help="Path to hetero .pt used for training")
    p.add_argument("--pair-csv", type=Path, default=None, help="pair_training_dataset.csv (default: from run training_config.json)")
    p.add_argument(
        "--gt-dir",
        type=Path,
        default=None,
        help="Use every *.json in this directory (default: skip filenames containing 'report').",
    )
    p.add_argument(
        "--gt-path",
        type=Path,
        action="append",
        default=None,
        metavar="PATH",
        help="Ground-truth JSON (repeat for multiple files, e.g. merged + default).",
    )
    p.add_argument(
        "--gt-include-report-json",
        action="store_true",
        help="With --gt-dir, also include *report*.json (not cluster maps; usually useless for this analysis).",
    )
    p.add_argument("--output-dir", type=Path, default=None, help="Output root (default: <run-dir>/pair_score_separation)")
    p.add_argument("--checkpoint", type=str, default="best_model.pt", help="Checkpoint filename under run_dir/models/")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument(
        "--low-score-max",
        type=float,
        default=0.15,
        help="Low-score band upper bound inclusive (default 0.15; scores at mid_min go to mid band).",
    )
    p.add_argument(
        "--mid-score-min",
        type=float,
        default=0.15,
        help="Mid-band lower bound exclusive (default 0.15).",
    )
    p.add_argument(
        "--mid-score-max",
        type=float,
        default=0.50,
        help="Mid-band upper bound inclusive (default 0.50).",
    )
    p.add_argument("--high-score-min", type=float, default=0.8, help="High-score band lower bound (exclusive).")
    p.add_argument(
        "--community-cut-score-min",
        type=float,
        default=0.30,
        help="Community cut zone lower bound inclusive (descriptive analysis).",
    )
    p.add_argument(
        "--community-cut-score-max",
        type=float,
        default=0.50,
        help="Community cut zone upper bound inclusive (descriptive analysis).",
    )
    p.add_argument("--anchor-run-dir", type=Path, default=None, help="Optional anchor graph run dir (contains anchor_graph_nodes.csv) for shared-evidence diagnostics.")
    p.add_argument(
        "--misp-json-path",
        type=Path,
        default=None,
        help="MISP lake JSON for subject/body export (default: pipeline_config datasets/graph misp_json_path).",
    )
    p.add_argument(
        "--misp-translated-json-path",
        type=Path,
        default=None,
        help="Optional translated subject/body sidecar (translate_misp_email_texts_to_en.py output).",
    )
    p.add_argument(
        "--skip-email-text-export",
        action="store_true",
        help="Do not write subject/body manual-review CSV/JSONL artifacts.",
    )
    p.add_argument(
        "--email-text-preview-chars",
        type=int,
        default=500,
        help="Max characters per body preview column in the review CSV.",
    )
    p.add_argument(
        "--email-text-wrap-width",
        type=int,
        default=88,
        help="Wrap width for subject/body line arrays in review JSONL.",
    )
    p.add_argument(
        "--no-to-undirected",
        action="store_true",
        help="Load graph without ToUndirected (default: undirected, matching training)",
    )
    p.add_argument(
        "--skip-rescued-collapsed-analysis",
        action="store_true",
        help="Skip rescued vs collapsed same-campaign unlabeled analysis.",
    )
    p.add_argument(
        "--skip-mid-band-analysis",
        action="store_true",
        help="Skip mid-band / full frontier analysis (low/mid/high unlabeled cohorts).",
    )
    p.add_argument(
        "--skip-frontier-analysis",
        action="store_true",
        help="Alias for --skip-mid-band-analysis (full pair frontier diagnostic).",
    )
    p.add_argument(
        "--skip-high-cross-analysis",
        action="store_true",
        help="Skip high-scoring cross-campaign unlabeled diagnostic (GNN-only FP analysis).",
    )
    p.add_argument(
        "--high-cross-score-min",
        type=float,
        default=0.80,
        help="Min score for high_cross_unlabeled / high_same_unlabeled cohorts (default 0.80).",
    )
    p.add_argument(
        "--mid-cross-score-min",
        type=float,
        default=0.70,
        help="Lower bound for mid_cross_unlabeled vs high_cross comparison (default 0.70).",
    )
    p.add_argument(
        "--rescued-score-min",
        type=float,
        default=0.80,
        help="Score >= this marks rescued_same_unlabeled (default 0.80).",
    )
    p.add_argument(
        "--collapsed-score-max",
        type=float,
        default=0.10,
        help="Score <= this marks collapsed_same_unlabeled (default 0.10).",
    )
    p.add_argument(
        "--emit-debug-json",
        action="store_true",
        help="Write detail_* separator/joint/twohop JSON under debug_json/ (off by default).",
    )
    p.add_argument(
        "--emit-debug-csv",
        action="store_true",
        help="Write detail_* tables and debug_* pair dumps under debug_csv/ (off by default).",
    )
    p.add_argument(
        "--emit-debug-html",
        action="store_true",
        help="Write debug_* secondary review HTML under debug_csv/ (off by default).",
    )
    p.add_argument(
        "--emit-review-jsonl",
        action="store_true",
        help="Also write JSONL review exports (requires --emit-debug-csv; off by default).",
    )
    args = p.parse_args(argv)

    if bool(args.gt_dir) == bool(args.gt_path):
        raise SystemExit("Provide exactly one of: --gt-dir, or one or more --gt-path.")

    run_dir = args.run_dir.resolve()
    repo_root = Path(__file__).resolve().parents[2]
    from seed_candidate_workflow.utils.pair_model_inference import (
        resolve_pair_dataset_csv_path,
        resolve_pair_supervision_run_artifacts,
    )

    edge_scores_csv = (run_dir / "edge_gnn_pair_scores.csv").resolve()
    if not edge_scores_csv.is_file():
        try:
            resolve_pair_supervision_run_artifacts(
                run_dir,
                checkpoint_name=str(args.checkpoint),
                project_root=repo_root,
            )
        except FileNotFoundError as exc:
            raise SystemExit(str(exc)) from exc

    try:
        pair_csv = resolve_pair_dataset_csv_path(
            run_dir,
            pair_csv=args.pair_csv,
            project_root=repo_root,
        )
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from exc

    gt_paths: list[Path] = []
    if args.gt_path is not None:
        gt_paths = [Path(p).resolve() for p in args.gt_path]
    elif args.gt_dir is not None:
        gt_paths.extend(
            _gt_json_paths_from_dir(
                args.gt_dir, include_report_json=bool(args.gt_include_report_json)
            )
        )
    seen: set[Path] = set()
    deduped: list[Path] = []
    for p in gt_paths:
        r = p.resolve()
        if r not in seen:
            seen.add(r)
            deduped.append(r)
    gt_paths = deduped
    if not gt_paths:
        raise SystemExit("No ground-truth files resolved; use --gt-path or --gt-dir.")

    export_flags = ExportFlags.from_cli(
        emit_debug_json=bool(args.emit_debug_json),
        emit_debug_csv=bool(args.emit_debug_csv),
        emit_debug_html=bool(args.emit_debug_html),
        emit_review_jsonl=bool(args.emit_review_jsonl),
    )
    out = run_pair_score_separation_analysis(
        run_dir=run_dir,
        graph_pt=args.graph_pt.resolve(),
        pair_csv=pair_csv,
        gt_paths=gt_paths,
        output_dir=args.output_dir,
        checkpoint_name=args.checkpoint,
        device=args.device,
        to_undirected=not bool(args.no_to_undirected),
        low_score_max=float(args.low_score_max),
        mid_score_min=float(args.mid_score_min),
        mid_score_max=float(args.mid_score_max),
        high_score_min=float(args.high_score_min),
        community_cut_score_min=float(args.community_cut_score_min),
        community_cut_score_max=float(args.community_cut_score_max),
        enable_mid_band_frontier_analysis=not (
            bool(args.skip_mid_band_analysis) or bool(args.skip_frontier_analysis)
        ),
        enable_frontier_analysis=not (
            bool(args.skip_mid_band_analysis) or bool(args.skip_frontier_analysis)
        ),
        anchor_run_dir=args.anchor_run_dir,
        misp_json_path=args.misp_json_path,
        misp_translated_json_path=args.misp_translated_json_path,
        skip_email_text_export=bool(args.skip_email_text_export),
        email_text_preview_chars=int(args.email_text_preview_chars),
        email_text_wrap_width=int(args.email_text_wrap_width),
        enable_rescued_collapsed_analysis=not bool(args.skip_rescued_collapsed_analysis),
        enable_high_cross_unlabeled_analysis=not bool(args.skip_high_cross_analysis),
        high_cross_score_min=float(args.high_cross_score_min),
        mid_cross_score_min=float(args.mid_cross_score_min),
        rescued_score_min=float(args.rescued_score_min),
        collapsed_score_max=float(args.collapsed_score_max),
        export_flags=export_flags,
    )
    print(
        json.dumps(
            {
                "wrote": out["summary_path"],
                "manifest": out.get("manifest_path"),
                "primary_outputs": out.get("primary_outputs"),
                "band_csv": out["band_csv_path"],
                "low_band_separator_json": out["low_separator_summary_path"],
                "low_band_separator_csv": out["low_separator_csv_path"],
                "low_band_joint_separator_json": out["low_joint_separator_summary_path"],
                "low_band_joint_separator_csv": out["low_joint_separator_csv_path"],
                "high_band_separator_json": out["high_separator_summary_path"],
                "high_band_separator_csv": out["high_separator_csv_path"],
                "high_band_joint_separator_json": out["high_joint_separator_summary_path"],
                "high_band_joint_separator_csv": out["high_joint_separator_csv_path"],
                "high_band_false_positive_pairs_csv": out["high_false_positive_pairs_csv_path"],
                "high_band_true_positive_pairs_csv": out["high_true_positive_pairs_csv_path"],
                "high_band_false_positive_artifact_summary_csv": out[
                    "high_false_positive_artifact_summary_csv_path"
                ],
                "high_band_false_positive_summary_json": out["high_false_positive_summary_json_path"],
                "high_band_false_positive_pairs_for_review_csv": out.get(
                    "high_false_positive_pairs_for_review_csv_path"
                ),
                "high_band_false_positive_pairs_for_review_jsonl": out.get(
                    "high_false_positive_pairs_for_review_jsonl_path"
                ),
                "high_band_semantic_only_false_positive_pairs_for_review_jsonl": out.get(
                    "high_semantic_only_false_positive_pairs_for_review_jsonl_path"
                ),
                "high_band_false_positive_pairs_for_review_html": out.get(
                    "high_false_positive_pairs_for_review_html_path"
                ),
                "high_band_semantic_only_false_positive_pairs_for_review_html": out.get(
                    "high_semantic_only_false_positive_pairs_for_review_html_path"
                ),
                "cross_campaign_positive_pairs_for_review_html": out.get(
                    "cross_campaign_positive_pairs_for_review_html_path"
                ),
                "cross_campaign_positive_semantic_only_pairs_for_review_html": out.get(
                    "cross_campaign_positive_semantic_only_pairs_for_review_html_path"
                ),
                "low_band_unlabeled_pairs_csv": out.get("low_band_unlabeled_pairs_csv_path"),
                "low_band_twohop_channel_summary_json": out.get(
                    "low_band_twohop_channel_summary_json_path"
                ),
                "low_band_twohop_channel_summary_csv": out.get(
                    "low_band_twohop_channel_summary_csv_path"
                ),
                "low_band_unlabeled_pairs_for_review_html": out.get(
                    "low_band_unlabeled_pairs_for_review_html_path"
                ),
                "low_band_same_campaign_unlabeled_pairs_for_review_html": out.get(
                    "low_band_same_campaign_unlabeled_pairs_for_review_html_path"
                ),
                "low_band_cross_campaign_unlabeled_pairs_for_review_html": out.get(
                    "low_band_cross_campaign_unlabeled_pairs_for_review_html_path"
                ),
                "rescued_collapsed_summary_json": out.get("rescued_collapsed_summary_path"),
                "rescued_collapsed_table_csv": out.get("rescued_collapsed_table_path"),
                "rescued_collapsed_joint_json": out.get("rescued_collapsed_joint_path"),
                "rescued_collapsed_review_html": out.get("rescued_collapsed_review_html_path"),
                "plots_under": str(Path(out["output_dir"]) / "plots"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
