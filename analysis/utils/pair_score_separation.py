"""
Post-training score separation: same-campaign vs cross-campaign on GT-covered candidate pairs.

Loads a pair-supervision checkpoint, scores rows from pair_training_dataset.csv,
labels pairs using ground-truth JSON (email external_id -> campaign), and writes
plots + pair_score_separation_summary.json.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from analysis.utils.anchor_graph_helpers import load_anchor_graph_artifacts
from analysis.utils.raw_gnn_notebook import load_ground_truth_structures
from src.load_graph_data import load_hetero_pt
from src.model import HeteroSAGE
from src.pair_graph_sampling import sample_hetero_around_pair_endpoints
from src.pair_scorer import build_email_pair_mlp_scorer
from src.pair_train import (
    PAIR_FEATURE_COLUMNS,
    build_pair_feature_matrix,
    forward_encoder_and_pair_logits,
    iter_pair_batches,
    load_pair_training_dataframe,
)


def load_pair_supervision_for_inference(
    *,
    run_dir: Path,
    graph_pt: Path,
    checkpoint_name: str = "best_model.pt",
    device: str = "cpu",
    to_undirected: bool = True,
) -> dict[str, Any]:
    """
    Load training_config, hetero graph, HeteroSAGE + pair scorer for inference-only stages.
    Caller must have ``core/GNN`` on sys.path (same as pair training).
    """
    run_dir = Path(run_dir).resolve()
    graph_pt = Path(graph_pt).resolve()
    ckpt_path = run_dir / "models" / checkpoint_name
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    cfg_path = run_dir / "training_config.json"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"training_config.json not found under run_dir: {cfg_path}")
    with open(cfg_path, encoding="utf-8") as f:
        train_cfg = json.load(f)
    fanout = list(train_cfg.get("pair_fanout") or train_cfg.get("fanout") or [25, 15])
    pair_batch_size = int(train_cfg.get("pair_batch_size", 64))
    max_unique = int(train_cfg.get("pair_max_unique_emails_per_graph_batch", 2048))

    dev = torch.device(device)
    data = load_hetero_pt(str(graph_pt), to_undirected=to_undirected)
    data_cpu = data.to("cpu")
    metadata = data_cpu.metadata()

    ckpt = torch.load(str(ckpt_path), map_location=dev, weights_only=False)  # nosemgrep
    enc = train_cfg
    hidden = int(enc.get("hidden", 128))
    out_dim = int(enc.get("out_dim", 128))
    layers = int(enc.get("layers", 2))
    dropout = float(enc.get("dropout", 0.0))

    model = HeteroSAGE(metadata=metadata, hidden=hidden, out=out_dim, layers=layers, dropout=dropout).to(dev)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)

    pair_feat_dim = int(enc.get("pair_feature_dim_passed_to_scorer") or len(PAIR_FEATURE_COLUMNS))
    use_exp = bool(enc.get("pair_scorer_use_explicit_features", True))
    if not use_exp:
        pair_feat_dim = 0
    pair_scorer = build_email_pair_mlp_scorer(out_dim, pair_feat_dim, train_cfg).to(dev)
    pair_scorer.load_state_dict(ckpt["pair_scorer_state_dict"], strict=True)

    return {
        "train_cfg": train_cfg,
        "model": model,
        "pair_scorer": pair_scorer,
        "data_cpu": data_cpu,
        "fanout": fanout,
        "pair_batch_size": pair_batch_size,
        "max_unique_emails": max_unique,
        "device": dev,
        "checkpoint_path": str(ckpt_path),
        "training_config_path": str(cfg_path),
    }


def _sanitize_filename_stem(name: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9._-]+", "_", name).strip("_")
    return s[:120] if s else "gt"


def _quantiles_dict(x: np.ndarray, qs: tuple[float, ...]) -> dict[str, float]:
    if x.size == 0:
        return {f"q{int(q * 100)}": float("nan") for q in qs}
    return {f"q{int(q * 100)}": float(np.quantile(x, q)) for q in qs}


def _safe_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    try:
        from sklearn.metrics import roc_auc_score
    except ImportError:
        return None
    if y_true.size < 2 or len(np.unique(y_true)) < 2:
        return None
    return float(roc_auc_score(y_true, y_score))


@torch.no_grad()
def score_pair_rows(
    *,
    model: HeteroSAGE,
    pair_scorer: torch.nn.Module,
    data_cpu: Any,
    df_work: pd.DataFrame,
    device: torch.device,
    fanout: list[int],
    pair_batch_size: int,
    max_unique_emails: int,
    with_logits: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Sigmoid scores aligned to df_work row order (NaN if endpoint batch mapping failed).

    If ``with_logits`` is True, returns ``(pu_score, pu_logit)`` arrays of the same shape.
    """
    model.eval()
    pair_scorer.eval()
    n = len(df_work)
    scores = np.full(n, np.nan, dtype=np.float64)
    logits_out = np.full(n, np.nan, dtype=np.float64) if with_logits else None
    for chunk, gi, gj in iter_pair_batches(df_work, pair_batch_size, max_unique_emails):
        sample = sample_hetero_around_pair_endpoints(data_cpu, gi, gj, fanout)
        feats: torch.Tensor | None = None
        if pair_scorer.use_explicit_pair_features:
            feats = torch.from_numpy(build_pair_feature_matrix(chunk))
        logits, ok_m, _, _ = forward_encoder_and_pair_logits(
            model, pair_scorer, sample, feats, device
        )
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        log_np = logits.detach().cpu().numpy().reshape(-1)
        ok_np = ok_m.cpu().numpy().astype(bool)
        row_ids = chunk["_row"].to_numpy(dtype=np.int64, copy=False)
        for i in range(len(row_ids)):
            if ok_np[i]:
                ri = int(row_ids[i])
                scores[ri] = float(probs[i])
                if logits_out is not None:
                    logits_out[ri] = float(log_np[i])
    if with_logits:
        return scores, logits_out  # type: ignore[return-value]
    return scores


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
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if scores.size == 0:
        ax.text(0.5, 0.5, f"No scored pairs ({cohort_label})", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Number of pairs")
        fig.tight_layout()
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
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
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
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
            project_root / "analysis" / "output" / "graph_bundles" / run_id / "anchor" / run_id
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
        "same_low_value": s,
        "cross_low_value": c,
        "difference_same_minus_cross": diff,
        "abs_difference": (abs(diff) if diff is not None else None),
        "enrichment_same_over_cross": enrich,
    }


def _build_low_band_separator_for_gt(
    *,
    gt_path: Path,
    band_diag: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    groups = band_diag.get("groups") or {}
    same = groups.get("same_campaign_low_score") or {}
    cross = groups.get("cross_campaign_low_score") or {}
    n_total_low_eval = int(band_diag.get("n_gt_covered_candidate_pairs_with_finite_scores") or 0)
    n_same = int(same.get("n_edges") or 0)
    n_cross = int(cross.get("n_edges") or 0)
    n_low = int(n_same + n_cross)

    out: dict[str, Any] = {
        "gt_path": str(gt_path.resolve()),
        "low_band_thresholds": band_diag.get("band_definitions", {}).get("low", {}),
        "counts": {
            "n_same_campaign_low_score": n_same,
            "n_cross_campaign_low_score": n_cross,
            "n_total_low_band_gt_covered_pairs": n_low,
            "n_total_gt_covered_candidate_pairs_with_finite_scores": n_total_low_eval,
            "fraction_gt_covered_pairs_that_are_low_band_same": (
                float(n_same / n_total_low_eval) if n_total_low_eval > 0 else None
            ),
            "fraction_gt_covered_pairs_that_are_low_band_cross": (
                float(n_cross / n_total_low_eval) if n_total_low_eval > 0 else None
            ),
        },
    }

    rows: list[dict[str, Any]] = []

    prov_keys = [
        "from_semantic",
        "from_rare_artifact",
        "from_2hop",
        "from_component",
        "source_count_eq_1",
        "source_count_eq_2",
        "source_count_ge_3",
        "same_seed_component_flag",
        "cross_seed_component_flag",
    ]
    prov_out: dict[str, Any] = {}
    for k in prov_keys:
        same_v = ((same.get("provenance") or {}).get(k) or {}).get("fraction")
        cross_v = ((cross.get("provenance") or {}).get(k) or {}).get("fraction")
        cmp = _compare_fraction_metric(same_v=same_v, cross_v=cross_v)
        prov_out[k] = cmp
        rows.append(
            {
                "gt_path": str(gt_path.resolve()),
                "metric_group": "provenance",
                "metric_name": k,
                "same_low_value": cmp["same_low_value"],
                "cross_low_value": cmp["cross_low_value"],
                "difference": cmp["difference_same_minus_cross"],
                "enrichment": cmp["enrichment_same_over_cross"],
                "abs_difference": cmp["abs_difference"],
            }
        )
    out["provenance_comparison"] = prov_out

    feature_keys = [
        "semantic_cosine_max",
        "rare_artifact_rarity_max",
        "twohop_rarity_max",
        "component_cosine_max",
        "time_gap_seconds_min",
    ]
    feat_out: dict[str, Any] = {}
    for k in feature_keys:
        ssum = (same.get("feature_summaries") or {}).get(k) or {}
        csum = (cross.get("feature_summaries") or {}).get(k) or {}
        ms = _safe_float(ssum.get("mean"))
        mc = _safe_float(csum.get("mean"))
        med_s = _safe_float(ssum.get("median"))
        med_c = _safe_float(csum.get("median"))
        feat_out[k] = {
            "mean_same_low": ms,
            "mean_cross_low": mc,
            "median_same_low": med_s,
            "median_cross_low": med_c,
            "difference_in_means_same_minus_cross": (ms - mc) if (ms is not None and mc is not None) else None,
            "difference_in_medians_same_minus_cross": (
                (med_s - med_c) if (med_s is not None and med_c is not None) else None
            ),
            "n_missing_same_low": int(ssum.get("n_missing") or 0),
            "n_missing_cross_low": int(csum.get("n_missing") or 0),
            "n_non_null_same_low": int(ssum.get("n_non_null") or 0),
            "n_non_null_cross_low": int(csum.get("n_non_null") or 0),
        }
        rows.append(
            {
                "gt_path": str(gt_path.resolve()),
                "metric_group": "feature_mean",
                "metric_name": k,
                "same_low_value": ms,
                "cross_low_value": mc,
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

    shared_keys = [
        "shared_url",
        "shared_sender",
        "shared_attachment",
        "shared_sender_domain",
        "shared_domain",
        "shared_stem",
    ]
    shared_out: dict[str, Any] = {}
    for k in shared_keys:
        same_v = ((same.get("shared_evidence") or {}).get(k) or {}).get("fraction_edges_with_at_least_1")
        cross_v = ((cross.get("shared_evidence") or {}).get(k) or {}).get("fraction_edges_with_at_least_1")
        cmp = _compare_fraction_metric(same_v=same_v, cross_v=cross_v)
        shared_out[k] = cmp
        rows.append(
            {
                "gt_path": str(gt_path.resolve()),
                "metric_group": "shared_evidence",
                "metric_name": k,
                "same_low_value": cmp["same_low_value"],
                "cross_low_value": cmp["cross_low_value"],
                "difference": cmp["difference_same_minus_cross"],
                "enrichment": cmp["enrichment_same_over_cross"],
                "abs_difference": cmp["abs_difference"],
            }
        )
    out["shared_evidence_comparison"] = shared_out

    # Ranked separators: provenance + shared evidence + feature mean deltas
    ranked = [r for r in rows if r.get("abs_difference") is not None]
    ranked.sort(key=lambda r: float(r["abs_difference"]), reverse=True)
    top = ranked[:10]
    out["ranked_separators_top10"] = [
        {
            "rank": i + 1,
            "metric_group": r["metric_group"],
            "metric_name": r["metric_name"],
            "same_low_value": r["same_low_value"],
            "cross_low_value": r["cross_low_value"],
            "difference_same_minus_cross": r["difference"],
            "abs_difference": r["abs_difference"],
            "enrichment_same_over_cross": r["enrichment"],
        }
        for i, r in enumerate(top)
    ]

    return out, rows


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
) -> dict[str, Any]:
    ns = int(base_same.sum())
    nc = int(base_cross.sum())
    fs = float((cond_same & base_same).sum() / ns) if ns > 0 else None
    fc = float((cond_cross & base_cross).sum() / nc) if nc > 0 else None
    diff = (fs - fc) if (fs is not None and fc is not None) else None
    return {
        "same_low_value": fs,
        "cross_low_value": fc,
        "difference_same_minus_cross": diff,
        "abs_difference": (abs(diff) if diff is not None else None),
        "enrichment_same_over_cross": _safe_enrichment(fs, fc),
        "n_same_low": ns,
        "n_cross_low": nc,
    }


def _build_low_band_joint_separator_for_gt(
    *,
    gt_path: Path,
    df_eval: pd.DataFrame,
    same_low_mask_eval: np.ndarray,
    cross_low_mask_eval: np.ndarray,
    low_max: float,
    nodes_by_email: dict[str, dict[str, set[str]]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """
    Joint separator analysis between same_campaign_low_score and cross_campaign_low_score.
    """
    out_rows: list[dict[str, Any]] = []
    n_eval = int(len(df_eval))
    if n_eval == 0:
        empty = {
            "gt_path": str(gt_path.resolve()),
            "low_band_thresholds": {"min_inclusive": 0.0, "max_inclusive": float(low_max)},
            "counts": {
                "n_same_campaign_low_score": 0,
                "n_cross_campaign_low_score": 0,
                "n_total_low_band_gt_covered_pairs": 0,
            },
            "binary_joint_comparisons": {},
            "semantic_bucket_analysis": {},
            "candidate_rule_analysis": {},
            "ranked_joint_separators_top15": [],
        }
        return empty, out_rows

    # Base booleans from pair rows
    fs = df_eval.get("from_semantic", False).fillna(False).astype(bool).to_numpy()
    f2 = df_eval.get("from_2hop", False).fillna(False).astype(bool).to_numpy()
    fc = df_eval.get("from_component", False).fillna(False).astype(bool).to_numpy()
    sc = pd.to_numeric(df_eval.get("source_count"), errors="coerce")
    sem = pd.to_numeric(df_eval.get("semantic_cosine_max"), errors="coerce")

    # Shared-evidence derived booleans at row level.
    n = len(df_eval)
    has_shared_sender = np.zeros(n, dtype=bool)
    has_shared_stem = np.zeros(n, dtype=bool)
    has_shared_sender_domain = np.zeros(n, dtype=bool)
    has_shared_url = np.zeros(n, dtype=bool)
    has_shared_attachment = np.zeros(n, dtype=bool)
    has_shared_domain = np.zeros(n, dtype=bool)
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

    # base masks
    same_low = same_low_mask_eval.astype(bool)
    cross_low = cross_low_mask_eval.astype(bool)
    n_same = int(same_low.sum())
    n_cross = int(cross_low.sum())
    n_low = int(n_same + n_cross)

    condition_defs: list[tuple[str, np.ndarray]] = [
        ("from_semantic_AND_shared_sender", fs & has_shared_sender),
        ("from_semantic_AND_NOT_shared_sender", fs & ~has_shared_sender),
        ("from_2hop_AND_shared_sender", f2 & has_shared_sender),
        ("from_2hop_AND_NOT_shared_sender", f2 & ~has_shared_sender),
        ("from_component_AND_shared_sender", fc & has_shared_sender),
        ("from_component_AND_NOT_shared_sender", fc & ~has_shared_sender),
        ("from_semantic_AND_from_2hop", fs & f2),
        ("from_semantic_AND_NOT_from_2hop", fs & ~f2),
        ("from_2hop_AND_NOT_from_semantic", f2 & ~fs),
        ("from_component_AND_NOT_from_semantic", fc & ~fs),
        ("from_component_AND_from_2hop", fc & f2),
        ("shared_sender_AND_shared_stem", has_shared_sender & has_shared_stem),
        ("shared_sender_AND_NOT_shared_stem", has_shared_sender & ~has_shared_stem),
        ("shared_sender_domain_AND_NOT_shared_sender", has_shared_sender_domain & ~has_shared_sender),
        ("shared_sender_domain_AND_shared_sender", has_shared_sender_domain & has_shared_sender),
    ]

    bin_out: dict[str, Any] = {}
    for name, cond in condition_defs:
        cmp = _cmp_from_masks(
            cond_same=cond,
            base_same=same_low,
            cond_cross=cond,
            base_cross=cross_low,
        )
        bin_out[name] = cmp
        out_rows.append(
            {
                "gt_path": str(gt_path.resolve()),
                "analysis_section": "binary_joint_comparisons",
                "condition_name": name,
                "same_low_value": cmp["same_low_value"],
                "cross_low_value": cmp["cross_low_value"],
                "difference": cmp["difference_same_minus_cross"],
                "enrichment": cmp["enrichment_same_over_cross"],
                "abs_difference": cmp["abs_difference"],
            }
        )

    # Bucketed semantic analysis
    bucket_defs: list[tuple[str, np.ndarray]] = [
        ("semantic_lt_0_91", sem.lt(0.91).fillna(False).to_numpy()),
        ("semantic_0_91_to_0_93", sem.ge(0.91).fillna(False).to_numpy() & sem.lt(0.93).fillna(False).to_numpy()),
        ("semantic_0_93_to_0_95", sem.ge(0.93).fillna(False).to_numpy() & sem.lt(0.95).fillna(False).to_numpy()),
        ("semantic_ge_0_95", sem.ge(0.95).fillna(False).to_numpy()),
    ]
    sem_out: dict[str, Any] = {}
    for bname, bmask in bucket_defs:
        cmp_base = _cmp_from_masks(cond_same=bmask, base_same=same_low, cond_cross=bmask, base_cross=cross_low)
        sem_out[bname] = {"bucket": cmp_base}
        out_rows.append(
            {
                "gt_path": str(gt_path.resolve()),
                "analysis_section": "semantic_bucket_analysis",
                "condition_name": bname,
                "same_low_value": cmp_base["same_low_value"],
                "cross_low_value": cmp_base["cross_low_value"],
                "difference": cmp_base["difference_same_minus_cross"],
                "enrichment": cmp_base["enrichment_same_over_cross"],
                "abs_difference": cmp_base["abs_difference"],
            }
        )
        # Crossed with sender and 2hop
        crossed = [
            (f"{bname}_AND_shared_sender", bmask & has_shared_sender),
            (f"{bname}_AND_NOT_shared_sender", bmask & ~has_shared_sender),
            (f"{bname}_AND_from_2hop", bmask & f2),
            (f"{bname}_AND_NOT_from_2hop", bmask & ~f2),
        ]
        for cname, cmask in crossed:
            cmp = _cmp_from_masks(cond_same=cmask, base_same=same_low, cond_cross=cmask, base_cross=cross_low)
            sem_out[bname][cname] = cmp
            out_rows.append(
                {
                    "gt_path": str(gt_path.resolve()),
                    "analysis_section": "semantic_bucket_analysis",
                    "condition_name": cname,
                    "same_low_value": cmp["same_low_value"],
                    "cross_low_value": cmp["cross_low_value"],
                    "difference": cmp["difference_same_minus_cross"],
                    "enrichment": cmp["enrichment_same_over_cross"],
                    "abs_difference": cmp["abs_difference"],
                }
            )

    # Candidate rule templates
    rule_defs: list[tuple[str, np.ndarray]] = [
        ("likely_positive__from_semantic_AND_shared_sender", fs & has_shared_sender),
        ("likely_positive__from_semantic_AND_semantic_ge_0_93", fs & sem.ge(0.93).fillna(False).to_numpy()),
        ("likely_positive__from_semantic_AND_shared_sender_AND_NOT_from_2hop", fs & has_shared_sender & ~f2),
        ("likely_positive__shared_sender_AND_NOT_from_2hop", has_shared_sender & ~f2),
        ("likely_negative__from_2hop_AND_NOT_shared_sender", f2 & ~has_shared_sender),
        ("likely_negative__from_2hop_AND_NOT_from_semantic", f2 & ~fs),
        ("likely_negative__from_component_AND_NOT_shared_sender", fc & ~has_shared_sender),
        ("likely_negative__shared_sender_domain_AND_NOT_shared_sender", has_shared_sender_domain & ~has_shared_sender),
    ]
    rule_out: dict[str, Any] = {}
    for rname, rmask in rule_defs:
        cmp = _cmp_from_masks(cond_same=rmask, base_same=same_low, cond_cross=rmask, base_cross=cross_low)
        rule_out[rname] = cmp
        out_rows.append(
            {
                "gt_path": str(gt_path.resolve()),
                "analysis_section": "candidate_rule_analysis",
                "condition_name": rname,
                "same_low_value": cmp["same_low_value"],
                "cross_low_value": cmp["cross_low_value"],
                "difference": cmp["difference_same_minus_cross"],
                "enrichment": cmp["enrichment_same_over_cross"],
                "abs_difference": cmp["abs_difference"],
            }
        )

    ranked = [r for r in out_rows if r.get("abs_difference") is not None]
    ranked.sort(key=lambda r: float(r["abs_difference"]), reverse=True)
    ranked_top = ranked[:15]

    out = {
        "gt_path": str(gt_path.resolve()),
        "low_band_thresholds": {"min_inclusive": 0.0, "max_inclusive": float(low_max)},
        "counts": {
            "n_same_campaign_low_score": n_same,
            "n_cross_campaign_low_score": n_cross,
            "n_total_low_band_gt_covered_pairs": n_low,
        },
        "binary_joint_comparisons": bin_out,
        "semantic_bucket_analysis": sem_out,
        "candidate_rule_analysis": rule_out,
        "ranked_joint_separators_top15": [
            {
                "rank": i + 1,
                "analysis_section": r["analysis_section"],
                "condition_name": r["condition_name"],
                "same_low_value": r["same_low_value"],
                "cross_low_value": r["cross_low_value"],
                "difference_same_minus_cross": r["difference"],
                "abs_difference": r["abs_difference"],
                "enrichment_same_over_cross": r["enrichment"],
            }
            for i, r in enumerate(ranked_top)
        ],
    }
    return out, out_rows


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
    low_score_max: float = 0.4,
    high_score_min: float = 0.8,
    anchor_run_dir: Path | None = None,
) -> dict[str, Any]:
    run_dir = Path(run_dir).resolve()
    graph_pt = Path(graph_pt).resolve()
    pair_csv = Path(pair_csv).resolve()
    out_dir = (output_dir or (run_dir / "pair_score_separation")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    df, _stats = load_pair_training_dataframe(pair_csv)
    df_work = df.reset_index(drop=True)
    df_work["_row"] = np.arange(len(df_work), dtype=np.int64)

    cross_comp = None
    if "cross_seed_component_flag" in df_work.columns:
        cross_comp = df_work["cross_seed_component_flag"].fillna(False).astype(bool).to_numpy()

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
    )
    project_root = Path(__file__).resolve().parents[2]
    nodes_by_email, shared_ctx = _load_anchor_nodes_by_email(
        pair_csv=pair_csv,
        project_root=project_root,
        explicit_anchor_run_dir=anchor_run_dir,
    )

    per_gt: list[dict[str, Any]] = []
    band_rows: list[dict[str, Any]] = []
    low_sep_rows: list[dict[str, Any]] = []
    low_sep_per_gt: list[dict[str, Any]] = []
    low_joint_rows: list[dict[str, Any]] = []
    low_joint_per_gt: list[dict[str, Any]] = []
    for gt_path in gt_paths:
        gt_path = Path(gt_path).resolve()
        label_map, _eid_row, _camp = load_ground_truth_structures(gt_path)
        label_map = {str(k): v for k, v in label_map.items()}

        ei = df_work["email_i"].astype(str).values
        ej = df_work["email_j"].astype(str).values
        n = len(df_work)
        scored = np.isfinite(scores)
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

        cc_plot_same: Path | None = None
        cc_plot_cross: Path | None = None
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
        summary["plot_same_campaign"] = str(plot_same.relative_to(out_dir))
        summary["plot_cross_campaign"] = str(plot_cross.relative_to(out_dir))
        if cc_plot_same is not None and cc_plot_cross is not None:
            summary["plot_cross_component_same_campaign"] = str(cc_plot_same.relative_to(out_dir))
            summary["plot_cross_component_cross_campaign"] = str(cc_plot_cross.relative_to(out_dir))
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
        same_low_eval = same_eval & low.to_numpy(dtype=bool, copy=False)
        cross_low_eval = cross_eval & low.to_numpy(dtype=bool, copy=False)
        low_joint, low_joint_table = _build_low_band_joint_separator_for_gt(
            gt_path=gt_path,
            df_eval=df_eval,
            same_low_mask_eval=same_low_eval,
            cross_low_mask_eval=cross_low_eval,
            low_max=low_score_max,
            nodes_by_email=nodes_by_email,
        )
        low_joint_per_gt.append(low_joint)
        low_joint_rows.extend(low_joint_table)
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
            "high_score_band": [float(high_score_min), 1.0],
        },
        "shared_evidence_context": shared_ctx,
        "per_gt": per_gt,
        "n_pair_rows_scored": int(len(df_work)),
        "n_finite_scores": int(np.isfinite(scores).sum()),
    }
    summary_path = out_dir / "pair_score_separation_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
    band_csv_path = out_dir / "pair_score_band_diagnostics.csv"
    pd.DataFrame(band_rows).to_csv(band_csv_path, index=False)
    low_sep_summary_path = out_dir / "pair_low_band_separator_summary.json"
    low_sep_payload = {
        "run_dir": str(run_dir),
        "graph_pt": str(graph_pt),
        "pair_csv": str(pair_csv),
        "checkpoint": str(bundle["checkpoint_path"]),
        "band_config": {
            "low_score_band": [0.0, float(low_score_max)],
            "high_score_band": [float(high_score_min), 1.0],
        },
        "per_gt": low_sep_per_gt,
    }
    with open(low_sep_summary_path, "w", encoding="utf-8") as f:
        json.dump(low_sep_payload, f, indent=2, default=str)
    low_sep_csv_path = out_dir / "pair_low_band_separator_table.csv"
    pd.DataFrame(low_sep_rows).to_csv(low_sep_csv_path, index=False)
    low_joint_summary_path = out_dir / "pair_low_band_joint_separator_summary.json"
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
    with open(low_joint_summary_path, "w", encoding="utf-8") as f:
        json.dump(low_joint_payload, f, indent=2, default=str)
    low_joint_csv_path = out_dir / "pair_low_band_joint_separator_table.csv"
    pd.DataFrame(low_joint_rows).to_csv(low_joint_csv_path, index=False)

    return {
        "output_dir": str(out_dir),
        "summary_path": str(summary_path),
        "band_csv_path": str(band_csv_path),
        "low_separator_summary_path": str(low_sep_summary_path),
        "low_separator_csv_path": str(low_sep_csv_path),
        "low_joint_separator_summary_path": str(low_joint_summary_path),
        "low_joint_separator_csv_path": str(low_joint_csv_path),
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
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument(
        "--gt-dir",
        type=Path,
        default=None,
        help="Use every *.json in this directory (default: skip filenames containing 'report').",
    )
    g.add_argument(
        "--gt-path",
        type=Path,
        default=None,
        help="Analyze exactly one ground-truth JSON file.",
    )
    p.add_argument(
        "--gt-include-report-json",
        action="store_true",
        help="With --gt-dir, also include *report*.json (not cluster maps; usually useless for this analysis).",
    )
    p.add_argument("--output-dir", type=Path, default=None, help="Output root (default: <run-dir>/pair_score_separation)")
    p.add_argument("--checkpoint", type=str, default="best_model.pt", help="Checkpoint filename under run_dir/models/")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--low-score-max", type=float, default=0.4, help="Low-score band upper bound (inclusive).")
    p.add_argument("--high-score-min", type=float, default=0.8, help="High-score band lower bound (exclusive).")
    p.add_argument("--anchor-run-dir", type=Path, default=None, help="Optional anchor graph run dir (contains anchor_graph_nodes.csv) for shared-evidence diagnostics.")
    p.add_argument(
        "--no-to-undirected",
        action="store_true",
        help="Load graph without ToUndirected (default: undirected, matching training)",
    )
    args = p.parse_args(argv)

    run_dir = args.run_dir.resolve()
    cfg_path = run_dir / "training_config.json"
    if not cfg_path.is_file():
        raise SystemExit(f"Missing {cfg_path}")
    with open(cfg_path, encoding="utf-8") as f:
        tc = json.load(f)
    pair_csv = args.pair_csv
    if pair_csv is None:
        raw = tc.get("pair_dataset_csv")
        if not raw:
            raise SystemExit("pair_dataset_csv not in training_config.json; pass --pair-csv")
        pair_csv = Path(raw)
        if not pair_csv.is_absolute():
            repo = Path(__file__).resolve().parents[2]
            pair_csv = (repo / pair_csv).resolve()

    gt_paths: list[Path] = []
    if args.gt_path is not None:
        gt_paths = [args.gt_path.resolve()]
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
        high_score_min=float(args.high_score_min),
        anchor_run_dir=args.anchor_run_dir,
    )
    print(
        json.dumps(
            {
                "wrote": out["summary_path"],
                "band_csv": out["band_csv_path"],
                "low_band_separator_json": out["low_separator_summary_path"],
                "low_band_separator_csv": out["low_separator_csv_path"],
                "low_band_joint_separator_json": out["low_joint_separator_summary_path"],
                "low_band_joint_separator_csv": out["low_joint_separator_csv_path"],
                "plots_under": str(Path(out["output_dir"]) / "plots"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
