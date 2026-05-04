from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd

from seed_candidate_workflow.utils.raw_gnn_notebook import load_ground_truth_structures


DEFAULT_THRESHOLDS = [0.0, 0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.90]


def _sanitize_stem(name: str) -> str:
    out = "".join(ch if (ch.isalnum() or ch in "._-") else "_" for ch in str(name))
    out = out.strip("._-")
    return out or "gt"


def _to_bool(df: pd.DataFrame, col: str, default: bool = False) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype=bool)
    return df[col].fillna(False).astype(bool)


def _to_float(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=np.float64)
    return pd.to_numeric(df[col], errors="coerce")


def _null_json(x: Any) -> Any:
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return None
    if isinstance(x, dict):
        return {k: _null_json(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_null_json(v) for v in x]
    return x


def _frac(num: int, den: int) -> float | None:
    if den <= 0:
        return None
    return float(num / den)


def _cross_same_seed_component_masks(df: pd.DataFrame) -> tuple[pd.Series, pd.Series, str]:
    """
    Cross / same seed-component structure for retention (pair-training schema).

    Prefer explicit flag columns; if absent, infer from non-negative seed_component_i/j.
    """
    has_ids = "seed_component_i" in df.columns and "seed_component_j" in df.columns
    if "cross_seed_component_flag" in df.columns and "same_seed_component_flag" in df.columns:
        cross = _to_bool(df, "cross_seed_component_flag", default=False)
        same = _to_bool(df, "same_seed_component_flag", default=False)
        return cross, same, "csv_flag_columns"
    if has_ids:
        sci = pd.to_numeric(df["seed_component_i"], errors="coerce")
        scj = pd.to_numeric(df["seed_component_j"], errors="coerce")
        both_ids = sci.notna() & scj.notna() & (sci >= 0) & (scj >= 0)
        cross = both_ids & (sci != scj)
        same = both_ids & (sci == scj)
        return (
            cross.fillna(False).astype(bool),
            same.fillna(False).astype(bool),
            "inferred_seed_component_ids",
        )
    z = pd.Series(False, index=df.index, dtype=bool)
    return z, z, "missing_no_flags_or_ids"


def _campaign_connectivity_statuses(
    *,
    campaign_to_members: dict[Any, list[str]],
    same_kept_pairs: set[tuple[str, str]],
) -> dict[str, int]:
    n_size_ge_2 = 0
    n_reconnectable = 0
    n_fully = 0
    n_partially = 0
    n_singleton_fractured = 0

    for _cid, members_raw in campaign_to_members.items():
        members = sorted({str(x) for x in members_raw if str(x).strip()})
        if len(members) < 2:
            continue
        n_size_ge_2 += 1
        g = nx.Graph()
        g.add_nodes_from(members)
        mem_set = set(members)
        for a, b in same_kept_pairs:
            if a in mem_set and b in mem_set:
                g.add_edge(a, b)
        comps = list(nx.connected_components(g))
        if not comps:
            n_singleton_fractured += 1
            continue
        largest = max((len(c) for c in comps), default=1)
        if largest >= 2:
            n_reconnectable += 1
        if len(comps) == 1 and largest == len(members):
            n_fully += 1
        elif largest >= 2:
            n_partially += 1
        else:
            n_singleton_fractured += 1

    return {
        "n_gt_campaigns_size_ge_2": n_size_ge_2,
        "n_gt_campaigns_reconnectable": n_reconnectable,
        "n_gt_campaigns_fully_connected": n_fully,
        "n_gt_campaigns_partially_connected": n_partially,
        "n_gt_campaigns_singleton_fractured": n_singleton_fractured,
    }


def run_pu_threshold_retention_analysis(
    *,
    scored_pairs_csv: Path,
    gt_paths: list[Path],
    thresholds: list[float] | None = None,
    output_dir: Path | None = None,
    keep_seeds_always: bool = True,
    make_plots: bool = True,
) -> dict[str, Any]:
    scored_pairs_csv = Path(scored_pairs_csv).resolve()
    if not scored_pairs_csv.is_file():
        raise FileNotFoundError(f"scored_pairs_csv not found: {scored_pairs_csv}")
    if not gt_paths:
        raise ValueError("Provide at least one GT path")
    gt_paths = [Path(p).resolve() for p in gt_paths]
    thresholds = [float(x) for x in (thresholds or DEFAULT_THRESHOLDS)]
    thresholds = sorted(set(thresholds))

    out_dir = (output_dir or scored_pairs_csv.parent / "pu_threshold_retention").resolve()
    os.makedirs(out_dir, exist_ok=True)
    # Short name keeps full PNG paths under Windows MAX_PATH when nested under long bundle dirs.
    plots_dir = (out_dir / "plots").resolve()
    if make_plots:
        os.makedirs(plots_dir, exist_ok=True)

    df = pd.read_csv(scored_pairs_csv, low_memory=False)
    if "email_i" not in df.columns or "email_j" not in df.columns:
        raise ValueError("scored_pairs_csv must include email_i and email_j")

    df_work = df.copy()
    df_work["email_i"] = df_work["email_i"].astype(str)
    df_work["email_j"] = df_work["email_j"].astype(str)
    is_candidate = _to_bool(df_work, "is_candidate_pair", default=True)
    trusted_seed = _to_bool(df_work, "trusted_seed_edge", default=False)
    if "trusted_seed_edge" not in df_work.columns:
        trusted_seed = _to_bool(df_work, "is_seed_pair", default=False) | _to_bool(
            df_work, "from_seed", default=False
        )
    pu_score = _to_float(df_work, "pu_score")
    cross_component, same_component, mask_src = _cross_same_seed_component_masks(df_work)

    source_flags = {
        "from_seed": _to_bool(df_work, "from_seed", default=False),
        "from_semantic": _to_bool(df_work, "from_semantic", default=False),
        "from_rare_artifact": _to_bool(df_work, "from_rare_artifact", default=False),
        "from_component": _to_bool(df_work, "from_component", default=False),
        "from_2hop": _to_bool(df_work, "from_2hop", default=False),
    }
    source_count = _to_float(df_work, "source_count").fillna(0.0)
    non_seed_mask = ~trusted_seed

    per_gt: list[dict[str, Any]] = []
    rows_csv: list[dict[str, Any]] = []

    for gt_path in gt_paths:
        label_map, _eid_row, campaign_to_members = load_ground_truth_structures(gt_path)
        label_map = {str(k): v for k, v in label_map.items()}

        ei = df_work["email_i"].values
        ej = df_work["email_j"].values
        li = np.array([label_map.get(str(x)) is not None for x in ei], dtype=bool)
        lj = np.array([label_map.get(str(x)) is not None for x in ej], dtype=bool)
        both = li & lj & is_candidate.to_numpy(dtype=bool)
        camp_i = np.array([label_map.get(str(x)) for x in ei], dtype=object)
        camp_j = np.array([label_map.get(str(x)) for x in ej], dtype=object)
        same = both & (camp_i == camp_j)
        cross = both & (camp_i != camp_j)

        n_gt_cov = int(both.sum())
        n_same = int(same.sum())
        n_cross = int(cross.sum())

        both_np = both.astype(bool, copy=False)
        cc_np = cross_component.to_numpy(dtype=bool, copy=False)
        ssc_np = same_component.to_numpy(dtype=bool, copy=False)
        n_gt_cov_cc = int((both_np & cc_np).sum())
        n_gt_cov_ssc = int((both_np & ssc_np).sum())

        gt_section: dict[str, Any] = {
            "gt_path": str(gt_path),
            "n_gt_covered_pairs_total": n_gt_cov,
            "n_same_campaign_pairs_total": n_same,
            "n_cross_campaign_pairs_total": n_cross,
            "n_gt_covered_cross_component_pairs_total": n_gt_cov_cc,
            "n_gt_covered_same_seed_component_pairs_total": n_gt_cov_ssc,
            "cross_seed_component_mask_source": mask_src,
            "per_threshold_metrics": [],
        }

        for t in thresholds:
            keep_non_seed = non_seed_mask.to_numpy(dtype=bool) & pu_score.ge(float(t)).to_numpy(dtype=bool)
            keep = keep_non_seed.copy()
            if keep_seeds_always:
                keep = keep | trusted_seed.to_numpy(dtype=bool)

            same_kept = same & keep
            cross_kept = cross & keep
            n_same_kept = int(same_kept.sum())
            n_cross_kept = int(cross_kept.sum())
            n_kept_cov = int((both & keep).sum())
            kept_precision = _frac(n_same_kept, n_kept_cov)

            cc_same_kept = int((same_kept & cross_component.to_numpy(dtype=bool)).sum())
            cc_cross_kept = int((cross_kept & cross_component.to_numpy(dtype=bool)).sum())
            cc_total_kept = cc_same_kept + cc_cross_kept
            cc_precision = _frac(cc_same_kept, cc_total_kept)

            same_kept_pairs: set[tuple[str, str]] = set()
            idxs = np.where(same_kept)[0]
            for i in idxs:
                a, b = str(ei[i]), str(ej[i])
                same_kept_pairs.add((a, b) if a <= b else (b, a))
            camp_stats = _campaign_connectivity_statuses(
                campaign_to_members=campaign_to_members,
                same_kept_pairs=same_kept_pairs,
            )

            surviving_non_seed = is_candidate.to_numpy(dtype=bool) & non_seed_mask.to_numpy(dtype=bool) & keep
            n_non_seed_surv = int(surviving_non_seed.sum())
            class_counts = {
                "semantic": int((surviving_non_seed & source_flags["from_semantic"].to_numpy(dtype=bool)).sum()),
                "rare_artifact": int((surviving_non_seed & source_flags["from_rare_artifact"].to_numpy(dtype=bool)).sum()),
                "component": int((surviving_non_seed & source_flags["from_component"].to_numpy(dtype=bool)).sum()),
                "twohop": int((surviving_non_seed & source_flags["from_2hop"].to_numpy(dtype=bool)).sum()),
                "multi_source": int((surviving_non_seed & source_count.ge(2).to_numpy(dtype=bool)).sum()),
                "cross_component": int((surviving_non_seed & cross_component.to_numpy(dtype=bool)).sum()),
                "internal_same_seed_component": int((surviving_non_seed & same_component.to_numpy(dtype=bool)).sum()),
            }
            class_fracs = {
                k: _frac(v, n_non_seed_surv) for k, v in class_counts.items()
            }

            thr_row: dict[str, Any] = {
                "threshold": float(t),
                "n_same_campaign_pairs_kept": n_same_kept,
                "n_cross_campaign_pairs_kept": n_cross_kept,
                "same_campaign_pair_recall_at_threshold": _frac(n_same_kept, n_same),
                "cross_campaign_pair_retention_at_threshold": _frac(n_cross_kept, n_cross),
                "kept_pair_precision": kept_precision,
                "n_cross_component_same_campaign_kept": cc_same_kept,
                "n_cross_component_cross_campaign_kept": cc_cross_kept,
                "cross_component_pair_precision": cc_precision,
                **camp_stats,
                "pct_gt_campaigns_reconnectable": _frac(
                    camp_stats["n_gt_campaigns_reconnectable"],
                    camp_stats["n_gt_campaigns_size_ge_2"],
                ),
                "pct_gt_campaigns_fully_connected": _frac(
                    camp_stats["n_gt_campaigns_fully_connected"],
                    camp_stats["n_gt_campaigns_size_ge_2"],
                ),
                "pct_gt_campaigns_partially_connected": _frac(
                    camp_stats["n_gt_campaigns_partially_connected"],
                    camp_stats["n_gt_campaigns_size_ge_2"],
                ),
                "pct_gt_campaigns_singleton_fractured": _frac(
                    camp_stats["n_gt_campaigns_singleton_fractured"],
                    camp_stats["n_gt_campaigns_size_ge_2"],
                ),
                "n_surviving_non_seed_candidate_edges": n_non_seed_surv,
                "surviving_non_seed_class_counts": class_counts,
                "surviving_non_seed_class_fractions": class_fracs,
            }
            gt_section["per_threshold_metrics"].append(thr_row)

            row_flat = {
                "gt_path": str(gt_path),
                "threshold": float(t),
                "n_gt_covered_pairs_total": n_gt_cov,
                "n_same_campaign_pairs_total": n_same,
                "n_cross_campaign_pairs_total": n_cross,
                **{k: v for k, v in thr_row.items() if not isinstance(v, dict)},
                **{f"class_count_{k}": v for k, v in class_counts.items()},
                **{f"class_frac_{k}": class_fracs[k] for k in class_fracs},
            }
            rows_csv.append(row_flat)

        if make_plots and gt_section["per_threshold_metrics"]:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            m = gt_section["per_threshold_metrics"]
            xs = [float(x["threshold"]) for x in m]
            stem = _sanitize_stem(Path(str(gt_path)).stem)

            def _plot(ys: list[float | None], ys2: list[float | None] | None, labels: tuple[str, str] | tuple[str], name: str, ylabel: str) -> None:
                os.makedirs(plots_dir, exist_ok=True)
                fig, ax = plt.subplots(figsize=(7.5, 4.5))
                ax.plot(xs, [np.nan if v is None else float(v) for v in ys], marker="o", label=labels[0])
                if ys2 is not None:
                    ax.plot(xs, [np.nan if v is None else float(v) for v in ys2], marker="o", label=labels[1])
                ax.set_xlabel("PU score threshold")
                ax.set_ylabel(ylabel)
                ax.set_title(f"{name} (GT: {Path(str(gt_path)).name})")
                ax.grid(alpha=0.2)
                ax.legend()
                fig.tight_layout()
                out_png = (plots_dir / f"{name}_{stem}.png").resolve()
                os.makedirs(out_png.parent, exist_ok=True)
                fig.savefig(str(out_png), dpi=120, bbox_inches="tight")
                plt.close(fig)

            _plot(
                [x["same_campaign_pair_recall_at_threshold"] for x in m],
                [x["cross_campaign_pair_retention_at_threshold"] for x in m],
                ("same recall", "cross retention"),
                "threshold_vs_pair_retention",
                "fraction",
            )
            _plot(
                [x["kept_pair_precision"] for x in m],
                None,
                ("kept pair precision",),
                "threshold_vs_kept_precision",
                "precision",
            )
            _plot(
                [x["pct_gt_campaigns_reconnectable"] for x in m],
                None,
                ("campaign reconnectability",),
                "threshold_vs_campaign_reconnectability",
                "fraction",
            )
            _plot(
                [float(x["n_cross_component_same_campaign_kept"]) for x in m],
                [float(x["n_cross_component_cross_campaign_kept"]) for x in m],
                ("cross-comp same kept", "cross-comp cross kept"),
                "threshold_vs_cross_component_counts",
                "count",
            )

        per_gt.append(gt_section)

    payload: dict[str, Any] = {
        "metadata": {
            "scored_pairs_csv": str(scored_pairs_csv),
            "output_dir": str(out_dir),
            "keep_seeds_always": bool(keep_seeds_always),
            "analysis_scope": "GT-covered candidate pairs with thresholding on non-seed pu_score",
            "cross_seed_component_mask_source": mask_src,
        },
        "thresholds": thresholds,
        "per_gt": per_gt,
    }
    p_json = out_dir / "pu_threshold_retention_summary.json"
    p_csv = out_dir / "pu_threshold_retention.csv"
    p_json.write_text(json.dumps(_null_json(payload), indent=2), encoding="utf-8")
    pd.DataFrame(rows_csv).to_csv(p_csv, index=False)

    return {
        "output_dir": str(out_dir),
        "summary_json": str(p_json),
        "summary_csv": str(p_csv),
        "plots_dir": str(plots_dir) if make_plots else None,
    }

