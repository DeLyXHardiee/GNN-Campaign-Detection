from __future__ import annotations

import json
import math
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from analysis.utils import graph_structure_helpers as gh
from analysis.utils.anchor_candidate_eval_helpers import _pair
from analysis.utils.anchor_candidate_rare_artifact_helpers import _resolve_latest_seed_dir
from analysis.utils.anchor_graph_community_helpers import run_anchor_multi_gt_community_sweep
from analysis.utils.anchor_graph_helpers import load_anchor_graph_artifacts


def _null_json(x: Any) -> Any:
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return None
    if isinstance(x, dict):
        return {k: _null_json(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_null_json(v) for v in x]
    return x


def _percentile_bounds(
    series: pd.Series,
    mask: pd.Series,
    p_lo: float,
    p_hi: float,
) -> tuple[float, float]:
    vals = pd.to_numeric(series.loc[mask], errors="coerce").dropna()
    if vals.empty:
        return 0.0, 0.0
    lo = float(np.percentile(vals, p_lo))
    hi = float(np.percentile(vals, p_hi))
    return lo, hi


def _norm_minmax(v: float, lo: float, hi: float) -> float:
    if math.isnan(v) or hi <= lo:
        return 0.0
    return float(np.clip((v - lo) / (hi - lo), 0.0, 1.0))


def _load_json_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def run_anchor_scored_clustering_stage(config: dict[str, Any]) -> dict[str, Any]:
    """
    Build scored clustering edges from candidate_union + seed artifacts, then run
    the existing multi-GT community sweep on the thresholded edge list.
    """
    config = dict(config)
    pipeline_config_path = str(config.pop("_pipeline_config_path", "") or "").strip() or None

    run_cfg = config.get("run") or {}
    scoring_cfg = config.get("scoring") or {}
    comm_cfg_in = config.get("community") or {}
    comm_path = str(config.get("community_config_path") or "").strip()

    project_root = gh.find_project_root()
    graph_run_id = str(run_cfg.get("graph_run_id") or "").strip()
    if not graph_run_id:
        raise ValueError("run.graph_run_id is required")

    cand_dir = str(run_cfg.get("candidate_output_dir") or "").strip()
    if not cand_dir:
        raise ValueError("run.candidate_output_dir is required")
    candidate_output_dir = Path(cand_dir).expanduser()
    if not candidate_output_dir.is_absolute():
        candidate_output_dir = (project_root / candidate_output_dir).resolve()
    else:
        candidate_output_dir = candidate_output_dir.resolve()
    if not candidate_output_dir.is_dir():
        raise FileNotFoundError(f"candidate_output_dir not found: {candidate_output_dir}")

    p_union = candidate_output_dir / "candidate_union.csv"
    if not p_union.is_file():
        raise FileNotFoundError(f"Missing candidate_union.csv: {p_union}")

    anchor_output_root = Path(
        run_cfg.get("anchor_output_root") or (project_root / "analysis" / "output" / "anchor_graph")
    ).expanduser().resolve()
    anchor_run_dir = (anchor_output_root / graph_run_id).resolve()
    nodes_df, _edges_anchor, _cand, _sum, _g = load_anchor_graph_artifacts(anchor_run_dir, load_graph_pickle=False)
    node_ids = set(nodes_df["external_id"].astype(str).tolist())

    seed_stage_dir_override = str(run_cfg.get("seed_stage_dir") or "").strip()
    if seed_stage_dir_override:
        seed_dir = Path(seed_stage_dir_override).expanduser()
        if not seed_dir.is_absolute():
            seed_dir = (project_root / seed_dir).resolve()
        else:
            seed_dir = seed_dir.resolve()
        if not seed_dir.is_dir():
            raise FileNotFoundError(f"run.seed_stage_dir not found: {seed_dir}")
    else:
        seed_output_root = Path(
            run_cfg.get("seed_output_root") or (project_root / "analysis" / "output" / "anchor_seeds")
        ).expanduser().resolve()
        seed_prefix = str(run_cfg.get("seed_stage_name_prefix") or "seed_generation_")
        seed_dir = _resolve_latest_seed_dir(
            seed_output_root=seed_output_root,
            graph_run_id=graph_run_id,
            seed_stage_name_prefix=seed_prefix,
        )

    p_seed_all = seed_dir / "seed_edges_all.csv"
    seed_edges_df = pd.read_csv(p_seed_all, low_memory=False) if p_seed_all.is_file() else pd.DataFrame()

    score_rule_version = str(scoring_cfg.get("score_rule_version") or "v0_handcrafted")
    non_seed_min = float(scoring_cfg.get("non_seed_min_edge_weight", 0.35))
    p_lo = float(scoring_cfg.get("rarity_percentile_low", 5.0))
    p_hi = float(scoring_cfg.get("rarity_percentile_high", 95.0))

    w_sem = float(scoring_cfg.get("w_semantic", 0.42))
    w_rare = float(scoring_cfg.get("w_rare_artifact", 0.28))
    w_2hop = float(scoring_cfg.get("w_twohop", 0.22))
    w_comp = float(scoring_cfg.get("w_component", 0.08))
    component_scale = float(scoring_cfg.get("component_scale", 0.12))
    w_multi = float(scoring_cfg.get("w_multi_source", 1.0))
    w_time = float(scoring_cfg.get("w_time_penalty", 1.0))

    seed_floor = float(scoring_cfg.get("seed_weight_floor", 0.88))
    seed_base = float(scoring_cfg.get("seed_weight_base", 0.92))
    seed_rarity_scale = float(scoring_cfg.get("seed_evidence_rarity_scale", 0.08))
    use_seed_rarity = bool(scoring_cfg.get("use_seed_evidence_rarity", True))

    union = pd.read_csv(p_union, low_memory=False)
    notes_missing: list[str] = []
    for col in [
        "from_seed",
        "from_rare_artifact",
        "from_semantic",
        "from_component",
        "from_2hop",
        "source_count",
        "semantic_cosine_max",
        "component_cosine_max",
        "rare_artifact_rarity_max",
        "twohop_rarity_max",
        "time_gap_seconds_min",
    ]:
        if col not in union.columns:
            union[col] = np.nan if col != "source_count" else 0
            notes_missing.append(f"missing_column_filled:{col}")

    union["email_i"] = union["email_i"].astype(str)
    union["email_j"] = union["email_j"].astype(str)
    union["email_a"] = union.apply(lambda r: _pair(str(r["email_i"]), str(r["email_j"]))[0], axis=1)
    union["email_b"] = union.apply(lambda r: _pair(str(r["email_i"]), str(r["email_j"]))[1], axis=1)

    bool_cols = ["from_seed", "from_rare_artifact", "from_semantic", "from_component", "from_2hop"]
    for c in bool_cols:
        union[c] = union[c].fillna(False).astype(bool)

    union["source_count"] = pd.to_numeric(union["source_count"], errors="coerce").fillna(0).astype(int)

    rare_lo, rare_hi = _percentile_bounds(
        union["rare_artifact_rarity_max"],
        union["from_rare_artifact"].astype(bool),
        p_lo,
        p_hi,
    )
    hop_lo, hop_hi = _percentile_bounds(
        union["twohop_rarity_max"],
        union["from_2hop"].astype(bool),
        p_lo,
        p_hi,
    )

    seed_pair_max_rarity: dict[tuple[str, str], float] = {}
    if use_seed_rarity and not seed_edges_df.empty and "evidence_rarity" in seed_edges_df.columns:
        se = seed_edges_df.copy()
        se["email_i"] = se["email_i"].astype(str)
        se["email_j"] = se["email_j"].astype(str)
        for a, b, rv in zip(se["email_i"], se["email_j"], se["evidence_rarity"], strict=False):
            v = float(pd.to_numeric(rv, errors="coerce"))
            if math.isnan(v):
                continue
            pk = _pair(a, b)
            seed_pair_max_rarity[pk] = max(seed_pair_max_rarity.get(pk, float("-inf")), v)
    rarity_vals = [v for v in seed_pair_max_rarity.values() if not math.isnan(v)]
    if len(rarity_vals) >= 2:
        sr_lo = float(np.percentile(rarity_vals, p_lo))
        sr_hi = float(np.percentile(rarity_vals, p_hi))
        if sr_hi <= sr_lo:
            sr_lo, sr_hi = 0.0, 1.0
    else:
        sr_lo, sr_hi = 0.0, 1.0

    agg = (
        union.groupby(["email_a", "email_b"], as_index=False)
        .agg(
            from_seed=("from_seed", "max"),
            from_rare_artifact=("from_rare_artifact", "max"),
            from_semantic=("from_semantic", "max"),
            from_component=("from_component", "max"),
            from_2hop=("from_2hop", "max"),
            source_count=("source_count", "max"),
            semantic_cosine_max=("semantic_cosine_max", "max"),
            component_cosine_max=("component_cosine_max", "max"),
            rare_artifact_rarity_max=("rare_artifact_rarity_max", "max"),
            twohop_rarity_max=("twohop_rarity_max", "max"),
            time_gap_seconds_min=("time_gap_seconds_min", "min"),
        )
    )

    n_pairs_before = int(len(union))
    n_unique_pairs = int(len(agg))

    rows_all: list[dict[str, Any]] = []
    for _, r in agg.iterrows():
        a, b = str(r["email_a"]), str(r["email_b"])
        pk = (a, b)
        fs = bool(r["from_seed"])

        sem = float(pd.to_numeric(r["semantic_cosine_max"], errors="coerce"))
        if math.isnan(sem):
            sem = 0.0
        sem = float(np.clip(sem, 0.0, 1.0))

        rv = float(pd.to_numeric(r["rare_artifact_rarity_max"], errors="coerce"))
        r_norm = _norm_minmax(rv, rare_lo, rare_hi) if not math.isnan(rv) else 0.0

        hv = float(pd.to_numeric(r["twohop_rarity_max"], errors="coerce"))
        h_norm = _norm_minmax(hv, hop_lo, hop_hi) if not math.isnan(hv) else 0.0

        cc = float(pd.to_numeric(r["component_cosine_max"], errors="coerce"))
        if math.isnan(cc):
            cc = 0.0
        comp_part = float(np.clip(cc, 0.0, 1.0)) * component_scale

        sc = int(r["source_count"])
        multi_bonus = w_multi * min(0.05, 0.01 * max(0, sc - 1))

        tg = float(pd.to_numeric(r["time_gap_seconds_min"], errors="coerce"))
        if math.isnan(tg) or tg < 0:
            time_pen = 0.0
        else:
            time_pen = w_time * min(0.08, math.log10(1.0 + tg) / 12.0)

        non_seed_score = w_sem * sem + w_rare * r_norm + w_2hop * h_norm + w_comp * comp_part + multi_bonus - time_pen
        non_seed_score = float(np.clip(non_seed_score, 0.0, 1.0))

        seed_joined = pk in seed_pair_max_rarity
        if fs and use_seed_rarity and seed_joined:
            raw_sr = seed_pair_max_rarity[pk]
            seed_rarity_norm = _norm_minmax(raw_sr, sr_lo, sr_hi)
            seed_w = max(seed_floor, min(1.0, seed_base + seed_rarity_scale * seed_rarity_norm))
        elif fs:
            seed_w = seed_base
        else:
            seed_w = 0.0

        if fs:
            edge_pre = max(seed_w, non_seed_score)
        else:
            edge_pre = non_seed_score

        rows_all.append(
            {
                "email_a": a,
                "email_b": b,
                "edge_weight": edge_pre,
                "from_seed": fs,
                "from_rare_artifact": bool(r["from_rare_artifact"]),
                "from_semantic": bool(r["from_semantic"]),
                "from_component": bool(r["from_component"]),
                "from_2hop": bool(r["from_2hop"]),
                "source_count": sc,
                "score_rule_version": score_rule_version,
            }
        )

    df_all = pd.DataFrame(rows_all)
    n_seed_pairs_read = int(union["from_seed"].sum()) if not union.empty else 0
    n_seed_unique = int(df_all["from_seed"].sum()) if not df_all.empty else 0

    p_all = candidate_output_dir / "scored_clustering_edges_all.csv"
    df_all.to_csv(p_all, index=False)

    df_thr = df_all[(df_all["from_seed"]) | (df_all["edge_weight"] >= non_seed_min)].copy()
    p_thr = candidate_output_dir / "scored_clustering_edges.csv"
    df_thr.to_csv(p_thr, index=False)

    touched = set()
    for a, b in zip(df_thr["email_a"].astype(str), df_thr["email_b"].astype(str), strict=False):
        touched.add(a)
        touched.add(b)
    isolated = int(len(node_ids - touched))

    wts = pd.to_numeric(df_all["edge_weight"], errors="coerce").dropna()
    wts_thr = pd.to_numeric(df_thr["edge_weight"], errors="coerce").dropna()

    def _q(x: pd.Series, qs: list[float]) -> dict[str, float]:
        if x.empty:
            return {f"p{int(q*100)}": float("nan") for q in qs}
        return {f"p{int(q*100)}": float(np.quantile(x, q)) for q in qs}

    combo_ctr = Counter()
    for _, r in df_all.iterrows():
        key = "|".join(
            sorted(
                [lbl for lbl, col in [("seed", "from_seed"), ("rare", "from_rare_artifact"), ("sem", "from_semantic"), ("comp", "from_component"), ("2hop", "from_2hop")] if bool(r.get(col))]
            )
        ) or "none"
        combo_ctr[key] += 1
    top_combos = combo_ctr.most_common(12)

    source_diag: dict[str, Any] = {}
    for lbl, col in [
        ("seed", "from_seed"),
        ("rare_artifact", "from_rare_artifact"),
        ("semantic", "from_semantic"),
        ("component", "from_component"),
        ("2hop", "from_2hop"),
    ]:
        sub = df_all[df_all[col]]
        surv = df_thr[df_thr[col]]
        sw = pd.to_numeric(sub["edge_weight"], errors="coerce").dropna()
        source_diag[lbl] = {
            "n_pairs_involving": int(len(sub)),
            "mean_edge_weight": float(sw.mean()) if len(sw) else None,
            "median_edge_weight": float(sw.median()) if len(sw) else None,
            "n_surviving_threshold": int(len(surv)),
            "survival_rate": float(len(surv) / max(1, len(sub))) if len(sub) else None,
        }

    created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    score_rule_doc = {
        "score_rule_version": score_rule_version,
        "philosophy": (
            "Hand-crafted v0: seed edges get strong weight (base + optional evidence_rarity norm); "
            "non-seed = weighted sum of normalized semantic cosine, rare and 2hop rarity (percentile min-max on union), "
            "small scaled component cosine, multi-source bonus, mild time-gap penalty; clipped to [0,1]. "
            "Final pair weight = max(seed_component, non_seed_score) when from_seed else non_seed_score."
        ),
        "seed_weighting": {
            "use_seed_evidence_rarity": use_seed_rarity,
            "seed_weight_floor": seed_floor,
            "seed_weight_base": seed_base,
            "seed_evidence_rarity_scale": seed_rarity_scale,
            "seed_rarity_percentiles_on_seed_pairs": {"p_lo": p_lo, "p_hi": p_hi},
        },
        "non_seed_terms": {
            "w_semantic": w_sem,
            "w_rare_artifact": w_rare,
            "w_twohop": w_2hop,
            "w_component": w_comp,
            "component_scale": component_scale,
            "w_multi_source": w_multi,
            "multi_bonus_cap": 0.05,
            "w_time_penalty": w_time,
            "time_penalty_cap": 0.08,
            "rarity_normalization": f"min-max using p{p_lo}-p{p_hi} on union rows with respective from_* flag for rare and 2hop columns",
        },
        "component_downweight_note": "component_cosine enters as clip(cos,0,1)*component_scale with small w_component so it cannot dominate",
    }

    summary: dict[str, Any] = {
        "metadata": {
            "created_at_utc": created_at,
            "graph_run_id": graph_run_id,
            "candidate_output_dir": str(candidate_output_dir),
            "seed_stage_dir": str(seed_dir),
            "anchor_run_dir": str(anchor_run_dir),
            "score_rule_version": score_rule_version,
            "pipeline_config_path": pipeline_config_path,
        },
        "inputs": {
            "n_candidate_union_rows": n_pairs_before,
            "n_unique_pairs_after_dedup": n_unique_pairs,
            "n_seed_pairs_in_union_rows": n_seed_pairs_read,
            "n_unique_pairs_with_from_seed": n_seed_unique,
            "seed_edges_all_path": str(p_seed_all),
            "n_seed_pairs_with_evidence_rarity_join": int(
                sum(
                    1
                    for _, r in agg.iterrows()
                    if _pair(str(r["email_a"]), str(r["email_b"])) in seed_pair_max_rarity
                )
            ),
        },
        "score_rule": score_rule_doc,
        "pre_threshold_graph": {
            "n_edges": int(len(df_all)),
            "n_seed_edges": int(df_all["from_seed"].sum()),
            "n_non_seed_edges": int((~df_all["from_seed"]).sum()),
            "edge_weight_quantiles": _null_json(_q(wts, [0.1, 0.25, 0.5, 0.75, 0.9, 1.0])),
            "top_source_flag_combinations": [{"key": k, "count": v} for k, v in top_combos],
        },
        "post_threshold_graph": {
            "non_seed_min_edge_weight": non_seed_min,
            "n_edges_kept": int(len(df_thr)),
            "n_seed_edges_kept": int(df_thr["from_seed"].sum()),
            "n_non_seed_edges_kept": int((~df_thr["from_seed"]).sum()),
            "edge_weight_quantiles": _null_json(_q(wts_thr, [0.1, 0.25, 0.5, 0.75, 0.9, 1.0])),
            "n_unique_emails_touched": int(len(touched)),
            "n_anchor_graph_nodes": int(len(node_ids)),
            "n_isolated_anchor_nodes": isolated,
        },
        "source_weight_diagnostics": source_diag,
        "thresholds_used": {
            "non_seed_min_edge_weight": non_seed_min,
            "community_sweep_weight_thresholds": None,
        },
        "notes": notes_missing
        + [f"scored_edges_written:{p_thr.name}", f"community_input_edge_count:{len(df_thr)}"]
        + (
            ["seed_edges_all_missing_evidence_rarity_column_using_seed_weight_base_only"]
            if (use_seed_rarity and not seed_edges_df.empty and "evidence_rarity" not in seed_edges_df.columns)
            else []
        ),
    }

    if comm_path:
        comm_base_path = Path(comm_path).expanduser()
        if not comm_base_path.is_absolute():
            comm_base_path = (project_root / comm_base_path).resolve()
        community_cfg = _load_json_config(comm_base_path)
    else:
        community_cfg = deepcopy(comm_cfg_in)
        if not community_cfg.get("ground_truth", {}).get("paths"):
            raise ValueError("community_config_path or community.ground_truth.paths is required")

    community_cfg = deepcopy(community_cfg)
    community_cfg.setdefault("run", {})
    community_cfg["run"]["graph_run_id"] = graph_run_id
    community_cfg["run"]["anchor_output_root"] = str(anchor_output_root)
    community_cfg["run"]["custom_edges_csv"] = str(p_thr.resolve())
    community_cfg["run"]["community_output_parent_dir"] = str(candidate_output_dir.resolve())
    community_cfg.setdefault("output", {})
    community_cfg["output"]["stage_name"] = str(
        community_cfg["output"].get("stage_name") or "scored_clustering_community_sweep"
    )
    sweep_merged = community_cfg.setdefault("sweep", {})
    sweep_merged["weight_thresholds"] = list(config.get("community_sweep_weight_thresholds", [0.0]))

    comm_res = run_anchor_multi_gt_community_sweep(community_cfg)
    summary["thresholds_used"]["community_sweep_weight_thresholds"] = community_cfg.get("sweep", {}).get(
        "weight_thresholds"
    )

    p_summary = candidate_output_dir / "scored_clustering_graph_summary.json"
    p_summary.write_text(json.dumps(_null_json(summary), indent=2, ensure_ascii=False), encoding="utf-8")

    return {
        "pipeline_config_path": pipeline_config_path,
        "scored_clustering_edges_csv": str(p_thr.resolve()),
        "scored_clustering_edges_all_csv": str(p_all.resolve()),
        "scored_clustering_graph_summary_json": str(p_summary.resolve()),
        "community_sweep_output_dir": comm_res.get("output_dir"),
        "community_summary_json": comm_res.get("summary_json"),
    }
