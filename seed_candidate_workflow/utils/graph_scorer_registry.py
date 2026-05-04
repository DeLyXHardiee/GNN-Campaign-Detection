from __future__ import annotations

import json
import math
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
import pandas as pd

from seed_candidate_workflow.utils.anchor_candidate_eval_helpers import _pair
from seed_candidate_workflow.utils.pair_graph_contract import (
    GRAPH_KIND_SEMANTIC_SHARD,
    GRAPH_KIND_SEED_CANDIDATE,
    ensure_scored_contract,
    validate_score_mode_target_compatibility,
)
from seed_candidate_workflow.utils import graph_structure_helpers as gh
from seed_candidate_workflow.utils.scorer_diagnostics_core import basic_score_diagnostics

ScorerFn = Callable[..., pd.DataFrame]


@dataclass(frozen=True)
class ScorerSpec:
    name: str
    fn: ScorerFn
    graph_kinds: tuple[str, ...]
    param_keys: tuple[str, ...]
    supports_thresholded_output: bool = False
    required_payload_keys: tuple[str, ...] = ()


@dataclass
class ScorerResult:
    scored_all: pd.DataFrame
    scored_filtered: pd.DataFrame | None
    metadata: dict[str, Any]


class SeedCandidatePayload(TypedDict):
    candidate_union_df: pd.DataFrame
    seed_edges_df: pd.DataFrame


class SemanticShardPayload(TypedDict):
    shard_edges_df: pd.DataFrame


def _ensure_gnn_on_path() -> None:
    root = gh.find_project_root()
    for p in (str(root), str(root / "core"), str(root / "core" / "GNN")):
        if p not in sys.path:
            sys.path.insert(0, p)


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


def score_seed_candidate_handcrafted(
    *,
    candidate_union_df: pd.DataFrame,
    seed_edges_df: pd.DataFrame,
    scoring_cfg: dict[str, Any],
) -> pd.DataFrame:
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
    score_mode = str(scoring_cfg.get("score_mode") or "seed_candidate_handcrafted_v1")

    union = candidate_union_df.copy()
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
    union["email_i"] = union["email_i"].astype(str)
    union["email_j"] = union["email_j"].astype(str)
    union["email_a"] = union.apply(lambda r: _pair(str(r["email_i"]), str(r["email_j"]))[0], axis=1)
    union["email_b"] = union.apply(lambda r: _pair(str(r["email_i"]), str(r["email_j"]))[1], axis=1)
    for c in ["from_seed", "from_rare_artifact", "from_semantic", "from_component", "from_2hop"]:
        union[c] = union[c].fillna(False).astype(bool)
    union["source_count"] = pd.to_numeric(union["source_count"], errors="coerce").fillna(0).astype(int)

    rare_lo, rare_hi = _percentile_bounds(union["rare_artifact_rarity_max"], union["from_rare_artifact"].astype(bool), p_lo, p_hi)
    hop_lo, hop_hi = _percentile_bounds(union["twohop_rarity_max"], union["from_2hop"].astype(bool), p_lo, p_hi)

    seed_pair_max_rarity: dict[tuple[str, str], float] = {}
    if use_seed_rarity and not seed_edges_df.empty and "evidence_rarity" in seed_edges_df.columns:
        se = seed_edges_df.copy()
        for a, b, rv in zip(se["email_i"].astype(str), se["email_j"].astype(str), se["evidence_rarity"], strict=False):
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

    rows: list[dict[str, Any]] = []
    for _, r in agg.iterrows():
        a, b = str(r["email_a"]), str(r["email_b"])
        pk = (a, b)
        fs = bool(r["from_seed"])
        sem = float(np.clip(float(pd.to_numeric(r["semantic_cosine_max"], errors="coerce") or 0.0), 0.0, 1.0))
        rv = float(pd.to_numeric(r["rare_artifact_rarity_max"], errors="coerce"))
        hv = float(pd.to_numeric(r["twohop_rarity_max"], errors="coerce"))
        cc = float(pd.to_numeric(r["component_cosine_max"], errors="coerce") or 0.0)
        r_norm = _norm_minmax(rv, rare_lo, rare_hi) if not math.isnan(rv) else 0.0
        h_norm = _norm_minmax(hv, hop_lo, hop_hi) if not math.isnan(hv) else 0.0
        comp_part = float(np.clip(cc, 0.0, 1.0)) * component_scale
        sc = int(r["source_count"])
        multi_bonus = w_multi * min(0.05, 0.01 * max(0, sc - 1))
        tg = float(pd.to_numeric(r["time_gap_seconds_min"], errors="coerce"))
        time_pen = 0.0 if (math.isnan(tg) or tg < 0) else w_time * min(0.08, math.log10(1.0 + tg) / 12.0)
        non_seed_score = float(np.clip(w_sem * sem + w_rare * r_norm + w_2hop * h_norm + w_comp * comp_part + multi_bonus - time_pen, 0.0, 1.0))
        if fs and use_seed_rarity and pk in seed_pair_max_rarity:
            raw_sr = seed_pair_max_rarity[pk]
            seed_rarity_norm = _norm_minmax(raw_sr, sr_lo, sr_hi)
            seed_w = max(seed_floor, min(1.0, seed_base + seed_rarity_scale * seed_rarity_norm))
        elif fs:
            seed_w = seed_base
        else:
            seed_w = 0.0
        edge_weight = max(seed_w, non_seed_score) if fs else non_seed_score
        rows.append(
            {
                "email_i": a,
                "email_j": b,
                "edge_weight": edge_weight,
                "score_mode": score_mode,
                "from_seed": fs,
                "from_rare_artifact": bool(r["from_rare_artifact"]),
                "from_semantic": bool(r["from_semantic"]),
                "from_component": bool(r["from_component"]),
                "from_2hop": bool(r["from_2hop"]),
                "source_count": sc,
            }
        )
    return ensure_scored_contract(pd.DataFrame(rows))


def _apply_non_seed_weight_transform(
    pu_scores: pd.Series,
    *,
    weight_mode: str,
) -> pd.Series:
    s = pd.to_numeric(pu_scores, errors="coerce")
    if weight_mode == "raw_score":
        return s
    if weight_mode == "raw_score_squared":
        return s.pow(2)
    if weight_mode == "raw_score_cubed":
        return s.pow(3)
    raise ValueError(
        "Unsupported weight_mode: "
        f"{weight_mode!r}. Expected one of: raw_score, raw_score_squared, raw_score_cubed"
    )


def score_seed_candidate_pu(
    *,
    dedup_pairs_df: pd.DataFrame | None = None,
    candidate_union_df: pd.DataFrame | None = None,
    scoring_cfg: dict[str, Any] | None = None,
    seed_edge_weight: float | None = None,
    weight_mode: str | None = None,
    export_non_seed_min_pu_score: float | None = None,
    score_mode: str = "seed_candidate_pu_v1",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    PU scorer contract for seed-candidate graphs.

    Extension contract:
    - Input must be canonical PairGraph identity (`email_i`,`email_j`) plus provenance flags.
    - Scorers may consume either:
      1) precomputed pair scores (`dedup_pairs_df` path), or
      2) raw candidate union + scorer config (`candidate_union_df` + `scoring_cfg`) and
         run inference internally.
    - Output MUST return canonical scored PairGraphs with:
      `email_i,email_j,edge_weight,score_mode`.

    Returns `(all_scored_rows, thresholded_scored_rows)`.
    """
    if dedup_pairs_df is None:
        if candidate_union_df is None:
            raise ValueError("score_seed_candidate_pu requires dedup_pairs_df or candidate_union_df")
        scfg = dict(scoring_cfg or {})
        pu_cfg = dict(scfg.get("pu_run") or {})
        _ensure_gnn_on_path()
        from seed_candidate_workflow.utils.pair_model_inference import (
            load_pair_supervision_for_inference,
            score_pair_rows,
        )
        from src.pair_train import load_pair_training_dataframe, resolve_pair_dataset_csv

        project_root = gh.find_project_root()
        pu_run_dir = Path(str(pu_cfg.get("run_dir") or "")).expanduser()
        if not pu_run_dir.is_absolute():
            pu_run_dir = (project_root / pu_run_dir).resolve()
        else:
            pu_run_dir = pu_run_dir.resolve()
        if not pu_run_dir.is_dir():
            raise FileNotFoundError(f"pu_run.run_dir not found: {pu_run_dir}")

        graph_pt = Path(str(pu_cfg.get("graph_pt") or "")).expanduser()
        if not graph_pt.is_absolute():
            graph_pt = (project_root / graph_pt).resolve()
        else:
            graph_pt = graph_pt.resolve()
        if not graph_pt.is_file():
            raise FileNotFoundError(f"pu_run.graph_pt not found: {graph_pt}")

        checkpoint_name = str(pu_cfg.get("checkpoint", "best_model.pt"))
        device = str(pu_cfg.get("device", "cpu"))
        to_undirected = not bool(pu_cfg.get("no_to_undirected", False))

        raw_pair = str(pu_cfg.get("pair_dataset_csv") or "").strip()
        if raw_pair:
            pair_csv = Path(raw_pair).expanduser()
            if not pair_csv.is_absolute():
                pair_csv = (project_root / pair_csv).resolve()
            else:
                pair_csv = pair_csv.resolve()
        else:
            tc_path = pu_run_dir / "training_config.json"
            tc = json.loads(tc_path.read_text(encoding="utf-8"))
            raw_ds = tc.get("pair_dataset_csv")
            if not raw_ds:
                raise ValueError("pu_run.pair_dataset_csv missing and not in training_config.json")
            pair_csv = resolve_pair_dataset_csv(str(raw_ds), project_root=project_root)

        df_pair, _ = load_pair_training_dataframe(pair_csv)
        df_work = df_pair.reset_index(drop=True).copy()
        df_work["_row"] = np.arange(len(df_work), dtype=np.int64)
        bundle = load_pair_supervision_for_inference(
            run_dir=pu_run_dir,
            graph_pt=graph_pt,
            checkpoint_name=checkpoint_name,
            device=device,
            to_undirected=to_undirected,
        )
        scored_tup = score_pair_rows(
            model=bundle["model"],
            pair_scorer=bundle["pair_scorer"],
            data_cpu=bundle["data_cpu"],
            df_work=df_work,
            device=bundle["device"],
            fanout=bundle["fanout"],
            pair_batch_size=bundle["pair_batch_size"],
            max_unique_emails=bundle["max_unique_emails"],
            with_logits=False,
        )
        pu_score = scored_tup[0] if isinstance(scored_tup, tuple) else scored_tup
        score_map: dict[tuple[str, str], float] = {}
        for a, b, s in zip(df_work["email_i"].astype(str), df_work["email_j"].astype(str), pu_score, strict=False):
            sf = float(s) if np.isfinite(s) else float("nan")
            if math.isnan(sf):
                continue
            pk = _pair(a, b)
            if pk in score_map:
                score_map[pk] = max(score_map[pk], sf)
            else:
                score_map[pk] = sf

        union = candidate_union_df.copy()
        union["email_i"] = union["email_i"].astype(str)
        union["email_j"] = union["email_j"].astype(str)
        for c in ("from_seed", "from_semantic", "from_rare_artifact", "from_component", "from_2hop"):
            if c not in union.columns:
                union[c] = False
            union[c] = union[c].fillna(False).astype(bool)
        if "source_count" not in union.columns:
            union["source_count"] = 0
        union["source_count"] = pd.to_numeric(union["source_count"], errors="coerce").fillna(0).astype(int)
        union["email_a"] = union.apply(lambda r: _pair(str(r["email_i"]), str(r["email_j"]))[0], axis=1)
        union["email_b"] = union.apply(lambda r: _pair(str(r["email_i"]), str(r["email_j"]))[1], axis=1)
        union["pu_score"] = union.apply(
            lambda r: score_map.get((str(r["email_a"]), str(r["email_b"])), float("nan")),
            axis=1,
        )
        cdf = (
            union.sort_values(["email_a", "email_b", "from_seed", "pu_score"], ascending=[True, True, False, False])
            .drop_duplicates(subset=["email_a", "email_b"], keep="first")
            .copy()
        )
        dedup_pairs_df = cdf[
            [
                "email_a",
                "email_b",
                "pu_score",
                "from_seed",
                "from_semantic",
                "from_rare_artifact",
                "from_component",
                "from_2hop",
                "source_count",
            ]
        ]
        seed_edge_weight = float(scfg.get("seed_edge_weight", scfg.get("seed_weight", 1.0)))
        weight_mode = str(scfg.get("weight_mode", "raw_score"))
        export_non_seed_min_pu_score = float(scfg.get("export_non_seed_min_pu_score", 0.0))

    if seed_edge_weight is None or weight_mode is None or export_non_seed_min_pu_score is None:
        raise ValueError("PU scorer missing required weight parameters")
    cdf = dedup_pairs_df.copy()
    cdf["edge_weight"] = np.where(
        cdf["from_seed"],
        float(seed_edge_weight),
        _apply_non_seed_weight_transform(cdf["pu_score"], weight_mode=weight_mode),
    )
    cdf["score_mode"] = str(score_mode)
    mask_keep = cdf["from_seed"] | (
        pd.to_numeric(cdf["pu_score"], errors="coerce").ge(float(export_non_seed_min_pu_score))
        & pd.to_numeric(cdf["pu_score"], errors="coerce").notna()
    )
    cdf_kept = cdf.loc[mask_keep].copy()
    out_cols = [
        "email_a",
        "email_b",
        "edge_weight",
        "from_seed",
        "from_semantic",
        "from_rare_artifact",
        "from_component",
        "from_2hop",
        "source_count",
        "score_mode",
    ]
    all_out = cdf[out_cols].rename(columns={"email_a": "email_i", "email_b": "email_j"})
    thr_out = cdf_kept[out_cols].rename(columns={"email_a": "email_i", "email_b": "email_j"})
    return ensure_scored_contract(all_out), ensure_scored_contract(thr_out)


def score_semantic_shard_handcrafted(
    *,
    shard_edges_df: pd.DataFrame,
    scoring_cfg: dict[str, Any] | None = None,
    score_mode: str = "semantic_shard_handcrafted_v1",
) -> pd.DataFrame:
    """Compute shard-edge weights from semantic/infra/temporal components."""
    cfg = dict(scoring_cfg or {})
    w_sem = float(cfg.get("w_semantic", 0.45))
    w_infra = float(cfg.get("w_infra", 0.45))
    w_temporal = float(cfg.get("w_temporal", 0.10))
    min_weight = float(cfg.get("min_edge_weight", 0.0))

    df = shard_edges_df.copy()
    if "shard_a" not in df.columns or "shard_b" not in df.columns:
        if "email_i" in df.columns and "email_j" in df.columns:
            df["shard_a"] = df["email_i"].astype(str)
            df["shard_b"] = df["email_j"].astype(str)
        else:
            raise ValueError("semantic shard scorer requires shard_a/shard_b or email_i/email_j columns")
    df["shard_a"] = df["shard_a"].astype(str)
    df["shard_b"] = df["shard_b"].astype(str)
    sem = pd.to_numeric(df.get("centroid_cosine"), errors="coerce").fillna(0.0).clip(lower=0.0)
    infra = pd.to_numeric(df.get("infra_score"), errors="coerce").fillna(0.0).clip(lower=0.0)
    temporal = pd.to_numeric(df.get("temporal_score"), errors="coerce").fillna(0.0).clip(lower=0.0)
    edge_weight = (w_sem * sem) + (w_infra * infra) + (w_temporal * temporal)
    df["edge_weight"] = pd.to_numeric(edge_weight, errors="coerce").fillna(0.0)
    if min_weight > 0.0:
        df = df[df["edge_weight"] >= float(min_weight)].copy()
    df["score_mode"] = str(score_mode)
    df["email_i"] = df["shard_a"].astype(str)
    df["email_j"] = df["shard_b"].astype(str)
    _ = ensure_scored_contract(df[["email_i", "email_j", "edge_weight", "score_mode"]])
    return df


def score_semantic_shard_affine(
    *,
    shard_edges_df: pd.DataFrame,
    scoring_cfg: dict[str, Any] | None = None,
    score_mode: str = "semantic_shard_affine_v1",
) -> pd.DataFrame:
    """Affine transform of existing shard edge_weight with optional clipping."""
    cfg = dict(scoring_cfg or {})
    scale = float(cfg.get("scale", 1.0))
    bias = float(cfg.get("bias", 0.0))
    clip_min = float(cfg.get("clip_min", 0.0))
    clip_max_raw = cfg.get("clip_max")
    clip_max = None if clip_max_raw in (None, "") else float(clip_max_raw)

    df = shard_edges_df.copy()
    if "shard_a" not in df.columns or "shard_b" not in df.columns:
        if "email_i" in df.columns and "email_j" in df.columns:
            df["shard_a"] = df["email_i"].astype(str)
            df["shard_b"] = df["email_j"].astype(str)
        else:
            raise ValueError("semantic shard scorer requires shard_a/shard_b or email_i/email_j columns")
    base = pd.to_numeric(df.get("edge_weight"), errors="coerce").fillna(0.0)
    ew = (base * scale) + bias
    ew = ew.clip(lower=clip_min)
    if clip_max is not None:
        ew = ew.clip(upper=float(clip_max))
    df["edge_weight"] = pd.to_numeric(ew, errors="coerce").fillna(0.0)
    df["score_mode"] = str(score_mode)
    df["email_i"] = df["shard_a"].astype(str)
    df["email_j"] = df["shard_b"].astype(str)
    _ = ensure_scored_contract(df[["email_i", "email_j", "edge_weight", "score_mode"]])
    return df


SCORER_REGISTRY: dict[str, ScorerFn] = {
    "seed_candidate_handcrafted_v1": score_seed_candidate_handcrafted,
    "seed_candidate_pu_v1": score_seed_candidate_pu,
    "semantic_shard_handcrafted_v1": score_semantic_shard_handcrafted,
    "semantic_shard_affine_v1": score_semantic_shard_affine,
}

SCORER_SPECS: dict[str, ScorerSpec] = {
    "seed_candidate_handcrafted_v1": ScorerSpec(
        name="seed_candidate_handcrafted_v1",
        fn=score_seed_candidate_handcrafted,
        graph_kinds=(GRAPH_KIND_SEED_CANDIDATE,),
        param_keys=("handcrafted",),
        required_payload_keys=("candidate_union_df", "seed_edges_df"),
    ),
    "seed_candidate_pu_v1": ScorerSpec(
        name="seed_candidate_pu_v1",
        fn=score_seed_candidate_pu,
        graph_kinds=(GRAPH_KIND_SEED_CANDIDATE,),
        param_keys=("pu",),
        supports_thresholded_output=True,
        required_payload_keys=("candidate_union_df",),
    ),
    "semantic_shard_handcrafted_v1": ScorerSpec(
        name="semantic_shard_handcrafted_v1",
        fn=score_semantic_shard_handcrafted,
        graph_kinds=(GRAPH_KIND_SEMANTIC_SHARD,),
        param_keys=("semantic_shard_handcrafted", "semantic_shard"),
        required_payload_keys=("shard_edges_df",),
    ),
    "semantic_shard_affine_v1": ScorerSpec(
        name="semantic_shard_affine_v1",
        fn=score_semantic_shard_affine,
        graph_kinds=(GRAPH_KIND_SEMANTIC_SHARD,),
        param_keys=("semantic_shard_affine", "semantic_shard"),
        required_payload_keys=("shard_edges_df",),
    ),
}


def validate_scorer_target(score_mode: str, graph_kind: str) -> None:
    # Centralized compatibility check so orchestration can stay graph-target agnostic.
    validate_score_mode_target_compatibility(score_mode=score_mode, graph_kind=graph_kind)


SCORER_TARGET_GRAPH_KINDS: dict[str, tuple[str, ...]] = {
    "seed_candidate_handcrafted_v1": (GRAPH_KIND_SEED_CANDIDATE,),
    "seed_candidate_pu_v1": (GRAPH_KIND_SEED_CANDIDATE,),
    "semantic_shard_handcrafted_v1": (GRAPH_KIND_SEMANTIC_SHARD,),
    "semantic_shard_affine_v1": (GRAPH_KIND_SEMANTIC_SHARD,),
}


def resolve_score_params(score_mode: str, params_root: dict[str, Any]) -> dict[str, Any]:
    spec = SCORER_SPECS.get(str(score_mode))
    if spec is None:
        return {}
    for k in spec.param_keys:
        v = params_root.get(k)
        if isinstance(v, dict):
            return dict(v)
    return {}


def apply_scorer(
    *,
    score_mode: str,
    graph_kind: str,
    score_params: dict[str, Any],
    payload: dict[str, Any],
    diagnostics_cfg: dict[str, Any] | None = None,
) -> ScorerResult:
    validate_scorer_target(score_mode=score_mode, graph_kind=graph_kind)
    if score_mode not in SCORER_REGISTRY:
        raise ValueError(f"Unknown score_mode {score_mode!r}. Available: {sorted(SCORER_REGISTRY)}")
    spec = SCORER_SPECS[score_mode]
    miss = [k for k in spec.required_payload_keys if k not in payload]
    if miss:
        raise ValueError(f"Missing scorer payload keys for {score_mode}: {miss}")
    fn = SCORER_REGISTRY[score_mode]
    if score_mode == "seed_candidate_handcrafted_v1":
        scored_all = fn(
            candidate_union_df=payload["candidate_union_df"],
            seed_edges_df=payload["seed_edges_df"],
            scoring_cfg=score_params,
        )
        metadata: dict[str, Any] = {"score_mode": score_mode}
        if diagnostics_cfg and diagnostics_cfg.get("enabled"):
            dr = basic_score_diagnostics(
                score_mode=score_mode,
                graph_kind=graph_kind,
                scored_df=scored_all,
                score_col="edge_weight",
            )
            metadata["diagnostics"] = {
                "enabled": True,
                "summary": {
                    "input_stats": dr.input_stats,
                    "output_stats": dr.output_stats,
                    "provenance_stats": dr.provenance_stats,
                    "scorer_specific": dr.scorer_specific,
                },
            }
        return ScorerResult(scored_all=scored_all, scored_filtered=None, metadata=metadata)
    if score_mode == "seed_candidate_pu_v1":
        scored_all, scored_filtered = fn(
            candidate_union_df=payload["candidate_union_df"],
            scoring_cfg=score_params,
            score_mode=score_mode,
        )
        metadata = {"score_mode": score_mode}
        if diagnostics_cfg and diagnostics_cfg.get("enabled"):
            dr = basic_score_diagnostics(
                score_mode=score_mode,
                graph_kind=graph_kind,
                scored_df=scored_all,
                score_col="edge_weight",
            )
            metadata["diagnostics"] = {
                "enabled": True,
                "summary": {
                    "input_stats": dr.input_stats,
                    "output_stats": dr.output_stats,
                    "provenance_stats": dr.provenance_stats,
                    "scorer_specific": dr.scorer_specific,
                },
            }
        return ScorerResult(
            scored_all=scored_all,
            scored_filtered=scored_filtered,
            metadata=metadata,
        )
    if score_mode in {"semantic_shard_handcrafted_v1", "semantic_shard_affine_v1"}:
        scored_all = fn(
            shard_edges_df=payload["shard_edges_df"],
            scoring_cfg=score_params,
            score_mode=score_mode,
        )
        metadata = {"score_mode": score_mode}
        if diagnostics_cfg and diagnostics_cfg.get("enabled"):
            dr = basic_score_diagnostics(
                score_mode=score_mode,
                graph_kind=graph_kind,
                scored_df=scored_all,
                score_col="edge_weight",
            )
            metadata["diagnostics"] = {
                "enabled": True,
                "summary": {
                    "input_stats": dr.input_stats,
                    "output_stats": dr.output_stats,
                    "provenance_stats": dr.provenance_stats,
                    "scorer_specific": dr.scorer_specific,
                },
            }
        return ScorerResult(scored_all=scored_all, scored_filtered=None, metadata=metadata)
    raise ValueError(f"Unsupported score_mode envelope: {score_mode!r}")
