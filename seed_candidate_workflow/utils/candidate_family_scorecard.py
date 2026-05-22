"""
Candidate-family scorecard: screen rule templates / provenance families for the seed-candidate graph.

Family-level metrics (no full GNN retrain):
  - same-campaign gain vs current graph
  - cross-campaign contamination
  - oracle ceiling delta (GT-same candidate edges only)
  - graph-only community delta (unweighted Louvain on seed-candidate topology)
  - learnability potential on newly added pairs
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - optional dependency

    def tqdm(iterable=None, **kwargs):  # type: ignore[misc]
        return iterable if iterable is not None else []

from seed_candidate_workflow.utils import graph_structure_helpers as gh
from seed_candidate_workflow.utils.anchor_candidate_eval_helpers import (
    _compute_candidate_oracle_ceiling_for_gt,
    _pairs_from_df,
)
from seed_candidate_workflow.utils.anchor_graph_community_helpers import (
    evaluate_external_metrics,
    run_weighted_email_community_detection,
)
from seed_candidate_workflow.utils.candidate_family_pair_enrichment import enrich_gt_pair_dataframe
from seed_candidate_workflow.utils.candidate_family_rules import (
    _bool_column,
    columns_available_for_rule,
    eval_family_rule_expr,
)
from seed_candidate_workflow.utils.candidate_family_scorecard_catalog import CATALOG_REGISTRY
from seed_candidate_workflow.utils.gt_edge_structure_analysis import (
    _pair_key,
    _resolve_candidate_union_csv,
    _score_candidate_rule_row,
    build_gt_pair_dataframe,
    resolve_gt_paths,
)
from seed_candidate_workflow.utils.raw_gnn_notebook import load_ground_truth_structures


def _bool_series(df: pd.DataFrame, col: str) -> np.ndarray:
    return _bool_column(df, col)


@dataclass
class CandidateFamilySpec:
    family_name: str
    rule_expression: str | None = None
    provenance_column: str | None = None
    edges_csv: Path | None = None
    category: str = ""
    mode: str = "hypothetical_add"
    """hypothetical_add: GT-screened rule edges not in current graph; provenance_slice: union column."""


@dataclass
class RecommendationThresholds:
    min_new_same_pairs: int = 5
    min_oracle_v_gain: float = 0.005
    min_precision_like_new: float = 0.65
    max_cross_new_capture_rate: float = 0.05
    max_graph_only_fraction_of_oracle: float = 0.85
    min_learnability_score: float = 0.15
    weak_gain_max_new_same: int = 3


@dataclass
class CandidateFamilyScorecardRunConfig:
    gt_paths: list[Path]
    graph_pt: Path
    meta_json: Path
    embeddings_json: Path | None = None
    pair_training_csv: Path | None = None
    candidate_union_csv: Path | None = None
    seed_candidate_edges_csv: Path | None = None
    anchor_run_dir: Path | None = None
    out_dir: Path = field(default_factory=lambda: Path("output/analysis/candidate_family_scorecard"))
    families: list[CandidateFamilySpec] = field(default_factory=list)
    max_same_pairs: int = 8000
    max_cross_pairs: int = 8000
    seed: int = 0
    min_support: int = 10
    community_method: str = "louvain"
    community_resolution: float = 1.0
    community_seed: int = 0
    thresholds: RecommendationThresholds = field(default_factory=RecommendationThresholds)
    family_catalog: str | None = None
    misp_json: Path | None = None
    admitting_evidence_dir: Path | None = None


LEARNABILITY_FEATURES: tuple[str, ...] = (
    "semantic_cosine",
    "subject_cosine",
    "body_cosine",
    "path_token_jaccard_combined",
    "sender_localpart_norm_jaccard",
    "body_token_jaccard",
    "subject_token_jaccard",
    "n_shared_core_channels",
    "support_count_excl_domain_and_root_stem",
    "rarity_weighted_support_sum",
    "has_shared_sender",
    "has_shared_stem",
    "has_shared_url",
    "twohop_rarity_max",
    "semantic_cosine_max",
)


def _pairs_to_edges_df(pairs: set[tuple[str, str]], *, weight: float = 1.0) -> pd.DataFrame:
    if not pairs:
        return pd.DataFrame(columns=["email_a", "email_b", "edge_weight"])
    rows = [{"email_a": a, "email_b": b, "edge_weight": float(weight)} for a, b in pairs]
    return pd.DataFrame(rows)


def _load_pair_keys_from_csv(path: Path) -> set[tuple[str, str]]:
    df = pd.read_csv(path, low_memory=False)
    return _pairs_from_df(df)


def _resolve_seed_candidate_edges(
    *,
    explicit: Path | None,
    candidate_union_csv: Path | None,
    pair_training_csv: Path | None,
    project_root: Path,
) -> tuple[set[tuple[str, str]], str]:
    if explicit is not None and explicit.is_file():
        return _load_pair_keys_from_csv(explicit), str(explicit.resolve())

    if candidate_union_csv is not None and candidate_union_csv.is_file():
        return _load_pair_keys_from_csv(candidate_union_csv), str(candidate_union_csv.resolve())

    cu = _resolve_candidate_union_csv(
        explicit=None,
        pair_training_csv=pair_training_csv,
        project_root=project_root,
    )
    if cu is not None and cu.is_file():
        return _load_pair_keys_from_csv(cu), str(cu.resolve())

    if pair_training_csv is not None and pair_training_csv.is_file():
        pt = pd.read_csv(pair_training_csv, low_memory=False)
        if {"email_i", "email_j"}.issubset(pt.columns):
            mask = pd.Series(True, index=pt.index)
            if "is_seed_pair" in pt.columns:
                mask |= pt["is_seed_pair"].fillna(False).astype(bool)
            if "is_candidate_pair" in pt.columns:
                mask |= pt["is_candidate_pair"].fillna(False).astype(bool)
            sub = pt.loc[mask, ["email_i", "email_j"]]
            return _pairs_from_df(sub), str(pair_training_csv.resolve())

    raise FileNotFoundError(
        "Could not resolve seed-candidate / candidate-union edge CSV. "
        "Set candidate_union_csv or seed_candidate_edges_csv in config."
    )


def _gt_label_map(gt_path: Path, meta_json: Path) -> dict[str, str]:
    label_map, _eid_row_gt, _campaign_to_members = load_ground_truth_structures(gt_path)
    meta = gh.load_meta(meta_json)
    eid_row = gh.external_id_to_row(meta)
    return {str(eid): str(camp) for eid, camp in label_map.items() if str(eid) in eid_row}


def _graph_only_v_measure(
    *,
    gt_map: dict[str, str],
    graph_pairs: set[tuple[str, str]],
    method: str,
    resolution: float,
    seed: int,
) -> dict[str, float]:
    node_ids = sorted(gt_map.keys())
    if len(node_ids) < 2:
        return {"homogeneity": float("nan"), "completeness": float("nan"), "v_measure": float("nan")}

    edges_df = _pairs_to_edges_df(graph_pairs)
    email_to_comm, _info = run_weighted_email_community_detection(
        node_ids=node_ids,
        edges_df=edges_df,
        method=method,
        resolution=resolution,
        min_edge_weight=0.0,
        weight_col="edge_weight",
        seed=seed,
        use_edge_weights_in_partitioning=False,
        apply_threshold_filter=False,
    )
    metrics = evaluate_external_metrics(pred_map=email_to_comm, gt_label_map=gt_map)
    return {
        "homogeneity": float(metrics["homogeneity"]),
        "completeness": float(metrics["completeness"]),
        "v_measure": float(metrics["v_measure"]),
    }


def _cohens_d(same: np.ndarray, cross: np.ndarray) -> float | None:
    same = same[np.isfinite(same)]
    cross = cross[np.isfinite(cross)]
    if same.size < 2 or cross.size < 2:
        return None
    m_s, m_c = float(np.mean(same)), float(np.mean(cross))
    v_s, v_c = float(np.var(same, ddof=1)), float(np.var(cross, ddof=1))
    pooled = np.sqrt(((same.size - 1) * v_s + (cross.size - 1) * v_c) / max(same.size + cross.size - 2, 1))
    if pooled <= 0:
        return None
    return float(abs(m_s - m_c) / pooled)


def _learnability_block(df: pd.DataFrame, mask: np.ndarray) -> dict[str, Any]:
    sub = df.loc[mask].copy()
    if sub.empty:
        return {
            "n_new_pairs": 0,
            "n_new_same": 0,
            "n_new_cross": 0,
            "learnability_score": None,
            "feature_separations": {},
        }

    same_m = sub["gt_same_campaign"].fillna(False).astype(bool).to_numpy()
    cross_m = ~same_m
    feature_seps: dict[str, Any] = {}
    d_scores: list[float] = []

    for feat in LEARNABILITY_FEATURES:
        col = feat
        if col == "semantic_cosine" and col not in sub.columns and "semantic_cosine_max" in sub.columns:
            col = "semantic_cosine_max"
        if col not in sub.columns:
            continue
        from seed_candidate_workflow.utils.candidate_family_rules import _as_series

        vals = pd.to_numeric(_as_series(sub, col), errors="coerce").to_numpy(dtype=np.float64)
        d = _cohens_d(vals[same_m], vals[cross_m])
        if d is not None:
            d_scores.append(d)
        same_vals = vals[same_m]
        cross_vals = vals[cross_m]
        feature_seps[feat] = {
            "mean_same": float(np.nanmean(same_vals)) if np.any(np.isfinite(same_vals)) else None,
            "mean_cross": float(np.nanmean(cross_vals)) if np.any(np.isfinite(cross_vals)) else None,
            "cohens_d": d,
        }

    learnability = float(np.clip(np.mean(d_scores), 0.0, 3.0) / 3.0) if d_scores else None
    return {
        "n_new_pairs": int(len(sub)),
        "n_new_same": int(same_m.sum()),
        "n_new_cross": int(cross_m.sum()),
        "learnability_score": learnability,
        "feature_separations": feature_seps,
    }


def _assign_recommendation(
    *,
    n_new_same: int,
    n_new_cross: int,
    precision_like_new: float | None,
    cross_new_capture_rate: float | None,
    oracle_v_gain: float | None,
    graph_only_v_gain: float | None,
    learnability_score: float | None,
    already_in_graph: bool,
    th: RecommendationThresholds,
) -> tuple[str, str]:
    if already_in_graph:
        return (
            "reference_in_graph",
            "Family edges are already present in the current candidate union (provenance slice).",
        )

    if n_new_same <= th.weak_gain_max_new_same:
        return (
            "weak_gain",
            f"Adds at most {n_new_same} new same-campaign GT pairs on the evaluation sample.",
        )

    prec = precision_like_new if precision_like_new is not None else 0.0
    cross_rate = cross_new_capture_rate if cross_new_capture_rate is not None else 1.0
    if prec < th.min_precision_like_new or cross_rate > th.max_cross_new_capture_rate:
        return (
            "too_noisy",
            "Low precision on new pairs or high cross-campaign capture on the GT sample.",
        )

    o_gain = oracle_v_gain if oracle_v_gain is not None else 0.0
    g_gain = graph_only_v_gain if graph_only_v_gain is not None else 0.0
    if o_gain < th.min_oracle_v_gain:
        return (
            "weak_gain",
            f"Oracle V-measure gain ({o_gain:.4f}) below threshold {th.min_oracle_v_gain}.",
        )

    if o_gain > 0 and g_gain >= o_gain * th.max_graph_only_fraction_of_oracle:
        return (
            "too_clean_graph_only",
            "Graph-only community detection captures most of the oracle ceiling gain; "
            "little headroom for learned scoring.",
        )

    learn = learnability_score if learnability_score is not None else 0.0
    if learn < th.min_learnability_score:
        return (
            "too_noisy",
            f"New same vs cross pairs overlap on features (learnability={learn:.3f}).",
        )

    return (
        "promising_for_learning",
        "Raises oracle ceiling with tolerable contamination and leaves graph-only headroom.",
    )


def _family_edges_from_spec(
    spec: CandidateFamilySpec,
    *,
    gt_df: pd.DataFrame,
    candidate_union_df: pd.DataFrame | None,
    graph_pairs: set[tuple[str, str]],
) -> tuple[set[tuple[str, str]], np.ndarray | None, str]:
    """Return (all_family_pairs, cond_on_gt_df or None, mode_note)."""
    if spec.edges_csv is not None:
        p = spec.edges_csv
        if not p.is_file():
            raise FileNotFoundError(f"edges_csv not found for {spec.family_name}: {p}")
        return _load_pair_keys_from_csv(p), None, "edges_csv"

    if spec.provenance_column:
        if candidate_union_df is None:
            raise ValueError(
                f"provenance_column={spec.provenance_column!r} requires candidate_union_csv."
            )
        col = spec.provenance_column
        if col not in candidate_union_df.columns:
            raise ValueError(f"Column {col!r} missing from candidate_union for {spec.family_name}")
        sub = candidate_union_df[candidate_union_df[col].fillna(False).astype(bool)]
        pairs = _pairs_from_df(sub[["email_i", "email_j"]])
        return pairs, None, "provenance_slice"

    if spec.rule_expression:
        cond = eval_family_rule_expr(gt_df, spec.rule_expression)
        keys: set[tuple[str, str]] = set()
        for pos, idx in enumerate(gt_df.index):
            if not cond[pos]:
                continue
            r = gt_df.loc[idx]
            keys.add(_pair_key(str(r["email_i"]), str(r["email_j"])))
        return keys, cond, "hypothetical_add"

    raise ValueError(f"Family {spec.family_name!r} needs rule_expression, provenance_column, or edges_csv.")


def score_one_family(
    spec: CandidateFamilySpec,
    *,
    gt_file: str,
    gt_df: pd.DataFrame,
    gt_map: dict[str, str],
    graph_pairs: set[tuple[str, str]],
    candidate_union_df: pd.DataFrame | None,
    n_same_total: int,
    n_cross_total: int,
    oracle_baseline: dict[str, Any],
    graph_baseline: dict[str, float],
    cfg: CandidateFamilyScorecardRunConfig,
) -> dict[str, Any]:
    th = cfg.thresholds
    family_pairs, cond, mode_note = _family_edges_from_spec(
        spec,
        gt_df=gt_df,
        candidate_union_df=candidate_union_df,
        graph_pairs=graph_pairs,
    )

    if "in_current_candidate_union" in gt_df.columns:
        in_graph = _bool_series(gt_df, "in_current_candidate_union")
    else:
        in_graph = np.zeros(len(gt_df), dtype=bool)
        for pos, idx in enumerate(gt_df.index):
            r = gt_df.loc[idx]
            in_graph[pos] = _pair_key(str(r["email_i"]), str(r["email_j"])) in graph_pairs

    already_in_graph = spec.mode == "provenance_slice" or (
        spec.provenance_column is not None and family_pairs.issubset(graph_pairs)
    )

    same_mask = gt_df["gt_same_campaign"].fillna(False).astype(bool).to_numpy()
    cross_mask = ~same_mask

    rule_hits_on_sample = 0
    same_captured_on_sample: int | None = None
    cross_captured_on_sample: int | None = None
    rule_hits_already_in_union = 0

    if cond is not None:
        new_mask = cond & ~in_graph
        rule_hits_on_sample = int(cond.sum())
        rule_hits_already_in_union = int((cond & in_graph).sum())
        row = _score_candidate_rule_row(
            cond,
            same_mask,
            cross_mask,
            in_graph,
            rule_name=spec.family_name,
            rule_expression=spec.rule_expression or "",
            rule_category=spec.category,
            gt_file=gt_file,
            n_same_total=n_same_total,
            n_cross_total=n_cross_total,
        )
        n_new_same = int(row.get("same_pairs_new_not_in_union") or 0)
        n_new_cross = int(row.get("cross_pairs_new_not_in_union") or 0)
        same_captured_on_sample = int(row.get("same_pairs_captured") or 0)
        cross_captured_on_sample = int(row.get("cross_pairs_captured") or 0)
        precision_like_new = row.get("precision_like_new")
        precision_like_on_sample = row.get("precision_like")
        same_gain_rate = row.get("same_new_capture_rate")
        cross_gain_rate = row.get("cross_new_capture_rate")
        same_capture_rate_on_sample = row.get("same_capture_rate")
        cross_capture_rate_on_sample = row.get("cross_capture_rate")
        learn_block = _learnability_block(gt_df, new_mask)
    else:
        fam_on_sample = np.zeros(len(gt_df), dtype=bool)
        for pos, idx in enumerate(gt_df.index):
            r = gt_df.loc[idx]
            fam_on_sample[pos] = _pair_key(str(r["email_i"]), str(r["email_j"])) in family_pairs
        new_mask = fam_on_sample & ~in_graph
        rule_hits_on_sample = int(fam_on_sample.sum())
        rule_hits_already_in_union = int((fam_on_sample & in_graph).sum())
        n_new_same = int((new_mask & same_mask).sum())
        n_new_cross = int((new_mask & cross_mask).sum())
        same_captured_on_sample = int((fam_on_sample & same_mask).sum())
        cross_captured_on_sample = int((fam_on_sample & cross_mask).sum())
        denom = n_new_same + n_new_cross
        precision_like_new = float(n_new_same / denom) if denom > 0 else None
        denom_sample = same_captured_on_sample + cross_captured_on_sample
        precision_like_on_sample = (
            float(same_captured_on_sample / denom_sample) if denom_sample > 0 else None
        )
        same_gain_rate = float(n_new_same / n_same_total) if n_same_total else None
        cross_gain_rate = float(n_new_cross / n_cross_total) if n_cross_total else None
        same_capture_rate_on_sample = (
            float(same_captured_on_sample / n_same_total) if n_same_total else None
        )
        cross_capture_rate_on_sample = (
            float(cross_captured_on_sample / n_cross_total) if n_cross_total else None
        )
        learn_block = _learnability_block(gt_df, new_mask)

    new_pair_keys: set[tuple[str, str]] = set()
    for pos, idx in enumerate(gt_df.index):
        if not new_mask[pos]:
            continue
        r = gt_df.loc[idx]
        new_pair_keys.add(_pair_key(str(r["email_i"]), str(r["email_j"])))

    aug_pairs = set(graph_pairs)
    if not already_in_graph:
        aug_pairs |= {
            p
            for p in new_pair_keys
            if p[0] in gt_map and p[1] in gt_map and gt_map[p[0]] == gt_map[p[1]]
        }

    oracle_aug = _compute_candidate_oracle_ceiling_for_gt(
        gt_path=gt_file,
        gt_map=gt_map,
        union_pairs=aug_pairs,
    )
    oracle_v_base = float(oracle_baseline.get("v_measure") or 0.0)
    oracle_v_aug = float(oracle_aug.get("v_measure") or 0.0)
    oracle_v_gain = oracle_v_aug - oracle_v_base if np.isfinite(oracle_v_aug) and np.isfinite(oracle_v_base) else None

    graph_aug_pairs = set(graph_pairs)
    if not already_in_graph:
        graph_aug_pairs |= new_pair_keys
    graph_aug = _graph_only_v_measure(
        gt_map=gt_map,
        graph_pairs=graph_aug_pairs,
        method=cfg.community_method,
        resolution=cfg.community_resolution,
        seed=cfg.community_seed,
    )
    g_v_base = float(graph_baseline.get("v_measure") or 0.0)
    g_v_aug = float(graph_aug.get("v_measure") or 0.0)
    graph_only_v_gain = g_v_aug - g_v_base if np.isfinite(g_v_aug) and np.isfinite(g_v_base) else None

    recommendation, rec_reason = _assign_recommendation(
        n_new_same=n_new_same,
        n_new_cross=n_new_cross,
        precision_like_new=precision_like_new,
        cross_new_capture_rate=cross_gain_rate,
        oracle_v_gain=oracle_v_gain,
        graph_only_v_gain=graph_only_v_gain,
        learnability_score=learn_block.get("learnability_score"),
        already_in_graph=already_in_graph,
        th=th,
    )

    return {
        "gt_file": gt_file,
        "family_name": spec.family_name,
        "category": spec.category,
        "mode": spec.mode or mode_note,
        "rule_expression": spec.rule_expression,
        "provenance_column": spec.provenance_column,
        "n_family_pairs_total": int(len(family_pairs)),
        "n_rule_hits_on_gt_sample": rule_hits_on_sample,
        "n_rule_hits_already_in_union": rule_hits_already_in_union,
        "n_same_pairs_captured_on_gt_sample": same_captured_on_sample,
        "n_cross_pairs_captured_on_gt_sample": cross_captured_on_sample,
        "precision_like_on_gt_sample": precision_like_on_sample,
        "same_capture_rate_on_gt_sample": same_capture_rate_on_sample,
        "cross_capture_rate_on_gt_sample": cross_capture_rate_on_sample,
        "n_new_same_pairs": n_new_same,
        "n_new_cross_pairs": n_new_cross,
        "same_gain_rate": same_gain_rate,
        "cross_gain_rate": cross_gain_rate,
        "precision_like_new": precision_like_new,
        "oracle_homogeneity_baseline": oracle_baseline.get("homogeneity"),
        "oracle_completeness_baseline": oracle_baseline.get("completeness"),
        "oracle_v_measure_baseline": oracle_v_base,
        "oracle_homogeneity_aug": oracle_aug.get("homogeneity"),
        "oracle_completeness_aug": oracle_aug.get("completeness"),
        "oracle_v_measure_aug": oracle_v_aug,
        "oracle_v_gain": oracle_v_gain,
        "graph_only_homogeneity_baseline": graph_baseline.get("homogeneity"),
        "graph_only_completeness_baseline": graph_baseline.get("completeness"),
        "graph_only_v_measure_baseline": g_v_base,
        "graph_only_homogeneity_aug": graph_aug.get("homogeneity"),
        "graph_only_completeness_aug": graph_aug.get("completeness"),
        "graph_only_v_measure_aug": graph_aug.get("v_measure"),
        "graph_only_v_gain": graph_only_v_gain,
        "learnability_score": learn_block.get("learnability_score"),
        "n_new_pairs_evaluated": learn_block.get("n_new_pairs"),
        "recommended_action": recommendation,
        "recommendation_reason": rec_reason,
        "already_in_current_graph": already_in_graph,
        "evaluation_note": (
            "GT pair sample + candidate-union topology. n_new_same_pairs counts rule "
            "hits on the sample that are not already in the candidate union; "
            "n_same_pairs_captured_on_gt_sample counts all same-campaign hits on the "
            "sample (including pairs already in the union)."
        ),
    }


def _parse_families(raw: Iterable[dict[str, Any]], project_root: Path) -> list[CandidateFamilySpec]:
    out: list[CandidateFamilySpec] = []
    for item in raw:
        edges = item.get("edges_csv")
        edges_path = None
        if edges:
            p = Path(str(edges))
            edges_path = p if p.is_absolute() else (project_root / p).resolve()
        out.append(
            CandidateFamilySpec(
                family_name=str(item["family_name"]),
                rule_expression=item.get("rule_expression"),
                provenance_column=item.get("provenance_column"),
                edges_csv=edges_path,
                category=str(item.get("category") or ""),
                mode=str(item.get("mode") or "hypothetical_add"),
            )
        )
    return out


def _default_families_from_gt_rules() -> list[dict[str, Any]]:
    """Curated hypothetical templates (subset of gt_edge_structure candidate rules)."""
    templates = [
        ("semantic_ge_0_90_AND_shared_sender", "B_semantic_plus_support"),
        ("semantic_ge_0_90_AND_shared_stem", "B_semantic_plus_support"),
        ("semantic_ge_0_90_AND_shared_url", "B_semantic_plus_support"),
        ("semantic_band_0_85_0_90_AND_shared_sender", "B_semantic_plus_support"),
        ("semantic_band_0_85_0_90_AND_n_shared_core_channels_ge_1", "B_semantic_plus_support"),
        ("from_2hop_AND_semantic_ge_0_90", "C_structural_frontier"),
        ("from_component_AND_semantic_ge_0_90", "C_structural_frontier"),
        ("from_2hop_AND_shared_sender", "C_structural_frontier"),
        ("shared_stem", "D_shared_channel"),
        ("n_shared_core_channels_ge_2", "D_shared_channel"),
    ]
    return [
        {
            "family_name": name,
            "rule_expression": name,
            "category": cat,
            "mode": "hypothetical_add",
        }
        for name, cat in templates
    ]


def run_candidate_family_scorecard(cfg: CandidateFamilyScorecardRunConfig) -> dict[str, Any]:
    project_root = gh.find_project_root()
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    graph_pairs, graph_source = _resolve_seed_candidate_edges(
        explicit=cfg.seed_candidate_edges_csv,
        candidate_union_csv=cfg.candidate_union_csv,
        pair_training_csv=cfg.pair_training_csv,
        project_root=project_root,
    )

    cu_df: pd.DataFrame | None = None
    cu_path = cfg.candidate_union_csv
    if cu_path is None:
        cu_path = _resolve_candidate_union_csv(
            explicit=None,
            pair_training_csv=cfg.pair_training_csv,
            project_root=project_root,
        )
    if cu_path is not None and cu_path.is_file():
        cu_df = pd.read_csv(cu_path, low_memory=False)

    catalog_skipped: list[dict[str, Any]] = []
    families = cfg.families
    if not families and cfg.family_catalog:
        builder = CATALOG_REGISTRY.get(str(cfg.family_catalog))
        if builder is None:
            raise ValueError(f"Unknown family_catalog: {cfg.family_catalog!r}")
        fam_raw, catalog_skipped = builder()
        families = _parse_families(fam_raw, project_root)
    if not families:
        families = _parse_families(_default_families_from_gt_rules(), project_root)

    all_rows: list[dict[str, Any]] = []
    runtime_skipped: list[dict[str, Any]] = []
    detail_rows: list[dict[str, Any]] = []
    per_gt_summaries: list[dict[str, Any]] = []

    print(
        f"[scorecard] {len(families)} families × {len(cfg.gt_paths)} GT file(s); "
        f"graph edges={len(graph_pairs):,}"
    )

    for gt_path in tqdm(cfg.gt_paths, desc="GT files", unit="gt"):
        gt_file = gt_path.name
        gt_map = _gt_label_map(gt_path, cfg.meta_json)

        print(f"[scorecard] {gt_file}: building GT pair sample …")
        gt_df, coverage = build_gt_pair_dataframe(
            gt_path=gt_path,
            meta_json=cfg.meta_json,
            graph_pt=cfg.graph_pt,
            max_same_pairs=cfg.max_same_pairs,
            max_cross_pairs=cfg.max_cross_pairs,
            seed=cfg.seed,
            embeddings_json=cfg.embeddings_json,
            anchor_run_dir=cfg.anchor_run_dir,
            pair_training_csv=cfg.pair_training_csv,
            candidate_union_csv=cfg.candidate_union_csv or cu_path,
            project_root=project_root,
        )
        print(f"[scorecard] {gt_file}: enriching {len(gt_df):,} pairs …")
        gt_df, enrich_meta = enrich_gt_pair_dataframe(
            gt_df,
            anchor_run_dir=cfg.anchor_run_dir,
            pair_training_csv=cfg.pair_training_csv,
            embeddings_json=cfg.embeddings_json,
            project_root=project_root,
            misp_json=cfg.misp_json,
            admitting_evidence_dir=cfg.admitting_evidence_dir,
        )
        coverage["pair_enrichment"] = enrich_meta
        n_same = int(gt_df["gt_same_campaign"].sum())
        n_cross = int(len(gt_df) - n_same)

        print(f"[scorecard] {gt_file}: baselines (oracle + graph-only Louvain) …")
        oracle_baseline = _compute_candidate_oracle_ceiling_for_gt(
            gt_path=str(gt_path),
            gt_map=gt_map,
            union_pairs=graph_pairs,
        )
        graph_baseline = _graph_only_v_measure(
            gt_map=gt_map,
            graph_pairs=graph_pairs,
            method=cfg.community_method,
            resolution=cfg.community_resolution,
            seed=cfg.community_seed,
        )

        gt_rows: list[dict[str, Any]] = []
        family_bar = tqdm(
            families,
            desc=f"Families ({gt_file})",
            unit="family",
            leave=True,
        )
        for spec in family_bar:
            family_bar.set_postfix_str(str(spec.family_name)[:48], refresh=False)
            if spec.rule_expression:
                ok, missing = columns_available_for_rule(gt_df, spec.rule_expression)
                if not ok:
                    runtime_skipped.append(
                        {
                            "gt_file": gt_file,
                            "family_name": spec.family_name,
                            "rule_expression": spec.rule_expression,
                            "category": spec.category,
                            "reason": f"missing_columns: {missing}",
                        }
                    )
                    continue
            try:
                row = score_one_family(
                    spec,
                    gt_file=gt_file,
                    gt_df=gt_df,
                    gt_map=gt_map,
                    graph_pairs=graph_pairs,
                    candidate_union_df=cu_df,
                    n_same_total=n_same,
                    n_cross_total=n_cross,
                    oracle_baseline=oracle_baseline,
                    graph_baseline=graph_baseline,
                    cfg=cfg,
                )
                gt_rows.append(row)
                detail_rows.append({**row, "coverage": coverage})
            except Exception as exc:
                gt_rows.append(
                    {
                        "gt_file": gt_file,
                        "family_name": spec.family_name,
                        "recommended_action": "error",
                        "recommendation_reason": str(exc),
                    }
                )

        # Rank within GT: promising first, then by oracle_v_gain
        action_rank = {
            "promising_for_learning": 0,
            "too_clean_graph_only": 1,
            "too_noisy": 2,
            "weak_gain": 3,
            "reference_in_graph": 4,
            "error": 5,
        }

        def _sort_key(r: dict[str, Any]) -> tuple:
            act = str(r.get("recommended_action") or "error")
            return (
                action_rank.get(act, 9),
                -(float(r.get("oracle_v_gain") or 0.0)),
                -(float(r.get("n_new_same_pairs") or 0.0)),
            )

        gt_rows_sorted = sorted(gt_rows, key=_sort_key)
        for rank, r in enumerate(gt_rows_sorted, start=1):
            r["rank_within_gt"] = rank
        all_rows.extend(gt_rows_sorted)

        by_action: dict[str, list[str]] = {}
        for r in gt_rows_sorted:
            act = str(r.get("recommended_action") or "unknown")
            by_action.setdefault(act, []).append(str(r.get("family_name")))

        per_gt_summaries.append(
            {
                "gt_file": gt_file,
                "n_gt_pairs_sampled": int(len(gt_df)),
                "n_same_pairs_sampled": n_same,
                "n_cross_pairs_sampled": n_cross,
                "graph_edge_source": graph_source,
                "n_graph_edges": int(len(graph_pairs)),
                "oracle_baseline": {
                    "v_measure": oracle_baseline.get("v_measure"),
                    "homogeneity": oracle_baseline.get("homogeneity"),
                    "completeness": oracle_baseline.get("completeness"),
                },
                "graph_only_baseline": graph_baseline,
                "coverage": coverage,
                "families_by_recommendation": by_action,
                "top_promising": [
                    r["family_name"]
                    for r in gt_rows_sorted
                    if r.get("recommended_action") == "promising_for_learning"
                ][:10],
            }
        )

    print(f"[scorecard] Writing outputs to {cfg.out_dir} …")
    scorecard_df = pd.DataFrame(all_rows)
    scorecard_path = cfg.out_dir / "candidate_family_scorecard.csv"
    scorecard_df.to_csv(scorecard_path, index=False)

    details_path = cfg.out_dir / "candidate_family_scorecard_details.csv"
    pd.DataFrame(detail_rows).to_csv(details_path, index=False)

    skipped_all = list(catalog_skipped) + list(runtime_skipped)
    skipped_path = cfg.out_dir / "candidate_family_scorecard_skipped.json"
    skipped_path.write_text(
        json.dumps(skipped_all, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "graph_pt": str(cfg.graph_pt.resolve()),
        "meta_json": str(cfg.meta_json.resolve()),
        "graph_edge_source": graph_source,
        "n_graph_edges": int(len(graph_pairs)),
        "n_gt_files": len(cfg.gt_paths),
        "n_families_configured": len(families),
        "family_catalog": cfg.family_catalog,
        "families_skipped_catalog": catalog_skipped,
        "families_skipped_runtime": runtime_skipped,
        "community_baseline": {
            "method": cfg.community_method,
            "resolution": cfg.community_resolution,
            "unweighted": True,
        },
        "thresholds": {
            "min_new_same_pairs": cfg.thresholds.min_new_same_pairs,
            "min_oracle_v_gain": cfg.thresholds.min_oracle_v_gain,
            "min_precision_like_new": cfg.thresholds.min_precision_like_new,
            "max_cross_new_capture_rate": cfg.thresholds.max_cross_new_capture_rate,
            "max_graph_only_fraction_of_oracle": cfg.thresholds.max_graph_only_fraction_of_oracle,
            "min_learnability_score": cfg.thresholds.min_learnability_score,
            "weak_gain_max_new_same": cfg.thresholds.weak_gain_max_new_same,
        },
        "recommendation_legend": {
            "promising_for_learning": (
                "Raises oracle ceiling, moderate contamination, graph-only does not absorb all gain."
            ),
            "too_clean_graph_only": "Graph-only community captures most oracle gain; scorer has little headroom.",
            "too_noisy": "Too many cross-campaign pairs or poor feature separation on new edges.",
            "weak_gain": "Too few new same-campaign pairs or negligible oracle lift.",
            "reference_in_graph": "Provenance family already represented in candidate union.",
        },
        "per_gt": per_gt_summaries,
        "global_shortlist": {
            "promising_for_learning": sorted(
                {
                    str(r["family_name"])
                    for r in all_rows
                    if r.get("recommended_action") == "promising_for_learning"
                }
            ),
            "too_clean_graph_only": sorted(
                {
                    str(r["family_name"])
                    for r in all_rows
                    if r.get("recommended_action") == "too_clean_graph_only"
                }
            ),
            "too_noisy": sorted(
                {str(r["family_name"]) for r in all_rows if r.get("recommended_action") == "too_noisy"}
            ),
            "weak_gain": sorted(
                {str(r["family_name"]) for r in all_rows if r.get("recommended_action") == "weak_gain"}
            ),
        },
    }

    summary_path = cfg.out_dir / "candidate_family_scorecard_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    return {
        **summary,
        "output_paths": {
            "scorecard_csv": str(scorecard_path.resolve()),
            "summary_json": str(summary_path.resolve()),
            "details_csv": str(details_path.resolve()),
            "skipped_json": str(skipped_path.resolve()),
        },
    }
