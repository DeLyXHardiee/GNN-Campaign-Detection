"""
Dedup-GT vs expanded-GT community evaluation gap analysis.

Read-only: loads existing anchor community outputs, ground truth JSON files,
dedup collapse mapping, optional candidate / pair-training artifacts, and
summarizes where expanded-email evaluation degrades relative to representative-level GT.
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from seed_candidate_workflow.utils import community_eval_contract as cec
from seed_candidate_workflow.utils import graph_structure_helpers as gh
from seed_candidate_workflow.utils import semantic_supernode_gt_metrics as ssgt
from seed_candidate_workflow.utils.anchor_graph_community_helpers import (
    run_weighted_email_community_detection,
)
from seed_candidate_workflow.utils.raw_gnn_notebook import load_ground_truth_structures


def _slugify(s: str) -> str:
    t = re.sub(r"[^A-Za-z0-9._-]+", "_", str(s).strip())
    t = re.sub(r"_+", "_", t).strip("_.-")
    return t or "unknown"


def load_member_to_rep_from_external_id_map(path: Path) -> dict[str, str]:
    """external_id -> representative_external_id (every row, including singletons)."""
    path = path.expanduser().resolve()
    out: dict[str, str] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return out
        for row in reader:
            eid = str(row.get("external_id") or "").strip()
            rep = str(row.get("representative_external_id") or "").strip()
            if eid and rep:
                out[eid] = rep
    return out


def _pred_map_from_assignment(node_ids: list[str], email_to_comm: dict[str, int]) -> dict[str, int]:
    return {str(eid): int(email_to_comm[str(eid)]) for eid in node_ids}


def _undirected_pair_key(a: str, b: str) -> tuple[str, str]:
    x, y = str(a), str(b)
    return (x, y) if x <= y else (y, x)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_best_row_from_anchor_best_json(path: Path) -> dict[str, Any]:
    raw = _load_json(path)
    br = raw.get("best_row") or {}
    if not isinstance(br, dict) or not br:
        raise ValueError(f"best_row missing or empty in {path}")
    return dict(br)


def _resolve_project_path(raw: str | Path, project_root: Path) -> Path:
    p = Path(str(raw).strip()).expanduser()
    return p if p.is_absolute() else (project_root / p).resolve()


def _pick_best_row_for_gt_from_summary(
    summary_path: Path,
    gt_path: Path,
    *,
    project_root: Path,
    gt_label: str = "target",
) -> tuple[dict[str, Any], str]:
    """
    Recover the sweep ``best_row`` for a GT file from ``anchor_community_multi_gt_summary.json``.

    1. Prefer an entry whose ``gt_path`` resolves to the same file as ``gt_path``.
    2. Else if exactly one GT row exists in the summary, use its ``best_row`` (fallback; recorded in source).
    3. Else if exactly one row matches ``gt_path.name`` (basename), use that row.
    """
    raw = _load_json(summary_path)
    target = gt_path.resolve()
    target_name = target.name

    def _rows_per() -> list[dict[str, Any]]:
        return [r for r in (raw.get("per_ground_truth_outputs") or []) if isinstance(r, dict)]

    def _rows_br() -> list[dict[str, Any]]:
        return [r for r in (raw.get("best_rows_by_gt") or []) if isinstance(r, dict)]

    for row in _rows_per():
        gp_raw = row.get("gt_path")
        if not gp_raw:
            continue
        gp = _resolve_project_path(str(gp_raw), project_root).resolve()
        br = row.get("best_row")
        if gp == target and isinstance(br, dict) and br:
            return dict(br), f"multi_gt_summary:{summary_path.name}:per_ground_truth_outputs:exact_gt_path"

    for row in _rows_br():
        gp_raw = row.get("gt_path")
        if not gp_raw:
            continue
        gp = _resolve_project_path(str(gp_raw), project_root).resolve()
        if gp != target:
            continue
        br = {k: v for k, v in row.items() if k not in ("gt_path", "gt_slug")}
        if br:
            return br, f"multi_gt_summary:{summary_path.name}:best_rows_by_gt:exact_gt_path"

    name_hits: list[tuple[dict[str, Any], str]] = []
    for row in _rows_per():
        gp_raw = row.get("gt_path")
        if not gp_raw:
            continue
        gp = _resolve_project_path(str(gp_raw), project_root).resolve()
        br = row.get("best_row")
        if gp.name == target_name and isinstance(br, dict) and br:
            name_hits.append((dict(br), str(gp)))
    if len(name_hits) == 1:
        return name_hits[0][0], f"multi_gt_summary:{summary_path.name}:unique_basename:{name_hits[0][1]}"

    per = _rows_per()
    if len(per) == 1:
        br = per[0].get("best_row")
        if isinstance(br, dict) and br:
            return dict(br), (
                f"multi_gt_summary:{summary_path.name}:sole_gt_row(per_ground_truth_outputs); "
                f"{gt_label}_gt_path_not_in_summary_using_same_partition_params_as:{per[0].get('gt_path')}"
            )

    br_rows = _rows_br()
    if len(br_rows) == 1:
        row = br_rows[0]
        br = {k: v for k, v in row.items() if k not in ("gt_path", "gt_slug")}
        if br:
            return br, (
                f"multi_gt_summary:{summary_path.name}:sole_gt_row(best_rows_by_gt); "
                f"{gt_label}_gt_path_not_in_summary_using_same_partition_params_as:{row.get('gt_path')}"
            )

    known: list[str] = []
    for row in br_rows:
        if row.get("gt_path"):
            known.append(str(_resolve_project_path(str(row["gt_path"]), project_root).resolve()))
    for row in per:
        if row.get("gt_path"):
            known.append(str(_resolve_project_path(str(row["gt_path"]), project_root).resolve()))
    raise ValueError(
        f"No usable community sweep row for {gt_label} GT {target} (basename {target_name!r}). "
        f"Summary gt_paths: {sorted(set(known))!r}. "
        "Pass a multi-GT summary that includes this GT, a valid anchor_community_best__*.json, "
        "or add best_row to the JSON config."
    )


def _pick_best_row_for_dedup_gt_from_summary(
    summary_path: Path,
    dedup_gt_path: Path,
    *,
    project_root: Path,
) -> tuple[dict[str, Any], str]:
    return _pick_best_row_for_gt_from_summary(
        summary_path, dedup_gt_path, project_root=project_root, gt_label="dedup"
    )


def _pick_best_row_for_expanded_gt_from_summary(
    summary_path: Path,
    expanded_gt_path: Path,
    *,
    project_root: Path,
) -> tuple[dict[str, Any], str]:
    return _pick_best_row_for_gt_from_summary(
        summary_path, expanded_gt_path, project_root=project_root, gt_label="expanded"
    )


def _partition_params_from_best_row(best_row: dict[str, Any]) -> dict[str, Any]:
    return {
        "method": str(best_row.get("method") or "leiden").strip().lower(),
        "resolution": float(best_row.get("resolution", 1.0)),
        "min_edge_weight": float(best_row.get("min_edge_weight", 0.0)),
        "weight_col": str(best_row.get("weight_col") or "edge_weight"),
        "use_edge_weights_in_partitioning": bool(best_row.get("use_edge_weights_in_partitioning", True)),
    }


def _partition_cache_key(params: dict[str, Any], *, use_w: bool, apply_thr: bool, seed: int) -> tuple[Any, ...]:
    return (
        params["method"],
        params["resolution"],
        params["min_edge_weight"],
        use_w,
        apply_thr,
        seed,
    )


def _metrics_subset(metrics: dict[str, Any]) -> dict[str, Any]:
    keys = ("n_eval", "homogeneity", "completeness", "v_measure", "coverage_gt", "coverage_predictions")
    return {k: float(metrics[k]) if k in metrics and metrics[k] is not None else metrics.get(k) for k in keys}


def _sweep_metrics_from_best_row(best_row: dict[str, Any]) -> dict[str, Any] | None:
    """Metrics stored on a community-sweep best_row (if present)."""
    if not any(k in best_row for k in ("v_measure", "homogeneity", "completeness")):
        return None
    return _metrics_subset(
        {
            "n_eval": best_row.get("n_eval"),
            "homogeneity": best_row.get("homogeneity"),
            "completeness": best_row.get("completeness"),
            "v_measure": best_row.get("v_measure"),
            "coverage_gt": best_row.get("coverage_gt"),
            "coverage_predictions": best_row.get("coverage_predictions"),
        }
    )


def _expanded_minus_dedup_deltas(
    metrics_on_expanded_gt: dict[str, Any],
    metrics_on_dedup_gt: dict[str, Any],
    *,
    prefix: str = "",
) -> dict[str, float]:
    p = f"{prefix}" if prefix else ""
    return {
        f"{p}expanded_minus_dedup_v_measure": float(
            metrics_on_expanded_gt["v_measure"] - metrics_on_dedup_gt["v_measure"]
        ),
        f"{p}expanded_minus_dedup_completeness": float(
            metrics_on_expanded_gt["completeness"] - metrics_on_dedup_gt["completeness"]
        ),
        f"{p}expanded_minus_dedup_homogeneity": float(
            metrics_on_expanded_gt["homogeneity"] - metrics_on_dedup_gt["homogeneity"]
        ),
    }


def _partitions_equal(p1: dict[str, Any], p2: dict[str, Any]) -> bool:
    return (
        str(p1["method"]) == str(p2["method"])
        and float(p1["resolution"]) == float(p2["resolution"])
        and float(p1["min_edge_weight"]) == float(p2["min_edge_weight"])
    )


def _partition_diff(p1: dict[str, Any], p2: dict[str, Any]) -> dict[str, Any]:
    return {
        "method": {"expanded_best": p1["method"], "dedup_best": p2["method"], "equal": p1["method"] == p2["method"]},
        "resolution": {
            "expanded_best": p1["resolution"],
            "dedup_best": p2["resolution"],
            "delta_expanded_minus_dedup": float(p1["resolution"] - p2["resolution"]),
        },
        "min_edge_weight": {
            "expanded_best": p1["min_edge_weight"],
            "dedup_best": p2["min_edge_weight"],
            "delta_expanded_minus_dedup": float(p1["min_edge_weight"] - p2["min_edge_weight"]),
        },
    }


def _interpret_same_partition(
    selection_label: str,
    deltas: dict[str, float],
) -> str:
    vm = float(deltas.get("expanded_minus_dedup_v_measure", 0.0))
    if vm > 0.005:
        direction = "expanded GT scores higher than dedup GT"
    elif vm < -0.005:
        direction = "dedup GT scores higher than expanded GT"
    else:
        direction = "expanded and dedup GT scores are nearly tied"
    return (
        f"Using the {selection_label} partition (one fixed prediction, two GT views), "
        f"{direction} on V-measure (Δ={vm:+.4f})."
    )


def _interpret_best_to_best(
    *,
    same_partition: bool,
    deltas: dict[str, float],
) -> str:
    vm = float(deltas.get("best_to_best_v_measure_delta", 0.0))
    if same_partition:
        base = "Each GT view selected the same sweep-best partition; "
    else:
        base = "Each GT view selected its own sweep-best partition (settings may differ); "
    if vm > 0.005:
        tail = "expanded GT achieves a higher best V-measure than dedup GT."
    elif vm < -0.005:
        tail = "dedup GT achieves a higher best V-measure than expanded GT."
    else:
        tail = "best achievable V-measure is nearly the same on both GT views."
    return base + tail


def _load_community_context_from_multi_summary(
    summary_path: Path,
    *,
    project_root: Path,
) -> dict[str, Any]:
    """Paths and flags from anchor_community_multi_gt_summary.json."""
    raw = _load_json(summary_path)
    anchor_run_dir = Path(str(raw.get("anchor_run_dir") or "")).expanduser()
    if not anchor_run_dir.is_absolute():
        anchor_run_dir = (project_root / anchor_run_dir).resolve()
    else:
        anchor_run_dir = anchor_run_dir.resolve()
    ce = raw.get("custom_edges_csv")
    scored_edges: Path | None = None
    if ce:
        p = Path(str(ce)).expanduser()
        scored_edges = p if p.is_absolute() else (project_root / p).resolve()
    exp = raw.get("gt_metric_email_expansion") or {}
    mapping_path_raw = str(exp.get("member_expansion_mapping_path") or "").strip()
    mapping_path: Path | None = None
    if mapping_path_raw:
        mp = Path(mapping_path_raw).expanduser()
        mapping_path = mp if mp.is_absolute() else (project_root / mp).resolve()
    return {
        "anchor_run_dir": anchor_run_dir,
        "scored_edges_csv": scored_edges,
        "gt_metric_email_expansion": exp,
        "graph_id": raw.get("graph_id"),
        "weight_col": str(raw.get("weight_col") or "edge_weight"),
        "use_edge_weights_in_partitioning": bool(raw.get("use_edge_weights_in_partitioning", True)),
        "apply_threshold_filter": bool(raw.get("apply_threshold_filter", True)),
        "seed": int(raw.get("seed", 0)),
        "resolved_mapping_path": mapping_path,
    }


def _dominant_pred_share(labels: list[int]) -> tuple[int, float]:
    if not labels:
        return 0, 0.0
    ctr = Counter(labels)
    top_c, top_n = ctr.most_common(1)[0]
    return int(len(ctr)), float(top_n / len(labels))


def _per_campaign_pred_stats(
    member_ids: list[str],
    pred: dict[str, int],
) -> dict[str, Any]:
    labs = [int(pred[str(m)]) for m in member_ids if str(m) in pred]
    n_cov = len(labs)
    n_par = len(member_ids)
    n_distinct, dom_share = _dominant_pred_share(labs)
    return {
        "n_members": int(n_par),
        "n_members_with_prediction": int(n_cov),
        "coverage_fraction": float(n_cov / max(1, n_par)),
        "n_pred_communities": int(n_distinct),
        "largest_pred_community_share": float(dom_share),
    }


@dataclass
class DedupExpandedGapConfig:
    project_root: Path
    gt_dedup_json: Path
    gt_expanded_json: Path
    dedup_collapse_out_dir: Path | None
    external_id_map_csv: Path | None
    anchor_run_dir: Path
    scored_edges_csv: Path
    community_multi_gt_summary_json: Path | None = None
    expanded_best_row: dict[str, Any] | None = None
    expanded_best_row_source: str = ""
    dedup_best_row: dict[str, Any] | None = None
    dedup_best_row_source: str = ""
    weight_col: str = "edge_weight"
    use_edge_weights_in_partitioning: bool = True
    apply_threshold_filter: bool = True
    seed: int = 0
    candidate_union_csv: Path | None = None
    pair_training_csv: Path | None = None
    score_threshold: float = 0.1
    top_lossy_campaigns: int = 40
    pair_feature_cols: tuple[str, ...] = field(
        default_factory=lambda: (
            "semantic_cosine_max",
            "source_count",
            "from_2hop",
            "from_semantic",
            "from_component",
            "body_token_jaccard",
            "body_char4gram_jaccard",
            "body_only_token_jaccard",
            "body_only_char4gram_jaccard",
            "path_token_jaccard_combined",
            "sender_localpart_norm_jaccard",
            "same_seed_component_flag",
            "cross_seed_component_flag",
        )
    )


def _build_undirected_pair_set_from_edges(df: pd.DataFrame) -> set[tuple[str, str]]:
    out: set[tuple[str, str]] = set()
    for _, r in df.iterrows():
        a = str(r.get("email_i") or r.get("email_a") or "")
        b = str(r.get("email_j") or r.get("email_b") or "")
        if a and b and a != b:
            out.add(_undirected_pair_key(a, b))
    return out


def _edge_weight_lookup(df: pd.DataFrame, wcol: str) -> dict[tuple[str, str], float]:
    out: dict[tuple[str, str], float] = {}
    for _, r in df.iterrows():
        a = str(r.get("email_i") or r.get("email_a") or "")
        b = str(r.get("email_j") or r.get("email_b") or "")
        if not a or not b:
            continue
        k = _undirected_pair_key(a, b)
        w = float(pd.to_numeric(r.get(wcol), errors="coerce") or 0.0)
        out[k] = max(out.get(k, 0.0), w)
    return out


def _pair_training_lookup(df: pd.DataFrame, feature_cols: Iterable[str]) -> dict[tuple[str, str], dict[str, Any]]:
    if df.empty or "email_i" not in df.columns or "email_j" not in df.columns:
        return {}
    out: dict[tuple[str, str], dict[str, Any]] = {}
    cols = [c for c in feature_cols if c in df.columns]
    for _, r in df.iterrows():
        k = _undirected_pair_key(str(r["email_i"]), str(r["email_j"]))
        out[k] = {c: r.get(c) for c in cols}
    return out


def _campaign_gap_contribution(
    *,
    n_expanded: int,
    dom_dedup: float,
    dom_exp: float,
) -> float:
    """Heuristic mass lost when dominant share drops after expansion."""
    return float(n_expanded) * max(0.0, float(dom_dedup) - float(dom_exp))


def _classify_failure_modes(
    link_rows: list[dict[str, Any]],
    campaign_rows: list[dict[str, Any]],
    *,
    score_threshold: float,
) -> dict[str, Any]:
    n_links = len(link_rows)
    n_missing = sum(1 for r in link_rows if r.get("link_status") == "missing_from_candidate")
    n_low = sum(1 for r in link_rows if r.get("link_status") == "present_low_score")
    n_high = sum(1 for r in link_rows if r.get("link_status") == "present_high_score")
    n_same = sum(1 for r in link_rows if r.get("same_pred_community"))
    n_split = n_links - n_same

    masses = [float(r.get("gap_mass", 0.0) or 0.0) for r in campaign_rows]
    total_mass = float(sum(masses))
    masses_sorted = sorted(masses, reverse=True)
    top5 = float(sum(masses_sorted[:5])) if masses_sorted else 0.0
    frac_top5 = top5 / total_mass if total_mass > 0 else 0.0

    n_small = sum(1 for m in masses if 0 < m < 5.0)
    n_camp = len([m for m in masses if m > 0])

    tags: list[str] = []
    if n_links and n_missing / n_links > 0.35:
        tags.append("missing_candidate_representative_links")
    if n_links and n_low / max(1, n_missing + n_low) > 0.4 and n_low >= n_missing:
        tags.append("low_scoring_representative_links")
    if n_split > n_same and n_high > n_split * 0.25:
        tags.append("community_fragmentation_despite_edges")
    if frac_top5 >= 0.5 and n_camp >= 5:
        tags.append("few_large_campaigns_dominate_gap")
    if n_camp >= 10 and n_small / max(1, n_camp) > 0.5 and frac_top5 < 0.45:
        tags.append("many_small_campaigns_dominate_gap")

    plain = []
    if "missing_candidate_representative_links" in tags:
        plain.append(
            "A substantial share of representative pairs in lossy campaigns never appear "
            "in the candidate graph; recovering those bridges or widening candidate families "
            "is likely high leverage."
        )
    if "low_scoring_representative_links" in tags:
        plain.append(
            "Many representative pairs exist in the candidate graph but with scores below "
            "the community-detection threshold; scorer or feature work on this frontier is relevant."
        )
    if "community_fragmentation_despite_edges" in tags:
        plain.append(
            "Some representative pairs have strong scores yet still land in different "
            "predicted communities—partitioning or thresholding may be washing out intra-campaign links."
        )
    if "few_large_campaigns_dominate_gap" in tags:
        plain.append(
            "A handful of large campaigns account for most of the coherence loss mass; "
            "deep-diving those IDs first will explain most of the dedup-vs-expanded gap."
        )
    if "many_small_campaigns_dominate_gap" in tags:
        plain.append(
            "The gap is spread across many smaller campaigns; broad candidate/scorer improvements "
            "may be more appropriate than a single-campaign fix."
        )
    if not plain:
        plain.append(
            "Inspect dedup_vs_expanded_gap_campaigns.csv for dominant_share and fragmentation deltas; "
            "link-level evidence is in dedup_vs_expanded_gap_representative_links.csv."
        )

    return {
        "failure_mechanism_tags": tags,
        "link_stats": {
            "n_representative_pairs_analyzed": n_links,
            "n_missing_from_candidate": n_missing,
            "n_present_low_score": n_low,
            "n_present_high_score": n_high,
            "n_same_pred_community": n_same,
            "n_split_across_pred_communities": n_split,
            "score_threshold_used": float(score_threshold),
        },
        "campaign_mass_stats": {
            "n_campaigns_positive_gap_mass": n_camp,
            "total_gap_mass": total_mass,
            "fraction_gap_mass_top5_campaigns": frac_top5,
            "n_campaigns_small_gap_mass_lt_5": n_small,
        },
        "plain_english_takeaways": plain,
    }


def _run_partition_predictions(
    *,
    node_ids: list[str],
    edges_df: pd.DataFrame,
    best_row: dict[str, Any],
    cfg: DedupExpandedGapConfig,
    gid_to_members: dict[str, list[str]],
    partition_cache: dict[tuple[Any, ...], tuple[dict[str, int], dict[str, int], dict[str, Any]]],
) -> tuple[dict[str, int], dict[str, int], dict[str, Any]]:
    params = _partition_params_from_best_row(best_row)
    wcol = str(cfg.weight_col)
    use_w = bool(cfg.use_edge_weights_in_partitioning)
    cache_key = _partition_cache_key(params, use_w=use_w, apply_thr=bool(cfg.apply_threshold_filter), seed=int(cfg.seed))
    if cache_key in partition_cache:
        return partition_cache[cache_key]

    email_to_comm, part_info = run_weighted_email_community_detection(
        node_ids=node_ids,
        edges_df=edges_df,
        method=params["method"],
        resolution=params["resolution"],
        min_edge_weight=params["min_edge_weight"],
        weight_col=wcol,
        seed=int(cfg.seed),
        use_edge_weights_in_partitioning=use_w,
        apply_threshold_filter=bool(cfg.apply_threshold_filter),
    )
    pred_graph = _pred_map_from_assignment(node_ids, email_to_comm)
    pred_expanded = ssgt.expand_pred_map_for_gt_eval(pred_graph, gid_to_members)
    out = (pred_graph, pred_expanded, part_info)
    partition_cache[cache_key] = out
    return out


def _evaluate_fixed_partition_on_both_gts(
    *,
    pred_graph: dict[str, int],
    pred_expanded: dict[str, int],
    gt_dedup_map: dict[str, str],
    gt_exp_map: dict[str, str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    metrics_dedup = _metrics_subset(
        cec.evaluate_external_metrics(
            gt_label_map=gt_dedup_map,
            pred_label_map=pred_graph,
            n_predictions_total=len(pred_graph),
        )
    )
    metrics_exp = _metrics_subset(
        cec.evaluate_external_metrics(
            gt_label_map=gt_exp_map,
            pred_label_map=pred_expanded,
            n_predictions_total=len(pred_expanded),
        )
    )
    return metrics_dedup, metrics_exp


def _build_same_partition_block(
    *,
    label: str,
    best_row: dict[str, Any],
    selection_source: str,
    metrics_on_dedup_gt: dict[str, Any],
    metrics_on_expanded_gt: dict[str, Any],
    partition_info: dict[str, Any],
) -> dict[str, Any]:
    deltas = _expanded_minus_dedup_deltas(metrics_on_expanded_gt, metrics_on_dedup_gt)
    return {
        "selection_label": label,
        "selection_source": selection_source,
        "partition_parameters": _partition_params_from_best_row(best_row),
        "partition_info_replayed": partition_info,
        "metrics_on_dedup_gt": metrics_on_dedup_gt,
        "metrics_on_expanded_gt": metrics_on_expanded_gt,
        "deltas_expanded_minus_dedup": deltas,
        "interpretation": _interpret_same_partition(label, deltas),
    }


def run_dedup_vs_expanded_gap_analysis(cfg: DedupExpandedGapConfig) -> dict[str, Any]:
    nodes_df = pd.read_csv(cfg.anchor_run_dir / "anchor_graph_nodes.csv", low_memory=False)
    node_ids = [str(x) for x in nodes_df["external_id"].astype(str).tolist()]
    node_set = set(node_ids)

    edges_df = pd.read_csv(cfg.scored_edges_csv, low_memory=False)
    if "email_i" in edges_df.columns and "email_j" in edges_df.columns:
        edges_df = edges_df.copy()
        edges_df["email_a"] = edges_df["email_i"].astype(str)
        edges_df["email_b"] = edges_df["email_j"].astype(str)
    elif "email_a" not in edges_df.columns or "email_b" not in edges_df.columns:
        raise ValueError("scored edges must have email_i/email_j or email_a/email_b")
    wcol = str(cfg.weight_col)
    if wcol not in edges_df.columns:
        raise ValueError(f"weight column {wcol!r} not in scored edges CSV {cfg.scored_edges_csv}")

    if cfg.expanded_best_row is None or cfg.dedup_best_row is None:
        raise ValueError("expanded_best_row and dedup_best_row are required (resolve via community summary or best JSON).")

    gid_to_members: dict[str, list[str]] | None = None
    if cfg.external_id_map_csv and cfg.external_id_map_csv.is_file():
        gid_to_members = ssgt.load_dedup_collapse_member_table_from_external_id_map(cfg.external_id_map_csv)
    elif cfg.dedup_collapse_out_dir and cfg.dedup_collapse_out_dir.is_dir():
        gid_to_members = ssgt.load_dedup_collapse_member_table_from_out_dir(cfg.dedup_collapse_out_dir)
    if not gid_to_members:
        raise ValueError(
            "Member expansion table is required (set dedup_collapse_out_dir or external_id_map_csv)."
        )

    gt_dedup_map, _e1, camp_dedup = load_ground_truth_structures(cfg.gt_dedup_json)
    gt_exp_map, _e2, camp_exp = load_ground_truth_structures(cfg.gt_expanded_json)

    if cfg.external_id_map_csv and cfg.external_id_map_csv.is_file():
        map_csv = cfg.external_id_map_csv
    elif cfg.dedup_collapse_out_dir and (cfg.dedup_collapse_out_dir / "external_id_map.csv").is_file():
        map_csv = cfg.dedup_collapse_out_dir / "external_id_map.csv"
    else:
        raise ValueError("Need external_id_map.csv (via external_id_map_csv or dedup_collapse_out_dir).")
    member_to_rep = load_member_to_rep_from_external_id_map(map_csv)
    if not member_to_rep:
        raise ValueError("external_id_map.csv did not yield member_to_rep mapping.")

    partition_cache: dict[tuple[Any, ...], tuple[dict[str, int], dict[str, int], dict[str, Any]]] = {}

    pred_exp_sel, pred_expanded_sel, part_exp = _run_partition_predictions(
        node_ids=node_ids,
        edges_df=edges_df,
        best_row=cfg.expanded_best_row,
        cfg=cfg,
        gid_to_members=gid_to_members,
        partition_cache=partition_cache,
    )
    pred_ded_sel, pred_expanded_ded_sel, part_ded = _run_partition_predictions(
        node_ids=node_ids,
        edges_df=edges_df,
        best_row=cfg.dedup_best_row,
        cfg=cfg,
        gid_to_members=gid_to_members,
        partition_cache=partition_cache,
    )

    metrics_ded_on_exp_part, metrics_exp_on_exp_part = _evaluate_fixed_partition_on_both_gts(
        pred_graph=pred_exp_sel,
        pred_expanded=pred_expanded_sel,
        gt_dedup_map=gt_dedup_map,
        gt_exp_map=gt_exp_map,
    )
    metrics_ded_on_ded_part, metrics_exp_on_ded_part = _evaluate_fixed_partition_on_both_gts(
        pred_graph=pred_ded_sel,
        pred_expanded=pred_expanded_ded_sel,
        gt_dedup_map=gt_dedup_map,
        gt_exp_map=gt_exp_map,
    )

    same_partition_comparison = {
        "expanded_selected": _build_same_partition_block(
            label="expanded_selected",
            best_row=cfg.expanded_best_row,
            selection_source=cfg.expanded_best_row_source,
            metrics_on_dedup_gt=metrics_ded_on_exp_part,
            metrics_on_expanded_gt=metrics_exp_on_exp_part,
            partition_info=part_exp,
        ),
        "dedup_selected": _build_same_partition_block(
            label="dedup_selected",
            best_row=cfg.dedup_best_row,
            selection_source=cfg.dedup_best_row_source,
            metrics_on_dedup_gt=metrics_ded_on_ded_part,
            metrics_on_expanded_gt=metrics_exp_on_ded_part,
            partition_info=part_ded,
        ),
    }

    exp_params = _partition_params_from_best_row(cfg.expanded_best_row)
    ded_params = _partition_params_from_best_row(cfg.dedup_best_row)
    same_part_for_best = _partitions_equal(exp_params, ded_params)

    sweep_exp = _sweep_metrics_from_best_row(cfg.expanded_best_row)
    sweep_ded = _sweep_metrics_from_best_row(cfg.dedup_best_row)
    if sweep_exp is None:
        sweep_exp = metrics_exp_on_exp_part
    if sweep_ded is None:
        sweep_ded = metrics_ded_on_ded_part

    b2b_deltas = {
        "best_to_best_v_measure_delta": float(sweep_exp["v_measure"] - sweep_ded["v_measure"]),
        "best_to_best_completeness_delta": float(sweep_exp["completeness"] - sweep_ded["completeness"]),
        "best_to_best_homogeneity_delta": float(sweep_exp["homogeneity"] - sweep_ded["homogeneity"]),
    }
    best_to_best_comparison = {
        "expanded_gt_best": {
            "gt_json": str(cfg.gt_expanded_json.resolve()),
            "selection_source": cfg.expanded_best_row_source,
            "partition_parameters": exp_params,
            "metrics_at_best_for_this_gt": sweep_exp,
            "note": "Metrics from community sweep best_row for expanded GT (each GT optimizes its own partition).",
        },
        "dedup_gt_best": {
            "gt_json": str(cfg.gt_dedup_json.resolve()),
            "selection_source": cfg.dedup_best_row_source,
            "partition_parameters": ded_params,
            "metrics_at_best_for_this_gt": sweep_ded,
            "note": "Metrics from community sweep best_row for dedup GT.",
        },
        "same_partition": same_part_for_best,
        "partition_parameter_diff": _partition_diff(exp_params, ded_params) if not same_part_for_best else None,
        "deltas_expanded_best_minus_dedup_best": b2b_deltas,
        "interpretation": _interpret_best_to_best(same_partition=same_part_for_best, deltas=b2b_deltas),
    }

    pred_graph = pred_exp_sel
    pred_expanded = pred_expanded_sel
    structural_partition = {
        "comparison_mode": "same_partition",
        "submode": "expanded_selected",
        "partition_parameters": exp_params,
        "selection_source": cfg.expanded_best_row_source,
    }

    candidate_pairs = _build_undirected_pair_set_from_edges(edges_df)
    union_pairs = candidate_pairs
    if cfg.candidate_union_csv and cfg.candidate_union_csv.is_file():
        cu = pd.read_csv(cfg.candidate_union_csv, low_memory=False)
        if "email_i" not in cu.columns and "email_a" in cu.columns:
            cu = cu.copy()
            cu["email_i"] = cu["email_a"].astype(str)
            cu["email_j"] = cu["email_b"].astype(str)
        union_pairs = _build_undirected_pair_set_from_edges(cu)

    weight_by_pair = _edge_weight_lookup(edges_df, wcol)

    pair_feats: dict[tuple[str, str], dict[str, Any]] = {}
    if cfg.pair_training_csv and cfg.pair_training_csv.is_file():
        ptdf = pd.read_csv(cfg.pair_training_csv, low_memory=False)
        pair_feats = _pair_training_lookup(ptdf, cfg.pair_feature_cols)

    thr = float(cfg.score_threshold)

    campaign_ids = sorted(set(camp_exp.keys()) | set(camp_dedup.keys()), key=lambda x: (str(type(x)), x))
    camp_rows: list[dict[str, Any]] = []
    for cid in campaign_ids:
        members_exp = list(camp_exp.get(cid, []))
        members_dedup = list(camp_dedup.get(cid, []))
        n_exp = len(members_exp)
        n_rep_gt_dedup = len(members_dedup)

        reps_in_graph = sorted({member_to_rep.get(m, m) for m in members_exp if member_to_rep.get(m, m) in node_set})
        dedup_view_ids = [m for m in members_dedup if m in node_set]
        if not dedup_view_ids and reps_in_graph:
            dedup_view_ids = reps_in_graph

        st_dedup = _per_campaign_pred_stats(dedup_view_ids, pred_graph)
        st_exp = _per_campaign_pred_stats(members_exp, pred_expanded)

        dom_d = float(st_dedup["largest_pred_community_share"])
        dom_e = float(st_exp["largest_pred_community_share"])
        gap_dom = dom_e - dom_d
        gap_frag = int(st_exp["n_pred_communities"]) - int(st_dedup["n_pred_communities"])
        gap_mass = _campaign_gap_contribution(
            n_expanded=n_exp,
            dom_dedup=dom_d,
            dom_exp=dom_e,
        )

        pred_comms_dedup = sorted({pred_graph[m] for m in dedup_view_ids if m in pred_graph})
        pred_comms_exp = sorted({pred_expanded[m] for m in members_exp if m in pred_expanded})

        camp_rows.append(
            {
                "campaign_id": cid,
                "n_expanded_emails": n_exp,
                "n_dedup_gt_rows": n_rep_gt_dedup,
                "n_representatives_in_anchor_graph": len(reps_in_graph),
                "n_pred_communities_dedup_view": int(st_dedup["n_pred_communities"]),
                "n_pred_communities_expanded_view": int(st_exp["n_pred_communities"]),
                "delta_n_pred_communities_exp_minus_dedup": int(gap_frag),
                "largest_pred_community_share_dedup_view": dom_d,
                "largest_pred_community_share_expanded_view": dom_e,
                "delta_largest_share_expanded_minus_dedup": float(gap_dom),
                "gap_mass_coherence_loss": float(gap_mass),
                "dedup_view_members_used": len(dedup_view_ids),
                "expanded_members_with_pred": int(st_exp["n_members_with_prediction"]),
                "pred_community_ids_dedup_view": "|".join(str(x) for x in pred_comms_dedup),
                "pred_community_ids_expanded_view": "|".join(str(x) for x in pred_comms_exp),
                "representative_external_ids_in_graph": "|".join(reps_in_graph),
            }
        )

    camp_df = pd.DataFrame(camp_rows)
    camp_df = camp_df.sort_values("delta_largest_share_expanded_minus_dedup", ascending=True).reset_index(drop=True)

    lossy = camp_df.head(int(cfg.top_lossy_campaigns))
    lossy_ids = {row["campaign_id"] for _, row in lossy.iterrows()}

    link_rows: list[dict[str, Any]] = []
    for cid in sorted(lossy_ids, key=lambda x: (str(type(x)), x)):
        members_exp = list(camp_exp.get(cid, []))
        reps = sorted({member_to_rep.get(m, m) for m in members_exp if member_to_rep.get(m, m) in node_set})
        if len(reps) < 2:
            continue
        for i, r1 in enumerate(reps):
            for r2 in reps[i + 1 :]:
                pk = _undirected_pair_key(r1, r2)
                in_union = pk in union_pairs
                w = float(weight_by_pair.get(pk, 0.0))
                if not in_union:
                    status = "missing_from_candidate"
                elif w < thr:
                    status = "present_low_score"
                else:
                    status = "present_high_score"
                same = bool(pred_graph.get(r1) == pred_graph.get(r2))
                feats = pair_feats.get(pk, {})
                row: dict[str, Any] = {
                    "campaign_id": cid,
                    "representative_i": r1,
                    "representative_j": r2,
                    "pred_community_i": int(pred_graph[r1]) if r1 in pred_graph else -1,
                    "pred_community_j": int(pred_graph[r2]) if r2 in pred_graph else -1,
                    "same_pred_community": same,
                    "in_candidate_union": bool(pk in union_pairs),
                    "in_scored_edges_csv": bool(pk in candidate_pairs),
                    "edge_weight_max": w,
                    "link_status": status,
                }
                for k, v in feats.items():
                    row[f"pair_{k}"] = v
                link_rows.append(row)

    link_df = pd.DataFrame(link_rows)
    class_block = _classify_failure_modes(link_rows, camp_rows, score_threshold=thr)

    evaluation_mode_diagnostics = {
        "community_multi_gt_summary_json": str(cfg.community_multi_gt_summary_json.resolve())
        if cfg.community_multi_gt_summary_json
        else None,
        "expanded_selected": {
            "selection_source": cfg.expanded_best_row_source,
            "partition_parameters": exp_params,
        },
        "dedup_selected": {
            "selection_source": cfg.dedup_best_row_source,
            "partition_parameters": ded_params,
        },
        "best_to_best_same_partition": same_part_for_best,
        "scored_edges_csv": str(cfg.scored_edges_csv.resolve()),
        "gt_dedup_json": str(cfg.gt_dedup_json.resolve()),
        "gt_expanded_json": str(cfg.gt_expanded_json.resolve()),
        "legacy_v1_global_metrics_note": (
            "v1 global_metrics compared expanded-GT metrics on member-expanded predictions "
            "against dedup-GT metrics on representative-level predictions under one partition; "
            "that mixed GT views and prediction levels. Use same_partition_comparison and "
            "best_to_best_comparison instead."
        ),
    }

    summary: dict[str, Any] = {
        "schema": "dedup_vs_expanded_gap_summary_v2",
        "inputs": {
            "gt_dedup_json": str(cfg.gt_dedup_json.resolve()),
            "gt_expanded_json": str(cfg.gt_expanded_json.resolve()),
            "anchor_run_dir": str(cfg.anchor_run_dir.resolve()),
            "scored_edges_csv": str(cfg.scored_edges_csv.resolve()),
            "community_multi_gt_summary_json": str(cfg.community_multi_gt_summary_json.resolve())
            if cfg.community_multi_gt_summary_json
            else None,
            "dedup_collapse_out_dir": str(cfg.dedup_collapse_out_dir.resolve())
            if cfg.dedup_collapse_out_dir
            else None,
            "external_id_map_csv": str(cfg.external_id_map_csv.resolve())
            if cfg.external_id_map_csv
            else None,
            "member_external_id_map_csv_resolved": str(map_csv.resolve()),
            "candidate_union_csv": str(cfg.candidate_union_csv.resolve()) if cfg.candidate_union_csv else None,
            "pair_training_csv": str(cfg.pair_training_csv.resolve()) if cfg.pair_training_csv else None,
        },
        "same_partition_comparison": same_partition_comparison,
        "best_to_best_comparison": best_to_best_comparison,
        "same_partition_interpretation": same_partition_comparison["expanded_selected"]["interpretation"],
        "best_to_best_interpretation": best_to_best_comparison["interpretation"],
        "evaluation_mode_diagnostics": evaluation_mode_diagnostics,
        "structural_analysis": {
            **structural_partition,
            "failure_classification": class_block,
            "best_to_best_note": (
                "Representative-link and lossy-campaign tables use the expanded-selected partition only. "
                "If best-to-best partitions differ, structural conclusions may not hold for the dedup-optimal partition."
                if not same_part_for_best
                else "Best-to-best uses the same partition as expanded-selected; structural analysis aligns with both."
            ),
        },
        "lossy_campaign_analysis": {
            "comparison_mode": "same_partition",
            "submode": "expanded_selected",
            "partition_parameters": exp_params,
            "description": (
                "Campaign ranking and representative-pair tables answer: given one fixed prediction "
                "(expanded-selected partition), what coherence is lost when evaluating duplicate-expanded GT?"
            ),
        },
        "questions": {
            "A_where_does_gap_come_from": {
                "few_large_campaigns": class_block["campaign_mass_stats"]["fraction_gap_mass_top5_campaigns"] >= 0.5
                and class_block["campaign_mass_stats"]["n_campaigns_positive_gap_mass"] >= 5,
                "many_small_campaigns": "many_small_campaigns_dominate_gap"
                in class_block["failure_mechanism_tags"],
                "many_dedup_representatives": float((camp_df["n_representatives_in_anchor_graph"] > 3).mean())
                > 0.25
                if len(camp_df)
                else False,
                "missing_bridges_between_representatives": "missing_candidate_representative_links"
                in class_block["failure_mechanism_tags"],
            },
            "B_failing_representative_links": {
                "mostly_absent_from_candidate": class_block["link_stats"]["n_missing_from_candidate"]
                > class_block["link_stats"]["n_present_low_score"],
                "mostly_present_but_low_scoring": class_block["link_stats"]["n_present_low_score"]
                >= class_block["link_stats"]["n_missing_from_candidate"],
                "community_split_despite_strong_edges": "community_fragmentation_despite_edges"
                in class_block["failure_mechanism_tags"],
            },
            "C_implications_next_steps": class_block["plain_english_takeaways"],
        },
        "failure_classification": class_block,
    }

    return {
        "summary": summary,
        "campaign_table": camp_df,
        "representative_link_table": link_df,
    }


def write_dedup_vs_expanded_gap_outputs(
    result: dict[str, Any],
    out_dir: Path,
    *,
    write_html: bool = False,
) -> dict[str, str]:
    out_dir = out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "summary_json": str(out_dir / "dedup_vs_expanded_gap_summary.json"),
        "campaigns_csv": str(out_dir / "dedup_vs_expanded_gap_campaigns.csv"),
        "representative_links_csv": str(out_dir / "dedup_vs_expanded_gap_representative_links.csv"),
    }
    (out_dir / "dedup_vs_expanded_gap_summary.json").write_text(
        json.dumps(result["summary"], indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    result["campaign_table"].to_csv(paths["campaigns_csv"], index=False)
    result["representative_link_table"].to_csv(paths["representative_links_csv"], index=False)
    if write_html:
        html_p = out_dir / "dedup_vs_expanded_gap_lossy_campaigns.html"
        _write_simple_lossy_html(result["campaign_table"], result["representative_link_table"], html_p)
        paths["lossy_html"] = str(html_p)
    return paths


def _write_simple_lossy_html(camp_df: pd.DataFrame, link_df: pd.DataFrame, path: Path) -> None:
    top = camp_df.head(50).to_html(index=False, float_format=lambda x: f"{x:.4f}")
    link_sample = link_df.head(500).to_html(index=False, float_format=lambda x: f"{x:.4f}")
    body = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Dedup vs expanded gap</title></head>
<body>
<h2>Most lossy campaigns (by delta largest pred share)</h2>
{top}
<h2>Representative pairs (sample)</h2>
{link_sample}
</body></html>"""
    path.write_text(body, encoding="utf-8")


def build_config_from_cli_and_json(
    *,
    project_root: Path,
    config_json: dict[str, Any] | None,
    overrides: dict[str, Any],
) -> DedupExpandedGapConfig:
    cfg = dict(config_json or {})
    cfg.update({k: v for k, v in overrides.items() if v is not None and str(v).strip() != ""})

    def _p(key: str, default: str | None = None) -> Path | None:
        raw = cfg.get(key, default)
        if raw is None or str(raw).strip() == "":
            return None
        p = Path(str(raw).strip()).expanduser()
        return p if p.is_absolute() else (project_root / p).resolve()

    gt_dedup = _p("gt_dedup_json")
    gt_exp = _p("gt_expanded_json")
    if gt_dedup is None or not gt_dedup.is_file():
        raise ValueError("gt_dedup_json must point to an existing file.")
    if gt_exp is None or not gt_exp.is_file():
        raise ValueError("gt_expanded_json must point to an existing file.")

    anchor = _p("anchor_run_dir")
    scored = _p("scored_edges_csv")
    mgs_path = _p("community_multi_gt_summary_json")
    if mgs_path and mgs_path.is_file():
        ctx0 = _load_community_context_from_multi_summary(mgs_path, project_root=project_root)
        if anchor is None or not anchor.is_dir():
            ar = ctx0.get("anchor_run_dir")
            if isinstance(ar, Path) and ar.is_dir():
                anchor = ar
        if scored is None or not scored.is_file():
            se = ctx0.get("scored_edges_csv")
            if se is not None and Path(se).is_file():
                scored = Path(se)

    if anchor is None or not anchor.is_dir():
        raise ValueError("anchor_run_dir required (directory with anchor_graph_nodes.csv).")
    if scored is None or not scored.is_file():
        raise ValueError("scored_edges_csv required (email_i/email_j + edge_weight).")

    def _resolve_one_best(
        *,
        gt_path: Path,
        json_key: str,
        pick_label: str,
        config_best_key: str | None = None,
    ) -> tuple[dict[str, Any], str]:
        jp = _p(json_key)
        if jp and jp.is_file():
            return _load_best_row_from_anchor_best_json(jp), str(jp.resolve())
        if mgs_path and mgs_path.is_file():
            try:
                return _pick_best_row_for_gt_from_summary(
                    mgs_path, gt_path, project_root=project_root, gt_label=pick_label
                )
            except ValueError:
                pass
        if config_best_key:
            br = cfg.get(config_best_key)
            if isinstance(br, dict) and br:
                return dict(br), f"config_json.{config_best_key}"
        legacy = cfg.get("best_row")
        if isinstance(legacy, dict) and legacy and pick_label == "dedup":
            return dict(legacy), "config_json.best_row"
        raise ValueError(
            f"Could not resolve {pick_label} GT best_row. Provide {json_key}, "
            "community_multi_gt_summary_json containing this GT, or config best_row / dedup_best_row."
        )

    expanded_best_row, expanded_best_row_source = _resolve_one_best(
        gt_path=gt_exp,
        json_key="expanded_best_json",
        pick_label="expanded",
        config_best_key="expanded_best_row",
    )
    try:
        dedup_best_row, dedup_best_row_source = _resolve_one_best(
            gt_path=gt_dedup,
            json_key="dedup_best_json",
            pick_label="dedup",
            config_best_key="dedup_best_row",
        )
    except ValueError as dedup_exc:
        dedup_json = _p("dedup_best_json")
        if dedup_json and not dedup_json.is_file():
            raise ValueError(f"dedup_best_json not found: {dedup_json.resolve()}") from dedup_exc
        raise

    dedup_dir = _p("dedup_collapse_out_dir")
    ext_map = _p("external_id_map_csv")
    if dedup_dir is None and ext_map is None:
        if mgs_path and mgs_path.is_file():
            ctx = _load_community_context_from_multi_summary(mgs_path, project_root=project_root)
            exp = ctx.get("gt_metric_email_expansion") or {}
            mp = exp.get("member_expansion_mapping_path")
            if mp:
                p = Path(str(mp)).expanduser()
                p = p if p.is_absolute() else (project_root / p).resolve()
                if p.is_dir():
                    dedup_dir = p
                elif p.is_file() and p.name.lower() == "external_id_map.csv":
                    ext_map = p
    if dedup_dir is None and ext_map is None:
        raise ValueError("Set dedup_collapse_out_dir or external_id_map_csv (or community_multi_gt_summary_json).")

    cand = _p("candidate_union_csv")
    pair_csv = _p("pair_training_csv")

    return DedupExpandedGapConfig(
        project_root=project_root,
        gt_dedup_json=gt_dedup,
        gt_expanded_json=gt_exp,
        dedup_collapse_out_dir=dedup_dir,
        external_id_map_csv=ext_map,
        anchor_run_dir=anchor,
        scored_edges_csv=scored,
        community_multi_gt_summary_json=mgs_path if mgs_path and mgs_path.is_file() else None,
        expanded_best_row=expanded_best_row,
        expanded_best_row_source=expanded_best_row_source,
        dedup_best_row=dedup_best_row,
        dedup_best_row_source=dedup_best_row_source,
        weight_col=str(cfg.get("weight_col", "edge_weight")),
        use_edge_weights_in_partitioning=bool(cfg.get("use_edge_weights_in_partitioning", True)),
        apply_threshold_filter=bool(cfg.get("apply_threshold_filter", True)),
        seed=int(cfg.get("seed", 0)),
        candidate_union_csv=cand,
        pair_training_csv=pair_csv,
        score_threshold=float(cfg.get("score_threshold", 0.1)),
        top_lossy_campaigns=int(cfg.get("top_lossy_campaigns", 40)),
    )
