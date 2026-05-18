"""
GT-induced pairwise evidence analysis: same-campaign vs cross-campaign structure.

Samples labeled email pairs from ground truth (graph-aligned), computes channel /
semantic / provenance statistics, compares to current pipeline rule templates, and
emits JSON + CSV summaries with actionable recommendations.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from seed_candidate_workflow.utils import graph_structure_helpers as gh
from seed_candidate_workflow.utils.anchor_graph_helpers import load_anchor_graph_artifacts
from seed_candidate_workflow.utils.raw_gnn_notebook import load_ground_truth_structures
from seed_candidate_workflow.utils.scorer_diagnostics_rules import (
    BINARY_CONDITION_RULES_DEFAULT,
    CANDIDATE_RULES_DEFAULT,
)

# Graph artifact type -> pair column prefix
CORE_ARTIFACT_TYPES: tuple[str, ...] = (
    "sender",
    "email_domain",
    "url",
    "domain",
    "stem",
    "attachment",
)
ROUTING_ARTIFACT_TYPES: tuple[str, ...] = (
    "received_host",
    "return_path_domain",
    "origin_ip",
)

CHANNEL_TO_BOOL_COL: dict[str, str] = {
    "sender": "has_shared_sender",
    "email_domain": "has_shared_sender_domain",
    "url": "has_shared_url",
    "domain": "has_shared_domain",
    "stem": "has_shared_stem",
    "attachment": "has_shared_attachment",
    "received_host": "has_shared_received_host",
    "return_path_domain": "has_shared_return_path_domain",
    "origin_ip": "has_shared_origin_ip",
}

DEFAULT_COSINE_BUCKETS: tuple[tuple[str, float | None, float | None], ...] = (
    ("cosine_lt_0_85", None, 0.85),
    ("cosine_0_85_to_0_90", 0.85, 0.90),
    ("cosine_0_90_to_0_93", 0.90, 0.93),
    ("cosine_0_93_to_0_95", 0.93, 0.95),
    ("cosine_ge_0_95", 0.95, None),
)

PROVENANCE_COLS: tuple[str, ...] = (
    "from_semantic",
    "from_rare_artifact",
    "from_2hop",
    "from_component",
    "from_shared_stem_highconf",
    "semantic_cosine_max",
    "rare_artifact_rarity_max",
    "twohop_rarity_max",
    "component_cosine_max",
    "time_gap_seconds_min",
    "source_count",
)

CANDIDATE_UNION_PROVENANCE_COLS: tuple[str, ...] = (
    "from_seed",
    "from_rare_artifact",
    "from_shared_stem_highconf",
    "from_semantic",
    "from_component",
    "from_2hop",
)

WEAK_SEED_CHANNELS: tuple[str, ...] = (
    "has_shared_sender",
    "has_shared_sender_domain",
    "has_shared_domain",
    "has_shared_stem",
    "has_shared_return_path_domain",
    "has_shared_received_host",
    "has_shared_origin_ip",
)


@dataclass
class GtEdgeStructureRunConfig:
    gt_paths: list[Path]
    graph_pt: Path
    meta_json: Path
    embeddings_json: Path | None = None
    pair_training_csv: Path | None = None
    candidate_union_csv: Path | None = None
    anchor_run_dir: Path | None = None
    anchor_graph_config: Path | None = None
    anchor_seed_config: Path | None = None
    anchor_candidate_config: Path | None = None
    out_dir: Path = field(default_factory=lambda: Path("output/analysis/gt_edge_structure"))
    max_same_pairs: int = 8000
    max_cross_pairs: int = 8000
    seed: int = 0
    min_support: int = 30
    frontier_max_abs_diff: float = 0.15
    cosine_buckets: tuple[tuple[str, float | None, float | None], ...] = DEFAULT_COSINE_BUCKETS
    top_joint_combinations: int = 25
    min_new_same_pairs_for_recommendation: int = 5


def _pair_key(a: str, b: str) -> tuple[str, str]:
    aa, bb = str(a).strip(), str(b).strip()
    return (aa, bb) if aa <= bb else (bb, aa)


def _safe_enrichment(same_v: float | None, cross_v: float | None) -> float | None:
    if same_v is None or cross_v is None or cross_v == 0.0:
        return None
    return float(same_v / cross_v)


def _rate_on_mask(mask: np.ndarray, base: np.ndarray) -> tuple[float | None, int]:
    n = int(base.sum())
    if n == 0:
        return None, 0
    return float((mask & base).sum() / n), int((mask & base).sum())


def _cmp_condition(
    cond: np.ndarray,
    same_mask: np.ndarray,
    cross_mask: np.ndarray,
) -> dict[str, Any]:
    same_rate, support_same = _rate_on_mask(cond, same_mask)
    cross_rate, support_cross = _rate_on_mask(cond, cross_mask)
    diff = (
        (same_rate - cross_rate)
        if same_rate is not None and cross_rate is not None
        else None
    )
    support_total = int(cond.sum())
    precision_like: float | None = None
    if support_total > 0:
        precision_like = float((cond & same_mask).sum() / support_total)
    lift = _safe_enrichment(same_rate, cross_rate)
    return {
        "same_rate": same_rate,
        "cross_rate": cross_rate,
        "difference_same_minus_cross": diff,
        "enrichment_same_over_cross": lift,
        "precision_like": precision_like,
        "support_same": support_same,
        "support_cross": support_cross,
        "support_total": support_total,
        "same_capture": same_rate,
        "cross_contamination": cross_rate,
        "lift": lift,
    }


def _table_row(
    *,
    gt_file: str,
    section: str,
    metric_id: str,
    stats: dict[str, Any],
    notes: str = "",
) -> dict[str, Any]:
    return {
        "gt_file": gt_file,
        "section": section,
        "metric_id": metric_id,
        "same_rate": stats.get("same_rate"),
        "cross_rate": stats.get("cross_rate"),
        "difference_same_minus_cross": stats.get("difference_same_minus_cross"),
        "enrichment_same_over_cross": stats.get("enrichment_same_over_cross"),
        "precision_like": stats.get("precision_like"),
        "support_same": stats.get("support_same"),
        "support_cross": stats.get("support_cross"),
        "support_total": stats.get("support_total"),
        "notes": notes,
    }


def _resolve_embeddings_json(
    *,
    explicit: Path | None,
    anchor_run_dir: Path | None,
    project_root: Path,
) -> Path | None:
    if explicit is not None and explicit.is_file():
        return explicit.resolve()
    if anchor_run_dir is not None:
        cfg_path = anchor_run_dir / "anchor_graph_run_config.json"
        if cfg_path.is_file():
            try:
                cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
                emb = str((cfg.get("inputs") or {}).get("embeddings_json") or "").strip()
                if emb:
                    p = Path(emb)
                    if not p.is_absolute():
                        p = project_root / p
                    if p.is_file():
                        return p.resolve()
            except Exception:
                pass
    default = project_root / "core" / "utils" / "embeddings" / "output" / "embeddings.json"
    return default.resolve() if default.is_file() else None


def _load_embeddings(path: Path) -> dict[str, np.ndarray]:
    """Load subject+body embeddings from embedder cache JSON (no PyG dependency)."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    by_key = payload.get("by_key")
    if not isinstance(by_key, dict):
        raise ValueError(f"Invalid embeddings cache at {path}: missing dict 'by_key'.")
    id_to_emb: dict[str, np.ndarray] = {}
    for k, v in by_key.items():
        if not isinstance(v, dict):
            continue
        subj = np.asarray(v.get("subj") or [], dtype=np.float32).reshape(-1)
        body = np.asarray(v.get("body") or [], dtype=np.float32).reshape(-1)
        if subj.size == 0 and body.size == 0:
            continue
        eid = str(v.get("external_id") or k)
        id_to_emb[eid] = np.concatenate([subj, body], axis=0)
    if not id_to_emb:
        raise ValueError(f"No subject/body vectors in embeddings cache: {path}")
    return id_to_emb


def _l2_normalize_rows(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    return mat / norms


def _cosine_for_pairs(
    pairs: list[tuple[int, int]],
    row_to_eid: list[str],
    id_to_emb: dict[str, np.ndarray],
) -> np.ndarray:
    out = np.full(len(pairs), np.nan, dtype=np.float64)
    for k, (i, j) in enumerate(pairs):
        ei = row_to_eid[i]
        ej = row_to_eid[j]
        vi = id_to_emb.get(ei)
        vj = id_to_emb.get(ej)
        if vi is None or vj is None:
            continue
        ni = np.linalg.norm(vi)
        nj = np.linalg.norm(vj)
        if ni <= 0 or nj <= 0:
            continue
        out[k] = float(np.dot(vi, vj) / (ni * nj))
    return out


def _load_nodes_by_email_from_anchor(anchor_run_dir: Path) -> dict[str, dict[str, set[str]]]:
    nodes_df, _, _, _, _ = load_anchor_graph_artifacts(
        anchor_run_dir.resolve(),
        load_graph_pickle=False,
    )
    if "external_id" not in nodes_df.columns:
        return {}
    col_map = {
        "sender_set": "has_shared_sender",
        "sender_email_domain_set": "has_shared_sender_domain",
        "url_set": "has_shared_url",
        "domain_set": "has_shared_domain",
        "stem_set": "has_shared_stem",
        "attachment_set": "has_shared_attachment",
    }

    def _cell_set(v: Any) -> set[str]:
        if isinstance(v, set):
            return {str(x) for x in v if str(x).strip()}
        if isinstance(v, list):
            return {str(x) for x in v if str(x).strip()}
        if isinstance(v, str) and v.strip():
            return {v.strip()}
        return set()

    out: dict[str, dict[str, set[str]]] = {}
    for _, row in nodes_df.iterrows():
        eid = str(row.get("external_id") or "").strip()
        if not eid:
            continue
        entry: dict[str, set[str]] = {}
        for src_col in col_map:
            if src_col in nodes_df.columns:
                entry[src_col] = _cell_set(row.get(src_col))
        out[eid] = entry
    return out


def _share_from_sets(
    eid_i: str,
    eid_j: str,
    nodes_by_email: dict[str, dict[str, set[str]]],
    src_col: str,
) -> tuple[bool, int]:
    na = nodes_by_email.get(eid_i, {}).get(src_col) or set()
    nb = nodes_by_email.get(eid_j, {}).get(src_col) or set()
    inter = na & nb
    return len(inter) > 0, len(inter)


def _share_from_graph(
    i: int,
    j: int,
    email_sets: dict[str, list[set[int]]],
    artifact_type: str,
) -> tuple[bool, int]:
    if artifact_type not in email_sets:
        return False, 0
    si = email_sets[artifact_type][i]
    sj = email_sets[artifact_type][j]
    inter = si & sj
    return len(inter) > 0, len(inter)


_ARTIFACT_TO_ANCHOR_COL: dict[str, str] = {
    "sender": "sender_set",
    "email_domain": "sender_email_domain_set",
    "url": "url_set",
    "domain": "domain_set",
    "stem": "stem_set",
    "attachment": "attachment_set",
}


def _resolve_candidate_union_csv(
    *,
    explicit: Path | None,
    pair_training_csv: Path | None,
    project_root: Path,
) -> Path | None:
    if explicit is not None and explicit.is_file():
        return explicit.resolve()
    if pair_training_csv is None or not pair_training_csv.is_file():
        return None
    # .../graph_bundles/<graph_id>/pair_training/<graph_id>/pair_training_dataset.csv
    graph_id = pair_training_csv.parent.name
    bundle_root = pair_training_csv.parent.parent.parent
    cand_parent = bundle_root / "candidate" / graph_id
    if not cand_parent.is_dir():
        return None
    gens = sorted(
        [d for d in cand_parent.iterdir() if d.is_dir() and d.name.startswith("candidate_generation")],
        key=lambda d: d.stat().st_mtime,
    )
    if not gens:
        return None
    p = gens[-1] / "candidate_union.csv"
    return p.resolve() if p.is_file() else None


def _join_candidate_union(
    df: pd.DataFrame,
    candidate_union_csv: Path,
) -> dict[str, Any]:
    cu = pd.read_csv(candidate_union_csv)
    if cu.empty or "email_i" not in cu.columns or "email_j" not in cu.columns:
        raise ValueError(f"candidate_union.csv missing email_i/email_j: {candidate_union_csv}")
    cu = cu.copy()
    cu["_pk"] = cu.apply(
        lambda r: _pair_key(str(r["email_i"]), str(r["email_j"])),
        axis=1,
    )
    cu = cu.drop_duplicates(subset=["_pk"], keep="first")
    union_keys = set(cu["_pk"].tolist())
    in_union = np.zeros(len(df), dtype=bool)
    joined_union = 0
    for idx, r in df.iterrows():
        pk = _pair_key(str(r["email_i"]), str(r["email_j"]))
        if pk in union_keys:
            in_union[idx] = True
            joined_union += 1
    df["in_current_candidate_union"] = in_union

    prov_cols = [c for c in CANDIDATE_UNION_PROVENANCE_COLS if c in cu.columns]
    if prov_cols:
        prov_map = cu.set_index("_pk")[prov_cols].to_dict("index")
        for col in prov_cols:
            if col not in df.columns:
                df[col] = False
        for idx, r in df.iterrows():
            pk = _pair_key(str(r["email_i"]), str(r["email_j"]))
            hit = prov_map.get(pk)
            if not hit:
                continue
            for c in prov_cols:
                val = hit.get(c)
                if pd.notna(val) and bool(val):
                    df.at[idx, c] = True

    return {
        "joined": True,
        "candidate_union_csv": str(candidate_union_csv.resolve()),
        "n_union_pairs_total": int(len(cu)),
        "n_gt_sample_pairs_in_union": int(joined_union),
        "fraction_gt_sample_in_union": float(joined_union / max(len(df), 1)),
        "provenance_columns_merged": prov_cols,
    }


def _load_graph_email_sets(
    graph_pt: Path,
    *,
    to_undirected: bool = True,
) -> dict[str, list[set[int]]]:
    """Load per-email artifact sets from hetero graph (requires torch-geometric)."""
    try:
        data = gh.load_hetero(graph_pt, to_undirected=to_undirected)
    except ImportError as exc:
        raise ImportError(
            "Loading the hetero graph requires torch-geometric. Either install project "
            "dependencies (pip install -r requirements.txt) or provide --anchor-run-dir "
            "with a completed anchor graph run so shared evidence can be read from nodes.csv."
        ) from exc
    return gh.build_email_artifact_sets(data)


def build_gt_pair_dataframe(
    *,
    gt_path: Path,
    meta_json: Path,
    graph_pt: Path,
    max_same_pairs: int,
    max_cross_pairs: int,
    seed: int,
    embeddings_json: Path | None,
    anchor_run_dir: Path | None,
    pair_training_csv: Path | None,
    candidate_union_csv: Path | None,
    project_root: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    label_map, _eid_row_gt, campaign_to_members = load_ground_truth_structures(gt_path)
    label_map = {str(k): v for k, v in label_map.items()}

    meta = json.loads(meta_json.read_text(encoding="utf-8"))
    eid_row = gh.external_id_to_row(meta)
    row_to_eid = gh.email_external_id_list(meta)

    # Restrict GT campaigns to graph emails
    campaign_to_members_g: dict[Any, list[str]] = {}
    for cid, members in campaign_to_members.items():
        kept = [e for e in members if e in eid_row]
        if len(kept) >= 2:
            campaign_to_members_g[cid] = kept

    same_pairs, cross_pairs = gh.sample_campaign_email_pairs(
        campaign_to_members_g,
        eid_row,
        max_same_pairs=max_same_pairs,
        max_diff_pairs=max_cross_pairs,
        seed=seed,
    )

    nodes_by_email: dict[str, dict[str, set[str]]] = {}
    if anchor_run_dir is not None and anchor_run_dir.is_dir():
        nodes_by_email = _load_nodes_by_email_from_anchor(anchor_run_dir)

    email_sets: dict[str, list[set[int]]] = {}
    graph_load_note = ""
    if nodes_by_email:
        graph_load_note = (
            "Shared core channels from anchor nodes.csv; hetero graph not loaded."
        )
    else:
        email_sets = _load_graph_email_sets(graph_pt)
        graph_load_note = "Shared channels from hetero graph (anchor run not available)."

    emb_path = _resolve_embeddings_json(
        explicit=embeddings_json,
        anchor_run_dir=anchor_run_dir,
        project_root=project_root,
    )
    id_to_emb: dict[str, np.ndarray] = {}
    if emb_path is not None:
        id_to_emb = _load_embeddings(emb_path)

    all_pairs = [(p, True) for p in same_pairs] + [(p, False) for p in cross_pairs]
    rows: list[dict[str, Any]] = []

    for (i, j), is_same in all_pairs:
        eid_i, eid_j = row_to_eid[i], row_to_eid[j]
        rec: dict[str, Any] = {
            "email_i": eid_i,
            "email_j": eid_j,
            "graph_email_idx_i": i,
            "graph_email_idx_j": j,
            "gt_same_campaign": bool(is_same),
            "gt_campaign_i": label_map.get(eid_i),
            "gt_campaign_j": label_map.get(eid_j),
        }

        n_core_shared = 0
        for art in CORE_ARTIFACT_TYPES:
            bool_col = CHANNEL_TO_BOOL_COL[art]
            anchor_col = _ARTIFACT_TO_ANCHOR_COL[art]
            if nodes_by_email:
                shared, cnt = _share_from_sets(eid_i, eid_j, nodes_by_email, anchor_col)
            else:
                shared, cnt = _share_from_graph(i, j, email_sets, art)
            rec[bool_col] = shared
            rec[f"shared_{art}_count"] = cnt
            if shared:
                n_core_shared += 1

        for art in ROUTING_ARTIFACT_TYPES:
            bool_col = CHANNEL_TO_BOOL_COL.get(art, f"has_shared_{art}")
            if email_sets:
                shared, cnt = _share_from_graph(i, j, email_sets, art)
            else:
                shared, cnt = False, 0
            rec[bool_col] = shared
            rec[f"shared_{art}_count"] = cnt

        rec["n_shared_core_channels"] = n_core_shared

        rows.append(rec)

    df = pd.DataFrame(rows)
    all_idx_pairs = same_pairs + cross_pairs
    cos_arr = _cosine_for_pairs(all_idx_pairs, row_to_eid, id_to_emb)
    df["semantic_cosine"] = cos_arr

    # Provenance join from pair training
    prov_stats: dict[str, Any] = {"joined": False}
    if pair_training_csv is not None and pair_training_csv.is_file():
        pt = pd.read_csv(pair_training_csv)
        keep_cols = [c for c in PROVENANCE_COLS if c in pt.columns]
        if "email_i" in pt.columns and "email_j" in pt.columns and keep_cols:
            pt = pt.copy()
            keys = pt.apply(
                lambda r: _pair_key(str(r["email_i"]), str(r["email_j"])),
                axis=1,
            )
            pt["_pk"] = keys
            pt = pt.drop_duplicates(subset=["_pk"], keep="first")
            prov_map = pt.set_index("_pk")[keep_cols].to_dict("index")
            joined = 0
            for idx, r in df.iterrows():
                pk = _pair_key(str(r["email_i"]), str(r["email_j"]))
                hit = prov_map.get(pk)
                if hit is None:
                    continue
                joined += 1
                for c in keep_cols:
                    df.at[idx, c] = hit.get(c)
            prov_stats = {
                "joined": True,
                "pair_training_csv": str(pair_training_csv.resolve()),
                "n_rows_with_provenance": joined,
                "fraction_with_provenance": float(joined / max(len(df), 1)),
                "columns_joined": keep_cols,
            }

    union_stats: dict[str, Any] = {"joined": False}
    cu_path = candidate_union_csv
    if cu_path is None:
        cu_path = _resolve_candidate_union_csv(
            explicit=None,
            pair_training_csv=pair_training_csv,
            project_root=project_root,
        )
    if cu_path is not None and cu_path.is_file():
        union_stats = _join_candidate_union(df, cu_path)
    else:
        df["in_current_candidate_union"] = False

    n_gt_in_graph = sum(1 for e in label_map if e in eid_row)
    n_theoretical_same = sum(
        (len(m) * (len(m) - 1)) // 2 for m in campaign_to_members_g.values()
    )
    coverage = {
        "gt_path": str(gt_path.resolve()),
        "embeddings_json": str(emb_path) if emb_path else None,
        "anchor_run_dir": str(anchor_run_dir.resolve()) if anchor_run_dir else None,
        "n_gt_labeled_emails_in_graph": n_gt_in_graph,
        "n_campaigns_with_ge_2_graph_emails": len(campaign_to_members_g),
        "n_same_pairs_sampled": len(same_pairs),
        "n_cross_pairs_sampled": len(cross_pairs),
        "n_theoretical_within_campaign_pairs": n_theoretical_same,
        "fraction_theoretical_same_pairs_sampled": (
            float(len(same_pairs) / max(n_theoretical_same, 1))
        ),
        "max_same_pairs_cap": max_same_pairs,
        "max_cross_pairs_cap": max_cross_pairs,
        "seed": seed,
        "provenance_join": prov_stats,
        "candidate_union_join": union_stats,
        "used_anchor_node_sets": bool(nodes_by_email),
        "evidence_source_note": graph_load_note,
        "routing_channels_from_graph": bool(email_sets),
    }
    return df, coverage


def _masks_from_df(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    same = df["gt_same_campaign"].astype(bool).to_numpy()
    cross = ~same
    return same, cross


def _channel_marginals(
    df: pd.DataFrame,
    *,
    channels: Iterable[str],
    gt_file: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    same_mask, cross_mask = _masks_from_df(df)
    table_rows: list[dict[str, Any]] = []
    summary_channels: list[dict[str, Any]] = []
    for ch in channels:
        col = CHANNEL_TO_BOOL_COL.get(ch, f"has_shared_{ch}")
        if col not in df.columns:
            continue
        cond = df[col].fillna(False).astype(bool).to_numpy()
        stats = _cmp_condition(cond, same_mask, cross_mask)
        stats["channel"] = ch
        stats["column"] = col
        summary_channels.append(stats)
        table_rows.append(
            _table_row(
                gt_file=gt_file,
                section="channel_marginal",
                metric_id=ch,
                stats=stats,
            )
        )
    return summary_channels, table_rows


def _cosine_bucket_analysis(
    df: pd.DataFrame,
    *,
    buckets: tuple[tuple[str, float | None, float | None], ...],
    gt_file: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    same_mask, cross_mask = _masks_from_df(df)
    cos = pd.to_numeric(df["semantic_cosine"], errors="coerce").to_numpy()
    finite = np.isfinite(cos)
    bucket_rows: list[dict[str, Any]] = []
    table_rows: list[dict[str, Any]] = []

    for name, lo, hi in buckets:
        mask = finite.copy()
        if lo is not None:
            mask &= cos >= float(lo)
        if hi is not None:
            mask &= cos < float(hi)
        stats = _cmp_condition(mask, same_mask, cross_mask)
        stats["bucket"] = name
        stats["lo_inclusive"] = lo
        stats["hi_exclusive"] = hi
        bucket_rows.append(stats)
        table_rows.append(
            _table_row(
                gt_file=gt_file,
                section="cosine_bucket",
                metric_id=name,
                stats=stats,
            )
        )

    contamination_note = ""
    if len(bucket_rows) >= 2:
        best_jump = 0.0
        jump_bucket = ""
        for i in range(1, len(bucket_rows)):
            prev_c = bucket_rows[i - 1].get("cross_rate") or 0.0
            cur_c = bucket_rows[i].get("cross_rate") or 0.0
            jump = cur_c - prev_c
            if jump > best_jump:
                best_jump = jump
                jump_bucket = bucket_rows[i].get("bucket", "")
        if jump_bucket:
            contamination_note = (
                f"Cross-campaign rate rises most entering bucket {jump_bucket} "
                f"(delta={best_jump:.4f})."
            )

    return {
        "buckets": bucket_rows,
        "n_pairs_with_finite_cosine": int(finite.sum()),
        "contamination_slope_note": contamination_note,
    }, table_rows


def _eval_rule_expr(df: pd.DataFrame, expr: str) -> np.ndarray:
    """Evaluate AND/NOT rule on dataframe boolean/numeric columns."""
    n = len(df)
    terms = expr.split("_AND_")
    if not terms:
        return np.zeros(n, dtype=bool)

    bool_terms = _build_bool_terms(df)
    cos = pd.to_numeric(df.get("semantic_cosine"), errors="coerce").to_numpy()

    out = np.ones(n, dtype=bool)
    for tok in terms:
        neg = tok.startswith("NOT_")
        key = tok[4:] if neg else tok

        if key == "semantic_ge_0_90":
            base = np.isfinite(cos) & (cos >= 0.90)
        elif key == "semantic_ge_0_92":
            base = np.isfinite(cos) & (cos >= 0.92)
        elif key == "semantic_ge_0_93":
            base = np.isfinite(cos) & (cos >= 0.93)
        elif key == "semantic_ge_0_95":
            base = np.isfinite(cos) & (cos >= 0.95)
        elif key == "semantic_ge_0_97":
            base = np.isfinite(cos) & (cos >= 0.97)
        elif key == "semantic_band_0_85_0_90":
            base = np.isfinite(cos) & (cos >= 0.85) & (cos < 0.90)
        elif key == "n_shared_core_channels_ge_1":
            ncol = pd.to_numeric(df.get("n_shared_core_channels"), errors="coerce").fillna(0)
            base = (ncol >= 1).to_numpy()
        elif key == "n_shared_core_channels_ge_2":
            ncol = pd.to_numeric(df.get("n_shared_core_channels"), errors="coerce").fillna(0)
            base = (ncol >= 2).to_numpy()
        elif key == "shared_url":
            base = df["has_shared_url"].fillna(False).astype(bool).to_numpy()
        elif key == "shared_domain":
            base = df["has_shared_domain"].fillna(False).astype(bool).to_numpy()
        elif key == "pipe_semantic_reciprocal_ge_0_90":
            base = np.isfinite(cos) & (cos >= 0.90)
        elif key == "pipe_component_expansion_ge_0_90":
            comp = bool_terms.get("from_component", np.zeros(n, dtype=bool))
            base = comp & np.isfinite(cos) & (cos >= 0.90)
        elif key == "pipe_2hop_bounded":
            base = bool_terms.get("from_2hop", np.zeros(n, dtype=bool))
        elif key == "pipe_shared_stem_highconf":
            if "from_shared_stem_highconf" in df.columns:
                base = df["from_shared_stem_highconf"].fillna(False).astype(bool).to_numpy()
            else:
                stem = df["has_shared_stem"].fillna(False).astype(bool).to_numpy()
                base = stem & np.isfinite(cos) & (cos >= 0.93)
        elif key == "sender_domain_only":
            base = (
                df["has_shared_sender_domain"].fillna(False).astype(bool).to_numpy()
                & ~df["has_shared_sender"].fillna(False).astype(bool).to_numpy()
            )
        elif key == "shared_attachment":
            base = df["has_shared_attachment"].fillna(False).astype(bool).to_numpy()
        elif key == "shared_stem":
            base = df["has_shared_stem"].fillna(False).astype(bool).to_numpy()
        elif key == "shared_sender":
            base = df["has_shared_sender"].fillna(False).astype(bool).to_numpy()
        elif key == "weak_channel_ge_1":
            wc = np.zeros(n, dtype=bool)
            for c in WEAK_SEED_CHANNELS:
                if c in df.columns:
                    wc |= df[c].fillna(False).astype(bool).to_numpy()
            base = wc
        elif key == "corroborated_seed_approx":
            sem97 = np.isfinite(cos) & (cos >= 0.97)
            wc = np.zeros(n, dtype=bool)
            for c in WEAK_SEED_CHANNELS:
                if c in df.columns:
                    wc |= df[c].fillna(False).astype(bool).to_numpy()
            base = sem97 & wc
        elif key in bool_terms:
            base = bool_terms[key]
        else:
            base = np.zeros(n, dtype=bool)

        out &= ~base if neg else base
    return out


def _build_bool_terms(df: pd.DataFrame) -> dict[str, np.ndarray]:
    n = len(df)
    terms: dict[str, np.ndarray] = {}
    for col in (
        "has_shared_sender",
        "has_shared_stem",
        "has_shared_sender_domain",
        "has_shared_url",
        "has_shared_attachment",
        "has_shared_domain",
    ):
        if col in df.columns:
            terms[col.replace("has_shared_", "shared_")] = (
                df[col].fillna(False).astype(bool).to_numpy()
            )
            terms[col] = terms[col.replace("has_shared_", "shared_")]

    for prov in (
        "from_semantic",
        "from_rare_artifact",
        "from_2hop",
        "from_component",
        "from_shared_stem_highconf",
    ):
        if prov in df.columns:
            terms[prov] = df[prov].fillna(False).astype(bool).to_numpy()
        else:
            terms[prov] = np.zeros(n, dtype=bool)

    # Alias for rule parser
    terms["shared_sender"] = terms.get(
        "shared_sender", df.get("has_shared_sender", pd.Series(False, index=df.index))
    )
    if isinstance(terms["shared_sender"], pd.Series):
        terms["shared_sender"] = terms["shared_sender"].fillna(False).astype(bool).to_numpy()
    terms["shared_stem"] = terms.get(
        "shared_stem",
        df.get("has_shared_stem", pd.Series(False, index=df.index))
        .fillna(False)
        .astype(bool)
        .to_numpy(),
    )
    terms["shared_sender_domain"] = terms.get(
        "shared_sender_domain",
        df.get("has_shared_sender_domain", pd.Series(False, index=df.index))
        .fillna(False)
        .astype(bool)
        .to_numpy(),
    )
    return terms


def _resolve_candidate_rule(rule_name: str) -> str | None:
    """Map CANDIDATE_RULES_DEFAULT names to eval expressions."""
    mapping = {
        "likely_positive__from_semantic_AND_shared_sender": "from_semantic_AND_shared_sender",
        "likely_positive__from_semantic_AND_semantic_ge_0_93": "from_semantic_AND_semantic_ge_0_93",
        "likely_positive__from_semantic_AND_shared_sender_AND_NOT_from_2hop": (
            "from_semantic_AND_shared_sender_AND_NOT_from_2hop"
        ),
        "likely_positive__shared_sender_AND_NOT_from_2hop": "shared_sender_AND_NOT_from_2hop",
        "likely_negative__from_2hop_AND_NOT_shared_sender": "from_2hop_AND_NOT_shared_sender",
        "likely_negative__from_2hop_AND_NOT_from_semantic": "from_2hop_AND_NOT_from_semantic",
        "likely_negative__from_component_AND_NOT_shared_sender": (
            "from_component_AND_NOT_shared_sender"
        ),
        "likely_negative__shared_sender_domain_AND_NOT_shared_sender": "sender_domain_only",
    }
    return mapping.get(rule_name)


def _rule_scorecard(
    df: pd.DataFrame,
    rule_defs: list[tuple[str, str]],
    *,
    gt_file: str,
    min_support: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    same_mask, cross_mask = _masks_from_df(df)
    scorecard: list[dict[str, Any]] = []
    table_rows: list[dict[str, Any]] = []

    for rule_id, expr in rule_defs:
        cond = _eval_rule_expr(df, expr)
        stats = _cmp_condition(cond, same_mask, cross_mask)
        if stats["support_total"] < min_support:
            continue
        row = {
            "gt_file": gt_file,
            "rule_id": rule_id,
            "rule_expression": expr,
            **stats,
        }
        scorecard.append(row)
        table_rows.append(
            _table_row(
                gt_file=gt_file,
                section="rule_scorecard",
                metric_id=rule_id,
                stats=stats,
                notes=expr,
            )
        )

    scorecard.sort(
        key=lambda r: (
            -(r.get("lift") or 0.0),
            -(r.get("same_capture") or 0.0),
            r.get("rule_id", ""),
        )
    )
    return scorecard, table_rows


def _candidate_rule_definitions() -> list[tuple[str, str, str]]:
    """(rule_name, expression, category) curated templates for candidate/seed design."""
    rules: list[tuple[str, str, str]] = []

    def add(category: str, items: list[tuple[str, str]]) -> None:
        for name, expr in items:
            rules.append((name, expr, category))

    add(
        "A_strong_semantic",
        [
            ("semantic_ge_0_90", "semantic_ge_0_90"),
            ("semantic_ge_0_92", "semantic_ge_0_92"),
            ("semantic_ge_0_93", "semantic_ge_0_93"),
            ("semantic_ge_0_95", "semantic_ge_0_95"),
        ],
    )
    add(
        "B_semantic_plus_support",
        [
            ("semantic_ge_0_90_AND_shared_sender", "semantic_ge_0_90_AND_shared_sender"),
            ("semantic_ge_0_90_AND_shared_stem", "semantic_ge_0_90_AND_shared_stem"),
            ("semantic_ge_0_90_AND_shared_url", "semantic_ge_0_90_AND_shared_url"),
            ("semantic_ge_0_90_AND_shared_attachment", "semantic_ge_0_90_AND_shared_attachment"),
            (
                "semantic_ge_0_90_AND_shared_sender_domain",
                "semantic_ge_0_90_AND_shared_sender_domain",
            ),
            (
                "semantic_ge_0_90_AND_n_shared_core_channels_ge_1",
                "semantic_ge_0_90_AND_n_shared_core_channels_ge_1",
            ),
            (
                "semantic_ge_0_90_AND_n_shared_core_channels_ge_2",
                "semantic_ge_0_90_AND_n_shared_core_channels_ge_2",
            ),
            (
                "semantic_band_0_85_0_90_AND_shared_sender",
                "semantic_band_0_85_0_90_AND_shared_sender",
            ),
            (
                "semantic_band_0_85_0_90_AND_shared_stem",
                "semantic_band_0_85_0_90_AND_shared_stem",
            ),
            (
                "semantic_band_0_85_0_90_AND_n_shared_core_channels_ge_1",
                "semantic_band_0_85_0_90_AND_n_shared_core_channels_ge_1",
            ),
        ],
    )
    add(
        "C_structural_frontier",
        [
            ("from_2hop_AND_semantic_ge_0_90", "from_2hop_AND_semantic_ge_0_90"),
            ("from_component_AND_semantic_ge_0_90", "from_component_AND_semantic_ge_0_90"),
            ("from_2hop_AND_shared_sender", "from_2hop_AND_shared_sender"),
            ("from_component_AND_shared_stem", "from_component_AND_shared_stem"),
            ("from_2hop_AND_NOT_shared_sender", "from_2hop_AND_NOT_shared_sender"),
            ("from_component_AND_NOT_shared_sender", "from_component_AND_NOT_shared_sender"),
            ("from_2hop_AND_NOT_from_semantic", "from_2hop_AND_NOT_from_semantic"),
            ("from_component_AND_NOT_from_semantic", "from_component_AND_NOT_from_semantic"),
        ],
    )
    add(
        "D_shared_channel",
        [
            ("shared_sender", "shared_sender"),
            ("shared_stem", "shared_stem"),
            ("shared_url", "shared_url"),
            ("shared_attachment", "shared_attachment"),
            ("shared_sender_domain", "shared_sender_domain"),
            ("shared_domain", "shared_domain"),
            ("n_shared_core_channels_ge_1", "n_shared_core_channels_ge_1"),
            ("n_shared_core_channels_ge_2", "n_shared_core_channels_ge_2"),
        ],
    )
    add(
        "E_pipeline_approx",
        [
            ("pipe_semantic_reciprocal_ge_0_90", "pipe_semantic_reciprocal_ge_0_90"),
            ("pipe_component_expansion_ge_0_90", "pipe_component_expansion_ge_0_90"),
            ("pipe_2hop_bounded", "pipe_2hop_bounded"),
            ("pipe_shared_stem_highconf", "pipe_shared_stem_highconf"),
            ("from_semantic", "from_semantic"),
            ("from_component", "from_component"),
            ("from_2hop", "from_2hop"),
            ("from_shared_stem_highconf", "from_shared_stem_highconf"),
        ],
    )

    seen: set[str] = set()
    out: list[tuple[str, str, str]] = []
    for name, expr, cat in rules:
        if name in seen:
            continue
        seen.add(name)
        out.append((name, expr, cat))
    return out


def _score_candidate_rule_row(
    cond: np.ndarray,
    same_mask: np.ndarray,
    cross_mask: np.ndarray,
    union_mask: np.ndarray | None,
    *,
    rule_name: str,
    rule_expression: str,
    rule_category: str,
    gt_file: str,
    n_same_total: int,
    n_cross_total: int,
) -> dict[str, Any]:
    same_cap = int((cond & same_mask).sum())
    cross_cap = int((cond & cross_mask).sum())
    support_total = int(cond.sum())
    support_same = int((cond & same_mask).sum())
    support_cross = int((cond & cross_mask).sum())

    same_capture_rate = float(same_cap / n_same_total) if n_same_total else None
    cross_capture_rate = float(cross_cap / n_cross_total) if n_cross_total else None
    precision_like = (
        float(same_cap / support_total) if support_total > 0 else None
    )
    lift = _safe_enrichment(same_capture_rate, cross_capture_rate)

    row: dict[str, Any] = {
        "gt_file": gt_file,
        "rule_name": rule_name,
        "rule_category": rule_category,
        "rule_expression": rule_expression,
        "same_pairs_captured": same_cap,
        "cross_pairs_captured": cross_cap,
        "same_capture_rate": same_capture_rate,
        "cross_capture_rate": cross_capture_rate,
        "precision_like": precision_like,
        "lift": lift,
        "support_total": support_total,
        "support_same": support_same,
        "support_cross": support_cross,
        # legacy aliases for expanded rule_scorecard.csv
        "rule_id": rule_name,
        "same_rate": same_capture_rate,
        "cross_rate": cross_capture_rate,
        "same_capture": same_capture_rate,
        "cross_contamination": cross_capture_rate,
        "enrichment_same_over_cross": lift,
    }

    if union_mask is not None:
        new_cond = cond & ~union_mask
        same_new = int((new_cond & same_mask).sum())
        cross_new = int((new_cond & cross_mask).sum())
        denom_new = same_new + cross_new
        row.update(
            {
                "same_pairs_new_not_in_union": same_new,
                "cross_pairs_new_not_in_union": cross_new,
                "same_new_capture_rate": (
                    float(same_new / n_same_total) if n_same_total else None
                ),
                "cross_new_capture_rate": (
                    float(cross_new / n_cross_total) if n_cross_total else None
                ),
                "precision_like_new": (
                    float(same_new / denom_new) if denom_new > 0 else None
                ),
            }
        )
    else:
        row.update(
            {
                "same_pairs_new_not_in_union": None,
                "cross_pairs_new_not_in_union": None,
                "same_new_capture_rate": None,
                "cross_new_capture_rate": None,
                "precision_like_new": None,
            }
        )
    return row


def _candidate_rule_scorecard(
    df: pd.DataFrame,
    *,
    gt_file: str,
    min_support: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    same_mask, cross_mask = _masks_from_df(df)
    n_same = int(same_mask.sum())
    n_cross = int(cross_mask.sum())
    union_mask: np.ndarray | None = None
    if "in_current_candidate_union" in df.columns:
        union_mask = df["in_current_candidate_union"].fillna(False).astype(bool).to_numpy()

    scorecard: list[dict[str, Any]] = []
    table_rows: list[dict[str, Any]] = []

    for rule_name, expr, category in _candidate_rule_definitions():
        cond = _eval_rule_expr(df, expr)
        if int(cond.sum()) < min_support:
            continue
        row = _score_candidate_rule_row(
            cond,
            same_mask,
            cross_mask,
            union_mask,
            rule_name=rule_name,
            rule_expression=expr,
            rule_category=category,
            gt_file=gt_file,
            n_same_total=n_same,
            n_cross_total=n_cross,
        )
        scorecard.append(row)
        stats = {
            "same_rate": row["same_capture_rate"],
            "cross_rate": row["cross_capture_rate"],
            "precision_like": row["precision_like"],
            "support_same": row["support_same"],
            "support_cross": row["support_cross"],
            "support_total": row["support_total"],
            "enrichment_same_over_cross": row["lift"],
        }
        table_rows.append(
            _table_row(
                gt_file=gt_file,
                section="candidate_rule_scorecard",
                metric_id=rule_name,
                stats=stats,
                notes=f"{category}; {expr}",
            )
        )

    scorecard.sort(
        key=lambda r: (
            -(r.get("same_pairs_new_not_in_union") or 0),
            -(r.get("precision_like_new") or r.get("precision_like") or 0.0),
            -(r.get("lift") or 0.0),
        )
    )
    return scorecard, table_rows


def _generate_candidate_rule_recommendations(
    scorecard: list[dict[str, Any]],
    *,
    min_new_same_pairs: int,
    union_joined: bool,
) -> dict[str, Any]:
    recommended_seed: list[str] = []
    recommended_candidate: list[str] = []
    frontier_risky: list[str] = []
    reject_noisy: list[str] = []
    redundant_low_novelty: list[str] = []

    for r in scorecard:
        name = r.get("rule_name", "")
        prec = r.get("precision_like")
        prec_new = r.get("precision_like_new")
        same_new = r.get("same_pairs_new_not_in_union")
        cross_new = r.get("cross_pairs_new_not_in_union")
        cross_new_rate = r.get("cross_new_capture_rate")
        same_new_rate = r.get("same_new_capture_rate")
        cat = r.get("rule_category", "")

        line_base = (
            f"{name} [{cat}]: precision={prec:.3f}" if prec is not None else f"{name} [{cat}]"
        )
        if union_joined and prec_new is not None:
            line = (
                f"{line_base}, precision_on_new={prec_new:.3f}, "
                f"new_same={same_new}, new_cross={cross_new}, "
                f"new_same_rate={same_new_rate:.4f}, new_cross_rate={cross_new_rate:.4f}"
            )
        else:
            line = line_base

        if union_joined and same_new is not None:
            if int(same_new or 0) < min_new_same_pairs and int(r.get("same_pairs_captured") or 0) >= min_new_same_pairs:
                redundant_low_novelty.append(
                    f"{line} — mostly redundant vs current candidate union."
                )
                continue

        if prec_new is not None and union_joined:
            if prec_new >= 0.90 and (cross_new_rate or 1) <= 0.02 and (same_new_rate or 0) >= 0.02:
                recommended_seed.append(
                    f"{line} — strong seed-like: high precision on new pairs, low new cross contamination."
                )
                continue
            if (
                prec_new >= 0.75
                and (cross_new_rate or 1) <= 0.04
                and int(same_new or 0) >= min_new_same_pairs
            ):
                recommended_candidate.append(
                    f"{line} — good candidate broadening: adds new same-campaign mass with tolerable contamination."
                )
                continue
            if (cross_new_rate or 0) >= 0.08 or prec_new < 0.55:
                reject_noisy.append(
                    f"{line} — too noisy for promotion (high new cross rate or low precision on new pairs)."
                )
                continue
            if prec_new >= 0.55 or cat == "C_structural_frontier":
                frontier_risky.append(
                    f"{line} — frontier-only: useful for unlabeled mining, not seed promotion."
                )
                continue
        else:
            if prec is not None and prec >= 0.90 and (r.get("cross_capture_rate") or 1) <= 0.03:
                recommended_seed.append(f"{line} — high GT precision (union novelty not available).")
            elif prec is not None and prec >= 0.75:
                recommended_candidate.append(f"{line} — moderate GT precision (union novelty not available).")
            elif prec is not None and prec < 0.55:
                reject_noisy.append(f"{line} — low GT precision.")

    top_new_same = sorted(
        scorecard,
        key=lambda x: (-(x.get("same_pairs_new_not_in_union") or 0), -(x.get("precision_like_new") or 0)),
    )[:12]

    return {
        "union_novelty_available": union_joined,
        "recommended_seed_like_additions": recommended_seed[:15],
        "recommended_candidate_broadening": recommended_candidate[:15],
        "frontier_only_or_risky": frontier_risky[:15],
        "reject_too_noisy": reject_noisy[:15],
        "redundant_vs_current_union": redundant_low_novelty[:15],
        "top_rules_by_new_same_pairs": [
            {
                "rule_name": t.get("rule_name"),
                "same_pairs_new_not_in_union": t.get("same_pairs_new_not_in_union"),
                "precision_like_new": t.get("precision_like_new"),
                "cross_new_capture_rate": t.get("cross_new_capture_rate"),
            }
            for t in top_new_same
        ],
        "interpretation_note": (
            "Prioritize rules with high precision_like_new and meaningful "
            "same_pairs_new_not_in_union; demote rules that only duplicate current union mass."
        ),
    }


def _joint_combination_analysis(
    df: pd.DataFrame,
    *,
    gt_file: str,
    min_support: int,
    top_n: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    joint_defs: list[tuple[str, str]] = [
        ("semantic_ge_0_95", "semantic_ge_0_95"),
        ("semantic_ge_0_93", "semantic_ge_0_93"),
        ("semantic_ge_0_90", "semantic_ge_0_90"),
        ("semantic_ge_0_93_AND_shared_sender", "semantic_ge_0_93_AND_shared_sender"),
        ("semantic_ge_0_95_AND_shared_sender", "semantic_ge_0_95_AND_shared_sender"),
        ("semantic_ge_0_93_AND_shared_stem", "semantic_ge_0_93_AND_shared_stem"),
        ("shared_attachment", "shared_attachment"),
        ("shared_stem", "shared_stem"),
        ("sender_domain_only", "sender_domain_only"),
        ("from_2hop_AND_semantic_ge_0_90", "from_2hop_AND_semantic_ge_0_90"),
        ("from_2hop_AND_NOT_from_semantic", "from_2hop_AND_NOT_from_semantic"),
        ("from_component_AND_shared_sender", "from_component_AND_shared_sender"),
        ("corroborated_seed_approx", "corroborated_seed_approx"),
    ]
    joint_defs.extend((n, n) for n in BINARY_CONDITION_RULES_DEFAULT)

    rows, table_rows = _rule_scorecard(
        df, joint_defs, gt_file=gt_file, min_support=min_support
    )
    for r in rows:
        r["section"] = "joint_combination"
    top_by_enrich = sorted(
        rows,
        key=lambda r: (-(r.get("enrichment_same_over_cross") or 0.0), -(r.get("support_total") or 0)),
    )[:top_n]
    top_by_same = sorted(
        rows,
        key=lambda r: (-(r.get("same_rate") or 0.0), (r.get("cross_rate") or 1.0)),
    )[:top_n]

    # n_shared_core_channels buckets
    same_mask, cross_mask = _masks_from_df(df)
    channel_count_rows: list[dict[str, Any]] = []
    if "n_shared_core_channels" in df.columns:
        for n_ch in sorted(df["n_shared_core_channels"].dropna().unique()):
            cond = df["n_shared_core_channels"].eq(n_ch).to_numpy()
            stats = _cmp_condition(cond, same_mask, cross_mask)
            stats["n_shared_core_channels"] = int(n_ch)
            channel_count_rows.append(stats)
            table_rows.append(
                _table_row(
                    gt_file=gt_file,
                    section="joint_combination",
                    metric_id=f"n_shared_core_channels_eq_{int(n_ch)}",
                    stats=stats,
                )
            )

    return {
        "top_by_enrichment": top_by_enrich[:top_n],
        "top_by_same_rate": top_by_same[:top_n],
        "n_shared_core_channels_distribution": channel_count_rows,
    }, table_rows


def _frontier_analysis(
    df: pd.DataFrame,
    *,
    gt_file: str,
    min_support: int,
    max_abs_diff: float,
    rule_defs: list[tuple[str, str]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    same_mask, cross_mask = _masks_from_df(df)
    table_rows: list[dict[str, Any]] = []
    scored: list[dict[str, Any]] = []

    for rule_id, expr in rule_defs:
        cond = _eval_rule_expr(df, expr)
        stats = _cmp_condition(cond, same_mask, cross_mask)
        if stats["support_total"] < min_support:
            continue
        sr = stats.get("same_rate")
        cr = stats.get("cross_rate")
        if sr is None or cr is None:
            continue
        abs_diff = abs(sr - cr)
        entry = {
            "rule_id": rule_id,
            "rule_expression": expr,
            **stats,
            "abs_same_minus_cross": abs_diff,
        }
        scored.append(entry)
        table_rows.append(
            _table_row(
                gt_file=gt_file,
                section="frontier",
                metric_id=rule_id,
                stats=stats,
                notes=f"abs_diff={abs_diff:.4f}",
            )
        )

    ambiguous = [
        s
        for s in scored
        if s["abs_same_minus_cross"] <= max_abs_diff
        and (s.get("support_total") or 0) >= min_support
    ]
    ambiguous.sort(key=lambda s: s["abs_same_minus_cross"])

    easy_same = [
        s
        for s in scored
        if (s.get("same_rate") or 0) >= 0.5
        and (s.get("cross_rate") or 1) <= 0.05
        and (s.get("support_total") or 0) >= min_support
    ]
    easy_same.sort(key=lambda s: (-(s.get("same_rate") or 0), s.get("cross_rate") or 1))

    easy_cross = [
        s
        for s in scored
        if (s.get("cross_rate") or 0) >= 0.15
        and (s.get("same_rate") or 1) <= 0.3
        and (s.get("support_total") or 0) >= min_support
    ]
    easy_cross.sort(key=lambda s: (-(s.get("cross_rate") or 0), s.get("same_rate") or 1))

    return {
        "frontier_max_abs_diff": max_abs_diff,
        "ambiguous_overlap": ambiguous[:20],
        "easy_same": easy_same[:15],
        "easy_cross_negative": easy_cross[:15],
    }, table_rows


def _load_config_threshold_audit(
    *,
    anchor_graph_config: Path | None,
    anchor_seed_config: Path | None,
    anchor_candidate_config: Path | None,
    project_root: Path,
) -> dict[str, Any]:
    audit: dict[str, Any] = {"anchor_graph": {}, "anchor_seed": {}, "anchor_candidate": {}}

    def _read(p: Path | None) -> dict[str, Any]:
        if p is None or not p.is_file():
            return {}
        return json.loads(p.read_text(encoding="utf-8"))

    ag = _read(anchor_graph_config)
    if ag:
        sem = (ag.get("channels") or {}).get("channel_settings", {}).get("semantic", {})
        audit["anchor_graph"] = {
            "config_path": str(anchor_graph_config.resolve()),
            "semantic_min_cos": sem.get("min_cos"),
            "semantic_top_k": sem.get("top_k"),
            "note": "Anchor semantic candidates use min_cos and per-email top_k.",
        }

    sd = _read(anchor_seed_config)
    if sd:
        gens = (sd.get("seeds") or {}).get("generators") or []
        for g in gens:
            if g.get("name") == "corroborated_v1":
                ss = g.get("semantic_support") or {}
                audit["anchor_seed"]["corroborated_v1"] = {
                    "min_semantic_score": ss.get("min_semantic_score"),
                    "require_min_support_channels": g.get("require_min_support_channels"),
                    "min_non_semantic_support_channels": ss.get(
                        "min_non_semantic_support_channels"
                    ),
                    "weak_channels": g.get("weak_channels"),
                }
            if g.get("name") == "hard_v1":
                audit["anchor_seed"]["hard_v1_rules"] = g.get("rules")

    cand = _read(anchor_candidate_config)
    if cand:
        for g in (cand.get("candidates") or {}).get("generators") or []:
            name = g.get("name")
            cfg = g.get("config") or {}
            if name == "semantic_reciprocal_v1":
                audit["anchor_candidate"]["semantic_reciprocal_v1"] = {
                    "semantic_min_cos": cfg.get("semantic_min_cos"),
                    "semantic_top_k": cfg.get("semantic_top_k"),
                }
            elif name == "component_expansion_v1":
                audit["anchor_candidate"]["component_expansion_v1"] = {
                    "semantic_centroid_min_cos": cfg.get("semantic_centroid_min_cos"),
                    "singleton_semantic_min_cos": cfg.get("singleton_semantic_min_cos"),
                }
            elif name == "2hop_bounded_v1":
                audit["anchor_candidate"]["2hop_bounded_v1"] = {
                    "semantic_min_cos_contradiction": cfg.get("semantic_min_cos"),
                    "max_total_pairs": cfg.get("max_total_pairs"),
                }

    audit["project_root"] = str(project_root.resolve())
    return audit


def _generate_recommendations(
    *,
    channel_marginals: list[dict[str, Any]],
    cosine_analysis: dict[str, Any],
    scorecard: list[dict[str, Any]],
    joint: dict[str, Any],
    frontier: dict[str, Any],
    config_audit: dict[str, Any],
) -> dict[str, list[str]]:
    anchor_recs: list[str] = []
    seed_recs: list[str] = []
    cand_recs: list[str] = []
    train_recs: list[str] = []
    frontier_recs: list[str] = []

    for ch in channel_marginals:
        enrich = ch.get("enrichment_same_over_cross")
        sr = ch.get("same_rate")
        cr = ch.get("cross_rate")
        if enrich is not None and enrich >= 1.5 and sr is not None and cr is not None:
            train_recs.append(
                f"Channel '{ch.get('channel')}': enrichment={enrich:.2f}, "
                f"same_rate={sr:.3f}, cross_rate={cr:.3f} — strong GT separator; "
                "ensure it remains explicit in pair-training features."
            )

    for rule in scorecard[:12]:
        lift = rule.get("lift")
        sc = rule.get("same_capture")
        cc = rule.get("cross_contamination")
        rid = rule.get("rule_id", "")
        if lift is None or sc is None or cc is None:
            continue
        if lift >= 2.0 and sc >= 0.2 and cc <= 0.05:
            seed_recs.append(
                f"Promote rule '{rid}': lift={lift:.2f}, same_capture={sc:.3f}, "
                f"cross_contamination={cc:.3f}."
            )
        elif cc >= 0.15:
            cand_recs.append(
                f"Flag rule '{rid}': cross_contamination={cc:.3f} is high even if "
                f"same_capture={sc:.3f}; tighten or require additional corroboration."
            )

    ag_sem = (config_audit.get("anchor_graph") or {}).get("semantic_min_cos")
    if ag_sem is not None:
        anchor_recs.append(
            f"Current anchor semantic min_cos={ag_sem}; compare to cosine_bucket "
            "contamination_slope_note before lowering further."
        )

    contam = cosine_analysis.get("contamination_slope_note") or ""
    if contam:
        anchor_recs.append(contam)

    for item in (joint.get("top_by_enrichment") or [])[:5]:
        rid = item.get("rule_id")
        enrich = item.get("enrichment_same_over_cross")
        cc = item.get("cross_contamination")
        if rid and enrich and cc is not None:
            cand_recs.append(
                f"Joint pattern '{rid}': enrichment={enrich:.2f}, "
                f"cross_contamination={cc:.3f} — candidate for stricter candidate gate."
            )

    for item in (frontier.get("ambiguous_overlap") or [])[:5]:
        sr = item.get("same_rate")
        cr = item.get("cross_rate")
        frontier_recs.append(
            f"Ambiguous frontier '{item.get('rule_id')}': same_rate={sr}, "
            f"cross_rate={cr} — use for unlabeled/frontier mining, not seeds."
        )

    if not seed_recs:
        seed_recs.append(
            "No rule met promote thresholds (lift>=2, same_capture>=0.2, cross<=0.05); "
            "prefer combination rules over single weak channels."
        )

    return {
        "anchor_threshold_recommendations": anchor_recs,
        "seed_recommendations": seed_recs,
        "candidate_recommendations": cand_recs,
        "training_feature_recommendations": train_recs,
        "unlabeled_frontier_recommendations": frontier_recs,
    }


def _all_rule_definitions() -> list[tuple[str, str]]:
    rules: list[tuple[str, str]] = []
    rules.extend((n, n) for n in BINARY_CONDITION_RULES_DEFAULT)
    rules.extend(
        [
            ("cosine_ge_0_90", "semantic_ge_0_90"),
            ("cosine_ge_0_93", "semantic_ge_0_93"),
            ("cosine_ge_0_95", "semantic_ge_0_95"),
            ("cosine_ge_0_97", "semantic_ge_0_97"),
            ("cosine_ge_0_93_AND_shared_sender", "semantic_ge_0_93_AND_shared_sender"),
            ("cosine_ge_0_95_AND_shared_sender", "semantic_ge_0_95_AND_shared_sender"),
            ("shared_attachment", "shared_attachment"),
            ("shared_stem", "shared_stem"),
            ("sender_domain_only", "sender_domain_only"),
            ("from_2hop_AND_semantic_ge_0_90", "from_2hop_AND_semantic_ge_0_90"),
            ("corroborated_seed_approx", "corroborated_seed_approx"),
        ]
    )
    for cr in CANDIDATE_RULES_DEFAULT:
        expr = _resolve_candidate_rule(cr)
        if expr:
            rules.append((cr, expr))
    # dedupe
    seen: set[str] = set()
    out: list[tuple[str, str]] = []
    for rid, expr in rules:
        if rid in seen:
            continue
        seen.add(rid)
        out.append((rid, expr))
    return out


def analyze_gt_file(
    cfg: GtEdgeStructureRunConfig,
    gt_path: Path,
    *,
    project_root: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    df, coverage = build_gt_pair_dataframe(
        gt_path=gt_path,
        meta_json=cfg.meta_json,
        graph_pt=cfg.graph_pt,
        max_same_pairs=cfg.max_same_pairs,
        max_cross_pairs=cfg.max_cross_pairs,
        seed=cfg.seed,
        embeddings_json=cfg.embeddings_json,
        anchor_run_dir=cfg.anchor_run_dir,
        pair_training_csv=cfg.pair_training_csv,
        candidate_union_csv=cfg.candidate_union_csv,
        project_root=project_root,
    )

    gt_file = gt_path.name
    table_rows: list[dict[str, Any]] = []

    core_channels, ch_rows = _channel_marginals(
        df, channels=CORE_ARTIFACT_TYPES, gt_file=gt_file
    )
    table_rows.extend(ch_rows)
    routing_channels, rt_rows = _channel_marginals(
        df, channels=ROUTING_ARTIFACT_TYPES, gt_file=gt_file
    )
    table_rows.extend(rt_rows)

    cosine_analysis, cos_rows = _cosine_bucket_analysis(
        df, buckets=cfg.cosine_buckets, gt_file=gt_file
    )
    table_rows.extend(cos_rows)

    rule_defs = _all_rule_definitions()
    scorecard, sc_rows = _rule_scorecard(
        df, rule_defs, gt_file=gt_file, min_support=cfg.min_support
    )
    table_rows.extend(sc_rows)

    candidate_scorecard, cand_rows = _candidate_rule_scorecard(
        df, gt_file=gt_file, min_support=cfg.min_support
    )
    table_rows.extend(cand_rows)
    union_joined = bool((coverage.get("candidate_union_join") or {}).get("joined"))
    candidate_recommendations = _generate_candidate_rule_recommendations(
        candidate_scorecard,
        min_new_same_pairs=cfg.min_new_same_pairs_for_recommendation,
        union_joined=union_joined,
    )

    joint, joint_rows = _joint_combination_analysis(
        df,
        gt_file=gt_file,
        min_support=cfg.min_support,
        top_n=cfg.top_joint_combinations,
    )
    table_rows.extend(joint_rows)

    frontier, frontier_rows = _frontier_analysis(
        df,
        gt_file=gt_file,
        min_support=cfg.min_support,
        max_abs_diff=cfg.frontier_max_abs_diff,
        rule_defs=rule_defs,
    )
    table_rows.extend(frontier_rows)

    config_audit = _load_config_threshold_audit(
        anchor_graph_config=cfg.anchor_graph_config,
        anchor_seed_config=cfg.anchor_seed_config,
        anchor_candidate_config=cfg.anchor_candidate_config,
        project_root=project_root,
    )

    recommendations = _generate_recommendations(
        channel_marginals=core_channels,
        cosine_analysis=cosine_analysis,
        scorecard=scorecard,
        joint=joint,
        frontier=frontier,
        config_audit=config_audit,
    )

    summary = {
        "gt_file": gt_file,
        "gt_path": str(gt_path.resolve()),
        "global_summary": coverage,
        "channel_marginals": {
            "core": core_channels,
            "routing_noisy": routing_channels,
        },
        "cosine_bucket_analysis": cosine_analysis,
        "joint_combinations": joint,
        "frontier_analysis": frontier,
        "current_rules_vs_gt": {
            "config_threshold_audit": config_audit,
            "rule_scorecard_top20": scorecard[:20],
        },
        "candidate_rule_scorecard": {
            "n_rules_evaluated": len(candidate_scorecard),
            "top_by_new_same_pairs": candidate_scorecard[:20],
            "candidate_union_join": coverage.get("candidate_union_join"),
        },
        "candidate_rule_recommendations": candidate_recommendations,
        "recommendations": recommendations,
    }
    return summary, table_rows, scorecard, candidate_scorecard


def resolve_gt_paths(
    *,
    gt_json: list[Path] | None,
    gt_dir: Path | None,
    gt_set: str | None,
    project_root: Path,
) -> list[Path]:
    paths: list[Path] = []
    if gt_json:
        paths.extend(Path(p).resolve() for p in gt_json)
    if gt_dir is not None:
        d = gt_dir.resolve()
        if d.is_dir():
            paths.extend(sorted(d.glob("*.json")))
    if gt_set:
        reg_path = project_root / "seed_candidate_workflow" / "configs" / "experiments" / "gt_sets.json"
        reg = json.loads(reg_path.read_text(encoding="utf-8"))
        for rel in reg.get(gt_set) or []:
            p = Path(rel)
            if not p.is_absolute():
                p = project_root / p
            paths.append(p.resolve())
    if not paths:
        default = project_root / "data" / "groundtruth" / "ground_truth.json"
        if default.is_file():
            paths.append(default.resolve())
    # dedupe preserve order
    seen: set[str] = set()
    out: list[Path] = []
    for p in paths:
        s = str(p)
        if s not in seen and p.is_file():
            seen.add(s)
            out.append(p)
    if not out:
        raise FileNotFoundError("No ground-truth JSON files resolved.")
    return out


def run_gt_edge_structure_analysis(cfg: GtEdgeStructureRunConfig) -> dict[str, Any]:
    project_root = gh.find_project_root()
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    per_gt: list[dict[str, Any]] = []
    all_table: list[dict[str, Any]] = []
    all_scorecard: list[dict[str, Any]] = []
    all_candidate_scorecard: list[dict[str, Any]] = []

    for gt_path in cfg.gt_paths:
        summary, table_rows, scorecard, candidate_scorecard = analyze_gt_file(
            cfg, gt_path, project_root=project_root
        )
        per_gt.append(summary)
        all_table.extend(table_rows)
        for sc in scorecard:
            sc["gt_file"] = gt_path.name
            all_scorecard.append(sc)
        for sc in candidate_scorecard:
            sc["gt_file"] = gt_path.name
            all_candidate_scorecard.append(sc)

    result = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "graph_pt": str(cfg.graph_pt.resolve()),
        "meta_json": str(cfg.meta_json.resolve()),
        "n_gt_files": len(cfg.gt_paths),
        "per_gt": per_gt,
    }

    summary_path = cfg.out_dir / "gt_edge_structure_analysis_summary.json"
    table_path = cfg.out_dir / "gt_edge_structure_analysis_table.csv"
    scorecard_path = cfg.out_dir / "gt_edge_structure_rule_scorecard.csv"
    candidate_scorecard_path = cfg.out_dir / "gt_candidate_rule_scorecard.csv"

    summary_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    pd.DataFrame(all_table).to_csv(table_path, index=False)
    # Expanded scorecard: legacy diagnostic rules + curated candidate-rule templates.
    combined_scorecard = all_scorecard + all_candidate_scorecard
    pd.DataFrame(combined_scorecard).to_csv(scorecard_path, index=False)
    pd.DataFrame(all_candidate_scorecard).to_csv(candidate_scorecard_path, index=False)

    result["output_paths"] = {
        "summary_json": str(summary_path.resolve()),
        "table_csv": str(table_path.resolve()),
        "rule_scorecard_csv": str(scorecard_path.resolve()),
        "candidate_rule_scorecard_csv": str(candidate_scorecard_path.resolve()),
    }
    return result
