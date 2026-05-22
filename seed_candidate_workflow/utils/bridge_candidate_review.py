"""
Bridge-candidate explainability: enrichment, band analysis, and dedicated HTML review cards.
"""

from __future__ import annotations

import html
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from seed_candidate_workflow.utils.bridge_candidate_experiment import canonical_pair
from seed_candidate_workflow.utils.pair_score_separation import (
    _PAIR_SHARED_CHANNEL_DEFS,
    _admitting_evidence_section_html,
    _email_pane_html,
    _format_pair_metric_value,
    _inject_semantic_cosine_for_manual_review,
    _pair_shared_evidence_detail,
    _resolve_embeddings_json_for_review,
)

BRIDGE_EXPLICIT_FEATURE_COLS: tuple[str, ...] = (
    "semantic_cosine_max",
    "body_token_jaccard",
    "body_char4gram_jaccard",
    "body_only_token_jaccard",
    "body_only_char4gram_jaccard",
    "path_token_jaccard_combined",
    "url_path_token_jaccard",
    "stem_path_token_jaccard",
    "sender_localpart_norm_jaccard",
    "time_gap_seconds_min",
    "source_count",
    "shared_sender_count",
    "shared_stem_count",
    "shared_url_count",
    "shared_attachment_count",
    "shared_sender_domain_count",
    "shared_domain_count",
    "n_shared_core_channels",
    "same_seed_component_flag",
    "cross_seed_component_flag",
    "from_seed",
    "from_semantic",
    "from_2hop",
    "from_component",
    "from_rare_artifact",
)

BRIDGE_RETRIEVAL_COLS: tuple[str, ...] = (
    "retrieval_channels",
    "n_retrieval_channels",
    "retrieval_semantic_cosine",
    "retrieval_semantic_rank",
    "retrieval_body_only_token_jaccard",
    "retrieval_body_only_rank",
    "retrieval_path_token_jaccard",
    "retrieval_path_rank",
)

BRIDGE_LATENT_COLS: tuple[str, ...] = (
    "scorer_encoder_cosine",
    "scorer_encoder_l2_distance",
    "scorer_encoder_dot_product",
    "scorer_encoder_backend",
    "embedding_cosine_subj_body",
    "embedding_l2_distance_subj_body",
    "embedding_dot_product_subj_body",
    "graph_email_x_cosine",
    "graph_email_x_l2_distance",
    "gnn_encoder_cosine",
    "gnn_encoder_l2_distance",
)

SUSPICIOUS_HIGH_SCORE_THRESHOLD: float = 0.9
SUSPICIOUS_WEAK_SEMANTIC: float = 0.5
SUSPICIOUS_WEAK_BODY: float = 0.25
SUSPICIOUS_WEAK_PATH: float = 0.05
SUSPICIOUS_WEAK_LATENT: float = 0.5

BRIDGE_GRAPH_CONTEXT_COLS: tuple[str, ...] = (
    "already_in_candidate_graph",
    "same_seed_component_flag",
    "cross_seed_component_flag",
    "email_i_seed_component_id",
    "email_j_seed_component_id",
    "same_predicted_component",
    "email_graph_common_neighbor_count",
)

BRIDGE_CARD_METRIC_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "Retrieval provenance",
        (
            "n_retrieval_channels",
            "retrieval_semantic_cosine",
            "retrieval_semantic_rank",
            "retrieval_body_only_token_jaccard",
            "retrieval_body_only_rank",
            "retrieval_path_token_jaccard",
            "retrieval_path_rank",
        ),
    ),
    (
        "Explicit scorer features",
        BRIDGE_EXPLICIT_FEATURE_COLS,
    ),
    (
        "Latent / embedding diagnostics (scorer + static embeddings)",
        (
            "scorer_encoder_backend",
            "scorer_encoder_cosine",
            "scorer_encoder_l2_distance",
            "scorer_encoder_dot_product",
            "embedding_cosine_subj_body",
            "embedding_l2_distance_subj_body",
            "graph_email_x_cosine",
            "graph_email_x_l2_distance",
        ),
    ),
    (
        "Graph context",
        BRIDGE_GRAPH_CONTEXT_COLS,
    ),
)


def _cosine_l2_dot(a: np.ndarray, b: np.ndarray) -> tuple[float | None, float | None, float | None]:
    if a.shape != b.shape or a.size == 0:
        return None, None, None
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na <= 0 or nb <= 0:
        return None, None, None
    cos = float(np.dot(a, b) / (na * nb))
    l2 = float(np.linalg.norm(a - b))
    dot = float(np.dot(a, b))
    return cos, l2, dot


def load_misp_node_sets_by_email(
    *,
    project_root: Path,
    misp_json_path: Path | None = None,
) -> tuple[dict[str, dict[str, set[str]]], dict[str, Any]]:
    """Build anchor-compatible node artifact sets from MISP lake JSON (fallback when no graph bundle)."""
    from seed_candidate_workflow.utils.pair_score_separation import _resolve_default_misp_json_path

    try:
        from analysis.scripts.misp_email_text_catalog import load_misp_events_list
        from core.feature_set_extraction.url_extraction_utils import (
            parse_url_host_and_registrable_domain,
        )
        from graph.common import parse_misp_events
        from preprocessing.utils.url_extractor import parse_url_components
    except Exception as exc:
        return {}, {"available": False, "reason": f"misp_import_failed:{exc}"}

    misp_path = Path(misp_json_path) if misp_json_path else _resolve_default_misp_json_path(project_root)
    if not misp_path.is_file():
        return {}, {"available": False, "reason": f"misp_json_not_found:{misp_path}"}

    events = load_misp_events_list(misp_path)
    parsed = parse_misp_events(events)
    out: dict[str, dict[str, set[str]]] = {}
    for ev in parsed:
        eid = str(ev.get("external_id") or "").strip()
        if not eid:
            continue
        senders = {str(s).strip().lower() for s in (ev.get("senders") or []) if str(s).strip()}
        urls = {str(u).strip() for u in (ev.get("urls") or []) if str(u).strip()}
        attachments = {str(a).strip() for a in (ev.get("attachments") or []) if str(a).strip()}
        sender_domains: set[str] = set()
        registrable_domains: set[str] = set()
        stems: set[str] = set()
        for s in senders:
            if "@" in s:
                sender_domains.add(s.split("@", 1)[1].lower())
        for u in urls:
            try:
                _host, reg, reg_ok = parse_url_host_and_registrable_domain(u)
                if reg_ok and reg:
                    registrable_domains.add(str(reg).lower())
                comp = parse_url_components(u)
                stem = str(comp.get("stem") or "").strip()
                if stem and stem != "/":
                    stems.add(stem)
            except Exception:
                continue
        out[eid] = {
            "sender_set": senders,
            "url_set": urls,
            "attachment_set": attachments,
            "sender_email_domain_set": sender_domains,
            "domain_set": registrable_domains,
            "stem_set": stems,
            "html_structure_fingerprint_set": set(),
            "received_host_set": set(),
        }
    return out, {
        "available": bool(out),
        "source": "misp_parsed",
        "misp_json_path": str(misp_path.resolve()),
        "n_emails": int(len(out)),
    }


def resolve_bridge_nodes_by_email(
    *,
    project_root: Path,
    run_dir: Path,
    graph_pt: Path,
    pair_csv: Path | None,
    candidate_union_csv: Path | None,
) -> tuple[dict[str, dict[str, set[str]]], dict[str, Any], Path | None]:
    """Load anchor node sets; try run_id and hetero-graph stem graph bundles."""
    from seed_candidate_workflow.utils.pair_training_dataset_helpers import (
        _load_anchor_node_sets_by_email,
    )

    graph_stem = Path(graph_pt).stem.replace("_hetero", "")
    graph_ids = []
    for gid in (run_dir.name, graph_stem):
        if gid and gid not in graph_ids:
            graph_ids.append(gid)

    tried: list[str] = []
    for gid in graph_ids:
        cand = (
            project_root
            / "seed_candidate_workflow"
            / "output"
            / "graph_bundles"
            / gid
            / "candidate"
            / gid
            / "candidate_union.csv"
        )
        if not cand.is_file():
            tried.append(str(cand))
            continue
        nodes, meta = _load_anchor_node_sets_by_email(
            candidate_union_csv=cand,
            graph_id=gid,
            project_root=project_root,
        )
        if nodes:
            meta = dict(meta or {})
            meta["resolved_graph_id"] = gid
            meta["candidate_union_csv"] = str(cand)
            return nodes, meta, cand

    if candidate_union_csv is not None and Path(candidate_union_csv).is_file():
        nodes, meta = _load_anchor_node_sets_by_email(
            candidate_union_csv=Path(candidate_union_csv),
            graph_id=run_dir.name,
            project_root=project_root,
        )
        if nodes:
            return nodes, meta or {}, Path(candidate_union_csv)

    misp_nodes, misp_meta = load_misp_node_sets_by_email(project_root=project_root)
    if misp_nodes:
        misp_meta = dict(misp_meta)
        misp_meta["fallback"] = "misp_parsed_after_anchor_bundle_missing"
        misp_meta["tried_anchor_paths"] = tried
        return misp_nodes, misp_meta, None

    return {}, {"available": False, "tried_paths": tried, "misp_fallback": misp_meta}, None


def _build_shared_evidence_lines(row: pd.Series) -> tuple[str, str, str]:
    """direct_shared_evidence_lines, admitting_evidence_lines, shared_artifacts_brief."""
    direct: list[str] = []
    for _ac, short in _PAIR_SHARED_CHANNEL_DEFS:
        if bool(row.get(f"has_shared_{short}")):
            vals = str(row.get(f"shared_{short}_values") or "").strip()
            cnt = row.get(f"shared_{short}_count")
            direct.append(f"{short}: {vals}" if vals else f"{short} (count={cnt})")
    direct_txt = "\n".join(direct) if direct else "(no shared artifacts on anchor nodes)"
    brief = "; ".join(direct[:6]) if direct else "none"
    admit = str(row.get("admitting_evidence_lines") or "").strip()
    if not admit:
        admit = "(no candidate-family admitting rows — bridge pair was not in union)"
    return direct_txt, admit, brief


def _attach_shared_evidence_columns(
    df: pd.DataFrame,
    *,
    nodes_by_email: dict[str, dict[str, set[str]]],
) -> pd.DataFrame:
    if df.empty or not nodes_by_email:
        out = df.copy()
        out["anchor_context_missing"] = True
        for _ac, short in _PAIR_SHARED_CHANNEL_DEFS:
            out[f"has_shared_{short}"] = False
            out[f"shared_{short}_count"] = 0
            out[f"shared_{short}_values"] = ""
        out["n_shared_core_channels"] = 0
        out["direct_shared_evidence_lines"] = "(anchor node context unavailable)"
        out["admitting_evidence_lines"] = ""
        out["shared_artifacts_brief"] = "none"
        return out

    rows_out: list[dict[str, Any]] = []
    for _, r in df.iterrows():
        rec = dict(r)
        detail = _pair_shared_evidence_detail(str(r["email_i"]), str(r["email_j"]), nodes_by_email)
        rec.update(detail)
        dlines, alines, brief = _build_shared_evidence_lines(pd.Series(rec))
        rec["direct_shared_evidence_lines"] = dlines
        rec["admitting_evidence_lines"] = alines
        rec["shared_artifacts_brief"] = brief
        rows_out.append(rec)
    return pd.DataFrame(rows_out)


def _attach_embedding_diagnostics(
    df: pd.DataFrame,
    *,
    id_to_emb: dict[str, np.ndarray],
) -> pd.DataFrame:
    out = df.copy()
    cos_l: list[float | None] = []
    l2_l: list[float | None] = []
    dot_l: list[float | None] = []
    for _, r in out.iterrows():
        vi = id_to_emb.get(str(r["email_i"]))
        vj = id_to_emb.get(str(r["email_j"]))
        if vi is None or vj is None:
            cos_l.append(None)
            l2_l.append(None)
            dot_l.append(None)
        else:
            c, l2, d = _cosine_l2_dot(vi, vj)
            cos_l.append(c)
            l2_l.append(l2)
            dot_l.append(d)
    out["embedding_cosine_subj_body"] = cos_l
    out["embedding_l2_distance_subj_body"] = l2_l
    out["embedding_dot_product_subj_body"] = dot_l
    return out


def _attach_graph_email_x_diagnostics(
    df: pd.DataFrame,
    *,
    graph_pt: Path,
    to_undirected: bool,
) -> pd.DataFrame:
    from seed_candidate_workflow.utils import graph_structure_helpers as gh

    out = df.copy()
    try:
        data = gh.load_hetero(graph_pt, to_undirected=to_undirected)
        x = data["email"].x.detach().cpu().numpy().astype(np.float64)
    except Exception:
        out["graph_email_x_cosine"] = np.nan
        out["graph_email_x_l2_distance"] = np.nan
        return out

    gi = pd.to_numeric(out["graph_email_idx_i"], errors="coerce")
    gj = pd.to_numeric(out["graph_email_idx_j"], errors="coerce")
    cos_vals: list[float | None] = []
    l2_vals: list[float | None] = []
    for i, j in zip(gi, gj, strict=False):
        if pd.isna(i) or pd.isna(j):
            cos_vals.append(None)
            l2_vals.append(None)
            continue
        ii, jj = int(i), int(j)
        if ii < 0 or jj < 0 or ii >= x.shape[0] or jj >= x.shape[0]:
            cos_vals.append(None)
            l2_vals.append(None)
            continue
        c, l2, _ = _cosine_l2_dot(x[ii], x[jj])
        cos_vals.append(c)
        l2_vals.append(l2)
    out["graph_email_x_cosine"] = cos_vals
    out["graph_email_x_l2_distance"] = l2_vals
    return out


def _attach_time_gap_from_text_catalog(
    df: pd.DataFrame,
    *,
    text_catalog: dict[str, dict[str, str]],
) -> pd.DataFrame:
    """Compute time_gap_seconds_min from MISP timestamps when column missing or all-null."""
    out = df.copy()
    existing = pd.to_numeric(out.get("time_gap_seconds_min"), errors="coerce")
    if existing.notna().any():
        out["time_gap_source"] = np.where(existing.notna(), "upstream", "unavailable")
        return out

    try:
        from graph.common import to_unix_ts
    except Exception:
        out["time_gap_seconds_min"] = np.nan
        out["time_gap_source"] = "unavailable"
        return out

    gaps: list[float | None] = []
    for _, r in out.iterrows():
        ti = text_catalog.get(str(r["email_i"]), {}) or {}
        tj = text_catalog.get(str(r["email_j"]), {}) or {}
        ts_i = to_unix_ts(ti.get("date_raw") or ti.get("timestamp_utc"))
        ts_j = to_unix_ts(tj.get("date_raw") or tj.get("timestamp_utc"))
        if ts_i is None or ts_j is None:
            gaps.append(None)
        else:
            gaps.append(float(abs(int(ts_i) - int(ts_j))))
    out["time_gap_seconds_min"] = gaps
    out["time_gap_source"] = np.where(pd.to_numeric(out["time_gap_seconds_min"], errors="coerce").notna(), "misp_timestamps", "unavailable")
    return out


def _attach_path_features_from_nodes(
    df: pd.DataFrame,
    *,
    nodes_by_email: dict[str, dict[str, set[str]]],
) -> pd.DataFrame:
    if df.empty or not nodes_by_email:
        return df
    from seed_candidate_workflow.utils.pair_similarity_features import attach_path_jaccard_features_to_dataframe

    return attach_path_jaccard_features_to_dataframe(
        df, nodes_by_email=nodes_by_email, prefer_existing=False
    )


def _attach_scorer_encoder_diagnostics(
    df: pd.DataFrame,
    *,
    run_dir: Path,
    graph_pt: Path,
    device: str,
    checkpoint_name: str,
    to_undirected: bool,
    max_gnn_pairs: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Scorer-side latent similarity (vectors the pair MLP uses before the final nonlinearity).

    For ``mlp_raw_email_x`` backend this matches ``graph_email.x`` cosine/L2 on all rows.
    For GNN backend, computes encoder output cosine/L2 on up to ``max_gnn_pairs`` highest-score rows.
    """
    out = df.copy()
    meta: dict[str, Any] = {"status": "skipped"}
    for c in (
        "scorer_encoder_cosine",
        "scorer_encoder_l2_distance",
        "scorer_encoder_dot_product",
        "gnn_encoder_cosine",
        "gnn_encoder_l2_distance",
    ):
        if c not in out.columns:
            out[c] = np.nan

    if "graph_email_x_cosine" not in out.columns:
        out = _attach_graph_email_x_diagnostics(out, graph_pt=graph_pt, to_undirected=to_undirected)

    try:
        from seed_candidate_workflow.utils.pair_model_inference import load_pair_supervision_for_inference
        from src.pair_graph_sampling import sample_hetero_around_pair_endpoints

        ctx = load_pair_supervision_for_inference(
            run_dir=run_dir,
            graph_pt=graph_pt,
            checkpoint_name=checkpoint_name,
            device=device,
            to_undirected=to_undirected,
        )
        backend = str(ctx.get("pair_encoder_backend") or "unknown")
        out["scorer_encoder_backend"] = backend
        meta["pair_encoder_backend"] = backend

        if backend == "mlp_raw_email_x" or ctx.get("model") is None:
            out["scorer_encoder_cosine"] = pd.to_numeric(out["graph_email_x_cosine"], errors="coerce")
            out["scorer_encoder_l2_distance"] = pd.to_numeric(out["graph_email_x_l2_distance"], errors="coerce")
            gi = pd.to_numeric(out["graph_email_idx_i"], errors="coerce")
            gj = pd.to_numeric(out["graph_email_idx_j"], errors="coerce")
            dots: list[float | None] = []
            x = None
            try:
                from seed_candidate_workflow.utils import graph_structure_helpers as gh

                data = gh.load_hetero(graph_pt, to_undirected=to_undirected)
                x = data["email"].x.detach().cpu().numpy().astype(np.float64)
            except Exception:
                x = None
            for i, j in zip(gi, gj, strict=False):
                if x is None or pd.isna(i) or pd.isna(j):
                    dots.append(None)
                    continue
                ii, jj = int(i), int(j)
                if ii < 0 or jj < 0 or ii >= x.shape[0] or jj >= x.shape[0]:
                    dots.append(None)
                else:
                    dots.append(float(np.dot(x[ii], x[jj])))
            out["scorer_encoder_dot_product"] = dots
            meta["status"] = "ok"
            meta["note"] = "scorer_encoder mirrors graph email.x (mlp_raw_email_x backend)"
            return out, meta

        import torch

        model = ctx["model"]
        model.eval()
        dev = ctx["device"]
        fanout = ctx["fanout"]
        data_cpu = ctx["data_cpu"]
        gi = pd.to_numeric(out["graph_email_idx_i"], errors="coerce")
        gj = pd.to_numeric(out["graph_email_idx_j"], errors="coerce")
        ok = gi.notna() & gj.notna()
        sub_idx = out.loc[ok].sort_values("score", ascending=False).head(int(max_gnn_pairs)).index
        bs = 256
        sub = out.loc[sub_idx].copy()
        sub["_row"] = np.arange(len(sub), dtype=np.int64)
        for start in range(0, len(sub), bs):
            chunk = sub.iloc[start : start + bs]
            gii = chunk["graph_email_idx_i"].astype(int).to_numpy()
            gjj = chunk["graph_email_idx_j"].astype(int).to_numpy()
            sample = sample_hetero_around_pair_endpoints(data_cpu, gii, gjj, fanout)
            hetero_batch = sample.hetero_batch.to(dev)
            with torch.no_grad():
                h = model(hetero_batch.x_dict, hetero_batch.edge_index_dict)
                z_all = h["email"]
                li = sample.pair_local_i.to(dev).clamp(min=0)
                lj = sample.pair_local_j.to(dev).clamp(min=0)
                zi = z_all[li].detach().cpu().numpy()
                zj = z_all[lj].detach().cpu().numpy()
            for k, idx in enumerate(chunk.index):
                if not bool(sample.pair_ok_mask[k]):
                    continue
                c, l2, dot = _cosine_l2_dot(zi[k], zj[k])
                if c is not None:
                    out.at[idx, "scorer_encoder_cosine"] = c
                    out.at[idx, "gnn_encoder_cosine"] = c
                if l2 is not None:
                    out.at[idx, "scorer_encoder_l2_distance"] = l2
                    out.at[idx, "gnn_encoder_l2_distance"] = l2
                if dot is not None:
                    out.at[idx, "scorer_encoder_dot_product"] = dot
        meta["status"] = "ok"
        meta["n_gnn_encoder_pairs_computed"] = int(len(sub_idx))
        meta["note"] = "gnn encoder cosine/L2 on highest-score pairs (see max_gnn_pairs)"
        return out, meta
    except Exception as exc:
        meta["status"] = "error"
        meta["error"] = str(exc)
        out["scorer_encoder_backend"] = "unknown"
        return out, meta


def _fill_bridge_display_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Backfill scorer/display columns that are often NaN on bridge rows (non-union pairs)."""
    out = df.copy()
    if "semantic_cosine_max" not in out.columns:
        out["semantic_cosine_max"] = np.nan
    sem_max = pd.to_numeric(out["semantic_cosine_max"], errors="coerce")
    if "retrieval_semantic_cosine" in out.columns:
        retr = pd.to_numeric(out["retrieval_semantic_cosine"], errors="coerce")
        fill = sem_max.isna() & retr.notna()
        out.loc[fill, "semantic_cosine_max"] = retr.loc[fill]
    if "embedding_cosine_subj_body" in out.columns:
        emb = pd.to_numeric(out["embedding_cosine_subj_body"], errors="coerce")
        fill2 = pd.to_numeric(out["semantic_cosine_max"], errors="coerce").isna() & emb.notna()
        out.loc[fill2, "semantic_cosine_max"] = emb.loc[fill2]
    if "semantic_cosine_for_display" in out.columns:
        disp = pd.to_numeric(out["semantic_cosine_for_display"], errors="coerce")
        fill3 = disp.isna() & pd.to_numeric(out["semantic_cosine_max"], errors="coerce").notna()
        out.loc[fill3, "semantic_cosine_for_display"] = out.loc[fill3, "semantic_cosine_max"]
    elif "semantic_cosine_max" in out.columns:
        out["semantic_cosine_for_display"] = pd.to_numeric(out["semantic_cosine_max"], errors="coerce")
    return out


def _attach_gt_campaign_columns(
    df: pd.DataFrame,
    *,
    label_map: dict[str, Any],
) -> pd.DataFrame:
    out = df.copy()
    out["gt_campaign_i"] = out["email_i"].astype(str).map(lambda x: label_map.get(x))
    out["gt_campaign_j"] = out["email_j"].astype(str).map(lambda x: label_map.get(x))
    return out


def _build_candidate_neighbor_index(
    candidate_union_csv: Path | None,
) -> dict[str, set[str]]:
    if candidate_union_csv is None or not Path(candidate_union_csv).is_file():
        return {}
    adj: dict[str, set[str]] = {}
    df = pd.read_csv(candidate_union_csv, usecols=["email_i", "email_j"], low_memory=False)
    for ei, ej in zip(df["email_i"].astype(str), df["email_j"].astype(str), strict=False):
        adj.setdefault(ei, set()).add(ej)
        adj.setdefault(ej, set()).add(ei)
    return adj


def _load_email_component_ids(
    *,
    project_root: Path,
    run_dir: Path,
    candidate_union_csv: Path | None,
    graph_id: str,
) -> dict[str, int]:
    from seed_candidate_workflow.utils.anchor_graph_helpers import load_anchor_graph_artifacts
    from seed_candidate_workflow.utils.pair_training_dataset_helpers import _infer_anchor_run_dir

    cand = candidate_union_csv
    if cand is None:
        return {}
    run_anchor = _infer_anchor_run_dir(
        candidate_union_csv=Path(cand),
        graph_id=graph_id,
        project_root=project_root,
    )
    if run_anchor is None:
        run_anchor = run_dir
    try:
        members_df, _, _, _, _ = load_anchor_graph_artifacts(run_anchor, load_graph_pickle=False)
    except Exception:
        return {}
    if not {"external_id", "component_id"}.issubset(members_df.columns):
        return {}
    out: dict[str, int] = {}
    for eid, cid in zip(
        members_df["external_id"].astype(str),
        pd.to_numeric(members_df["component_id"], errors="coerce"),
        strict=False,
    ):
        if pd.notna(cid):
            out[eid] = int(cid)
    return out


def _attach_graph_context_columns(
    df: pd.DataFrame,
    *,
    connected: set[tuple[str, str]],
    candidate_union_csv: Path | None,
    component_by_email: dict[str, int],
    neighbor_index: dict[str, set[str]],
) -> pd.DataFrame:
    out = df.copy()
    already: list[bool] = []
    cn: list[int] = []
    comp_i: list[int | None] = []
    comp_j: list[int | None] = []
    same_comp: list[bool | None] = []
    for _, r in out.iterrows():
        ei, ej = str(r["email_i"]), str(r["email_j"])
        pk = canonical_pair(ei, ej)
        already.append(pk in connected if pk is not None else False)
        ni = neighbor_index.get(ei) or set()
        nj = neighbor_index.get(ej) or set()
        cn.append(int(len(ni & nj)))
        ci = component_by_email.get(ei)
        cj = component_by_email.get(ej)
        comp_i.append(ci)
        comp_j.append(cj)
        if ci is not None and cj is not None:
            same_comp.append(ci == cj)
        else:
            same_comp.append(None)
    out["already_in_candidate_graph"] = already
    out["email_graph_common_neighbor_count"] = cn
    out["email_i_seed_component_id"] = comp_i
    out["email_j_seed_component_id"] = comp_j
    out["same_predicted_component"] = same_comp
    if "same_seed_component_flag" not in out.columns:
        out["same_seed_component_flag"] = [s is True for s in same_comp]
    if "cross_seed_component_flag" not in out.columns:
        out["cross_seed_component_flag"] = [
            (s is False) if s is not None else False for s in same_comp
        ]
    return out


def enrich_bridge_dataframe_for_review(
    df: pd.DataFrame,
    *,
    project_root: Path,
    run_dir: Path,
    graph_pt: Path,
    connected: set[tuple[str, str]] | None = None,
    candidate_union_csv: Path | None,
    pair_csv: Path | None,
    id_to_emb: dict[str, np.ndarray] | None = None,
    to_undirected: bool = True,
    compute_gnn_latent_max_rows: int = 5000,
    device: str = "cpu",
    checkpoint_name: str = "best_model.pt",
    misp_json_path: Path | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Full explainability columns for bridge CSV + HTML."""
    out = df.copy()
    nodes, nodes_meta, resolved_cand = resolve_bridge_nodes_by_email(
        project_root=project_root,
        run_dir=run_dir,
        graph_pt=graph_pt,
        pair_csv=pair_csv,
        candidate_union_csv=candidate_union_csv,
    )
    cand_path = Path(candidate_union_csv) if candidate_union_csv else resolved_cand
    neighbor_index = _build_candidate_neighbor_index(cand_path)
    comp_map = _load_email_component_ids(
        project_root=project_root,
        run_dir=run_dir,
        candidate_union_csv=cand_path,
        graph_id=run_dir.name,
    )
    out = _attach_graph_context_columns(
        out,
        connected=connected or set(),
        candidate_union_csv=cand_path,
        component_by_email=comp_map,
        neighbor_index=neighbor_index,
    )
    out = _attach_shared_evidence_columns(out, nodes_by_email=nodes)
    if nodes:
        out = _attach_path_features_from_nodes(out, nodes_by_email=nodes)

    text_catalog: dict[str, dict[str, str]] = {}
    try:
        from seed_candidate_workflow.utils.pair_similarity_features import load_misp_text_catalog_for_pairs

        text_catalog, _cat_meta = load_misp_text_catalog_for_pairs(
            project_root=project_root,
            misp_json_path=misp_json_path,
        )
    except Exception:
        text_catalog = {}
    out = _attach_time_gap_from_text_catalog(out, text_catalog=text_catalog)

    emb = id_to_emb or {}
    if not emb:
        ep = _resolve_embeddings_json_for_review(project_root)
        if ep and ep.is_file():
            from seed_candidate_workflow.utils.bridge_candidate_experiment import (
                _load_embeddings_by_external_id,
            )

            emb = _load_embeddings_by_external_id(ep)
    out = _attach_embedding_diagnostics(out, id_to_emb=emb)
    out = _attach_graph_email_x_diagnostics(out, graph_pt=graph_pt, to_undirected=to_undirected)
    out, scorer_meta = _attach_scorer_encoder_diagnostics(
        out,
        run_dir=run_dir,
        graph_pt=graph_pt,
        device=device,
        checkpoint_name=checkpoint_name,
        to_undirected=to_undirected,
        max_gnn_pairs=int(compute_gnn_latent_max_rows),
    )

    out = _inject_semantic_cosine_for_manual_review(out)
    out = _fill_bridge_display_feature_columns(out)
    if "semantic_cosine" not in out.columns and "semantic_cosine_for_display" in out.columns:
        out["semantic_cosine"] = out["semantic_cosine_for_display"]

    for col in BRIDGE_EXPLICIT_FEATURE_COLS:
        if col not in out.columns:
            out[col] = np.nan
    for col in BRIDGE_RETRIEVAL_COLS:
        if col not in out.columns:
            out[col] = np.nan

    meta = {
        "nodes_meta": nodes_meta,
        "resolved_candidate_union_csv": str(resolved_cand) if resolved_cand else None,
        "n_with_anchor_context": int((out.get("anchor_context_missing") == False).sum())  # noqa: E712
        if "anchor_context_missing" in out.columns
        else 0,
        "scorer_encoder_meta": scorer_meta,
        "text_catalog_n": int(len(text_catalog)),
    }
    return out, meta


def bridge_feature_population_diagnostics(df: pd.DataFrame) -> dict[str, Any]:
    n = int(len(df))

    def _pop(cols: tuple[str, ...]) -> dict[str, Any]:
        block: dict[str, Any] = {}
        for c in cols:
            if c not in df.columns:
                block[c] = {"present": False, "n_non_null": 0}
            else:
                s = df[c]
                if s.dtype == bool or c.startswith("has_shared_") or c.endswith("_flag"):
                    nn = int(s.fillna(False).astype(bool).sum())
                elif c in ("retrieval_channels", "direct_shared_evidence_lines", "shared_artifacts_brief"):
                    nn = int(s.astype(str).str.strip().replace("nan", "").ne("").sum())
                else:
                    nn = int(pd.to_numeric(s, errors="coerce").notna().sum())
                block[c] = {
                    "present": True,
                    "n_non_null": nn,
                    "fraction_non_null": float(nn / n) if n else None,
                }
        return block

    return {
        "n_bridge_pairs": n,
        "shared_artifact_columns": _pop(
            tuple(f"has_shared_{s}" for _a, s in _PAIR_SHARED_CHANNEL_DEFS)
            + tuple(f"shared_{s}_values" for _a, s in _PAIR_SHARED_CHANNEL_DEFS)
        ),
        "retrieval_provenance_columns": _pop(BRIDGE_RETRIEVAL_COLS),
        "explicit_scorer_feature_columns": _pop(BRIDGE_EXPLICIT_FEATURE_COLS),
        "latent_embedding_columns": _pop(BRIDGE_LATENT_COLS),
        "time_gap_columns": _pop(("time_gap_seconds_min", "time_gap_source")),
        "graph_context_columns": _pop(BRIDGE_GRAPH_CONTEXT_COLS),
        "review_line_columns": _pop(
            ("direct_shared_evidence_lines", "shared_artifacts_brief", "semantic_cosine_for_display")
        ),
    }


def _score_band(score: float | None, *, high: float = 0.9, mid_lo: float = 0.5, low: float = 0.2) -> str:
    if score is None or not np.isfinite(score):
        return "unscored"
    s = float(score)
    if s >= high:
        return "high_bridge"
    if s > low and s < high:
        return "mid_bridge" if s >= mid_lo else "low_bridge"
    if s <= low:
        return "low_bridge"
    return "mid_bridge"


def build_bridge_band_analysis(
    df: pd.DataFrame,
    *,
    label_map: dict[str, Any] | None = None,
) -> dict[str, Any]:
    out = df.copy()
    scores = pd.to_numeric(out["score"], errors="coerce")
    out["bridge_score_band"] = [_score_band(float(s) if pd.notna(s) else None) for s in scores]

    if label_map:
        rels: list[str | None] = []
        for _, r in out.iterrows():
            ci = label_map.get(str(r["email_i"]))
            cj = label_map.get(str(r["email_j"]))
            if ci is None or cj is None:
                rels.append(None)
            elif ci == cj:
                rels.append("same_campaign")
            else:
                rels.append("cross_campaign")
        out["gt_relation"] = rels

    compare_cols = list(BRIDGE_RETRIEVAL_COLS) + list(BRIDGE_EXPLICIT_FEATURE_COLS) + list(BRIDGE_LATENT_COLS)
    compare_cols += [
        "has_shared_sender",
        "has_shared_stem",
        "has_shared_url",
        "has_shared_html_fp",
        "n_shared_core_channels",
    ]

    cohorts: dict[str, Any] = {}
    for band in ("high_bridge", "mid_bridge", "low_bridge"):
        sub = out.loc[out["bridge_score_band"] == band]
        if sub.empty:
            cohorts[band] = {"n_pairs": 0}
            continue
        block: dict[str, Any] = {
            "n_pairs": int(len(sub)),
            "score_mean": float(pd.to_numeric(sub["score"], errors="coerce").mean()),
            "retrieval_channel_counts": sub["retrieval_channels"].astype(str).value_counts().head(15).to_dict()
            if "retrieval_channels" in sub.columns
            else {},
            "n_retrieval_channels_mean": float(sub["n_retrieval_channels"].mean())
            if "n_retrieval_channels" in sub.columns
            else None,
        }
        for c in compare_cols:
            if c not in sub.columns:
                continue
            s = pd.to_numeric(sub[c], errors="coerce")
            if s.notna().any():
                block[f"mean_{c}"] = float(s.mean())
        if label_map and "gt_relation" in sub.columns:
            cov = sub["gt_relation"].notna()
            block["n_gt_covered"] = int(cov.sum())
            block["gt_same_fraction"] = float(
                (sub.loc[cov, "gt_relation"] == "same_campaign").mean()
            ) if cov.any() else None
            block["gt_cross_fraction"] = float(
                (sub.loc[cov, "gt_relation"] == "cross_campaign").mean()
            ) if cov.any() else None
            if "retrieval_channels" in sub.columns and cov.any():
                gt_sub = sub.loc[cov]
                by_ch: dict[str, dict[str, float | int]] = {}
                for ch, grp in gt_sub.groupby(gt_sub["retrieval_channels"].astype(str), dropna=False):
                    by_ch[str(ch)] = {
                        "n": int(len(grp)),
                        "gt_same_fraction": float((grp["gt_relation"] == "same_campaign").mean()),
                        "gt_cross_fraction": float((grp["gt_relation"] == "cross_campaign").mean()),
                    }
                block["gt_by_retrieval_channel"] = by_ch
            if "n_retrieval_channels" in sub.columns and cov.any():
                gt_sub = sub.loc[cov]
                by_n: dict[str, dict[str, float | int]] = {}
                for nch, grp in gt_sub.groupby("n_retrieval_channels", dropna=False):
                    by_n[str(int(nch))] = {
                        "n": int(len(grp)),
                        "gt_same_fraction": float((grp["gt_relation"] == "same_campaign").mean()),
                    }
                block["gt_by_n_retrieval_channels"] = by_n
        cohorts[band] = block

    return {
        "band_definitions": {
            "high_bridge": "score >= 0.90",
            "mid_bridge": "0.50 <= score < 0.90",
            "low_bridge": "score <= 0.20",
        },
        "cohorts": cohorts,
        "retrieval_channel_counts_all": out["retrieval_channels"].astype(str).value_counts().head(25).to_dict()
        if "retrieval_channels" in out.columns
        else {},
    }


def _explicit_feature_mass(row: pd.Series) -> dict[str, float | bool]:
    sem = pd.to_numeric(row.get("semantic_cosine_max"), errors="coerce")
    body = pd.to_numeric(row.get("body_token_jaccard"), errors="coerce")
    path = pd.to_numeric(row.get("path_token_jaccard_combined"), errors="coerce")
    n_core = int(pd.to_numeric(row.get("n_shared_core_channels"), errors="coerce") or 0)
    has_shared = any(bool(row.get(f"has_shared_{s}")) for _a, s in _PAIR_SHARED_CHANNEL_DEFS)
    return {
        "semantic_ok": bool(pd.notna(sem) and float(sem) >= SUSPICIOUS_WEAK_SEMANTIC),
        "body_ok": bool(pd.notna(body) and float(body) >= SUSPICIOUS_WEAK_BODY),
        "path_ok": bool(pd.notna(path) and float(path) >= SUSPICIOUS_WEAK_PATH),
        "shared_ok": bool(n_core > 0 or has_shared),
        "semantic": float(sem) if pd.notna(sem) else None,
        "body": float(body) if pd.notna(body) else None,
        "path": float(path) if pd.notna(path) else None,
    }


def _latent_feature_mass(row: pd.Series) -> dict[str, float | bool]:
    scorer = pd.to_numeric(row.get("scorer_encoder_cosine"), errors="coerce")
    emb = pd.to_numeric(row.get("embedding_cosine_subj_body"), errors="coerce")
    if pd.isna(scorer):
        scorer = pd.to_numeric(row.get("graph_email_x_cosine"), errors="coerce")
    return {
        "scorer_ok": bool(pd.notna(scorer) and float(scorer) >= SUSPICIOUS_WEAK_LATENT),
        "embedding_ok": bool(pd.notna(emb) and float(emb) >= SUSPICIOUS_WEAK_LATENT),
        "scorer_cosine": float(scorer) if pd.notna(scorer) else None,
        "embedding_cosine": float(emb) if pd.notna(emb) else None,
    }


def build_bridge_suspicious_high_score_analysis(
    df: pd.DataFrame,
    *,
    label_map: dict[str, Any] | None = None,
    score_threshold: float = SUSPICIOUS_HIGH_SCORE_THRESHOLD,
) -> dict[str, Any]:
    """
    High-score bridges with weak explicit features — split into latent-explained vs genuinely suspicious.
    """
    scores = pd.to_numeric(df["score"], errors="coerce")
    hi = df.loc[scores >= float(score_threshold)].copy()
    if hi.empty:
        return {"n_high_score": 0, "cohorts": {}}

    latent_explained: list[dict[str, Any]] = []
    suspicious: list[dict[str, Any]] = []
    for idx, r in hi.iterrows():
        expl = _explicit_feature_mass(r)
        lat = _latent_feature_mass(r)
        weak_explicit = not (expl["semantic_ok"] or expl["body_ok"] or expl["path_ok"] or expl["shared_ok"])
        if not weak_explicit:
            continue
        rec = {
            "email_i": str(r["email_i"]),
            "email_j": str(r["email_j"]),
            "score": float(r["score"]),
            "retrieval_channels": str(r.get("retrieval_channels") or ""),
            "retrieval_semantic_cosine": r.get("retrieval_semantic_cosine"),
            "explicit": expl,
            "latent": lat,
            "gt_relation": None,
        }
        if label_map:
            ci = label_map.get(str(r["email_i"]))
            cj = label_map.get(str(r["email_j"]))
            if ci is not None and cj is not None:
                rec["gt_relation"] = "same_campaign" if ci == cj else "cross_campaign"
        if lat["scorer_ok"] or lat["embedding_ok"]:
            rec["classification"] = "latent_explained_high_score"
            latent_explained.append(rec)
        else:
            rec["classification"] = "suspicious_weak_explicit_and_latent"
            suspicious.append(rec)

    def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
        if not rows:
            return {"n": 0}
        chans: Counter[str] = Counter()
        gt_same = gt_cross = 0
        gt_n = 0
        scorer_vals: list[float] = []
        emb_vals: list[float] = []
        for rec in rows:
            for ch in str(rec.get("retrieval_channels") or "").split("|"):
                if ch:
                    chans[ch] += 1
            if rec.get("gt_relation") == "same_campaign":
                gt_same += 1
                gt_n += 1
            elif rec.get("gt_relation") == "cross_campaign":
                gt_cross += 1
                gt_n += 1
            lat = rec.get("latent") or {}
            if lat.get("scorer_cosine") is not None:
                scorer_vals.append(float(lat["scorer_cosine"]))
            if lat.get("embedding_cosine") is not None:
                emb_vals.append(float(lat["embedding_cosine"]))
        return {
            "n": int(len(rows)),
            "retrieval_channel_counts": dict(chans.most_common(15)),
            "gt_same": int(gt_same),
            "gt_cross": int(gt_cross),
            "gt_n_covered": int(gt_n),
            "mean_scorer_encoder_cosine": float(np.mean(scorer_vals)) if scorer_vals else None,
            "mean_embedding_cosine_subj_body": float(np.mean(emb_vals)) if emb_vals else None,
            "examples": rows[:30],
        }

    return {
        "score_threshold": float(score_threshold),
        "definition_weak_explicit": (
            f"semantic_cosine_max < {SUSPICIOUS_WEAK_SEMANTIC} AND body_token_jaccard < {SUSPICIOUS_WEAK_BODY} "
            f"AND path_token_jaccard_combined < {SUSPICIOUS_WEAK_PATH} AND no shared core artifacts"
        ),
        "definition_latent_explained": f"weak explicit but scorer_encoder_cosine or embedding_cosine_subj_body >= {SUSPICIOUS_WEAK_LATENT}",
        "n_high_score": int(len(hi)),
        "n_weak_explicit_high_score": int(len(latent_explained) + len(suspicious)),
        "latent_explained_high_score": _summarize(latent_explained),
        "suspicious_weak_explicit_and_latent": _summarize(suspicious),
    }


def build_bridge_trustworthiness_recommendation(
    df: pd.DataFrame,
    *,
    band_analysis: dict[str, Any],
    population_diag: dict[str, Any],
    suspicious_analysis: dict[str, Any] | None = None,
) -> dict[str, Any]:
    scores = pd.to_numeric(df["score"], errors="coerce")
    hi = df.loc[scores >= 0.9].copy()
    trustworthy_channels: Counter[str] = Counter()

    if not hi.empty:
        for _, r in hi.iterrows():
            for ch in str(r.get("retrieval_channels") or "").split("|"):
                if ch:
                    trustworthy_channels[ch] += 1

    susp = suspicious_analysis or build_bridge_suspicious_high_score_analysis(df)
    top_ch = [c for c, _ in trustworthy_channels.most_common(5)]
    hi_cohort = (band_analysis.get("cohorts") or {}).get("high_bridge") or {}
    n_true_susp = int((susp.get("suspicious_weak_explicit_and_latent") or {}).get("n") or 0)
    n_latent_ok = int((susp.get("latent_explained_high_score") or {}).get("n") or 0)

    return {
        "A_are_top_scoring_bridges_explainable": (
            "High-score bridges are explainable when retrieval_semantic_cosine / "
            "scorer_encoder_cosine (pair MLP input) or shared artifacts are strong; "
            "see latent_explained vs suspicious cohorts in bridge_candidate_suspicious_high_score_summary.json."
        ),
        "B_trustworthy_retrieval_channels": top_ch or ["semantic", "body_only"],
        "C_typical_high_score_evidence": hi_cohort,
        "D_suspicious_high_score_analysis": susp,
        "n_suspicious_weak_explicit_and_latent": n_true_susp,
        "n_latent_explained_weak_explicit_high_score": n_latent_ok,
        "E_suggested_first_bridge_addition_threshold": (
            "Start with score >= 0.90 AND (scorer_encoder_cosine >= 0.75 OR n_shared_core_channels >= 1 "
            "OR retrieval_semantic_cosine >= 0.80); exclude suspicious_weak_explicit_and_latent cohort; "
            "review bridge_candidates_high_score_for_review.html."
        ),
        "population_diag_summary": {
            "anchor_or_misp_nodes": population_diag.get("shared_artifact_columns", {})
            .get("has_shared_sender", {})
            .get("fraction_non_null"),
            "scorer_encoder_populated": population_diag.get("latent_embedding_columns", {})
            .get("scorer_encoder_cosine", {})
            .get("fraction_non_null"),
        },
    }


def _format_bridge_metric_value(row: pd.Series, col: str, *, prec: int = 3) -> str:
    """Like pair review formatter but labels truly-missing bridge features explicitly."""
    if col == "time_gap_seconds_min" and str(row.get("time_gap_source") or "") == "unavailable":
        return "unavailable"
    if col == "semantic_cosine_max":
        v = pd.to_numeric(row.get("semantic_cosine_max"), errors="coerce")
        if pd.isna(v) and pd.notna(pd.to_numeric(row.get("semantic_cosine_for_display"), errors="coerce")):
            return _format_pair_metric_value(row, "semantic_cosine_for_display", prec=prec)
    val = _format_pair_metric_value(row, col, prec=prec)
    if val == "—" and col in BRIDGE_RETRIEVAL_COLS and "retrieval_channels" in row.index:
        ch = str(row.get("retrieval_channels") or "")
        if col.startswith("retrieval_semantic") and "semantic" not in ch:
            return "n/a (channel)"
        if col.startswith("retrieval_body") and "body_only" not in ch:
            return "n/a (channel)"
        if col.startswith("retrieval_path") and "path" not in ch:
            return "n/a (channel)"
    return val


def _bridge_score_explanation_html(row: pd.Series) -> str:
    score = pd.to_numeric(row.get("score"), errors="coerce")
    if pd.isna(score) or float(score) < 0.85:
        return ""
    expl = _explicit_feature_mass(row)
    lat = _latent_feature_mass(row)
    lines: list[str] = []
    lines.append(f"Retrieval: {row.get('retrieval_channels', 'none')} (n={row.get('n_retrieval_channels', '—')})")
    if expl["shared_ok"]:
        lines.append("Explicit: shared artifacts present")
    elif expl["semantic_ok"] or expl["body_ok"]:
        lines.append("Explicit: semantic/body similarity supports score")
    else:
        lines.append("Explicit: weak shared/body/path signal")
    if lat["scorer_ok"]:
        lines.append(
            f"Latent (scorer input): strong — scorer_encoder_cosine={_format_bridge_metric_value(row, 'scorer_encoder_cosine')} "
            f"backend={row.get('scorer_encoder_backend', '?')}"
        )
    elif lat["embedding_ok"]:
        lines.append(
            f"Latent (static emb): embedding_cosine_subj_body={_format_bridge_metric_value(row, 'embedding_cosine_subj_body')}"
        )
    else:
        lines.append("Latent: weak — inspect suspicious-high-score summary if score ≥ 0.9")
    items = "".join(f"<li>{html.escape(ln)}</li>" for ln in lines)
    return (
        '<section class="bridge-score-explanation"><h4>Why this score?</h4>'
        f"<ul class=\"evidence-list\">{items}</ul></section>"
    )


def _bridge_metric_groups_html(row: pd.Series) -> str:
    blocks: list[str] = []
    for title, cols in BRIDGE_CARD_METRIC_GROUPS:
        chips = "".join(
            f'<span class="metric-chip"><strong>{html.escape(c)}</strong> '
            f"{html.escape(_format_bridge_metric_value(row, c))}</span>"
            for c in cols
        )
        blocks.append(
            f'<div class="metric-group"><div class="metric-group-title">{html.escape(title)}</div>'
            f'<div class="metric-chips">{chips}</div></div>'
        )
    ch = html.escape(str(row.get("retrieval_channels") or "none"))
    blocks.insert(
        0,
        '<div class="metric-group"><div class="metric-group-title">Retrieval channel combo</div>'
        f'<div class="metric-chips"><span class="metric-chip"><code>{ch}</code></span></div></div>',
    )
    return f'<div class="metric-groups">{"".join(blocks)}</div>'


def _bridge_retrieval_section_html(row: pd.Series) -> str:
    lines = [
        f"Channels: {row.get('retrieval_channels', 'none')}",
        f"N channels: {row.get('n_retrieval_channels', '—')}",
        f"Semantic retrieval cos: {_format_bridge_metric_value(row, 'retrieval_semantic_cosine')}",
        f"Semantic rank: {_format_bridge_metric_value(row, 'retrieval_semantic_rank', prec=0)}",
        f"Body-only retrieval jac: {_format_bridge_metric_value(row, 'retrieval_body_only_token_jaccard')}",
        f"Body-only rank: {_format_bridge_metric_value(row, 'retrieval_body_only_rank', prec=0)}",
        f"Path retrieval jac: {_format_bridge_metric_value(row, 'retrieval_path_token_jaccard')}",
        f"Path rank: {_format_bridge_metric_value(row, 'retrieval_path_rank', prec=0)}",
    ]
    items = "".join(f"<li>{html.escape(ln)}</li>" for ln in lines)
    return (
        '<section class="bridge-retrieval"><h4>Retrieval provenance</h4>'
        f"<ul class=\"evidence-list\">{items}</ul></section>"
    )


def _bridge_pair_card_html(
    *,
    pair_idx: int,
    row: pd.Series,
    email_text_by_eid: dict[str, dict[str, str]],
) -> str:
    score_s = _format_bridge_metric_value(row, "score", prec=4)
    gt_rel = html.escape(str(row.get("gt_relation") or "not GT-covered"))
    regime = html.escape(str(row.get("retrieval_channels") or "bridge"))
    tgap = _format_bridge_metric_value(row, "time_gap_seconds_min", prec=0)
    explain_html = _bridge_score_explanation_html(row)
    already = "no (bridge)" if not bool(row.get("already_in_candidate_graph")) else "YES"
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
    shared_lines = str(row.get("direct_shared_evidence_lines") or "").strip()
    shared_block = (
        '<section class="bridge-shared"><h4>Shared evidence / attributes</h4>'
        f'<pre class="shared-pre">{html.escape(shared_lines)}</pre></section>'
    )
    evidence_html = _admitting_evidence_section_html(row)
    retrieval_html = _bridge_retrieval_section_html(row)
    metrics_html = _bridge_metric_groups_html(row)
    filter_val = str(row.get("bridge_review_band") or row.get("retrieval_channels") or "bridge")
    return f"""
    <section class="pair-card regime-bridge" id="pair-{pair_idx}"
      data-filter-value="{html.escape(filter_val)}" data-regime="bridge">
      <header class="pair-header">
        <h2>Bridge pair {pair_idx + 1}</h2>
        <div class="meta-grid meta-grid-core">
          <span><strong>Model score</strong> {score_s}</span>
          <span><strong>GT relation</strong> {gt_rel}</span>
          <span><strong>Already in graph</strong> {already}</span>
          <span><strong>Time gap</strong> {tgap}</span>
          <span><strong>Retrieval</strong> <code>{regime}</code></span>
        </div>
        {explain_html}
        {retrieval_html}
        {shared_block}
        {metrics_html}
        {evidence_html}
        <p class="review-prompt">Bridge candidate (missing edge). Verify retrieval provenance, shared artifacts, explicit features, and latent similarity explain the score.</p>
      </header>
      <div class="pair-columns">
        {pane_i}
        {pane_j}
      </div>
    </section>
    """


def _bridge_review_css() -> str:
    return """
    :root {
      --bg: #0f1419; --panel: #1a2332; --panel2: #243044; --text: #e7ecf3;
      --muted: #9aa8bc; --accent: #6cb6ff; --warn: #f0a020; --border: #334155;
    }
    * { box-sizing: border-box; }
    body { margin: 0; font-family: "Segoe UI", system-ui, sans-serif; background: var(--bg); color: var(--text); line-height: 1.45; }
    .layout { display: grid; grid-template-columns: 280px 1fr; min-height: 100vh; }
    .sidebar { position: sticky; top: 0; height: 100vh; overflow-y: auto; padding: 1rem; background: var(--panel); border-right: 1px solid var(--border); }
    .sidebar h1 { font-size: 1rem; margin: 0 0 0.25rem; }
    .sidebar .subtitle { font-size: 0.8rem; color: var(--muted); margin: 0 0 1rem; }
    .filters { display: flex; flex-wrap: wrap; gap: 0.35rem; margin-bottom: 0.75rem; }
    .filter-btn { font-size: 0.72rem; padding: 0.25rem 0.5rem; border-radius: 4px; border: 1px solid var(--border); background: var(--panel2); color: var(--text); cursor: pointer; }
    .filter-btn.active { border-color: var(--accent); color: var(--accent); }
    .toc { display: flex; flex-direction: column; gap: 0.35rem; }
    .toc-item { display: block; padding: 0.45rem 0.5rem; border-radius: 6px; text-decoration: none; color: var(--text); border: 1px solid transparent; font-size: 0.78rem; }
    .toc-item:hover { border-color: var(--border); background: var(--panel2); }
    .main { padding: 1.25rem 1.5rem 3rem; max-width: 56rem; }
    .pair-card { margin-bottom: 2.5rem; padding-bottom: 2rem; border-bottom: 2px solid var(--border); }
    .pair-card.hidden { display: none; }
    .metric-groups { display: grid; gap: 0.55rem; margin: 0.5rem 0 0.75rem; }
    .metric-group-title { font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.04em; color: var(--muted); margin-bottom: 0.25rem; }
    .metric-chips { display: flex; flex-wrap: wrap; gap: 0.35rem 0.65rem; }
    .metric-chip { font-size: 0.78rem; padding: 0.2rem 0.5rem; background: var(--panel2); border: 1px solid var(--border); border-radius: 4px; color: var(--muted); }
    .metric-chip strong { color: var(--text); font-weight: 600; margin-right: 0.25rem; }
    .bridge-retrieval { margin: 0.5rem 0; padding: 0.55rem 0.75rem; background: #141c28; border: 1px solid var(--border); border-radius: 6px; }
    .bridge-retrieval h4, .bridge-shared h4, .bridge-score-explanation h4 { margin: 0 0 0.35rem; font-size: 0.85rem; color: var(--accent); }
    .bridge-score-explanation { margin: 0.5rem 0; padding: 0.55rem 0.75rem; background: #1a2a1e; border: 1px solid #2d5a3a; border-radius: 6px; }
    .bridge-shared { margin: 0.5rem 0; padding: 0.55rem 0.75rem; background: #141c28; border: 1px solid var(--border); border-radius: 6px; }
    .shared-pre { margin: 0; white-space: pre-wrap; font-size: 0.8rem; color: var(--text); font-family: inherit; }
    .meta-grid { display: flex; flex-wrap: wrap; gap: 0.5rem 1.25rem; font-size: 0.85rem; color: var(--muted); margin-bottom: 0.5rem; }
    .meta-grid strong { color: var(--text); }
    .pair-columns { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }
    @media (max-width: 1100px) { .layout { grid-template-columns: 1fr; } .pair-columns { grid-template-columns: 1fr; } }
    """


def write_bridge_review_html(
    df: pd.DataFrame,
    *,
    out_path: Path,
    email_text_by_eid: dict[str, dict[str, str]],
    title: str,
    subtitle: str,
) -> None:
    """Write bridge-specific review HTML with grouped explainability sections."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if df.empty:
        out_path.write_text(
            f"<!DOCTYPE html><html><head><meta charset='utf-8'/><title>{html.escape(title)}</title></head>"
            f"<body><h1>{html.escape(title)}</h1><p>No pairs.</p></body></html>",
            encoding="utf-8",
        )
        return

    review = df.copy()
    if "bridge_review_band" not in review.columns:
        scores = pd.to_numeric(review["score"], errors="coerce")
        review["bridge_review_band"] = [
            _score_band(float(s) if pd.notna(s) else None) for s in scores
        ]
    if "fp_regime" not in review.columns:
        review["fp_regime"] = review["bridge_review_band"]

    toc_items: list[str] = []
    cards: list[str] = []
    filter_values = sorted(review["fp_regime"].astype(str).unique().tolist())
    for pair_idx, (_, row) in enumerate(review.iterrows()):
        filt_val = str(row.get("fp_regime") or "bridge")
        score_s = _format_pair_metric_value(row, "score", prec=3)
        ch = html.escape(str(row.get("retrieval_channels") or ""))
        toc_items.append(
            f'<a class="toc-item" href="#pair-{pair_idx}" data-filter-value="{html.escape(filt_val)}">'
            f"#{pair_idx + 1} score {score_s} · {ch}</a>"
        )
        cards.append(
            _bridge_pair_card_html(
                pair_idx=pair_idx,
                row=row,
                email_text_by_eid=email_text_by_eid,
            )
        )

    value_filters = "".join(
        f'<button type="button" class="filter-btn" data-filter="{html.escape(r)}">{html.escape(r)}</button>'
        for r in filter_values
    )
    doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{html.escape(title)}</title>
  <style>{_bridge_review_css()}</style>
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
      <nav class="toc">{"".join(toc_items)}</nav>
    </aside>
    <main class="main">{"".join(cards)}</main>
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


def export_bridge_review_artifacts(
    df: pd.DataFrame,
    *,
    out_root: Path,
    email_catalog: dict[str, dict[str, str]],
    label_map: dict[str, Any] | None = None,
    review_meta: dict[str, Any] | None = None,
    score_threshold_high: float = 0.9,
    high_score_max_rows: int = 500,
    mid_score_max_rows: int = 300,
    low_score_max_rows: int = 200,
) -> dict[str, Any]:
    """Write band/suspicious summaries, debug JSON, enriched CSV, and review HTML cohorts."""
    out_root = Path(out_root)
    debug_json = out_root / "debug_json"
    debug_csv = out_root / "debug_csv"
    debug_json.mkdir(parents=True, exist_ok=True)
    debug_csv.mkdir(parents=True, exist_ok=True)

    pop_diag = bridge_feature_population_diagnostics(df)
    band_analysis = build_bridge_band_analysis(df, label_map=label_map)
    suspicious = build_bridge_suspicious_high_score_analysis(df, label_map=label_map, score_threshold=score_threshold_high)
    trust_rec = build_bridge_trustworthiness_recommendation(
        df,
        band_analysis=band_analysis,
        population_diag=pop_diag,
        suspicious_analysis=suspicious,
    )

    df.to_csv(out_root / "bridge_candidate_scores.csv", index=False)
    with open(out_root / "bridge_candidate_band_analysis_summary.json", "w", encoding="utf-8") as f:
        json.dump(band_analysis, f, indent=2, default=str)
    with open(out_root / "bridge_candidate_suspicious_high_score_summary.json", "w", encoding="utf-8") as f:
        json.dump(suspicious, f, indent=2, default=str)
    with open(debug_json / "bridge_feature_population_diagnostics.json", "w", encoding="utf-8") as f:
        json.dump(pop_diag, f, indent=2, default=str)
    with open(debug_json / "bridge_trustworthiness_recommendation.json", "w", encoding="utf-8") as f:
        json.dump(trust_rec, f, indent=2, default=str)
    if review_meta:
        with open(debug_json / "bridge_review_meta.json", "w", encoding="utf-8") as f:
            json.dump(review_meta, f, indent=2, default=str)

    scores = pd.to_numeric(df["score"], errors="coerce")
    susp_rows = suspicious.get("suspicious_weak_explicit_and_latent", {}).get("examples") or []
    if susp_rows:
        pd.DataFrame(susp_rows).to_csv(debug_csv / "bridge_suspicious_high_score_examples.csv", index=False)

    export_paths: dict[str, str] = {
        "bridge_candidate_scores_csv": str(out_root / "bridge_candidate_scores.csv"),
        "band_analysis_json": str(out_root / "bridge_candidate_band_analysis_summary.json"),
        "suspicious_high_score_json": str(out_root / "bridge_candidate_suspicious_high_score_summary.json"),
    }

    if email_catalog:
        hi = df.loc[scores >= float(score_threshold_high)].sort_values("score", ascending=False).head(
            int(high_score_max_rows)
        )
        if not hi.empty:
            p = out_root / "bridge_candidates_high_score_for_review.html"
            write_bridge_review_html(
                hi,
                out_path=p,
                email_text_by_eid=email_catalog,
                title="High-score bridge candidates",
                subtitle=(
                    f"Score >= {score_threshold_high} — retrieval, shared evidence, explicit features, "
                    "scorer_encoder_cosine (MLP input), static embeddings."
                ),
            )
            export_paths["high_score_review_html"] = str(p)
        mid = df.loc[(scores >= 0.5) & (scores < float(score_threshold_high))].sort_values(
            "score", ascending=False
        ).head(int(mid_score_max_rows))
        if not mid.empty:
            p = out_root / "bridge_candidates_mid_score_for_review.html"
            write_bridge_review_html(
                mid,
                out_path=p,
                email_text_by_eid=email_catalog,
                title="Mid-score bridge candidates",
                subtitle="Scores in [0.50, 0.90).",
            )
            export_paths["mid_score_review_html"] = str(p)
        lo = df.loc[scores <= 0.2].sort_values("score", ascending=True).head(int(low_score_max_rows))
        if not lo.empty:
            p = out_root / "bridge_candidates_low_score_for_review.html"
            write_bridge_review_html(
                lo,
                out_path=p,
                email_text_by_eid=email_catalog,
                title="Low-score bridge candidates",
                subtitle="Score <= 0.20.",
            )
            export_paths["low_score_review_html"] = str(p)

    return {
        "bridge_feature_population_diagnostics": pop_diag,
        "bridge_band_analysis": band_analysis,
        "bridge_suspicious_high_score_analysis": suspicious,
        "bridge_trustworthiness_recommendation": trust_rec,
        "export_paths": export_paths,
    }
