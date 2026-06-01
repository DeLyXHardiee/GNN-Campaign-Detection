"""
Semantic supernode collapse: merge cosine-threshold clusters into one graph email node
per component (singletons unchanged), union infra fields, mean-pooled BERT in embeddings overlay.

See plan: mapping JSON + collapsed ``parse_misp_events``-shaped emails + ``embeddings.json``
keys aligned to ``graph_external_id`` for anchor semantic cosine.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from .common import parse_misp_events, to_str, to_unix_ts


GRAPH_EXTERNAL_ID_PREFIX = "sem_sn_"


def stable_supernode_graph_external_id(member_external_ids: Sequence[str]) -> str:
    """Deterministic id: ``sem_sn_`` + first 16 hex chars of SHA-256 of sorted member ids."""
    members = sorted({str(x).strip() for x in member_external_ids if str(x).strip()})
    if not members:
        raise ValueError("supernode requires at least one member external_id")
    h = hashlib.sha256("\n".join(members).encode("utf-8")).hexdigest()[:16]
    return f"{GRAPH_EXTERNAL_ID_PREFIX}{h}"


def _ordered_union_lists(*seqs: Iterable[Any]) -> list[Any]:
    seen: set[str] = set()
    out: list[Any] = []
    for seq in seqs:
        for x in seq or []:
            s = str(x).strip()
            if not s or s in seen:
                continue
            seen.add(s)
            out.append(x)
    return out


def _unique_structure_fingerprints_in_member_order(html_parts: list[dict[str, Any]]) -> list[str]:
    """Distinct ``structure_fingerprint`` hex strings across members (identical values kept once)."""
    seen: set[str] = set()
    out: list[str] = []
    for h in html_parts:
        if not isinstance(h, dict):
            continue
        v = to_str(h.get("structure_fingerprint", "")).strip().lower()
        if v and v not in seen:
            seen.add(v)
            out.append(v)
    return out


def _merge_html_tag_counts(html_parts: list[dict[str, Any]]) -> dict[str, Any]:
    """Merge ``tag_counts`` across members; base html from first non-empty member.

    ``structure_fingerprint``: every **distinct** member fingerprint is kept in
    ``structure_fingerprints`` (order = first occurrence in member list). The scalar
    ``structure_fingerprint`` is set to the first distinct value for backward-compatible
    single-field consumers (e.g. fixed-length HTML feature block).
    """
    base: dict[str, Any] = {}
    tag_acc: dict[str, int] = {}
    for h in html_parts:
        if not isinstance(h, dict):
            continue
        if not base and h:
            base = deepcopy(h)
        tc = h.get("tag_counts") or {}
        if isinstance(tc, dict):
            for k, v in tc.items():
                try:
                    tag_acc[str(k)] = tag_acc.get(str(k), 0) + int(v or 0)
                except (TypeError, ValueError):
                    continue
    if not base:
        base = {}
    if tag_acc:
        base = dict(base)
        base["tag_counts"] = tag_acc
    fps = _unique_structure_fingerprints_in_member_order(html_parts)
    if fps:
        base = dict(base)
        base["structure_fingerprints"] = fps
        base["structure_fingerprint"] = fps[0]
    return base


def merge_parsed_email_dicts(
    members: list[dict[str, Any]],
    *,
    graph_external_id: str,
    representative_external_id: str,
) -> dict[str, Any]:
    """
    Merge normalized email dicts (``parse_misp_events`` output shape) for one supernode.

    Policy (documented in ``semantic_supernode_mapping.json`` meta):
    - ``senders``, ``receivers``, ``urls``, ``attachments``: ordered union (dedupe).
    - ``html``: ``tag_counts`` merged across members; **distinct** ``structure_fingerprint``
      values kept in ``html.structure_fingerprints`` (same value not duplicated); scalar
      ``structure_fingerprint`` set to the first distinct for legacy single-field readers.
      Other ``html`` keys (e.g. ``tree_stats``) come from the first non-empty member dict.
    - ``subject``, ``body``, ``email_info``, ``css``, ``return_path``,
      ``auth_*``: from the **representative** member (``representative_external_id``).
    - ``attachment_metadata``: concatenated lists.
    - ``date``: ISO/date string from the member with **maximum** unix timestamp.
    - ``EMAIL_BOOL_ATTR_KEYS`` string fields: ``\"true\"`` if any member is ``\"true\"`` (case-insensitive).
    - ``external_id`` set to ``graph_external_id``.
    """
    if not members:
        raise ValueError("merge_parsed_email_dicts: empty members")
    rep = next((m for m in members if str(m.get("external_id", "")).strip() == representative_external_id), None)
    if rep is None:
        rep = members[0]

    senders = _ordered_union_lists(*[m.get("senders") or [] for m in members])
    receivers = _ordered_union_lists(*[m.get("receivers") or [] for m in members])
    urls = _ordered_union_lists(*[m.get("urls") or [] for m in members])
    raw_atts = [m.get("attachments") or [] for m in members]
    attachments = _ordered_union_lists(*raw_atts)

    md: list[Any] = []
    for m in members:
        meta = m.get("attachment_metadata") or []
        if isinstance(meta, list):
            md.extend(meta)

    best_date = ""
    best_ts = float("-inf")
    for m in members:
        d = str(m.get("date") or "")
        ts = float(to_unix_ts(d)) if d else float("-inf")
        if ts >= best_ts and d:
            best_ts = ts
            best_date = d

    html_parts = [m.get("html") or {} for m in members if isinstance(m.get("html"), dict)]
    merged_html = _merge_html_tag_counts(html_parts) if html_parts else (rep.get("html") or {})

    bool_keys = (
        "cyrillic_domain",
        "contains_symbols",
        "body_has_tracking_url",
        "body_has_tracking_image",
        "body_has_tracking_pixel",
        "body_has_unsubscribe_link",
        "domain_is_common_webprovided",
    )
    bool_merged: dict[str, str] = {}
    for k in bool_keys:
        any_true = any(
            str(m.get(k, "") or "").strip().lower() == "true" for m in members
        )
        bool_merged[k] = "true" if any_true else "false"

    hops_parts = [m.get("received_hops") or [] for m in members]
    received_hops: list[Any] = []
    hop_seen: set[str] = set()
    for part in hops_parts:
        if not isinstance(part, list):
            continue
        for hop in part:
            key = json.dumps(hop, sort_keys=True, ensure_ascii=False) if isinstance(hop, dict) else str(hop)
            if key in hop_seen:
                continue
            hop_seen.add(key)
            received_hops.append(hop)

    out = {
        "email_info": rep.get("email_info", ""),
        "email_index": rep.get("email_index", 0),
        "external_id": graph_external_id,
        "senders": senders,
        "receivers": receivers,
        "subject": str(rep.get("subject") or ""),
        "body": str(rep.get("body") or ""),
        "html": merged_html if merged_html else (rep.get("html") or {}),
        "css": deepcopy(rep.get("css") or {}),
        "attachments": attachments,
        "attachment_metadata": md,
        "urls": urls,
        "date": best_date or str(rep.get("date") or ""),
        "received_hops": received_hops,
        **bool_merged,
        "return_path": deepcopy(rep.get("return_path") or {}),
        "auth_spf": str(rep.get("auth_spf") or ""),
        "auth_dkim": str(rep.get("auth_dkim") or ""),
        "auth_dmarc": str(rep.get("auth_dmarc") or ""),
    }
    return out


def _load_embeddings_by_key(path: Path) -> tuple[dict[str, Any], int, int]:
    data = json.loads(path.read_text(encoding="utf-8"))
    by_key = data.get("by_key") or {}
    if not isinstance(by_key, dict):
        by_key = {}
    return (
        by_key,
        int(data.get("subj_dim") or 0),
        int(data.get("body_dim") or 0),
    )


def _mean_vec(rows: list[list[float]]) -> list[float]:
    if not rows:
        return []
    a = np.asarray(rows, dtype=np.float64)
    return a.mean(axis=0).astype(np.float32).tolist()


def write_embeddings_overlay(
    *,
    source_embeddings_json: Path,
    output_dir: Path,
    nodes: list[dict[str, Any]],
    l2_normalize_after_mean: bool = False,
) -> Path:
    """
    Write ``output_dir/embeddings.json`` with one ``by_key`` entry per graph node.

    ``nodes`` entries: ``{"graph_external_id", "kind", "member_external_ids"}``.
    Singletons copy source vectors; supernodes mean-pool subject/body separately.
    """
    by_src, subj_dim, body_dim = _load_embeddings_by_key(source_embeddings_json)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "embeddings.json"
    new_by: dict[str, Any] = {}
    model_name = "intfloat/multilingual-e5-large"
    try:
        prev = json.loads(Path(source_embeddings_json).read_text(encoding="utf-8"))
        model_name = str(prev.get("model") or model_name)
    except Exception:
        pass

    for node in nodes:
        gid = str(node["graph_external_id"]).strip()
        kind = str(node.get("kind") or "")
        members = [str(x).strip() for x in (node.get("member_external_ids") or []) if str(x).strip()]
        if not gid:
            continue
        if kind == "singleton":
            if len(members) != 1:
                raise ValueError(f"singleton {gid!r} must have exactly one member")
            src = by_src.get(members[0])
            if not isinstance(src, dict):
                raise KeyError(f"Missing embeddings for singleton member {members[0]!r}")
            new_by[gid] = {
                "subj": list(src.get("subj") or []),
                "body": list(src.get("body") or []),
                "external_id": gid,
            }
            continue

        subj_rows: list[list[float]] = []
        body_rows: list[list[float]] = []
        for mid in members:
            src = by_src.get(mid)
            if not isinstance(src, dict):
                raise KeyError(f"Missing embeddings for supernode member {mid!r} (graph node {gid!r})")
            sj = src.get("subj")
            bd = src.get("body")
            if sj:
                subj_rows.append([float(x) for x in sj])
            if bd:
                body_rows.append([float(x) for x in bd])
        subj_m = _mean_vec(subj_rows)
        body_m = _mean_vec(body_rows)
        if l2_normalize_after_mean:
            if subj_m:
                v = np.asarray(subj_m, dtype=np.float64)
                n = np.linalg.norm(v)
                if n > 0:
                    subj_m = (v / n).astype(np.float32).tolist()
            if body_m:
                v = np.asarray(body_m, dtype=np.float64)
                n = np.linalg.norm(v)
                if n > 0:
                    body_m = (v / n).astype(np.float32).tolist()
        new_by[gid] = {"subj": subj_m, "body": body_m, "external_id": gid}

    if subj_dim == 0 and new_by:
        first = next(iter(new_by.values()))
        sj = first.get("subj") or []
        if isinstance(sj, list) and sj:
            subj_dim = len(sj)
    if body_dim == 0 and new_by:
        first = next(iter(new_by.values()))
        bd = first.get("body") or []
        if isinstance(bd, list) and bd:
            body_dim = len(bd)

    payload = {
        "model": model_name,
        "subj_dim": subj_dim,
        "body_dim": body_dim,
        "by_key": new_by,
        "meta": {
            "source_embeddings_json": str(source_embeddings_json.resolve()),
            "semantic_supernode_overlay": True,
            "l2_normalize_after_mean": bool(l2_normalize_after_mean),
        },
    }
    out_path.write_text(json.dumps(payload, indent=0, separators=(",", ":")), encoding="utf-8")
    return out_path


def build_collapsed_emails_and_nodes(
    *,
    misp_events: list[dict],
    clusters_csv: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """
    Returns ``(emails_for_graph, node_records_for_mapping, meta)``.

    ``node_records`` are rows for ``semantic_supernode_mapping.json`` under ``nodes``.
    """
    all_emails = parse_misp_events(misp_events)
    by_ext: dict[str, dict[str, Any]] = {}
    for em in all_emails:
        eid = str(em.get("external_id", "")).strip()
        if eid:
            by_ext[eid] = em

    df = pd.read_csv(clusters_csv)
    required = {"email_id", "cluster_id", "cluster_size", "representative_email_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"clusters CSV missing columns: {sorted(missing)}")

    node_rows: list[dict[str, Any]] = []
    merged_emails: list[dict[str, Any]] = []
    warnings: list[str] = []

    for cid, g in df.groupby("cluster_id", sort=True):
        rows = g.sort_values("email_id")
        member_ids = [str(x).strip() for x in rows["email_id"].tolist() if str(x).strip()]
        member_ids = sorted(set(member_ids))
        rep = str(rows["representative_email_id"].iloc[0]).strip()
        size = int(rows["cluster_size"].iloc[0])
        if len(member_ids) != size:
            warnings.append(
                f"cluster_id={cid}: cluster_size={size} but unique email_id count={len(member_ids)}"
            )

        missing_m = [m for m in member_ids if m not in by_ext]
        if missing_m:
            warnings.append(f"cluster_id={cid}: skipping {len(missing_m)} members not in MISP parse")
            member_ids = [m for m in member_ids if m in by_ext]
        if not member_ids:
            continue

        if len(member_ids) == 1:
            gid = member_ids[0]
            em = deepcopy(by_ext[gid])
            node_rows.append(
                {
                    "graph_external_id": gid,
                    "kind": "singleton",
                    "member_external_ids": member_ids,
                    "semantic_cluster_id": int(cid) if pd.notna(cid) else None,
                    "cluster_size": 1,
                    "representative_email_id": rep,
                }
            )
            merged_emails.append(em)
            continue

        gid = stable_supernode_graph_external_id(member_ids)
        mem_objs = [by_ext[m] for m in member_ids]
        merged = merge_parsed_email_dicts(
            mem_objs,
            graph_external_id=gid,
            representative_external_id=rep if rep in member_ids else member_ids[0],
        )
        node_rows.append(
            {
                "graph_external_id": gid,
                "kind": "supernode",
                "member_external_ids": member_ids,
                "semantic_cluster_id": int(cid) if pd.notna(cid) else None,
                "cluster_size": len(member_ids),
                "representative_email_id": rep,
            }
        )
        merged_emails.append(merged)

    merged_emails.sort(key=lambda e: str(e.get("external_id", "")))
    node_rows.sort(key=lambda r: str(r["graph_external_id"]))

    meta = {
        "merge_policies": {
            "lists": "ordered_union_dedupe",
            "subject_body_html_css_return_path_auth": "representative_member_except_html_tag_counts_and_distinct_fingerprints",
            "html_tag_counts": "sum_across_members",
            "html_structure_fingerprint": "distinct_member_values_in_order_first_for_scalar_field",
            "date": "max_unix_timestamp_across_members",
            "bool_string_flags": "any_true",
            "synthetic_id": f"{GRAPH_EXTERNAL_ID_PREFIX}<sha256(sorted_member_ids)[:16]>",
        },
        "n_input_emails_parsed": len(all_emails),
        "n_graph_email_nodes": len(merged_emails),
        "n_singletons": sum(1 for r in node_rows if r.get("kind") == "singleton"),
        "n_supernodes": sum(1 for r in node_rows if r.get("kind") == "supernode"),
        "warnings": warnings,
    }
    return merged_emails, node_rows, meta


def write_semantic_supernode_mapping(
    path: Path,
    *,
    nodes: list[dict[str, Any]],
    meta: dict[str, Any],
    clusters_csv: str,
    misp_json: str,
    source_embeddings_json: str,
) -> None:
    payload = {
        "schema_version": 1,
        "clusters_csv": clusters_csv,
        "misp_json": misp_json,
        "source_embeddings_json": source_embeddings_json,
        "meta": meta,
        "nodes": nodes,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def build_semantic_supernode_graph(
    *,
    misp_json_path: str | Path,
    clusters_csv: Path,
    source_embeddings_json: Path,
    out_dir: str | Path,
    out_name: str,
    embeddings_overlay_dir: str | Path | None = None,
    l2_normalize_after_mean: bool = False,
    schema: Any = None,
    exclude_nodes: Sequence[Any] | None = None,
    degree_node_filter: Any = None,
    email_feature_projection: Any = None,
    zero_email_timestamps: bool = False,
    collapse_enabled: bool = True,
    max_misp_events: int | None = None,
) -> tuple[Any, str, str, Path]:
    """
    Full pipeline: collapsed emails + mapping + embeddings overlay + ``build_graph`` path.

    Returns ``(graph, graph_path, meta_path, mapping_json_path)``.
    """
    from .graph_builder_pytorch import build_graph

    misp_path = Path(misp_json_path)
    with misp_path.open("r", encoding="utf-8") as f:
        misp_events = json.load(f)
    if max_misp_events is not None and max_misp_events > 0:
        misp_events = misp_events[: int(max_misp_events)]

    emails, nodes, collapse_meta = build_collapsed_emails_and_nodes(
        misp_events=misp_events,
        clusters_csv=Path(clusters_csv),
    )
    overlay_parent = Path(embeddings_overlay_dir) if embeddings_overlay_dir else Path(out_dir)
    overlay_parent.mkdir(parents=True, exist_ok=True)
    overlay_dir = overlay_parent / "semantic_supernode_embeddings_overlay"
    if overlay_dir.exists():
        shutil.rmtree(overlay_dir)
    overlay_dir.mkdir(parents=True, exist_ok=True)

    write_embeddings_overlay(
        source_embeddings_json=Path(source_embeddings_json),
        output_dir=overlay_dir,
        nodes=nodes,
        l2_normalize_after_mean=l2_normalize_after_mean,
    )

    mapping_path = Path(out_dir) / "semantic_supernode_mapping.json"
    write_semantic_supernode_mapping(
        mapping_path,
        nodes=nodes,
        meta=collapse_meta,
        clusters_csv=str(Path(clusters_csv).resolve()),
        misp_json=str(misp_path.resolve()),
        source_embeddings_json=str(Path(source_embeddings_json).resolve()),
    )

    graph, graph_path, meta_path = build_graph(
        parsed_emails=emails,
        misp_json_path=None,
        misp_events=None,
        out_dir=str(out_dir),
        out_name=out_name,
        schema=schema,
        exclude_nodes=exclude_nodes,
        degree_node_filter=degree_node_filter,
        embeddings_output_dir=str(overlay_dir),
        max_misp_events=None,
        email_feature_projection=email_feature_projection,
        zero_email_timestamps=zero_email_timestamps,
        collapse_enabled=collapse_enabled,
    )
    return graph, graph_path, meta_path, mapping_path


def build_graph_from_semantic_supernode_pipeline_config(
    cfg: dict[str, Any],
    *,
    graph_settings: Any,
    misp_json_path: str,
    max_misp_events: int | None,
) -> tuple[Any, str, str]:
    """
    Used by ``core.main.run_graph_creation`` when ``semantic_supernode.enabled`` is true.

    ``graph_settings`` is a :class:`GraphBuildSettings` instance (for projection / filters).
    """
    from config.pipeline_config import EmailFeatureProjectionSettings

    sn = cfg.get("semantic_supernode") or {}
    clusters = sn.get("clusters_csv")
    if not clusters:
        raise ValueError("semantic_supernode.enabled requires semantic_supernode.clusters_csv")
    root = Path(__file__).resolve().parent.parent.parent
    clusters_path = Path(str(clusters).strip())
    if not clusters_path.is_absolute():
        clusters_path = (root / clusters_path).resolve()

    src_emb = sn.get("source_embeddings_json") or "core/utils/embeddings/output/embeddings.json"
    src_emb_path = Path(str(src_emb).strip())
    if not src_emb_path.is_absolute():
        src_emb_path = (root / src_emb_path).resolve()
    if not src_emb_path.is_file():
        raise FileNotFoundError(f"semantic_supernode source_embeddings_json not found: {src_emb_path}")

    out_dir = Path(graph_settings.output_dir)
    stem = sn.get("hetero_graph_stem") or graph_settings.hetero_graph_stem or "semantic_supernode"
    out_name = f"{stem}_hetero.pt"
    l2 = bool(sn.get("l2_normalize_mean_bert", False))

    graph, gp, mp, _mapping = build_semantic_supernode_graph(
        misp_json_path=misp_json_path,
        clusters_csv=clusters_path,
        source_embeddings_json=src_emb_path,
        out_dir=out_dir,
        out_name=out_name,
        embeddings_overlay_dir=sn.get("embeddings_overlay_parent_dir") or out_dir,
        l2_normalize_after_mean=l2,
        exclude_nodes=graph_settings.exclude_node_types,
        degree_node_filter=graph_settings.degree_node_filter,
        email_feature_projection=graph_settings.email_feature_projection or EmailFeatureProjectionSettings(),
        zero_email_timestamps=graph_settings.zero_email_timestamps,
        collapse_enabled=graph_settings.collapse_enabled,
        max_misp_events=max_misp_events,
    )
    return graph, gp, mp
