"""
Build and store a Memgraph-compatible graph from MISP JSON events.

- Mirrors the heterogeneous schema used in graph_builder_pytorch, but as labeled nodes & relationships.
- Connects to Memgraph over Bolt using the Neo4j Python driver (works with Memgraph's openCypher).
- Deduplicates component nodes (Sender, Receiver, Week, Subject, Url, Domain, Stem, EmailDomain).
- Stores a lightweight set of properties for each node for inspection/filters.

Node labels and key properties:
- Email { eid: int, info: str, date: str, body_len: float }
- Sender { key: str }
- Receiver { key: str }
- Week { key: str }
- Subject { key: str }
- Url { key: str, full_url: str }
- Domain { key: str }
- Stem { key: str }
- EmailDomain { key: str }

Relationship types:
- (Email)-[:HAS_SENDER]->(Sender)
- (Email)-[:HAS_RECEIVER]->(Receiver)
- (Email)-[:IN_WEEK]->(Week)
- (Email)-[:HAS_SUBJECT]->(Subject)
- (Email)-[:HAS_URL]->(Url)
- (Url)-[:HAS_DOMAIN]->(Domain)
- (Url)-[:HAS_STEM]->(Stem)
- (Sender)-[:FROM_DOMAIN]->(EmailDomain)
- (Receiver)-[:FROM_DOMAIN]->(EmailDomain)

Usage:
    from core.graph.graph_builder_memgraph import build_memgraph
    build_memgraph(misp_json_path="data/misp/trec07_misp.json")

Memgraph connection defaults to bolt://localhost:7687 without auth. Override via args.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

# Reuse parsing helpers from the PyTorch graph builder
from .graph_builder_pytorch import (
    _parse_misp_events,
    _extract_week_key,
    _extract_email_domain,
)
from utils.url_extractor import parse_url_components

try:
    from neo4j import GraphDatabase  # type: ignore
except Exception as e:  # pragma: no cover
    raise ImportError(
        "The 'neo4j' Python driver is required for Memgraph connectivity. Install with: pip install neo4j"
    ) from e


def _load_misp_json(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _with_tx(session, cypher: str, parameters: Optional[Dict[str, Any]] = None) -> None:
    session.run(cypher, parameters or {})


def _create_indexes(session) -> None:
    # Use Memgraph's supported index syntax. Some versions don't support IF NOT EXISTS.
    index_statements = [
        "CREATE INDEX ON :Email(eid)",
        "CREATE INDEX ON :Sender(key)",
        "CREATE INDEX ON :Receiver(key)",
        "CREATE INDEX ON :Week(key)",
        "CREATE INDEX ON :Subject(key)",
        "CREATE INDEX ON :Url(key)",
        "CREATE INDEX ON :Domain(key)",
        "CREATE INDEX ON :Stem(key)",
        "CREATE INDEX ON :EmailDomain(key)",
    ]
    for stmt in index_statements:
        try:
            _with_tx(session, stmt)
        except Exception as e:  # Ignore "already exists" style errors across versions
            msg = str(e).lower()
            if "exist" in msg or "already" in msg:
                continue
            raise


def _clear_graph(session) -> None:
    _with_tx(session, "MATCH (n) DETACH DELETE n")


def _prepare_entities(misp_events: List[dict]):
    emails = _parse_misp_events(misp_events)

    sender_to_idx: Dict[str, int] = {}
    receiver_to_idx: Dict[str, int] = {}
    week_to_idx: Dict[str, int] = {}
    subject_to_idx: Dict[str, int] = {}
    url_to_idx: Dict[str, int] = {}
    domain_to_idx: Dict[str, int] = {}
    stem_to_idx: Dict[str, int] = {}
    email_domain_to_idx: Dict[str, int] = {}

    url_components: Dict[str, Tuple[str, str, str]] = {}

    # Pre-scan for unique components
    for em in emails:
        sender = em.get("sender")
        if sender:
            sender_to_idx.setdefault(sender, len(sender_to_idx))
            s_dom = _extract_email_domain(sender)
            if s_dom:
                email_domain_to_idx.setdefault(s_dom, len(email_domain_to_idx))

        for r in em.get("receivers", []) or []:
            if r:
                receiver_to_idx.setdefault(r, len(receiver_to_idx))
                r_dom = _extract_email_domain(r)
                if r_dom:
                    email_domain_to_idx.setdefault(r_dom, len(email_domain_to_idx))

        wk = _extract_week_key(em.get("date", ""))
        if wk:
            week_to_idx.setdefault(wk, len(week_to_idx))

        subj = em.get("subject", "")
        if subj:
            subject_to_idx.setdefault(subj, len(subject_to_idx))

        for u in em.get("urls", []) or []:
            if u:
                url_to_idx.setdefault(u, len(url_to_idx))
                parsed = parse_url_components(u)
                dom = parsed.get("domain", "")
                stem = parsed.get("stem", "")
                full_url = parsed.get("full_url", u)
                if dom:
                    domain_to_idx.setdefault(dom, len(domain_to_idx))
                if stem:
                    stem_to_idx.setdefault(stem, len(stem_to_idx))
                url_components[u] = (dom, stem, full_url)

    return {
        "emails": emails,
        "sender_to_idx": sender_to_idx,
        "receiver_to_idx": receiver_to_idx,
        "week_to_idx": week_to_idx,
        "subject_to_idx": subject_to_idx,
        "url_to_idx": url_to_idx,
        "domain_to_idx": domain_to_idx,
        "stem_to_idx": stem_to_idx,
        "email_domain_to_idx": email_domain_to_idx,
        "url_components": url_components,
    }


def _batch_create_nodes(session, label: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    # Use UNWIND + MERGE to insert in batch
    cypher = f"""
    UNWIND $rows AS row
    MERGE (n:{label} {{ {','.join([f'{k}: row.{k}' for k in rows[0].keys()])} }})
    """
    _with_tx(session, cypher, {"rows": rows})


def _batch_create_edges(session, rel: str, rows: List[Dict[str, Any]],
                        left_label: str, left_key: str,
                        right_label: str, right_key: str) -> None:
    if not rows:
        return
    cypher = f"""
    UNWIND $rows AS row
    MATCH (l:{left_label} {{{left_key}: row.l}})
    MATCH (r:{right_label} {{{right_key}: row.r}})
    MERGE (l)-[:{rel}]->(r)
    """
    _with_tx(session, cypher, {"rows": rows})


def build_memgraph(
    *,
    misp_events: Optional[List[dict]] = None,
    misp_json_path: Optional[str] = None,
    mg_uri: str = "bolt://localhost:7687",
    mg_user: Optional[str] = None,
    mg_password: Optional[str] = None,
    clear: bool = True,
    create_indexes: bool = True,
) -> Dict[str, Any]:
    """
    Build the Memgraph graph and store it via Bolt.

    Returns a summary dict with counts.
    """
    if misp_events is None and misp_json_path is None:
        raise ValueError("Provide either misp_events or misp_json_path")
    if misp_events is None:
        misp_events = _load_misp_json(misp_json_path)  # type: ignore[arg-type]

    prep = _prepare_entities(misp_events)

    # Prepare node rows
    email_rows: List[Dict[str, Any]] = []
    for eid, em in enumerate(prep["emails"]):
        email_rows.append(
            {
                "eid": int(eid),
                "info": em.get("email_info", ""),
                "date": em.get("date", ""),
                "body_len": float(len(em.get("body", ""))),
            }
        )

    sender_rows = [{"key": k} for k in prep["sender_to_idx"].keys()]
    receiver_rows = [{"key": k} for k in prep["receiver_to_idx"].keys()]
    week_rows = [{"key": k} for k in prep["week_to_idx"].keys()]
    subject_rows = [{"key": k} for k in prep["subject_to_idx"].keys()]
    url_rows = []
    for url, (_, _, full_url) in prep["url_components"].items():
        url_rows.append({"key": url, "full_url": full_url})
    domain_rows = [{"key": k} for k in prep["domain_to_idx"].keys()]
    stem_rows = [{"key": k} for k in prep["stem_to_idx"].keys()]
    email_domain_rows = [{"key": k} for k in prep["email_domain_to_idx"].keys()]

    # Prepare edge rows
    has_sender_rows: List[Dict[str, Any]] = []
    has_receiver_rows: List[Dict[str, Any]] = []
    in_week_rows: List[Dict[str, Any]] = []
    has_subject_rows: List[Dict[str, Any]] = []
    has_url_rows: List[Dict[str, Any]] = []

    url_domain_rows: List[Dict[str, Any]] = []
    url_stem_rows: List[Dict[str, Any]] = []

    sender_email_domain_rows: List[Dict[str, Any]] = []
    receiver_email_domain_rows: List[Dict[str, Any]] = []

    for eid, em in enumerate(prep["emails"]):
        if em.get("sender") and em["sender"] in prep["sender_to_idx"]:
            has_sender_rows.append({"l": int(eid), "r": em["sender"]})

        for r in em.get("receivers", []) or []:
            if r and r in prep["receiver_to_idx"]:
                has_receiver_rows.append({"l": int(eid), "r": r})

        wk = _extract_week_key(em.get("date", ""))
        if wk and wk in prep["week_to_idx"]:
            in_week_rows.append({"l": int(eid), "r": wk})

        subj = em.get("subject", "")
        if subj and subj in prep["subject_to_idx"]:
            has_subject_rows.append({"l": int(eid), "r": subj})

        for u in em.get("urls", []) or []:
            if u and u in prep["url_to_idx"]:
                has_url_rows.append({"l": int(eid), "r": u})

    for url, (dom, stem, _full) in prep["url_components"].items():
        if dom and dom in prep["domain_to_idx"]:
            url_domain_rows.append({"l": url, "r": dom})
        if stem and stem in prep["stem_to_idx"]:
            url_stem_rows.append({"l": url, "r": stem})

    for sender in prep["sender_to_idx"].keys():
        s_dom = _extract_email_domain(sender)
        if s_dom and s_dom in prep["email_domain_to_idx"]:
            sender_email_domain_rows.append({"l": sender, "r": s_dom})

    for receiver in prep["receiver_to_idx"].keys():
        r_dom = _extract_email_domain(receiver)
        if r_dom and r_dom in prep["email_domain_to_idx"]:
            receiver_email_domain_rows.append({"l": receiver, "r": r_dom})

    # Connect and write to Memgraph
    driver = GraphDatabase.driver(mg_uri, auth=(mg_user, mg_password) if mg_user or mg_password else None)
    with driver.session(database=None) as session:  # Memgraph ignores database name
        if clear:
            _clear_graph(session)
        if create_indexes:
            _create_indexes(session)

        # Create nodes (batched)
        _batch_create_nodes(session, "Email", email_rows)
        _batch_create_nodes(session, "Sender", sender_rows)
        _batch_create_nodes(session, "Receiver", receiver_rows)
        _batch_create_nodes(session, "Week", week_rows)
        _batch_create_nodes(session, "Subject", subject_rows)
        _batch_create_nodes(session, "Url", url_rows)
        _batch_create_nodes(session, "Domain", domain_rows)
        _batch_create_nodes(session, "Stem", stem_rows)
        _batch_create_nodes(session, "EmailDomain", email_domain_rows)

        # Create edges (batched)
        _batch_create_edges(session, "HAS_SENDER", has_sender_rows, "Email", "eid", "Sender", "key")
        _batch_create_edges(session, "HAS_RECEIVER", has_receiver_rows, "Email", "eid", "Receiver", "key")
        _batch_create_edges(session, "IN_WEEK", in_week_rows, "Email", "eid", "Week", "key")
        _batch_create_edges(session, "HAS_SUBJECT", has_subject_rows, "Email", "eid", "Subject", "key")
        _batch_create_edges(session, "HAS_URL", has_url_rows, "Email", "eid", "Url", "key")
        _batch_create_edges(session, "HAS_DOMAIN", url_domain_rows, "Url", "key", "Domain", "key")
        _batch_create_edges(session, "HAS_STEM", url_stem_rows, "Url", "key", "Stem", "key")
        _batch_create_edges(session, "FROM_DOMAIN", sender_email_domain_rows, "Sender", "key", "EmailDomain", "key")
        _batch_create_edges(session, "FROM_DOMAIN", receiver_email_domain_rows, "Receiver", "key", "EmailDomain", "key")

    driver.close()

    return {
        "nodes": {
            "Email": len(email_rows),
            "Sender": len(sender_rows),
            "Receiver": len(receiver_rows),
            "Week": len(week_rows),
            "Subject": len(subject_rows),
            "Url": len(url_rows),
            "Domain": len(domain_rows),
            "Stem": len(stem_rows),
            "EmailDomain": len(email_domain_rows),
        },
        "edges": {
            "HAS_SENDER": len(has_sender_rows),
            "HAS_RECEIVER": len(has_receiver_rows),
            "IN_WEEK": len(in_week_rows),
            "HAS_SUBJECT": len(has_subject_rows),
            "HAS_URL": len(has_url_rows),
            "HAS_DOMAIN": len(url_domain_rows),
            "HAS_STEM": len(url_stem_rows),
            "FROM_DOMAIN(sender)": len(sender_email_domain_rows),
            "FROM_DOMAIN(receiver)": len(receiver_email_domain_rows),
        },
    }


__all__ = ["build_memgraph"]
