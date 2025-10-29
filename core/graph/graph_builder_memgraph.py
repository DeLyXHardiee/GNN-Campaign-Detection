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

from .graph_schema import GraphSchema, DEFAULT_SCHEMA
from .assembler import assemble_misp_graph_ir

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


def _create_indexes(session, schema: GraphSchema) -> None:
    # Use Memgraph's supported index syntax. Some versions don't support IF NOT EXISTS.
    N = schema.nodes
    index_statements = [
        f"CREATE INDEX ON :{N['email'].memgraph}({N['email'].memgraph_id_key})",
        f"CREATE INDEX ON :{N['sender'].memgraph}({N['sender'].memgraph_id_key})",
        f"CREATE INDEX ON :{N['receiver'].memgraph}({N['receiver'].memgraph_id_key})",
        f"CREATE INDEX ON :{N['week'].memgraph}({N['week'].memgraph_id_key})",
        f"CREATE INDEX ON :{N['subject'].memgraph}({N['subject'].memgraph_id_key})",
        f"CREATE INDEX ON :{N['url'].memgraph}({N['url'].memgraph_id_key})",
        f"CREATE INDEX ON :{N['domain'].memgraph}({N['domain'].memgraph_id_key})",
        f"CREATE INDEX ON :{N['stem'].memgraph}({N['stem'].memgraph_id_key})",
        f"CREATE INDEX ON :{N['email_domain'].memgraph}({N['email_domain'].memgraph_id_key})",
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


# Legacy _prepare_entities removed in favor of shared assembler (Graph IR)


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
    schema: Optional[GraphSchema] = None,
) -> Dict[str, Any]:
    """
    Build the Memgraph graph and store it via Bolt.

    Returns a summary dict with counts.
    """
    if misp_events is None and misp_json_path is None:
        raise ValueError("Provide either misp_events or misp_json_path")
    if misp_events is None:
        misp_events = _load_misp_json(misp_json_path)  # type: ignore[arg-type]

    schema = schema or DEFAULT_SCHEMA
    N = schema.nodes
    E = schema.edges

    ir = assemble_misp_graph_ir(misp_events, schema=schema)

    # Prepare node rows
    email_rows: List[Dict[str, Any]] = []
    email_meta = ir.nodes["email"].index_to_meta or []
    n_emails = len(email_meta)
    get_attr = lambda k: (ir.email_attrs.get(k) or [0] * n_emails)
    # Build both raw and normalized arrays
    ts_raw = get_attr("ts")
    ts_norm = ir.email_attrs.get("ts_minmax") or [0.0] * n_emails
    len_body_raw = get_attr("len_body")
    len_body_norm = ir.email_attrs.get("len_body_z") or [0.0] * n_emails

    for eid, em in enumerate(email_meta):
        email_rows.append(
            {
                "eid": int(eid),
                "date": em.get("date", ""),
                # raw + normalized
                "ts": int(ts_raw[eid]) if eid < len(ts_raw) else 0,
                "ts_minmax": float(ts_norm[eid]) if eid < len(ts_norm) else 0.0,
                "n_urls": int(get_attr("n_urls")[eid]),
                "len_body": int(len_body_raw[eid]) if eid < len(len_body_raw) else 0,
                "len_body_z": float(len_body_norm[eid]) if eid < len(len_body_norm) else 0.0,
            }
        )

    # Sender nodes with docfreq
    sender_rows = []
    snd_meta = ir.nodes["sender"].index_to_string or []
    snd_attrs = ir.nodes["sender"].attrs
    for i, s in enumerate(snd_meta):
        row = {N["sender"].memgraph_id_key: s}
        if snd_attrs.get("docfreq"):
            row["docfreq"] = int(snd_attrs["docfreq"][i])
        sender_rows.append(row)

    # Receiver nodes with docfreq
    receiver_rows = []
    rcv_meta = ir.nodes["receiver"].index_to_string or []
    rcv_attrs = ir.nodes["receiver"].attrs
    for i, s in enumerate(rcv_meta):
        row = {N["receiver"].memgraph_id_key: s}
        if rcv_attrs.get("docfreq"):
            row["docfreq"] = int(rcv_attrs["docfreq"][i])
        receiver_rows.append(row)

    week_rows = [{N["week"].memgraph_id_key: s} for s in (ir.nodes["week"].index_to_string or [])]

    # Subject nodes with length props
    subject_rows = []
    subj_meta = ir.nodes["subject"].index_to_string or []
    subj_attrs = ir.nodes["subject"].attrs
    for i, s in enumerate(subj_meta):
        row = {N["subject"].memgraph_id_key: s}
        if subj_attrs.get("len_subject"):
            row["len_subject"] = int(subj_attrs["len_subject"][i])
        if subj_attrs.get("len_subject_z"):
            row["len_subject_z"] = float(subj_attrs["len_subject_z"][i])
        subject_rows.append(row)

    # URL nodes with docfreq (no redundant full_url property)
    url_rows = []
    url_meta = ir.nodes["url"].index_to_string or []
    url_attrs = ir.nodes["url"].attrs
    for i, url in enumerate(url_meta):
        row = {N["url"].memgraph_id_key: url}
        if url_attrs.get("docfreq"):
            row["docfreq"] = int(url_attrs["docfreq"][i])
        url_rows.append(row)
    # Domain rows with attributes
    domain_rows = []
    domain_meta = ir.nodes["domain"].index_to_string or []
    d_attrs = ir.nodes["domain"].attrs
    for i, s in enumerate(domain_meta):
        row = {N["domain"].memgraph_id_key: s}
        if d_attrs.get("docfreq"):
            row["docfreq"] = int(d_attrs["docfreq"][i])
        # We won't store x_lex vector in Memgraph by default to keep properties lean
        domain_rows.append(row)

    # Stem rows with attributes
    stem_rows = []
    stem_meta = ir.nodes["stem"].index_to_string or []
    s_attrs = ir.nodes["stem"].attrs
    for i, s in enumerate(stem_meta):
        row = {N["stem"].memgraph_id_key: s}
        if s_attrs.get("docfreq"):
            row["docfreq"] = int(s_attrs["docfreq"][i])
        stem_rows.append(row)

    # EmailDomain rows with attributes
    email_domain_rows = []
    ed_meta = ir.nodes["email_domain"].index_to_string or []
    ed_attrs = ir.nodes["email_domain"].attrs
    for i, s in enumerate(ed_meta):
        row = {N["email_domain"].memgraph_id_key: s}
        if ed_attrs.get("docfreq_sender"):
            row["docfreq_sender"] = int(ed_attrs["docfreq_sender"][i])
        if ed_attrs.get("docfreq_receiver"):
            row["docfreq_receiver"] = int(ed_attrs["docfreq_receiver"][i])
        email_domain_rows.append(row)

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

    # Build edge rows from IR
    def add_email_edge_rows(edge_key: str, right_node_key: str, rows: List[Dict[str, Any]]):
        src, dst = ir.edges[edge_key]
        right_meta = ir.nodes[right_node_key].index_to_string or []
        for l, r in zip(src, dst):
            rows.append({"l": int(l), "r": right_meta[r]})

    add_email_edge_rows("has_sender", "sender", has_sender_rows)
    add_email_edge_rows("has_receiver", "receiver", has_receiver_rows)
    add_email_edge_rows("in_week", "week", in_week_rows)
    add_email_edge_rows("has_subject", "subject", has_subject_rows)
    add_email_edge_rows("has_url", "url", has_url_rows)

    def add_string_edge_rows(edge_key: str, left_node_key: str, right_node_key: str, rows: List[Dict[str, Any]]):
        src, dst = ir.edges[edge_key]
        left_meta = ir.nodes[left_node_key].index_to_string or []
        right_meta = ir.nodes[right_node_key].index_to_string or []
        for l, r in zip(src, dst):
            rows.append({"l": left_meta[l], "r": right_meta[r]})

    add_string_edge_rows("url_has_domain", "url", "domain", url_domain_rows)
    add_string_edge_rows("url_has_stem", "url", "stem", url_stem_rows)

    add_string_edge_rows("sender_from_domain", "sender", "email_domain", sender_email_domain_rows)
    add_string_edge_rows("receiver_from_domain", "receiver", "email_domain", receiver_email_domain_rows)

    # Connect and write to Memgraph
    driver = GraphDatabase.driver(mg_uri, auth=(mg_user, mg_password) if mg_user or mg_password else None)
    with driver.session(database=None) as session:  # Memgraph ignores database name
        if clear:
            _clear_graph(session)
        if create_indexes:
            _create_indexes(session, schema)

        # Create nodes (batched)
        _batch_create_nodes(session, N["email"].memgraph, email_rows)
        _batch_create_nodes(session, N["sender"].memgraph, sender_rows)
        _batch_create_nodes(session, N["receiver"].memgraph, receiver_rows)
        _batch_create_nodes(session, N["week"].memgraph, week_rows)
        _batch_create_nodes(session, N["subject"].memgraph, subject_rows)
        _batch_create_nodes(session, N["url"].memgraph, url_rows)
        _batch_create_nodes(session, N["domain"].memgraph, domain_rows)
        _batch_create_nodes(session, N["stem"].memgraph, stem_rows)
        _batch_create_nodes(session, N["email_domain"].memgraph, email_domain_rows)

        # Create edges (batched)
        def add_edges(edge_key: str, rows: List[Dict[str, Any]]):
            e = E[edge_key]
            _batch_create_edges(
                session,
                e.memgraph_type,
                rows,
                e.memgraph_left_label,
                e.memgraph_left_key,
                e.memgraph_right_label,
                e.memgraph_right_key,
            )

        add_edges("has_sender", has_sender_rows)
        add_edges("has_receiver", has_receiver_rows)
        add_edges("in_week", in_week_rows)
        add_edges("has_subject", has_subject_rows)
        add_edges("has_url", has_url_rows)
        add_edges("url_has_domain", url_domain_rows)
        add_edges("url_has_stem", url_stem_rows)
        add_edges("sender_from_domain", sender_email_domain_rows)
        add_edges("receiver_from_domain", receiver_email_domain_rows)

    driver.close()

    return {
        "nodes": {
            N["email"].memgraph: len(email_rows),
            N["sender"].memgraph: len(sender_rows),
            N["receiver"].memgraph: len(receiver_rows),
            N["week"].memgraph: len(week_rows),
            N["subject"].memgraph: len(subject_rows),
            N["url"].memgraph: len(url_rows),
            N["domain"].memgraph: len(domain_rows),
            N["stem"].memgraph: len(stem_rows),
            N["email_domain"].memgraph: len(email_domain_rows),
        },
        "edges": {
            E["has_sender"].memgraph_type: len(has_sender_rows),
            E["has_receiver"].memgraph_type: len(has_receiver_rows),
            E["in_week"].memgraph_type: len(in_week_rows),
            E["has_subject"].memgraph_type: len(has_subject_rows),
            E["has_url"].memgraph_type: len(has_url_rows),
            E["url_has_domain"].memgraph_type: len(url_domain_rows),
            E["url_has_stem"].memgraph_type: len(url_stem_rows),
            f"{E['sender_from_domain'].memgraph_type}(sender)": len(sender_email_domain_rows),
            f"{E['receiver_from_domain'].memgraph_type}(receiver)": len(receiver_email_domain_rows),
        },
    }


__all__ = ["build_memgraph"]
