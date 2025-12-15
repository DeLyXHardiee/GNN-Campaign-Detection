from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

from .graph_schema import GraphSchema, DEFAULT_SCHEMA
from .assembler import assemble_misp_graph_ir
from .graph_filter import NodeType, filter_graph_ir

try:
    from neo4j import GraphDatabase 
except Exception as e: 
    raise ImportError(
        "The 'neo4j' Python driver is required for Memgraph connectivity. Install with: pip install neo4j"
    ) from e


def _load_misp_json(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _with_tx(session, cypher: str, parameters: Optional[Dict[str, Any]] = None) -> None:
    session.run(cypher, parameters or {})

def _create_indexes(session, schema: GraphSchema) -> None:
    N = schema.nodes
    index_statements = [
        f"CREATE INDEX ON :{N['email'].memgraph}({N['email'].memgraph_id_key})",
        f"CREATE INDEX ON :{N['sender'].memgraph}({N['sender'].memgraph_id_key})",
        f"CREATE INDEX ON :{N['receiver'].memgraph}({N['receiver'].memgraph_id_key})",
        f"CREATE INDEX ON :{N['week'].memgraph}({N['week'].memgraph_id_key})",
        f"CREATE INDEX ON :{N['url'].memgraph}({N['url'].memgraph_id_key})",
        f"CREATE INDEX ON :{N['domain'].memgraph}({N['domain'].memgraph_id_key})",
        f"CREATE INDEX ON :{N['stem'].memgraph}({N['stem'].memgraph_id_key})",
        f"CREATE INDEX ON :{N['email_domain'].memgraph}({N['email_domain'].memgraph_id_key})",
    ]
    for stmt in index_statements:
        try:
            _with_tx(session, stmt)
        except Exception as e: 
            msg = str(e).lower()
            if "exist" in msg or "already" in msg:
                continue
            raise


def _clear_graph(session) -> None:
    _with_tx(session, "MATCH (n) DETACH DELETE n")

def _batch_create_nodes(session, label: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
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


def _prepare_node_rows_from_ir(ir: Any, schema: GraphSchema) -> Dict[str, List[Dict[str, Any]]]:
    N = schema.nodes
    out: Dict[str, List[Dict[str, Any]]] = {}

    email_rows: List[Dict[str, Any]] = []
    email_node = ir.nodes.get("email")
    email_meta = (email_node and email_node.index_to_meta) or []
    n_emails = len(email_meta)
    get_attr = lambda k: (ir.email_attrs.get(k) or [0] * n_emails)
    ts_raw = get_attr("ts")
    len_body_raw = get_attr("len_body")
    subj_dim_arr = ir.email_attrs.get("x_text_subject_dim") or [0] * n_emails
    body_dim_arr = ir.email_attrs.get("x_text_body_dim") or [0] * n_emails
    len_subject_arr = ir.email_attrs.get("len_subject") or [0] * n_emails
    for eid, em in enumerate(email_meta):
        email_rows.append(
            {
                "eid": int(eid),
                "email_index": em.get("email_index", int(eid)),
                "date": em.get("date", ""),
                "ts": int(ts_raw[eid]) if eid < len(ts_raw) else 0,
                "n_urls": int(get_attr("n_urls")[eid]),
                "len_body": int(len_body_raw[eid]) if eid < len(len_body_raw) else 0,
                "x_text_subject_dim": int(subj_dim_arr[eid]) if eid < len(subj_dim_arr) else 0,
                "x_text_body_dim": int(body_dim_arr[eid]) if eid < len(body_dim_arr) else 0,
                "len_subject": int(len_subject_arr[eid]) if eid < len(len_subject_arr) else 0,
            }
        )
    out[N["email"].memgraph] = email_rows

    # Helper to pack simple string-keyed nodes with optional attributes aligned by index
    def pack_string_nodes(node_key: str, extra_fields: Dict[str, List[Any]] = None) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        node = ir.nodes.get(node_key)
        meta = (node and node.index_to_string) or []
        attrs = (node and node.attrs) or {}
        id_key = N[node_key].memgraph_id_key
        for i, s in enumerate(meta):
            row = {id_key: s}
            if extra_fields:
                for k, arr in extra_fields.items():
                    if arr and i < len(arr):
                        row[k] = arr[i] if not isinstance(arr[i], bool) else int(arr[i])
            if attrs:
                for k in ("docfreq", "len_subject", "docfreq_sender", "docfreq_receiver"):
                    vals = attrs.get(k)
                    if vals is not None and i < len(vals):
                        row[k] = vals[i]

            rows.append(row)
        return rows

    out[N["sender"].memgraph] = pack_string_nodes("sender")
    out[N["receiver"].memgraph] = pack_string_nodes("receiver")
    out[N["week"].memgraph] = pack_string_nodes("week")
    out[N["url"].memgraph] = pack_string_nodes("url")
    out[N["domain"].memgraph] = pack_string_nodes("domain")
    out[N["stem"].memgraph] = pack_string_nodes("stem")
    out[N["email_domain"].memgraph] = pack_string_nodes("email_domain")

    return out


def _prepare_edge_rows_from_ir(ir: Any, schema: GraphSchema) -> Dict[str, List[Dict[str, Any]]]:
    N = schema.nodes
    E = schema.edges
    out: Dict[str, List[Dict[str, Any]]] = {e.memgraph_type: [] for e in E.values()}

    def add_email_edge_rows(edge_key: str, right_node_key: str, mem_type: str):
        if edge_key not in ir.edges:
            return
        rows = out[mem_type]
        src, dst = ir.edges[edge_key]
        right_node = ir.nodes.get(right_node_key)
        right_meta = (right_node and right_node.index_to_string) or []
        for l, r in zip(src, dst):
            rows.append({"l": int(l), "r": right_meta[r]})

    def add_string_edge_rows(edge_key: str, left_node_key: str, right_node_key: str, mem_type: str):
        if edge_key not in ir.edges:
            return
        
        rows = out[mem_type]
        src, dst = ir.edges[edge_key]
        left_node = ir.nodes.get(left_node_key)
        right_node = ir.nodes.get(right_node_key)
        left_meta = (left_node and left_node.index_to_string) or []
        right_meta = (right_node and right_node.index_to_string) or []

        for l, r in zip(src, dst):
            rows.append({"l": left_meta[l], "r": right_meta[r]})

    add_email_edge_rows("has_sender", "sender", E["has_sender"].memgraph_type)
    add_email_edge_rows("has_receiver", "receiver", E["has_receiver"].memgraph_type)
    add_email_edge_rows("in_week", "week", E["in_week"].memgraph_type)
    add_email_edge_rows("has_url", "url", E["has_url"].memgraph_type)
    add_email_edge_rows("has_domain", "domain", E["has_domain"].memgraph_type)
    add_email_edge_rows("has_stem", "stem", E["has_stem"].memgraph_type)
    add_string_edge_rows("sender_from_domain", "sender", "email_domain", E["sender_from_domain"].memgraph_type)
    add_string_edge_rows("receiver_from_domain", "receiver", "email_domain", E["receiver_from_domain"].memgraph_type)

    return out


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
    exclude_nodes: Optional[list[NodeType]] = None,
) -> Dict[str, Any]:

    if misp_events is None and misp_json_path is None:
        raise ValueError("Provide either misp_events or misp_json_path")
    if misp_events is None:
        misp_events = _load_misp_json(misp_json_path)

    schema = schema or DEFAULT_SCHEMA
    N = schema.nodes
    E = schema.edges

    ir = assemble_misp_graph_ir(misp_events, schema=schema)
    if exclude_nodes:
        ir = filter_graph_ir(ir, exclude_nodes=NodeType.canonical_set(exclude_nodes), schema=schema)

    node_rows_by_label = _prepare_node_rows_from_ir(ir, schema)
    edge_rows_by_type = _prepare_edge_rows_from_ir(ir, schema)

    driver = GraphDatabase.driver(mg_uri, auth=(mg_user, mg_password) if mg_user or mg_password else None)
    with driver.session(database=None) as session:  
        if clear:
            _clear_graph(session)
        if create_indexes:
            _create_indexes(session, schema)

        for label, rows in node_rows_by_label.items():
            _batch_create_nodes(session, label, rows)

        def add_edges(edge_key: str):
            e = E[edge_key]
            rows = edge_rows_by_type.get(e.memgraph_type, [])
            _batch_create_edges(
                session,
                e.memgraph_type,
                rows,
                e.memgraph_left_label,
                e.memgraph_left_key,
                e.memgraph_right_label,
                e.memgraph_right_key,
            )

        for edge_key in [
            "has_sender",
            "has_receiver",
            "in_week",
            "has_url",
            "has_domain",
            "has_stem",
            "sender_from_domain",
            "receiver_from_domain",
        ]:
            add_edges(edge_key)

    driver.close()

    return {
        "nodes": {
            k: len(v) for k, v in node_rows_by_label.items()
        },
        "edges": {
            **{t: len(rows) for t, rows in edge_rows_by_type.items()},
        },
    }


__all__ = ["build_memgraph"]
