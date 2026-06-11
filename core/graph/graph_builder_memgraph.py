from __future__ import annotations

import json
import os
from collections.abc import Sequence
from typing import Any, Dict, List, Optional, Tuple

from .graph_schema import GraphSchema, DEFAULT_SCHEMA
from .assembler import assemble_misp_graph_ir, AUTH_ATTR_KEYS
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
    index_statements = [
        f"CREATE INDEX ON :{node.memgraph}({node.memgraph_id_key})"
        for node in schema.nodes.values()
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
    _email_bool_attrs = (
        "cyrillic_domain",
        "contains_symbols",
        "body_has_tracking_url",
        "body_has_tracking_image",
        "body_has_tracking_pixel",
        "body_has_unsubscribe_link",
        "domain_is_common_webprovided",
    )
    get_attr = lambda k: (ir.email_attrs.get(k) or [0] * n_emails)
    ts_raw = get_attr("ts")
    len_body_raw = get_attr("len_body")
    subj_dim_arr = ir.email_attrs.get("x_text_subject_dim") or [0] * n_emails
    body_dim_arr = ir.email_attrs.get("x_text_body_dim") or [0] * n_emails
    len_subject_arr = ir.email_attrs.get("len_subject") or [0] * n_emails
    for eid, em in enumerate(email_meta):
        row: Dict[str, Any] = {
            "eid": int(eid),
            "email_index": em.get("email_index", int(eid)),
            "external_id": str(em.get("external_id") or ""),
            "date": em.get("date", ""),
            "ts": int(ts_raw[eid]) if eid < len(ts_raw) else 0,
            "n_urls": int(get_attr("n_urls")[eid]),
            "len_body": int(len_body_raw[eid]) if eid < len(len_body_raw) else 0,
            "x_text_subject_dim": int(subj_dim_arr[eid]) if eid < len(subj_dim_arr) else 0,
            "x_text_body_dim": int(body_dim_arr[eid]) if eid < len(body_dim_arr) else 0,
            "len_subject": int(len_subject_arr[eid]) if eid < len(len_subject_arr) else 0,
        }
        for k in _email_bool_attrs:
            arr = get_attr(k)
            row[k] = int(arr[eid]) if eid < len(arr) else 0
        for k in AUTH_ATTR_KEYS:
            arr = get_attr(k)
            row[k] = str(arr[eid]) if eid < len(arr) and arr[eid] is not None else ""
        email_rows.append(row)
    out[N["email"].memgraph] = email_rows

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

    for node_key, node_map in schema.nodes.items():
        if node_key == "email":
            continue
        out[node_map.memgraph] = pack_string_nodes(node_key)

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

    for edge_key, edge_map in E.items():
        if edge_map.edge_strategy == "email_to_entity" or edge_map.src == "email":
            add_email_edge_rows(edge_key, edge_map.dst, edge_map.memgraph_type)
        else:
            add_string_edge_rows(edge_key, edge_map.src, edge_map.dst, edge_map.memgraph_type)

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
    exclude_nodes: Optional[Sequence[NodeType | str]] = None,
    embeddings_output_dir: Optional[str] = None,
    max_misp_events: Optional[int] = None,
    zero_email_timestamps: bool = False,
    filter_popular_domains: bool = True,
) -> Dict[str, Any]:

    if misp_events is None and misp_json_path is None:
        raise ValueError("Provide either misp_events or misp_json_path")
    if misp_events is None:
        misp_events = _load_misp_json(misp_json_path)

    if max_misp_events is not None and max_misp_events > 0:
        misp_events = misp_events[:max_misp_events]

    schema = schema or DEFAULT_SCHEMA
    N = schema.nodes
    E = schema.edges

    pop_domains: frozenset = frozenset()
    if filter_popular_domains:
        import sys as _sys, os as _os
        _core = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
        if _core not in _sys.path:
            _sys.path.insert(0, _core)
        from feature_set_extraction.domain_lists_loader import load_url_intelligence_sets
        pop_domains = frozenset(load_url_intelligence_sets().get("popular_domains", set()))
    ir = assemble_misp_graph_ir(
        misp_events,
        schema=schema,
        embeddings_output_dir=embeddings_output_dir,
        zero_email_timestamps=zero_email_timestamps,
        popular_domains=pop_domains,
    )
    if exclude_nodes:
        ir = filter_graph_ir(ir, exclude_nodes=NodeType.canonical_set(exclude_nodes, schema=schema), schema=schema)

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

        for edge_key in E:
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
