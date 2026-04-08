from __future__ import annotations

from dataclasses import replace
from enum import Enum
from typing import Dict, Iterable, Set

from .assembler import GraphIR
from .graph_schema import GraphSchema


class NodeType(Enum):
    EMAIL = "email"
    SENDER = "sender"
    RECEIVER = "receiver"
    URL = "url"
    DOMAIN = "domain"
    STEM = "stem"
    EMAIL_DOMAIN = "email_domain"
    ATTACHMENT = "attachment"
    ORIGIN_IP = "origin_ip"
    RECEIVED_HOST = "received_host"
    RETURN_PATH_EMAIL = "return_path_email"
    RETURN_PATH_DOMAIN = "return_path_domain"

    @classmethod
    def canonical_set(cls, items: Iterable["NodeType | str"], schema: GraphSchema | None = None) -> Set[str]:
        out: Set[str] = set()
        for item in items:
            if isinstance(item, NodeType):
                out.add(item.value)
            else:
                out.add(str(item))
        if schema is not None:
            out = {n for n in out if n in schema.nodes}
        return out


def filter_graph_ir(ir: GraphIR, *, exclude_nodes: Set[str], schema: GraphSchema) -> GraphIR:
    """Return a pruned GraphIR by removing selected node types and any edges touching them.

    - exclude_nodes: set of canonical node type names to remove (e.g., {"url", "domain"}).
    - schema: used to inspect edge endpoints.
    """
    if not exclude_nodes:
        return ir

    new_nodes = {k: v for k, v in ir.nodes.items() if k not in exclude_nodes}

    new_edges = {}
    for ek, (src_idx, dst_idx) in ir.edges.items():
        try:
            e = schema.edge(ek)
        except KeyError:
            continue
        if e.src in exclude_nodes or e.dst in exclude_nodes:
            continue
        new_edges[ek] = (src_idx, dst_idx)

    return replace(ir, nodes=new_nodes, edges=new_edges)


def filter_graph_ir_by_degree(
    ir: GraphIR,
    *,
    schema: GraphSchema,
    strength: float,
    target_node_types: Set[str] | None = None,
    min_degree: int = 2,
) -> GraphIR:
    """
    Remove high-degree nodes using a strength-controlled threshold.

    - ``strength`` in [0, 1]: low values remove only top-degree nodes; higher values
      lower the degree threshold and remove more nodes.
    - ``target_node_types``: optional canonical node types to filter. Defaults to all.
    - ``min_degree``: lower bound for removals to avoid pruning low-degree nodes.
    """
    if strength <= 0.0:
        return ir

    s = max(0.0, min(1.0, float(strength)))
    min_deg = max(0, int(min_degree))
    targets = (
        {nt for nt in target_node_types if nt in ir.nodes}
        if target_node_types is not None
        else set(ir.nodes.keys())
    )
    if not targets:
        return ir

    degrees: Dict[str, list[int]] = {
        nt: [0] * len(node.x) for nt, node in ir.nodes.items()
    }
    for edge_key, (src_idx, dst_idx) in ir.edges.items():
        edge = schema.edges.get(edge_key)
        if edge is None:
            continue
        src_deg = degrees.get(edge.src)
        dst_deg = degrees.get(edge.dst)
        if src_deg is None or dst_deg is None:
            continue
        for s_i, d_i in zip(src_idx, dst_idx):
            if 0 <= s_i < len(src_deg):
                src_deg[s_i] += 1
            if 0 <= d_i < len(dst_deg):
                dst_deg[d_i] += 1

    remove_indices: Dict[str, Set[int]] = {}
    for nt in targets:
        deg = degrees.get(nt, [])
        if not deg:
            continue
        # Strength maps to quantile: 0.0 => near-max threshold, 1.0 => min threshold.
        q = 1.0 - s
        sorted_deg = sorted(deg)
        q_idx = int(q * (len(sorted_deg) - 1)) if sorted_deg else 0
        threshold = sorted_deg[q_idx] if sorted_deg else 0
        to_remove = {i for i, d in enumerate(deg) if d > threshold and d >= min_deg}
        if to_remove:
            remove_indices[nt] = to_remove

    if not remove_indices:
        return ir

    new_nodes = {}
    old_to_new_by_type: Dict[str, Dict[int, int]] = {}
    for nt, node in ir.nodes.items():
        removed = remove_indices.get(nt, set())
        kept_old = [i for i in range(len(node.x)) if i not in removed]
        old_to_new = {old_i: new_i for new_i, old_i in enumerate(kept_old)}
        old_to_new_by_type[nt] = old_to_new

        new_x = [node.x[i] for i in kept_old]
        new_attrs = {
            k: [vals[i] for i in kept_old if i < len(vals)]
            for k, vals in node.attrs.items()
        }
        new_index_to_string = (
            [node.index_to_string[i] for i in kept_old if node.index_to_string and i < len(node.index_to_string)]
            if node.index_to_string is not None
            else None
        )
        new_index_to_meta = (
            [node.index_to_meta[i] for i in kept_old if node.index_to_meta and i < len(node.index_to_meta)]
            if node.index_to_meta is not None
            else None
        )
        new_index = {}
        if new_index_to_string is not None:
            new_index = {v: i for i, v in enumerate(new_index_to_string)}

        new_nodes[nt] = type(node)(
            index=new_index,
            x=new_x,
            index_to_string=new_index_to_string,
            index_to_meta=new_index_to_meta,
            attrs=new_attrs,
        )

    new_edges = {}
    for edge_key, (src_idx, dst_idx) in ir.edges.items():
        edge = schema.edges.get(edge_key)
        if edge is None:
            continue
        src_map = old_to_new_by_type.get(edge.src, {})
        dst_map = old_to_new_by_type.get(edge.dst, {})
        out_src: list[int] = []
        out_dst: list[int] = []
        for s_i, d_i in zip(src_idx, dst_idx):
            ns = src_map.get(s_i)
            nd = dst_map.get(d_i)
            if ns is None or nd is None:
                continue
            out_src.append(ns)
            out_dst.append(nd)
        new_edges[edge_key] = (out_src, out_dst)

    # email_attrs (external_id, ts, etc.) is aligned to email node row order; keep it in sync
    # when emails are removed so metadata / clustering match data["email"].num_nodes.
    new_email_attrs = ir.email_attrs
    if "email" in remove_indices and remove_indices["email"]:
        email_node = ir.nodes.get("email")
        if email_node is not None:
            n_email = len(email_node.x)
            kept_email_idx = [i for i in range(n_email) if i not in remove_indices["email"]]
            sliced: Dict[str, list] = {}
            for k, vals in ir.email_attrs.items():
                if isinstance(vals, list):
                    sliced[k] = [vals[i] for i in kept_email_idx if i < len(vals)]
                else:
                    sliced[k] = vals
            new_email_attrs = sliced

    return replace(ir, nodes=new_nodes, edges=new_edges, email_attrs=new_email_attrs)
