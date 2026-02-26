from __future__ import annotations

from dataclasses import replace
from enum import Enum
from typing import Iterable, Set

from .assembler import GraphIR
from .graph_schema import GraphSchema


class NodeType(Enum):
    EMAIL = "email"
    SENDER = "sender"
    RECEIVER = "receiver"
    WEEK = "week"
    URL = "url"
    DOMAIN = "domain"
    STEM = "stem"
    EMAIL_DOMAIN = "email_domain"
    ATTACHMENT = "attachment"

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
