"""
Graph builder for PyTorch Geometric Heterogeneous graphs from MISP JSON.

Capabilities:
- Accepts input either as an in-memory list of MISP events or from a JSON file path.
- Builds a HeteroData graph with email nodes as central hubs connected to component nodes.
- Node types: 'email', 'sender', 'receiver', 'week', 'url', 'domain', 'stem', 'email_domain'.
- Edges:
    - ('email', 'has_sender', 'sender')
    - ('email', 'has_receiver', 'receiver')
    - ('email', 'in_week', 'week') - emails are grouped by ISO week
    - ('email', 'has_url', 'url')
    - ('url', 'has_domain', 'domain')
    - ('url', 'has_stem', 'stem')
    - ('sender', 'from_domain', 'email_domain')
    - ('receiver', 'from_domain', 'email_domain')
- Component nodes are deduplicated: multiple emails sharing the same sender, week, etc. 
    will have edges to the same component node.
- URLs are parsed into domain and stem components for better deduplication.
- Email addresses are normalized (lowercase, angle brackets removed) and connected to their 
    domain nodes (email_domain) to increase graph connectivity.
 - Email features include normalized scalars (ts_minmax, len_body_z, n_urls_z, len_subject_z) and optional TF-IDF.
- Creates simple numeric features for nodes (lengths) to keep tensors valid.
- Saves both the graph (.pt via torch.save) and a companion metadata JSON mapping node indices to original strings.

If torch or torch_geometric are not installed, import errors will clearly explain how to install them.
"""
from __future__ import annotations

import json
import os
import sys
from typing import Dict, List, Optional, Tuple, Any, TYPE_CHECKING

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Shared schema configuration and assembler
from .graph_schema import GraphSchema, DEFAULT_SCHEMA
from .assembler import assemble_misp_graph_ir
from .graph_filter import NodeType, filter_graph_ir

if TYPE_CHECKING:  # For type checkers only; avoids runtime import requirement
    import torch  # type: ignore
    from torch_geometric.data import HeteroData  # type: ignore
else:
    torch = None  # type: ignore
    HeteroData = Any  # type: ignore

from .common import (
    parse_misp_events,
    extract_week_key,
    extract_email_domain,
)


def _ensure_torch():
    global torch
    if torch is None:
        try:
            import torch as _torch  # type: ignore
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "PyTorch is required. Install with: pip install torch --index-url https://download.pytorch.org/whl/cpu"
            ) from e
        torch = _torch  # type: ignore
    return torch


def _ensure_heterodata():
    global HeteroData
    if HeteroData is Any:
        try:
            from torch_geometric.data import HeteroData as _HeteroData  # type: ignore
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "PyTorch Geometric is required. Install with: pip install torch-geometric"
            ) from e
        HeteroData = _HeteroData  # type: ignore
    return HeteroData


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _load_misp_json(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


"""
Important: Helper functions moved to core.graph.common to be shared with the IR assembler.
This file now focuses on rendering HeteroData from the shared Graph IR.
"""


def _merge_features_with_attrs(base: List[List[float]], attr_vals: Dict[str, Any], keys: List[str]) -> List[List[float]]:
    """Concatenate selected attribute columns (scalars or vectors) to a base feature matrix.

    - base: NxD list of floats
    - attr_vals: mapping of attribute name -> per-node values (list or list of lists)
    - keys: attribute names to append in order
    Returns a new Nx(D+sum(attr_dims)) matrix.
    """
    if not base:
        return []
    n = len(base)
    extras_per_row: List[List[List[float]]] = [[] for _ in range(n)]
    for k in keys:
        vals = attr_vals.get(k)
        if vals is None:
            continue
        if isinstance(vals, list) and len(vals) > 0 and isinstance(vals[0], (list, tuple)):
            for i in range(n):
                v = vals[i] if i < len(vals) else []
                extras_per_row[i].append([float(x) for x in v])
        else:
            for i in range(n):
                x = float(vals[i]) if i < len(vals) else 0.0
                extras_per_row[i].append([x])
    out: List[List[float]] = []
    for i in range(n):
        row = list(base[i])
        for chunk in extras_per_row[i]:
            row.extend(chunk)
        out.append(row)
    return out


def _set_node_features_from_ir(data: Any, ir: Any, schema: GraphSchema) -> None:
    """Populate node feature tensors on a HeteroData object from Graph IR."""
    HData = _ensure_heterodata()
    torch_lib = _ensure_torch()
    N = schema.nodes

    # Email: use IR-provided features as-is (normalized only)
    if "email" not in ir.nodes:
        # No emails; create empty type
        data[N["email"].pyg].num_nodes = 0
        return
    email_x = ir.nodes["email"].x
    if email_x:
        data[N["email"].pyg].x = torch_lib.tensor(email_x, dtype=torch_lib.float)
    else:
        data[N["email"].pyg].num_nodes = 0

    def set_simple(node_key: str, extra_keys: List[str] = None):
        if node_key not in ir.nodes:
            # Skip entirely when filtered out
            return
        x = ir.nodes[node_key].x
        if x:
            attrs = ir.nodes[node_key].attrs
            if attrs and extra_keys:
                x = _merge_features_with_attrs(x, attrs, extra_keys)
            data[N[node_key].pyg].x = torch_lib.tensor(x, dtype=torch_lib.float)
        else:
            data[N[node_key].pyg].num_nodes = 0

    # Use normalized docfreqs for sender/receiver
    set_simple("sender", ["docfreq_z"])
    set_simple("receiver", ["docfreq_z"])
    set_simple("week")
    # url has raw docfreq; domain/stem have lexical vectors + docfreqs; email_domain uses normalized docfreqs
    set_simple("url", ["docfreq"])
    set_simple("domain", ["x_lex", "docfreq"])
    set_simple("stem", ["x_lex", "docfreq"])
    set_simple("email_domain", ["x_lex", "docfreq_sender_z", "docfreq_receiver_z"])


def _set_edges_from_ir(data: Any, ir: Any, schema: GraphSchema) -> None:
    """Populate edge_index for all canonical edges from Graph IR."""
    torch_lib = _ensure_torch()
    N = schema.nodes

    def set_edges(edge_key: str):
        if edge_key not in ir.edges:
            return
        e = schema.edge(edge_key)
        src, dst = ir.edges[edge_key]
        if src:
            data[N[e.src].pyg, e.rel_pyg, N[e.dst].pyg].edge_index = torch_lib.tensor([src, dst], dtype=torch_lib.long)

    for ek in [
        "has_sender",
        "has_receiver",
        "in_week",
        "has_url",
        "url_has_domain",
        "url_has_stem",
        "sender_from_domain",
        "receiver_from_domain",
    ]:
        set_edges(ek)


def _build_metadata_from_ir(data: Any, ir: Any, schema: GraphSchema) -> Dict[str, Any]:
    """Construct the metadata dict summarizing node maps, feature shapes, and edge counts."""
    N = schema.nodes
    sender_meta = (ir.nodes.get("sender") and ir.nodes["sender"].index_to_string) or []
    receiver_meta = (ir.nodes.get("receiver") and ir.nodes["receiver"].index_to_string) or []
    week_meta = (ir.nodes.get("week") and ir.nodes["week"].index_to_string) or []
    url_meta = (ir.nodes.get("url") and ir.nodes["url"].index_to_string) or []
    domain_meta = (ir.nodes.get("domain") and ir.nodes["domain"].index_to_string) or []
    stem_meta = (ir.nodes.get("stem") and ir.nodes["stem"].index_to_string) or []
    email_domain_meta = (ir.nodes.get("email_domain") and ir.nodes["email_domain"].index_to_string) or []
    email_meta = (ir.nodes.get("email") and ir.nodes["email"].index_to_meta) or []

    meta = {
        "node_maps": {
            N["email"].pyg: {"index_to_meta": email_meta},
            N["sender"].pyg: {"index_to_string": sender_meta},
            N["receiver"].pyg: {"index_to_string": receiver_meta},
            N["week"].pyg: {"index_to_string": week_meta},
            N["url"].pyg: {"index_to_string": url_meta},
            N["domain"].pyg: {"index_to_string": domain_meta},
            N["stem"].pyg: {"index_to_string": stem_meta},
            N["email_domain"].pyg: {"index_to_string": email_domain_meta},
        },
        "feature_shapes": {
            N["email"].pyg: list(data[N["email"].pyg].x.shape) if "x" in data[N["email"].pyg] else [0, 0],
            N["sender"].pyg: list(data[N["sender"].pyg].x.shape) if "x" in data[N["sender"].pyg] else [0, 0],
            N["receiver"].pyg: list(data[N["receiver"].pyg].x.shape) if "x" in data[N["receiver"].pyg] else [0, 0],
            N["week"].pyg: list(data[N["week"].pyg].x.shape) if "x" in data[N["week"].pyg] else [0, 0],
            N["url"].pyg: list(data[N["url"].pyg].x.shape) if "x" in data[N["url"].pyg] else [0, 0],
            N["domain"].pyg: list(data[N["domain"].pyg].x.shape) if "x" in data[N["domain"].pyg] else [0, 0],
            N["stem"].pyg: list(data[N["stem"].pyg].x.shape) if "x" in data[N["stem"].pyg] else [0, 0],
            N["email_domain"].pyg: list(data[N["email_domain"].pyg].x.shape) if "x" in data[N["email_domain"].pyg] else [0, 0],
        },
        "edge_counts": {
            f"{N['email'].pyg}->{N['sender'].pyg}:{schema.edge('has_sender').rel_pyg}": len(ir.edges.get('has_sender', ([], []))[0]),
            f"{N['email'].pyg}->{N['receiver'].pyg}:{schema.edge('has_receiver').rel_pyg}": len(ir.edges.get('has_receiver', ([], []))[0]),
            f"{N['email'].pyg}->{N['week'].pyg}:{schema.edge('in_week').rel_pyg}": len(ir.edges.get('in_week', ([], []))[0]),
            f"{N['email'].pyg}->{N['url'].pyg}:{schema.edge('has_url').rel_pyg}": len(ir.edges.get('has_url', ([], []))[0]),
            f"{N['url'].pyg}->{N['domain'].pyg}:{schema.edge('url_has_domain').rel_pyg}": len(ir.edges.get('url_has_domain', ([], []))[0]),
            f"{N['url'].pyg}->{N['stem'].pyg}:{schema.edge('url_has_stem').rel_pyg}": len(ir.edges.get('url_has_stem', ([], []))[0]),
            f"{N['sender'].pyg}->{N['email_domain'].pyg}:{schema.edge('sender_from_domain').rel_pyg}": len(ir.edges.get('sender_from_domain', ([], []))[0]),
            f"{N['receiver'].pyg}->{N['email_domain'].pyg}:{schema.edge('receiver_from_domain').rel_pyg}": len(ir.edges.get('receiver_from_domain', ([], []))[0]),
        },
    }
    return meta


def build_hetero_graph_from_misp(
    misp_events: List[dict],
    *,
    schema: Optional[GraphSchema] = None,
    exclude_nodes: Optional[list[NodeType]] = None,
) -> Tuple[Any, Dict[str, Any]]:
    """
    Build a HeteroData graph from a list of MISP events.
    
        New schema: Email nodes are central hubs connected to component nodes:
        - Node types: email, sender, receiver, week, url, domain, stem, email_domain
    - Edge types: 
      - (email, has_sender, sender)
      - (email, has_receiver, receiver)
      - (email, in_week, week)
      - (email, has_url, url)
      - (url, has_domain, domain)
      - (url, has_stem, stem)
      - (sender, from_domain, email_domain)
      - (receiver, from_domain, email_domain)
    
    Components are deduplicated: multiple emails sharing the same sender/receiver/week/etc. 
    will have edges to the same component node. URLs are decomposed into domain and stem.
    Email addresses are normalized (lowercase, angle brackets removed) and connected to 
    their domain nodes to increase connectivity.
    
    Email features include normalized scalars: ts_minmax, len_body_z,
    n_urls_z, len_subject_z, and optional TF-IDF of subject/body.

    Returns (graph, metadata) where metadata contains mappings for node indices.
    """
    schema = schema or DEFAULT_SCHEMA
    # Resolve node labels per backend
    N = schema.nodes
    ir = assemble_misp_graph_ir(misp_events, schema=schema)
    if exclude_nodes:
        ir = filter_graph_ir(ir, exclude_nodes=NodeType.canonical_set(exclude_nodes), schema=schema)

    HData = _ensure_heterodata()
    data = HData()

    _set_node_features_from_ir(data, ir, schema)
    _set_edges_from_ir(data, ir, schema)
    metadata = _build_metadata_from_ir(data, ir, schema)
    return data, metadata


def save_graph(
    graph: Any,
    metadata: Dict[str, Any],
    out_dir: str = "results",
    out_name: str = "hetero_graph.pt",
) -> Tuple[str, str]:
    """
    Serialize the graph to disk and save companion metadata JSON.

    Returns (graph_path, metadata_path)
    """
    _ensure_dir(out_dir)
    graph_path = os.path.join(out_dir, out_name)

    # Save graph
    torch_lib = _ensure_torch()
    torch_lib.save(graph, graph_path)

    # Save metadata JSON
    meta_path = os.path.splitext(graph_path)[0] + ".meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    return graph_path, meta_path


def build_graph(
    *,
    misp_events: Optional[List[dict]] = None,
    misp_json_path: Optional[str] = None,
    out_dir: str = "results",
    out_name: Optional[str] = None,
    schema: Optional[GraphSchema] = None,
    exclude_nodes: Optional[list[NodeType]] = None,
) -> Tuple[Any, str, str]:
    """
    Convenience entrypoint.

    One of misp_events or misp_json_path must be provided.
    Returns (graph, graph_path, metadata_path)
    """
    if misp_events is None and misp_json_path is None:
        raise ValueError("Provide either misp_events (in-memory) or misp_json_path (file path).")

    if misp_events is None:
        misp_events = _load_misp_json(misp_json_path)  # type: ignore[arg-type]

    graph, metadata = build_hetero_graph_from_misp(misp_events, schema=schema, exclude_nodes=exclude_nodes)

    # Determine output name
    if out_name is None:
        # Derive from file name if available
        if misp_json_path:
            base = os.path.splitext(os.path.basename(misp_json_path))[0]
            out_name = f"{base}_hetero.pt"
        else:
            out_name = "hetero_graph.pt"

    graph_path, meta_path = save_graph(graph, metadata, out_dir=out_dir, out_name=out_name)
    return graph, graph_path, meta_path


def load_graph(graph_path: str) -> Any:
    """Load a serialized HeteroData graph (.pt)."""
    torch_lib = _ensure_torch()
    return torch_lib.load(graph_path, weights_only=False)


__all__ = [
    "build_hetero_graph_from_misp",
    "build_graph",
    "save_graph",
    "load_graph",
]
