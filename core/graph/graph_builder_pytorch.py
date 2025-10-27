"""
Graph builder for PyTorch Geometric Heterogeneous graphs from MISP JSON.

Capabilities:
- Accepts input either as an in-memory list of MISP events or from a JSON file path.
- Builds a HeteroData graph with email nodes as central hubs connected to component nodes.
- Node types: 'email', 'sender', 'receiver', 'week', 'subject', 'url', 'domain', 'stem', 'email_domain'.
- Edges:
  - ('email', 'has_sender', 'sender')
  - ('email', 'has_receiver', 'receiver')
  - ('email', 'in_week', 'week') - emails are grouped by ISO week
  - ('email', 'has_subject', 'subject')
  - ('email', 'has_url', 'url')
  - ('url', 'has_domain', 'domain')
  - ('url', 'has_stem', 'stem')
  - ('sender', 'from_domain', 'email_domain')
  - ('receiver', 'from_domain', 'email_domain')
- Component nodes are deduplicated: multiple emails sharing the same sender, week, subject, etc. 
  will have edges to the same component node.
- URLs are parsed into domain and stem components for better deduplication.
- Email addresses are normalized (lowercase, angle brackets removed) and connected to their 
  domain nodes (email_domain) to increase graph connectivity.
- Email features include body length. Week nodes group emails by ISO calendar week.
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


def build_hetero_graph_from_misp(
    misp_events: List[dict],
    *,
    schema: Optional[GraphSchema] = None,
) -> Tuple[Any, Dict[str, Any]]:
    """
    Build a HeteroData graph from a list of MISP events.
    
    New schema: Email nodes are central hubs connected to component nodes:
    - Node types: email, sender, receiver, week, subject, url, domain, stem, email_domain
    - Edge types: 
      - (email, has_sender, sender)
      - (email, has_receiver, receiver)
      - (email, in_week, week)
      - (email, has_subject, subject)
      - (email, has_url, url)
      - (url, has_domain, domain)
      - (url, has_stem, stem)
      - (sender, from_domain, email_domain)
      - (receiver, from_domain, email_domain)
    
    Components are deduplicated: multiple emails sharing the same sender/receiver/week/etc. 
    will have edges to the same component node. URLs are decomposed into domain and stem.
    Email addresses are normalized (lowercase, angle brackets removed) and connected to 
    their domain nodes to increase connectivity.
    
    Email features include body length as a numeric feature.

    Returns (graph, metadata) where metadata contains mappings for node indices.
    """
    schema = schema or DEFAULT_SCHEMA
    # Resolve node labels per backend
    N = schema.nodes  # shorthand

    N = schema.nodes  # shorthand
    ir = assemble_misp_graph_ir(misp_events, schema=schema)

    # Pull ordered features and meta from IR
    email_x = ir.nodes["email"].x
    sender_x = ir.nodes["sender"].x
    receiver_x = ir.nodes["receiver"].x
    week_x = ir.nodes["week"].x
    subject_x = ir.nodes["subject"].x
    url_x = ir.nodes["url"].x
    domain_x = ir.nodes["domain"].x
    stem_x = ir.nodes["stem"].x
    email_domain_x = ir.nodes["email_domain"].x

    HData = _ensure_heterodata()
    torch_lib = _ensure_torch()
    data = HData()

    # Set node features for email and attach additional attributes
    if email_x:
        data[N["email"].pyg].x = torch_lib.tensor(email_x, dtype=torch_lib.float)
        # Attach requested attributes
        attrs = ir.email_attrs
        if attrs:
            if attrs.get("x_text"):
                data[N["email"].pyg]["x_text"] = torch_lib.tensor(attrs["x_text"], dtype=torch_lib.float)
            if attrs.get("ts") is not None:
                data[N["email"].pyg]["ts"] = torch_lib.tensor(attrs["ts"], dtype=torch_lib.int64)
            if attrs.get("n_urls") is not None:
                data[N["email"].pyg]["n_urls"] = torch_lib.tensor(attrs["n_urls"], dtype=torch_lib.int16)
            if attrs.get("len_subject") is not None:
                data[N["email"].pyg]["len_subject"] = torch_lib.tensor(attrs["len_subject"], dtype=torch_lib.int32)
            if attrs.get("len_body") is not None:
                data[N["email"].pyg]["len_body"] = torch_lib.tensor(attrs["len_body"], dtype=torch_lib.int32)
    else:
        data[N["email"].pyg].num_nodes = 0

    # Set node features for each component type
    if sender_x:
        data[N["sender"].pyg].x = torch_lib.tensor(sender_x, dtype=torch_lib.float)
    else:
        data[N["sender"].pyg].num_nodes = 0

    if receiver_x:
        data[N["receiver"].pyg].x = torch_lib.tensor(receiver_x, dtype=torch_lib.float)
    else:
        data[N["receiver"].pyg].num_nodes = 0

    if week_x:
        data[N["week"].pyg].x = torch_lib.tensor(week_x, dtype=torch_lib.float)
    else:
        data[N["week"].pyg].num_nodes = 0

    if subject_x:
        data[N["subject"].pyg].x = torch_lib.tensor(subject_x, dtype=torch_lib.float)
    else:
        data[N["subject"].pyg].num_nodes = 0

    if url_x:
        data[N["url"].pyg].x = torch_lib.tensor(url_x, dtype=torch_lib.float)
    else:
        data[N["url"].pyg].num_nodes = 0

    if domain_x:
        data[N["domain"].pyg].x = torch_lib.tensor(domain_x, dtype=torch_lib.float)
    else:
        data[N["domain"].pyg].num_nodes = 0

    if stem_x:
        data[N["stem"].pyg].x = torch_lib.tensor(stem_x, dtype=torch_lib.float)
    else:
        data[N["stem"].pyg].num_nodes = 0
    
    if email_domain_x:
        data[N["email_domain"].pyg].x = torch_lib.tensor(email_domain_x, dtype=torch_lib.float)
    else:
        data[N["email_domain"].pyg].num_nodes = 0

    # Set edges from IR
    def set_edges(edge_key: str):
        e = schema.edge(edge_key)
        src, dst = ir.edges[edge_key]
        if src:
            data[N[e.src].pyg, e.rel_pyg, N[e.dst].pyg].edge_index = torch_lib.tensor(
                [src, dst], dtype=torch_lib.long
            )

    for ek in [
        "has_sender",
        "has_receiver",
        "in_week",
        "has_subject",
        "has_url",
        "url_has_domain",
        "url_has_stem",
        "sender_from_domain",
        "receiver_from_domain",
    ]:
        set_edges(ek)

    # Ordered meta arrays from IR
    sender_meta = ir.nodes["sender"].index_to_string or []
    receiver_meta = ir.nodes["receiver"].index_to_string or []
    week_meta = ir.nodes["week"].index_to_string or []
    subject_meta = ir.nodes["subject"].index_to_string or []
    url_meta = ir.nodes["url"].index_to_string or []
    domain_meta = ir.nodes["domain"].index_to_string or []
    stem_meta = ir.nodes["stem"].index_to_string or []
    email_domain_meta = ir.nodes["email_domain"].index_to_string or []
    email_meta = ir.nodes["email"].index_to_meta or []

    # Attribute shapes for email (optional diagnostics)
    email_attr_shapes: Dict[str, List[int]] = {}
    if "x_text" in data[N["email"].pyg]:
        email_attr_shapes["x_text"] = list(data[N["email"].pyg]["x_text"].shape)
    if "ts" in data[N["email"].pyg]:
        email_attr_shapes["ts"] = list(data[N["email"].pyg]["ts"].shape)
    if "n_urls" in data[N["email"].pyg]:
        email_attr_shapes["n_urls"] = list(data[N["email"].pyg]["n_urls"].shape)
    if "len_subject" in data[N["email"].pyg]:
        email_attr_shapes["len_subject"] = list(data[N["email"].pyg]["len_subject"].shape)
    if "len_body" in data[N["email"].pyg]:
        email_attr_shapes["len_body"] = list(data[N["email"].pyg]["len_body"].shape)

    metadata = {
        "node_maps": {
            N["email"].pyg: {"index_to_meta": email_meta},
            N["sender"].pyg: {"index_to_string": sender_meta},
            N["receiver"].pyg: {"index_to_string": receiver_meta},
            N["week"].pyg: {"index_to_string": week_meta},
            N["subject"].pyg: {"index_to_string": subject_meta},
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
            N["subject"].pyg: list(data[N["subject"].pyg].x.shape) if "x" in data[N["subject"].pyg] else [0, 0],
            N["url"].pyg: list(data[N["url"].pyg].x.shape) if "x" in data[N["url"].pyg] else [0, 0],
            N["domain"].pyg: list(data[N["domain"].pyg].x.shape) if "x" in data[N["domain"].pyg] else [0, 0],
            N["stem"].pyg: list(data[N["stem"].pyg].x.shape) if "x" in data[N["stem"].pyg] else [0, 0],
            N["email_domain"].pyg: list(data[N["email_domain"].pyg].x.shape) if "x" in data[N["email_domain"].pyg] else [0, 0],
        },
        "email_attr_shapes": email_attr_shapes,
        "edge_counts": {
            f"{N['email'].pyg}->{N['sender'].pyg}:{schema.edge('has_sender').rel_pyg}": len(ir.edges['has_sender'][0]),
            f"{N['email'].pyg}->{N['receiver'].pyg}:{schema.edge('has_receiver').rel_pyg}": len(ir.edges['has_receiver'][0]),
            f"{N['email'].pyg}->{N['week'].pyg}:{schema.edge('in_week').rel_pyg}": len(ir.edges['in_week'][0]),
            f"{N['email'].pyg}->{N['subject'].pyg}:{schema.edge('has_subject').rel_pyg}": len(ir.edges['has_subject'][0]),
            f"{N['email'].pyg}->{N['url'].pyg}:{schema.edge('has_url').rel_pyg}": len(ir.edges['has_url'][0]),
            f"{N['url'].pyg}->{N['domain'].pyg}:{schema.edge('url_has_domain').rel_pyg}": len(ir.edges['url_has_domain'][0]),
            f"{N['url'].pyg}->{N['stem'].pyg}:{schema.edge('url_has_stem').rel_pyg}": len(ir.edges['url_has_stem'][0]),
            f"{N['sender'].pyg}->{N['email_domain'].pyg}:{schema.edge('sender_from_domain').rel_pyg}": len(ir.edges['sender_from_domain'][0]),
            f"{N['receiver'].pyg}->{N['email_domain'].pyg}:{schema.edge('receiver_from_domain').rel_pyg}": len(ir.edges['receiver_from_domain'][0]),
        },
    }

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

    graph, metadata = build_hetero_graph_from_misp(misp_events, schema=schema)

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
