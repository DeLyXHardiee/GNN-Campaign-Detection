"""
Graph builder for PyTorch Geometric Heterogeneous graphs from MISP JSON.

Capabilities:
- Accepts input either as an in-memory list of MISP events or from a JSON file path.
- Builds a HeteroData graph with email nodes as central hubs connected to component nodes.
- Node types: 'email', 'sender', 'receiver', 'week', 'url', 'domain', 'stem', 'email_domain', 'attachment'.
- Edges:
    - ('email', 'has_sender', 'sender')
    - ('email', 'has_receiver', 'receiver')
    - ('email', 'in_week', 'week') - emails are grouped by ISO week
    - ('email', 'has_url', 'url')
    - ('url', 'has_domain', 'domain')
    - ('url', 'has_stem', 'stem')
    - ('sender', 'from_domain', 'email_domain')
    - ('receiver', 'from_domain', 'email_domain')
    - ('email', 'has_attachment', 'attachment')
- Component nodes are deduplicated: multiple emails sharing the same sender, week, etc. 
    will have edges to the same component node.
- URLs are parsed into domain and stem components for better deduplication.
- Email addresses are normalized (lowercase, angle brackets removed) and connected to their 
    domain nodes (email_domain) to increase graph connectivity.
- Email features include normalized scalars (ts_minmax, len_body_z, n_urls_z, len_subject_z) and optional SBERT embeddings.
- Creates simple numeric features for nodes (lengths) to keep tensors valid.
- Saves both the graph (.pt via torch.save) and a companion metadata JSON mapping node indices to original strings.
"""
from __future__ import annotations

import json
import os
import sys
from typing import Dict, List, Optional, Tuple, Any, TYPE_CHECKING

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .graph_schema import GraphSchema, DEFAULT_SCHEMA
from .assembler import assemble_misp_graph_ir
from .graph_filter import NodeType, filter_graph_ir
from .normalizer import normalize_graph
from .feature_projection import (
    SCALAR_COUNT,
    HTML_CSS_LEN,
    BOOL_ATTR_COUNT,
    AUTH_ONEHOT_DIM,
    EmailFeatureProjectionModule,
)
from preprocessing.utils.defang import sanitize_for_json

# Fixed seed for email feature projection so graph builds are reproducible
_EMAIL_PROJECTION_SEED = 42


if TYPE_CHECKING:  
    import torch 
    from torch_geometric.data import HeteroData
else:
    torch = None 
    HeteroData = Any

def _ensure_torch():
    global torch
    if torch is None:
        try:
            import torch as _torch
        except Exception as e:
            raise ImportError(
                "PyTorch is required. Install with: pip install torch --index-url https://download.pytorch.org/whl/cpu"
            ) from e
        torch = _torch  
    return torch


def _ensure_heterodata():
    global HeteroData
    if HeteroData is Any:
        try:
            from torch_geometric.data import HeteroData as _HeteroData 
        except Exception as e:
            raise ImportError(
                "PyTorch Geometric is required. Install with: pip install torch-geometric"
            ) from e
        HeteroData = _HeteroData 
    return HeteroData


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _load_misp_json(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _merge_features_with_attrs(base: List[List[float]], attr_vals: Dict[str, Any], keys: List[str]) -> List[List[float]]:
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


def _infer_email_embedding_dims(total_dim: int) -> Tuple[int, int]:
    """Infer subj_dim and body_dim from raw email feature dimension (layout from feature_projection)."""
    text_dim = total_dim - SCALAR_COUNT - HTML_CSS_LEN - BOOL_ATTR_COUNT - AUTH_ONEHOT_DIM
    if text_dim <= 0:
        return 0, 0
    half = text_dim // 2
    return half, half


def _set_node_features_from_ir(data: Any, ir: Any, schema: GraphSchema) -> None:
    _ensure_heterodata()
    torch_lib = _ensure_torch()
    N = schema.nodes

    if "email" not in ir.nodes:
        data[N["email"].pyg].num_nodes = 0
        return
    email_x = ir.nodes["email"].x
    if email_x:
        raw = torch_lib.tensor(email_x, dtype=torch_lib.float)
        total_dim = raw.size(1)
        subj_dim, body_dim = _infer_email_embedding_dims(total_dim)
        # Apply balanced projection so BERT is down-projected and other features up-projected
        torch_lib.manual_seed(_EMAIL_PROJECTION_SEED)
        proj = EmailFeatureProjectionModule(
            subj_dim=subj_dim,
            body_dim=body_dim,
            bert_out_dim=128,
            other_out_dim=32,
        )
        data[N["email"].pyg].x = proj(raw)
        email_meta = ir.nodes["email"].index_to_meta or []
        data[N["email"].pyg].external_id = [
            str(m.get("external_id") or "") for m in email_meta
        ]
    else:
        data[N["email"].pyg].num_nodes = 0

    def set_simple(node_key: str, extra_keys: List[str] = None):
        if node_key not in ir.nodes:
            return
        x = ir.nodes[node_key].x
        if x:
            attrs = ir.nodes[node_key].attrs
            if attrs and extra_keys:
                x = _merge_features_with_attrs(x, attrs, extra_keys)
            data[N[node_key].pyg].x = torch_lib.tensor(x, dtype=torch_lib.float)
        else:
            data[N[node_key].pyg].num_nodes = 0

    for node_key, node_map in schema.nodes.items():
        if node_key == "email":
            continue
        set_simple(node_key, list(node_map.extra_attr_keys))



def _set_edges_from_ir(data: Any, ir: Any, schema: GraphSchema) -> None:
    torch_lib = _ensure_torch()
    N = schema.nodes

    def set_edges(edge_key: str):
        if edge_key not in ir.edges:
            return
        e = schema.edge(edge_key)
        src, dst = ir.edges[edge_key]
        if src:
            data[N[e.src].pyg, e.rel_pyg, N[e.dst].pyg].edge_index = torch_lib.tensor([src, dst], dtype=torch_lib.long)

    for ek in schema.edges:
        set_edges(ek)


def _build_metadata_from_ir(data: Any, ir: Any, schema: GraphSchema) -> Dict[str, Any]:
    """Construct the metadata dict summarizing node maps, feature shapes, and edge counts."""
    N = schema.nodes
    email_meta = (ir.nodes.get("email") and ir.nodes["email"].index_to_meta) or []
    node_maps: Dict[str, Dict[str, Any]] = {N["email"].pyg: {"index_to_meta": email_meta}}
    feature_shapes: Dict[str, List[int]] = {}
    edge_counts: Dict[str, int] = {}

    for node_key, node_map in schema.nodes.items():
        pyg_label = node_map.pyg
        if node_key != "email":
            meta = (ir.nodes.get(node_key) and ir.nodes[node_key].index_to_string) or []
            node_maps[pyg_label] = {"index_to_string": meta}
        feature_shapes[pyg_label] = list(data[pyg_label].x.shape) if "x" in data[pyg_label] else [0, 0]

    for edge_key, edge in schema.edges.items():
        src_label = N[edge.src].pyg
        dst_label = N[edge.dst].pyg
        count_key = f"{src_label}->{dst_label}:{edge.rel_pyg}"
        edge_counts[count_key] = len(ir.edges.get(edge_key, ([], []))[0])

    meta: Dict[str, Any] = {
        "node_maps": node_maps,
        "feature_shapes": feature_shapes,
        "edge_counts": edge_counts,
    }
    if getattr(ir, "email_attrs", None):
        meta["email_attrs"] = ir.email_attrs
    return meta


def build_hetero_graph_from_misp(
    misp_events: List[dict],
    *,
    schema: Optional[GraphSchema] = None,
    exclude_nodes: Optional[list[NodeType | str]] = None,
    embeddings_output_dir: Optional[str] = None,
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
    n_urls_z, len_subject_z, and optional SBERT embeddings of subject/body.

    Returns (graph, metadata) where metadata contains mappings for node indices.
    """
    schema = schema or DEFAULT_SCHEMA
    N = schema.nodes
    ir = assemble_misp_graph_ir(
        misp_events,
        schema=schema,
        embeddings_output_dir=embeddings_output_dir,
    )
    if exclude_nodes:
        ir = filter_graph_ir(ir, exclude_nodes=NodeType.canonical_set(exclude_nodes, schema=schema), schema=schema)

    HData = _ensure_heterodata()
    data = HData()

    _set_node_features_from_ir(data, ir, schema)
    _set_edges_from_ir(data, ir, schema)

    data = normalize_graph(data)
    
    metadata = _build_metadata_from_ir(data, ir, schema)
    return data, metadata


def save_graph(
    graph: Any,
    metadata: Dict[str, Any],
    out_dir: str = "results",
    out_name: str = "hetero_graph.pt",
) -> Tuple[str, str]:
    
    _ensure_dir(out_dir)
    graph_path = os.path.join(out_dir, out_name)

    torch_lib = _ensure_torch()
    torch_lib.save(graph, graph_path)

    meta_path = os.path.splitext(graph_path)[0] + ".meta.json"
    metadata_sanitized = sanitize_for_json(metadata)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata_sanitized, f, indent=2, ensure_ascii=False)

    return graph_path, meta_path


def build_graph(
    *,
    misp_events: Optional[List[dict]] = None,
    misp_json_path: Optional[str] = None,
    out_dir: str = "results",
    out_name: Optional[str] = None,
    schema: Optional[GraphSchema] = None,
    exclude_nodes: Optional[list[NodeType | str]] = None,
    embeddings_output_dir: Optional[str] = None,
) -> Tuple[Any, str, str]:
   
    if misp_events is None and misp_json_path is None:
        raise ValueError("Provide either misp_events (in-memory) or misp_json_path (file path).")

    if misp_events is None:
        misp_events = _load_misp_json(misp_json_path)

    graph, metadata = build_hetero_graph_from_misp(
        misp_events,
        schema=schema,
        exclude_nodes=exclude_nodes,
        embeddings_output_dir=embeddings_output_dir,
    )

    if out_name is None:
        if misp_json_path:
            base = os.path.splitext(os.path.basename(misp_json_path))[0]
            out_name = f"{base}_hetero.pt"
        else:
            out_name = "hetero_graph.pt"

    graph_path, meta_path = save_graph(graph, metadata, out_dir=out_dir, out_name=out_name)
    return graph, graph_path, meta_path


def load_graph(graph_path: str) -> Any:
    torch_lib = _ensure_torch()
    return torch_lib.load(graph_path, weights_only=False)


__all__ = [
    "build_hetero_graph_from_misp",
    "build_graph",
    "save_graph",
    "load_graph",
]
