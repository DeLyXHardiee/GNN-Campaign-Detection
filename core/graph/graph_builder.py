"""
Graph builder for PyTorch Geometric Heterogeneous graphs from MISP JSON.

Capabilities:
- Accepts input either as an in-memory list of MISP events or from a JSON file path.
- Builds a HeteroData graph with node types: 'email', 'address', 'url'.
- Adds edges:
  - ('address', 'sends', 'email') for sender relationships
  - ('address', 'receives', 'email') for receiver relationships
  - ('email', 'contains', 'url') for URL references
- Creates simple numeric features for nodes (lengths and counts) to keep tensors valid.
- Saves both the graph (.pt via torch.save) and a companion metadata JSON mapping node indices to original strings.

If torch or torch_geometric are not installed, import errors will clearly explain how to install them.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple, Any, TYPE_CHECKING

if TYPE_CHECKING:  # For type checkers only; avoids runtime import requirement
    import torch  # type: ignore
    from torch_geometric.data import HeteroData  # type: ignore
else:
    torch = None  # type: ignore
    HeteroData = Any  # type: ignore


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


def _to_str(val: Any) -> str:
    """Convert value to a safe string, returning empty for None/NaN."""
    if isinstance(val, str):
        return val
    if val is None:
        return ""
    # Handle NaN
    try:
        if isinstance(val, float) and val != val:
            return ""
    except Exception:
        return ""
    try:
        return str(val)
    except Exception:
        return ""


def _parse_misp_events(misp_events: List[dict]) -> List[Dict[str, Any]]:
    """
    Normalize MISP events to a simpler structure per email event.

    Output list of dicts with keys:
    - email_info: str
    - sender: Optional[str]
    - receivers: List[str]
    - subject: str
    - body: str
    - urls: List[str]
    - date: str
    """
    normalized: List[Dict[str, Any]] = []
    for ev in misp_events:
        event = ev.get("Event", {})
        info = event.get("info", "")
        attrs = event.get("Attribute", []) or []

        sender = None
        receivers: List[str] = []
        subject = ""
        body = ""
        urls: List[str] = []
        date = ""

        for attr in attrs:
            a_type = (attr or {}).get("type", "")
            raw_val = (attr or {}).get("value", "")
            val = _to_str(raw_val)
            if a_type == "email-src":
                sender = val if val.strip() else None
            elif a_type == "email-dst":
                if val.strip():
                    receivers.append(val)
            elif a_type == "email-subject":
                subject = val
            elif a_type == "email-body":
                body = val
            elif a_type == "url":
                if val.strip():
                    urls.append(val)
            elif a_type == "email-date":
                date = val

        normalized.append(
            {
                "email_info": info,
                "sender": sender,
                "receivers": receivers,
                "subject": subject,
                "body": body,
                "urls": urls,
                "date": date,
            }
        )

    return normalized


def build_hetero_graph_from_misp(
    misp_events: List[dict],
) -> Tuple[Any, Dict[str, Any]]:
    """
    Build a HeteroData graph from a list of MISP events.

    Returns (graph, metadata) where metadata contains mappings for node indices.
    """
    emails = _parse_misp_events(misp_events)

    # Unique entities and indices
    address_to_idx: Dict[str, int] = {}
    url_to_idx: Dict[str, int] = {}

    # Pre-scan to register unique addresses and urls
    for em in emails:
        if em.get("sender"):
            address_to_idx.setdefault(em["sender"], len(address_to_idx))
        for r in em.get("receivers", []) or []:
            address_to_idx.setdefault(r, len(address_to_idx))
        for u in em.get("urls", []) or []:
            url_to_idx.setdefault(u, len(url_to_idx))

    # Node features
    email_x: List[List[float]] = []
    address_x: List[List[float]] = [[float(len(str(addr)))] for addr in sorted(address_to_idx, key=lambda x: address_to_idx[x])]
    url_x: List[List[float]] = [[float(len(str(u)))] for u in sorted(url_to_idx, key=lambda x: url_to_idx[x])]

    # Edges
    sends_src: List[int] = []  # address -> email
    sends_dst: List[int] = []
    receives_src: List[int] = []  # address -> email (receivers)
    receives_dst: List[int] = []
    contains_src: List[int] = []  # email -> url
    contains_dst: List[int] = []

    # Meta maps
    email_meta: List[Dict[str, Any]] = []
    address_meta: List[str] = [None] * len(address_to_idx)
    for addr, idx in address_to_idx.items():
        address_meta[idx] = addr
    url_meta: List[str] = [None] * len(url_to_idx)
    for u, idx in url_to_idx.items():
        url_meta[idx] = u

    for email_idx, em in enumerate(emails):
        subj_len = float(len(em.get("subject", "")))
        body_len = float(len(em.get("body", "")))
        num_urls = float(len(em.get("urls", []) or []))
        num_receivers = float(len(em.get("receivers", []) or []))
        email_x.append([subj_len, body_len, num_urls, num_receivers])

        # sender edge
        if em.get("sender"):
            a_idx = address_to_idx[em["sender"]]
            sends_src.append(a_idx)
            sends_dst.append(email_idx)

        # receiver edges
        for r in em.get("receivers", []) or []:
            a_idx = address_to_idx[r]
            receives_src.append(a_idx)
            receives_dst.append(email_idx)

        # url edges
        for u in em.get("urls", []) or []:
            if u in url_to_idx:
                contains_src.append(email_idx)
                contains_dst.append(url_to_idx[u])

        email_meta.append(
            {
                "info": em.get("email_info", ""),
                "sender": em.get("sender"),
                "receivers": em.get("receivers", []),
                "date": em.get("date", ""),
            }
        )

    HData = _ensure_heterodata()
    torch_lib = _ensure_torch()
    data = HData()

    # Set node features
    if email_x:
        data["email"].x = torch_lib.tensor(email_x, dtype=torch_lib.float)
    else:
        data["email"].num_nodes = 0

    if address_x:
        data["address"].x = torch_lib.tensor(address_x, dtype=torch_lib.float)
    else:
        data["address"].num_nodes = 0

    if url_x:
        data["url"].x = torch_lib.tensor(url_x, dtype=torch_lib.float)
    else:
        data["url"].num_nodes = 0

    # Set edges
    if sends_src:
        data["address", "sends", "email"].edge_index = torch_lib.tensor(
            [sends_src, sends_dst], dtype=torch_lib.long
        )
    if receives_src:
        data["address", "receives", "email"].edge_index = torch_lib.tensor(
            [receives_src, receives_dst], dtype=torch_lib.long
        )
    if contains_src:
        data["email", "contains", "url"].edge_index = torch_lib.tensor(
            [contains_src, contains_dst], dtype=torch_lib.long
        )

    metadata = {
        "node_maps": {
            "email": {"index_to_meta": email_meta},
            "address": {"index_to_string": address_meta},
            "url": {"index_to_string": url_meta},
        },
        "feature_shapes": {
            "email": list(data["email"].x.shape) if "x" in data["email"] else [0, 0],
            "address": list(data["address"].x.shape) if "x" in data["address"] else [0, 0],
            "url": list(data["url"].x.shape) if "x" in data["url"] else [0, 0],
        },
        "edge_counts": {
            "address->email:sends": len(sends_src),
            "address->email:receives": len(receives_src),
            "email->url:contains": len(contains_src),
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

    graph, metadata = build_hetero_graph_from_misp(misp_events)

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
    return torch_lib.load(graph_path)


__all__ = [
    "build_hetero_graph_from_misp",
    "build_graph",
    "save_graph",
    "load_graph",
]
