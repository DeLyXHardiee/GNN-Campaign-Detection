"""
Graph builder for PyTorch Geometric Heterogeneous graphs from MISP JSON.

Capabilities:
- Accepts input either as an in-memory list of MISP events or from a JSON file path.
- Builds a HeteroData graph with email nodes as central hubs connected to component nodes.
- Node types: 'email', 'sender', 'receiver', 'week', 'subject', 'url', 'domain', 'stem'.
- Edges (all from email to components):
  - ('email', 'has_sender', 'sender')
  - ('email', 'has_receiver', 'receiver')
  - ('email', 'in_week', 'week') - emails are grouped by ISO week
  - ('email', 'has_subject', 'subject')
  - ('email', 'has_url', 'url')
  - ('url', 'has_domain', 'domain')
  - ('url', 'has_stem', 'stem')
- Component nodes are deduplicated: multiple emails sharing the same sender, week, subject, etc. 
  will have edges to the same component node.
- URLs are parsed into domain and stem components for better deduplication.
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

# Add utils to path for URL extractor
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from utils.url_extractor import parse_url_components
except ImportError:
    # Fallback if import fails
    def parse_url_components(url: str) -> Dict[str, Any]:
        return {"full_url": url, "domain": "", "stem": "", "scheme": ""}

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


def _extract_week_key(date_str: str) -> Optional[str]:
    """
    Extract a week identifier from a date string.
    
    Returns: "YYYY-Www" format (e.g., "2007-W15") for ISO week, or None if parsing fails.
    """
    if not date_str or not date_str.strip():
        return None
    
    try:
        from datetime import datetime
        # Common email date formats - try parsing various patterns
        for fmt in [
            "%a, %d %b %Y %H:%M:%S %z",  # Standard RFC 2822: Sun, 08 Apr 2007 21:00:48 +0300
            "%a, %d %b %Y %H:%M:%S %Z",
            "%d %b %Y %H:%M:%S %z",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d",
        ]:
            try:
                dt = datetime.strptime(date_str.strip(), fmt)
                # Get ISO calendar: (year, week_number, weekday)
                iso_cal = dt.isocalendar()
                return f"{iso_cal[0]}-W{iso_cal[1]:02d}"
            except ValueError:
                continue
        
        # If all formats fail, return None
        return None
    except Exception:
        return None


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
    
    New schema: Email nodes are central hubs connected to component nodes:
    - Node types: email, sender, receiver, week, subject, url, domain, stem
    - Edge types: 
      - (email, has_sender, sender)
      - (email, has_receiver, receiver)
      - (email, in_week, week)
      - (email, has_subject, subject)
      - (email, has_url, url)
      - (url, has_domain, domain)
      - (url, has_stem, stem)
    
    Components are deduplicated: multiple emails sharing the same sender/receiver/week/etc. 
    will have edges to the same component node. URLs are decomposed into domain and stem.
    
    Email features include body length as a numeric feature.

    Returns (graph, metadata) where metadata contains mappings for node indices.
    """
    emails = _parse_misp_events(misp_events)

    # Unique entities and indices for each component type
    sender_to_idx: Dict[str, int] = {}
    receiver_to_idx: Dict[str, int] = {}
    week_to_idx: Dict[str, int] = {}
    subject_to_idx: Dict[str, int] = {}
    url_to_idx: Dict[str, int] = {}
    domain_to_idx: Dict[str, int] = {}
    stem_to_idx: Dict[str, int] = {}
    
    # Track URL -> (domain, stem) mapping for edge creation
    url_components: Dict[str, Tuple[str, str]] = {}

    # Pre-scan to register unique components
    for em in emails:
        sender = em.get("sender")
        if sender:
            sender_to_idx.setdefault(sender, len(sender_to_idx))
        
        for r in em.get("receivers", []) or []:
            if r:
                receiver_to_idx.setdefault(r, len(receiver_to_idx))
        
        # Extract week from date
        date_val = em.get("date", "")
        week_key = _extract_week_key(date_val)
        if week_key:
            week_to_idx.setdefault(week_key, len(week_to_idx))
        
        subj = em.get("subject", "")
        if subj:
            subject_to_idx.setdefault(subj, len(subject_to_idx))
        
        # Parse URLs into components
        for u in em.get("urls", []) or []:
            if u:
                url_to_idx.setdefault(u, len(url_to_idx))
                
                # Parse URL components
                parsed = parse_url_components(u)
                domain = parsed.get("domain", "")
                stem = parsed.get("stem", "")
                
                if domain:
                    domain_to_idx.setdefault(domain, len(domain_to_idx))
                if stem:
                    stem_to_idx.setdefault(stem, len(stem_to_idx))
                
                # Store mapping
                url_components[u] = (domain, stem)

    # Node features (simple length-based for now)
    email_x: List[List[float]] = []
    sender_x: List[List[float]] = [[float(len(str(s)))] for s in sorted(sender_to_idx, key=lambda x: sender_to_idx[x])]
    receiver_x: List[List[float]] = [[float(len(str(r)))] for r in sorted(receiver_to_idx, key=lambda x: receiver_to_idx[x])]
    week_x: List[List[float]] = [[float(idx)] for idx in range(len(week_to_idx))]  # Week index as feature
    subject_x: List[List[float]] = [[float(len(str(s)))] for s in sorted(subject_to_idx, key=lambda x: subject_to_idx[x])]
    url_x: List[List[float]] = [[float(len(str(u)))] for u in sorted(url_to_idx, key=lambda x: url_to_idx[x])]
    domain_x: List[List[float]] = [[float(len(str(d)))] for d in sorted(domain_to_idx, key=lambda x: domain_to_idx[x])]
    stem_x: List[List[float]] = [[float(len(str(s)))] for s in sorted(stem_to_idx, key=lambda x: stem_to_idx[x])]

    # Edge lists (email -> component)
    has_sender_src: List[int] = []
    has_sender_dst: List[int] = []
    has_receiver_src: List[int] = []
    has_receiver_dst: List[int] = []
    in_week_src: List[int] = []
    in_week_dst: List[int] = []
    has_subject_src: List[int] = []
    has_subject_dst: List[int] = []
    has_url_src: List[int] = []
    has_url_dst: List[int] = []
    
    # URL -> domain/stem edges
    url_to_domain_src: List[int] = []
    url_to_domain_dst: List[int] = []
    url_to_stem_src: List[int] = []
    url_to_stem_dst: List[int] = []

    # Meta maps for reverse lookup
    sender_meta: List[str] = [None] * len(sender_to_idx)
    for s, idx in sender_to_idx.items():
        sender_meta[idx] = s
    receiver_meta: List[str] = [None] * len(receiver_to_idx)
    for r, idx in receiver_to_idx.items():
        receiver_meta[idx] = r
    week_meta: List[str] = [None] * len(week_to_idx)
    for w, idx in week_to_idx.items():
        week_meta[idx] = w
    subject_meta: List[str] = [None] * len(subject_to_idx)
    for s, idx in subject_to_idx.items():
        subject_meta[idx] = s
    url_meta: List[str] = [None] * len(url_to_idx)
    for u, idx in url_to_idx.items():
        url_meta[idx] = u
    domain_meta: List[str] = [None] * len(domain_to_idx)
    for d, idx in domain_to_idx.items():
        domain_meta[idx] = d
    stem_meta: List[str] = [None] * len(stem_to_idx)
    for s, idx in stem_to_idx.items():
        stem_meta[idx] = s

    # Email metadata
    email_meta: List[Dict[str, Any]] = []

    # Build edges: iterate emails and create edges to their components
    for email_idx, em in enumerate(emails):
        # Email node feature: [body_length]
        body_len = float(len(em.get("body", "")))
        email_x.append([body_len])

        email_meta.append({
            "info": em.get("email_info", ""),
            "index": email_idx,
            "date": em.get("date", ""),
        })

        # has_sender edge
        if em.get("sender") and em["sender"] in sender_to_idx:
            has_sender_src.append(email_idx)
            has_sender_dst.append(sender_to_idx[em["sender"]])

        # has_receiver edges
        for r in em.get("receivers", []) or []:
            if r and r in receiver_to_idx:
                has_receiver_src.append(email_idx)
                has_receiver_dst.append(receiver_to_idx[r])

        # in_week edge
        date_val = em.get("date", "")
        week_key = _extract_week_key(date_val)
        if week_key and week_key in week_to_idx:
            in_week_src.append(email_idx)
            in_week_dst.append(week_to_idx[week_key])

        # has_subject edge
        subj = em.get("subject", "")
        if subj and subj in subject_to_idx:
            has_subject_src.append(email_idx)
            has_subject_dst.append(subject_to_idx[subj])

        # has_url edges
        for u in em.get("urls", []) or []:
            if u and u in url_to_idx:
                has_url_src.append(email_idx)
                has_url_dst.append(url_to_idx[u])

    # Create URL -> domain and URL -> stem edges
    for url, (domain, stem) in url_components.items():
        url_idx = url_to_idx[url]
        
        if domain and domain in domain_to_idx:
            url_to_domain_src.append(url_idx)
            url_to_domain_dst.append(domain_to_idx[domain])
        
        if stem and stem in stem_to_idx:
            url_to_stem_src.append(url_idx)
            url_to_stem_dst.append(stem_to_idx[stem])

    HData = _ensure_heterodata()
    torch_lib = _ensure_torch()
    data = HData()

    # Set node features for email
    if email_x:
        data["email"].x = torch_lib.tensor(email_x, dtype=torch_lib.float)
    else:
        data["email"].num_nodes = 0

    # Set node features for each component type
    if sender_x:
        data["sender"].x = torch_lib.tensor(sender_x, dtype=torch_lib.float)
    else:
        data["sender"].num_nodes = 0

    if receiver_x:
        data["receiver"].x = torch_lib.tensor(receiver_x, dtype=torch_lib.float)
    else:
        data["receiver"].num_nodes = 0

    if week_x:
        data["week"].x = torch_lib.tensor(week_x, dtype=torch_lib.float)
    else:
        data["week"].num_nodes = 0

    if subject_x:
        data["subject"].x = torch_lib.tensor(subject_x, dtype=torch_lib.float)
    else:
        data["subject"].num_nodes = 0

    if url_x:
        data["url"].x = torch_lib.tensor(url_x, dtype=torch_lib.float)
    else:
        data["url"].num_nodes = 0

    if domain_x:
        data["domain"].x = torch_lib.tensor(domain_x, dtype=torch_lib.float)
    else:
        data["domain"].num_nodes = 0

    if stem_x:
        data["stem"].x = torch_lib.tensor(stem_x, dtype=torch_lib.float)
    else:
        data["stem"].num_nodes = 0

    # Set edges
    if has_sender_src:
        data["email", "has_sender", "sender"].edge_index = torch_lib.tensor(
            [has_sender_src, has_sender_dst], dtype=torch_lib.long
        )
    if has_receiver_src:
        data["email", "has_receiver", "receiver"].edge_index = torch_lib.tensor(
            [has_receiver_src, has_receiver_dst], dtype=torch_lib.long
        )
    if in_week_src:
        data["email", "in_week", "week"].edge_index = torch_lib.tensor(
            [in_week_src, in_week_dst], dtype=torch_lib.long
        )
    if has_subject_src:
        data["email", "has_subject", "subject"].edge_index = torch_lib.tensor(
            [has_subject_src, has_subject_dst], dtype=torch_lib.long
        )
    if has_url_src:
        data["email", "has_url", "url"].edge_index = torch_lib.tensor(
            [has_url_src, has_url_dst], dtype=torch_lib.long
        )
    if url_to_domain_src:
        data["url", "has_domain", "domain"].edge_index = torch_lib.tensor(
            [url_to_domain_src, url_to_domain_dst], dtype=torch_lib.long
        )
    if url_to_stem_src:
        data["url", "has_stem", "stem"].edge_index = torch_lib.tensor(
            [url_to_stem_src, url_to_stem_dst], dtype=torch_lib.long
        )

    metadata = {
        "node_maps": {
            "email": {"index_to_meta": email_meta},
            "sender": {"index_to_string": sender_meta},
            "receiver": {"index_to_string": receiver_meta},
            "week": {"index_to_string": week_meta},
            "subject": {"index_to_string": subject_meta},
            "url": {"index_to_string": url_meta},
            "domain": {"index_to_string": domain_meta},
            "stem": {"index_to_string": stem_meta},
        },
        "feature_shapes": {
            "email": list(data["email"].x.shape) if "x" in data["email"] else [0, 0],
            "sender": list(data["sender"].x.shape) if "x" in data["sender"] else [0, 0],
            "receiver": list(data["receiver"].x.shape) if "x" in data["receiver"] else [0, 0],
            "week": list(data["week"].x.shape) if "x" in data["week"] else [0, 0],
            "subject": list(data["subject"].x.shape) if "x" in data["subject"] else [0, 0],
            "url": list(data["url"].x.shape) if "x" in data["url"] else [0, 0],
            "domain": list(data["domain"].x.shape) if "x" in data["domain"] else [0, 0],
            "stem": list(data["stem"].x.shape) if "x" in data["stem"] else [0, 0],
        },
        "edge_counts": {
            "email->sender:has_sender": len(has_sender_src),
            "email->receiver:has_receiver": len(has_receiver_src),
            "email->week:in_week": len(in_week_src),
            "email->subject:has_subject": len(has_subject_src),
            "email->url:has_url": len(has_url_src),
            "url->domain:has_domain": len(url_to_domain_src),
            "url->stem:has_stem": len(url_to_stem_src),
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
    return torch_lib.load(graph_path, weights_only=False)


__all__ = [
    "build_hetero_graph_from_misp",
    "build_graph",
    "save_graph",
    "load_graph",
]
