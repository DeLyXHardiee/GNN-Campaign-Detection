"""
Backend-agnostic graph assembler.

Takes MISP events and the shared schema, and produces a simple intermediate
representation (Graph IR) with:
- unique nodes per canonical type (with index order, features, and metadata)
- edge lists per canonical relationship (source/destination indices)

Both the PyTorch-Geometric and Memgraph builders render from this IR,
so changes to how the graph is derived from data live here in one place.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .graph_schema import GraphSchema, DEFAULT_SCHEMA
from .common import (
    parse_misp_events,
    extract_week_key,
    extract_email_domain,
    parse_url_components,
    to_unix_ts,
)


@dataclass
class NodeIR:
    index: Dict[str, int]
    x: List[List[float]]  # simple numeric features to keep tensors valid
    index_to_string: Optional[List[str]] = None  # for non-email nodes
    index_to_meta: Optional[List[Dict[str, Any]]] = None  # for emails


@dataclass
class GraphIR:
    nodes: Dict[str, NodeIR]  # keyed by canonical node type
    edges: Dict[str, Tuple[List[int], List[int]]]  # keyed by canonical edge name
    email_attrs: Dict[str, List[Any]]  # additional attributes for email nodes


def assemble_misp_graph_ir(misp_events: List[dict], *, schema: Optional[GraphSchema] = None) -> GraphIR:
    schema = schema or DEFAULT_SCHEMA

    emails = parse_misp_events(misp_events)

    # Unique entities and indices for each component type
    sender_to_idx: Dict[str, int] = {}
    receiver_to_idx: Dict[str, int] = {}
    week_to_idx: Dict[str, int] = {}
    subject_to_idx: Dict[str, int] = {}
    url_to_idx: Dict[str, int] = {}
    domain_to_idx: Dict[str, int] = {}
    stem_to_idx: Dict[str, int] = {}
    email_domain_to_idx: Dict[str, int] = {}

    # Track URL -> (domain, stem) mapping for edge creation
    url_components: Dict[str, Tuple[str, str]] = {}

    # Pre-scan to register unique components
    for em in emails:
        sender = em.get("sender")
        if sender:
            sender_to_idx.setdefault(sender, len(sender_to_idx))
            sender_domain = extract_email_domain(sender)
            if sender_domain:
                email_domain_to_idx.setdefault(sender_domain, len(email_domain_to_idx))

        for r in em.get("receivers", []) or []:
            if r:
                receiver_to_idx.setdefault(r, len(receiver_to_idx))
                receiver_domain = extract_email_domain(r)
                if receiver_domain:
                    email_domain_to_idx.setdefault(receiver_domain, len(email_domain_to_idx))

        week_key = extract_week_key(em.get("date", ""))
        if week_key:
            week_to_idx.setdefault(week_key, len(week_to_idx))

        subj = em.get("subject", "")
        if subj:
            subject_to_idx.setdefault(subj, len(subject_to_idx))

        for u in em.get("urls", []) or []:
            if u:
                url_to_idx.setdefault(u, len(url_to_idx))
                parsed = parse_url_components(u)
                domain = parsed.get("domain", "")
                stem = parsed.get("stem", "")
                if domain:
                    domain_to_idx.setdefault(domain, len(domain_to_idx))
                if stem:
                    stem_to_idx.setdefault(stem, len(stem_to_idx))
                url_components[u] = (domain, stem)

    # Features and meta arrays in index order
    def ordered_keys(d: Dict[str, int]) -> List[str]:
        return [k for k, _ in sorted(d.items(), key=lambda kv: kv[1])]

    sender_meta = ordered_keys(sender_to_idx)
    receiver_meta = ordered_keys(receiver_to_idx)
    week_meta = ordered_keys(week_to_idx)
    subject_meta = ordered_keys(subject_to_idx)
    url_meta = ordered_keys(url_to_idx)
    domain_meta = ordered_keys(domain_to_idx)
    stem_meta = ordered_keys(stem_to_idx)
    email_domain_meta = ordered_keys(email_domain_to_idx)

    sender_x = [[float(len(s))] for s in sender_meta]
    receiver_x = [[float(len(s))] for s in receiver_meta]
    week_x = [[float(idx)] for idx in range(len(week_meta))]
    subject_x = [[float(len(s))] for s in subject_meta]
    url_x = [[float(len(u))] for u in url_meta]
    domain_x = [[float(len(d))] for d in domain_meta]
    stem_x = [[float(len(s))] for s in stem_meta]
    email_domain_x = [[float(len(d))] for d in email_domain_meta]

    # Build edges and email features/meta
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
    url_to_domain_src: List[int] = []
    url_to_domain_dst: List[int] = []
    url_to_stem_src: List[int] = []
    url_to_stem_dst: List[int] = []
    sender_to_email_domain_src: List[int] = []
    sender_to_email_domain_dst: List[int] = []
    receiver_to_email_domain_src: List[int] = []
    receiver_to_email_domain_dst: List[int] = []

    email_x: List[List[float]] = []
    email_meta: List[Dict[str, Any]] = []
    # Email attributes to expose separately
    email_ts: List[int] = []
    email_n_urls: List[int] = []
    email_len_subject: List[int] = []
    email_len_body: List[int] = []
    TEXT_EMBED_DIM = 384
    email_x_text: List[List[float]] = []

    for email_idx, em in enumerate(emails):
        # Email feature and meta
        subj = em.get("subject", "")
        body = em.get("body", "")
        urls = em.get("urls", []) or []
        email_meta.append({
            "info": em.get("email_info", ""),
            "index": email_idx,
            "date": em.get("date", ""),
        })
        # Derived attributes
        ts_val = to_unix_ts(em.get("date", ""))
        email_ts.append(ts_val)
        # Unique URL domains per email
        domains = set()
        for u in urls:
            if not u:
                continue
            comp = parse_url_components(u)
            d = comp.get("domain", "")
            if d:
                domains.add(d)
        n_urls_val = int(len(domains))
        email_n_urls.append(n_urls_val)
        len_subject_val = int(len(subj))
        len_body_val = int(len(body))
        email_len_subject.append(len_subject_val)
        email_len_body.append(len_body_val)
        # Text embedding placeholder (zeros); replace with real model if available
        email_x_text.append([0.0] * TEXT_EMBED_DIM)

        # Construct primary numeric feature vector (x) for PyG graph
        # Order: [len_subject, len_body, n_urls, ts]
        email_x.append([
            float(len_subject_val),
            float(len_body_val),
            float(n_urls_val),
            float(ts_val),
        ])

        if em.get("sender") and em["sender"] in sender_to_idx:
            has_sender_src.append(email_idx)
            has_sender_dst.append(sender_to_idx[em["sender"]])

        for r in em.get("receivers", []) or []:
            if r and r in receiver_to_idx:
                has_receiver_src.append(email_idx)
                has_receiver_dst.append(receiver_to_idx[r])

        wk = extract_week_key(em.get("date", ""))
        if wk and wk in week_to_idx:
            in_week_src.append(email_idx)
            in_week_dst.append(week_to_idx[wk])

        subj = em.get("subject", "")
        if subj and subj in subject_to_idx:
            has_subject_src.append(email_idx)
            has_subject_dst.append(subject_to_idx[subj])

        for u in em.get("urls", []) or []:
            if u and u in url_to_idx:
                has_url_src.append(email_idx)
                has_url_dst.append(url_to_idx[u])

    for url, (domain, stem) in url_components.items():
        url_idx = url_to_idx[url]
        if domain and domain in domain_to_idx:
            url_to_domain_src.append(url_idx)
            url_to_domain_dst.append(domain_to_idx[domain])
        if stem and stem in stem_to_idx:
            url_to_stem_src.append(url_idx)
            url_to_stem_dst.append(stem_to_idx[stem])

    for sender, s_idx in sender_to_idx.items():
        s_dom = extract_email_domain(sender)
        if s_dom and s_dom in email_domain_to_idx:
            sender_to_email_domain_src.append(s_idx)
            sender_to_email_domain_dst.append(email_domain_to_idx[s_dom])

    for receiver, r_idx in receiver_to_idx.items():
        r_dom = extract_email_domain(receiver)
        if r_dom and r_dom in email_domain_to_idx:
            receiver_to_email_domain_src.append(r_idx)
            receiver_to_email_domain_dst.append(email_domain_to_idx[r_dom])

    nodes: Dict[str, NodeIR] = {
        "email": NodeIR(index={}, x=email_x, index_to_meta=email_meta),
        "sender": NodeIR(index=sender_to_idx, x=sender_x, index_to_string=sender_meta),
        "receiver": NodeIR(index=receiver_to_idx, x=receiver_x, index_to_string=receiver_meta),
        "week": NodeIR(index=week_to_idx, x=week_x, index_to_string=week_meta),
        "subject": NodeIR(index=subject_to_idx, x=subject_x, index_to_string=subject_meta),
        "url": NodeIR(index=url_to_idx, x=url_x, index_to_string=url_meta),
        "domain": NodeIR(index=domain_to_idx, x=domain_x, index_to_string=domain_meta),
        "stem": NodeIR(index=stem_to_idx, x=stem_x, index_to_string=stem_meta),
        "email_domain": NodeIR(index=email_domain_to_idx, x=email_domain_x, index_to_string=email_domain_meta),
    }

    edges: Dict[str, Tuple[List[int], List[int]]] = {
        "has_sender": (has_sender_src, has_sender_dst),
        "has_receiver": (has_receiver_src, has_receiver_dst),
        "in_week": (in_week_src, in_week_dst),
        "has_subject": (has_subject_src, has_subject_dst),
        "has_url": (has_url_src, has_url_dst),
        "url_has_domain": (url_to_domain_src, url_to_domain_dst),
        "url_has_stem": (url_to_stem_src, url_to_stem_dst),
        "sender_from_domain": (sender_to_email_domain_src, sender_to_email_domain_dst),
        "receiver_from_domain": (receiver_to_email_domain_src, receiver_to_email_domain_dst),
    }

    email_attrs = {
        "ts": email_ts,
        "n_urls": email_n_urls,
        "len_subject": email_len_subject,
        "len_body": email_len_body,
        "x_text": email_x_text,
    }
    return GraphIR(nodes=nodes, edges=edges, email_attrs=email_attrs)


__all__ = ["GraphIR", "NodeIR", "assemble_misp_graph_ir"]
