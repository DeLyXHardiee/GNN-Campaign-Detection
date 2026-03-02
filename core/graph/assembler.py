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

from dataclasses import dataclass, field
import hashlib
import math
from typing import Any, Callable, Dict, List, Optional, Tuple, Set

from .graph_schema import GraphSchema, DEFAULT_SCHEMA
from .common import (
    parse_misp_events,
    extract_week_key,
    extract_email_domain,
    parse_url_components,
    to_unix_ts,
    compute_lexical_features,
    is_freemail_domain,
)


def create_html_css_features(html_data: dict, css_data: dict, tag_bin_count: int = 16) -> List[float]:
    """Create a fixed-length HTML/CSS feature vector for one email."""
    features: List[float] = []
    html_data = html_data if isinstance(html_data, dict) else {}
    css_data = css_data if isinstance(css_data, dict) else {}

    tree = html_data.get("tree_stats", {}) or {}
    total_elements = int(tree.get("total_elements", 0) or 0)
    total = float(total_elements or 1)
    forms = float(tree.get("forms", 0) or 0)
    password_fields = float(tree.get("password_fields", 0) or 0)

    # Block A: Structural complexity
    features.extend(
        [
            math.log1p(total_elements),
            float(tree.get("max_depth", 0) or 0),
            float(tree.get("avg_depth", 0.0) or 0.0),
            math.log1p(forms),
            math.log1p(password_fields),
            math.log1p(float(tree.get("hidden_elements", 0) or 0)),
            math.log1p(float(tree.get("external_scripts", 0) or 0)),
            float(tree.get("link_ratio", 0.0) or 0.0),
            float(tree.get("image_ratio", 0.0) or 0.0),
            (password_fields / forms) if forms else 0.0,
        ]
    )

    # Block B: Hashed tag distribution (stable hash, not Python's randomized hash())
    tag_bins = [0.0] * tag_bin_count
    tag_counts = html_data.get("tag_counts", {}) or {}
    if isinstance(tag_counts, dict):
        for tag, count in tag_counts.items():
            try:
                idx = int.from_bytes(
                    hashlib.blake2b(str(tag).encode("utf-8", "ignore"), digest_size=8).digest(),
                    byteorder="big",
                    signed=False,
                ) % tag_bin_count
                tag_bins[idx] += float(count or 0)
            except Exception:
                continue
    tag_bins = [v / total for v in tag_bins]
    features.extend(tag_bins)

    # Block C: Structural SimHash bytes
    fingerprint_hex = str(html_data.get("structure_fingerprint", "") or "").strip().lower()
    try:
        fingerprint_int = int(fingerprint_hex, 16) if fingerprint_hex else 0
    except Exception:
        fingerprint_int = 0
    for i in range(8):
        byte = (fingerprint_int >> (8 * i)) & 0xFF
        features.append(float(byte) / 255.0)

    # Block D: CSS features
    style = css_data.get("style_features", {}) or {}
    features.extend(
        [
            math.log1p(float(style.get("unique_color_count", 0) or 0)),
            1.0 if style.get("uses_position_absolute") else 0.0,
            1.0 if style.get("uses_z_index") else 0.0,
            1.0 if style.get("uses_media_queries") else 0.0,
            math.log1p(float(style.get("unique_class_count", 0) or 0)),
            float(style.get("class_entropy", 0.0) or 0.0),
        ]
    )
    return features


@dataclass
class NodeIR:
    index: Dict[str, int]
    x: List[List[float]]  # simple numeric features to keep tensors valid
    index_to_string: Optional[List[str]] = None  # for non-email nodes
    index_to_meta: Optional[List[Dict[str, Any]]] = None  # for emails
    attrs: Dict[str, List[Any]] = field(default_factory=dict)  # aligned to node order


@dataclass
class GraphIR:
    nodes: Dict[str, NodeIR]  # keyed by canonical node type
    edges: Dict[str, Tuple[List[int], List[int]]]  # keyed by canonical edge name
    email_attrs: Dict[str, List[Any]]  # additional attributes for email nodes


def _ordered_keys(d: Dict[str, int]) -> List[str]:
    """Return keys ordered by their assigned index values."""
    return [k for k, _ in sorted(d.items(), key=lambda kv: kv[1])]


def _is_valid_stem(stem: str) -> bool:
    """Return True if URL stem should be represented as a node."""
    s = (stem or "").strip()
    return bool(s) and s != "/"


def _as_email_list(value: Any) -> List[str]:
    """Normalize scalar/list email field to a unique list preserving order."""
    if isinstance(value, list):
        raw_vals = value
    elif value:
        raw_vals = [value]
    else:
        raw_vals = []
    out: List[str] = []
    seen: Set[str] = set()
    for v in raw_vals:
        s = str(v).strip().lower()
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


@dataclass
class ProviderRegistry:
    """Registry for declarative assembler providers."""
    node_indexers: Dict[str, Callable[[List[Dict[str, Any]]], Dict[str, int]]]
    edge_builders: Dict[str, Callable[[Dict[str, Any], Dict[str, Dict[str, int]], Dict[str, List[int]], Dict[str, Dict[str, Set[int]]], str], None]]
    node_feature_builders: Dict[str, Callable[[str, Dict[str, Dict[str, int]], Dict[str, Tuple[str, str]], Dict[str, Dict[str, Set[int]]]], Tuple[List[List[float]], List[str], Dict[str, List[Any]]]]]


def _dedup_index(values: List[str]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for v in values:
        if v:
            out.setdefault(v, len(out))
    return out


def _field_values_for_node(email: Dict[str, Any], node_key: str) -> List[str]:
    if node_key == "sender":
        return _as_email_list(email.get("senders"))
    if node_key == "receiver":
        return _as_email_list(email.get("receivers"))
    if node_key == "attachment":
        return _as_email_list(email.get("attachments"))
    if node_key == "url":
        return _as_email_list(email.get("urls"))
    if node_key == "origin_ip":
        hops = email.get("received_hops") or []
        return [str(h.get("origin_ip", "")).strip() for h in hops if isinstance(h, dict) and str(h.get("origin_ip", "")).strip()]
    if node_key == "received_host":
        hops = email.get("received_hops") or []
        vals: List[str] = []
        for h in hops:
            if not isinstance(h, dict):
                continue
            for key in ("helo_host", "by_host"):
                v = str(h.get(key, "")).strip()
                if v:
                    vals.append(v)
        return vals
    return _as_email_list(email.get(f"{node_key}s") or email.get(node_key))


def _node_indexers() -> Dict[str, Callable[[List[Dict[str, Any]]], Dict[str, int]]]:
    def sender(emails: List[Dict[str, Any]]) -> Dict[str, int]:
        vals: List[str] = []
        for em in emails:
            vals.extend(_as_email_list(em.get("senders")))
        return _dedup_index(vals)

    def receiver(emails: List[Dict[str, Any]]) -> Dict[str, int]:
        vals: List[str] = []
        for em in emails:
            vals.extend(_as_email_list(em.get("receivers")))
        return _dedup_index(vals)

    def week(emails: List[Dict[str, Any]]) -> Dict[str, int]:
        vals = [extract_week_key(em.get("date", "")) or "" for em in emails]
        return _dedup_index(vals)

    def url(emails: List[Dict[str, Any]]) -> Dict[str, int]:
        vals: List[str] = []
        for em in emails:
            vals.extend(_as_email_list(em.get("urls")))
        return _dedup_index(vals)

    def domain(emails: List[Dict[str, Any]]) -> Dict[str, int]:
        vals: List[str] = []
        for em in emails:
            for u in _as_email_list(em.get("urls")):
                d = parse_url_components(u).get("domain", "")
                if d:
                    vals.append(d)
        return _dedup_index(vals)

    def stem(emails: List[Dict[str, Any]]) -> Dict[str, int]:
        vals: List[str] = []
        for em in emails:
            for u in _as_email_list(em.get("urls")):
                s = parse_url_components(u).get("stem", "")
                if _is_valid_stem(s):
                    vals.append(s)
        return _dedup_index(vals)

    def email_domain(emails: List[Dict[str, Any]]) -> Dict[str, int]:
        vals: List[str] = []
        for em in emails:
            for sender_addr in _as_email_list(em.get("senders")):
                d = extract_email_domain(sender_addr)
                if d and not is_freemail_domain(d):
                    vals.append(d)
            for receiver_addr in _as_email_list(em.get("receivers")):
                d = extract_email_domain(receiver_addr)
                if d and not is_freemail_domain(d):
                    vals.append(d)
        return _dedup_index(vals)

    def attachment(emails: List[Dict[str, Any]]) -> Dict[str, int]:
        vals: List[str] = []
        for em in emails:
            vals.extend(_as_email_list(em.get("attachments")))
        return _dedup_index(vals)

    def origin_ip(emails: List[Dict[str, Any]]) -> Dict[str, int]:
        vals: List[str] = []
        for em in emails:
            for h in em.get("received_hops") or []:
                if isinstance(h, dict):
                    v = str(h.get("origin_ip", "")).strip()
                    if v:
                        vals.append(v)
        return _dedup_index(vals)

    def received_host(emails: List[Dict[str, Any]]) -> Dict[str, int]:
        vals: List[str] = []
        for em in emails:
            for h in em.get("received_hops") or []:
                if isinstance(h, dict):
                    for key in ("helo_host", "by_host"):
                        v = str(h.get(key, "")).strip()
                        if v:
                            vals.append(v)
        return _dedup_index(vals)

    return {
        "sender": sender,
        "receiver": receiver,
        "week": week,
        "url": url,
        "domain": domain,
        "stem": stem,
        "email_domain": email_domain,
        "attachment": attachment,
        "origin_ip": origin_ip,
        "received_host": received_host,
    }


def _edge_builders() -> Dict[str, Callable[[Dict[str, Any], Dict[str, Dict[str, int]], Dict[str, List[int]], Dict[str, Dict[str, Set[int]]], str], None]]:
    def email_to_entity(
        email_ctx: Dict[str, Any],
        indices: Dict[str, Dict[str, int]],
        edges_idx: Dict[str, List[int]],
        docfreq_maps: Dict[str, Dict[str, Set[int]]],
        edge_name: str,
    ) -> None:
        email_idx = int(email_ctx["email_idx"])
        em = email_ctx["email"]
        dst_key = email_ctx["edge_dst"]
        values = _field_values_for_node(em, dst_key)
        for value in values:
            if value in indices.get(dst_key, {}):
                edges_idx[f"{edge_name}_src"].append(email_idx)
                edges_idx[f"{edge_name}_dst"].append(indices[dst_key][value])
                if dst_key == "sender":
                    docfreq_maps["sender_email_sets"].setdefault(value, set()).add(email_idx)
                elif dst_key == "receiver":
                    docfreq_maps["receiver_email_sets"].setdefault(value, set()).add(email_idx)
                elif dst_key == "url":
                    docfreq_maps["url_email_sets"].setdefault(value, set()).add(email_idx)
                elif dst_key == "attachment":
                    docfreq_maps["attachment_email_sets"].setdefault(value, set()).add(email_idx)
                elif dst_key == "origin_ip":
                    docfreq_maps["origin_ip_email_sets"].setdefault(value, set()).add(email_idx)
                elif dst_key == "received_host":
                    docfreq_maps["received_host_email_sets"].setdefault(value, set()).add(email_idx)

    def email_to_week_from_date(
        email_ctx: Dict[str, Any],
        indices: Dict[str, Dict[str, int]],
        edges_idx: Dict[str, List[int]],
        _docfreq_maps: Dict[str, Dict[str, Set[int]]],
        edge_name: str,
    ) -> None:
        email_idx = int(email_ctx["email_idx"])
        em = email_ctx["email"]
        wk = extract_week_key(em.get("date", ""))
        if wk and wk in indices.get("week", {}):
            edges_idx[f"{edge_name}_src"].append(email_idx)
            edges_idx[f"{edge_name}_dst"].append(indices["week"][wk])

    def email_to_domain_from_urls(
        email_ctx: Dict[str, Any],
        indices: Dict[str, Dict[str, int]],
        edges_idx: Dict[str, List[int]],
        docfreq_maps: Dict[str, Dict[str, Set[int]]],
        edge_name: str,
    ) -> None:
        email_idx = int(email_ctx["email_idx"])
        em = email_ctx["email"]
        for u in _as_email_list(em.get("urls")):
            d = parse_url_components(u).get("domain", "")
            if not d:
                continue
            docfreq_maps["domain_email_sets"].setdefault(d, set()).add(email_idx)
            if d in indices.get("domain", {}):
                edges_idx[f"{edge_name}_src"].append(email_idx)
                edges_idx[f"{edge_name}_dst"].append(indices["domain"][d])

    def email_to_stem_from_urls(
        email_ctx: Dict[str, Any],
        indices: Dict[str, Dict[str, int]],
        edges_idx: Dict[str, List[int]],
        docfreq_maps: Dict[str, Dict[str, Set[int]]],
        edge_name: str,
    ) -> None:
        email_idx = int(email_ctx["email_idx"])
        em = email_ctx["email"]
        for u in _as_email_list(em.get("urls")):
            s = parse_url_components(u).get("stem", "")
            if not _is_valid_stem(s):
                continue
            docfreq_maps["stem_email_sets"].setdefault(s, set()).add(email_idx)
            if s in indices.get("stem", {}):
                edges_idx[f"{edge_name}_src"].append(email_idx)
                edges_idx[f"{edge_name}_dst"].append(indices["stem"][s])

    return {
        "email_to_entity": email_to_entity,
        "email_to_week_from_date": email_to_week_from_date,
        "email_to_domain_from_urls": email_to_domain_from_urls,
        "email_to_stem_from_urls": email_to_stem_from_urls,
    }


def _node_feature_builders() -> Dict[str, Callable[[str, Dict[str, Dict[str, int]], Dict[str, Tuple[str, str]], Dict[str, Dict[str, Set[int]]]], Tuple[List[List[float]], List[str], Dict[str, List[Any]]]]]:
    def sender(node_key: str, indices: Dict[str, Dict[str, int]], _url_components: Dict[str, Tuple[str, str]], docfreq_maps: Dict[str, Dict[str, Set[int]]]):
        meta = _ordered_keys(indices[node_key])
        x = [[float(len(v))] for v in meta]
        attrs = {"docfreq": [len(docfreq_maps["sender_email_sets"].get(v, set())) for v in meta]}
        return x, meta, attrs

    def receiver(node_key: str, indices: Dict[str, Dict[str, int]], _url_components: Dict[str, Tuple[str, str]], docfreq_maps: Dict[str, Dict[str, Set[int]]]):
        meta = _ordered_keys(indices[node_key])
        x = [[float(len(v))] for v in meta]
        attrs = {"docfreq": [len(docfreq_maps["receiver_email_sets"].get(v, set())) for v in meta]}
        return x, meta, attrs

    def week(node_key: str, indices: Dict[str, Dict[str, int]], _url_components: Dict[str, Tuple[str, str]], _docfreq_maps: Dict[str, Dict[str, Set[int]]]):
        meta = _ordered_keys(indices[node_key])
        x = [[float(i)] for i, _ in enumerate(meta)]
        return x, meta, {}

    def url(node_key: str, indices: Dict[str, Dict[str, int]], url_components: Dict[str, Tuple[str, str]], docfreq_maps: Dict[str, Dict[str, Set[int]]]):
        meta = _ordered_keys(indices[node_key])
        x: List[List[float]] = []
        for u in meta:
            stem = (url_components.get(u) or ("", ""))[1] if u in url_components else parse_url_components(u).get("stem", "/")
            x.append([float(len(stem or "/"))])
        attrs = {
            "x_lex": [compute_lexical_features(u) for u in meta],
            "docfreq": [len(docfreq_maps["url_email_sets"].get(u, set())) for u in meta],
        }
        return x, meta, attrs

    def domain(node_key: str, indices: Dict[str, Dict[str, int]], _url_components: Dict[str, Tuple[str, str]], docfreq_maps: Dict[str, Dict[str, Set[int]]]):
        meta = _ordered_keys(indices[node_key])
        x_lex = [compute_lexical_features(d) for d in meta]
        x = [[float(v[7] if len(v) > 7 else 0.0)] for v in x_lex]
        attrs = {
            "x_lex": x_lex,
            "docfreq": [len(docfreq_maps["domain_email_sets"].get(d, set())) for d in meta],
        }
        return x, meta, attrs

    def stem(node_key: str, indices: Dict[str, Dict[str, int]], _url_components: Dict[str, Tuple[str, str]], docfreq_maps: Dict[str, Dict[str, Set[int]]]):
        meta = _ordered_keys(indices[node_key])
        x = [[float(len(s))] for s in meta]
        attrs = {
            "x_lex": [compute_lexical_features(s) for s in meta],
            "docfreq": [len(docfreq_maps["stem_email_sets"].get(s, set())) for s in meta],
        }
        return x, meta, attrs

    def email_domain(node_key: str, indices: Dict[str, Dict[str, int]], _url_components: Dict[str, Tuple[str, str]], docfreq_maps: Dict[str, Dict[str, Set[int]]]):
        meta = _ordered_keys(indices[node_key])
        x = [[float(len(d))] for d in meta]
        attrs = {
            "x_lex": [compute_lexical_features(d) for d in meta],
            "docfreq_sender": [len(docfreq_maps["email_domain_sender_sets"].get(d, set())) for d in meta],
            "docfreq_receiver": [len(docfreq_maps["email_domain_receiver_sets"].get(d, set())) for d in meta],
        }
        return x, meta, attrs

    def attachment(node_key: str, indices: Dict[str, Dict[str, int]], _url_components: Dict[str, Tuple[str, str]], docfreq_maps: Dict[str, Dict[str, Set[int]]]):
        meta = _ordered_keys(indices[node_key])
        x = [[float(len(v))] for v in meta]
        attrs = {"docfreq": [len(docfreq_maps["attachment_email_sets"].get(v, set())) for v in meta]}
        return x, meta, attrs

    def _str_len_docfreq(docfreq_key: str):
        def _builder(node_key: str, indices: Dict[str, Dict[str, int]], _url_components: Dict[str, Tuple[str, str]], docfreq_maps: Dict[str, Dict[str, Set[int]]]):
            meta = _ordered_keys(indices[node_key])
            x = [[float(len(v))] for v in meta]
            attrs = {"docfreq": [len(docfreq_maps[docfreq_key].get(v, set())) for v in meta]}
            return x, meta, attrs
        return _builder

    return {
        "sender": sender,
        "receiver": receiver,
        "week": week,
        "url": url,
        "domain": domain,
        "stem": stem,
        "email_domain": email_domain,
        "attachment": attachment,
        "origin_ip": _str_len_docfreq("origin_ip_email_sets"),
        "received_host": _str_len_docfreq("received_host_email_sets"),
    }


DEFAULT_PROVIDER_REGISTRY = ProviderRegistry(
    node_indexers=_node_indexers(),
    edge_builders=_edge_builders(),
    node_feature_builders=_node_feature_builders(),
)


def index_entities(
    emails: List[Dict[str, Any]],
    schema: GraphSchema,
    registry: ProviderRegistry = DEFAULT_PROVIDER_REGISTRY,
) -> Dict[str, Any]:
    """Registry-driven entity indexing pass."""
    indices: Dict[str, Dict[str, int]] = {}
    for node_key in schema.nodes:
        if node_key == "email":
            continue
        provider = registry.node_indexers.get(node_key)
        if provider is None:
            indices[node_key] = {}
            continue
        indices[node_key] = provider(emails)

    url_components: Dict[str, Tuple[str, str]] = {}
    for em in emails:
        for u in _as_email_list(em.get("urls")):
            comp = parse_url_components(u)
            url_components[u] = (comp.get("domain", ""), comp.get("stem", ""))

    out: Dict[str, Any] = dict(indices)
    out["url_components"] = url_components
    return out


def materialize_edges(
    emails: List[Dict[str, Any]],
    indices: Dict[str, Dict[str, int]],
    schema: GraphSchema,
    registry: ProviderRegistry = DEFAULT_PROVIDER_REGISTRY,
) -> Tuple[Dict[str, List[int]], List[Dict[str, Any]], Dict[str, List[Any]], Dict[str, Dict[str, Set[int]]]]:
    edges_idx: Dict[str, List[int]] = {}
    for edge_name in schema.edges:
        edges_idx[f"{edge_name}_src"] = []
        edges_idx[f"{edge_name}_dst"] = []

    email_meta: List[Dict[str, Any]] = []
    email_attrs_raw: Dict[str, List[Any]] = {
        "ts": [],
        "n_urls": [],
        "len_subject": [],
        "len_body": [],
        "x_html_css": [],
    }
    docfreq_maps: Dict[str, Dict[str, Set[int]]] = {
        "domain_email_sets": {},
        "stem_email_sets": {},
        "email_domain_sender_sets": {},
        "email_domain_receiver_sets": {},
        "url_email_sets": {},
        "attachment_email_sets": {},
        "sender_email_sets": {},
        "receiver_email_sets": {},
        "origin_ip_email_sets": {},
        "received_host_email_sets": {},
    }

    for email_idx, em in enumerate(emails):
        urls = _as_email_list(em.get("urls"))
        email_meta.append(
            {
                "info": em.get("email_info", ""),
                "index": email_idx,
                "email_index": em.get("email_index", email_idx),
                "date": em.get("date", ""),
            }
        )
        email_attrs_raw["ts"].append(to_unix_ts(em.get("date", "")))
        email_attrs_raw["len_subject"].append(int(len(em.get("subject", "") or "")))
        email_attrs_raw["len_body"].append(int(len(em.get("body", "") or "")))
        domains = {
            parse_url_components(u).get("domain", "")
            for u in urls
            if parse_url_components(u).get("domain", "")
        }
        email_attrs_raw["n_urls"].append(int(len(domains)))
        email_attrs_raw["x_html_css"].append(
            create_html_css_features(
                em.get("html", {}) or {},
                em.get("css", {}) or {},
            )
        )

        for sender in _as_email_list(em.get("senders")):
            d = extract_email_domain(sender)
            if d and not is_freemail_domain(d):
                docfreq_maps["email_domain_sender_sets"].setdefault(d, set()).add(email_idx)
        for receiver in _as_email_list(em.get("receivers")):
            d = extract_email_domain(receiver)
            if d and not is_freemail_domain(d):
                docfreq_maps["email_domain_receiver_sets"].setdefault(d, set()).add(email_idx)

        for edge_name, edge_map in schema.edges.items():
            provider = registry.edge_builders.get(edge_map.edge_strategy)
            if provider is None:
                continue
            provider(
                {
                    "email_idx": email_idx,
                    "email": em,
                    "edge_src": edge_map.src,
                    "edge_dst": edge_map.dst,
                },
                indices,
                edges_idx,
                docfreq_maps,
                edge_name,
            )

    return edges_idx, email_meta, email_attrs_raw, docfreq_maps


def _connect_email_entities_to_domains(
    sender_to_idx: Dict[str, int],
    receiver_to_idx: Dict[str, int],
    email_domain_to_idx: Dict[str, int],
) -> Tuple[List[int], List[int], List[int], List[int]]:

    sender_src: List[int] = []
    sender_dst: List[int] = []
    receiver_src: List[int] = []
    receiver_dst: List[int] = []
    for sender, s_idx in sender_to_idx.items():
        s_dom = extract_email_domain(sender)
        if s_dom and s_dom in email_domain_to_idx:
            sender_src.append(s_idx)
            sender_dst.append(email_domain_to_idx[s_dom])
    for receiver, r_idx in receiver_to_idx.items():
        r_dom = extract_email_domain(receiver)
        if r_dom and r_dom in email_domain_to_idx:
            receiver_src.append(r_idx)
            receiver_dst.append(email_domain_to_idx[r_dom])
    return sender_src, sender_dst, receiver_src, receiver_dst


def build_node_features(
    emails: List[Dict[str, Any]],
    schema: GraphSchema,
    indices: Dict[str, Dict[str, int]],
    url_components: Dict[str, Tuple[str, str]],
    docfreq_maps: Dict[str, Dict[str, Set[int]]],
    registry: ProviderRegistry = DEFAULT_PROVIDER_REGISTRY,
    embeddings_output_dir: Optional[str] = None,
) -> Tuple[
    Dict[str, List[List[float]]],
    Dict[str, List[str]],
    Dict[str, Dict[str, List[Any]]],
    List[List[float]],
    List[List[float]],
    int,
    int,
]:
    node_x: Dict[str, List[List[float]]] = {}
    node_meta: Dict[str, List[str]] = {}
    node_attrs: Dict[str, Dict[str, List[Any]]] = {}
    for node_key in schema.nodes:
        if node_key == "email":
            continue
        provider = registry.node_feature_builders.get(node_key)
        if provider is None:
            meta = _ordered_keys(indices.get(node_key, {}))
            node_x[node_key] = [[float(len(v))] for v in meta]
            node_meta[node_key] = meta
            node_attrs[node_key] = {}
            continue
        x, meta, attrs = provider(node_key, indices, url_components, docfreq_maps)
        node_x[node_key] = x
        node_meta[node_key] = meta
        node_attrs[node_key] = attrs

    from .embeddings import DEFAULT_OUTPUT_DIR, get_embeddings

    out_dir = embeddings_output_dir if embeddings_output_dir else str(DEFAULT_OUTPUT_DIR)
    subj_vecs, body_vecs, subj_dim, body_dim = get_embeddings(emails, output_dir=out_dir)

    return node_x, node_meta, node_attrs, subj_vecs, body_vecs, subj_dim, body_dim




def _build_email_feature_matrix(
    ts: List[float],
    len_body: List[float],
    n_urls: List[float],
    len_subject: List[float],
    subj_vecs: List[List[float]],
    body_vecs: List[List[float]],
    html_css_vecs: List[List[float]],
) -> List[List[float]]:
    """Construct the email feature matrix using raw scalars + text embeddings.

    Order: [ts, len_body, n_urls, len_subject, SBERT(subject), SBERT(body)]
    """
    n_emails = max(
        len(ts),
        len(len_body),
        len(n_urls),
        len(len_subject),
        len(subj_vecs) if subj_vecs else 0,
        len(body_vecs) if body_vecs else 0,
        len(html_css_vecs) if html_css_vecs else 0,
    )
    email_x: List[List[float]] = []
    for i in range(n_emails):
        row: List[float] = [
            float(ts[i]) if i < len(ts) else 0.0,
            float(len_body[i]) if i < len(len_body) else 0.0,
            float(n_urls[i]) if i < len(n_urls) else 0.0,
            float(len_subject[i]) if i < len(len_subject) else 0.0,
        ]
        if subj_vecs:
            row.extend(subj_vecs[i] if i < len(subj_vecs) else [])
        if body_vecs:
            row.extend(body_vecs[i] if i < len(body_vecs) else [])
        if html_css_vecs:
            row.extend(html_css_vecs[i] if i < len(html_css_vecs) else [])
        email_x.append(row)
    return email_x



def _assemble_nodes(
    schema: GraphSchema,
    node_x: Dict[str, List[List[float]]],
    node_meta: Dict[str, List[str]],
    node_attrs: Dict[str, Dict[str, List[Any]]],
    indices: Dict[str, Dict[str, int]],
    email_meta: List[Dict[str, Any]],
    email_x: List[List[float]],
) -> Dict[str, NodeIR]:
    nodes: Dict[str, NodeIR] = {"email": NodeIR(index={}, x=email_x, index_to_meta=email_meta)}
    for node_key in schema.nodes:
        if node_key == "email":
            continue
        nodes[node_key] = NodeIR(
            index=indices.get(node_key, {}),
            x=node_x.get(node_key, []),
            index_to_string=node_meta.get(node_key, []),
            attrs=node_attrs.get(node_key, {}),
        )
    return nodes


def _assemble_edges(
    schema: GraphSchema,
    edges_idx: Dict[str, List[int]],
    snd_dom_src: List[int], snd_dom_dst: List[int],
    rcv_dom_src: List[int], rcv_dom_dst: List[int],
) -> Dict[str, Tuple[List[int], List[int]]]:
    edges: Dict[str, Tuple[List[int], List[int]]] = {}
    for edge_name in schema.edges:
        src_key = f"{edge_name}_src"
        dst_key = f"{edge_name}_dst"
        if src_key in edges_idx and dst_key in edges_idx:
            edges[edge_name] = (edges_idx[src_key], edges_idx[dst_key])
    edges["sender_from_domain"] = (snd_dom_src, snd_dom_dst)
    edges["receiver_from_domain"] = (rcv_dom_src, rcv_dom_dst)
    return edges


def _assemble_email_attrs(
    email_meta: List[Dict[str, Any]],
    email_attrs_raw: Dict[str, List[Any]],
    subj_dim: int,
    body_dim: int,
    subj_vecs: List[List[float]],
    body_vecs: List[List[float]],
) -> Dict[str, Any]:
    n_emails = len(email_meta) or 0
    x_text: List[List[float]] = []
    if subj_dim > 0 or body_dim > 0:
        for i in range(n_emails):
            comb: List[float] = []
            if subj_vecs:
                comb.extend(subj_vecs[i] if i < len(subj_vecs) else [0.0] * subj_dim)
            if body_vecs:
                comb.extend(body_vecs[i] if i < len(body_vecs) else [0.0] * body_dim)
            x_text.append(comb)
    
    # Note: 'features' (x) are the primary input for GNNs.
    # 'attrs' are supplementary raw values or metadata used for:
    # 1. Debugging/inspection (e.g. raw timestamps)
    # 2. Custom feature engineering in downstream tasks
    # 3. Filtering or stratification during analysis
    return {
        "ts": email_attrs_raw["ts"],
        "n_urls": email_attrs_raw["n_urls"],
        "len_body": email_attrs_raw["len_body"],
        "len_subject": email_attrs_raw.get("len_subject", []),
        "x_html_css": email_attrs_raw.get("x_html_css", []),
        "x_text": x_text if x_text and (len(x_text[0]) > 0 if x_text else False) else [],
    }


def _compute_degrees(ir: GraphIR, schema: GraphSchema, node_type: str) -> List[int]:
    """Compute total degree (in + out) for all nodes of a given type."""
    node = ir.nodes.get(node_type)
    if not node:
        return []
    num_nodes = len(node.x)
    degrees = [0] * num_nodes
    
    for edge_name, (srcs, dsts) in ir.edges.items():
        edge_def = schema.edges.get(edge_name)
        if not edge_def: 
            continue
            
        if edge_def.src == node_type:
            for idx in srcs:
                if idx < num_nodes: degrees[idx] += 1
        
        if edge_def.dst == node_type:
            for idx in dsts:
                if idx < num_nodes: degrees[idx] += 1
                
    return degrees


def _perform_collapse(ir: GraphIR, schema: GraphSchema, parent_type: str, child_type: str, edge_name: str) -> bool:
    if parent_type not in ir.nodes or child_type not in ir.nodes or edge_name not in ir.edges:
        return False
        
    parent_node = ir.nodes[parent_type]
    child_node = ir.nodes[child_type]
    src_indices, dst_indices = ir.edges[edge_name]
    
    degrees = _compute_degrees(ir, schema, child_type)
    
    collapsible_children = set()
    parent_to_collapsed_children = {} 
    
    for p, c in zip(src_indices, dst_indices):
        if c < len(degrees) and degrees[c] == 1:
            collapsible_children.add(c)
            if p not in parent_to_collapsed_children:
                parent_to_collapsed_children[p] = []
            parent_to_collapsed_children[p].append(c)
            
    if not collapsible_children:
        return False
        
    child_dim = len(child_node.x[0]) if child_node.x else 0
    if child_dim > 0:
        for i in range(len(parent_node.x)):
            if i in parent_to_collapsed_children:
                agg = [0.0] * child_dim
                for c_idx in parent_to_collapsed_children[i]:
                    c_feat = child_node.x[c_idx]
                    for k in range(child_dim):
                        agg[k] += c_feat[k]
                parent_node.x[i].extend(agg)
            else:
                parent_node.x[i].extend([0.0] * child_dim)
                
    old_to_new = {}
    new_x = []
    new_index_to_string = []
    new_index_map = {}
    new_attrs = {k: [] for k in child_node.attrs}
    
    kept_count = 0
    original_strings = child_node.index_to_string or []
    
    for i in range(len(child_node.x)):
        if i in collapsible_children:
            continue
            
        old_to_new[i] = kept_count
        new_x.append(child_node.x[i])
        
        if i < len(original_strings):
            s = original_strings[i]
            new_index_to_string.append(s)
            new_index_map[s] = kept_count
            
        for k, v_list in child_node.attrs.items():
            if i < len(v_list):
                new_attrs[k].append(v_list[i])
                
        kept_count += 1
        
    child_node.x = new_x
    child_node.index = new_index_map
    child_node.index_to_string = new_index_to_string
    child_node.attrs = new_attrs
    
    for ename, (esrc, edst) in ir.edges.items():
        edef = schema.edges.get(ename)
        if not edef: continue
        
        if edef.src == child_type:
            new_srcs, new_dsts = [], []
            for s, d in zip(esrc, edst):
                if s in old_to_new:
                    new_srcs.append(old_to_new[s])
                    new_dsts.append(d)
            ir.edges[ename] = (new_srcs, new_dsts)
            
        elif edef.dst == child_type:
            new_srcs, new_dsts = [], []
            for s, d in zip(esrc, edst):
                if d in old_to_new:
                    new_srcs.append(s)
                    new_dsts.append(old_to_new[d])
            ir.edges[ename] = (new_srcs, new_dsts)
            
    return True


def _collapse_graph_ir(ir: GraphIR, schema: GraphSchema) -> GraphIR:
    """
    Iteratively collapse 1:1 mappings where a child node is connected only to a single parent
    and has no other edges.
    """
    collapse_specs = list(schema.collapse_rules)
    
    while True:
        something_changed = False
        for parent_type, child_type, edge_name in collapse_specs:
            if _perform_collapse(ir, schema, parent_type, child_type, edge_name):
                something_changed = True
                
        if not something_changed:
            break
            
    return ir


def assemble_misp_graph_ir(
    misp_events: List[dict],
    *,
    schema: Optional[GraphSchema] = None,
    embeddings_output_dir: Optional[str] = None,
) -> GraphIR:
    """Assemble a backend-agnostic Graph IR from raw MISP events.

    High-level steps:
    1) Parse/normalize MISP events.
    2) Index unique component entities and URL parts.
    3) Build email->component edges and raw email attributes.
    4) Compute per-node features/attributes and text vectors.
    5) Assemble nodes, edges, and email_attrs blocks.
    """
    schema = schema or DEFAULT_SCHEMA
    emails = parse_misp_events(misp_events)
    indexed = index_entities(emails, schema, DEFAULT_PROVIDER_REGISTRY)
    indices = {k: v for k, v in indexed.items() if k != "url_components"}
    url_components = indexed["url_components"]
    edges_idx, email_meta, email_attrs_raw, docfreq_maps = materialize_edges(
        emails, indices, schema, DEFAULT_PROVIDER_REGISTRY
    )
    snd_dom_src, snd_dom_dst, rcv_dom_src, rcv_dom_dst = _connect_email_entities_to_domains(
        indices["sender"], indices["receiver"], indices["email_domain"]
    )
    (
        node_x,
        node_meta,
        node_attrs,
        subj_vecs,
        body_vecs,
        subj_dim,
        body_dim,
    ) = build_node_features(
        emails,
        schema,
        indices,
        url_components,
        docfreq_maps,
        DEFAULT_PROVIDER_REGISTRY,
        embeddings_output_dir=embeddings_output_dir,
    )

    # Use raw attributes for feature matrix construction
    # Normalization happens later in the pipeline (e.g. via normalizer.py)
    email_x = _build_email_feature_matrix(
        [float(v) for v in email_attrs_raw["ts"]],
        [float(v) for v in email_attrs_raw["len_body"]],
        [float(v) for v in email_attrs_raw["n_urls"]],
        [float(v) for v in email_attrs_raw["len_subject"]],
        subj_vecs,
        body_vecs,
        [list(v) if isinstance(v, list) else [] for v in email_attrs_raw.get("x_html_css", [])],
    )


    nodes = _assemble_nodes(
        schema,
        node_x,
        node_meta,
        node_attrs,
        indices,
        email_meta,
        email_x,
    )

    edges = _assemble_edges(
        schema,
        edges_idx,
        snd_dom_src, snd_dom_dst,
        rcv_dom_src, rcv_dom_dst,
    )

    email_attrs = _assemble_email_attrs(
        email_meta,
        email_attrs_raw,
        subj_dim,
        body_dim,
        subj_vecs,
        body_vecs,
    )

    ir = GraphIR(nodes=nodes, edges=edges, email_attrs=email_attrs)
    return _collapse_graph_ir(ir, schema)


__all__ = ["GraphIR", "NodeIR", "assemble_misp_graph_ir"]
