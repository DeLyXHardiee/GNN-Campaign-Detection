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
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

from tqdm import tqdm

from tqdm import tqdm

from .graph_schema import GraphSchema, DEFAULT_SCHEMA
from .url_skip_superspreaders import resolve_url_skip_superspreaders_patterns
try:
    from core.feature_set_extraction.url_extraction_utils import shard_url_infra_classify
except ModuleNotFoundError:
    from feature_set_extraction.url_extraction_utils import shard_url_infra_classify
from .common import (
    parse_misp_events,
    extract_email_domain,
    parse_url_components,
    to_unix_ts,
    compute_lexical_features,
    is_freemail_domain,
    to_str,
    auth_triple_to_onehot,
    is_sha256_hex,
)

# Authentication-Results header components (spf, dkim, dmarc) stored as string attributes on email nodes
AUTH_ATTR_KEYS = ("auth_spf", "auth_dkim", "auth_dmarc")

# Boolean email attributes (string "true"/"false" in data, stored as 0/1 in features and attrs)
EMAIL_BOOL_ATTR_KEYS = (
    "cyrillic_domain",
    "contains_symbols",
    "body_has_tracking_url",
    "body_has_tracking_image",
    "body_has_tracking_pixel",
    "body_has_unsubscribe_link",
    "domain_is_common_webprovided",
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


def _url_should_skip_for_superspreader(url: str, skip_substrings: Tuple[str, ...]) -> bool:
    """True if ``url`` contains any non-empty substring from the skip list (``url`` is lowercased in data)."""
    if not skip_substrings:
        return False
    u = url or ""
    return any(p and p in u for p in skip_substrings)


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
        vals = _as_email_list(email.get("attachments"))
        return [v for v in vals if is_sha256_hex(v)]
    if node_key == "html_structure_fingerprint":
        html = email.get("html") or {}
        if not isinstance(html, dict):
            return []
        v = to_str(html.get("structure_fingerprint", "")).strip().lower()
        return [v] if v else []
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
    if node_key == "helo_host":
        hops = email.get("received_hops") or []
        return [
            str(h.get("helo_host", "")).strip()
            for h in hops
            if isinstance(h, dict) and str(h.get("helo_host", "")).strip()
        ]
    if node_key == "return_path_email":
        rp = email.get("return_path") or {}
        if not isinstance(rp, dict):
            return []
        v = to_str(rp.get("email", "")).strip().lower()
        return [v] if v else []
    if node_key == "return_path_domain":
        rp = email.get("return_path") or {}
        if not isinstance(rp, dict):
            return []
        v = to_str(rp.get("domain", "")).strip().lower()
        return [v] if v else []
    return _as_email_list(email.get(f"{node_key}s") or email.get(node_key))


def _node_indexers(
    url_skip_substrings: Tuple[str, ...] = (),
    popular_domains: frozenset = frozenset(),
) -> Dict[str, Callable[[List[Dict[str, Any]]], Dict[str, int]]]:
    def _is_popular_url(u: str) -> bool:
        return bool(popular_domains) and shard_url_infra_classify(u, popular_domains)[0] == "benign"

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

    def url(emails: List[Dict[str, Any]]) -> Dict[str, int]:
        vals: List[str] = []
        for em in emails:
            for u in _as_email_list(em.get("urls")):
                if _url_should_skip_for_superspreader(u, url_skip_substrings):
                    continue
                if _is_popular_url(u):
                    continue
                vals.append(u)
        return _dedup_index(vals)

    def domain(emails: List[Dict[str, Any]]) -> Dict[str, int]:
        vals: List[str] = []
        for em in emails:
            for u in _as_email_list(em.get("urls")):
                if _is_popular_url(u):
                    continue
                d = parse_url_components(u).get("domain", "")
                if d:
                    vals.append(d)
        return _dedup_index(vals)

    def stem(emails: List[Dict[str, Any]]) -> Dict[str, int]:
        vals: List[str] = []
        for em in emails:
            for u in _as_email_list(em.get("urls")):
                if _is_popular_url(u):
                    continue
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
            vals.extend(_field_values_for_node(em, "attachment"))
        return _dedup_index(vals)

    def html_structure_fingerprint(emails: List[Dict[str, Any]]) -> Dict[str, int]:
        vals: List[str] = []
        for em in emails:
            vals.extend(_field_values_for_node(em, "html_structure_fingerprint"))
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

    def helo_host(emails: List[Dict[str, Any]]) -> Dict[str, int]:
        vals: List[str] = []
        for em in emails:
            vals.extend(_field_values_for_node(em, "helo_host"))
        return _dedup_index(vals)

    def return_path_email(emails: List[Dict[str, Any]]) -> Dict[str, int]:
        vals: List[str] = []
        for em in emails:
            vals.extend(_field_values_for_node(em, "return_path_email"))
        return _dedup_index(vals)

    def return_path_domain(emails: List[Dict[str, Any]]) -> Dict[str, int]:
        vals: List[str] = []
        for em in emails:
            vals.extend(_field_values_for_node(em, "return_path_domain"))
        return _dedup_index(vals)

    return {
        "sender": sender,
        "receiver": receiver,
        "url": url,
        "domain": domain,
        "stem": stem,
        "email_domain": email_domain,
        "attachment": attachment,
        "html_structure_fingerprint": html_structure_fingerprint,
        "origin_ip": origin_ip,
        "received_host": received_host,
        "helo_host": helo_host,
        "return_path_email": return_path_email,
        "return_path_domain": return_path_domain,
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
                elif dst_key == "html_structure_fingerprint":
                    docfreq_maps["html_structure_fingerprint_email_sets"].setdefault(value, set()).add(email_idx)
                elif dst_key == "origin_ip":
                    docfreq_maps["origin_ip_email_sets"].setdefault(value, set()).add(email_idx)
                elif dst_key == "received_host":
                    docfreq_maps["received_host_email_sets"].setdefault(value, set()).add(email_idx)
                elif dst_key == "helo_host":
                    docfreq_maps["helo_host_email_sets"].setdefault(value, set()).add(email_idx)
                elif dst_key == "return_path_email":
                    docfreq_maps["return_path_email_email_sets"].setdefault(value, set()).add(email_idx)
                elif dst_key == "return_path_domain":
                    docfreq_maps["return_path_domain_email_sets"].setdefault(value, set()).add(email_idx)

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

    def email_to_email_domain(
        email_ctx: Dict[str, Any],
        indices: Dict[str, Dict[str, int]],
        edges_idx: Dict[str, List[int]],
        docfreq_maps: Dict[str, Dict[str, Set[int]]],
        edge_name: str,
    ) -> None:
        email_idx = int(email_ctx["email_idx"])
        em = email_ctx["email"]
        email_domain_idx = indices.get("email_domain", {})
        seen: set = set()
        for addr in [
            *_as_email_list(em.get("senders")),
            *_as_email_list(em.get("receivers")),
        ]:
            d = extract_email_domain(addr)
            if d and not is_freemail_domain(d) and d in email_domain_idx and d not in seen:
                edges_idx[f"{edge_name}_src"].append(email_idx)
                edges_idx[f"{edge_name}_dst"].append(email_domain_idx[d])
                seen.add(d)

    return {
        "email_to_entity": email_to_entity,
        "email_to_domain_from_urls": email_to_domain_from_urls,
        "email_to_stem_from_urls": email_to_stem_from_urls,
        "email_to_email_domain": email_to_email_domain,
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
        "url": url,
        "domain": domain,
        "stem": stem,
        "email_domain": email_domain,
        "attachment": attachment,
        "html_structure_fingerprint": _str_len_docfreq("html_structure_fingerprint_email_sets"),
        "origin_ip": _str_len_docfreq("origin_ip_email_sets"),
        "received_host": _str_len_docfreq("received_host_email_sets"),
        "helo_host": _str_len_docfreq("helo_host_email_sets"),
        "return_path_email": _str_len_docfreq("return_path_email_email_sets"),
        "return_path_domain": _str_len_docfreq("return_path_domain_email_sets"),
    }


def default_provider_registry(
    url_skip_substrings: Tuple[str, ...] = (),
    popular_domains: frozenset = frozenset(),
) -> ProviderRegistry:
    return ProviderRegistry(
        node_indexers=_node_indexers(url_skip_substrings, popular_domains),
        edge_builders=_edge_builders(),
        node_feature_builders=_node_feature_builders(),
    )


DEFAULT_PROVIDER_REGISTRY = default_provider_registry()


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
    for em in tqdm(emails, total=len(emails), desc="Indexing URL components"):
        for u in _as_email_list(em.get("urls")):
            comp = parse_url_components(u)
            url_components[u] = (comp.get("domain", ""), comp.get("stem", ""))

    url_keys = set(indices.get("url") or {})
    url_components = {u: c for u, c in url_components.items() if u in url_keys}

    out: Dict[str, Any] = dict(indices)
    out["url_components"] = url_components
    return out


def materialize_edges(
    emails: List[Dict[str, Any]],
    indices: Dict[str, Dict[str, int]],
    schema: GraphSchema,
    registry: ProviderRegistry = DEFAULT_PROVIDER_REGISTRY,
    *,
    zero_email_timestamps: bool = False,
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
        **{k: [] for k in EMAIL_BOOL_ATTR_KEYS},
        **{k: [] for k in AUTH_ATTR_KEYS},
    }
    docfreq_maps: Dict[str, Dict[str, Set[int]]] = {
        "domain_email_sets": {},
        "stem_email_sets": {},
        "email_domain_sender_sets": {},
        "email_domain_receiver_sets": {},
        "url_email_sets": {},
        "attachment_email_sets": {},
        "html_structure_fingerprint_email_sets": {},
        "sender_email_sets": {},
        "receiver_email_sets": {},
        "origin_ip_email_sets": {},
        "received_host_email_sets": {},
        "helo_host_email_sets": {},
        "return_path_email_email_sets": {},
        "return_path_domain_email_sets": {},
    }

    for email_idx, em in enumerate(
        tqdm(emails, total=len(emails), desc="Materializing edges & email attrs")
    ):
        urls = _as_email_list(em.get("urls"))
        # external_id: MISP id for joining to ground truth; not used in feature matrix
        ext_id = em.get("external_id")
        email_meta.append(
            {
                "info": em.get("email_info", ""),
                "index": email_idx,
                "email_index": em.get("email_index", email_idx),
                "date": em.get("date", ""),
                "external_id": str(ext_id) if ext_id is not None else "",
            }
        )
        email_attrs_raw["ts"].append(
            0 if zero_email_timestamps else to_unix_ts(em.get("date", ""))
        )
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
        for k in EMAIL_BOOL_ATTR_KEYS:
            raw = str(em.get(k, "") or "").strip().lower()
            email_attrs_raw[k].append(1 if raw == "true" else 0)
        for k in AUTH_ATTR_KEYS:
            email_attrs_raw[k].append(str(em.get(k, "") or "").strip())

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

    from utils.embeddings import DEFAULT_OUTPUT_DIR, get_embeddings

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
    bool_attr_rows: Optional[List[List[float]]] = None,
    auth_onehot_rows: Optional[List[List[float]]] = None,
) -> List[List[float]]:
    """Construct the email feature matrix using raw scalars + text embeddings + boolean attrs + auth one-hot.

    Order: [ts, len_body, n_urls, len_subject, SBERT(subject), SBERT(body), html_css, bool_attrs(7), auth_onehot(18)]
    """
    n_emails = max(
        len(ts),
        len(len_body),
        len(n_urls),
        len(len_subject),
        len(subj_vecs) if subj_vecs else 0,
        len(body_vecs) if body_vecs else 0,
        len(html_css_vecs) if html_css_vecs else 0,
        len(bool_attr_rows) if bool_attr_rows else 0,
        len(auth_onehot_rows) if auth_onehot_rows else 0,
    )
    email_x: List[List[float]] = []
    for i in tqdm(range(n_emails), total=n_emails, desc="Building email feature matrix"):
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
        if bool_attr_rows and i < len(bool_attr_rows):
            row.extend(float(v) for v in bool_attr_rows[i])
        if auth_onehot_rows and i < len(auth_onehot_rows):
            row.extend(float(v) for v in auth_onehot_rows[i])
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
) -> Dict[str, Tuple[List[int], List[int]]]:
    edges: Dict[str, Tuple[List[int], List[int]]] = {}
    for edge_name in schema.edges:
        src_key = f"{edge_name}_src"
        dst_key = f"{edge_name}_dst"
        if src_key in edges_idx and dst_key in edges_idx:
            edges[edge_name] = (edges_idx[src_key], edges_idx[dst_key])
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
        for i in tqdm(range(n_emails), total=n_emails, desc="Assembling email text attrs"):
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
    out: Dict[str, Any] = {
        "ts": email_attrs_raw["ts"],
        "n_urls": email_attrs_raw["n_urls"],
        "len_body": email_attrs_raw["len_body"],
        "len_subject": email_attrs_raw.get("len_subject", []),
        "x_html_css": email_attrs_raw.get("x_html_css", []),
        "x_text": x_text if x_text and (len(x_text[0]) > 0 if x_text else False) else [],
    }
    for k in EMAIL_BOOL_ATTR_KEYS:
        out[k] = email_attrs_raw.get(k, [])
    for k in AUTH_ATTR_KEYS:
        out[k] = email_attrs_raw.get(k, [])
    out["external_id"] = [str(m.get("external_id") or "") for m in email_meta]
    return out


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
    zero_email_timestamps: bool = False,
    url_skip_superspreaders_path: Optional[str] = None,
    url_skip_substrings: Optional[Sequence[str]] = None,
    popular_domains: Optional[frozenset] = None,
) -> GraphIR:
    """Assemble a backend-agnostic Graph IR from raw MISP events.

    High-level steps:
    1) Parse/normalize MISP events.
    2) Index unique component entities and URL parts.
    3) Build email->component edges and raw email attributes.
    4) Compute per-node features/attributes and text vectors.
    5) Assemble nodes, edges, and email_attrs blocks.

    URLs whose string contains any line from ``core/graph/url_skip_superspreaders.txt``
    (substring match, case-insensitive) are omitted as ``url`` nodes; email→domain and
    email→stem edges for those URLs are unchanged. Pass ``url_skip_substrings`` to
    override the file (including ``[]`` to skip loading the file).
    """
    schema = schema or DEFAULT_SCHEMA
    url_skip_patterns = resolve_url_skip_superspreaders_patterns(
        path=url_skip_superspreaders_path,
        inline_substrings=url_skip_substrings,
    )
    pop_domains = popular_domains if popular_domains is not None else frozenset()
    registry = default_provider_registry(url_skip_patterns, pop_domains)
    emails = parse_misp_events(misp_events)
    #iterate through emails and print urls if not empty
    '''
    for email in emails:
        urls = _as_email_list(email.get("urls"))
        if len(urls) > 0 and email.get("external_id") == "trec_28184":
            print(f"Email index {email.get('email_index', 'N/A')} has URLs: {urls}")
    '''
    indexed = index_entities(emails, schema, registry)
    indices = {k: v for k, v in indexed.items() if k != "url_components"}
    url_components = indexed["url_components"]
    edges_idx, email_meta, email_attrs_raw, docfreq_maps = materialize_edges(
        emails,
        indices,
        schema,
        registry,
        zero_email_timestamps=zero_email_timestamps,
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
        registry,
        embeddings_output_dir=embeddings_output_dir,
    )

    # Use raw attributes for feature matrix construction
    # Normalization happens later in the pipeline (e.g. via normalizer.py)
    n_emails = len(email_attrs_raw["ts"])
    bool_attr_rows: List[List[float]] = [
        [float(email_attrs_raw.get(k, [0] * n_emails)[i]) for k in EMAIL_BOOL_ATTR_KEYS]
        for i in range(n_emails)
    ]
    auth_spf = email_attrs_raw.get("auth_spf", [""] * n_emails)
    auth_dkim = email_attrs_raw.get("auth_dkim", [""] * n_emails)
    auth_dmarc = email_attrs_raw.get("auth_dmarc", [""] * n_emails)
    auth_onehot_rows: List[List[float]] = [
        auth_triple_to_onehot(
            auth_spf[i] if i < len(auth_spf) else "",
            auth_dkim[i] if i < len(auth_dkim) else "",
            auth_dmarc[i] if i < len(auth_dmarc) else "",
        )
        for i in range(n_emails)
    ]
    email_x = _build_email_feature_matrix(
        [float(v) for v in email_attrs_raw["ts"]],
        [float(v) for v in email_attrs_raw["len_body"]],
        [float(v) for v in email_attrs_raw["n_urls"]],
        [float(v) for v in email_attrs_raw["len_subject"]],
        subj_vecs,
        body_vecs,
        [list(v) if isinstance(v, list) else [] for v in email_attrs_raw.get("x_html_css", [])],
        bool_attr_rows=bool_attr_rows,
        auth_onehot_rows=auth_onehot_rows,
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

    edges = _assemble_edges(schema, edges_idx)

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
