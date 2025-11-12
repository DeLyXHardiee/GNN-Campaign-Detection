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
from typing import Any, Dict, List, Optional, Tuple, Set

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
from .normalizer import zscore_list, minmax_list

# ----------------------------------------------------------------------------
# Graph IR data structures
# ----------------------------------------------------------------------------


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


def _index_uniques_and_url_components(
    emails: List[Dict[str, Any]]
) -> Tuple[
    Dict[str, int], Dict[str, int], Dict[str, int], Dict[str, int], Dict[str, int], Dict[str, int], Dict[str, int], Dict[str, int],
    Dict[str, Tuple[str, str]],
]:
    """Scan emails once to index unique entities and collect URL components.

    Returns dicts mapping string -> index for senders/receivers/weeks/subjects/urls/domains/stems/email_domains,
    along with a url_components map: url -> (domain, stem).
    """
    sender_to_idx: Dict[str, int] = {}
    receiver_to_idx: Dict[str, int] = {}
    week_to_idx: Dict[str, int] = {}
    subject_to_idx: Dict[str, int] = {}
    url_to_idx: Dict[str, int] = {}
    domain_to_idx: Dict[str, int] = {}
    stem_to_idx: Dict[str, int] = {}
    email_domain_to_idx: Dict[str, int] = {}
    url_components: Dict[str, Tuple[str, str]] = {}    # Track URL -> (domain, stem) mapping for edge creation

    for em in emails:
        sender = em.get("sender")
        if sender:
            sender_to_idx.setdefault(sender, len(sender_to_idx))
            sender_domain = extract_email_domain(sender)
            if sender_domain and not is_freemail_domain(sender_domain):
                email_domain_to_idx.setdefault(sender_domain, len(email_domain_to_idx))

        for r in em.get("receivers", []) or []:
            if r:
                receiver_to_idx.setdefault(r, len(receiver_to_idx))
                receiver_domain = extract_email_domain(r)
                if receiver_domain and not is_freemail_domain(receiver_domain):
                    email_domain_to_idx.setdefault(receiver_domain, len(email_domain_to_idx))

        week_key = extract_week_key(em.get("date", ""))
        if week_key:
            week_to_idx.setdefault(week_key, len(week_to_idx))

        subj = em.get("subject", "")
        if subj:
            subject_to_idx.setdefault(subj, len(subject_to_idx))

        for u in em.get("urls", []) or []:
            if not u:
                continue
            url_to_idx.setdefault(u, len(url_to_idx))
            comp = parse_url_components(u)
            d = comp.get("domain", "")
            s = comp.get("stem", "")
            if d:
                domain_to_idx.setdefault(d, len(domain_to_idx))
            if s:
                stem_to_idx.setdefault(s, len(stem_to_idx))
            url_components[u] = (d, s)

    return (
        sender_to_idx,
        receiver_to_idx,
        week_to_idx,
        subject_to_idx,
        url_to_idx,
        domain_to_idx,
        stem_to_idx,
        email_domain_to_idx,
        url_components,
    )


def _collect_edges_and_email_attrs(
    emails: List[Dict[str, Any]],
    sender_to_idx: Dict[str, int],
    receiver_to_idx: Dict[str, int],
    week_to_idx: Dict[str, int],
    subject_to_idx: Dict[str, int],
    url_to_idx: Dict[str, int],
    email_domain_to_idx: Dict[str, int],
) -> Tuple[
    Dict[str, List[int]],
    List[Dict[str, Any]],
    Dict[str, List[int]],
    Dict[str, Set[int]],
]:
    """Build email->component edges and gather raw email attributes and per-node docfreq sets."""
    # Edge indices
    edges_idx: Dict[str, List[int]] = {
        "has_sender_src": [],
        "has_sender_dst": [],
        "has_receiver_src": [],
        "has_receiver_dst": [],
        "in_week_src": [],
        "in_week_dst": [],
        "has_subject_src": [],
        "has_subject_dst": [],
        "has_url_src": [],
        "has_url_dst": [],
    }

    # Email meta and raw attributes
    email_meta: List[Dict[str, Any]] = []
    email_attrs_raw: Dict[str, List[int]] = {
        "ts": [],
        "n_urls": [],
        "len_subject": [],
        "len_body": [],
    }

    # Per-node sets for docfreq/statistics
    docfreq_sets: Dict[str, Set[int]] = {
        "domain": set(),  # tracked per domain value via map below
        "stem": set(),
    }
    # String-keyed maps to sets
    domain_email_sets: Dict[str, Set[int]] = {}
    stem_email_sets: Dict[str, Set[int]] = {}
    email_domain_sender_sets: Dict[str, Set[int]] = {}
    email_domain_receiver_sets: Dict[str, Set[int]] = {}
    url_email_sets: Dict[str, Set[int]] = {}
    sender_email_sets: Dict[str, Set[int]] = {}
    receiver_email_sets: Dict[str, Set[int]] = {}

    for email_idx, em in enumerate(emails):
        subj = em.get("subject", "")
        body = em.get("body", "")
        urls = em.get("urls", []) or []

        email_meta.append({
            "info": em.get("email_info", ""),
            "index": email_idx,
            "date": em.get("date", ""),
        })

        # Raw attributes
        ts_val = to_unix_ts(em.get("date", ""))
        email_attrs_raw["ts"].append(ts_val)

        domains: Set[str] = set()
        for u in urls:
            if not u:
                continue
            comp = parse_url_components(u)
            d = comp.get("domain", "")
            if d:
                domains.add(d)
                domain_email_sets.setdefault(d, set()).add(email_idx)
            s = comp.get("stem", "")
            if s:
                stem_email_sets.setdefault(s, set()).add(email_idx)

        email_attrs_raw["n_urls"].append(int(len(domains)))
        email_attrs_raw["len_subject"].append(int(len(subj)))
        email_attrs_raw["len_body"].append(int(len(body)))

        # Edges from email to components
        if em.get("sender") and em["sender"] in sender_to_idx:
            edges_idx["has_sender_src"].append(email_idx)
            edges_idx["has_sender_dst"].append(sender_to_idx[em["sender"]])
            sender_email_sets.setdefault(em["sender"], set()).add(email_idx)
            s_dom = extract_email_domain(em["sender"])
            if s_dom and not is_freemail_domain(s_dom):
                email_domain_sender_sets.setdefault(s_dom, set()).add(email_idx)

        for r in em.get("receivers", []) or []:
            if r and r in receiver_to_idx:
                edges_idx["has_receiver_src"].append(email_idx)
                edges_idx["has_receiver_dst"].append(receiver_to_idx[r])
                receiver_email_sets.setdefault(r, set()).add(email_idx)
            r_dom = extract_email_domain(r)
            if r_dom and not is_freemail_domain(r_dom):
                email_domain_receiver_sets.setdefault(r_dom, set()).add(email_idx)

        wk = extract_week_key(em.get("date", ""))
        if wk and wk in week_to_idx:
            edges_idx["in_week_src"].append(email_idx)
            edges_idx["in_week_dst"].append(week_to_idx[wk])

        if subj and subj in subject_to_idx:
            edges_idx["has_subject_src"].append(email_idx)
            edges_idx["has_subject_dst"].append(subject_to_idx[subj])

        for u in urls:
            if u and u in url_to_idx:
                edges_idx["has_url_src"].append(email_idx)
                edges_idx["has_url_dst"].append(url_to_idx[u])
                url_email_sets.setdefault(u, set()).add(email_idx)

    # Return also the docfreq maps used downstream
    docfreq_maps: Dict[str, Dict[str, Set[int]]] = {
        "domain_email_sets": domain_email_sets,
        "stem_email_sets": stem_email_sets,
        "email_domain_sender_sets": email_domain_sender_sets,
        "email_domain_receiver_sets": email_domain_receiver_sets,
        "url_email_sets": url_email_sets,
        "sender_email_sets": sender_email_sets,
        "receiver_email_sets": receiver_email_sets,
    }

    return edges_idx, email_meta, email_attrs_raw, docfreq_maps


def _build_url_component_edges(
    url_components: Dict[str, Tuple[str, str]],
    url_to_idx: Dict[str, int],
    domain_to_idx: Dict[str, int],
    stem_to_idx: Dict[str, int],
) -> Tuple[List[int], List[int], List[int], List[int]]:
    """Create url->domain and url->stem edge index lists."""
    url_to_domain_src: List[int] = []
    url_to_domain_dst: List[int] = []
    url_to_stem_src: List[int] = []
    url_to_stem_dst: List[int] = []
    for url, (domain, stem) in url_components.items():
        u_idx = url_to_idx[url]
        if domain and domain in domain_to_idx:
            url_to_domain_src.append(u_idx)
            url_to_domain_dst.append(domain_to_idx[domain])
        if stem and stem in stem_to_idx:
            url_to_stem_src.append(u_idx)
            url_to_stem_dst.append(stem_to_idx[stem])
    return url_to_domain_src, url_to_domain_dst, url_to_stem_src, url_to_stem_dst


def _connect_email_entities_to_domains(
    sender_to_idx: Dict[str, int],
    receiver_to_idx: Dict[str, int],
    email_domain_to_idx: Dict[str, int],
) -> Tuple[List[int], List[int], List[int], List[int]]:
    """Create edges from sender/receiver nodes to their email domain nodes."""
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


def _compute_node_attributes_and_features(
    sender_to_idx: Dict[str, int],
    receiver_to_idx: Dict[str, int],
    week_to_idx: Dict[str, int],
    subject_to_idx: Dict[str, int],
    url_to_idx: Dict[str, int],
    domain_to_idx: Dict[str, int],
    stem_to_idx: Dict[str, int],
    email_domain_to_idx: Dict[str, int],
    url_components: Dict[str, Tuple[str, str]],
    docfreq_maps: Dict[str, Dict[str, Set[int]]],
    email_attrs_raw: Dict[str, List[int]],
    emails: List[Dict[str, Any]],
) -> Tuple[
    Dict[str, List[List[float]]],
    Dict[str, List[str]],
    Dict[str, Dict[str, List[Any]]],
    Dict[str, Any],
    List[List[float]],
    List[List[float]],
    int,
    int,
]:
    """Compute per-node x features and attributes, plus email normalization and text vectors."""
    # Ordered metadata lists
    sender_meta = _ordered_keys(sender_to_idx)
    receiver_meta = _ordered_keys(receiver_to_idx)
    week_meta = _ordered_keys(week_to_idx)
    subject_meta = _ordered_keys(subject_to_idx)
    url_meta = _ordered_keys(url_to_idx)
    domain_meta = _ordered_keys(domain_to_idx)
    stem_meta = _ordered_keys(stem_to_idx)
    email_domain_meta = _ordered_keys(email_domain_to_idx)

    # Base numeric x for simple nodes
    sender_x = [[float(len(s))] for s in sender_meta]
    receiver_x = [[float(len(s))] for s in receiver_meta]
    week_x = [[float(idx)] for idx in range(len(week_meta))]
    subject_x = [[float(len(s))] for s in subject_meta]
    stem_x = [[float(len(s))] for s in stem_meta]
    email_domain_x = [[float(len(d))] for d in email_domain_meta]

    # URL x: path length zscore
    url_path_lens: List[float] = []
    for u in url_meta:
        if u in url_components:
            _, stem = url_components[u]
            url_path_lens.append(float(len(stem or "/")))
        else:
            comp = parse_url_components(u)
            url_path_lens.append(float(len(comp.get("stem", "/"))))
    url_path_len_z, _, _ = zscore_list(url_path_lens)
    url_x = [[float(url_path_len_z[i])] for i in range(len(url_path_len_z))]

    # Domain attrs
    domain_x_lex: List[List[float]] = [compute_lexical_features(d) for d in domain_meta]
    domain_entropies: List[float] = [v[7] if len(v) > 7 else 0.0 for v in domain_x_lex]
    domain_entropy_z, _, _ = zscore_list(domain_entropies)
    domain_x = [[float(domain_entropy_z[i])] for i in range(len(domain_entropy_z))]
    domain_docfreq: List[int] = [len(docfreq_maps["domain_email_sets"].get(d, set())) for d in domain_meta]

    # Stem attrs
    stem_x_lex: List[List[float]] = [compute_lexical_features(s) for s in stem_meta]
    stem_docfreq: List[int] = [len(docfreq_maps["stem_email_sets"].get(s, set())) for s in stem_meta]

    # Email domain attrs
    email_domain_x_lex: List[List[float]] = [compute_lexical_features(d) for d in email_domain_meta]
    email_domain_docfreq_sender: List[int] = [len(docfreq_maps["email_domain_sender_sets"].get(d, set())) for d in email_domain_meta]
    email_domain_docfreq_receiver: List[int] = [len(docfreq_maps["email_domain_receiver_sets"].get(d, set())) for d in email_domain_meta]

    # Subject normalization
    subject_lens: List[int] = [int(len(s)) for s in subject_meta]
    subject_len_z, _, _ = zscore_list([float(v) for v in subject_lens])
    subject_x = [[float(subject_len_z[i])] for i in range(len(subject_len_z))]

    # Email normalization
    len_body_z, _, _ = zscore_list([float(v) for v in email_attrs_raw["len_body"]])
    ts_minmax, _, _ = minmax_list([float(v) for v in email_attrs_raw["ts"]])

    # Text vectorization (TF-IDF)
    TEXT_SUBJ_MAX_FEATS = 128
    TEXT_BODY_MAX_FEATS = 256
    subj_vecs: List[List[float]] = []
    body_vecs: List[List[float]] = []
    subj_dim = 0
    body_dim = 0
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
        subj_corpus: List[str] = [(em.get("subject") or "").strip() for em in emails]
        body_corpus: List[str] = [(em.get("body") or "").strip() for em in emails]
        if any(bool(t) for t in subj_corpus):
            subj_vec = TfidfVectorizer(
                max_features=TEXT_SUBJ_MAX_FEATS,
                ngram_range=(1, 2),
                min_df=1,
                stop_words="english",
                strip_accents="unicode",
                lowercase=True,
            ).fit_transform(subj_corpus)
            subj_dim = int(subj_vec.shape[1])
            if subj_dim > 0:
                subj_vecs = [row.toarray().astype("float32")[0].tolist() for row in subj_vec]
        if any(bool(t) for t in body_corpus):
            body_vec = TfidfVectorizer(
                max_features=TEXT_BODY_MAX_FEATS,
                ngram_range=(1, 1),
                min_df=1,
                stop_words="english",
                strip_accents="unicode",
                lowercase=True,
            ).fit_transform(body_corpus)
            body_dim = int(body_vec.shape[1])
            if body_dim > 0:
                body_vecs = [row.toarray().astype("float32")[0].tolist() for row in body_vec]
    except Exception:
        subj_vecs, body_vecs = [], []
        subj_dim = body_dim = 0

    node_x: Dict[str, List[List[float]]] = {
        "sender": sender_x,
        "receiver": receiver_x,
        "week": week_x,
        "subject": subject_x,
        "url": url_x,
        "domain": domain_x,
        "stem": stem_x,
        "email_domain": email_domain_x,
    }
    node_meta: Dict[str, List[str]] = {
        "sender": sender_meta,
        "receiver": receiver_meta,
        "week": week_meta,
        "subject": subject_meta,
        "url": url_meta,
        "domain": domain_meta,
        "stem": stem_meta,
        "email_domain": email_domain_meta,
    }
    node_attrs: Dict[str, Dict[str, List[Any]]] = {
        "sender": {"docfreq": [len(docfreq_maps["sender_email_sets"].get(s, set())) for s in sender_meta]},
        "receiver": {"docfreq": [len(docfreq_maps["receiver_email_sets"].get(r, set())) for r in receiver_meta]},
        "subject": {"len_subject": subject_lens, "len_subject_z": [float(z) for z in subject_len_z]},
        "url": {"docfreq": [len(docfreq_maps["url_email_sets"].get(u, set())) for u in url_meta]},
        "domain": {"x_lex": domain_x_lex, "docfreq": domain_docfreq},
        "stem": {"x_lex": stem_x_lex, "docfreq": stem_docfreq},
        "email_domain": {
            "x_lex": email_domain_x_lex,
            "docfreq_sender": email_domain_docfreq_sender,
            "docfreq_receiver": email_domain_docfreq_receiver,
        },
    }
    email_norm: Dict[str, Any] = {"len_body_z": len_body_z, "ts_minmax": ts_minmax}

    return node_x, node_meta, node_attrs, email_norm, subj_vecs, body_vecs, subj_dim, body_dim


def _build_email_feature_matrix(
    email_len_body: List[int],
    email_n_urls: List[int],
    ts_minmax: List[float],
    len_body_z: List[float],
    subj_vecs: List[List[float]],
    body_vecs: List[List[float]],
    subj_dim: int,
    body_dim: int,
) -> List[List[float]]:
    """Construct the final email feature matrix in the agreed order."""
    n_emails = len(email_len_body)
    email_x: List[List[float]] = []
    for i in range(n_emails):
        row: List[float] = [
            float(email_len_body[i]) if i < len(email_len_body) else 0.0,
            float(email_n_urls[i]) if i < len(email_n_urls) else 0.0,
            float(ts_minmax[i]) if i < len(ts_minmax) else 0.0,
            float(len_body_z[i]) if i < len(len_body_z) else 0.0,
        ]
        if subj_vecs:
            row.extend(subj_vecs[i] if i < len(subj_vecs) else [0.0] * subj_dim)
        if body_vecs:
            row.extend(body_vecs[i] if i < len(body_vecs) else [0.0] * body_dim)
        email_x.append(row)
    return email_x


def _assemble_nodes(
    node_x: Dict[str, List[List[float]]],
    node_meta: Dict[str, List[str]],
    node_attrs: Dict[str, Dict[str, List[Any]]],
    indices: Dict[str, Dict[str, int]],
    email_meta: List[Dict[str, Any]],
    email_x: List[List[float]],
) -> Dict[str, NodeIR]:
    """Create the nodes dict for GraphIR from parts."""
    return {
        "email": NodeIR(index={}, x=email_x, index_to_meta=email_meta),
        "sender": NodeIR(index=indices["sender"], x=node_x["sender"], index_to_string=node_meta["sender"],
                          attrs=node_attrs.get("sender", {})),
        "receiver": NodeIR(index=indices["receiver"], x=node_x["receiver"], index_to_string=node_meta["receiver"],
                            attrs=node_attrs.get("receiver", {})),
        "week": NodeIR(index=indices["week"], x=node_x["week"], index_to_string=node_meta["week"]),
        "subject": NodeIR(index=indices["subject"], x=node_x["subject"], index_to_string=node_meta["subject"],
                           attrs=node_attrs.get("subject", {})),
        "url": NodeIR(index=indices["url"], x=node_x["url"], index_to_string=node_meta["url"],
                       attrs=node_attrs.get("url", {})),
        "domain": NodeIR(index=indices["domain"], x=node_x["domain"], index_to_string=node_meta["domain"],
                          attrs=node_attrs.get("domain", {})),
        "stem": NodeIR(index=indices["stem"], x=node_x["stem"], index_to_string=node_meta["stem"],
                        attrs=node_attrs.get("stem", {})),
        "email_domain": NodeIR(index=indices["email_domain"], x=node_x["email_domain"], index_to_string=node_meta["email_domain"],
                                 attrs=node_attrs.get("email_domain", {})),
    }


def _assemble_edges(
    edges_idx: Dict[str, List[int]],
    url_dom_src: List[int], url_dom_dst: List[int],
    url_stem_src: List[int], url_stem_dst: List[int],
    snd_dom_src: List[int], snd_dom_dst: List[int],
    rcv_dom_src: List[int], rcv_dom_dst: List[int],
) -> Dict[str, Tuple[List[int], List[int]]]:
    """Create the edges dict for GraphIR from parts."""
    return {
        "has_sender": (edges_idx["has_sender_src"], edges_idx["has_sender_dst"]),
        "has_receiver": (edges_idx["has_receiver_src"], edges_idx["has_receiver_dst"]),
        "in_week": (edges_idx["in_week_src"], edges_idx["in_week_dst"]),
        "has_subject": (edges_idx["has_subject_src"], edges_idx["has_subject_dst"]),
        "has_url": (edges_idx["has_url_src"], edges_idx["has_url_dst"]),
        "url_has_domain": (url_dom_src, url_dom_dst),
        "url_has_stem": (url_stem_src, url_stem_dst),
        "sender_from_domain": (snd_dom_src, snd_dom_dst),
        "receiver_from_domain": (rcv_dom_src, rcv_dom_dst),
    }


def _assemble_email_attrs(
    email_meta: List[Dict[str, Any]],
    email_attrs_raw: Dict[str, List[int]],
    email_norm: Dict[str, Any],
    subj_dim: int,
    body_dim: int,
    subj_vecs: List[List[float]],
    body_vecs: List[List[float]],
) -> Dict[str, Any]:
    """Create the email attributes dict, including optional text vectors and dims."""
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
    return {
        "ts": email_attrs_raw["ts"],
        "n_urls": email_attrs_raw["n_urls"],
        "len_body": email_attrs_raw["len_body"],
        "x_text": x_text if x_text and (len(x_text[0]) > 0 if x_text else False) else [],
        "x_text_subject_dim": [int(subj_dim)] * n_emails,
        "x_text_body_dim": [int(body_dim)] * n_emails,
        "len_body_z": email_norm.get("len_body_z", []),
        "ts_minmax": email_norm.get("ts_minmax", []),
    }


def assemble_misp_graph_ir(misp_events: List[dict], *, schema: Optional[GraphSchema] = None) -> GraphIR:
    """Assemble a backend-agnostic Graph IR from raw MISP events.

    High-level steps:
    1) Parse/normalize MISP events.
    2) Index unique component entities and URL parts.
    3) Build email->component edges and raw email attributes.
    4) Compute per-node features/attributes, email normalization, and text vectors.
    5) Assemble nodes, edges, and email_attrs blocks.
    """
    schema = schema or DEFAULT_SCHEMA
    emails = parse_misp_events(misp_events)

    (
        sender_to_idx,
        receiver_to_idx,
        week_to_idx,
        subject_to_idx,
        url_to_idx,
        domain_to_idx,
        stem_to_idx,
        email_domain_to_idx,
        url_components,
    ) = _index_uniques_and_url_components(emails)

    edges_idx, email_meta, email_attrs_raw, docfreq_maps = _collect_edges_and_email_attrs(
        emails,
        sender_to_idx,
        receiver_to_idx,
        week_to_idx,
        subject_to_idx,
        url_to_idx,
        email_domain_to_idx,
    )

    url_dom_src, url_dom_dst, url_stem_src, url_stem_dst = _build_url_component_edges(
        url_components, url_to_idx, domain_to_idx, stem_to_idx
    )

    snd_dom_src, snd_dom_dst, rcv_dom_src, rcv_dom_dst = _connect_email_entities_to_domains(
        sender_to_idx, receiver_to_idx, email_domain_to_idx
    )

    (
        node_x,
        node_meta,
        node_attrs,
        email_norm,
        subj_vecs,
        body_vecs,
        subj_dim,
        body_dim,
    ) = _compute_node_attributes_and_features(
        sender_to_idx,
        receiver_to_idx,
        week_to_idx,
        subject_to_idx,
        url_to_idx,
        domain_to_idx,
        stem_to_idx,
        email_domain_to_idx,
        url_components,
        docfreq_maps,
        email_attrs_raw,
        emails,
    )

    email_x = _build_email_feature_matrix(
        email_attrs_raw["len_body"],
        email_attrs_raw["n_urls"],
        email_norm.get("ts_minmax", []),
        email_norm.get("len_body_z", []),
        subj_vecs,
        body_vecs,
        subj_dim,
        body_dim,
    )

    nodes = _assemble_nodes(
        node_x,
        node_meta,
        node_attrs,
        {
            "sender": sender_to_idx,
            "receiver": receiver_to_idx,
            "week": week_to_idx,
            "subject": subject_to_idx,
            "url": url_to_idx,
            "domain": domain_to_idx,
            "stem": stem_to_idx,
            "email_domain": email_domain_to_idx,
        },
        email_meta,
        email_x,
    )

    edges = _assemble_edges(
        edges_idx,
        url_dom_src, url_dom_dst,
        url_stem_src, url_stem_dst,
        snd_dom_src, snd_dom_dst,
        rcv_dom_src, rcv_dom_dst,
    )

    email_attrs = _assemble_email_attrs(
        email_meta,
        email_attrs_raw,
        email_norm,
        subj_dim,
        body_dim,
        subj_vecs,
        body_vecs,
    )

    return GraphIR(nodes=nodes, edges=edges, email_attrs=email_attrs)


__all__ = ["GraphIR", "NodeIR", "assemble_misp_graph_ir"]
