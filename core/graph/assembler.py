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
    Dict[str, int], Dict[str, int], Dict[str, int], Dict[str, int], Dict[str, int], Dict[str, int], Dict[str, int],
    Dict[str, Tuple[str, str]],
]:
    """Scan emails once to index unique entities and collect URL components.

    Returns dicts mapping string -> index for senders/receivers/weeks/urls/domains/stems/email_domains,
    along with a url_components map: url -> (domain, stem).
    """
    sender_to_idx: Dict[str, int] = {}
    receiver_to_idx: Dict[str, int] = {}
    week_to_idx: Dict[str, int] = {}
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
    url_to_idx: Dict[str, int],
    domain_to_idx: Dict[str, int],
    stem_to_idx: Dict[str, int],
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
        "has_url_src": [],
        "has_url_dst": [],
        "has_domain_src": [],
        "has_domain_dst": [],
        "has_stem_src": [],
        "has_stem_dst": [],
    }

    email_meta: List[Dict[str, Any]] = []
    email_attrs_raw: Dict[str, List[int]] = {
        "ts": [],
        "n_urls": [],
        "len_subject": [],
        "len_body": [],
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
            "email_index": em.get("email_index", email_idx),
            "date": em.get("date", ""),
        })

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
                if d in domain_to_idx:
                    edges_idx["has_domain_src"].append(email_idx)
                    edges_idx["has_domain_dst"].append(domain_to_idx[d])
            s = comp.get("stem", "")
            if s:
                stem_email_sets.setdefault(s, set()).add(email_idx)
                if s in stem_to_idx:
                    edges_idx["has_stem_src"].append(email_idx)
                    edges_idx["has_stem_dst"].append(stem_to_idx[s])

        email_attrs_raw["n_urls"].append(int(len(domains)))
        email_attrs_raw["len_subject"].append(int(len(subj)))
        email_attrs_raw["len_body"].append(int(len(body)))

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

        for u in urls:
            if u and u in url_to_idx:
                edges_idx["has_url_src"].append(email_idx)
                edges_idx["has_url_dst"].append(url_to_idx[u])
                url_email_sets.setdefault(u, set()).add(email_idx)

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


def _compute_node_attributes_and_features(
    sender_to_idx: Dict[str, int],
    receiver_to_idx: Dict[str, int],
    week_to_idx: Dict[str, int],
    url_to_idx: Dict[str, int],
    domain_to_idx: Dict[str, int],
    stem_to_idx: Dict[str, int],
    email_domain_to_idx: Dict[str, int],
    url_components: Dict[str, Tuple[str, str]],
    docfreq_maps: Dict[str, Dict[str, Set[int]]],
    emails: List[Dict[str, Any]],
) -> Tuple[
    Dict[str, List[List[float]]],
    Dict[str, List[str]],
    Dict[str, Dict[str, List[Any]]],
    List[List[float]],
    List[List[float]],
    int,
    int,
]:
    sender_meta = _ordered_keys(sender_to_idx)
    receiver_meta = _ordered_keys(receiver_to_idx)
    week_meta = _ordered_keys(week_to_idx)
    url_meta = _ordered_keys(url_to_idx)
    domain_meta = _ordered_keys(domain_to_idx)
    stem_meta = _ordered_keys(stem_to_idx)
    email_domain_meta = _ordered_keys(email_domain_to_idx)

    sender_len = [float(len(s)) for s in sender_meta]
    sender_x = [[sender_len[i]] for i in range(len(sender_len))]

    receiver_len = [float(len(r)) for r in receiver_meta]
    receiver_x = [[receiver_len[i]] for i in range(len(receiver_len))]

    week_indices = [float(idx) for idx in range(len(week_meta))]
    week_x = [[week_indices[i]] for i in range(len(week_indices))]

    stem_len = [float(len(s)) for s in stem_meta]
    stem_x = [[stem_len[i]] for i in range(len(stem_len))]

    email_domain_len = [float(len(d)) for d in email_domain_meta]
    email_domain_x = [[email_domain_len[i]] for i in range(len(email_domain_len))]

    url_path_lens: List[float] = []
    for u in url_meta:
        if u in url_components:
            _, stem = url_components[u]
            url_path_lens.append(float(len(stem or "/")))
        else:
            comp = parse_url_components(u)
            url_path_lens.append(float(len(comp.get("stem", "/"))))
    url_x = [[float(url_path_lens[i])] for i in range(len(url_path_lens))]
    url_docfreq: List[int] = [len(docfreq_maps["url_email_sets"].get(u, set())) for u in url_meta]
    url_x_lex: List[List[float]] = [compute_lexical_features(u) for u in url_meta]

    domain_x_lex: List[List[float]] = [compute_lexical_features(d) for d in domain_meta]
    domain_entropies: List[float] = [v[7] if len(v) > 7 else 0.0 for v in domain_x_lex]
    domain_x = [[float(domain_entropies[i])] for i in range(len(domain_entropies))]
    domain_docfreq: List[int] = [len(docfreq_maps["domain_email_sets"].get(d, set())) for d in domain_meta]

    stem_x_lex: List[List[float]] = [compute_lexical_features(s) for s in stem_meta]
    stem_docfreq: List[int] = [len(docfreq_maps["stem_email_sets"].get(s, set())) for s in stem_meta]

    email_domain_x_lex: List[List[float]] = [compute_lexical_features(d) for d in email_domain_meta]
    email_domain_docfreq_sender: List[int] = [len(docfreq_maps["email_domain_sender_sets"].get(d, set())) for d in email_domain_meta]
    email_domain_docfreq_receiver: List[int] = [len(docfreq_maps["email_domain_receiver_sets"].get(d, set())) for d in email_domain_meta]

    sender_docfreq: List[int] = [len(docfreq_maps["sender_email_sets"].get(s, set())) for s in sender_meta]
    receiver_docfreq: List[int] = [len(docfreq_maps["receiver_email_sets"].get(r, set())) for r in receiver_meta]

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
        "url": url_x,
        "domain": domain_x,
        "stem": stem_x,
        "email_domain": email_domain_x,
    }
    node_meta: Dict[str, List[str]] = {
        "sender": sender_meta,
        "receiver": receiver_meta,
        "week": week_meta,
        "url": url_meta,
        "domain": domain_meta,
        "stem": stem_meta,
        "email_domain": email_domain_meta,
    }
    node_attrs: Dict[str, Dict[str, List[Any]]] = {
        "sender": {"docfreq": sender_docfreq},
        "receiver": {"docfreq": receiver_docfreq},
        "url": {
            "x_lex": url_x_lex,
            "docfreq": url_docfreq,
        },
        "domain": {
            "x_lex": domain_x_lex,
            "docfreq": domain_docfreq,
        },
        "stem": {
            "x_lex": stem_x_lex,
            "docfreq": stem_docfreq,
        },
        "email_domain": {
            "x_lex": email_domain_x_lex,
            "docfreq_sender": email_domain_docfreq_sender,
            "docfreq_receiver": email_domain_docfreq_receiver,
        },
    }

    return node_x, node_meta, node_attrs, subj_vecs, body_vecs, subj_dim, body_dim




def _build_email_feature_matrix(
    ts: List[float],
    len_body: List[float],
    n_urls: List[float],
    len_subject: List[float],
    subj_vecs: List[List[float]],
    body_vecs: List[List[float]],
) -> List[List[float]]:
    """Construct the email feature matrix using raw scalars + TF-IDF vectors.

    Order: [ts, len_body, n_urls, len_subject, TFIDF(subject), TFIDF(body)]
    """
    n_emails = max(len(ts), len(len_body), len(n_urls), len(len_subject), len(subj_vecs) if subj_vecs else 0, len(body_vecs) if body_vecs else 0)
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
    return {
        "email": NodeIR(index={}, x=email_x, index_to_meta=email_meta),
        "sender": NodeIR(index=indices["sender"], x=node_x["sender"], index_to_string=node_meta["sender"],
                          attrs=node_attrs.get("sender", {})),
        "receiver": NodeIR(index=indices["receiver"], x=node_x["receiver"], index_to_string=node_meta["receiver"],
                            attrs=node_attrs.get("receiver", {})),
        "week": NodeIR(index=indices["week"], x=node_x["week"], index_to_string=node_meta["week"]),
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
    snd_dom_src: List[int], snd_dom_dst: List[int],
    rcv_dom_src: List[int], rcv_dom_dst: List[int],
) -> Dict[str, Tuple[List[int], List[int]]]:
    return {
        "has_sender": (edges_idx["has_sender_src"], edges_idx["has_sender_dst"]),
        "has_receiver": (edges_idx["has_receiver_src"], edges_idx["has_receiver_dst"]),
        "in_week": (edges_idx["in_week_src"], edges_idx["in_week_dst"]),
        "has_url": (edges_idx["has_url_src"], edges_idx["has_url_dst"]),
        "has_domain": (edges_idx["has_domain_src"], edges_idx["has_domain_dst"]),
        "has_stem": (edges_idx["has_stem_src"], edges_idx["has_stem_dst"]),
        "sender_from_domain": (snd_dom_src, snd_dom_dst),
        "receiver_from_domain": (rcv_dom_src, rcv_dom_dst),
    }


def _assemble_email_attrs(
    email_meta: List[Dict[str, Any]],
    email_attrs_raw: Dict[str, List[int]],
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
    # Define hierarchy of collapses (Parent, Child, Edge)
    # Order matters slightly for efficiency, but loop handles dependencies.
    collapse_specs = [
        ("sender", "email_domain", "sender_from_domain"),
        ("receiver", "email_domain", "receiver_from_domain"),
        ("email", "sender", "has_sender"),
        ("email", "receiver", "has_receiver"),
        ("email", "url", "has_url"),
        ("email", "domain", "has_domain"),
        ("email", "stem", "has_stem"),
    ]
    
    while True:
        something_changed = False
        for parent_type, child_type, edge_name in collapse_specs:
            if _perform_collapse(ir, schema, parent_type, child_type, edge_name):
                something_changed = True
                
        if not something_changed:
            break
            
    return ir


def assemble_misp_graph_ir(misp_events: List[dict], *, schema: Optional[GraphSchema] = None) -> GraphIR:
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

    (
        sender_to_idx,
        receiver_to_idx,
        week_to_idx,
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
        url_to_idx,
        domain_to_idx,
        stem_to_idx,
        email_domain_to_idx,
    )

    snd_dom_src, snd_dom_dst, rcv_dom_src, rcv_dom_dst = _connect_email_entities_to_domains(
        sender_to_idx, receiver_to_idx, email_domain_to_idx
    )

    (
        node_x,
        node_meta,
        node_attrs,
        subj_vecs,
        body_vecs,
        subj_dim,
        body_dim,
    ) = _compute_node_attributes_and_features(
        sender_to_idx,
        receiver_to_idx,
        week_to_idx,
        url_to_idx,
        domain_to_idx,
        stem_to_idx,
        email_domain_to_idx,
        url_components,
        docfreq_maps,
        emails,
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
    )


    nodes = _assemble_nodes(
        node_x,
        node_meta,
        node_attrs,
        {
            "sender": sender_to_idx,
            "receiver": receiver_to_idx,
            "week": week_to_idx,
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
