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
from typing import Any, Dict, List, Optional, Tuple

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
    # URL and Domain x will be set after we compute their normalized features below
    url_x: List[List[float]] = []
    domain_x: List[List[float]] = []
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
    # Text vectorization settings (subject/body TF-IDF reduced dims)
    TEXT_SUBJ_MAX_FEATS = 128
    TEXT_BODY_MAX_FEATS = 256
    # For backward-compat, we'll also expose a combined x_text attribute
    email_x_text: List[List[float]] = []

    # Pre-allocate structures for per-node statistics
    domain_email_sets: Dict[str, set] = {}

    stem_email_sets: Dict[str, set] = {}

    email_domain_sender_sets: Dict[str, set] = {}
    email_domain_receiver_sets: Dict[str, set] = {}
    url_email_sets: Dict[str, set] = {}
    sender_email_sets: Dict[str, set] = {}
    receiver_email_sets: Dict[str, set] = {}

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
                # Track domain per-email and ts stats
                domain_email_sets.setdefault(d, set()).add(email_idx)
                # Track stem stats if present
                s = comp.get("stem", "")
                if s:
                    stem_email_sets.setdefault(s, set()).add(email_idx)
        n_urls_val = int(len(domains))
        email_n_urls.append(n_urls_val)
        len_subject_val = int(len(subj))
        len_body_val = int(len(body))
        email_len_subject.append(len_subject_val)
        email_len_body.append(len_body_val)
        # We'll compute text vectors for subject and body after the initial pass

        # email_x will be filled after normalization below

        if em.get("sender") and em["sender"] in sender_to_idx:
            has_sender_src.append(email_idx)
            has_sender_dst.append(sender_to_idx[em["sender"]])
            sender_email_sets.setdefault(em["sender"], set()).add(email_idx)
            s_dom = extract_email_domain(em["sender"])
            if s_dom and not is_freemail_domain(s_dom):
                email_domain_sender_sets.setdefault(s_dom, set()).add(email_idx)

        for r in em.get("receivers", []) or []:
            if r and r in receiver_to_idx:
                has_receiver_src.append(email_idx)
                has_receiver_dst.append(receiver_to_idx[r])
                receiver_email_sets.setdefault(r, set()).add(email_idx)
            r_dom = extract_email_domain(r)
            if r_dom and not is_freemail_domain(r_dom):
                email_domain_receiver_sets.setdefault(r_dom, set()).add(email_idx)

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
                url_email_sets.setdefault(u, set()).add(email_idx)

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

    # -------------------------
    # Build per-node attributes arrays aligned to meta order
    # and compute normalized features requested
    # -------------------------
    # Domain node attrs
    domain_x_lex: List[List[float]] = [compute_lexical_features(d) for d in domain_meta]
    # entropy is index 7 in x_lex
    domain_entropies: List[float] = [v[7] if len(v) > 7 else 0.0 for v in domain_x_lex]
    domain_entropy_z, domain_entropy_mean, domain_entropy_std = zscore_list(domain_entropies)
    # Use normalized entropy as the primary domain x feature (shape [N,1])
    domain_x = [[float(domain_entropy_z[i])] for i in range(len(domain_entropy_z))]
    domain_docfreq: List[int] = [len(domain_email_sets.get(d, set())) for d in domain_meta]

    # Stem node attrs
    stem_x_lex: List[List[float]] = [compute_lexical_features(s) for s in stem_meta]
    stem_docfreq: List[int] = [len(stem_email_sets.get(s, set())) for s in stem_meta]

    # Email domain attrs (shared node type for sender/receiver domain roles)
    email_domain_x_lex: List[List[float]] = [compute_lexical_features(d) for d in email_domain_meta]
    email_domain_docfreq_sender: List[int] = [len(email_domain_sender_sets.get(d, set())) for d in email_domain_meta]
    email_domain_docfreq_receiver: List[int] = [len(email_domain_receiver_sets.get(d, set())) for d in email_domain_meta]

    # URL node path_len z-score as primary x
    # Build path (stem) lengths aligned to url_meta
    url_path_lens: List[float] = []
    for u in url_meta:
        if u in url_components:
            _, stem = url_components[u]
            url_path_lens.append(float(len(stem or "/")))
        else:
            comp = parse_url_components(u)
            url_path_lens.append(float(len(comp.get("stem", "/"))))
    url_path_len_z, url_path_len_mean, url_path_len_std = zscore_list(url_path_lens)
    url_x = [[float(url_path_len_z[i])] for i in range(len(url_path_len_z))]

    # Email feature normalization
    # ts -> min-max [0,1], len_body -> z-score
    len_body_z, _len_body_mean, _len_body_std = zscore_list([float(v) for v in email_len_body])
    ts_minmax, _ts_min, _ts_max = minmax_list([float(v) for v in email_ts])


    # Subject node normalization/attrs
    subject_lens: List[int] = [int(len(s)) for s in subject_meta]
    subject_len_z, _s_mean, _s_std = zscore_list([float(v) for v in subject_lens])
    subject_x = [[float(subject_len_z[i])] for i in range(len(subject_len_z))]

    # --------------------------------------------------
    # Subject/Body text vectorization per email (TF-IDF)
    # We vectorize subject and body separately with reduced dimensions, then
    # concatenate into the email.x feature vector along with numeric scalars.
    # If scikit-learn is unavailable or corpus is empty, we skip text features.
    # --------------------------------------------------
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
                ngram_range=(1, 2),  # short texts benefit from bigrams
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
                ngram_range=(1, 1),  # cap size for longer bodies
                min_df=1,
                stop_words="english",
                strip_accents="unicode",
                lowercase=True,
            ).fit_transform(body_corpus)
            body_dim = int(body_vec.shape[1])
            if body_dim > 0:
                body_vecs = [row.toarray().astype("float32")[0].tolist() for row in body_vec]
    except Exception:
        # Leave text vectors empty if vectorization fails (pipeline remains robust)
        subj_vecs, body_vecs = [], []
        subj_dim = body_dim = 0

    # Build final email.x by concatenating numeric scalars and text vectors
    # Order: [len_body_raw, n_urls_raw, ts_minmax, len_body_z] + subj_vec + body_vec
    email_x = []
    n_emails_final = len(email_len_subject)
    for i in range(n_emails_final):
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

    # For compatibility, also expose a combined x_text (subject+body) if available
    if subj_dim > 0 or body_dim > 0:
        email_x_text = []
        for i in range(n_emails_final):
            comb: List[float] = []
            if subj_vecs:
                comb.extend(subj_vecs[i] if i < len(subj_vecs) else [0.0] * subj_dim)
            if body_vecs:
                comb.extend(body_vecs[i] if i < len(body_vecs) else [0.0] * body_dim)
            email_x_text.append(comb)

    nodes: Dict[str, NodeIR] = {
        "email": NodeIR(index={}, x=email_x, index_to_meta=email_meta),
        "sender": NodeIR(index=sender_to_idx, x=sender_x, index_to_string=sender_meta,
                          attrs={
                              "docfreq": [len(sender_email_sets.get(s, set())) for s in sender_meta],
                          }),
        "receiver": NodeIR(index=receiver_to_idx, x=receiver_x, index_to_string=receiver_meta,
                            attrs={
                                "docfreq": [len(receiver_email_sets.get(r, set())) for r in receiver_meta],
                            }),
        "week": NodeIR(index=week_to_idx, x=week_x, index_to_string=week_meta),
        "subject": NodeIR(index=subject_to_idx, x=subject_x, index_to_string=subject_meta,
                           attrs={
                               "len_subject": subject_lens,
                               "len_subject_z": [float(z) for z in subject_len_z],
                           }),
        "url": NodeIR(index=url_to_idx, x=url_x, index_to_string=url_meta,
                       attrs={
                           "docfreq": [len(url_email_sets.get(u, set())) for u in url_meta],
                       }),
        "domain": NodeIR(index=domain_to_idx, x=domain_x, index_to_string=domain_meta,
                          attrs={
                              "x_lex": domain_x_lex,
                              "docfreq": domain_docfreq,
                          }),
        "stem": NodeIR(index=stem_to_idx, x=stem_x, index_to_string=stem_meta,
                        attrs={
                            "x_lex": stem_x_lex,
                            "docfreq": stem_docfreq,
                        }),
        "email_domain": NodeIR(index=email_domain_to_idx, x=email_domain_x, index_to_string=email_domain_meta,
                                 attrs={
                                     "x_lex": email_domain_x_lex,
                                     "docfreq_sender": email_domain_docfreq_sender,
                                     "docfreq_receiver": email_domain_docfreq_receiver,
                                 }),
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
        "len_body": email_len_body,
        # Only attach x_text if we actually built a non-empty vector per email
        "x_text": email_x_text if email_x_text and len(email_x_text[0]) > 0 else [],
        # Include per-email dimension indicators for Memgraph convenience
        "x_text_subject_dim": [int(subj_dim)] * (len(email_meta) or 0),
        "x_text_body_dim": [int(body_dim)] * (len(email_meta) or 0),
        # normalized variants (email only)
        "len_body_z": len_body_z,
        "ts_minmax": ts_minmax,
    }
    return GraphIR(nodes=nodes, edges=edges, email_attrs=email_attrs)


__all__ = ["GraphIR", "NodeIR", "assemble_misp_graph_ir"]
