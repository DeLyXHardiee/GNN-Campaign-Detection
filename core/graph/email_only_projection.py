"""
Project a heterogeneous GraphIR into a single node-type (email) HeteroData with one
(email, aggregated, email) edge type: each unordered email pair has at most one
logical link, with edge_attr weight combining all shared-infrastructure contributions.

Each time two emails co-occur on the same infrastructure node (or same email_domain
path), that occurrence adds the configured IR weight; multiple occurrences are
combined per pair using pair_weight_aggregation (sum, max, or mean).
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Mapping, Optional, Set, Tuple

from .graph_schema import GraphSchema

# Documentary: which IR edge keys are considered (output is always one relation name)
IR_EDGE_KEYS: frozenset[str] = frozenset(
    {
        "has_sender",
        "has_receiver",
        "has_url",
        "has_domain",
        "has_stem",
        "has_attachment",
        "has_origin_ip",
        "has_received_host",
        "has_return_path_email",
        "has_return_path_domain",
        "sender_from_domain",
        "receiver_from_domain",
    }
)

# Backward compat: previously used for multi-relation output names
DEFAULT_IR_TO_RELATION: Dict[str, str] = {
    "has_sender": "shares_sender",
    "has_receiver": "shares_receiver",
    "has_url": "shares_url",
    "has_domain": "shares_domain",
    "has_stem": "shares_stem",
    "has_attachment": "shares_attachment",
    "has_origin_ip": "shares_origin_ip",
    "has_received_host": "shares_received_host",
    "has_return_path_email": "shares_return_path_email",
    "has_return_path_domain": "shares_return_path_domain",
    "sender_from_domain": "shares_sender_domain",
    "receiver_from_domain": "shares_receiver_domain",
}


def _build_reverse_bipartite(
    src: List[int], dst: List[int]
) -> Dict[int, List[int]]:
    m: Dict[int, List[int]] = defaultdict(list)
    for a, b in zip(src, dst):
        m[int(b)].append(int(a))
    for k, vs in m.items():
        m[k] = sorted(set(vs))
    return dict(m)


def _emails_sharing_bipartite_edges(
    email_src: List[int],
    infra_dst: List[int],
    *,
    min_emails: int,
) -> List[Tuple[int, int, int]]:
    m = _build_reverse_bipartite(email_src, infra_dst)
    out: List[Tuple[int, int, int]] = []
    for k, ems in m.items():
        if len(ems) < min_emails:
            continue
        for ii in range(len(ems)):
            for jj in range(ii + 1, len(ems)):
                a, b = ems[ii], ems[jj]
                if a > b:
                    a, b = b, a
                out.append((a, b, k))
    return out


def _emails_sharing_email_domain(
    has_entity: Tuple[List[int], List[int]],
    entity_to_domain: Tuple[List[int], List[int]],
    *,
    min_emails: int,
) -> List[Tuple[int, int, int]]:
    e_m, e_ent = has_entity[0], has_entity[1]
    ent_s, d_d = entity_to_domain[0], entity_to_domain[1]
    entity_to_domain_map: Dict[int, int] = {}
    for e, d in zip(ent_s, d_d):
        entity_to_domain_map[int(e)] = int(d)
    domain_to_emails: Dict[int, Set[int]] = defaultdict(set)
    for em_i, ent_i in zip(e_m, e_ent):
        d = entity_to_domain_map.get(int(ent_i))
        if d is None:
            continue
        domain_to_emails[d].add(int(em_i))
    out: List[Tuple[int, int, int]] = []
    for d, ems in domain_to_emails.items():
        srt = sorted(ems)
        if len(srt) < min_emails:
            continue
        for ii in range(len(srt)):
            for jj in range(ii + 1, len(srt)):
                out.append((srt[ii], srt[jj], d))
    return out


def _to_bidirectional_weighted(
    ordered_pairs: List[Tuple[int, int, float]],
    torch_lib: Any,
) -> Tuple[Any, Any, int]:
    """
    (a, b) with a < b and weight w -> two directed edges (a,b), (b,a) each with edge_attr w.
    """
    if not ordered_pairs:
        t = torch_lib
        return (
            t.empty((2, 0), dtype=t.long),
            t.empty((0, 1), dtype=t.float),
            0,
        )
    rows0: List[int] = []
    rows1: List[int] = []
    wlist: List[float] = []
    for a, b, w in ordered_pairs:
        rows0.extend([a, b])
        rows1.extend([b, a])
        wf = float(w)
        wlist.extend([wf, wf])
    ei = torch_lib.tensor([rows0, rows1], dtype=torch_lib.long)
    ea = torch_lib.tensor(wlist, dtype=torch_lib.float).view(-1, 1)
    return ei, ea, ei.size(1)


def _aggregate_contributions(
    contribs: List[float],
    mode: str,
) -> float:
    if not contribs:
        return 0.0
    if mode == "sum":
        return float(sum(contribs))
    if mode == "max":
        return float(max(contribs))
    if mode == "mean":
        return float(sum(contribs)) / float(len(contribs))
    return float(sum(contribs))


def project_ir_to_email_only(
    ir: Any,
    _schema: GraphSchema,
    *,
    enabled_ir_edges: Optional[Set[str]] = None,
    relation_renames: Optional[Mapping[str, str]] = None,  # noqa: ARG001
    relation_weights: Optional[Mapping[str, float]] = None,
    min_emails_per_infra: int = 2,
    aggregated_edge_name: str = "aggregated",
    pair_weight_aggregation: str = "sum",
) -> Tuple[Any, Dict[str, Any]]:
    """
    Build a HeteroData with a single (email, ``aggregated_edge_name``, email) edge type
    and scalar edge_attr = aggregated per-pair weight.
    """
    from torch_geometric.data import HeteroData  # type: ignore

    _ = relation_renames  # legacy; single aggregated edge name comes from config
    torch_mod = __import__("torch", fromlist=["torch"])
    pwa = (pair_weight_aggregation or "sum").strip().lower()
    if pwa not in ("sum", "max", "mean"):
        pwa = "sum"
    rel_name = (aggregated_edge_name or "aggregated").strip() or "aggregated"
    w_ir: Dict[str, float] = {}
    for k in IR_EDGE_KEYS:
        if relation_weights and k in relation_weights:
            w_ir[k] = max(0.0, float(relation_weights[k]))
        else:
            w_ir[k] = 1.0
    if enabled_ir_edges is not None:
        to_use: Set[str] = {
            k
            for k in enabled_ir_edges
            if k in IR_EDGE_KEYS and ir.edges and (k in ir.edges)
        }
    else:
        to_use = {k for k in IR_EDGE_KEYS if ir.edges and (k in ir.edges)}

    empty_email_only_meta = {
        "ir_sources": sorted(to_use),
        "relation_weights": {k: w_ir.get(k, 1.0) for k in to_use},
        "min_emails_per_infra": min_emails_per_infra,
        "pair_weight_aggregation": pwa,
        "aggregated_edge_name": rel_name,
    }
    empty_meta: Dict[str, Any] = {
        "node_maps": {"email": {"index_to_meta": []}},
        "feature_shapes": {"email": [0, 0]},
        "edge_counts": {},
        "graph_mode": "email_only",
        "email_only": empty_email_only_meta,
    }

    if not ir.nodes or "email" not in ir.nodes or not ir.nodes["email"].x:
        H = HeteroData()
        H["email"].num_nodes = 0
        return H, empty_meta

    num_emails = len(ir.nodes["email"].x)
    # pair (min, max) -> list of per-occurrence weights
    pair_contrib: Dict[Tuple[int, int], List[float]] = defaultdict(list)

    bipartite_keys: Set[str] = {
        "has_sender",
        "has_receiver",
        "has_url",
        "has_domain",
        "has_stem",
        "has_attachment",
        "has_origin_ip",
        "has_received_host",
        "has_return_path_email",
        "has_return_path_domain",
    }
    for k in bipartite_keys:
        if k not in to_use:
            continue
        e = ir.edges.get(k) if ir.edges else None
        if not e:
            continue
        src, dst = e[0], e[1]
        wk = w_ir.get(k, 1.0)
        for a, b, _ in _emails_sharing_bipartite_edges(
            src, dst, min_emails=min_emails_per_infra
        ):
            pair_contrib[(a, b)].append(wk)

    if "sender_from_domain" in to_use and ir.edges:
        if "has_sender" in ir.edges and "sender_from_domain" in ir.edges:
            wk = w_ir.get("sender_from_domain", 1.0)
            for a, b, _ in _emails_sharing_email_domain(
                (ir.edges["has_sender"][0], ir.edges["has_sender"][1]),
                (ir.edges["sender_from_domain"][0], ir.edges["sender_from_domain"][1]),
                min_emails=min_emails_per_infra,
            ):
                pair_contrib[(a, b)].append(wk)
    if "receiver_from_domain" in to_use and ir.edges:
        if "has_receiver" in ir.edges and "receiver_from_domain" in ir.edges:
            wk = w_ir.get("receiver_from_domain", 1.0)
            for a, b, _ in _emails_sharing_email_domain(
                (ir.edges["has_receiver"][0], ir.edges["has_receiver"][1]),
                (ir.edges["receiver_from_domain"][0], ir.edges["receiver_from_domain"][1]),
                min_emails=min_emails_per_infra,
            ):
                pair_contrib[(a, b)].append(wk)

    ordered: List[Tuple[int, int, float]] = []
    for (a, b), contribs in pair_contrib.items():
        wab = _aggregate_contributions(contribs, pwa)
        if wab > 0.0:
            ordered.append((a, b, wab))

    data = HeteroData()
    data["email"].num_nodes = num_emails
    email_meta = (ir.nodes.get("email") and ir.nodes["email"].index_to_meta) or []
    ei, ea, ecount = _to_bidirectional_weighted(ordered, torch_mod)
    data["email", rel_name, "email"].edge_index = ei
    if ecount > 0:
        data["email", rel_name, "email"].edge_attr = ea

    edge_counts = {f"email->email:{rel_name}": ecount}
    meta: Dict[str, Any] = {
        "node_maps": {"email": {"index_to_meta": email_meta}},
        "feature_shapes": {},
        "edge_counts": edge_counts,
        "graph_mode": "email_only",
        "email_only": {
            "ir_sources": sorted(to_use),
            "relation_weights": {k: w_ir.get(k, 1.0) for k in to_use},
            "min_emails_per_infra": min_emails_per_infra,
            "pair_weight_aggregation": pwa,
            "aggregated_edge_name": rel_name,
        },
    }
    if getattr(ir, "email_attrs", None) is not None:
        meta["email_attrs"] = ir.email_attrs
    return data, meta


__all__ = [
    "DEFAULT_IR_TO_RELATION",
    "IR_EDGE_KEYS",
    "project_ir_to_email_only",
]
