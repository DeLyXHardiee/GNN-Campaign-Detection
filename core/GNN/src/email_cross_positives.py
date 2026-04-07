"""
CPU-side precompute + batch mining for conservative cross-email positive pairs.

Used only by the standalone contrastive objective.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Set, Tuple

from torch_geometric.data import HeteroData

from .email_safe_negatives import (
    build_channel_memberships,
    build_raw_email_feature_norm,
)

_EMPTY_MEMBERS: Set[int] = set()


@dataclass
class EmailCrossPositivePrecompute:
    num_emails: int
    raw_x_norm: Any  # torch.Tensor on CPU
    channel_memberships: Dict[str, List[Set[int]]]

    def raw_cosine(self, gi: int, gj: int) -> float:
        return float((self.raw_x_norm[gi] * self.raw_x_norm[gj]).sum().item())


def _infer_num_email_nodes(data: HeteroData, primary_ntype: str) -> int:
    st = data[primary_ntype]
    if getattr(st, "num_nodes", None) is not None:
        return int(st.num_nodes)
    if st.x is not None:
        return int(st.x.size(0))
    raise ValueError(
        f"Cannot infer num_nodes for {primary_ntype!r}: missing num_nodes and x."
    )


def build_email_cross_positive_precompute(
    data: HeteroData,
    primary_ntype: str = "email",
) -> EmailCrossPositivePrecompute:
    n_email = _infer_num_email_nodes(data, primary_ntype)
    raw_norm = build_raw_email_feature_norm(data, primary_ntype)
    if int(raw_norm.size(0)) != n_email:
        raise ValueError(
            f"raw feature rows ({raw_norm.size(0)}) != inferred email num_nodes ({n_email})."
        )
    ch_m = build_channel_memberships(data, primary_ntype)
    return EmailCrossPositivePrecompute(
        num_emails=n_email,
        raw_x_norm=raw_norm,
        channel_memberships=ch_m,
    )


def _set_intersection_preview(a: Set[int], b: Set[int], max_items: int = 3) -> List[int]:
    inter = sorted(a & b)
    return [int(x) for x in inter[:max_items]]


def _rule_match(
    gi: int,
    gj: int,
    pre: EmailCrossPositivePrecompute,
    rules: Sequence[str],
    cross_positive_raw_cosine_min: float,
) -> Tuple[bool, str, Dict[str, Any]]:
    m = pre.channel_memberships

    def _members(channel: str, gidx: int) -> Set[int]:
        arr = m.get(channel)
        if arr is None:
            return _EMPTY_MEMBERS
        if not (0 <= int(gidx) < len(arr)):
            return _EMPTY_MEMBERS
        return arr[int(gidx)]

    sender_i = _members("sender", gi)
    sender_j = _members("sender", gj)
    email_dom_i = _members("email_domain", gi)
    email_dom_j = _members("email_domain", gj)
    url_i = _members("url", gi)
    url_j = _members("url", gj)
    dom_i = _members("domain", gi)
    dom_j = _members("domain", gj)
    stem_i = _members("stem", gi)
    stem_j = _members("stem", gj)
    raw_cos = pre.raw_cosine(gi, gj)

    for rule in rules:
        if rule == "same_url":
            shared = _set_intersection_preview(url_i, url_j)
            if shared:
                return True, rule, {"shared_url_ids": shared}
        elif rule == "same_sender_and_email_domain":
            shared_sender = _set_intersection_preview(sender_i, sender_j)
            shared_email_dom = _set_intersection_preview(email_dom_i, email_dom_j)
            if shared_sender and shared_email_dom:
                return True, rule, {
                    "shared_sender_ids": shared_sender,
                    "shared_email_domain_ids": shared_email_dom,
                }
        elif rule == "same_domain_and_stem":
            shared_domain = _set_intersection_preview(dom_i, dom_j)
            shared_stem = _set_intersection_preview(stem_i, stem_j)
            if shared_domain and shared_stem:
                return True, rule, {
                    "shared_domain_ids": shared_domain,
                    "shared_stem_ids": shared_stem,
                }
        elif rule == "same_stem_and_raw_cos":
            shared_stem = _set_intersection_preview(stem_i, stem_j)
            if shared_stem and raw_cos >= float(cross_positive_raw_cosine_min):
                return True, rule, {
                    "shared_stem_ids": shared_stem,
                    "raw_cos_min": float(cross_positive_raw_cosine_min),
                }
        elif rule == "same_exact_stem":
            shared_stem = _set_intersection_preview(stem_i, stem_j)
            if shared_stem:
                return True, rule, {"shared_stem_ids": shared_stem}
    return False, "", {}


def mine_cross_email_positives_per_anchor(
    anchor_global_ids: Sequence[int],
    pre: EmailCrossPositivePrecompute,
    *,
    positive_rules: Sequence[str],
    max_cross_positives_per_anchor: int,
    cross_positive_raw_cosine_min: float = 0.0,
) -> Tuple[List[List[int]], Dict[str, Any]]:
    """
    For each anchor batch row i, return batch indices j (!= i) that qualify as conservative
    cross-email positives according to enabled graph rules.
    """
    bid = [int(x) for x in anchor_global_ids]
    b = len(bid)
    k_max = max(int(max_cross_positives_per_anchor), 0)

    pos_lists: List[List[int]] = [[] for _ in range(b)]
    anchors_with_pos = 0
    total_selected = 0
    debug_pairs: List[Dict[str, Any]] = []
    selected_raw_sum = 0.0
    selected_raw_n = 0
    rule_counts: Dict[str, int] = {}

    for i in range(b):
        gi = bid[i]
        candidates: List[Tuple[float, int, str, Dict[str, Any]]] = []
        for j in range(b):
            if j == i:
                continue
            gj = bid[j]
            if not (0 <= gi < pre.num_emails and 0 <= gj < pre.num_emails):
                continue
            matched, rule_name, evidence = _rule_match(
                gi,
                gj,
                pre,
                positive_rules,
                cross_positive_raw_cosine_min,
            )
            if not matched:
                continue
            rc = pre.raw_cosine(gi, gj)
            candidates.append((rc, j, rule_name, evidence))

        # Conservative ranking: prefer stronger semantic support first.
        candidates.sort(key=lambda t: (-t[0], t[1]))
        chosen = candidates[:k_max] if k_max > 0 else []
        pos_lists[i] = [j for _, j, _, _ in chosen]

        if pos_lists[i]:
            anchors_with_pos += 1
        total_selected += len(pos_lists[i])

        for rc, j, rule_name, evidence in chosen:
            selected_raw_sum += float(rc)
            selected_raw_n += 1
            rule_counts[rule_name] = int(rule_counts.get(rule_name, 0)) + 1
            if len(debug_pairs) < 5:
                debug_pairs.append(
                    {
                        "anchor_batch_i": i,
                        "pos_batch_j": j,
                        "global_i": gi,
                        "global_j": bid[j],
                        "rule": rule_name,
                        "raw_cos": float(rc),
                        "evidence": evidence,
                    }
                )

    stats: Dict[str, Any] = {
        "n_anchors": b,
        "total_selected_cross_positive_slots": int(total_selected),
        "mean_cross_positives_per_anchor": float(total_selected) / max(b, 1),
        "frac_anchors_with_cross_positive": float(anchors_with_pos) / max(b, 1),
        "frac_anchors_without_cross_positive": float(b - anchors_with_pos) / max(b, 1),
        "mean_raw_cos_cross_positive": (
            float(selected_raw_sum) / float(selected_raw_n)
            if selected_raw_n > 0
            else float("nan")
        ),
        "cross_positive_rule_counts": rule_counts,
        "debug_cross_positive_pairs": debug_pairs,
    }
    return pos_lists, stats


def validate_cross_positive_pair(
    pre: EmailCrossPositivePrecompute,
    gi: int,
    gj: int,
    *,
    positive_rules: Sequence[str],
    cross_positive_raw_cosine_min: float,
) -> Tuple[bool, str, Dict[str, Any]]:
    return _rule_match(gi, gj, pre, positive_rules, cross_positive_raw_cosine_min)

