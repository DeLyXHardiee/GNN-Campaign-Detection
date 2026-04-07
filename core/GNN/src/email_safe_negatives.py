"""
CPU-side precompute for **safe** email–email negatives: disjoint infrastructure channels +
raw ``email.x`` cosine screen. Used by graph-native contrastive training (not VICReg).
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Set

from torch_geometric.data import HeteroData
import torch


@dataclass
class EmailSafeNegativePrecompute:
    num_emails: int
    raw_x_norm: torch.Tensor  # [N, F] float32 CPU, row L2-normalized
    channel_memberships: Dict[str, List[Set[int]]]

    def shares_selected_infrastructure(
        self,
        gi: int,
        gj: int,
        channels: Sequence[str],
    ) -> bool:
        for ch in channels:
            if ch not in self.channel_memberships:
                continue
            m = self.channel_memberships[ch]
            if gi < 0 or gj < 0 or gi >= len(m) or gj >= len(m):
                continue
            if m[gi] & m[gj]:
                return True
        return False

    def raw_cosine(self, gi: int, gj: int) -> float:
        return float((self.raw_x_norm[gi] * self.raw_x_norm[gj]).sum().item())

    def is_safe_negative_pair(
        self,
        gi: int,
        gj: int,
        *,
        channels: Sequence[str],
        raw_cosine_threshold: float,
    ) -> bool:
        if gi == gj:
            return False
        if not (0 <= gi < self.num_emails and 0 <= gj < self.num_emails):
            return False
        if self.shares_selected_infrastructure(gi, gj, channels):
            return False
        if self.raw_cosine(gi, gj) >= float(raw_cosine_threshold):
            return False
        return True


def _infer_num_email_nodes(data: HeteroData, primary_ntype: str) -> int:
    st = data[primary_ntype]
    if getattr(st, "num_nodes", None) is not None:
        return int(st.num_nodes)
    if st.x is not None:
        return int(st.x.size(0))
    raise ValueError(
        f"Cannot infer num_nodes for {primary_ntype!r}: missing num_nodes and x."
    )


def build_channel_memberships(
    data: HeteroData,
    primary_ntype: str = "email",
) -> Dict[str, List[Set[int]]]:
    """
    Per global email id, artifact id sets per channel. ``email_domain`` is **sender-only**:
    email → sender → from_domain → email_domain (receiver path excluded).
    """
    n_email = _infer_num_email_nodes(data, primary_ntype)
    out: Dict[str, List[Set[int]]] = defaultdict(lambda: [set() for _ in range(n_email)])

    for src_t, rel, dst_t in data.edge_types:
        ei = data[src_t, rel, dst_t].edge_index
        if ei is None or ei.numel() == 0:
            continue
        if src_t == primary_ntype and dst_t != primary_ntype:
            emails = ei[0].long().cpu().tolist()
            arts = ei[1].long().cpu().tolist()
            buckets = out[dst_t]
            for e, a in zip(emails, arts):
                if 0 <= int(e) < n_email:
                    buckets[int(e)].add(int(a))
        elif dst_t == primary_ntype and src_t != primary_ntype:
            arts = ei[0].long().cpu().tolist()
            emails = ei[1].long().cpu().tolist()
            buckets = out[src_t]
            for a, e in zip(arts, emails):
                if 0 <= int(e) < n_email:
                    buckets[int(e)].add(int(a))

    s_to_d: Dict[int, Set[int]] = defaultdict(set)
    if ("sender", "from_domain", "email_domain") in data.edge_types:
        ei_sd = data["sender", "from_domain", "email_domain"].edge_index
        if ei_sd is not None and ei_sd.numel() > 0:
            for s, d in zip(
                ei_sd[0].long().cpu().tolist(),
                ei_sd[1].long().cpu().tolist(),
            ):
                s_to_d[int(s)].add(int(d))

    if s_to_d:
        es_key = (primary_ntype, "has_sender", "sender")
        if es_key in data.edge_types:
            ei_es = data[es_key].edge_index
            if ei_es is not None and ei_es.numel() > 0:
                buckets_ed = out["email_domain"]
                for e, s in zip(
                    ei_es[0].long().cpu().tolist(),
                    ei_es[1].long().cpu().tolist(),
                ):
                    if not (0 <= int(e) < n_email):
                        continue
                    for dom in s_to_d.get(int(s), ()):
                        buckets_ed[int(e)].add(int(dom))

    return dict(out)


def build_raw_email_feature_norm(
    data: HeteroData,
    primary_ntype: str = "email",
) -> torch.Tensor:
    st = data[primary_ntype]
    if st.x is None:
        raise ValueError(
            f"{primary_ntype}.x is required for safe negative raw cosine filter."
        )
    x = st.x.float().cpu()
    nrm = x.norm(dim=1, keepdim=True).clamp(min=1e-12)
    return x / nrm


def build_email_safe_negative_precompute(
    data: HeteroData,
    primary_ntype: str = "email",
) -> EmailSafeNegativePrecompute:
    n_email = _infer_num_email_nodes(data, primary_ntype)
    raw_norm = build_raw_email_feature_norm(data, primary_ntype)
    if int(raw_norm.size(0)) != n_email:
        raise ValueError(
            f"raw feature rows ({raw_norm.size(0)}) != inferred email num_nodes ({n_email})."
        )
    ch_m = build_channel_memberships(data, primary_ntype)
    return EmailSafeNegativePrecompute(
        num_emails=n_email,
        raw_x_norm=raw_norm,
        channel_memberships=ch_m,
    )


def hard_safe_negatives_per_anchor(
    anchor_global_ids: Sequence[int],
    pre: EmailSafeNegativePrecompute,
    *,
    channels: Sequence[str],
    raw_cosine_threshold: float,
    max_negatives_per_anchor: int,
) -> tuple[list[list[int]], dict[str, Any]]:
    """
    For each batch position ``i``, return batch indices ``j`` (others in batch) that are **safe**
    negatives for anchor ``i``, taking the **hardest** safe negatives first: highest raw cosine
    among pairs that still pass the infrastructure and raw-threshold filters.

    Returns:
        neg_lists: length ``B`` list of lists of batch indices ``j``
        stats: diagnostics (reject counts, means, optional examples)
    """
    bid = [int(x) for x in anchor_global_ids]
    b = len(bid)
    thr = float(raw_cosine_threshold)
    k_max = int(max_negatives_per_anchor)

    reject_infra = 0
    reject_cosine = 0
    neg_lists: list[list[int]] = [[] for _ in range(b)]
    eligible_count_per_i: list[int] = []

    for i in range(b):
        gi = bid[i]
        candidates: list[tuple[float, int]] = []
        for j in range(b):
            if j == i:
                continue
            gj = bid[j]
            if not (0 <= gi < pre.num_emails and 0 <= gj < pre.num_emails):
                continue
            if pre.shares_selected_infrastructure(gi, gj, channels):
                reject_infra += 1
                continue
            rc = pre.raw_cosine(gi, gj)
            if rc >= thr:
                reject_cosine += 1
                continue
            candidates.append((rc, j))
        candidates.sort(key=lambda t: (-t[0], t[1]))
        chosen = [j for _, j in candidates[:k_max]]
        neg_lists[i] = chosen
        eligible_count_per_i.append(len(candidates))

    n_elig = sum(eligible_count_per_i)
    n_chosen = sum(len(x) for x in neg_lists)
    n_with_any = sum(1 for L in neg_lists if len(L) > 0)
    raw_sel_sum = 0.0
    raw_sel_n = 0
    for i in range(b):
        gi = bid[i]
        for j in neg_lists[i]:
            gj = bid[j]
            raw_sel_sum += pre.raw_cosine(gi, gj)
            raw_sel_n += 1
    mean_raw_selected = (
        float(raw_sel_sum) / float(raw_sel_n) if raw_sel_n > 0 else float("nan")
    )

    stats: dict[str, Any] = {
        "n_anchors": b,
        "reject_infra_ordered": reject_infra,
        "reject_cosine_ordered": reject_cosine,
        "total_eligible_safe_slots": int(n_elig),
        "total_selected_negative_slots": int(n_chosen),
        "mean_eligible_safe_per_anchor": float(n_elig) / max(b, 1),
        "mean_selected_hard_per_anchor": float(n_chosen) / max(b, 1),
        "frac_anchors_with_any_selected_neg": float(n_with_any) / max(b, 1),
        "frac_anchors_with_zero_selected_neg": float(b - n_with_any) / max(b, 1),
        "mean_raw_cos_selected": mean_raw_selected,
        "n_selected_pairs_batch": int(raw_sel_n),
    }

    ex: list[dict[str, Any]] = []
    for i in range(min(b, 3)):
        if not neg_lists[i]:
            continue
        j = neg_lists[i][0]
        gi, gj = bid[i], bid[j]
        ex.append(
            {
                "anchor_batch_i": i,
                "neg_batch_j": j,
                "global_i": gi,
                "global_j": gj,
                "raw_cos": pre.raw_cosine(gi, gj),
                "infra_disjoint": not pre.shares_selected_infrastructure(gi, gj, channels),
            }
        )
    stats["examples"] = ex

    dbg_pairs: list[dict[str, Any]] = []
    for i in range(b):
        if not neg_lists[i]:
            continue
        for j in neg_lists[i]:
            gi, gj = bid[i], bid[j]
            dbg_pairs.append(
                {
                    "anchor_batch_i": i,
                    "neg_batch_j": j,
                    "global_i": gi,
                    "global_j": gj,
                    "raw_cos": pre.raw_cosine(gi, gj),
                    "infra_disjoint": not pre.shares_selected_infrastructure(
                        gi, gj, channels
                    ),
                }
            )
            if len(dbg_pairs) >= 5:
                break
        if len(dbg_pairs) >= 5:
            break
    stats["debug_neg_pairs"] = dbg_pairs
    return neg_lists, stats
