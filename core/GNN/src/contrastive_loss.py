"""
Filtered InfoNCE variants for standalone contrastive training.
"""
from __future__ import annotations

import math
from typing import List, Tuple

import torch
import torch.nn.functional as F


def filtered_multi_positive_nt_xent_symmetric(
    z1: torch.Tensor,
    z2: torch.Tensor,
    pos_lists: List[List[int]],
    neg_lists: List[List[int]],
    temperature: float,
) -> Tuple[torch.Tensor, dict]:
    """
    Symmetric filtered InfoNCE with multiple positives:
    - Direction v1->v2 for anchor i uses positives z2[pos_lists[i]]
      and negatives z2[neg_lists[i]].
    - Direction v2->v1 mirrors the same index lists.

    `pos_lists[i]` should include i (same-email across views), plus optional
    cross-email positives in the batch. `neg_lists[i]` should contain only safe
    negatives for anchor i.
    """
    if z1.shape != z2.shape or z1.dim() != 2:
        raise ValueError(f"Expected z1/z2 [B,D] same shape, got {z1.shape} vs {z2.shape}")
    b = z1.size(0)
    if len(pos_lists) != b or len(neg_lists) != b:
        raise ValueError(
            f"Expected pos/neg list length {b}, got {len(pos_lists)} / {len(neg_lists)}"
        )
    tau = float(temperature)
    if tau <= 0:
        raise ValueError("temperature must be > 0")

    def one_direction(anchor: torch.Tensor, other: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        total = anchor.new_zeros(())
        same_pos_sum = 0.0
        cross_pos_sum = 0.0
        cross_pos_n = 0
        neg_sum = 0.0
        neg_n = 0
        n_cross_anchor = 0
        used = 0

        for i in range(b):
            pos = [int(j) for j in pos_lists[i] if 0 <= int(j) < b]
            if not pos:
                pos = [i]
            if i not in pos:
                pos = [i] + pos
            # Preserve order while deduplicating.
            pos = list(dict.fromkeys(pos))
            pos_set = set(pos)

            neg = [int(j) for j in neg_lists[i] if 0 <= int(j) < b and int(j) not in pos_set]
            a = anchor[i]
            same_pos_cos = float((a * other[i]).sum().detach().item())
            same_pos_sum += same_pos_cos

            cross_js = [j for j in pos if j != i]
            if cross_js:
                n_cross_anchor += 1
                cross_cos = (other[torch.tensor(cross_js, device=anchor.device)] * a.unsqueeze(0)).sum(
                    dim=-1
                )
                cross_pos_sum += float(cross_cos.sum().detach().item())
                cross_pos_n += int(cross_cos.numel())

            pos_idx = torch.tensor(pos, device=anchor.device, dtype=torch.long)
            pos_logits = (other[pos_idx] * a.unsqueeze(0)).sum(dim=-1) / tau

            if neg:
                neg_idx = torch.tensor(neg, device=anchor.device, dtype=torch.long)
                neg_cos = (other[neg_idx] * a.unsqueeze(0)).sum(dim=-1)
                neg_sum += float(neg_cos.sum().detach().item())
                neg_n += int(neg_cos.numel())
                neg_logits = neg_cos / tau
                all_logits = torch.cat([pos_logits, neg_logits], dim=0)
            else:
                all_logits = pos_logits

            total = total + (torch.logsumexp(all_logits, dim=0) - torch.logsumexp(pos_logits, dim=0))
            used += 1

        denom = float(max(used, 1))
        return total / denom, {
            "same_pos_mean_cos": same_pos_sum / max(b, 1),
            "cross_pos_mean_cos": (cross_pos_sum / cross_pos_n) if cross_pos_n > 0 else float("nan"),
            "neg_mean_cos": (neg_sum / neg_n) if neg_n > 0 else float("nan"),
            "mean_cross_positives_per_anchor": float(cross_pos_n) / max(b, 1),
            "frac_anchors_with_cross_positive": float(n_cross_anchor) / max(b, 1),
        }

    l12, m12 = one_direction(z1, z2)
    l21, m21 = one_direction(z2, z1)
    cross12 = float(m12["cross_pos_mean_cos"])
    cross21 = float(m21["cross_pos_mean_cos"])
    neg12 = float(m12["neg_mean_cos"])
    neg21 = float(m21["neg_mean_cos"])

    def _mean_ignore_nan(a: float, b: float) -> float:
        vals = [x for x in (a, b) if not math.isnan(float(x))]
        return float(sum(vals) / len(vals)) if vals else float("nan")

    out = {
        "same_pos_mean_cos": 0.5 * (m12["same_pos_mean_cos"] + m21["same_pos_mean_cos"]),
        "cross_pos_mean_cos": _mean_ignore_nan(cross12, cross21),
        "neg_mean_cos": _mean_ignore_nan(neg12, neg21),
        "mean_cross_positives_per_anchor": 0.5
        * (m12["mean_cross_positives_per_anchor"] + m21["mean_cross_positives_per_anchor"]),
        "frac_anchors_with_cross_positive": 0.5
        * (m12["frac_anchors_with_cross_positive"] + m21["frac_anchors_with_cross_positive"]),
        # Backward-compatible alias used by some logs.
        "pos_mean_cos": 0.5 * (m12["same_pos_mean_cos"] + m21["same_pos_mean_cos"]),
    }
    return 0.5 * (l12 + l21), out


def filtered_nt_xent_symmetric(
    z1: torch.Tensor,
    z2: torch.Tensor,
    neg_lists: List[List[int]],
    temperature: float,
) -> Tuple[torch.Tensor, dict]:
    """
    z1, z2: [B, D] L2-normalized anchor embeddings for view1 / view2.

    For each anchor i, positives are (z1[i], z2[i]). Negatives for direction view1→view2
    are z2[j] for j in neg_lists[i] (safe hard negatives). Symmetric loss averages
    view1→view2 and view2→view1.

    If neg_lists[i] is empty, falls back to (1 - cos(z1[i], z2[i])) so two views still align.

    Returns:
        loss, dict with pos_mean_cos, neg_mean_cos (where negatives exist)
    """
    if z1.shape != z2.shape or z1.dim() != 2:
        raise ValueError(f"Expected z1/z2 [B,D] same shape, got {z1.shape} vs {z2.shape}")
    b = z1.size(0)
    if len(neg_lists) != b:
        raise ValueError(f"neg_lists length {len(neg_lists)} != batch {b}")
    tau = float(temperature)
    if tau <= 0:
        raise ValueError("temperature must be > 0")

    def one_direction(
        anchor: torch.Tensor,
        positive_other: torch.Tensor,
    ) -> Tuple[torch.Tensor, float, float, int, int]:
        total = anchor.new_zeros(())
        pos_cos_acc = 0.0
        neg_cos_acc = 0.0
        n_neg_terms = 0
        n_pos_only = 0
        for i in range(b):
            negs = neg_lists[i]
            a = anchor[i]
            p = positive_other[i]
            pos_cos = (a * p).sum()
            pos_cos_acc += float(pos_cos.detach().item())
            if not negs:
                total = total + (1.0 - pos_cos)
                n_pos_only += 1
                continue
            idx = torch.tensor(negs, device=anchor.device, dtype=torch.long)
            nk = positive_other[idx]
            neg_cos = (nk * a.unsqueeze(0)).sum(dim=-1)
            neg_cos_acc += float(neg_cos.mean().detach().item())
            n_neg_terms += 1
            logits = torch.cat([(pos_cos / tau).unsqueeze(0), neg_cos / tau], dim=0).unsqueeze(0)
            target = torch.zeros(1, dtype=torch.long, device=anchor.device)
            total = total + F.cross_entropy(logits, target)
        denom = float(b)
        pos_mean = pos_cos_acc / max(b, 1)
        neg_mean = neg_cos_acc / max(n_neg_terms, 1) if n_neg_terms else float("nan")
        return total / denom, pos_mean, neg_mean, n_neg_terms, n_pos_only

    l12, p1, n1, nt1, po1 = one_direction(z1, z2)
    l21, p2, n2, nt2, po2 = one_direction(z2, z1)
    loss = 0.5 * (l12 + l21)
    return loss, {
        "pos_mean_cos": 0.5 * (p1 + p2),
        "neg_mean_cos": 0.5 * (n1 + n2) if nt1 or nt2 else float("nan"),
        "n_anchors_with_neg": nt1,
        "n_anchors_pos_only": po1 + po2,
    }
