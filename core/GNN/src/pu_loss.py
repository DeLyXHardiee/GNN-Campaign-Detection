"""
Positive-unlabeled (PU) loss for pair supervision (nnPU-style conservative risk).

Reference formulation (logistic / BCE-with-logits):
  R_p^+  = E_P[ l(f(x), y=1) ]
  R_p^-  = E_P[ l(f(x), y=0) ]
  R_u^-  = E_U[ l(f(x), y=0) ]
  R_n    = max(0, R_u^- - pi_p * R_p^-)   (non-negative correction)
  L      = pi_p * R_p^+ + R_n

Unlabeled rows are not treated as negatives; only R_u^- enters via the corrected negative risk.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F

PAIR_LOSS_PLACEHOLDER_BCE = "placeholder_bce"
PAIR_LOSS_NNPU = "nnpu"
PAIR_LOSS_NNPU_WITH_RELIABLE_NEGATIVES = "nnpu_with_reliable_negatives"


def resolve_pair_loss_type(training_cfg: dict[str, Any]) -> str:
    """Default to nnPU when unset. Legacy: infer placeholder from ``pair_placeholder_loss_mode``."""
    raw = training_cfg.get("pair_loss_type")
    if raw is not None and str(raw).strip() != "":
        s = str(raw).lower().strip()
        if s in ("placeholder_bce", "placeholder", "bce"):
            return PAIR_LOSS_PLACEHOLDER_BCE
        if s in ("nnpu", "pu", "nnpu_binary"):
            return PAIR_LOSS_NNPU
        if s in ("nnpu_with_reliable_negatives", "nnpu_rn", "pu_rn"):
            return PAIR_LOSS_NNPU_WITH_RELIABLE_NEGATIVES
        raise ValueError(
            f"Unknown pair_loss_type: {raw!r}; use {PAIR_LOSS_PLACEHOLDER_BCE!r}, {PAIR_LOSS_NNPU!r}, "
            f"or {PAIR_LOSS_NNPU_WITH_RELIABLE_NEGATIVES!r}."
        )
    # Backward compatibility: explicit placeholder mode in config without pair_loss_type
    legacy = str(training_cfg.get("pair_placeholder_loss_mode", "") or "").lower()
    if "bce_pos_vs_unlabeled" in legacy:
        return PAIR_LOSS_PLACEHOLDER_BCE
    return PAIR_LOSS_NNPU


def _exclusive_pu_masks(is_positive: torch.Tensor, is_unlabeled: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Positive rows take precedence if both flags were ever true."""
    pos_m = is_positive.clone()
    unl_m = is_unlabeled & ~pos_m
    return pos_m, unl_m


def exclusive_pair_masks(
    is_positive: torch.Tensor, is_unlabeled: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Public alias for endpoint-aligned positive vs unlabeled rows."""
    return _exclusive_pu_masks(is_positive, is_unlabeled)


def nnpu_binary_loss(
    logits: torch.Tensor,
    is_positive: torch.Tensor,
    is_unlabeled: torch.Tensor,
    *,
    pi_p: float,
    non_negative: bool = True,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    nnPU risk on a minibatch (already endpoint-filtered logits/masks).

    Returns (loss, diagnostics) with detached scalar diagnostics for logging.
    """
    if logits.dim() != 1:
        logits = logits.view(-1)
    pos_m, unl_m = exclusive_pair_masks(is_positive, is_unlabeled)
    n_p = int(pos_m.sum().item())
    n_u = int(unl_m.sum().item())

    if n_p > 0:
        lp = logits[pos_m]
        r_p_pos = F.binary_cross_entropy_with_logits(lp, torch.ones_like(lp), reduction="mean")
        r_p_neg = F.binary_cross_entropy_with_logits(lp, torch.zeros_like(lp), reduction="mean")
    else:
        r_p_pos = logits.new_zeros(())
        r_p_neg = logits.new_zeros(())

    if n_u > 0:
        lu = logits[unl_m]
        r_u_neg = F.binary_cross_entropy_with_logits(lu, torch.zeros_like(lu), reduction="mean")
    else:
        r_u_neg = logits.new_zeros(())

    pi = float(pi_p)
    if not (0.0 < pi < 1.0):
        raise ValueError(f"pu_class_prior (pi_p) must be in (0, 1), got {pi_p}")

    neg_risk_raw = r_u_neg - pi * r_p_neg
    neg_risk = torch.relu(neg_risk_raw) if non_negative else neg_risk_raw
    loss = pi * r_p_pos + neg_risk

    with torch.no_grad():
        mean_pos = torch.sigmoid(logits[pos_m]).mean().item() if n_p > 0 else float("nan")
        mean_unl = torch.sigmoid(logits[unl_m]).mean().item() if n_u > 0 else float("nan")
        sep = (mean_pos - mean_unl) if n_p > 0 and n_u > 0 else float("nan")

    diag: dict[str, Any] = {
        "n_positive": n_p,
        "n_unlabeled": n_u,
        "r_p_pos": float(r_p_pos.detach().item()),
        "r_p_neg": float(r_p_neg.detach().item()),
        "r_u_neg": float(r_u_neg.detach().item()),
        "neg_risk_raw": float(neg_risk_raw.detach().item()),
        "neg_risk_after_nn": float(neg_risk.detach().item()),
        "total_nnpu_loss": float(loss.detach().item()),
        "pi_p": pi,
        "pu_non_negative": bool(non_negative),
        "mean_pos_prob": mean_pos,
        "mean_unl_prob": mean_unl,
        "score_separation": sep,
    }
    return loss, diag


def nnpu_with_reliable_negatives_loss(
    logits: torch.Tensor,
    is_positive: torch.Tensor,
    is_unlabeled: torch.Tensor,
    is_reliable_negative: torch.Tensor,
    *,
    pi_p: float,
    reliable_negative_loss_weight: float = 1.0,
    non_negative: bool = True,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    nnPU risk on P and U only (reliable negatives excluded from U), plus weighted BCE-to-0 on N.

    ``L = nnPU(P, U') + lambda_neg * E_N[BCE(logit, 0)]`` where ``U' = U & ~N`` (and exclusivity with P).
    """
    if logits.dim() != 1:
        logits = logits.view(-1)
    neg_m = is_reliable_negative & ~is_positive
    unl_for_pu = is_unlabeled & ~is_positive & ~is_reliable_negative
    loss_pu, diag_pu = nnpu_binary_loss(
        logits,
        is_positive,
        unl_for_pu,
        pi_p=pi_p,
        non_negative=non_negative,
    )
    n_n = int(neg_m.sum().item())
    if n_n > 0:
        ln = logits[neg_m]
        bce_n = F.binary_cross_entropy_with_logits(ln, torch.zeros_like(ln), reduction="mean")
    else:
        bce_n = logits.new_zeros(())

    w = float(reliable_negative_loss_weight)
    loss = loss_pu + w * bce_n

    diag: dict[str, Any] = {
        **diag_pu,
        "n_reliable_negative": n_n,
        "neg_supervised_bce": float(bce_n.detach().item()),
        "reliable_negative_loss_weight": w,
        "total_hybrid_loss": float(loss.detach().item()),
    }
    return loss, diag


def placeholder_bce_pos_vs_unlabeled_as_neg(
    logits: torch.Tensor,
    is_positive: torch.Tensor,
    is_unlabeled: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Legacy structural loss (mis-specified for PU). Kept for debugging only."""
    pos_m, unl_m = exclusive_pair_masks(is_positive, is_unlabeled)
    y = pos_m.float()
    loss = F.binary_cross_entropy_with_logits(logits, y, reduction="mean")
    with torch.no_grad():
        pred = (torch.sigmoid(logits) >= 0.5).float()
        acc = (pred == y).float().mean().item()
        n_p = int(pos_m.sum().item())
        n_u = int(unl_m.sum().item())
        mean_pos = torch.sigmoid(logits[pos_m]).mean().item() if n_p > 0 else float("nan")
        mean_unl = torch.sigmoid(logits[unl_m]).mean().item() if n_u > 0 else float("nan")
        sep = (mean_pos - mean_unl) if n_p > 0 and n_u > 0 else float("nan")
    diag: dict[str, Any] = {
        "placeholder_acc": acc,
        "n_positive": n_p,
        "n_unlabeled": n_u,
        "mean_pos_prob": mean_pos,
        "mean_unl_prob": mean_unl,
        "score_separation": sep,
    }
    return loss, diag


def compute_pair_loss(
    logits: torch.Tensor,
    is_positive: torch.Tensor,
    is_unlabeled: torch.Tensor,
    pair_loss_type: str,
    *,
    pi_p: float = 0.1,
    pu_non_negative: bool = True,
    is_reliable_negative: torch.Tensor | None = None,
    reliable_negative_loss_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if pair_loss_type == PAIR_LOSS_PLACEHOLDER_BCE:
        return placeholder_bce_pos_vs_unlabeled_as_neg(logits, is_positive, is_unlabeled)
    if pair_loss_type == PAIR_LOSS_NNPU:
        return nnpu_binary_loss(
            logits,
            is_positive,
            is_unlabeled,
            pi_p=pi_p,
            non_negative=pu_non_negative,
        )
    if pair_loss_type == PAIR_LOSS_NNPU_WITH_RELIABLE_NEGATIVES:
        if is_reliable_negative is None:
            raise ValueError("is_reliable_negative is required for nnpu_with_reliable_negatives")
        return nnpu_with_reliable_negatives_loss(
            logits,
            is_positive,
            is_unlabeled,
            is_reliable_negative,
            pi_p=pi_p,
            reliable_negative_loss_weight=reliable_negative_loss_weight,
            non_negative=pu_non_negative,
        )
    raise ValueError(f"Unknown pair_loss_type: {pair_loss_type!r}")


def _is_finite(x: Any) -> bool:
    if isinstance(x, float):
        return not math.isnan(x)
    return isinstance(x, (int, float))


def aggregate_epoch_pu_stats(batches: list[dict[str, Any]], pair_loss_type: str) -> dict[str, float]:
    """Mean over batches for loss-like terms; sums for counts; weighted means for probabilities."""
    if not batches:
        return {}
    out: dict[str, Any] = {}
    usable = [
        b
        for b in batches
        if int(b.get("n_positive", 0)) + int(b.get("n_unlabeled", 0)) + int(b.get("n_reliable_negative", 0)) > 0
    ]
    if not usable:
        return out

    if pair_loss_type == PAIR_LOSS_NNPU:
        for k in ("r_p_pos", "r_p_neg", "r_u_neg", "neg_risk_raw", "neg_risk_after_nn", "total_nnpu_loss"):
            vs = [float(b[k]) for b in usable if k in b]
            if vs:
                out[f"epoch_mean_{k}"] = float(sum(vs) / len(vs))
    if pair_loss_type == PAIR_LOSS_NNPU_WITH_RELIABLE_NEGATIVES:
        for k in ("r_p_pos", "r_p_neg", "r_u_neg", "neg_risk_raw", "neg_risk_after_nn", "total_nnpu_loss"):
            vs = [float(b[k]) for b in usable if k in b]
            if vs:
                out[f"epoch_mean_{k}"] = float(sum(vs) / len(vs))
        bce_ns = [float(b["neg_supervised_bce"]) for b in usable if "neg_supervised_bce" in b]
        if bce_ns:
            out["epoch_mean_neg_supervised_bce"] = float(sum(bce_ns) / len(bce_ns))
        totals = [float(b.get("total_hybrid_loss", b.get("total_nnpu_loss", float("nan")))) for b in usable]
        totals_f = [t for t in totals if t == t]
        if totals_f:
            out["epoch_mean_total_hybrid_loss"] = float(sum(totals_f) / len(totals_f))
    if pair_loss_type == PAIR_LOSS_PLACEHOLDER_BCE:
        accs = [float(b["placeholder_acc"]) for b in usable if "placeholder_acc" in b]
        if accs:
            out["epoch_placeholder_acc"] = float(sum(accs) / len(accs))

    out["epoch_sum_n_positive"] = float(sum(int(b.get("n_positive", 0)) for b in batches))
    out["epoch_sum_n_unlabeled"] = float(sum(int(b.get("n_unlabeled", 0)) for b in batches))
    out["epoch_sum_n_reliable_negative"] = float(sum(int(b.get("n_reliable_negative", 0)) for b in batches))

    w_pos = 0.0
    s_pos = 0.0
    w_unl = 0.0
    s_unl = 0.0
    for b in batches:
        np_ = int(b.get("n_positive", 0))
        nu_ = int(b.get("n_unlabeled", 0))
        mp = b.get("mean_pos_prob")
        if np_ > 0 and _is_finite(mp):
            s_pos += float(mp) * np_
            w_pos += np_
        mu = b.get("mean_unl_prob")
        if nu_ > 0 and _is_finite(mu):
            s_unl += float(mu) * nu_
            w_unl += nu_
    out["epoch_mean_pos_prob"] = float(s_pos / w_pos) if w_pos > 0 else float("nan")
    out["epoch_mean_unl_prob"] = float(s_unl / w_unl) if w_unl > 0 else float("nan")
    if _is_finite(out["epoch_mean_pos_prob"]) and _is_finite(out["epoch_mean_unl_prob"]):
        out["epoch_score_separation"] = float(out["epoch_mean_pos_prob"] - out["epoch_mean_unl_prob"])
    else:
        out["epoch_score_separation"] = float("nan")
    return out
