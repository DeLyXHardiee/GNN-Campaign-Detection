"""
Graph-native contrastive / metric learning for email-centric heterogeneous graphs.
Two augmented views, HeteroSAGE + projector, multi-positive filtered InfoNCE with
cross-email campaign-aware positives + hard safe negatives.
"""
from __future__ import annotations

import csv
import json
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm

from .contrastive_loss import filtered_multi_positive_nt_xent_symmetric
from .email_cross_positives import (
    EmailCrossPositivePrecompute,
    build_email_cross_positive_precompute,
    mine_cross_email_positives_per_anchor,
    validate_cross_positive_pair,
)
from .email_safe_negatives import (
    EmailSafeNegativePrecompute,
    build_email_safe_negative_precompute,
    hard_safe_negatives_per_anchor,
)
from .model import HeteroSAGE
from .model_io import save_contrastive_checkpoint
from .vicreg_modules import (
    VicRegProjector,
    augment_hetero_batch,
    extract_anchor_email_embeddings,
    extract_anchor_global_email_ids,
    make_email_anchor_loaders,
    split_email_node_indices,
)

CONTRASTIVE_DEBUG = os.getenv("CONTRASTIVE_DEBUG", "0") == "1"


def _assert_same_anchor_globals_view12(
    v1_store: Any,
    v2_store: Any,
    email_loader_input_nodes: torch.Tensor,
    *,
    context: str,
) -> torch.Tensor:
    g1 = extract_anchor_global_email_ids(
        v1_store,
        input_id=getattr(v1_store, "input_id", None),
        email_loader_input_nodes=email_loader_input_nodes,
    )
    g2 = extract_anchor_global_email_ids(
        v2_store,
        input_id=getattr(v2_store, "input_id", None),
        email_loader_input_nodes=email_loader_input_nodes,
    )
    if not torch.equal(g1, g2):
        n = min(32, g1.numel())
        raise RuntimeError(
            f"CONTRASTIVE SANITY CHECK FAILED ({context}): anchor global ids differ between views.\n"
            f"  view1[:{n}]={g1[:n].tolist()}\n  view2[:{n}]={g2[:n].tolist()}"
        )
    return g1


def _assert_selected_negatives_valid(
    pre: EmailSafeNegativePrecompute,
    bid: List[int],
    neg_lists: List[List[int]],
    channels: List[str],
    raw_cosine_threshold: float,
    *,
    context: str,
) -> None:
    thr = float(raw_cosine_threshold)
    for i, js in enumerate(neg_lists):
        gi = bid[i]
        for j in js:
            gj = bid[j]
            if pre.shares_selected_infrastructure(gi, gj, channels):
                shared = [c for c in channels if c in pre.channel_memberships and (pre.channel_memberships[c][gi] & pre.channel_memberships[c][gj])]
                raise RuntimeError(
                    f"CONTRASTIVE SANITY CHECK FAILED ({context}): selected pair "
                    f"(batch {i},{j}) globals ({gi},{gj}) overlaps infrastructure on channels={shared}."
                )
            rc = pre.raw_cosine(gi, gj)
            if rc >= thr:
                raise RuntimeError(
                    f"CONTRASTIVE SANITY CHECK FAILED ({context}): selected pair "
                    f"(batch {i},{j}) globals ({gi},{gj}) raw_cos={rc:.6f} >= threshold={thr}."
                )


def _assert_cross_positives_valid(
    pre: EmailCrossPositivePrecompute,
    bid: List[int],
    cross_pos_lists: List[List[int]],
    positive_rules: List[str],
    cross_positive_raw_cosine_min: float,
    *,
    context: str,
) -> None:
    for i, js in enumerate(cross_pos_lists):
        gi = bid[i]
        for j in js:
            if j == i:
                raise RuntimeError(
                    f"CONTRASTIVE SANITY CHECK FAILED ({context}): cross-positive "
                    f"(batch {i},{j}) points to itself."
                )
            gj = bid[j]
            matched, rule, _evidence = validate_cross_positive_pair(
                pre,
                gi,
                gj,
                positive_rules=positive_rules,
                cross_positive_raw_cosine_min=float(cross_positive_raw_cosine_min),
            )
            if not matched:
                raise RuntimeError(
                    f"CONTRASTIVE SANITY CHECK FAILED ({context}): selected cross-positive "
                    f"(batch {i},{j}) globals ({gi},{gj}) does not satisfy any enabled positive rule."
                )
            if (
                rule == "same_stem_and_raw_cos"
                and pre.raw_cosine(gi, gj) < float(cross_positive_raw_cosine_min)
            ):
                raise RuntimeError(
                    f"CONTRASTIVE SANITY CHECK FAILED ({context}): selected cross-positive "
                    f"(batch {i},{j}) globals ({gi},{gj}) raw_cos below minimum."
                )


def _fmt_float_or_nan(x: float, nd: int = 4) -> str:
    if isinstance(x, float) and math.isnan(x):
        return "nan"
    return f"{float(x):.{nd}f}"


def _csv_metric_cell(x: Any) -> Any:
    """Empty CSV cell for NaN epoch aggregates (no selected-neg batches)."""
    if isinstance(x, float) and math.isnan(x):
        return ""
    return x


def train_epoch_contrastive(
    device: torch.device,
    model: nn.Module,
    projector: nn.Module,
    optimizer: torch.optim.Optimizer,
    loader,
    *,
    email_loader_input_nodes: torch.Tensor,
    pos_pre: EmailCrossPositivePrecompute,
    positive_rules: List[str],
    max_cross_positives_per_anchor: int,
    cross_positive_raw_cosine_min: float,
    neg_pre: EmailSafeNegativePrecompute,
    negative_channels: List[str],
    raw_cosine_threshold: float,
    max_negatives_per_anchor: int,
    temperature: float,
    feat_mask_prob: float,
    edge_drop_probs: Optional[Dict[str, float]],
    email_semantic_mask_prob: float,
    email_semantic_mask_mode: str,
    email_semantic_block: Optional[Tuple[int, int]],
    primary_ntype: str,
    torch_seed: int,
    epoch: int,
    contrastive_debug: bool = False,
    contrastive_debug_num_batches: int = 3,
) -> Tuple[Dict[str, float], int]:
    model.train()
    projector.train()
    totals = {
        "total": 0.0,
        "same_pos_mean_cos": 0.0,
        "cross_pos_mean_cos": 0.0,
        "neg_mean_cos": 0.0,
        "mean_cross_positive_per_anchor": 0.0,
        "frac_anchors_with_cross_positive": 0.0,
        "mean_eligible_safe": 0.0,
        "mean_selected_hard": 0.0,
        "frac_anchors_with_neg": 0.0,
        "frac_anchors_without_neg": 0.0,
    }
    n_batches = 0
    skipped = 0
    raw_mean_sum_batches = 0.0
    raw_mean_n_batches = 0
    cross_raw_mean_sum_batches = 0.0
    cross_raw_mean_n_batches = 0

    pbar = tqdm(loader, desc="Contrastive train", leave=True)
    for batch_idx, batch in enumerate(pbar):
        try:
            if not hasattr(batch[primary_ntype], "batch_size"):
                raise AttributeError(f"batch[{primary_ntype!r}] missing batch_size")
            bs = int(batch[primary_ntype].batch_size)
        except Exception:
            skipped += 1
            continue
        if bs < 2:
            skipped += 1
            continue

        do_dbg = contrastive_debug and batch_idx < contrastive_debug_num_batches
        g1 = torch.Generator(device="cpu")
        g2 = torch.Generator(device="cpu")
        g1.manual_seed(torch_seed + epoch * 1_000_003 + batch_idx * 17 + 1)
        g2.manual_seed(torch_seed + epoch * 1_000_003 + batch_idx * 17 + 2)

        base_cpu = batch.clone()
        v1 = augment_hetero_batch(
            base_cpu.clone(),
            feat_mask_prob,
            edge_drop_probs,
            generator=g1,
            email_semantic_mask_prob=email_semantic_mask_prob,
            email_semantic_mask_mode=email_semantic_mask_mode,
            email_semantic_block=email_semantic_block,
            primary_ntype=primary_ntype,
        ).to(device)
        v2 = augment_hetero_batch(
            base_cpu.clone(),
            feat_mask_prob,
            edge_drop_probs,
            generator=g2,
            email_semantic_mask_prob=email_semantic_mask_prob,
            email_semantic_mask_mode=email_semantic_mask_mode,
            email_semantic_block=email_semantic_block,
            primary_ntype=primary_ntype,
        ).to(device)

        optimizer.zero_grad(set_to_none=True)
        h1 = model(v1.x_dict, v1.edge_index_dict)
        h2 = model(v2.x_dict, v2.edge_index_dict)

        try:
            a1 = extract_anchor_email_embeddings(
                h1[primary_ntype],
                v1[primary_ntype],
                input_id=getattr(v1[primary_ntype], "input_id", None),
                email_loader_input_nodes=email_loader_input_nodes,
            )
            a2 = extract_anchor_email_embeddings(
                h2[primary_ntype],
                v2[primary_ntype],
                input_id=getattr(v2[primary_ntype], "input_id", None),
                email_loader_input_nodes=email_loader_input_nodes,
            )
            if do_dbg:
                anchor_gid = _assert_same_anchor_globals_view12(
                    v1[primary_ntype],
                    v2[primary_ntype],
                    email_loader_input_nodes,
                    context=f"train epoch={epoch} batch={batch_idx}",
                )
            else:
                anchor_gid = extract_anchor_global_email_ids(
                    v1[primary_ntype],
                    input_id=getattr(v1[primary_ntype], "input_id", None),
                    email_loader_input_nodes=email_loader_input_nodes,
                )
        except ValueError:
            skipped += 1
            continue

        cross_pos_lists, pstats = mine_cross_email_positives_per_anchor(
            anchor_gid.tolist(),
            pos_pre,
            positive_rules=positive_rules,
            max_cross_positives_per_anchor=max_cross_positives_per_anchor,
            cross_positive_raw_cosine_min=cross_positive_raw_cosine_min,
        )
        pos_lists = [
            list(dict.fromkeys([i] + [int(j) for j in cross_pos_lists[i] if int(j) != i]))
            for i in range(len(cross_pos_lists))
        ]

        neg_lists, nstats = hard_safe_negatives_per_anchor(
            anchor_gid.tolist(),
            neg_pre,
            channels=negative_channels,
            raw_cosine_threshold=raw_cosine_threshold,
            max_negatives_per_anchor=max_negatives_per_anchor,
        )

        if do_dbg:
            _assert_cross_positives_valid(
                pos_pre,
                anchor_gid.tolist(),
                cross_pos_lists,
                positive_rules,
                cross_positive_raw_cosine_min,
                context=f"train epoch={epoch} batch={batch_idx}",
            )
            _assert_selected_negatives_valid(
                neg_pre,
                anchor_gid.tolist(),
                neg_lists,
                negative_channels,
                raw_cosine_threshold,
                context=f"train epoch={epoch} batch={batch_idx}",
            )

        z1 = F.normalize(projector(a1), dim=1, eps=1e-8)
        z2 = F.normalize(projector(a2), dim=1, eps=1e-8)
        loss, m = filtered_multi_positive_nt_xent_symmetric(
            z1, z2, pos_lists, neg_lists, temperature
        )
        loss.backward()
        optimizer.step()

        totals["total"] += float(loss.detach().item())
        totals["same_pos_mean_cos"] += float(m.get("same_pos_mean_cos", 0.0))
        mcross = m.get("cross_pos_mean_cos", float("nan"))
        totals["cross_pos_mean_cos"] += (
            0.0 if (isinstance(mcross, float) and math.isnan(mcross)) else float(mcross)
        )
        ng = m.get("neg_mean_cos", float("nan"))
        totals["neg_mean_cos"] += (
            0.0 if (isinstance(ng, float) and math.isnan(ng)) else float(ng)
        )
        totals["mean_cross_positive_per_anchor"] += float(
            pstats.get("mean_cross_positives_per_anchor", 0.0)
        )
        totals["frac_anchors_with_cross_positive"] += float(
            pstats.get("frac_anchors_with_cross_positive", 0.0)
        )
        totals["mean_eligible_safe"] += float(nstats.get("mean_eligible_safe_per_anchor", 0.0))
        totals["mean_selected_hard"] += float(nstats.get("mean_selected_hard_per_anchor", 0.0))
        totals["frac_anchors_with_neg"] += float(
            nstats.get("frac_anchors_with_any_selected_neg", 0.0)
        )
        totals["frac_anchors_without_neg"] += float(
            nstats.get("frac_anchors_with_zero_selected_neg", 0.0)
        )
        mraw_b = nstats.get("mean_raw_cos_selected")
        if mraw_b is not None and isinstance(mraw_b, (int, float)) and not math.isnan(
            float(mraw_b)
        ):
            raw_mean_sum_batches += float(mraw_b)
            raw_mean_n_batches += 1
        pcross_raw_b = pstats.get("mean_raw_cos_cross_positive")
        if pcross_raw_b is not None and isinstance(pcross_raw_b, (int, float)) and not math.isnan(
            float(pcross_raw_b)
        ):
            cross_raw_mean_sum_batches += float(pcross_raw_b)
            cross_raw_mean_n_batches += 1

        if do_dbg:
            gids = anchor_gid.tolist()
            bsz = len(gids)
            print(f"=== CONTRASTIVE DEBUG train epoch={epoch} batch={batch_idx} bs={bsz} ===")
            print(f"  anchor_global_ids (first20): {gids[:20]}")
            print(
                "  positives: same global id per row across views (asserted equal). "
                f"Sample pos pairs (batch_idx, global_id): "
                f"{[(i, gids[i]) for i in range(min(5, bsz))]}"
            )
            print(
                f"  cross-positives: mean/anchor={pstats.get('mean_cross_positives_per_anchor'):.3f} "
                f"frac_with_cross_pos={pstats.get('frac_anchors_with_cross_positive'):.3f} "
                f"total_selected={pstats.get('total_selected_cross_positive_slots')} "
                f"mean_raw_cos={_fmt_float_or_nan(float(pstats.get('mean_raw_cos_cross_positive', float('nan'))))}"
            )
            print(
                f"  cross-positive rule counts: {pstats.get('cross_positive_rule_counts', {})}"
            )
            print("  sample cross-positive pairs (rule + evidence):")
            for ex in pstats.get("debug_cross_positive_pairs", []) or []:
                print(
                    f"    anchor_g={ex.get('global_i')} pos_g={ex.get('global_j')} "
                    f"rule={ex.get('rule')} raw_cos={float(ex.get('raw_cos')):.4f} "
                    f"evidence={ex.get('evidence')}"
                )
            print(
                f"  mining: mean_eligible_safe/anchor={nstats.get('mean_eligible_safe_per_anchor'):.3f} "
                f"mean_selected_hard/anchor={nstats.get('mean_selected_hard_per_anchor'):.3f}"
            )
            print(
                f"  batch slot totals: eligible_safe={nstats.get('total_eligible_safe_slots')} "
                f"selected_hard={nstats.get('total_selected_negative_slots')} "
                f"(n_selected_pairs={nstats.get('n_selected_pairs_batch')})"
            )
            print(
                f"  frac_anchors_with_any_neg={nstats.get('frac_anchors_with_any_selected_neg'):.3f} "
                f"frac_with_zero_neg={nstats.get('frac_anchors_with_zero_selected_neg'):.3f} "
                f"mean_raw_cos(selected pairs, this batch)={_fmt_float_or_nan(float(mraw_b)) if (mraw_b is not None and isinstance(mraw_b, (int, float)) and not math.isnan(float(mraw_b))) else 'nan'}"
            )
            print(
                f"  rejects (ordered i,j pairs): infra={nstats.get('reject_infra_ordered')} "
                f"raw_cos>={raw_cosine_threshold}: {nstats.get('reject_cosine_ordered')}"
            )
            print("  sample selected negatives (verify infra_disjoint + raw_cos<thresh):")
            for ex in nstats.get("debug_neg_pairs", []) or []:
                print(
                    f"    anchor_g={ex.get('global_i')} neg_g={ex.get('global_j')} "
                    f"raw_cos={ex.get('raw_cos'):.4f} infra_disjoint={ex.get('infra_disjoint')}"
                )
            print(
                f"  loss batch: same_pos_cos(proj)={m.get('same_pos_mean_cos'):.4f} "
                f"cross_pos_cos(proj)={_fmt_float_or_nan(float(m.get('cross_pos_mean_cos', float('nan'))))} "
                f"neg_cos(proj mean)={_fmt_float_or_nan(float(m.get('neg_mean_cos', float('nan'))))} "
                f"loss={float(loss.detach().item()):.4f}"
            )

        n_batches += 1

    if n_batches == 0:
        return {k: 0.0 for k in totals}, skipped
    out = {k: totals[k] / n_batches for k in totals}
    out["pos_mean_cos"] = out["same_pos_mean_cos"]
    out["mean_selected_neg_raw_cos"] = (
        raw_mean_sum_batches / raw_mean_n_batches
        if raw_mean_n_batches > 0
        else float("nan")
    )
    out["mean_cross_pos_raw_cos"] = (
        cross_raw_mean_sum_batches / cross_raw_mean_n_batches
        if cross_raw_mean_n_batches > 0
        else float("nan")
    )
    return out, skipped


@torch.no_grad()
def eval_epoch_contrastive(
    device: torch.device,
    model: nn.Module,
    projector: nn.Module,
    loader,
    *,
    email_loader_input_nodes: torch.Tensor,
    pos_pre: EmailCrossPositivePrecompute,
    positive_rules: List[str],
    max_cross_positives_per_anchor: int,
    cross_positive_raw_cosine_min: float,
    neg_pre: EmailSafeNegativePrecompute,
    negative_channels: List[str],
    raw_cosine_threshold: float,
    max_negatives_per_anchor: int,
    temperature: float,
    feat_mask_prob: float,
    edge_drop_probs: Optional[Dict[str, float]],
    email_semantic_mask_prob: float,
    email_semantic_mask_mode: str,
    email_semantic_block: Optional[Tuple[int, int]],
    primary_ntype: str,
    torch_seed: int,
    epoch: int,
    split_name: str = "val",
    contrastive_debug: bool = False,
    contrastive_debug_num_batches: int = 3,
) -> Tuple[Dict[str, float], int]:
    model.eval()
    projector.eval()
    totals = {
        "total": 0.0,
        "same_pos_mean_cos": 0.0,
        "cross_pos_mean_cos": 0.0,
        "neg_mean_cos": 0.0,
        "mean_cross_positive_per_anchor": 0.0,
        "frac_anchors_with_cross_positive": 0.0,
        "mean_eligible_safe": 0.0,
        "mean_selected_hard": 0.0,
        "frac_anchors_with_neg": 0.0,
        "frac_anchors_without_neg": 0.0,
    }
    n_batches = 0
    skipped = 0
    raw_mean_sum_batches = 0.0
    raw_mean_n_batches = 0
    cross_raw_mean_sum_batches = 0.0
    cross_raw_mean_n_batches = 0

    pbar = tqdm(loader, desc=f"Contrastive {split_name}", leave=True)
    for batch_idx, batch in enumerate(pbar):
        try:
            bs = int(batch[primary_ntype].batch_size)
        except Exception:
            skipped += 1
            continue
        if bs < 2:
            skipped += 1
            continue

        do_dbg = contrastive_debug and batch_idx < contrastive_debug_num_batches
        g1 = torch.Generator(device="cpu")
        g2 = torch.Generator(device="cpu")
        g1.manual_seed(torch_seed + 9_000_001 + epoch + batch_idx + 1)
        g2.manual_seed(torch_seed + 9_000_001 + epoch + batch_idx + 2)

        base_cpu = batch.clone()
        v1 = augment_hetero_batch(
            base_cpu.clone(),
            feat_mask_prob,
            edge_drop_probs,
            generator=g1,
            email_semantic_mask_prob=email_semantic_mask_prob,
            email_semantic_mask_mode=email_semantic_mask_mode,
            email_semantic_block=email_semantic_block,
            primary_ntype=primary_ntype,
        ).to(device)
        v2 = augment_hetero_batch(
            base_cpu.clone(),
            feat_mask_prob,
            edge_drop_probs,
            generator=g2,
            email_semantic_mask_prob=email_semantic_mask_prob,
            email_semantic_mask_mode=email_semantic_mask_mode,
            email_semantic_block=email_semantic_block,
            primary_ntype=primary_ntype,
        ).to(device)

        h1 = model(v1.x_dict, v1.edge_index_dict)
        h2 = model(v2.x_dict, v2.edge_index_dict)
        try:
            a1 = extract_anchor_email_embeddings(
                h1[primary_ntype],
                v1[primary_ntype],
                input_id=getattr(v1[primary_ntype], "input_id", None),
                email_loader_input_nodes=email_loader_input_nodes,
            )
            a2 = extract_anchor_email_embeddings(
                h2[primary_ntype],
                v2[primary_ntype],
                input_id=getattr(v2[primary_ntype], "input_id", None),
                email_loader_input_nodes=email_loader_input_nodes,
            )
            if do_dbg:
                anchor_gid = _assert_same_anchor_globals_view12(
                    v1[primary_ntype],
                    v2[primary_ntype],
                    email_loader_input_nodes,
                    context=f"{split_name} epoch={epoch} batch={batch_idx}",
                )
            else:
                anchor_gid = extract_anchor_global_email_ids(
                    v1[primary_ntype],
                    input_id=getattr(v1[primary_ntype], "input_id", None),
                    email_loader_input_nodes=email_loader_input_nodes,
                )
        except ValueError:
            skipped += 1
            continue

        cross_pos_lists, pstats = mine_cross_email_positives_per_anchor(
            anchor_gid.tolist(),
            pos_pre,
            positive_rules=positive_rules,
            max_cross_positives_per_anchor=max_cross_positives_per_anchor,
            cross_positive_raw_cosine_min=cross_positive_raw_cosine_min,
        )
        pos_lists = [
            list(dict.fromkeys([i] + [int(j) for j in cross_pos_lists[i] if int(j) != i]))
            for i in range(len(cross_pos_lists))
        ]

        neg_lists, nstats = hard_safe_negatives_per_anchor(
            anchor_gid.tolist(),
            neg_pre,
            channels=negative_channels,
            raw_cosine_threshold=raw_cosine_threshold,
            max_negatives_per_anchor=max_negatives_per_anchor,
        )

        if do_dbg:
            _assert_cross_positives_valid(
                pos_pre,
                anchor_gid.tolist(),
                cross_pos_lists,
                positive_rules,
                cross_positive_raw_cosine_min,
                context=f"{split_name} epoch={epoch} batch={batch_idx}",
            )
            _assert_selected_negatives_valid(
                neg_pre,
                anchor_gid.tolist(),
                neg_lists,
                negative_channels,
                raw_cosine_threshold,
                context=f"{split_name} epoch={epoch} batch={batch_idx}",
            )

        z1 = F.normalize(projector(a1), dim=1, eps=1e-8)
        z2 = F.normalize(projector(a2), dim=1, eps=1e-8)
        loss, m = filtered_multi_positive_nt_xent_symmetric(
            z1, z2, pos_lists, neg_lists, temperature
        )

        totals["total"] += float(loss.item())
        totals["same_pos_mean_cos"] += float(m.get("same_pos_mean_cos", 0.0))
        mcross = m.get("cross_pos_mean_cos", float("nan"))
        totals["cross_pos_mean_cos"] += (
            0.0 if (isinstance(mcross, float) and math.isnan(mcross)) else float(mcross)
        )
        ng = m.get("neg_mean_cos", float("nan"))
        totals["neg_mean_cos"] += (
            0.0 if (isinstance(ng, float) and math.isnan(ng)) else float(ng)
        )
        totals["mean_cross_positive_per_anchor"] += float(
            pstats.get("mean_cross_positives_per_anchor", 0.0)
        )
        totals["frac_anchors_with_cross_positive"] += float(
            pstats.get("frac_anchors_with_cross_positive", 0.0)
        )
        totals["mean_eligible_safe"] += float(nstats.get("mean_eligible_safe_per_anchor", 0.0))
        totals["mean_selected_hard"] += float(nstats.get("mean_selected_hard_per_anchor", 0.0))
        totals["frac_anchors_with_neg"] += float(
            nstats.get("frac_anchors_with_any_selected_neg", 0.0)
        )
        totals["frac_anchors_without_neg"] += float(
            nstats.get("frac_anchors_with_zero_selected_neg", 0.0)
        )
        mraw_b = nstats.get("mean_raw_cos_selected")
        if mraw_b is not None and isinstance(mraw_b, (int, float)) and not math.isnan(
            float(mraw_b)
        ):
            raw_mean_sum_batches += float(mraw_b)
            raw_mean_n_batches += 1
        pcross_raw_b = pstats.get("mean_raw_cos_cross_positive")
        if pcross_raw_b is not None and isinstance(pcross_raw_b, (int, float)) and not math.isnan(
            float(pcross_raw_b)
        ):
            cross_raw_mean_sum_batches += float(pcross_raw_b)
            cross_raw_mean_n_batches += 1

        if do_dbg:
            gids = anchor_gid.tolist()
            bsz = len(gids)
            print(f"=== CONTRASTIVE DEBUG {split_name} epoch={epoch} batch={batch_idx} bs={bsz} ===")
            print(f"  anchor_global_ids (first20): {gids[:20]}")
            print(
                "  positives: same global id per row across views (asserted equal). "
                f"Sample pos pairs (batch_idx, global_id): "
                f"{[(i, gids[i]) for i in range(min(5, bsz))]}"
            )
            print(
                f"  cross-positives: mean/anchor={pstats.get('mean_cross_positives_per_anchor'):.3f} "
                f"frac_with_cross_pos={pstats.get('frac_anchors_with_cross_positive'):.3f} "
                f"total_selected={pstats.get('total_selected_cross_positive_slots')} "
                f"mean_raw_cos={_fmt_float_or_nan(float(pstats.get('mean_raw_cos_cross_positive', float('nan'))))}"
            )
            print(
                f"  cross-positive rule counts: {pstats.get('cross_positive_rule_counts', {})}"
            )
            print("  sample cross-positive pairs (rule + evidence):")
            for ex in pstats.get("debug_cross_positive_pairs", []) or []:
                print(
                    f"    anchor_g={ex.get('global_i')} pos_g={ex.get('global_j')} "
                    f"rule={ex.get('rule')} raw_cos={float(ex.get('raw_cos')):.4f} "
                    f"evidence={ex.get('evidence')}"
                )
            print(
                f"  mining: mean_eligible_safe/anchor={nstats.get('mean_eligible_safe_per_anchor'):.3f} "
                f"mean_selected_hard/anchor={nstats.get('mean_selected_hard_per_anchor'):.3f}"
            )
            print(
                f"  batch slot totals: eligible_safe={nstats.get('total_eligible_safe_slots')} "
                f"selected_hard={nstats.get('total_selected_negative_slots')} "
                f"(n_selected_pairs={nstats.get('n_selected_pairs_batch')})"
            )
            print(
                f"  frac_anchors_with_any_neg={nstats.get('frac_anchors_with_any_selected_neg'):.3f} "
                f"frac_with_zero_neg={nstats.get('frac_anchors_with_zero_selected_neg'):.3f} "
                f"mean_raw_cos(selected pairs, this batch)={_fmt_float_or_nan(float(mraw_b)) if (mraw_b is not None and isinstance(mraw_b, (int, float)) and not math.isnan(float(mraw_b))) else 'nan'}"
            )
            print(
                f"  rejects (ordered i,j pairs): infra={nstats.get('reject_infra_ordered')} "
                f"raw_cos>={raw_cosine_threshold}: {nstats.get('reject_cosine_ordered')}"
            )
            print("  sample selected negatives (verify infra_disjoint + raw_cos<thresh):")
            for ex in nstats.get("debug_neg_pairs", []) or []:
                print(
                    f"    anchor_g={ex.get('global_i')} neg_g={ex.get('global_j')} "
                    f"raw_cos={ex.get('raw_cos'):.4f} infra_disjoint={ex.get('infra_disjoint')}"
                )
            print(
                f"  loss batch: same_pos_cos(proj)={m.get('same_pos_mean_cos'):.4f} "
                f"cross_pos_cos(proj)={_fmt_float_or_nan(float(m.get('cross_pos_mean_cos', float('nan'))))} "
                f"neg_cos(proj mean)={_fmt_float_or_nan(float(m.get('neg_mean_cos', float('nan'))))} "
                f"loss={float(loss.item()):.4f}"
            )
        n_batches += 1

    if n_batches == 0:
        return {k: 0.0 for k in totals}, skipped
    out = {k: totals[k] / n_batches for k in totals}
    out["pos_mean_cos"] = out["same_pos_mean_cos"]
    out["mean_selected_neg_raw_cos"] = (
        raw_mean_sum_batches / raw_mean_n_batches
        if raw_mean_n_batches > 0
        else float("nan")
    )
    out["mean_cross_pos_raw_cos"] = (
        cross_raw_mean_sum_batches / cross_raw_mean_n_batches
        if cross_raw_mean_n_batches > 0
        else float("nan")
    )
    return out, skipped


def run_contrastive_training(
    DEVICE: torch.device,
    TORCH_SEED: int,
    data,
    *,
    primary_ntype: str = "email",
    hidden: int = 128,
    out_dim: int = 128,
    layers: int = 2,
    dropout: float = 0.1,
    fanout=None,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    epochs: int = 5,
    lr: float = 1e-3,
    wd: float = 1e-4,
    anchor_batch_size: Optional[int] = None,
    batch_size: Optional[int] = None,
    model_save_name: str = "best_model.pt",
    early_stopping_patience: int = 5,
    lr_reduce_patience: int = 5,
    lr_reduce_factor: float = 0.5,
    lr_reduce_min: float = 0.0,
    run_dir=None,
    runs_parent=None,
    models_subdir: str = "models",
    metrics_csv: str = "metrics.csv",
    training_config_json: str = "training_config.json",
    feat_mask_prob: float = 0.08,
    edge_drop_probs: Optional[Dict[str, float]] = None,
    email_semantic_mask_prob: float = 0.05,
    email_semantic_mask_mode: str = "block_zero",
    email_semantic_block: Optional[Tuple[int, int]] = None,
    projector_hidden_dim: Optional[int] = None,
    projector_out_dim: Optional[int] = None,
    save_epoch_checkpoints: bool = False,
    contrastive_temperature: float = 0.07,
    contrastive_raw_cosine_threshold: float = 0.30,
    contrastive_max_negatives_per_anchor: int = 16,
    contrastive_use_negative_channels: Optional[List[str]] = None,
    contrastive_use_positive_rules: Optional[List[str]] = None,
    contrastive_max_cross_positives_per_anchor: int = 4,
    contrastive_cross_positive_raw_cosine_min: float = 0.20,
    contrastive_debug_anchor_matching: bool = False,
    contrastive_debug_num_batches: int = 3,
):
    if fanout is None:
        fanout = [15, 10]
    anchor_bs = int(anchor_batch_size if anchor_batch_size is not None else (batch_size or 256))
    ph = int(projector_hidden_dim if projector_hidden_dim is not None else out_dim)
    po = int(projector_out_dim if projector_out_dim is not None else out_dim)
    neg_ch_list: List[str] = list(
        contrastive_use_negative_channels
        if contrastive_use_negative_channels is not None
        else ["sender", "url", "domain", "stem", "email_domain"]
    )
    pos_rule_list: List[str] = list(
        contrastive_use_positive_rules
        if contrastive_use_positive_rules is not None
        else [
            "same_url",
            "same_sender_and_email_domain",
            "same_domain_and_stem",
        ]
    )
    if edge_drop_probs is None:
        edge_drop_probs = {"default": 0.05}

    if run_dir is None:
        parent = Path(runs_parent) if runs_parent is not None else Path("outputs")
        parent.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_dir = parent / f"run_{timestamp}"
    else:
        run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = run_dir / models_subdir
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    metrics_csv_path = os.path.join(run_dir, metrics_csv)
    with open(metrics_csv_path, mode="w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "epoch",
                "train_total",
                "val_total",
                "train_avg_cross_pos_per_anchor",
                "val_avg_cross_pos_per_anchor",
                "train_frac_anchors_with_cross_pos",
                "val_frac_anchors_with_cross_pos",
                "train_avg_eligible_negs_per_anchor",
                "val_avg_eligible_negs_per_anchor",
                "train_avg_selected_negs_per_anchor",
                "val_avg_selected_negs_per_anchor",
                "train_frac_anchors_with_neg",
                "val_frac_anchors_with_neg",
                "train_frac_anchors_without_neg",
                "val_frac_anchors_without_neg",
                "train_avg_cross_pos_raw_cos",
                "val_avg_cross_pos_raw_cos",
                "train_avg_selected_neg_raw_cos",
                "val_avg_selected_neg_raw_cos",
                "train_avg_same_pos_cos",
                "val_avg_same_pos_cos",
                "train_avg_cross_pos_cos",
                "val_avg_cross_pos_cos",
                "train_avg_neg_proj_cos",
                "val_avg_neg_proj_cos",
                "train_skipped_batches",
                "val_skipped_batches",
            ]
        )

    data_cpu = data.to("cpu")
    print("Contrastive training | metadata:", data_cpu.metadata())
    neg_pre = build_email_safe_negative_precompute(data_cpu, primary_ntype)
    pos_pre = build_email_cross_positive_precompute(data_cpu, primary_ntype)

    num_email = int(data_cpu[primary_ntype].num_nodes)
    train_idx, val_idx, test_idx = split_email_node_indices(
        num_email, TORCH_SEED, val_ratio, test_ratio
    )

    loaders = make_email_anchor_loaders(
        data_cpu,
        primary_ntype,
        train_idx,
        val_idx,
        test_idx,
        fanout,
        layers,
        anchor_bs,
        num_workers=0,
    )

    model = HeteroSAGE(
        metadata=data_cpu.metadata(), hidden=hidden, out=out_dim, layers=layers, dropout=dropout
    ).to(DEVICE)
    projector = VicRegProjector(in_dim=out_dim, hidden_dim=ph, out_dim=po).to(DEVICE)

    opt = torch.optim.AdamW(
        list(model.parameters()) + list(projector.parameters()), lr=lr, weight_decay=wd
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt,
        mode="min",
        factor=lr_reduce_factor,
        patience=lr_reduce_patience,
        min_lr=lr_reduce_min,
    )

    encoder_config = {
        "hidden": hidden,
        "out_dim": out_dim,
        "layers": layers,
        "dropout": dropout,
    }
    contrastive_hparams = {
        "contrastive_temperature": float(contrastive_temperature),
        "contrastive_raw_cosine_threshold": float(contrastive_raw_cosine_threshold),
        "contrastive_max_negatives_per_anchor": int(contrastive_max_negatives_per_anchor),
        "contrastive_use_negative_channels": neg_ch_list,
        "contrastive_use_positive_rules": pos_rule_list,
        "contrastive_max_cross_positives_per_anchor": int(
            contrastive_max_cross_positives_per_anchor
        ),
        "contrastive_cross_positive_raw_cosine_min": float(
            contrastive_cross_positive_raw_cosine_min
        ),
        "contrastive_feat_mask_prob": float(feat_mask_prob),
        "contrastive_edge_drop_probs": edge_drop_probs,
        "contrastive_email_semantic_mask_prob": float(email_semantic_mask_prob),
        "contrastive_email_semantic_mask_mode": str(email_semantic_mask_mode),
        "contrastive_email_semantic_block": (
            list(email_semantic_block) if email_semantic_block is not None else None
        ),
        "contrastive_projector_hidden_dim": ph,
        "contrastive_projector_out_dim": po,
    }
    anchor_loader_params = {
        "anchor_batch_size": anchor_bs,
        "fanout": list(fanout) if isinstance(fanout, (list, tuple)) else fanout,
        "primary_ntype": primary_ntype,
        "num_gnn_layers": layers,
    }

    training_record: Dict[str, Any] = {
        "training_objective": "contrastive",
        "torch_seed": TORCH_SEED,
        "primary_ntype": primary_ntype,
        "encoder": encoder_config,
        "fanout": fanout,
        "anchor_batch_size": anchor_bs,
        "email_split": {
            "val_ratio": val_ratio,
            "test_ratio": test_ratio,
            "num_email_nodes": num_email,
            "note": "train/val/test are disjoint global email node indices; loaders use full graph for neighborhoods.",
        },
        "epochs": epochs,
        "lr": lr,
        "wd": wd,
        "early_stopping_patience": early_stopping_patience,
        "lr_reduce_patience": lr_reduce_patience,
        "lr_reduce_factor": lr_reduce_factor,
        "lr_reduce_min": lr_reduce_min,
        "model_save_name": model_save_name,
        "save_epoch_checkpoints": bool(save_epoch_checkpoints),
        "contrastive": contrastive_hparams,
        "anchor_loader": anchor_loader_params,
    }
    with open(run_dir / training_config_json, "w", encoding="utf-8") as f:
        json.dump(training_record, f, indent=2)

    dbg = bool(contrastive_debug_anchor_matching or CONTRASTIVE_DEBUG)
    best_val = float("inf")
    patience_counter = 0
    best_state = None

    print(
        f"Starting contrastive training (tau={contrastive_temperature}, "
        f"raw_cos_thresh={contrastive_raw_cosine_threshold}, K_neg={contrastive_max_negatives_per_anchor}, "
        f"rules={pos_rule_list}, K_cross_pos={contrastive_max_cross_positives_per_anchor})"
    )
    for epoch in range(1, epochs + 1):
        tr, tr_skip = train_epoch_contrastive(
            DEVICE,
            model,
            projector,
            opt,
            loaders["train"],
            email_loader_input_nodes=train_idx,
            pos_pre=pos_pre,
            positive_rules=pos_rule_list,
            max_cross_positives_per_anchor=contrastive_max_cross_positives_per_anchor,
            cross_positive_raw_cosine_min=contrastive_cross_positive_raw_cosine_min,
            neg_pre=neg_pre,
            negative_channels=neg_ch_list,
            raw_cosine_threshold=contrastive_raw_cosine_threshold,
            max_negatives_per_anchor=contrastive_max_negatives_per_anchor,
            temperature=contrastive_temperature,
            feat_mask_prob=feat_mask_prob,
            edge_drop_probs=edge_drop_probs,
            email_semantic_mask_prob=email_semantic_mask_prob,
            email_semantic_mask_mode=email_semantic_mask_mode,
            email_semantic_block=email_semantic_block,
            primary_ntype=primary_ntype,
            torch_seed=TORCH_SEED,
            epoch=epoch,
            contrastive_debug=dbg,
            contrastive_debug_num_batches=contrastive_debug_num_batches,
        )
        va, va_skip = eval_epoch_contrastive(
            DEVICE,
            model,
            projector,
            loaders["val"],
            email_loader_input_nodes=val_idx,
            pos_pre=pos_pre,
            positive_rules=pos_rule_list,
            max_cross_positives_per_anchor=contrastive_max_cross_positives_per_anchor,
            cross_positive_raw_cosine_min=contrastive_cross_positive_raw_cosine_min,
            neg_pre=neg_pre,
            negative_channels=neg_ch_list,
            raw_cosine_threshold=contrastive_raw_cosine_threshold,
            max_negatives_per_anchor=contrastive_max_negatives_per_anchor,
            temperature=contrastive_temperature,
            feat_mask_prob=feat_mask_prob,
            edge_drop_probs=edge_drop_probs,
            email_semantic_mask_prob=email_semantic_mask_prob,
            email_semantic_mask_mode=email_semantic_mask_mode,
            email_semantic_block=email_semantic_block,
            primary_ntype=primary_ntype,
            torch_seed=TORCH_SEED,
            epoch=epoch,
            split_name="val",
            contrastive_debug=dbg,
            contrastive_debug_num_batches=contrastive_debug_num_batches,
        )
        va_loss = va["total"]
        print(
            f"Epoch {epoch:02d} | train loss {tr['total']:.4f} | val {va_loss:.4f} "
            f"[skip tr={tr_skip} val={va_skip}]"
        )
        print(
            f"         cross-pos (mean/batch): per_anchor tr {tr['mean_cross_positive_per_anchor']:.3f} "
            f"val {va['mean_cross_positive_per_anchor']:.3f} | "
            f"frac_anchor_with_cross_pos tr {tr['frac_anchors_with_cross_positive']:.3f} "
            f"val {va['frac_anchors_with_cross_positive']:.3f}"
        )
        print(
            f"         mining (mean/batch): elig/anchor train {tr['mean_eligible_safe']:.3f} val {va['mean_eligible_safe']:.3f} | "
            f"selected/anchor tr {tr['mean_selected_hard']:.3f} val {va['mean_selected_hard']:.3f}"
        )
        print(
            f"         anchors w/ neg: train {tr['frac_anchors_with_neg']:.3f} val {va['frac_anchors_with_neg']:.3f} | "
            f"w/o neg: tr {tr['frac_anchors_without_neg']:.3f} val {va['frac_anchors_without_neg']:.3f}"
        )
        print(
            f"         raw_cos(cross-pos) mean: tr {_fmt_float_or_nan(float(tr.get('mean_cross_pos_raw_cos', float('nan'))))} "
            f"val {_fmt_float_or_nan(float(va.get('mean_cross_pos_raw_cos', float('nan'))))}"
        )
        print(
            f"         raw_cos(selected) mean: tr {_fmt_float_or_nan(float(tr.get('mean_selected_neg_raw_cos', float('nan'))))} "
            f"val {_fmt_float_or_nan(float(va.get('mean_selected_neg_raw_cos', float('nan'))))} | "
            f"same_pos_cos(proj): tr {tr['same_pos_mean_cos']:.4f} val {va['same_pos_mean_cos']:.4f} | "
            f"cross_pos_cos(proj): tr {_fmt_float_or_nan(float(tr.get('cross_pos_mean_cos', float('nan'))))} "
            f"val {_fmt_float_or_nan(float(va.get('cross_pos_mean_cos', float('nan'))))} | "
            f"neg_cos(proj): tr {_fmt_float_or_nan(float(tr['neg_mean_cos']))} val {_fmt_float_or_nan(float(va['neg_mean_cos']))}"
        )

        with open(metrics_csv_path, mode="a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    epoch,
                    tr["total"],
                    va_loss,
                    tr["mean_cross_positive_per_anchor"],
                    va["mean_cross_positive_per_anchor"],
                    tr["frac_anchors_with_cross_positive"],
                    va["frac_anchors_with_cross_positive"],
                    tr["mean_eligible_safe"],
                    va["mean_eligible_safe"],
                    tr["mean_selected_hard"],
                    va["mean_selected_hard"],
                    tr["frac_anchors_with_neg"],
                    va["frac_anchors_with_neg"],
                    tr["frac_anchors_without_neg"],
                    va["frac_anchors_without_neg"],
                    _csv_metric_cell(tr.get("mean_cross_pos_raw_cos", float("nan"))),
                    _csv_metric_cell(va.get("mean_cross_pos_raw_cos", float("nan"))),
                    _csv_metric_cell(tr.get("mean_selected_neg_raw_cos", float("nan"))),
                    _csv_metric_cell(va.get("mean_selected_neg_raw_cos", float("nan"))),
                    tr["same_pos_mean_cos"],
                    va["same_pos_mean_cos"],
                    _csv_metric_cell(tr.get("cross_pos_mean_cos", float("nan"))),
                    _csv_metric_cell(va.get("cross_pos_mean_cos", float("nan"))),
                    _csv_metric_cell(tr["neg_mean_cos"]),
                    _csv_metric_cell(va["neg_mean_cos"]),
                    tr_skip,
                    va_skip,
                ]
            )

        if bool(save_epoch_checkpoints) and (epoch % 5 == 0 or epoch == 1):
            save_contrastive_checkpoint(
                save_dir=ckpt_dir,
                filename=f"model_epoch_{epoch}.pt",
                model=model,
                projector=projector,
                optimizer=opt,
                epoch=epoch,
                patience_counter=patience_counter,
                encoder_config=encoder_config,
                data_metadata=data_cpu.metadata(),
                torch_seed=TORCH_SEED,
                email_train_idx=train_idx.cpu(),
                email_val_idx=val_idx.cpu(),
                email_test_idx=test_idx.cpu(),
                contrastive_hparams=contrastive_hparams,
                anchor_loader_params=anchor_loader_params,
                optimizer_state_dict=opt.state_dict(),
                val_contrastive_total=float(va_loss),
                best_val_contrastive_total=float(best_val),
            )

        if va_loss < best_val:
            best_val = va_loss
            patience_counter = 0
            best_state = {"model": model.state_dict(), "projector": projector.state_dict()}
            save_contrastive_checkpoint(
                save_dir=ckpt_dir,
                filename=model_save_name,
                model=model,
                projector=projector,
                optimizer=opt,
                epoch=epoch,
                patience_counter=patience_counter,
                encoder_config=encoder_config,
                data_metadata=data_cpu.metadata(),
                torch_seed=TORCH_SEED,
                email_train_idx=train_idx.cpu(),
                email_val_idx=val_idx.cpu(),
                email_test_idx=test_idx.cpu(),
                contrastive_hparams=contrastive_hparams,
                anchor_loader_params=anchor_loader_params,
                optimizer_state_dict=opt.state_dict(),
                val_contrastive_total=float(va_loss),
                best_val_contrastive_total=float(best_val),
            )
            print(f"Best val contrastive checkpoint saved to {ckpt_dir / model_save_name}")
        else:
            patience_counter += 1

        prev_lr = opt.param_groups[0]["lr"]
        scheduler.step(va_loss)
        new_lr = opt.param_groups[0]["lr"]
        if new_lr < prev_lr:
            print(f"Learning rate reduced from {prev_lr:.2e} to {new_lr:.2e}")

        if patience_counter >= early_stopping_patience:
            print(f"Early stopping after {epoch} epochs (validation contrastive loss plateau).")
            break

    if best_state:
        model.load_state_dict(best_state["model"])
        projector.load_state_dict(best_state["projector"])

    te, te_skip = eval_epoch_contrastive(
        DEVICE,
        model,
        projector,
        loaders["test"],
        email_loader_input_nodes=test_idx,
        pos_pre=pos_pre,
        positive_rules=pos_rule_list,
        max_cross_positives_per_anchor=contrastive_max_cross_positives_per_anchor,
        cross_positive_raw_cosine_min=contrastive_cross_positive_raw_cosine_min,
        neg_pre=neg_pre,
        negative_channels=neg_ch_list,
        raw_cosine_threshold=contrastive_raw_cosine_threshold,
        max_negatives_per_anchor=contrastive_max_negatives_per_anchor,
        temperature=contrastive_temperature,
        feat_mask_prob=feat_mask_prob,
        edge_drop_probs=edge_drop_probs,
        email_semantic_mask_prob=email_semantic_mask_prob,
        email_semantic_mask_mode=email_semantic_mask_mode,
        email_semantic_block=email_semantic_block,
        primary_ntype=primary_ntype,
        torch_seed=TORCH_SEED,
        epoch=epochs + 1,
        split_name="test",
        contrastive_debug=False,
        contrastive_debug_num_batches=0,
    )
    print(
        f"[Test] loss {te['total']:.4f} elig/anchor {te['mean_eligible_safe']:.3f} "
        f"selected/anchor {te['mean_selected_hard']:.3f} frac_w_neg {te['frac_anchors_with_neg']:.3f} "
        f"cross_pos/anchor {te['mean_cross_positive_per_anchor']:.3f} "
        f"raw_cos(sel) {_fmt_float_or_nan(float(te.get('mean_selected_neg_raw_cos', float('nan'))))} "
        f"same_pos_cos {te['same_pos_mean_cos']:.4f} "
        f"cross_pos_cos {_fmt_float_or_nan(float(te.get('cross_pos_mean_cos', float('nan'))))} "
        f"[skipped {te_skip}]"
    )

    splits = {"email_train_idx": train_idx, "email_val_idx": val_idx, "email_test_idx": test_idx}
    return model, projector, loaders, splits


__all__ = [
    "run_contrastive_training",
    "train_epoch_contrastive",
    "eval_epoch_contrastive",
]
