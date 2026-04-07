"""
VICReg self-supervised training for heterogeneous email-centric graphs (anchor NeighborLoader).
No link-prediction scaffolding on this path.
"""
from __future__ import annotations

import csv
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn
from tqdm import tqdm

from .model import HeteroSAGE
from .model_io import save_vicreg_checkpoint
from .vicreg_modules import (
    VicRegProjector,
    augment_hetero_batch,
    extract_anchor_email_embeddings,
    make_email_anchor_loaders,
    split_email_node_indices,
    vicreg_loss,
)

VICREG_DEBUG = os.getenv("VICREG_DEBUG", "0") == "1"

def _tensor_preview(t: Optional[torch.Tensor], max_vals: int = 20) -> str:
    if t is None:
        return "<absent>"
    if not isinstance(t, torch.Tensor):
        return f"<non-tensor: {type(t).__name__}>"
    flat = t.view(-1)
    preview = flat[:max_vals].detach().cpu().tolist()
    return f"dtype={t.dtype} shape={tuple(t.shape)} first{min(max_vals, flat.numel())}={preview}"


def _email_store_debug_summary(store: Any, label: str) -> None:
    print(f"--- {label} email store ---")
    bs = getattr(store, "batch_size", None)
    print(f"batch_size: {bs} ({type(bs).__name__})")
    print(f"input_id: {_tensor_preview(getattr(store, 'input_id', None))}")
    print(f"n_id: {_tensor_preview(getattr(store, 'n_id', None))}")


def train_epoch_vicreg(
    device: torch.device,
    model: nn.Module,
    projector: nn.Module,
    optimizer: torch.optim.Optimizer,
    loader,
    *,
    email_loader_input_nodes: torch.Tensor,
    w_inv: float,
    w_var: float,
    w_cov: float,
    feat_mask_prob: float,
    edge_drop_probs: Optional[Dict[str, float]],
    email_full_zero_prob: float,
    email_semantic_mask_prob: float,
    email_semantic_mask_mode: str,
    email_semantic_apply_to: str,
    email_semantic_block: Optional[Tuple[int, int]],
    primary_ntype: str,
    torch_seed: int,
    epoch: int,
    vicreg_debug_anchor_matching: bool = False,
    vicreg_debug_num_batches: int = 3,
) -> Tuple[Dict[str, float], int]:
    """
    Loader batches must stay on CPU until after augmentation.
    Returns (averaged metric dict, number of skipped batches).
    """
    model.train()
    projector.train()
    totals = {"total": 0.0, "inv": 0.0, "var": 0.0, "cov": 0.0}
    n_batches = 0
    skipped = 0

    pbar = tqdm(loader, desc="VICReg train", leave=True)
    for batch_idx, batch in enumerate(pbar):
        try:
            if not hasattr(batch[primary_ntype], "batch_size"):
                raise AttributeError(f"batch[{primary_ntype!r}] missing batch_size")
            bs = int(batch[primary_ntype].batch_size)
        except Exception:
            skipped += 1
            if VICREG_DEBUG and batch_idx < 3:
                try:
                    node_types = list(getattr(batch, "node_types", []))
                except Exception:
                    node_types = "<unavailable>"
                print(
                    f"[VICReg DEBUG][train] batch_idx={batch_idx} failed bs check. "
                    f"primary_ntype={primary_ntype!r} node_types={node_types}"
                )
            continue
        if bs < 2:
            skipped += 1
            if VICREG_DEBUG and batch_idx < 3:
                print(f"[VICReg DEBUG][train] batch_idx={batch_idx} skipped: bs={bs} (<2)")
            continue

        do_anchor_debug = (
            vicreg_debug_anchor_matching and batch_idx < vicreg_debug_num_batches
        )

        g1 = torch.Generator(device="cpu")
        g2 = torch.Generator(device="cpu")
        g1.manual_seed(torch_seed + epoch * 1_000_003 + batch_idx * 17 + 1)
        g2.manual_seed(torch_seed + epoch * 1_000_003 + batch_idx * 17 + 2)

        base_cpu = batch.clone()
        if do_anchor_debug:
            print(f"=== VICREG ANCHOR DEBUG: TRAIN BATCH {batch_idx} ===")
            print(f"epoch={epoch} primary_ntype={primary_ntype!r}")
            print(
                f"B) Namespace check: email_loader_input_nodes (train split) "
                f"shape={tuple(email_loader_input_nodes.shape)} "
                f"first20={_tensor_preview(email_loader_input_nodes.view(-1), 20)}"
            )
            _email_store_debug_summary(base_cpu[primary_ntype], "base")

        v1 = augment_hetero_batch(
            base_cpu.clone(),
            feat_mask_prob,
            edge_drop_probs,
            generator=g1,
            email_full_zero_prob=email_full_zero_prob,
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
            email_full_zero_prob=email_full_zero_prob,
            email_semantic_mask_prob=email_semantic_mask_prob,
            email_semantic_mask_mode=email_semantic_mask_mode,
            email_semantic_block=email_semantic_block,
            primary_ntype=primary_ntype,
        ).to(device)

        if (
            do_anchor_debug
            and batch_idx == 0
            and float(email_full_zero_prob) > 0.0
        ):
            xe1 = v1[primary_ntype].x
            xe2 = v2[primary_ntype].x
            fr1 = float((xe1 == 0).all(dim=1).float().mean().item()) if xe1 is not None and xe1.numel() > 0 else 0.0
            fr2 = float((xe2 == 0).all(dim=1).float().mean().item()) if xe2 is not None and xe2.numel() > 0 else 0.0
            print(
                "Z) Full email feature zeroing debug (train): "
                f"p={float(email_full_zero_prob):.4f} "
                f"fraction_email_rows_fully_zero view1={fr1:.4f} view2={fr2:.4f}"
            )
        if (
            do_anchor_debug
            and batch_idx == 0
            and str(email_semantic_mask_mode).lower().strip() != "none"
            and float(email_semantic_mask_prob) > 0.0
            and email_semantic_block is not None
        ):
            s0, s1 = int(email_semantic_block[0]), int(email_semantic_block[1])
            b1 = v1[primary_ntype].x[:, s0:s1]
            b2 = v2[primary_ntype].x[:, s0:s1]
            mode = str(email_semantic_mask_mode).lower().strip()
            if mode == "block_zero":
                z1 = float((b1 == 0).all(dim=1).float().mean().item())
                z2 = float((b2 == 0).all(dim=1).float().mean().item())
                z_name = "fraction_email_nodes_full_block_zeroed"
            else:
                z1 = float((b1 == 0).float().mean().item())
                z2 = float((b2 == 0).float().mean().item())
                z_name = "fraction_semantic_block_values_zero"
            print(
                "S) Semantic block masking debug (train): "
                f"block=[{s0},{s1}) mode={mode} p={float(email_semantic_mask_prob):.4f} "
                f"{z_name} view1={z1:.4f} view2={z2:.4f}"
            )

        optimizer.zero_grad(set_to_none=True)
        h1 = model(v1.x_dict, v1.edge_index_dict)
        h2 = model(v2.x_dict, v2.edge_index_dict)

        if do_anchor_debug:
            print("A) Batch-level summary:")
            print(f"  epoch={epoch} batch_idx={batch_idx} primary_ntype={primary_ntype!r}")
            print(f"  view1: h_email rows={tuple(h1[primary_ntype].shape)}")
            print(f"  view2: h_email rows={tuple(h2[primary_ntype].shape)}")
            _email_store_debug_summary(v1[primary_ntype], "view1")
            _email_store_debug_summary(v2[primary_ntype], "view2")
        try:
            if do_anchor_debug:
                a_base_rows = None
                # Base forward + extraction is debug-only; failure should not change training skipping logic.
                was_training = model.training
                model.eval()
                with torch.no_grad():
                    base_for_debug = base_cpu.clone().to(device)
                    h_base = model(base_for_debug.x_dict, base_for_debug.edge_index_dict)
                    print(f"  base encoder output: h_email rows={tuple(h_base[primary_ntype].shape)}")
                    try:
                        print("--- base extraction ---")
                        _a_base, a_base_rows = extract_anchor_email_embeddings(
                            h_base[primary_ntype],
                            base_for_debug[primary_ntype],
                            input_id=getattr(base_for_debug[primary_ntype], "input_id", None),
                            email_loader_input_nodes=email_loader_input_nodes,
                            debug_anchor_matching=True,
                            debug_tag="base",
                            return_anchor_row_indices=True,
                        )
                    except ValueError as e_base:
                        print(f"[VICREG DEBUG] base anchor extraction failed (debug-only): {e_base}")
                if was_training:
                    model.train()

                print("--- view1 extraction ---")
                a1, a1_rows = extract_anchor_email_embeddings(
                    # view1 extraction
                    h1[primary_ntype],
                    v1[primary_ntype],
                    input_id=getattr(v1[primary_ntype], "input_id", None),
                    email_loader_input_nodes=email_loader_input_nodes,
                    debug_anchor_matching=True,
                    debug_tag="view1",
                    return_anchor_row_indices=True,
                )
                print("--- view2 extraction ---")
                a2, a2_rows = extract_anchor_email_embeddings(
                    # view2 extraction
                    h2[primary_ntype],
                    v2[primary_ntype],
                    input_id=getattr(v2[primary_ntype], "input_id", None),
                    email_loader_input_nodes=email_loader_input_nodes,
                    debug_anchor_matching=True,
                    debug_tag="view2",
                    return_anchor_row_indices=True,
                )
            else:
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
        except ValueError as e:
            skipped += 1
            if VICREG_DEBUG and batch_idx < 3:
                # Print just enough info to see why anchor extraction fails.
                store = v1[primary_ntype]
                input_id = getattr(store, "input_id", None)
                n_id = getattr(store, "n_id", None)
                input_id_numel = None if input_id is None else int(input_id.numel())
                n_id_numel = None if n_id is None else int(n_id.numel())
                try:
                    input_id_head = (
                        input_id.view(-1)[:5].detach().cpu().tolist() if input_id is not None else None
                    )
                except Exception:
                    input_id_head = "<unavailable>"
                try:
                    n_id_head = (
                        n_id[:5].detach().cpu().tolist() if n_id is not None else None
                    )
                except Exception:
                    n_id_head = "<unavailable>"

                print(
                    f"[VICReg DEBUG][train] batch_idx={batch_idx} anchor extraction failed: {e}\n"
                    f"  bs={bs}  h1_rows={tuple(h1[primary_ntype].shape)}\n"
                    f"  input_id_numel={input_id_numel} input_id_head={input_id_head}\n"
                    f"  n_id_numel={n_id_numel} n_id_head={n_id_head}"
                )
            continue

        if do_anchor_debug:
            store_base = base_cpu[primary_ntype]
            store_v1 = v1[primary_ntype]
            store_v2 = v2[primary_ntype]
            bs_local = int(getattr(store_v1, "batch_size"))

            n_id_base = getattr(store_base, "n_id", None)
            input_id_base = getattr(store_base, "input_id", None)
            base_equal = False
            if n_id_base is not None and input_id_base is not None:
                n_head = n_id_base.view(-1)[:bs_local].detach().cpu()
                i_head = input_id_base.view(-1)[:bs_local].detach().cpu()
                base_equal = torch.equal(n_head, i_head)

            print("E) Leading-row assumption check:")
            if n_id_base is not None:
                print(f"  base n_id[:bs] first20={_tensor_preview(n_id_base.view(-1)[:bs_local], 20)}")
            print(f"  base input_id[:bs] first20={_tensor_preview(input_id_base.view(-1)[:bs_local], 20) if input_id_base is not None else '<absent>'}")
            print(f"  base n_id[:bs] == input_id[:bs] exactly: {base_equal}")

            n_id1 = getattr(store_v1, "n_id", None)
            input_id1 = getattr(store_v1, "input_id", None)
            if n_id1 is not None:
                v1_nhead = n_id1.view(-1)[:bs_local].detach().cpu()
            else:
                v1_nhead = None
            if input_id1 is not None:
                v1_ihead = input_id1.view(-1)[:bs_local].detach().cpu()
            else:
                v1_ihead = None
            v1_equal = (
                v1_nhead is not None
                and v1_ihead is not None
                and v1_nhead.numel() == v1_ihead.numel()
                and torch.equal(v1_nhead, v1_ihead)
            )
            print(f"  view1 n_id[:bs] first20={_tensor_preview(v1_nhead, 20)}")
            print(f"  view1 input_id[:bs] first20={_tensor_preview(v1_ihead, 20) if v1_ihead is not None else '<absent>'}")
            print(f"  view1 n_id[:bs] == input_id[:bs] exactly: {v1_equal}")

            n_id2 = getattr(store_v2, "n_id", None)
            input_id2 = getattr(store_v2, "input_id", None)
            if n_id2 is not None:
                v2_nhead = n_id2.view(-1)[:bs_local].detach().cpu()
            else:
                v2_nhead = None
            if input_id2 is not None:
                v2_ihead = input_id2.view(-1)[:bs_local].detach().cpu()
            else:
                v2_ihead = None
            v2_equal = (
                v2_nhead is not None
                and v2_ihead is not None
                and v2_nhead.numel() == v2_ihead.numel()
                and torch.equal(v2_nhead, v2_ihead)
            )
            print(f"  view2 n_id[:bs] first20={_tensor_preview(v2_nhead, 20)}")
            print(f"  view2 input_id[:bs] first20={_tensor_preview(v2_ihead, 20) if v2_ihead is not None else '<absent>'}")
            print(f"  view2 n_id[:bs] == input_id[:bs] exactly: {v2_equal}")

            # View-level checks + extracted identity sequence comparison.
            if n_id1 is not None:
                ids1 = n_id1.view(-1)[a1_rows].detach().cpu()
            else:
                ids1 = a1_rows.detach().cpu()
            if n_id2 is not None:
                ids2 = n_id2.view(-1)[a2_rows].detach().cpu()
            else:
                ids2 = a2_rows.detach().cpu()

            if a_base_rows is not None:
                a_base_rows_cpu = a_base_rows.detach().cpu()
                if n_id_base is not None:
                    ids_base = n_id_base.view(-1)[a_base_rows_cpu].detach().cpu()
                else:
                    ids_base = a_base_rows_cpu
                print("--- base extracted anchors ---")
                print(f"  base anchor ids (first20)={_tensor_preview(ids_base[:bs_local], 20)}")
                print(
                    f"  base extracted rows == range(bs): {torch.equal(a_base_rows_cpu, torch.arange(bs_local))}"
                )
                print(f"  base ids == view1 ids (same order): {torch.equal(ids_base, ids1)}")
                print(f"  base ids == view2 ids (same order): {torch.equal(ids_base, ids2)}")
            else:
                print("--- base extracted anchors ---")
                print("  base anchor extraction unavailable (debug-only failure).")

            print("--- D) Anchor identity comparison across views ---")
            print(f"  view1: a1.shape={tuple(a1.shape)} a1_rows.shape={tuple(a1_rows.shape)}")
            print(f"  view2: a2.shape={tuple(a2.shape)} a2_rows.shape={tuple(a2_rows.shape)}")
            print(f"  view1 anchor ids (first20)={_tensor_preview(ids1[:bs_local], 20)}")
            print(f"  view2 anchor ids (first20)={_tensor_preview(ids2[:bs_local], 20)}")
            print(f"  anchors count view1={int(ids1.numel())} unique={int(torch.unique(ids1).numel())}")
            print(f"  anchors count view2={int(ids2.numel())} unique={int(torch.unique(ids2).numel())}")
            same_seq = torch.equal(ids1, ids2)
            print(f"  view1_ids == view2_ids (same order): {same_seq}")
            if not same_seq:
                print("ANCHOR MISMATCH BETWEEN VIEWS")

            # Whether the current extraction rows are contiguous (range(bs)).
            a1_rows_cpu = a1_rows.detach().cpu()
            a2_rows_cpu = a2_rows.detach().cpu()
            print("E) Extracted anchor row contiguity:")
            print(f"  view1 extracted rows == range(bs): {torch.equal(a1_rows_cpu, torch.arange(bs_local))}")
            print(f"  view2 extracted rows == range(bs): {torch.equal(a2_rows_cpu, torch.arange(bs_local))}")

            print("F) Strong checks (debug, split-aware extraction):")
            print(
                f"  anchor_global_count={int(ids1.numel())} batch_size={bs_local} "
                f"match={int(ids1.numel()) == bs_local}"
            )
            print(f"  mean(a1)={float(a1.detach().mean().cpu()):.6f} mean(a2)={float(a2.detach().mean().cpu()):.6f}")
            if int(ids1.numel()) != bs_local or int(ids2.numel()) != bs_local:
                print("VICREG ANCHOR ASSERT FAILED: extracted anchor count != batch_size")
                assert False
            if not torch.equal(ids1, ids2):
                print(
                    "VICREG ANCHOR ASSERT FAILED: view1 vs view2 global id sequences differ "
                    f"(first delta may be in tail; view1[:20]={ids1[:20].tolist()} view2[:20]={ids2[:20].tolist()})"
                )
                assert False

        z1 = projector(a1)
        z2 = projector(a2)
        loss, parts = vicreg_loss(z1, z2, w_inv=w_inv, w_var=w_var, w_cov=w_cov)
        loss.backward()
        optimizer.step()

        totals["inv"] += float(parts["inv"].item())
        totals["var"] += float(parts["var"].item())
        totals["cov"] += float(parts["cov"].item())
        totals["total"] += float(parts["total"].item())

        n_batches += 1

    if n_batches == 0:
        return {k: 0.0 for k in totals}, skipped
    return {k: totals[k] / n_batches for k in totals}, skipped


@torch.no_grad()
def eval_epoch_vicreg(
    device: torch.device,
    model: nn.Module,
    projector: nn.Module,
    loader,
    *,
    email_loader_input_nodes: torch.Tensor,
    w_inv: float,
    w_var: float,
    w_cov: float,
    feat_mask_prob: float,
    edge_drop_probs: Optional[Dict[str, float]],
    email_full_zero_prob: float,
    email_full_zero_apply_to: str,
    email_semantic_mask_prob: float,
    email_semantic_mask_mode: str,
    email_semantic_apply_to: str,
    email_semantic_block: Optional[Tuple[int, int]],
    primary_ntype: str,
    torch_seed: int,
    epoch: int,
    split_name: str = "val",
    vicreg_debug_anchor_matching: bool = False,
    vicreg_debug_num_batches: int = 3,
) -> Tuple[Dict[str, float], int]:
    model.eval()
    projector.eval()
    totals = {"total": 0.0, "inv": 0.0, "var": 0.0, "cov": 0.0}
    n_batches = 0
    skipped = 0
    apply_full_zero_on_eval = str(email_full_zero_apply_to).lower().strip() == "train_and_eval"
    apply_semantic_on_eval = str(email_semantic_apply_to).lower().strip() == "train_and_eval"

    pbar = tqdm(loader, desc=f"VICReg {split_name}", leave=True)
    for batch_idx, batch in enumerate(pbar):
        try:
            if not hasattr(batch[primary_ntype], "batch_size"):
                raise AttributeError(f"batch[{primary_ntype!r}] missing batch_size")
            bs = int(batch[primary_ntype].batch_size)
        except Exception:
            skipped += 1
            if VICREG_DEBUG and batch_idx < 3:
                try:
                    node_types = list(getattr(batch, "node_types", []))
                except Exception:
                    node_types = "<unavailable>"
                print(
                    f"[VICReg DEBUG][{split_name}] batch_idx={batch_idx} failed bs check. "
                    f"primary_ntype={primary_ntype!r} node_types={node_types}"
                )
            continue
        if bs < 2:
            skipped += 1
            if VICREG_DEBUG and batch_idx < 3:
                print(f"[VICReg DEBUG][{split_name}] batch_idx={batch_idx} skipped: bs={bs} (<2)")
            continue

        do_anchor_debug = (
            vicreg_debug_anchor_matching and batch_idx < vicreg_debug_num_batches
        )

        g1 = torch.Generator(device="cpu")
        g2 = torch.Generator(device="cpu")
        g1.manual_seed(torch_seed + 9_000_001 + epoch + batch_idx + 1)
        g2.manual_seed(torch_seed + 9_000_001 + epoch + batch_idx + 2)

        base_cpu = batch.clone()
        if do_anchor_debug:
            print(f"=== VICREG ANCHOR DEBUG: {split_name.upper()} BATCH {batch_idx} ===")
            print(f"epoch={epoch} primary_ntype={primary_ntype!r}")
            print(
                f"B) Namespace check: email_loader_input_nodes ({split_name} split) "
                f"shape={tuple(email_loader_input_nodes.shape)} "
                f"first20={_tensor_preview(email_loader_input_nodes.view(-1), 20)}"
            )
            _email_store_debug_summary(base_cpu[primary_ntype], "base")
        v1 = augment_hetero_batch(
            base_cpu.clone(),
            feat_mask_prob,
            edge_drop_probs,
            generator=g1,
            email_full_zero_prob=(email_full_zero_prob if apply_full_zero_on_eval else 0.0),
            email_semantic_mask_prob=(email_semantic_mask_prob if apply_semantic_on_eval else 0.0),
            email_semantic_mask_mode=(email_semantic_mask_mode if apply_semantic_on_eval else "none"),
            email_semantic_block=email_semantic_block,
            primary_ntype=primary_ntype,
        ).to(device)
        v2 = augment_hetero_batch(
            base_cpu.clone(),
            feat_mask_prob,
            edge_drop_probs,
            generator=g2,
            email_full_zero_prob=(email_full_zero_prob if apply_full_zero_on_eval else 0.0),
            email_semantic_mask_prob=(email_semantic_mask_prob if apply_semantic_on_eval else 0.0),
            email_semantic_mask_mode=(email_semantic_mask_mode if apply_semantic_on_eval else "none"),
            email_semantic_block=email_semantic_block,
            primary_ntype=primary_ntype,
        ).to(device)

        if (
            do_anchor_debug
            and batch_idx == 0
            and apply_full_zero_on_eval
            and float(email_full_zero_prob) > 0.0
        ):
            xe1 = v1[primary_ntype].x
            xe2 = v2[primary_ntype].x
            fr1 = float((xe1 == 0).all(dim=1).float().mean().item()) if xe1 is not None and xe1.numel() > 0 else 0.0
            fr2 = float((xe2 == 0).all(dim=1).float().mean().item()) if xe2 is not None and xe2.numel() > 0 else 0.0
            print(
                f"Z) Full email feature zeroing debug ({split_name}): "
                f"p={float(email_full_zero_prob):.4f} "
                f"fraction_email_rows_fully_zero view1={fr1:.4f} view2={fr2:.4f}"
            )
        if (
            do_anchor_debug
            and batch_idx == 0
            and apply_semantic_on_eval
            and str(email_semantic_mask_mode).lower().strip() != "none"
            and float(email_semantic_mask_prob) > 0.0
            and email_semantic_block is not None
        ):
            s0, s1 = int(email_semantic_block[0]), int(email_semantic_block[1])
            b1 = v1[primary_ntype].x[:, s0:s1]
            b2 = v2[primary_ntype].x[:, s0:s1]
            mode = str(email_semantic_mask_mode).lower().strip()
            if mode == "block_zero":
                z1 = float((b1 == 0).all(dim=1).float().mean().item())
                z2 = float((b2 == 0).all(dim=1).float().mean().item())
                z_name = "fraction_email_nodes_full_block_zeroed"
            else:
                z1 = float((b1 == 0).float().mean().item())
                z2 = float((b2 == 0).float().mean().item())
                z_name = "fraction_semantic_block_values_zero"
            print(
                f"S) Semantic block masking debug ({split_name}): "
                f"block=[{s0},{s1}) mode={mode} p={float(email_semantic_mask_prob):.4f} "
                f"{z_name} view1={z1:.4f} view2={z2:.4f}"
            )

        h1 = model(v1.x_dict, v1.edge_index_dict)
        h2 = model(v2.x_dict, v2.edge_index_dict)

        if do_anchor_debug:
            print("A) Batch-level summary:")
            print(f"  view1: h_email rows={tuple(h1[primary_ntype].shape)}")
            print(f"  view2: h_email rows={tuple(h2[primary_ntype].shape)}")
            _email_store_debug_summary(v1[primary_ntype], "view1")
            _email_store_debug_summary(v2[primary_ntype], "view2")
        try:
            if do_anchor_debug:
                # Base forward + anchor extraction is debug-only.
                base_for_debug = base_cpu.clone().to(device)
                h_base = model(base_for_debug.x_dict, base_for_debug.edge_index_dict)
                print(f"  base encoder output: h_email rows={tuple(h_base[primary_ntype].shape)}")
                print("--- base extraction ---")
                _a_base, a_base_rows = extract_anchor_email_embeddings(
                    h_base[primary_ntype],
                    base_for_debug[primary_ntype],
                    input_id=getattr(base_for_debug[primary_ntype], "input_id", None),
                    email_loader_input_nodes=email_loader_input_nodes,
                    debug_anchor_matching=True,
                    debug_tag="base",
                    return_anchor_row_indices=True,
                )
                print("--- view1 extraction ---")
                a1, a1_rows = extract_anchor_email_embeddings(
                    h1[primary_ntype],
                    v1[primary_ntype],
                    input_id=getattr(v1[primary_ntype], "input_id", None),
                    email_loader_input_nodes=email_loader_input_nodes,
                    debug_anchor_matching=True,
                    debug_tag="view1",
                    return_anchor_row_indices=True,
                )
                print("--- view2 extraction ---")
                a2, a2_rows = extract_anchor_email_embeddings(
                    h2[primary_ntype],
                    v2[primary_ntype],
                    input_id=getattr(v2[primary_ntype], "input_id", None),
                    email_loader_input_nodes=email_loader_input_nodes,
                    debug_anchor_matching=True,
                    debug_tag="view2",
                    return_anchor_row_indices=True,
                )
            else:
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
        except ValueError as e:
            skipped += 1
            if VICREG_DEBUG and batch_idx < 3:
                store = v1[primary_ntype]
                input_id = getattr(store, "input_id", None)
                n_id = getattr(store, "n_id", None)
                input_id_numel = None if input_id is None else int(input_id.numel())
                n_id_numel = None if n_id is None else int(n_id.numel())
                try:
                    input_id_head = (
                        input_id.view(-1)[:5].detach().cpu().tolist() if input_id is not None else None
                    )
                except Exception:
                    input_id_head = "<unavailable>"
                try:
                    n_id_head = n_id[:5].detach().cpu().tolist() if n_id is not None else None
                except Exception:
                    n_id_head = "<unavailable>"

                print(
                    f"[VICReg DEBUG][{split_name}] batch_idx={batch_idx} anchor extraction failed: {e}\n"
                    f"  bs={bs}  h1_rows={tuple(h1[primary_ntype].shape)}\n"
                    f"  input_id_numel={input_id_numel} input_id_head={input_id_head}\n"
                    f"  n_id_numel={n_id_numel} n_id_head={n_id_head}"
                )
            continue

        if do_anchor_debug:
            store_base = base_cpu[primary_ntype]
            store_v1 = v1[primary_ntype]
            store_v2 = v2[primary_ntype]
            bs_local = int(getattr(store_v1, "batch_size"))

            n_id_base = getattr(store_base, "n_id", None)
            input_id_base = getattr(store_base, "input_id", None)
            base_equal = False
            if n_id_base is not None and input_id_base is not None:
                base_equal = torch.equal(
                    n_id_base.view(-1)[:bs_local].detach().cpu(),
                    input_id_base.view(-1)[:bs_local].detach().cpu(),
                )
            print("E) Leading-row assumption check:")
            if n_id_base is not None:
                print(f"  base n_id[:bs] first20={_tensor_preview(n_id_base.view(-1)[:bs_local], 20)}")
            print(
                f"  base input_id[:bs] first20={_tensor_preview(input_id_base.view(-1)[:bs_local], 20) if input_id_base is not None else '<absent>'}"
            )
            print(f"  base n_id[:bs] == input_id[:bs] exactly: {base_equal}")

            # Prepare view1/view2 ids for leading-row checks.
            n_id1 = getattr(store_v1, "n_id", None)
            n_id2 = getattr(store_v2, "n_id", None)

            if n_id1 is not None:
                v1_ihead = getattr(store_v1, "input_id", None)
                if v1_ihead is not None:
                    v1_ihead = v1_ihead.view(-1)[:bs_local].detach().cpu()
                v1_equal = (
                    n_id1.view(-1)[:bs_local].detach().cpu().numel() == 0
                    or (v1_ihead is not None and torch.equal(n_id1.view(-1)[:bs_local].detach().cpu(), v1_ihead))
                )
                print(f"  view1 n_id[:bs] first20={_tensor_preview(n_id1.view(-1)[:bs_local].detach().cpu(), 20)}")
                print(
                    f"  view1 input_id[:bs] first20={_tensor_preview(v1_ihead, 20) if v1_ihead is not None else '<absent>'}"
                )
                print(f"  view1 n_id[:bs] == input_id[:bs] exactly: {v1_equal}")

            if n_id2 is not None:
                v2_ihead = getattr(store_v2, "input_id", None)
                if v2_ihead is not None:
                    v2_ihead = v2_ihead.view(-1)[:bs_local].detach().cpu()
                v2_equal = (
                    n_id2.view(-1)[:bs_local].detach().cpu().numel() == 0
                    or (v2_ihead is not None and torch.equal(n_id2.view(-1)[:bs_local].detach().cpu(), v2_ihead))
                )
                print(f"  view2 n_id[:bs] first20={_tensor_preview(n_id2.view(-1)[:bs_local].detach().cpu(), 20)}")
                print(
                    f"  view2 input_id[:bs] first20={_tensor_preview(v2_ihead, 20) if v2_ihead is not None else '<absent>'}"
                )
                print(f"  view2 n_id[:bs] == input_id[:bs] exactly: {v2_equal}")

            n_id1 = getattr(store_v1, "n_id", None)
            n_id2 = getattr(store_v2, "n_id", None)
            if n_id1 is not None:
                ids1 = n_id1.view(-1)[a1_rows].detach().cpu()
            else:
                ids1 = a1_rows.detach().cpu()
            if n_id2 is not None:
                ids2 = n_id2.view(-1)[a2_rows].detach().cpu()
            else:
                ids2 = a2_rows.detach().cpu()

            a_base_rows_cpu = a_base_rows.detach().cpu()
            if n_id_base is not None:
                ids_base = n_id_base.view(-1)[a_base_rows_cpu].detach().cpu()
            else:
                ids_base = a_base_rows_cpu
            print("--- base extracted anchors ---")
            print(f"  base anchor ids (first20)={_tensor_preview(ids_base[:bs_local], 20)}")
            print(
                f"  base extracted rows == range(bs): {torch.equal(a_base_rows_cpu, torch.arange(bs_local))}"
            )
            print(f"  base ids == view1 ids (same order): {torch.equal(ids_base, ids1)}")
            print(f"  base ids == view2 ids (same order): {torch.equal(ids_base, ids2)}")

            print("--- D) Anchor identity comparison across views ---")
            print(f"  view1 anchor ids (first20)={_tensor_preview(ids1[:bs_local], 20)}")
            print(f"  view2 anchor ids (first20)={_tensor_preview(ids2[:bs_local], 20)}")
            print(f"  anchors count view1={int(ids1.numel())} unique={int(torch.unique(ids1).numel())}")
            print(f"  anchors count view2={int(ids2.numel())} unique={int(torch.unique(ids2).numel())}")
            same_seq = torch.equal(ids1, ids2)
            print(f"  view1_ids == view2_ids (same order): {same_seq}")
            if not same_seq:
                print("ANCHOR MISMATCH BETWEEN VIEWS")

            a1_rows_cpu = a1_rows.detach().cpu()
            a2_rows_cpu = a2_rows.detach().cpu()
            print("E) Extracted anchor row contiguity:")
            print(f"  view1 extracted rows == range(bs): {torch.equal(a1_rows_cpu, torch.arange(bs_local))}")
            print(f"  view2 extracted rows == range(bs): {torch.equal(a2_rows_cpu, torch.arange(bs_local))}")

            print("F) Strong checks (debug, split-aware extraction):")
            print(
                f"  anchor_global_count={int(ids1.numel())} batch_size={bs_local} "
                f"match={int(ids1.numel()) == bs_local}"
            )
            if int(ids1.numel()) != bs_local or int(ids2.numel()) != bs_local:
                print("VICREG ANCHOR ASSERT FAILED: extracted anchor count != batch_size")
                assert False
            if not torch.equal(ids1, ids2):
                print(
                    "VICREG ANCHOR ASSERT FAILED: view1 vs view2 global id sequences differ "
                    f"(view1[:20]={ids1[:20].tolist()} view2[:20]={ids2[:20].tolist()})"
                )
                assert False

        z1 = projector(a1)
        z2 = projector(a2)
        _loss, parts = vicreg_loss(z1, z2, w_inv=w_inv, w_var=w_var, w_cov=w_cov)
        totals["inv"] += float(parts["inv"].item())
        totals["var"] += float(parts["var"].item())
        totals["cov"] += float(parts["cov"].item())
        totals["total"] += float(parts["total"].item())

        n_batches += 1

    if n_batches == 0:
        return {k: 0.0 for k in totals}, skipped
    return {k: totals[k] / n_batches for k in totals}, skipped


def run_vicreg_training(
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
    w_inv: float = 25.0,
    w_var: float = 25.0,
    w_cov: float = 1.0,
    feat_mask_prob: float = 0.05,
    edge_drop_probs: Optional[Dict[str, float]] = None,
    email_full_zero_prob: float = 0.15,
    email_full_zero_apply_to: str = "train_only",
    projector_hidden_dim: Optional[int] = None,
    projector_out_dim: Optional[int] = None,
    save_epoch_checkpoints: bool = True,
    email_semantic_mask_prob: float = 0.0,
    email_semantic_mask_mode: str = "none",
    email_semantic_apply_to: str = "train_only",
    email_semantic_block: Optional[Tuple[int, int]] = None,
    vicreg_debug_anchor_matching: bool = False,
    vicreg_debug_num_batches: int = 3,
):
    if fanout is None:
        fanout = [15, 10]
    anchor_bs = int(anchor_batch_size if anchor_batch_size is not None else (batch_size or 256))
    ph = int(projector_hidden_dim if projector_hidden_dim is not None else out_dim)
    po = int(projector_out_dim if projector_out_dim is not None else out_dim)
    sem_mode = str(email_semantic_mask_mode).lower().strip()
    sem_apply = str(email_semantic_apply_to).lower().strip()
    sem_prob = float(email_semantic_mask_prob)
    if sem_mode not in {"none", "block_zero", "feature_mask"}:
        raise ValueError(
            f"Invalid vicreg_email_semantic_mask_mode={email_semantic_mask_mode!r}; "
            "expected one of: 'none', 'block_zero', 'feature_mask'."
        )
    if sem_apply not in {"train_only", "train_and_eval"}:
        raise ValueError(
            f"Invalid vicreg_email_semantic_apply_to={email_semantic_apply_to!r}; "
            "expected one of: 'train_only', 'train_and_eval'."
        )
    if sem_prob < 0.0 or sem_prob > 1.0:
        raise ValueError(
            f"vicreg_email_semantic_mask_prob must be in [0,1], got {sem_prob}."
        )
    full_zero_prob = float(email_full_zero_prob)
    full_zero_apply = str(email_full_zero_apply_to).lower().strip()
    if not (0.0 <= full_zero_prob <= 1.0):
        raise ValueError(
            f"vicreg_email_full_zero_prob must be in [0,1], got {full_zero_prob}."
        )
    if full_zero_apply not in {"train_only", "train_and_eval"}:
        raise ValueError(
            f"Invalid vicreg_email_full_zero_apply_to={email_full_zero_apply_to!r}; "
            "expected one of: 'train_only', 'train_and_eval'."
        )

    if sem_mode != "none" and sem_prob > 0.0:
        if email_semantic_block is None:
            raise ValueError(
                "Semantic masking requested but vicreg_email_semantic_block is not set. "
                "Provide [start_idx, end_idx] for email feature semantic block."
            )
        if len(email_semantic_block) != 2:
            raise ValueError(
                f"vicreg_email_semantic_block must have 2 items, got {email_semantic_block!r}."
            )
        b0, b1 = int(email_semantic_block[0]), int(email_semantic_block[1])
        if not (0 <= b0 < b1):
            raise ValueError(
                f"Invalid vicreg_email_semantic_block={email_semantic_block!r}; expected [start, end) with 0 <= start < end."
            )

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
                "train_vicreg_total",
                "train_inv",
                "train_var",
                "train_cov",
                "train_skipped_batches",
                "val_vicreg_total",
                "val_inv",
                "val_var",
                "val_cov",
                "val_skipped_batches",
            ]
        )

    data_cpu = data.to("cpu")
    print("VICReg training | metadata:", data_cpu.metadata())

    if sem_mode != "none" and sem_prob > 0.0:
        if primary_ntype not in data_cpu.node_types:
            raise ValueError(
                f"primary_ntype={primary_ntype!r} not present in graph; cannot apply semantic masking."
            )
        x_email = data_cpu[primary_ntype].x
        if x_email is None or x_email.dim() != 2:
            raise ValueError(
                f"{primary_ntype}.x must be a 2D tensor for semantic masking; got {None if x_email is None else tuple(x_email.shape)}."
            )
        b0, b1 = int(email_semantic_block[0]), int(email_semantic_block[1])
        if not (0 <= b0 < b1 <= int(x_email.size(1))):
            raise ValueError(
                f"vicreg_email_semantic_block [{b0}, {b1}) is out of bounds for "
                f"{primary_ntype}.x feature dim {int(x_email.size(1))}."
            )
        sem_block = (b0, b1)
    else:
        sem_block = None

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
    vicreg_hparams = {
        "vicreg_weight_invariance": w_inv,
        "vicreg_weight_variance": w_var,
        "vicreg_weight_covariance": w_cov,
        "vicreg_feat_mask_prob": feat_mask_prob,
        "vicreg_edge_drop_probs": edge_drop_probs,
        "vicreg_projector_hidden_dim": ph,
        "vicreg_projector_out_dim": po,
        "vicreg_email_full_zero_prob": full_zero_prob,
        "vicreg_email_full_zero_apply_to": full_zero_apply,
        "vicreg_email_semantic_mask_prob": sem_prob,
        "vicreg_email_semantic_mask_mode": sem_mode,
        "vicreg_email_semantic_apply_to": sem_apply,
        "vicreg_email_semantic_block": list(sem_block) if sem_block is not None else None,
    }
    anchor_loader_params = {
        "anchor_batch_size": anchor_bs,
        "fanout": list(fanout) if isinstance(fanout, (list, tuple)) else fanout,
        "primary_ntype": primary_ntype,
        "num_gnn_layers": layers,
    }

    training_record: Dict[str, Any] = {
        "training_objective": "vicreg",
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
        "vicreg": vicreg_hparams,
        "anchor_loader": anchor_loader_params,
    }
    with open(run_dir / training_config_json, "w", encoding="utf-8") as f:
        json.dump(training_record, f, indent=2)

    best_val = float("inf")
    patience_counter = 0
    best_state = None

    print("Starting VICReg training")
    for epoch in range(1, epochs + 1):
        tr, tr_skip = train_epoch_vicreg(
            DEVICE,
            model,
            projector,
            opt,
            loaders["train"],
            email_loader_input_nodes=train_idx,
            w_inv=w_inv,
            w_var=w_var,
            w_cov=w_cov,
            feat_mask_prob=feat_mask_prob,
            edge_drop_probs=edge_drop_probs,
            email_full_zero_prob=full_zero_prob,
            email_semantic_mask_prob=sem_prob,
            email_semantic_mask_mode=sem_mode,
            email_semantic_apply_to=sem_apply,
            email_semantic_block=sem_block,
            primary_ntype=primary_ntype,
            torch_seed=TORCH_SEED,
            epoch=epoch,
            vicreg_debug_anchor_matching=vicreg_debug_anchor_matching,
            vicreg_debug_num_batches=vicreg_debug_num_batches,
        )
        va, va_skip = eval_epoch_vicreg(
            DEVICE,
            model,
            projector,
            loaders["val"],
            email_loader_input_nodes=val_idx,
            w_inv=w_inv,
            w_var=w_var,
            w_cov=w_cov,
            feat_mask_prob=feat_mask_prob,
            edge_drop_probs=edge_drop_probs,
            email_full_zero_prob=full_zero_prob,
            email_full_zero_apply_to=full_zero_apply,
            email_semantic_mask_prob=sem_prob,
            email_semantic_mask_mode=sem_mode,
            email_semantic_apply_to=sem_apply,
            email_semantic_block=sem_block,
            primary_ntype=primary_ntype,
            torch_seed=TORCH_SEED,
            epoch=epoch,
            split_name="val",
            vicreg_debug_anchor_matching=vicreg_debug_anchor_matching,
            vicreg_debug_num_batches=vicreg_debug_num_batches,
        )
        va_loss = va["total"]
        print(
            f"Epoch {epoch:02d} | train VICReg {tr['total']:.4f} "
            f"(inv {tr['inv']:.4f} var {tr['var']:.4f} cov {tr['cov']:.4f}) "
            f"[skipped {tr_skip}] | val {va_loss:.4f} [skipped {va_skip}]"
        )

        with open(metrics_csv_path, mode="a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    epoch,
                    tr["total"],
                    tr["inv"],
                    tr["var"],
                    tr["cov"],
                    tr_skip,
                    va["total"],
                    va["inv"],
                    va["var"],
                    va["cov"],
                    va_skip,
                ]
            )

        if bool(save_epoch_checkpoints) and (epoch % 5 == 0 or epoch == 1):
            save_vicreg_checkpoint(
                save_dir=ckpt_dir,
                filename=f"model_epoch_{epoch}.pt",
                model=model,
                projector=projector,
                optimizer=opt,
                epoch=epoch,
                val_vicreg_total=float(va_loss),
                best_val_vicreg_total=float(best_val),
                patience_counter=patience_counter,
                encoder_config=encoder_config,
                data_metadata=data_cpu.metadata(),
                torch_seed=TORCH_SEED,
                email_train_idx=train_idx.cpu(),
                email_val_idx=val_idx.cpu(),
                email_test_idx=test_idx.cpu(),
                vicreg_hparams=vicreg_hparams,
                anchor_loader_params=anchor_loader_params,
                optimizer_state_dict=opt.state_dict(),
            )

        if va_loss < best_val:
            best_val = va_loss
            patience_counter = 0
            best_state = {
                "model": model.state_dict(),
                "projector": projector.state_dict(),
            }
            save_vicreg_checkpoint(
                save_dir=ckpt_dir,
                filename=model_save_name,
                model=model,
                projector=projector,
                optimizer=opt,
                epoch=epoch,
                val_vicreg_total=float(va_loss),
                best_val_vicreg_total=float(best_val),
                patience_counter=patience_counter,
                encoder_config=encoder_config,
                data_metadata=data_cpu.metadata(),
                torch_seed=TORCH_SEED,
                email_train_idx=train_idx.cpu(),
                email_val_idx=val_idx.cpu(),
                email_test_idx=test_idx.cpu(),
                vicreg_hparams=vicreg_hparams,
                anchor_loader_params=anchor_loader_params,
                optimizer_state_dict=opt.state_dict(),
            )
            print(f"Best val VICReg checkpoint saved to {ckpt_dir / model_save_name}")
        else:
            patience_counter += 1

        prev_lr = opt.param_groups[0]["lr"]
        scheduler.step(va_loss)
        new_lr = opt.param_groups[0]["lr"]
        if new_lr < prev_lr:
            print(f"Learning rate reduced from {prev_lr:.2e} to {new_lr:.2e}")

        if patience_counter >= early_stopping_patience:
            print(f"Early stopping after {epoch} epochs (validation total loss plateau).")
            break

    if best_state:
        model.load_state_dict(best_state["model"])
        projector.load_state_dict(best_state["projector"])

    te, te_skip = eval_epoch_vicreg(
        DEVICE,
        model,
        projector,
        loaders["test"],
        email_loader_input_nodes=test_idx,
        w_inv=w_inv,
        w_var=w_var,
        w_cov=w_cov,
        feat_mask_prob=feat_mask_prob,
        edge_drop_probs=edge_drop_probs,
        email_full_zero_prob=full_zero_prob,
        email_full_zero_apply_to=full_zero_apply,
        email_semantic_mask_prob=sem_prob,
        email_semantic_mask_mode=sem_mode,
        email_semantic_apply_to=sem_apply,
        email_semantic_block=sem_block,
        primary_ntype=primary_ntype,
        torch_seed=TORCH_SEED,
        epoch=epochs + 1,
        split_name="test",
        vicreg_debug_anchor_matching=vicreg_debug_anchor_matching,
        vicreg_debug_num_batches=vicreg_debug_num_batches,
    )
    print(
        f"[Test] VICReg total {te['total']:.4f} | inv {te['inv']:.4f} "
        f"var {te['var']:.4f} cov {te['cov']:.4f} [skipped {te_skip} batches]"
    )

    splits = {
        "email_train_idx": train_idx,
        "email_val_idx": val_idx,
        "email_test_idx": test_idx,
    }
    return model, projector, loaders, splits


__all__ = ["run_vicreg_training", "train_epoch_vicreg", "eval_epoch_vicreg"]
