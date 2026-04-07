"""
VICReg helpers: projector, loss, hetero batch augmentation, anchor extraction, email NeighborLoaders.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader


def vicreg_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
    w_inv: float = 25.0,
    w_var: float = 25.0,
    w_cov: float = 1.0,
    *,
    std_floor: float = 1.0,
    eps: float = 1e-4,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    VICReg loss on two projected batches [B, D].
    Returns (total_loss, dict of detached scalars for logging).
    """
    if z1.shape != z2.shape:
        raise ValueError(f"z1/z2 shape mismatch: {z1.shape} vs {z2.shape}")
    if z1.dim() != 2:
        raise ValueError(f"Expected z1 2D [B,D], got {z1.shape}")

    inv = F.mse_loss(z1, z2)

    def var_term(z: torch.Tensor) -> torch.Tensor:
        std = torch.sqrt(z.var(dim=0, unbiased=True) + eps)
        return torch.mean(F.relu(std_floor - std))

    var = var_term(z1) + var_term(z2)

    def cov_term(z: torch.Tensor) -> torch.Tensor:
        n, d = z.shape
        if n <= 1:
            return z.new_zeros(())
        zc = z - z.mean(dim=0, keepdim=True)
        cov = (zc.T @ zc) / (n - 1)
        off_diag = cov - torch.diag(torch.diag(cov))
        return (off_diag ** 2).sum() / d

    cov = cov_term(z1) + cov_term(z2)

    total = w_inv * inv + w_var * var + w_cov * cov
    metrics = {
        "inv": inv.detach(),
        "var": var.detach(),
        "cov": cov.detach(),
        "total": total.detach(),
    }
    return total, metrics


class VicRegProjector(nn.Module):
    """MLP projector for VICReg only; encoder embeddings stay the representation."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _edge_drop_prob(et: Tuple[str, str, str], edge_drop_probs: Optional[Dict[str, float]]) -> float:
    if not edge_drop_probs:
        return 0.05
    key = "__".join(et)
    if key in edge_drop_probs:
        return float(edge_drop_probs[key])
    return float(edge_drop_probs.get("default", 0.05))


def augment_hetero_batch(
    batch: HeteroData,
    feat_mask_prob: float,
    edge_drop_probs: Optional[Dict[str, float]] = None,
    *,
    generator: Optional[torch.Generator] = None,
    email_full_zero_prob: float = 0.0,
    email_semantic_mask_prob: float = 0.0,
    email_semantic_mask_mode: str = "none",
    email_semantic_block: Optional[Tuple[int, int]] = None,
    primary_ntype: str = "email",
) -> HeteroData:
    """
    Non-in-place augmentation: edge dropout + feature masking.
    Preserves node ordering and n_id / batch_size / input_id on node stores.
    """
    out = batch.clone()

    mode = str(email_semantic_mask_mode).lower().strip()
    for store_key in out.node_types:
        if not hasattr(out[store_key], "x") or out[store_key].x is None:
            continue
        x = out[store_key].x
        if not (feat_mask_prob <= 0 or x.numel() == 0):
            prob = torch.full(
                x.shape, 1.0 - feat_mask_prob, device=x.device, dtype=torch.float32
            )
            if generator is not None:
                mask = torch.bernoulli(prob, generator=generator)
            else:
                mask = torch.bernoulli(prob)
            out[store_key].x = x * mask.to(dtype=x.dtype)

        # Optional full-row email feature dropout:
        # sometimes remove semantic signal entirely so structure/artifacts must carry the view.
        if (
            store_key == primary_ntype
            and float(email_full_zero_prob) > 0.0
            and out[store_key].x is not None
            and out[store_key].x.numel() > 0
        ):
            pz = float(email_full_zero_prob)
            keep_prob = torch.full(
                (out[store_key].x.size(0),), 1.0 - pz, device=out[store_key].x.device, dtype=torch.float32
            )
            if generator is not None:
                keep_rows = torch.bernoulli(keep_prob, generator=generator).view(-1, 1)
            else:
                keep_rows = torch.bernoulli(keep_prob).view(-1, 1)
            out[store_key].x = out[store_key].x * keep_rows.to(dtype=out[store_key].x.dtype)

        # Optional VICReg-only semantic block weakening for email features.
        if (
            store_key == primary_ntype
            and mode != "none"
            and float(email_semantic_mask_prob) > 0.0
        ):
            if email_semantic_block is None:
                raise ValueError(
                    "email_semantic_mask requested but email_semantic_block is None."
                )
            start, end = int(email_semantic_block[0]), int(email_semantic_block[1])
            if not (0 <= start < end <= x.size(1)):
                raise ValueError(
                    f"Invalid email_semantic_block [{start}, {end}) for email feature dim {x.size(1)}."
                )
            p = float(email_semantic_mask_prob)
            blk = out[store_key].x[:, start:end]
            if blk.numel() == 0:
                continue
            if mode == "block_zero":
                keep_prob = torch.full(
                    (blk.size(0),), 1.0 - p, device=blk.device, dtype=torch.float32
                )
                if generator is not None:
                    keep_rows = torch.bernoulli(keep_prob, generator=generator).view(-1, 1)
                else:
                    keep_rows = torch.bernoulli(keep_prob).view(-1, 1)
                out[store_key].x[:, start:end] = blk * keep_rows.to(dtype=blk.dtype)
            elif mode == "feature_mask":
                keep_prob = torch.full(
                    blk.shape, 1.0 - p, device=blk.device, dtype=torch.float32
                )
                if generator is not None:
                    keep = torch.bernoulli(keep_prob, generator=generator)
                else:
                    keep = torch.bernoulli(keep_prob)
                out[store_key].x[:, start:end] = blk * keep.to(dtype=blk.dtype)
            else:
                raise ValueError(
                    f"Unknown email_semantic_mask_mode={email_semantic_mask_mode!r}; "
                    "expected one of: 'none', 'block_zero', 'feature_mask'."
                )

    for et in out.edge_types:
        ei = out[et].edge_index
        if ei.numel() == 0:
            continue
        p = _edge_drop_prob(et, edge_drop_probs)
        if p <= 0:
            continue
        e = ei.size(1)
        if generator is not None:
            keep = torch.bernoulli(
                torch.full((e,), 1.0 - p, device=ei.device, dtype=torch.float32),
                generator=generator,
            ).bool()
        else:
            keep = torch.rand(e, device=ei.device) > p
        if keep.sum() == 0 and e > 0:
            keep[torch.randint(0, e, (1,), device=ei.device)] = True
        out[et].edge_index = ei[:, keep]

    return out


def _batch_size_on_store(store: Any) -> int:
    bs = getattr(store, "batch_size", None)
    if bs is None:
        raise ValueError(
            "Missing batch_size on email node store; expected NeighborLoader hetero batch."
        )
    if isinstance(bs, torch.Tensor):
        return int(bs.item())
    return int(bs)


def _anchor_email_row_indices_cpu(
    email_store: Any,
    *,
    input_id: Optional[torch.Tensor] = None,
    email_loader_input_nodes: Optional[torch.Tensor] = None,
    debug_anchor_matching: bool = False,
    debug_tag: str = "",
) -> torch.Tensor:
    """
    Map each loader seed to its row index in ``email_store.n_id`` / encoder email matrix.
    Requires ``n_id`` and ``input_id`` on the store (split-aware matching).
    """
    def _pv(t: Optional[torch.Tensor], max_vals: int = 20) -> str:
        if t is None:
            return "<absent>"
        if not isinstance(t, torch.Tensor):
            return f"<non-tensor: {type(t).__name__}>"
        flat = t.view(-1)
        preview = flat[:max_vals].detach().cpu().tolist()
        return f"dtype={t.dtype} shape={tuple(t.shape)} first{min(max_vals, flat.numel())}={preview}"

    tag = f"{':'+debug_tag}" if debug_tag else ""

    bs = _batch_size_on_store(email_store)
    if bs < 2:
        raise ValueError(f"Need at least 2 anchor emails in batch; got batch_size={bs}")
    nid = getattr(email_store, "n_id", None)
    iid = input_id if input_id is not None else getattr(email_store, "input_id", None)
    if iid is None or nid is None:
        raise ValueError(
            "_anchor_email_row_indices_cpu requires both n_id and input_id on the email store."
        )

    if int(iid.numel()) != bs:
        raise ValueError(
            f"input_id length ({iid.numel()}) must equal email batch_size ({bs})."
        )
    if nid.numel() < bs:
        raise ValueError(
            f"email n_id length ({nid.numel()}) < batch_size ({bs}); cannot match seeds."
        )

    nid_cpu = nid.view(-1).detach().cpu()
    iid_cpu = iid.view(-1).detach().cpu().long()

    if email_loader_input_nodes is not None:
        split_cpu = email_loader_input_nodes.detach().cpu().view(-1)
        n_split = int(split_cpu.numel())
        if torch.any(iid_cpu < 0) or torch.any(iid_cpu >= n_split):
            raise ValueError(
                f"input_id out of range for email_loader_input_nodes (len={n_split}); "
                f"min input_id={int(iid_cpu.min())} max={int(iid_cpu.max())}"
            )
        seed_global_cpu = split_cpu[iid_cpu]
        if debug_anchor_matching:
            print(
                f"[_anchor_email_row_indices_cpu{tag}] namespace=split_input_nodes\n"
                f"  email_loader_input_nodes shape={tuple(split_cpu.shape)} "
                f"first20_split={split_cpu[:20].tolist()}\n"
                f"  raw input_id(first20)={iid_cpu[:20].tolist()}\n"
                f"  seed_global_ids = split[input_id] (first20)={seed_global_cpu[:20].tolist()}"
            )
        matches = nid_cpu[:, None] == seed_global_cpu[None, :].to(dtype=nid_cpu.dtype)
        counts = matches.sum(dim=0)
        if debug_anchor_matching:
            print(
                f"[_anchor_email_row_indices_cpu{tag}] match_counts_per_anchor (first20)="
                f"{counts[:20].tolist()}"
            )
        if not torch.all(counts == 1):
            bad = (counts != 1).nonzero(as_tuple=False).view(-1)
            bad_list = bad[:10].tolist()
            raise ValueError(
                "Anchor row resolution via split_idx[input_id] failed: each seed global id must "
                f"appear exactly once in n_id. Bad anchor positions (first 10): {bad_list}, "
                f"counts (first 10): {counts[:10].tolist()}"
            )
        anchor_row_idxs_cpu = matches.to(dtype=torch.long).argmax(dim=0)
        if debug_anchor_matching:
            pos_preview = anchor_row_idxs_cpu[:20].tolist()
            print(
                f"[_anchor_email_row_indices_cpu{tag}] decision=split_then_match_n_id\n"
                f"  matched_row_indices(first20)={pos_preview}\n"
                f"  n_id[row](first20)={nid_cpu[anchor_row_idxs_cpu[:20]].tolist()}"
            )
    else:
        matches = nid_cpu[:, None] == iid_cpu.to(dtype=nid_cpu.dtype)[None, :]
        counts = matches.sum(dim=0)
        if not torch.all(counts == 1):
            raise ValueError(
                "Anchor rows: pass email_loader_input_nodes (train/val/test idx tensor) "
                "when NeighborLoader input_id indexes the split, or ensure input_id values are "
                "global node ids present exactly once in n_id. "
                f"counts (first 10): {counts[:10].tolist()}"
            )
        anchor_row_idxs_cpu = matches.to(dtype=torch.long).argmax(dim=0)
        if debug_anchor_matching:
            print(
                f"[_anchor_email_row_indices_cpu{tag}] decision=global_ids_no_split_tensor\n"
                f"  local_row_idxs(first20)={anchor_row_idxs_cpu[:20].tolist()}"
            )

    return anchor_row_idxs_cpu


def extract_anchor_global_email_ids(
    email_store: Any,
    *,
    input_id: Optional[torch.Tensor] = None,
    email_loader_input_nodes: Optional[torch.Tensor] = None,
) -> torch.LongTensor:
    """
    Global graph email node id per anchor seed ``[B]`` (CPU int64), aligned with
    :func:`extract_anchor_email_embeddings` on the same batch store.
    """
    bs = _batch_size_on_store(email_store)
    if bs < 2:
        raise ValueError(f"Need at least 2 anchor emails in batch; got batch_size={bs}")
    nid = getattr(email_store, "n_id", None)
    iid = input_id if input_id is not None else getattr(email_store, "input_id", None)

    if iid is None or nid is None:
        if iid is None:
            raise ValueError(
                "extract_anchor_global_email_ids cannot infer globals without input_id; "
                "expected NeighborLoader email store with input_id (and n_id for split loaders)."
            )
        return iid.view(-1).detach().cpu().long()[:bs]

    row_cpu = _anchor_email_row_indices_cpu(
        email_store,
        input_id=iid,
        email_loader_input_nodes=email_loader_input_nodes,
        debug_anchor_matching=False,
        debug_tag="",
    )
    nid_cpu = nid.view(-1).detach().cpu()
    return nid_cpu[row_cpu].long()


def extract_anchor_email_embeddings(
    h_email: torch.Tensor,
    email_store: Any,
    *,
    input_id: Optional[torch.Tensor] = None,
    email_loader_input_nodes: Optional[torch.Tensor] = None,
    debug_anchor_matching: bool = False,
    debug_tag: str = "",
    return_anchor_row_indices: bool = False,
) -> torch.Tensor:
    """
    Extract encoder outputs for seed (anchor) email nodes only.

    For ``NeighborLoader(..., input_nodes=(email_type, split_idx))``, PyG typically sets
    ``email_store.input_id`` to **indices into ``split_idx``** (batch-local positions),
    **not** global graph node ids. Global ids for those seeds are
    ``split_idx[input_id]``. The subgraph stores global ids in ``email_store.n_id``.

    Pass ``email_loader_input_nodes`` as the same 1D tensor passed to the loader
    (e.g. ``train_idx``, ``val_idx``, ``test_idx``). Anchor rows are found by matching
    ``split_idx[input_id]`` against ``n_id``.

    If ``email_loader_input_nodes`` is omitted, ``input_id`` is interpreted as **global**
    node ids (legacy / rare). No "local embedding row index" fallback is used.
    """
    def _pv(t: Optional[torch.Tensor], max_vals: int = 20) -> str:
        if t is None:
            return "<absent>"
        if not isinstance(t, torch.Tensor):
            return f"<non-tensor: {type(t).__name__}>"
        flat = t.view(-1)
        preview = flat[:max_vals].detach().cpu().tolist()
        return f"dtype={t.dtype} shape={tuple(t.shape)} first{min(max_vals, flat.numel())}={preview}"

    tag = f"{':'+debug_tag}" if debug_tag else ""

    bs = _batch_size_on_store(email_store)
    if bs < 2:
        raise ValueError(f"Need at least 2 anchor emails in batch; got batch_size={bs}")
    nid = getattr(email_store, "n_id", None)
    iid = input_id if input_id is not None else getattr(email_store, "input_id", None)

    if iid is None or nid is None:
        if h_email.size(0) < bs:
            raise ValueError(
                f"email embedding rows ({h_email.size(0)}) < anchor batch_size ({bs})."
            )
        anchor_row_idxs_cpu = torch.arange(bs, dtype=torch.long, device=torch.device("cpu"))
        if debug_anchor_matching:
            print(
                f"[extract_anchor_email_embeddings{tag}] "
                f"decision=legacy_first_bs (input_id and/or n_id missing)\n"
                f"  h_email_rows={tuple(h_email.shape)} bs={bs}"
            )
        anchor_row_idxs = anchor_row_idxs_cpu.to(device=h_email.device)
        anchors = h_email[anchor_row_idxs].clone()
        if return_anchor_row_indices:
            return anchors, anchor_row_idxs
        return anchors

    if int(iid.numel()) != bs:
        raise ValueError(
            f"input_id length ({iid.numel()}) must equal email batch_size ({bs})."
        )
    if nid.numel() < bs:
        raise ValueError(
            f"email n_id length ({nid.numel()}) < batch_size ({bs}); cannot match seeds."
        )
    if h_email.size(0) != nid.numel():
        raise ValueError(
            f"email embedding rows ({h_email.size(0)}) != email_store.n_id.numel ({nid.numel()}); "
            "cannot map embeddings back to seed ids."
        )

    anchor_row_idxs_cpu = _anchor_email_row_indices_cpu(
        email_store,
        input_id=iid,
        email_loader_input_nodes=email_loader_input_nodes,
        debug_anchor_matching=debug_anchor_matching,
        debug_tag=debug_tag.replace(":", "") if debug_tag else "",
    )

    anchor_row_idxs = anchor_row_idxs_cpu.to(device=h_email.device)
    anchors = h_email[anchor_row_idxs].clone()
    if return_anchor_row_indices:
        return anchors, anchor_row_idxs
    return anchors


def split_email_node_indices(
    num_nodes: int,
    seed: int,
    val_ratio: float,
    test_ratio: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Random permutation split into train / val / test index tensors (global email indices)."""
    if num_nodes < 3:
        raise ValueError(f"Need at least 3 email nodes for split; got {num_nodes}")
    g = torch.Generator()
    g.manual_seed(int(seed))
    perm = torch.randperm(num_nodes, generator=g)
    n_test = max(1, int(round(test_ratio * num_nodes)))
    n_val = max(1, int(round(val_ratio * num_nodes)))
    # VICReg skips batches with fewer than 2 seed emails; a val/test split of size 1
    # would always yield skipped eval. When the graph is large enough, reserve at least
    # 2 nodes for val and test so NeighborLoader batches are usable.
    if num_nodes >= 6:
        n_val = max(n_val, 2)
        n_test = max(n_test, 2)
    if n_val + n_test >= num_nodes:
        n_val = max(1, num_nodes // 3)
        n_test = max(1, num_nodes // 3)
        while n_val + n_test >= num_nodes and n_test > 1:
            n_test -= 1
        while n_val + n_test >= num_nodes and n_val > 1:
            n_val -= 1
    test_idx = perm[:n_test]
    val_idx = perm[n_test : n_test + n_val]
    train_idx = perm[n_test + n_val :]
    if train_idx.numel() < 2:
        raise ValueError(
            "Train split has too few email nodes; reduce val_ratio/test_ratio or use a larger graph."
        )
    return train_idx, val_idx, test_idx


def _fanout_per_hop(fanout: Any, num_hops: int) -> List[int]:
    if not isinstance(fanout, (list, tuple)) or len(fanout) == 0:
        return [10] * num_hops
    f = [int(x) for x in fanout]
    if len(f) < num_hops:
        f = f + [f[-1]] * (num_hops - len(f))
    return f[:num_hops]


def make_email_anchor_loaders(
    data: HeteroData,
    primary_ntype: str,
    train_idx: torch.Tensor,
    val_idx: torch.Tensor,
    test_idx: torch.Tensor,
    fanout: Any,
    num_gnn_layers: int,
    anchor_batch_size: int,
    *,
    num_workers: int = 0,
) -> Dict[str, NeighborLoader]:
    """NeighborLoader over full graph, seeding on email node indices per split."""
    if primary_ntype not in data.node_types:
        raise ValueError(
            f"primary_ntype {primary_ntype!r} not in graph node types {data.node_types}"
        )
    hops = _fanout_per_hop(fanout, int(num_gnn_layers))
    num_neighbors = {et: hops for et in data.edge_types}

    data_cpu = data.cpu()
    loaders: Dict[str, NeighborLoader] = {}
    for name, idx, shuffle in (
        ("train", train_idx, True),
        ("val", val_idx, False),
        ("test", test_idx, False),
    ):
        # drop_last=True on train drops the only batch when len(train_idx) < anchor_batch_size,
        # which yields zero training steps. Partial batches still satisfy bs>=2 when splits are sized correctly.
        loaders[name] = NeighborLoader(
            data_cpu,
            num_neighbors=num_neighbors,
            batch_size=int(anchor_batch_size),
            input_nodes=(primary_ntype, idx),
            shuffle=shuffle,
            num_workers=num_workers,
            drop_last=False,
        )
    return loaders


__all__ = [
    "VicRegProjector",
    "augment_hetero_batch",
    "extract_anchor_email_embeddings",
    "extract_anchor_global_email_ids",
    "make_email_anchor_loaders",
    "split_email_node_indices",
    "vicreg_loss",
]
