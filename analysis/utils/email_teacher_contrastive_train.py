"""
Student MLP encoder + supervised contrastive loss + epoch helpers.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from analysis.utils.email_teacher_contrastive_data import (
    CommunityAwareBatchIterator,
    HardNegativeIndexSklearn,
)


class EmailEncoderMLP(nn.Module):
    """MLP email encoder with L2-normalized outputs (legacy checkpoints)."""

    def __init__(
        self,
        input_dim: int,
        *,
        hidden_dim_1: int = 512,
        hidden_dim_2: int = 256,
        output_dim: int = 128,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim_1),
            nn.LayerNorm(hidden_dim_1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim_1, hidden_dim_2),
            nn.LayerNorm(hidden_dim_2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim_2, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x)
        return F.normalize(z, dim=1, eps=1e-12)


class EmailEncoderResidual(nn.Module):
    """
    Residual refinement: L2-normalize(proj(x) + alpha * delta(x)).

    The MLP predicts a **delta** in embedding space; ``alpha`` keeps the correction small
    so the student refines rather than replaces the projected raw features.
    """

    def __init__(
        self,
        input_dim: int,
        *,
        hidden_dim: int = 256,
        output_dim: int = 128,
        dropout: float = 0.1,
        residual_alpha: float = 0.2,
        learnable_alpha: bool = False,
        num_mlp_hidden_layers: int = 1,
        hidden_dim_2: int | None = None,
    ) -> None:
        super().__init__()
        self.proj = nn.Linear(input_dim, output_dim)
        n_h = int(max(1, num_mlp_hidden_layers))
        layers: list[nn.Module] = []
        in_d = input_dim
        for h in range(n_h):
            hdim = int(hidden_dim_2) if (h == 1 and hidden_dim_2 is not None) else int(hidden_dim)
            layers.append(nn.Linear(in_d, hdim))
            layers.append(nn.LayerNorm(hdim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            in_d = hdim
        layers.append(nn.Linear(in_d, output_dim))
        self.mlp = nn.Sequential(*layers)
        if learnable_alpha:
            self.residual_alpha = nn.Parameter(torch.tensor(float(residual_alpha), dtype=torch.float32))
        else:
            self.register_buffer(
                "residual_alpha",
                torch.tensor(float(residual_alpha), dtype=torch.float32),
                persistent=True,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_proj = self.proj(x)
        delta = self.mlp(x)
        z_pre = x_proj + self.residual_alpha * delta
        return F.normalize(z_pre, dim=1, eps=1e-12)


def build_student_from_train_config(cfg: dict[str, Any], input_dim: int) -> nn.Module:
    """
    Instantiate student from ``config_train.json``-style dict.

    ``STUDENT_ARCH``: ``"residual"`` or ``"mlp"``. If missing, ``"mlp"`` is assumed so
    older checkpoints without this key still load.
    """
    arch = str(cfg.get("STUDENT_ARCH", "mlp")).lower()
    if arch == "mlp":
        h2 = cfg.get("HIDDEN_DIM_2_MLP", cfg.get("HIDDEN_DIM_2", 256))
        return EmailEncoderMLP(
            input_dim,
            hidden_dim_1=int(cfg.get("HIDDEN_DIM_1", 512)),
            hidden_dim_2=int(h2),
            output_dim=int(cfg.get("OUTPUT_DIM", 128)),
            dropout=float(cfg.get("DROPOUT", 0.15)),
        )
    h2 = cfg.get("HIDDEN_DIM_2")
    return EmailEncoderResidual(
        input_dim,
        hidden_dim=int(cfg.get("HIDDEN_DIM", 256)),
        output_dim=int(cfg.get("OUTPUT_DIM", 128)),
        dropout=float(cfg.get("DROPOUT", 0.1)),
        residual_alpha=float(cfg.get("RESIDUAL_ALPHA", 0.2)),
        learnable_alpha=bool(cfg.get("RESIDUAL_ALPHA_LEARNABLE", False)),
        num_mlp_hidden_layers=int(cfg.get("NUM_MLP_HIDDEN_LAYERS", 1)),
        hidden_dim_2=int(h2) if h2 is not None else None,
    )


def supervised_contrastive_loss(
    z: torch.Tensor,
    labels: torch.Tensor,
    *,
    temperature: float = 0.07,
) -> torch.Tensor:
    """SupCon-style loss with multiple positives per anchor (same integer label)."""
    device = z.device
    z = F.normalize(z, dim=1, eps=1e-12)
    sim = (z @ z.t()) / temperature
    n = z.size(0)
    if n < 2:
        return torch.zeros((), device=device, requires_grad=True)

    lab = labels.view(-1, 1)
    mask_same = (lab == lab.t()).float().to(device)
    eye = torch.eye(n, device=device)
    logits_mask = 1.0 - eye
    mask_pos = mask_same * logits_mask

    exp_sim = torch.exp(sim) * logits_mask
    denom = exp_sim.sum(dim=1).clamp_min(1e-12)
    num = (torch.exp(sim) * mask_pos).sum(dim=1)
    valid = mask_pos.sum(dim=1) > 0
    if not torch.any(valid):
        return torch.zeros((), device=device, requires_grad=True)
    loss = -torch.log(num.clamp_min(1e-12) / denom)
    return loss[valid].mean()


def batch_cross_shard_positive_stats(
    shards: list[str],
    labels: torch.Tensor,
) -> tuple[float, float]:
    """Fraction of anchors with ≥1 same-label different-shard peer; mean count of such peers."""
    n = len(shards)
    if n < 2:
        return float("nan"), float("nan")
    lab = labels.cpu().numpy()
    sh = np.array([str(s) for s in shards], dtype=object)
    ok = 0
    counts: list[int] = []
    for i in range(n):
        same = lab == lab[i]
        diff_shard = sh != sh[i]
        c = int(np.sum(same & diff_shard & (np.arange(n) != i)))
        counts.append(c)
        if c > 0:
            ok += 1
    return ok / n, float(np.mean(counts))


@torch.no_grad()
def embedding_cosine_separation(z: torch.Tensor, labels: torch.Tensor) -> tuple[float, float]:
    """Mean cosine among same-label pairs vs different-label pairs (off-diagonal)."""
    z = F.normalize(z, dim=1, eps=1e-12)
    cos = z @ z.t()
    n = z.size(0)
    if n < 2:
        return float("nan"), float("nan")
    lab = labels.view(-1, 1)
    same = (lab == lab.t()).float()
    eye = torch.eye(n, device=z.device)
    same_m = same * (1 - eye)
    diff_m = (1 - same) * (1 - eye)
    ns = same_m.sum().clamp_min(1.0)
    nd = diff_m.sum().clamp_min(1.0)
    return float((cos * same_m).sum() / ns), float((cos * diff_m).sum() / nd)


def teacher_cluster_to_labels(cluster_ids: list[str] | np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
    ids = [str(x) for x in cluster_ids]
    uniq = sorted(set(ids))
    m = {u: i for i, u in enumerate(uniq)}
    y = np.array([m[e] for e in ids], dtype=np.int64)
    return y, m


def _infinite_batches(it: CommunityAwareBatchIterator) -> Iterator[list[str]]:
    while True:
        yield next(it)


def _extend_batch_hard_negatives(
    idxs: list[int],
    shards_sel: list[str],
    shard_np: np.ndarray,
    eid_order: list[str],
    eid_to_pos: dict[str, int],
    hard_neg_index: HardNegativeIndexSklearn | None,
    rng: np.random.Generator,
    *,
    use_hard_negatives: bool,
    hard_negative_fraction: float,
    hard_negative_topk: int,
) -> tuple[list[int], list[str], list[tuple[int, int]], int]:
    """
    Append ``different-teacher`` neighbors that are semantically near in **raw** feature space.

    Returns extended index/shard lists, (anchor_local, hard_local) index pairs in the **extended**
    batch, and original batch length.
    """
    n_orig = len(idxs)
    if (
        not use_hard_negatives
        or hard_neg_index is None
        or hard_negative_fraction <= 0.0
        or n_orig < 2
    ):
        return idxs, shards_sel, [], n_orig

    n_add = max(0, int(round(n_orig * float(hard_negative_fraction))))
    if n_add == 0:
        return idxs, shards_sel, [], n_orig

    in_batch = set(idxs)
    out_idxs = list(idxs)
    out_shards = list(shards_sel)
    hard_pairs: list[tuple[int, int]] = []
    g_to_local = {g: i for i, g in enumerate(idxs)}
    attempts = 0
    max_attempts = max(n_add * 20, 50)

    while len(hard_pairs) < n_add and attempts < max_attempts:
        attempts += 1
        a_idx = int(rng.choice(np.array(idxs, dtype=np.int64)))
        eid = str(eid_order[a_idx])
        try:
            hard_eids = hard_neg_index.sample_hard(eid, k=int(hard_negative_topk), rng=rng)
        except KeyError:
            continue
        al = g_to_local.get(int(a_idx))
        if al is None:
            continue
        for he in hard_eids:
            p = eid_to_pos.get(str(he))
            if p is None or p in in_batch:
                continue
            new_local = len(out_idxs)
            out_idxs.append(int(p))
            out_shards.append(str(shard_np[int(p)]))
            in_batch.add(int(p))
            hard_pairs.append((al, new_local))
            g_to_local[int(p)] = new_local
            break

    return out_idxs, out_shards, hard_pairs, n_orig


def _hard_vs_random_cosine(
    z: torch.Tensor,
    yi: torch.Tensor,
    hard_pairs: list[tuple[int, int]],
    n_orig: int,
    rng: np.random.Generator,
    *,
    n_rand_pairs: int = 32,
) -> tuple[float, float]:
    """Mean cosine for (anchor, hard_neg) vs random different-community pairs in the batch head."""
    if n_orig < 2:
        return float("nan"), float("nan")
    lab = yi[:n_orig].detach().cpu().numpy()
    zc = z.detach()
    hard_vals: list[float] = []
    for i, j in hard_pairs:
        if i < n_orig and j < zc.size(0):
            hard_vals.append(float((zc[i] * zc[j]).sum().cpu()))
    mean_hard = float(np.mean(hard_vals)) if hard_vals else float("nan")

    rand_vals: list[float] = []
    tries = 0
    while len(rand_vals) < min(n_rand_pairs, n_orig * (n_orig - 1)) and tries < n_rand_pairs * 8:
        tries += 1
        i = int(rng.integers(0, n_orig))
        j = int(rng.integers(0, n_orig))
        if i == j or lab[i] == lab[j]:
            continue
        rand_vals.append(float((zc[i] * zc[j]).sum().cpu()))
    mean_rand = float(np.mean(rand_vals)) if rand_vals else float("nan")
    return mean_hard, mean_rand


def train_one_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    x: torch.Tensor,
    y: torch.Tensor,
    shard_np: np.ndarray,
    eid_order: list[str],
    batch_it: CommunityAwareBatchIterator,
    *,
    rng_step: np.random.Generator,
    steps: int,
    temperature: float,
    desc: str,
    hard_neg_index: HardNegativeIndexSklearn | None = None,
    use_hard_negatives: bool = False,
    hard_negative_fraction: float = 0.0,
    hard_negative_topk: int = 32,
) -> dict[str, float]:
    model.train()
    eid_to_pos = {str(e): i for i, e in enumerate(eid_order)}
    loss_acc = []
    cross_frac_acc = []
    cross_mean_acc = []
    hard_cos_acc: list[float] = []
    rand_cos_acc: list[float] = []

    gen = _infinite_batches(batch_it)
    pbar = tqdm(range(steps), desc=desc)
    for _ in pbar:
        raw_batch = next(gen)
        idxs: list[int] = []
        shards_sel: list[str] = []
        for e in raw_batch:
            e = str(e)
            p = eid_to_pos.get(e)
            if p is None:
                continue
            idxs.append(p)
            shards_sel.append(str(shard_np[p]))

        if len(idxs) < 2:
            continue
        idxs, shards_sel, hard_pairs, n_orig = _extend_batch_hard_negatives(
            idxs,
            shards_sel,
            shard_np,
            eid_order,
            eid_to_pos,
            hard_neg_index,
            rng_step,
            use_hard_negatives=use_hard_negatives,
            hard_negative_fraction=hard_negative_fraction,
            hard_negative_topk=hard_negative_topk,
        )
        xi = x[idxs].to(device, non_blocking=True)
        yi = y[idxs].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        z = model(xi)
        loss = supervised_contrastive_loss(z, yi, temperature=temperature)
        loss.backward()
        optimizer.step()
        loss_acc.append(float(loss.detach().cpu()))
        cf, cm = batch_cross_shard_positive_stats(shards_sel, yi)
        cross_frac_acc.append(cf)
        cross_mean_acc.append(cm)
        if hard_pairs:
            mh, mr = _hard_vs_random_cosine(z, yi, hard_pairs, n_orig, rng_step)
            if np.isfinite(mh):
                hard_cos_acc.append(mh)
            if np.isfinite(mr):
                rand_cos_acc.append(mr)
        pbar.set_postfix(loss=np.mean(loss_acc[-10:]))

    out: dict[str, float] = {
        "loss": float(np.mean(loss_acc)) if loss_acc else float("nan"),
        "batch_cross_shard_positive_frac": float(np.nanmean(cross_frac_acc))
        if cross_frac_acc
        else float("nan"),
        "batch_cross_shard_pos_mean": float(np.nanmean(cross_mean_acc))
        if cross_mean_acc
        else float("nan"),
    }
    if hard_cos_acc:
        out["train_mean_cos_hard_neg"] = float(np.nanmean(hard_cos_acc))
    if rand_cos_acc:
        out["train_mean_cos_rand_neg"] = float(np.nanmean(rand_cos_acc))
    return out


@torch.no_grad()
def eval_epoch(
    model: nn.Module,
    device: torch.device,
    x: torch.Tensor,
    y: torch.Tensor,
    shard_np: np.ndarray,
    eid_order: list[str],
    batch_it: CommunityAwareBatchIterator,
    *,
    steps: int,
    temperature: float,
    desc: str,
) -> dict[str, float]:
    model.eval()
    eid_to_pos = {str(e): i for i, e in enumerate(eid_order)}
    loss_acc = []
    pos_cos_acc = []
    neg_cos_acc = []
    cross_frac_acc: list[float] = []
    cross_mean_acc: list[float] = []
    gen = _infinite_batches(batch_it)
    pbar = tqdm(range(steps), desc=desc)
    for _ in pbar:
        raw_batch = next(gen)
        idxs: list[int] = []
        shards_sel: list[str] = []
        for e in raw_batch:
            e = str(e)
            p = eid_to_pos.get(e)
            if p is None:
                continue
            idxs.append(p)
            shards_sel.append(str(shard_np[p]))
        if len(idxs) < 2:
            continue
        xi = x[idxs].to(device, non_blocking=True)
        yi = y[idxs].to(device, non_blocking=True)
        z = model(xi)
        loss = supervised_contrastive_loss(z, yi, temperature=temperature)
        loss_acc.append(float(loss.cpu()))
        pc, nc = embedding_cosine_separation(z, yi)
        pos_cos_acc.append(pc)
        neg_cos_acc.append(nc)
        cf, cm = batch_cross_shard_positive_stats(shards_sel, yi)
        cross_frac_acc.append(cf)
        cross_mean_acc.append(cm)
        pbar.set_postfix(loss=np.mean(loss_acc[-10:]))

    return {
        "loss": float(np.mean(loss_acc)) if loss_acc else float("nan"),
        "val_pos_cos_mean": float(np.nanmean(pos_cos_acc)) if pos_cos_acc else float("nan"),
        "val_neg_cos_mean": float(np.nanmean(neg_cos_acc)) if neg_cos_acc else float("nan"),
        "val_batch_cross_shard_positive_frac": float(np.nanmean(cross_frac_acc))
        if cross_frac_acc
        else float("nan"),
        "val_batch_cross_shard_pos_mean": float(np.nanmean(cross_mean_acc))
        if cross_mean_acc
        else float("nan"),
    }


def export_embeddings(
    model: nn.Module,
    device: torch.device,
    x: torch.Tensor,
    *,
    batch_size: int = 2048,
) -> np.ndarray:
    model.eval()
    outs: list[np.ndarray] = []
    n = x.size(0)
    with torch.no_grad():
        for i in range(0, n, batch_size):
            sl = x[i : i + batch_size].to(device, non_blocking=True)
            outs.append(model(sl).cpu().numpy())
    return np.concatenate(outs, axis=0)


def save_training_history_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in keys})


def save_checkpoint(
    path: Path,
    *,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler_state: dict | None,
    meta: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler_state,
            "meta": meta,
        },
        path,
    )
