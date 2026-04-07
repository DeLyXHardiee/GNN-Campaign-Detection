"""
Homogeneous shard-graph **VICReg-style** node representation learning + HDBSCAN eval.

Two augmented views, skip-GraphSAGE encoders, VICReg objective (invariance / variance / covariance).
Used by ``analysis/shard_graph_homogeneous_gnn_experiment.ipynb``.
"""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv

from analysis.utils.semantic_shard_step3_helpers import evaluate_external_metrics, map_shards_to_email_predictions


def _parse_member_ids(raw: Any) -> list[str]:
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return []
    s = str(raw).strip()
    if not s or s.lower() in ("nan", "none", "[]"):
        return []
    try:
        v = ast.literal_eval(s)
        if isinstance(v, (list, tuple)):
            return [str(x).strip() for x in v if str(x).strip()]
    except (ValueError, SyntaxError, TypeError):
        pass
    return []


def build_shard_node_features(
    nodes_df: pd.DataFrame,
    id_to_embedding: dict[str, np.ndarray],
    *,
    include_log_size: bool = False,
) -> tuple[np.ndarray, list[str], dict[str, int]]:
    rows: list[np.ndarray] = []
    shard_ids: list[str] = []
    sizes: list[float] = []

    for _, r in nodes_df.iterrows():
        sid = str(r["shard_id"])
        members = _parse_member_ids(r.get("member_external_ids"))
        vecs = [
            np.asarray(id_to_embedding[e], dtype=np.float32)
            for e in members
            if e in id_to_embedding
        ]
        if not vecs:
            continue
        emb = np.mean(np.stack(vecs, axis=0), axis=0)
        rows.append(emb)
        shard_ids.append(sid)
        sizes.append(float(len(members)))

    if not rows:
        raise ValueError("No shards with any cached member embedding.")

    X = np.stack(rows, axis=0).astype(np.float32)
    nrm = np.linalg.norm(X, axis=1, keepdims=True)
    nrm[nrm == 0] = 1.0
    X = X / nrm

    if include_log_size:
        s = np.log1p(np.asarray(sizes, dtype=np.float32)).reshape(-1, 1)
        s = StandardScaler().fit_transform(s).astype(np.float32)
        X = np.concatenate([X, s], axis=1)

    idx_map = {s: i for i, s in enumerate(shard_ids)}
    return X, shard_ids, idx_map


def build_homogeneous_shard_data(
    shard_ids: list[str],
    idx_map: dict[str, int],
    edges_df: pd.DataFrame,
    x: np.ndarray,
    *,
    weight_col: str = "edge_weight",
    min_edge_weight: float = 0.0,
) -> Data:
    src: list[int] = []
    dst: list[int] = []
    ed = edges_df
    if not ed.empty and weight_col in ed.columns:
        ed = ed[ed[weight_col] >= float(min_edge_weight)]
    if not ed.empty:
        for _, r in ed.iterrows():
            a, b = str(r["shard_a"]), str(r["shard_b"])
            if a not in idx_map or b not in idx_map:
                continue
            ia, ib = idx_map[a], idx_map[b]
            if ia == ib:
                continue
            src.extend([ia, ib])
            dst.extend([ib, ia])

    edge_index = torch.empty((2, 0), dtype=torch.long)
    if src:
        edge_index = torch.tensor([src, dst], dtype=torch.long)

    data = Data(
        x=torch.tensor(x, dtype=torch.float32),
        edge_index=edge_index,
    )
    data.num_nodes = int(x.shape[0])
    return data


def edge_dropout(
    edge_index: torch.Tensor,
    drop_prob: float,
    rng: np.random.Generator,
) -> torch.Tensor:
    """Undirected-aware: drop each unordered edge (and both directions) i.i.d."""
    if edge_index.numel() == 0 or drop_prob <= 0:
        return edge_index.clone()
    e = edge_index.detach().cpu().numpy()
    undir: set[tuple[int, int]] = set()
    for i in range(e.shape[1]):
        u, v = int(e[0, i]), int(e[1, i])
        if u == v:
            continue
        a, b = (u, v) if u < v else (v, u)
        undir.add((a, b))
    pairs = list(undir)
    if not pairs:
        return edge_index.clone()
    mask = rng.random(len(pairs)) > float(drop_prob)
    src: list[int] = []
    dst: list[int] = []
    for (u, v), k in zip(pairs, mask):
        if k:
            src.extend([u, v])
            dst.extend([v, u])
    if not src:
        return edge_index.clone()
    return torch.tensor([src, dst], dtype=torch.long, device=edge_index.device)


def feature_dropout_mask(
    x: torch.Tensor,
    drop_prob: float,
    rng: np.random.Generator,
) -> torch.Tensor:
    """Element-wise Bernoulli mask, expectation preserved."""
    if drop_prob <= 0:
        return x
    m = torch.as_tensor(
        rng.binomial(1, 1.0 - float(drop_prob), size=x.shape),
        dtype=x.dtype,
        device=x.device,
    )
    scale = 1.0 / max(1.0 - float(drop_prob), 1e-6)
    return x * m * scale


def augment_graph_view(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    rng: np.random.Generator,
    *,
    edge_drop_prob: float,
    feat_drop_prob: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    ei = edge_dropout(edge_index, edge_drop_prob, rng)
    xa = feature_dropout_mask(x.clone(), feat_drop_prob, rng)
    return xa, ei


@dataclass
class HomogeneousShardGNNConfig:
    run_id: str
    step2_dir: str
    step1_assignments_csv: str
    embeddings_json: str
    ground_truth_json: str
    include_log_size: bool = False
    skip_alpha: float = 0.5
    seed: int = 42
    hidden_dim: int = 128
    out_dim: int = 64
    n_layers: int = 2
    dropout: float = 0.2
    lr: float = 1e-3
    weight_decay: float = 3e-4
    max_epochs: int = 40
    scheduler_patience: int = 5
    early_stopping_patience: int = 10
    save_checkpoint_every_epoch: bool = True
    save_embeddings_every_epoch: bool = False
    hdbscan_min_cluster_size: int = 5
    hdbscan_min_samples: int | None = None
    eval_every_epochs: int = 1
    min_edge_weight: float = 0.0
    # Augmentations (light)
    edge_dropout_prob: float = 0.08
    feature_dropout_prob: float = 0.1
    # VICReg weights (Bardes et al. use λ≈25, μ≈25, ν≈1; defaults are a reasonable starting point)
    vicreg_lambda: float = 25.0
    vicreg_mu: float = 25.0
    vicreg_nu: float = 1.0
    # Std floor per dimension in the variance term (mean ReLU(γ - std)^2).
    vicreg_gamma: float = 1.0
    # If True, L2-normalize rows before HDBSCAN (optional comparison; default raw z_pre)
    cluster_l2_normalize: bool = False


class HomogeneousGraphSAGE(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        *,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        if n_layers < 2:
            raise ValueError("n_layers must be >= 2")
        self.dropout = float(dropout)
        self.n_layers = int(n_layers)
        dims = [in_dim] + [hidden_dim] * (n_layers - 1) + [out_dim]
        self.convs = nn.ModuleList(
            SAGEConv(dims[i], dims[i + 1]) for i in range(n_layers)
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = x
        for li, conv in enumerate(self.convs):
            h = conv(h, edge_index)
            if li < len(self.convs) - 1:
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
        return h


class SkipGraphSAGEModel(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        *,
        n_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.gnn = HomogeneousGraphSAGE(
            in_dim, hidden_dim, out_dim, n_layers=n_layers, dropout=dropout
        )
        self.input_proj = nn.Linear(in_dim, out_dim)

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, *, skip_alpha: float
    ) -> torch.Tensor:
        h = self.gnn(x, edge_index)
        xp = self.input_proj(x)
        a = float(skip_alpha)
        return a * h + (1.0 - a) * xp


def vicreg_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
    *,
    lam: float,
    mu: float,
    nu: float,
    gamma: float,
    eps: float = 1e-4,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    VICReg between two [N, D] matrices (no L2 normalization applied to z1/z2).
    Returns (weighted total, dict of unweighted component scalars for logging).
    """
    inv = F.mse_loss(z1, z2)

    def variance_term(z: torch.Tensor) -> torch.Tensor:
        # Per-dim std along nodes (population)
        std = torch.sqrt(z.var(dim=0, unbiased=False) + eps)
        return torch.mean(F.relu(float(gamma) - std) ** 2)

    v = variance_term(z1) + variance_term(z2)

    def covariance_term(z: torch.Tensor) -> torch.Tensor:
        n = z.size(0)
        if n <= 1:
            return z.new_zeros(())
        zc = z - z.mean(dim=0, keepdim=True)
        c = (zc.T @ zc) / (n - 1)
        d = int(z.size(1))
        off = c - torch.diag(torch.diag(c))
        return (off**2).sum() / float(d)

    c = covariance_term(z1) + covariance_term(z2)
    total = float(lam) * inv + float(mu) * v + float(nu) * c
    parts = {"inv": inv.detach(), "var": v.detach(), "cov": c.detach()}
    return total, parts


def embedding_collapse_mean_std(z_np: np.ndarray) -> float:
    """Mean per-dimension std across nodes (higher => less collapse)."""
    if z_np.size == 0:
        return float("nan")
    return float(np.mean(np.std(z_np.astype(np.float64), axis=0)))


def embedding_offdiag_cov_frobenius_per_dim(z_np: np.ndarray) -> float:
    """VICReg-style off-diagonal covariance energy: ||offdiag(C)||_F^2 / D on centered z."""
    if z_np.size == 0 or z_np.shape[0] <= 1:
        return float("nan")
    z = z_np.astype(np.float64)
    z = z - z.mean(axis=0, keepdims=True)
    n = z.shape[0]
    c = (z.T @ z) / (n - 1)
    d = z.shape[1]
    off = c - np.diag(np.diag(c))
    return float(np.sum(off**2) / max(d, 1))


def embedding_random_pair_mean_cosine(
    z_np: np.ndarray, rng: np.random.Generator, n_pairs: int = 4096
) -> float:
    """Mean cosine of random distinct node pairs (diagnostic for cone collapse)."""
    n = z_np.shape[0]
    if n < 2:
        return float("nan")
    z = z_np.astype(np.float64)
    norms = np.linalg.norm(z, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    zn = z / norms
    m = min(int(n_pairs), n * (n - 1) // 2)
    if m <= 0:
        return float("nan")
    cos_vals: list[float] = []
    for _ in range(m):
        i, j = rng.integers(0, n, size=2)
        if i == j:
            j = (j + 1) % n
            if j == i:
                j = (j + 1) % n
        cos_vals.append(float(np.dot(zn[i], zn[j])))
    return float(np.mean(cos_vals))


@torch.no_grad()
def shard_embeddings_to_email_metrics(
    shard_ids: list[str],
    z_np: np.ndarray,
    assignments_df: pd.DataFrame,
    gt_label_map: dict[str, Any],
    *,
    min_cluster_size: int,
    min_samples: int | None,
    l2_normalize_rows: bool = False,
) -> dict[str, Any]:
    z_fit = z_np.astype(np.float64)
    if l2_normalize_rows:
        nrm = np.linalg.norm(z_fit, axis=1, keepdims=True)
        nrm[nrm == 0] = 1.0
        z_fit = z_fit / nrm
    try:
        import hdbscan  # type: ignore

        cl = hdbscan.HDBSCAN(
            min_cluster_size=int(min_cluster_size),
            min_samples=None if min_samples is None else int(min_samples),
            metric="euclidean",
        )
        lab = cl.fit_predict(z_fit)
    except Exception as e:
        return {
            "error": str(e),
            "homogeneity": float("nan"),
            "completeness": float("nan"),
            "v_measure": float("nan"),
            "n_eval": 0.0,
            "coverage_gt": float("nan"),
            "coverage_assignments": float("nan"),
            "n_clusters": 0,
            "n_noise_shards": 0,
        }

    shard_to_cluster = {shard_ids[i]: int(lab[i]) for i in range(len(shard_ids))}
    email_df = map_shards_to_email_predictions(assignments_df, shard_to_cluster)
    m = evaluate_external_metrics(email_df, gt_label_map)
    uniq = set(lab.tolist())
    n_noise = int(np.sum(lab == -1))
    n_meaningful = int(len(uniq - {-1}))
    return {
        **m,
        "n_clusters": float(n_meaningful),
        "n_noise_shards": float(n_noise),
        "n_shard_nodes": float(len(shard_ids)),
    }


def save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
    epoch: int,
    metrics: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": int(epoch),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "metrics": metrics,
        },
        path,
    )


def load_ground_truth_map(path: str | Path) -> dict[str, Any]:
    from analysis.utils.raw_gnn_notebook import load_ground_truth_structures

    gt, _, _ = load_ground_truth_structures(Path(path))
    return {str(k): v for k, v in gt.items()}


def train_homogeneous_shard_gnn(
    cfg: HomogeneousShardGNNConfig,
    data: Data,
    shard_ids: list[str],
    assignments_df: pd.DataFrame,
    gt_label_map: dict[str, Any],
    out_dir: Path,
    device: torch.device,
    *,
    tqdm_epochs: Any | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    **VICReg** on two augmented views. No L2 normalization on encoder outputs inside the loss.
    Scheduler + early stopping on **val_loss** (same VICReg with fixed val RNG).
    Tracks **best clustering** (max V-measure) separately → ``embeddings_clustering_best.npy``.
    ``embeddings_best.npy`` / ``best.pt`` follow **lowest val_loss**.

    Saved embeddings are **raw** skip-blend outputs unless ``cfg.cluster_l2_normalize`` is True
    only affects HDBSCAN inside ``shard_embeddings_to_email_metrics`` during training eval
    (saved arrays remain raw).
    """
    out_dir = Path(out_dir).expanduser().resolve()
    chk = out_dir / "checkpoints"
    emb_dir = out_dir / "embeddings"
    chk.mkdir(parents=True, exist_ok=True)
    if cfg.save_embeddings_every_epoch:
        emb_dir.mkdir(parents=True, exist_ok=True)

    in_dim = int(data.x.shape[1])
    model = SkipGraphSAGEModel(
        in_dim,
        cfg.hidden_dim,
        cfg.out_dim,
        n_layers=int(cfg.n_layers),
        dropout=cfg.dropout,
    ).to(device)

    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    n_nodes = int(data.num_nodes)
    skip_a = float(cfg.skip_alpha)
    edp = float(cfg.edge_dropout_prob)
    fdp = float(cfg.feature_dropout_prob)
    vlam = float(cfg.vicreg_lambda)
    vmu = float(cfg.vicreg_mu)
    vnu = float(cfg.vicreg_nu)
    vgamma = float(cfg.vicreg_gamma)
    clust_l2 = bool(cfg.cluster_l2_normalize)

    n_dir_e = int(edge_index.shape[1])
    n_undir = n_dir_e // 2
    print(
        f"[VICReg shard] n_nodes={n_nodes}, directed_edges={n_dir_e} (~{n_undir} undirected), "
        f"edge_drop={edp}, feat_drop={fdp}, λ={vlam} μ={vmu} ν={vnu} γ={vgamma}"
    )

    if n_dir_e == 0:
        summary = {
            "best_loss_epoch": -1,
            "best_val_loss": float("inf"),
            "best_clustering_epoch": -1,
            "best_v_measure": float("nan"),
            "final_epoch": 0,
            "stop_reason": "no_edges",
            "stall_epochs_without_val_loss_improvement": 0,
        }
        with open(out_dir / "training_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        return [], summary

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(cfg.lr),
        weight_decay=float(cfg.weight_decay),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=int(cfg.scheduler_patience),
        threshold=1e-5,
    )

    rng_master = np.random.default_rng(int(cfg.seed))
    torch.manual_seed(int(cfg.seed))

    history: list[dict[str, Any]] = []
    best_val_loss = float("inf")
    best_loss_epoch = -1
    stall = 0
    best_vm = float("nan")
    best_cluster_epoch = -1

    assignments_df = assignments_df.copy()
    assignments_df["external_id"] = assignments_df["external_id"].astype(str)
    assignments_df["shard_id"] = assignments_df["shard_id"].astype(str)

    epoch_iter = range(1, int(cfg.max_epochs) + 1)
    if tqdm_epochs is None:
        try:
            from tqdm.auto import tqdm as _tqdm  # type: ignore

            epoch_iter = _tqdm(epoch_iter, desc="epochs")
        except Exception:
            pass
    else:
        epoch_iter = tqdm_epochs(epoch_iter, desc="epochs")

    stop_reason = "max_epochs"
    pbar = epoch_iter if hasattr(epoch_iter, "set_postfix") else None

    for epoch in epoch_iter:
        model.train()
        rng_t = np.random.default_rng(rng_master.integers(0, 2**31))
        rng_v = np.random.default_rng(rng_master.integers(0, 2**31))
        x1, e1 = augment_graph_view(x, edge_index, rng_t, edge_drop_prob=edp, feat_drop_prob=fdp)
        x2, e2 = augment_graph_view(x, edge_index, rng_v, edge_drop_prob=edp, feat_drop_prob=fdp)

        z1 = model(x1, e1, skip_alpha=skip_a)
        z2 = model(x2, e2, skip_alpha=skip_a)
        loss, parts = vicreg_loss(
            z1,
            z2,
            lam=vlam,
            mu=vmu,
            nu=vnu,
            gamma=vgamma,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        train_loss = float(loss.detach().cpu())

        model.eval()
        val_loss = float("nan")
        with torch.no_grad():
            rng_val_a = np.random.default_rng(int(cfg.seed) + 17_017 * int(epoch))
            rng_val_b = np.random.default_rng(int(cfg.seed) + 23_023 * int(epoch))
            xv1, ev1 = augment_graph_view(
                x, edge_index, rng_val_a, edge_drop_prob=edp, feat_drop_prob=fdp
            )
            xv2, ev2 = augment_graph_view(
                x, edge_index, rng_val_b, edge_drop_prob=edp, feat_drop_prob=fdp
            )
            z1v = model(xv1, ev1, skip_alpha=skip_a)
            z2v = model(xv2, ev2, skip_alpha=skip_a)
            vtot, vparts = vicreg_loss(
                z1v, z2v, lam=vlam, mu=vmu, nu=vnu, gamma=vgamma
            )
            val_loss = float(vtot.cpu())

        row: dict[str, Any] = {
            "epoch": int(epoch),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "vicreg_train_inv": float(parts["inv"].cpu()),
            "vicreg_train_var": float(parts["var"].cpu()),
            "vicreg_train_cov": float(parts["cov"].cpu()),
        }

        if not np.isnan(val_loss):
            scheduler.step(val_loss)

        do_eval = (epoch % max(1, int(cfg.eval_every_epochs)) == 0) or (epoch == 1)
        stopped = False

        if do_eval:
            model.eval()
            with torch.no_grad():
                z_np = model(x, edge_index, skip_alpha=skip_a).cpu().numpy()

            row["diag_emb_mean_std_dim"] = embedding_collapse_mean_std(z_np)
            row["diag_offdiag_cov_per_dim"] = embedding_offdiag_cov_frobenius_per_dim(z_np)
            row["diag_emb_norm_mean"] = float(np.linalg.norm(z_np, axis=1).mean())
            row["diag_emb_norm_std"] = float(np.linalg.norm(z_np, axis=1).std())
            diag_rng = np.random.default_rng(int(cfg.seed) + 91_919 * int(epoch))
            row["diag_rand_pair_cosine"] = embedding_random_pair_mean_cosine(
                z_np, diag_rng, n_pairs=4096
            )

            cl_metrics = shard_embeddings_to_email_metrics(
                shard_ids,
                z_np,
                assignments_df,
                gt_label_map,
                min_cluster_size=cfg.hdbscan_min_cluster_size,
                min_samples=cfg.hdbscan_min_samples,
                l2_normalize_rows=clust_l2,
            )
            for k in (
                "homogeneity",
                "completeness",
                "v_measure",
                "n_eval",
                "coverage_gt",
                "coverage_assignments",
                "n_clusters",
                "n_noise_shards",
                "n_shard_nodes",
            ):
                if k in cl_metrics and isinstance(cl_metrics[k], (int, float, np.floating)):
                    row[f"val_{k}"] = float(cl_metrics[k])
            vm = float(cl_metrics.get("v_measure", float("nan")))
            row["val_v_measure"] = vm

            if not np.isnan(vm) and (np.isnan(best_vm) or vm > best_vm + 1e-12):
                best_vm = vm
                best_cluster_epoch = int(epoch)
                np.save(out_dir / "embeddings_clustering_best.npy", z_np)

            if not np.isnan(val_loss) and val_loss < best_val_loss - 1e-7:
                best_val_loss = float(val_loss)
                best_loss_epoch = int(epoch)
                stall = 0
                save_checkpoint(
                    chk / "best.pt",
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    metrics={
                        "val_loss": val_loss,
                        "val_v_measure": float(vm),
                        "train_loss": train_loss,
                    },
                )
                np.save(out_dir / "embeddings_best.npy", z_np)
            elif not np.isnan(val_loss):
                stall += 1

            if cfg.save_embeddings_every_epoch:
                np.save(emb_dir / f"epoch_{epoch:04d}.npy", z_np)

            es_pat = int(cfg.early_stopping_patience)
            if es_pat > 0 and not np.isnan(val_loss) and stall >= es_pat:
                stopped = True
                stop_reason = "early_stop_val_loss"
        else:
            row["val_v_measure"] = float("nan")

        if pbar is not None:
            vl = row["val_loss"]
            pf: dict[str, str] = {
                "trn": f"{row['train_loss']:.4f}",
                "vl": f"{float(vl):.4f}" if not np.isnan(float(vl)) else "nan",
            }
            vm = float(row.get("val_v_measure", float("nan")))
            if not np.isnan(vm):
                pf["Vm"] = f"{vm:.3f}"
            pbar.set_postfix(pf)

        if cfg.save_checkpoint_every_epoch:
            save_checkpoint(
                chk / f"epoch_{epoch:04d}.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                metrics={
                    "train_loss": row["train_loss"],
                    "val_loss": row.get("val_loss", float("nan")),
                    "val_v_measure": row.get("val_v_measure", float("nan")),
                },
            )

        history.append(row)
        if stopped:
            break

    model.eval()
    with torch.no_grad():
        z_final = model(x, edge_index, skip_alpha=skip_a).cpu().numpy()
    np.save(out_dir / "embeddings_final.npy", z_final)

    summary = {
        "best_loss_epoch": int(best_loss_epoch),
        "best_val_loss": float(best_val_loss),
        "best_clustering_epoch": int(best_cluster_epoch),
        "best_v_measure": float(best_vm),
        "final_epoch": int(history[-1]["epoch"]) if history else 0,
        "stop_reason": stop_reason,
        "stall_epochs_without_val_loss_improvement": int(stall),
        "cluster_l2_normalize": clust_l2,
    }
    with open(out_dir / "training_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    if stop_reason == "early_stop_val_loss":
        print(
            f"[VICReg shard val_loss early stop] stall={stall} (patience={cfg.early_stopping_patience})."
        )
    return history, summary
