"""
Shard-graph GNN utilities (Step 4): load Step-2 artifacts, node features, weighted encoders, link-prediction training.

This module trains on the **semantic shard graph** from Steps 1–2 (plus optional Step-3 best-setting metadata),
not on the legacy heterogeneous email graph.
"""

from __future__ import annotations

import json
import math
import copy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch import Tensor, nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import scatter


@dataclass
class Step4TrainConfig:
    """Serializable training hyperparameters + paths."""

    step2_dir: str
    step3_best_json: str | None
    min_edge_weight: float
    train_frac: float
    val_frac: float
    test_frac: float
    neg_sampling_ratio: float
    hidden_dim: int
    out_dim: int
    dropout: float
    lr: float
    weight_decay: float
    epochs: int
    seed: int
    edge_weight_transform: str  # "identity" | "log1p" | "divide_max"
    device: str
    # "dot" = raw inner product (scale-sensitive); "cosine" = L2-normalize z before dot (matches Step-5 Path B geometry)
    link_score: str = "dot"
    # Training stability / preventing overfitting
    best_metric_key: str = "val_ap"  # maximize by default
    best_metric_mode: str = "max"  # "max" | "min"
    best_metric_min_delta: float = 1e-4
    early_stopping_patience_epochs: int = 10
    # Reduce LR after no improvement for N epochs (keyed to same best_metric_key)
    scheduler_patience_epochs: int = 5
    scheduler_factor: float = 0.5
    scheduler_min_lr: float = 1e-6
    show_epoch_progress: bool = True

    def to_jsonable(self) -> dict[str, Any]:
        return asdict(self)


def load_step2_graph_summary(step2_dir: str | Path) -> dict[str, Any]:
    p = Path(step2_dir).expanduser().resolve() / "semantic_shard_step2_graph_summary.json"
    if not p.is_file():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def load_step2_shard_bundle(
    step2_dir: str | Path,
    *,
    centroid_norm_l2: bool = True,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, dict[str, Any] | None]:
    """
    Load nodes CSV, weighted edges CSV, centroid matrix, and optional graph summary.

    Centroid row i aligns with nodes CSV row i (do not reorder one without the other).
    """
    d = Path(step2_dir).expanduser().resolve()
    nodes = pd.read_csv(d / "semantic_shard_step2_nodes.csv")
    edges = pd.read_csv(d / "semantic_shard_step2_edges_weighted.csv")
    cent = np.load(d / "semantic_shard_step2_centroids.npy")
    nodes["shard_id"] = nodes["shard_id"].astype(str)
    edges["shard_a"] = edges["shard_a"].astype(str)
    edges["shard_b"] = edges["shard_b"].astype(str)
    if len(nodes) != len(cent):
        raise ValueError(
            f"nodes CSV ({len(nodes)}) and centroids ({len(cent)}) length mismatch — "
            "expected same row order as Step-2 save."
        )
    if centroid_norm_l2:
        n = np.linalg.norm(cent, axis=1, keepdims=True)
        n[n == 0.0] = 1.0
        cent = (cent / n).astype(np.float32)
    summary = load_step2_graph_summary(d)
    return nodes, cent, edges, summary or None


def load_step3_best_setting(step3_best_json: str | Path) -> dict[str, Any]:
    p = Path(step3_best_json).expanduser().resolve()
    return json.loads(p.read_text(encoding="utf-8"))


DEFAULT_SCALAR_COLUMNS = (
    "size",
    "within_cos_mean",
    "within_cos_median",
    "centroid_dist_mean",
    "centroid_dist_median",
    "n_unique_senders",
    "n_unique_sender_email_domains",
    "n_unique_urls",
    "n_unique_domains",
    "n_unique_stems",
    "n_unique_attachments",
    "ts_span_seconds",
)


def build_shard_node_feature_matrix(
    nodes_df: pd.DataFrame,
    centroid_mat: np.ndarray,
    *,
    scalar_columns: tuple[str, ...] = DEFAULT_SCALAR_COLUMNS,
    log1p_size: bool = True,
    log1p_counts: bool = True,
    fillna: float = 0.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Concatenate L2-normalized semantic centroid with standardized scalar shard statistics.

    Excludes GT-derived columns by default (not in `DEFAULT_SCALAR_COLUMNS`).
    """
    cent = np.asarray(centroid_mat, dtype=np.float32)
    cols_present = [c for c in scalar_columns if c in nodes_df.columns]
    Xb = np.zeros((len(nodes_df), len(cols_present)), dtype=np.float64)
    for j, c in enumerate(cols_present):
        # copy=True: pandas/Arrow may expose a read-only buffer; we assign NaN fills in-place.
        v = pd.to_numeric(nodes_df[c], errors="coerce").to_numpy(dtype=np.float64, copy=True)
        if c == "size" and log1p_size:
            v = np.log1p(np.clip(v, 0.0, None))
        elif log1p_counts and c.startswith("n_unique_"):
            v = np.log1p(np.clip(v, 0.0, None))
        elif c == "ts_span_seconds":
            v = np.log1p(np.clip(v, 0.0, None))
        m = np.isnan(v)
        v[m] = fillna
        Xb[:, j] = v
    scaler = StandardScaler()
    Xs = scaler.fit_transform(Xb).astype(np.float32)
    X = np.concatenate([cent, Xs], axis=1).astype(np.float32)
    meta = {
        "centroid_dim": int(cent.shape[1]),
        "scalar_columns": cols_present,
        "n_scalars": int(len(cols_present)),
        "total_dim": int(X.shape[1]),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
    }
    return X, meta


def undirected_edge_index_and_weight(
    shard_ids: list[str],
    edges_df: pd.DataFrame,
    *,
    min_edge_weight: float = 0.0,
    shard_id_to_idx: dict[str, int] | None = None,
) -> tuple[Tensor, Tensor, list[tuple[int, int, float]]]:
    """
    Build bidirectional edge_index and per-directed-edge weights.

    Returns canonical triples (i, j, w) with i < j once each (undirected logical edge),
    plus duplicated directions in edge_index / edge_weight.
    """
    if shard_id_to_idx is None:
        shard_id_to_idx = {s: i for i, s in enumerate(shard_ids)}
    rows: list[tuple[int, int, float]] = []
    seen: set[tuple[int, int]] = set()
    use = edges_df[edges_df["edge_weight"] >= float(min_edge_weight)].copy()
    for _, r in use.iterrows():
        a, b = str(r["shard_a"]), str(r["shard_b"])
        if a == b:
            continue
        ia, ib = shard_id_to_idx.get(a), shard_id_to_idx.get(b)
        if ia is None or ib is None:
            continue
        i, j = (ia, ib) if ia < ib else (ib, ia)
        key = (i, j)
        if key in seen:
            continue
        seen.add(key)
        w = float(r["edge_weight"])
        rows.append((i, j, w))
    if not rows:
        return (
            torch.empty((2, 0), dtype=torch.long),
            torch.empty((0,), dtype=torch.float32),
            [],
        )
    src: list[int] = []
    dst: list[int] = []
    ew: list[float] = []
    for i, j, w in rows:
        src.extend([i, j])
        dst.extend([j, i])
        ew.extend([w, w])
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_weight = torch.tensor(ew, dtype=torch.float32)
    return edge_index, edge_weight, rows


def transform_edge_weights(edge_weight: Tensor, mode: str) -> Tensor:
    w = edge_weight.clamp(min=0.0)
    if mode == "identity":
        return w
    if mode == "log1p":
        return torch.log1p(w)
    if mode == "divide_max":
        m = float(w.max().item()) if w.numel() else 1.0
        return w / (m + 1e-8)
    raise ValueError(f"Unknown edge_weight_transform: {mode!r}")


class WeightedSAGEConv(MessagePassing):
    """One GraphSAGE-style layer with weighted mean aggregation over neighbors."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(aggr="add")
        self.lin = nn.Linear(2 * in_channels, out_channels)

    def forward(self, x: Tensor, edge_index: Tensor, edge_weight: Tensor) -> Tensor:
        row, col = edge_index[0], edge_index[1]
        deg = scatter(edge_weight, row, dim=0, reduce="sum", dim_size=x.size(0))
        norm = edge_weight * deg[row].clamp(min=1e-8).reciprocal()
        neigh = self.propagate(edge_index, x=x, norm=norm)
        return self.lin(torch.cat([x, neigh], dim=-1)).relu()

    def message(self, x_j: Tensor, norm: Tensor) -> Tensor:
        return norm.view(-1, 1) * x_j


class ShardGraphSAGEEncoder(nn.Module):
    """Two-layer weighted GraphSAGE-style encoder."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.conv1 = WeightedSAGEConv(in_dim, hidden_dim)
        self.conv2 = WeightedSAGEConv(hidden_dim, out_dim)
        self.dropout = float(dropout)

    def forward(self, x: Tensor, edge_index: Tensor, edge_weight: Tensor) -> Tensor:
        h = self.conv1(x, edge_index, edge_weight)
        h = nn.functional.dropout(h, p=self.dropout, training=self.training)
        h = self.conv2(h, edge_index, edge_weight)
        return h


class DotLinkDecoder(nn.Module):
    """Dot-product logits for undirected edges (symmetric)."""

    @staticmethod
    def forward(
        z: Tensor,
        edge_label_index: Tensor,
        *,
        score_mode: str = "dot",
    ) -> Tensor:
        s, t = edge_label_index[0], edge_label_index[1]
        mode = str(score_mode).lower().strip()
        if mode == "cosine":
            z = F.normalize(z, p=2, dim=-1, eps=1e-8)
        elif mode != "dot":
            raise ValueError(f"Unknown score_mode: {score_mode!r} (use 'dot' or 'cosine')")
        return (z[s] * z[t]).sum(dim=-1)


def split_edge_indices(
    *,
    canonical_rows: list[tuple[int, int, float]],
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return index arrays into canonical_rows for train / val / test (leakage-safe split on positives)."""
    if not math.isclose(train_frac + val_frac + test_frac, 1.0, rel_tol=1e-5):
        raise ValueError("train_frac + val_frac + test_frac must sum to 1")
    n = len(canonical_rows)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_train = int(round(train_frac * n))
    n_val = int(round(val_frac * n))
    i_tr = perm[:n_train]
    i_va = perm[n_train : n_train + n_val]
    i_te = perm[n_train + n_val :]
    return i_tr, i_va, i_te


def bidirectional_from_canonical_subset(
    canonical_rows: list[tuple[int, int, float]], idx: np.ndarray
) -> tuple[Tensor, Tensor]:
    src: list[int] = []
    dst: list[int] = []
    ew: list[float] = []
    for t in idx:
        i, j, w = canonical_rows[int(t)]
        src.extend([i, j])
        dst.extend([j, i])
        ew.extend([w, w])
    if not src:
        return torch.empty((2, 0), dtype=torch.long), torch.empty((0,), dtype=torch.float32)
    return torch.tensor([src, dst], dtype=torch.long), torch.tensor(ew, dtype=torch.float32)


def single_dir_pairs(
    canonical_rows: list[tuple[int, int, float]], idx: np.ndarray
) -> Tensor:
    pairs: list[list[int]] = []
    for t in idx:
        i, j, _w = canonical_rows[int(t)]
        pairs.append([i, j])
    if not pairs:
        return torch.empty((2, 0), dtype=torch.long)
    a = np.array(pairs, dtype=np.int64).T
    return torch.from_numpy(a)


def sample_negative_edges(
    *,
    num_nodes: int,
    num_samples: int,
    pos_pairs_undirected: set[tuple[int, int]],
    rng: np.random.Generator,
    max_tries_factor: int = 50,
) -> Tensor:
    """Sample unique undirected non-edges (single direction i<j stored as (i,j))."""
    out: list[tuple[int, int]] = []
    tries = 0
    max_tries = max(1000, num_samples * max_tries_factor)
    while len(out) < num_samples and tries < max_tries:
        tries += 1
        i = int(rng.integers(0, num_nodes))
        j = int(rng.integers(0, num_nodes))
        if i == j:
            continue
        a, b = (i, j) if i < j else (j, i)
        if (a, b) in pos_pairs_undirected:
            continue
        if (a, b) in out:
            continue
        out.append((a, b))
    if len(out) < num_samples:
        raise RuntimeError(
            f"Could sample only {len(out)} negatives of {num_samples}; graph may be very dense."
        )
    arr = np.array(out[:num_samples], dtype=np.int64).T
    return torch.from_numpy(arr)


def link_prediction_metrics(
    z: Tensor,
    pos_idx: Tensor,
    neg_idx: Tensor,
    *,
    score_mode: str = "dot",
) -> dict[str, float]:
    with torch.no_grad():
        pos_s = DotLinkDecoder.forward(z, pos_idx, score_mode=score_mode).cpu().numpy()
        neg_s = DotLinkDecoder.forward(z, neg_idx, score_mode=score_mode).cpu().numpy()
    y = np.concatenate([np.ones_like(pos_s), np.zeros_like(neg_s)])
    s = np.concatenate([pos_s, neg_s])
    out: dict[str, float] = {
        "score_pos_mean": float(pos_s.mean()) if pos_s.size else float("nan"),
        "score_neg_mean": float(neg_s.mean()) if neg_s.size else float("nan"),
    }
    if y.size and len(np.unique(y)) > 1:
        out["auroc"] = float(roc_auc_score(y, s))
        out["ap"] = float(average_precision_score(y, s))
    else:
        out["auroc"] = float("nan")
        out["ap"] = float("nan")
    return out


def train_shard_link_predictor(
    *,
    X: Tensor,
    canonical_rows: list[tuple[int, int, float]],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    cfg: Step4TrainConfig,
) -> dict[str, Any]:
    """
    Train encoder with train-positive subgraph; validate/test on held-out positives vs negatives.
    """
    device = torch.device(
        "cuda"
        if str(cfg.device).startswith("cuda") and torch.cuda.is_available()
        else "cpu"
    )
    if not canonical_rows:
        raise ValueError("No edges to split — cannot train link predictor.")
    rng = np.random.default_rng(cfg.seed)
    torch.manual_seed(cfg.seed)

    pos_undir: set[tuple[int, int]] = {(i, j) for i, j, _ in canonical_rows}

    ei_train, ew_train = bidirectional_from_canonical_subset(canonical_rows, train_idx)
    ew_train_t = transform_edge_weights(ew_train.to(device), cfg.edge_weight_transform)

    encoder = ShardGraphSAGEEncoder(
        in_dim=int(X.size(1)),
        hidden_dim=cfg.hidden_dim,
        out_dim=cfg.out_dim,
        dropout=cfg.dropout,
    ).to(device)
    decoder = DotLinkDecoder()
    score_mode = str(cfg.link_score).lower().strip()
    opt = torch.optim.Adam(
        encoder.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    train_pos = single_dir_pairs(canonical_rows, train_idx).to(device)
    val_pos = single_dir_pairs(canonical_rows, val_idx).to(device)
    test_pos = single_dir_pairs(canonical_rows, test_idx).to(device)

    n_neg_tr = max(1, int(len(train_idx) * cfg.neg_sampling_ratio))
    n_neg_va = max(1, int(len(val_idx) * cfg.neg_sampling_ratio))
    n_neg_te = max(1, int(len(test_idx) * cfg.neg_sampling_ratio))

    neg_train = sample_negative_edges(
        num_nodes=int(X.size(0)),
        num_samples=n_neg_tr,
        pos_pairs_undirected=pos_undir,
        rng=rng,
    ).to(device)
    neg_val = sample_negative_edges(
        num_nodes=int(X.size(0)),
        num_samples=n_neg_va,
        pos_pairs_undirected=pos_undir,
        rng=rng,
    ).to(device)
    neg_test = sample_negative_edges(
        num_nodes=int(X.size(0)),
        num_samples=n_neg_te,
        pos_pairs_undirected=pos_undir,
        rng=rng,
    ).to(device)

    Xd = X.to(device)
    ei_train_d = ei_train.to(device)
    crit = nn.BCEWithLogitsLoss()
    history: list[dict[str, float]] = []

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt,
        mode=str(cfg.best_metric_mode).lower().strip(),
        factor=float(cfg.scheduler_factor),
        patience=int(cfg.scheduler_patience_epochs),
        min_lr=float(cfg.scheduler_min_lr),
    )

    best_state: dict[str, Tensor] | None = None
    best_epoch: int | None = None
    best_metric_val = -float("inf") if str(cfg.best_metric_mode).lower() == "max" else float("inf")
    epochs_without_improve = 0

    epoch_iter = range(1, int(cfg.epochs) + 1)
    pbar = None
    if bool(cfg.show_epoch_progress):
        try:
            from tqdm.auto import tqdm  # type: ignore

            pbar = tqdm(epoch_iter, total=int(cfg.epochs), desc="Epochs")
            epoch_iter = pbar
        except Exception:
            pbar = None

    for epoch in epoch_iter:
        encoder.train()
        opt.zero_grad(set_to_none=True)
        z = encoder(Xd, ei_train_d, ew_train_t)
        pos_logits = decoder.forward(z, train_pos, score_mode=score_mode)
        neg_logits = decoder.forward(z, neg_train, score_mode=score_mode)
        logits = torch.cat([pos_logits, neg_logits], dim=0)
        y = torch.cat(
            [
                torch.ones_like(pos_logits),
                torch.zeros_like(neg_logits),
            ],
            dim=0,
        )
        loss = crit(logits, y)
        loss.backward()
        opt.step()

        encoder.eval()
        with torch.no_grad():
            z_val = encoder(Xd, ei_train_d, ew_train_t)
            v_logits = torch.cat(
                [
                    decoder.forward(z_val, val_pos, score_mode=score_mode),
                    decoder.forward(z_val, neg_val, score_mode=score_mode),
                ],
                dim=0,
            )
            yv = torch.cat(
                [
                    torch.ones(val_pos.size(1), device=device),
                    torch.zeros(neg_val.size(1), device=device),
                ]
            )
            val_loss = float(crit(v_logits, yv).item())
            m_tr = link_prediction_metrics(
                z_val, val_pos, neg_val, score_mode=score_mode
            )  # naming: reuse helper on val set

        rec = {
            "epoch": float(epoch),
            "train_loss": float(loss.item()),
            "val_loss": val_loss,
            "val_auroc": m_tr["auroc"],
            "val_ap": m_tr["ap"],
            "val_score_pos_mean": m_tr["score_pos_mean"],
            "val_score_neg_mean": m_tr["score_neg_mean"],
        }
        history.append(rec)

        # Early stopping + LR scheduling keyed to the same metric.
        curr_metric = float(rec.get(cfg.best_metric_key, float("nan")))
        is_finite = math.isfinite(curr_metric)
        improved = False
        if is_finite:
            if str(cfg.best_metric_mode).lower() == "max":
                improved = curr_metric > (best_metric_val + float(cfg.best_metric_min_delta))
            else:
                improved = curr_metric < (best_metric_val - float(cfg.best_metric_min_delta))

        if improved:
            best_metric_val = curr_metric
            best_epoch = int(epoch)
            epochs_without_improve = 0
            # Store on CPU to reduce GPU memory pressure.
            best_state = {k: v.detach().cpu().clone() for k, v in encoder.state_dict().items()}
            if pbar is not None:
                pbar.write(
                    f"[best] epoch={best_epoch} {cfg.best_metric_key}={best_metric_val:.6f} "
                    f"(val_loss={val_loss:.4f}, val_auroc={float(rec['val_auroc']):.4f})"
                )
            else:
                print(
                    f"[best] epoch={best_epoch} {cfg.best_metric_key}={best_metric_val:.6f} "
                    f"(val_loss={val_loss:.4f}, val_auroc={float(rec['val_auroc']):.4f})"
                )
        else:
            epochs_without_improve += 1

        # Scheduler step should not receive NaN.
        if is_finite:
            scheduler.step(curr_metric)
        else:
            scheduler.step(best_metric_val)

        if pbar is not None:
            lr_now = float(opt.param_groups[0]["lr"])
            pbar.set_postfix(
                train_loss=f"{rec['train_loss']:.4f}",
                val_loss=f"{rec['val_loss']:.4f}",
                val_ap=f"{rec['val_ap']:.4f}",
                val_auroc=f"{rec['val_auroc']:.4f}",
                lr=f"{lr_now:.2e}",
            )

        if int(epochs_without_improve) >= int(cfg.early_stopping_patience_epochs):
            if pbar is not None:
                pbar.write(
                    f"Early stopping: no improvement in {cfg.best_metric_key} for "
                    f"{cfg.early_stopping_patience_epochs} epochs."
                )
            else:
                print(
                    f"Early stopping: no improvement in {cfg.best_metric_key} for "
                    f"{cfg.early_stopping_patience_epochs} epochs."
                )
            break

    if best_state is not None:
        encoder.load_state_dict(best_state, strict=True)

    encoder.eval()
    with torch.no_grad():
        z_final_train = encoder(Xd, ei_train_d, ew_train_t)
        test_logits = torch.cat(
            [
                decoder.forward(z_final_train, test_pos, score_mode=score_mode),
                decoder.forward(z_final_train, neg_test, score_mode=score_mode),
            ],
            dim=0,
        )
        yt = torch.cat(
            [
                torch.ones(test_pos.size(1), device=device),
                torch.zeros(neg_test.size(1), device=device),
            ]
        )
        test_loss = float(crit(test_logits, yt).item())
        test_metrics = link_prediction_metrics(
            z_final_train, test_pos, neg_test, score_mode=score_mode
        )

    return {
        "encoder": encoder,
        "history": history,
        "test_loss": test_loss,
        "test_metrics": test_metrics,
        "neg_train": neg_train.cpu(),
        "neg_val": neg_val.cpu(),
        "neg_test": neg_test.cpu(),
        "best_epoch": best_epoch,
        "best_val_metric": best_metric_val,
    }


@torch.no_grad()
def encode_full_shard_graph(
    encoder: ShardGraphSAGEEncoder,
    X: Tensor,
    edge_index: Tensor,
    edge_weight: Tensor,
    edge_weight_transform: str,
    device: str,
) -> np.ndarray:
    encoder.eval()
    dev = torch.device(
        "cuda" if str(device).startswith("cuda") and torch.cuda.is_available() else "cpu"
    )
    ew = transform_edge_weights(edge_weight.to(dev), edge_weight_transform)
    z = encoder(X.to(dev), edge_index.to(dev), ew)
    return z.cpu().numpy().astype(np.float32)


def save_step4_artifacts(
    *,
    output_dir: str | Path,
    shard_ids: list[str],
    refined_embeddings: np.ndarray,
    encoder: ShardGraphSAGEEncoder,
    train_cfg: Step4TrainConfig,
    feature_schema: dict[str, Any],
    history: list[dict[str, float]],
    split_meta: dict[str, Any],
    test_metrics: dict[str, float],
    split_idx: dict[str, np.ndarray] | None = None,
) -> dict[str, str]:
    out = Path(output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "semantic_shard_step4_refined_embeddings.npy", refined_embeddings)
    (out / "semantic_shard_step4_shard_ids.json").write_text(
        json.dumps(shard_ids, indent=2), encoding="utf-8"
    )
    (out / "semantic_shard_step4_train_config.json").write_text(
        json.dumps(train_cfg.to_jsonable(), indent=2), encoding="utf-8"
    )
    (out / "semantic_shard_step4_feature_schema.json").write_text(
        json.dumps(feature_schema, indent=2), encoding="utf-8"
    )
    if split_idx:
        p_npz = out / "semantic_shard_step4_edge_split_indices.npz"
        np.savez(p_npz, **{k: np.asarray(v, dtype=np.int64) for k, v in split_idx.items()})
        split_meta = {
            **split_meta,
            "edge_split_indices_npz": str(p_npz.resolve()),
            "edge_split_keys": list(split_idx.keys()),
        }
    (out / "semantic_shard_step4_edge_split.json").write_text(
        json.dumps(split_meta, indent=2), encoding="utf-8"
    )
    (out / "semantic_shard_step4_test_metrics.json").write_text(
        json.dumps(test_metrics, indent=2), encoding="utf-8"
    )
    pd.DataFrame(history).to_csv(out / "semantic_shard_step4_train_history.csv", index=False)
    torch.save(
        {
            "encoder_state_dict": encoder.state_dict(),
            "train_cfg": train_cfg.to_jsonable(),
        },
        out / "semantic_shard_step4_model.pt",
    )
    paths = {
        "embeddings_npy": str(out / "semantic_shard_step4_refined_embeddings.npy"),
        "shard_ids_json": str(out / "semantic_shard_step4_shard_ids.json"),
        "model_pt": str(out / "semantic_shard_step4_model.pt"),
        "train_config_json": str(out / "semantic_shard_step4_train_config.json"),
        "history_csv": str(out / "semantic_shard_step4_train_history.csv"),
    }
    if split_idx:
        paths["edge_split_npz"] = str(out / "semantic_shard_step4_edge_split_indices.npz")
    return paths
