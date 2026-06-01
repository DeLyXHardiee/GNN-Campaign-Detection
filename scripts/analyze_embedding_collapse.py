"""
Embedding-collapse analysis for a GNN link-prediction run.

Generates (all saved under <run_dir>/collapse_analysis/):
  training_curves.png         – val loss + val accuracy vs epoch
  pca_scree.png               – variance explained by each PC (top-N bar + line)
  pca_cumulative.png          – cumulative explained variance ratio
  pca_dim_std.png             – per-embedding-dimension std, sorted descending
  tsne_embeddings.png         – 2-D t-SNE, points coloured by GT campaign
  norm_histogram.png          – distribution of L2 embedding norms
  cosine_histogram.png        – pairwise cosine similarity distribution (subsample)
  pc_correlation_heatmap.png  – Pearson r between top-k PCs and email attrs + graph degree
  pc_campaign_eta2.png        – between-campaign variance fraction (η²) per PC
  pc_loadings.png             – which embedding dims drive each PC
  pca_summary.json            – numeric PCA stats (effective rank, PR, 90/95 thresholds)

Usage (from repo root, with venv active)::

    python scripts/analyze_embedding_collapse.py \\
        --run-dir output/runs/gnn_embeddings_clustering_train_full_graph \\
        --graph-pt core/graph/output/incidents-lake-misp-url-fixed_hetero.pt \\
        --gt-json data/groundtruth/ground_truth_merged.json

All args have defaults matching the run above, so plain invocation works too:

    python scripts/analyze_embedding_collapse.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
_GNN_SRC = _REPO / "core" / "GNN" / "src"
_GNN_ROOT = _REPO / "core" / "GNN"
_CORE = _REPO / "core"

for p in (_GNN_ROOT, _GNN_SRC, _CORE, _REPO):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_external_ids(meta_json: Path) -> list[str]:
    with open(meta_json, encoding="utf-8") as f:
        meta = json.load(f)
    xs = meta.get("email_attrs", {}).get("external_id")
    if not xs:
        raise ValueError(f"No email_attrs.external_id in {meta_json}")
    return [str(x) for x in xs]


def _load_gt_label_map(gt_json: Path) -> dict[str, int | str]:
    with open(gt_json, encoding="utf-8") as f:
        data = json.load(f)
    label_map: dict[str, int | str] = {}
    for raw_key, emails in (data.get("clusters") or {}).items():
        cid_str = raw_key.split("/")[-1] if "/" in raw_key else raw_key
        try:
            cid: int | str = int(cid_str)
        except ValueError:
            cid = cid_str
        if not isinstance(emails, list):
            continue
        for em in emails:
            if not isinstance(em, dict):
                continue
            eid = em.get("external_id")
            if eid is None:
                continue
            eid = str(eid)
            if eid not in label_map:
                label_map[eid] = cid
    return label_map


def _load_model_and_graph(run_dir: Path, graph_pt: Path, device_str: str = "cpu"):
    import torch
    from src.load_graph_data import load_hetero_pt
    from src.model import HeteroSAGE

    ckpt_path = run_dir / "models" / "best_model.pt"
    ckpt = torch.load(str(ckpt_path), map_location=device_str, weights_only=False)

    cfg = ckpt.get("config", {})
    hidden = int(cfg.get("hidden", 128))
    out_dim = int(cfg.get("out_dim", 128))
    layers = int(cfg.get("layers", 2))
    dropout = float(cfg.get("dropout", 0.0))
    metadata = ckpt.get("data_metadata")
    if metadata is None:
        raise ValueError("Checkpoint missing data_metadata; cannot rebuild model.")

    import torch as _t
    device = _t.device(device_str)
    model = HeteroSAGE(metadata=metadata, hidden=hidden, out=out_dim, layers=layers, dropout=dropout).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    data = load_hetero_pt(str(graph_pt), to_undirected=True)
    data_cpu = data.to("cpu")
    return model, data_cpu, device


@np.errstate(invalid="ignore", divide="ignore")
def _extract_embeddings(model, data_cpu, device, ids: list[str]) -> np.ndarray:
    """Return (N, D) float64 embedding matrix aligned to `ids`."""
    import torch

    from src.clustering.clustering_helpers import extract_email_embeddings

    id_to_emb = extract_email_embeddings(model, data_cpu, device, ids)
    return np.stack([id_to_emb[eid] for eid in ids], axis=0).astype(np.float64)


# ---------------------------------------------------------------------------
# PCA stats
# ---------------------------------------------------------------------------

def _fit_pca(X: np.ndarray, top_n: int = 32):
    """Fit full PCA and return (pca_object, stats_dict)."""
    from sklearn.decomposition import PCA

    n, d = X.shape
    n_comp = min(n, d)
    pca = PCA(n_components=n_comp, svd_solver="full")
    pca.fit(X)
    ev = np.asarray(pca.explained_variance_, dtype=np.float64)
    evr = np.asarray(pca.explained_variance_ratio_, dtype=np.float64)

    def _eff_rank_shannon(ev):
        ev = ev[ev > 0]
        p = ev / ev.sum()
        return float(np.exp(-np.sum(p * np.log(p + 1e-300))))

    def _participation_ratio(ev):
        ev = ev[ev > 0]
        return float(ev.sum() ** 2 / np.dot(ev, ev))

    def _k_for_target(evr, target):
        cum = np.cumsum(evr)
        hit = np.nonzero(cum >= target)[0]
        return int(hit[0] + 1) if hit.size else int(evr.size)

    stats = {
        "n_samples": int(n),
        "n_features": int(d),
        "explained_variance": ev[:top_n].tolist(),
        "explained_variance_ratio": evr[:top_n].tolist(),
        "cumulative_evr_full": np.cumsum(evr).tolist(),
        "effective_rank_shannon": _eff_rank_shannon(ev),
        "participation_ratio": _participation_ratio(ev),
        "k_90pct": _k_for_target(evr, 0.90),
        "k_95pct": _k_for_target(evr, 0.95),
        "dim_std_sorted_desc": float(np.std(X, axis=0, ddof=1).max()),
    }
    return pca, stats


def _pca_stats(X: np.ndarray, top_n: int = 32) -> dict:
    _, stats = _fit_pca(X, top_n=top_n)
    return stats


# ---------------------------------------------------------------------------
# PC interpretation helpers
# ---------------------------------------------------------------------------

def _load_email_scalar_attrs(meta_json: Path, n_emails: int) -> pd.DataFrame:
    """Load scalar email attributes from meta.json as a (n_emails, F) DataFrame."""
    with open(meta_json, encoding="utf-8") as f:
        meta = json.load(f)
    attrs = meta.get("email_attrs", {})

    scalar_keys = [
        "ts", "n_urls", "len_body", "len_subject",
        "cyrillic_domain", "contains_symbols",
        "body_has_tracking_url", "body_has_tracking_image",
        "body_has_tracking_pixel", "body_has_unsubscribe_link",
        "domain_is_common_webprovided",
        "auth_spf", "auth_dkim", "auth_dmarc",
    ]
    rows: dict[str, np.ndarray] = {}
    for k in scalar_keys:
        val = attrs.get(k)
        if isinstance(val, list) and len(val) == n_emails:
            try:
                rows[k] = np.array([float(v) if v is not None else np.nan for v in val])
            except (TypeError, ValueError):
                pass

    df = pd.DataFrame(rows)
    df = df.loc[:, df.notna().any()]
    return df


def _compute_email_degrees(data_cpu, n_emails: int) -> pd.DataFrame:
    """Count per-email outgoing edges to each neighbor node type."""
    rows: dict[str, np.ndarray] = {}
    for (src_type, _rel, dst_type), ei in data_cpu.edge_index_dict.items():
        if src_type != "email":
            continue
        src_idx = ei[0].numpy()
        counts = np.bincount(src_idx, minlength=n_emails).astype(np.float32)
        rows[f"deg_{dst_type}"] = counts
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plotting functions
# ---------------------------------------------------------------------------

_FIG_W = 7.0
_FIG_H = 4.5
_DPI = 150
_BLUE = "#1f77b4"
_ORANGE = "#ff7f0e"
_GREEN = "#2ca02c"


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {path}")


def plot_training_curves(metrics_csv: Path, out_dir: Path) -> None:
    df = pd.read_csv(metrics_csv).sort_values("epoch").reset_index(drop=True)
    epochs = df["epoch"].to_numpy()

    fig, axes = plt.subplots(1, 2, figsize=(_FIG_W * 1.6, _FIG_H))

    ax = axes[0]
    if "train_loss" in df.columns:
        ax.plot(epochs, df["train_loss"].to_numpy(), color=_BLUE, linewidth=1.8, label="train loss")
    ax.plot(epochs, df["val_loss"].to_numpy(), color=_ORANGE, linewidth=1.8, label="val loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (BCE)")
    ax.set_title("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    if "train_acc" in df.columns:
        ax.plot(epochs, df["train_acc"].to_numpy(), color=_BLUE, linewidth=1.8, label="train acc")
    ax.plot(epochs, df["val_acc"].to_numpy(), color=_GREEN, linewidth=1.8, label="val acc")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy")
    ax.set_ylim(0.0, 1.0)
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle("Link-prediction training curves", fontsize=12)
    fig.tight_layout()
    _save(fig, out_dir / "training_curves.png")


def plot_pca_scree(stats: dict, out_dir: Path, top_n: int = 32) -> None:
    evr = np.asarray(stats["explained_variance_ratio"])[:top_n]
    pcs = np.arange(1, len(evr) + 1)
    cum = np.cumsum(evr)

    fig, ax = plt.subplots(figsize=(_FIG_W, _FIG_H))
    ax.bar(pcs, evr * 100, color=_BLUE, alpha=0.75, label="Individual")
    ax2 = ax.twinx()
    ax2.plot(pcs, cum * 100, color=_ORANGE, linewidth=1.8, marker="o", markersize=3, label="Cumulative")
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Explained variance (%)")
    ax2.set_ylabel("Cumulative explained variance (%)", color=_ORANGE)
    ax2.tick_params(axis="y", labelcolor=_ORANGE)
    ax.set_title(
        f"PCA scree plot  (eff. rank={stats['effective_rank_shannon']:.1f}, "
        f"PR={stats['participation_ratio']:.1f}, "
        f"90% in {stats['k_90pct']} PCs)"
    )
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="center right")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    _save(fig, out_dir / "pca_scree.png")


def plot_pca_cumulative(stats: dict, out_dir: Path) -> None:
    cum = np.asarray(stats["cumulative_evr_full"])
    pcs = np.arange(1, len(cum) + 1)

    fig, ax = plt.subplots(figsize=(_FIG_W, _FIG_H))
    ax.plot(pcs, cum * 100, color=_BLUE, linewidth=1.8)
    ax.axhline(90, color=_ORANGE, linestyle="--", linewidth=1.2, label="90%")
    ax.axhline(95, color=_GREEN, linestyle="--", linewidth=1.2, label="95%")
    ax.axvline(stats["k_90pct"], color=_ORANGE, linestyle=":", linewidth=1.0)
    ax.axvline(stats["k_95pct"], color=_GREEN, linestyle=":", linewidth=1.0)
    ax.set_xlabel("Number of principal components")
    ax.set_ylabel("Cumulative explained variance (%)")
    ax.set_title("Cumulative explained variance")
    ax.set_ylim(0, 102)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, out_dir / "pca_cumulative.png")


def plot_dim_std(X: np.ndarray, out_dir: Path) -> None:
    stds = np.sort(np.std(X, axis=0, ddof=1))[::-1]
    dims = np.arange(1, len(stds) + 1)

    fig, ax = plt.subplots(figsize=(_FIG_W, _FIG_H))
    ax.bar(dims, stds, color=_BLUE, alpha=0.75, width=1.0)
    ax.set_xlabel("Embedding dimension (sorted by std)")
    ax.set_ylabel("Standard deviation")
    ax.set_title(
        f"Per-dimension std (D={len(stds)}; "
        f"max={stds[0]:.4f}, median={np.median(stds):.4f}, min={stds[-1]:.4f})"
    )
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    _save(fig, out_dir / "pca_dim_std.png")


def plot_norm_histogram(X: np.ndarray, out_dir: Path) -> None:
    norms = np.linalg.norm(X, axis=1)

    fig, ax = plt.subplots(figsize=(_FIG_W, _FIG_H))
    ax.hist(norms, bins=60, color=_BLUE, alpha=0.8, edgecolor="white", linewidth=0.3)
    ax.axvline(float(np.mean(norms)), color=_ORANGE, linewidth=1.5, label=f"mean={np.mean(norms):.3f}")
    ax.set_xlabel("L2 norm")
    ax.set_ylabel("Count")
    ax.set_title(
        f"Distribution of embedding L2 norms  "
        f"(std={np.std(norms):.4f}, cv={np.std(norms)/max(np.mean(norms),1e-9):.4f})"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, out_dir / "norm_histogram.png")


def plot_cosine_histogram(X: np.ndarray, out_dir: Path, n_sample: int = 3000, seed: int = 42) -> None:
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    n_sample = min(n_sample, n)
    idx = rng.choice(n, size=n_sample, replace=False)
    Xs = X[idx]
    norms = np.linalg.norm(Xs, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    Xn = Xs / norms
    gram = Xn @ Xn.T
    iu = np.triu_indices(n_sample, k=1)
    sims = gram[iu]

    fig, ax = plt.subplots(figsize=(_FIG_W, _FIG_H))
    ax.hist(sims, bins=80, color=_BLUE, alpha=0.8, edgecolor="white", linewidth=0.3)
    ax.axvline(float(np.mean(sims)), color=_ORANGE, linewidth=1.5, label=f"mean={np.mean(sims):.3f}")
    ax.set_xlabel("Cosine similarity")
    ax.set_ylabel("Pair count")
    ax.set_title(
        f"Pairwise cosine similarity ({n_sample} sampled points)\n"
        f"mean={np.mean(sims):.3f}, std={np.std(sims):.4f}"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, out_dir / "cosine_histogram.png")


def plot_tsne(
    X: np.ndarray,
    ids: list[str],
    label_map: dict | None,
    out_dir: Path,
    max_points: int = 6000,
    seed: int = 42,
) -> None:
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler

    n = X.shape[0]
    if n > max_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, size=max_points, replace=False)
        X_sub = X[idx]
        ids_sub = [ids[i] for i in idx]
    else:
        X_sub = X
        ids_sub = ids

    Xs = StandardScaler().fit_transform(X_sub)
    n_sub = Xs.shape[0]
    perp = float(max(5, min(30, (n_sub - 1) // 3)))
    xy = TSNE(n_components=2, random_state=seed, perplexity=perp, init="pca", learning_rate="auto").fit_transform(Xs)

    fig, ax = plt.subplots(figsize=(8, 7))
    if label_map is None:
        ax.scatter(xy[:, 0], xy[:, 1], s=8, c=_BLUE, alpha=0.55, linewidths=0)
    else:
        labels = [label_map.get(eid) for eid in ids_sub]
        unlabeled = np.array([L is None for L in labels])
        labeled = ~unlabeled
        if unlabeled.any():
            ax.scatter(xy[unlabeled, 0], xy[unlabeled, 1], s=6, c="0.82", alpha=0.35, linewidths=0, label="no GT", zorder=1)
        if labeled.any():
            uniq = sorted({labels[i] for i in range(n_sub) if labels[i] is not None}, key=str)
            code = {v: j for j, v in enumerate(uniq)}
            codes = np.array([code[labels[i]] for i in range(n_sub) if labels[i] is not None], dtype=np.int32)
            nc = len(uniq)
            cmap = plt.colormaps["nipy_spectral"].resampled(max(nc, 2))
            sc = ax.scatter(xy[labeled, 0], xy[labeled, 1], s=10, c=codes, cmap=cmap, vmin=0, vmax=max(nc - 1, 1), alpha=0.85, linewidths=0, zorder=2)
            cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("campaign index")
        ax.set_title(
            f"t-SNE embedding space  ({int(labeled.sum())} labelled / {n_sub} points)"
            if label_map else f"t-SNE embedding space  ({n_sub} points)"
        )

    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    _save(fig, out_dir / "tsne_embeddings.png")


# ---------------------------------------------------------------------------
# PC interpretation plots
# ---------------------------------------------------------------------------

def plot_pc_correlation_heatmap(
    pc_scores: np.ndarray,
    attr_df: pd.DataFrame,
    degree_df: pd.DataFrame,
    out_dir: Path,
) -> None:
    """Pearson correlation between top-k PC scores and email metadata / connectivity."""
    top_k = pc_scores.shape[1]
    pc_cols = [f"PC{i + 1}" for i in range(top_k)]
    pc_df = pd.DataFrame(pc_scores, columns=pc_cols)

    feature_df = pd.concat(
        [attr_df.reset_index(drop=True), degree_df.reset_index(drop=True)], axis=1
    )
    # Drop columns where there is no variance
    feature_df = feature_df.loc[:, feature_df.std(ddof=0) > 0]

    combined = pd.concat([pc_df, feature_df], axis=1)
    corr = combined.corr().loc[pc_cols, feature_df.columns]

    n_feat = len(feature_df.columns)
    fig_h = max(_FIG_H, n_feat * 0.34 + 1.5)
    fig, ax = plt.subplots(figsize=(_FIG_W * 1.2, fig_h))
    im = ax.imshow(corr.values.T, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(pc_cols)))
    ax.set_xticklabels(pc_cols)
    ax.set_yticks(range(n_feat))
    ax.set_yticklabels(list(feature_df.columns), fontsize=8)
    ax.set_title("Pearson r: top PCs vs email attributes & graph connectivity")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="r")
    for i in range(len(pc_cols)):
        for j in range(n_feat):
            val = float(corr.values[i, j])
            if np.isfinite(val):
                ax.text(
                    i, j, f"{val:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if abs(val) > 0.5 else "black",
                )
    fig.tight_layout()
    _save(fig, out_dir / "pc_correlation_heatmap.png")


def plot_pc_campaign_eta2(
    pc_scores: np.ndarray,
    ids: list[str],
    label_map: dict,
    out_dir: Path,
) -> None:
    """Bar chart of η² (between-campaign variance fraction) for each PC."""
    top_k = pc_scores.shape[1]
    labels = np.array([label_map.get(eid) for eid in ids])
    has_label = labels != None  # noqa: E711
    if has_label.sum() < 10:
        return

    s_scores = pc_scores[has_label]
    s_labels = labels[has_label]
    uniq = [c for c in sorted(set(s_labels), key=str) if c is not None]

    eta2 = []
    for pc_i in range(top_k):
        col = s_scores[:, pc_i]
        grand_mean = col.mean()
        ss_total = float(np.sum((col - grand_mean) ** 2))
        ss_between = float(sum(
            np.sum(s_labels == c) * (col[s_labels == c].mean() - grand_mean) ** 2
            for c in uniq
        ))
        eta2.append(ss_between / ss_total if ss_total > 0 else 0.0)

    pc_labels = [f"PC{i + 1}" for i in range(top_k)]
    fig, ax = plt.subplots(figsize=(_FIG_W, _FIG_H))
    bars = ax.bar(pc_labels, eta2, color=_BLUE, alpha=0.8)
    ax.bar_label(bars, fmt="{:.3f}", padding=3, fontsize=9)
    ax.set_ylim(0, min(1.05, max(eta2) * 1.25 + 0.05))
    ax.set_xlabel("Principal component")
    ax.set_ylabel("η² (between-campaign variance / total variance)")
    ax.set_title(
        f"Campaign separability per PC  ({len(uniq)} campaigns, {has_label.sum()} emails)\n"
        "η² ≈ 1 → PC perfectly separates campaigns; ≈ 0 → no separation"
    )
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    _save(fig, out_dir / "pc_campaign_eta2.png")


def plot_pc_loadings(pca, out_dir: Path, top_k: int = 5, top_dims: int = 30) -> None:
    """Heatmap of which embedding dimensions contribute most to each PC."""
    components = pca.components_[:top_k]
    D = components.shape[1]
    top_dims = min(top_dims, D)

    # Select dims with largest max-abs loading across the top-k PCs
    max_abs = np.max(np.abs(components), axis=0)
    top_idx = np.sort(np.argsort(max_abs)[::-1][:top_dims])
    sub = components[:, top_idx]

    vmax = float(np.abs(sub).max()) or 1.0
    fig_h = max(2.5, top_k * 0.55 + 1.5)
    fig, ax = plt.subplots(figsize=(_FIG_W * 1.4, fig_h))
    im = ax.imshow(sub, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_yticks(range(top_k))
    ax.set_yticklabels([f"PC{i + 1}" for i in range(top_k)])
    ax.set_xticks(range(len(top_idx)))
    ax.set_xticklabels([f"d{i}" for i in top_idx], rotation=90, fontsize=7)
    ax.set_xlabel("Embedding dimension index")
    ax.set_ylabel("Principal component")
    ax.set_title(
        f"PC loadings — top {top_dims} most influential dims (of {D} total)\n"
        "Red = positive loading, blue = negative"
    )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="loading")
    fig.tight_layout()
    _save(fig, out_dir / "pc_loadings.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    _RUN = "output/runs/gnn_embeddings_clustering_train_full_graph_fixed_leak (5)"
    _GRAPH = "core/graph/output/incidents-lake-misp-url-fixed_hetero.pt"
    _GT = "data/groundtruth/ground_truth_merged.json"

    p = argparse.ArgumentParser(description="Embedding-collapse analysis for a GNN run.")
    p.add_argument("--run-dir", type=Path, default=Path(_RUN))
    p.add_argument("--graph-pt", type=Path, default=Path(_GRAPH))
    p.add_argument("--gt-json", type=Path, default=Path(_GT))
    p.add_argument("--out-subdir", default="collapse_analysis")
    p.add_argument("--device", default="cpu")
    p.add_argument("--max-tsne-points", type=int, default=6000)
    p.add_argument("--top-pcs", type=int, default=32, help="PCs shown in scree plot")
    p.add_argument("--cosine-sample", type=int, default=3000)
    p.add_argument("--interpret-pcs", type=int, default=5, help="Number of PCs to interpret")
    args = p.parse_args(argv)

    run_dir = (_REPO / args.run_dir).resolve() if not args.run_dir.is_absolute() else args.run_dir.resolve()
    graph_pt = (_REPO / args.graph_pt).resolve() if not args.graph_pt.is_absolute() else args.graph_pt.resolve()
    gt_json = (_REPO / args.gt_json).resolve() if not args.gt_json.is_absolute() else args.gt_json.resolve()
    out_dir = run_dir / args.out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Embedding-collapse analysis ===")
    print(f"Run dir : {run_dir}")
    print(f"Graph   : {graph_pt}")
    print(f"GT      : {gt_json}")
    print(f"Output  : {out_dir}\n")

    # 1. Training curves
    print("[1/6] Training curves …")
    metrics_csv = run_dir / "metrics.csv"
    plot_training_curves(metrics_csv, out_dir)

    # 2. Load model + extract embeddings (once)
    print("[2/6] Loading model and extracting embeddings …")
    meta_json = graph_pt.with_suffix(".meta.json")
    external_ids = _load_external_ids(meta_json)
    model, data_cpu, device = _load_model_and_graph(run_dir, graph_pt, args.device)
    X = _extract_embeddings(model, data_cpu, device, external_ids)
    print(f"       Embedding matrix: {X.shape}  (N={X.shape[0]}, D={X.shape[1]})")

    # 3. PCA
    print("[3/9] PCA analysis …")
    pca_obj, stats = _fit_pca(X, top_n=args.top_pcs)
    print(f"       effective_rank_shannon = {stats['effective_rank_shannon']:.2f}")
    print(f"       participation_ratio    = {stats['participation_ratio']:.2f}")
    print(f"       90% variance in {stats['k_90pct']} PCs, 95% in {stats['k_95pct']} PCs")

    summary_path = out_dir / "pca_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"  saved → {summary_path}")

    plot_pca_scree(stats, out_dir, top_n=args.top_pcs)
    plot_pca_cumulative(stats, out_dir)
    plot_dim_std(X, out_dir)

    # 4. Embedding norm histogram
    print("[4/9] Norm histogram …")
    plot_norm_histogram(X, out_dir)

    # 5. Cosine similarity histogram
    print("[5/9] Cosine similarity histogram …")
    plot_cosine_histogram(X, out_dir, n_sample=args.cosine_sample)

    # 6. t-SNE
    print("[6/9] t-SNE projection …")
    label_map: dict | None = None
    if gt_json.is_file():
        label_map = _load_gt_label_map(gt_json)
        print(f"       GT campaigns loaded: {len(set(label_map.values()))} campaigns, {len(label_map)} labelled emails")
    else:
        print("       No GT file found — t-SNE will be uncoloured")
    plot_tsne(X, external_ids, label_map, out_dir, max_points=args.max_tsne_points)

    # 7. PC interpretation — correlation with email attributes
    print("[7/9] PC interpretation — correlation heatmap …")
    n_emails = X.shape[0]
    top_k = min(args.interpret_pcs, pca_obj.n_components_)
    pc_scores = pca_obj.transform(X)[:, :top_k]
    attr_df = _load_email_scalar_attrs(meta_json, n_emails)
    degree_df = _compute_email_degrees(data_cpu, n_emails)
    plot_pc_correlation_heatmap(pc_scores, attr_df, degree_df, out_dir)

    # 8. PC interpretation — campaign separability (η²)
    print("[8/9] PC interpretation — campaign separability (η²) …")
    if label_map is not None:
        plot_pc_campaign_eta2(pc_scores, external_ids, label_map, out_dir)
    else:
        print("       Skipped (no GT label map)")

    # 9. PC loadings
    print("[9/9] PC loadings heatmap …")
    plot_pc_loadings(pca_obj, out_dir, top_k=top_k)

    print(f"\nDone. All plots saved under:\n  {out_dir}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
