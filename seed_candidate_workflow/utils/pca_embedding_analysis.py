"""
Principal component analysis on trained email-node embeddings (pair-supervision encoder).

Extracts the same full-graph email embeddings as ``plot_embedding_space``, fits
``sklearn.decomposition.PCA``, and writes a JSON summary with:

- Per-component **explained variance** and **explained variance ratio** for the first
  10 principal components (or fewer if the rank is smaller)
- **Cumulative** explained variance for PCs 1..10
- **Effective rank** (Shannon / entropy) and **participation ratio** from the
  full spectrum of explained variances
- Counts of components needed to reach 90% and 95% explained variance
- Optional: marginal variance of the top 10 *original* embedding dimensions
  (diagonal of the covariance matrix, before rotation)

Output defaults to ``<run_dir>/pca_analysis/pca_summary.json``.

Example::

    python -m seed_candidate_workflow.utils.pca_embedding_analysis \\
      --run-dir output/runs/my_run \\
      --graph-pt core/graph/output/incidents-lake-misp_hetero.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.decomposition import PCA

_GNN_ROOT = Path(__file__).resolve().parents[2] / "core" / "GNN"
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_GNN_ROOT) not in sys.path:
    sys.path.insert(0, str(_GNN_ROOT))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from seed_candidate_workflow.utils.pair_model_inference import load_pair_supervision_for_inference  # noqa: E402
from seed_candidate_workflow.utils.raw_gnn_notebook import load_email_external_ids  # noqa: E402


def _subsample_ids(
    ids: list[str],
    *,
    max_points: int,
    random_state: int,
) -> list[str]:
    if max_points <= 0 or len(ids) <= max_points:
        return ids
    rng = np.random.default_rng(random_state)
    idx = rng.choice(len(ids), size=max_points, replace=False)
    return [ids[int(i)] for i in sorted(idx)]


def _finite_positive(ev: np.ndarray) -> np.ndarray:
    x = np.asarray(ev, dtype=np.float64)
    x = x[np.isfinite(x) & (x > 0.0)]
    return x


def effective_rank_shannon(explained_variance: np.ndarray) -> float:
    """
    exp(entropy) of normalized eigenvalues; ~1 if variance is concentrated on one PC,
    ~n if spread uniformly across n PCs.
    """
    ev = _finite_positive(explained_variance)
    if ev.size == 0:
        return float("nan")
    p = ev / ev.sum()
    h = -float(np.sum(p * np.log(p + 1e-300)))
    return float(np.exp(h))


def participation_ratio(explained_variance: np.ndarray) -> float:
    """
    (sum λ_i)^2 / sum λ_i^2  over nonzero eigenvalues; same as 1 / Herfindahl of variance shares.
    """
    ev = _finite_positive(explained_variance)
    if ev.size == 0:
        return float("nan")
    s = float(ev.sum())
    if s <= 0.0:
        return float("nan")
    return float(s**2 / float(np.dot(ev, ev)))


def smallest_k_cumulative_ratio(ratios: np.ndarray, target: float) -> int:
    """Smallest k such that sum(ratios[:k]) >= target (1-based k)."""
    r = np.asarray(ratios, dtype=np.float64)
    if r.size == 0:
        return 0
    cum = np.cumsum(r)
    hit = np.nonzero(cum >= float(target))[0]
    if hit.size == 0:
        return int(r.size)
    return int(hit[0] + 1)


def _top_n_original_dim_variance(X: np.ndarray, n: int) -> list[dict[str, Any]]:
    """Marginal variance per input dimension (columns of X), descending."""
    var = np.var(X, axis=0, ddof=0)
    d = int(X.shape[1])
    order = np.argsort(-var)
    out: list[dict[str, Any]] = []
    for rank, j in enumerate(order[: min(n, d)], start=1):
        ji = int(j)
        out.append(
            {
                "rank": rank,
                "embedding_dimension_index": ji,
                "marginal_variance": float(var[ji]),
            }
        )
    return out


def build_pca_summary(
    X: np.ndarray,
    *,
    top_pc_count: int = 10,
    original_dim_top_n: int = 10,
) -> dict[str, Any]:
    """
    Fit PCA on X (n_samples x n_features); return a JSON-serializable summary dict.
    """
    n, d = X.shape
    if n < 2:
        raise ValueError("PCA requires at least 2 samples.")
    n_comp_fit = min(n, d)
    pca = PCA(n_components=n_comp_fit, svd_solver="full")
    pca.fit(X)

    ev = np.asarray(pca.explained_variance_, dtype=np.float64)
    evr = np.asarray(pca.explained_variance_ratio_, dtype=np.float64)

    k10 = min(top_pc_count, ev.size)
    top_pc: list[dict[str, Any]] = []
    for i in range(k10):
        top_pc.append(
            {
                "principal_component": i + 1,
                "explained_variance": float(ev[i]),
                "explained_variance_ratio": float(evr[i]),
                "cumulative_explained_variance_ratio": float(np.sum(evr[: i + 1])),
            }
        )

    cum_top = float(np.sum(evr[:k10])) if k10 else 0.0

    return {
        "n_samples": int(n),
        "n_features": int(d),
        "n_components_fitted": int(ev.size),
        "n_top_pcs_reported": int(k10),
        "top_principal_components": top_pc,
        "cumulative_explained_variance_ratio_top_reported_pcs": cum_top,
        "total_explained_variance_sum": float(ev.sum()),
        "effective_rank_shannon": effective_rank_shannon(ev),
        "participation_ratio": participation_ratio(ev),
        "components_for_90pct_variance": smallest_k_cumulative_ratio(evr, 0.90),
        "components_for_95pct_variance": smallest_k_cumulative_ratio(evr, 0.95),
        "original_dimensions_top_by_marginal_variance": _top_n_original_dim_variance(
            X, original_dim_top_n
        ),
    }


def write_pca_embedding_analysis(
    *,
    run_dir: Path | str,
    graph_pt: Path | str,
    max_points: int = 0,
    random_state: int = 42,
    out_subdir: str = "pca_analysis",
    checkpoint_name: str = "best_model.pt",
    device: str = "cpu",
    to_undirected: bool = True,
    top_pc_count: int = 10,
    original_dim_top_n: int = 10,
) -> dict[str, Any]:
    run = Path(run_dir).expanduser().resolve()
    gpath = Path(graph_pt).expanduser().resolve()
    meta_path = gpath.with_suffix(".meta.json")
    if not meta_path.is_file():
        raise FileNotFoundError(f"Graph metadata not found (expected {meta_path})")

    external_ids = load_email_external_ids(meta_path)
    bundle = load_pair_supervision_for_inference(
        run_dir=run,
        graph_pt=gpath,
        checkpoint_name=checkpoint_name,
        device=device,
        to_undirected=to_undirected,
    )

    from src.clustering.clustering_helpers import extract_email_embeddings  # noqa: E402

    ids_full = [str(x) for x in external_ids]
    id_to_emb = extract_email_embeddings(
        bundle["model"],
        bundle["data_cpu"],
        bundle["device"],
        ids_full,
    )
    sampled_ids = (
        _subsample_ids(ids_full, max_points=max_points, random_state=random_state)
        if max_points > 0
        else ids_full
    )
    X = np.stack([id_to_emb[eid] for eid in sampled_ids], axis=0)

    summary = build_pca_summary(
        X,
        top_pc_count=top_pc_count,
        original_dim_top_n=original_dim_top_n,
    )

    out_dir = run / out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "pca_summary.json"

    payload = {
        "run_dir": str(run),
        "graph_pt": str(gpath),
        "checkpoint": checkpoint_name,
        "n_emails_in_sample": len(sampled_ids),
        "max_points_cap": max_points if max_points > 0 else None,
        "random_state": random_state if max_points > 0 else None,
        "pca": summary,
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return {
        "run_dir": run,
        "summary_path": summary_path,
        "payload": payload,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="PCA summary (top PCs, effective rank, participation ratio) on encoder email embeddings.",
    )
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--graph-pt", type=Path, required=True)
    p.add_argument(
        "--max-points",
        type=int,
        default=0,
        help="Subsample this many emails for PCA (0 = use all; full encoder pass still runs once).",
    )
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--out-subdir", type=str, default="pca_analysis")
    p.add_argument("--checkpoint", type=str, default="best_model.pt")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--top-pc-count", type=int, default=10, help="Report first K PCs (default: 10)")
    p.add_argument(
        "--original-dim-top-n",
        type=int,
        default=10,
        help="Report top N raw embedding coordinates by marginal variance (default: 10)",
    )
    p.add_argument(
        "--no-to-undirected",
        action="store_true",
        help="Load graph without ToUndirected (default: undirected, matching training)",
    )
    args = p.parse_args(argv)

    out = write_pca_embedding_analysis(
        run_dir=args.run_dir,
        graph_pt=args.graph_pt,
        max_points=args.max_points,
        random_state=args.random_state,
        out_subdir=args.out_subdir,
        checkpoint_name=args.checkpoint,
        device=args.device,
        to_undirected=not args.no_to_undirected,
        top_pc_count=args.top_pc_count,
        original_dim_top_n=args.original_dim_top_n,
    )
    print(out["summary_path"])
    pca = out["payload"]["pca"]
    k_rep = pca.get("n_top_pcs_reported", 0)
    cum = pca.get("cumulative_explained_variance_ratio_top_reported_pcs")
    print(f"cumulative explained variance ratio (first {k_rep} PCs): {cum}")
    print(f"effective_rank_shannon: {pca['effective_rank_shannon']}")
    print(f"participation_ratio: {pca['participation_ratio']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
