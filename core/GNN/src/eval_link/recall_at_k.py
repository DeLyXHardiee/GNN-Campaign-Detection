"""
Recall@K analysis for link prediction.
Lower-level: operates on already-loaded device, model, and splits.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from ..embed import embed_with_graph
from ..load_graph_data import load_hetero_pt
from ..model_io import get_models_dir, load_full_run, select_device


def recall_at_k_mrr(
    h: dict,
    edge_type: tuple,
    test_edges: torch.Tensor,
    K: int = 20,
    use_dot: bool = True,
    restrict_to_sources_with_pos: bool = True,
) -> dict[str, Any]:
    """
    Compute Recall@K and MRR for one edge type.
    h: dict {node_type: embeddings [N_t, d]} (same device)
    edge_type: (src_type, rel, dst_type)
    test_edges: edge_index [2, E_test]
    """
    src_t, _, dst_t = edge_type
    S = h[src_t]
    D = h[dst_t]

    device = S.device
    test_edges = test_edges.to(device)

    if not use_dot:
        S = F.normalize(S, p=2, dim=1)
        D = F.normalize(D, p=2, dim=1)

    Ns, Nd = S.size(0), D.size(0)
    gt = torch.zeros((Ns, Nd), dtype=torch.bool, device=device)
    gt[test_edges[0], test_edges[1]] = True

    if restrict_to_sources_with_pos:
        src_mask = gt.any(dim=1)
    else:
        src_mask = torch.ones(Ns, dtype=torch.bool, device=device)

    S_eval = S[src_mask]
    gt_eval = gt[src_mask]
    scores = S_eval @ D.T
    K = min(K, Nd)
    topk = torch.topk(scores, K, dim=1).indices
    hits = gt_eval.gather(1, topk)
    recall_k = hits.any(dim=1).float().mean().item()

    has = hits.any(dim=1)
    first_pos = torch.argmax(hits.int(), dim=1)
    ranks = torch.full((hits.size(0),), float("inf"), device=device)
    ranks[has] = (first_pos[has] + 1).float()
    mask = ranks != float("inf")
    mrr = (1.0 / ranks[mask]).mean().item() if mask.any() else 0.0

    return {
        "recall@K": recall_k,
        "MRR": mrr,
        "K": K,
        "n_eval_sources": int(hits.size(0)),
    }


def topk_eval_with_splits(
    device: torch.device,
    model: torch.nn.Module,
    splits: dict,
    edge_types: list,
    K: int = 20,
    use_dot: bool = True,
) -> dict[tuple, dict]:
    """Compute Recall@K and MRR for all edge types at a single K."""
    h_train = embed_with_graph(device, model, splits["train_graph"])
    results = {}
    for et in edge_types:
        res = recall_at_k_mrr(
            h_train, et, splits["test_pos"][et], K=K, use_dot=use_dot
        )
        results[et] = res
    return results


def topk_for_source(
    h: dict,
    et: tuple,
    src_id: int,
    K: int = 20,
    cosine: bool = True,
) -> tuple[list, list]:
    """Top-K predicted neighbors for a single source node. Returns (indices, scores)."""
    src_t, _, dst_t = et
    S = h[src_t]
    D = h[dst_t]
    if cosine:
        S = F.normalize(S, p=2, dim=1)
        D = F.normalize(D, p=2, dim=1)
    s = S[src_id : src_id + 1]
    scores = (s @ D.T).squeeze(0).cpu()
    vals, idxs = torch.topk(scores, min(K, D.size(0)))
    return idxs.tolist(), vals.tolist()


def _et_to_label(et: tuple) -> str:
    return "_".join(str(x) for x in et)


def plot_recall_curves(
    recall_curves: dict,
    K_list: list[int],
    output_path: str | Path,
    use_dot: bool,
) -> str:
    """
    Save one combined Recall@K plot (one line per edge type).
    Returns the saved plot path.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = plt.cm.tab20(range(len(recall_curves)))
    for (et, vals), color in zip(recall_curves.items(), colors):
        label = f"{et[0]}→{et[2]}"
        ax.plot(K_list, vals, label=label, color=color)
    ax.set_xlabel("K")
    ax.set_ylabel("Recall@K")
    ax.set_title("Recall@K (cosine)" if not use_dot else "Recall@K (dot)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize="x-small", loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return str(output_path)


def run_recall_at_k_analysis(
    device: torch.device,
    model: torch.nn.Module,
    splits: dict,
    output_dir: str | Path,
    K_list: list[int] | None = None,
    use_dot: bool = False,
) -> dict[str, Any]:
    """
    Run Recall@K analysis: compute recall curves per edge type, save plot and JSON.
    Does NOT load graph or run from disk; expects already-loaded model and splits.

    use_dot=False means cosine similarity (L2-normalized); use_dot=True means dot product.

    Returns a result dict with:
      - recall_curves: dict mapping edge_type -> list of Recall@K values (one per K in K_list)
      - mrr_at_max_k: dict mapping edge_type -> MRR at largest K (optional)
      - plot_path: path to saved recall curve plot
      - metrics_path: path to saved JSON
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if K_list is None:
        K_list = [1, 10, 20, 40, 60, 80, 100]

    edge_types = splits["sup_ets"]
    h_train = embed_with_graph(device, model, splits["train_graph"])

    recall_curves = {et: [] for et in edge_types}
    mrr_at_max_k = {}

    for et in edge_types:
        for K in K_list:
            res = recall_at_k_mrr(
                h_train, et, splits["test_pos"][et], K=K, use_dot=use_dot
            )
            recall_curves[et].append(res["recall@K"])
        res_max = recall_at_k_mrr(
            h_train, et, splits["test_pos"][et], K=max(K_list), use_dot=use_dot
        )
        mrr_at_max_k[et] = res_max["MRR"]

    # One combined plot: x=K, y=Recall@K, one line per edge type
    plot_path = plot_recall_curves(
        recall_curves=recall_curves,
        K_list=K_list,
        output_path=output_dir / "recall_at_k.png",
        use_dot=use_dot,
    )

    # Machine-readable: curves and MRR (string keys for JSON serialization)
    recall_curves_serializable = {_et_to_label(et): v for et, v in recall_curves.items()}
    mrr_at_max_k_serializable = {_et_to_label(et): v for et, v in mrr_at_max_k.items()}    
    metrics_serializable = {
        "K_list": K_list,
        "use_dot": use_dot,
        "recall_curves": recall_curves_serializable,
        "mrr_at_max_K": mrr_at_max_k_serializable,
    }
    metrics_path = output_dir / "recall_at_k_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_serializable, f, indent=2)

    return {
        "recall_curves": recall_curves_serializable,
        "mrr_at_max_k": mrr_at_max_k_serializable,
        "plot_path": str(plot_path),
        "metrics_path": str(metrics_path),
    }


def run_recall_at_k_from_run(
    data_path: str | Path,
    filename: str = "best_model.pt",
    device=None,
    output_dir: str | Path | None = None,
    K_list: list[int] | None = None,
    use_dot: bool = False,
    to_undirected: bool = True,
) -> dict[str, Any]:
    """
    Load graph from filepath via load_hetero_pt, reconstruct run via load_full_run,
    run Recall@K analysis and save outputs.

    Args:
        data_path: Path to the saved HeteroData .pt file (passed to load_hetero_pt).
        filename: Checkpoint filename under get_models_dir().
        device: Torch device; if None, auto-selected.
        output_dir: Where to save analysis outputs. If None, uses
                    <models_dir>/analysis_<stem>/recall_at_k/.
        K_list: List of K values for Recall@K. If None, uses [1, 10, 20, 40, 60, 80, 100].
        use_dot: If False, use cosine similarity; if True, use dot product.
        to_undirected: Passed to load_hetero_pt (default True).

    Returns:
        Result dict from run_recall_at_k_analysis (recall_curves, plot_path, metrics_path, etc.).
    """
    data = load_hetero_pt(path=str(Path(data_path).expanduser()), to_undirected=to_undirected)
    device = select_device(device)
    model, predictor, loaders, splits, _ = load_full_run(
        data=data, device=device, filename=filename
    )
    if output_dir is None:
        stem = Path(filename).stem
        output_dir = get_models_dir() / f"analysis_{stem}" / "recall_at_k"
    return run_recall_at_k_analysis(
        device=device,
        model=model,
        splits=splits,
        output_dir=output_dir,
        K_list=K_list,
        use_dot=use_dot,
    )
