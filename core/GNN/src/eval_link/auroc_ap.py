"""
AUROC / Average Precision analysis for link prediction.
Lower-level: operates on already-loaded device, model, predictor, test loaders.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score

from ..model import DistMultPredictor


def collect_auroc_ap_scores(
    device: torch.device,
    model: torch.nn.Module,
    predictor: torch.nn.Module,
    loaders_test: dict,
) -> dict[tuple, dict[str, float]]:
    """
    Compute AUROC and AP per edge type using test loaders.
    Returns dict mapping edge_type -> {'auroc': float, 'ap': float}.
    """
    model.eval()
    predictor.eval()
    all_scores = {}
    with torch.no_grad():
        for et, loader in loaders_test.items():
            ys, yhats = [], []
            for batch in loader:
                batch = batch.to(device)
                h = model(batch.x_dict, batch.edge_index_dict)
                idx = batch[et].edge_label_index
                y = batch[et].edge_label.float().cpu().numpy()
                src_type, _, dst_type = et
                s = h[src_type][idx[0]]
                d = h[dst_type][idx[1]]
                if isinstance(predictor, DistMultPredictor):
                    logits = predictor(s, d, edge_type=et).cpu().numpy()
                else:
                    logits = predictor(s, d).cpu().numpy()
                ys.append(y)
                yhats.append(logits)
            y = np.concatenate(ys)
            z = np.concatenate(yhats)
            auroc = roc_auc_score(y, z)
            ap = average_precision_score(y, z)
            all_scores[et] = {"auroc": float(auroc), "ap": float(ap)}
    return all_scores


def collect_auroc_ap_scores_and_distributions(
    device: torch.device,
    model: torch.nn.Module,
    predictor: torch.nn.Module,
    loaders_test: dict,
) -> tuple[dict, dict]:
    """
    Like collect_auroc_ap_scores but also returns score arrays per edge type
    for plotting: pos_scores and neg_scores (lists of float).
    Returns (all_scores, distributions) where distributions[et] = {'pos_scores': [...], 'neg_scores': [...]}.
    """
    model.eval()
    predictor.eval()
    all_scores = {}
    distributions = {}
    with torch.no_grad():
        for et, loader in loaders_test.items():
            ys, yhats = [], []
            for batch in loader:
                batch = batch.to(device)
                h = model(batch.x_dict, batch.edge_index_dict)
                idx = batch[et].edge_label_index
                y = batch[et].edge_label.float().cpu().numpy()
                src_type, _, dst_type = et
                s = h[src_type][idx[0]]
                d = h[dst_type][idx[1]]
                if isinstance(predictor, DistMultPredictor):
                    logits = predictor(s, d, edge_type=et).cpu().numpy()
                else:
                    logits = predictor(s, d).cpu().numpy()
                ys.append(y)
                yhats.append(logits)
            y = np.concatenate(ys)
            z = np.concatenate(yhats)
            auroc = roc_auc_score(y, z)
            ap = average_precision_score(y, z)
            all_scores[et] = {"auroc": float(auroc), "ap": float(ap)}
            pos_scores = z[y >= 0.5].tolist()
            neg_scores = z[y < 0.5].tolist()
            distributions[et] = {"pos_scores": pos_scores, "neg_scores": neg_scores}
    return all_scores, distributions


def _et_to_label(et: tuple) -> str:
    """Convert edge type tuple to a safe filename/label."""
    return "_".join(str(x) for x in et)


def run_auroc_ap_analysis(
    device: torch.device,
    model: torch.nn.Module,
    predictor: torch.nn.Module,
    loaders_test: dict,
    output_dir: str | Path,
) -> dict[str, Any]:
    """
    Run AUROC/AP analysis, save per-edge-type score distribution plots and JSON metrics.
    Does NOT load graph or run from disk; expects already-loaded runtime objects.

    Returns a result dict with:
      - metrics: dict mapping edge_type (as string key) -> {auroc, ap}
      - plot_paths: list of paths to saved distribution plots (one per edge type)
      - metrics_path: path to saved JSON
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_scores, distributions = collect_auroc_ap_scores_and_distributions(
        device, model, predictor, loaders_test
    )

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_paths = []
    for et, dist in distributions.items():
        label = _et_to_label(et)
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.hist(dist["pos_scores"], bins=50, alpha=0.6, label="Positive")
        ax.hist(dist["neg_scores"], bins=50, alpha=0.6, label="Negative")
        ax.set_title(f"Score distributions for {et}")
        ax.set_xlabel("Score")
        ax.legend()
        path = output_dir / f"score_dist_{label}.png"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        plot_paths.append(str(path))

    # Machine-readable metrics: use string keys for JSON
    metrics_serializable = {_et_to_label(et): v for et, v in all_scores.items()}
    metrics_path = output_dir / "auroc_ap_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_serializable, f, indent=2)

    return {
        "metrics": all_scores,
        "plot_paths": plot_paths,
        "metrics_path": str(metrics_path),
    }


def run_auroc_ap_from_run(
    data_path: str | Path,
    filename: str = "best_model.pt",
    device=None,
    output_dir: str | Path | None = None,
    to_undirected: bool = True,
) -> dict[str, Any]:
    """
    Load graph from filepath via load_hetero_pt, reconstruct run via load_full_run,
    run AUROC/AP analysis and save outputs.

    Args:
        data_path: Path to the saved HeteroData .pt file (passed to load_hetero_pt).
        filename: Checkpoint filename under get_models_dir() (e.g. 'best_model.pt').
        device: Torch device; if None, auto-selected.
        output_dir: Where to save analysis outputs. If None, uses
                    <models_dir>/analysis_<stem>/auroc_ap/.
        to_undirected: Passed to load_hetero_pt (default True).

    Returns:
        Result dict from run_auroc_ap_analysis (metrics, plot_paths, metrics_path).
    """
    from ..load_graph_data import load_hetero_pt
    from ..model_io import get_models_dir, load_full_run, select_device

    data = load_hetero_pt(path=str(Path(data_path).expanduser()), to_undirected=to_undirected)
    device = select_device(device)
    model, predictor, loaders, splits, _ = load_full_run(
        data=data, device=device, filename=filename
    )
    if output_dir is None:
        stem = Path(filename).stem
        output_dir = get_models_dir() / f"analysis_{stem}" / "auroc_ap"
    return run_auroc_ap_analysis(
        device=device,
        model=model,
        predictor=predictor,
        loaders_test=loaders["test"],
        output_dir=output_dir,
    )
