"""
2D visualization of trained email-node embeddings (pair-supervision encoder).

Runs the HeteroSAGE encoder on the full hetero graph, optionally subsamples nodes,
projects with **t-SNE** (scikit-learn) or **UMAP** (requires ``pip install umap-learn``),
and saves a PNG under ``<run_dir>/plots/``.

Optional ground-truth JSON (``--gt-json`` or ``--gt-dir``) colors points by campaign;
emails without a label are drawn in light gray behind labeled points.

Example::

    python -m seed_candidate_workflow.utils.plot_embedding_space \\
      --run-dir output/runs/my_run \\
      --graph-pt path/to/graph_hetero.pt \\
      --method tsne --max-points 5000

    # With campaign colors (first wins across multiple GT files in a directory)::
    python -m seed_candidate_workflow.utils.plot_embedding_space \\
      --run-dir output/runs/my_run \\
      --graph-pt path/to/graph_hetero.pt \\
      --gt-dir data/groundtruth
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

_GNN_ROOT = Path(__file__).resolve().parents[2] / "core" / "GNN"
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_GNN_ROOT) not in sys.path:
    sys.path.insert(0, str(_GNN_ROOT))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from seed_candidate_workflow.utils.pair_model_inference import (  # noqa: E402
    load_pair_supervision_for_inference,
)
from seed_candidate_workflow.utils.raw_gnn_notebook import (  # noqa: E402
    load_email_external_ids,
    load_ground_truth_structures,
)


def _gt_json_paths_from_dir(gt_dir: Path, *, include_report_json: bool) -> list[Path]:
    d = gt_dir.resolve()
    if not d.is_dir():
        raise ValueError(f"--gt-dir is not a directory: {d}")
    paths = sorted(d.glob("*.json"))
    if not include_report_json:
        paths = [p for p in paths if "report" not in p.name.lower()]
    return paths


def merge_label_maps_first_wins(gt_paths: list[Path]) -> dict[str, Any]:
    """Merge GT maps: first file in sorted order wins for each external_id."""
    merged: dict[str, Any] = {}
    for p in gt_paths:
        lm, _, _ = load_ground_truth_structures(p)
        for k, v in lm.items():
            sk = str(k)
            if sk not in merged:
                merged[sk] = v
    return merged


def _resolve_gt_label_map(
    *,
    gt_json: Path | None,
    gt_dir: Path | None,
    include_report_json: bool,
) -> dict[str, Any] | None:
    if gt_json is None and gt_dir is None:
        return None
    paths: list[Path] = []
    if gt_json is not None:
        paths = [gt_json.resolve()]
    elif gt_dir is not None:
        paths = _gt_json_paths_from_dir(gt_dir, include_report_json=include_report_json)
    if not paths:
        raise ValueError("No ground-truth JSON files found.")
    return merge_label_maps_first_wins(paths)


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


def project_2d(
    X: np.ndarray,
    *,
    method: str,
    random_state: int,
    tsne_perplexity: float | None,
    umap_n_neighbors: int | None,
) -> np.ndarray:
    if X.shape[0] < 2:
        raise ValueError("Need at least 2 points for a 2D projection.")
    Xs = StandardScaler().fit_transform(X)
    n = Xs.shape[0]
    if method == "tsne":
        perp = tsne_perplexity
        if perp is None:
            perp = float(max(5, min(30, (n - 1) // 3)))
        perp = min(perp, float(n - 1) - 1e-6)
        return TSNE(
            n_components=2,
            random_state=random_state,
            perplexity=perp,
            init="pca",
            learning_rate="auto",
        ).fit_transform(Xs)
    if method == "umap":
        try:
            import umap  # type: ignore[import-untyped]
        except ImportError as e:
            raise ImportError(
                "UMAP requires the umap-learn package. Install with: pip install umap-learn"
            ) from e
        n_neighbors = umap_n_neighbors if umap_n_neighbors is not None else min(15, max(2, n - 1))
        n_neighbors = min(n_neighbors, n - 1)
        reducer = umap.UMAP(
            n_components=2,
            random_state=random_state,
            n_neighbors=n_neighbors,
            min_dist=0.1,
        )
        return reducer.fit_transform(Xs)
    raise ValueError(f"Unknown method: {method!r} (use 'tsne' or 'umap')")


def _scatter_embedding_plot(
    xy: np.ndarray,
    ids: list[str],
    label_map: dict[str, Any] | None,
    *,
    title: str,
    out_path: Path,
    dpi: int,
) -> None:
    n = len(ids)
    if xy.shape != (n, 2):
        raise ValueError("xy shape must match ids length")

    fig, ax = plt.subplots(figsize=(9, 7))
    if label_map is None:
        ax.scatter(xy[:, 0], xy[:, 1], s=10, c="#4c72b0", alpha=0.65, linewidths=0)
        ax.set_title(title)
    else:
        labels: list[Any | None] = [label_map.get(str(i)) for i in ids]
        unlabeled = np.array([L is None for L in labels], dtype=bool)
        labeled = ~unlabeled
        if unlabeled.any():
            ax.scatter(
                xy[unlabeled, 0],
                xy[unlabeled, 1],
                s=8,
                c="0.82",
                alpha=0.45,
                linewidths=0,
                label="no GT label",
                zorder=1,
            )
        if labeled.any():
            uniq_vals = sorted({labels[i] for i in range(n) if labels[i] is not None}, key=str)
            val_to_code = {v: j for j, v in enumerate(uniq_vals)}
            codes = np.array(
                [val_to_code[labels[i]] for i in range(n) if labels[i] is not None],
                dtype=np.int32,
            )
            xy_l = xy[labeled]
            n_cat = len(uniq_vals)
            cmap = plt.colormaps["nipy_spectral"].resampled(max(n_cat, 2))
            sc = ax.scatter(
                xy_l[:, 0],
                xy_l[:, 1],
                s=12,
                c=codes,
                cmap=cmap,
                vmin=0,
                vmax=max(n_cat - 1, 1),
                alpha=0.85,
                linewidths=0,
                zorder=2,
            )
            cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("campaign index (sorted)")
        ax.set_title(f"{title}\n({int(labeled.sum())} labeled / {n} points)" if label_map else title)
        if label_map is not None and unlabeled.any():
            ax.legend(loc="best", fontsize=8)

    ax.set_xlabel("dim 1")
    ax.set_ylabel("dim 2")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def write_embedding_space_plot(
    *,
    run_dir: Path | str,
    graph_pt: Path | str,
    method: str = "tsne",
    max_points: int = 8000,
    random_state: int = 42,
    plots_subdir: str = "plots",
    checkpoint_name: str = "best_model.pt",
    device: str = "cpu",
    to_undirected: bool = True,
    gt_json: Path | None = None,
    gt_dir: Path | None = None,
    gt_include_report_json: bool = False,
    dpi: int = 120,
    output_stem: str = "embedding_space",
    tsne_perplexity: float | None = None,
    umap_n_neighbors: int | None = None,
) -> dict[str, Any]:
    """
    Extract encoder email embeddings for the graph, project to 2D, write PNG under run_dir/plots_subdir.
    """
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
    sampled_ids = _subsample_ids(ids_full, max_points=max_points, random_state=random_state)
    X = np.stack([id_to_emb[eid] for eid in sampled_ids], axis=0)

    label_map = _resolve_gt_label_map(
        gt_json=gt_json,
        gt_dir=gt_dir,
        include_report_json=gt_include_report_json,
    )

    xy = project_2d(
        X,
        method=method,
        random_state=random_state,
        tsne_perplexity=tsne_perplexity,
        umap_n_neighbors=umap_n_neighbors,
    )

    plots_dir = run / plots_subdir
    out_png = plots_dir / f"{output_stem}_{method}.png"
    title = f"Email embeddings ({method.upper()})\n{run.name}"
    _scatter_embedding_plot(xy, sampled_ids, label_map, title=title, out_path=out_png, dpi=dpi)

    meta_out = plots_dir / f"{output_stem}_{method}.meta.json"
    gt_ref: str | None = None
    if gt_json is not None:
        gt_ref = str(Path(gt_json).resolve())
    elif gt_dir is not None:
        gt_ref = str(Path(gt_dir).resolve())
    payload = {
        "run_dir": str(run),
        "graph_pt": str(gpath),
        "method": method,
        "n_points": len(sampled_ids),
        "max_points_cap": max_points,
        "random_state": random_state,
        "checkpoint": checkpoint_name,
        "plot_path": str(out_png),
        "ground_truth_source": gt_ref,
        "n_gt_labels_used": len(label_map) if label_map is not None else 0,
    }
    with open(meta_out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return {
        "run_dir": run,
        "plot_path": out_png,
        "meta_path": meta_out,
        "n_points": len(sampled_ids),
        "method": method,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Plot 2D embedding space (t-SNE or UMAP) from a pair-supervision run.",
    )
    p.add_argument("--run-dir", type=Path, required=True, help="Training run directory (models + training_config.json)")
    p.add_argument("--graph-pt", type=Path, required=True, help="HeteroData .pt used when training")
    p.add_argument("--method", choices=("tsne", "umap"), default="tsne", help="Dimensionality reduction (default: tsne)")
    p.add_argument("--max-points", type=int, default=8000, help="Random subsample size for speed (default: 8000)")
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--plots-subdir", type=str, default="plots")
    p.add_argument("--checkpoint", type=str, default="best_model.pt")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--dpi", type=int, default=120)
    p.add_argument("--output-stem", type=str, default="embedding_space", help="Output filenames: <stem>_<method>.png")
    p.add_argument(
        "--tsne-perplexity",
        type=float,
        default=None,
        help="Override t-SNE perplexity (default: min(30, (n-1)/3), capped by n)",
    )
    p.add_argument(
        "--umap-n-neighbors",
        type=int,
        default=None,
        help="Override UMAP n_neighbors (default: min(15, n-1))",
    )
    p.add_argument("--gt-json", type=Path, default=None, help="Single ground-truth JSON for campaign coloring")
    p.add_argument(
        "--gt-dir",
        type=Path,
        default=None,
        help="Directory of *.json; merge labels with first-wins (skips *report* unless flag below)",
    )
    p.add_argument(
        "--gt-include-report-json",
        action="store_true",
        help="With --gt-dir, include filenames containing 'report'",
    )
    p.add_argument(
        "--no-to-undirected",
        action="store_true",
        help="Load graph without ToUndirected (default matches training: undirected)",
    )
    args = p.parse_args(argv)

    if args.gt_json is not None and args.gt_dir is not None:
        p.error("Use only one of --gt-json or --gt-dir")

    out = write_embedding_space_plot(
        run_dir=args.run_dir,
        graph_pt=args.graph_pt,
        method=args.method,
        max_points=args.max_points,
        random_state=args.random_state,
        plots_subdir=args.plots_subdir,
        checkpoint_name=args.checkpoint,
        device=args.device,
        to_undirected=not args.no_to_undirected,
        gt_json=args.gt_json,
        gt_dir=args.gt_dir,
        gt_include_report_json=bool(args.gt_include_report_json),
        dpi=args.dpi,
        output_stem=args.output_stem,
        tsne_perplexity=args.tsne_perplexity,
        umap_n_neighbors=args.umap_n_neighbors,
    )
    print(f"Wrote {out['plot_path']}")
    print(f"Wrote {out['meta_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
