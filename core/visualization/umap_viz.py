"""
2D UMAP projection of GNN email embeddings for visualization (ground-truth campaign colors).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

_CORE = Path(__file__).resolve().parents[1]
if str(_CORE) not in sys.path:
    sys.path.insert(0, str(_CORE))
_GNN_ROOT = _CORE / "GNN"
if str(_GNN_ROOT) not in sys.path:
    sys.path.insert(0, str(_GNN_ROOT))

from clustering.clusteringMetrics import extract_ground_truth_labels  # noqa: E402

# Distinct palette (tab20-style); cycles for many campaigns
_CAMPAIGN_PALETTE = [
    "#e6194b",
    "#3cb44b",
    "#ffe119",
    "#4363d8",
    "#f58231",
    "#911eb4",
    "#46f0f0",
    "#f032e6",
    "#bcf60c",
    "#fabebe",
    "#008080",
    "#e6beff",
    "#9a6324",
    "#fffac8",
    "#800000",
    "#aaffc3",
    "#808000",
    "#ffd8b1",
    "#000075",
    "#9a9a9a",
]

_GREY_NO_GT = "#b0b0b0"


def _color_for_campaign(
    campaign_id: Any,
    campaign_to_index: dict[Any, int],
) -> str:
    idx = campaign_to_index[campaign_id]
    return _CAMPAIGN_PALETTE[idx % len(_CAMPAIGN_PALETTE)]


def build_umap_payload(
    *,
    graph_path: str | Path,
    checkpoint_path: str | Path,
    ground_truth_path: str | Path | None,
    device_pref: str | None,
    to_undirected: bool,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42,
) -> dict[str, Any]:
    """
    Load graph + checkpoint, run email embeddings through the GNN, UMAP to 2D,
    assign colors from ground-truth campaign id when present; otherwise grey.
    """
    from src.clustering.clustering_helpers import extract_email_embeddings  # noqa: E402
    from src.load_graph_data import load_hetero_pt  # noqa: E402
    from src.model_io import load_model_checkpoint, select_device  # noqa: E402

    graph_path = Path(graph_path).expanduser().resolve()
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    meta_path = graph_path.with_suffix(".meta.json")

    if not meta_path.is_file():
        return {
            "error": f"Graph metadata not found: {meta_path}",
            "points": [],
            "legend": [],
        }

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    email_external_ids = meta.get("email_attrs", {}).get("external_id")
    if not email_external_ids:
        return {
            "error": f"No email_attrs.external_id in {meta_path}",
            "points": [],
            "legend": [],
        }

    import torch

    pref = torch.device(device_pref) if isinstance(device_pref, str) and device_pref else None
    device = select_device(pref)

    data = load_hetero_pt(str(graph_path), to_undirected=bool(to_undirected))
    model, predictor, checkpoint = load_model_checkpoint(
        device=device, metadata=data.metadata(), filename=str(checkpoint_path)
    )
    _ = predictor, checkpoint

    id_to_emb = extract_email_embeddings(
        model, data, device, email_external_ids
    )
    ordered_ids = list(email_external_ids)
    n = len(ordered_ids)
    if n < 2:
        return {
            "error": "Need at least 2 email nodes for UMAP.",
            "points": [],
            "legend": [],
        }

    X = np.stack(
        [np.asarray(id_to_emb[str(eid)], dtype=np.float64) for eid in ordered_ids],
        axis=0,
    )

    gt_map: dict[str, Any] = {}
    if ground_truth_path and str(ground_truth_path).strip():
        gt_p = Path(ground_truth_path).expanduser().resolve()
        if gt_p.is_file():
            try:
                gt_map = extract_ground_truth_labels(str(gt_p))
            except Exception as exc:
                return {
                    "error": f"Ground truth load failed: {exc}",
                    "points": [],
                    "legend": [],
                }

    # Stable color index per campaign id appearing in GT labels for emails in graph
    campaigns_in_data: set[Any] = set()
    for eid in ordered_ids:
        sid = str(eid)
        if sid in gt_map:
            campaigns_in_data.add(gt_map[sid])
    sorted_camps = sorted(campaigns_in_data, key=lambda x: (str(type(x)), str(x)))
    campaign_to_index: dict[Any, int] = {c: i for i, c in enumerate(sorted_camps)}

    try:
        import umap
    except ImportError as exc:
        return {
            "error": "umap-learn is not installed. Install with: pip install umap-learn",
            "points": [],
            "legend": [],
        }

    n_neighbors_eff = max(2, min(int(n_neighbors), n - 1))
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors_eff,
        min_dist=float(min_dist),
        random_state=random_state,
        metric="euclidean",
    )
    try:
        coords = reducer.fit_transform(X)
    except Exception as exc:
        return {
            "error": f"UMAP failed: {exc}",
            "points": [],
            "legend": [],
        }

    legend = [
        {"campaign": c, "color": _color_for_campaign(c, campaign_to_index)}
        for c in sorted_camps
    ]

    points: list[dict[str, Any]] = []
    for i, eid in enumerate(ordered_ids):
        sid = str(eid)
        has_gt = sid in gt_map
        camp = gt_map[sid] if has_gt else None
        fill = (
            _color_for_campaign(camp, campaign_to_index)
            if has_gt and camp is not None
            else _GREY_NO_GT
        )
        points.append(
            {
                "external_id": sid,
                "x": float(coords[i, 0]),
                "y": float(coords[i, 1]),
                "has_ground_truth": bool(has_gt),
                "ground_truth_campaign": camp,
                "color": fill,
            }
        )

    return {
        "error": None,
        "params": {
            "n_neighbors": n_neighbors_eff,
            "min_dist": float(min_dist),
            "metric": "euclidean",
            "random_state": random_state,
        },
        "n_emails": n,
        "embedding_dim": int(X.shape[1]) if X.size else 0,
        "points": points,
        "legend": legend,
        "no_ground_truth_color": _GREY_NO_GT,
    }


def merge_umap_into_visualization_json(
    *,
    viz_json_path: Path,
    umap_payload: dict[str, Any],
) -> None:
    """Read/write ``data.json`` and set/update the ``umap`` key."""
    viz_json_path = Path(viz_json_path)
    if not viz_json_path.is_file():
        raise FileNotFoundError(f"Visualization JSON not found: {viz_json_path}")
    with open(viz_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    data["umap"] = umap_payload
    with open(viz_json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def write_umap_projection_image(
    *,
    umap_payload: dict[str, Any],
    output_path: str | Path,
    figsize: tuple[float, float] = (11, 7),
    dpi: int = 150,
    point_size: float = 8.0,
) -> Path | None:
    """
    Save a PNG scatter plot of the UMAP layout (same colors as JSON / web UI).

    Returns the path written, or ``None`` if there is nothing to plot (error payload
    or empty points).
    """
    if umap_payload.get("error"):
        return None
    points: list[dict[str, Any]] = umap_payload.get("points") or []
    if len(points) < 1:
        return None

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import patches as mpatches

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    xs = [p["x"] for p in points]
    ys = [p["y"] for p in points]
    colors = [p.get("color") or "#888888" for p in points]

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(xs, ys, c=colors, s=point_size, alpha=0.85, linewidths=0, edgecolors="none")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title("GNN email embeddings (UMAP 2D)")
    params = umap_payload.get("params") or {}
    cap = (
        f"n={umap_payload.get('n_emails', len(points))}, "
        f"emb_dim={umap_payload.get('embedding_dim', '?')}, "
        f"n_neighbors={params.get('n_neighbors', '?')}, "
        f"min_dist={params.get('min_dist', '?')}"
    )
    ax.text(
        0.02,
        0.98,
        cap,
        transform=ax.transAxes,
        fontsize=8,
        verticalalignment="top",
        color="#444",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85, "edgecolor": "#ccc"},
    )
    ax.grid(True, alpha=0.25, linestyle="--")

    legend_rows = umap_payload.get("legend") or []
    grey = umap_payload.get("no_ground_truth_color") or _GREY_NO_GT
    if legend_rows:
        handles = [
            mpatches.Patch(color=row["color"], label=f"Campaign {row['campaign']}")
            for row in legend_rows[:40]
        ]
        handles.append(mpatches.Patch(color=grey, label="Not in ground truth"))
        ax.legend(
            handles=handles,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            fontsize=7,
            frameon=True,
            title="Ground truth",
        )
    else:
        handles = [mpatches.Patch(color=grey, label="No GT labels in file")]
        ax.legend(handles=handles, loc="upper right", fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, format="png", bbox_inches="tight")
    plt.close(fig)
    return output_path
