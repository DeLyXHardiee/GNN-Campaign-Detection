"""
Stage-3 evaluation helpers: HDBSCAN + external metrics for email contrastive runs.

GT is only passed into metric computation (see clusteringMetrics.compute_all_metrics).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from core.clustering.clusteringMetrics import _emb_matrix_from_id_to_embedding, compute_all_metrics


def load_embedding_meta_and_array(
    run_dir: Path | str, stem: str
) -> tuple[list[dict[str, Any]], np.ndarray] | None:
    """
    Load ``embedding_meta.json`` and ``{stem}.npy`` (e.g. ``embeddings_best_val``).

    Returns None if either file is missing.
    """
    run_dir = Path(run_dir)
    meta_path = run_dir / "embedding_meta.json"
    arr_path = run_dir / f"{stem}.npy"
    if not meta_path.is_file() or not arr_path.is_file():
        return None
    meta: list[dict[str, Any]] = json.loads(meta_path.read_text(encoding="utf-8"))
    X = np.load(arr_path)
    if len(meta) != len(X):
        raise ValueError(f"{stem}: embedding_meta rows {len(meta)} != {len(X)} rows in npy")
    return meta, X


def meta_matrix_to_id_to_emb(
    meta: list[dict[str, Any]],
    X: np.ndarray,
    mask: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Map external_id -> row vector; optional boolean mask aligned with meta order."""
    out: dict[str, np.ndarray] = {}
    for i, row in enumerate(meta):
        if mask is not None and not bool(mask[i]):
            continue
        r = int(row.get("row", i))
        eid = str(row["external_id"]).strip()
        if not eid:
            continue
        out[eid] = np.asarray(X[r], dtype=np.float64)
    return out


def matrix_clustering_sanity(X: np.ndarray, *, tag: str = "") -> dict[str, Any]:
    """
    Quick checks before HDBSCAN: finiteness, shape, row L2-norm distribution (after preprocessing, norms matter for cosine-style runs).
    """
    X = np.asarray(X, dtype=np.float64)
    n, d = X.shape
    finite = bool(np.isfinite(X).all())
    norms = np.linalg.norm(X, axis=1)
    return {
        "tag": tag,
        "n_rows": int(n),
        "n_dim": int(d),
        "all_finite": finite,
        "row_l2_norm_p10": float(np.percentile(norms, 10)) if n else float("nan"),
        "row_l2_norm_p50": float(np.percentile(norms, 50)) if n else float("nan"),
        "row_l2_norm_p90": float(np.percentile(norms, 90)) if n else float("nan"),
        "n_near_zero_rows": int((norms < 1e-10).sum()),
    }


def gt_id_set_overlap(
    id_to_embedding_map: dict[str, np.ndarray],
    label_map: dict[str, Any],
) -> dict[str, Any]:
    """Diagnostics: why ``n_eval`` can be 0 (ID join mismatch vs empty GT)."""
    ke = {str(k).strip() for k in id_to_embedding_map if str(k).strip()}
    kg = {str(k).strip() for k in label_map if str(k).strip()}
    inter = ke & kg
    only_e = sorted(ke - kg)[:5]
    only_g = sorted(kg - ke)[:5]
    return {
        "n_gt_labeled_emails": len(kg),
        "n_embedding_ids": len(ke),
        "n_intersection": len(inter),
        "sample_ids_in_embeddings_not_in_gt": only_e,
        "sample_ids_in_gt_not_in_embeddings": only_g,
    }


def hdbscan_evaluate(
    id_to_embedding_map: dict[str, np.ndarray],
    ground_truth_labels: dict[str, Any],
    *,
    min_cluster_size: int,
    min_samples: int | None,
    variant: str,
    metric: str = "euclidean",
    l2_normalize_rows: bool = False,
    standardize_columns: bool = False,
    cluster_selection_epsilon: float = 0.0,
    allow_single_cluster: bool = False,
    cluster_selection_method: str = "eom",
) -> tuple[dict[str, Any], np.ndarray, list[str]]:
    """
    Fit HDBSCAN on sorted embedding matrix, then compute packaged metrics + return labels / id order.

    For high-dimensional vectors (e.g. 128-d student embeddings), **euclidean** often yields all-noise
    clusters; prefer ``metric='cosine'`` and/or ``l2_normalize_rows=True``.

    **Cosine note:** some scikit-learn builds do not expose ``'cosine'`` to the fast distance path that
    ``hdbscan`` uses, which raises ``ValueError: Unrecognized metric 'cosine'``. In that case we
    **L2-normalize rows** and run HDBSCAN with ``metric='euclidean'`` (monotone with cosine distance on
    the sphere for pairwise ordering).

    ``standardize_columns`` (optional): z-score each column **before** row L2 / cosine handling. Use the
    same flag for all variants if you want a fair comparison; it can reduce HDBSCAN noise when raw
    feature dimensions have very different scales (less often needed for already projected SBERT blocks).

    ``cluster_selection_method``: ``\"eom\"`` (library default) vs ``\"leaf\"``. ``leaf`` often assigns
    fewer points as noise (more, smaller clusters) on embedding-like data.
    """
    import hdbscan  # type: ignore

    sorted_ids, embeddings = _emb_matrix_from_id_to_embedding(id_to_embedding_map)
    X = np.asarray(embeddings, dtype=np.float64, order="C")
    if standardize_columns:
        from sklearn.preprocessing import StandardScaler

        X = StandardScaler().fit_transform(X)
    metric_s = str(metric).strip().lower()
    want_cosine = metric_s in ("cosine", "cosine_distance")
    do_l2 = bool(l2_normalize_rows or want_cosine)
    if do_l2:
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        np.maximum(norms, 1e-12, out=norms)
        X = X / norms
    # Cosine → euclidean on unit vectors (avoids sklearn DistanceMetric 'cosine' KeyError in some envs).
    metric_for_hdbscan = "euclidean" if want_cosine else metric_s
    csm = str(cluster_selection_method).strip().lower()
    if csm not in ("eom", "leaf"):
        raise ValueError(f"cluster_selection_method must be 'eom' or 'leaf', got {cluster_selection_method!r}")
    base_kw: dict[str, Any] = dict(
        min_cluster_size=int(min_cluster_size),
        min_samples=None if min_samples is None else int(min_samples),
        metric=metric_for_hdbscan,
        cluster_selection_epsilon=float(cluster_selection_epsilon),
        allow_single_cluster=bool(allow_single_cluster),
    )
    try:
        clusterer = hdbscan.HDBSCAN(**base_kw, cluster_selection_method=csm)
    except TypeError:
        # Older hdbscan builds may omit cluster_selection_method
        clusterer = hdbscan.HDBSCAN(**base_kw)
        csm = "eom (fallback)"
    labels = clusterer.fit_predict(X)
    m = compute_all_metrics(id_to_embedding_map, labels, ground_truth_labels)
    m.update(
        {
            "clustering_type": "hdbscan",
            "min_cluster_size": int(min_cluster_size),
            "min_samples": None if min_samples is None else int(min_samples),
            "n_embeddings": int(len(sorted_ids)),
            "variant": variant,
            "hdbscan_metric": str(metric),
            "hdbscan_metric_used": metric_for_hdbscan,
            "l2_normalize_rows": do_l2,
            "standardize_columns": bool(standardize_columns),
            "cluster_selection_epsilon": float(cluster_selection_epsilon),
            "allow_single_cluster": bool(allow_single_cluster),
            "cluster_selection_method": str(csm),
            "n_eval": int(m["n_samples"]),
            "coverage_gt": float(m["coverage_ground_truth"]),
            "coverage_assignments": float(m["coverage_all"]),
        }
    )
    return m, np.asarray(labels, dtype=np.int64), sorted_ids


def load_graph_paths_from_feature_info(run_dir: Path | str) -> tuple[Path, Path, str]:
    """Read ``feature_load_info.json`` (stage 2) for graph checkpoint + resolved feature mode."""
    run_dir = Path(run_dir)
    p = run_dir / "feature_load_info.json"
    if not p.is_file():
        raise FileNotFoundError(f"Missing {p} (run stage 2 from this RUN_DIR first)")
    raw = json.loads(p.read_text(encoding="utf-8"))
    tr = raw.get("train") or {}
    graph_pt = Path(tr["graph_pt"])
    meta_json = Path(tr["meta_json"])
    mode = str(tr.get("feature_mode_resolved", "auto"))
    return graph_pt, meta_json, mode


def infer_student_embeddings_full_graph(
    run_dir: Path | str,
    *,
    graph_pt: Path | str,
    meta_json: Path | str | None,
    external_ids: list[str],
    feature_mode: str,
    checkpoint: str = "best",
    batch_size: int = 4096,
    device: str | None = None,
    to_undirected: bool = True,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """
    Run the saved student encoder (residual or legacy MLP from ``config_train.json``) on graph
    email features for each listed ``external_id``.

    Training ``embedding_meta.json`` / ``embeddings_*.npy`` often cover only a no–GT train/val split.
    This path loads the **same** feature slice as stage 2 for **all** graph emails and applies the
    checkpoint, enabling HDBSCAN + GT metrics on held-out labels without retraining.
    """
    import torch

    from analysis.utils.email_teacher_contrastive_features import load_graph_email_features_for_external_ids
    from analysis.utils.email_teacher_contrastive_train import build_student_from_train_config, export_embeddings

    run_dir = Path(run_dir)
    if checkpoint == "best":
        ck_path = run_dir / "checkpoint_best.pt"
    elif checkpoint == "final":
        ck_path = run_dir / "checkpoint_final.pt"
    else:
        raise ValueError(f"checkpoint must be 'best' or 'final', got {checkpoint!r}")
    if not ck_path.is_file():
        raise FileNotFoundError(f"Missing {ck_path}")

    cfg_path = run_dir / "config_train.json"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Missing {cfg_path} (need training hyperparameters for MLP shape)")

    X, mask, finfo = load_graph_email_features_for_external_ids(
        graph_pt,
        meta_json,
        external_ids,
        feature_mode=feature_mode,
        to_undirected=to_undirected,
    )
    input_dim = int(X.shape[1])
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    model = build_student_from_train_config(cfg, input_dim)

    map_dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dev = torch.device(map_dev)
    try:
        blob = torch.load(ck_path, map_location=dev, weights_only=False)
    except TypeError:
        blob = torch.load(ck_path, map_location=dev)
    model.load_state_dict(blob["model_state_dict"])
    model.to(dev).eval()

    present = np.asarray(mask, dtype=bool)
    if not np.any(present):
        return {}, {
            "checkpoint": str(ck_path),
            "n_requested": int(len(external_ids)),
            "n_present": 0,
            "n_missing": int(finfo.get("n_missing", 0)),
        }

    x_np = np.asarray(X[present], dtype=np.float32)
    eids_present = [str(external_ids[i]) for i in np.flatnonzero(present)]
    x_t = torch.from_numpy(x_np).float()
    emb = export_embeddings(model, dev, x_t, batch_size=int(batch_size))

    out: dict[str, np.ndarray] = {}
    for i, eid in enumerate(eids_present):
        out[eid] = np.asarray(emb[i], dtype=np.float64)

    info: dict[str, Any] = {
        "checkpoint": str(ck_path.resolve()),
        "device": str(dev),
        "input_dim": input_dim,
        "feature_mode_resolved": finfo.get("feature_mode_resolved"),
        "n_requested": int(len(external_ids)),
        "n_present": int(present.sum()),
        "n_missing": int(finfo.get("n_missing", 0)),
        "embedding_dim": int(emb.shape[1]) if emb.size else 0,
    }
    return out, info


def cluster_size_counts(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Counts per predicted cluster id (excludes noise -1). Returns (cluster_ids, counts)."""
    labs = np.asarray(labels)
    noise = labs == -1
    usable = labs[~noise]
    if usable.size == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    uniq, counts = np.unique(usable, return_counts=True)
    order = np.argsort(-counts)
    return uniq[order], counts[order]


def cosine_same_vs_diff_campaign(
    id_to_emb: dict[str, np.ndarray],
    label_map: dict[str, Any],
    *,
    rng: np.random.Generator,
    max_points: int = 2500,
) -> dict[str, float]:
    """Mean cosine similarity on upper triangle: same-GT pairs vs different-GT pairs."""
    ids = sorted([eid for eid in id_to_emb if eid in label_map], key=str)
    if len(ids) < 2:
        return {"mean_cos_same": float("nan"), "mean_cos_diff": float("nan"), "n_points_used": 0.0}
    if len(ids) > max_points:
        pick = rng.choice(np.asarray(ids, dtype=object), size=max_points, replace=False)
        ids = sorted([str(x) for x in pick], key=str)
    X = np.stack([id_to_emb[i] for i in ids], axis=0).astype(np.float64, copy=False)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1e-12, norms)
    Xn = X / norms
    sim = Xn @ Xn.T
    labs = np.array([label_map[i] for i in ids])
    iu = np.triu_indices(sim.shape[0], k=1)
    s = sim[iu]
    same_mask = labs[iu[0]] == labs[iu[1]]
    n_same = int(same_mask.sum())
    n_diff = int((~same_mask).sum())
    return {
        "mean_cos_same": float(np.mean(s[same_mask])) if n_same else float("nan"),
        "mean_cos_diff": float(np.mean(s[~same_mask])) if n_diff else float("nan"),
        "n_points_used": float(len(ids)),
        "n_pairs_same": float(n_same),
        "n_pairs_diff": float(n_diff),
    }
