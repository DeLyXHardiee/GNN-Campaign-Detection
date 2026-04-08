"""
Pre-cluster emails using concatenated subject + body embedding vectors.

Used by the assembler when ``email_preclustering`` is enabled in pipeline config.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np

from config.pipeline_config import EmailPreclusteringSettings


@dataclass(frozen=True)
class PreclusterResult:
    """Per-email cluster id in ``0 .. n_clusters-1`` after noise relabeling."""

    labels: np.ndarray  # int64, shape (n_emails,)
    n_clusters: int
    raw_noise_mask: np.ndarray  # bool: True where algorithm assigned noise before singleton policy


def _stack_vectors(
    subj_vecs: Sequence[Sequence[float]],
    body_vecs: Sequence[Sequence[float]],
    subj_dim: int,
    body_dim: int,
) -> np.ndarray:
    n = max(len(subj_vecs), len(body_vecs))
    if n == 0:
        return np.zeros((0, subj_dim + body_dim), dtype=np.float64)
    rows: List[List[float]] = []
    for i in range(n):
        s = list(subj_vecs[i]) if i < len(subj_vecs) else [0.0] * subj_dim
        b = list(body_vecs[i]) if i < len(body_vecs) else [0.0] * body_dim
        if len(s) < subj_dim:
            s = s + [0.0] * (subj_dim - len(s))
        if len(b) < body_dim:
            b = b + [0.0] * (body_dim - len(b))
        rows.append(s[:subj_dim] + b[:body_dim])
    return np.asarray(rows, dtype=np.float64)


def _remap_contiguous(labels: np.ndarray) -> tuple[np.ndarray, int]:
    if labels.size == 0:
        return labels.astype(np.int64), 0
    uniq = np.unique(labels)
    mapping = {int(v): i for i, v in enumerate(uniq)}
    out = np.array([mapping[int(x)] for x in labels], dtype=np.int64)
    return out, len(uniq)


def _apply_singleton_noise_policy(labels: np.ndarray, _rng: np.random.Generator) -> tuple[np.ndarray, int]:
    """Map HDBSCAN/DBSCAN noise (-1) to unique cluster ids."""
    out = labels.astype(np.int64, copy=True)
    mask = out == -1
    max_lab = int(out.max()) if out.size else -1
    noise_idx = np.where(mask)[0]
    next_id = max_lab + 1
    for _pos, j in enumerate(noise_idx.tolist()):
        out[j] = next_id
        next_id += 1
    n_clusters = int(out.max()) + 1 if out.size else 0
    return out, n_clusters


def precluster_email_embeddings(
    subj_vecs: List[List[float]],
    body_vecs: List[List[float]],
    subj_dim: int,
    body_dim: int,
    settings: EmailPreclusteringSettings,
) -> PreclusterResult:
    """
    Cluster rows of [subj || body] with HDBSCAN or DBSCAN.

    When ``noise_policy`` is ``singleton``, each noise point becomes its own cluster id
    so every email retains a graph node (via its cluster).
    """
    rng = np.random.default_rng(int(settings.random_seed))
    X = _stack_vectors(subj_vecs, body_vecs, subj_dim, body_dim)
    n = X.shape[0]
    if n == 0:
        return PreclusterResult(
            labels=np.zeros((0,), dtype=np.int64),
            n_clusters=0,
            raw_noise_mask=np.zeros((0,), dtype=bool),
        )

    if settings.l2_normalize_embedding:
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        X = X / norms

    raw_noise = np.zeros((n,), dtype=bool)
    algo = settings.algorithm

    if algo == "hdbscan":
        import hdbscan

        min_samples = settings.min_samples if settings.min_samples is not None else max(1, settings.min_cluster_size)
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=int(settings.min_cluster_size),
            min_samples=int(min_samples),
            metric="euclidean",
            core_dist_n_jobs=-1,
        )
        labels = clusterer.fit_predict(X).astype(np.int64)
        raw_noise = labels == -1
    elif algo == "dbscan":
        from sklearn.cluster import DBSCAN

        clusterer = DBSCAN(
            eps=float(settings.dbscan_eps),
            min_samples=int(settings.dbscan_min_samples),
            metric="euclidean",
            n_jobs=-1,
        )
        labels = clusterer.fit_predict(X).astype(np.int64)
        raw_noise = labels == -1
    else:
        raise ValueError(f"Unknown preclustering algorithm: {algo!r}")

    if settings.noise_policy == "singleton":
        labels_final, _n = _apply_singleton_noise_policy(labels, rng)
    else:
        labels_final = labels

    # Dense ids 0..K-1 for downstream node indexing
    labels_final, n_clusters = _remap_contiguous(labels_final)

    return PreclusterResult(
        labels=labels_final,
        n_clusters=n_clusters,
        raw_noise_mask=raw_noise,
    )
