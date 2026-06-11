"""
Embedding-map store for featureset clustering.

Lazily builds and caches ``external_id → np.ndarray`` maps for each
(feature-set name, n_components) pair on first access via :func:`get_embedding_map`.

All preprocessing settings are read from
:data:`config.pipeline_config.FEATURESET_CLUSTERING_CONFIG`; no parameters need
to be threaded through call sites.
"""

from __future__ import annotations

import json
from pathlib import Path

import config.blas_env  # noqa: F401 — must be imported before NumPy

import numpy as np

from config.pipeline_config import FEATURESET_CLUSTERING_CONFIG
from feature_set_extraction.cluster_comparison.clusteringCommonFunctions import (
    preprocess_for_clustering,
    record_cluster_id,
    remove_outliers_from_matrix,
    scale_and_normalize_matrix,
)

_PACKAGE_DIR = Path(__file__).resolve().parent.parent                                

_CACHE: dict[tuple[str, int], dict[str, np.ndarray]] = {}
_SKIP_MESSAGES: dict[tuple[str, int], str] = {}



def _featuresets_dir() -> Path:
    return _PACKAGE_DIR / "output" / "featuresets"


def _load_records(fs_name: str):
    """Return (records, path) or (None, path) when the file is missing."""
    path = _featuresets_dir() / f"{FEATURESET_CLUSTERING_CONFIG.dataset_base}-{fs_name}.json"
    if not path.exists():
        return None, path
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)[:8000], path


def _build_embedding_map(records, n_components: int) -> dict[str, np.ndarray]:
    """Preprocess records into an external_id -> embedding dict."""
    idxs = [record_cluster_id(r) for r in records]
    if FEATURESET_CLUSTERING_CONFIG.remove_outliers:
        # Extract raw features without scaling so outlier detection runs on
        # uncontaminated data, then scale the cleaned matrix.
        X, _ = preprocess_for_clustering(
            records,
            FEATURESET_CLUSTERING_CONFIG.max_tfidf_features,
            n_components=n_components,
            scaler_type="none",
            l2_normalize=False,
        )
        X, keep_mask, removed = remove_outliers_from_matrix(
            X, contamination=FEATURESET_CLUSTERING_CONFIG.outlier_contamination
        )
        idxs = [idx for idx, keep in zip(idxs, keep_mask) if keep]
        print(
            f"  Removed {removed} outliers "
            f"(contamination={FEATURESET_CLUSTERING_CONFIG.outlier_contamination})"
        )
        X = scale_and_normalize_matrix(X)
    else:
        X, _ = preprocess_for_clustering(
            records,
            FEATURESET_CLUSTERING_CONFIG.max_tfidf_features,
            n_components=n_components,
        )
    return {eid: np.asarray(vec, dtype=np.float64) for eid, vec in zip(idxs, X)}



def get_embedding_map(
    fs_name: str,
    n_components: int,
) -> tuple[dict[str, np.ndarray] | None, str | None]:
    """
    Return ``(embedding_map, None)`` for *fs_name* / *n_components*, building and
    caching it on first access.  Returns ``(None, skip_message)`` if the source
    file is missing or the resulting map is empty.
    """
    key = (fs_name, n_components)
    if key in _CACHE:
        return _CACHE[key], None
    if key in _SKIP_MESSAGES:
        return None, _SKIP_MESSAGES[key]

    records, fs_path = _load_records(fs_name)
    if records is None:
        msg = f"{fs_name}: SKIPPED (file not found: {fs_path})"
        _SKIP_MESSAGES[key] = msg
        return None, msg

    embedding_map = _build_embedding_map(records, n_components)
    if not embedding_map:
        msg = f"{fs_name}: SKIPPED (empty embedding map)"
        _SKIP_MESSAGES[key] = msg
        return None, msg

    _CACHE[key] = embedding_map
    return embedding_map, None


def warm_embedding_cache(feature_sets: list[str], n_components_values: list[int]) -> None:
    """
    Pre-build embeddings for all (feature set, n_components) combinations, logging
    progress to stdout.  Useful to front-load preprocessing before the clustering
    sweep begins so that every :func:`get_embedding_map` call during the sweep is
    a pure cache hit.
    """
    total = len(n_components_values) * len(feature_sets)
    done = 0
    print(
        f"\nBuilding embedding cache: {len(n_components_values)} n_components × "
        f"{len(feature_sets)} feature sets = {total} combinations "
        f"(shared across all clustering sweeps)\n"
    )
    for n_components in n_components_values:
        for fs_name in feature_sets:
            done += 1
            key = (fs_name, n_components)
            if key in _CACHE or key in _SKIP_MESSAGES:
                continue
            print(
                f"  [{done}/{total}] {fs_name}, n_components={n_components} "
                f"(preprocess + optional outlier removal)…"
            )
            embedding_map, skip_msg = get_embedding_map(fs_name, n_components)
            if skip_msg:
                print(f"  [{done}/{total}] {skip_msg}")
            else:
                print(
                    f"  [{done}/{total}] {fs_name}, n_components={n_components} "
                    f"→ cached {len(embedding_map)} embeddings"
                )

    n_ok = len(_CACHE)
    n_skip = len(_SKIP_MESSAGES)
    print(
        f"\nEmbedding cache ready: {n_ok} usable, {n_skip} skipped "
        f"(total keys {n_ok + n_skip}).\n"
    )
