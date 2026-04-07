"""
Step 5: evaluate downstream use of refined shard-graph GNN embeddings (clustering + edge refinement).

Does not retrain the GNN — consumes Step-4 saved artifacts only.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from analysis.utils import semantic_shard_step3_helpers as s3


def load_step4_bundle(step4_dir: str | Path) -> dict[str, Any]:
    """Load refined embeddings, shard id order, and training config JSON."""
    d = Path(step4_dir).expanduser().resolve()
    emb_p = d / "semantic_shard_step4_refined_embeddings.npy"
    sid_p = d / "semantic_shard_step4_shard_ids.json"
    cfg_p = d / "semantic_shard_step4_train_config.json"
    feat_p = d / "semantic_shard_step4_feature_schema.json"
    if not emb_p.is_file():
        raise FileNotFoundError(f"Missing Step-4 embeddings: {emb_p}")
    if not sid_p.is_file():
        raise FileNotFoundError(f"Missing Step-4 shard ids: {sid_p}")
    Z = np.load(emb_p)
    shard_ids: list[str] = json.loads(sid_p.read_text(encoding="utf-8"))
    cfg = json.loads(cfg_p.read_text(encoding="utf-8")) if cfg_p.is_file() else {}
    feat = json.loads(feat_p.read_text(encoding="utf-8")) if feat_p.is_file() else {}
    model_p = d / "semantic_shard_step4_model.pt"
    return {
        "dir": str(d),
        "Z": Z.astype(np.float32),
        "shard_ids": [str(s) for s in shard_ids],
        "train_config": cfg,
        "feature_schema": feat,
        "model_pt": str(model_p) if model_p.is_file() else None,
    }


def l2_normalize_rows(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0.0] = 1.0
    return X / n


def ensure_no_zero_embedding_rows(X: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    """
    Sklearn AgglomerativeClustering(metric='cosine') rejects all-zero rows.
    Refined GNN rows can be exactly zero; replace those rows with a harmless unit direction.
    """
    X = np.asarray(X, dtype=np.float64, copy=True)
    norms = np.linalg.norm(X, axis=1)
    mask = norms < float(eps)
    if np.any(mask):
        d = X.shape[1]
        fill = np.ones(d, dtype=np.float64) / np.sqrt(max(d, 1))
        X[mask, :] = fill
    return X


def prepare_normalized_shard_embeddings(Z: np.ndarray) -> np.ndarray:
    """L2-normalize shard rows and fix zero vectors for cosine/clustering stability."""
    return ensure_no_zero_embedding_rows(l2_normalize_rows(Z))


def validate_embedding_alignment(
    *,
    assignments_df: pd.DataFrame,
    shard_ids_step4: list[str],
    Z: np.ndarray,
) -> dict[str, Any]:
    """Check coverage: every shard in assignments should appear in Step-4 shard list."""
    adf = assignments_df.copy()
    adf["shard_id"] = adf["shard_id"].astype(str)
    shards_in_assign = set(adf["shard_id"].unique())
    shards_in_emb = set(shard_ids_step4)
    missing_emb = sorted(shards_in_assign - shards_in_emb)
    extra_emb = sorted(shards_in_emb - shards_in_assign)
    ok = len(missing_emb) == 0 and len(Z) == len(shard_ids_step4)
    return {
        "n_rows_Z": int(Z.shape[0]),
        "embed_dim": int(Z.shape[1]) if Z.ndim == 2 else 0,
        "n_unique_shards_assignments": len(shards_in_assign),
        "n_shards_step4": len(shard_ids_step4),
        "missing_shards_in_step4": missing_emb[:50],
        "n_missing_shards_in_step4": len(missing_emb),
        "n_extra_shards_only_in_step4": len(extra_emb),
        "aligned": bool(ok),
    }


def shard_id_to_index_map(shard_ids: list[str]) -> dict[str, int]:
    return {str(s): i for i, s in enumerate(shard_ids)}


def edge_cosine_similarities(
    edges_df: pd.DataFrame,
    Z_norm: np.ndarray,
    sid_to_idx: dict[str, int],
) -> np.ndarray:
    """Per-row cosine similarity for (shard_a, shard_b) using L2-normalized Z."""
    cos = np.zeros(len(edges_df), dtype=np.float64)
    ia = edges_df["shard_a"].astype(str).map(sid_to_idx).to_numpy()
    ib = edges_df["shard_b"].astype(str).map(sid_to_idx).to_numpy()
    for k, (i, j) in enumerate(zip(ia, ib, strict=False)):
        if pd.isna(i) or pd.isna(j):
            cos[k] = 0.0
            continue
        ii, jj = int(i), int(j)
        cos[k] = float(np.dot(Z_norm[ii], Z_norm[jj]))
    return cos


def refine_edge_weights(
    edge_weight: np.ndarray,
    cos_sim: np.ndarray,
    *,
    alpha: float,
    sim_mode: str = "cosine_clipped01",
    weight_scale: str = "none",
) -> np.ndarray:
    """
    new_score = alpha * old_edge_weight + (1 - alpha) * embedding_similarity_term

    sim_mode:
      - cosine_clipped01: similarity = clip((1+cos)/2, 0, 1) so it mixes cleanly with nonnegative weights
      - cosine_relu: similarity = max(0, cos)

    weight_scale:
      - none: use raw edge weights (legacy; similarity term often dominated when weights sit in a narrow band)
      - minmax: linearly scale edge weights to [0, 1] before blending so alpha interpolates fairly vs sim_term
    """
    w = np.asarray(edge_weight, dtype=np.float64)
    c = np.asarray(cos_sim, dtype=np.float64)
    ws = str(weight_scale).lower().strip()
    if ws == "minmax":
        wmin, wmax = float(w.min()), float(w.max())
        if wmax - wmin > 1e-12:
            w = (w - wmin) / (wmax - wmin)
        else:
            w = np.zeros_like(w)
    elif ws != "none":
        raise ValueError(f"Unknown weight_scale: {weight_scale!r}")
    if sim_mode == "cosine_clipped01":
        sim = np.clip(0.5 * (1.0 + c), 0.0, 1.0)
    elif sim_mode == "cosine_relu":
        sim = np.clip(c, 0.0, 1.0)
    else:
        raise ValueError(f"Unknown sim_mode: {sim_mode!r}")
    a = float(alpha)
    return (a * w + (1.0 - a) * sim).astype(np.float64)


def disambiguate_hdbscan_noise(
    labels: np.ndarray,
    *,
    mode: str = "single_cluster",
) -> np.ndarray:
    """
    HDBSCAN uses -1 for noise.

    - single_cluster (default): all noise shards share one extra cluster id (avoids exploding
      predicted communities — per-singleton noise was crushing homogeneity / completeness).
    - singleton: legacy behavior — each noise point becomes its own cluster id.
    """
    lab = np.asarray(labels, dtype=np.int64).copy()
    m = str(mode).lower().strip()
    if m == "single_cluster":
        noise = lab == -1
        if not np.any(noise):
            return lab
        new_id = int(lab.max()) + 1
        lab[noise] = new_id
        return lab
    if m == "singleton":
        nxt = int(lab.max()) + 1 if lab.size else 0
        for i in range(len(lab)):
            if lab[i] == -1:
                lab[i] = nxt
                nxt += 1
        return lab
    raise ValueError(f"Unknown disambiguate mode: {mode!r}")


def cluster_shards_hdbscan(
    Z: np.ndarray,
    *,
    min_cluster_size: int,
    min_samples: int,
    metric: str = "euclidean",
) -> tuple[np.ndarray, str]:
    try:
        import hdbscan  # type: ignore

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=int(min_cluster_size),
            min_samples=int(min_samples),
            metric=str(metric),
        )
        labels = clusterer.fit_predict(np.asarray(Z, dtype=np.float64))
        return np.asarray(labels, dtype=np.int64), f"hdbscan_{metric}"
    except Exception as e:
        raise RuntimeError(f"HDBSCAN failed: {e}") from e


def cluster_shards_agglomerative_cosine(
    Z_norm: np.ndarray,
    *,
    distance_threshold: float,
) -> tuple[np.ndarray, str]:
    from sklearn.cluster import AgglomerativeClustering

    n = len(Z_norm)
    if n < 2:
        return np.zeros(n, dtype=np.int64), "agglomerative_cosine"
    X = ensure_no_zero_embedding_rows(np.asarray(Z_norm, dtype=np.float64))
    # linkage average + cosine distance on L2-normalized points
    clus = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=float(distance_threshold),
        metric="cosine",
        linkage="average",
    )
    labels = clus.fit_predict(X)
    return np.asarray(labels, dtype=np.int64), "agglomerative_cosine"


def labels_to_shard_comm_map(shard_ids: list[str], labels: np.ndarray) -> dict[str, int]:
    out: dict[str, int] = {}
    for sid, lab in zip(shard_ids, np.asarray(labels), strict=False):
        out[str(sid)] = int(lab)
    return out


def email_pred_from_shard_labels(
    assignments_df: pd.DataFrame,
    shard_to_label: dict[str, int],
    *,
    pred_col: str = "pred_community",
) -> pd.DataFrame:
    """Map shard-level cluster ids to emails (same pattern as Step-3 community map)."""
    out = assignments_df.copy()
    out["external_id"] = out["external_id"].astype(str)
    out["shard_id"] = out["shard_id"].astype(str)
    out[pred_col] = out["shard_id"].map(lambda s: int(shard_to_label.get(str(s), -1)))
    return out


def sweep_path_a_embedding_clusters(
    *,
    Z: np.ndarray,
    shard_ids: list[str],
    assignments_df: pd.DataFrame,
    gt_label_map: dict[str, Any],
    hdbscan_grid: list[tuple[int, int]] | None = None,
    hdbscan_metric: str = "euclidean",
    hdbscan_noise_mode: str = "single_cluster",
    agg_thresholds: list[float] | None = None,
    pred_col: str = "pred_shard_cluster",
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, dict[str, int]]]:
    """
    Path A: cluster shard rows of Z; map to emails; evaluate with sklearn metrics.

    **HDBSCAN is required** for Path A (install `hdbscan`). Supplemental agglomerative
    sweeps are optional diagnostics after HDBSCAN settings are evaluated.

    Use ``hdbscan_noise_mode="single_cluster"`` (default) so outliers do not each become
    their own community (which destroys homogeneity/completeness).
    """
    try:
        import hdbscan  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "Path A requires `hdbscan`. Install with: pip install hdbscan"
        ) from e

    Zn = prepare_normalized_shard_embeddings(Z)
    rows: list[dict[str, Any]] = []
    email_by_key: dict[str, pd.DataFrame] = {}
    shard_map_by_key: dict[str, dict[str, int]] = {}

    if hdbscan_grid is None:
        hdbscan_grid = [(5, 3), (8, 3), (10, 5), (15, 5), (20, 8)]

    hdbscan_ok = 0
    for mcs, ms in hdbscan_grid:
        name = f"A_hdbscan_mcs{mcs}_ms{ms}"
        try:
            labels, backend = cluster_shards_hdbscan(
                Zn,
                min_cluster_size=mcs,
                min_samples=ms,
                metric=hdbscan_metric,
            )
            labels = disambiguate_hdbscan_noise(labels, mode=hdbscan_noise_mode)
            hdbscan_ok += 1
        except Exception as ex:
            warnings.warn(f"HDBSCAN skipped for {name}: {ex}", UserWarning, stacklevel=2)
            continue
        smap = labels_to_shard_comm_map(shard_ids, labels)
        email_df = email_pred_from_shard_labels(assignments_df, smap, pred_col=pred_col)
        m = s3.evaluate_external_metrics(
            email_df.rename(columns={pred_col: "pred_community"}), gt_label_map
        )
        rows.append(
            {
                "path": "A_embedding_cluster",
                "setting_key": name,
                "backend": backend,
                "n_shard_clusters": int(len(set(smap.values()))),
                "n_pred_communities_email": int(
                    email_df[pred_col].nunique(),
                ),
                **m,
            }
        )
        email_by_key[name] = email_df.copy()
        shard_map_by_key[name] = dict(smap)

    if hdbscan_ok == 0:
        raise RuntimeError(
            "No HDBSCAN runs succeeded; check min_cluster_size/min_samples vs n_shards or embedding variance."
        )

    if agg_thresholds is None:
        agg_thresholds = [0.25, 0.5, 0.75, 1.0]

    for th in agg_thresholds:
        name = f"A_agg_cos_dt{th:.2f}"
        try:
            labels, backend = cluster_shards_agglomerative_cosine(Zn, distance_threshold=th)
        except Exception as ex:
            warnings.warn(f"Agglomerative skipped for {name}: {ex}", UserWarning, stacklevel=2)
            continue
        smap = labels_to_shard_comm_map(shard_ids, labels)
        email_df = email_pred_from_shard_labels(assignments_df, smap, pred_col=pred_col)
        m = s3.evaluate_external_metrics(
            email_df.rename(columns={pred_col: "pred_community"}), gt_label_map
        )
        rows.append(
            {
                "path": "A_embedding_cluster",
                "setting_key": name,
                "backend": backend,
                "n_shard_clusters": int(len(set(smap.values()))),
                "n_pred_communities_email": int(email_df[pred_col].nunique()),
                **m,
            }
        )
        email_by_key[name] = email_df.copy()
        shard_map_by_key[name] = dict(smap)

    return pd.DataFrame(rows), email_by_key, shard_map_by_key


def sweep_path_b_refined_edges(
    *,
    edges_df: pd.DataFrame,
    shard_ids: list[str],
    Z: np.ndarray,
    assignments_df: pd.DataFrame,
    gt_label_map: dict[str, Any],
    alphas: list[float],
    sim_mode: str,
    resolutions: list[float],
    min_edge_weights: list[float],
    refine_weight_scale: str = "none",
    cd_method: str = "louvain",
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    """
    Path B: refine edge_weight, rerun weighted community detection (Step-3 style).
    Returns results table, email predictions per key, refined edge frames per key.
    """
    sid_to_idx = shard_id_to_index_map(shard_ids)
    Zn = prepare_normalized_shard_embeddings(Z)
    ed = edges_df.copy()
    ed["shard_a"] = ed["shard_a"].astype(str)
    ed["shard_b"] = ed["shard_b"].astype(str)
    ed["edge_weight_original"] = pd.to_numeric(ed["edge_weight"], errors="coerce").fillna(0.0)
    cos = edge_cosine_similarities(ed, Zn, sid_to_idx)
    ed["embedding_cosine"] = cos

    rows: list[dict[str, Any]] = []
    email_by_key: dict[str, pd.DataFrame] = {}
    refined_edges_by_key: dict[str, pd.DataFrame] = {}

    for alpha in alphas:
        rw = refine_edge_weights(
            ed["edge_weight_original"].to_numpy(),
            cos,
            alpha=float(alpha),
            sim_mode=sim_mode,
            weight_scale=refine_weight_scale,
        )
        ed_a = ed.copy()
        ed_a["edge_weight"] = rw
        ed_a["refinement_alpha"] = float(alpha)
        alpha_key = f"B_alpha{alpha:.2f}"
        refined_edges_by_key[alpha_key] = ed_a[
            ["shard_a", "shard_b", "edge_weight_original", "embedding_cosine", "edge_weight"]
        ].copy()

        for wcut in min_edge_weights:
            for res in resolutions:
                key = f"{alpha_key}_w{wcut:.3f}_r{res:.3f}"
                shard_to_comm, info = s3.run_weighted_community_detection(
                    shard_ids=shard_ids,
                    edges_df=ed_a,
                    method=cd_method,
                    resolution=float(res),
                    min_edge_weight=float(wcut),
                    seed=0,
                )
                email_df = s3.map_shards_to_email_predictions(assignments_df, shard_to_comm)
                m = s3.evaluate_external_metrics(email_df, gt_label_map)
                rows.append(
                    {
                        "path": "B_refined_edges_cd",
                        "setting_key": key,
                        "alpha": float(alpha),
                        "sim_mode": sim_mode,
                        "refine_weight_scale": refine_weight_scale,
                        "min_edge_weight": float(wcut),
                        "resolution": float(res),
                        "method_used": info["method_used"],
                        "n_communities": float(info["n_communities"]),
                        "n_pred_communities_email": int(email_df["pred_community"].nunique()),
                        **m,
                    }
                )
                email_by_key[key] = email_df

    return pd.DataFrame(rows), email_by_key, refined_edges_by_key


def save_step5_outputs(
    *,
    output_dir: str | Path,
    compare_df: pd.DataFrame,
    path_a_table: pd.DataFrame,
    path_b_table: pd.DataFrame,
    best_summary: dict[str, Any],
    extra_tables: dict[str, pd.DataFrame] | None = None,
) -> dict[str, str]:
    out = Path(output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    compare_df.to_csv(out / "semantic_shard_step5_compare_all.csv", index=False)
    paths["compare_all"] = str(out / "semantic_shard_step5_compare_all.csv")
    path_a_table.to_csv(out / "semantic_shard_step5_path_a_sweep.csv", index=False)
    paths["path_a_sweep"] = str(out / "semantic_shard_step5_path_a_sweep.csv")
    path_b_table.to_csv(out / "semantic_shard_step5_path_b_sweep.csv", index=False)
    paths["path_b_sweep"] = str(out / "semantic_shard_step5_path_b_sweep.csv")
    (out / "semantic_shard_step5_best_summary.json").write_text(
        json.dumps(best_summary, indent=2), encoding="utf-8"
    )
    paths["best_summary"] = str(out / "semantic_shard_step5_best_summary.json")
    for name, df in (extra_tables or {}).items():
        if df is None or df.empty:
            continue
        p = out / f"semantic_shard_step5_{name}.csv"
        df.to_csv(p, index=False)
        paths[name] = str(p)
    return paths


def drilldown_campaign_email_table(
    *,
    campaign_id: Any,
    campaign_to_members: dict[Any, list[str]],
    assignments_df: pd.DataFrame,
    baseline_pred_map: dict[str, int],
    path_a_pred_map: dict[str, int],
    path_b_pred_map: dict[str, int],
    shard_labels_a: dict[str, int] | None = None,
    shard_comm_b: dict[str, int] | None = None,
) -> pd.DataFrame:
    """Per-email view for one GT campaign: baseline vs path A vs path B."""
    members = [str(x) for x in campaign_to_members.get(campaign_id, [])]
    adf = assignments_df.copy()
    adf["external_id"] = adf["external_id"].astype(str)
    sub = adf[adf["external_id"].isin(set(members))].copy()
    sub["baseline_pred"] = sub["external_id"].map(baseline_pred_map)
    sub["path_a_pred"] = sub["external_id"].map(path_a_pred_map)
    sub["path_b_pred"] = sub["external_id"].map(path_b_pred_map)
    if shard_labels_a is not None:
        sub["path_a_shard_cluster"] = sub["shard_id"].astype(str).map(shard_labels_a)
    if shard_comm_b is not None:
        sub["path_b_shard_community"] = sub["shard_id"].astype(str).map(shard_comm_b)
    return sub.sort_values(["shard_id", "external_id"])


def drilldown_edges_for_shards(
    edges_df: pd.DataFrame,
    *,
    shard_subset: set[str],
) -> pd.DataFrame:
    """Filter to edges whose endpoints lie in shard_subset; preserve auxiliary columns."""
    e = edges_df.copy()
    e["shard_a"] = e["shard_a"].astype(str)
    e["shard_b"] = e["shard_b"].astype(str)
    m = e["shard_a"].isin(shard_subset) & e["shard_b"].isin(shard_subset)
    sub = e.loc[m].copy()
    sort_cols = [c for c in ("refined_edge_weight", "edge_weight", "embedding_cosine") if c in sub.columns]
    if sort_cols:
        return sub.sort_values(sort_cols[0], ascending=False)
    return sub
