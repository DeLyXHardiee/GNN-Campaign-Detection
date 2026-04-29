from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class EmbeddingCacheSummary:
    model: str
    subj_dim: int
    body_dim: int
    n_entries_in_by_key: int
    n_entries_with_any_vector: int
    key_examples: list[str]
    entry_fields: list[str]


def _l2_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0.0] = 1.0
    return x / n


def load_transformer_cache(
    embeddings_json_path: str | Path,
) -> tuple[dict[str, Any], dict[str, np.ndarray], EmbeddingCacheSummary]:
    p = Path(embeddings_json_path).expanduser().resolve()
    with open(p, "r", encoding="utf-8") as f:
        payload = json.load(f)

    by_key = payload.get("by_key") or {}
    if not isinstance(by_key, dict):
        raise ValueError("Invalid embeddings cache format: `by_key` must be a dict.")

    id_to_vec: dict[str, np.ndarray] = {}
    sample_fields: set[str] = set()
    for k, v in by_key.items():
        if not isinstance(v, dict):
            continue
        sample_fields.update(v.keys())
        subj = np.asarray(v.get("subj") or [], dtype=np.float32).reshape(-1)
        body = np.asarray(v.get("body") or [], dtype=np.float32).reshape(-1)
        if subj.size == 0 and body.size == 0:
            continue
        eid = str(v.get("external_id") or k)
        id_to_vec[eid] = np.concatenate([subj, body], axis=0)

    summary = EmbeddingCacheSummary(
        model=str(payload.get("model") or ""),
        subj_dim=int(payload.get("subj_dim") or 0),
        body_dim=int(payload.get("body_dim") or 0),
        n_entries_in_by_key=int(len(by_key)),
        n_entries_with_any_vector=int(len(id_to_vec)),
        key_examples=[str(k) for k in list(by_key.keys())[:5]],
        entry_fields=sorted(sample_fields),
    )
    return payload, id_to_vec, summary


def align_embedding_ids(
    id_to_vec: dict[str, np.ndarray],
    graph_external_ids: list[str],
    gt_label_map: dict[str, Any],
) -> dict[str, Any]:
    emb_ids = set(map(str, id_to_vec.keys()))
    graph_ids = set(map(str, graph_external_ids))
    gt_ids = set(map(str, gt_label_map.keys()))
    return {
        "n_embeddings": int(len(emb_ids)),
        "n_graph_emails": int(len(graph_ids)),
        "n_gt_emails": int(len(gt_ids)),
        "n_emb_on_graph": int(len(emb_ids & graph_ids)),
        "n_emb_on_gt": int(len(emb_ids & gt_ids)),
        "n_emb_on_graph_and_gt": int(len(emb_ids & graph_ids & gt_ids)),
    }


def cluster_semantic_shards_hdbscan(
    id_to_vec: dict[str, np.ndarray],
    *,
    min_cluster_size: int = 2,
    min_samples: int | None = None,
    fallback_cosine_distance_threshold: float = 0.22,
) -> pd.DataFrame:
    ids = sorted(id_to_vec.keys())
    if not ids:
        raise ValueError("No embeddings to cluster.")
    x = np.stack([id_to_vec[eid] for eid in ids]).astype(np.float32)
    x = _l2_rows(x)
    try:
        import hdbscan  # type: ignore

        cl = hdbscan.HDBSCAN(
            min_cluster_size=int(min_cluster_size),
            min_samples=None if min_samples is None else int(min_samples),
        )
        labels = cl.fit_predict(x)
        method = "hdbscan"
    except Exception:
        # Conservative fallback: hierarchical cosine clustering then mark tiny clusters as noise.
        from sklearn.cluster import AgglomerativeClustering

        cl = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=float(fallback_cosine_distance_threshold),
            metric="cosine",
            linkage="average",
        )
        raw_labels = cl.fit_predict(x).astype(int)
        ct = Counter(raw_labels.tolist())
        labels = np.array(
            [lab if ct[int(lab)] >= int(min_cluster_size) else -1 for lab in raw_labels],
            dtype=np.int64,
        )
        method = "agglomerative_cosine_fallback"

    df = pd.DataFrame({"external_id": ids, "cluster_label": labels.astype(int)})
    df["is_noise"] = df["cluster_label"] == -1
    df.attrs["cluster_method"] = method
    return df


def build_shard_assignments(
    clustered_df: pd.DataFrame,
    *,
    noise_as_singleton: bool = True,
) -> pd.DataFrame:
    out = clustered_df.copy()
    if noise_as_singleton:
        out["shard_id"] = [
            f"shard_{int(lab)}" if int(lab) >= 0 else f"noise_{eid}"
            for eid, lab in zip(out["external_id"], out["cluster_label"])
        ]
    else:
        out["shard_id"] = [
            f"shard_{int(lab)}" if int(lab) >= 0 else "noise"
            for lab in out["cluster_label"]
        ]
    return out


def _sampled_within_cosine(
    member_ids: list[str],
    id_to_vec: dict[str, np.ndarray],
    *,
    max_pairs: int = 3000,
    rng_seed: int = 0,
) -> tuple[float, float]:
    if len(member_ids) < 2:
        return float("nan"), float("nan")
    x = _l2_rows(np.stack([id_to_vec[e] for e in member_ids]).astype(np.float32))
    pairs = [(i, j) for i in range(len(member_ids)) for j in range(i + 1, len(member_ids))]
    if len(pairs) > max_pairs:
        rng = np.random.default_rng(rng_seed)
        pick = rng.choice(len(pairs), size=max_pairs, replace=False)
        pairs = [pairs[int(i)] for i in pick]
    vals = np.array([float(np.dot(x[i], x[j])) for i, j in pairs], dtype=np.float64)
    return float(np.mean(vals)), float(np.median(vals))


def shard_quality_tables(
    assignments_df: pd.DataFrame,
    id_to_vec: dict[str, np.ndarray],
    gt_label_map: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    gt_map = {str(k): v for k, v in gt_label_map.items()}

    for shard_id, g in assignments_df.groupby("shard_id", sort=False):
        members = [str(x) for x in g["external_id"].tolist()]
        size = int(len(members))
        non_noise_count = int((~g["is_noise"]).sum()) if "is_noise" in g.columns else size
        member_gt = [gt_map[e] for e in members if e in gt_map]
        n_gt = int(len(member_gt))
        if n_gt > 0:
            ct = Counter(member_gt)
            dom_campaign, dom_n = ct.most_common(1)[0]
            dom_frac = float(dom_n / n_gt)
            n_campaigns = int(len(ct))
        else:
            dom_campaign, dom_frac, n_campaigns = None, float("nan"), 0

        cos_mean, cos_median = _sampled_within_cosine(members, id_to_vec)
        rows.append(
            {
                "shard_id": shard_id,
                "size": size,
                "non_noise_member_count": non_noise_count,
                "n_members_with_gt": n_gt,
                "n_gt_campaigns_touched": n_campaigns,
                "dominant_campaign": dom_campaign,
                "dominant_campaign_fraction": dom_frac,
                "within_cos_mean": cos_mean,
                "within_cos_median": cos_median,
            }
        )

    shard_df = pd.DataFrame(rows).sort_values(["size", "dominant_campaign_fraction"], ascending=[False, False])
    if shard_df.empty:
        return shard_df, pd.DataFrame(columns=["metric", "value"])

    overall = {
        "n_emails_assigned": int(len(assignments_df)),
        "n_shards_total": int(shard_df["shard_id"].nunique()),
        "n_singleton_shards": int((shard_df["size"] == 1).sum()),
        "frac_singleton_shards": float((shard_df["size"] == 1).mean()),
        "mean_shard_size": float(shard_df["size"].mean()),
        "median_shard_size": float(shard_df["size"].median()),
        "n_non_singleton_shards": int((shard_df["size"] >= 2).sum()),
        "mean_within_cos_non_singleton": float(shard_df.loc[shard_df["size"] >= 2, "within_cos_mean"].mean()),
        "median_within_cos_non_singleton": float(shard_df.loc[shard_df["size"] >= 2, "within_cos_median"].median()),
        # NOTE: this denominator is ALL shards (including shards with no GT-covered members).
        "frac_shards_pure_ge_0.90": float((shard_df["dominant_campaign_fraction"] >= 0.90).mean()),
        "frac_shards_pure_ge_0.95": float((shard_df["dominant_campaign_fraction"] >= 0.95).mean()),
        "frac_shards_mixed_gt_gt2_campaigns": float((shard_df["n_gt_campaigns_touched"] >= 2).mean()),
    }
    gt_cov = shard_df[shard_df["n_members_with_gt"] > 0].copy()
    overall["n_shards_with_gt_coverage"] = int(len(gt_cov))
    overall["frac_shards_pure_ge_0.90_gt_covered"] = (
        float((gt_cov["dominant_campaign_fraction"] >= 0.90).mean()) if not gt_cov.empty else float("nan")
    )
    overall["frac_shards_pure_ge_0.95_gt_covered"] = (
        float((gt_cov["dominant_campaign_fraction"] >= 0.95).mean()) if not gt_cov.empty else float("nan")
    )
    overall_df = pd.DataFrame([{"metric": k, "value": v} for k, v in overall.items()])
    return shard_df.reset_index(drop=True), overall_df


def campaign_split_by_shards(
    assignments_df: pd.DataFrame,
    gt_label_map: dict[str, Any],
) -> pd.DataFrame:
    gt_map = {str(k): v for k, v in gt_label_map.items()}
    with_gt = assignments_df[assignments_df["external_id"].map(lambda x: str(x) in gt_map)].copy()
    with_gt["campaign_id"] = with_gt["external_id"].map(lambda x: gt_map[str(x)])
    rows: list[dict[str, Any]] = []
    for cid, g in with_gt.groupby("campaign_id", sort=False):
        shard_counts = g["shard_id"].value_counts()
        top = int(shard_counts.iloc[0]) if len(shard_counts) else 0
        n = int(len(g))
        rows.append(
            {
                "campaign_id": cid,
                "campaign_size": n,
                "n_shards_touched": int(shard_counts.size),
                "largest_shard_overlap": top,
                "dominant_shard_fraction": float(top / max(1, n)),
                "fragmentation_score": float(1.0 - (top / max(1, n))),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["fragmentation_score", "campaign_size"],
        ascending=[False, False],
    ).reset_index(drop=True)


def save_shard_step1_artifacts(
    *,
    output_dir: str | Path,
    assignments_df: pd.DataFrame,
    shard_summary_df: pd.DataFrame,
    overall_df: pd.DataFrame,
) -> dict[str, str]:
    out = Path(output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    p_assign = out / "semantic_shard_step1_assignments.csv"
    p_shards = out / "semantic_shard_step1_shard_summary.csv"
    p_overall = out / "semantic_shard_step1_overall_summary.csv"
    assignments_df.to_csv(p_assign, index=False)
    shard_summary_df.to_csv(p_shards, index=False)
    overall_df.to_csv(p_overall, index=False)
    return {
        "assignments_csv": str(p_assign),
        "shard_summary_csv": str(p_shards),
        "overall_summary_csv": str(p_overall),
    }
