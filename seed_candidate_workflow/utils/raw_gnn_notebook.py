"""
Notebook support: load config-driven paths, align RAW/GNN graph embeddings with ground truth,
cluster (HDBSCAN), fragmentation stats, nearest-neighbor metrics, and pairwise comparisons.

Reuses:
- core.config.pipeline_config (load, resolve paths, default graph)
- core.GNN.steps.gnn_pipeline_helpers.resolve_gnn_paths
- core.clustering.clusteringMetrics (GT labels, external metrics, _aligned_true_predictived_labels)
- core.GNN.src.clustering.clustering_helpers (extract_email_embeddings, extract_raw_email_embeddings)
"""
from __future__ import annotations

import json
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np
import pandas as pd



def find_project_root(start: Path | None = None) -> Path:
    p = (start or Path.cwd()).resolve()
    for d in (p, *p.parents):
        if (d / "pipeline_config.json").is_file():
            return d
    raise FileNotFoundError(
        "Could not find pipeline_config.json; run the notebook from the repo root "
        "or from seed_candidate_workflow/ (parent must contain pipeline_config.json)."
    )


def ensure_core_gnn_paths(project_root: Path) -> None:
    core = project_root / "core"
    gnn = core / "GNN"
    for x in (core, gnn):
        s = str(x.resolve())
        if s not in sys.path:
            sys.path.insert(0, s)


@dataclass
class AnalysisPaths:
    project_root: Path
    graph_pt: Path
    meta_json: Path
    checkpoint_pt: Path
    ground_truth_json: Path
    run_dir: Path
    device: str
    to_undirected: bool
    hdbscan_min_cluster_size: int
    hdbscan_min_samples: int | None


def resolve_analysis_paths(project_root: Path | None = None) -> AnalysisPaths:
    root = project_root or find_project_root()
    ensure_core_gnn_paths(root)
    from config.pipeline_config import load_pipeline_config
    from steps.gnn_pipeline_helpers import load_gnn_cfg, resolve_gnn_paths

    cfg = load_pipeline_config(project_root=root)
    g = load_gnn_cfg(cfg, project_root=root)
    run_dir, ckpt, graph, gt = resolve_gnn_paths(
        cfg=cfg,
        run_dir=None,
        runs_parent=None,
        checkpoint_path=None,
        graph_path=None,
        ground_truth_path=None,
        require_ground_truth=True,
        project_root=root,
    )
    hcfg = cfg.get("gnn_clustering", {}).get("config", {}).get("hdbscan", {}) or {}
    mcs_list = hcfg.get("min_cluster_size_values") or [5]
    mcs = int(mcs_list[0])
    ms = hcfg.get("min_samples")
    ms = None if ms is None else int(ms)
    return AnalysisPaths(
        project_root=root,
        graph_pt=Path(graph),
        meta_json=Path(graph).with_suffix(".meta.json"),
        checkpoint_pt=Path(ckpt),
        ground_truth_json=Path(gt),
        run_dir=Path(run_dir),
        device=str(cfg.get("device") or "cpu"),
        to_undirected=bool(cfg.get("to_undirected", True)),
        hdbscan_min_cluster_size=mcs,
        hdbscan_min_samples=ms,
    )


def load_email_external_ids(meta_json: Path) -> list[str]:
    with open(meta_json, "r", encoding="utf-8") as f:
        meta = json.load(f)
    xs = meta.get("email_attrs", {}).get("external_id")
    if not xs:
        raise ValueError(f"No email_attrs.external_id in {meta_json}")
    return [str(x.item() if isinstance(x, np.generic) else x) for x in xs]


def parse_campaign_key(raw_key: str) -> Any:
    cluster_id_str = raw_key.split("/")[-1] if "/" in raw_key else raw_key
    try:
        return int(cluster_id_str)
    except ValueError:
        return cluster_id_str


def load_ground_truth_structures(
    gt_path: str | Path,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[Any, list[str]]]:
    """
    Align with extract_ground_truth_labels: first occurrence of external_id wins;
    duplicates are skipped (no overwrite).
    """
    with open(gt_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    label_map: dict[str, Any] = {}
    eid_to_row: dict[str, dict[str, Any]] = {}
    campaign_to_members: dict[Any, list[str]] = defaultdict(list)

    for raw_key, emails in (data.get("clusters") or {}).items():
        cid = parse_campaign_key(str(raw_key))
        if not isinstance(emails, list):
            continue
        for em in emails:
            if not isinstance(em, dict):
                continue
            eid = em.get("external_id")
            if eid is None:
                continue
            eid = str(eid)
            if eid in label_map:
                continue
            label_map[eid] = cid
            row = dict(em)
            row["_campaign_id"] = cid
            eid_to_row[eid] = row
            campaign_to_members[cid].append(eid)

    return label_map, eid_to_row, dict(campaign_to_members)


def restrict_embedding_map(
    id_to_emb: dict[str, np.ndarray], allowed: set[str]
) -> dict[str, np.ndarray]:
    allow = {str(x) for x in allowed}
    return {str(k): id_to_emb[k] for k in id_to_emb if str(k) in allow}


def extract_raw_gnn_maps(
    *,
    graph_pt: Path,
    checkpoint_pt: Path,
    external_ids: list[str],
    device: str,
    to_undirected: bool,
):
    import torch
    from src.clustering.clustering_helpers import (
        extract_email_embeddings,
        extract_raw_email_embeddings,
    )
    from src.load_graph_data import load_hetero_pt
    from src.model_io import load_model_checkpoint, select_device

    data = load_hetero_pt(path=str(graph_pt), to_undirected=to_undirected)
    dev = select_device(torch.device(device) if device else None)
    model, _pred, _ckpt = load_model_checkpoint(
        device=dev, metadata=data.metadata(), filename=str(checkpoint_pt)
    )
    raw = extract_raw_email_embeddings(data, external_ids=external_ids)
    gnn = extract_email_embeddings(model, data, dev, external_ids=external_ids)
    return data, raw, gnn


def run_hdbscan_get_labels(
    id_to_emb: dict[str, np.ndarray],
    min_cluster_size: int,
    min_samples: int | None = None,
) -> tuple[list[str], np.ndarray]:
    import hdbscan  # type: ignore

    from core.clustering.clusteringMetrics import _emb_matrix_from_id_to_embedding

    sorted_ids, X = _emb_matrix_from_id_to_embedding(id_to_emb)
    clust = hdbscan.HDBSCAN(
        min_cluster_size=int(min_cluster_size),
        min_samples=None if min_samples is None else int(min_samples),
    )
    labels = clust.fit_predict(X)
    return sorted_ids, labels


def eid_label_map(sorted_ids: list[str], labels: np.ndarray) -> dict[str, int]:
    return {eid: int(lab) for eid, lab in zip(sorted_ids, np.asarray(labels))}


def external_scores_subset(
    sorted_ids: list[str],
    labels: np.ndarray,
    ground_truth: dict[str, Any],
) -> dict[str, float]:
    from core.clustering.clusteringMetrics import (
        compute_external_metrics,
        _aligned_true_predicted_labels,
    )

    tr, pr = _aligned_true_predicted_labels(
        sorted_ids=list(sorted_ids),
        labels=np.asarray(labels),
        ground_truth_labels=ground_truth,
    )
    return compute_external_metrics(tr, pr)


def campaign_fragmentation_table(
    campaign_to_members: dict[Any, list[str]],
    eid_to_pred: dict[str, int],
    *,
    name: str,
) -> list[dict[str, Any]]:
    rows = []
    for cid, members in sorted(campaign_to_members.items(), key=lambda x: str(x[0])):
        m = [str(x) for x in members]
        assigned = [
            eid for eid in m if eid in eid_to_pred and eid_to_pred[eid] != -1
        ]
        preds = [eid_to_pred[eid] for eid in assigned]
        if not preds:
            rows.append(
                {
                    "campaign_id": cid,
                    "model": name,
                    "n_members_gt": len(m),
                    "n_assigned_non_noise": 0,
                    "n_pred_clusters_touching": 0,
                    "largest_pred_overlap": 0,
                    "dominant_fraction": 0.0,
                }
            )
            continue
        ct = Counter(preds)
        top = max(ct.values())
        rows.append(
            {
                "campaign_id": cid,
                "model": name,
                "n_members_gt": len(m),
                "n_assigned_non_noise": len(assigned),
                "n_pred_clusters_touching": len(ct),
                "largest_pred_overlap": int(top),
                "dominant_fraction": float(top) / max(1, len(m)),
            }
        )
    return rows


def _l2_normalize_rows(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n


def _emb_matrix_for_ids(
    ids: list[str], id_to_emb: dict[str, np.ndarray]
) -> np.ndarray:
    return np.stack([np.asarray(id_to_emb[i], dtype=np.float64).ravel() for i in ids])


def neighbor_metrics(
    labeled_ids: list[str],
    id_to_emb: dict[str, np.ndarray],
    id_to_campaign: dict[str, Any],
    k_list: tuple[int, ...] = (5, 10),
    *,
    name: str,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    """Recall@k and mean NN purity@k in embedding space (cosine ~ dot after L2 norm)."""
    ids = [str(x) for x in labeled_ids]
    X = _l2_normalize_rows(_emb_matrix_for_ids(ids, id_to_emb))
    sims = X @ X.T
    np.fill_diagonal(sims, -np.inf)
    per_campaign_rows: list[dict[str, Any]] = []
    recalls = {f"recall_at_{k}": [] for k in k_list}
    purities = {f"nn_purity_at_{k}": [] for k in k_list}
    ranks_first_same: list[int] = []

    camp_indices: dict[Any, list[int]] = defaultdict(list)
    for i, eid in enumerate(ids):
        camp_indices[id_to_campaign[eid]].append(i)

    for ci, eid in enumerate(ids):
        true_c = id_to_campaign[eid]
        order = np.argsort(-sims[ci])
        same_others = [j for j in order if id_to_campaign[ids[j]] == true_c]
        if same_others:
            ranks_first_same.append(int(np.where(order == same_others[0])[0][0]) + 1)
        for k in k_list:
            top = order[:k]
            same = sum(1 for j in top if id_to_campaign[ids[j]] == true_c)
            recalls[f"recall_at_{k}"].append(same / max(1, min(k, len(top))))
            purities[f"nn_purity_at_{k}"].append(same / max(1, k))

    agg = {"model": name, "n_labeled": len(ids)}
    for k in k_list:
        agg[f"recall_at_{k}_mean"] = float(np.mean(recalls[f"recall_at_{k}"])) if recalls[f"recall_at_{k}"] else 0.0
        agg[f"nn_purity_at_{k}_mean"] = float(np.mean(purities[f"nn_purity_at_{k}"])) if purities[f"nn_purity_at_{k}"] else 0.0
    agg["median_rank_first_same_campaign_nn"] = float(np.median(ranks_first_same)) if ranks_first_same else float("nan")

    for camp, cidx in sorted(camp_indices.items(), key=lambda x: str(x[0])):
        if len(cidx) < 2:
            continue
        sub_sim = sims[np.ix_(cidx, cidx)].copy()
        np.fill_diagonal(sub_sim, -np.inf)
        kmax = max(k_list)
        recall_vals = []
        purity_vals = []
        for row_i, gi in enumerate(cidx):
            order = np.argsort(-sub_sim[row_i])
            top = order[: min(kmax, len(order))]
            same = sum(1 for j in top if id_to_campaign[ids[cidx[j]]] == camp)
            recall_vals.append(same / max(1, min(kmax, len(top))))
            purity_vals.append(same / max(1, kmax))
        per_campaign_rows.append(
            {
                "campaign_id": camp,
                "model": name,
                "size": len(cidx),
                f"mean_intra_recall_at_{kmax}": float(np.mean(recall_vals)),
                f"mean_intra_purity_at_{kmax}": float(np.mean(purity_vals)),
            }
        )

    return agg, per_campaign_rows


def domain_hint(url: str) -> str:
    try:
        p = urlparse(str(url).strip())
        if p.netloc:
            return p.netloc.lower()
        return str(url).split("/")[0].lower()
    except Exception:
        return ""


def truncate_preview(text: str, max_chars: int = 140) -> str:
    t = " ".join(str(text).split())
    if len(t) <= max_chars:
        return t
    return t[: max_chars - 3] + "..."


def urls_list_from_row(r: dict[str, Any]) -> list[str]:
    urls = r.get("urls") or []
    if isinstance(urls, str):
        urls = [urls]
    return [str(u) for u in urls if u]


def row_email_previews(
    eid: str, eid_to_row: dict[str, dict[str, Any]], body_chars: int = 140
) -> dict[str, str]:
    r = eid_to_row.get(eid) or {}
    urls = urls_list_from_row(r)
    doms_set: set[str] = set()
    for u in urls:
        d = domain_hint(u)
        if d:
            doms_set.add(d)
    doms = sorted(doms_set)
    return {
        "sender": truncate_preview(str(r.get("sender") or ""), 100),
        "subject": truncate_preview(
            str(r.get("subject_translated") or r.get("subject") or ""), 120
        ),
        "body_preview": truncate_preview(
            str(r.get("body_translated") or r.get("body") or ""), body_chars
        ),
        "urls_preview": ", ".join(truncate_preview(u, 72) for u in urls[:5]),
        "domains_preview": truncate_preview(", ".join(doms[:10]), 120),
    }


def intra_campaign_embedding_pairwise_stats(
    member_eids: list[str],
    id_to_emb: dict[str, np.ndarray],
    *,
    max_pairs: int = 800,
    seed: int = 0,
) -> tuple[float, float]:
    """Mean and median cosine similarity over sampled within-campaign pairs (L2 rows)."""
    m = [str(e) for e in member_eids if str(e) in id_to_emb]
    if len(m) < 2:
        return float("nan"), float("nan")
    X = _l2_normalize_rows(_emb_matrix_for_ids(m, id_to_emb))
    rng = random.Random(seed)
    pairs = [(i, j) for i in range(len(m)) for j in range(i + 1, len(m))]
    rng.shuffle(pairs)
    pairs = pairs[:max_pairs]
    if not pairs:
        return float("nan"), float("nan")
    cos = [float(np.dot(X[i], X[j])) for i, j in pairs]
    return float(np.mean(cos)), float(np.median(cos))


def campaign_descriptors(
    campaign_to_members: dict[Any, list[str]],
    eid_to_row: dict[str, dict[str, Any]],
    id_to_emb_raw: dict[str, np.ndarray],
    id_to_emb_gnn: dict[str, np.ndarray],
    *,
    intra_pair_max_samples: int = 800,
    jaccard_pair_cap: int = 120,
) -> list[dict[str, Any]]:
    """
    Campaign-level proxies for content vs infrastructure spread.

    `intra_cosine_mean_raw` / `_gnn` and medians are **mean/median pairwise cosine**
    among labeled members in **graph RAW** vs **GNN** embedding spaces (L2-normalized).
    They are *not* separate sentence-transformer body vectors unless those were used
    to build `data[\"email\"].x` / the checkpoint.
    """
    rows = []
    for cid, members in campaign_to_members.items():
        m = [str(x) for x in members]
        senders = set()
        urls_all = set()
        bodies = []
        subjects = []
        for eid in m:
            r = eid_to_row.get(eid) or {}
            senders.add(str(r.get("sender") or ""))
            urls = r.get("urls") or []
            if isinstance(urls, str):
                urls = [urls]
            for u in urls:
                d = domain_hint(u)
                if d:
                    urls_all.add(d)
            bodies.append(str(r.get("body") or r.get("body_translated") or ""))
            subjects.append(
                str(r.get("subject_translated") or r.get("subject") or "")
            )

        def jac(a: str, b: str) -> float:
            sa, sb = set(a.lower().split()), set(b.lower().split())
            if not sa and not sb:
                return 1.0
            inter = len(sa & sb)
            uni = len(sa | sb)
            return inter / max(1, uni)

        def sampled_mean_jac(texts: list[str]) -> float:
            if len(texts) < 2:
                return 0.0
            pairs: list[tuple[int, int]] = [
                (i, j) for i in range(len(texts)) for j in range(i + 1, len(texts))
            ]
            rng = random.Random(0)
            rng.shuffle(pairs)
            pairs = pairs[:jaccard_pair_cap]
            if not pairs:
                return 0.0
            return float(np.mean([jac(texts[i], texts[j]) for i, j in pairs]))

        avg_body_jac = sampled_mean_jac(bodies)
        avg_subject_jac = sampled_mean_jac(subjects)
        mn_raw, med_raw = intra_campaign_embedding_pairwise_stats(
            m, id_to_emb_raw, max_pairs=intra_pair_max_samples, seed=0
        )
        mn_gnn, med_gnn = intra_campaign_embedding_pairwise_stats(
            m, id_to_emb_gnn, max_pairs=intra_pair_max_samples, seed=0
        )

        rows.append(
            {
                "campaign_id": cid,
                "size": len(m),
                "n_unique_senders": len([x for x in senders if x]),
                "n_unique_url_domains": len(urls_all),
                "avg_body_jaccard_sampled": float(avg_body_jac),
                "avg_subject_jaccard_sampled": float(avg_subject_jac),
                "intra_cosine_mean_raw": mn_raw,
                "intra_cosine_median_raw": med_raw,
                "intra_cosine_mean_gnn": mn_gnn,
                "intra_cosine_median_gnn": med_gnn,
            }
        )
    return rows


def pairwise_similarity_samples(
    labeled_ids: list[str],
    id_to_emb_raw: dict[str, np.ndarray],
    id_to_emb_gnn: dict[str, np.ndarray],
    id_to_campaign: dict[str, Any],
    *,
    max_same_pairs: int = 4000,
    max_diff_pairs: int = 4000,
    seed: int = 0,
) -> dict[str, Any]:
    rng = random.Random(seed)
    ids = [str(x) for x in labeled_ids]
    Xr = _l2_normalize_rows(_emb_matrix_for_ids(ids, id_to_emb_raw))
    Xg = _l2_normalize_rows(_emb_matrix_for_ids(ids, id_to_emb_gnn))
    camp_to_idx: dict[Any, list[int]] = defaultdict(list)
    for i, eid in enumerate(ids):
        camp_to_idx[id_to_campaign[eid]].append(i)

    same_idx: list[tuple[int, int]] = []
    for _, idxs in camp_to_idx.items():
        if len(idxs) < 2:
            continue
        pairs = [(idxs[a], idxs[b]) for a in range(len(idxs)) for b in range(a + 1, len(idxs))]
        rng.shuffle(pairs)
        same_idx.extend(pairs[:200])
    rng.shuffle(same_idx)
    same_idx = same_idx[:max_same_pairs]

    diff_idx = []
    camps = list(camp_to_idx.keys())
    for _ in range(max_diff_pairs * 3):
        if len(diff_idx) >= max_diff_pairs:
            break
        c1, c2 = rng.sample(camps, 2) if len(camps) >= 2 else (camps[0], camps[0])
        if c1 == c2:
            continue
        i = rng.choice(camp_to_idx[c1])
        j = rng.choice(camp_to_idx[c2])
        diff_idx.append((i, j))
    diff_idx = diff_idx[:max_diff_pairs]

    def cosims(X, pairs):
        return np.array([float(np.dot(X[i], X[j])) for i, j in pairs], dtype=np.float64)

    return {
        "same_pairs": same_idx,
        "diff_pairs": diff_idx,
        "raw_same_cos": cosims(Xr, same_idx),
        "raw_diff_cos": cosims(Xr, diff_idx),
        "gnn_same_cos": cosims(Xg, same_idx),
        "gnn_diff_cos": cosims(Xg, diff_idx),
    }


def find_extreme_pair_examples(
    samples: dict[str, Any],
    ids: list[str],
    *,
    top_n: int = 8,
) -> dict[str, list[dict[str, Any]]]:
    """Concrete pairs where cosine similarity differs most between RAW and GNN."""

    def pack(
        order: np.ndarray,
        plist: list[tuple[int, int]],
        rc: np.ndarray,
        gc: np.ndarray,
        gap: np.ndarray,
    ) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for t in order[:top_n]:
            ti = int(t)
            i, j = plist[ti]
            out.append(
                {
                    "eid_a": ids[i],
                    "eid_b": ids[j],
                    "gap_raw_minus_gnn_cos": float(gap[ti]),
                    "cos_raw": float(rc[ti]),
                    "cos_gnn": float(gc[ti]),
                }
            )
        return out

    same = samples["same_pairs"]
    diff = samples["diff_pairs"]
    r_s, g_s = samples["raw_same_cos"], samples["gnn_same_cos"]
    r_d, g_d = samples["raw_diff_cos"], samples["gnn_diff_cos"]
    gap_s = r_s - g_s
    gap_d = r_d - g_d

    return {
        "same_campaign_raw_more_similar_than_gnn": pack(
            np.argsort(-gap_s), same, r_s, g_s, gap_s
        ),
        "same_campaign_gnn_more_similar_than_raw": pack(
            np.argsort(gap_s), same, r_s, g_s, gap_s
        ),
        "diff_campaign_raw_more_similar_than_gnn": pack(
            np.argsort(-gap_d), diff, r_d, g_d, gap_d
        ),
        "diff_campaign_gnn_more_similar_than_raw": pack(
            np.argsort(gap_d), diff, r_d, g_d, gap_d
        ),
    }


def top_k_neighbors(
    query_eid: str,
    pool_ids: list[str],
    id_to_emb: dict[str, np.ndarray],
    k: int = 8,
) -> list[tuple[str, float]]:
    """Cosine similarity to query after L2-normalizing rows (pool must include query)."""
    pool = [str(x) for x in pool_ids]
    q = str(query_eid)
    X = _l2_normalize_rows(_emb_matrix_for_ids(pool, id_to_emb))
    qi = pool.index(q)
    sims = X @ X[qi]
    sims[qi] = -np.inf
    order = np.argsort(-sims)
    out = []
    for j in order[:k]:
        out.append((pool[int(j)], float(sims[int(j)])))
    return out


def format_pair_row(eid_a: str, eid_b: str, eid_to_row: dict[str, dict]) -> dict[str, Any]:
    def one(eid):
        r = eid_to_row.get(eid) or {}
        urls = r.get("urls") or []
        if isinstance(urls, str):
            urls = [urls]
        return {
            "external_id": eid,
            "subject": (r.get("subject_translated") or r.get("subject") or "")[:200],
            "sender": r.get("sender"),
            "urls": urls[:5],
            "campaign_id": r.get("_campaign_id"),
        }

    return {"a": one(eid_a), "b": one(eid_b)}


def extreme_pair_inspection_table(
    packed_pairs: list[dict[str, Any]],
    *,
    pair_type: str,
    id_to_campaign: dict[str, Any],
    eid_to_row: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    rows_out: list[dict[str, Any]] = []
    for r in packed_pairs:
        ea, eb = str(r["eid_a"]), str(r["eid_b"])
        pa = row_email_previews(ea, eid_to_row)
        pb = row_email_previews(eb, eid_to_row)
        cr = r.get("cos_raw")
        cg = r.get("cos_gnn")
        gap = r.get("gap_raw_minus_gnn_cos")
        try:
            cos_readable = (
                f"RAW {float(cr):.4f} | GNN {float(cg):.4f} | (raw - gnn) {float(gap):+.4f}"
            )
        except (TypeError, ValueError):
            cos_readable = f"RAW {cr} | GNN {cg} | (raw - gnn) {gap}"
        rows_out.append(
            {
                "pair_type": pair_type,
                "external_id_1": ea,
                "external_id_2": eb,
                "campaign_id_1": id_to_campaign.get(ea),
                "campaign_id_2": id_to_campaign.get(eb),
                "raw_cosine": cr,
                "gnn_cosine": cg,
                "cosine_gap_raw_minus_gnn": gap,
                "cosine_similarity": cos_readable,
                "sender_1": pa["sender"],
                "sender_2": pb["sender"],
                "subject_1": pa["subject"],
                "subject_2": pb["subject"],
                "body_preview_1": pa["body_preview"],
                "body_preview_2": pb["body_preview"],
                "urls_1": pa["urls_preview"],
                "urls_2": pb["urls_preview"],
                "domains_1": pa["domains_preview"],
                "domains_2": pb["domains_preview"],
            }
        )
    col_order = [
        "pair_type",
        "external_id_1",
        "external_id_2",
        "campaign_id_1",
        "campaign_id_2",
        "raw_cosine",
        "gnn_cosine",
        "cosine_gap_raw_minus_gnn",
        "cosine_similarity",
        "sender_1",
        "sender_2",
        "subject_1",
        "subject_2",
        "body_preview_1",
        "body_preview_2",
        "urls_1",
        "urls_2",
        "domains_1",
        "domains_2",
    ]
    return pd.DataFrame(rows_out, columns=col_order)


def extreme_pair_inspection_tables(
    extremes: dict[str, list[dict[str, Any]]],
    id_to_campaign: dict[str, Any],
    eid_to_row: dict[str, dict[str, Any]],
) -> dict[str, pd.DataFrame]:
    key_to_pair_type = {
        "same_campaign_raw_more_similar_than_gnn": "same_campaign",
        "same_campaign_gnn_more_similar_than_raw": "same_campaign",
        "diff_campaign_raw_more_similar_than_gnn": "different_campaign",
        "diff_campaign_gnn_more_similar_than_raw": "different_campaign",
    }
    return {
        k: extreme_pair_inspection_table(
            extremes[k], pair_type=key_to_pair_type[k], id_to_campaign=id_to_campaign, eid_to_row=eid_to_row
        )
        for k in key_to_pair_type
        if k in extremes
    }


def campaign_size_bucket_map(
    campaign_to_members: dict[Any, list[str]],
) -> tuple[dict[Any, str], dict[str, float]]:
    """
    Deterministic tertiles on campaign sizes (members count): small / medium / large.
    Thresholds are the 1/3 and 2/3 quantiles of the per-campaign size distribution.
    """
    items = [(cid, len(m)) for cid, m in campaign_to_members.items()]
    if not items:
        return {}, {"p33_size": float("nan"), "p67_size": float("nan")}
    sizes = np.array([s for _, s in items], dtype=np.float64)
    t1, t2 = np.quantile(sizes, [1.0 / 3.0, 2.0 / 3.0])
    out: dict[Any, str] = {}
    for cid, s in items:
        if s <= t1:
            out[cid] = "small"
        elif s <= t2:
            out[cid] = "medium"
        else:
            out[cid] = "large"
    return out, {"p33_size": float(t1), "p67_size": float(t2)}


def stratified_fragmentation_summary(frag_with_bucket: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for b in ("small", "medium", "large"):
        sub = frag_with_bucket[frag_with_bucket["size_bucket"] == b]
        if sub.empty:
            continue
        vc = sub["winner_fragmentation"].value_counts()
        rows.append(
            {
                "size_bucket": b,
                "n_campaigns": int(len(sub)),
                "mean_n_pred_clusters_raw": float(sub["n_pred_clusters_touching_raw"].mean()),
                "mean_n_pred_clusters_gnn": float(sub["n_pred_clusters_touching_gnn"].mean()),
                "mean_dominant_frac_raw": float(sub["dominant_fraction_raw"].mean()),
                "mean_dominant_frac_gnn": float(sub["dominant_fraction_gnn"].mean()),
                "n_wins_raw": int(vc.get("RAW", 0)),
                "n_wins_gnn": int(vc.get("GNN", 0)),
                "n_ties": int(vc.get("tie", 0)),
            }
        )
    return pd.DataFrame(rows)


def merge_per_campaign_neighbor_rows(
    nnCamp_raw: list[dict[str, Any]],
    nnCamp_gnn: list[dict[str, Any]],
) -> pd.DataFrame:
    dr = pd.DataFrame(nnCamp_raw)
    dg = pd.DataFrame(nnCamp_gnn)
    kcols = [c for c in dr.columns if c.startswith("mean_intra_")]
    m = dr.merge(dg, on="campaign_id", suffixes=("_raw", "_gnn"), how="inner")
    for c in kcols:
        br, bg = f"{c}_raw", f"{c}_gnn"
        if br in m.columns and bg in m.columns:
            m[f"delta_{c}"] = m[bg] - m[br]
    return m


def stratified_neighbor_summary(
    nnMerged: pd.DataFrame,
    bucket_map: dict[Any, str],
) -> pd.DataFrame:
    df = nnMerged.copy()
    df["size_bucket"] = df["campaign_id"].map(bucket_map)
    recall_col = [c for c in df.columns if c.startswith("delta_mean_intra_recall_at_")]
    purity_col = [c for c in df.columns if c.startswith("delta_mean_intra_purity_at_")]
    d_recall = recall_col[-1] if recall_col else None
    d_purity = purity_col[-1] if purity_col else None
    rows = []
    for b in ("small", "medium", "large"):
        sub = df[df["size_bucket"] == b]
        if sub.empty:
            continue
        row: dict[str, Any] = {
            "size_bucket": b,
            "n_campaigns": int(len(sub)),
        }
        if d_recall and d_recall in sub:
            row["mean_delta_intra_recall"] = float(sub[d_recall].mean())
        if d_purity and d_purity in sub:
            row["mean_delta_intra_purity"] = float(sub[d_purity].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def pairwise_cosine_summary_table(samples: dict[str, Any]) -> pd.DataFrame:
    def summarize(arr: np.ndarray) -> dict[str, float]:
        a = np.asarray(arr, dtype=np.float64)
        if a.size == 0:
            return {"n": 0, "mean": float("nan"), "median": float("nan"), "p10": float("nan"), "p90": float("nan")}
        return {
            "n": int(a.size),
            "mean": float(np.mean(a)),
            "median": float(np.median(a)),
            "p10": float(np.quantile(a, 0.10)),
            "p90": float(np.quantile(a, 0.90)),
        }

    rows = []
    for model, rs, rd in (
        ("RAW", samples["raw_same_cos"], samples["raw_diff_cos"]),
        ("GNN", samples["gnn_same_cos"], samples["gnn_diff_cos"]),
    ):
        for label, arr in (("same_campaign", rs), ("different_campaign", rd)):
            s = summarize(arr)
            rows.append({"model": model, "pair_type": label, **s})
    return pd.DataFrame(rows)


def pairwise_scatter_frame(
    samples: dict[str, Any],
    *,
    max_same: int = 4000,
    max_diff: int = 4000,
    seed: int = 0,
) -> pd.DataFrame:
    rng = random.Random(seed)
    rs = np.asarray(samples["raw_same_cos"], dtype=np.float64)
    gs = np.asarray(samples["gnn_same_cos"], dtype=np.float64)
    rd = np.asarray(samples["raw_diff_cos"], dtype=np.float64)
    gd = np.asarray(samples["gnn_diff_cos"], dtype=np.float64)
    n_s, n_d = int(rs.size), int(rd.size)
    idx_s = rng.sample(range(n_s), min(max_same, n_s)) if n_s else []
    idx_d = rng.sample(range(n_d), min(max_diff, n_d)) if n_d else []
    rows: list[dict[str, Any]] = []
    for t in idx_s:
        rows.append(
            {
                "same_campaign": True,
                "raw_cosine": float(rs[t]),
                "gnn_cosine": float(gs[t]),
            }
        )
    for t in idx_d:
        rows.append(
            {
                "same_campaign": False,
                "raw_cosine": float(rd[t]),
                "gnn_cosine": float(gd[t]),
            }
        )
    return pd.DataFrame(rows)


def drilldown_member_table(
    campaign_id: Any,
    campaign_to_members: dict[Any, list[str]],
    eid_to_row: dict[str, dict[str, Any]],
    pred_raw: dict[str, int],
    pred_gnn: dict[str, int],
) -> pd.DataFrame:
    m = [str(x) for x in campaign_to_members.get(campaign_id, [])]
    rows = []
    for eid in m:
        pv = row_email_previews(eid, eid_to_row)
        rows.append(
            {
                "external_id": eid,
                "sender": pv["sender"],
                "subject": pv["subject"],
                "body_preview": pv["body_preview"],
                "urls_preview": pv["urls_preview"],
                "domains_preview": pv["domains_preview"],
                "raw_cluster_id": pred_raw.get(eid),
                "gnn_cluster_id": pred_gnn.get(eid),
            }
        )
    return pd.DataFrame(rows)


def neighbor_ranking_table(
    query_eid: str,
    pool_ids: list[str],
    id_to_emb: dict[str, np.ndarray],
    id_to_campaign: dict[str, Any],
    eid_to_row: dict[str, dict[str, Any]],
    *,
    k: int = 8,
    model_name: str = "",
) -> pd.DataFrame:
    ranked = top_k_neighbors(query_eid, pool_ids, id_to_emb, k=k)
    rows = []
    for rank, (ne_eid, sim) in enumerate(ranked, start=1):
        pv = row_email_previews(ne_eid, eid_to_row)
        rows.append(
            {
                "model": model_name,
                "query_external_id": str(query_eid),
                "rank": rank,
                "neighbor_external_id": ne_eid,
                "neighbor_campaign_id": id_to_campaign.get(ne_eid),
                "cosine_similarity": sim,
                "sender": pv["sender"],
                "subject": pv["subject"],
                "body_preview": pv["body_preview"],
                "urls_preview": pv["urls_preview"],
                "domains_preview": pv["domains_preview"],
            }
        )
    return pd.DataFrame(rows)


def drilldown_neighbor_comparison_block(
    query_eids: list[str],
    pool_ids: list[str],
    id_to_emb_raw: dict[str, np.ndarray],
    id_to_emb_gnn: dict[str, np.ndarray],
    id_to_campaign: dict[str, Any],
    eid_to_row: dict[str, dict[str, Any]],
    *,
    k: int = 8,
) -> pd.DataFrame:
    parts = []
    for q in query_eids:
        parts.append(
            neighbor_ranking_table(
                q, pool_ids, id_to_emb_raw, id_to_campaign, eid_to_row, k=k, model_name="RAW"
            )
        )
        parts.append(
            neighbor_ranking_table(
                q, pool_ids, id_to_emb_gnn, id_to_campaign, eid_to_row, k=k, model_name="GNN"
            )
        )
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def drilldown_fragmentation_summary(
    campaign_id: Any,
    campaign_to_members: dict[Any, list[str]],
    pred_raw: dict[str, int],
    pred_gnn: dict[str, int],
) -> pd.DataFrame:
    rows = []
    for name, pmap in (("RAW", pred_raw), ("GNN", pred_gnn)):
        m = [str(x) for x in campaign_to_members.get(campaign_id, [])]
        assigned = [eid for eid in m if eid in pmap and pmap[eid] != -1]
        preds = [pmap[eid] for eid in assigned]
        if not preds:
            rows.append(
                {
                    "campaign_id": campaign_id,
                    "model": name,
                    "n_members_gt": len(m),
                    "n_assigned_non_noise": 0,
                    "n_pred_clusters_touching": 0,
                    "largest_pred_overlap": 0,
                    "dominant_fraction": 0.0,
                }
            )
            continue
        ct = Counter(preds)
        top = max(ct.values())
        rows.append(
            {
                "campaign_id": campaign_id,
                "model": name,
                "n_members_gt": len(m),
                "n_assigned_non_noise": len(assigned),
                "n_pred_clusters_touching": len(ct),
                "largest_pred_overlap": int(top),
                "dominant_fraction": float(top) / max(1, len(m)),
            }
        )
    return pd.DataFrame(rows)


def top_campaign_shortlists(
    frag: pd.DataFrame,
    *,
    nn_merged: pd.DataFrame | None = None,
    top_n: int = 10,
) -> dict[str, pd.DataFrame]:
    """Compact pick-lists for qualitative follow-up (requires merged `frag` table)."""
    col_keep = [
        "campaign_id",
        "n_members_gt_raw",
        "n_pred_clusters_touching_raw",
        "n_pred_clusters_touching_gnn",
        "dominant_fraction_raw",
        "dominant_fraction_gnn",
        "delta_pred_clusters",
    ]
    col_keep = [c for c in col_keep if c in frag.columns]
    out: dict[str, pd.DataFrame] = {}
    if "delta_pred_clusters" in frag.columns:
        out["gnn_less_fragmented"] = frag.nsmallest(top_n, "delta_pred_clusters")[
            col_keep
        ].copy()
        out["raw_less_fragmented"] = frag.nlargest(top_n, "delta_pred_clusters")[
            col_keep
        ].copy()
    if "n_members_gt_raw" in frag.columns:
        out["largest_campaigns"] = frag.nlargest(top_n, "n_members_gt_raw")[
            col_keep
        ].copy()
    if nn_merged is not None:
        recall_deltas = [
            c for c in nn_merged.columns if c.startswith("delta_mean_intra_recall_at_")
        ]
        if recall_deltas:
            col = sorted(recall_deltas)[-1]
            size_col = "size_raw" if "size_raw" in nn_merged.columns else "size"
            base = ["campaign_id", size_col, col]
            base = [c for c in base if c in nn_merged.columns]
            out["top_intra_recall_delta_gnn_minus_raw"] = nn_merged.nlargest(
                top_n, col
            )[base].copy()
            out["top_intra_recall_delta_raw_minus_gnn"] = nn_merged.nsmallest(
                top_n, col
            )[base].copy()
    return out
