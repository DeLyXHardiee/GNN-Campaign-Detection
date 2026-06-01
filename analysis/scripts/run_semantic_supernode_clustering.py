"""
First-pass semantic **redundancy** clustering on existing email BERT embeddings.

Builds a cosine-threshold graph (optional mutual top-k) and clusters by connected
components. Writes membership CSV + a summary JSON with purity-first GT
diagnostics. Optionally writes **MISP-backed inspection JSON** for the three
largest clusters (``subject_lines`` / ``body_lines`` per email for file readability) unless
``--skip-misp-inspection-export``. Translated text from ``translate_misp_email_texts_to_en.py``
is **off by default** (raw MISP for inspection). Pass ``--use-misp-translated-inspection``
to prefer English from ``--misp-translated-json`` when that file exists.

This does **not** rebuild graphs, seeds, or training — it only validates whether
conservative semantic super-nodes are trustworthy enough for a later pipeline stage.
"""
from __future__ import annotations

import argparse
import json
import sys
import textwrap
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

_script_dir = Path(__file__).resolve().parent
if str(_script_dir) not in sys.path:
    sys.path.insert(0, str(_script_dir))

from misp_email_text_catalog import (
    find_project_root,
    load_misp_subject_body_by_external_id,
    load_translated_email_text_by_external_id,
)

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.metrics import homogeneity_completeness_v_measure
from sklearn.neighbors import kneighbors_graph, radius_neighbors_graph



def _wrap_display_text(text: str, width: int) -> str:
    """Break long lines for JSON inspection (preserves blank lines between paragraphs)."""
    if width <= 0 or not text:
        return text
    parts: list[str] = []
    for para in text.splitlines():
        if not para.strip():
            parts.append("")
            continue
        parts.append(
            textwrap.fill(
                para,
                width=width,
                break_long_words=True,
                replace_whitespace=False,
            )
        )
    return "\n".join(parts)


def _inspection_text_as_line_list(text: str, wrap_width: int) -> list[str]:
    """
    Lines for inspection JSON. Using a list (not a single string with ``\\n``)
    lets ``json.dumps(..., indent=2)`` put each line on its own row in the file.
    """
    raw = text or ""
    if wrap_width <= 0:
        return raw.splitlines()
    return _wrap_display_text(raw, wrap_width).splitlines()


def most_dissimilar_pair_by_embedding(
    X: np.ndarray,
    member_eids: list[str],
    idx: np.ndarray,
    *,
    exact_max_n: int = 2500,
    subsample: int = 400,
) -> dict[str, Any] | None:
    """
    Pair of members with **minimum** cosine similarity (most embedding-dissimilar).
    Exact if cluster size <= exact_max_n; else approximate over a fixed random subset.
    """
    n = len(member_eids)
    if n < 2 or idx.size < 2:
        return None

    def pack_result(i_local: int, j_local: int, method: str) -> dict[str, Any]:
        ea, eb = member_eids[i_local], member_eids[j_local]
        sim = float(X[idx[i_local]] @ X[idx[j_local]])
        return {
            "external_id_a": ea,
            "external_id_b": eb,
            "cosine_similarity": sim,
            "method": method,
        }

    if n <= exact_max_n:
        M = X[idx]
        sims = M @ M.T
        np.fill_diagonal(sims, np.inf)
        flat = int(np.argmin(sims))
        li, lj = np.unravel_index(flat, sims.shape)
        if li == lj:
            return None
        return pack_result(int(li), int(lj), "exact_min_pairwise_all_members")

    rng = np.random.default_rng(42)
    sub = np.sort(rng.choice(n, size=min(subsample, n), replace=False))
    idx_s = idx[sub]
    M = X[idx_s]
    sims = M @ M.T
    np.fill_diagonal(sims, np.inf)
    flat = int(np.argmin(sims))
    li, lj = np.unravel_index(flat, sims.shape)
    ia, ib = int(sub[li]), int(sub[lj])
    sim = float(X[idx[ia]] @ X[idx[ib]])
    return {
        "external_id_a": member_eids[ia],
        "external_id_b": member_eids[ib],
        "cosine_similarity": sim,
        "method": f"approximate_min_over_random_{len(sub)}_of_{n}_members",
    }


def _min_exact_pairwise_cosine(X: np.ndarray, idx: np.ndarray) -> float:
    if idx.size < 2:
        return float("nan")
    M = X[idx]
    g = M @ M.T
    np.fill_diagonal(g, np.inf)
    return float(g.min())


def cluster_embedding_overview(
    X: np.ndarray, idx: np.ndarray, *, max_exact_min_pairs: int = 200
) -> dict[str, Any]:
    """Pairwise cosine stats on L2-normalized rows (idx = row indices into X)."""
    n = int(idx.size)
    if n < 2:
        return {
            "n_emails": n,
            "mean_pairwise_cosine_similarity": float("nan"),
            "min_pairwise_cosine_similarity": float("nan"),
            "max_pairwise_cosine_distance": float("nan"),
            "note": "singleton_cluster_no_pairwise_stats",
        }
    mean_cos = _mean_intra_cluster_cosine(X, idx)
    if n <= max_exact_min_pairs:
        min_cos = _min_exact_pairwise_cosine(X, idx)
    else:
        min_cos = _min_sampled_intra_cosine(X, idx, max_pairs=8000)
    max_dist = float(1.0 - min_cos) if np.isfinite(min_cos) else float("nan")
    return {
        "n_emails": n,
        "mean_pairwise_cosine_similarity": float(mean_cos),
        "min_pairwise_cosine_similarity": float(min_cos),
        "max_pairwise_cosine_distance": max_dist,
        "min_cosine_note": (
            "exact_all_pairs"
            if n <= max_exact_min_pairs
            else "min_is_sampled_for_large_clusters_see_script"
        ),
    }


def export_largest_cluster_inspection_jsons(
    *,
    out_dir: Path,
    ids: list[str],
    labels: np.ndarray,
    X: np.ndarray,
    id_index: dict[str, int],
    rep_map: dict[int, str],
    misp_by_eid: dict[str, dict[str, str]],
    misp_source: Path,
    translated_by_eid: dict[str, dict[str, str]] | None,
    translated_source: Path | None,
    cosine_threshold: float,
    inspection_text_wrap_width: int,
    n_ranks: int = 3,
) -> list[Path]:
    """
    Write one JSON per rank (1=largest …) with cluster_overview + emails
    (external_id, subject_lines, body_lines). Line arrays keep the file readable;
    join with ``\\n`` if you need a single string. Translated sidecar used only when caller loads it.
    """
    n_clusters = int(labels.max()) + 1 if labels.size else 0
    written: list[Path] = []
    for rank in range(1, min(n_ranks, n_clusters) + 1):
        cid = rank - 1
        member_eids = [eid for eid in ids if int(labels[id_index[eid]]) == cid]
        idx = np.array([id_index[e] for e in member_eids], dtype=np.int64)
        emb_stats = cluster_embedding_overview(X, idx)

        emails_out: list[dict[str, Any]] = []
        n_hit_raw = 0
        n_used_translated = 0
        for eid in member_eids:
            raw_rec = misp_by_eid.get(eid)
            if raw_rec is not None:
                n_hit_raw += 1
            ru = raw_rec or {"subject": "", "body": ""}
            if translated_by_eid and eid in translated_by_eid:
                tr = translated_by_eid[eid]
                subject = (str(tr.get("subject") or "").strip() or str(ru.get("subject") or ""))
                body = (str(tr.get("body") or "").strip() or str(ru.get("body") or ""))
                n_used_translated += 1
            else:
                subject = str(ru.get("subject") or "")
                body = str(ru.get("body") or "")
            emails_out.append(
                {
                    "external_id": eid,
                    "subject_lines": _inspection_text_as_line_list(
                        subject, inspection_text_wrap_width
                    ),
                    "body_lines": _inspection_text_as_line_list(
                        body, inspection_text_wrap_width
                    ),
                }
            )

        pair_info = most_dissimilar_pair_by_embedding(X, member_eids, idx)
        by_eid = {e["external_id"]: e for e in emails_out}
        ordered: list[dict[str, Any]] = []
        seen: set[str] = set()
        if pair_info:
            for key in ("external_id_a", "external_id_b"):
                eid = str(pair_info.get(key) or "")
                if eid and eid in by_eid and eid not in seen:
                    ordered.append(dict(by_eid[eid]))
                    seen.add(eid)
        for e in emails_out:
            eid = e["external_id"]
            if eid not in seen:
                ordered.append(dict(e))
                seen.add(eid)
        emails_out = ordered

        overview: dict[str, Any] = {
            "rank_by_cluster_size": rank,
            "semantic_cluster_id": cid,
            "cluster_size": len(member_eids),
            "representative_email_id": rep_map.get(cid, ""),
            "clustering_cosine_threshold": cosine_threshold,
            "embedding_cosine_similarity": emb_stats,
            "most_dissimilar_pair_by_embedding": pair_info,
            "misp_source_path": str(misp_source.resolve()),
            "translated_catalog_path": str(translated_source.resolve())
            if translated_source
            else None,
            "inspection_text_preference": (
                "translated_then_raw_misp" if translated_by_eid else "raw_misp_only"
            ),
            "n_emails_found_in_misp_source": n_hit_raw,
            "n_emails_missing_from_misp_source": len(member_eids) - n_hit_raw,
            "n_emails_using_translated_text": n_used_translated,
            "inspection_text_wrap_width": inspection_text_wrap_width,
            "inspection_email_text_shape": "subject_lines_and_body_lines_are_json_arrays_for_readable_exports",
        }

        payload = {"cluster_overview": overview, "emails": emails_out}
        fname = f"semantic_supernode_inspect_rank{rank}_largest_cluster.json"
        out_path = out_dir / fname
        out_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        written.append(out_path)
    return written


def load_ground_truth_structures(
    gt_path: str | Path,
) -> tuple[dict[str, Any], dict[Any, list[str]]]:
    """Return (email -> campaign_id, campaign_id -> member emails)."""
    with open(gt_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    label_map: dict[str, Any] = {}
    campaign_to_members: dict[Any, list[str]] = defaultdict(list)

    for raw_key, emails in (data.get("clusters") or {}).items():
        try:
            cid = int(raw_key)
        except (TypeError, ValueError):
            cid = str(raw_key)
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
            campaign_to_members[cid].append(eid)

    return label_map, dict(campaign_to_members)


def load_text_embeddings_json(path: Path) -> tuple[list[str], np.ndarray]:
    """
    Load email-level vectors from embeddings.json (subject + body concatenated).

    Matches core/utils/OLS/OLS.py::load_text_embeddings and graph/cluster_stage
    conventions: keys under `by_key`, each entry may use `external_id` or key.
    """
    with path.open("r", encoding="utf-8-sig") as f:
        payload = json.load(f)
    by_key = payload.get("by_key")
    if not isinstance(by_key, dict):
        raise ValueError(f"Expected object 'by_key' in {path}")

    ids: list[str] = []
    rows: list[np.ndarray] = []
    seen: set[str] = set()

    reverse_index: dict[str, dict[str, Any]] = {}
    for key, entry in by_key.items():
        if not isinstance(entry, dict):
            continue
        ext = entry.get("external_id")
        if ext is not None:
            reverse_index[str(ext).strip()] = entry
        external_id = str(ext).strip() if ext is not None else str(key).strip()
        if not external_id or external_id in seen:
            continue
        subj = entry.get("subj") or []
        body = entry.get("body") or []
        vec = np.asarray([*subj, *body], dtype=np.float64)
        if vec.size == 0:
            continue
        seen.add(external_id)
        ids.append(external_id)
        rows.append(vec)

    for external_id, entry in reverse_index.items():
        if external_id in seen:
            continue
        if not isinstance(entry, dict):
            continue
        subj = entry.get("subj") or []
        body = entry.get("body") or []
        vec = np.asarray([*subj, *body], dtype=np.float64)
        if vec.size == 0:
            continue
        seen.add(external_id)
        ids.append(external_id)
        rows.append(vec)

    if not rows:
        raise ValueError(f"No usable embedding rows in {path}")

    X = np.vstack(rows)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    X = X / norms
    return ids, X


def _symmetrize_sparse(A: csr_matrix) -> csr_matrix:
    A = A.tocsr()
    AT = A.transpose().tocsr()
    return (A + AT).astype(bool).astype(np.float32)


def build_threshold_adjacency(
    X: np.ndarray,
    cosine_threshold: float,
    *,
    mutual_top_k: int | None,
    n_jobs: int,
) -> csr_matrix:
    """Undirected adjacency: edge if cosine similarity >= threshold (and optional mutual top-k)."""
    if not (0.0 < cosine_threshold <= 1.0):
        raise ValueError("cosine_threshold must be in (0, 1].")
    radius = float(1.0 - cosine_threshold)
    adj = radius_neighbors_graph(
        X,
        radius=radius,
        mode="connectivity",
        metric="cosine",
        include_self=False,
        n_jobs=n_jobs,
    )
    adj = _symmetrize_sparse(adj).astype(np.float32)

    if mutual_top_k is not None and mutual_top_k > 0:
        n = X.shape[0]
        k = min(int(mutual_top_k), max(1, n - 1))
        knn = kneighbors_graph(
            X,
            n_neighbors=k,
            mode="connectivity",
            metric="cosine",
            include_self=False,
            n_jobs=n_jobs,
        )
        mutual = knn.multiply(knn.transpose())
        adj = adj.multiply(mutual).astype(np.float32)

    adj.setdiag(0)
    adj.eliminate_zeros()
    return adj


def apply_max_component_size(
    adj: csr_matrix,
    X: np.ndarray,
    cosine_threshold: float,
    max_size: int,
) -> csr_matrix:
    """
    Greedy edge addition (Kruskal-style): edges sorted by descending cosine;
    merge only if resulting component size <= max_size.

    Replaces plain threshold-graph CC when chaining produces oversized components.
    """
    if max_size < 2:
        return adj
    adj = adj.tocoo()
    rows = adj.row.astype(np.int64)
    cols = adj.col.astype(np.int64)
    mask = rows < cols
    rows, cols = rows[mask], cols[mask]
    if rows.size == 0:
        return csr_matrix((X.shape[0], X.shape[0]), dtype=np.float32)

    sims = np.sum(X[rows] * X[cols], axis=1)
    order = np.argsort(-sims)
    rows, cols, sims = rows[order], cols[order], sims[order]

    n = X.shape[0]
    parent = np.arange(n, dtype=np.int64)
    size = np.ones(n, dtype=np.int64)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return int(x)

    def union(a: int, b: int) -> bool:
        ra, rb = find(a), find(b)
        if ra == rb:
            return False
        if size[ra] + size[rb] > max_size:
            return False
        if size[ra] < size[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        size[ra] += size[rb]
        return True

    kept_rows: list[int] = []
    kept_cols: list[int] = []
    for i in range(len(rows)):
        r, c = int(rows[i]), int(cols[i])
        if sims[i] + 1e-12 < cosine_threshold:
            continue
        if union(r, c):
            kept_rows.extend((r, c))
            kept_cols.extend((c, r))

    data = np.ones(len(kept_rows), dtype=np.float32)
    out = csr_matrix((data, (kept_rows, kept_cols)), shape=(n, n), dtype=np.float32)
    return out


def cluster_connected_components(adj: csr_matrix) -> np.ndarray:
    n_components, labels = connected_components(adj, directed=False, connection="weak")
    return labels.astype(np.int64)


def _percentile(vals: np.ndarray, q: float) -> float:
    if vals.size == 0:
        return float("nan")
    return float(np.percentile(vals, q))


def _hist_counts(sizes: np.ndarray) -> dict[str, int]:
    bins = [1, 2, 3, 4, 5, 6, 11, 21, 51, 10**9]
    labels = ["1", "2", "3", "4", "5", "6-10", "11-20", "21-50", "51+"]
    out: dict[str, int] = {lab: 0 for lab in labels}
    for s in sizes:
        for i in range(len(bins) - 1):
            if bins[i] <= s < bins[i + 1]:
                out[labels[i]] += 1
                break
    return out


def _mean_intra_cluster_cosine(X: np.ndarray, idx: np.ndarray) -> float:
    """Exact mean pairwise cosine for L2-normalized rows in idx (n>=2)."""
    if idx.size < 2:
        return float("nan")
    M = X[idx]
    s = M.sum(axis=0)
    sq = float(s @ s)
    n = idx.size
    return float((sq - n) / (n * (n - 1)))


def _min_sampled_intra_cosine(X: np.ndarray, idx: np.ndarray, max_pairs: int = 4000) -> float:
    if idx.size < 2:
        return float("nan")
    n = idx.size
    if n <= 200:
        M = X[idx]
        g = M @ M.T
        np.fill_diagonal(g, np.inf)
        return float(g.min())
    rng = np.random.default_rng(0)
    best = 1.0
    for _ in range(max_pairs):
        i, j = rng.integers(0, n, size=2)
        if i == j:
            continue
        c = float(X[idx[i]] @ X[idx[j]])
        if c < best:
            best = c
    return float(best)


def representative_per_cluster(
    X: np.ndarray, labels: np.ndarray, ids: list[str]
) -> dict[int, str]:
    """Pick email id whose vector is most aligned with the cluster mean direction."""
    out: dict[int, str] = {}
    n_clust = int(labels.max()) + 1 if labels.size else 0
    for c in range(n_clust):
        mask = labels == c
        idx = np.flatnonzero(mask)
        if idx.size == 0:
            continue
        M = X[idx]
        centroid = M.mean(axis=0)
        nc = np.linalg.norm(centroid)
        if nc < 1e-12:
            out[c] = ids[int(idx[0])]
            continue
        centroid = centroid / nc
        scores = M @ centroid
        best_local = int(idx[int(np.argmax(scores))])
        out[c] = ids[best_local]
    return out


def collect_gt_paths(gt_arg: Path) -> list[Path]:
    if gt_arg.is_dir():
        paths = sorted(gt_arg.glob("*.json"))
        if not paths:
            raise FileNotFoundError(f"No JSON files in GT directory: {gt_arg}")
        return paths
    if not gt_arg.is_file():
        raise FileNotFoundError(f"GT path not found: {gt_arg}")
    return [gt_arg]


def evaluate_gt_file(
    gt_path: Path,
    ids: list[str],
    labels: np.ndarray,
    X: np.ndarray,
    id_index: dict[str, int],
) -> dict[str, Any]:
    label_map, campaign_to_members = load_ground_truth_structures(gt_path)

    true_ids: list[str] = []
    true_y: list[Any] = []
    pred_y: list[int] = []
    for eid in ids:
        if eid not in label_map:
            continue
        true_ids.append(eid)
        true_y.append(label_map[eid])
        pred_y.append(int(labels[id_index[eid]]))

    n_labeled = len(true_y)
    out: dict[str, Any] = {
        "ground_truth_file": str(gt_path.resolve()),
        "n_emails_with_gt": n_labeled,
        "n_emails_embedding_only": len(ids) - n_labeled,
    }

    if n_labeled < 2:
        out["note"] = "Too few labeled emails for external clustering metrics."
        return out

    uniq_pred = sorted(set(pred_y))
    pred_remap = {p: i for i, p in enumerate(uniq_pred)}
    pred_arr = np.array([pred_remap[p] for p in pred_y], dtype=np.int64)

    true_str = np.array([str(t) for t in true_y], dtype=object)
    h, co, vm = homogeneity_completeness_v_measure(true_str, pred_arr)
    out["homogeneity"] = float(h)
    out["completeness"] = float(co)
    out["v_measure"] = float(vm)
    out["interpretation"] = (
        "Purity-first redundancy clustering: target **high homogeneity**; lower "
        "completeness is acceptable if clusters stay tight and pure."
    )

    cluster_to_gt_counts: dict[int, Counter] = defaultdict(Counter)
    for eid, py in zip(true_ids, pred_y):
        cluster_to_gt_counts[py][label_map[eid]] += 1

    purities: list[float] = []
    for _c, ctr in cluster_to_gt_counts.items():
        tot = sum(ctr.values())
        if tot == 0:
            continue
        purities.append(ctr.most_common(1)[0][1] / tot)
    pur_arr = np.array(purities, dtype=np.float64) if purities else np.array([])

    out["dominant_gt_purity"] = {
        "mean": float(pur_arr.mean()) if pur_arr.size else float("nan"),
        "median": float(np.median(pur_arr)) if pur_arr.size else float("nan"),
        "fraction_clusters_purity_ge_0.95": float(np.mean(pur_arr >= 0.95))
        if pur_arr.size
        else float("nan"),
        "fraction_clusters_purity_ge_0.99": float(np.mean(pur_arr >= 0.99))
        if pur_arr.size
        else float("nan"),
        "n_semantic_clusters_with_any_gt_member": len(cluster_to_gt_counts),
    }

    spans: list[int] = []
    for _cid, members in campaign_to_members.items():
        clusters_hit = set()
        for m in members:
            if m not in id_index:
                continue
            clusters_hit.add(int(labels[id_index[m]]))
        if not clusters_hit:
            continue
        spans.append(len(clusters_hit))
    sp = np.array(spans, dtype=np.int64) if spans else np.array([])

    out["gt_campaign_fragmentation"] = {
        "n_gt_campaigns_with_any_embedded_member": int(sp.size),
        "mean_clusters_per_gt_campaign": float(sp.mean()) if sp.size else float("nan"),
        "median_clusters_per_gt_campaign": float(np.median(sp)) if sp.size else float("nan"),
        "p90_clusters_per_gt_campaign": _percentile(sp.astype(float), 90) if sp.size else float("nan"),
        "fraction_gt_campaigns_in_single_semantic_cluster": float(np.mean(sp == 1))
        if sp.size
        else float("nan"),
        "fraction_gt_campaigns_spanning_2_to_5_clusters": float(np.mean((sp >= 2) & (sp <= 5)))
        if sp.size
        else float("nan"),
        "fraction_gt_campaigns_spanning_gt_5_clusters": float(np.mean(sp > 5))
        if sp.size
        else float("nan"),
    }

    cl_indices: dict[int, list[int]] = defaultdict(list)
    for i, lab in enumerate(labels.tolist()):
        cl_indices[int(lab)].append(i)

    intra_means: list[float] = []
    intra_mins: list[float] = []
    for _c, idx_list in cl_indices.items():
        if len(idx_list) < 2:
            continue
        idx = np.array(idx_list, dtype=np.int64)
        intra_means.append(_mean_intra_cluster_cosine(X, idx))
        intra_mins.append(_min_sampled_intra_cosine(X, idx))

    im = np.array(intra_means, dtype=np.float64) if intra_means else np.array([])
    imin = np.array(intra_mins, dtype=np.float64) if intra_mins else np.array([])
    out["intra_cluster_cosine_similarity"] = {
        "mean_of_cluster_means": float(im.mean()) if im.size else float("nan"),
        "median_of_cluster_means": float(np.median(im)) if im.size else float("nan"),
        "mean_of_cluster_min_estimates": float(imin.mean()) if imin.size else float("nan"),
        "median_of_cluster_min_estimates": float(np.median(imin)) if imin.size else float("nan"),
        "n_clusters_size_ge_2": int(im.size),
        "note_min_is_exact_for_n_le_200_else_sampled": True,
    }

    return out


def run(
    *,
    project_root: Path,
    embeddings_path: Path,
    gt_paths: Iterable[Path] | None,
    out_dir: Path,
    cosine_threshold: float,
    mutual_top_k: int | None,
    max_component_size: int | None,
    n_jobs: int,
    misp_json_path: Path | None,
    misp_translated_json_path: Path | None,
    use_misp_translated_inspection: bool,
    inspection_text_wrap_width: int,
    export_misp_inspection: bool,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    ids, X = load_text_embeddings_json(embeddings_path)
    id_index = {e: i for i, e in enumerate(ids)}
    n = len(ids)

    adj = build_threshold_adjacency(
        X, cosine_threshold, mutual_top_k=mutual_top_k, n_jobs=n_jobs
    )
    n_edges_undirected = int(adj.nnz // 2)
    clustering_mode = "connected_components_on_threshold_graph"
    if max_component_size is not None and max_component_size > 0:
        adj = apply_max_component_size(adj, X, cosine_threshold, max_component_size)
        clustering_mode = (
            f"greedy_merge_by_descending_cosine_with_component_cap_{max_component_size}"
        )

    n_edges_final = int(adj.nnz // 2)
    labels = cluster_connected_components(adj)
    sizes = Counter(labels.tolist())
    sorted_clust = sorted(sizes.keys(), key=lambda c: -sizes[c])
    remap = {old: i for i, old in enumerate(sorted_clust)}
    labels = np.array([remap[int(x)] for x in labels], dtype=np.int64)

    cluster_sizes = np.array([np.sum(labels == c) for c in range(labels.max() + 1)], dtype=np.int64)
    rep_map = representative_per_cluster(X, labels, ids)

    rows = []
    for i, eid in enumerate(ids):
        c = int(labels[i])
        rep_eid = rep_map[c]
        ri = id_index[rep_eid]
        cos_to_rep = float(X[i] @ X[ri])
        rows.append(
            {
                "email_id": eid,
                "cluster_id": c,
                "cluster_size": int(cluster_sizes[c]),
                "representative_email_id": rep_eid,
                "cosine_to_representative": cos_to_rep,
            }
        )
    df = pd.DataFrame(rows)
    csv_path = out_dir / "semantic_supernode_clusters.csv"
    df.to_csv(csv_path, index=False)

    n_clusters = int(labels.max() + 1) if n else 0
    counts = cluster_sizes
    singletons = int(np.sum(counts == 1))
    n_nonsingleton_clusters = int(np.sum(counts >= 2))
    nonsingleton_emails = int(sum(int(c) for c in counts if c >= 2))

    redundant_pairs_removed = int(sum(int(c * (c - 1) // 2) for c in counts if c >= 2))
    super_nodes = n_clusters
    reduction_ratio = float(n / super_nodes) if super_nodes else float("nan")

    summary: dict[str, Any] = {
        "purpose": (
            "Semantic **redundancy-compression** groups (super-email candidates), "
            "not campaign clustering and not the final downstream graph."
        ),
        "evaluation_framing": (
            "Clusters are evaluated as semantic redundancy-compression groups for a "
            "future super-email node rebuild — not as campaign recovery. Success is "
            "primarily high homogeneity / purity, meaningful redundant-pair removal, "
            "sane cluster sizes, and tight intra-cluster cosine; completeness is secondary."
        ),
        "success_criteria_notes": [
            "Very high homogeneity / dominant-label purity",
            "Meaningful reduction in email-node count and redundant within-cluster pairs",
            "Cluster sizes not absurdly large; non-singleton clusters materially present",
            "Tight intra-cluster cosine (watch min for chaining)",
            "Completeness is secondary at this stage",
        ],
        "config": {
            "embeddings_path": str(embeddings_path.resolve()),
            "embedding_dim": int(X.shape[1]),
            "cosine_threshold": cosine_threshold,
            "mutual_top_k": mutual_top_k,
            "max_component_size": max_component_size,
            "clustering_mode": clustering_mode,
            "n_jobs": n_jobs,
            "misp_json_path": str(misp_json_path.resolve()) if misp_json_path else None,
            "misp_translated_json_path": str(misp_translated_json_path.resolve())
            if misp_translated_json_path
            else None,
            "use_misp_translated_inspection": use_misp_translated_inspection,
            "inspection_text_wrap_width": inspection_text_wrap_width,
            "export_misp_inspection": export_misp_inspection,
        },
        "similarity_graph": {
            "n_undirected_edges_before_optional_component_cap": n_edges_undirected,
            "n_undirected_edges_after_optional_component_cap": n_edges_final,
        },
        "counts": {
            "n_email_embeddings": n,
            "n_semantic_supernodes": super_nodes,
            "n_singleton_clusters": singletons,
            "n_non_singleton_clusters": n_nonsingleton_clusters,
            "singleton_rate": float(singletons / n_clusters) if n_clusters else float("nan"),
            "n_emails_in_non_singleton_clusters": nonsingleton_emails,
            "fraction_emails_in_clusters_size_ge_2": float(nonsingleton_emails / n)
            if n
            else float("nan"),
        },
        "redundancy_reduction": {
            "n_original_email_nodes": n,
            "n_supernodes": super_nodes,
            "reduction_ratio_emails_over_supernodes": reduction_ratio,
            "redundant_pairs_removed_estimate_sum_n_choose_2": redundant_pairs_removed,
        },
        "cluster_size_distribution": {
            "max": int(counts.max()) if counts.size else 0,
            "p50": _percentile(counts.astype(float), 50) if counts.size else float("nan"),
            "p90": _percentile(counts.astype(float), 90) if counts.size else float("nan"),
            "p95": _percentile(counts.astype(float), 95) if counts.size else float("nan"),
            "histogram": _hist_counts(counts),
        },
        "artifacts": {
            "cluster_membership_csv": str(csv_path.resolve()),
        },
    }

    if gt_paths:
        summary["ground_truth_evaluations"] = []
        for gtp in gt_paths:
            summary["ground_truth_evaluations"].append(
                evaluate_gt_file(gtp, ids, labels, X, id_index)
            )

    inspection_paths: list[str] = []
    inspection_note: str | None = None
    if export_misp_inspection and misp_json_path is not None:
        if not misp_json_path.is_file():
            inspection_note = f"misp_json_not_found_skipped_inspection_export: {misp_json_path}"
        else:
            misp_map = load_misp_subject_body_by_external_id(
                misp_json_path, project_root=project_root
            )
            translated_by_eid: dict[str, dict[str, str]] | None = None
            translated_path_used: Path | None = None
            if (
                use_misp_translated_inspection
                and misp_translated_json_path is not None
                and misp_translated_json_path.is_file()
            ):
                translated_by_eid = load_translated_email_text_by_external_id(
                    misp_translated_json_path
                )
                translated_path_used = misp_translated_json_path
            paths = export_largest_cluster_inspection_jsons(
                out_dir=out_dir,
                ids=ids,
                labels=labels,
                X=X,
                id_index=id_index,
                rep_map=rep_map,
                misp_by_eid=misp_map,
                misp_source=misp_json_path,
                translated_by_eid=translated_by_eid,
                translated_source=translated_path_used,
                cosine_threshold=cosine_threshold,
                inspection_text_wrap_width=inspection_text_wrap_width,
                n_ranks=3,
            )
            inspection_paths = [str(p.resolve()) for p in paths]
    elif export_misp_inspection:
        inspection_note = "misp_json_path_not_set_skipped_inspection_export"

    if inspection_paths:
        summary["artifacts"]["inspection_cluster_json"] = inspection_paths
    if inspection_note:
        summary["inspection_export_note"] = inspection_note

    json_path = out_dir / "semantic_supernode_cluster_summary.json"
    summary["artifacts"]["summary_json"] = str(json_path.resolve())
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=False)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    root = find_project_root()
    default_emb = root / "core" / "utils" / "embeddings" / "output" / "embeddings.json"
    default_out = root / "output" / "analysis" / "semantic_supernode_clustering"
    default_misp = root / "data" / "misp" / "incidents-lake-misp.json"
    default_translated = root / "data" / "misp" / "incidents-lake-misp-text-en.by_external_id.json"

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--embeddings",
        type=Path,
        default=default_emb,
        help="Path to embeddings.json (email-level subject+body BERT vectors).",
    )
    p.add_argument(
        "--gt",
        type=Path,
        default=None,
        help="Ground-truth JSON file or directory of *.json (same schema as pipeline GT).",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=default_out,
        help="Output directory for CSV + summary JSON.",
    )
    p.add_argument(
        "--cosine-threshold",
        type=float,
        default=0.95,
        help="Connect emails if cosine similarity >= this value (main tuning knob).",
    )
    p.add_argument(
        "--mutual-top-k",
        type=int,
        default=None,
        help="If set, keep an edge only if it is also a mutual top-k cosine neighbor.",
    )
    p.add_argument(
        "--max-component-size",
        type=int,
        default=None,
        help=(
            "Optional safety cap: use greedy merging by descending cosine instead of "
            "plain CC so no component exceeds this size (may reduce chaining)."
        ),
    )
    p.add_argument(
        "--misp-json",
        type=Path,
        default=default_misp,
        help=(
            "MISP incidents JSON (lake format) for subject/body inspection exports. "
            "Ignored if --skip-misp-inspection-export is set."
        ),
    )
    p.add_argument(
        "--misp-translated-json",
        type=Path,
        default=default_translated,
        help=(
            "Sidecar JSON from translate_misp_email_texts_to_en.py. Used only when "
            "--use-misp-translated-inspection is set and this path exists."
        ),
    )
    p.add_argument(
        "--use-misp-translated-inspection",
        action="store_true",
        help=(
            "Use translated subject/body from --misp-translated-json when present. "
            "Default is raw MISP only (faster iteration while translation runs)."
        ),
    )
    p.add_argument(
        "--inspection-text-wrap-width",
        type=int,
        default=100,
        help=(
            "Wrap width for inspection JSON: subject_lines/body_lines are split so each "
            "physical line is a short JSON string (0 = no wrap, only existing newlines)."
        ),
    )
    p.add_argument(
        "--skip-misp-inspection-export",
        action="store_true",
        help="Do not write semantic_supernode_inspect_rank*.json files.",
    )
    p.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Parallel jobs for sklearn neighbor queries (-1 = all cores).",
    )
    p.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional JSON file with keys matching long option names (cosine_threshold, ...).",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.config is not None:
        cfg = json.loads(args.config.read_text(encoding="utf-8"))
        path_keys = {"embeddings", "gt", "out_dir", "misp_json", "misp_translated_json"}
        for k, v in cfg.items():
            key = k.replace("-", "_")
            if key in path_keys and v is not None:
                setattr(args, key, Path(str(v)))
            elif key == "gt" and v is None:
                args.gt = None
            elif hasattr(args, key):
                setattr(args, key, v)

    if not args.embeddings.is_file():
        raise FileNotFoundError(
            f"Embeddings not found: {args.embeddings}\n"
            "Generate or copy embeddings.json into core/utils/embeddings/output first."
        )

    gt_list: list[Path] | None = None
    if args.gt is not None:
        gt_list = collect_gt_paths(args.gt)

    run(
        project_root=find_project_root(),
        embeddings_path=args.embeddings,
        gt_paths=gt_list,
        out_dir=args.out_dir,
        cosine_threshold=args.cosine_threshold,
        mutual_top_k=args.mutual_top_k,
        max_component_size=args.max_component_size,
        n_jobs=args.n_jobs,
        misp_json_path=None if args.skip_misp_inspection_export else args.misp_json,
        misp_translated_json_path=None if args.skip_misp_inspection_export else args.misp_translated_json,
        use_misp_translated_inspection=(
            False if args.skip_misp_inspection_export else args.use_misp_translated_inspection
        ),
        inspection_text_wrap_width=args.inspection_text_wrap_width,
        export_misp_inspection=not args.skip_misp_inspection_export,
    )
    print(f"Wrote {args.out_dir / 'semantic_supernode_clusters.csv'}")
    print(f"Wrote {args.out_dir / 'semantic_supernode_cluster_summary.json'}")
    if not args.skip_misp_inspection_export and args.misp_json.is_file():
        for r in (1, 2, 3):
            pth = args.out_dir / f"semantic_supernode_inspect_rank{r}_largest_cluster.json"
            if pth.is_file():
                print(f"Wrote {pth}")


if __name__ == "__main__":
    main()
