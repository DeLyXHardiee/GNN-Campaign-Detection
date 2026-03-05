from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import umap
from collections import Counter
import re
from collections import Counter
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.metrics.cluster import homogeneity_score, completeness_score, v_measure_score
from collections import Counter
from sklearn.decomposition import PCA


def visualize_email_clusters(email_vecs, labels, max_points=5000, title="Email embeddings (2D)"):
    labels = np.asarray(labels)
    email_vecs = np.asarray(email_vecs)

    # Subsample consistently
    if len(email_vecs) > max_points:
        idx = np.random.choice(len(email_vecs), size=max_points, replace=False)
        vecs_plot = email_vecs[idx]
        labels_plot = labels[idx]
    else:
        vecs_plot = email_vecs
        labels_plot = labels

    # 2D projection for plotting only
    coords = None
    try:
        reducer = umap.UMAP(n_components=2, random_state=42, metric="cosine")
        coords = reducer.fit_transform(vecs_plot)
    except Exception:
        coords = PCA(n_components=2, random_state=42).fit_transform(vecs_plot)

    # Summary should be computed on the FULL labels (not subsample)
    non_noise = labels[labels >= 0]
    n_clusters = len(set(non_noise))
    noise_count = int((labels == -1).sum())
    sizes = Counter(non_noise)
    largest = max(sizes.values()) if sizes else 0
    avg_size = float(np.mean(list(sizes.values()))) if sizes else 0.0

    print("📊 Clustering Summary (FULL dataset):")
    print(f" - Number of clusters (excluding noise): {n_clusters}")
    print(f" - Number of noise emails: {noise_count}")
    print(f" - Average cluster size: {avg_size:.2f}")
    print(f" - Largest cluster size: {largest}")

    # Plot uses subsample
    plt.figure(figsize=(9, 6))
    plt.scatter(coords[:, 0], coords[:, 1], c=labels_plot, cmap="tab20", s=6, alpha=0.7)
    plt.title(title)
    plt.xticks([]); plt.yticks([])
    plt.show()


def list_top_clusters(labels: np.ndarray, top_n: int = 10, include_noise: bool = False) -> List[Tuple[int, int]]:
    """
    Returns a list of (cluster_id, size) sorted by size descending.
    By default excludes noise cluster (-1).
    """
    labels = np.asarray(labels)
    if include_noise:
        counts = Counter(labels)
    else:
        counts = Counter(labels[labels >= 0])
    return counts.most_common(top_n)


def get_cluster_indices(labels: np.ndarray, cluster_id: int) -> np.ndarray:
    """
    Returns the indices of items belonging to `cluster_id`.
    """
    labels = np.asarray(labels)
    return np.where(labels == cluster_id)[0]


def show_cluster_emails(
    bodies: List[str],
    labels: np.ndarray,
    cluster_id: int,
    *,
    max_emails: int = 20,
    max_chars: int = 500,
    shuffle: bool = False,
    show_indices: bool = True,
) -> np.ndarray:
    """
    Prints up to `max_emails` email bodies in the given cluster.
    Returns the indices of all emails in that cluster (not just the shown ones).
    """
    idx = get_cluster_indices(labels, cluster_id)

    if idx.size == 0:
        print(f"No emails found for cluster_id={cluster_id}")
        return idx

    if shuffle:
        idx = idx.copy()
        np.random.shuffle(idx)

    print(f"\n=== Cluster {cluster_id} ===")
    print(f"Size: {len(idx)} emails")
    print(f"Showing up to {min(max_emails, len(idx))} examples:\n")

    for n, i in enumerate(idx[:max_emails], start=1):
        header = f"--- {n}"
        if show_indices:
            header += f" (idx={i})"
        header += " ---"
        print(header)
        print(bodies[i][:max_chars])
        print()

    return idx


def show_largest_cluster_emails(
    bodies: List[str],
    labels: np.ndarray,
    *,
    rank: int = 1,
    max_emails: int = 20,
    max_chars: int = 500,
    shuffle: bool = False,
) -> int:
    """
    Finds the `rank`-th largest non-noise cluster and prints its emails.
    Returns the chosen cluster_id.
    """
    top = list_top_clusters(labels, top_n=max(rank, 10), include_noise=False)
    if not top:
        print("No non-noise clusters found.")
        return -1

    if rank < 1 or rank > len(top):
        raise ValueError(f"rank must be in [1, {len(top)}], got {rank}")

    cluster_id, size = top[rank - 1]
    show_cluster_emails(
        bodies, labels, cluster_id,
        max_emails=max_emails,
        max_chars=max_chars,
        shuffle=shuffle,
        show_indices=True
    )
    return cluster_id


EmailId = Union[int, str]
CampaignId = Union[int, str]


# -----------------------------
# A) Load manual ground truth
# -----------------------------
def load_campaign_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"campaign_label", "email_ids"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"campaigns.csv missing columns: {missing}. Found: {list(df.columns)}")
    return df


def _parse_campaign_label(raw: str) -> CampaignId:
    """
    Turns labels like "Campaign 12" into 12, otherwise keeps the string.
    """
    raw = str(raw).strip()
    m = re.search(r"\d+", raw)
    return int(m.group()) if m else raw


def _parse_email_ids(raw: str) -> List[EmailId]:
    """
    Parses a comma-separated list of ids from campaigns.csv.
    Tries int first; falls back to string.
    """
    if raw is None:
        return []
    items = [x.strip() for x in str(raw).split(",")]
    out: List[EmailId] = []
    for it in items:
        if not it:
            continue
        try:
            out.append(int(it))
        except ValueError:
            out.append(it)
    return out


def build_ground_truth_map(campaigns_df: pd.DataFrame) -> Dict[EmailId, CampaignId]:
    """
    Returns mapping: email_id -> campaign_id
    """
    gt: Dict[EmailId, CampaignId] = {}
    for _, row in campaigns_df.iterrows():
        cid = _parse_campaign_label(row.get("campaign_label", ""))
        ids = _parse_email_ids(row.get("email_ids", ""))
        for eid in ids:
            gt[eid] = cid
    return gt


# -----------------------------------------
# B) Internal metrics (no ground truth)
# -----------------------------------------
def compute_internal_metrics(
    embeddings: np.ndarray,
    pred_labels: np.ndarray,
) -> Dict[str, float]:
    """
    Computes silhouette, Davies-Bouldin, Calinski-Harabasz on clustered (non-noise) points.
    Returns safe defaults if not computable.
    """
    embeddings = np.asarray(embeddings)
    pred_labels = np.asarray(pred_labels)

    mask = pred_labels != -1
    usable_labels = pred_labels[mask]

    # Need at least 2 clusters and >1 sample
    uniq = np.unique(usable_labels)
    if mask.sum() > 1 and len(uniq) > 1:
        sil = float(silhouette_score(embeddings[mask], usable_labels))
        db = float(davies_bouldin_score(embeddings[mask], usable_labels))
        ch = float(calinski_harabasz_score(embeddings[mask], usable_labels))
    else:
        sil, db, ch = -1.0, float("inf"), 0.0

    n_clusters = int(len(set(pred_labels)) - (1 if -1 in pred_labels else 0))
    n_noise = int((pred_labels == -1).sum())

    return {
        "silhouette": sil,
        "davies_bouldin": db,
        "calinski_harabasz": ch,
        "n_clusters": n_clusters,
        "n_noise": n_noise,
    }


# -----------------------------------------
# C) External metrics (needs ground truth)
# -----------------------------------------
def compute_external_metrics(
    pred_labels: np.ndarray,
    email_ids: np.ndarray,
    ground_truth_map: Dict[EmailId, CampaignId],
    ignore_noise: bool = True,
) -> Dict[str, float]:
    """
    Computes homogeneity / completeness / v-measure using only emails that:
      - exist in ground_truth_map, and
      - (optionally) are not noise (-1)

    IMPORTANT: This requires that `email_ids[i]` matches the IDs used in campaigns.csv.
    Most often, email_ids is just np.arange(N) (row indices).
    """
    pred_labels = np.asarray(pred_labels)
    email_ids = np.asarray(email_ids)

    # Select emails that have a ground truth campaign label
    has_gt = np.array([eid in ground_truth_map for eid in email_ids], dtype=bool)

    if ignore_noise:
        mask = has_gt & (pred_labels != -1)
    else:
        mask = has_gt

    if mask.sum() == 0:
        # No overlap between clustering output and ground truth ids
        return {
            "homogeneity": 0.0,
            "completeness": 0.0,
            "v_measure": 0.0,
            "n_labeled_used": 0,
            "coverage_labeled": 0.0,
            "n_labeled_total": int(has_gt.sum()),
        }

    # Build aligned arrays
    y_true = np.array([ground_truth_map[eid] for eid in email_ids[mask]])
    y_pred = pred_labels[mask]

    # sklearn metrics accept non-numeric labels too (strings) but numpy array dtype=object is ok
    hom = float(homogeneity_score(y_true, y_pred))
    comp = float(completeness_score(y_true, y_pred))
    vmeas = float(v_measure_score(y_true, y_pred))

    n_labeled_total = int(has_gt.sum())
    n_labeled_used = int(mask.sum())

    # coverage among labeled = how many labeled emails were assigned a non-noise cluster
    if ignore_noise:
        coverage = n_labeled_used / max(1, n_labeled_total)
    else:
        coverage = 1.0  # if you include noise in evaluation mask, "used" == "total"

    return {
        "homogeneity": hom,
        "completeness": comp,
        "v_measure": vmeas,
        "n_labeled_used": n_labeled_used,
        "n_labeled_total": n_labeled_total,
        "coverage_labeled": float(coverage),
    }


# -----------------------------------------
# D) One convenience function
# -----------------------------------------
def evaluate_clustering(
    embeddings: np.ndarray,
    pred_labels: np.ndarray,
    ground_truth_path: str,
    email_ids: Optional[np.ndarray] = None,
    ignore_noise_for_external: bool = True,
) -> Dict[str, float]:
    """
    Computes both internal + external metrics, returns a single dict.

    - embeddings: (N, D)
    - pred_labels: (N,) with -1 as noise
    - email_ids: (N,) ids corresponding to each row (defaults to np.arange(N))
    - ground_truth_path: path to manually labeled campaigns file
    """
    embeddings = np.asarray(embeddings)
    pred_labels = np.asarray(pred_labels)

    if email_ids is None:
        email_ids = np.arange(len(pred_labels))
    else:
        email_ids = np.asarray(email_ids)

    if len(pred_labels) != embeddings.shape[0] or len(email_ids) != embeddings.shape[0]:
        raise ValueError(
            f"Alignment error: embeddings has {embeddings.shape[0]} rows, "
            f"labels has {len(pred_labels)}, email_ids has {len(email_ids)}"
        )

    campaigns_df = load_campaign_csv(ground_truth_path)
    gt_map = build_ground_truth_map(campaigns_df)

    internal = compute_internal_metrics(embeddings, pred_labels)
    external = compute_external_metrics(
        pred_labels=pred_labels,
        email_ids=email_ids,
        ground_truth_map=gt_map,
        ignore_noise=ignore_noise_for_external,
    )

    return {**internal, **external}


