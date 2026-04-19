"""
Teacher-community pseudo labels for email-level supervised contrastive experiments.

Parses JSON shaped like ``data/pseudo_ground_truth_no_gt_shard_graph.json``:
top-level ``clusters`` maps teacher community id -> list of email records.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import pandas as pd


@dataclass
class TeacherLookupTables:
    """Index structures for positive / negative sampling."""

    email_to_teacher_cluster: dict[str, str]
    email_to_shard: dict[str, str]
    teacher_cluster_to_emails: dict[str, list[str]]
    teacher_cluster_to_shards: dict[str, list[str]]
    shard_to_emails: dict[str, list[str]]

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "email_to_teacher_cluster": dict(self.email_to_teacher_cluster),
            "email_to_shard": dict(self.email_to_shard),
            "teacher_cluster_to_emails": {
                k: list(v) for k, v in self.teacher_cluster_to_emails.items()
            },
            "teacher_cluster_to_shards": {
                k: list(v) for k, v in self.teacher_cluster_to_shards.items()
            },
            "shard_to_emails": {k: list(v) for k, v in self.shard_to_emails.items()},
        }


def parse_teacher_community_json(path: Path | str) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Flatten ``clusters`` into one row per email.

    Columns: external_id, teacher_cluster_id, shard_id, hdbscan_cluster_label, is_hdbscan_noise

    Returns ``(df, stats)``. Duplicate ``external_id`` rows are dropped (first wins).
    """
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        doc = json.load(f)
    clusters = doc.get("clusters")
    if not isinstance(clusters, Mapping):
        raise ValueError("teacher JSON must contain object 'clusters'")

    rows: list[dict[str, Any]] = []
    for cid_raw, members in clusters.items():
        tcid = str(cid_raw)
        if not isinstance(members, list):
            raise ValueError(f"clusters[{tcid!r}] must be a list")
        for rec in members:
            if not isinstance(rec, Mapping):
                raise ValueError(f"clusters[{tcid!r}] entries must be objects")
            rows.append(
                {
                    "external_id": str(rec["external_id"]),
                    "teacher_cluster_id": tcid,
                    "shard_id": str(rec["shard_id"]),
                    "hdbscan_cluster_label": int(rec["hdbscan_cluster_label"]),
                    "is_hdbscan_noise": bool(rec["is_hdbscan_noise"]),
                }
            )

    df = pd.DataFrame(rows)
    stats: dict[str, Any] = {"n_rows_raw": int(len(df))}
    if df.empty:
        stats["n_duplicate_external_ids_dropped"] = 0
        return df, stats

    n_before = len(df)
    dup_mask = df["external_id"].duplicated(keep=False)
    stats["n_rows_with_duplicate_id"] = int(dup_mask.sum())
    df = df.drop_duplicates(subset=["external_id"], keep="first").reset_index(drop=True)
    stats["n_duplicate_external_ids_dropped"] = int(n_before - len(df))
    stats["n_rows_final"] = int(len(df))
    return df, stats


def build_lookup_tables(df: pd.DataFrame) -> TeacherLookupTables:
    if df.empty:
        return TeacherLookupTables({}, {}, {}, {}, {})

    email_to_teacher_cluster = dict(
        zip(df["external_id"].astype(str), df["teacher_cluster_id"].astype(str), strict=False)
    )
    email_to_shard = dict(
        zip(df["external_id"].astype(str), df["shard_id"].astype(str), strict=False)
    )

    teacher_cluster_to_emails: dict[str, list[str]] = defaultdict(list)
    teacher_cluster_to_shards: dict[str, set[str]] = defaultdict(set)
    shard_to_emails: dict[str, list[str]] = defaultdict(list)

    for _, r in df.iterrows():
        eid = str(r["external_id"])
        tc = str(r["teacher_cluster_id"])
        sh = str(r["shard_id"])
        teacher_cluster_to_emails[tc].append(eid)
        teacher_cluster_to_shards[tc].add(sh)
        shard_to_emails[sh].append(eid)

    return TeacherLookupTables(
        email_to_teacher_cluster=email_to_teacher_cluster,
        email_to_shard=email_to_shard,
        teacher_cluster_to_emails={k: v for k, v in teacher_cluster_to_emails.items()},
        teacher_cluster_to_shards={k: sorted(v) for k, v in teacher_cluster_to_shards.items()},
        shard_to_emails={k: v for k, v in shard_to_emails.items()},
    )


@dataclass
class TrainValSplit:
    train_teacher_clusters: list[str]
    val_teacher_clusters: list[str]
    train_df: pd.DataFrame
    val_df: pd.DataFrame


def split_communities_train_val(
    df: pd.DataFrame,
    rng: np.random.Generator,
    *,
    val_fraction: float | None = None,
    val_n_communities: int | None = None,
) -> TrainValSplit:
    """
    Split by teacher community (not by email).

    Exactly one of ``val_fraction`` or ``val_n_communities`` should be set.
    """
    if (val_fraction is None) == (val_n_communities is None):
        raise ValueError("Set exactly one of val_fraction or val_n_communities")

    comms = sorted(df["teacher_cluster_id"].astype(str).unique().tolist())
    n = len(comms)
    comms_shuf = list(comms)
    rng.shuffle(comms_shuf)

    if val_fraction is not None:
        if n <= 1:
            n_val = 0
        else:
            n_val = max(1, int(round(val_fraction * n)))
            n_val = min(n_val, n - 1)
    else:
        assert val_n_communities is not None
        if n <= 1:
            n_val = 0
        else:
            n_val = min(max(1, int(val_n_communities)), n - 1)

    val_set = set(comms_shuf[:n_val]) if n_val else set()
    train_set = set(comms_shuf[n_val:]) if n_val else set(comms_shuf)

    train_df = df[df["teacher_cluster_id"].astype(str).isin(train_set)].copy()
    val_df = df[df["teacher_cluster_id"].astype(str).isin(val_set)].copy()

    return TrainValSplit(
        train_teacher_clusters=sorted(train_set),
        val_teacher_clusters=sorted(val_set),
        train_df=train_df.reset_index(drop=True),
        val_df=val_df.reset_index(drop=True),
    )


# backward-compatible name
split_teacher_communities_train_val = split_communities_train_val


def positive_counts_per_email(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """For each external_id: same-community different-shard / same-shard counts (excl. self)."""
    diff_ct: dict[str, int] = {}
    same_ct: dict[str, int] = {}

    for tc, g in df.groupby("teacher_cluster_id", sort=False):
        g = g.reset_index(drop=True)
        eids = g["external_id"].astype(str).tolist()
        shards = g["shard_id"].astype(str).tolist()
        by_shard: dict[str, list[str]] = defaultdict(list)
        for e, s in zip(eids, shards, strict=False):
            by_shard[s].append(e)

        per_shard_size = {s: len(v) for s, v in by_shard.items()}
        total = len(eids)
        for e, s in zip(eids, shards, strict=False):
            same = per_shard_size[s] - 1
            diff = total - per_shard_size[s]
            diff_ct[e] = max(0, diff)
            same_ct[e] = max(0, same)

    s_diff = pd.Series(diff_ct, name="n_pos_diff_shard")
    s_same = pd.Series(same_ct, name="n_pos_same_shard")
    return s_diff, s_same


def compute_positive_sampling_diagnostics(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        return {
            "n_emails": 0,
            "n_teacher_communities": 0,
            "emails_with_any_diff_shard_positive": 0,
            "fraction_emails_with_any_diff_shard_positive": float("nan"),
            "n_communities_single_shard_only": 0,
            "n_communities_multi_shard": 0,
            "n_communities_allow_cross_shard_positive": 0,
            "mean_diff_shard_candidates_per_anchor": float("nan"),
            "mean_same_shard_candidates_per_anchor": float("nan"),
            "median_diff_shard_candidates_per_anchor": float("nan"),
            "median_same_shard_candidates_per_anchor": float("nan"),
        }

    s_diff, s_same = positive_counts_per_email(df)
    idx = df["external_id"].astype(str)
    diff_aligned = idx.map(s_diff).fillna(0).astype(int)
    same_aligned = idx.map(s_same).fillna(0).astype(int)

    by_comm = df.groupby("teacher_cluster_id")["shard_id"].nunique()
    single_shard = int((by_comm <= 1).sum())
    multi_shard = int((by_comm > 1).sum())
    allow_cross = 0
    for tc, g in df.groupby("teacher_cluster_id"):
        if len(g) <= 1:
            continue
        if g["shard_id"].nunique() > 1:
            allow_cross += 1

    n_with_diff = int((diff_aligned > 0).sum())
    return {
        "n_emails": int(len(df)),
        "n_teacher_communities": int(df["teacher_cluster_id"].nunique()),
        "emails_with_any_diff_shard_positive": n_with_diff,
        "fraction_emails_with_any_diff_shard_positive": float(n_with_diff / len(df))
        if len(df)
        else 0.0,
        "n_communities_single_shard_only": single_shard,
        "n_communities_multi_shard": multi_shard,
        "n_communities_allow_cross_shard_positive": allow_cross,
        "mean_diff_shard_candidates_per_anchor": float(diff_aligned.mean()),
        "mean_same_shard_candidates_per_anchor": float(same_aligned.mean()),
        "median_diff_shard_candidates_per_anchor": float(diff_aligned.median()),
        "median_same_shard_candidates_per_anchor": float(same_aligned.median()),
    }


def compute_easy_negative_diagnostics(df: pd.DataFrame) -> dict[str, Any]:
    """Pool size per anchor: all emails in other teacher communities (within ``df``)."""
    if df.empty:
        return {
            "mean_easy_pool_size": float("nan"),
            "median_easy_pool_size": float("nan"),
            "min_easy_pool_size": float("nan"),
            "max_easy_pool_size": float("nan"),
            "n_anchors_with_zero_easy_pool": 0,
            "fraction_anchors_with_zero_easy_pool": float("nan"),
        }

    sizes = df.groupby("teacher_cluster_id")["external_id"].count().to_dict()
    total = len(df)
    pool = []
    for _, r in df.iterrows():
        tc = str(r["teacher_cluster_id"])
        pool.append(total - sizes[tc])
    arr = np.array(pool, dtype=np.int64)
    zero = int((arr == 0).sum())
    return {
        "mean_easy_pool_size": float(arr.mean()),
        "median_easy_pool_size": float(np.median(arr)),
        "min_easy_pool_size": int(arr.min()),
        "max_easy_pool_size": int(arr.max()),
        "n_anchors_with_zero_easy_pool": zero,
        "fraction_anchors_with_zero_easy_pool": float(zero / len(arr)) if len(arr) else float("nan"),
    }


class EasyNegativeSampler:
    """Random email from a different teacher community."""

    def __init__(
        self,
        df: pd.DataFrame,
        rng: np.random.Generator,
        *,
        id_column: str = "external_id",
        cluster_column: str = "teacher_cluster_id",
    ) -> None:
        self._df = df.reset_index(drop=True)
        self._rng = rng
        self._id_col = id_column
        self._cl_col = cluster_column
        self._by_cluster: dict[str, np.ndarray] = {}
        for tc, g in self._df.groupby(cluster_column):
            self._by_cluster[str(tc)] = g[id_column].astype(str).to_numpy()

        self._all_ids = self._df[id_column].astype(str).to_numpy()
        self._all_clusters = self._df[cluster_column].astype(str).to_numpy()

    def sample(self, anchor_external_id: str, *, exclude_self: bool = True) -> str:
        aid = str(anchor_external_id)
        row = self._df.loc[self._df[self._id_col].astype(str) == aid]
        if row.empty:
            raise KeyError(f"unknown external_id {anchor_external_id!r}")
        tc = str(row[self._cl_col].iloc[0])
        mask = self._all_clusters != tc
        if exclude_self:
            mask &= self._all_ids != aid
        cand = self._all_ids[mask]
        if cand.size == 0:
            raise RuntimeError(f"no easy negatives for anchor {aid!r}")
        i = int(self._rng.integers(0, cand.size))
        return str(cand[i])


class HardNegativeIndexSklearn:
    """
    Nearest-neighbor index over L2-normalized embeddings; returns different-cluster neighbors.

    Built only for emails present in ``external_ids``.
    """

    def __init__(
        self,
        external_ids: Sequence[str],
        embeddings: np.ndarray,
        email_to_cluster: Mapping[str, str],
        *,
        max_neighbors_query: int = 64,
    ) -> None:
        from sklearn.neighbors import NearestNeighbors

        if len(external_ids) != embeddings.shape[0]:
            raise ValueError("external_ids length must match embedding rows")
        self._eids = [str(x) for x in external_ids]
        self._idx_map = {e: i for i, e in enumerate(self._eids)}
        self._email_to_cluster = dict(email_to_cluster)
        self._X = np.asarray(embeddings, dtype=np.float64)
        norms = np.linalg.norm(self._X, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        self._Xn = self._X / norms
        self._nn = NearestNeighbors(n_neighbors=min(max_neighbors_query, len(self._eids)), metric="cosine")
        self._nn.fit(self._Xn)
        self._max_q = max_neighbors_query

    def sample_hard(
        self,
        anchor_external_id: str,
        k: int,
        rng: np.random.Generator,
    ) -> list[str]:
        eid = str(anchor_external_id)
        if eid not in self._idx_map:
            raise KeyError(eid)
        anchor_c = self._email_to_cluster[eid]
        i = self._idx_map[eid]
        dist, ind = self._nn.kneighbors(self._Xn[i : i + 1], n_neighbors=min(self._max_q, len(self._eids)))
        ind = ind[0].tolist()
        out: list[str] = []
        for j in ind:
            if j == i:
                continue
            cand = self._eids[j]
            if self._email_to_cluster.get(cand) == anchor_c:
                continue
            out.append(cand)
            if len(out) >= k:
                break
        rng.shuffle(out)
        return out[:k]


class CommunityAwareBatchIterator:
    """
    Yields batches of external_ids by sampling disjoint teacher communities.

    Each batch contains up to ``n_communities_per_batch`` communities and
    up to ``n_emails_per_community`` emails per community (without replacement
    per community when possible; if a community is smaller, all its emails).

    When ``prefer_cross_shard_positives`` is True, multi-shard communities are
    oversampled and within-community draws bias toward **distinct shards** so
    anchors are more likely to see same-teacher, different-shard positives.
    """

    def __init__(
        self,
        train_df: pd.DataFrame,
        rng: np.random.Generator,
        *,
        n_communities_per_batch: int,
        n_emails_per_community: int,
        prefer_cross_shard_positives: bool = False,
        multi_shard_oversample_factor: float = 1.0,
        min_distinct_shards_per_multi_shard_community_in_batch: int | None = None,
    ) -> None:
        if train_df.empty:
            raise ValueError("train_df is empty")
        self._rng = rng
        self._b_comm = int(n_communities_per_batch)
        self._b_per = int(n_emails_per_community)
        self._prefer_cross = bool(prefer_cross_shard_positives)
        self._multi_shard_oversample_factor = float(multi_shard_oversample_factor)
        self._min_distinct_shards = min_distinct_shards_per_multi_shard_community_in_batch
        self._by_c: dict[str, np.ndarray] = {}
        self._tc_shard_eids: dict[str, dict[str, np.ndarray]] = {}
        self._tc_n_shards: dict[str, int] = {}
        for tc, g in train_df.groupby("teacher_cluster_id"):
            tc = str(tc)
            g = g.reset_index(drop=True)
            self._by_c[tc] = g["external_id"].astype(str).to_numpy()
            by_s: dict[str, list[str]] = defaultdict(list)
            for eid, sid in zip(
                g["external_id"].astype(str),
                g["shard_id"].astype(str),
                strict=False,
            ):
                by_s[sid].append(eid)
            self._tc_shard_eids[tc] = {s: np.array(v, dtype=object) for s, v in by_s.items()}
            self._tc_n_shards[tc] = len(by_s)

        self._communities = list(self._by_c.keys())
        if self._prefer_cross and self._multi_shard_oversample_factor > 1.0:
            extra: list[str] = []
            nrep = int(max(0, round(self._multi_shard_oversample_factor) - 1))
            for tc in self._communities:
                if self._tc_n_shards.get(tc, 0) > 1:
                    extra.extend([tc] * nrep)
            frac = self._multi_shard_oversample_factor - np.floor(self._multi_shard_oversample_factor)
            if frac > 1e-9:
                for tc in self._communities:
                    if self._tc_n_shards.get(tc, 0) > 1 and rng.random() < frac:
                        extra.append(tc)
            self._communities = self._communities + extra
        if self._b_comm < 1 or self._b_per < 1:
            raise ValueError("batch sizes must be >= 1")

    def _sample_community_emails(self, tc: str) -> list[str]:
        rng = self._rng
        members = self._by_c[tc]
        budget = self._b_per
        if members.size <= budget:
            return members.tolist()

        n_shards = self._tc_n_shards.get(tc, 0)
        if not self._prefer_cross or n_shards <= 1:
            idx = rng.choice(members.size, size=budget, replace=False)
            return [str(members[i]) for i in idx]

        shards_map = self._tc_shard_eids[tc]
        shard_ids = list(shards_map.keys())
        min_d = self._min_distinct_shards
        if min_d is None:
            min_d = min(2, n_shards, budget)
        else:
            min_d = int(min_d)
        min_d = max(1, min(min_d, n_shards, budget))

        picked_shard_idx = rng.choice(len(shard_ids), size=min_d, replace=False)
        picked_shards = [shard_ids[j] for j in picked_shard_idx]
        out: list[str] = []
        used: set[str] = set()
        shard_counts: Counter[str] = Counter()
        for sh in picked_shards:
            arr = shards_map[sh]
            choice = str(rng.choice(arr))
            out.append(choice)
            used.add(choice)
            shard_counts[sh] += 1
        remaining = budget - len(out)

        while remaining > 0 and len(used) < members.size:
            weights = np.array([1.0 / (shard_counts[s] + 0.5) for s in shard_ids], dtype=np.float64)
            wsum = float(weights.sum())
            if wsum <= 0:
                break
            weights /= wsum
            sh = str(rng.choice(shard_ids, p=weights))
            arr = shards_map[sh]
            avail = [str(x) for x in arr.tolist() if str(x) not in used]
            if not avail:
                shard_counts[sh] += 1000
                continue
            choice = str(rng.choice(np.array(avail, dtype=object)))
            out.append(choice)
            used.add(choice)
            shard_counts[sh] += 1
            remaining -= 1
        return out[:budget]

    def __iter__(self) -> Iterator[list[str]]:
        return self

    def __next__(self) -> list[str]:
        rng = self._rng
        comms = list(self._communities)
        rng.shuffle(comms)
        picked = comms[: self._b_comm]
        batch: list[str] = []
        for tc in picked:
            batch.extend(self._sample_community_emails(str(tc)))
        return batch

    def iter_epoch(self) -> Iterator[list[str]]:
        """Cover each train community roughly once per epoch (chunked batches)."""
        rng = self._rng
        comms = list(self._communities)
        rng.shuffle(comms)
        for i in range(0, len(comms), self._b_comm):
            chunk = comms[i : i + self._b_comm]
            batch: list[str] = []
            for tc in chunk:
                batch.extend(self._sample_community_emails(str(tc)))
            yield batch


def split_summary_dict(split: TrainValSplit) -> dict[str, Any]:
    def comm_sizes(frame: pd.DataFrame) -> list[int]:
        if frame.empty:
            return []
        return frame.groupby("teacher_cluster_id").size().astype(int).tolist()

    return {
        "n_train_teacher_communities": len(split.train_teacher_clusters),
        "n_val_teacher_communities": len(split.val_teacher_clusters),
        "train_teacher_clusters": list(split.train_teacher_clusters),
        "val_teacher_clusters": list(split.val_teacher_clusters),
        "n_train_emails": int(len(split.train_df)),
        "n_val_emails": int(len(split.val_df)),
        "train_community_sizes": comm_sizes(split.train_df),
        "val_community_sizes": comm_sizes(split.val_df),
    }


def build_hard_negative_index_from_features(
    train_df: pd.DataFrame,
    feature_matrix: np.ndarray,
    *,
    max_neighbors_query: int = 64,
) -> HardNegativeIndexSklearn:
    """
    NN index over **raw** train email features (L2-normalized inside ``HardNegativeIndexSklearn``).

    Rows of ``feature_matrix`` must align with ``train_df`` order.
    """
    if len(train_df) != feature_matrix.shape[0]:
        raise ValueError("train_df rows must match feature_matrix row count")
    eids = train_df["external_id"].astype(str).tolist()
    email_to_cluster = dict(
        zip(train_df["external_id"].astype(str), train_df["teacher_cluster_id"].astype(str), strict=False)
    )
    return HardNegativeIndexSklearn(
        eids,
        np.asarray(feature_matrix, dtype=np.float64),
        email_to_cluster,
        max_neighbors_query=max_neighbors_query,
    )


def try_build_hard_negative_index_from_graph(
    df: pd.DataFrame,
    graph_pt: Path,
    meta_json: Path | None,
    rng: np.random.Generator,
    *,
    embedding_dim: int = 128,
    max_neighbors_query: int = 64,
    sample_k_test: int = 3,
) -> tuple[HardNegativeIndexSklearn | None, dict[str, Any]]:
    """
    Optional: load ``email.x[:, :embedding_dim]`` and fit NN index for df rows.

    Returns (index_or_none, diagnostics_dict).
    """
    diag: dict[str, Any] = {"built": False, "reason": None}
    meta_path = meta_json or Path(graph_pt).with_suffix(".meta.json")
    try:
        from analysis.utils.graph_structure_helpers import external_id_to_row, load_hetero, load_meta
    except ImportError as e:
        diag["reason"] = f"import_error: {e}"
        return None, diag

    if not Path(graph_pt).is_file():
        diag["reason"] = "missing_graph_pt"
        return None, diag
    if not Path(meta_path).is_file():
        diag["reason"] = "missing_meta_json"
        return None, diag

    try:
        meta = load_meta(meta_path)
        row_map = external_id_to_row(meta)
        data = load_hetero(graph_pt, to_undirected=True)
        x = data["email"].x
        if x is None:
            diag["reason"] = "no_email_x"
            return None, diag
        import torch

        xnp = x.detach().cpu().numpy()
        d = min(embedding_dim, xnp.shape[1])
        xnp = xnp[:, :d]
    except Exception as e:  # noqa: BLE001
        diag["reason"] = f"load_failed: {e!r}"
        return None, diag

    want = set(df["external_id"].astype(str))
    present = [e for e in df["external_id"].astype(str).tolist() if e in row_map]
    missing = sorted(want - set(present))
    if len(present) < 2:
        diag["reason"] = "too_few_emails_in_graph"
        diag["n_missing_ids"] = len(missing)
        return None, diag

    rows_idx = np.array([row_map[e] for e in present], dtype=np.int64)
    emb = xnp[rows_idx]
    email_to_cluster = dict(zip(df["external_id"].astype(str), df["teacher_cluster_id"].astype(str), strict=False))
    # restrict map to present
    email_to_cluster = {e: email_to_cluster[e] for e in present}

    try:
        idx_obj = HardNegativeIndexSklearn(
            present,
            emb,
            email_to_cluster,
            max_neighbors_query=max_neighbors_query,
        )
    except ImportError as e:
        diag["reason"] = f"sklearn_missing: {e!r}"
        return None, diag
    diag["built"] = True
    diag["embedding_dim_used"] = int(d)
    diag["n_indexed_emails"] = len(present)
    diag["n_missing_ids"] = len(missing)
    diag["missing_sample"] = missing[:8]

    # smoke test
    test_e = present[0]
    try:
        got = idx_obj.sample_hard(test_e, sample_k_test, rng)
        diag["smoke_test"] = {"status": "ok", "k_returned": len(got)}
    except Exception as e:  # noqa: BLE001
        diag["smoke_test"] = {"status": "error", "error": repr(e)}

    return idx_obj, diag
