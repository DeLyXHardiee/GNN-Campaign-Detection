"""
Ranking pair sampling for Method 1 V2 (agreement-stratified, some endpoint-shared hard pairs).
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd


def build_adjacency_row_indices(edges_df: pd.DataFrame) -> dict[str, list[int]]:
    """Adjacency: shard -> list of incident edge row indices (unweighted)."""
    edges_df = edges_df.copy()
    edges_df["shard_a"] = edges_df["shard_a"].astype(str)
    edges_df["shard_b"] = edges_df["shard_b"].astype(str)
    edges_df = edges_df.reset_index(drop=True)
    adj_idx: dict[str, list[int]] = defaultdict(list)
    for i, r in edges_df.iterrows():
        a, b = str(r["shard_a"]), str(r["shard_b"])
        adj_idx[a].append(i)
        adj_idx[b].append(i)
    return dict(adj_idx)


def sample_ranking_pairs(
    edges_df: pd.DataFrame,
    agreement: np.ndarray,
    rng: np.random.Generator,
    *,
    n_pairs: int,
    n_quantile_bins: int = 5,
    fraction_endpoint_hard: float = 0.35,
    min_agreement_gap: float = 1e-4,
    index_pool: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample pairs (idx_hi, idx_lo) such that agreement[idx_hi] > agreement[idx_lo].

    If ``index_pool`` is set, both indices are restricted to that subset (e.g. train-only
    or val-only edges). Uses full ``edges_df`` for adjacency on endpoint-hard pairs.

    Returns:
        idx_hi, idx_lo: int arrays of shape (n_pairs,)
    """
    edges_df = edges_df.reset_index(drop=True)
    n = len(edges_df)
    if n < 2:
        raise ValueError("need at least 2 edges")

    pool_set: set[int] | None = None
    pool_list: list[int] | None = None
    if index_pool is not None:
        pool_list = sorted({int(x) for x in np.asarray(index_pool).ravel() if 0 <= int(x) < n})
        if len(pool_list) < 2:
            raise ValueError("index_pool must contain at least two valid edge row indices")
        pool_set = set(pool_list)

    def _filt(xs: list[int]) -> list[int]:
        if pool_set is None:
            return xs
        out = [x for x in xs if x in pool_set]
        return out if out else pool_list  # type: ignore[return-value]

    agree = np.asarray(agreement, dtype=np.float64)
    n_bins = int(min(n_quantile_bins, max(2, n // 10)))
    try:
        qs = np.quantile(agree, np.linspace(0, 1, n_bins + 1))
        qs[0] -= 1e-9
        qs[-1] += 1e-9
        bin_id = np.digitize(agree, qs[1:-1], right=False)
    except Exception:
        bin_id = np.zeros(n, dtype=np.int64)

    by_bin: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        if pool_set is None or i in pool_set:
            by_bin[int(bin_id[i])].append(i)

    adj_idx = build_adjacency_row_indices(edges_df)

    idx_hi = np.empty(n_pairs, dtype=np.int64)
    idx_lo = np.empty(n_pairs, dtype=np.int64)

    n_hard = int(round(n_pairs * fraction_endpoint_hard))
    n_strat = n_pairs - n_hard

    # Stratified: pick hi from upper bins, lo from lower bins
    for t in range(n_strat):
        hi_bin = int(rng.integers(max(1, n_bins // 2), max(2, n_bins)))
        lo_bin = int(rng.integers(0, max(1, n_bins // 2)))
        pool_hi = _filt(by_bin.get(hi_bin, pool_list if pool_list is not None else list(range(n))))
        pool_lo = _filt(by_bin.get(lo_bin, pool_list if pool_list is not None else list(range(n))))
        if not pool_hi:
            pool_hi = _filt(list(range(n)))
        if not pool_lo:
            pool_lo = _filt(list(range(n)))
        ih = int(rng.choice(pool_hi))
        il = int(rng.choice(pool_lo))
        if agree[ih] <= agree[il] + min_agreement_gap:
            for _ in range(20):
                ih = int(rng.choice(pool_hi))
                il = int(rng.choice(pool_lo))
                if agree[ih] > agree[il] + min_agreement_gap:
                    break
        if agree[ih] <= agree[il]:
            ih, il = il, ih
        if agree[ih] <= agree[il]:
            if pool_list is not None:
                ih, il = int(pool_list[0]), int(pool_list[1])
            else:
                ih, il = 0, min(1, n - 1)
        idx_hi[t], idx_lo[t] = ih, il

    # Endpoint-hard: same shard u, neighbors v1 v2
    shard_list = np.unique(
        np.concatenate([edges_df["shard_a"].astype(str).values, edges_df["shard_b"].astype(str).values])
    )
    for t in range(n_strat, n_pairs):
        ok = False
        for _ in range(50):
            u = str(rng.choice(shard_list))
            nbrs = adj_idx.get(u, [])
            if len(nbrs) < 2:
                continue
            i1, i2 = rng.choice(nbrs, size=2, replace=False)
            if pool_set is not None and (i1 not in pool_set or i2 not in pool_set):
                continue
            a1, a2 = agree[i1], agree[i2]
            if abs(a1 - a2) < min_agreement_gap:
                continue
            if a1 > a2:
                idx_hi[t], idx_lo[t] = i1, i2
            else:
                idx_hi[t], idx_lo[t] = i2, i1
            ok = True
            break
        if not ok:
            pl = pool_list if pool_list is not None else list(range(n))
            ih, il = rng.choice(pl, size=2, replace=False)
            ih, il = int(ih), int(il)
            if agree[ih] <= agree[il]:
                ih, il = il, ih
            idx_hi[t], idx_lo[t] = ih, il

    return idx_hi, idx_lo
