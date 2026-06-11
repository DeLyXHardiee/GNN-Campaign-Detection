"""
Near-duplicate email detection (SimHash + optional fuzzy ratio).

Shared by ``scripts/deduplicate_ground_truth.py`` and GNN pair training.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any

try:
    from rapidfuzz import fuzz as _rfuzz

    _FUZZY_BACKEND = "rapidfuzz"

    def _fuzzy_ratio(a: str, b: str) -> float:
        return _rfuzz.ratio(a, b) / 100.0

except ImportError:
    _FUZZY_BACKEND = "difflib.SequenceMatcher"

    def _fuzzy_ratio(a: str, b: str) -> float:
        return SequenceMatcher(None, a, b).ratio()


def normalize_ws(s: str) -> str:
    return " ".join(s.split())


def normalize_attachments(att: Any) -> str:
    if att is None:
        return ""
    if isinstance(att, list):
        parts = [str(x).strip() for x in att if str(x).strip()]
        return "|".join(sorted(parts))
    return normalize_ws(str(att))


def canonical_similarity_string(record: dict[str, Any], *, body_max_chars: int | None) -> str:
    """Single string used for SimHash / fuzzy ratio (order is fixed)."""
    sender = normalize_ws(str(record.get("sender") or ""))
    receiver = normalize_ws(str(record.get("receiver") or ""))
    date = normalize_ws(str(record.get("date") or ""))
    subject = normalize_ws(str(record.get("subject") or ""))
    body = normalize_ws(str(record.get("body") or ""))
    if body_max_chars is not None and len(body) > body_max_chars:
        body = body[:body_max_chars]
    urls = normalize_ws(str(record.get("urls") or ""))
    attachments = normalize_attachments(record.get("attachments"))
    return "\x1f".join((sender, receiver, date, subject, body, urls, attachments))


def simhash_64(text: str) -> int:
    """64-bit SimHash (Charikar), token = whitespace-delimited word."""
    if not text.strip():
        return 0
    vec = [0] * 64
    for tok in text.split():
        h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
        for bit in range(64):
            vec[bit] += 1 if (h >> bit) & 1 else -1
    out = 0
    for bit in range(64):
        if vec[bit] > 0:
            out |= 1 << bit
    return out


def hamming_u64(a: int, b: int) -> int:
    return (a ^ b).bit_count()


class UnionFind:
    def __init__(self, n: int) -> None:
        self.p = list(range(n))
        self.r = [0] * n

    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.r[ra] < self.r[rb]:
            ra, rb = rb, ra
        self.p[rb] = ra
        if self.r[ra] == self.r[rb]:
            self.r[ra] += 1


@dataclass(frozen=True)
class FlatItem:
    cluster_key: str
    index_in_cluster: int
    email: dict[str, Any]
    norm: str
    simh: int


def iter_flat_items(
    clusters: dict[str, Any],
    *,
    body_max_chars: int | None,
) -> Iterator[FlatItem]:
    for ck in sorted(clusters.keys()):
        emails = clusters[ck]
        if not isinstance(emails, list):
            continue
        for idx, email in enumerate(emails):
            if not isinstance(email, dict):
                continue
            norm = canonical_similarity_string(email, body_max_chars=body_max_chars)
            yield FlatItem(
                cluster_key=ck,
                index_in_cluster=idx,
                email=email,
                norm=norm,
                simh=simhash_64(norm),
            )


def merge_duplicates(
    items: list[FlatItem],
    *,
    max_hamming: int,
    min_fuzzy_ratio: float | None,
) -> UnionFind:
    n = len(items)
    uf = UnionFind(n)
    if n < 2:
        return uf

    for i in range(n):
        for j in range(i + 1, n):
            h = hamming_u64(items[i].simh, items[j].simh)
            if h > max_hamming:
                continue
            if min_fuzzy_ratio is not None:
                r = _fuzzy_ratio(items[i].norm, items[j].norm)
                if r < min_fuzzy_ratio:
                    continue
            uf.union(i, j)
    return uf


def dedup_flat(
    items: list[FlatItem],
    *,
    max_hamming: int,
    min_fuzzy_ratio: float | None,
) -> tuple[set[int], int]:
    """
    Returns (indices_to_keep, duplicate_rows_merged).
    Keeps the smallest index in each union-find component.
    """
    uf = merge_duplicates(items, max_hamming=max_hamming, min_fuzzy_ratio=min_fuzzy_ratio)
    n = len(items)
    root_to_min: dict[int, int] = {}
    for i in range(n):
        r = uf.find(i)
        if r not in root_to_min or i < root_to_min[r]:
            root_to_min[r] = i
    keep = set(root_to_min.values())
    removed = n - len(keep)
    return keep, removed


def rebuild_clusters(
    clusters: dict[str, Any],
    flat: list[FlatItem],
    keep_indices: set[int],
) -> dict[str, list[dict[str, Any]]]:
    """Rebuild cluster map preserving relative order within each cluster for kept rows."""
    out: dict[str, list[dict[str, Any]]] = {k: [] for k in clusters}
    for i, fi in enumerate(flat):
        if i not in keep_indices:
            continue
        out[fi.cluster_key].append(fi.email)
    return {k: v for k, v in out.items() if v}


FUZZY_BACKEND = _FUZZY_BACKEND
