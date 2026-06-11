"""
Deduplicate (or thin) ground-truth JSON by similarity on selected email fields.

Uses only: sender, receiver, date, subject, body, urls, attachments.
Does not read project data paths by default; pass --input / --output explicitly.

Near-duplicate detection (default):
  64-bit SimHash over whitespace tokens; merge when Hamming distance <= --max-hamming.
  Optional: --min-fuzzy-ratio requires difflib/rapidfuzz ratio >= threshold for merges
  (reduces SimHash false positives).

Exact duplicates (after normalization) always merge (Hamming 0).

Examples:
  python scripts/deduplicate_ground_truth.py \\
    --input data/groundtruth/ground_truth.json \\
    --output data/groundtruth/ground_truth_dedup.json \\
    --max-hamming 13

  # MISP lake + GT id remap for dedup_task_identity (current experiment track):
  python scripts/misp_lake_dedup/collapse_misp_lake_strict_duplicates.py \\
    --collapse-signature-type strict_task_message_identity

  # Stricter text agreement (~80% token overlap sense via ratio):
  python scripts/deduplicate_ground_truth.py -i in.json -o out.json \\
    --max-hamming 20 --min-fuzzy-ratio 0.8

  # Dedupe only within each campaign cluster:
  python scripts/deduplicate_ground_truth.py -i in.json -o out.json --scope per-cluster
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
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


def _normalize_ws(s: str) -> str:
    return " ".join(s.split())


def _normalize_attachments(att: Any) -> str:
    if att is None:
        return ""
    if isinstance(att, list):
        parts = [str(x).strip() for x in att if str(x).strip()]
        return "|".join(sorted(parts))
    return _normalize_ws(str(att))


def canonical_similarity_string(record: dict[str, Any], *, body_max_chars: int | None) -> str:
    """Single string used for SimHash / fuzzy ratio (order is fixed)."""
    sender = _normalize_ws(str(record.get("sender") or ""))
    receiver = _normalize_ws(str(record.get("receiver") or ""))
    date = _normalize_ws(str(record.get("date") or ""))
    subject = _normalize_ws(str(record.get("subject") or ""))
    body = _normalize_ws(str(record.get("body") or ""))
    if body_max_chars is not None and len(body) > body_max_chars:
        body = body[:body_max_chars]
    urls = _normalize_ws(str(record.get("urls") or ""))
    attachments = _normalize_attachments(record.get("attachments"))
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


def _iter_flat_items(
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


def _merge_duplicates(
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
    Returns (indices_to_keep), duplicate_edges_merged).
    Keeps the smallest index in each union-find component.
    """
    uf = _merge_duplicates(items, max_hamming=max_hamming, min_fuzzy_ratio=min_fuzzy_ratio)
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


def run(
    *,
    input_path: Path,
    output_path: Path,
    scope: str,
    max_hamming: int,
    min_fuzzy_ratio: float | None,
    body_max_chars: int | None,
) -> dict[str, Any]:
    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    clusters = data.get("clusters")
    if not isinstance(clusters, dict):
        raise SystemExit("Input JSON must contain an object 'clusters'.")

    total_before = 0
    total_removed = 0
    new_clusters: dict[str, list[dict[str, Any]]] = {}

    if scope == "global":
        flat = list(_iter_flat_items(clusters, body_max_chars=body_max_chars))
        total_before = len(flat)
        keep, removed = dedup_flat(
            flat,
            max_hamming=max_hamming,
            min_fuzzy_ratio=min_fuzzy_ratio,
        )
        total_removed = removed
        new_clusters = rebuild_clusters(clusters, flat, keep)
    else:
        for ck in sorted(clusters.keys()):
            emails = clusters[ck]
            if not isinstance(emails, list):
                continue
            flat: list[FlatItem] = []
            for idx, email in enumerate(emails):
                if not isinstance(email, dict):
                    continue
                norm = canonical_similarity_string(email, body_max_chars=body_max_chars)
                flat.append(
                    FlatItem(
                        cluster_key=ck,
                        index_in_cluster=idx,
                        email=email,
                        norm=norm,
                        simh=simhash_64(norm),
                    )
                )
            total_before += len(flat)
            if len(flat) < 2:
                if flat:
                    new_clusters[ck] = [fi.email for fi in flat]
                continue
            keep, removed = dedup_flat(
                flat,
                max_hamming=max_hamming,
                min_fuzzy_ratio=min_fuzzy_ratio,
            )
            total_removed += removed
            kept_emails = [flat[i].email for i in range(len(flat)) if i in keep]
            if kept_emails:
                new_clusters[ck] = kept_emails

    out_obj = {"clusters": new_clusters}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(out_obj, f, indent=2, ensure_ascii=False)
        f.write("\n")

    return {
        "emails_before": total_before,
        "emails_after": total_before - total_removed,
        "removed": total_removed,
        "removal_fraction": (total_removed / total_before) if total_before else 0.0,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", "-i", type=Path, required=True, help="Source ground_truth.json")
    p.add_argument("--output", "-o", type=Path, required=True, help="Destination JSON path")
    p.add_argument(
        "--scope",
        choices=("global", "per-cluster"),
        default="global",
        help="Merge duplicates across all campaigns (global) or only within each cluster key.",
    )
    p.add_argument(
        "--max-hamming",
        type=int,
        default=13,
        help="SimHash Hamming threshold (0=exact normalized match only). Default 13 is a moderate merge rate; "
        "raise (e.g. 18–22) to remove more near-duplicates, lower to be stricter.",
    )
    p.add_argument(
        "--min-fuzzy-ratio",
        type=float,
        default=None,
        metavar="R",
        help="If set (e.g. 0.8), only merge when rapidfuzz/difflib ratio on the canonical string is >= R "
        "(in addition to SimHash Hamming <= max-hamming). Slower but closer to '80%% similar text'.",
    )
    p.add_argument(
        "--body-max-chars",
        type=int,
        default=None,
        metavar="N",
        help="Truncate body contribution to N characters before similarity (speed; optional).",
    )
    args = p.parse_args()

    if args.max_hamming < 0 or args.max_hamming > 64:
        print("--max-hamming must be between 0 and 64", file=sys.stderr)
        sys.exit(2)
    if args.min_fuzzy_ratio is not None and not (0.0 < args.min_fuzzy_ratio <= 1.0):
        print("--min-fuzzy-ratio must be in (0, 1]", file=sys.stderr)
        sys.exit(2)

    stats = run(
        input_path=args.input,
        output_path=args.output,
        scope=args.scope,
        max_hamming=args.max_hamming,
        min_fuzzy_ratio=args.min_fuzzy_ratio,
        body_max_chars=args.body_max_chars,
    )
    print(
        f"Wrote {args.output}\n"
        f"  emails: {stats['emails_before']} -> {stats['emails_after']} "
        f"(removed {stats['removed']}, {stats['removal_fraction']:.1%})\n"
        f"  scope={args.scope}, max_hamming={args.max_hamming}, "
        f"min_fuzzy_ratio={args.min_fuzzy_ratio}, fuzzy_backend={_FUZZY_BACKEND}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
