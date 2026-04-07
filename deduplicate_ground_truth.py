#!/usr/bin/env python3
"""
Standalone utility: deduplicate ground-truth campaign clusters using
normalized text (exact) + cached SBERT vectors from embeddings.json (semantic).

Does not modify the input ground-truth file. Not wired into the ML pipeline.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import string
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    import numpy as np

    _HAS_NP = True
except ImportError:
    _HAS_NP = False


# ---------------------------------------------------------------------------
# Text normalization
# ---------------------------------------------------------------------------

_WS_RE = re.compile(r"\s+")
_PUNCT_TABLE = str.maketrans({c: " " for c in string.punctuation})


def normalize_text(s: str, *, strip_punctuation: bool) -> str:
    t = (s or "").lower().strip()
    if strip_punctuation:
        t = t.translate(_PUNCT_TABLE)
    t = _WS_RE.sub(" ", t).strip()
    return t


def choose_subject_body(
    rec: Dict[str, Any], *, prefer_translated: bool
) -> Tuple[str, str]:
    if prefer_translated:
        subj = (rec.get("subject_translated") or "").strip() or (
            rec.get("subject") or ""
        )
        body = (rec.get("body_translated") or "").strip() or (rec.get("body") or "")
    else:
        subj = rec.get("subject") or ""
        body = rec.get("body") or ""
    return str(subj), str(body)


def combined_subject_body(norm_subj: str, norm_body: str) -> str:
    return f"{norm_subj}\n{norm_body}"


# ---------------------------------------------------------------------------
# Embeddings index (embeddings.json)
# ---------------------------------------------------------------------------

VectorSource = str  # "body" | "subject_only" | "concat_subj_body" | "none"


def _l2_norm(vec: Sequence[float]) -> float:
    if _HAS_NP:
        a = np.asarray(vec, dtype=np.float64)
        return float(np.linalg.norm(a))
    return math.sqrt(sum(float(x) * float(x) for x in vec))


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b:
        return 0.0
    if _HAS_NP:
        va = np.asarray(a, dtype=np.float64)
        vb = np.asarray(b, dtype=np.float64)
        na, nb = float(np.linalg.norm(va)), float(np.linalg.norm(vb))
        if na <= 0.0 or nb <= 0.0:
            return 0.0
        return float(np.dot(va, vb) / (na * nb))
    da, db = 0.0, 0.0
    s = 0.0
    for i in range(min(len(a), len(b))):
        x, y = float(a[i]), float(b[i])
        s += x * y
        da += x * x
        db += y * y
    if da <= 0.0 or db <= 0.0:
        return 0.0
    return s / (math.sqrt(da) * math.sqrt(db))


def choose_embedding_vector(
    entry: Dict[str, Any],
) -> Tuple[Optional[List[float]], VectorSource]:
    """
    Pick the semantic vector for deduplication.

    Prefer `body` when present and non-empty — it is the most
    content-representative signal for near-duplicate email bodies.

    If `body` is missing or empty but `subj` exists, use subject-only.

    If both are non-empty but `body` were empty (handled above), we already
    returned body branch when len(body)>0.

    If both vectors are missing, return (None, "none").

    When `body` is missing or an empty list but `subj` is non-empty, we use the
    subject embedding only (still from the same cache entry).
    """
    subj = entry.get("subj") or []
    body = entry.get("body") or []
    if not isinstance(subj, list):
        subj = []
    if not isinstance(body, list):
        body = []

    if len(body) > 0:
        return [float(x) for x in body], "body"
    if len(subj) > 0:
        return [float(x) for x in subj], "subject_only"

    return None, "none"


def load_embeddings_index(embeddings_path: Path) -> Tuple[Dict[str, Any], str]:
    """
    Load embeddings.json. Returns metadata+index mapping external_id -> (vec, source).

    Keys: entry['external_id'] if set, else the by_key key (same convention
    as core.utils.embeddings.embedder).
    """
    with embeddings_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    meta = {
        "model": data.get("model"),
        "subj_dim": data.get("subj_dim"),
        "body_dim": data.get("body_dim"),
    }
    by_key = data.get("by_key") or {}
    index: Dict[str, Tuple[Optional[List[float]], VectorSource]] = {}
    if not isinstance(by_key, dict):
        return meta, index
    # Deterministic iteration order from JSON dict (Py3.7+ insertion order);
    # sort keys for reproducibility across runs/loaders.
    for k in sorted(by_key.keys(), key=str):
        entry = by_key[k]
        if not isinstance(entry, dict):
            continue
        eid = str((entry.get("external_id") or k or "")).strip()
        if not eid:
            continue
        vec, src = choose_embedding_vector(entry)
        index[eid] = (vec, src)
    return meta, index


def get_vec_for_email(
    external_id: str,
    emb_index: Dict[str, Tuple[Optional[List[float]], VectorSource]],
) -> Tuple[Optional[List[float]], VectorSource]:
    tup = emb_index.get(str(external_id).strip())
    if not tup:
        return None, "none"
    return tup


# ---------------------------------------------------------------------------
# Union-Find
# ---------------------------------------------------------------------------


class UnionFind:
    def __init__(self, n: int) -> None:
        self._p = list(range(n))
        self._r = [0] * n

    def find(self, x: int) -> int:
        while self._p[x] != x:
            self._p[x] = self._p[self._p[x]]
            x = self._p[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self._r[ra] < self._r[rb]:
            self._p[ra] = rb
        elif self._r[ra] > self._r[rb]:
            self._p[rb] = ra
        else:
            self._p[rb] = ra
            self._r[ra] += 1


def _rep_index(
    group: List[int],
    records: List[Dict[str, Any]],
    norm_bodies: List[str],
) -> int:
    """Smallest distance_to_centroid; tie-break lexicographic external_id."""

    def sort_key(i: int) -> Tuple[float, str]:
        rec = records[i]
        d = rec.get("distance_to_centroid")
        eid = str(rec.get("external_id") or "")
        if d is None:
            return (float("inf"), eid)
        try:
            return (float(d), eid)
        except (TypeError, ValueError):
            return (float("inf"), eid)

    # Deterministic: sort by key then by index for full stability
    return min(group, key=lambda i: (sort_key(i), i))


def _exact_removal_reason(i: int, r: int, norm_bodies: List[str]) -> str:
    if norm_bodies[i] and norm_bodies[i] == norm_bodies[r]:
        return "exact_body"
    return "exact_subject_body"


# ---------------------------------------------------------------------------
# Per-cluster deduplication
# ---------------------------------------------------------------------------


@dataclass
class ClusterDedupResult:
    kept_records: List[Dict[str, Any]]
    removed: List[Dict[str, Any]]
    dropped_below_min: bool
    exact_removed_count: int = 0
    semantic_removed_count: int = 0


def dedupe_one_cluster(
    records: List[Dict[str, Any]],
    emb_index: Dict[str, Tuple[Optional[List[float]], VectorSource]],
    *,
    semantic_threshold: float,
    prefer_translated: bool,
    strip_punctuation: bool,
    match_nonempty_body_only: bool,
) -> ClusterDedupResult:
    """Deduplicate records within a single cluster (independent of others)."""
    n = len(records)
    removed: List[Dict[str, Any]] = []
    if n == 0:
        return ClusterDedupResult([], [], True, 0, 0)

    norm_bodies: List[str] = []
    norm_combined: List[str] = []
    for rec in records:
        sj, bd = choose_subject_body(rec, prefer_translated=prefer_translated)
        nb = normalize_text(bd, strip_punctuation=strip_punctuation)
        ns = normalize_text(sj, strip_punctuation=strip_punctuation)
        norm_bodies.append(nb)
        norm_combined.append(combined_subject_body(ns, nb))

    # --- Phase 1: exact ---
    uf1 = UnionFind(n)
    # Bucket by norm_body
    body_buckets: Dict[str, List[int]] = {}
    for i, nb in enumerate(norm_bodies):
        if match_nonempty_body_only and not nb:
            continue
        body_buckets.setdefault(nb, []).append(i)
    for indices in body_buckets.values():
        idxs = sorted(indices)
        for a, b in zip(idxs, idxs[1:]):
            uf1.union(a, b)

    # Bucket by norm_combined
    combo_buckets: Dict[str, List[int]] = {}
    for i, nc in enumerate(norm_combined):
        combo_buckets.setdefault(nc, []).append(i)
    for indices in combo_buckets.values():
        idxs = sorted(indices)
        for a, b in zip(idxs, idxs[1:]):
            uf1.union(a, b)

    groups1: Dict[int, List[int]] = {}
    for i in range(n):
        root = uf1.find(i)
        groups1.setdefault(root, []).append(i)
    survivors1: List[int] = []
    for _root, grp in sorted(groups1.items(), key=lambda kv: min(kv[1])):
        grp_sorted = sorted(grp)
        r = _rep_index(grp_sorted, records, norm_bodies)
        survivors1.append(r)
        for i in grp_sorted:
            if i != r:
                rid = str(records[i].get("external_id") or "")
                krid = str(records[r].get("external_id") or "")
                removed.append(
                    {
                        "external_id": rid,
                        "representative_kept_external_id": krid,
                        "reason": _exact_removal_reason(i, r, norm_bodies),
                        "similarity": None,
                    }
                )

    exact_removed = len(removed)

    # --- Phase 2: semantic on survivors1 (indices into `records`) ---
    sem_removed_count = 0
    if len(survivors1) <= 1:
        kept_idx = list(survivors1)
    else:
        m = len(survivors1)
        vecs: List[Optional[List[float]]] = []
        for j in survivors1:
            eid = str(records[j].get("external_id") or "")
            v, _src = get_vec_for_email(eid, emb_index)
            vecs.append(v if v and _l2_norm(v) > 0 else None)

        uf2 = UnionFind(m)
        for a in range(m):
            if vecs[a] is None:
                continue
            for b in range(a + 1, m):
                if vecs[b] is None:
                    continue
                cs = _cosine(vecs[a], vecs[b])  # type: ignore[arg-type]
                if cs >= semantic_threshold:
                    uf2.union(a, b)

        groups2: Dict[int, List[int]] = {}
        for i in range(m):
            groups2.setdefault(uf2.find(i), []).append(i)

        kept_survivor_positions: List[int] = []
        for _root, grp in sorted(groups2.items(), key=lambda kv: min(kv[1])):
            grp_sorted = sorted(grp)
            orig_indices = [survivors1[i] for i in grp_sorted]
            r_orig = _rep_index(sorted(orig_indices), records, norm_bodies)
            r_local_candidates = [li for li in grp_sorted if survivors1[li] == r_orig]
            r_local = min(r_local_candidates)
            kept_survivor_positions.append(r_orig)
            for li in grp_sorted:
                if li == r_local:
                    continue
                j_orig = survivors1[li]
                rid = str(records[j_orig].get("external_id") or "")
                krid = str(records[r_orig].get("external_id") or "")
                sim = _cosine(vecs[li], vecs[r_local])  # type: ignore[arg-type]
                removed.append(
                    {
                        "external_id": rid,
                        "representative_kept_external_id": krid,
                        "reason": "semantic_near_duplicate",
                        "similarity": round(sim, 6),
                    }
                )
                sem_removed_count += 1

        kept_idx = sorted(set(kept_survivor_positions))

    kept_records = [records[i] for i in sorted(kept_idx)]
    dropped = len(kept_records) < 2
    return ClusterDedupResult(
        kept_records=kept_records,
        removed=removed,
        dropped_below_min=dropped,
        exact_removed_count=exact_removed,
        semantic_removed_count=sem_removed_count,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run(
    input_path: Path,
    embeddings_path: Path,
    output_path: Path,
    report_path: Path,
    *,
    semantic_threshold: float,
    prefer_translated: bool,
    strip_punctuation: bool,
    match_nonempty_body_only: bool,
) -> None:
    with input_path.open("r", encoding="utf-8") as f:
        gt = json.load(f)
    clusters_raw = gt.get("clusters")
    if not isinstance(clusters_raw, dict):
        raise ValueError("Ground truth must contain a top-level 'clusters' object.")

    emb_meta, emb_index = load_embeddings_index(embeddings_path)

    out_clusters: Dict[str, List[Dict[str, Any]]] = {}
    per_campaign: Dict[str, Any] = {}

    total_in = 0
    total_out = 0
    campaigns_before = len(clusters_raw)
    campaigns_after = 0
    removed_exact = 0
    removed_sem = 0
    dropped_campaigns = 0

    for ckey in sorted(clusters_raw.keys(), key=str):
        emails = clusters_raw[ckey]
        if not isinstance(emails, list):
            continue
        records = [e for e in emails if isinstance(e, dict)]
        total_in += len(records)
        res = dedupe_one_cluster(
            records,
            emb_index,
            semantic_threshold=semantic_threshold,
            prefer_translated=prefer_translated,
            strip_punctuation=strip_punctuation,
            match_nonempty_body_only=match_nonempty_body_only,
        )
        removed_exact += res.exact_removed_count
        removed_sem += res.semantic_removed_count

        kept_ids = [str(r.get("external_id") or "") for r in res.kept_records]
        per_campaign[ckey] = {
            "original_size": len(records),
            "deduplicated_size": len(res.kept_records),
            "kept_external_ids": kept_ids,
            "removed": res.removed,
            "dropped_below_min_size_2": res.dropped_below_min,
        }

        if res.dropped_below_min:
            dropped_campaigns += 1
        else:
            out_clusters[ckey] = res.kept_records
            total_out += len(res.kept_records)
            campaigns_after += 1

    report = {
        "input_ground_truth_path": str(input_path.resolve()),
        "embeddings_path": str(embeddings_path.resolve()),
        "embedding_file_model": emb_meta.get("model"),
        "embedding_file_subj_dim": emb_meta.get("subj_dim"),
        "embedding_file_body_dim": emb_meta.get("body_dim"),
        "semantic_threshold_cosine": semantic_threshold,
        "prefer_translated": prefer_translated,
        "strip_punctuation": strip_punctuation,
        "match_nonempty_body_only_for_exact_body_bucket": match_nonempty_body_only,
        "numpy_used_for_similarity": _HAS_NP,
        "embedding_vector_policy": (
            "Prefer non-empty `body` vector from embeddings.json; if body vector "
            "is empty, use `subj` only (subject_only). Entries missing from the "
            "embedding index skip semantic pairing but remain for exact dedup."
        ),
        "totals": {
            "campaigns_before": campaigns_before,
            "campaigns_after": campaigns_after,
            "campaigns_removed_below_2_emails": dropped_campaigns,
            "emails_before": total_in,
            "emails_after": total_out,
            "removed_exact_duplicate": removed_exact,
            "removed_semantic_near_duplicate": removed_sem,
        },
        "per_campaign": per_campaign,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump({"clusters": out_clusters}, f, indent=2, ensure_ascii=False)

    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Deduplicate ground-truth clusters (exact text + semantic near-dup) "
            "using cached embeddings.json. Does not alter the input file."
        )
    )
    p.add_argument("--input", required=True, type=Path, help="Input ground-truth JSON")
    p.add_argument(
        "--embeddings",
        required=True,
        type=Path,
        help="Path to embeddings.json (e.g. core/utils/embeddings/output/embeddings.json)",
    )
    p.add_argument(
        "--output", required=True, type=Path, help="Output deduplicated ground-truth JSON"
    )
    p.add_argument(
        "--report", required=True, type=Path, help="Output deduplication report JSON"
    )
    p.add_argument(
        "--semantic-threshold",
        type=float,
        default=0.98,
        help="Cosine similarity threshold for semantic near-duplicates (default: 0.985)",
    )
    p.add_argument(
        "--prefer-translated",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Prefer subject_translated/body_translated when non-empty (default: true)",
    )
    p.add_argument(
        "--strip-punctuation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Conservatively map punctuation to spaces before exact-match (default: true)",
    )
    p.add_argument(
        "--match-nonempty-body-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "For exact body matching, only merge when normalized body is non-empty "
            "(avoids merging unrelated empty-body rows). Default: true"
        ),
    )
    return p


def main() -> None:
    args = build_argparser().parse_args()
    if not args.input.exists():
        raise SystemExit(f"Input not found: {args.input}")
    if not args.embeddings.exists():
        raise SystemExit(f"Embeddings not found: {args.embeddings}")
    inp_resolved = args.input.resolve()
    out_resolved = args.output.resolve()
    rep_resolved = args.report.resolve()
    if inp_resolved == out_resolved:
        raise SystemExit("Refusing to overwrite: --output must differ from --input")
    if inp_resolved == rep_resolved:
        raise SystemExit("Refusing to overwrite: --report must differ from --input")
    run(
        args.input,
        args.embeddings,
        args.output,
        args.report,
        semantic_threshold=float(args.semantic_threshold),
        prefer_translated=bool(args.prefer_translated),
        strip_punctuation=bool(args.strip_punctuation),
        match_nonempty_body_only=bool(args.match_nonempty_body_only),
    )


if __name__ == "__main__":
    main()
