"""
Apply merge decisions exported from manual_review_browser.html to a ground-truth JSON.

The browser stores a document like:
  {"version": 1, "merge_edges": [{"left_campaign": ..., "right_campaign": ..., ...}, ...]}

Each edge means: unify the two GT campaigns (transitive closure across all edges).
Output is a new JSON with the same top-level shape as the input (``clusters`` map).

Usage:
  python analysis/scripts/apply_manual_campaign_merges.py \\
    --gt data/groundtruth/ground_truth.json \\
    --merges analysis/output/semantic_shard_oracle_headroom/manual_campaign_merges.json \\
    --out data/groundtruth/ground_truth_merged.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from analysis.utils.raw_gnn_notebook import parse_campaign_key


def canonical_str(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, bool):
        return "true" if x else "false"
    if isinstance(x, int):
        return str(x)
    if isinstance(x, float):
        if x.is_integer():
            return str(int(x))
        return str(x)
    return str(x).strip()


class UnionFind:
    def __init__(self) -> None:
        self._p: dict[str, str] = {}

    def find(self, x: str) -> str:
        self._p.setdefault(x, x)
        if self._p[x] != x:
            self._p[x] = self.find(self._p[x])
        return self._p[x]

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if ra < rb:
            self._p[rb] = ra
        else:
            self._p[ra] = rb


def _load_merge_edges(merges_path: Path) -> list[dict[str, Any]]:
    data = json.loads(merges_path.read_text(encoding="utf-8"))
    edges = data.get("merge_edges")
    if not isinstance(edges, list):
        raise ValueError("merges file must contain a list 'merge_edges'")
    out: list[dict[str, Any]] = []
    for e in edges:
        if not isinstance(e, dict):
            continue
        out.append(e)
    return out


def _build_campaign_email_lists(gt_path: Path) -> tuple[dict[str, list[dict[str, Any]]], dict[str, str]]:
    """
    Returns:
      canon_cid -> list of email dicts (order preserved within each raw cluster walk)
      canon_cid -> one representative raw cluster key from the file (for stable singleton naming)
    """
    raw = json.loads(gt_path.read_text(encoding="utf-8"))
    clusters = raw.get("clusters") or {}
    if not isinstance(clusters, dict):
        raise ValueError("ground truth must have object 'clusters'")

    cid_to_emails: dict[str, list[dict[str, Any]]] = defaultdict(list)
    cid_to_raw: dict[str, str] = {}

    for raw_key, emails in clusters.items():
        if not isinstance(emails, list):
            continue
        cid = parse_campaign_key(str(raw_key))
        ck = canonical_str(cid)
        if ck not in cid_to_raw:
            cid_to_raw[ck] = str(raw_key)
        for em in emails:
            if isinstance(em, dict):
                cid_to_emails[ck].append(dict(em))

    return dict(cid_to_emails), cid_to_raw


def _dedupe_emails_preserve_order(emails: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for em in emails:
        eid = em.get("external_id")
        if eid is None:
            continue
        s = str(eid)
        if s in seen:
            continue
        seen.add(s)
        out.append(em)
    return out


def _merged_cluster_key(component_cids: list[str], cid_to_raw: dict[str, str]) -> str:
    if len(component_cids) == 1:
        ck = component_cids[0]
        return cid_to_raw.get(ck, ck)
    raw_keys = sorted({cid_to_raw.get(c, c) for c in component_cids})
    h = hashlib.md5("|".join(raw_keys).encode("utf-8")).hexdigest()[:14]
    return f"merged__{h}"


def apply_merges(
    gt_path: Path,
    merges_path: Path,
    out_path: Path,
    *,
    pretty: bool = True,
) -> dict[str, Any]:
    cid_to_emails, cid_to_raw = _build_campaign_email_lists(gt_path)
    edges = _load_merge_edges(merges_path)

    uf = UnionFind()
    for ck in cid_to_emails:
        uf.find(ck)

    n_skipped = 0
    for e in edges:
        lc = e.get("left_campaign")
        rc = e.get("right_campaign")
        if lc is None or rc is None:
            n_skipped += 1
            continue
        a, b = canonical_str(lc), canonical_str(rc)
        if not a or not b:
            n_skipped += 1
            continue
        if a == b:
            continue
        uf.union(a, b)

    components: dict[str, list[str]] = defaultdict(list)
    for ck in cid_to_emails:
        components[uf.find(ck)].append(ck)

    new_clusters: dict[str, list[dict[str, Any]]] = {}
    for _root, cids in components.items():
        all_emails: list[dict[str, Any]] = []
        for ck in sorted(cids, key=str):
            all_emails.extend(cid_to_emails.get(ck, []))
        deduped = _dedupe_emails_preserve_order(all_emails)
        out_key = _merged_cluster_key(sorted(cids, key=str), cid_to_raw)
        if out_key in new_clusters:
            raise RuntimeError(f"duplicate output cluster key {out_key!r}")
        new_clusters[out_key] = deduped

    raw_gt = json.loads(gt_path.read_text(encoding="utf-8"))
    out_doc = {k: v for k, v in raw_gt.items() if k != "clusters"}
    out_doc["clusters"] = new_clusters
    out_doc["_merge_metadata"] = {
        "source_gt": str(gt_path.as_posix()),
        "merges_file": str(merges_path.as_posix()),
        "n_merge_edges_read": len(edges),
        "n_merge_edges_skipped_missing": n_skipped,
        "n_output_clusters": len(new_clusters),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    indent = 2 if pretty else None
    out_path.write_text(json.dumps(out_doc, indent=indent, ensure_ascii=False) + "\n", encoding="utf-8")
    return out_doc["_merge_metadata"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gt", type=Path, required=True, help="Input ground_truth.json")
    ap.add_argument("--merges", type=Path, required=True, help="manual_campaign_merges.json from browser export")
    ap.add_argument("--out", type=Path, required=True, help="Output merged ground truth JSON")
    ap.add_argument("--compact", action="store_true", help="Write compact JSON (no indent)")
    args = ap.parse_args()
    meta = apply_merges(args.gt, args.merges, args.out, pretty=not args.compact)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
