"""
URL node analysis for saved PyG heterogeneous graphs.

For each `url` node, reports the URL string (from graph metadata) and the number of
incident edges on (`email`, `has_url`, `url`) — i.e. how many email–url links touch that
node (parallel links count separately; distinct-email counts are in ``UrlNodeRow``).
"""
from __future__ import annotations

import csv
import json
import os
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from math import comb
from pathlib import Path
from typing import Any, List, Optional, Tuple
from urllib.parse import urlparse

_CORE_ROOT = Path(__file__).resolve().parents[2]
if str(_CORE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CORE_ROOT))

from graph.utils.graph_metrics import _safe_load_graph, load_graph_metadata

_EMAIL_HAS_URL_EDGE = ("email", "has_url", "url")


@dataclass(frozen=True)
class UrlNodeRow:
    url_index: int
    url: str
    email_edge_degree: int
    distinct_email_count: int
    parsed_host: str
    parsed_scheme: str


@dataclass(frozen=True)
class UrlCampaignPairRow:
    """Implicit email-email pairs induced by co-occurring on the same URL node."""

    url_index: int
    url: str
    email_edge_degree: int
    distinct_email_count: int
    parsed_host: str
    n_gt_labeled_emails: int
    n_email_pairs_total: int
    n_same_campaign_pairs: int
    n_cross_campaign_pairs: int
    n_unlabeled_pairs: int
    frac_same_among_gt_pairs: float
    frac_cross_among_gt_pairs: float


def _url_node_count(graph: Any, url_strings: List[str]) -> int:
    n_meta = len(url_strings)
    try:
        if "url" in graph and hasattr(graph["url"], "x") and graph["url"].x is not None:
            return max(n_meta, int(graph["url"].x.size(0)))
    except Exception:
        pass
    return n_meta


def _sanitize_one_line(text: str) -> str:
    return text.replace("\r", " ").replace("\n", " ").strip()


def _parse_campaign_key(raw: str) -> Any:
    s = str(raw).strip()
    if "/" in s:
        s = s.rsplit("/", 1)[-1]
    try:
        return int(s)
    except ValueError:
        return s


def load_gt_label_map(gt_path: str | Path) -> dict[str, Any]:
    """``external_id`` -> campaign id (first GT occurrence wins)."""
    gt_path = Path(gt_path).expanduser().resolve()
    data = json.loads(gt_path.read_text(encoding="utf-8"))
    label_map: dict[str, Any] = {}
    for raw_key, emails in (data.get("clusters") or {}).items():
        cid = _parse_campaign_key(str(raw_key))
        if not isinstance(emails, list):
            continue
        for em in emails:
            if not isinstance(em, dict):
                continue
            eid = em.get("external_id")
            if eid is None:
                continue
            eid = str(eid).strip()
            if not eid or eid in label_map:
                continue
            label_map[eid] = cid
    return label_map


def _email_node_index_to_external_id(metadata: dict[str, Any]) -> dict[int, str]:
    out: dict[int, str] = {}
    for i, row in enumerate((metadata.get("node_maps") or {}).get("email", {}).get("index_to_meta") or []):
        if not isinstance(row, dict):
            continue
        eid = str(row.get("external_id") or "").strip()
        if eid:
            out[int(i)] = eid
    return out


def count_campaign_pairs_for_emails(
    external_ids: list[str],
    label_map: dict[str, Any],
) -> tuple[int, int, int, int, int]:
    """
    For unordered pairs among ``external_ids`` sharing a URL, return:

    ``(n_gt_labeled_emails, n_pairs_total, n_same, n_cross, n_unlabeled)``.
    """
    n = len(external_ids)
    if n < 2:
        n_lab = sum(1 for e in external_ids if e in label_map)
        return n_lab, 0, 0, 0, 0

    n_pairs_total = comb(n, 2)
    campaigns = [label_map.get(e) for e in external_ids]
    n_gt_labeled = sum(1 for c in campaigns if c is not None)
    labeled_campaigns = [c for c in campaigns if c is not None]
    n_gt_pairs = comb(n_gt_labeled, 2) if n_gt_labeled >= 2 else 0
    by_campaign = Counter(labeled_campaigns)
    n_same = sum(comb(c, 2) for c in by_campaign.values() if c >= 2)
    n_cross = n_gt_pairs - n_same
    n_unlabeled = n_pairs_total - n_gt_pairs
    return n_gt_labeled, int(n_pairs_total), int(n_same), int(n_cross), int(n_unlabeled)


def collect_url_campaign_pair_rows(
    graph_path: str,
    meta_path: str,
    gt_path: str | Path,
) -> tuple[List[UrlCampaignPairRow], dict[str, Any]]:
    """
    Per URL node: count implicit email-email pairs from shared URL attachment.

    - **same_campaign**: both emails have GT labels and same campaign id
    - **cross_campaign**: both labeled, different campaigns
    - **unlabeled**: at least one email lacks a GT label
    """
    gt_path = Path(gt_path).expanduser().resolve()
    label_map = load_gt_label_map(gt_path)
    url_rows, base_summary = collect_url_node_rows(graph_path, meta_path)
    metadata = load_graph_metadata(meta_path)
    idx_to_eid = _email_node_index_to_external_id(metadata)

    graph = _safe_load_graph(graph_path)
    n_urls = len(url_rows)
    emails_by_url_sets: dict[int, set[str]] = defaultdict(set)
    if _EMAIL_HAS_URL_EDGE in getattr(graph, "edge_types", []):
        edge_index = graph[_EMAIL_HAS_URL_EDGE].edge_index
        if edge_index is not None and edge_index.numel() > 0:
            for e_idx, u_idx in zip(
                edge_index[0].tolist(), edge_index[1].tolist(), strict=False
            ):
                u = int(u_idx)
                if 0 <= u < n_urls:
                    eid = idx_to_eid.get(int(e_idx))
                    if eid:
                        emails_by_url_sets[u].add(eid)

    emails_by_url_list: dict[int, list[str]] = {
        u: sorted(s) for u, s in emails_by_url_sets.items()
    }

    out_rows: list[UrlCampaignPairRow] = []
    totals = Counter()
    for base in url_rows:
        eids = emails_by_url_list.get(base.url_index, [])
        n_lab, n_tot, n_same, n_cross, n_unlab = count_campaign_pairs_for_emails(eids, label_map)
        if n_tot > 0 and n_lab >= 2:
            gt_pairs = comb(n_lab, 2)
            frac_same = float(n_same / gt_pairs) if gt_pairs else float("nan")
            frac_cross = float(n_cross / gt_pairs) if gt_pairs else float("nan")
        else:
            frac_same = float("nan")
            frac_cross = float("nan")
        out_rows.append(
            UrlCampaignPairRow(
                url_index=base.url_index,
                url=base.url,
                email_edge_degree=base.email_edge_degree,
                distinct_email_count=base.distinct_email_count,
                parsed_host=base.parsed_host,
                n_gt_labeled_emails=n_lab,
                n_email_pairs_total=n_tot,
                n_same_campaign_pairs=n_same,
                n_cross_campaign_pairs=n_cross,
                n_unlabeled_pairs=n_unlab,
                frac_same_among_gt_pairs=frac_same,
                frac_cross_among_gt_pairs=frac_cross,
            )
        )
        totals["n_email_pairs_total"] += n_tot
        totals["n_same_campaign_pairs"] += n_same
        totals["n_cross_campaign_pairs"] += n_cross
        totals["n_unlabeled_pairs"] += n_unlab

    summary = dict(base_summary)
    summary["gt_path"] = str(gt_path)
    summary["n_gt_labeled_emails_in_map"] = len(label_map)
    summary.update({k: int(v) for k, v in totals.items()})
    if totals["n_email_pairs_total"] > 0:
        gt_pairs = totals["n_same_campaign_pairs"] + totals["n_cross_campaign_pairs"]
        summary["frac_same_among_all_pairs"] = float(
            totals["n_same_campaign_pairs"] / totals["n_email_pairs_total"]
        )
        summary["frac_cross_among_all_pairs"] = float(
            totals["n_cross_campaign_pairs"] / totals["n_email_pairs_total"]
        )
        summary["frac_unlabeled_among_all_pairs"] = float(
            totals["n_unlabeled_pairs"] / totals["n_email_pairs_total"]
        )
        if gt_pairs > 0:
            summary["frac_same_among_gt_pairs"] = float(
                totals["n_same_campaign_pairs"] / gt_pairs
            )
            summary["frac_cross_among_gt_pairs"] = float(
                totals["n_cross_campaign_pairs"] / gt_pairs
            )
    return out_rows, summary


def write_url_campaign_pair_rows_csv(
    rows: List[UrlCampaignPairRow],
    output_csv_path: str,
    *,
    encoding: str = "utf-8",
) -> str:
    out = Path(output_csv_path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    sorted_rows = sorted(
        rows,
        key=lambda r: (-(r.n_cross_campaign_pairs + r.n_same_campaign_pairs), -r.email_edge_degree, r.url),
    )
    fieldnames = [
        "rank",
        "url_index",
        "email_edge_degree",
        "distinct_email_count",
        "n_gt_labeled_emails",
        "n_email_pairs_total",
        "n_same_campaign_pairs",
        "n_cross_campaign_pairs",
        "n_unlabeled_pairs",
        "frac_same_among_gt_pairs",
        "frac_cross_among_gt_pairs",
        "parsed_host",
        "url",
    ]
    with open(out, "w", encoding=encoding, newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for rank, row in enumerate(sorted_rows, start=1):
            w.writerow(
                {
                    "rank": rank,
                    "url_index": row.url_index,
                    "email_edge_degree": row.email_edge_degree,
                    "distinct_email_count": row.distinct_email_count,
                    "n_gt_labeled_emails": row.n_gt_labeled_emails,
                    "n_email_pairs_total": row.n_email_pairs_total,
                    "n_same_campaign_pairs": row.n_same_campaign_pairs,
                    "n_cross_campaign_pairs": row.n_cross_campaign_pairs,
                    "n_unlabeled_pairs": row.n_unlabeled_pairs,
                    "frac_same_among_gt_pairs": row.frac_same_among_gt_pairs,
                    "frac_cross_among_gt_pairs": row.frac_cross_among_gt_pairs,
                    "parsed_host": row.parsed_host,
                    "url": row.url,
                }
            )
    return str(out)


def write_url_campaign_pair_totals_csv(
    summary: dict[str, Any],
    output_csv_path: str,
    *,
    encoding: str = "utf-8",
) -> str:
    """Single-row CSV with global totals across all URL-induced email pairs."""
    out = Path(output_csv_path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "gt_path": summary.get("gt_path", ""),
        "n_url_nodes": summary.get("n_url_nodes", 0),
        "n_gt_labeled_emails_in_map": summary.get("n_gt_labeled_emails_in_map", 0),
        "n_email_pairs_total": summary.get("n_email_pairs_total", 0),
        "n_same_campaign_pairs": summary.get("n_same_campaign_pairs", 0),
        "n_cross_campaign_pairs": summary.get("n_cross_campaign_pairs", 0),
        "n_unlabeled_pairs": summary.get("n_unlabeled_pairs", 0),
        "frac_same_among_all_pairs": summary.get("frac_same_among_all_pairs", ""),
        "frac_cross_among_all_pairs": summary.get("frac_cross_among_all_pairs", ""),
        "frac_unlabeled_among_all_pairs": summary.get("frac_unlabeled_among_all_pairs", ""),
        "frac_same_among_gt_pairs": summary.get("frac_same_among_gt_pairs", ""),
        "frac_cross_among_gt_pairs": summary.get("frac_cross_among_gt_pairs", ""),
    }
    with open(out, "w", encoding=encoding, newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        w.writeheader()
        w.writerow(row)
    return str(out)


def _parse_url_bits(url_text: str) -> tuple[str, str]:
    s = (url_text or "").strip()
    if not s:
        return "", ""
    probe = s if "://" in s else f"https://{s}"
    try:
        p = urlparse(probe)
    except ValueError:
        return "", ""
    return (p.scheme or "").lower(), (p.hostname or "").lower()


def collect_url_node_rows(
    graph_path: str,
    meta_path: str,
) -> tuple[List[UrlNodeRow], dict[str, Any]]:
    """
    Return one ``UrlNodeRow`` per URL node index plus a small summary dict.

    ``email_edge_degree`` counts all (`email`, `has_url`, `url`) edges into the URL node.
    ``distinct_email_count`` counts unique source email indices on those edges.
    """
    metadata = load_graph_metadata(meta_path)
    url_strings = list(
        (metadata.get("node_maps") or {}).get("url", {}).get("index_to_string") or []
    )

    graph = _safe_load_graph(graph_path)
    n_urls = _url_node_count(graph, url_strings)

    edge_degrees = Counter()
    emails_by_url: dict[int, set[int]] = defaultdict(set)
    if _EMAIL_HAS_URL_EDGE in getattr(graph, "edge_types", []):
        store = graph[_EMAIL_HAS_URL_EDGE]
        edge_index = getattr(store, "edge_index", None)
        if edge_index is not None and edge_index.numel() > 0:
            src = edge_index[0].tolist()
            dst = edge_index[1].tolist()
            for e_idx, u_idx in zip(src, dst, strict=False):
                u = int(u_idx)
                if 0 <= u < n_urls:
                    edge_degrees[u] += 1
                    emails_by_url[u].add(int(e_idx))

    rows: List[UrlNodeRow] = []
    for i in range(n_urls):
        label = url_strings[i] if i < len(url_strings) else ""
        scheme, host = _parse_url_bits(label)
        rows.append(
            UrlNodeRow(
                url_index=i,
                url=label,
                email_edge_degree=int(edge_degrees[i]),
                distinct_email_count=len(emails_by_url[i]),
                parsed_scheme=scheme,
                parsed_host=host,
            )
        )

    deg_vals = [r.email_edge_degree for r in rows]
    summary: dict[str, Any] = {
        "graph_path": str(Path(graph_path).expanduser().resolve()),
        "meta_path": str(Path(meta_path).expanduser().resolve()),
        "n_url_nodes": int(n_urls),
        "n_url_strings_in_meta": int(len(url_strings)),
        "n_email_has_url_edges": int(sum(deg_vals)),
        "n_url_nodes_degree_zero": int(sum(1 for d in deg_vals if d == 0)),
        "n_url_nodes_degree_one": int(sum(1 for d in deg_vals if d == 1)),
        "max_email_edge_degree": int(max(deg_vals) if deg_vals else 0),
        "edge_counts_from_meta": (metadata.get("edge_counts") or {}),
        "node_counts_from_meta": (metadata.get("node_counts") or {}),
    }
    if deg_vals:
        summary["median_email_edge_degree"] = float(sorted(deg_vals)[len(deg_vals) // 2])
    return rows, summary


def collect_url_email_degrees(
    graph_path: str,
    meta_path: str,
) -> List[Tuple[str, int]]:
    """
    Return one row per URL node index: (url_text, degree_to_email_nodes).

    `degree_to_email_nodes` is the count of edges in (`email`, `has_url`, `url`) whose
    destination is that URL node (standard incident degree on that bipartite relation).
    """
    rows, _ = collect_url_node_rows(graph_path, meta_path)
    return [(r.url, r.email_edge_degree) for r in rows]


def write_url_node_rows_csv(
    rows: List[UrlNodeRow],
    output_csv_path: str,
    *,
    encoding: str = "utf-8",
) -> str:
    """Write URL rows sorted by ``email_edge_degree`` desc, then URL asc."""
    out = Path(output_csv_path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    sorted_rows = sorted(rows, key=lambda r: (-r.email_edge_degree, r.url))
    fieldnames = [
        "rank",
        "url_index",
        "email_edge_degree",
        "distinct_email_count",
        "parsed_scheme",
        "parsed_host",
        "url",
    ]
    with open(out, "w", encoding=encoding, newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for rank, row in enumerate(sorted_rows, start=1):
            w.writerow(
                {
                    "rank": rank,
                    "url_index": row.url_index,
                    "email_edge_degree": row.email_edge_degree,
                    "distinct_email_count": row.distinct_email_count,
                    "parsed_scheme": row.parsed_scheme,
                    "parsed_host": row.parsed_host,
                    "url": row.url,
                }
            )
    return str(out)


def write_url_analysis_bundle(
    graph_path: str,
    meta_path: str,
    output_csv_path: str,
    *,
    summary_json_path: Optional[str] = None,
    encoding: str = "utf-8",
) -> dict[str, Any]:
    """Write ranked URL CSV and optional summary JSON; return paths + summary."""
    rows, summary = collect_url_node_rows(graph_path, meta_path)
    csv_path = write_url_node_rows_csv(rows, output_csv_path, encoding=encoding)
    summary = dict(summary)
    summary["output_csv"] = csv_path
    summary["top_20_by_degree"] = [
        {
            "rank": i + 1,
            "url_index": r.url_index,
            "email_edge_degree": r.email_edge_degree,
            "distinct_email_count": r.distinct_email_count,
            "parsed_host": r.parsed_host,
            "url": _sanitize_one_line(r.url)[:500],
        }
        for i, r in enumerate(
            sorted(rows, key=lambda x: (-x.email_edge_degree, x.url))[:20]
        )
    ]
    if summary_json_path:
        jp = Path(summary_json_path).expanduser().resolve()
        jp.parent.mkdir(parents=True, exist_ok=True)
        jp.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding=encoding)
        summary["output_summary_json"] = str(jp)
    return summary


def write_url_email_degrees_report(
    graph_path: str,
    meta_path: str,
    output_txt_path: str,
    *,
    encoding: str = "utf-8",
) -> Tuple[str, int]:
    """
    Write a tab-separated text file: `email_edge_degree` then `url` (one line per URL node).
    Rows are sorted by degree descending, then URL ascending for ties.

    Creates parent directories for `output_txt_path` if needed.
    Returns ``(resolved_output_path, number_of_url_nodes)``.
    """
    pairs = collect_url_email_degrees(graph_path, meta_path)
    rows = [(u, d) for u, d in pairs]
    rows.sort(key=lambda r: (-r[1], r[0]))
    out = Path(output_txt_path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    with open(out, "w", encoding=encoding, newline="\n") as f:
        f.write("email_edge_degree\turl\n")
        for url_text, deg in rows:
            f.write(f"{deg}\t{_sanitize_one_line(url_text)}\n")
    return str(out), len(rows)


if __name__ == "__main__":
    _here = Path(__file__).resolve().parent
    _default_meta = _here.parent / "output" / "incidents-lake-misp_hetero.meta.json"
    _default_graph = _here.parent / "output" / "incidents-lake-misp_hetero.pt"

    if len(sys.argv) > 1:
        meta_p = sys.argv[1]
        graph_p = sys.argv[2] if len(sys.argv) > 2 else str(_default_graph)
        out_p = sys.argv[3] if len(sys.argv) > 3 else str(_here / "url_email_degrees.txt")
    else:
        meta_p = str(_default_meta)
        graph_p = str(_default_graph)
        out_p = str(_here / "url_email_degrees.txt")

    if not os.path.isfile(meta_p):
        print(f"Error: metadata not found: {meta_p}")
        print(
            "Usage: python url_analysis.py <meta.json> [graph.pt] [output.txt]\n"
            "Degree column counts edges (email, has_url, url) incident on each URL node."
        )
        sys.exit(1)
    if not os.path.isfile(graph_p):
        print(f"Error: graph not found: {graph_p}")
        sys.exit(1)

    written, n = write_url_email_degrees_report(graph_p, meta_p, out_p)
    print(f"Wrote {written} ({n} url nodes)")
