"""
URL node analysis for saved PyG heterogeneous graphs.

For each `url` node, reports the URL string (from graph metadata) and the number of
incident edges on (`email`, `has_url`, `url`) — i.e. how many email–url links touch that
node (parallel links count separately; use distinct-email counts only if you extend this).
"""
from __future__ import annotations

import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, List, Tuple

_CORE_ROOT = Path(__file__).resolve().parents[2]
if str(_CORE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CORE_ROOT))

from graph.utils.graph_metrics import _safe_load_graph, load_graph_metadata

_EMAIL_HAS_URL_EDGE = ("email", "has_url", "url")


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


def collect_url_email_degrees(
    graph_path: str,
    meta_path: str,
) -> List[Tuple[str, int]]:
    """
    Return one row per URL node index: (url_text, degree_to_email_nodes).

    `degree_to_email_nodes` is the count of edges in (`email`, `has_url`, `url`) whose
    destination is that URL node (standard incident degree on that bipartite relation).
    """
    metadata = load_graph_metadata(meta_path)
    url_strings = list(
        (metadata.get("node_maps") or {}).get("url", {}).get("index_to_string") or []
    )

    graph = _safe_load_graph(graph_path)
    n_urls = _url_node_count(graph, url_strings)

    degrees = Counter()
    if _EMAIL_HAS_URL_EDGE in getattr(graph, "edge_types", []):
        store = graph[_EMAIL_HAS_URL_EDGE]
        edge_index = getattr(store, "edge_index", None)
        if edge_index is not None and edge_index.numel() > 0:
            for u in edge_index[1].tolist():
                u = int(u)
                if 0 <= u < n_urls:
                    degrees[u] += 1

    rows: List[Tuple[str, int]] = []
    for i in range(n_urls):
        label = url_strings[i] if i < len(url_strings) else ""
        rows.append((label, int(degrees[i])))
    return rows


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
    rows = collect_url_email_degrees(graph_path, meta_path)
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
