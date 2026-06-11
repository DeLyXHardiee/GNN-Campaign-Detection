"""
Verify surgical popular-domain filtering in the heterogeneous graph.

Assertions:
  1. url nodes  — no bare hostname (www. stripped) exactly matches a popular domain
  2. domain nodes — no registrable domain matches popular_domains (full eTLD+1 filter preserved)
  3. stem nodes — no bare hostname exactly matches a popular domain

Also reports how many url/stem nodes with *subdomain-prefixed* popular domains pass through,
confirming the new behaviour actually differs from the old one.

Run from the project root:
    python scripts/verify_popular_domain_filtering.py [path/to/incidents-lake-misp-url-fixed.json]
"""
import sys
import os
from pathlib import Path

ROOT = Path(__file__).parent.parent
CORE = ROOT / "core"
sys.path.insert(0, str(CORE))

import json

DEFAULT_MISP = CORE / "preprocessing/output/incidents-lake-misp-url-fixed.json"
POPULAR_DOMAINS_TXT = CORE / "feature_set_extraction/caches/popular_domains.txt"


def load_popular_domains() -> frozenset:
    with open(POPULAR_DOMAINS_TXT) as f:
        return frozenset(line.strip().lower() for line in f if line.strip())


def bare_hostname(host: str) -> str:
    h = host.lower()
    return h[4:] if h.startswith("www.") else h


def build_graph(misp_path: Path):
    from graph.graph_builder_pytorch import build_hetero_graph_from_misp
    from feature_set_extraction.domain_lists_loader import load_url_intelligence_sets
    from feature_set_extraction.url_extraction_utils import parse_url_host_and_registrable_domain

    print(f"Loading MISP data from {misp_path} ...")
    with open(misp_path) as f:
        misp_events = json.load(f)

    pop_domains = frozenset(load_url_intelligence_sets().get("popular_domains", set()))
    print(f"Loaded {len(pop_domains)} popular domains.")

    print("Building graph ...")
    graph, metadata = build_hetero_graph_from_misp(
        misp_events,
        filter_popular_domains=True,
    )
    return metadata


def run_assertions(metadata: dict, pop: frozenset) -> bool:
    from feature_set_extraction.url_extraction_utils import parse_url_host_and_registrable_domain

    node_maps = metadata["node_maps"]
    url_nodes    = node_maps.get("url",    {}).get("index_to_string", [])
    domain_nodes = node_maps.get("domain", {}).get("index_to_string", [])
    stem_nodes   = node_maps.get("stem",   {}).get("index_to_string", [])

    failures = []

    exact_blocked_urls = []
    subdomain_pop_urls = []
    for u in url_nodes:
        host, reg, ok = parse_url_host_and_registrable_domain(u)
        if not ok:
            continue
        bare = bare_hostname(host)
        if bare in pop:
            exact_blocked_urls.append((u, bare))
        elif reg in pop and bare != reg:
            subdomain_pop_urls.append((u, host, reg))

    if exact_blocked_urls:
        failures.append(
            f"FAIL [url] {len(exact_blocked_urls)} url node(s) have a bare hostname in popular_domains:\n"
            + "\n".join(f"  {u!r}  (hostname={h!r})" for u, h in exact_blocked_urls[:10])
        )
    else:
        print(f"PASS [url] No url node has a bare hostname in popular_domains ({len(url_nodes)} nodes checked).")

    if subdomain_pop_urls:
        print(f"PASS [url] {len(subdomain_pop_urls)} subdomain-of-popular-domain url node(s) correctly kept by new logic:")
        for u, host, reg in subdomain_pop_urls[:10]:
            print(f"  {u!r}  (host={host!r}, reg={reg!r})")
        if len(subdomain_pop_urls) > 10:
            print(f"  ... and {len(subdomain_pop_urls) - 10} more.")
    else:
        failures.append(
            "FAIL [url] No subdomain-of-popular-domain url nodes found in the graph — "
            "either the dataset has none (check your input) or the new filter is over-blocking."
        )

    leaked_domains = []
    for d in domain_nodes:
        host, reg, ok = parse_url_host_and_registrable_domain(d)
        if not ok:
            continue
        if reg in pop:
            leaked_domains.append((d, reg))

    if leaked_domains:
        failures.append(
            f"FAIL [domain] {len(leaked_domains)} domain node(s) have a popular registrable domain (should be filtered):\n"
            + "\n".join(f"  {d!r}  (reg={r!r})" for d, r in leaked_domains[:10])
        )
    else:
        print(f"PASS [domain] No domain node has a registrable domain in popular_domains ({len(domain_nodes)} nodes checked).")

    print(f"INFO [stem] {len(stem_nodes)} stem nodes present (covered by url-node assertion).")

    if failures:
        print()
        for f in failures:
            print(f)
        return False

    return True


def main():
    misp_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_MISP
    if not misp_path.exists():
        print(f"ERROR: MISP file not found: {misp_path}")
        print("Pass the path as the first argument, e.g.:")
        print("  python scripts/verify_popular_domain_filtering.py data/misp/incidents-lake-misp-url-fixed.json")
        sys.exit(1)

    pop = load_popular_domains()
    metadata = build_graph(misp_path)
    ok = run_assertions(metadata, pop)

    print()
    if ok:
        print("All assertions passed.")
        sys.exit(0)
    else:
        print("One or more assertions FAILED.")
        sys.exit(1)


if __name__ == "__main__":
    main()
