"""
Graph metrics and statistics checker for heterogeneous email graphs.

Now filter-aware: all functions gracefully handle missing node/edge types
when graphs are built with excluded nodes.

Provides utilities to analyze graph structure, node counts, edge counts,
and identify top entities (URLs, domains, stems, senders, receivers, etc.).
Also generates a Markdown report for easy sharing.
"""
from typing import Dict, List, Tuple, Any, Optional
from collections import Counter, deque
import json
import os
from datetime import datetime


def load_graph_metadata(meta_path: str) -> Dict[str, Any]:
    """Load graph metadata from JSON file."""
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def print_graph_overview(metadata: Dict[str, Any]) -> None:
    """Print high-level graph statistics."""
    print("=" * 70)
    print("GRAPH OVERVIEW")
    print("=" * 70)
    
    feature_shapes = metadata.get("feature_shapes", {})
    edge_counts = metadata.get("edge_counts", {})
    
    print("\nNode Counts:")
    total_nodes = 0
    for node_type, shape in feature_shapes.items():
        count = shape[0] if shape else 0
        total_nodes += count
        print(f"  {node_type:12s}: {count:8,d} nodes")
    print(f"  {'TOTAL':12s}: {total_nodes:8,d} nodes")
    
    print("\nEdge Counts:")
    total_edges = 0
    for edge_name, count in edge_counts.items():
        total_edges += count
        print(f"  {edge_name:35s}: {count:8,d} edges")
    print(f"  {'TOTAL':35s}: {total_edges:8,d} edges")
    
    print()


def _md_graph_overview(metadata: Dict[str, Any]) -> str:
    feature_shapes = metadata.get("feature_shapes", {})
    edge_counts = metadata.get("edge_counts", {})

    lines = ["## Graph Overview", ""]
    lines.append("### Node Counts")
    total_nodes = 0
    for node_type, shape in feature_shapes.items():
        count = shape[0] if shape else 0
        total_nodes += count
        lines.append(f"- {node_type}: {count:,} nodes")
    lines.append(f"- TOTAL: {total_nodes:,} nodes")

    lines.append("")
    lines.append("### Edge Counts")
    total_edges = 0
    for edge_name, count in edge_counts.items():
        total_edges += count
        lines.append(f"- {edge_name}: {count:,} edges")
    lines.append(f"- TOTAL: {total_edges:,} edges")
    lines.append("")
    return "\n".join(lines)


def get_top_urls(metadata: Dict[str, Any], top_n: int = 5) -> List[Tuple[str, int]]:
    """
    Get the top N most frequently referenced URLs.
    
    Returns list of (url, count) tuples sorted by count descending.
    """
    node_maps = metadata.get("node_maps", {})
    email_meta = node_maps.get("email", {}).get("index_to_meta", [])
    edge_counts_dict = metadata.get("edge_counts", {})
    
    # Check if we have URL edges
    has_url_edges = edge_counts_dict.get("email->url:has_url", 0) > 0
    
    if not has_url_edges:
        return []
    
    # We need to count URL references from the graph structure
    # For simplicity, we'll use the url node mapping and assume frequency correlates with index
    # In a real implementation, we'd load the actual graph to count edge references
    
    url_strings = node_maps.get("url", {}).get("index_to_string", [])
    
    # Return URLs with their indices (as a proxy for frequency for now)
    # A better approach would be to load the actual graph and count edges
    if url_strings:
        # For now, return all URLs with placeholder counts
        # TODO: Load actual graph to get real edge counts per URL
        return [(url, 1) for url in url_strings[:top_n]]
    
    return []


def get_top_receivers_from_graph(graph_path: str, metadata: Dict[str, Any], top_n: int = 5) -> List[Tuple[str, int]]:
    """Top receivers by number of incoming emails using edge counts."""
    try:
        import torch
        graph = torch.load(graph_path, weights_only=False)
        receiver_strings = metadata.get("node_maps", {}).get("receiver", {}).get("index_to_string", [])
        if not receiver_strings or ("email", "has_receiver", "receiver") not in getattr(graph, "edge_types", []):
            return []
        idxs = graph["email", "has_receiver", "receiver"].edge_index[1].tolist()
        c = Counter()
        for i in idxs:
            if 0 <= i < len(receiver_strings):
                c[receiver_strings[i]] += 1
        return c.most_common(top_n)
    except Exception:
        return []


def count_url_references_from_graph(graph_path: str, metadata: Dict[str, Any], top_n: int = 5) -> List[Tuple[str, int]]:
    """
    Count URL references by loading the actual graph and analyzing edge_index.
    
    Args:
        graph_path: Path to the .pt graph file
        metadata: Graph metadata dict
        top_n: Number of top URLs to return
    
    Returns:
        List of (url, count) tuples sorted by count descending
    """
    try:
        import torch
        
        # Load the graph with weights_only=False for PyTorch 2.6+ compatibility
        graph = torch.load(graph_path, weights_only=False)
        
        # Get URL node strings
        url_strings = metadata.get("node_maps", {}).get("url", {}).get("index_to_string", [])
        
        if not url_strings:
            return []
        
        # Count how many emails reference each URL
        url_counts = Counter()
        
        # Check if the edge type exists
        if ("email", "has_url", "url") in graph.edge_types:
            edge_index = graph["email", "has_url", "url"].edge_index
            
            # edge_index[1] contains URL node indices
            url_indices = edge_index[1].tolist()
            
            for url_idx in url_indices:
                if 0 <= url_idx < len(url_strings):
                    url_counts[url_strings[url_idx]] += 1
        
        # Return top N
        return url_counts.most_common(top_n)
        
    except ImportError:
        print("Warning: torch not available, cannot count URL references from graph")
        return []
    except Exception as e:
        print(f"Warning: Could not load graph: {e}")
        return []


def print_top_urls(metadata: Dict[str, Any], graph_path: Optional[str] = None, top_n: int = 5) -> None:
    """Print the top N most referenced URLs."""
    print("=" * 70)
    print(f"TOP {top_n} MOST REFERENCED URLs")
    print("=" * 70)
    
    if graph_path:
        top_urls = count_url_references_from_graph(graph_path, metadata, top_n)
    else:
        # Fallback: just list URLs from metadata
        url_strings = metadata.get("node_maps", {}).get("url", {}).get("index_to_string", [])
        top_urls = [(url, 0) for url in url_strings[:top_n]]
    
    if not top_urls:
        print("\nNo URLs found in the graph.")
        print()
        return
    
    for i, (url, count) in enumerate(top_urls, 1):
        if count > 0:
            print(f"{i:2d}. {url:60s} ({count:4d} emails)")
        else:
            print(f"{i:2d}. {url}")
    
    print()


def _md_top_list(title: str, pairs: List[Tuple[str, int]], unit: str) -> str:
    lines = [f"## {title}", ""]
    if not pairs:
        lines.append("No data available.")
        lines.append("")
        return "\n".join(lines)
    for i, (label, count) in enumerate(pairs, 1):
        suffix = f" ({count} {unit})" if count > 0 else ""
        lines.append(f"- {i}. {label}{suffix}")
    lines.append("")
    return "\n".join(lines)


def print_top_domains(metadata: Dict[str, Any], graph_path: Optional[str] = None, top_n: int = 5) -> None:
    """Print the top N most referenced domains."""
    print("=" * 70)
    print(f"TOP {top_n} MOST REFERENCED DOMAINS")
    print("=" * 70)
    
    domain_strings = metadata.get("node_maps", {}).get("domain", {}).get("index_to_string", [])
    
    if not domain_strings:
        print("\nNo domains found in the graph.")
        print()
        return
    
    if graph_path:
        try:
            import torch
            graph = torch.load(graph_path, weights_only=False)
            
            domain_counts = Counter()
            
            if ("url", "has_domain", "domain") in graph.edge_types:
                edge_index = graph["url", "has_domain", "domain"].edge_index
                domain_indices = edge_index[1].tolist()
                
                for domain_idx in domain_indices:
                    if 0 <= domain_idx < len(domain_strings):
                        domain_counts[domain_strings[domain_idx]] += 1
            
            top_domains = domain_counts.most_common(top_n)
            
            for i, (domain, count) in enumerate(top_domains, 1):
                print(f"{i:2d}. {domain:50s} ({count:4d} URLs)")
        except Exception as e:
            print(f"\nCould not load graph for domain counting: {e}")
            for i, domain in enumerate(domain_strings[:top_n], 1):
                print(f"{i:2d}. {domain}")
    else:
        for i, domain in enumerate(domain_strings[:top_n], 1):
            print(f"{i:2d}. {domain}")
    
    print()


def get_top_stems_from_graph(graph_path: str, metadata: Dict[str, Any], top_n: int = 5) -> List[Tuple[str, int]]:
    """Top stems by URL references."""
    try:
        import torch
        graph = torch.load(graph_path, weights_only=False)
        stem_strings = metadata.get("node_maps", {}).get("stem", {}).get("index_to_string", [])
        if not stem_strings or ("url", "has_stem", "stem") not in getattr(graph, "edge_types", []):
            return []
        idxs = graph["url", "has_stem", "stem"].edge_index[1].tolist()
        c = Counter()
        for i in idxs:
            if 0 <= i < len(stem_strings):
                c[stem_strings[i]] += 1
        return c.most_common(top_n)
    except Exception:
        return []


def print_top_senders(metadata: Dict[str, Any], graph_path: Optional[str] = None, top_n: int = 5) -> None:
    """Print the top N most active senders."""
    print("=" * 70)
    print(f"TOP {top_n} MOST ACTIVE SENDERS")
    print("=" * 70)
    
    sender_strings = metadata.get("node_maps", {}).get("sender", {}).get("index_to_string", [])
    
    if not sender_strings:
        print("\nNo senders found in the graph.")
        print()
        return
    
    if graph_path:
        try:
            import torch
            graph = torch.load(graph_path, weights_only=False)
            
            sender_counts = Counter()
            
            if ("email", "has_sender", "sender") in graph.edge_types:
                edge_index = graph["email", "has_sender", "sender"].edge_index
                sender_indices = edge_index[1].tolist()
                
                for sender_idx in sender_indices:
                    if 0 <= sender_idx < len(sender_strings):
                        sender_counts[sender_strings[sender_idx]] += 1
            
            top_senders = sender_counts.most_common(top_n)
            
            for i, (sender, count) in enumerate(top_senders, 1):
                print(f"{i:2d}. {sender:60s} ({count:4d} emails)")
        except Exception as e:
            print(f"\nCould not load graph for sender counting: {e}")
            for i, sender in enumerate(sender_strings[:top_n], 1):
                print(f"{i:2d}. {sender}")
    else:
        for i, sender in enumerate(sender_strings[:top_n], 1):
            print(f"{i:2d}. {sender}")
    
    print()


def print_top_receivers(metadata: Dict[str, Any], graph_path: Optional[str] = None, top_n: int = 5) -> None:
    print("=" * 70)
    print(f"TOP {top_n} MOST ACTIVE RECEIVERS")
    print("=" * 70)

    receiver_strings = metadata.get("node_maps", {}).get("receiver", {}).get("index_to_string", [])
    if not receiver_strings:
        print("\nNo receivers found in the graph.")
        print()
        return

    pairs: List[Tuple[str, int]] = []
    if graph_path:
        pairs = get_top_receivers_from_graph(graph_path, metadata, top_n)
    else:
        pairs = [(r, 0) for r in receiver_strings[:top_n]]

    if not pairs:
        print("\nNo receiver edges found.")
        print()
        return

    for i, (receiver, count) in enumerate(pairs, 1):
        if count > 0:
            print(f"{i:2d}. {receiver:60s} ({count:4d} emails)")
        else:
            print(f"{i:2d}. {receiver}")
    print()


def print_week_distribution(metadata: Dict[str, Any], graph_path: Optional[str] = None) -> None:
    """Print email distribution across weeks."""
    print("=" * 70)
    print("EMAIL DISTRIBUTION BY WEEK")
    print("=" * 70)
    
    week_strings = metadata.get("node_maps", {}).get("week", {}).get("index_to_string", [])
    
    if not week_strings:
        print("\nNo week data found in the graph.")
        print()
        return
    
    if graph_path:
        try:
            import torch
            graph = torch.load(graph_path, weights_only=False)
            
            week_counts = Counter()
            
            if ("email", "in_week", "week") in graph.edge_types:
                edge_index = graph["email", "in_week", "week"].edge_index
                week_indices = edge_index[1].tolist()
                
                for week_idx in week_indices:
                    if 0 <= week_idx < len(week_strings):
                        week_counts[week_strings[week_idx]] += 1
            
            # Sort by week string (chronological)
            sorted_weeks = sorted(week_counts.items())
            
            print()
            for week, count in sorted_weeks:
                bar = "█" * min(50, count // 10)
                print(f"{week:12s}: {count:5d} emails {bar}")
        except Exception as e:
            print(f"\nCould not load graph for week distribution: {e}")
            print(f"Total weeks: {len(week_strings)}")
    else:
        print(f"\nTotal weeks: {len(week_strings)}")
    
    print()


def _md_week_distribution(metadata: Dict[str, Any], graph_path: Optional[str]) -> str:
    week_strings = metadata.get("node_maps", {}).get("week", {}).get("index_to_string", [])
    lines = ["## Email Distribution by Week", ""]
    if not week_strings:
        lines.append("No week data found in the graph.")
        lines.append("")
        return "\n".join(lines)
    if not graph_path:
        lines.append(f"Total weeks: {len(week_strings)}")
        lines.append("")
        return "\n".join(lines)
    try:
        import torch
        graph = torch.load(graph_path, weights_only=False)
        if ("email", "in_week", "week") not in getattr(graph, "edge_types", []):
            lines.append("No in_week edges present.")
            lines.append("")
            return "\n".join(lines)
        edge_index = graph["email", "in_week", "week"].edge_index
        week_indices = edge_index[1].tolist()
        c = Counter(week_indices)
        sorted_weeks = sorted(((week_strings[i], count) for i, count in c.items() if 0 <= i < len(week_strings)))
        lines.append("")
        lines.append("```")
        for week, count in sorted_weeks:
            bar = "█" * min(50, count // 10)
            lines.append(f"{week:12s}: {count:5d} emails {bar}")
        lines.append("```")
        lines.append("")
        return "\n".join(lines)
    except Exception as e:
        return "\n".join(lines + [f"Could not load graph for week distribution: {e}", ""]) 


def analyze_graph(meta_path: str, graph_path: Optional[str] = None) -> None:
    """
    Complete graph analysis with all metrics and write a Markdown report.

    Args:
        meta_path: Path to metadata JSON file
        graph_path: Optional path to .pt graph file for detailed analysis
    """
    metadata = load_graph_metadata(meta_path)

    # Build markdown report
    sections: List[str] = []
    sections.append(f"# Graph Analysis Report\n\nGenerated: {datetime.utcnow().isoformat()}Z\n")
    sections.append(_md_graph_overview(metadata))

    # Top entities (robust to filtering)
    # URLs
    urls = count_url_references_from_graph(graph_path, metadata, 5) if graph_path else []
    if not urls:
        url_strings = metadata.get("node_maps", {}).get("url", {}).get("index_to_string", [])
        urls = [(u, 0) for u in url_strings[:5]]
    sections.append(_md_top_list("Top URLs", urls, "emails"))

    # Domains
    try:
        import torch
        dom_pairs: List[Tuple[str,int]] = []
        if graph_path:
            graph = torch.load(graph_path, weights_only=False)
            domain_strings = metadata.get("node_maps", {}).get("domain", {}).get("index_to_string", [])
            if domain_strings and ("url", "has_domain", "domain") in getattr(graph, "edge_types", []):
                idxs = graph["url", "has_domain", "domain"].edge_index[1].tolist()
                c = Counter(i for i in idxs if 0 <= i < len(domain_strings))
                dom_pairs = [(domain_strings[i], cnt) for i, cnt in c.most_common(5)]
        if not dom_pairs:
            domain_strings = metadata.get("node_maps", {}).get("domain", {}).get("index_to_string", [])
            dom_pairs = [(d, 0) for d in domain_strings[:5]]
    except Exception:
        dom_pairs = []
    sections.append(_md_top_list("Top Domains", dom_pairs, "URLs"))

    # Stems
    stem_pairs = get_top_stems_from_graph(graph_path, metadata, 5) if graph_path else []
    if not stem_pairs:
        stem_strings = metadata.get("node_maps", {}).get("stem", {}).get("index_to_string", [])
        stem_pairs = [(s, 0) for s in stem_strings[:5]]
    sections.append(_md_top_list("Top Stems", stem_pairs, "URLs"))

    # Senders
    send_pairs: List[Tuple[str,int]] = []
    if graph_path:
        try:
            import torch
            graph = torch.load(graph_path, weights_only=False)
            sender_strings = metadata.get("node_maps", {}).get("sender", {}).get("index_to_string", [])
            if sender_strings and ("email", "has_sender", "sender") in getattr(graph, "edge_types", []):
                idxs = graph["email", "has_sender", "sender"].edge_index[1].tolist()
                c = Counter(i for i in idxs if 0 <= i < len(sender_strings))
                send_pairs = [(sender_strings[i], cnt) for i, cnt in c.most_common(5)]
        except Exception:
            send_pairs = []
    if not send_pairs:
        sender_strings = metadata.get("node_maps", {}).get("sender", {}).get("index_to_string", [])
        send_pairs = [(s, 0) for s in sender_strings[:5]]
    sections.append(_md_top_list("Top Senders", send_pairs, "emails"))

    # Receivers
    recv_pairs = get_top_receivers_from_graph(graph_path, metadata, 5) if graph_path else []
    if not recv_pairs:
        receiver_strings = metadata.get("node_maps", {}).get("receiver", {}).get("index_to_string", [])
        recv_pairs = [(r, 0) for r in receiver_strings[:5]]
    sections.append(_md_top_list("Top Receivers", recv_pairs, "emails"))

    # Week distribution
    sections.append(_md_week_distribution(metadata, graph_path))

    # Connected components
    # We'll reuse the existing print function logic by recomputing here as MD
    comp_md_lines = ["## Connected Components", ""]
    if graph_path:
        try:
            import torch
            graph = torch.load(graph_path, weights_only=False)
            node_types = list(graph.node_types)
            offsets = {}
            off = 0
            for nt in node_types:
                num = graph[nt].num_nodes if hasattr(graph[nt], "num_nodes") else (graph[nt].x.size(0) if "x" in graph[nt] else 0)
                offsets[nt] = off
                off += int(num)
            total = off
            adj = [[] for _ in range(total)]
            for src_t, rel, dst_t in graph.edge_types:
                edge_index = graph[src_t, rel, dst_t].edge_index
                if edge_index is None:
                    continue
                src_idx = edge_index[0].tolist()
                dst_idx = edge_index[1].tolist()
                bs = offsets[src_t]
                bd = offsets[dst_t]
                for s_i, d_i in zip(src_idx, dst_idx):
                    s_g = bs + int(s_i)
                    d_g = bd + int(d_i)
                    adj[s_g].append(d_g)
                    adj[d_g].append(s_g)
            visited = [False]*total
            sizes: List[int] = []
            for nid in range(total):
                if visited[nid]:
                    continue
                if not adj[nid]:
                    visited[nid] = True
                    sizes.append(1)
                    continue
                dq = deque([nid])
                visited[nid] = True
                size = 0
                while dq:
                    v = dq.popleft()
                    size += 1
                    for nb in adj[v]:
                        if not visited[nb]:
                            visited[nb] = True
                            dq.append(nb)
                sizes.append(size)
            sizes.sort(reverse=True)
            top = sizes[:5]
            if not top:
                comp_md_lines.append("No components found.")
            else:
                for i, sz in enumerate(top, 1):
                    comp_md_lines.append(f"- {i}. {sz:,d} nodes")
        except Exception as e:
            comp_md_lines.append(f"Could not compute components: {e}")
    else:
        comp_md_lines.append("Graph path not provided.")
    comp_md_lines.append("")
    sections.append("\n".join(comp_md_lines))

    # Write report
    out_dir = os.path.join("core", "utils", "results")
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(meta_path))[0] or "graph"
    out_path = os.path.join(out_dir, f"{base}_analysis.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(sections))

    print(f"\nSaved analysis report: {out_path}\n")


def print_connected_components(metadata: Dict[str, Any], graph_path: Optional[str], top_n: int = 5) -> None:
    """Load the HeteroData graph and compute connected components across all node types.

    The HeteroData stores edge_index per edge type; we build an undirected adjacency across
    node-type/index pairs and perform BFS to find connected components. We then print the
    sizes of the top N largest components.
    """
    print("=" * 70)
    print(f"TOP {top_n} LARGEST CONNECTED COMPONENTS")
    print("=" * 70)

    if not graph_path:
        print("\nNo graph file provided; cannot compute connected components.\n")
        return

    try:
        import torch

        graph = torch.load(graph_path, weights_only=False)

        # Build mapping from (node_type, local_idx) -> global id
        node_types = list(graph.node_types)
        node_type_offsets = {}
        offset = 0
        for nt in node_types:
            num = graph[nt].num_nodes if hasattr(graph[nt], "num_nodes") else (graph[nt].x.size(0) if "x" in graph[nt] else 0)
            node_type_offsets[nt] = offset
            offset += int(num)

        total_nodes = offset

        # Build adjacency list as list of lists
        adj = [[] for _ in range(total_nodes)]

        # Iterate over all edge types and add undirected edges
        for src_type, rel, dst_type in graph.edge_types:
            ekey = (src_type, rel, dst_type)
            if ekey not in graph.edge_types:
                continue
            edge_index = graph[src_type, rel, dst_type].edge_index
            if edge_index is None:
                continue
            src_indices = edge_index[0].tolist()
            dst_indices = edge_index[1].tolist()
            base_src = node_type_offsets[src_type]
            base_dst = node_type_offsets[dst_type]

            for s_i, d_i in zip(src_indices, dst_indices):
                s_global = base_src + int(s_i)
                d_global = base_dst + int(d_i)
                # undirected
                adj[s_global].append(d_global)
                adj[d_global].append(s_global)

        # BFS to compute components
        visited = [False] * total_nodes
        comps = []
        from collections import deque

        for nid in range(total_nodes):
            if visited[nid]:
                continue
            if not adj[nid]:
                # isolated node
                visited[nid] = True
                comps.append(1)
                continue

            q = deque([nid])
            visited[nid] = True
            size = 0
            while q:
                v = q.popleft()
                size += 1
                for nb in adj[v]:
                    if not visited[nb]:
                        visited[nb] = True
                        q.append(nb)

            comps.append(size)

        comps.sort(reverse=True)
        top = comps[:top_n]

        if not top:
            print("\nNo components found in graph.\n")
            return

        for i, sz in enumerate(top, 1):
            print(f"{i:2d}. {sz:,d} nodes")

        print()

    except Exception as e:
        print(f"Could not compute connected components: {e}\n")


if __name__ == "__main__":
    import sys
    import os
    
    # Default paths
    default_meta = os.path.join("results", "trec07_misp_hetero.meta.json")
    default_graph = os.path.join("results", "trec07_misp_hetero.pt")
    
    if len(sys.argv) > 1:
        meta_path = sys.argv[1]
        graph_path = sys.argv[2] if len(sys.argv) > 2 else None
    else:
        meta_path = default_meta
        graph_path = default_graph if os.path.exists(default_graph) else None
    
    if not os.path.exists(meta_path):
        print(f"Error: Metadata file not found: {meta_path}")
        print(f"\nUsage: python graph_metrics.py [meta_path] [graph_path]")
        sys.exit(1)
    
    analyze_graph(meta_path, graph_path)
