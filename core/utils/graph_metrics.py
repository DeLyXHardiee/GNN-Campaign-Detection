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
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_max_degree_node_in_largest_component(graph_path: str, metadata: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Find the node with the highest total degree (incoming + outgoing across all edge types)
    within the largest connected component of the heterogeneous graph.

    Returns a dictionary with keys: node_type, local_index, global_index, degree, component_size, label.
    Returns None if the graph is empty or cannot be loaded.
    """
    try:
        import torch

        graph = torch.load(graph_path, weights_only=False)

        node_types = list(getattr(graph, "node_types", []))
        if not node_types:
            return None

        inferred_counts: Dict[str, int] = {nt: 0 for nt in node_types}
        for (src_t, rel, dst_t) in getattr(graph, "edge_types", []):
            store = graph[src_t, rel, dst_t]
            edge_index = getattr(store, "edge_index", None)
            if edge_index is None or edge_index.numel() == 0:
                continue
            src_max = int(edge_index[0].max().item()) if edge_index[0].numel() > 0 else -1
            dst_max = int(edge_index[1].max().item()) if edge_index[1].numel() > 0 else -1
            if src_max >= 0:
                inferred_counts[src_t] = max(inferred_counts[src_t], src_max + 1)
            if dst_max >= 0:
                inferred_counts[dst_t] = max(inferred_counts[dst_t], dst_max + 1)
        for nt in node_types:
            try:
                if "x" in graph[nt]:
                    inferred_counts[nt] = max(inferred_counts[nt], int(graph[nt].x.size(0)))
            except Exception:
                pass

        offsets: Dict[str, int] = {}
        off = 0
        for nt in node_types:
            offsets[nt] = off
            off += int(inferred_counts.get(nt, 0))
        total = off
        if total == 0:
            return None

        adj: List[List[int]] = [[] for _ in range(total)]
        deg = [0] * total
        for src_t, rel, dst_t in getattr(graph, "edge_types", []):
            store = graph[src_t, rel, dst_t]
            edge_index = getattr(store, "edge_index", None)
            if edge_index is None or edge_index.numel() == 0:
                continue
            bs = offsets[src_t]
            bd = offsets[dst_t]
            src_idx = edge_index[0].tolist()
            dst_idx = edge_index[1].tolist()
            for s_i, d_i in zip(src_idx, dst_idx):
                s_g = bs + int(s_i)
                d_g = bd + int(d_i)
                if 0 <= s_g < total and 0 <= d_g < total:
                    adj[s_g].append(d_g)
                    adj[d_g].append(s_g)
                    deg[s_g] += 1
                    deg[d_g] += 1

        visited = [False] * total
        comp_id = [-1] * total
        comp_sizes: List[int] = []
        current_id = 0
        for nid in range(total):
            if visited[nid]:
                continue
            from collections import deque
            q = deque([nid])
            visited[nid] = True
            comp_id[nid] = current_id
            size = 0
            while q:
                v = q.popleft()
                size += 1
                for nb in adj[v]:
                    if not visited[nb]:
                        visited[nb] = True
                        comp_id[nb] = current_id
                        q.append(nb)
            comp_sizes.append(size)
            current_id += 1

        if not comp_sizes:
            return None
        largest_c = max(range(len(comp_sizes)), key=lambda i: comp_sizes[i])
        largest_size = comp_sizes[largest_c]

        best_global = None
        best_deg = -1
        for gid in range(total):
            if comp_id[gid] != largest_c:
                continue
            if deg[gid] > best_deg:
                best_deg = deg[gid]
                best_global = gid

        if best_global is None:
            return None

        # Map global index back to node_type and local_index
        best_type = node_types[0]
        best_local = 0
        for nt in node_types:
            base = offsets[nt]
            count = inferred_counts.get(nt, 0)
            if base <= best_global < base + count:
                best_type = nt
                best_local = best_global - base
                break

        label: Optional[str] = None
        nm = metadata.get("node_maps", {}).get(best_type, {})
        idx2str = nm.get("index_to_string")
        if isinstance(idx2str, list) and 0 <= best_local < len(idx2str):
            label = idx2str[best_local]

        return {
            "node_type": best_type,
            "local_index": int(best_local),
            "global_index": int(best_global),
            "degree": int(best_deg),
            "component_size": int(largest_size),
            "label": label,
        }
    except Exception:
        return None

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
    node_maps = metadata.get("node_maps", {})
    email_meta = node_maps.get("email", {}).get("index_to_meta", [])
    edge_counts_dict = metadata.get("edge_counts", {})
    
    has_url_edges = edge_counts_dict.get("email->url:has_url", 0) > 0
    
    if not has_url_edges:
        return []
    
    # We need to count URL references from the graph structure
    # For simplicity, we use the url node mapping and assume frequency correlates with index
    url_strings = node_maps.get("url", {}).get("index_to_string", [])
    
    # Return URLs with their indices (as a proxy for frequency for now)
    # A better approach would be to load the actual graph and count edges
    if url_strings:
        # For now, return all URLs with placeholder counts
        # TODO: Load actual graph to get real edge counts per URL
        return [(url, 1) for url in url_strings[:top_n]]
    
    return []


def get_top_receivers_from_graph(graph_path: str, metadata: Dict[str, Any], top_n: int = 5) -> List[Tuple[str, int]]:
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
    try:
        import torch
        
        graph = torch.load(graph_path, weights_only=False)
        
        url_strings = metadata.get("node_maps", {}).get("url", {}).get("index_to_string", [])
        
        if not url_strings:
            return []
        
        url_counts = Counter()
        
        if ("email", "has_url", "url") in graph.edge_types:
            edge_index = graph["email", "has_url", "url"].edge_index
            
            # edge_index[1] contains URL node indices
            url_indices = edge_index[1].tolist()
            
            for url_idx in url_indices:
                if 0 <= url_idx < len(url_strings):
                    url_counts[url_strings[url_idx]] += 1
        
        return url_counts.most_common(top_n)
        
    except ImportError:
        print("Warning: torch not available, cannot count URL references from graph")
        return []
    except Exception as e:
        print(f"Warning: Could not load graph: {e}")
        return []


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


def get_top_stems_from_graph(graph_path: str, metadata: Dict[str, Any], top_n: int = 5) -> List[Tuple[str, int]]:
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
    

def md_random_node_features_sample(meta_path: str, graph_path: Optional[str] = None, sample_size: int = 5) -> str:
    import random

    metadata = load_graph_metadata(meta_path)
    feature_shapes = metadata.get("feature_shapes", {})
    node_maps = metadata.get("node_maps", {})

    samples_md = ["## Random Node Feature Samples", ""]

    try:
        import torch
        graph = None
        if graph_path:
            graph = torch.load(graph_path, weights_only=False)

        for node_type, shape in feature_shapes.items():
            count = shape[0] if shape else 0
            if count == 0:
                continue
            samples_md.append(f"### Node Type: {node_type} (Total Nodes: {count})")
            samples_md.append("")
            indices = random.sample(range(count), min(sample_size, count))
            header = ["Index"] + [f"Feat_{i}" for i in range(shape[1])] if len(shape) > 1 else ["Index", "Feature"]
            samples_md.append("| " + " | ".join(header) + " |")
            samples_md.append("|" + " --- |" * len(header))
            for idx in indices:
                feat_values = []
                if graph and node_type in graph.node_types:
                    try:
                        feat_tensor = graph[node_type].x
                        if feat_tensor is not None and idx < feat_tensor.size(0):
                            feat_row = feat_tensor[idx].tolist()
                            feat_values = [f"{v:.4f}" if isinstance(v, float) else str(v) for v in feat_row]
                    except Exception:
                        pass
                if not feat_values:
                    feat_values = ["N/A"] * (shape[1] if len(shape) > 1 else 1)
                row = [str(idx)] + feat_values
                samples_md.append("| " + " | ".join(row) + " |")
            samples_md.append("")

    except ImportError:
        samples_md.append("Warning: torch not available, cannot sample node features.")
    except Exception as e:
        samples_md.append(f"Warning: Could not load graph for feature sampling: {e}")

    if len(samples_md) == 2:
        samples_md.append("No node features were found in metadata to sample.")

    return "\n".join(samples_md)


def analyze_graph(meta_path: str, graph_path: Optional[str] = None) -> None:
    metadata = load_graph_metadata(meta_path)

    sections: List[str] = []
    sections.append(f"# Graph Analysis Report\n\nGenerated: {datetime.utcnow().isoformat()}Z\n")
    sections.append(_md_graph_overview(metadata))

    urls = count_url_references_from_graph(graph_path, metadata, 5) if graph_path else []
    if not urls:
        url_strings = metadata.get("node_maps", {}).get("url", {}).get("index_to_string", [])
        urls = [(u, 0) for u in url_strings[:5]]
    sections.append(_md_top_list("Top URLs", urls, "emails"))

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

    stem_pairs = get_top_stems_from_graph(graph_path, metadata, 5) if graph_path else []
    if not stem_pairs:
        stem_strings = metadata.get("node_maps", {}).get("stem", {}).get("index_to_string", [])
        stem_pairs = [(s, 0) for s in stem_strings[:5]]
    sections.append(_md_top_list("Top Stems", stem_pairs, "URLs"))

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

    recv_pairs = get_top_receivers_from_graph(graph_path, metadata, 5) if graph_path else []
    if not recv_pairs:
        receiver_strings = metadata.get("node_maps", {}).get("receiver", {}).get("index_to_string", [])
        recv_pairs = [(r, 0) for r in receiver_strings[:5]]
    sections.append(_md_top_list("Top Receivers", recv_pairs, "emails"))

    sections.append(_md_week_distribution(metadata, graph_path))

    comp_md_lines = ["## Connected Components", ""]
    if graph_path:
        try:
            import torch
            graph = torch.load(graph_path, weights_only=False)

            node_types = list(graph.node_types)
            inferred_counts: Dict[str, int] = {nt: 0 for nt in node_types}
            for (src_t, rel, dst_t) in getattr(graph, "edge_types", []):
                store = graph[src_t, rel, dst_t]
                edge_index = getattr(store, "edge_index", None)
                if edge_index is None:
                    continue
                if edge_index.numel() == 0:
                    continue
                src_max = int(edge_index[0].max().item()) if edge_index[0].numel() > 0 else -1
                dst_max = int(edge_index[1].max().item()) if edge_index[1].numel() > 0 else -1
                if src_max >= 0:
                    inferred_counts[src_t] = max(inferred_counts[src_t], src_max + 1)
                if dst_max >= 0:
                    inferred_counts[dst_t] = max(inferred_counts[dst_t], dst_max + 1)
            for nt in node_types:
                try:
                    if "x" in graph[nt]:
                        inferred_counts[nt] = max(inferred_counts[nt], int(graph[nt].x.size(0)))
                except Exception:
                    pass

            offsets: Dict[str, int] = {}
            off = 0
            for nt in node_types:
                offsets[nt] = off
                off += int(inferred_counts.get(nt, 0))
            total = off

            if total == 0:
                comp_md_lines.append("No nodes present to compute components.")
            else:
                adj: List[List[int]] = [[] for _ in range(total)]
                for src_t, rel, dst_t in getattr(graph, "edge_types", []):
                    store = graph[src_t, rel, dst_t]
                    edge_index = getattr(store, "edge_index", None)
                    if edge_index is None or edge_index.numel() == 0:
                        continue
                    bs = offsets[src_t]
                    bd = offsets[dst_t]
                    src_idx = edge_index[0].tolist()
                    dst_idx = edge_index[1].tolist()
                    for s_i, d_i in zip(src_idx, dst_idx):
                        s_g = bs + int(s_i)
                        d_g = bd + int(d_i)
                        if 0 <= s_g < total and 0 <= d_g < total:
                            adj[s_g].append(d_g)
                            adj[d_g].append(s_g)

                visited = [False] * total
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

    maxdeg_md = ["## Max-Degree Node in Largest Component", ""]
    info = get_max_degree_node_in_largest_component(graph_path, metadata) if graph_path else None
    if info is None:
        maxdeg_md.append("No data available (graph missing or empty).")
    else:
        maxdeg_md.append(f"- Node type: {info['node_type']}")
        maxdeg_md.append(f"- Local index: {info['local_index']}")
        maxdeg_md.append(f"- Global index: {info['global_index']}")
        maxdeg_md.append(f"- Degree: {info['degree']}")
        maxdeg_md.append(f"- Component size: {info['component_size']}")
        if info.get("label"):
            maxdeg_md.append(f"- Label: {info['label']}")
    maxdeg_md.append("")
    sections.append("\n".join(maxdeg_md))

    samples_md = ["## Random Node Feature Samples", ""]
    sample_data = md_random_node_features_sample(meta_path, graph_path)
    if sample_data is None:
        samples_md.append("No data available for node feature samples.")
    else:
        samples_md.append(sample_data)
    sections.append("\n".join(samples_md))

    out_dir = os.path.join("results")
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(meta_path))[0] or "graph"
    out_path = os.path.join(out_dir, f"{base}_analysis.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(sections))

    print(f"\nSaved analysis report: {out_path}\n")


if __name__ == "__main__":
    import sys
    import os
    
    default_meta = os.path.join("../results", "TREC-07-misp_hetero.meta.json")
    default_graph = os.path.join("../results", "TREC-07-misp_hetero.pt")
    
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
