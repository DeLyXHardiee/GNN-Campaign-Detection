from typing import Dict, List, Tuple

import torch
from torch_geometric.data import HeteroData


def _summarize_degrees(deg: torch.Tensor, multi_threshold: int) -> Dict[str, float]:
    """Return degree bucket counts for a 1D degree tensor."""
    total = deg.numel()
    deg0 = int((deg == 0).sum())
    deg1 = int((deg == 1).sum())
    deg_multi = int((deg >= multi_threshold).sum())
    pct = lambda x: (x / total * 100.0) if total else 0.0  # noqa: E731
    return {
        "total_nodes": total,
        "deg0": deg0,
        "deg1": deg1,
        f"deg>={multi_threshold}": deg_multi,
        "pct_deg1": pct(deg1),
        f"pct_deg>={multi_threshold}": pct(deg_multi),
    }


def summarize_relation_connectivity(
    data: HeteroData, multi_threshold: int = 2
) -> Tuple[List[Dict[str, object]], Dict[str, Dict[str, float]]]:
    """
    For each relation, bucket nodes by degree (0, 1, >= multi_threshold).

    Returns:
        relation_rows: list of dicts with per-edge-type stats
        overall_node_stats: dict mapping node type -> degree bucket counts across all relations
    """
    relation_rows: List[Dict[str, object]] = []

    for edge_type in data.edge_types:
        src_type, rel, dst_type = edge_type
        edge_index = data[edge_type].edge_index
        src, dst = edge_index

        src_deg = torch.bincount(src, minlength=data[src_type].num_nodes)
        dst_deg = torch.bincount(dst, minlength=data[dst_type].num_nodes)

        relation_rows.append(
            {
                "edge_type": edge_type,
                "relation": rel,
                "num_edges": edge_index.size(1),
                "src_type": src_type,
                "dst_type": dst_type,
                "src_counts": _summarize_degrees(src_deg, multi_threshold),
                "dst_counts": _summarize_degrees(dst_deg, multi_threshold),
            }
        )

    # Aggregate degrees across all relations for each node type
    node_degrees = {
        ntype: torch.zeros(data[ntype].num_nodes, dtype=torch.long)
        for ntype in data.node_types
    }
    for edge_type in data.edge_types:
        src_type, _, dst_type = edge_type
        edge_index = data[edge_type].edge_index
        src, dst = edge_index
        node_degrees[src_type].index_add_(0, src, torch.ones_like(src))
        node_degrees[dst_type].index_add_(0, dst, torch.ones_like(dst))

    overall_node_stats = {
        ntype: _summarize_degrees(deg, multi_threshold)
        for ntype, deg in node_degrees.items()
    }

    return relation_rows, overall_node_stats


def print_connectivity_report(data: HeteroData, multi_threshold: int = 2) -> None:
    """
    Convenience wrapper to print per-relation connectivity stats to stdout.
    """
    relation_rows, overall_node_stats = summarize_relation_connectivity(
        data, multi_threshold=multi_threshold
    )

    print("=== Per-relation connectivity ===")
    for row in relation_rows:
        et = row["edge_type"]
        print(f"\nRelation: {et}  (|E|={row['num_edges']})")

        src_counts = row["src_counts"]
        dst_counts = row["dst_counts"]
        print(
            f"  src[{row['src_type']}]: {src_counts['total_nodes']} nodes "
            f"(this relation goes OUT from these nodes)"
        )
        print(
            f"    - no outgoing edges in this relation (deg0): {src_counts['deg0']}"
        )
        print(
            f"    - exactly 1 outgoing edge (deg1): {src_counts['deg1']} "
            f"({src_counts['pct_deg1']:.1f}%)"
        )
        print(
            f"    - {multi_threshold}+ outgoing edges (deg>={multi_threshold}): "
            f"{src_counts[f'deg>={multi_threshold}']} "
            f"({src_counts[f'pct_deg>={multi_threshold}']:.1f}%)"
        )

        print(
            f"  dst[{row['dst_type']}]: {dst_counts['total_nodes']} nodes "
            f"(this relation comes IN to these nodes)"
        )
        print(
            f"    - never used as destination (deg0): {dst_counts['deg0']}"
        )
        print(
            f"    - used by exactly 1 source (deg1): {dst_counts['deg1']} "
            f"({dst_counts['pct_deg1']:.1f}%)"
        )
        print(
            f"    - used by {multi_threshold}+ sources (deg>={multi_threshold}): "
            f"{dst_counts[f'deg>={multi_threshold}']} "
            f"({dst_counts[f'pct_deg>={multi_threshold}']:.1f}%)"
        )

    print("\n=== Node degrees aggregated across all relations ===")
    for ntype, counts in overall_node_stats.items():
        print(
            f"  {ntype}: "
            f"deg0={counts['deg0']}, "
            f"deg1={counts['deg1']} "
            f"( {counts['pct_deg1']:.1f}% ), "
            f"deg>={multi_threshold}={counts[f'deg>={multi_threshold}']} "
            f"( {counts[f'pct_deg>={multi_threshold}']:.1f}% ), "
            f"total={counts['total_nodes']}"
        )
