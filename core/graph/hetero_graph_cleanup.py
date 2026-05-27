"""Utilities for cleaning HeteroData artifacts after graph build or load."""

from __future__ import annotations

from typing import Any


def drop_inactive_hetero_node_types(data: Any) -> Any:
    """
    Remove node types that have no feature matrix and no non-empty incident edges.

    PyG creates empty ``NodeStorage`` entries when code touches ``data[ntype]`` for a
    type that was excluded at build time (e.g. via ``exclude_node_types``). Those
    placeholders break ``to_hetero`` because they never appear as edge destinations.
    """
    node_types = list(getattr(data, "node_types", []))
    edge_types = list(getattr(data, "edge_types", []))
    to_drop: list[str] = []

    for ntype in node_types:
        store = data[ntype]
        has_features = False
        if "x" in store:
            x = store.x
            has_features = x is not None and getattr(x, "numel", lambda: 0)() > 0

        if has_features:
            continue

        has_edges = False
        for src_t, _rel, dst_t in edge_types:
            if src_t != ntype and dst_t != ntype:
                continue
            ei = data[src_t, _rel, dst_t].edge_index
            if ei is not None and getattr(ei, "numel", lambda: 0)() > 0:
                has_edges = True
                break

        if not has_edges:
            to_drop.append(ntype)

    for ntype in to_drop:
        del data[ntype]

    return data
