import json
import os
import pytest

from core.graph.graph_schema import DEFAULT_SCHEMA


torch_geometric = pytest.importorskip("torch_geometric", reason="torch-geometric not installed in this env")
from core.graph.graph_builder_pytorch import build_hetero_graph_from_misp


def test_pyg_uses_shared_schema():
    data_path = os.path.join("data", "misp", "trec07_misp.json")
    assert os.path.exists(data_path), "Expected sample MISP JSON at data/misp/trec07_misp.json"
    with open(data_path, "r", encoding="utf-8") as f:
        events = json.load(f)

    graph, metadata = build_hetero_graph_from_misp(events, schema=DEFAULT_SCHEMA)

    # All node types from schema should exist in the graph (possibly with 0 nodes)
    for node in DEFAULT_SCHEMA.nodes.values():
        assert node.pyg in graph.node_types

    # All edge types from schema should be recognized, but may be missing if there are no edges
    for edge in DEFAULT_SCHEMA.edges.values():
        et = (DEFAULT_SCHEMA.nodes[edge.src].pyg, edge.rel_pyg, DEFAULT_SCHEMA.nodes[edge.dst].pyg)
        # If there are edges, the type must exist
        # Find a safe proxy from metadata using our formatted keys
        key = f"{DEFAULT_SCHEMA.nodes[edge.src].pyg}->{DEFAULT_SCHEMA.nodes[edge.dst].pyg}:{edge.rel_pyg}"
        count = metadata["edge_counts"].get(key, 0)
        if count > 0:
            assert et in graph.edge_types
