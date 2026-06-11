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

    for node in DEFAULT_SCHEMA.nodes.values():
        assert node.pyg in graph.node_types

    for edge in DEFAULT_SCHEMA.edges.values():
        et = (DEFAULT_SCHEMA.nodes[edge.src].pyg, edge.rel_pyg, DEFAULT_SCHEMA.nodes[edge.dst].pyg)
        key = f"{DEFAULT_SCHEMA.nodes[edge.src].pyg}->{DEFAULT_SCHEMA.nodes[edge.dst].pyg}:{edge.rel_pyg}"
        count = metadata["edge_counts"].get(key, 0)
        if count > 0:
            assert et in graph.edge_types
