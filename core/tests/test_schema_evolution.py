from dataclasses import replace

import pytest

from core.graph.assembler import assemble_misp_graph_ir
from core.graph.graph_builder_pytorch import build_hetero_graph_from_misp
from core.graph.graph_schema import DEFAULT_SCHEMA, EdgeMapping, GraphSchema, NodeMapping, validate_schema


def _sample_events():
    return [
        {
            "Event": {
                "info": "schema-evolution",
                "email_index": 1,
                "Attribute": [
                    {"type": "from", "value": "alice@example.com"},
                    {"type": "to", "value": "bob@example.com"},
                    {"type": "attachment", "value": ["abc123"]},
                ],
            }
        }
    ]


def _schema_with_campaign() -> GraphSchema:
    nodes = dict(DEFAULT_SCHEMA.nodes)
    nodes["campaign"] = NodeMapping(
        canonical="campaign",
        pyg="campaign",
        memgraph="Campaign",
        memgraph_id_key="key",
        feature_strategy="str_len",
        extra_attr_keys=(),
    )
    edges = dict(DEFAULT_SCHEMA.edges)
    edges["has_campaign"] = EdgeMapping(
        canonical="has_campaign",
        src="email",
        rel_pyg="has_campaign",
        dst="campaign",
        memgraph_type="HAS_CAMPAIGN",
        memgraph_left_label="Email",
        memgraph_left_key="eid",
        memgraph_right_label="Campaign",
        memgraph_right_key="key",
        edge_strategy="email_to_entity",
    )
    schema = GraphSchema(nodes=nodes, edges=edges, collapse_rules=DEFAULT_SCHEMA.collapse_rules)
    validate_schema(schema)
    return schema


def test_custom_schema_adds_new_node_and_edge_without_assembler_changes():
    schema = _schema_with_campaign()
    ir = assemble_misp_graph_ir(_sample_events(), schema=schema)
    assert "campaign" in ir.nodes
    assert ir.nodes["campaign"].x == []
    assert "has_campaign" in ir.edges
    assert ir.edges["has_campaign"] == ([], [])


def test_collapse_rules_are_schema_driven():
    schema = replace(
        DEFAULT_SCHEMA,
        collapse_rules=DEFAULT_SCHEMA.collapse_rules + (("email", "attachment", "has_attachment"),),
    )
    ir = assemble_misp_graph_ir(_sample_events(), schema=schema)
    assert "attachment" in ir.nodes
    assert len(ir.nodes["attachment"].x) == 0


def test_pyg_builder_reflects_schema_nodes_and_edges():
    torch_geometric = pytest.importorskip("torch_geometric", reason="torch-geometric not installed in this env")
    _ = torch_geometric
    schema = _schema_with_campaign()
    graph, metadata = build_hetero_graph_from_misp(_sample_events(), schema=schema)
    assert "campaign" in graph.node_types
    assert "campaign" in metadata["node_maps"]
    key = "email->campaign:has_campaign"
    assert key in metadata["edge_counts"]
