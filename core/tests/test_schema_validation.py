from dataclasses import replace

import pytest

from core.graph.graph_schema import DEFAULT_SCHEMA, EdgeMapping, GraphSchema, validate_schema


def test_default_schema_is_valid():
    validate_schema(DEFAULT_SCHEMA)


def test_schema_validation_rejects_unknown_edge_endpoints():
    bad_edges = dict(DEFAULT_SCHEMA.edges)
    bad_edges["has_sender"] = replace(bad_edges["has_sender"], src="unknown_node")
    bad_schema = GraphSchema(
        nodes=DEFAULT_SCHEMA.nodes,
        edges=bad_edges,
        collapse_rules=DEFAULT_SCHEMA.collapse_rules,
    )
    with pytest.raises(ValueError, match="unknown src node"):
        validate_schema(bad_schema)


def test_schema_validation_rejects_mismatched_memgraph_labels():
    bad_edges = dict(DEFAULT_SCHEMA.edges)
    bad_edges["has_sender"] = replace(
        bad_edges["has_sender"],
        memgraph_right_label="WrongLabel",
    )
    bad_schema = GraphSchema(
        nodes=DEFAULT_SCHEMA.nodes,
        edges=bad_edges,
        collapse_rules=DEFAULT_SCHEMA.collapse_rules,
    )
    with pytest.raises(ValueError, match="right label"):
        validate_schema(bad_schema)
