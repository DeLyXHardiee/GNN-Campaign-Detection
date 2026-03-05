from core.graph.assembler import assemble_misp_graph_ir
from core.graph.graph_filter import NodeType, filter_graph_ir
from core.graph.graph_schema import DEFAULT_SCHEMA


def _sample_events():
    return [
        {
            "Event": {
                "info": "filter-test",
                "email_index": 1,
                "Attribute": [
                    {"type": "from", "value": "alice@example.com"},
                    {"type": "to", "value": "bob@example.com"},
                    {"type": "attachment", "value": ["h1"]},
                    {"type": "url", "value": "https://example.com/path"},
                ],
            }
        }
    ]


def test_filter_accepts_string_node_names_and_prunes_edges():
    ir = assemble_misp_graph_ir(_sample_events(), schema=DEFAULT_SCHEMA)
    filtered = filter_graph_ir(
        ir,
        exclude_nodes=NodeType.canonical_set(["attachment", "url"], schema=DEFAULT_SCHEMA),
        schema=DEFAULT_SCHEMA,
    )
    assert "attachment" not in filtered.nodes
    assert "url" not in filtered.nodes
    assert "has_attachment" not in filtered.edges
    assert "has_url" not in filtered.edges
