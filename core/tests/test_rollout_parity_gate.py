from core.graph.assembler import assemble_misp_graph_ir
from core.graph.graph_schema import DEFAULT_SCHEMA


def _sample_events():
    return [
        {
            "Event": {
                "info": "parity-test",
                "email_index": 1,
                "Attribute": [
                    {"type": "from", "value": "alice@example.com"},
                    {"type": "to", "value": "bob@example.com"},
                    {"type": "url", "value": "https://example.com/path"},
                    {"type": "attachment", "value": ["h1", "h2"]},
                ],
            }
        }
    ]


def _shape_summary(ir):
    return {
        "nodes": {k: len(v.x) for k, v in ir.nodes.items()},
        "edges": {k: len(v[0]) for k, v in ir.edges.items()},
    }


def test_schema_pipeline_matches_legacy_pipeline(monkeypatch):
    events = _sample_events()

    monkeypatch.setenv("GRAPH_BUILDER_USE_SCHEMA_PIPELINE", "1")
    ir_schema = assemble_misp_graph_ir(events, schema=DEFAULT_SCHEMA)

    monkeypatch.setenv("GRAPH_BUILDER_USE_SCHEMA_PIPELINE", "0")
    ir_legacy = assemble_misp_graph_ir(events, schema=DEFAULT_SCHEMA)

    assert _shape_summary(ir_schema) == _shape_summary(ir_legacy)
