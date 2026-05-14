from core.graph.assembler import assemble_misp_graph_ir
from core.graph.graph_schema import DEFAULT_SCHEMA


def _events_with_two_urls():
    return [
        {
            "Event": {
                "info": "skip-superspreader-test",
                "email_index": 1,
                "Attribute": [
                    {"type": "from", "value": "alice@example.com"},
                    {"type": "to", "value": "bob@example.com"},
                    {"type": "url", "value": "https://keep.example/path"},
                    {"type": "url", "value": "https://drop.example/spreader-track-id-999/extra"},
                ],
            }
        }
    ]


def test_url_skip_superspreaders_removes_matching_url_nodes_only():
    events = _events_with_two_urls()
    ir_all = assemble_misp_graph_ir(events, schema=DEFAULT_SCHEMA, url_skip_substrings=[])
    ir_skip = assemble_misp_graph_ir(
        events, schema=DEFAULT_SCHEMA, url_skip_substrings=["spreader-track-id-999"]
    )

    assert len(ir_all.nodes["url"].index) == 2
    assert len(ir_skip.nodes["url"].index) == 1
    kept = (ir_skip.nodes["url"].index_to_string or [""])[0]
    assert "spreader-track-id-999" not in kept
    assert "keep.example" in kept

    assert "keep.example" in ir_skip.nodes["domain"].index
    assert "drop.example" in ir_skip.nodes["domain"].index
