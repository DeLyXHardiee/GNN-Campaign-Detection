from core.graph.assembler import assemble_misp_graph_ir
from core.graph.graph_schema import DEFAULT_SCHEMA


def test_attachment_nodes_are_unique_and_linked_to_emails():
    events = [
        {
            "Event": {
                "info": "att-1",
                "email_index": 1,
                "Attribute": [
                    {"type": "from", "value": "alice@example.com"},
                    {"type": "to", "value": "bob@example.com"},
                    {"type": "attachment", "value": ["AAA", "bbb", "aaa"]},
                ],
            }
        },
        {
            "Event": {
                "info": "att-2",
                "email_index": 2,
                "Attribute": [
                    {"type": "from", "value": "carol@example.com"},
                    {"type": "to", "value": "dave@example.com"},
                    {"type": "attachment", "value": ["aaa", "ccc"]},
                ],
            }
        },
    ]

    ir = assemble_misp_graph_ir(events, schema=DEFAULT_SCHEMA)

    attachment_node = ir.nodes["attachment"]
    attachments = attachment_node.index_to_string or []
    assert set(attachments) == {"aaa", "bbb", "ccc"}

    src, dst = ir.edges["has_attachment"]
    assert len(src) == 4
    assert len(dst) == 4
