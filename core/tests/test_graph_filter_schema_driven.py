from core.graph.assembler import GraphIR, NodeIR, assemble_misp_graph_ir
from core.graph.graph_filter import NodeType, filter_graph_ir, filter_graph_ir_by_degree
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


def test_degree_filter_strength_removes_more_high_degree_nodes():
    # email(0..2) -> received_host(0..2)
    # degrees(received_host): [1, 2, 4]
    ir = GraphIR(
        nodes={
            "email": NodeIR(index={}, x=[[0.0], [0.0], [0.0]], index_to_meta=[{}, {}, {}]),
            "received_host": NodeIR(
                index={"h1": 0, "h2": 1, "h3": 2},
                x=[[1.0], [1.0], [1.0]],
                index_to_string=["h1", "h2", "h3"],
                attrs={"docfreq": [1, 2, 3]},
            ),
        },
        edges={
            "has_received_host": (
                [0, 0, 1, 1, 2, 2, 2],  # email idx
                [0, 1, 1, 2, 2, 2, 2],  # received_host idx
            )
        },
        email_attrs={},
    )

    low = filter_graph_ir_by_degree(
        ir,
        schema=DEFAULT_SCHEMA,
        strength=0.2,
        target_node_types={"received_host"},
        min_degree=2,
    )
    high = filter_graph_ir_by_degree(
        ir,
        schema=DEFAULT_SCHEMA,
        strength=0.8,
        target_node_types={"received_host"},
        min_degree=2,
    )

    assert len(low.nodes["received_host"].x) == 2
    assert len(high.nodes["received_host"].x) == 1
    assert len(low.edges["has_received_host"][0]) > len(high.edges["has_received_host"][0])


def test_degree_filter_slices_email_attrs_when_emails_removed():
    # Three emails; email 0 links to three URLs -> degree 3; others link to one URL -> degree 1.
    # Low strength removes only the highest-degree email; email_attrs must stay aligned with nodes.
    ir = GraphIR(
        nodes={
            "email": NodeIR(index={}, x=[[0.0], [0.0], [0.0]], index_to_meta=[{}, {}, {}]),
            "url": NodeIR(
                index={},
                x=[[1.0], [1.0], [1.0]],
                index_to_string=["u0", "u1", "u2"],
            ),
        },
        edges={
            "has_url": (
                [0, 0, 0, 1, 2],
                [0, 1, 2, 0, 0],
            ),
        },
        email_attrs={
            "external_id": ["a", "b", "c"],
            "ts": [1.0, 2.0, 3.0],
        },
    )
    out = filter_graph_ir_by_degree(
        ir,
        schema=DEFAULT_SCHEMA,
        strength=0.2,
        target_node_types={"email"},
        min_degree=1,
    )
    assert len(out.nodes["email"].x) == 2
    assert out.email_attrs["external_id"] == ["b", "c"]
    assert out.email_attrs["ts"] == [2.0, 3.0]
    assert len(out.email_attrs["external_id"]) == len(out.nodes["email"].x)
