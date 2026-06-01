import json
import os

from core.graph.assembler import assemble_misp_graph_ir
from core.graph.graph_schema import DEFAULT_SCHEMA


def test_email_x_text_vectors_present_and_nonzero():
    data_path = os.path.join("data", "misp", "trec07_misp.json")
    assert os.path.exists(data_path), "Expected sample MISP JSON at data/misp/trec07_misp.json"
    with open(data_path, "r", encoding="utf-8") as f:
        events = json.load(f)

    ir = assemble_misp_graph_ir(events, schema=DEFAULT_SCHEMA)

    x_text = ir.email_attrs.get("x_text")
    assert isinstance(x_text, list)
    if x_text:                    
        n_emails = len(ir.nodes["email"].index_to_meta or [])
        assert len(x_text) == n_emails
        assert len(x_text[0]) > 0
        assert any(sum(abs(v) for v in row) > 0.0 for row in x_text)
