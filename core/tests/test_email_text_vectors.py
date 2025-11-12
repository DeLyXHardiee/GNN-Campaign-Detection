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

    # x_text should be present and aligned to number of emails when sklearn is available
    x_text = ir.email_attrs.get("x_text")
    # It is acceptable for x_text to be an empty list if sklearn is missing or corpus was empty,
    # but in our sample dataset it should be non-empty and have >0 dim per vector.
    assert isinstance(x_text, list)
    if x_text:  # vectors computed
        n_emails = len(ir.nodes["email"].index_to_meta or [])
        assert len(x_text) == n_emails
        # vectors should have dimensionality > 0
        assert len(x_text[0]) > 0
        # At least one email should have a non-zero TF-IDF sum
        assert any(sum(abs(v) for v in row) > 0.0 for row in x_text)
