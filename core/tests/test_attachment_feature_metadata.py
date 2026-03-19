from email.message import EmailMessage

from core.feature_set_extraction.feature_set_extraction import extract_attachment_features
from core.graph.common import parse_misp_events
from core.preprocessing.attachment_parser import extract_attachment_metadata_from_email


def test_extract_attachment_metadata_from_email_includes_size_and_type():
    msg = EmailMessage()
    msg["From"] = "alice@example.com"
    msg["To"] = "bob@example.com"
    msg["Subject"] = "Attachment test"
    msg.set_content("hello")

    payload = b"ABC123"
    msg.add_attachment(
        payload,
        maintype="application",
        subtype="pdf",
        filename="invoice.pdf",
    )

    metadata = extract_attachment_metadata_from_email(msg.as_bytes())
    assert len(metadata) == 1
    assert metadata[0]["size_bytes"] == len(payload)
    assert metadata[0]["content_type"] == "application/pdf"
    assert len(metadata[0]["sha256"]) == 64


def test_parse_misp_events_keeps_attachment_metadata_field():
    events = [
        {
            "Event": {
                "info": "test-attachment-meta",
                "email_index": 7,
                "external_id": "evt-7",
                "Attribute": [
                    {"type": "from", "value": "alice@example.com"},
                    {"type": "to", "value": "bob@example.com"},
                    {"type": "attachments", "value": ["hash-a"]},
                    {
                        "type": "attachments_meta",
                        "value": [
                            {
                                "sha256": "hash-a",
                                "size_bytes": 42,
                                "content_type": "application/pdf",
                            }
                        ],
                    },
                ],
            }
        }
    ]

    parsed = parse_misp_events(events)
    assert len(parsed) == 1
    assert parsed[0]["attachments"] == ["hash-a"]
    assert parsed[0]["attachment_metadata"] == [
        {"sha256": "hash-a", "size_bytes": 42, "content_type": "application/pdf"}
    ]


def test_extract_attachment_features_computes_size_list_and_type_features():
    features = extract_attachment_features(
        ["h1", "h2"],
        [
            {"sha256": "h1", "size_bytes": 10, "content_type": "application/pdf"},
            {"sha256": "h2", "size_bytes": 30, "content_type": "image/png"},
        ],
    )

    assert features["has_attachments"] == 1
    assert features["num_attachments"] == 2
    assert features["attachment_sizes_bytes"] == [10, 30]
    assert features["attachment_types"] == ["application/pdf", "image/png"]
    assert features["attachment_top_level_types"] == "application image"
