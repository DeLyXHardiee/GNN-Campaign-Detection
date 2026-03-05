from core.graph.common import parse_misp_events


def test_parse_misp_events_splits_from_and_to_lists():
    events = [
        {
            "Event": {
                "info": "list-address-test",
                "email_index": 1,
                "Attribute": [
                    {
                        "type": "from",
                        "value": ["Alice <alice@example.com>", "bob@example.org"],
                    },
                    {
                        "type": "to",
                        "value": ["carol@example.net", "Dave <dave@example.io>"],
                    },
                ],
            }
        }
    ]

    parsed = parse_misp_events(events)
    assert len(parsed) == 1
    row = parsed[0]

    assert row["senders"] == ["alice@example.com", "bob@example.org"]
    assert "sender" not in row
    assert row["receivers"] == ["carol@example.net", "dave@example.io"]
