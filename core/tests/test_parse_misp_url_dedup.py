"""URL deduplication when parsing MISP events for the graph."""

from core.graph.common import parse_misp_events


def test_parse_misp_events_dedupes_defanged_and_plain_same_url():
    events = [
        {
            "Event": {
                "external_id": "e1",
                "email_index": 0,
                "Attribute": [
                    {
                        "type": "url",
                        "value": [
                            "hxxps://phish.evil.com/path?q=1",
                        ],
                    },
                    {
                        "type": "body",
                        "value": "Click https://phish.evil.com/path?q=1 for details",
                    },
                ],
            }
        }
    ]
    parsed = parse_misp_events(events)
    assert len(parsed) == 1
    urls = parsed[0]["urls"]
    assert len(urls) == 1
    assert urls[0].startswith("hxxps://")
    assert "phish.evil.com" in urls[0]


def test_parse_misp_events_trailing_slash_dedup_one_defanged():
    events = [
        {
            "Event": {
                "external_id": "e2",
                "email_index": 0,
                "Attribute": [
                    {"type": "url", "value": ["https://same.example.com"]},
                    {"type": "body", "value": "see https://same.example.com/ end"},
                ],
            }
        }
    ]
    parsed = parse_misp_events(events)
    urls = parsed[0]["urls"]
    assert len(urls) == 1
    assert "same.example.com" in urls[0]
    assert urls[0].startswith("hxxps://")
