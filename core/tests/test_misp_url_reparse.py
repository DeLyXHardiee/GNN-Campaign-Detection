"""Tests for in-place MISP JSON URL refresh."""

from core.preprocessing.utils.url_extractor import (
    collect_urls_for_misp_event_attributes,
    refresh_urls_in_misp_events,
)


def test_collect_urls_from_body_and_list_unsubscribe():
    attrs = [
        {"type": "body", "value": "see hxxps://evil.com/x?q=1 ok"},
        {"type": "header_List-Unsubscribe", "value": "<https://unsub.example.com/optout>"},
        {"type": "url", "value": ["old", "garbage"]},
    ]
    urls = collect_urls_for_misp_event_attributes(attrs)
    assert "https://evil.com/x?q=1" in urls
    assert "https://unsub.example.com/optout" in urls
    assert "garbage" not in urls


def test_collect_urls_skips_structured_html_dict():
    attrs = [
        {"type": "body", "value": "visit https://only-text.com/a"},
        {"type": "html", "value": {"tag_counts": {"td": 1}, "tree_stats": {}, "structure_fingerprint": "x"}},
    ]
    urls = collect_urls_for_misp_event_attributes(attrs)
    assert urls == ["https://only-text.com/a"]


def test_refresh_urls_inserts_url_attr_if_missing():
    events = [
        {
            "Event": {
                "Attribute": [
                    {"type": "body", "value": "visit https://new.example.com/a"},
                ],
            }
        }
    ]
    refresh_urls_in_misp_events(events)
    attrs = events[0]["Event"]["Attribute"]
    url_attr = next(a for a in attrs if a.get("type") == "url")
    assert url_attr["value"] == ["https://new.example.com/a"]
