

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

COMMON_HTML_TAGS = {
    "a", "abbr", "address", "area", "article", "aside", "audio",
    "b", "base", "bdi", "bdo", "blockquote", "body", "br", "button",
    "canvas", "caption", "cite", "code", "col", "colgroup",
    "data", "datalist", "dd", "del", "details", "dfn", "dialog", "div", "dl", "dt",
    "em", "embed",
    "fieldset", "figcaption", "figure", "footer", "form",
    "h1", "h2", "h3", "h4", "h5", "h6", "head", "header", "hgroup", "hr", "html",
    "i", "iframe", "img", "input", "ins",
    "kbd",
    "label", "legend", "li", "link",
    "main", "map", "mark", "menu", "meta", "meter",
    "nav", "noscript",
    "object", "ol", "optgroup", "option", "output",
    "p", "param", "picture", "pre", "progress",
    "q",
    "rp", "rt", "ruby",
    "s", "samp", "script", "search", "section", "select", "slot", "small", "source", "span", "strong", "style", "sub", "summary", "sup",
    "table", "tbody", "td", "template", "textarea", "tfoot", "th", "thead", "time", "title", "tr", "track",
    "u", "ul",
    "var", "video",
    "wbr",
}

def _iter_misp_events(payload: Any) -> Iterable[Dict[str, Any]]:
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict):
                if "Event" in item and isinstance(item["Event"], dict):
                    yield item["Event"]
                elif "Attribute" in item:
                    yield item
        return

    if isinstance(payload, dict):
        if "Event" in payload:
            event = payload["Event"]
            if isinstance(event, list):
                for e in event:
                    if isinstance(e, dict):
                        yield e
            elif isinstance(event, dict):
                yield event
            return

        for key in ("response", "events"):
            items = payload.get(key)
            if isinstance(items, list):
                for item in items:
                    if isinstance(item, dict):
                        if "Event" in item and isinstance(item["Event"], dict):
                            yield item["Event"]
                        elif "Attribute" in item:
                            yield item


def _filter_tag_counts(tag_counts: Any) -> Tuple[Any, int]:
    if not isinstance(tag_counts, dict):
        return tag_counts, 0

    filtered = {k: v for k, v in tag_counts.items() if str(k).lower() in COMMON_HTML_TAGS}
    removed = len(tag_counts) - len(filtered)
    return filtered, removed


def _sanitize_html_value(value: Any) -> Tuple[Any, int, int]:
    """Return (new_value, removed_tag_count, html_objects_touched)."""
    if not isinstance(value, dict):
        return value, 0, 0

    tag_counts = value.get("tag_counts")
    if not isinstance(tag_counts, dict):
        return value, 0, 1

    filtered, removed = _filter_tag_counts(tag_counts)
    if removed > 0:
        value = dict(value)
        value["tag_counts"] = filtered

    return value, removed, 1


def sanitize_payload(payload: Any) -> Dict[str, int]:
    stats = {
        "events_seen": 0,
        "attributes_seen": 0,
        "direct_records_seen": 0,
        "html_objects_seen": 0,
        "tags_removed": 0,
        "attributes_updated": 0,
        "direct_records_updated": 0,
    }

    if isinstance(payload, list):
        for record in payload:
            if not (isinstance(record, dict) and record.get("external_id")):
                continue
            stats["direct_records_seen"] += 1
            if "html" in record:
                new_html, removed, touched = _sanitize_html_value(record.get("html"))
                stats["html_objects_seen"] += touched
                stats["tags_removed"] += removed
                if removed > 0:
                    record["html"] = new_html
                    stats["direct_records_updated"] += 1

    for event in _iter_misp_events(payload):
        stats["events_seen"] += 1
        attributes = event.get("Attribute", [])
        if not isinstance(attributes, list):
            continue

        for attr in attributes:
            if not isinstance(attr, dict):
                continue
            stats["attributes_seen"] += 1
            if attr.get("type") != "html":
                continue

            new_value, removed, touched = _sanitize_html_value(attr.get("value"))
            stats["html_objects_seen"] += touched
            stats["tags_removed"] += removed
            if removed > 0:
                attr["value"] = new_value
                stats["attributes_updated"] += 1

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Sanitize MISP html.tag_counts by dropping non-common HTML tags "
            "in Event/Attribute payloads (or direct-record datasets)."
        )
    )
    parser.add_argument("--misp-path", required=True, help="Path to input MISP JSON file")
    parser.add_argument(
        "--output-path",
        default=None,
        help="Path to write sanitized JSON. If omitted, updates input file in place.",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="When writing in place, do not create a .bak backup file.",
    )
    args = parser.parse_args()



    in_path = Path(args.misp_path)
    if not in_path.exists() or not in_path.is_file():
        raise FileNotFoundError(f"MISP JSON file not found: {in_path}")

    with in_path.open("r", encoding="utf-8-sig") as f:
        payload = json.load(f)

    stats = sanitize_payload(payload)

    out_path = Path(args.output_path) if args.output_path else in_path

    if out_path == in_path and not args.no_backup:
        backup_path = in_path.with_suffix(in_path.suffix + ".bak")
        backup_path.write_text(in_path.read_text(encoding="utf-8-sig"), encoding="utf-8-sig")
        print(f"Backup created: {backup_path}")

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("Sanitization complete")
    print(f"Input: {in_path}")
    print(f"Output: {out_path}")
    print(f"Events seen: {stats['events_seen']}")
    print(f"Attributes seen: {stats['attributes_seen']}")
    print(f"Direct records seen: {stats['direct_records_seen']}")
    print(f"HTML objects seen: {stats['html_objects_seen']}")
    print(f"Attributes updated: {stats['attributes_updated']}")
    print(f"Direct records updated: {stats['direct_records_updated']}")
    print(f"Tag entries removed: {stats['tags_removed']}")


if __name__ == "__main__":
    main()
