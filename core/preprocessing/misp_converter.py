from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List


def _add_attr(attributes: List[Dict[str, Any]], attr_type: str, value: Any, category: str, relation: str = "") -> None:
    text = "" if value is None else str(value).strip()
    if not text:
        return

    attr: Dict[str, Any] = {
        "type": attr_type,
        "value": text,
        "category": category,
    }
    if relation:
        attr["object_relation"] = relation
    attributes.append(attr)


def incidents_to_misp_events(incidents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []

    for idx, incident in enumerate(incidents):
        headers = incident.get("email_headers", {}) or {}
        attributes: List[Dict[str, Any]] = []

        _add_attr(attributes, "email-src", headers.get("From", ""), "Payload delivery")
        _add_attr(attributes, "email-dst", headers.get("To", ""), "Payload delivery")
        _add_attr(attributes, "email-subject", incident.get("subject", ""), "Payload delivery")
        _add_attr(attributes, "email-date", incident.get("date_sent", ""), "Payload delivery")
        _add_attr(attributes, "email-body", incident.get("email_body", ""), "Payload delivery")

        for field in [
            "category",
            "rfc_defects",
            "cyrillic_domain",
            "contains_symbols",
            "body_has_tracking_url",
            "body_has_tracking_image",
            "body_has_tracking_pixel",
            "body_has_unsubscribe_link",
            "domain_is_common_webprovided",
        ]:
            _add_attr(attributes, "text", incident.get(field, ""), "External analysis", field)

        for header_key in [
            "Received",
            "Return-Path",
            "Content-Type",
            "Received-SPF",
            "DKIM-Signature",
            "List-Unsubscribe",
            "Authentication-Results",
        ]:
            _add_attr(
                attributes,
                "text",
                headers.get(header_key, ""),
                "External analysis",
                f"header_{header_key}",
            )

        event = {
            "Event": {
                "info": f"Incident {incident.get('external_id', idx)}",
                "email_index": int(incident.get("record_index", idx)),
                "external_id": incident.get("external_id", ""),
                "Attribute": attributes,
            }
        }
        events.append(event)

    return events


def write_misp_events_securely(misp_events: List[Dict[str, Any]], output_path: str) -> None:
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    temp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=str(destination.parent),
            delete=False,
            suffix=".tmp",
        ) as tmp:
            temp_path = tmp.name
            json.dump(misp_events, tmp, indent=2, ensure_ascii=False)
            tmp.flush()
            os.fsync(tmp.fileno())

        os.replace(temp_path, destination)
        try:
            os.chmod(destination, 0o600)
        except Exception:
            # Best effort; permission models vary across operating systems.
            pass
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


def incidents_to_misp_file(incidents: List[Dict[str, Any]], output_path: str) -> List[Dict[str, Any]]:
    misp_events = incidents_to_misp_events(incidents)
    write_misp_events_securely(misp_events, output_path)
    return misp_events


__all__ = ["incidents_to_misp_events", "write_misp_events_securely", "incidents_to_misp_file"]
