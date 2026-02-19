from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List


def _clean_text(value: Any) -> str:
    text = "" if value is None else str(value)
    # Replace invalid Unicode surrogate code points to avoid JSON encoding failures.
    return text.encode("utf-8", errors="replace").decode("utf-8").strip()


def _sanitize_structure(value: Any) -> Any:
    if isinstance(value, dict):
        return {(_clean_text(k) if isinstance(k, str) else k): _sanitize_structure(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitize_structure(v) for v in value]
    if isinstance(value, str):
        return _clean_text(value)
    return value


def _add_attr(attributes: List[Dict[str, Any]], attr_type: str, value: Any) -> None:
    text = _clean_text(value)

    attr: Dict[str, Any] = {
        "type": attr_type,
        "value": text,
    }
    attributes.append(attr)


def incidents_to_misp_events(incidents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []

    for idx, incident in enumerate(incidents):
        headers = incident.get("email_headers", {}) or {}
        attributes: List[Dict[str, Any]] = []

        _add_attr(attributes, "from", headers.get("From", ""))
        _add_attr(attributes, "to", headers.get("To", ""))
        _add_attr(attributes, "subject", incident.get("subject", ""))
        _add_attr(attributes, "date", incident.get("date_sent", ""))
        _add_attr(attributes, "body", incident.get("email_body", ""))
        _add_attr(attributes, "html", incident.get("email_html", ""))
        _add_attr(attributes, "date_sent", incident.get("date_sent", ""))

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
            _add_attr(attributes, field, incident.get(field, ""))

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
                f"header_{header_key}",
                headers.get(header_key, ""),
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
        sanitized_events = _sanitize_structure(misp_events)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=str(destination.parent),
            delete=False,
            suffix=".tmp",
        ) as tmp:
            temp_path = tmp.name
            json.dump(sanitized_events, tmp, indent=2, ensure_ascii=False)
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
