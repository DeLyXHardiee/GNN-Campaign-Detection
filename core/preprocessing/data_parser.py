from __future__ import annotations

import ast
import csv
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

try:
    import mailparser  # type: ignore
except Exception:
    mailparser = None

_MAILPARSER_LOG_CONFIGURED = False


INCIDENT_FIELDS = [
    "subject",
    "category",
    "date_sent",
    "rfc_defects",
    "cyrillic_domain",
    "contains_symbols",
    "body_has_tracking_url",
    "body_has_tracking_image",
    "body_has_tracking_pixel",
    "body_has_unsubscribe_link",
    "domain_is_common_webprovided",
]

HEADER_FIELDS = [
    "To",
    "From",
    "Received",
    "Return-Path",
    "Content-Type",
    "Received-SPF",
    "DKIM-Signature",
    "List-Unsubscribe",
    "Authentication-Results",
]


def _normalize_scalar(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, dict)):
        try:
            return json.dumps(value, ensure_ascii=False, sort_keys=True).strip()
        except Exception:
            return str(value).strip()
    return str(value).strip()


def _normalize_bool_like(value: Any) -> str:
    normalized = _normalize_scalar(value).lower()
    if normalized in {"1", "true", "yes", "y"}:
        return "true"
    if normalized in {"0", "false", "no", "n"}:
        return "false"
    return _normalize_scalar(value)


def _try_parse_email_headers(raw_headers: str) -> Dict[str, Any]:
    if not raw_headers or not raw_headers.strip():
        return {}

    try:
        parsed = json.loads(raw_headers)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    try:
        parsed = ast.literal_eval(raw_headers)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    return {}


def _try_parse_mapping(raw_value: Any) -> Dict[str, Any]:
    if isinstance(raw_value, dict):
        return raw_value
    if raw_value is None:
        return {}

    raw_text = _normalize_scalar(raw_value)
    if not raw_text:
        return {}

    try:
        parsed = json.loads(raw_text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    try:
        parsed = ast.literal_eval(raw_text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    return {}


def _extract_selected_headers(headers: Dict[str, Any]) -> Dict[str, str]:
    lower_map = {str(k).strip().lower(): _normalize_scalar(v) for k, v in headers.items()}
    selected: Dict[str, str] = {}
    for field in HEADER_FIELDS:
        selected[field] = lower_map.get(field.lower(), "")
    return selected


def _decode_email_body(raw_bytes: bytes) -> str:
    # Keep this as passive byte-to-text decoding only; no parsing/execution.
    try:
        return raw_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return raw_bytes.decode("latin-1", errors="replace")


def _normalize_header_value(value: Any) -> str:
    if isinstance(value, list):
        return " | ".join(_normalize_scalar(v) for v in value if _normalize_scalar(v))
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return _normalize_scalar(value)


def _extract_headers_from_mailparser(parsed_mail: Any) -> Dict[str, str]:
    raw_headers = getattr(parsed_mail, "headers", None)
    collected: Dict[str, Any] = {}

    if isinstance(raw_headers, dict):
        collected.update(raw_headers)
    elif isinstance(raw_headers, list):
        for item in raw_headers:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                key = _normalize_scalar(item[0])
                val = item[1]
                if not key:
                    continue
                existing = collected.get(key)
                if existing is None:
                    collected[key] = val
                else:
                    existing_text = _normalize_header_value(existing)
                    new_text = _normalize_header_value(val)
                    collected[key] = " | ".join([x for x in [existing_text, new_text] if x])

    # Fill a few common fields from dedicated properties if headers dict lacks them.
    if "To" not in collected:
        collected["To"] = getattr(parsed_mail, "to", [])
    if "From" not in collected:
        collected["From"] = getattr(parsed_mail, "from_", [])
    if "Received" not in collected:
        collected["Received"] = getattr(parsed_mail, "received", [])
    if "Return-Path" not in collected:
        collected["Return-Path"] = getattr(parsed_mail, "return_path", "")
    if "Content-Type" not in collected:
        collected["Content-Type"] = getattr(parsed_mail, "content_type", "")
    if "Received-SPF" not in collected:
        collected["Received-SPF"] = getattr(parsed_mail, "received_spf", "")
    if "DKIM-Signature" not in collected:
        collected["DKIM-Signature"] = getattr(parsed_mail, "dkim_signature", "")
    if "List-Unsubscribe" not in collected:
        collected["List-Unsubscribe"] = getattr(parsed_mail, "list_unsubscribe", "")
    if "Authentication-Results" not in collected:
        collected["Authentication-Results"] = getattr(parsed_mail, "authentication_results", "")

    normalized: Dict[str, str] = {}
    for key, val in collected.items():
        k = _normalize_scalar(key)
        if not k:
            continue
        normalized[k] = _normalize_header_value(val)

    return _extract_selected_headers(normalized)


def _extract_body_from_mailparser(parsed_mail: Any, raw_bytes: bytes) -> str:
    text_plain = getattr(parsed_mail, "text_plain", None)
    if isinstance(text_plain, list):
        joined = "\n\n".join(_normalize_scalar(part) for part in text_plain if _normalize_scalar(part))
        if joined:
            return joined

    body = _normalize_scalar(getattr(parsed_mail, "body", ""))
    if body:
        return body

    return _decode_email_body(raw_bytes)


def _parse_body_and_headers_with_mailparser(raw_bytes: bytes) -> Tuple[str, Dict[str, str]]:
    if not raw_bytes:
        return "", {field: "" for field in HEADER_FIELDS}
    if mailparser is None:
        return _decode_email_body(raw_bytes), {field: "" for field in HEADER_FIELDS}

    try:
        _configure_mailparser_logging()
        parsed_mail = mailparser.parse_from_bytes(raw_bytes)
        body_text = _extract_body_from_mailparser(parsed_mail, raw_bytes)
        headers = _extract_headers_from_mailparser(parsed_mail)
        return body_text, headers
    except Exception:
        # Fail-closed: keep passive text fallback and empty selected headers.
        return _decode_email_body(raw_bytes), {field: "" for field in HEADER_FIELDS}


def _configure_mailparser_logging() -> None:
    global _MAILPARSER_LOG_CONFIGURED
    if _MAILPARSER_LOG_CONFIGURED:
        return

    # Silence verbose parser warnings ("calendar not handled", ambiguous matches)
    # while still allowing hard failures to be raised as exceptions.
    for logger_name in ("mailparser", "mailparser.mail", "mailparser.utils"):
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.ERROR)
        logger.propagate = False
    _MAILPARSER_LOG_CONFIGURED = True


def _read_body_file_bytes(path: Path) -> bytes:
    try:
        return path.read_bytes()
    except OSError:
        # Best-effort long-path fallback on Windows.
        if os.name == "nt":
            try:
                resolved = str(path.resolve())
                long_path = resolved if resolved.startswith("\\\\?\\") else f"\\\\?\\{resolved}"
                with open(long_path, "rb") as file_obj:
                    return file_obj.read()
            except OSError:
                return b""
        return b""


def _index_email_body_files(bodies_dir: Path) -> Tuple[Dict[str, Path], Dict[str, Path]]:
    by_filename: Dict[str, Path] = {}
    by_stem: Dict[str, Path] = {}

    for root, _, files in os.walk(bodies_dir):
        root_path = Path(root)
        for filename in files:
            path = root_path / filename
            by_filename.setdefault(filename, path)
            stem = path.stem
            if stem and stem not in by_stem:
                by_stem[stem] = path

    return by_filename, by_stem


def _get_value_from_row_or_properties(row: Dict[str, Any], properties: Dict[str, Any], key: str) -> Any:
    if key in properties and properties.get(key) is not None:
        return properties.get(key)
    return row.get(key, "")


def parse_incidents_with_email_bodies(incidents_csv_path: str, bodies_dir: str) -> List[Dict[str, Any]]:
    incidents_path = Path(incidents_csv_path)
    body_folder = Path(bodies_dir)

    if not incidents_path.exists():
        raise FileNotFoundError(f"Incidents CSV not found: {incidents_path}")
    if not body_folder.exists() or not body_folder.is_dir():
        raise FileNotFoundError(f"Email body folder not found: {body_folder}")

    body_files_by_name, body_files_by_stem = _index_email_body_files(body_folder)
    parsed_incidents: List[Dict[str, Any]] = []

    max_csv_field_size = sys.maxsize
    while True:
        try:
            csv.field_size_limit(max_csv_field_size)
            break
        except OverflowError:
            max_csv_field_size = max_csv_field_size // 10

    with incidents_path.open("r", encoding="utf-8-sig", errors="replace", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        for row_idx, row in enumerate(reader):
            external_id = _normalize_scalar(row.get("external_id"))
            properties = _try_parse_mapping(row.get("properties", ""))

            body_text = ""
            parser_headers = {field: "" for field in HEADER_FIELDS}
            if external_id:
                body_file = body_files_by_name.get(external_id) or body_files_by_stem.get(external_id)
                if body_file is not None:
                    raw_body_bytes = _read_body_file_bytes(body_file)
                    body_text, parser_headers = _parse_body_and_headers_with_mailparser(raw_body_bytes)

            headers_from_properties = _extract_selected_headers(_try_parse_mapping(properties.get("emailHeaders")))
            headers_raw = _normalize_scalar(row.get("emailHeaders"))
            csv_headers = _extract_selected_headers(_try_parse_email_headers(headers_raw))
            selected_headers = {
                field: headers_from_properties.get(field, "") or csv_headers.get(field, "") or parser_headers.get(field, "")
                for field in HEADER_FIELDS
            }

            incident: Dict[str, Any] = {
                "record_index": row_idx,
                "external_id": external_id,
                "email_body": body_text,
                "email_headers": selected_headers,
            }

            for field in INCIDENT_FIELDS:
                raw_val = _get_value_from_row_or_properties(row, properties, field)
                if field in {
                    "rfc_defects",
                    "cyrillic_domain",
                    "contains_symbols",
                    "body_has_tracking_url",
                    "body_has_tracking_image",
                    "body_has_tracking_pixel",
                    "body_has_unsubscribe_link",
                    "domain_is_common_webprovided",
                }:
                    incident[field] = _normalize_bool_like(raw_val)
                else:
                    incident[field] = _normalize_scalar(raw_val)

            parsed_incidents.append(incident)

    return parsed_incidents


__all__ = ["parse_incidents_with_email_bodies", "INCIDENT_FIELDS", "HEADER_FIELDS"]
