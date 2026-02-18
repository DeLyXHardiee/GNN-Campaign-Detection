from __future__ import annotations

import ast
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple


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

            body_text = ""
            if external_id:
                body_file = body_files_by_name.get(external_id) or body_files_by_stem.get(external_id)
                if body_file is not None:
                    body_text = _decode_email_body(_read_body_file_bytes(body_file))

            headers_raw = _normalize_scalar(row.get("emailHeaders"))
            selected_headers = _extract_selected_headers(_try_parse_email_headers(headers_raw))

            incident: Dict[str, Any] = {
                "record_index": row_idx,
                "external_id": external_id,
                "email_body": body_text,
                "email_headers": selected_headers,
            }

            for field in INCIDENT_FIELDS:
                raw_val = row.get(field, "")
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
