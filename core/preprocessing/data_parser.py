from __future__ import annotations

import ast
import csv
import ipaddress
import json
import logging
import os
import re
import sys
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from pathlib import Path
from typing import Any, Dict, List, Tuple
from preprocessing.body_parser import extract_body_html_css_without_headers
from preprocessing.attachment_parser import extract_attachment_hashes_from_email
from preprocessing.html_css_parser import parse_css_fast, parse_html_fast

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
    "List-Unsubscribe",
    "Authentication-Results",
    "X-Forefront-Antispam-Report",
    "X-MS-Exchange-Organization-SCL",
]

_FOREFRONT_ALLOWED_KEYS = {"CIP", "CTRY", "LANG", "SCL", "SFV", "CAT", "BCL", "PCL"}
_HOSTNAME_RE = re.compile(r"\b[a-z0-9][a-z0-9-]*(?:\.[a-z0-9-]+)+\b", re.IGNORECASE)
_IPV4_RE = re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b")
_TRACKING_QUERY_KEY_RE = re.compile(
    r"^(utm_.+|mc_.+|trk.*|fbclid|gclid|msclkid|cid|icid|mkt_tok|campaign|source|medium|content|"
    r"token|auth|signature|sig|hash|userid|user_id|uid)$",
    re.IGNORECASE,
)
_TOKENISH_SEGMENT_RE = re.compile(r"^[a-z0-9_-]{20,}$", re.IGNORECASE)
_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$",
    re.IGNORECASE,
)


def _default_header_value(field: str) -> Any:
    if field == "Received":
        return []
    if field == "Return-Path":
        return {"email": "", "domain": ""}
    return ""


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


def _extract_selected_headers(headers: Dict[str, Any]) -> Dict[str, Any]:
    lower_map = {str(k).strip().lower(): v for k, v in headers.items()}
    selected: Dict[str, Any] = {}
    for field in HEADER_FIELDS:
        value = lower_map.get(field.lower(), _default_header_value(field))
        if field == "Received":
            value = _filter_received_header(value)
        elif field == "Return-Path":
            value = _normalize_return_path(value)
        elif field == "Content-Type":
            value = _filter_content_type(value)
        elif field == "Received-SPF":
            value = _filter_received_spf(value)
        elif field == "List-Unsubscribe":
            value = _filter_list_unsubscribe(value)
        elif field == "Authentication-Results":
            value = _filter_authentication_results(value)
        elif field == "X-Forefront-Antispam-Report":
            value = _filter_forefront_antispam_report(value)
        else:
            value = _normalize_scalar(value)
        if value in ("", None):
            value = _default_header_value(field)
        selected[field] = value
    return selected


def _filter_forefront_antispam_report(value: str) -> str:
    text = _normalize_scalar(value)
    if not text:
        return ""

    kept_parts: List[str] = []
    for raw_part in text.split(";"):
        part = raw_part.strip()
        if not part:
            continue

        separator = ":"
        if ":" in part:
            key, raw_val = part.split(":", 1)
            separator = ":"
        elif "=" in part:
            key, raw_val = part.split("=", 1)
            separator = "="
        else:
            continue

        normalized_key = _normalize_scalar(key).upper()
        if normalized_key not in _FOREFRONT_ALLOWED_KEYS:
            continue

        normalized_val = _normalize_scalar(raw_val)
        kept_parts.append(f"{normalized_key}{separator}{normalized_val}")

    return "; ".join(kept_parts)


def _split_header_segments(value: Any) -> List[str]:
    if isinstance(value, list):
        segments: List[str] = []
        for item in value:
            item_text = _normalize_scalar(item)
            if not item_text:
                continue
            segments.extend([segment.strip() for segment in item_text.split("|") if segment.strip()])
        return segments
    text = _normalize_scalar(value)
    return [segment.strip() for segment in text.split("|") if segment.strip()]


def _normalize_whitespace(value: str) -> str:
    return re.sub(r"\s+", " ", _normalize_scalar(value)).strip()


def _contains_private_or_loopback_ip(value: str) -> bool:
    for raw_ip in _IPV4_RE.findall(value):
        try:
            parsed = ipaddress.ip_address(raw_ip)
        except ValueError:
            continue
        if parsed.is_private or parsed.is_loopback or parsed.is_link_local:
            return True
    return False


def _is_corporate_internal_hop(received_hop: str) -> bool:
    hop = _normalize_whitespace(received_hop).lower()
    if not hop:
        return False
    if _contains_private_or_loopback_ip(hop):
        return True

    for hostname in _HOSTNAME_RE.findall(hop):
        host = hostname.lower()
        if host.endswith((".local", ".lan", ".corp", ".internal")):
            return True

    for marker in (" from ", " by "):
        idx = hop.find(marker)
        if idx == -1:
            continue
        token = hop[idx + len(marker):].split(" ", 1)[0].strip("()[];")
        if token and "." not in token and token not in {"localhost"}:
            return True

    return " localhost" in hop or hop.startswith("localhost")


def _received_hop_fingerprint(received_hop: str) -> str:
    hop = _normalize_whitespace(received_hop).lower()
    if not hop:
        return ""
    parts: List[str] = []
    for label in ("from", "by", "with"):
        match = re.search(rf"\b{label}\s+([^\s;()]+)", hop)
        if match:
            parts.append(f"{label}={match.group(1)}")
    return "|".join(parts) if parts else hop


def _parse_received_hop(received_hop: str) -> Dict[str, str]:
    hop = _normalize_whitespace(received_hop)
    if not hop:
        return {"origin_ip": "", "helo_host": "", "by_host": "", "timestamp": ""}

    helo_host = ""
    by_host = ""
    origin_ip = ""
    timestamp = ""

    from_match = re.search(r"\bfrom\s+([^\s;()]+)", hop, flags=re.IGNORECASE)
    if from_match:
        helo_host = from_match.group(1).strip("()[];")

    by_match = re.search(r"\bby\s+([^\s;()]+)", hop, flags=re.IGNORECASE)
    if by_match:
        by_host = by_match.group(1).strip("()[];")

    bracketed_ip_match = re.search(r"\[(\d{1,3}(?:\.\d{1,3}){3})\]", hop)
    if bracketed_ip_match:
        origin_ip = bracketed_ip_match.group(1)
    else:
        ipv4_match = _IPV4_RE.search(hop)
        if ipv4_match:
            origin_ip = ipv4_match.group(0)

    if ";" in hop:
        timestamp = hop.split(";", 1)[1].strip()

    return {
        "origin_ip": origin_ip,
        "helo_host": helo_host.lower(),
        "by_host": by_host.lower(),
        "timestamp": timestamp,
    }


def _filter_received_header(value: Any) -> List[Dict[str, str]]:
    hops = _split_header_segments(value)
    if not hops:
        return []

    kept: List[Dict[str, str]] = []
    seen: set[str] = set()
    for raw_hop in hops:
        hop = _normalize_whitespace(raw_hop)
        if not hop:
            continue
        lowered = hop.lower()
        if _is_corporate_internal_hop(lowered):
            continue

        fingerprint = _received_hop_fingerprint(lowered)
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        kept.append(_parse_received_hop(hop))

    return kept


def _normalize_return_path(value: Any) -> Dict[str, str]:
    if isinstance(value, dict):
        email_value = _normalize_scalar(value.get("email", "")).lower().strip().strip("<>").strip()
        domain_value = _normalize_scalar(value.get("domain", "")).lower().strip()
        if not domain_value and "@" in email_value:
            domain_value = email_value.rsplit("@", 1)[1]
        return {"email": email_value, "domain": domain_value}

    text = _normalize_scalar(value).lower().strip()
    if not text:
        return {"email": "", "domain": ""}
    text = text.strip("<>").strip()
    if not text:
        return {"email": "", "domain": ""}

    if "|" in text:
        text = text.split("|", 1)[0].strip()

    email_match = re.search(r"([a-z0-9._%+\-]+@([a-z0-9.\-]+\.[a-z]{2,}))", text)
    if email_match:
        return {"email": email_match.group(1), "domain": email_match.group(2)}

    return {"email": text, "domain": ""}


def _filter_content_type(value: str) -> str:
    text = _normalize_scalar(value)
    if not text:
        return ""

    parts = [part.strip() for part in text.split(";") if part.strip()]
    if not parts:
        return ""
    kept_parts = [parts[0].lower()]
    for raw_part in parts[1:]:
        if "=" not in raw_part:
            kept_parts.append(raw_part)
            continue
        key, raw_val = raw_part.split("=", 1)
        normalized_key = key.strip().lower()
        if normalized_key in {"boundary", "charset"}:
            continue
        kept_parts.append(f"{normalized_key}={raw_val.strip()}")
    return "; ".join(kept_parts)


def _filter_received_spf(value: str) -> str:
    text = _normalize_scalar(value)
    if not text:
        return ""

    compact_parts: List[str] = []
    verdict_match = re.match(r"\s*([a-z-]+)\b", text, flags=re.IGNORECASE)
    if verdict_match:
        compact_parts.append(f"spf={verdict_match.group(1).lower()}")

    domain_match = re.search(r"domain of\s+([^\s;()]+)", text, flags=re.IGNORECASE)
    if domain_match:
        compact_parts.append(f"domain={domain_match.group(1).lower()}")

    allowed_keys = {"client-ip", "helo", "envelope-from", "mailfrom", "identity"}
    for raw_part in text.split(";"):
        part = raw_part.strip()
        if "=" not in part:
            continue
        key, raw_val = part.split("=", 1)
        normalized_key = key.strip().lower()
        if normalized_key not in allowed_keys:
            continue
        normalized_val = _normalize_scalar(raw_val).lower()
        if normalized_val:
            compact_parts.append(f"{normalized_key}={normalized_val}")

    deduped: List[str] = []
    seen: set[str] = set()
    for part in compact_parts:
        if part in seen:
            continue
        seen.add(part)
        deduped.append(part)
    return "; ".join(deduped)


def _is_token_like_text(value: str) -> bool:
    normalized = _normalize_scalar(value).strip()
    if not normalized:
        return False
    if _UUID_RE.match(normalized):
        return True
    return bool(_TOKENISH_SEGMENT_RE.match(normalized))


def _filter_query_params(pairs: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    kept: List[Tuple[str, str]] = []
    for key, val in pairs:
        normalized_key = _normalize_scalar(key).lower()
        normalized_val = _normalize_scalar(val)
        if _TRACKING_QUERY_KEY_RE.match(normalized_key):
            continue
        if "token" in normalized_key or "auth" in normalized_key:
            continue
        if _is_token_like_text(normalized_val):
            continue
        kept.append((key, val))
    return kept


def _strip_tokenized_path_segments(path: str) -> str:
    segments = path.split("/")
    cleaned_segments = [
        segment
        for segment in segments
        if segment == "" or not _is_token_like_text(segment)
    ]
    cleaned_path = "/".join(cleaned_segments)
    return cleaned_path or "/"


def _sanitize_unsubscribe_target(value: str) -> str:
    target = _normalize_scalar(value).strip().strip("<>").strip()
    if not target:
        return ""

    if target.lower().startswith(("http://", "https://", "mailto:")):
        parsed = urlsplit(target)
        filtered_pairs = _filter_query_params(parse_qsl(parsed.query, keep_blank_values=True))
        sanitized_path = _strip_tokenized_path_segments(parsed.path)
        sanitized_query = urlencode(filtered_pairs, doseq=True)
        return urlunsplit((parsed.scheme, parsed.netloc, sanitized_path, sanitized_query, ""))

    return target


def _filter_list_unsubscribe(value: str) -> str:
    text = _normalize_scalar(value)
    if not text:
        return ""

    extracted = re.findall(r"<([^>]+)>", text)
    raw_targets = extracted if extracted else [part.strip() for part in text.split(",")]
    sanitized_targets = [
        sanitized
        for sanitized in (_sanitize_unsubscribe_target(target) for target in raw_targets)
        if sanitized
    ]
    return " | ".join(sanitized_targets)


def _filter_authentication_results(value: str) -> str:
    text = _normalize_scalar(value)
    if not text:
        return ""

    compact_parts: List[str] = []
    for label in ("spf", "dkim", "dmarc"):
        match = re.search(rf"\b{label}\s*=\s*([a-z0-9_-]+)", text, flags=re.IGNORECASE)
        if match:
            compact_parts.append(f"{label}={match.group(1).lower()}")

    header_from_match = re.search(r"\bheader\.from=([^\s;]+)", text, flags=re.IGNORECASE)
    if header_from_match:
        compact_parts.append(f"header.from={header_from_match.group(1).lower()}")

    return "; ".join(compact_parts)


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


def _extract_headers_from_mailparser(parsed_mail: Any) -> Dict[str, Any]:
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


def _parse_body_and_headers_with_mailparser(
    raw_bytes: bytes,
) -> Tuple[str, Dict[str, Any], Dict[str, Any], List[str], Dict[str, Any]]:
    if not raw_bytes:
        return (
            "",
            {"tag_counts": {}, "tree_stats": {}, "structure_fingerprint": ""},
            {"style_features": {}},
            [],
            {field: _default_header_value(field) for field in HEADER_FIELDS},
        )

    body_text, html_text, css_text = extract_body_html_css_without_headers(raw_bytes)
    html_structure = parse_html_fast(html_text)
    css_structure = parse_css_fast(css_text)
    attachment_hashes = extract_attachment_hashes_from_email(raw_bytes)
    if mailparser is None:
        return body_text, html_structure, css_structure, attachment_hashes, {field: _default_header_value(field) for field in HEADER_FIELDS}

    try:
        _configure_mailparser_logging()
        parsed_mail = mailparser.parse_from_bytes(raw_bytes)
        headers = _extract_headers_from_mailparser(parsed_mail)
        return body_text, html_structure, css_structure, attachment_hashes, headers
    except Exception:
        # Keep body extracted from raw RFC email; fail closed on headers only.
        return body_text, html_structure, css_structure, attachment_hashes, {field: _default_header_value(field) for field in HEADER_FIELDS}


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


def parse_incidents_with_email_bodies(
    incidents_csv_path: str,
    bodies_dir: str,
    limit: int | None = None,
) -> List[Dict[str, Any]]:
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

    if limit is not None and limit <= 0:
        return []

    with incidents_path.open("r", encoding="utf-8-sig", errors="replace", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        for row_idx, row in enumerate(reader):
            external_id = _normalize_scalar(row.get("external_id"))
            properties = _try_parse_mapping(row.get("properties", ""))

            body_file = body_files_by_name.get(external_id) or body_files_by_stem.get(external_id) if external_id else None
            if body_file is None:
                # Skip emails that have no matching file in bodies_dir.
                continue

            body_text = ""
            html_text: Dict[str, Any] = {"tag_counts": {}, "tree_stats": {}, "structure_fingerprint": ""}
            css_text: Dict[str, Any] = {"style_features": {}}
            attachment_hashes: List[str] = []
            parser_headers = {field: _default_header_value(field) for field in HEADER_FIELDS}
            raw_body_bytes = _read_body_file_bytes(body_file)
            body_text, html_text, css_text, attachment_hashes, parser_headers = _parse_body_and_headers_with_mailparser(raw_body_bytes)

            headers_from_properties = _extract_selected_headers(_try_parse_mapping(properties.get("emailHeaders")))
            headers_raw = _normalize_scalar(row.get("emailHeaders"))
            csv_headers = _extract_selected_headers(_try_parse_email_headers(headers_raw))
            selected_headers = {
                field: (
                    headers_from_properties.get(field, _default_header_value(field))
                    or csv_headers.get(field, _default_header_value(field))
                    or parser_headers.get(field, _default_header_value(field))
                )
                for field in HEADER_FIELDS
            }

            incident: Dict[str, Any] = {
                "record_index": row_idx,
                "external_id": external_id,
                "email_body": body_text,
                "email_html": html_text,
                "email_css": css_text,
                "email_attachments": attachment_hashes,
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

            if limit is not None and len(parsed_incidents) >= limit:
                break

    return parsed_incidents


__all__ = ["parse_incidents_with_email_bodies", "INCIDENT_FIELDS", "HEADER_FIELDS"]
