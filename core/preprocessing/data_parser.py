from __future__ import annotations

import ast
import base64
import csv
import ipaddress
import json
import logging
import os
import re
import sys
from datetime import date, datetime, time, timedelta, timezone
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from pathlib import Path
from typing import Any, Collection, Dict, Iterator, List, Optional, Set, Tuple
from lake.client import LakeAPIClient
from preprocessing.body_parser import (
    extract_body_html_css_without_headers,
    extract_css_text_from_html,
)
from preprocessing.attachment_parser import extract_attachment_metadata_from_email
from preprocessing.html_css_parser import get_empty_html_structure, parse_css_fast, parse_html_fast
from preprocessing.utils.url_extractor import (
    deduplicate_urls,
    extract_urls_from_plain_and_html,
    extract_urls_from_text,
)

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
    sample_id: Optional[str] = None,
) -> Tuple[str, Dict[str, Any], Dict[str, Any], List[str], List[Dict[str, Any]], Dict[str, Any], str]:
    """Extract body, HTML/CSS structure, attachments, headers, and raw HTML for URL extraction.

    Never raises: on body/HTML parse failure returns empty body and empty structure, and logs.

    Returns:
        Tuple of ``(body_text, html_structure, css_structure, attachment_hashes,
        attachment_metadata, headers, html_raw)`` where ``html_raw`` is the defanged HTML
        string before structural parsing (empty if none).
    """
    empty_html = get_empty_html_structure()
    default_headers = {field: _default_header_value(field) for field in HEADER_FIELDS}
    if not raw_bytes:
        return (
            "",
            empty_html,
            {"style_features": {}},
            [],
            [],
            default_headers,
            "",
        )

    html_raw = ""
    try:
        body_text, html_text, css_text = extract_body_html_css_without_headers(raw_bytes)
        html_raw = html_text or ""
        html_structure = parse_html_fast(html_text, sample_id=sample_id)
        css_structure = parse_css_fast(css_text)
        attachment_metadata = extract_attachment_metadata_from_email(raw_bytes)
        attachment_hashes = [
            str(item.get("sha256", "")).strip()
            for item in attachment_metadata
            if str(item.get("sha256", "")).strip()
        ]
    except Exception as e:
        logging.warning(
            "Body/HTML extraction or parsing failed for sample %s: %s",
            sample_id if sample_id is not None else "(unknown)",
            e,
            exc_info=False,
        )
        body_text = ""
        html_raw = ""
        html_structure = get_empty_html_structure()
        css_structure = {"style_features": {}}
        attachment_hashes = []
        attachment_metadata = []

    if mailparser is None:
        return (
            body_text,
            html_structure,
            css_structure,
            attachment_hashes,
            attachment_metadata,
            default_headers,
            html_raw,
        )

    try:
        _configure_mailparser_logging()
        parsed_mail = mailparser.parse_from_bytes(raw_bytes)
        headers = _extract_headers_from_mailparser(parsed_mail)
        return (
            body_text,
            html_structure,
            css_structure,
            attachment_hashes,
            attachment_metadata,
            headers,
            html_raw,
        )
    except Exception:
        # Keep body extracted from raw RFC email; fail closed on headers only.
        return (
            body_text,
            html_structure,
            css_structure,
            attachment_hashes,
            attachment_metadata,
            default_headers,
            html_raw,
        )


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


def _normalize_category_allowlist(allowed_categories: Optional[Collection[str]]) -> Optional[Set[str]]:
    if not allowed_categories:
        return None
    normalized = {_normalize_scalar(c).lower() for c in allowed_categories if _normalize_scalar(c)}
    return normalized or None


def parse_incidents_with_email_bodies(
    incidents_csv_path: str,
    bodies_dir: str,
    limit: int | None = None,
    *,
    allowed_categories: Optional[Collection[str]] = None,
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
        limit = None

    category_allow = _normalize_category_allowlist(allowed_categories)

    with incidents_path.open("r", encoding="utf-8-sig", errors="replace", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        for row_idx, row in enumerate(reader):
            external_id = _normalize_scalar(row.get("external_id"))
            properties = _try_parse_mapping(row.get("properties", ""))

            if category_allow is not None:
                cat_raw = _get_value_from_row_or_properties(row, properties, "category")
                if _normalize_scalar(cat_raw).lower() not in category_allow:
                    continue

            body_file = body_files_by_name.get(external_id) or body_files_by_stem.get(external_id) if external_id else None
            if body_file is None:
                # Skip emails that have no matching file in bodies_dir.
                continue

            body_text = ""
            html_text: Dict[str, Any] = {"tag_counts": {}, "tree_stats": {}, "structure_fingerprint": ""}
            css_text: Dict[str, Any] = {"style_features": {}}
            attachment_hashes: List[str] = []
            attachment_metadata: List[Dict[str, Any]] = []
            parser_headers = {field: _default_header_value(field) for field in HEADER_FIELDS}
            raw_body_bytes = _read_body_file_bytes(body_file)
            body_text, html_text, css_text, attachment_hashes, attachment_metadata, parser_headers, html_raw = _parse_body_and_headers_with_mailparser(
                raw_body_bytes, sample_id=external_id
            )

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

            urls_from_body = extract_urls_from_plain_and_html(body_text, html_raw)
            list_unsub_str = _normalize_header_value(selected_headers.get("List-Unsubscribe", ""))
            urls_from_headers = extract_urls_from_text(list_unsub_str)
            email_urls = deduplicate_urls(urls_from_body + urls_from_headers)

            incident: Dict[str, Any] = {
                "record_index": row_idx,
                "external_id": external_id,
                "email_body": body_text,
                "email_html": html_text,
                "email_css": css_text,
                "email_attachments": attachment_hashes,
                "email_attachment_metadata": attachment_metadata,
                "email_headers": selected_headers,
                "email_urls": email_urls,
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


def _value_from_stream_row(row: Dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in row:
            return row.get(key)
    for key in keys:
        leaf_key = key.split(".")[-1]
        if leaf_key in row:
            return row.get(leaf_key)
    return ""


def _nested_get(data: Any, *keys: str) -> Any:
    current: Any = data
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _extract_analysis_mapping(row: Dict[str, Any], properties: Dict[str, Any]) -> Dict[str, Any]:
    analysis_raw = _value_from_stream_row(row, "analysis", "i.analysis")
    analysis_map = _try_parse_mapping(analysis_raw)
    if analysis_map:
        return analysis_map
    # Fallback for legacy payloads where analysis is embedded in properties.
    embedded = _try_parse_mapping(properties.get("analysis"))
    return embedded if embedded else {}


def _extract_incident_category(
    row: Dict[str, Any],
    properties: Dict[str, Any],
    analysis_map: Dict[str, Any],
) -> str:
    category_candidate = (
        analysis_map.get("category")
        or properties.get("category")
        or _value_from_stream_row(row, "category", "i.category")
    )
    normalized = _normalize_scalar(category_candidate).lower()
    if normalized:
        return normalized

    analysis_text = _normalize_scalar(_value_from_stream_row(row, "analysis", "i.analysis")).lower()
    return analysis_text if analysis_text in {"phishing", "scam"} else ""


# Smaller pages + two-stage join (see _iterate_lake_incident_rows_stream) reduce peak memory for large ``parsed`` JSON.
_DEFAULT_LAKE_PAGE_SIZE = 5000
_LAKE_DATE_ONLY_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _lake_to_naive_utc(dt: datetime) -> datetime:
    if dt.tzinfo is not None:
        return dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


def _lake_parse_timeframe_bound(value: Any) -> Optional[tuple[datetime, bool]]:
    """Parse ``preprocessing_lake`` start/end config. Returns (naive UTC datetime, is_date_only) or None."""
    if value is None:
        return None
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        if _LAKE_DATE_ONLY_RE.match(s):
            d = date.fromisoformat(s)
            return (datetime.combine(d, time.min), True)
        s2 = s[:-1] + "+00:00" if s.endswith("Z") else s
        try:
            dt = datetime.fromisoformat(s2)
        except ValueError as exc:
            raise ValueError(f"Invalid lake timeframe date/datetime string: {s!r}") from exc
        return (_lake_to_naive_utc(dt), False)
    if isinstance(value, date) and not isinstance(value, datetime):
        return (datetime.combine(value, time.min), True)
    if isinstance(value, datetime):
        return (_lake_to_naive_utc(value), False)
    raise TypeError(
        f"Lake timeframe value must be str, date, datetime, or null; got {type(value).__name__}."
    )


def _lake_created_at_filter_sql(start: Any, end: Any) -> str:
    """SQL ``AND`` lines for ``i.created_at`` (empty if both bounds unset). Date-only end is end-exclusive."""
    parts: list[str] = []
    start_parsed = _lake_parse_timeframe_bound(start)
    end_parsed = _lake_parse_timeframe_bound(end)
    if start_parsed is not None:
        dt, _ = start_parsed
        lit = dt.strftime("%Y-%m-%d %H:%M:%S")
        parts.append(f"i.created_at >= TIMESTAMP '{lit}'")
    if end_parsed is not None:
        dt, is_date_only = end_parsed
        if is_date_only:
            exclusive = datetime.combine(dt.date() + timedelta(days=1), time.min)
            lit = exclusive.strftime("%Y-%m-%d %H:%M:%S")
            parts.append(f"i.created_at < TIMESTAMP '{lit}'")
        else:
            lit = dt.strftime("%Y-%m-%d %H:%M:%S")
            parts.append(f"i.created_at <= TIMESTAMP '{lit}'")
    if not parts:
        return ""
    return "\n    AND " + "\n    AND ".join(parts)


def _lake_incident_join_page_sql(
    incidents_table: str,
    parsed_emails_table: str,
    created_at_filters: str,
    batch: int,
    offset: int,
) -> str:
    """SQL for one page of the two-stage incidents × parsed_emails join (shared by query and stream paths)."""
    inner_sql = f"""
SELECT
    i.external_id,
    i.type,
    i.analysis,
    i.severity,
    i.properties,
    i.title,
    i.created_at,
    sk.source_id
FROM {incidents_table} i
INNER JOIN (
    SELECT source_id FROM {parsed_emails_table}
) sk ON i.external_id = sk.source_id
WHERE
    LOWER(TRIM(CAST(i.type AS VARCHAR))) = 'phishing'
    AND NULLIF(TRIM(CAST(i.severity AS VARCHAR)), '') IS NOT NULL{created_at_filters}
ORDER BY i.created_at ASC, i.external_id ASC
LIMIT {int(batch)} OFFSET {int(offset)}
""".strip()

    return f"""
SELECT
    b.external_id,
    b.type,
    b.analysis,
    b.severity,
    b.properties,
    b.title,
    b.created_at,
    b.source_id,
    pe.parsed
FROM (
{inner_sql}
) AS b
INNER JOIN {parsed_emails_table} pe ON b.source_id = pe.source_id
""".strip()


def _iterate_lake_incident_rows_query(
    client: LakeAPIClient,
    incidents_table: str,
    parsed_emails_table: str,
    *,
    page_size: int = _DEFAULT_LAKE_PAGE_SIZE,
    sql_limit: Optional[int] = None,
    start_date: Any = None,
    end_date: Any = None,
) -> Iterator[Dict[str, Any]]:
    """Fetch joined incidents via :meth:`LakeAPIClient.query` (JSON ``/query``) in stable, bounded batches.

    Same pagination and row semantics as :func:`_iterate_lake_incident_rows_stream`; use that path for
    large results to avoid buffering full pages as JSON.
    """
    created_at_filters = _lake_created_at_filter_sql(start_date, end_date)
    offset = 0
    while True:
        batch = page_size
        if sql_limit is not None:
            remaining = sql_limit - offset
            if remaining <= 0:
                break
            batch = min(batch, remaining)

        page_sql = _lake_incident_join_page_sql(
            incidents_table, parsed_emails_table, created_at_filters, batch, offset
        )
        # API ``limit`` must not truncate the page; SQL already applies LIMIT/OFFSET.
        rows = client.query(page_sql, limit=max(batch, 1))
        n = 0
        for row in rows:
            if isinstance(row, dict):
                yield row
                n += 1
        if n == 0:
            break
        if n < batch:
            break
        offset += n
        if sql_limit is not None and offset >= sql_limit:
            break


def _iterate_lake_incident_rows_stream(
    client: LakeAPIClient,
    incidents_table: str,
    parsed_emails_table: str,
    *,
    page_size: int = _DEFAULT_LAKE_PAGE_SIZE,
    sql_limit: Optional[int] = None,
    start_date: Any = None,
    end_date: Any = None,
) -> Iterator[Dict[str, Any]]:
    """Fetch joined incidents via :meth:`LakeAPIClient.query_stream` (``/query/stream``) in stable, bounded batches.

    Memory / IO:

    - Pagination uses a **narrow** join: ``incidents`` × ``(SELECT source_id FROM parsed_emails)``
      so engines can **column-prune** the heavy ``parsed`` payload until the final join.
    - Only after ``LIMIT``/``OFFSET`` is applied do we join the full ``parsed_emails`` row for
      that page, so each request materializes at most ``page_size`` large JSON blobs.

    Each page is read as Arrow record batches, converted to the same ``dict`` row shape as the JSON
    ``/query`` path via ``RecordBatch.to_pylist()``.

    Ordering is ``i.created_at ASC, i.external_id ASC`` (stable, time-ordered pages).

    ``sql_limit`` caps total rows read from the lake (same role as the SQL ``LIMIT`` in the inner query), not
    filtered incident count.

    Optional ``start_date`` / ``end_date`` restrict ``i.created_at`` (see :func:`_lake_created_at_filter_sql`).
    """
    created_at_filters = _lake_created_at_filter_sql(start_date, end_date)
    offset = 0
    while True:
        batch = page_size
        if sql_limit is not None:
            remaining = sql_limit - offset
            if remaining <= 0:
                break
            batch = min(batch, remaining)

        page_sql = _lake_incident_join_page_sql(
            incidents_table, parsed_emails_table, created_at_filters, batch, offset
        )

        n = 0
        for arrow_batch in client.query_stream(page_sql):
            for row in arrow_batch.to_pylist():
                if isinstance(row, dict):
                    yield row
                    n += 1
        if n == 0:
            break
        if n < batch:
            break
        offset += n
        if sql_limit is not None and offset >= sql_limit:
            break


def _first_non_empty_str(data: Dict[str, Any], *keys: str) -> str:
    for key in keys:
        val = data.get(key)
        if val is None:
            continue
        if isinstance(val, str) and val.strip():
            return val
        if isinstance(val, (bytes, bytearray)):
            try:
                s = bytes(val).decode("utf-8", errors="replace")
            except Exception:
                continue
            if s.strip():
                return s
    return ""


def _lake_raw_email_bytes(parsed_payload: Dict[str, Any]) -> bytes:
    """Recover RFC822 bytes if the lake row stores them (matches CSV pipeline when present)."""
    for b64_key in ("raw_email_base64", "rfc822_base64", "raw_bytes_base64"):
        raw = parsed_payload.get(b64_key)
        if isinstance(raw, str) and raw.strip():
            try:
                return base64.b64decode(raw, validate=False)
            except Exception:
                continue
    raw = parsed_payload.get("raw_email") or parsed_payload.get("rfc822")
    if isinstance(raw, (bytes, bytearray)):
        return bytes(raw)
    if isinstance(raw, str) and raw.strip():
        return raw.encode("utf-8", errors="replace")
    return b""


def _lake_resolve_raw_html_css_strings(parsed_payload: Dict[str, Any]) -> Tuple[str, str]:
    """Best-effort raw HTML and CSS strings from a lake ``parsed`` payload."""
    raw_b = _lake_raw_email_bytes(parsed_payload)
    if raw_b:
        try:
            _, html_str, css_str = extract_body_html_css_without_headers(raw_b)
            return (html_str or "").strip(), (css_str or "").strip()
        except Exception:
            pass

    html_raw = _first_non_empty_str(
        parsed_payload,
        "html_body",
        "body_html",
        "html",
        "raw_html",
        "html_selectolax_cleaned",
        "html_cleaned",
    )
    enrichment = parsed_payload.get("enrichment")
    if not html_raw and isinstance(enrichment, dict):
        html_raw = _first_non_empty_str(
            enrichment,
            "html_body",
            "body_html",
            "html",
            "raw_html",
        )

    if not html_raw:
        body_only = parsed_payload.get("body")
        if isinstance(body_only, str) and body_only.strip():
            if re.search(r"<\s*[a-zA-Z!?/]", body_only):
                html_raw = body_only

    css_raw = _first_non_empty_str(parsed_payload, "css", "css_text", "inline_css")
    if not css_raw and isinstance(enrichment, dict):
        css_raw = _first_non_empty_str(enrichment, "css", "css_text", "inline_css")

    if not css_raw and html_raw:
        css_raw = extract_css_text_from_html(html_raw)

    return (html_raw or "").strip(), (css_raw or "").strip()


def _lake_html_css_structures_from_parsed_payload(
    parsed_payload: Dict[str, Any],
    sample_id: Optional[str] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any], str]:
    """Build ``email_html`` / ``email_css`` feature dicts like :func:`_parse_body_and_headers_with_mailparser`.

    Lake rows expose structured ``parsed`` JSON instead of raw body files; we try (in order):

    1. Full raw message bytes, if present → same extraction as CSV.
    2. HTML/CSS strings from common column names (and optional ``enrichment`` nesting).

    Returns:
        ``(html_structure, css_structure, html_raw)`` for reuse in URL extraction.
    """
    empty_html = get_empty_html_structure()
    empty_css: Dict[str, Any] = {"style_features": {}}

    html_raw, css_raw = _lake_resolve_raw_html_css_strings(parsed_payload)
    if not html_raw and not css_raw:
        return empty_html, empty_css, html_raw

    html_structure = parse_html_fast(html_raw, sample_id=sample_id) if html_raw.strip() else empty_html
    css_structure = parse_css_fast(css_raw) if css_raw.strip() else empty_css
    return html_structure, css_structure if css_structure else empty_css, html_raw


def _parse_incidents_from_lake_rows(
    rows: Iterator[Dict[str, Any]],
    *,
    limit: int | None,
    allowed_categories: Optional[Collection[str]],
) -> List[Dict[str, Any]]:
    """Build incident dicts from lake join rows (shared by query and stream fetchers)."""
    category_allow = _normalize_category_allowlist(allowed_categories) or {"phishing", "scam"}

    parsed_incidents: List[Dict[str, Any]] = []
    row_index = 0

    for row in rows:
        row_index += 1
        external_id = _normalize_scalar(_value_from_stream_row(row, "external_id", "i.external_id"))
        properties = _try_parse_mapping(_value_from_stream_row(row, "properties", "i.properties"))
        analysis_map = _extract_analysis_mapping(row, properties)
        incident_category = _extract_incident_category(row, properties, analysis_map)
        if incident_category not in category_allow:
            continue

        severity = _normalize_scalar(
            analysis_map.get("severity") or _value_from_stream_row(row, "severity", "i.severity")
        ).lower()
        if not severity:
            continue

        parsed_payload_raw = _value_from_stream_row(row, "parsed", "s.parsed")
        parsed_payload = (
            parsed_payload_raw
            if isinstance(parsed_payload_raw, dict)
            else _try_parse_mapping(parsed_payload_raw)
        )
        parsed_payload = parsed_payload if isinstance(parsed_payload, dict) else {}

        body_text = _normalize_scalar(
            parsed_payload.get("body_selectolax_cleaned") or parsed_payload.get("body")
        )
        html_text, css_text, html_raw_lake = _lake_html_css_structures_from_parsed_payload(
            parsed_payload,
            sample_id=external_id or None,
        )

        attachments_raw = parsed_payload.get("attachments")
        attachment_metadata: List[Dict[str, Any]] = (
            [a for a in attachments_raw if isinstance(a, dict)]
            if isinstance(attachments_raw, list)
            else []
        )
        attachment_hashes = [
            _normalize_scalar(a.get("content_hash_sha256"))
            for a in attachment_metadata
            if _normalize_scalar(a.get("content_hash_sha256"))
        ]

        headers_from_properties = _extract_selected_headers(_try_parse_mapping(properties.get("emailHeaders")))
        parsed_headers = parsed_payload.get("headers")
        row_headers = _extract_selected_headers(parsed_headers if isinstance(parsed_headers, dict) else {})
        selected_headers = {
            field: (
                headers_from_properties.get(field, _default_header_value(field))
                or row_headers.get(field, _default_header_value(field))
            )
            for field in HEADER_FIELDS
        }

        urls_from_body = extract_urls_from_plain_and_html(body_text, html_raw_lake)
        parsed_urls: List[str] = []
        body_analysis_urls = _nested_get(parsed_payload, "enrichment", "body_analysis", "urls")
        if isinstance(body_analysis_urls, list):
            for url_item in body_analysis_urls:
                if isinstance(url_item, dict):
                    url_text = _normalize_scalar(url_item.get("url"))
                    if url_text:
                        parsed_urls.append(url_text)
        list_unsub_str = _normalize_header_value(selected_headers.get("List-Unsubscribe", ""))
        urls_from_headers = extract_urls_from_text(list_unsub_str)
        email_urls = deduplicate_urls(parsed_urls + urls_from_body + urls_from_headers)

        incident: Dict[str, Any] = {
            "record_index": row_index - 1,
            "external_id": external_id,
            "email_body": body_text,
            "email_html": html_text,
            "email_css": css_text,
            "email_attachments": attachment_hashes,
            "email_attachment_metadata": attachment_metadata,
            "email_headers": selected_headers,
            "email_urls": email_urls,
            "subject": _normalize_scalar(
                parsed_payload.get("subject")
                or _value_from_stream_row(row, "title", "i.title")
            ),
            "category": incident_category,
            "date_sent": _normalize_scalar(
                parsed_payload.get("date_sent")
                or parsed_payload.get("date")
                or _value_from_stream_row(row, "created_at", "i.created_at")
            ),
            "rfc_defects": _normalize_bool_like(
                bool(_nested_get(parsed_payload, "enrichment", "rfc_compliance", "defects"))
            ),
            "cyrillic_domain": _normalize_bool_like(
                _nested_get(parsed_payload, "enrichment", "sender_profile", "sender_domain_cyrillic")
            ),
            "contains_symbols": _normalize_bool_like(
                _nested_get(parsed_payload, "enrichment", "sender_profile", "sender_domain_has_symbols")
            ),
            "body_has_tracking_url": _normalize_bool_like(
                _nested_get(parsed_payload, "enrichment", "body_analysis", "body_has_tracking_url")
            ),
            "body_has_tracking_image": _normalize_bool_like(
                _nested_get(parsed_payload, "enrichment", "body_analysis", "body_has_tracking_image")
            ),
            "body_has_tracking_pixel": _normalize_bool_like(
                _nested_get(parsed_payload, "enrichment", "body_analysis", "body_has_tracking_pixel")
            ),
            "body_has_unsubscribe_link": _normalize_bool_like(
                _nested_get(parsed_payload, "enrichment", "body_analysis", "body_has_unsubscribe_link")
            ),
            "domain_is_common_webprovided": _normalize_bool_like(
                _nested_get(parsed_payload, "enrichment", "sender_profile", "domain_is_common_webmail")
            ),
        }

        parsed_incidents.append(incident)

        if limit is not None and limit > 0 and len(parsed_incidents) >= limit:
            return parsed_incidents

    return parsed_incidents


def parse_incidents_from_lake_query(
    *,
    base_url: str,
    api_key: str,
    incidents_table: str = "intellagent.public.incidents",
    parsed_emails_table: str = "parsed_emails",
    limit: int | None = None,
    allowed_categories: Optional[Collection[str]] = None,
    start_date: Any = None,
    end_date: Any = None,
) -> List[Dict[str, Any]]:
    """Parse incidents using :meth:`LakeAPIClient.query` (JSON). Prefer :func:`parse_incidents_from_lake_stream` for large pulls."""
    if not base_url.strip():
        raise ValueError("base_url must be provided.")
    if not api_key.strip():
        raise ValueError("api_key must be provided.")

    sql_limit = limit if limit is not None and limit > 0 else None
    client = LakeAPIClient(base_url=base_url, api_key=api_key)
    return _parse_incidents_from_lake_rows(
        _iterate_lake_incident_rows_query(
            client,
            incidents_table=incidents_table,
            parsed_emails_table=parsed_emails_table,
            sql_limit=sql_limit,
            start_date=start_date,
            end_date=end_date,
        ),
        limit=limit,
        allowed_categories=allowed_categories,
    )


def parse_incidents_from_lake_stream(
    *,
    base_url: str,
    api_key: str,
    incidents_table: str = "intellagent.public.incidents",
    parsed_emails_table: str = "parsed_emails",
    limit: int | None = None,
    allowed_categories: Optional[Collection[str]] = None,
    start_date: Any = None,
    end_date: Any = None,
) -> List[Dict[str, Any]]:
    """Parse incidents using :meth:`LakeAPIClient.query_stream` (Arrow over ``/query/stream``)."""
    if not base_url.strip():
        raise ValueError("base_url must be provided.")
    if not api_key.strip():
        raise ValueError("api_key must be provided.")

    sql_limit = limit if limit is not None and limit > 0 else None
    client = LakeAPIClient(base_url=base_url, api_key=api_key)
    return _parse_incidents_from_lake_rows(
        _iterate_lake_incident_rows_stream(
            client,
            incidents_table=incidents_table,
            parsed_emails_table=parsed_emails_table,
            sql_limit=sql_limit,
            start_date=start_date,
            end_date=end_date,
        ),
        limit=limit,
        allowed_categories=allowed_categories,
    )


__all__ = [
    "parse_incidents_with_email_bodies",
    "parse_incidents_from_lake_query",
    "parse_incidents_from_lake_stream",
    "INCIDENT_FIELDS",
    "HEADER_FIELDS",
]
