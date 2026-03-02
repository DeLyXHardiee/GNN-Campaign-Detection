"""
Common helpers for normalizing MISP events and extracting features.
Shared across graph builders and the assembler to avoid duplication.
"""
from __future__ import annotations

import ast
import json
from typing import Any, Dict, List, Optional, Tuple
from typing import Set
from datetime import timezone
import math
import sys

sys.path.append('../preprocessing/utils')

from preprocessing.utils.url_extractor import parse_url_components, extract_urls_from_text
from .misp_attribute_schema import DEFAULT_MISP_ATTRIBUTE_SCHEMA


def to_str(val: Any) -> str:
    if isinstance(val, str):
        return val
    if val is None:
        return ""
    try:
        if isinstance(val, float) and val != val:
            return ""
    except Exception:
        return ""
    try:
        return str(val)
    except Exception:
        return ""


def extract_week_key(date_str: str) -> Optional[str]:
    if not date_str or not date_str.strip():
        return None
    try:
        from datetime import datetime
        for fmt in [
            "%a, %d %b %Y %H:%M:%S %z",
            "%a, %d %b %Y %H:%M:%S %Z",
            "%d %b %Y %H:%M:%S %z",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d",
        ]:
            try:
                dt = datetime.strptime(date_str.strip(), fmt)
                iso = dt.isocalendar()
                return f"{iso[0]}-W{iso[1]:02d}"
            except ValueError:
                continue
        return None
    except Exception:
        return None


def to_unix_ts(date_str: str) -> int:
    if not date_str or not date_str.strip():
        return 0
    try:
        from datetime import datetime
        for fmt in [
            "%a, %d %b %Y %H:%M:%S %z",
            "%a, %d %b %Y %H:%M:%S %Z",
            "%d %b %Y %H:%M:%S %z",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d",
        ]:
            try:
                dt = datetime.strptime(date_str.strip(), fmt)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return int(dt.timestamp())
            except ValueError:
                continue
        return 0
    except Exception:
        return 0


def normalize_email_address(email_str: str) -> str:
    if not email_str:
        return ""
    email_str = email_str.strip()
    if "<" in email_str and ">" in email_str:
        start = email_str.find("<")
        end = email_str.find(">", start)
        if end > start:
            email_str = email_str[start + 1 : end]
    return email_str.lower().strip()


def extract_email_domain(email_str: str) -> str:
    if not email_str or "@" not in email_str:
        return ""
    try:
        return email_str.split("@")[-1].strip().lower()
    except Exception:
        return ""

def extract_all_emails(text: str) -> List[str]:
    """Extract all email addresses from a free-form string.
    Returns lowercased addresses without surrounding spaces.
    """
    if not text:
        return []
    import re
    pattern = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
    matches = pattern.findall(text)
    out: List[str] = []
    seen: Set[str] = set()
    for m in matches:
        nm = normalize_email_address(m)
        if nm and nm not in seen:
            seen.add(nm)
            out.append(nm)
    if not out:
        for part in re.split(r"[,;]", text):
            nm = normalize_email_address(part)
            if nm and nm not in seen and "@" in nm:
                seen.add(nm)
                out.append(nm)
    return out


def _extract_urls_from_attr_value(value: Any) -> List[str]:
    """Extract URLs from scalar/list/dict MISP attribute values."""
    if isinstance(value, str):
        return extract_urls_from_text(value)

    if isinstance(value, list):
        urls: List[str] = []
        for item in value:
            urls.extend(_extract_urls_from_attr_value(item))
        return urls

    if isinstance(value, dict):
        urls = []
        for item in value.values():
            urls.extend(_extract_urls_from_attr_value(item))
        return urls

    text = to_str(value)
    return extract_urls_from_text(text) if text else []


def _extract_emails_from_attr_value(value: Any) -> List[str]:
    """Extract email addresses from scalar/list/dict MISP attribute values."""
    if isinstance(value, str):
        return extract_all_emails(value)

    if isinstance(value, list):
        emails: List[str] = []
        for item in value:
            emails.extend(_extract_emails_from_attr_value(item))
        return emails

    if isinstance(value, dict):
        emails: List[str] = []
        for item in value.values():
            emails.extend(_extract_emails_from_attr_value(item))
        return emails

    text = to_str(value)
    return extract_all_emails(text) if text else []


def _extract_strings_from_attr_value(value: Any) -> List[str]:
    """Extract scalar strings from scalar/list/dict values recursively."""
    if isinstance(value, str):
        s = value.strip()
        return [s] if s else []

    if isinstance(value, list):
        out: List[str] = []
        for item in value:
            out.extend(_extract_strings_from_attr_value(item))
        return out

    if isinstance(value, dict):
        out: List[str] = []
        for item in value.values():
            out.extend(_extract_strings_from_attr_value(item))
        return out

    text = to_str(value).strip()
    return [text] if text else []


def _normalize_hop_dict(h: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize a Received hop dict to lowercase keys (origin_ip, helo_host, by_host, timestamp)."""
    if not isinstance(h, dict):
        return {}
    return {str(k).strip().lower(): (to_str(v).strip() if v is not None else "") for k, v in h.items()}


def _coerce_received_hops(value: Any) -> List[Dict[str, Any]]:
    """Coerce MISP attribute value to list of Received hop dicts (origin_ip, helo_host, by_host, timestamp)."""
    if value is None:
        return []
    if isinstance(value, list):
        return [_normalize_hop_dict(h) for h in value if isinstance(h, dict)]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [_normalize_hop_dict(h) for h in parsed if isinstance(h, dict)]
        except Exception:
            pass
    return []


def _coerce_mapping(value: Any) -> Dict[str, Any]:
    """Best-effort conversion of a MISP attribute value into a dict."""
    if isinstance(value, dict):
        return value
    if value is None:
        return {}
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return {}
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
        return {}
    return {}


def parse_misp_events(misp_events: List[dict]) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for idx_ev, ev in enumerate(misp_events):
        event = ev.get("Event", {})
        info = event.get("info", "")
        email_index = event.get("email_index", idx_ev)
        attrs = event.get("Attribute", []) or []

        accum: Dict[str, List[str]] = {
            "senders": [],
            "receivers": [],
            "attachments": [],
            "urls": [],
        }
        accum_seen: Dict[str, Set[str]] = {
            "senders": set(),
            "receivers": set(),
            "attachments": set(),
            "urls": set(),
        }
        fields: Dict[str, Any] = {
            "subject": "",
            "body": "",
            "html": {},
            "css": {},
            "date": "",
            "received_hops": [],
        }

        for attr in attrs:
            a_type = to_str((attr or {}).get("type", "")).strip().lower()
            raw_val = (attr or {}).get("value", "")
            mapping = DEFAULT_MISP_ATTRIBUTE_SCHEMA.resolve(a_type)
            if mapping is None:
                continue

            extracted: Any
            if mapping.strategy == "email_list":
                extracted = _extract_emails_from_attr_value(raw_val)
                if not extracted:
                    fallback = normalize_email_address(to_str(raw_val))
                    extracted = [fallback] if fallback and "@" in fallback else []
            elif mapping.strategy == "url_list":
                extracted = _extract_urls_from_attr_value(raw_val)
            elif mapping.strategy == "string_list":
                extracted = _extract_strings_from_attr_value(raw_val)
            elif mapping.strategy == "dict_mapping":
                extracted = _coerce_mapping(raw_val)
            elif mapping.strategy == "received_list":
                extracted = _coerce_received_hops(raw_val)
            else:
                extracted = to_str(raw_val)

            if mapping.accumulate:
                values = extracted if isinstance(extracted, list) else [to_str(extracted)]
                for value in values:
                    item = to_str(value).strip()
                    if not item:
                        continue
                    if mapping.lowercase_items:
                        item = item.lower()
                    if item not in accum_seen[mapping.field]:
                        accum_seen[mapping.field].add(item)
                        accum[mapping.field].append(item)
            else:
                fields[mapping.field] = extracted

            if mapping.extract_urls_side_effect:
                for url in _extract_urls_from_attr_value(raw_val):
                    if url and url not in accum_seen["urls"]:
                        accum_seen["urls"].add(url)
                        accum["urls"].append(url)

        external_id = to_str(event.get("external_id", ""))
        normalized.append(
            {
                "email_info": info,
                "email_index": email_index,
                "external_id": external_id.strip() or str(email_index),
                "senders": accum["senders"],
                "receivers": accum["receivers"],
                "subject": fields["subject"],
                "body": fields["body"],
                "html": fields["html"],
                "css": fields["css"],
                "attachments": accum["attachments"],
                "urls": accum["urls"],
                "date": fields["date"],
                "received_hops": fields.get("received_hops", []),
            }
        )

    return normalized


__all__ = [
    "parse_url_components",
    "to_str",
    "extract_week_key",
    "to_unix_ts",
    "normalize_email_address",
    "extract_email_domain",
    "extract_all_emails",
    "parse_misp_events",
    "compute_lexical_features",
    "is_freemail_domain",
]


def compute_lexical_features(s: str) -> List[float]:
    """Compute a small lexical feature vector for a domain or stem string.

    Features (8 dims):
    - length
    - num_digits
    - num_hyphens
    - num_alpha
    - num_non_alnum
    - digit_ratio (digits/length)
    - hyphen_ratio (hyphens/length)
    - shannon_entropy (bits per char)
    """
    if not isinstance(s, str):
        s = str(s) if s is not None else ""
    s = s.strip().lower()
    L = float(len(s))
    if L <= 0:
        return [0.0] * 8
    num_digits = sum(ch.isdigit() for ch in s)
    num_hyphens = s.count('-')
    num_alpha = sum(ch.isalpha() for ch in s)
    num_non_alnum = sum(not ch.isalnum() for ch in s)
    digit_ratio = float(num_digits) / L
    hyphen_ratio = float(num_hyphens) / L
    # entropy
    freq: Dict[str, int] = {}
    for ch in s:
        freq[ch] = freq.get(ch, 0) + 1
    entropy = 0.0
    for cnt in freq.values():
        p = cnt / L
        entropy -= p * math.log2(p)
    return [
        L,
        float(num_digits),
        float(num_hyphens),
        float(num_alpha),
        float(num_non_alnum),
        float(digit_ratio),
        float(hyphen_ratio),
        float(entropy),
    ]


_FREEMAIL = {
    "gmail.com",
    "googlemail.com",
    "yahoo.com",
    "yahoo.co.uk",
    "outlook.com",
    "hotmail.com",
    "live.com",
    "aol.com",
    "icloud.com",
    "proton.me",
    "protonmail.com",
    "gmx.com",
    "yandex.com",
}


def is_freemail_domain(domain: str) -> bool:
    d = (domain or "").strip().lower()
    return d in _FREEMAIL
