"""
Common helpers for normalizing MISP events and extracting features.
Shared across graph builders and the assembler to avoid duplication.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from datetime import timezone


try:
    from utils.url_extractor import parse_url_components  # type: ignore
except Exception:  # pragma: no cover
    def parse_url_components(url: str) -> Dict[str, Any]:  # type: ignore
        return {"full_url": url, "domain": "", "stem": "", "scheme": ""}


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
    """Parse date string into UNIX timestamp seconds (UTC). Returns 0 if parsing fails."""
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
                # If naive, treat as UTC
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


def parse_misp_events(misp_events: List[dict]) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for ev in misp_events:
        event = ev.get("Event", {})
        info = event.get("info", "")
        attrs = event.get("Attribute", []) or []

        sender = None
        receivers: List[str] = []
        subject = ""
        body = ""
        urls: List[str] = []
        date = ""

        for attr in attrs:
            a_type = (attr or {}).get("type", "")
            raw_val = (attr or {}).get("value", "")
            val = to_str(raw_val)
            if a_type == "email-src":
                normalized_sender = normalize_email_address(val) if val.strip() else None
                sender = normalized_sender if normalized_sender else None
            elif a_type == "email-dst":
                if val.strip():
                    normalized_receiver = normalize_email_address(val)
                    if normalized_receiver:
                        receivers.append(normalized_receiver)
            elif a_type == "email-subject":
                subject = val
            elif a_type == "email-body":
                body = val
            elif a_type == "url":
                if val.strip():
                    urls.append(val)
            elif a_type == "email-date":
                date = val

        normalized.append(
            {
                "email_info": info,
                "sender": sender,
                "receivers": receivers,
                "subject": subject,
                "body": body,
                "urls": urls,
                "date": date,
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
    "parse_misp_events",
]
