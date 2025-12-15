"""
Common helpers for normalizing MISP events and extracting features.
Shared across graph builders and the assembler to avoid duplication.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
from typing import Set
from datetime import timezone
import math


try:  
    from core.utils.url_extractor import parse_url_components  
except Exception:
    try:  
        from utils.url_extractor import parse_url_components  
    except Exception:  
        def parse_url_components(url: str) -> Dict[str, Any]: 
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


def parse_misp_events(misp_events: List[dict]) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for idx_ev, ev in enumerate(misp_events):
        event = ev.get("Event", {})
        info = event.get("info", "")
        email_index = event.get("email_index", idx_ev)
        attrs = event.get("Attribute", []) or []

        sender: Optional[str] = None
        receivers: List[str] = []
        receiver_set: Set[str] = set()
        subject = ""
        body = ""
        urls: List[str] = []
        url_set: Set[str] = set()
        date = ""

        for attr in attrs:
            a_type = (attr or {}).get("type", "")
            raw_val = (attr or {}).get("value", "")
            val = to_str(raw_val)
            if a_type == "email-src":
                addrs = extract_all_emails(val)
                sender = addrs[0] if addrs else (normalize_email_address(val) if val.strip() else None)
            elif a_type in ("email-dst", "email-cc", "email-bcc"):
                if val.strip():
                    addrs = extract_all_emails(val)
                    for addr in addrs:
                        if addr not in receiver_set:
                            receiver_set.add(addr)
                            receivers.append(addr)
            elif a_type == "email-subject":
                subject = val
            elif a_type == "email-body":
                body = val
            elif a_type == "url":
                if val.strip() and val not in url_set:
                    url_set.add(val)
                    urls.append(val)
            elif a_type == "email-date":
                date = val

        normalized.append(
            {
                "email_info": info,
                "email_index": email_index,
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
