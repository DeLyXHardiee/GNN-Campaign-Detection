"""
Common helpers for normalizing MISP events and extracting features.
Shared across graph builders and the assembler to avoid duplication.
"""
from __future__ import annotations

import ast
import json
import re
from typing import Any, Dict, List, Optional, Tuple
from typing import Set
from datetime import datetime, timezone
import math

# Whole string is a scalar epoch (seconds or ms); avoids misparsing year-like fragments as floats.
_EPOCH_NUMERIC = re.compile(r"^-?\d+(?:\.\d+)?$")

try:
    from core.preprocessing.utils.defang import defang_url_string
except ModuleNotFoundError:
    from preprocessing.utils.defang import defang_url_string

try:
    from core.preprocessing.utils.url_extractor import (
        extract_urls_from_text,
        normalize_http_url,
        parse_url_components,
    )
except ModuleNotFoundError:
    from preprocessing.utils.url_extractor import (
        extract_urls_from_text,
        normalize_http_url,
        parse_url_components,
    )

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


def to_unix_ts(value: Any) -> int:
    """Convert email/event date values to UNIX seconds (since 1970-01-01 UTC).

    Accepts legacy numeric epoch seconds, ISO/RFC datetime strings (incl.
    ``YYYY-MM-DD HH:MM:SS+00:00`` from the lake), :class:`datetime`, and the
    RFC-2822-style strings handled previously.
    """
    if value is None:
        return 0
    if isinstance(value, datetime):
        dt = value
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        try:
            return int(dt.timestamp())
        except (OSError, OverflowError, ValueError):
            return 0
    if isinstance(value, (int, float)):
        try:
            if isinstance(value, float) and (value != value or math.isnan(value)):
                return 0
        except Exception:
            return 0
        v = float(value)
        if v > 1e12:
            v = v / 1000.0
        try:
            return int(v)
        except (OverflowError, ValueError):
            return 0

    date_str = to_str(value).strip()
    if not date_str:
        return 0

    if _EPOCH_NUMERIC.fullmatch(date_str):
        try:
            v = float(date_str)
            if v > 1e12:
                v = v / 1000.0
            return int(v)
        except (OverflowError, ValueError):
            return 0

    iso = date_str
    if iso.endswith("Z"):
        iso = iso[:-1] + "+00:00"
    iso_candidates = [iso]
    if len(iso) >= 19 and iso[4] == "-" and iso[7] == "-" and iso[10] == " ":
        iso_candidates.append(iso[:10] + "T" + iso[11:])

    for cand in iso_candidates:
        try:
            dt = datetime.fromisoformat(cand)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return int(dt.timestamp())
        except ValueError:
            continue

    try:
        for fmt in [
            "%a, %d %b %Y %H:%M:%S %z",
            "%a, %d %b %Y %H:%M:%S %Z",
            "%d %b %Y %H:%M:%S %z",
            "%Y-%m-%d %H:%M:%S%z",
            "%Y-%m-%d %H:%M:%S %z",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d",
        ]:
            try:
                dt = datetime.strptime(date_str, fmt)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return int(dt.timestamp())
            except ValueError:
                continue
    except Exception:
        return 0
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


# One-hot encoding for SPF/DKIM/DMARC for GNN message passing. "unknown" = missing/other.
AUTH_ONEHOT_VOCAB = ("pass", "fail", "neutral", "softfail", "none", "unknown")
AUTH_ONEHOT_DIM_PER_FIELD = len(AUTH_ONEHOT_VOCAB)
AUTH_ONEHOT_DIM = 3 * AUTH_ONEHOT_DIM_PER_FIELD  # spf + dkim + dmarc


def auth_value_to_onehot(value: str) -> List[float]:
    """Encode a single auth result (e.g. 'pass', 'fail') as a one-hot vector of length AUTH_ONEHOT_DIM_PER_FIELD."""
    s = (to_str(value) or "").strip().lower()
    if not s or s not in AUTH_ONEHOT_VOCAB:
        idx = AUTH_ONEHOT_VOCAB.index("unknown")
    else:
        idx = AUTH_ONEHOT_VOCAB.index(s)
    return [1.0 if i == idx else 0.0 for i in range(AUTH_ONEHOT_DIM_PER_FIELD)]


def auth_triple_to_onehot(spf: str, dkim: str, dmarc: str) -> List[float]:
    """Encode (spf, dkim, dmarc) as a concatenated one-hot vector of length AUTH_ONEHOT_DIM for email features."""
    out: List[float] = []
    out.extend(auth_value_to_onehot(spf))
    out.extend(auth_value_to_onehot(dkim))
    out.extend(auth_value_to_onehot(dmarc))
    return out


def parse_authentication_results(value: Any) -> Dict[str, str]:
    """Extract spf, dkim, dmarc from Authentication-Results header value.

    Accepts compact strings like 'spf=pass; dkim=pass; dmarc=pass' or raw header text.
    Returns dict with keys 'spf', 'dkim', 'dmarc'; missing values are ''.
    """
    import re
    text = to_str(value).strip() if value is not None else ""
    if not text:
        return {"spf": "", "dkim": "", "dmarc": ""}
    out: Dict[str, str] = {"spf": "", "dkim": "", "dmarc": ""}
    for label in ("spf", "dkim", "dmarc"):
        match = re.search(rf"\b{label}\s*=\s*([a-z0-9_-]+)", text, flags=re.IGNORECASE)
        if match:
            out[label] = match.group(1).lower()
    return out


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


def _coerce_mapping_list(value: Any) -> List[Dict[str, Any]]:
    """Best-effort conversion of a MISP attribute value into a list of dicts."""
    if value is None:
        return []

    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]

    if isinstance(value, dict):
        return [value]

    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [item for item in parsed if isinstance(item, dict)]
            if isinstance(parsed, dict):
                return [parsed]
        except Exception:
            pass
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, list):
                return [item for item in parsed if isinstance(item, dict)]
            if isinstance(parsed, dict):
                return [parsed]
        except Exception:
            pass

    return []


_SHA256_RE = re.compile(r"^[a-f0-9]{64}$", flags=re.IGNORECASE)


def is_sha256_hex(value: Any) -> bool:
    s = to_str(value).strip().lower()
    return bool(_SHA256_RE.fullmatch(s))


def extract_attachment_sha256s(raw_attachments: List[Any], attachment_metadata: List[Dict[str, Any]]) -> List[str]:
    """Normalize attachment indicators into distinct lowercase SHA256 hashes only."""
    out: List[str] = []
    seen: Set[str] = set()
    for value in raw_attachments:
        s = to_str(value).strip().lower()
        if not s or s in seen or not is_sha256_hex(s):
            continue
        seen.add(s)
        out.append(s)
    for meta in attachment_metadata:
        if not isinstance(meta, dict):
            continue
        s = to_str(meta.get("sha256", "")).strip().lower()
        if not s or s in seen or not is_sha256_hex(s):
            continue
        seen.add(s)
        out.append(s)
    return out


def _url_canonical_dedup_key(normalized: str) -> str:
    """Stable key for deduplicating equivalent http(s) URLs (trailing slash tolerant)."""
    return (normalized or "").strip().lower().rstrip("/")


def _append_url_once_canonical_defanged(
    accum_urls: List[str],
    canonical_seen: Set[str],
    raw: str,
) -> None:
    """
    Append a single defanged URL to ``accum_urls`` if ``raw`` normalizes to a new http(s) URL.

    Merges stored ``hxxps://`` attributes with freshly extracted ``https://`` strings so only
    one defanged form exists per canonical link.
    """
    norm = normalize_http_url(to_str(raw).strip())
    if not norm:
        return
    key = _url_canonical_dedup_key(norm)
    if not key or key in canonical_seen:
        return
    canonical_seen.add(key)
    accum_urls.append(defang_url_string(norm))


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
        }
        url_canonical_seen: Set[str] = set()
        fields: Dict[str, Any] = {
            "subject": "",
            "body": "",
            "html": {},
            "css": {},
            "attachment_metadata": [],
            "date": "",
            "received_hops": [],
            "return_path": {},
            "authentication_results": "",
            "auth_spf": "",
            "auth_dkim": "",
            "auth_dmarc": "",
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
                #extracted = _extract_urls_from_attr_value(raw_val)
                extracted = _extract_strings_from_attr_value(raw_val)
            elif mapping.strategy == "string_list":
                extracted = _extract_strings_from_attr_value(raw_val)
            elif mapping.strategy == "dict_mapping":
                extracted = _coerce_mapping(raw_val)
            elif mapping.strategy == "dict_list":
                extracted = _coerce_mapping_list(raw_val)
            elif mapping.strategy == "received_list":
                extracted = _coerce_received_hops(raw_val)
            else:
                extracted = to_str(raw_val)

            if mapping.accumulate:
                values = extracted if isinstance(extracted, list) else [to_str(extracted)]
                if mapping.field == "urls":
                    for value in values:
                        item = to_str(value).strip()
                        if not item:
                            continue
                        if mapping.lowercase_items:
                            item = item.lower()
                        _append_url_once_canonical_defanged(
                            accum["urls"], url_canonical_seen, item
                        )
                else:
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
                if mapping.field == "authentication_results":
                    auth = parse_authentication_results(extracted)
                    fields["auth_spf"] = auth.get("spf", "")
                    fields["auth_dkim"] = auth.get("dkim", "")
                    fields["auth_dmarc"] = auth.get("dmarc", "")

            if mapping.extract_urls_side_effect:
                for url in _extract_urls_from_attr_value(raw_val):
                    _append_url_once_canonical_defanged(accum["urls"], url_canonical_seen, url)

        external_id = to_str(event.get("external_id", ""))
        attachment_metadata = fields.get("attachment_metadata", [])
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
                "attachments": extract_attachment_sha256s(accum["attachments"], attachment_metadata),
                "attachment_metadata": attachment_metadata,
                "urls": accum["urls"],
                "date": fields["date"],
                "received_hops": fields.get("received_hops", []),
                "cyrillic_domain": fields.get("cyrillic_domain", ""),
                "contains_symbols": fields.get("contains_symbols", ""),
                "body_has_tracking_url": fields.get("body_has_tracking_url", ""),
                "body_has_tracking_image": fields.get("body_has_tracking_image", ""),
                "body_has_tracking_pixel": fields.get("body_has_tracking_pixel", ""),
                "body_has_unsubscribe_link": fields.get("body_has_unsubscribe_link", ""),
                "domain_is_common_webprovided": fields.get("domain_is_common_webprovided", ""),
                "return_path": fields.get("return_path", {}),
                "auth_spf": fields.get("auth_spf", ""),
                "auth_dkim": fields.get("auth_dkim", ""),
                "auth_dmarc": fields.get("auth_dmarc", ""),
            }
        )

    return normalized


__all__ = [
    "parse_url_components",
    "to_str",
    "to_unix_ts",
    "normalize_email_address",
    "extract_email_domain",
    "extract_all_emails",
    "is_sha256_hex",
    "extract_attachment_sha256s",
    "parse_misp_events",
    "parse_authentication_results",
    "auth_value_to_onehot",
    "auth_triple_to_onehot",
    "AUTH_ONEHOT_VOCAB",
    "AUTH_ONEHOT_DIM_PER_FIELD",
    "AUTH_ONEHOT_DIM",
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
