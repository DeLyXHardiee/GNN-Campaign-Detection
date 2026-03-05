from __future__ import annotations

import html
import re
from typing import Any, Dict, List, Optional, Iterable, Tuple
from urllib.parse import urlparse, parse_qsl, urlunparse
import tldextract
import numpy as np
from collections import defaultdict
from typing import Any, Dict, List


# ----------------------------
# Preprocessing (plaintext)
# ----------------------------

# quoted-printable soft line breaks can split URLs: "=\r\n" or "=\n"
_QP_SOFT_BREAK = re.compile(r"=\r?\n")

# zero-width chars sometimes appear in emails
_ZERO_WIDTH = re.compile(r"[\u200B-\u200D\uFEFF]")

# common defanging: hxxp / hxxps
_DEFANG_SCHEME = re.compile(r"(?i)\bhxxps?\b")

# common defanging for dots: example[.]com, example(.)com, [dot], etc.
_DEFANG_DOT = re.compile(r"(?i)(?:\[\.\]|\(\.\)|\{\.}|<\.>|\[dot]|\(dot\)|\{dot}|<dot>)")

# sometimes URLs appear as "http : //"
_DEFANG_COLON_SLASH = re.compile(r"(?i)\bhttps?\s*:\s*/\s*/")

# emails (to avoid treating domain inside user@domain.com as a URL)
_EMAIL_RE = re.compile(r"(?i)\b[a-z0-9._%+-]+@(?:[a-z0-9-]+\.)+[a-z]{2,24}\b")

# candidates: scheme URLs, www URLs, naked domains (conservative)
_SCHEME_URL_RE = re.compile(r"(?i)\bhttps?://[^\s<>()\"']+")
_WWW_URL_RE    = re.compile(r"(?i)\bwww\.[^\s<>()\"']+")

# Conservative naked domain match:
# - requires at least one dot
# - plausible TLD length
# - optional path/query
_DOMAIN_URL_RE = re.compile(
    r"""(?ix)
    \b
    (?:[a-z0-9-]{1,63}\.)+     # labels + dots
    [a-z]{2,24}                # TLD
    (?:/[^\s<>()"']+)?         # optional path/query
    """
)

# characters that often stick to URLs at the end in text
_TRAILING_PUNCT = '.,;:!?)]}>"\'…'
_LEADING_PUNCT = '(<[{"\''


def _preprocess_plaintext(text: str) -> str:
    if not text:
        return ""
    s = html.unescape(text)
    s = _QP_SOFT_BREAK.sub("", s)
    s = _ZERO_WIDTH.sub("", s)

    # defang scheme
    s = _DEFANG_SCHEME.sub(lambda m: "https" if m.group(0).lower() == "hxxps" else "http", s)

    # defang dots
    s = _DEFANG_DOT.sub(".", s)

    # defang "http : //"
    s = _DEFANG_COLON_SLASH.sub(lambda m: m.group(0).replace(" ", ""), s)

    return s


def _strip_junk(url: str) -> str:
    u = url.strip()

    # strip leading junk
    while u and u[0] in _LEADING_PUNCT:
        u = u[1:]

    # strip trailing junk
    while u and u[-1] in _TRAILING_PUNCT:
        u = u[:-1]

    return u.strip()


def _balance_brackets(url: str) -> str:
    # Remove unmatched trailing brackets/parens
    pairs = [("(", ")"), ("[", "]"), ("{", "}")]
    u = url
    for l, r in pairs:
        while u.endswith(r) and u.count(l) < u.count(r):
            u = u[:-1].rstrip()
    return u


def normalize_url_for_linking(url: str) -> Optional[str]:
    """
    Light normalization for *linking/campaign grouping*.
    Keeps path+query (often important for phishing), but:
    - ensures http/https scheme
    - lowercases scheme + host
    - removes fragments (#...)
    - replaces backslashes in paths
    """
    if not url:
        return None

    u = _balance_brackets(_strip_junk(url))

    # add scheme if starts with www. or naked domain
    low = u.lower()
    if low.startswith("www."):
        u = "http://" + u
    elif not (low.startswith("http://") or low.startswith("https://")):
        # if it looks like a domain, add http://
        if _DOMAIN_URL_RE.match(u) and not _EMAIL_RE.match(u):
            u = "http://" + u
        else:
            return None

    try:
        p = urlparse(u)
    except Exception:
        return None

    if p.scheme.lower() not in ("http", "https"):
        return None
    if not p.netloc:
        return None

    scheme = p.scheme.lower()
    host = p.netloc.lower().strip()
    path = (p.path or "").replace("\\", "/")
    query = p.query or ""
    fragment = ""  # drop

    return urlunparse((scheme, host, path, p.params or "", query, fragment))


def extract_urls_plaintext(body: str, *, normalize: bool = True) -> List[str]:
    """
    Very robust plaintext URL extraction.
    - Handles http(s)://, www., and conservative naked domains
    - Handles quoted-printable soft breaks, common defanging, html entities
    - Strips trailing punctuation and balances brackets
    - De-duplicates per email (preserves order)

    If normalize=True, returns normalized URLs (recommended for linking).
    """
    s = _preprocess_plaintext(body)

    # Find emails to avoid treating their domain part as URLs
    email_spans = [(m.start(), m.end()) for m in _EMAIL_RE.finditer(s)]

    def _inside_email(start: int, end: int) -> bool:
        for a, b in email_spans:
            if start >= a and end <= b:
                return True
        return False

    candidates: List[str] = []

    # scheme + www
    candidates.extend(m.group(0) for m in _SCHEME_URL_RE.finditer(s))
    candidates.extend(m.group(0) for m in _WWW_URL_RE.finditer(s))

    # naked domains: filter those that are inside email addresses
    for m in _DOMAIN_URL_RE.finditer(s):
        if not _inside_email(m.start(), m.end()):
            candidates.append(m.group(0))

    seen = set()
    out: List[str] = []
    for c in candidates:
        u = normalize_url_for_linking(c) if normalize else _balance_brackets(_strip_junk(c))
        if not u:
            continue
        if u not in seen:
            seen.add(u)
            out.append(u)

    return out


def build_email_url_map(
    bodies: List[str],
    email_ids: Optional[Iterable[int]] = None,
    *,
    normalize: bool = True,
) -> Dict[int, List[str]]:
    """
    Returns: { email_id: [url1, url2, ...] }

    If email_ids is None, uses row indices 0..N-1.
    """
    if email_ids is None:
        email_ids = range(len(bodies))

    url_map: Dict[int, List[str]] = {}
    for eid, body in zip(email_ids, bodies):
        url_map[int(eid)] = extract_urls_plaintext(body, normalize=normalize)

    return url_map


# A shared extractor instance (fast). cache_dir default is fine.
_TLD_EXTRACTOR = tldextract.TLDExtract(cache_dir=None)

import re
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse, parse_qsl

_TLD = tldextract.TLDExtract(cache_dir=None)

TRACKING_KEYS = {
    "fbclid", "gclid", "msclkid", "yclid", "igshid",
    "mc_cid", "mc_eid", "mkt_tok", "_hsenc", "_hsmi",
}
TRACKING_PREFIXES = ("utm_", "vero_", "rb_", "pk_", "sc_", "spm", "ref", "source")


def _is_tracking_key(k: str) -> bool:
    k = k.lower()
    return k in TRACKING_KEYS or any(k.startswith(p) for p in TRACKING_PREFIXES)


def _netloc_to_host(netloc: str) -> str:
    h = (netloc or "").strip().lower()
    if not h:
        return ""
    if "@" in h:
        h = h.split("@", 1)[1]
    if h.startswith("["):  # ipv6
        end = h.find("]")
        if end != -1:
            return h[1:end]
        return h.strip("[]")
    if ":" in h:
        h = h.split(":", 1)[0]
    return h


def _strip_www(host: str) -> str:
    return host[4:] if host.lower().startswith("www.") else host.lower()


# --- randomness heuristics (simple but effective) ---
_RE_UUID = re.compile(r"(?i)^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
_RE_HEX_LONG = re.compile(r"(?i)^[0-9a-f]{16,}$")
_RE_B64ISH = re.compile(r"^[A-Za-z0-9_\-+/=]{20,}$")
_RE_NUMFILE = re.compile(r"(?i)^\d{2,}\.(php|asp|aspx|jsp|html?)$")  # 2441.php
_RE_TN = re.compile(r"(?i)^t\d+$")  # t2 / t3 / t4


def _looks_random_token(s: str) -> bool:
    if not s:
        return False
    s = s.strip()
    if _RE_UUID.match(s):
        return True
    if _RE_HEX_LONG.match(s):
        return True
    if _RE_B64ISH.match(s) and sum(c.isdigit() for c in s) > 0 and sum(c.isalpha() for c in s) > 0:
        return True
    return False


def _mask_path_segment(seg: str) -> str:
    low = seg.lower()
    if _RE_NUMFILE.match(low):
        ext = low.split(".")[-1]
        return f"<num>.{ext}"
    if low.isdigit() and len(low) >= 2:
        return "<num>"
    if _looks_random_token(low):
        return "<rand>"
    return low


def _mask_query_key(k: str) -> str:
    low = k.lower()
    if _RE_TN.match(low):
        return "t<rand>"
    if _looks_random_token(low):
        return "<randkey>"
    return low


def url_to_artifacts_multi(url: str) -> Optional[Dict[str, Any]]:
    """
    Returns artifacts + 3 campaign-linking keys:
      - key_strict: host_key|full_masked_path|masked_query_keys
      - key_medium: host_key|stem2_masked|masked_query_keys
      - key_loose:  stem2_masked|masked_query_keys   (domain-rotation tolerant)
    """
    if not url:
        return None

    try:
        p = urlparse(url)
    except Exception:
        return None

    scheme = (p.scheme or "").lower()
    if scheme not in ("http", "https"):
        return None

    host = _strip_www(_netloc_to_host(p.netloc))
    if not host:
        return None

    ext = _TLD(host)
    registered_domain = f"{ext.domain}.{ext.suffix}" if ext.domain and ext.suffix else ""
    host_key = registered_domain if registered_domain else host  # fallback for IPs

    # path segments (masked)
    path = (p.path or "").replace("\\", "/")
    path = re.sub(r"/{2,}", "/", path)
    segs = [s for s in path.split("/") if s]
    segs_masked = [_mask_path_segment(s) for s in segs]

    full_masked_path = "/" + "/".join(segs_masked) if segs_masked else ""
    stem2_masked = "/".join(segs_masked[:2]) if segs_masked else ""

    # query keys (masked, tracking removed)
    q_pairs = parse_qsl(p.query or "", keep_blank_values=True)
    keys = [_mask_query_key(k) for (k, _) in q_pairs if k and not _is_tracking_key(k)]
    keys = sorted(set(keys))

    key_suffix = "&".join(keys)

    key_strict = f"{host_key}|{full_masked_path}|{key_suffix}"
    key_medium = f"{host_key}|{stem2_masked}|{key_suffix}"
    key_loose = f"{stem2_masked}|{key_suffix}"

    return {
        "url": url,
        "host": host,
        "registered_domain": registered_domain,
        "host_key": host_key,
        "path": path,
        "masked_path": full_masked_path,
        "masked_stem2": stem2_masked,
        "query_keys_masked": keys,
        "key_strict": key_strict,
        "key_medium": key_medium,
        "key_loose": key_loose,
    }


def build_email_url_artifact_map(url_map: Dict[int, List[str]]) -> Dict[int, List[Dict[str, Any]]]:
    out: Dict[int, List[Dict[str, Any]]] = {}
    for eid, urls in url_map.items():
        artifacts = []
        for u in urls:
            a = url_to_artifacts_multi(u)
            if a is not None:
                artifacts.append(a)
        out[eid] = artifacts
    return out


from typing import Any, Dict, Optional
from urllib.parse import urlparse, parse_qsl
import re

_TLD_EXTRACTOR = tldextract.TLDExtract(cache_dir=None)

TRACKING_KEYS = {
    "fbclid", "gclid", "msclkid", "yclid", "igshid",
    "mc_cid", "mc_eid", "mkt_tok", "_hsenc", "_hsmi",
}
TRACKING_PREFIXES = ("utm_", "vero_", "rb_", "pk_", "sc_", "spm", "ref", "source")

def _is_tracking_key(k: str) -> bool:
    k = k.lower()
    if k in TRACKING_KEYS:
        return True
    if any(k.startswith(p) for p in TRACKING_PREFIXES):
        return True
    return False

def _strip_www(host: str) -> str:
    host = host.lower()
    return host[4:] if host.startswith("www.") else host

def _netloc_to_host(netloc: str) -> str:
    """
    Turn a URL netloc into a clean host:
    - remove credentials user:pass@
    - handle ipv6 [::1]:443
    - remove :port
    """
    h = (netloc or "").strip().lower()
    if not h:
        return ""

    # drop credentials
    if "@" in h:
        h = h.split("@", 1)[1]

    # ipv6 like [2001:db8::1]:443
    if h.startswith("["):
        end = h.find("]")
        if end != -1:
            return h[1:end]  # inside brackets
        return h.strip("[]")

    # drop port
    if ":" in h:
        h = h.split(":", 1)[0]

    return h

def url_to_artifacts(url: str) -> Optional[Dict[str, Any]]:
    if not url:
        return None

    try:
        p = urlparse(url)
    except Exception:
        return None

    scheme = (p.scheme or "").lower()
    if scheme not in ("http", "https"):
        return None

    host = _netloc_to_host(p.netloc)
    if not host:
        return None

    host_no_www = _strip_www(host)

    # Domain parsing (works for normal domains; for IPs it yields empty domain/suffix)
    ext = _TLD_EXTRACTOR(host_no_www)
    subdomain = ext.subdomain or ""
    domain = ext.domain or ""
    suffix = ext.suffix or ""
    registered_domain = f"{domain}.{suffix}" if domain and suffix else ""

    # Choose a stable host key: prefer registered_domain, fallback to host_no_www (IP/localhost/etc.)
    host_key = registered_domain if registered_domain else host_no_www

    # Path artifacts
    path = (p.path or "").replace("\\", "/")
    path = re.sub(r"/{2,}", "/", path)
    segments = [seg for seg in path.split("/") if seg]
    path_stem_1 = segments[0].lower() if len(segments) >= 1 else ""
    path_stem_2 = "/".join(seg.lower() for seg in segments[:2]) if len(segments) >= 2 else path_stem_1
    path_last = segments[-1].lower() if segments else ""

    # Query artifacts (keys only; drop obvious tracking keys)
    q_pairs = parse_qsl(p.query or "", keep_blank_values=True)
    query_keys = sorted({k.lower() for k, _ in q_pairs if k and not _is_tracking_key(k)})

    # IMPORTANT: avoid meaningless keys like "||"
    # If we somehow have no host_key, skip. (Shouldn't happen now.)
    if not host_key:
        return None

    return {
        "url": url,
        "scheme": scheme,
        "host": host,
        "host_no_www": host_no_www,
        "registered_domain": registered_domain,
        "subdomain": subdomain,
        "domain": domain,
        "suffix": suffix,
        "path": path,
        "path_stem_1": path_stem_1,
        "path_stem_2": path_stem_2,
        "path_last": path_last,
        "query_keys": query_keys,
        "n_query_params": len(q_pairs)
    }


def invert_url_keys_multi(
    artifact_map: Dict[int, List[Dict[str, Any]]],
    key_field: str = "key_medium",
    min_emails: int = 2,
    require_nontrivial: bool = True,
) -> Dict[str, List[int]]:
    """
    Builds: key -> sorted email_ids
    Keeps only keys appearing in >= min_emails distinct emails.

    key_field: "key_strict" | "key_medium" | "key_loose"
    require_nontrivial: drop keys that are too generic (empty stem + empty keys)
    """
    key_to_emails: Dict[str, set[int]] = defaultdict(set)

    for eid, artifacts in artifact_map.items():
        keys_in_email = set()
        for a in artifacts:
            k = a.get(key_field)
            if not k:
                continue

            if require_nontrivial:
                # Format differs by key_field; use the artifact fields directly
                stem = a.get("masked_stem2", "")
                qk = a.get("query_keys_masked", [])
                if (not stem) and (not qk):
                    continue

            keys_in_email.add(k)

        for k in keys_in_email:
            key_to_emails[k].add(int(eid))

    return {k: sorted(v) for k, v in key_to_emails.items() if len(v) >= min_emails}





