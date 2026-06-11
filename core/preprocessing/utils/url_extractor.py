"""
URL extraction and parsing utilities for email campaign detection.

Extraction pipeline (mirrors mailparser → HTML + plain text, with JS-stack equivalents
in parentheses):

- **HTML** — ``href`` on ``a`` / ``area``, optional ``<base>`` resolution (lxml recover
  mode; cheerio-style).
- **Plain text** — linkify-it (same role as linkify-it in Node).
- **Normalize / validate** — ``urllib.parse``; only ``http`` / ``https`` with a real host
  on a **public suffix** (``tldextract`` / PSL), ``localhost``, or an IP — so dotted
  non-host tokens (e.g. ``td.padding`` from CSS) are not accepted.

Bodies from :mod:`preprocessing.body_parser` defang schemes (``hxxps://``, …). Extraction
refangs first so hosts and schemes are recoverable.

Requires **linkify-it-py** for plain-text URL discovery (no regex-based extraction).
"""
from __future__ import annotations

import html as html_module
import ipaddress
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urljoin, urlparse, urlunparse

import tldextract
from linkify_it import LinkifyIt

try:
    from lxml import html as lxml_html

    _LXML_HTML = lxml_html
except ImportError:
    _LXML_HTML = None

try:
    from bs4 import BeautifulSoup
    from bs4.element import Tag as Bs4Tag

    _BS4 = BeautifulSoup
except ImportError:
    _BS4 = None
    Bs4Tag = None  # type: ignore[misc, assignment]

from preprocessing.html_css_parser import _sanitize_html_for_parsing

_LOG = logging.getLogger(__name__)

_LINKIFY_IT = LinkifyIt()

_JUNK_SCHEME_PREFIXES = (
    "javascript:",
    "vbscript:",
    "data:",
    "about:",
    "file:",
)


def _replace_case_insensitive(text: str, needle: str, repl: str) -> str:
    """Replace every case-insensitive occurrence of ``needle`` with ``repl`` (no regex)."""
    if not text or not needle:
        return text
    lower = text.lower()
    needle_l = needle.lower()
    out: List[str] = []
    start = 0
    nlen = len(needle)
    while True:
        idx = lower.find(needle_l, start)
        if idx == -1:
            out.append(text[start:])
            break
        out.append(text[start:idx])
        out.append(repl)
        start = idx + nlen
    return "".join(out)


def refang_url_like_schemes(text: str) -> str:
    """Undo preprocessing defang so extractors see real schemes."""
    if not text:
        return ""
    out = _replace_case_insensitive(text, "hxxps://", "https://")
    out = _replace_case_insensitive(out, "hxxp://", "http://")
    return out


def _strip_wrapping_punct(url: str) -> str:
    s = (url or "").strip()
    while s and s[-1] in ")].,;:!?>'\"":
        s = s[:-1].rstrip()
    while s and s[0] in "(<\"'":
        s = s[1:].lstrip()
    return s.strip()


def _element_local_tag(tag: Any) -> str:
    if not isinstance(tag, str):
        return ""
    if tag.startswith("{"):
        return tag.rsplit("}", 1)[-1].lower()
    return tag.lower()


def _safe_urlparse(url: str):
    """``urlparse`` that returns ``None`` on malformed hosts (e.g. ``http://[.]/``)."""
    try:
        return urlparse(url)
    except ValueError:
        return None


def _host_acceptable_for_http_url(host: str) -> bool:
    """
    Reject dotted garbage (e.g. ``td.padding`` from CSS) that ``urlparse`` treats as a host.

    Requires a real ICANN/public suffix (via ``tldextract`` / PSL), or ``localhost``,
    or a literal IP address.
    """
    h = (host or "").strip().lower()
    if not h:
        return False
    if h == "localhost":
        return True
    ip_probe = h.strip("[]")
    try:
        ipaddress.ip_address(ip_probe)
        return True
    except ValueError:
        pass
    ext = tldextract.extract(h)
    if ext.suffix and ext.domain:
        return True
    return False


def normalize_http_url(raw: str) -> Optional[str]:
    """
    Return a canonical ``http``/``https`` URL, or ``None`` if not an acceptable web URL.

    Scheme-relative ``//host`` and host-only tokens get a usable scheme for parsing.
    Hosts without a recognized public suffix (e.g. ``td.padding``) are rejected so
    scheme-less tokens are not turned into bogus ``https`` URLs.
    """
    s = _strip_wrapping_punct(refang_url_like_schemes(raw))
    if not s:
        return None
    low = s.lower()
    if any(low.startswith(p) for p in _JUNK_SCHEME_PREFIXES):
        return None
    if low.startswith("#"):
        return None
    if low.startswith("/") and not low.startswith("//"):
        return None

    if s.startswith("//"):
        s = "https:" + s
    else:
        probe = _safe_urlparse(s)
        if probe is None:
            return None
        if not probe.scheme:
            s = "https://" + s

    parsed = _safe_urlparse(s)
    if parsed is None:
        return None

    scheme = (parsed.scheme or "").lower()
    if scheme not in {"http", "https"}:
        return None

    try:
        host = (parsed.hostname or "").strip().lower()
    except ValueError:
        return None
    if not host:
        return None
    if not _host_acceptable_for_http_url(host):
        return None

    netloc_parts: List[str] = []
    if parsed.username:
        auth = parsed.username
        if parsed.password:
            auth += ":" + parsed.password
        netloc_parts.append(auth + "@")
    netloc_parts.append(host)
    if parsed.port:
        netloc_parts.append(f":{parsed.port}")
    netloc = "".join(netloc_parts)

    path = parsed.path if parsed.path else "/"

    try:
        return urlunparse((scheme, netloc, path, parsed.params, parsed.query, parsed.fragment))
    except Exception:
        return None


def extract_urls_with_linkify(text: str) -> List[str]:
    """Collect URL strings from plain text using linkify-it's ``match`` / normalized ``url``."""
    if not text:
        return []
    t = refang_url_like_schemes(text)
    try:
        matches = _LINKIFY_IT.match(t)
    except Exception:
        _LOG.debug("linkify-it match failed", exc_info=True)
        return []
    if not matches:
        return []
    out: List[str] = []
    for m in matches:
        u = getattr(m, "url", None) or getattr(m, "text", None)
        if isinstance(u, str) and u.strip():
            out.append(u.strip())
    return out


def extract_hrefs_from_html(html: str) -> List[str]:
    """Collect ``href`` values from ``a`` / ``area``; resolve with first ``<base href>``."""
    if not html or not str(html).strip():
        return []
    h = refang_url_like_schemes(str(html))
    try:
        sanitized = _sanitize_html_for_parsing(h)
    except Exception:
        sanitized = h

    raw_hrefs: List[str] = []
    base_href = ""

    if _LXML_HTML is not None:
        try:
            parser = _LXML_HTML.HTMLParser(encoding="utf-8", recover=True)
            doc = _LXML_HTML.fromstring(sanitized, parser=parser)
            for el in doc.iter():
                ln = _element_local_tag(el.tag)
                if ln == "base":
                    bh = el.get("href")
                    if bh and str(bh).strip() and not base_href:
                        base_href = str(bh).strip()
                elif ln in ("a", "area"):
                    v = el.get("href")
                    if v and str(v).strip():
                        raw_hrefs.append(html_module.unescape(str(v).strip()))
        except Exception:
            _LOG.debug("lxml href extraction failed", exc_info=True)
            raw_hrefs = []

    if not raw_hrefs and _BS4 is not None:
        try:
            soup = _BS4(sanitized, "html.parser")
            btag = soup.find("base", href=True)
            if Bs4Tag is not None and isinstance(btag, Bs4Tag):
                bh = btag.get("href")
                if bh and str(bh).strip():
                    base_href = str(bh).strip()
            for tag in ("a", "area"):
                for el in soup.find_all(tag, href=True):
                    if Bs4Tag is not None and not isinstance(el, Bs4Tag):
                        continue
                    href_attr = el.get("href")
                    if href_attr and str(href_attr).strip():
                        raw_hrefs.append(html_module.unescape(str(href_attr).strip()))
        except Exception:
            _LOG.debug("bs4 href extraction failed", exc_info=True)

    resolved: List[str] = []
    for raw in raw_hrefs:
        joined = urljoin(base_href, raw) if base_href else raw
        if joined.strip():
            resolved.append(joined.strip())
    return resolved


def extract_urls_from_text(text: str) -> List[str]:
    """Extract validated ``http``/``https`` URLs from unstructured text."""
    out: List[str] = []
    for c in extract_urls_with_linkify(text):
        n = normalize_http_url(c)
        if n:
            out.append(n)
    return out


def extract_urls_from_plain_and_html(plain_body: str, html_body: str = "") -> List[str]:
    """Merge validated URLs from HTML ``href`` values and plain-text linkification."""
    seen: set[str] = set()
    ordered: List[str] = []

    def add(u: str) -> None:
        if u not in seen:
            seen.add(u)
            ordered.append(u)

    for u in extract_hrefs_from_html(html_body):
        n = normalize_http_url(u)
        if n:
            add(n)
    for u in extract_urls_from_text(plain_body or ""):
        add(u)
    return ordered


def parse_url_components(url: str) -> Dict[str, Any]:
    if not url:
        return {
            "full_url": "",
            "domain": "",
            "stem": "",
            "scheme": "",
        }

    raw_in = str(url).strip()

    try:
        s = _strip_wrapping_punct(refang_url_like_schemes(raw_in))
        if not s:
            return {
                "full_url": "",
                "domain": "",
                "stem": "",
                "scheme": "",
            }

        canonical = normalize_http_url(s)

        if canonical:
            parsed = urlparse(canonical)
            full_url_out = canonical
        else:
            parsing_url = s
            low = s.lower()
            if not low.startswith(("http://", "https://")):
                if s.startswith("//"):
                    parsing_url = "https:" + s
                else:
                    parsing_url = "http://" + s
            parsed = _safe_urlparse(parsing_url)
            if parsed is None:
                return {
                    "full_url": raw_in,
                    "domain": "",
                    "stem": "",
                    "scheme": "",
                }
            scheme_l = (parsed.scheme or "").lower()
            nl = parsed.netloc.lower() if parsed.netloc else ""
            normalized = urlunparse(
                (
                    scheme_l,
                    nl,
                    parsed.path,
                    parsed.params,
                    parsed.query,
                    parsed.fragment,
                )
            )
            full_url_out = normalized

        try:
            host = (parsed.hostname or "").strip().lower()
        except ValueError:
            host = ""
        domain = host
        if not domain and parsed.netloc:
            domain = parsed.netloc.lower()

        stem_parts: List[str] = []
        if parsed.path:
            stem_parts.append(parsed.path)
        if parsed.params:
            stem_parts.append(f";{parsed.params}")
        if parsed.query:
            stem_parts.append(f"?{parsed.query}")
        if parsed.fragment:
            stem_parts.append(f"#{parsed.fragment}")

        stem = "".join(stem_parts) if stem_parts else "/"

        return {
            "full_url": full_url_out,
            "domain": domain,
            "stem": stem,
            "scheme": (parsed.scheme or "").lower() if parsed.scheme else "",
        }
    except Exception:
        return {
            "full_url": raw_in,
            "domain": "",
            "stem": "",
            "scheme": "",
        }


def extract_and_parse_urls(text: str) -> List[Dict[str, Any]]:
    urls = extract_urls_from_text(text)
    return [parse_url_components(url) for url in urls]


def deduplicate_urls(url_list: List[str]) -> List[str]:
    seen = set()
    result = []
    for url in url_list:
        normalized = url.lower().rstrip("/")
        if normalized not in seen:
            seen.add(normalized)
            result.append(url)
    return result


def collect_urls_for_misp_event_attributes(attributes: List[Dict[str, Any]]) -> List[str]:
    """
    Re-derive ``http`` / ``https`` URLs from MISP-style ``Attribute`` rows (same field names
    as :func:`preprocessing.misp_converter.incidents_to_misp_events`).

    Uses ``body`` + optional raw ``html`` string + ``header_List-Unsubscribe``. If ``html``
    is only a structure dict (no raw HTML), href extraction is skipped; plain text and
    header URLs are still refreshed. Angle brackets in ``List-Unsubscribe`` (RFC 2369)
    are neutralized so linkify-it can see embedded ``https`` URIs.
    """
    body = ""
    html_raw = ""
    list_unsub = ""
    for attr in attributes:
        if not isinstance(attr, dict):
            continue
        t = attr.get("type")
        v = attr.get("value")
        if t == "body" and isinstance(v, str):
            body = v
        elif t == "html" and isinstance(v, str) and v.strip():
            html_raw = v
        elif t == "header_List-Unsubscribe" and isinstance(v, str):
            list_unsub = v.replace("<", " ").replace(">", " ")

    from_content = extract_urls_from_plain_and_html(body, html_raw)
    from_headers = extract_urls_from_text(list_unsub) if list_unsub.strip() else []
    return deduplicate_urls(from_content + from_headers)


def refresh_urls_in_misp_events(events: List[Dict[str, Any]]) -> int:
    """
    Update the ``url`` attribute list on each ``Event`` in place.

    Returns the number of events that had an ``Attribute`` list and were processed
    (including those whose URL list became empty).
    """
    processed = 0
    for item in events:
        if not isinstance(item, dict):
            continue
        event = item.get("Event")
        if not isinstance(event, dict):
            continue
        attrs = event.get("Attribute")
        if not isinstance(attrs, list):
            continue
        new_urls = collect_urls_for_misp_event_attributes(attrs)
        found = False
        for attr in attrs:
            if isinstance(attr, dict) and attr.get("type") == "url":
                attr["value"] = new_urls
                found = True
                break
        if not found:
            attrs.append({"type": "url", "value": new_urls})
        processed += 1
    return processed


def load_misp_events_from_json_file(path: str | Path) -> List[Dict[str, Any]]:
    """Load MISP export JSON: top-level list of ``{\"Event\": ...}`` or ``{\"response\": [...]}``."""
    raw = Path(path).read_text(encoding="utf-8")
    data = json.loads(raw)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        resp = data.get("response")
        if isinstance(resp, list):
            return resp
    raise ValueError(
        "Unsupported MISP JSON shape: expected a list of events or an object with a list "
        "'response' (MISP REST style)."
    )


def reparse_misp_json_urls_file(input_path: str | Path, output_path: str | Path | None = None) -> Tuple[int, Path]:
    """
    Reload ``misp.json``, recompute each email's ``url`` attributes, and write the result.

    URLs are stored normalized (``https://``) in memory; :func:`preprocessing.misp_converter.write_misp_events_securely`
    applies the same defanging used by the rest of the pipeline when writing JSON.

    Args:
        input_path: Source ``misp.json`` (list of ``{\"Event\": ...}``).
        output_path: Destination path; defaults to ``input_path`` (atomic replace).

    Returns:
        ``(number_of_events_processed, output_path)``.
    """
    from preprocessing.misp_converter import write_misp_events_securely

    inp = Path(input_path)
    out = Path(output_path) if output_path is not None else inp
    events = load_misp_events_from_json_file(inp)
    n = refresh_urls_in_misp_events(events)
    write_misp_events_securely(events, str(out))
    return n, out


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Re-extract URLs from an existing MISP JSON file (body + optional raw html "
            "+ List-Unsubscribe), without re-running full preprocessing."
        )
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Path to misp.json (list of Event objects)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output path (default: overwrite input atomically)",
    )
    args = parser.parse_args()
    try:
        n_events, written = reparse_misp_json_urls_file(args.input, args.output)
    except Exception as exc:  # pragma: no cover - CLI surface
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
    print(f"Processed {n_events} events. Wrote: {written}")
