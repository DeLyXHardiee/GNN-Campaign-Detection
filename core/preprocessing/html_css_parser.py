"""
HTML/CSS parsing for email preprocessing.

HTML parsing uses a robust, fault-tolerant approach suitable for malformed real-world
email HTML (e.g. Outlook conditional comments, SGML marked sections). The previous
implementation used Python's built-in html.parser.HTMLParser, which raises
AssertionError on invalid SGML marked sections (e.g. "unknown status keyword 'e'
in marked section"). We now prefer lxml.html (or BeautifulSoup with lxml/html5lib),
which recover from broken markup, and we never let a single bad email crash the pipeline.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import hashlib
import math

# --- Optional robust HTML backends (prefer lxml, then BeautifulSoup) ---
try:
    from lxml import html as lxml_html

    _LXML_AVAILABLE = True
except ImportError:
    _LXML_AVAILABLE = False
    lxml_html = None  # type: ignore[misc, assignment]

try:
    from bs4 import BeautifulSoup

    _BS4_AVAILABLE = True
except ImportError:
    _BS4_AVAILABLE = False
    BeautifulSoup = None  # type: ignore[misc, assignment]

# --- Lightweight regexes ---
_HEX_COLOR_RE = re.compile(r"(?i)#(?:[0-9a-f]{3}|[0-9a-f]{6})\b")
_CLASS_SELECTOR_RE = re.compile(r"\.([_a-zA-Z][-_a-zA-Z0-9]*)")

# SGML-style marked sections (e.g. <![[if ...]]>) cause stdlib HTMLParser to raise
# "unknown status keyword 'e' in marked section". Strip them before parsing.
_MARKED_SECTION_RE = re.compile(r"<!\[\[.*?\]\]>", re.DOTALL | re.IGNORECASE)

# Optional: strip other known-bad declaration-like fragments that can confuse parsers
# (Outlook conditional comments already look like comments; we only strip <![[ ]]> here
# to avoid removing useful structure.)
_FINGERPRINT_EXCLUDED_TAGS = {"script", "style"}

_LOG = logging.getLogger(__name__)

# Snippet length for error logs (avoid dumping huge bodies)
_HTML_SNIPPET_LEN = 500


def _fast_hash_64(token: str) -> int:
    digest = hashlib.blake2b(token.encode("utf-8", "ignore"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big", signed=False)


def _simhash_64(tokens: List[str]) -> str:
    if not tokens:
        return "0000000000000000"

    vector = [0] * 64
    counts = Counter(tokens)

    for token, weight in counts.items():
        hashed = _fast_hash_64(token)
        for bit in range(64):
            if (hashed >> bit) & 1:
                vector[bit] += weight
            else:
                vector[bit] -= weight

    fingerprint = 0
    for bit in range(64):
        if vector[bit] >= 0:
            fingerprint |= 1 << bit

    return f"{fingerprint:016x}"


def get_empty_html_structure() -> Dict[str, Any]:
    """Return the same schema as a successful parse, with empty/zero values.
    For use by callers when they need a safe fallback (e.g. body extraction failed).
    """
    return _empty_html_structure()


def _empty_html_structure() -> Dict[str, Any]:
    """Return the same schema as a successful parse, with empty/zero values.
    Used when input is empty or when parsing fails so downstream never sees missing keys.
    """
    return {
        "tag_counts": {},
        "tree_stats": {
            "total_elements": 0,
            "max_depth": 0,
            "avg_depth": 0.0,
            "forms": 0,
            "password_fields": 0,
            "hidden_elements": 0,
            "external_scripts": 0,
            "links": 0,
            "images": 0,
            "link_ratio": 0.0,
            "image_ratio": 0.0,
        },
        "structure_fingerprint": _simhash_64([]),
    }


def _sanitize_html_for_parsing(html_text: str) -> str:
    """Lightweight sanitization to improve robustness against malformed email HTML.
    Strips SGML marked sections (e.g. <![[...]]>) that trigger stdlib parser assertions.
    Does not strip normal HTML comments or alter DOM structure used for phishing features.
    """
    if not html_text:
        return html_text
    return _MARKED_SECTION_RE.sub("", html_text)


def _attr_map_lxml(element: Any) -> Dict[str, str]:
    """Build lowercased attr dict from an lxml element."""
    out: Dict[str, str] = {}
    for name, value in (element.attrib or {}).items():
        if name and isinstance(name, str):
            out[name.lower()] = (value or "").lower()
    return out


def _attr_map_bs4(tag: Any) -> Dict[str, str]:
    """Build lowercased attr dict from a BeautifulSoup tag."""
    out: Dict[str, str] = {}
    for name, value in (tag.attrs or {}).items():
        if isinstance(value, list):
            value = " ".join(str(v) for v in value)
        out[str(name).lower()] = (str(value) if value else "").lower()
    return out


def _counts_and_edges_from_lxml(
    root: Any,
) -> Tuple[Counter, List[str], List[int], int, int, int, int, int, int, int]:
    """Walk lxml tree and collect tag counts, parent->child edges (for fingerprint),
    depths, and feature counts. Excludes script/style from fingerprint edges.
    """
    tag_counter: Counter = Counter()
    edge_tokens: List[str] = []
    depth_values: List[int] = []
    counts = {
        "form": 0,
        "password": 0,
        "hidden": 0,
        "external_script": 0,
        "link": 0,
        "image": 0,
    }

    def walk(element: Any, parent_tag: Optional[str], depth: int) -> None:
        # lxml: element.tag can be a Cython function for Comment/PI nodes, not a string
        raw_tag = getattr(element, "tag", None)
        if not isinstance(raw_tag, str):
            for child in element:
                walk(child, parent_tag, depth + 1)
            return
        tag = raw_tag.lower().strip()
        if not tag:
            return
        tag_counter[tag] += 1
        depth_values.append(depth)

        if parent_tag is not None and tag not in _FINGERPRINT_EXCLUDED_TAGS:
            edge_tokens.append(f"{parent_tag}>{tag}")

        attrs = _attr_map_lxml(element)

        if tag == "form":
            counts["form"] += 1
        if tag == "input":
            if attrs.get("type") == "password":
                counts["password"] += 1
            if attrs.get("type") == "hidden":
                counts["hidden"] += 1
        if tag == "script":
            src = attrs.get("src", "")
            if src.startswith(("http://", "https://", "//")):
                counts["external_script"] += 1
        if tag == "a":
            counts["link"] += 1
        if tag == "img":
            counts["image"] += 1
        style = attrs.get("style", "")
        if "display:none" in style or "visibility:hidden" in style:
            counts["hidden"] += 1

        for child in element:
            walk(child, tag, depth + 1)

    walk(root, None, 1)
    return (
        tag_counter,
        edge_tokens,
        depth_values,
        counts["form"],
        counts["password"],
        counts["hidden"],
        counts["external_script"],
        counts["link"],
        counts["image"],
    )


def _counts_and_edges_from_bs4(
    soup: Any,
) -> Tuple[Counter, List[str], List[int], int, int, int, int, int, int, int]:
    """Walk BeautifulSoup tree and collect same stats as lxml path."""
    tag_counter: Counter = Counter()
    edge_tokens: List[str] = []
    depth_values: List[int] = []
    form_count = 0
    password_count = 0
    hidden_count = 0
    external_script_count = 0
    link_count = 0
    image_count = 0

    for tag in soup.find_all(True):
        raw_name = getattr(tag, "name", None)
        if not isinstance(raw_name, str):
            continue
        name = raw_name.lower().strip()
        if not name:
            continue
        tag_counter[name] += 1

        depth = 1
        p = tag.parent
        while getattr(p, "name", None) is not None and p != soup:
            depth += 1
            p = getattr(p, "parent", None)
        depth_values.append(depth)

        parent = tag.parent
        if parent and parent != soup:
            raw_parent_name = getattr(parent, "name", None)
            parent_name = raw_parent_name.lower().strip() if isinstance(raw_parent_name, str) else ""
            if parent_name and name not in _FINGERPRINT_EXCLUDED_TAGS:
                edge_tokens.append(f"{parent_name}>{name}")

        attrs = _attr_map_bs4(tag)

        if name == "form":
            form_count += 1
        if name == "input":
            if attrs.get("type") == "password":
                password_count += 1
            if attrs.get("type") == "hidden":
                hidden_count += 1
        if name == "script":
            src = attrs.get("src", "")
            if src.startswith(("http://", "https://", "//")):
                external_script_count += 1
        if name == "a":
            link_count += 1
        if name == "img":
            image_count += 1
        style = attrs.get("style", "")
        if "display:none" in style or "visibility:hidden" in style:
            hidden_count += 1

    return (
        tag_counter,
        edge_tokens,
        depth_values,
        form_count,
        password_count,
        hidden_count,
        external_script_count,
        link_count,
        image_count,
    )


def _build_result(
    tag_counter: Counter,
    edge_tokens: List[str],
    depth_values: List[int],
    form_count: int,
    password_count: int,
    hidden_count: int,
    external_script_count: int,
    link_count: int,
    image_count: int,
) -> Dict[str, Any]:
    """Build the standard parse result dict from collected stats."""
    total = len(depth_values)
    max_depth = max(depth_values) if depth_values else 0
    avg_depth = sum(depth_values) / len(depth_values) if depth_values else 0.0

    return {
        "tag_counts": dict(tag_counter),
        "tree_stats": {
            "total_elements": total,
            "max_depth": max_depth,
            "avg_depth": round(avg_depth, 4),
            "forms": form_count,
            "password_fields": password_count,
            "hidden_elements": hidden_count,
            "external_scripts": external_script_count,
            "links": link_count,
            "images": image_count,
            "link_ratio": link_count / total if total else 0.0,
            "image_ratio": image_count / total if total else 0.0,
        },
        "structure_fingerprint": _simhash_64(edge_tokens),
    }


def _parse_with_lxml(html_text: str) -> Dict[str, Any]:
    """Parse with lxml.html (recover mode by default). Returns same schema as parse_html_fast."""
    doc = lxml_html.fromstring(html_text)
    (
        tag_counter,
        edge_tokens,
        depth_values,
        form_count,
        password_count,
        hidden_count,
        external_script_count,
        link_count,
        image_count,
    ) = _counts_and_edges_from_lxml(doc)
    return _build_result(
        tag_counter,
        edge_tokens,
        depth_values,
        form_count,
        password_count,
        hidden_count,
        external_script_count,
        link_count,
        image_count,
    )


def _parse_with_bs4(html_text: str) -> Dict[str, Any]:
    """Parse with BeautifulSoup using lxml or html5lib for tolerance to bad markup.
    When lxml is not installed, html5lib is used (pip install html5lib).
    """
    parser = "lxml" if _LXML_AVAILABLE else "html5lib"
    soup = BeautifulSoup(html_text, parser)
    (
        tag_counter,
        edge_tokens,
        depth_values,
        form_count,
        password_count,
        hidden_count,
        external_script_count,
        link_count,
        image_count,
    ) = _counts_and_edges_from_bs4(soup)
    return _build_result(
        tag_counter,
        edge_tokens,
        depth_values,
        form_count,
        password_count,
        hidden_count,
        external_script_count,
        link_count,
        image_count,
    )


def parse_html_fast(
    html_text: str,
    sample_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Parse HTML into tag counts, tree stats, and structure fingerprint.

    Uses a robust parser (lxml.html preferred, else BeautifulSoup with lxml/html5lib)
    so malformed email HTML does not crash preprocessing. On any failure, returns
    a safe fallback structure with empty/zero values and logs the error.

    Output schema (unchanged for compatibility):
      - tag_counts: dict of tag name -> count
      - tree_stats: total_elements, max_depth, avg_depth, forms, password_fields,
        hidden_elements, external_scripts, links, images, link_ratio, image_ratio
      - structure_fingerprint: 16-char hex simhash of parent->child edge tokens
        (script/style excluded)

    Args:
        html_text: Raw HTML string (e.g. email body).
        sample_id: Optional identifier for the email/sample, included in error logs.
    """
    if not (html_text and html_text.strip()):
        return _empty_html_structure()

    sanitized = _sanitize_html_for_parsing(html_text)
    snippet = (sanitized[: _HTML_SNIPPET_LEN] + "…") if len(sanitized) > _HTML_SNIPPET_LEN else sanitized

    try:
        if _LXML_AVAILABLE:
            return _parse_with_lxml(sanitized)
        if _BS4_AVAILABLE:
            return _parse_with_bs4(sanitized)
        # No robust backend installed: return empty and log once so user can install lxml/beautifulsoup4
        _LOG.warning(
            "HTML parsing skipped: neither lxml nor beautifulsoup4 available. "
            "Install lxml (pip install lxml) or beautifulsoup4 (pip install beautifulsoup4) for robust parsing."
        )
        return _empty_html_structure()
    except Exception as e:
        _LOG.warning(
            "HTML parse failed for email/sample %s: %s. Snippet: %s",
            sample_id if sample_id is not None else "(unknown)",
            e,
            snippet[:200].replace("\n", " "),
            exc_info=False,
        )
        return _empty_html_structure()


def parse_css_fast(css_text: str) -> Dict[str, Any]:
    if not css_text:
        return {}

    colors = Counter(_HEX_COLOR_RE.findall(css_text.lower()))

    primary_color = ""
    if colors:
        primary_color = colors.most_common(1)[0][0]

    class_counter = Counter(
        cls.lower() for cls in _CLASS_SELECTOR_RE.findall(css_text)
    )

    total_classes = sum(class_counter.values())
    entropy = 0.0
    if total_classes:
        for count in class_counter.values():
            p = count / total_classes
            entropy -= p * math.log2(p)

    return {
        "style_features": {
            "unique_color_count": len(colors),
            "primary_color": primary_color,
            "uses_position_absolute": "position:absolute" in css_text.lower(),
            "uses_z_index": "z-index" in css_text.lower(),
            "uses_media_queries": "@media" in css_text.lower(),
            "unique_class_count": len(class_counter),
            "class_entropy": round(entropy, 4),
        }
    }
