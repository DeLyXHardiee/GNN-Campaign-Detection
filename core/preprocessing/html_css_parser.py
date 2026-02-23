from __future__ import annotations

from collections import Counter
from html.parser import HTMLParser
import hashlib
import math
import re
from typing import Dict, Any, List


# --- Lightweight regexes ---
_HEX_COLOR_RE = re.compile(r"(?i)#(?:[0-9a-f]{3}|[0-9a-f]{6})\b")
_CLASS_SELECTOR_RE = re.compile(r"\.([_a-zA-Z][-_a-zA-Z0-9]*)")

_FINGERPRINT_EXCLUDED_TAGS = {"script", "style"}


# --- Fast 64-bit hash (non-cryptographic use) ---
def _fast_hash_64(token: str) -> int:
    # Much faster than SHA256
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


class _FastHTMLElementParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()

        self.tag_counter = Counter()
        self.edge_tokens: List[str] = []
        self.depth_values: List[int] = []
        self._stack: List[str] = []

        self.total_elements = 0
        self.form_count = 0
        self.password_count = 0
        self.hidden_count = 0
        self.external_script_count = 0
        self.link_count = 0
        self.image_count = 0

    def handle_starttag(self, tag, attrs):
        tag = (tag or "").lower().strip()
        if not tag:
            return

        self.total_elements += 1
        self.tag_counter[tag] += 1

        current_depth = len(self._stack) + 1
        self.depth_values.append(current_depth)

        # Parent → child structural edge
        if self._stack:
            parent = self._stack[-1]
            if tag not in _FINGERPRINT_EXCLUDED_TAGS:
                self.edge_tokens.append(f"{parent}>{tag}")

        attr_map = {k.lower(): (v or "").lower() for k, v in attrs}

        if tag == "form":
            self.form_count += 1

        if tag == "input":
            if attr_map.get("type") == "password":
                self.password_count += 1
            if attr_map.get("type") == "hidden":
                self.hidden_count += 1

        if tag == "script":
            src = attr_map.get("src", "")
            if src.startswith(("http://", "https://", "//")):
                self.external_script_count += 1

        if tag == "a":
            self.link_count += 1

        if tag == "img":
            self.image_count += 1

        # Basic hidden detection
        style = attr_map.get("style", "")
        if "display:none" in style or "visibility:hidden" in style:
            self.hidden_count += 1

        self._stack.append(tag)

    def handle_endtag(self, tag):
        tag = (tag or "").lower().strip()
        if self._stack and tag in self._stack:
            while self._stack and self._stack[-1] != tag:
                self._stack.pop()
            if self._stack:
                self._stack.pop()


def parse_html_fast(html_text: str) -> Dict[str, Any]:
    if not html_text:
        return {}

    parser = _FastHTMLElementParser()
    parser.feed(html_text)
    parser.close()

    total = parser.total_elements
    max_depth = max(parser.depth_values) if parser.depth_values else 0
    avg_depth = sum(parser.depth_values) / len(parser.depth_values) if parser.depth_values else 0.0

    return {
        "tag_counts": dict(parser.tag_counter),
        "tree_stats": {
            "total_elements": total,
            "max_depth": max_depth,
            "avg_depth": round(avg_depth, 4),
            "forms": parser.form_count,
            "password_fields": parser.password_count,
            "hidden_elements": parser.hidden_count,
            "external_scripts": parser.external_script_count,
            "links": parser.link_count,
            "images": parser.image_count,
            "link_ratio": parser.link_count / total if total else 0.0,
            "image_ratio": parser.image_count / total if total else 0.0,
        },
        "structure_fingerprint": _simhash_64(parser.edge_tokens),
    }


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