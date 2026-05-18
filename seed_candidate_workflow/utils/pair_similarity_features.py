"""
Shared pair similarity features for training and low-band feature discovery.

Definitions match ``pair_low_band_feature_discovery`` analysis (path-token Jaccard,
normalized sender local-part similarity).
"""

from __future__ import annotations

import difflib
import re
from typing import Any

_ROOT_STEM = "/"
_DIGITS_RE = re.compile(r"\d+")
_SENDER_DISPLAY_RE = re.compile(r"^(.+?)\s*<([^>]+)>$")


def jaccard_similarity(a: set[str], b: set[str]) -> float:
    """Jaccard index; returns 0.0 when both sets are empty."""
    if not a and not b:
        return 0.0
    u = a | b
    if not u:
        return 0.0
    return float(len(a & b) / len(u))


def parse_sender_parts(sender: str) -> tuple[str, str, str]:
    """Return (local_part, domain, display_name)."""
    s = str(sender or "").strip()
    m = _SENDER_DISPLAY_RE.match(s)
    if m:
        display = m.group(1).strip().strip('"')
        addr = m.group(2).strip().lower()
    else:
        display = ""
        addr = s.lower()
    if "@" in addr:
        local, dom = addr.split("@", 1)
    else:
        local, dom = addr, ""
    return local, dom, display


def normalize_sender_localpart(local: str) -> str:
    return _DIGITS_RE.sub("", str(local or "").lower())


def first_sender_string(node: dict[str, Any]) -> str:
    ss = node.get("sender_set") or set()
    return str(next(iter(ss), "")) if ss else ""


def parse_url_path_tokens(url: str) -> tuple[str, list[str], int]:
    """Return (registrable_domain, path_tokens, path_depth)."""
    from core.feature_set_extraction.url_extraction_utils import parse_url_host_and_registrable_domain
    from core.preprocessing.utils.url_extractor import parse_url_components

    u = str(url or "").strip()
    if not u:
        return "", [], 0
    _host, reg, ok = parse_url_host_and_registrable_domain(u)
    reg = reg.lower() if ok else ""
    comp = parse_url_components(u)
    stem = str(comp.get("stem") or "").strip()
    parts = [p for p in stem.split("/") if p and p != _ROOT_STEM]
    depth = len(parts)
    tokens: list[str] = []
    for p in parts:
        p_norm = re.sub(r"\d{4,}", "<id>", p.lower())
        for t in re.split(r"[/_.-]+", p_norm):
            t = t.strip()
            if t and t not in ("<id>",):
                tokens.append(t)
    return reg, tokens, depth


def nontrivial_stems(stems: set[str]) -> set[str]:
    return {s for s in stems if s and s != _ROOT_STEM}


def path_token_set_for_node(node: dict[str, Any]) -> set[str]:
    """Union of URL path tokens and nontrivial stem tokens for one email."""
    url_tokens: list[str] = []
    for u in node.get("url_set") or set():
        _reg, toks, _dep = parse_url_path_tokens(str(u))
        url_tokens.extend(toks)
    path_tokens = set(url_tokens)
    stem_tokens: set[str] = set()
    for st in nontrivial_stems(node.get("stem_set") or set()):
        for t in re.split(r"[/_.-]+", str(st).lower()):
            if t:
                stem_tokens.add(t)
    return path_tokens | stem_tokens


def path_token_jaccard_combined_for_nodes(
    node_a: dict[str, Any],
    node_b: dict[str, Any],
) -> float:
    """
    Jaccard similarity of combined URL path + nontrivial stem path tokens.

    Same definition as feature-discovery ``path_token_jaccard_combined``.
    """
    return jaccard_similarity(path_token_set_for_node(node_a), path_token_set_for_node(node_b))


def sender_localpart_norm_similarity(local_a: str, local_b: str) -> float:
    """
    SequenceMatcher ratio on digit-stripped lowercased local-parts.

    Same definition as feature-discovery ``sender_localpart_norm_jaccard``.
    Returns 0.0 when both normalized local-parts are empty.
    """
    na = normalize_sender_localpart(local_a)
    nb = normalize_sender_localpart(local_b)
    if not na and not nb:
        return 0.0
    if not na or not nb:
        return 0.0
    return float(difflib.SequenceMatcher(None, na, nb).ratio())


def sender_localpart_norm_jaccard_for_nodes(
    node_a: dict[str, Any],
    node_b: dict[str, Any],
) -> float:
    """Normalized local-part similarity using the first sender on each email."""
    la, _, _ = parse_sender_parts(first_sender_string(node_a))
    lb, _, _ = parse_sender_parts(first_sender_string(node_b))
    return sender_localpart_norm_similarity(la, lb)
