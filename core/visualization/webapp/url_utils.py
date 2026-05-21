"""URL helpers for the visualization API (defanged MISP + body extraction)."""
from __future__ import annotations

import ast
import re
from typing import Any

_URL_PATTERN = re.compile(
    r"(?:https?://|hxxps?://|www\.)[^\s'\"<>]+",
    re.IGNORECASE,
)


def refang_url_string(text: str) -> str:
    if not text or not isinstance(text, str):
        return text
    out = re.sub(r"(?i)\bhxxps://", "https://", text)
    out = re.sub(r"(?i)\bhxxp://", "http://", out)
    return out


def defang_url_string(text: str) -> str:
    """Neutralize clickable schemes for safe display (not for extraction)."""
    if not text or not isinstance(text, str):
        return text
    out = re.sub(r"(?i)\bhttps://", "hxxps://", text)
    out = re.sub(r"(?i)\bhttp://", "hxxp://", out)
    out = re.sub(r"(?i)\bftp://", "fxp://", out)
    out = re.sub(r"(?i)\bmailto:", "mailt0:", out)
    return out


def extract_urls_from_text(text: str) -> list[str]:
    if not text:
        return []
    cleaned: list[str] = []
    for url in _URL_PATTERN.findall(refang_url_string(text)):
        u = refang_url_string(url.rstrip(".,;:!?)"))
        if u:
            cleaned.append(u)
    return cleaned


def _expand_url_value(raw: Any) -> list[str]:
    if isinstance(raw, list):
        out: list[str] = []
        for item in raw:
            out.extend(_expand_url_value(item))
        return out
    text = str(raw).strip()
    if not text:
        return []
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, (list, tuple)):
                out = []
                for item in parsed:
                    out.extend(_expand_url_value(item))
                return out
        except (ValueError, SyntaxError):
            pass
    return extract_urls_from_text(text)


def urls_for_email_row(row: dict[str, Any]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []

    def add(raw: Any) -> None:
        for candidate in _expand_url_value(raw):
            u = refang_url_string(candidate.strip())
            if not u or u in seen:
                continue
            seen.add(u)
            out.append(u)

    for field in ("urls", "email_urls"):
        for u in row.get(field) or []:
            add(u)

    for field in ("body", "email_info"):
        text = row.get(field) or ""
        if isinstance(text, str) and text.strip():
            for u in extract_urls_from_text(text):
                add(u)

    return sorted((defang_url_string(u) for u in out), key=str.casefold)
