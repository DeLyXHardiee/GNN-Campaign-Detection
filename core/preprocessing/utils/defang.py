"""
Defang URL-like schemes so they are not clickable in saved output.
Used by preprocessing (MISP output), graph (meta JSON), and feature-set extraction.
"""
from __future__ import annotations

import re
from typing import Any


def defang_url_string(text: str) -> str:
    """Neutralize common clickable URI schemes while preserving readable content."""
    if not text or not isinstance(text, str):
        return text
    out = re.sub(r"(?i)\bhttps://", "hxxps://", text)
    out = re.sub(r"(?i)\bhttp://", "hxxp://", out)
    out = re.sub(r"(?i)\bftp://", "fxp://", out)
    out = re.sub(r"(?i)\bmailto:", "mailt0:", out)
    return out


def refang_url_string(text: str) -> str:
    """Restore defanged URI schemes for extraction and display."""
    if not text or not isinstance(text, str):
        return text
    out = re.sub(r"(?i)\bhxxps://", "https://", text)
    out = re.sub(r"(?i)\bhxxp://", "http://", out)
    out = re.sub(r"(?i)\bfxp://", "ftp://", out)
    out = re.sub(r"(?i)\bmailt0:", "mailto:", out)
    return out


def sanitize_for_json(obj: Any) -> Any:
    """Recursively defang all string values in a structure (for safe JSON/file output)."""
    if isinstance(obj, str):
        return defang_url_string(obj)
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]
    return obj
