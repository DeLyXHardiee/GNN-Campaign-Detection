from __future__ import annotations

import html
from email import policy
from email.parser import BytesParser
import re
from typing import List


def _decode_payload(raw_payload: bytes | str | None, charset: str | None) -> str:
    if raw_payload is None:
        return ""
    if isinstance(raw_payload, str):
        return raw_payload
    if isinstance(raw_payload, bytes):
        if charset:
            try:
                return raw_payload.decode(charset, errors="replace")
            except Exception:
                pass
        try:
            return raw_payload.decode("utf-8", errors="replace")
        except Exception:
            return raw_payload.decode("latin-1", errors="replace")
    return ""


def _extract_after_header_block(raw_bytes: bytes) -> str:
    # RFC822 headers and body are separated by first blank line.
    # This fallback keeps body content while stripping top-level headers.
    for delimiter in (b"\r\n\r\n", b"\n\n"):
        idx = raw_bytes.find(delimiter)
        if idx != -1:
            body_bytes = raw_bytes[idx + len(delimiter) :]
            return _decode_payload(body_bytes, None).strip()
    return _decode_payload(raw_bytes, None).strip()


def _defang_url_like_text(text: str) -> str:
    # Neutralize common clickable URI schemes while preserving readable content.
    out = re.sub(r"(?i)\bhttps://", "hxxps://", text)
    out = re.sub(r"(?i)\bhttp://", "hxxp://", out)
    out = re.sub(r"(?i)\bftp://", "fxp://", out)
    out = re.sub(r"(?i)\bmailto:", "mailt0:", out)
    return out


def _inert_html_text(html_text: str) -> str:
    # Render HTML as inert text so no elements/links are clickable.
    return _defang_url_like_text(html.escape(html_text, quote=False))


def extract_body_without_headers(raw_bytes: bytes) -> str:
    """Return message body text without top-level headers.

    Includes both plain and HTML parts when available.
    """
    if not raw_bytes:
        return ""

    try:
        message = BytesParser(policy=policy.default).parsebytes(raw_bytes)
        parts: List[str] = []

        if message.is_multipart():
            for part in message.walk():
                if part.is_multipart():
                    continue
                content_disposition = (part.get_content_disposition() or "").lower()
                if content_disposition == "attachment":
                    continue
                content_type = (part.get_content_type() or "").lower()
                if content_type not in {"text/plain", "text/html"}:
                    continue
                payload = part.get_payload(decode=True)
                text = _decode_payload(payload, part.get_content_charset()).strip()
                if content_type == "text/html" and text:
                    text = _inert_html_text(text)
                elif content_type == "text/plain" and text:
                    text = _defang_url_like_text(text)
                if text:
                    parts.append(text)
        else:
            content_type = (message.get_content_type() or "").lower()
            payload = message.get_payload(decode=True)
            text = _decode_payload(payload, message.get_content_charset()).strip()
            if content_type == "text/html" and text:
                text = _inert_html_text(text)
            elif text:
                text = _defang_url_like_text(text)
            if text:
                parts.append(text)

        if parts:
            return "\n\n".join(parts).strip()
    except Exception:
        pass

    return _inert_html_text(_extract_after_header_block(raw_bytes))


__all__ = ["extract_body_without_headers"]
