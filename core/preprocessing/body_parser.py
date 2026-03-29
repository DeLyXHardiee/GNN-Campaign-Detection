from __future__ import annotations

from email import policy
from email.parser import BytesParser
from html.parser import HTMLParser
import html
import re
from typing import List, Tuple


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


class _HTMLToTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._parts: List[str] = []
        self._ignore_depth = 0

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, str | None]]) -> None:
        lowered = tag.lower()
        if lowered in {"style", "script"}:
            self._ignore_depth += 1
            return
        if lowered in {"br", "p", "div", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6"}:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        lowered = tag.lower()
        if lowered in {"style", "script"} and self._ignore_depth > 0:
            self._ignore_depth -= 1
            return
        if lowered in {"p", "div", "li", "tr"}:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._ignore_depth > 0:
            return
        if data:
            self._parts.append(data)

    def get_text(self) -> str:
        text = "".join(self._parts)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()


def _html_to_text(html_text: str) -> str:
    parser = _HTMLToTextParser()
    parser.feed(html_text)
    parser.close()
    text = html.unescape(parser.get_text())
    return _defang_url_like_text(text)


def _extract_css_from_html(html_text: str) -> str:
    css_parts: List[str] = []
    for match in re.finditer(r"(?is)<style\b[^>]*>(.*?)</style>", html_text):
        css_value = (match.group(1) or "").strip()
        if css_value:
            css_parts.append(css_value)

    for match in re.finditer(r'(?is)\sstyle\s*=\s*"([^"]*)"', html_text):
        css_value = (match.group(1) or "").strip()
        if css_value:
            css_parts.append(css_value)
    for match in re.finditer(r"(?is)\sstyle\s*=\s*'([^']*)'", html_text):
        css_value = (match.group(1) or "").strip()
        if css_value:
            css_parts.append(css_value)

    if not css_parts:
        return ""
    return _defang_url_like_text("\n\n".join(css_parts))


def extract_body_html_css_without_headers(raw_bytes: bytes) -> Tuple[str, str, str]:
    """Return (plain_body_text, html_body_text, css_text) without top-level headers."""
    if not raw_bytes:
        return "", "", ""

    try:
        message = BytesParser(policy=policy.default).parsebytes(raw_bytes)
        plain_parts: List[str] = []
        html_parts: List[str] = []

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
                if text:
                    if content_type == "text/plain":
                        plain_parts.append(_defang_url_like_text(text))
                    elif content_type == "text/html":
                        html_parts.append(_defang_url_like_text(text))
        else:
            content_type = (message.get_content_type() or "").lower()
            payload = message.get_payload(decode=True)
            text = _decode_payload(payload, message.get_content_charset()).strip()
            if text:
                if content_type == "text/html":
                    html_parts.append(_defang_url_like_text(text))
                else:
                    plain_parts.append(_defang_url_like_text(text))

        html_text = "\n\n".join(html_parts).strip()
        css_text = _extract_css_from_html(html_text) if html_text else ""
        if plain_parts:
            body_text = "\n\n".join(plain_parts).strip()
        elif html_text:
            body_text = _html_to_text(html_text)
        else:
            body_text = ""

        return body_text, html_text, css_text
    except Exception:
        pass

    raw_body = _extract_after_header_block(raw_bytes)
    looks_like_html = bool(re.search(r"<\s*html|<\s*body|<\s*[a-zA-Z][^>]*>", raw_body, flags=re.IGNORECASE))
    if looks_like_html:
        html_text = _defang_url_like_text(raw_body)
        return _html_to_text(html_text), html_text, _extract_css_from_html(html_text)
    return _defang_url_like_text(raw_body), "", ""


def extract_body_and_html_without_headers(raw_bytes: bytes) -> Tuple[str, str]:
    # Backward-compatible helper used by older call sites.
    body_text, html_text, _ = extract_body_html_css_without_headers(raw_bytes)
    return body_text, html_text


def extract_body_without_headers(raw_bytes: bytes) -> str:
    # Backward-compatible helper used by older call sites.
    body_text, _, _ = extract_body_html_css_without_headers(raw_bytes)
    return body_text


def extract_css_text_from_html(html_text: str) -> str:
    """Extract CSS from ``<style>`` blocks and inline ``style`` attributes in HTML."""
    if not html_text or not isinstance(html_text, str):
        return ""
    return _extract_css_from_html(html_text)


__all__ = [
    "extract_body_html_css_without_headers",
    "extract_body_and_html_without_headers",
    "extract_body_without_headers",
    "extract_css_text_from_html",
]
