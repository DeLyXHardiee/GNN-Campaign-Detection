from __future__ import annotations

import hashlib
from email import policy
from email.parser import BytesParser
from typing import List


def _to_bytes(payload: bytes | str | None) -> bytes:
    if payload is None:
        return b""
    if isinstance(payload, bytes):
        return payload
    if isinstance(payload, str):
        return payload.encode("utf-8", errors="replace")
    return b""


def extract_attachment_hashes_from_email(raw_bytes: bytes) -> List[str]:
    """Extract SHA-256 hashes of attachment payloads from an RFC email."""
    if not raw_bytes:
        return []

    try:
        message = BytesParser(policy=policy.default).parsebytes(raw_bytes)
    except Exception:
        return []

    hashes: List[str] = []
    seen: set[str] = set()

    for part in message.walk():
        if part.is_multipart():
            continue

        disposition = (part.get_content_disposition() or "").lower()
        filename = part.get_filename()
        content_type = (part.get_content_type() or "").lower()

        is_attachment = disposition == "attachment"
        looks_like_embedded_file = bool(filename) and content_type not in {"text/plain", "text/html"}
        if not is_attachment and not looks_like_embedded_file:
            continue

        payload = _to_bytes(part.get_payload(decode=True))
        if not payload:
            continue

        digest = hashlib.sha256(payload).hexdigest()
        if digest not in seen:
            seen.add(digest)
            hashes.append(digest)

    return hashes


__all__ = ["extract_attachment_hashes_from_email"]
