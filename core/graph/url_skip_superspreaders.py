"""Load substring patterns used to omit high-degree URL nodes from graph assembly."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

_GRAPH_DIR = Path(__file__).resolve().parent
DEFAULT_URL_SKIP_SUPERSPREADERS_FILE = "url_skip_superspreaders.txt"


def load_url_skip_superspreaders(path: Optional[Union[str, Path]] = None) -> Tuple[str, ...]:
    """Return non-empty lowercase lines from the skip file (``#`` starts a comment).

    If the file is missing or empty, returns an empty tuple.
    """
    resolved = Path(path) if path is not None else _GRAPH_DIR / DEFAULT_URL_SKIP_SUPERSPREADERS_FILE
    if not resolved.is_file():
        return ()
    patterns: list[str] = []
    text = resolved.read_text(encoding="utf-8")
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        token = line.lower()
        if token:
            patterns.append(token)
    return tuple(patterns)


def resolve_url_skip_superspreaders_patterns(
    *,
    path: Optional[Union[str, Path]] = None,
    inline_substrings: Optional[Sequence[str]] = None,
) -> Tuple[str, ...]:
    """Use ``inline_substrings`` when provided; otherwise load from ``path`` (or default file)."""
    if inline_substrings is not None:
        return tuple(str(s).strip().lower() for s in inline_substrings if str(s).strip())
    return load_url_skip_superspreaders(path)
