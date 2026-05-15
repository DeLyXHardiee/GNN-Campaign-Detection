"""
Load email subject/body keyed by external_id from MISP lake JSON or a translated
sidecar file (see translate_misp_email_texts_to_en.py).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


def find_project_root(start: Path | None = None) -> Path:
    p = (start or Path.cwd()).resolve()
    for d in (p, *p.parents):
        if (d / "pipeline_config.json").is_file():
            return d
    raise FileNotFoundError(
        "Could not find pipeline_config.json; run from repo root or a subdirectory."
    )


def _ensure_core_on_syspath(project_root: Path) -> None:
    core = project_root / "core"
    s = str(core.resolve())
    if s not in sys.path:
        sys.path.insert(0, s)


def load_misp_events_list(misp_json_path: Path) -> list[dict[str, Any]]:
    with misp_json_path.open("r", encoding="utf-8-sig") as f:
        raw = json.load(f)
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict):
        ev = raw.get("Events") or raw.get("response", {}).get("Event", [])
        if isinstance(ev, list):
            return ev
        if isinstance(ev, dict):
            return [ev]
    return []


def load_misp_subject_body_by_external_id(
    misp_json_path: Path, *, project_root: Path
) -> dict[str, dict[str, str]]:
    """First occurrence per external_id wins (same as graph email catalog)."""
    _ensure_core_on_syspath(project_root)
    from graph.common import parse_misp_events

    events = load_misp_events_list(misp_json_path)
    parsed = parse_misp_events(events)
    out: dict[str, dict[str, str]] = {}
    for ev in parsed:
        eid = str(ev.get("external_id") or "").strip()
        if not eid or eid in out:
            continue
        out[eid] = {
            "subject": str(ev.get("subject") or ""),
            "body": str(ev.get("body") or ""),
        }
    return out


def load_translated_email_text_by_external_id(path: Path) -> dict[str, dict[str, str]]:
    """
    Load English (or other) subject/body from a sidecar JSON written by
    translate_misp_email_texts_to_en.py.

    Expected shape: ``{"by_external_id": {"<eid>": {"subject": "...", "body": "..."}}}``.
    """
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Translated file must be a JSON object: {path}")
    by_eid = data.get("by_external_id")
    if not isinstance(by_eid, dict):
        raise ValueError(f"Translated file must contain object 'by_external_id': {path}")

    out: dict[str, dict[str, str]] = {}
    for eid, row in by_eid.items():
        if not isinstance(row, dict):
            continue
        eid_s = str(eid).strip()
        if not eid_s:
            continue
        out[eid_s] = {
            "subject": str(row.get("subject") or ""),
            "body": str(row.get("body") or ""),
        }
    return out
