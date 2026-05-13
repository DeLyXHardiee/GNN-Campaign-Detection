"""
Build ``visualization/data.json`` for the cluster inspection web UI.

Discovers every ``campaigns*.json`` artifact in the run directory (recursively),
joins them with email content from MISP JSON, and emits one entry per file under
``solutions`` so the webapp can render a tab per discovered solution.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

_CORE_ROOT = Path(__file__).resolve().parents[1]
if str(_CORE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CORE_ROOT))

from graph.common import parse_misp_events  # noqa: E402

_VIZ_DIRNAME = "visualization"
_CAMPAIGNS_GLOB = "campaigns*.json"


def _load_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_misp_events_list(misp_json_path: str) -> list[dict[str, Any]]:
    with open(misp_json_path, "r", encoding="utf-8") as f:
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


def build_email_catalog_from_misp(misp_json_path: str) -> dict[str, dict[str, Any]]:
    """external_id -> display fields for the UI."""
    events = _load_misp_events_list(misp_json_path)
    parsed = parse_misp_events(events)
    out: dict[str, dict[str, Any]] = {}
    for ev in parsed:
        eid = str(ev.get("external_id") or "").strip()
        if not eid:
            continue
        out[eid] = {
            "external_id": eid,
            "subject": ev.get("subject") or "",
            "date": ev.get("date") or "",
            "senders": list(ev.get("senders") or []),
            "receivers": list(ev.get("receivers") or []),
            "body": ev.get("body") or "",
            "urls": list(ev.get("urls") or []),
            "email_info": ev.get("email_info") or "",
        }
    return out


def _strip_campaigns_payload(raw: dict[str, Any] | None) -> dict[str, Any] | None:
    if not raw:
        return None
    camps = raw.get("campaigns") or []
    if not camps:
        return None
    return {
        "solution": raw.get("solution"),
        "algorithm": raw.get("algorithm"),
        "model": raw.get("model"),
        "feature_set": raw.get("feature_set"),
        "n_components": raw.get("n_components"),
        "params": raw.get("params") or {},
        "metrics": raw.get("metrics") or {},
        "n_campaigns": raw.get("n_campaigns", len(camps)),
        "n_noise": raw.get("n_noise"),
        "campaigns": camps,
    }


def _discover_campaigns_files(run_dir: Path) -> list[Path]:
    """All ``campaigns*.json`` files under ``run_dir`` excluding the visualization output."""
    out: list[Path] = []
    for p in sorted(run_dir.rglob(_CAMPAIGNS_GLOB)):
        if not p.is_file():
            continue
        try:
            rel_parts = p.relative_to(run_dir).parts
        except ValueError:
            rel_parts = ()
        if rel_parts and rel_parts[0] == _VIZ_DIRNAME:
            continue
        out.append(p)
    return out


def _solutions_from_run_dir(run_dir: Path) -> dict[str, dict[str, Any]]:
    """
    Map ``<rel-path-from-run-dir>`` -> stripped campaigns payload, with a
    UI ``label`` field set to the JSON file name (collisions disambiguated by
    appending the parent directory name).
    """
    solutions: dict[str, dict[str, Any]] = {}
    label_counts: dict[str, int] = defaultdict(int)

    for path in _discover_campaigns_files(run_dir):
        raw = _load_json_if_exists(path)
        payload = _strip_campaigns_payload(raw)
        if payload is None:
            continue
        rel = path.relative_to(run_dir).as_posix()
        label = path.name
        payload["label"] = label
        payload["file"] = rel
        solutions[rel] = payload
        label_counts[label] += 1

    for rel, payload in solutions.items():
        if label_counts[payload["label"]] > 1:
            parent = Path(rel).parent.name or "root"
            payload["label"] = f"{payload['label']} ({parent})"

    return solutions


def build_visualization_data(
    *,
    run_dir: str | Path,
    misp_json_path: str,
    include_attribute_similarity: bool = True,
) -> dict[str, Any]:
    """
    Assemble the webapp data file by discovering every ``campaigns*.json``
    artifact under ``run_dir`` and joining members with email content.

    Output shape:
    - ``solutions``: { <rel_path>: { label, file, campaigns, ... } }
    - ``emails``: external_id -> email fields
    - ``attribute_similarity``: optional { <rel_path>: { <cid>: { <eid>: {...} } } }
    """
    run_dir = Path(run_dir)
    emails = build_email_catalog_from_misp(misp_json_path)
    solutions = _solutions_from_run_dir(run_dir)

    payload: dict[str, Any] = {
        "run_dir": str(run_dir.resolve()),
        "misp_json_path": str(Path(misp_json_path).resolve()),
        "solutions": solutions,
        "emails": emails,
    }

    if include_attribute_similarity and solutions:
        try:
            from visualization.attribute_similarity import (
                build_attribute_similarity_sidecar,
            )
        except ModuleNotFoundError:
            from core.visualization.attribute_similarity import (
                build_attribute_similarity_sidecar,
            )

        try:
            sim = build_attribute_similarity_sidecar(
                solutions=solutions,
                emails=emails,
            )
            if sim:
                payload["attribute_similarity"] = sim
        except Exception as exc:
            payload["attribute_similarity_error"] = str(exc)

    return payload


def write_visualization_data_json(
    *,
    run_dir: str | Path,
    misp_json_path: str,
    out_name: str = "data.json",
    include_attribute_similarity: bool = True,
) -> Path:
    run_dir = Path(run_dir)
    payload = build_visualization_data(
        run_dir=run_dir,
        misp_json_path=misp_json_path,
        include_attribute_similarity=include_attribute_similarity,
    )
    viz_dir = run_dir / _VIZ_DIRNAME
    viz_dir.mkdir(parents=True, exist_ok=True)
    out_path = viz_dir / out_name
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return out_path
