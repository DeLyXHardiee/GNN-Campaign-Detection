"""
Build ``visualization/data.json`` for the cluster inspection web UI.

Joins campaign assignment artifacts (GNN / featureset) with email content from MISP JSON.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

_CORE_ROOT = Path(__file__).resolve().parents[1]
if str(_CORE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CORE_ROOT))

from graph.common import parse_misp_events  # noqa: E402


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


def build_visualization_data(
    *,
    run_dir: str | Path,
    misp_json_path: str,
    include_attribute_similarity: bool = True,
) -> dict[str, Any]:
    """
    Combine ``campaigns_gnn.json`` / ``campaigns_featureset.json`` (when present) with email text.

    Output shape:
    - ``gnn``: null or { algorithm, campaigns, ... }
    - ``featureset``: null or { ... }
    - ``emails``: map external_id -> email fields
    """
    run_dir = Path(run_dir)
    emails = build_email_catalog_from_misp(misp_json_path)

    gnn_path = run_dir / "clustering" / "campaigns_gnn.json"
    fs_path = run_dir / "featureset_clustering" / "campaigns_featureset.json"

    gnn_raw = _load_json_if_exists(gnn_path)
    fs_raw = _load_json_if_exists(fs_path)

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
            "n_non_noise_campaigns": raw.get("n_non_noise_campaigns"),
            "n_noise": raw.get("n_noise"),
            "campaigns": camps,
        }

    payload: dict[str, Any] = {
        "run_dir": str(run_dir.resolve()),
        "misp_json_path": str(Path(misp_json_path).resolve()),
        "gnn": _strip_campaigns_payload(gnn_raw),
        "featureset": _strip_campaigns_payload(fs_raw),
        "emails": emails,
    }

    if include_attribute_similarity:
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
                gnn=payload.get("gnn"),
                featureset=payload.get("featureset"),
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
    viz_dir = run_dir / "visualization"
    viz_dir.mkdir(parents=True, exist_ok=True)
    out_path = viz_dir / out_name
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return out_path
