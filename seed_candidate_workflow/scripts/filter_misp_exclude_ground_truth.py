"""
Drop MISP events whose canonical external_id appears in ground_truth.json.

Reads:
  - data/groundtruth/ground_truth.json (clusters -> list of email dicts with external_id)
  - data/misp/incidents-lake-misp-large.json (list of {"Event": ...} MISP events)

Writes:
  - data/incidents-lake-misp-large-no-ground-truth.json

Canonical id per event matches ``parse_misp_events`` in core/graph/common.py:
  external_id stripped string, or ``str(email_index)`` with email_index defaulting to
  the event's list index.

Usage (from repo root)::

    python seed_candidate_workflow/scripts/filter_misp_exclude_ground_truth.py

Or with explicit paths::

    python seed_candidate_workflow/scripts/filter_misp_exclude_ground_truth.py \\
        --ground-truth path/to/ground_truth.json \\
        --misp-in path/to/incidents.json \\
        --out path/to/out.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _to_str(val: Any) -> str:
    if val is None:
        return ""
    if isinstance(val, str):
        return val
    return str(val)


def external_ids_from_ground_truth(gt_path: Path) -> set[str]:
    """Same key resolution as ``load_ground_truth_structures`` (first occurrence wins)."""
    with open(gt_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    label_map: dict[str, Any] = {}
    for _raw_key, emails in (data.get("clusters") or {}).items():
        if not isinstance(emails, list):
            continue
        for em in emails:
            if not isinstance(em, dict):
                continue
            eid = em.get("external_id")
            if eid is None:
                continue
            eid_s = str(eid)
            if eid_s in label_map:
                continue
            label_map[eid_s] = None

    return set(label_map.keys())


def misp_event_canonical_id(ev: Any, list_index: int) -> str:
    """Align with ``parse_misp_events`` external_id assignment."""
    if not isinstance(ev, dict):
        return str(list_index)
    event = ev.get("Event")
    if not isinstance(event, dict):
        return str(list_index)
    email_index = event.get("email_index", list_index)
    ext = _to_str(event.get("external_id", "")).strip()
    return ext or str(email_index)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--ground-truth",
        type=Path,
        default=PROJECT_ROOT / "data" / "groundtruth" / "ground_truth.json",
    )
    p.add_argument(
        "--misp-in",
        type=Path,
        default=PROJECT_ROOT / "data" / "misp" / "incidents-lake-misp-large.json",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "data" / "incidents-lake-misp-large-no-ground-truth.json",
    )
    p.add_argument(
        "--indent",
        type=int,
        default=None,
        help="Pretty-print with this indent (default: compact JSON).",
    )
    args = p.parse_args()

    gt_path = args.ground_truth.expanduser().resolve()
    misp_path = args.misp_in.expanduser().resolve()
    out_path = args.out.expanduser().resolve()

    if not gt_path.is_file():
        raise FileNotFoundError(f"Ground truth not found: {gt_path}")
    if not misp_path.is_file():
        raise FileNotFoundError(f"MISP JSON not found: {misp_path}")

    gt_ids = external_ids_from_ground_truth(gt_path)

    with open(misp_path, "r", encoding="utf-8") as f:
        misp_events = json.load(f)

    if not isinstance(misp_events, list):
        raise TypeError(f"Expected MISP file to be a JSON array; got {type(misp_events).__name__}")

    kept: list[Any] = []
    removed_canon: list[str] = []
    for i, ev in enumerate(misp_events):
        cid = misp_event_canonical_id(ev, i)
        if cid in gt_ids:
            removed_canon.append(cid)
            continue
        kept.append(ev)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(kept, f, ensure_ascii=False, indent=args.indent)

    print(
        "ground_truth external_ids:",
        len(gt_ids),
        "| misp events in:",
        len(misp_events),
        "| removed (in GT):",
        len(removed_canon),
        "| kept:",
        len(kept),
    )
    print("wrote:", out_path)


if __name__ == "__main__":
    main()
