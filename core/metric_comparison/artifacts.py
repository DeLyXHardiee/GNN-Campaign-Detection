"""Load campaign JSON artifacts and build external_id -> cluster label maps."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _normalize_cluster_id(raw: Any) -> int:
    if isinstance(raw, bool):
        raise TypeError(f"unexpected boolean cluster id: {raw!r}")
    if isinstance(raw, int):
        return raw
    s = str(raw).strip()
    try:
        return int(s)
    except ValueError:
        # Stable hash for non-numeric ids (sklearn accepts int labels)
        return hash(s) % (2**31)


def pred_map_from_campaign_payload(payload: dict[str, Any]) -> dict[str, int]:
    """
    Build external_id -> predicted cluster id from ``campaigns`` list.

    Each campaign has ``id`` and ``member_external_ids``.
    """
    pred: dict[str, int] = {}
    campaigns = payload.get("campaigns") or []
    for camp in campaigns:
        cid = _normalize_cluster_id(camp.get("id"))
        for eid in camp.get("member_external_ids") or []:
            pred[str(eid)] = cid
    return pred


def load_campaign_artifact(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def strip_payload_meta(payload: dict[str, Any]) -> dict[str, Any]:
    """Subset of payload useful for comparison_summary.json."""
    keys = (
        "solution",
        "algorithm",
        "model",
        "feature_set",
        "n_components",
        "params",
        "metrics",
        "n_campaigns",
        "n_noise",
    )
    return {k: payload.get(k) for k in keys if k in payload}
