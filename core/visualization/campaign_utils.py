"""
Shared helpers for campaign (cluster) membership artifacts used by GNN / featureset visualization.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np


def _noise_singleton_campaign_id(external_id: str) -> str:
    """Stable singleton campaign id for a noise point."""
    return f"noise_{str(external_id)}"


def group_emails_by_cluster(
    sorted_ids: list[str],
    labels: np.ndarray | list[int],
    *,
    noise_as_singletons: bool = True,
) -> tuple[list[dict[str, Any]], int, int]:
    """
    Group cluster labels into campaign artifacts.

    When ``noise_as_singletons`` is true, every ``-1`` label is emitted as its own
    singleton campaign using a stable ``noise_<external_id>`` id. Returns
    ``(campaigns, n_noise, n_non_noise_campaigns)`` where each campaign is
    ``{"id": int | str, "member_external_ids": list[str], "size": int}``.
    """
    labels_arr = np.asarray(labels)
    if len(sorted_ids) != len(labels_arr):
        raise ValueError(
            f"len(sorted_ids)={len(sorted_ids)} != len(labels)={len(labels_arr)}"
        )
    by_cluster: dict[int, list[str]] = defaultdict(list)
    noise_campaigns: list[dict[str, Any]] = []
    n_noise = 0
    for eid, lab in zip(sorted_ids, labels_arr):
        li = int(lab)
        if li == -1:
            n_noise += 1
            if noise_as_singletons:
                noise_campaigns.append(
                    {
                        "id": _noise_singleton_campaign_id(str(eid)),
                        "member_external_ids": [str(eid)],
                        "size": 1,
                    }
                )
            continue
        by_cluster[li].append(str(eid))

    non_noise_campaigns = [
        {
            "id": cid,
            "member_external_ids": members,
            "size": len(members),
        }
        for cid, members in sorted(by_cluster.items(), key=lambda x: x[0])
    ]
    campaigns = non_noise_campaigns + noise_campaigns
    return campaigns, n_noise, len(non_noise_campaigns)


def build_campaign_artifact_payload(
    *,
    solution: str,
    algorithm: str,
    sorted_ids: list[str],
    labels: np.ndarray,
    params: dict[str, Any],
    metrics: dict[str, Any] | None = None,
    model_name: str | None = None,
    feature_set: str | None = None,
    n_components: int | None = None,
) -> dict[str, Any]:
    """Normalized JSON-serializable document for campaigns_*.json."""
    campaigns, n_noise, n_non_noise_campaigns = group_emails_by_cluster(sorted_ids, labels)
    payload: dict[str, Any] = {
        "solution": solution,
        "algorithm": algorithm,
        "params": params,
        "campaigns": campaigns,
        "n_campaigns": len(campaigns),
        "n_non_noise_campaigns": n_non_noise_campaigns,
        "n_noise": n_noise,
    }
    if metrics:
        payload["metrics"] = metrics
    if model_name is not None:
        payload["model"] = model_name
    if feature_set is not None:
        payload["feature_set"] = feature_set
    if n_components is not None:
        payload["n_components"] = int(n_components)
    return payload
