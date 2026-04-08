"""
Shared helpers for campaign (cluster) membership artifacts used by GNN / featureset visualization.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np


def group_emails_by_cluster(
    sorted_ids: list[str],
    labels: np.ndarray | list[int],
) -> tuple[list[dict[str, Any]], int]:
    """
    Group non-noise points by cluster label.

    Returns (campaigns, n_noise) where each campaign is
    ``{"id": int, "member_external_ids": list[str], "size": int}``.
    """
    labels_arr = np.asarray(labels)
    if len(sorted_ids) != len(labels_arr):
        raise ValueError(
            f"len(sorted_ids)={len(sorted_ids)} != len(labels)={len(labels_arr)}"
        )
    by_cluster: dict[int, list[str]] = defaultdict(list)
    n_noise = 0
    for eid, lab in zip(sorted_ids, labels_arr):
        li = int(lab)
        if li == -1:
            n_noise += 1
            continue
        by_cluster[li].append(str(eid))

    campaigns = [
        {
            "id": cid,
            "member_external_ids": members,
            "size": len(members),
        }
        for cid, members in sorted(by_cluster.items(), key=lambda x: x[0])
    ]
    return campaigns, n_noise


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
    campaigns, n_noise = group_emails_by_cluster(sorted_ids, labels)
    payload: dict[str, Any] = {
        "solution": solution,
        "algorithm": algorithm,
        "params": params,
        "campaigns": campaigns,
        "n_campaigns": len(campaigns),
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
