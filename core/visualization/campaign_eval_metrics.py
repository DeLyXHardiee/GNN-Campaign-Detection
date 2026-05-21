"""
Per-predicted-campaign external metrics vs ground truth (homogeneity, completeness, v-measure).

Homogeneity uses sklearn on cluster members (constant predicted label). Completeness uses
macro-averaged recall of each ground-truth class present in the cluster against global
class counts in the solution evaluation set.
"""
from __future__ import annotations

from collections import Counter
from typing import Any

from sklearn.metrics import homogeneity_score


def _gt_counts_for_solution(
    campaigns: list[dict[str, Any]],
    ground_truth: dict[str, Any],
) -> Counter[Any]:
    """Count ground-truth labels over all campaign members that have a GT label."""
    counts: Counter[Any] = Counter()
    for camp in campaigns:
        for eid in camp.get("member_external_ids") or []:
            key = str(eid)
            label = ground_truth.get(key)
            if label is None:
                label = ground_truth.get(eid)
            if label is not None:
                counts[label] += 1
    return counts


def compute_campaign_eval_metrics(
    campaign: dict[str, Any],
    *,
    ground_truth: dict[str, Any],
    gt_label_counts: Counter[Any],
) -> dict[str, Any]:
    """
    Return ``homogeneity``, ``completeness``, ``v_measure``, and ``n_eval`` for one campaign.

    Members without a ground-truth label are skipped. When ``n_eval`` is 0, metric fields are
    ``None``.
    """
    cid = campaign.get("id")
    members_raw = campaign.get("member_external_ids") or []
    y_true: list[Any] = []
    y_pred: list[Any] = []
    for eid in members_raw:
        key = str(eid)
        true = ground_truth.get(key)
        if true is None:
            true = ground_truth.get(eid)
        if true is None:
            continue
        y_true.append(true)
        y_pred.append(cid)

    n_eval = len(y_true)
    if n_eval == 0:
        return {
            "n_eval": 0,
            "homogeneity": None,
            "completeness": None,
            "v_measure": None,
        }

    hom = float(homogeneity_score(y_true, y_pred))

    comp_vals: list[float] = []
    for label in set(y_true):
        in_cluster = sum(1 for t in y_true if t == label)
        global_n = int(gt_label_counts.get(label, 0))
        if global_n > 0:
            comp_vals.append(in_cluster / global_n)
    completeness = float(sum(comp_vals) / len(comp_vals)) if comp_vals else float("nan")

    if n_eval == 1:
        hom = 1.0

    if not (completeness == completeness):
        vm = float("nan")
    elif hom + completeness == 0:
        vm = 0.0
    else:
        vm = float(2.0 * hom * completeness / (hom + completeness))

    return {
        "n_eval": n_eval,
        "homogeneity": hom,
        "completeness": completeness,
        "v_measure": vm,
    }


def enrich_campaigns_with_eval_metrics(
    campaigns: list[dict[str, Any]],
    ground_truth: dict[str, Any],
) -> None:
    """Attach eval metric fields to each campaign dict in place."""
    gt_counts = _gt_counts_for_solution(campaigns, ground_truth)
    for camp in campaigns:
        camp.update(
            compute_campaign_eval_metrics(
                camp,
                ground_truth=ground_truth,
                gt_label_counts=gt_counts,
            )
        )
