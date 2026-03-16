"""
Link prediction evaluation: AUROC/AP and Recall@K.
Lower-level functions take already-loaded runtime objects; wrappers load graph from path and call them.
"""
from .auroc_ap import (
    collect_auroc_ap_scores,
    collect_auroc_ap_scores_and_distributions,
    run_auroc_ap_analysis,
    run_auroc_ap_from_run,
)
from .recall_at_k import (
    recall_at_k_mrr,
    run_recall_at_k_analysis,
    run_recall_at_k_from_run,
    topk_eval_with_splits,
    topk_for_source,
)

__all__ = [
    "collect_auroc_ap_scores",
    "collect_auroc_ap_scores_and_distributions",
    "run_auroc_ap_analysis",
    "recall_at_k_mrr",
    "topk_eval_with_splits",
    "topk_for_source",
    "run_recall_at_k_analysis",
    "run_auroc_ap_from_run",
    "run_recall_at_k_from_run",
]
