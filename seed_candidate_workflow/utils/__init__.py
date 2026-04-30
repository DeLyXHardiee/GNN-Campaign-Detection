"""Analysis notebook helpers (RAW vs GNN campaign diagnostics).

Heavy imports (e.g. numpy) are deferred until a symbol from ``raw_gnn_notebook`` is
requested via ``from seed_candidate_workflow.utils import <name>`` or attribute access.
Submodule imports such as ``seed_candidate_workflow.utils.graph_structure_helpers`` do not load
``raw_gnn_notebook``.
"""

from __future__ import annotations

import importlib
from typing import Any

__all__ = [
    "AnalysisPaths",
    "campaign_descriptors",
    "campaign_fragmentation_table",
    "campaign_size_bucket_map",
    "drilldown_fragmentation_summary",
    "drilldown_member_table",
    "drilldown_neighbor_comparison_block",
    "ensure_core_gnn_paths",
    "extreme_pair_inspection_tables",
    "external_scores_subset",
    "extract_raw_gnn_maps",
    "find_extreme_pair_examples",
    "find_project_root",
    "format_pair_row",
    "intra_campaign_embedding_pairwise_stats",
    "load_email_external_ids",
    "load_ground_truth_structures",
    "merge_per_campaign_neighbor_rows",
    "neighbor_metrics",
    "neighbor_ranking_table",
    "pairwise_cosine_summary_table",
    "pairwise_scatter_frame",
    "pairwise_similarity_samples",
    "resolve_analysis_paths",
    "restrict_embedding_map",
    "row_email_previews",
    "run_hdbscan_get_labels",
    "stratified_fragmentation_summary",
    "stratified_neighbor_summary",
    "top_campaign_shortlists",
    "top_k_neighbors",
    "truncate_preview",
    "urls_list_from_row",
    "eid_label_map",
]


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    rn = importlib.import_module(".raw_gnn_notebook", __package__)
    return getattr(rn, name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
