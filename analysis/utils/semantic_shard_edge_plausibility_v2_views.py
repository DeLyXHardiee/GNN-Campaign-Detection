"""
Three-view (semantic, infra, temporal) scores and agreement teacher for Method 1 V2.

Agreement uses **only** these three views — never local graph structure.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.utils.semantic_shard_edge_refinement_method1 import (
    Method1RefinementConfig,
    build_method1_edge_feature_frame,
    compute_method1_view_scores,
)

VIEW_COLS = ("view_semantic", "view_infra", "view_temporal")


def build_view_scores_df(edges_df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-edge view scores in [0,1] using Method 1 view logic."""
    feat = build_method1_edge_feature_frame(edges_df, weight_col="edge_weight")
    cfg = Method1RefinementConfig(
        use_semantic_view=True,
        use_infra_view=True,
        use_temporal_view=True,
        use_local_structure=False,
    )
    return compute_method1_view_scores(feat, cfg=cfg)


def stack_view_matrix(views_df: pd.DataFrame) -> np.ndarray:
    """Shape (n_edges, 3) in order semantic, infra, temporal."""
    return np.column_stack(
        [views_df[c].to_numpy(dtype=np.float64) for c in VIEW_COLS]
    )


def compute_agreement_scalar(views_df: pd.DataFrame, eps: float = 1e-6) -> np.ndarray:
    """
    Unsupervised teacher ordering signal: geometric mean of the three views.

    Monotonic in each view on (0,1]^3; higher => more multi-view support.
    """
    m = stack_view_matrix(views_df)
    m = np.clip(m, eps, 1.0)
    g = np.exp(np.mean(np.log(m), axis=1))
    return np.clip(g, 0.0, 1.0)
