"""High-confidence shared-stem candidate pairs (strict rarity / DF gates)."""

from __future__ import annotations

from typing import Any

import pandas as pd

from seed_candidate_workflow.utils.anchor_candidate_rare_artifact_helpers import (
    generate_candidates_rare_artifact_v1,
)


def generate_candidates_shared_stem_highconf_v1(
    *,
    nodes_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    seed_pairs: set[tuple[str, str]] | None,
    generator_cfg: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
  Stem-only rare-artifact candidates with stricter IDF/DF than generic rare_artifact_v1.

  Provenance source label: ``shared_stem_highconf``.
  """
    cfg = dict(generator_cfg or {})
    effective = {
        "include_time_gap_seconds": bool(cfg.get("include_time_gap_seconds", True)),
        "min_artifact_idf": float(cfg.get("min_artifact_idf", 1.0)),
        "max_artifact_df": int(cfg.get("max_artifact_df", 8)),
        "max_shared_values_per_edge_per_artifact": int(
            cfg.get("max_shared_values_per_edge_per_artifact", 10)
        ),
        "max_candidate_rows": cfg.get("max_candidate_rows"),
        "artifact_specs": [
            {
                "artifact_type": "stem",
                "node_set_col": "stem_set",
                "overlap_base": "stem",
            }
        ],
    }
    df, diag = generate_candidates_rare_artifact_v1(
        nodes_df=nodes_df,
        edges_df=edges_df,
        seed_pairs=seed_pairs,
        generator_cfg=effective,
    )
    if not df.empty:
        df = df.copy()
        df["source"] = "shared_stem_highconf"
    diag = dict(diag or {})
    diag["generator"] = "shared_stem_highconf_v1"
    diag["effective_config"] = effective
    return df, diag
