from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from seed_candidate_workflow.utils.graph_scorer_registry import apply_scorer


def test_apply_scorer_emits_diagnostics_when_enabled() -> None:
    candidate_union_df = pd.DataFrame(
        {
            "email_i": ["a", "b"],
            "email_j": ["b", "c"],
            "from_seed": [False, False],
            "from_semantic": [True, False],
            "from_rare_artifact": [False, True],
            "from_component": [False, False],
            "from_2hop": [False, True],
            "source_count": [1, 2],
            "semantic_cosine_max": [0.95, 0.12],
            "component_cosine_max": [0.0, 0.0],
            "rare_artifact_rarity_max": [0.0, 0.8],
            "twohop_rarity_max": [0.0, 0.5],
            "time_gap_seconds_min": [20.0, 40.0],
        }
    )
    seed_edges_df = pd.DataFrame(columns=["email_i", "email_j", "evidence_rarity"])
    out = apply_scorer(
        score_mode="seed_candidate_handcrafted_v1",
        graph_kind="seed_candidate",
        score_params={},
        payload={"candidate_union_df": candidate_union_df, "seed_edges_df": seed_edges_df},
        diagnostics_cfg={"enabled": True},
    )
    assert out.scored_all.shape[0] == 2
    assert "diagnostics" in out.metadata
    diag = out.metadata["diagnostics"]
    assert diag["enabled"] is True
    assert diag["summary"]["output_stats"]["rows_total"] == 2

