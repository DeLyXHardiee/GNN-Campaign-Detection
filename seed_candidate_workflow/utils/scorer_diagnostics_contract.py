from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class GroundTruthSplit:
    both_labeled_mask: Any
    same_campaign_mask: Any
    cross_campaign_mask: Any


@dataclass(frozen=True)
class ScoredPairsInput:
    score_mode: str
    graph_kind: str
    scored_df: pd.DataFrame
    score_column: str = "edge_weight"


@dataclass
class DiagnosticsResult:
    scorer_name: str
    graph_kind: str
    score_mode: str
    input_stats: dict[str, Any] = field(default_factory=dict)
    output_stats: dict[str, Any] = field(default_factory=dict)
    provenance_stats: dict[str, Any] = field(default_factory=dict)
    scorer_specific: dict[str, Any] = field(default_factory=dict)

