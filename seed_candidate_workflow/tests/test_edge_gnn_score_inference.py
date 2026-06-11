from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from seed_candidate_workflow.utils.edge_gnn_score_inference import load_edge_gnn_pair_scores


def test_load_edge_gnn_pair_scores_canonical_and_dedup(tmp_path: Path) -> None:
    p = tmp_path / "edge_gnn_pair_scores.csv"
    pd.DataFrame(
        [
            {"email_i": "b@x.com", "email_j": "a@x.com", "pu_score": 0.25},
            {"email_i": "c@x.com", "email_j": "d@x.com", "pu_score": 0.75},
            {"email_i": "c@x.com", "email_j": "d@x.com", "pu_score": 0.75},
        ]
    ).to_csv(p, index=False)

    score_map, diag = load_edge_gnn_pair_scores(p, on_duplicate="keep_first")
    assert score_map[("a@x.com", "b@x.com")] == pytest.approx(0.25)
    assert score_map[("c@x.com", "d@x.com")] == pytest.approx(0.75)
    assert diag["num_scores"] == 2
    assert diag["num_duplicate_pairs"] == 1
    assert diag["num_invalid_scores"] == 0


def test_load_edge_gnn_pair_scores_rejects_conflicting_duplicates(tmp_path: Path) -> None:
    p = tmp_path / "edge_gnn_pair_scores.csv"
    pd.DataFrame(
        [
            {"email_i": "a@x.com", "email_j": "b@x.com", "pu_score": 0.1},
            {"email_i": "a@x.com", "email_j": "b@x.com", "pu_score": 0.9},
        ]
    ).to_csv(p, index=False)

    with pytest.raises(ValueError, match="Conflicting pu_score"):
        load_edge_gnn_pair_scores(p, on_duplicate="error")


def test_load_edge_gnn_pair_scores_skips_invalid(tmp_path: Path) -> None:
    p = tmp_path / "edge_gnn_pair_scores.csv"
    pd.DataFrame(
        [
            {"email_i": "a@x.com", "email_j": "b@x.com", "pu_score": "nan"},
            {"email_i": "c@x.com", "email_j": "d@x.com", "pu_score": 0.5},
        ]
    ).to_csv(p, index=False)

    score_map, diag = load_edge_gnn_pair_scores(p)
    assert len(score_map) == 1
    assert diag["num_invalid_scores"] == 1
