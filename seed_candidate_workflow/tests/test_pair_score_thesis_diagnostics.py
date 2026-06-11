from __future__ import annotations

import json

import numpy as np

from seed_candidate_workflow.utils.pair_score_thesis_diagnostics import (
    SLICE_NON_SEED,
    SLICE_SEED,
    build_statistics_rows,
    compute_thesis_pair_score_diagnostics,
    distribution_stats,
)


def test_distribution_stats_basic():
    x = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=float)
    st = distribution_stats(x)
    assert st["count"] == 5
    assert st["mean"] == 0.3
    assert st["median"] == 0.3
    assert st["min"] == 0.1
    assert st["max"] == 0.5
    assert st["iqr"] == 0.2


def test_compute_thesis_diagnostics_slices(tmp_path):
    import pandas as pd

    gt_path = tmp_path / "gt.json"
    gt_path.write_text(
        json.dumps(
            {
                "clusters": {
                    "c1": [
                        {"external_id": "a@x.com"},
                        {"external_id": "b@x.com"},
                    ],
                    "c2": [
                        {"external_id": "c@x.com"},
                        {"external_id": "d@x.com"},
                    ],
                }
            }
        ),
        encoding="utf-8",
    )
    df = pd.DataFrame(
        {
            "email_i": ["a@x.com", "a@x.com", "c@x.com", "a@x.com"],
            "email_j": ["b@x.com", "c@x.com", "d@x.com", "c@x.com"],
            "is_seed_pair": [True, True, False, False],
        }
    )
    scores = np.array([0.9, 0.1, 0.2, 0.8], dtype=float)
    out = compute_thesis_pair_score_diagnostics(df=df, scores=scores, gt_path=gt_path)
    rows = build_statistics_rows(out["slices"])
    assert len(rows) == 6
    seed_sl = next(s for s in out["slices"] if s["slice_id"] == SLICE_SEED)
    non_seed_sl = next(s for s in out["slices"] if s["slice_id"] == SLICE_NON_SEED)
    assert seed_sl["n_same_campaign"] == 1
    assert seed_sl["n_cross_campaign"] == 1
    assert non_seed_sl["n_same_campaign"] == 1
    assert non_seed_sl["n_cross_campaign"] == 1
