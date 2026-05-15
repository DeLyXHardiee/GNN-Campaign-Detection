from __future__ import annotations

import json
from pathlib import Path

from seed_candidate_workflow.utils.seed_bundle_health import run_health_checks


def test_health_checks_pass(tmp_path: Path) -> None:
    anchor = {
        "union_edges": {
            "metrics": {
                "n_components": 600,
                "component_size_distribution_top50": [120, 90, 80],
            }
        }
    }
    pair = {
        "pair_counts": {"n_unique_pairs_final": 50000},
        "component_context": {"n_pairs_same_seed_component": 18000},
    }
    pa = tmp_path / "anchor_seed_summary.json"
    pb = tmp_path / "pair_training_dataset_summary.json"
    pa.write_text(json.dumps(anchor), encoding="utf-8")
    pb.write_text(json.dumps(pair), encoding="utf-8")

    errs = run_health_checks(
        anchor_seed_summary=pa,
        pair_training_summary=pb,
    )
    assert errs == []


def test_health_checks_mega_component(tmp_path: Path) -> None:
    anchor = {
        "union_edges": {
            "metrics": {
                "n_components": 300,
                "component_size_distribution_top50": [2500, 100],
            }
        }
    }
    pa = tmp_path / "anchor_seed_summary.json"
    pa.write_text(json.dumps(anchor), encoding="utf-8")

    errs = run_health_checks(anchor_seed_summary=pa)
    assert any("largest component" in e for e in errs)


def test_health_checks_same_seed_fraction(tmp_path: Path) -> None:
    pair = {
        "pair_counts": {"n_unique_pairs_final": 10000},
        "component_context": {"n_pairs_same_seed_component": 8000},
    }
    pb = tmp_path / "pair_training_dataset_summary.json"
    pb.write_text(json.dumps(pair), encoding="utf-8")

    errs = run_health_checks(pair_training_summary=pb)
    assert any("same_seed_component fraction" in e for e in errs)
