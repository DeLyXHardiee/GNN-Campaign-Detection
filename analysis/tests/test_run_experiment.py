"""Tests for experiment runner (dry-run, ids, manifest)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from analysis.pipelines.run_experiment import run_experiment
from analysis.utils.config_run_fields import resolve_graph_id, resolve_scoring_run_id
from analysis.utils.pair_graph_contract import (
    ensure_unscored_contract,
    migrate_unscored_graph_id_column,
    validate_score_mode_target_compatibility,
)


def test_resolve_graph_id_requires_value() -> None:
    with pytest.raises(ValueError, match="graph_id"):
        resolve_graph_id({})
    with pytest.raises(ValueError, match="graph_id"):
        resolve_graph_id({"graph_id": ""})


def test_resolve_scoring_run_id_requires_value() -> None:
    with pytest.raises(ValueError, match="scoring_run_id"):
        resolve_scoring_run_id({})
    with pytest.raises(ValueError, match="scoring_run_id"):
        resolve_scoring_run_id({"scoring_run_id": "  "})


def _minimal_experiment_cfg(
    *,
    tmp_path: Path,
    mode: str,
    graph_bundle_root: Path,
    scoring_output_root: Path,
) -> dict:
    gt_path = tmp_path / "gt_sets.json"
    gt_path.write_text(json.dumps({"tset": ["data/groundtruth/x.json"]}), encoding="utf-8")
    return {
        "experiment": {
            "graph_id": "g_test",
            "scoring_run_id": "sc_test",
            "mode": mode,
        },
        "artifacts": {
            "graph_bundle_root": str(graph_bundle_root),
            "scoring_output_root": str(scoring_output_root),
        },
        "setup": {"enable": {}, "policy": {}, "paths": {}},
        "selection": {
            "score_targets": ["seed_candidate"],
            "gt_set": "tset",
            "gt_sets_path": str(gt_path),
        },
        "scoring": {"score_mode": "none", "params": {}},
        "community": {"sweep": {}},
    }


def test_run_experiment_dry_run_score_only_without_graph_bundle(tmp_path: Path) -> None:
    bundles = tmp_path / "graph_bundles"
    bundles.mkdir(parents=True, exist_ok=True)
    scoring = tmp_path / "scoring_runs"
    scoring.mkdir(parents=True, exist_ok=True)
    cfg = _minimal_experiment_cfg(
        tmp_path=tmp_path,
        mode="score_only",
        graph_bundle_root=bundles,
        scoring_output_root=scoring,
    )
    out = run_experiment(cfg, dry_run=True)
    assert out["dry_run"] is True
    assert out["manifest"]["graph_id"] == "g_test"
    assert out["manifest"]["scoring_run_id"] == "sc_test"
    assert out["manifest"]["manifest_version"] == "2.0"
    assert len(out["community_results"]) == 1
    assert out["community_results"][0]["community_result"].get("dry_run") is True
    assert "inputs" in out["community_results"][0]
    assert "artifacts" in out["community_results"][0]
    assert "metrics" in out["community_results"][0]
    assert len(out["community_results_legacy"]) == 1
    assert out["community_results_legacy"][0]["target"] == out["community_results"][0]["target"]
    p_manifest = Path(out["manifest_json"])
    assert p_manifest.is_file()


def test_run_experiment_score_only_non_dry_requires_bundle(tmp_path: Path) -> None:
    bundles = tmp_path / "graph_bundles"
    bundles.mkdir(parents=True, exist_ok=True)
    scoring = tmp_path / "scoring_runs"
    scoring.mkdir(parents=True, exist_ok=True)
    cfg = _minimal_experiment_cfg(
        tmp_path=tmp_path,
        mode="score_only",
        graph_bundle_root=bundles,
        scoring_output_root=scoring,
    )
    with pytest.raises(FileNotFoundError, match="Graph bundle not found"):
        run_experiment(cfg, dry_run=False)


def test_migrate_unscored_graph_id_column_from_legacy_name() -> None:
    df = pd.DataFrame(
        {
            "email_i": ["a@x.com", "b@x.com"],
            "email_j": ["b@x.com", "c@x.com"],
            "graph_kind": ["seed_candidate", "seed_candidate"],
            "graph_run_id": ["gid", "gid"],
            "from_seed": [True, False],
            "from_semantic": [False, True],
            "from_rare_artifact": [False, False],
            "from_component": [False, False],
            "from_2hop": [False, False],
            "source_count": [1, 1],
        }
    )
    out = ensure_unscored_contract(df)
    assert "graph_id" in out.columns
    assert "graph_run_id" not in out.columns
    assert out["graph_id"].tolist() == ["gid", "gid"]


def test_migrate_prefers_graph_id_when_both_present() -> None:
    df = pd.DataFrame(
        {
            "email_i": ["a@x.com"],
            "email_j": ["b@x.com"],
            "graph_kind": ["seed_candidate"],
            "graph_id": ["new"],
            "graph_run_id": ["old"],
            "from_seed": [False],
            "from_semantic": [False],
            "from_rare_artifact": [False],
            "from_component": [False],
            "from_2hop": [False],
            "source_count": [1],
        }
    )
    m = migrate_unscored_graph_id_column(df)
    assert "graph_run_id" not in m.columns
    assert m["graph_id"].tolist() == ["new"]


def test_run_experiment_setup_only_dry_run_writes_manifest(tmp_path: Path) -> None:
    bundles = tmp_path / "graph_bundles"
    bundles.mkdir(parents=True, exist_ok=True)
    scoring = tmp_path / "scoring_runs"
    scoring.mkdir(parents=True, exist_ok=True)
    cfg = _minimal_experiment_cfg(
        tmp_path=tmp_path,
        mode="setup_only",
        graph_bundle_root=bundles,
        scoring_output_root=scoring,
    )
    out = run_experiment(cfg, dry_run=True)
    assert out["dry_run"] is True
    assert out["setup_result"] is None
    assert Path(out["manifest_json"]).is_file()


def test_validate_score_mode_target_compatibility_for_semantic_shard() -> None:
    validate_score_mode_target_compatibility(
        score_mode="semantic_shard_handcrafted_v1",
        graph_kind="semantic_shard",
    )
    with pytest.raises(ValueError, match="semantic_shard"):
        validate_score_mode_target_compatibility(
            score_mode="semantic_shard_handcrafted_v1",
            graph_kind="anchor",
        )


def test_run_experiment_dry_run_semantic_shard_target(tmp_path: Path) -> None:
    bundles = tmp_path / "graph_bundles"
    bundles.mkdir(parents=True, exist_ok=True)
    scoring = tmp_path / "scoring_runs"
    scoring.mkdir(parents=True, exist_ok=True)
    cfg = _minimal_experiment_cfg(
        tmp_path=tmp_path,
        mode="score_only",
        graph_bundle_root=bundles,
        scoring_output_root=scoring,
    )
    cfg["selection"]["score_targets"] = ["semantic_shard"]
    out = run_experiment(cfg, dry_run=True)
    assert out["dry_run"] is True
    assert out["community_results"][0]["target"] == "semantic_shard"
    cr = out["community_results"][0]
    assert "inputs" in cr
    assert "artifacts" in cr
    assert "metrics" in cr
    assert "semantic_shard" in str(cr["inputs"]["edges_csv"])


def test_run_experiment_rejects_invalid_target(tmp_path: Path) -> None:
    bundles = tmp_path / "graph_bundles"
    bundles.mkdir(parents=True, exist_ok=True)
    scoring = tmp_path / "scoring_runs"
    scoring.mkdir(parents=True, exist_ok=True)
    cfg = _minimal_experiment_cfg(
        tmp_path=tmp_path,
        mode="score_only",
        graph_bundle_root=bundles,
        scoring_output_root=scoring,
    )
    cfg["selection"]["score_targets"] = ["not_a_target"]
    with pytest.raises(ValueError, match="Unsupported targets"):
        run_experiment(cfg, dry_run=True)
