from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.pipelines.graph_setup_pipeline import run_graph_setup
from analysis.utils.community_eval_contract import evaluate_external_metrics
from analysis.utils.semantic_shard_step3_helpers import best_sweep_metric_row


def test_best_sweep_metric_row_vmeasure_tiebreaks_by_completeness() -> None:
    sweep_df = pd.DataFrame(
        [
            {"setting_key": "a", "v_measure": 0.8, "completeness": 0.70, "homogeneity": 0.90, "n_eval": 50},
            {"setting_key": "b", "v_measure": 0.8, "completeness": 0.75, "homogeneity": 0.88, "n_eval": 50},
        ]
    )
    best = best_sweep_metric_row(sweep_df, metric="v_measure")
    assert str(best["setting_key"]) == "b"


def test_run_graph_setup_shard_only_creates_only_shard_dirs(tmp_path: Path, monkeypatch) -> None:
    def _fake_build(*, project_root, graph_id, out_dir, stage_cfg):  # type: ignore[no-untyped-def]
        out_dir.mkdir(parents=True, exist_ok=True)
        p_pair = out_dir / "semantic_shard_pairgraph_unscored.csv"
        pd.DataFrame(
            columns=[
                "email_i",
                "email_j",
                "graph_kind",
                "graph_id",
                "from_seed",
                "from_semantic",
                "from_rare_artifact",
                "from_component",
                "from_2hop",
                "source_count",
            ]
        ).to_csv(p_pair, index=False)
        return {"pairgraph_unscored_csv": str(p_pair)}

    monkeypatch.setattr(
        "analysis.pipelines.graph_setup_pipeline._build_semantic_shard_pairgraph",
        _fake_build,
    )

    bundle_root = tmp_path / "graph_bundles"
    cfg = {
        "enable": {
            "anchor": False,
            "seed": False,
            "candidate": False,
            "seed_candidate": False,
            "pair_training": False,
            "semantic_shard": True,
        },
        "policy": {"on_missing": "build", "on_present": "reuse"},
        "paths": {"semantic_shard": {"embeddings_json": "x", "graph_pt": "x", "meta_json": "x"}},
    }
    out = run_graph_setup(
        project_root=tmp_path,
        graph_id="g1",
        graph_bundle_root=bundle_root,
        setup_cfg=cfg,
    )

    b = Path(out["bundle_root"])
    assert b.is_dir()
    assert (b / "semantic_shard").is_dir()
    assert not (b / "anchor").exists()
    assert not (b / "seed").exists()
    assert not (b / "candidate").exists()
    assert not (b / "seed_candidate").exists()
    assert not (b / "pair_training").exists()


def test_shared_evaluator_returns_expected_coverage_fields() -> None:
    gt = {"a": 0, "b": 0, "c": 1}
    pred = {"a": 1, "b": 1, "c": 2, "d": 3}
    m = evaluate_external_metrics(gt_label_map=gt, pred_label_map=pred, n_predictions_total=4)
    assert m["n_eval"] == 3.0
    assert m["coverage_gt"] == 1.0
    assert m["coverage_predictions"] == 0.75
