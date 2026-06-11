"""Tests for dedup vs expanded GT gap analysis (synthetic fixtures)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from analysis.utils.dedup_vs_expanded_gap_analysis import (
    DedupExpandedGapConfig,
    run_dedup_vs_expanded_gap_analysis,
    write_dedup_vs_expanded_gap_outputs,
)


def _write_gt(path: Path, campaign_id: int, members: list[str]) -> None:
    key = f"label_store_x/{campaign_id}"
    payload = {
        "clusters": {
            key: [{"external_id": m} for m in members],
        }
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_run_dedup_vs_expanded_gap_smoke(tmp_path: Path) -> None:
    root = tmp_path
    anchor = root / "anchor"
    anchor.mkdir()
    pd.DataFrame({"external_id": ["a", "b"]}).to_csv(anchor / "anchor_graph_nodes.csv", index=False)

    pd.DataFrame(
        {
            "email_i": ["a", "b"],
            "email_j": ["b", "a"],
            "edge_weight": [1.0, 1.0],
        }
    ).to_csv(root / "edges.csv", index=False)

    pd.DataFrame(
        {
            "external_id": ["a", "b", "a_dup"],
            "representative_external_id": ["a", "b", "a"],
        }
    ).to_csv(root / "external_id_map.csv", index=False)

    p_dedup = root / "gt_dedup.json"
    p_exp = root / "gt_expanded.json"
    _write_gt(p_dedup, 1, ["a", "b"])
    _write_gt(p_exp, 1, ["a", "b", "a_dup"])

    best = {
        "method": "louvain",
        "resolution": 1.0,
        "min_edge_weight": 0.0,
        "use_edge_weights_in_partitioning": False,
        "homogeneity": 1.0,
        "completeness": 1.0,
        "v_measure": 1.0,
        "n_eval": 2.0,
        "coverage_gt": 1.0,
    }
    cfg = DedupExpandedGapConfig(
        project_root=root,
        gt_dedup_json=p_dedup,
        gt_expanded_json=p_exp,
        dedup_collapse_out_dir=None,
        external_id_map_csv=root / "external_id_map.csv",
        anchor_run_dir=anchor,
        scored_edges_csv=root / "edges.csv",
        expanded_best_row=dict(best),
        expanded_best_row_source="test",
        dedup_best_row=dict(best),
        dedup_best_row_source="test",
        use_edge_weights_in_partitioning=False,
        apply_threshold_filter=False,
        top_lossy_campaigns=10,
    )
    result = run_dedup_vs_expanded_gap_analysis(cfg)
    paths = write_dedup_vs_expanded_gap_outputs(result, root / "out")
    assert Path(paths["summary_json"]).is_file()
    summary = json.loads(Path(paths["summary_json"]).read_text(encoding="utf-8"))
    assert summary["schema"] == "dedup_vs_expanded_gap_summary_v2"
    assert "same_partition_comparison" in summary
    assert "best_to_best_comparison" in summary
    sp = summary["same_partition_comparison"]
    assert "expanded_selected" in sp
    assert "dedup_selected" in sp
    assert "deltas_expanded_minus_dedup" in sp["expanded_selected"]
    assert "metrics_on_dedup_gt" in sp["expanded_selected"]
    assert "metrics_on_expanded_gt" in sp["expanded_selected"]
    b2b = summary["best_to_best_comparison"]
    assert "deltas_expanded_best_minus_dedup_best" in b2b
    assert b2b["same_partition"] is True
    assert summary["lossy_campaign_analysis"]["comparison_mode"] == "same_partition"
    camp = result["campaign_table"]
    assert not camp.empty
    assert "delta_largest_share_expanded_minus_dedup" in camp.columns
