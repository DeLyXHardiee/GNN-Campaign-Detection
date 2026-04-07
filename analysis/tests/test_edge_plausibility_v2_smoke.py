"""Smoke test for Method 1 V2 plausibility pipeline (tiny graph)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.utils.semantic_shard_edge_plausibility_v2_config import EdgePlausibilityV2Config
from analysis.utils.semantic_shard_edge_plausibility_v2_train import train_and_score_edge_plausibility


def _tiny_edges():
    return pd.DataFrame(
        [
            {
                "shard_a": "s0",
                "shard_b": "s1",
                "centroid_cosine": 0.9,
                "infra_score": 0.8,
                "temporal_score": 0.7,
                "shared_url_count": 1,
                "shared_url_idf_sum": 0.5,
                "infra_contrib_url": 0.5,
                "url_jaccard": 0.2,
                "edge_weight": 0.9,
            },
            {
                "shard_a": "s1",
                "shard_b": "s2",
                "centroid_cosine": 0.2,
                "infra_score": 0.9,
                "temporal_score": 0.1,
                "shared_url_count": 0,
                "shared_url_idf_sum": 0.0,
                "infra_contrib_url": 0.9,
                "url_jaccard": 0.0,
                "edge_weight": 0.5,
            },
            {
                "shard_a": "s0",
                "shard_b": "s2",
                "centroid_cosine": 0.5,
                "infra_score": 0.4,
                "temporal_score": 0.5,
                "shared_url_count": 0,
                "shared_url_idf_sum": 0.0,
                "infra_contrib_url": 0.2,
                "url_jaccard": 0.0,
                "edge_weight": 0.4,
            },
        ]
    )


def _tiny_nodes():
    return pd.DataFrame(
        [
            {"shard_id": "s0", "size": 3, "n_unique_urls": 1, "ts_span_seconds": 100.0},
            {"shard_id": "s1", "size": 2, "n_unique_urls": 0, "ts_span_seconds": 50.0},
            {"shard_id": "s2", "size": 4, "n_unique_urls": 0, "ts_span_seconds": 200.0},
        ]
    )


def test_v2_train_smoke_cpu(tmp_path):
    edges = _tiny_edges()
    nodes = _tiny_nodes()
    cfg = EdgePlausibilityV2Config(
        random_seed=0,
        run_id="pytest_v2",
        epochs=2,
        batch_size=2,
        n_ranking_pairs_per_batch=4,
        output_root=str(tmp_path),
        log_gt_separation=False,
    )
    out = train_and_score_edge_plausibility(edges, nodes, cfg, device="cpu")
    df = out["scored_edges_df"]
    assert "edge_plausibility" in df.columns
    assert len(df) == 3
    assert df["edge_plausibility"].between(0.0, 1.0).all()
    meta_path = Path(tmp_path) / "pytest_v2" / "ranking_supervision_meta.json"
    assert meta_path.is_file()
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert "counts" in meta and "per_epoch_pair_supervision" in meta
    assert len(meta["per_epoch_pair_supervision"]) == 2
    assert "bucket_counts_by_split" in meta and meta["bucket_counts_by_split"]["full"]["n_edges_in_split"] == 3
    assert "positive_subpath_counts_by_split" in meta
