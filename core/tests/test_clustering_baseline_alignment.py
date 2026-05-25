import sys
from pathlib import Path

import numpy as np

_GNN_ROOT = Path(__file__).resolve().parents[1] / "GNN"
if str(_GNN_ROOT) not in sys.path:
    sys.path.insert(0, str(_GNN_ROOT))

from core.GNN.steps.cluster_stage import (
    _prepare_embedding_map_for_clustering,
    _resolve_bert_max_components,
)
from core.clustering.clusteringMetrics import compute_all_metrics, compute_external_metrics
from core.metric_comparison.artifacts import pred_map_from_campaign_payload
from core.visualization.campaign_utils import build_campaign_artifact_payload


def test_compute_all_metrics_treats_noise_as_singletons_for_external_scores():
    id_to_embedding_map = {
        "email_a": np.asarray([0.0, 0.0], dtype=np.float64),
        "email_b": np.asarray([1.0, 0.0], dtype=np.float64),
        "email_c": np.asarray([2.0, 0.0], dtype=np.float64),
    }
    labels = np.asarray([0, -1, -1], dtype=np.int64)
    ground_truth_labels = {
        "email_a": "campaign_alpha",
        "email_b": "campaign_beta",
        "email_c": "campaign_gamma",
    }

    metrics = compute_all_metrics(id_to_embedding_map, labels, ground_truth_labels)
    collapsed_noise = compute_external_metrics(
        ["campaign_alpha", "campaign_beta", "campaign_gamma"],
        [0, -1, -1],
    )

    assert metrics["homogeneity"] == 1.0
    assert metrics["homogeneity"] > collapsed_noise["homogeneity"]
    assert metrics["n_noise"] == 2
    assert metrics["n_clusters"] == 1
    assert metrics["coverage_ground_truth"] == 1 / 3


def test_campaign_artifact_round_trip_preserves_noise_singletons():
    payload = build_campaign_artifact_payload(
        solution="bert_embeddings",
        algorithm="dbscan",
        sorted_ids=["email_a", "email_b"],
        labels=np.asarray([-1, -1], dtype=np.int64),
        params={"epsilon": 0.3},
    )

    pred_map = pred_map_from_campaign_payload(payload)

    assert payload["n_noise"] == 2
    assert payload["n_non_noise_campaigns"] == 0
    assert payload["n_campaigns"] == 2
    assert [camp["id"] for camp in payload["campaigns"]] == ["noise_email_a", "noise_email_b"]
    assert set(pred_map) == {"email_a", "email_b"}
    assert pred_map["email_a"] != pred_map["email_b"]


def test_resolve_bert_max_components_allows_explicit_disable():
    assert _resolve_bert_max_components({}) == 256
    assert _resolve_bert_max_components({"max_components": None}) is None
    assert _resolve_bert_max_components({"max_components": 512}) == 512


def test_prepare_embedding_map_skips_svd_when_max_components_is_none():
    id_to_embedding_map = {
        "email_a": np.asarray([1.0, 0.0, 0.5, 0.25], dtype=np.float64),
        "email_b": np.asarray([0.0, 1.0, 0.25, 0.5], dtype=np.float64),
        "email_c": np.asarray([0.5, 0.5, 1.0, 0.0], dtype=np.float64),
    }

    full_width = _prepare_embedding_map_for_clustering(
        id_to_embedding_map=id_to_embedding_map,
        l2_normalize=False,
        max_components=None,
    )
    reduced_width = _prepare_embedding_map_for_clustering(
        id_to_embedding_map=id_to_embedding_map,
        l2_normalize=False,
        max_components=2,
    )

    assert len(next(iter(full_width.values()))) == 4
    assert len(next(iter(reduced_width.values()))) == 2
