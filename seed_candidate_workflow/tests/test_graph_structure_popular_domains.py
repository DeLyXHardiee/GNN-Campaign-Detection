"""Tests for exact-hostname popular-domain filtering in graph_structure_helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from seed_candidate_workflow.utils.graph_structure_helpers import ensure_core_gnn_on_path

ensure_core_gnn_on_path(PROJECT_ROOT)

from seed_candidate_workflow.utils import graph_structure_helpers as gh

_POPULAR = frozenset({"google.com", "facebook.com"})


@pytest.mark.parametrize(
    ("url", "expected_kind"),
    [
        ("https://www.google.com/search", "benign"),
        ("https://google.com/search", "benign"),
        ("https://groups.google.com/a/example/group", "kept"),
        ("https://docs.google.com/document/d/1", "kept"),
        ("https://example.com/path", "kept"),
    ],
)
def test_classify_url_for_popular_filter(url: str, expected_kind: str) -> None:
    kind, _ = gh.classify_url_for_popular_filter(url, _POPULAR)
    assert kind == expected_kind


def test_classify_url_for_popular_filter_malformed_empty() -> None:
    kind, label = gh.classify_url_for_popular_filter("", _POPULAR)
    assert kind == "malformed"
    assert label == ""


def test_url_matches_popular_domain_exact_host_only() -> None:
    matched, label = gh.url_matches_popular_domain("https://www.google.com/x", _POPULAR)
    assert matched is True
    assert label == "google.com"

    matched, _ = gh.url_matches_popular_domain(
        "https://groups.google.com/a/isa.pendlerty.de/group/rp/subscribe",
        _POPULAR,
    )
    assert matched is False


def test_classify_differs_from_registrable_domain_coarse_logic() -> None:
    from core.feature_set_extraction.url_extraction_utils import shard_url_infra_classify

    url = "https://groups.google.com/a/campaign/path"
    kind_exact, _ = gh.classify_url_for_popular_filter(url, _POPULAR)
    kind_coarse, _ = shard_url_infra_classify(url, _POPULAR)
    assert kind_exact == "kept"
    assert kind_coarse == "benign"
