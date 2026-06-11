"""Tests for body Jaccard generator output cache."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from seed_candidate_workflow.utils.body_jaccard_generator_cache import (
    cache_dir_for_generator_manifest,
    generator_output_manifest,
    save_cached_generator_output,
    try_load_cached_generator_output,
)
from seed_candidate_workflow.utils.body_similarity_cache import (
    build_body_similarity_content_fingerprint,
)


def test_generator_output_cache_round_trip(tmp_path: Path):
    misp = tmp_path / "misp.json"
    misp.write_text("{}", encoding="utf-8")
    email_fp = build_body_similarity_content_fingerprint(
        misp_json_path=misp,
        email_ids=["e1", "e2"],
        min_token_len=2,
        char_n=4,
    )
    manifest = generator_output_manifest(
        email_content_fp=email_fp,
        generator_name="body_char4gram_jaccard_highconf_v1",
        mode="char4",
        min_jaccard=0.25,
        max_candidate_rows=500_000,
        use_filtered_inverted_index=True,
        max_token_document_frequency=40,
        max_char4gram_document_frequency=60,
        prior_pair_pool_hash="abc123",
        semantic_band_pool_hash="def456",
    )
    cache_dir = cache_dir_for_generator_manifest(
        cache_root=tmp_path / "gen_cache",
        email_content_fp=email_fp,
        generator_manifest=manifest,
    )
    df = pd.DataFrame(
        [
            {
                "email_i": "e1",
                "email_j": "e2",
                "source": "body_char4gram_jaccard_highconf_v1",
                "body_char4gram_jaccard": 0.5,
                "body_token_jaccard": 0.3,
            }
        ]
    )
    save_cached_generator_output(df, cache_dir=cache_dir, manifest=manifest)
    loaded, diag = try_load_cached_generator_output(cache_dir, expected_manifest=manifest)
    assert diag["cache_status"] == "hit"
    assert loaded is not None
    assert len(loaded) == 1
