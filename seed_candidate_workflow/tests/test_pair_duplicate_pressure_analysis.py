from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from seed_candidate_workflow.utils.pair_duplicate_pressure_analysis import run_pair_duplicate_pressure


def test_run_pair_duplicate_pressure_smoke(tmp_path: Path) -> None:
    mem_rows = [
        {
            "external_id": "e1",
            "signature_type": "strict_full_email",
            "cluster_id": "clusterA_full",
            "signature_hash12": "aaaaaaaaaaaa",
            "group_size": 2,
            "cluster_rank_by_size": 1,
        },
        {
            "external_id": "e2",
            "signature_type": "strict_full_email",
            "cluster_id": "clusterA_full",
            "signature_hash12": "aaaaaaaaaaaa",
            "group_size": 2,
            "cluster_rank_by_size": 1,
        },
        {
            "external_id": "e1",
            "signature_type": "content_subject_body",
            "cluster_id": "clusterB_content",
            "signature_hash12": "bbbbbbbbbbbb",
            "group_size": 2,
            "cluster_rank_by_size": 1,
        },
        {
            "external_id": "e2",
            "signature_type": "content_subject_body",
            "cluster_id": "clusterB_content",
            "signature_hash12": "bbbbbbbbbbbb",
            "group_size": 2,
            "cluster_rank_by_size": 1,
        },
    ]
    p_mem = tmp_path / "email_duplicate_cluster.parquet"
    pd.DataFrame(mem_rows).to_parquet(p_mem, index=False)

    p_loaded = tmp_path / "misp_loaded_external_ids.parquet"
    pd.DataFrame({"external_id": ["e1", "e2", "e3"]}).to_parquet(p_loaded, index=False)

    p_meta = tmp_path / "g.meta.json"
    p_meta.write_text(json.dumps({"email_attrs": {"external_id": ["e1", "e2", "e3"]}}), encoding="utf-8")

    pair_rows = [
        {
            "email_i": "e1",
            "email_j": "e2",
            "graph_email_idx_i": 0,
            "graph_email_idx_j": 1,
            "pair_status": "positive",
            "is_seed_pair": True,
            "is_candidate_pair": True,
            "from_seed": True,
            "from_semantic": False,
            "from_component": False,
            "from_2hop": False,
            "from_rare_artifact": False,
            "source_count": 1,
            "semantic_cosine_max": 0.95,
            "cross_seed_component_flag": False,
            "same_seed_component_flag": True,
        },
        {
            "email_i": "e2",
            "email_j": "e3",
            "graph_email_idx_i": 1,
            "graph_email_idx_j": 2,
            "pair_status": "unlabeled",
            "is_seed_pair": False,
            "is_candidate_pair": True,
            "from_seed": False,
            "from_semantic": True,
            "from_component": True,
            "from_2hop": False,
            "from_rare_artifact": False,
            "source_count": 2,
            "semantic_cosine_max": 0.4,
            "cross_seed_component_flag": True,
            "same_seed_component_flag": False,
        },
    ]
    p_pair = tmp_path / "pair_training_dataset.csv"
    pd.DataFrame(pair_rows).to_csv(p_pair, index=False)

    out_dir = tmp_path / "out"
    summary = run_pair_duplicate_pressure(
        pair_csv=p_pair,
        membership_parquet=p_mem,
        out_dir=out_dir,
        graph_meta_json=p_meta,
        misp_loaded_ids_parquet=p_loaded,
        training_rows_only=True,
        apply_split=True,
        pair_val_ratio=0.1,
        pair_test_ratio=0.1,
        pair_split_seed=0,
        write_augmented_parquet=True,
    )

    assert (out_dir / "pair_duplicate_pressure_summary.json").is_file()
    assert summary["per_signature_type"]["strict_full_email"]["n_dup_same_cluster_rows"] == 1
    assert summary["per_signature_type"]["near_template_subject_body_sender"]["n_dup_neither"] == 2
    pot = summary["potential_vs_realized"]["strict_full_email"]
    assert pot["E_duplicate_potential_edges_in_graph"] == 1
    assert pot["E_pair_rows_dup_same_cluster_both_endpoints_in_graph"] == 1
    assert (out_dir / "pair_duplicate_labeled_rows.parquet").is_file()
