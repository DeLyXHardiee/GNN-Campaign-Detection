"""Tests for semantic supernode collapse (merge, ids, embeddings overlay)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from core.graph.semantic_supernode_collapse import (
    build_collapsed_emails_and_nodes,
    merge_parsed_email_dicts,
    stable_supernode_graph_external_id,
    write_embeddings_overlay,
)


def test_stable_supernode_id_is_deterministic():
    a = stable_supernode_graph_external_id(["z", "a", "m"])
    b = stable_supernode_graph_external_id(["m", "a", "z"])
    assert a == b
    assert a.startswith("sem_sn_")


def test_merge_unions_senders_and_sets_external_id():
    m1 = {
        "email_info": "",
        "email_index": 0,
        "external_id": "e1",
        "senders": ["a@x.com"],
        "receivers": ["b@x.com"],
        "subject": "s1",
        "body": "b1",
        "html": {},
        "css": {},
        "attachments": ["h1"],
        "attachment_metadata": [],
        "urls": ["http://a"],
        "date": "2020-01-01 00:00:00",
        "received_hops": [],
        "cyrillic_domain": "false",
        "contains_symbols": "false",
        "body_has_tracking_url": "false",
        "body_has_tracking_image": "false",
        "body_has_tracking_pixel": "false",
        "body_has_unsubscribe_link": "false",
        "domain_is_common_webprovided": "false",
        "return_path": {},
        "auth_spf": "",
        "auth_dkim": "",
        "auth_dmarc": "",
    }
    m2 = dict(m1)
    m2["external_id"] = "e2"
    m2["senders"] = ["c@x.com"]
    m2["urls"] = ["http://a", "http://b"]
    m2["body_has_tracking_url"] = "true"
    gid = stable_supernode_graph_external_id(["e1", "e2"])
    out = merge_parsed_email_dicts([m1, m2], graph_external_id=gid, representative_external_id="e1")
    assert out["external_id"] == gid
    assert set(out["senders"]) == {"a@x.com", "c@x.com"}
    assert out["subject"] == "s1"
    assert out["body_has_tracking_url"] == "true"


def test_merge_keeps_distinct_html_structure_fingerprints():
    """Supernodes must retain every distinct member structure_fingerprint (dedupe identical only)."""
    base_html = {"tag_counts": {"div": 1}, "tree_stats": {}, "structure_fingerprint": "aaaabbbbccccdddd"}
    m1 = {
        "email_info": "",
        "email_index": 0,
        "external_id": "e1",
        "senders": [],
        "receivers": [],
        "subject": "s1",
        "body": "b1",
        "html": dict(base_html),
        "css": {},
        "attachments": [],
        "attachment_metadata": [],
        "urls": [],
        "date": "2020-01-01 00:00:00",
        "received_hops": [],
        "cyrillic_domain": "false",
        "contains_symbols": "false",
        "body_has_tracking_url": "false",
        "body_has_tracking_image": "false",
        "body_has_tracking_pixel": "false",
        "body_has_unsubscribe_link": "false",
        "domain_is_common_webprovided": "false",
        "return_path": {},
        "auth_spf": "",
        "auth_dkim": "",
        "auth_dmarc": "",
    }
    m2 = dict(m1)
    m2["external_id"] = "e2"
    m2["html"] = {
        "tag_counts": {"a": 2},
        "tree_stats": {},
        "structure_fingerprint": "1111222233334444",
    }
    m3 = dict(m1)
    m3["external_id"] = "e3"
    m3["html"] = {
        "tag_counts": {},
        "tree_stats": {},
        "structure_fingerprint": "aaaabbbbccccdddd",
    }
    gid = stable_supernode_graph_external_id(["e1", "e2", "e3"])
    out = merge_parsed_email_dicts(
        [m1, m2, m3], graph_external_id=gid, representative_external_id="e1"
    )
    html = out.get("html") or {}
    assert html.get("structure_fingerprints") == [
        "aaaabbbbccccdddd",
        "1111222233334444",
    ]
    assert html.get("structure_fingerprint") == "aaaabbbbccccdddd"


def _two_misp_events():
    return [
        {
            "Event": {
                "info": "t1",
                "email_index": 0,
                "external_id": "ext_alpha",
                "Attribute": [
                    {"type": "from", "value": "a@phish.com"},
                    {"type": "email-subject", "value": "Sub A"},
                ],
            }
        },
        {
            "Event": {
                "info": "t2",
                "email_index": 1,
                "external_id": "ext_beta",
                "Attribute": [
                    {"type": "from", "value": "b@phish.com"},
                    {"type": "email-subject", "value": "Sub B"},
                ],
            }
        },
    ]


def test_build_collapsed_emails_supernode_and_singleton(tmp_path: Path):
    csv = tmp_path / "cl.csv"
    pd.DataFrame(
        {
            "email_id": ["ext_alpha", "ext_beta", "ext_gamma"],
            "cluster_id": [1, 1, 2],
            "cluster_size": [2, 2, 1],
            "representative_email_id": ["ext_alpha", "ext_alpha", "ext_gamma"],
            "cosine_to_representative": [1.0, 0.9, 1.0],
        }
    ).to_csv(csv, index=False)

    emails, nodes, meta = build_collapsed_emails_and_nodes(
        misp_events=_two_misp_events()
        + [
            {
                "Event": {
                    "info": "t3",
                    "email_index": 2,
                    "external_id": "ext_gamma",
                    "Attribute": [{"type": "from", "value": "solo@phish.com"}],
                }
            }
        ],
        clusters_csv=csv,
    )
    assert meta["n_graph_email_nodes"] == 2
    kinds = {n["graph_external_id"]: n["kind"] for n in nodes}
    assert kinds["ext_gamma"] == "singleton"
    sn_ids = [n["graph_external_id"] for n in nodes if n["kind"] == "supernode"]
    assert len(sn_ids) == 1
    assert sn_ids[0].startswith("sem_sn_")
    ext_ids = {e["external_id"] for e in emails}
    assert "ext_gamma" in ext_ids
    assert sn_ids[0] in ext_ids


def test_embeddings_overlay_mean_matches_expected(tmp_path: Path):
    src = tmp_path / "src.json"
    src.write_text(
        json.dumps(
            {
                "model": "test",
                "subj_dim": 2,
                "body_dim": 2,
                "by_key": {
                    "p": {"subj": [1.0, 0.0], "body": [0.0, 2.0], "external_id": "p"},
                    "q": {"subj": [3.0, 0.0], "body": [0.0, 4.0], "external_id": "q"},
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    gid = stable_supernode_graph_external_id(["p", "q"])
    nodes = [
        {"graph_external_id": gid, "kind": "supernode", "member_external_ids": ["p", "q"]},
        {"graph_external_id": "p", "kind": "singleton", "member_external_ids": ["p"]},
    ]
    out_json = write_embeddings_overlay(
        source_embeddings_json=src,
        output_dir=tmp_path / "ov",
        nodes=nodes,
        l2_normalize_after_mean=False,
    )
    data = json.loads(out_json.read_text(encoding="utf-8"))
    sn = data["by_key"][gid]
    np.testing.assert_allclose(sn["subj"], [2.0, 0.0], rtol=1e-5)
    np.testing.assert_allclose(sn["body"], [0.0, 3.0], rtol=1e-5)


def test_build_graph_parsed_emails_smoke(tmp_path: Path):
    torch_geometric = pytest.importorskip("torch_geometric", reason="torch-geometric required")
    _ = torch_geometric
    from core.config.pipeline_config import EmailFeatureProjectionSettings
    from core.graph.graph_builder_pytorch import build_graph

    m1 = {
        "email_info": "",
        "email_index": 0,
        "external_id": "smoke_a",
        "senders": ["s@x.com"],
        "receivers": [],
        "subject": "x",
        "body": "y",
        "html": {},
        "css": {},
        "attachments": [],
        "attachment_metadata": [],
        "urls": [],
        "date": "",
        "received_hops": [],
        "cyrillic_domain": "false",
        "contains_symbols": "false",
        "body_has_tracking_url": "false",
        "body_has_tracking_image": "false",
        "body_has_tracking_pixel": "false",
        "body_has_unsubscribe_link": "false",
        "domain_is_common_webprovided": "false",
        "return_path": {},
        "auth_spf": "",
        "auth_dkim": "",
        "auth_dmarc": "",
    }
    emb_path = tmp_path / "embeddings.json"
    emb_path.write_text(
        json.dumps(
            {
                "model": "test",
                "subj_dim": 2,
                "body_dim": 2,
                "by_key": {
                    "smoke_a": {"subj": [1.0, 0.0], "body": [0.0, 1.0], "external_id": "smoke_a"},
                },
            }
        ),
        encoding="utf-8",
    )
    graph, gp, mp = build_graph(
        parsed_emails=[m1],
        misp_json_path=None,
        misp_events=None,
        out_dir=str(tmp_path),
        out_name="smoke_hetero.pt",
        embeddings_output_dir=str(tmp_path),
        email_feature_projection=EmailFeatureProjectionSettings(seed=42, bert_out_dim=4, other_out_dim=4),
    )
    assert hasattr(graph, "node_types")
    assert Path(gp).is_file()
    assert Path(mp).is_file()
