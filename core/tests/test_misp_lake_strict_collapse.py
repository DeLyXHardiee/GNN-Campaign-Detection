"""Tests for strict MISP lake dedupe (collapse_misp_lake_strict_duplicates)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DEDUP_SCRIPTS_DIR = REPO / "scripts" / "misp_lake_dedup"
if str(DEDUP_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(DEDUP_SCRIPTS_DIR))

import misp_email_identity as mei  # noqa: E402
from collapse_misp_lake_strict_duplicates import remap_ground_truth_json, run_collapse  # noqa: E402


def _event(ext_id: str, subject: str, body: str, sender: str = "a@b.com") -> dict:
    return {
        "Event": {
            "external_id": ext_id,
            "Attribute": [
                {"type": "subject", "value": subject},
                {"type": "body", "value": body},
                {"type": "from", "value": sender},
            ],
        }
    }


def test_collapse_merges_strict_dup_lexicographic_representative(tmp_path: Path) -> None:
    """Two identical emails: smaller external_id becomes representative; solo row kept."""
    inp = tmp_path / "in.json"
    out_json = tmp_path / "out.json"
    out_dir = tmp_path / "sidecar"
    inp.write_text(
        json.dumps(
            [
                _event("zz_loser", "Subj", "Body1"),
                _event("aa_winner", "Subj", "Body1"),
                _event("solo_id", "Other", "Body2"),
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    run_collapse(
        input_json=inp,
        out_json=out_json,
        out_dir=out_dir,
        collapse_signature_type=mei.SIGNATURE_STRICT_FULL_EMAIL,
        max_events=None,
        top_k=5,
        ground_truth_in=None,
        ground_truth_out=None,
    )

    out_events = json.loads(out_json.read_text(encoding="utf-8"))
    assert len(out_events) == 2

    ids = sorted(mei._extract_email_record(raw, i).external_id for i, raw in enumerate(out_events))
    assert ids == ["aa_winner", "solo_id"]

    summ = json.loads((out_dir / "collapse_summary.json").read_text(encoding="utf-8"))
    assert summ["n_events_in"] == 3
    assert summ["n_events_out"] == 2
    assert summ["n_events_removed"] == 1
    assert summ["n_duplicate_clusters_merged"] == 1
    delta = summ["delta"]
    assert delta["estimated_intra_duplicate_easy_edges_removed"] == 1
    assert delta["all_possible_pairs_removed"] == 2                     

    clusters = json.loads((out_dir / "collapsed_clusters.json").read_text(encoding="utf-8"))
    assert len(clusters) == 1
    assert clusters[0]["representative_external_id"] == "aa_winner"
    assert set(clusters[0]["member_external_ids"]) == {"aa_winner", "zz_loser"}


def test_remap_ground_truth_dedupes_ids(tmp_path: Path) -> None:
    gt_in = tmp_path / "gt.json"
    gt_out = tmp_path / "gt_out.json"
    gt_in.write_text(
        json.dumps(
            {
                "clusters": {
                    "c1": [
                        {"external_id": "zz_loser"},
                        {"external_id": "aa_winner"},
                    ]
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    id_map = {"zz_loser": "aa_winner", "aa_winner": "aa_winner"}
    remap_ground_truth_json(gt_in, gt_out, id_map)
    data = json.loads(gt_out.read_text(encoding="utf-8"))
    assert data["clusters"]["c1"] == [{"external_id": "aa_winner"}]


def test_sig_strict_stable_across_extract_index() -> None:
    r0 = mei._extract_email_record(_event("id-a", "S", "B"), 0)
    r9 = mei._extract_email_record(_event("id-b", "S", "B"), 9)
    assert mei._sig_strict(r0) == mei._sig_strict(r9)
