from __future__ import annotations

from pathlib import Path

from seed_candidate_workflow.utils import semantic_supernode_gt_metrics as m


def test_expand_supernode_inherits_community(tmp_path: Path) -> None:
    mp = tmp_path / "semantic_supernode_mapping.json"
    mp.write_text(
        """
{
  "nodes": [
    {"graph_external_id": "e1", "kind": "singleton", "member_external_ids": ["e1"]},
    {"graph_external_id": "sem_sn_abc", "kind": "supernode", "member_external_ids": ["a", "b", "c"]}
  ]
}
""",
        encoding="utf-8",
    )
    tab = m.load_semantic_supernode_member_table(mp)
    pred = {"e1": 0, "sem_sn_abc": 7}
    ex = m.expand_pred_map_for_gt_eval(pred, tab)
    assert ex["e1"] == 0
    assert ex["a"] == ex["b"] == ex["c"] == 7
    assert len(ex) == 4


def test_member_emails_union() -> None:
    tab = {
        "e1": ["e1"],
        "sem_sn_x": ["a", "b"],
    }
    cov = m.member_emails_represented_by_graph_nodes(["e1", "sem_sn_x"], tab)
    assert cov == {"e1", "a", "b"}


def test_resolve_mapping_path_relative(tmp_path: Path) -> None:
    root = tmp_path / "proj"
    root.mkdir()
    p = m.resolve_optional_mapping_path(root, "data/map.json")
    assert p == root / "data" / "map.json"


def test_dedup_external_id_map_expansion(tmp_path: Path) -> None:
    p = tmp_path / "external_id_map.csv"
    p.write_text(
        "external_id,representative_external_id\n"
        "aa_winner,aa_winner\n"
        "zz_loser,aa_winner\n"
        "solo,solo\n",
        encoding="utf-8",
    )
    tab = m.load_dedup_collapse_member_table_from_external_id_map(p)
    assert tab["aa_winner"] == ["aa_winner", "zz_loser"]
    assert tab["solo"] == ["solo"]
    pred = {"aa_winner": 3, "solo": 1}
    ex = m.expand_pred_map_for_gt_eval(pred, tab)
    assert ex["zz_loser"] == 3
    assert ex["aa_winner"] == 3
    assert ex["solo"] == 1


def test_resolve_dedup_collapse_out_dir(tmp_path: Path) -> None:
    out_dir = tmp_path / "sidecar"
    out_dir.mkdir()
    (out_dir / "external_id_map.csv").write_text(
        "external_id,representative_external_id\nr1,r1\nr2,r1\n",
        encoding="utf-8",
    )
    tab = m.load_dedup_collapse_member_table_from_out_dir(out_dir)
    assert tab["r1"] == ["r1", "r2"]
    root = tmp_path / "proj"
    root.mkdir()
    got, path, src = m.resolve_gid_to_members_for_gt_eval(
        root, dedup_collapse_out_dir=str(out_dir)
    )
    assert src == "dedup_collapse_out_dir"
    assert path == out_dir.resolve()
    assert got == tab
