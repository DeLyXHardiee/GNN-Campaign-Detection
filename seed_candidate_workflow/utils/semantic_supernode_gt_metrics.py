"""
Ground-truth metric helpers when graph nodes represent more than one email.

Community detection still runs on graph node ids (representatives, ``sem_sn_*``,
etc.). For homogeneity / completeness / v-measure vs email-level ground truth,
predicted communities are expanded so every member email inherits the graph
node's community id.

Supports:
  - ``semantic_supernode_mapping.json`` (``nodes[].graph_external_id``)
  - MISP dedup collapse sidecars (``external_id_map.csv``, ``collapsed_clusters.json``)
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path


def load_semantic_supernode_member_table(mapping_path: Path) -> dict[str, list[str]]:
    """
    Load ``semantic_supernode_mapping.json`` and return
    ``graph_external_id -> member_external_ids``.
    """
    raw = json.loads(mapping_path.read_text(encoding="utf-8"))
    out: dict[str, list[str]] = {}
    for node in raw.get("nodes") or []:
        if not isinstance(node, dict):
            continue
        gid = str(node.get("graph_external_id") or "").strip()
        if not gid:
            continue
        mem = node.get("member_external_ids") or []
        if not isinstance(mem, list):
            continue
        members = [str(x).strip() for x in mem if str(x).strip()]
        if members:
            out[gid] = members
    return out


def member_emails_represented_by_graph_nodes(
    node_ids: list[str],
    gid_to_members: dict[str, list[str]],
) -> set[str]:
    """All member external_ids covered by the anchor graph's node list."""
    covered: set[str] = set()
    for nid in node_ids:
        s = str(nid)
        members = gid_to_members.get(s)
        if members:
            covered.update(members)
        else:
            covered.add(s)
    return covered


def expand_pred_map_for_gt_eval(
    pred_map: dict[str, int],
    gid_to_members: dict[str, list[str]],
) -> dict[str, int]:
    """
    Map graph-node communities to per-member-email communities for external metrics.

    Graph nodes not present in ``gid_to_members`` are copied through unchanged
    (backwards-compatible with non-supernode graphs if an incomplete mapping is
    accidentally passed).
    """
    expanded: dict[str, int] = {}
    for gid, cid in pred_map.items():
        sg = str(gid)
        members = gid_to_members.get(sg)
        if members:
            c = int(cid)
            for m in members:
                expanded[str(m)] = c
        else:
            expanded[sg] = int(cid)
    return expanded


def resolve_optional_mapping_path(project_root: Path, raw: str | None) -> Path | None:
    t = (raw or "").strip()
    if not t:
        return None
    p = Path(t).expanduser()
    if not p.is_absolute():
        p = (project_root / p).resolve()
    else:
        p = p.resolve()
    return p


def load_dedup_collapse_member_table_from_external_id_map(path: Path) -> dict[str, list[str]]:
    """
    Load ``representative_external_id -> [all member external_ids]`` from collapse CSV.

    Every row maps ``external_id`` to its representative; singletons map to themselves.
    """
    path = path.expanduser().resolve()
    by_rep: dict[str, set[str]] = defaultdict(set)
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return {}
        for row in reader:
            rep = str(row.get("representative_external_id") or "").strip()
            eid = str(row.get("external_id") or "").strip()
            if not rep or not eid:
                continue
            by_rep[rep].add(eid)
    return {rep: sorted(members) for rep, members in by_rep.items() if members}


def load_dedup_collapse_member_table_from_collapsed_clusters(path: Path) -> dict[str, list[str]]:
    """Load multi-member collapse clusters only (singleton reps are omitted)."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"collapsed_clusters.json must be a JSON array: {path}")
    out: dict[str, list[str]] = {}
    for cluster in raw:
        if not isinstance(cluster, dict):
            continue
        rep = str(cluster.get("representative_external_id") or "").strip()
        mem = cluster.get("member_external_ids") or []
        if not rep or not isinstance(mem, list):
            continue
        members = sorted({str(x).strip() for x in mem if str(x).strip()})
        if members:
            out[rep] = members
    return out


def merge_member_tables(*tables: dict[str, list[str]]) -> dict[str, list[str]]:
    """Union member lists per graph node id (later tables extend earlier)."""
    merged: dict[str, set[str]] = defaultdict(set)
    for tab in tables:
        for gid, members in tab.items():
            merged[str(gid)].update(str(m) for m in members)
    return {gid: sorted(members) for gid, members in merged.items() if members}


def load_member_expansion_table(path: Path) -> dict[str, list[str]]:
    """
    Load a graph-node -> member-email table from a known artifact path.

    Auto-detects format by filename; semantic supernode JSON is the fallback.
    """
    path = path.expanduser().resolve()
    name = path.name.lower()
    if name == "external_id_map.csv":
        return load_dedup_collapse_member_table_from_external_id_map(path)
    if name == "collapsed_clusters.json":
        return load_dedup_collapse_member_table_from_collapsed_clusters(path)
    if name.endswith(".json"):
        tab = load_semantic_supernode_member_table(path)
        if tab:
            return tab
        raw = json.loads(path.read_text(encoding="utf-8"))
        nodes = raw.get("nodes")
        if isinstance(nodes, list) and nodes:
            return load_semantic_supernode_member_table(path)
    raise ValueError(f"Unrecognized member-expansion mapping file: {path}")


def load_dedup_collapse_member_table_from_out_dir(out_dir: Path) -> dict[str, list[str]]:
    """
    Prefer ``external_id_map.csv`` (full lake coverage); merge ``collapsed_clusters.json``.
    """
    out_dir = out_dir.expanduser().resolve()
    p_csv = out_dir / "external_id_map.csv"
    p_clusters = out_dir / "collapsed_clusters.json"
    tables: list[dict[str, list[str]]] = []
    if p_csv.is_file():
        tables.append(load_dedup_collapse_member_table_from_external_id_map(p_csv))
    elif p_clusters.is_file():
        tables.append(load_dedup_collapse_member_table_from_collapsed_clusters(p_clusters))
    else:
        raise FileNotFoundError(
            f"dedup collapse out_dir missing external_id_map.csv and collapsed_clusters.json: {out_dir}"
        )
    if p_clusters.is_file() and p_csv.is_file():
        tables.append(load_dedup_collapse_member_table_from_collapsed_clusters(p_clusters))
    tab = merge_member_tables(*tables)
    if not tab:
        raise ValueError(f"dedup collapse out_dir produced empty member table: {out_dir}")
    return tab


def write_member_expansion_mapping_json(
    path: Path,
    *,
    gid_to_members: dict[str, list[str]],
    meta: dict[str, object] | None = None,
) -> None:
    """Write semantic-supernode-compatible mapping JSON for caching / reuse."""
    nodes = [
        {
            "graph_external_id": gid,
            "kind": "collapse_representative" if len(members) > 1 else "singleton",
            "member_external_ids": members,
        }
        for gid, members in sorted(gid_to_members.items(), key=lambda kv: kv[0])
    ]
    payload = {
        "schema_version": 1,
        "mapping_kind": "dedup_collapse_member_expansion",
        "meta": meta or {},
        "nodes": nodes,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def resolve_gid_to_members_for_gt_eval(
    project_root: Path,
    *,
    member_expansion_mapping_json: str | None = None,
    semantic_supernode_mapping_json: str | None = None,
    dedup_collapse_out_dir: str | None = None,
) -> tuple[dict[str, list[str]] | None, Path | None, str]:
    """
    Resolve optional member-expansion table from experiment/community config.

    Returns ``(gid_to_members or None, resolved_path or None, source_label)``.
    """
    for raw, label in (
        (member_expansion_mapping_json, "member_expansion_mapping_json"),
        (semantic_supernode_mapping_json, "semantic_supernode_mapping_json"),
    ):
        p = resolve_optional_mapping_path(project_root, raw)
        if p is None:
            continue
        if not p.is_file():
            return None, p, label
        return load_member_expansion_table(p), p, label

    out_raw = (dedup_collapse_out_dir or "").strip()
    if not out_raw:
        return None, None, ""
    p_dir = resolve_optional_mapping_path(project_root, out_raw)
    if p_dir is None:
        return None, None, ""
    if not p_dir.is_dir():
        return None, p_dir, "dedup_collapse_out_dir"
    return load_dedup_collapse_member_table_from_out_dir(p_dir), p_dir, "dedup_collapse_out_dir"
