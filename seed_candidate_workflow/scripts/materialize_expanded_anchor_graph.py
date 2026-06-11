"""
Union anchor_graph_edges_unscored.csv and seed_candidate_pairgraph_unscored.csv into one
broad unlearned pair-evidence graph for thesis community-detection baselines.

Does not change generator rules or recompute Jaccard; only deduplicates canonical pairs.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import networkx as nx
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from seed_candidate_workflow.utils import graph_structure_helpers as gh

DEFAULT_GRAPH_ID = "main_gnn_pu_1_no_ts_dedup_task_identity_13"
GRAPH_KIND = "anchor_expanded"


def _pair_key(a: str, b: str) -> tuple[str, str]:
    aa, bb = str(a).strip(), str(b).strip()
    return (aa, bb) if aa <= bb else (bb, aa)


def _load_pairs(path: Path) -> set[tuple[str, str]]:
    df = pd.read_csv(path, low_memory=False)
    if df.empty:
        return set()
    ei, ej = "email_i", "email_j"
    if ei not in df.columns or ej not in df.columns:
        raise ValueError(f"{path} missing email_i/email_j")
    out: set[tuple[str, str]] = set()
    for a, b in zip(df[ei].astype(str), df[ej].astype(str), strict=False):
        if str(a) == str(b):
            continue
        out.add(_pair_key(a, b))
    return out


def _graph_topology_stats(*, node_ids: list[str], pairs: set[tuple[str, str]]) -> dict[str, Any]:
    g = nx.Graph()
    g.add_nodes_from(node_ids)
    for a, b in pairs:
        if a in g and b in g:
            g.add_edge(a, b)
    deg = dict(g.degree())
    isolated = sum(1 for n in node_ids if deg.get(n, 0) == 0)
    comps = list(nx.connected_components(g))
    singleton_comps = sum(1 for c in comps if len(c) == 1)
    endpoints: set[str] = set()
    for a, b in pairs:
        endpoints.add(a)
        endpoints.add(b)
    return {
        "n_graph_nodes": int(len(node_ids)),
        "n_union_edges": int(len(pairs)),
        "n_distinct_edge_endpoints": int(len(endpoints)),
        "n_isolated_nodes_degree_0": int(isolated),
        "n_connected_components": int(nx.number_connected_components(g)),
        "n_singleton_components": int(singleton_comps),
    }


def materialize_expanded_anchor_graph(
    *,
    graph_id: str,
    graph_bundle_root: Path,
    out_csv: Path | None = None,
    out_summary: Path | None = None,
) -> dict[str, Any]:
    bundle = graph_bundle_root / graph_id
    anchor_dir = bundle / "anchor" / graph_id
    sc_dir = bundle / "seed_candidate" / graph_id

    p_nodes = anchor_dir / "anchor_graph_nodes.csv"
    p_anchor = anchor_dir / "anchor_graph_edges_unscored.csv"
    p_sc = sc_dir / "seed_candidate_pairgraph_unscored.csv"

    for p in (p_nodes, p_anchor, p_sc):
        if not p.is_file():
            raise FileNotFoundError(f"Required bundle artifact not found: {p}")

    nodes_df = pd.read_csv(p_nodes, low_memory=False)
    node_ids = sorted({str(x) for x in nodes_df["external_id"].astype(str).tolist()})

    anchor_pairs = _load_pairs(p_anchor)
    sc_pairs = _load_pairs(p_sc)
    overlap = anchor_pairs & sc_pairs
    anchor_only = anchor_pairs - sc_pairs
    sc_only = sc_pairs - anchor_pairs
    union_pairs = anchor_pairs | sc_pairs

    if sc_pairs - union_pairs:
        raise AssertionError("seed_candidate pairs not contained in union (unexpected)")

    rows: list[dict[str, Any]] = []
    for a, b in sorted(union_pairs):
        in_anchor = (a, b) in anchor_pairs
        in_sc = (a, b) in sc_pairs
        rows.append(
            {
                "email_i": a,
                "email_j": b,
                "email_a": a,
                "email_b": b,
                "edge_weight": 1.0,
                "in_original_anchor": bool(in_anchor),
                "in_seed_candidate": bool(in_sc),
                "from_both": bool(in_anchor and in_sc),
                "graph_kind": GRAPH_KIND,
                "graph_id": graph_id,
            }
        )
    edges_df = pd.DataFrame(rows)

    out_csv = out_csv or (anchor_dir / "anchor_graph_edges_expanded_unscored.csv")
    out_summary = out_summary or (anchor_dir / "anchor_graph_edges_expanded_summary.json")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    edges_df.to_csv(out_csv, index=False)

    topo = _graph_topology_stats(node_ids=node_ids, pairs=union_pairs)
    summary: dict[str, Any] = {
        "graph_id": graph_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "definition": (
            "Union of anchor_graph_edges_unscored.csv and seed_candidate_pairgraph_unscored.csv; "
            "canonical unordered pairs; edge_weight=1.0 for unlearned thesis baseline."
        ),
        "inputs": {
            "anchor_graph_nodes_csv": str(p_nodes.resolve()),
            "anchor_graph_edges_unscored_csv": str(p_anchor.resolve()),
            "seed_candidate_pairgraph_unscored_csv": str(p_sc.resolve()),
        },
        "outputs": {
            "anchor_graph_edges_expanded_unscored_csv": str(out_csv.resolve()),
            "anchor_graph_edges_expanded_summary_json": str(out_summary.resolve()),
        },
        "edge_counts": {
            "original_anchor_edges": int(len(anchor_pairs)),
            "seed_candidate_edges": int(len(sc_pairs)),
            "overlap_edges": int(len(overlap)),
            "anchor_only_edges": int(len(anchor_only)),
            "seed_candidate_only_edges": int(len(sc_only)),
            "union_edges": int(len(union_pairs)),
        },
        "subset_checks": {
            "seed_candidate_subset_of_union": bool(sc_pairs <= union_pairs),
            "seed_candidate_subset_of_anchor_only": bool(sc_pairs <= anchor_pairs),
            "n_seed_candidate_missing_from_original_anchor": int(len(sc_only)),
        },
        "topology": topo,
    }
    out_summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    summary["outputs"]["anchor_graph_edges_expanded_unscored_csv"] = str(out_csv)
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--graph-id", type=str, default=DEFAULT_GRAPH_ID)
    p.add_argument(
        "--graph-bundle-root",
        type=Path,
        default=Path("seed_candidate_workflow/output/graph_bundles"),
    )
    p.add_argument("--out-csv", type=Path, default=None)
    p.add_argument("--out-summary", type=Path, default=None)
    args = p.parse_args()

    root = gh.find_project_root()
    bundle_root = args.graph_bundle_root
    if not bundle_root.is_absolute():
        bundle_root = (root / bundle_root).resolve()

    out_csv = args.out_csv
    out_summary = args.out_summary
    if out_csv is not None and not out_csv.is_absolute():
        out_csv = (root / out_csv).resolve()
    if out_summary is not None and not out_summary.is_absolute():
        out_summary = (root / out_summary).resolve()

    summary = materialize_expanded_anchor_graph(
        graph_id=str(args.graph_id).strip(),
        graph_bundle_root=bundle_root,
        out_csv=out_csv,
        out_summary=out_summary,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
