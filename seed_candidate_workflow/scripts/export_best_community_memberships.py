"""Re-run best community partition and export external_id -> pred_community assignments."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
if str(_REPO / "core") not in sys.path:
    sys.path.insert(0, str(_REPO / "core"))
if str(_REPO / "core" / "GNN") not in sys.path:
    sys.path.insert(0, str(_REPO / "core" / "GNN"))

from seed_candidate_workflow.utils.anchor_graph_community_helpers import (  # noqa: E402
    map_email_predictions,
    run_weighted_email_community_detection,
)
from seed_candidate_workflow.utils.graph_scorer_registry import apply_scorer  # noqa: E402
from seed_candidate_workflow.utils import semantic_supernode_gt_metrics as member_expansion  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--run-id",
        default="final_14_only_mlp__timestamp_feature__early_stopping",
    )
    p.add_argument("--graph-id", default="main_gnn_pu_1_no_ts_dedup_task_identity_13")
    p.add_argument("--gt-slug", default="ground_truth")
    p.add_argument("--manifest", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument(
        "--dedup-collapse-out-dir",
        default="data/misp/misp_lake_dedup_task_identity",
        help="Directory with external_id_map.csv / collapsed_clusters.json for expanding graph-node ids.",
    )
    args = p.parse_args()

    run_id = str(args.run_id)
    graph_id = str(args.graph_id)
    gt_slug = str(args.gt_slug)
    best_json = _REPO / "output/runs" / run_id / "community" / f"anchor_community_best__{gt_slug}.json"
    if not best_json.is_file():
        raise FileNotFoundError(f"Missing best community JSON: {best_json}")

    out_dir = args.out_dir or (_REPO / "output/runs" / run_id / "community" / "exports")
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = args.manifest or (
        _REPO / "seed_candidate_workflow/configs/final_14_only_mlp/final_14_only_mlp.manifest.json"
    )
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8-sig"))

    best_payload = json.loads(best_json.read_text(encoding="utf-8"))
    br = dict(best_payload["best_row"])
    method = str(br["method"])
    resolution = float(br["resolution"])
    min_edge_weight = float(br["min_edge_weight"])
    print(
        f"Best: {method} resolution={resolution} min_edge_weight={min_edge_weight} "
        f"v_measure={float(br.get('v_measure', float('nan'))):.4f}"
    )

    bundle = _REPO / "seed_candidate_workflow/output/graph_bundles" / graph_id
    anchor_dir = bundle / "anchor" / graph_id
    sc_dir = bundle / "seed_candidate" / graph_id
    nodes_df = pd.read_csv(anchor_dir / "anchor_graph_nodes.csv", low_memory=False)
    edges_df = pd.read_csv(sc_dir / "seed_candidate_pairgraph_unscored.csv", low_memory=False)
    node_ids = nodes_df["external_id"].astype(str).tolist()
    print(f"Nodes: {len(node_ids):,}  Unscored edges: {len(edges_df):,}")

    score_params = {
        "pu_run": {
            "run_dir": f"output/runs/{run_id}",
            "graph_pt": str(manifest["graph_pt"]),
            "checkpoint": "best_model.pt",
            "pair_dataset_csv": str(manifest["final_pair_dataset_csv"]),
            "device": "cpu",
            "no_to_undirected": False,
        },
        "seed_edge_weight": 1.0,
        "weight_mode": "raw_score",
        "export_non_seed_min_pu_score": 0.0,
    }
    print("Scoring edges (PU model)...")
    sr = apply_scorer(
        score_mode="seed_candidate_pu_v1",
        graph_kind="seed_candidate",
        score_params=score_params,
        payload={"candidate_union_df": edges_df},
        diagnostics_cfg={},
    )
    scored = sr.scored_all.copy()
    scored["email_i"] = scored["email_i"].astype(str)
    scored["email_j"] = scored["email_j"].astype(str)
    scored["email_a"] = scored["email_i"]
    scored["email_b"] = scored["email_j"]

    email_to_comm, info = run_weighted_email_community_detection(
        node_ids=node_ids,
        edges_df=scored,
        method=method,
        resolution=resolution,
        min_edge_weight=min_edge_weight,
        weight_col="edge_weight",
        seed=0,
        use_edge_weights_in_partitioning=True,
        apply_threshold_filter=True,
    )
    graph_assign_df = map_email_predictions(node_ids, email_to_comm)
    graph_assign_csv = out_dir / "best_solution_graph_node_communities.csv"
    graph_assign_df.to_csv(graph_assign_csv, index=False)

    dedup_dir = Path(args.dedup_collapse_out_dir).expanduser()
    if not dedup_dir.is_absolute():
        dedup_dir = (_REPO / dedup_dir).resolve()
    else:
        dedup_dir = dedup_dir.resolve()
    gid_to_members = member_expansion.load_dedup_collapse_member_table_from_out_dir(dedup_dir)
    expanded_pred = member_expansion.expand_pred_map_for_gt_eval(email_to_comm, gid_to_members)
    graph_to_members = {
        str(gid): gid_to_members.get(str(gid), [str(gid)])
        for gid in node_ids
    }

    expanded_rows: list[dict[str, object]] = []
    for graph_external_id, members in graph_to_members.items():
        cid = int(email_to_comm[str(graph_external_id)])
        for external_id in members:
            expanded_rows.append(
                {
                    "external_id": str(external_id),
                    "pred_community": cid,
                    "graph_external_id": str(graph_external_id),
                }
            )
    expanded_assign_df = pd.DataFrame(expanded_rows).sort_values(
        ["pred_community", "external_id"],
        kind="stable",
    )
    assign_csv = out_dir / "best_solution_email_communities.csv"
    expanded_assign_df.to_csv(assign_csv, index=False)

    by_comm: dict[str, list[str]] = {}
    for eid, cid in expanded_pred.items():
        by_comm.setdefault(str(int(cid)), []).append(str(eid))
    for k in by_comm:
        by_comm[k] = sorted(set(by_comm[k]))
    grouped_json = out_dir / "best_solution_communities_grouped.json"
    grouped_json.write_text(json.dumps(by_comm, indent=2), encoding="utf-8")

    txt_path = out_dir / "best_solution_communities_grouped.txt"
    lines = [
        f"# {run_id}",
        f"# method={method} resolution={resolution} min_edge_weight={min_edge_weight}",
        f"# n_communities={len(by_comm)} n_expanded_emails={len(expanded_pred)}",
        f"# dedup_collapse_out_dir={dedup_dir}",
        "",
    ]
    for cid in sorted(by_comm, key=lambda x: int(x)):
        members = by_comm[cid]
        lines.append(f"## community {cid} (n={len(members)})")
        lines.extend(members)
        lines.append("")
    txt_path.write_text("\n".join(lines), encoding="utf-8")

    meta = {
        "run_id": run_id,
        "best_json": str(best_json),
        "best_row": br,
        "partition_info": info,
        "dedup_collapse_out_dir": str(dedup_dir),
        "n_graph_nodes": int(len(email_to_comm)),
        "n_expanded_member_emails": int(len(expanded_pred)),
        "n_communities": int(len(by_comm)),
        "outputs": {
            "expanded_flat_csv": str(assign_csv),
            "expanded_grouped_json": str(grouped_json),
            "expanded_grouped_txt": str(txt_path),
            "graph_node_flat_csv": str(graph_assign_csv),
        },
    }
    (out_dir / "export_manifest.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("Wrote:", graph_assign_csv)
    print("Wrote:", assign_csv)
    print("Wrote:", grouped_json)
    print("Wrote:", txt_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
