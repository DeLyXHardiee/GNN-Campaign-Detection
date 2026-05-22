"""
Strip selected node types (and incident edges) from a HeteroData .pt + .meta.json pair.

Used for ablations that remove hetero node types without rebuilding from MISP.
To restore full graph, point runs back at the source .pt (see manifest sidecar).

Example (_16 ablation: drop domain + html_structure_fingerprint from _12 graph):

  python core/graph/scripts/strip_hetero_node_types.py \\
    --input-pt core/graph/output/main_gnn_pu_1_no_ts_dedup_task_identity_12_hetero.pt \\
    --output-stem main_gnn_pu_1_no_ts_dedup_task_identity_16_no_domain_html_fp \\
    --strip domain html_structure_fingerprint
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch


def _load_hetero(pt_path: Path) -> Any:
    from torch_geometric.data import HeteroData

    data = torch.load(pt_path, map_location="cpu", weights_only=False)
    if not isinstance(data, HeteroData):
        raise TypeError(f"Expected HeteroData in {pt_path}, got {type(data)}")
    return data


def strip_hetero_node_types(
    data: Any,
    *,
    strip_types: tuple[str, ...],
) -> dict[str, Any]:
    """Remove node types and any edge keys touching them. Returns removal stats."""
    strip_set = set(strip_types)
    removed_edges: list[str] = []
    for et in list(data.edge_types):
        src, rel, dst = et
        if src in strip_set or dst in strip_set:
            del data[src, rel, dst]
            removed_edges.append(f"{src}->{dst}:{rel}")
    for nt in strip_types:
        if nt in data.node_types:
            del data[nt]
    return {
        "stripped_node_types": list(strip_types),
        "removed_edge_types": removed_edges,
        "remaining_node_types": list(data.node_types),
        "remaining_edge_types": [f"{s}->{d}:{r}" for s, r, d in data.edge_types],
    }


def _patch_meta(
    meta: dict[str, Any],
    *,
    strip_types: tuple[str, ...],
    stats: dict[str, Any],
    source_pt: Path,
    output_pt: Path,
) -> dict[str, Any]:
    out = dict(meta)
    strip_set = set(strip_types)
    node_maps = dict(out.get("node_maps") or {})
    for nt in strip_types:
        node_maps.pop(nt, None)
    out["node_maps"] = node_maps

    feat = dict(out.get("feature_shapes") or {})
    for nt in strip_types:
        feat.pop(nt, None)
    out["feature_shapes"] = feat

    ec = dict(out.get("edge_counts") or {})
    keys_drop = [k for k in ec if any(f"->{t}:" in k or k.startswith(f"{t}->") for t in strip_set)]
    for k in keys_drop:
        ec.pop(k, None)
    out["edge_counts"] = ec

    out["hetero_strip_manifest"] = {
        "source_graph_pt": str(source_pt.resolve()),
        "output_graph_pt": str(output_pt.resolve()),
        "stripped_node_types": list(strip_types),
        "removed_edge_types": stats.get("removed_edge_types"),
        "remaining_node_types": stats.get("remaining_node_types"),
        "remaining_edge_types": stats.get("remaining_edge_types"),
        "restore_hint": "Re-point pipeline graph_pt_path_override to the source_graph_pt to use domain/html nodes again.",
    }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Strip node types from a hetero graph checkpoint.")
    parser.add_argument("--input-pt", type=Path, required=True, help="Source *_hetero.pt")
    parser.add_argument(
        "--output-stem",
        type=str,
        required=True,
        help="Basename under core/graph/output (writes {stem}_hetero.pt and .meta.json)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("core/graph/output"),
        help="Directory for output artifacts",
    )
    parser.add_argument(
        "--strip",
        nargs="+",
        required=True,
        help="PyG node type labels to remove (e.g. domain html_structure_fingerprint)",
    )
    args = parser.parse_args()

    input_pt = args.input_pt.resolve()
    if not input_pt.is_file():
        raise FileNotFoundError(input_pt)

    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    output_pt = out_dir / f"{args.output_stem}_hetero.pt"
    output_meta = output_pt.with_suffix(".meta.json")
    input_meta = input_pt.with_suffix(".meta.json")

    data = _load_hetero(input_pt)
    stats = strip_hetero_node_types(data, strip_types=tuple(args.strip))

    torch.save(data, output_pt)

    meta: dict[str, Any] = {}
    if input_meta.is_file():
        meta = json.loads(input_meta.read_text(encoding="utf-8"))
    meta = _patch_meta(
        meta,
        strip_types=tuple(args.strip),
        stats=stats,
        source_pt=input_pt,
        output_pt=output_pt,
    )
    output_meta.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")

    manifest_path = out_dir / f"{args.output_stem}_hetero.strip_manifest.json"
    manifest_path.write_text(
        json.dumps(meta.get("hetero_strip_manifest") or {}, indent=2),
        encoding="utf-8",
    )

    print(f"Wrote {output_pt}")
    print(f"Wrote {output_meta}")
    print(f"Wrote {manifest_path}")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
