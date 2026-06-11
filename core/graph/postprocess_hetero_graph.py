from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import torch
from torch_geometric.data import HeteroData


_NODE_TYPE_ALIASES = {
    "attachments": "attachment",
}


def _canonical_node_types(node_types: Iterable[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for n in node_types:
        k = _NODE_TYPE_ALIASES.get(str(n), str(n))
        if k not in seen:
            seen.add(k)
            out.append(k)
    return out


def _load_metadata(meta_path: Path) -> Dict[str, Any]:
    if not meta_path.exists():
        return {}
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _subset_list(values: Any, keep_idx: torch.Tensor) -> Any:
    if not isinstance(values, list):
        return values
    idx = keep_idx.tolist()
    if len(values) < max(idx, default=-1) + 1:
        return values
    return [values[i] for i in idx]


def _compute_keep_masks(
    data: HeteroData,
    keep_node_types: List[str],
    *,
    primary_ntype: str = "email",
    remove_singleton_non_email: bool = True,
) -> Dict[str, torch.Tensor]:
    keep_masks: Dict[str, torch.Tensor] = {}
    for ntype in keep_node_types:
        n = int(data[ntype].num_nodes)
        keep_masks[ntype] = torch.ones(n, dtype=torch.bool)

    edge_types = [
        et for et in data.edge_types if et[0] in keep_masks and et[2] in keep_masks
    ]
    if not remove_singleton_non_email:
        return keep_masks

    changed = True
    while changed:
        changed = False
        for ntype in keep_node_types:
            if ntype == primary_ntype:
                continue
            alive = keep_masks[ntype]
            if int(alive.sum().item()) == 0:
                continue
            degree = torch.zeros(int(alive.numel()), dtype=torch.long)
            for et in edge_types:
                ei = data[et].edge_index
                if ei.numel() == 0:
                    continue
                src_t, _rel, dst_t = et
                src = ei[0].long()
                dst = ei[1].long()
                src_ok = keep_masks[src_t][src]
                dst_ok = keep_masks[dst_t][dst]
                edge_ok = src_ok & dst_ok
                if not bool(edge_ok.any()):
                    continue
                if src_t == ntype:
                    degree.scatter_add_(
                        0, src[edge_ok], torch.ones(int(edge_ok.sum().item()), dtype=torch.long)
                    )
                if dst_t == ntype:
                    degree.scatter_add_(
                        0, dst[edge_ok], torch.ones(int(edge_ok.sum().item()), dtype=torch.long)
                    )
            drop = alive & (degree <= 1)
            if bool(drop.any()):
                keep_masks[ntype] = alive & ~drop
                changed = True
    return keep_masks


def prune_hetero_graph(
    data: HeteroData,
    metadata: Dict[str, Any],
    *,
    keep_node_types: Iterable[str],
    primary_ntype: str = "email",
    remove_singleton_non_email: bool = True,
) -> Tuple[HeteroData, Dict[str, Any], Dict[str, Any]]:
    keep_types = _canonical_node_types(keep_node_types)
    present_keep_types = [n for n in keep_types if n in data.node_types]
    if primary_ntype not in present_keep_types:
        raise ValueError(f"primary_ntype {primary_ntype!r} must be in keep_node_types.")

    keep_masks = _compute_keep_masks(
        data,
        present_keep_types,
        primary_ntype=primary_ntype,
        remove_singleton_non_email=remove_singleton_non_email,
    )
    old_to_new: Dict[str, torch.Tensor] = {}
    keep_idx: Dict[str, torch.Tensor] = {}
    for ntype in present_keep_types:
        idx = torch.where(keep_masks[ntype])[0].long()
        keep_idx[ntype] = idx
        n_old = int(data[ntype].num_nodes)
        remap = torch.full((n_old,), -1, dtype=torch.long)
        remap[idx] = torch.arange(int(idx.numel()), dtype=torch.long)
        old_to_new[ntype] = remap

    out = HeteroData()
    for ntype in present_keep_types:
        old_store = data[ntype]
        out_store = out[ntype]
        idx = keep_idx[ntype]
        n_old = int(old_store.num_nodes)
        out_store.num_nodes = int(idx.numel())
        for key, value in old_store.items():
            if key == "num_nodes":
                continue
            if isinstance(value, torch.Tensor) and value.dim() > 0 and int(value.size(0)) == n_old:
                out_store[key] = value[idx].clone()
            else:
                out_store[key] = value

    kept_edge_types = [
        et for et in data.edge_types if et[0] in present_keep_types and et[2] in present_keep_types
    ]
    for et in kept_edge_types:
        src_t, rel, dst_t = et
        ei = data[et].edge_index
        if ei.numel() == 0:
            out[et].edge_index = ei.clone()
            continue
        src = ei[0].long()
        dst = ei[1].long()
        edge_ok = keep_masks[src_t][src] & keep_masks[dst_t][dst]
        src_new = old_to_new[src_t][src[edge_ok]]
        dst_new = old_to_new[dst_t][dst[edge_ok]]
        valid = (src_new >= 0) & (dst_new >= 0)
        out[et].edge_index = torch.stack([src_new[valid], dst_new[valid]], dim=0)

    new_meta: Dict[str, Any] = {}
    old_node_maps = metadata.get("node_maps", {}) if isinstance(metadata, dict) else {}
    new_node_maps: Dict[str, Dict[str, Any]] = {}
    for ntype in present_keep_types:
        key = ntype
        old_map = old_node_maps.get(key, {})
        idx = keep_idx[ntype]
        if "index_to_meta" in old_map:
            new_node_maps[key] = {"index_to_meta": _subset_list(old_map["index_to_meta"], idx)}
        elif "index_to_string" in old_map:
            new_node_maps[key] = {
                "index_to_string": _subset_list(old_map["index_to_string"], idx)
            }
        else:
            new_node_maps[key] = {}
    new_meta["node_maps"] = new_node_maps
    new_meta["feature_shapes"] = {
        ntype: (list(out[ntype].x.shape) if "x" in out[ntype] else [int(out[ntype].num_nodes), 0])
        for ntype in present_keep_types
    }
    edge_counts: Dict[str, int] = {}
    for et in out.edge_types:
        src_t, rel, dst_t = et
        edge_counts[f"{src_t}->{dst_t}:{rel}"] = int(out[et].edge_index.size(1))
    new_meta["edge_counts"] = edge_counts

    old_email_attrs = metadata.get("email_attrs", {}) if isinstance(metadata, dict) else {}
    if isinstance(old_email_attrs, dict) and primary_ntype in keep_idx:
        eidx = keep_idx[primary_ntype]
        new_email_attrs: Dict[str, Any] = {}
        for k, v in old_email_attrs.items():
            new_email_attrs[k] = _subset_list(v, eidx)
        new_meta["email_attrs"] = new_email_attrs

    stats = {
        "kept_node_types": present_keep_types,
        "node_counts_before": {n: int(data[n].num_nodes) for n in data.node_types},
        "node_counts_after": {n: int(out[n].num_nodes) for n in out.node_types},
    }
    return out, new_meta, stats


def prune_graph_file(
    input_graph_path: str | Path,
    *,
    output_graph_path: str | Path,
    keep_node_types: Iterable[str],
    primary_ntype: str = "email",
    remove_singleton_non_email: bool = True,
) -> Dict[str, Any]:
    in_path = Path(input_graph_path).expanduser().resolve()
    out_path = Path(output_graph_path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    data = torch.load(str(in_path), map_location="cpu", weights_only=False)
    if not isinstance(data, HeteroData):
        raise TypeError(f"Expected HeteroData at {in_path}, got {type(data)}")
    in_meta_path = in_path.with_suffix(".meta.json")
    out_meta_path = out_path.with_suffix(".meta.json")
    old_meta = _load_metadata(in_meta_path)

    new_graph, new_meta, stats = prune_hetero_graph(
        data,
        old_meta,
        keep_node_types=keep_node_types,
        primary_ntype=primary_ntype,
        remove_singleton_non_email=remove_singleton_non_email,
    )
    torch.save(new_graph, str(out_path))
    with open(out_meta_path, "w", encoding="utf-8") as f:
        json.dump(new_meta, f, indent=2)

    return {
        "input_graph_path": str(in_path),
        "output_graph_path": str(out_path),
        "input_meta_path": str(in_meta_path),
        "output_meta_path": str(out_meta_path),
        **stats,
    }

