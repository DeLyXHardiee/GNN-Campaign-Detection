from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


def resolve_bundle_dir(*, graph_bundle_root: Path, graph_id: str) -> Path:
    p = (graph_bundle_root / graph_id).resolve()
    if not p.is_dir():
        raise FileNotFoundError(f"Graph bundle not found for graph_id={graph_id}: {p}")
    return p


def resolve_target_edges_csv(*, bundle_dir: Path, graph_id: str, target: str) -> Path:
    target_l = str(target).strip().lower()
    if target_l == "seed_candidate":
        p = bundle_dir / "seed_candidate" / graph_id / "seed_candidate_pairgraph_unscored.csv"
    elif target_l == "candidate":
        cand_root = bundle_dir / "candidate" / graph_id
        dirs = [d for d in cand_root.iterdir() if d.is_dir() and d.name.startswith("candidate_generation_")]
        if not dirs:
            raise FileNotFoundError(f"No candidate stage dirs found under {cand_root}")
        p = max(dirs, key=lambda d: d.stat().st_mtime) / "candidate_union.csv"
    elif target_l == "seed":
        seed_root = bundle_dir / "seed" / graph_id
        dirs = [d for d in seed_root.iterdir() if d.is_dir() and d.name.startswith("seed_generation_")]
        if not dirs:
            raise FileNotFoundError(f"No seed stage dirs found under {seed_root}")
        p = max(dirs, key=lambda d: d.stat().st_mtime) / "seed_edges_all.csv"
    elif target_l == "anchor":
        p = bundle_dir / "anchor" / graph_id / "anchor_graph_edges_unscored.csv"
    elif target_l == "semantic_shard":
        p = bundle_dir / "semantic_shard" / graph_id / "semantic_shard_pairgraph_unscored.csv"
    else:
        raise ValueError(f"Unsupported score target: {target!r}")
    if not p.is_file():
        raise FileNotFoundError(f"Target edges file not found for target={target!r}: {p}")
    return p


def dry_run_planned_target_edges_csv(*, bundle_dir: Path, graph_id: str, target: str) -> Path:
    target_l = str(target).strip().lower()
    if target_l == "seed_candidate":
        return bundle_dir / "seed_candidate" / graph_id / "seed_candidate_pairgraph_unscored.csv"
    if target_l == "candidate":
        return bundle_dir / "candidate" / graph_id / "candidate_generation_dryrun" / "candidate_union.csv"
    if target_l == "seed":
        return bundle_dir / "seed" / graph_id / "seed_generation_dryrun" / "seed_edges_all.csv"
    if target_l == "anchor":
        return bundle_dir / "anchor" / graph_id / "anchor_graph_edges_unscored.csv"
    if target_l == "semantic_shard":
        return bundle_dir / "semantic_shard" / graph_id / "semantic_shard_pairgraph_unscored.csv"
    raise ValueError(f"Unsupported score target: {target!r}")


TARGET_EDGE_RESOLVERS = {
    "actual": resolve_target_edges_csv,
    "dry_run": dry_run_planned_target_edges_csv,
}


@dataclass(frozen=True)
class TargetSpec:
    name: str
    executor_kind: str


TARGET_REGISTRY: dict[str, TargetSpec] = {
    "anchor": TargetSpec(name="anchor", executor_kind="anchor_like"),
    "seed": TargetSpec(name="seed", executor_kind="anchor_like"),
    "candidate": TargetSpec(name="candidate", executor_kind="anchor_like"),
    "seed_candidate": TargetSpec(name="seed_candidate", executor_kind="anchor_like"),
    "semantic_shard": TargetSpec(name="semantic_shard", executor_kind="semantic_shard"),
}


def resolve_target_spec(target: str) -> TargetSpec:
    t = str(target).strip().lower()
    if t not in TARGET_REGISTRY:
        raise ValueError(f"Unsupported target {target!r}")
    return TARGET_REGISTRY[t]
