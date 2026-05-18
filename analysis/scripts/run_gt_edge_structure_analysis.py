#!/usr/bin/env python3
"""
GT edge-structure analysis: same-campaign vs cross-campaign pairwise evidence.

Example (repo root):

  python analysis/scripts/run_gt_edge_structure_analysis.py --config analysis/configs/gt_edge_structure.default.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from seed_candidate_workflow.utils import graph_structure_helpers as gh
from seed_candidate_workflow.utils.gt_edge_structure_analysis import (
    DEFAULT_COSINE_BUCKETS,
    GtEdgeStructureRunConfig,
    resolve_gt_paths,
    run_gt_edge_structure_analysis,
)


def _load_json_config(path: Path) -> dict:
    return json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))


def _resolve_path(project_root: Path, raw: str | None) -> Path | None:
    if not raw or not str(raw).strip():
        return None
    p = Path(str(raw).strip())
    return p if p.is_absolute() else (project_root / p).resolve()


def _paths_from_pipeline(project_root: Path) -> tuple[Path, Path]:
    paths = gh.resolve_graph_analysis_paths(project_root)
    return paths.graph_pt, paths.meta_json


def _build_config_from_args(args: argparse.Namespace, project_root: Path) -> GtEdgeStructureRunConfig:
    cfg_json: dict = {}
    if args.config is not None:
        cfg_json = _load_json_config(args.config)

    def _get(key: str, default: Any = None):
        return cfg_json.get(key, default)

    graph_pt = _resolve_path(project_root, args.graph_pt or _get("graph_pt"))
    meta_json = _resolve_path(project_root, args.meta_json or _get("meta_json"))
    if graph_pt is None or meta_json is None:
        graph_pt, meta_json = _paths_from_pipeline(project_root)

    gt_paths = resolve_gt_paths(
        gt_json=[Path(p) for p in (args.gt_json or [])] or None,
        gt_dir=Path(args.gt_dir) if args.gt_dir else None,
        gt_set=args.gt_set or _get("gt_set"),
        project_root=project_root,
    )
    if not gt_paths and _get("gt_paths"):
        gt_paths = [
            _resolve_path(project_root, p)
            for p in _get("gt_paths")
            if _resolve_path(project_root, p) is not None
        ]

    buckets_raw = _get("cosine_buckets")
    buckets = DEFAULT_COSINE_BUCKETS
    if isinstance(buckets_raw, list) and buckets_raw:
        buckets = tuple(
            (str(b[0]), b[1] if b[1] is None else float(b[1]), b[2] if b[2] is None else float(b[2]))
            for b in buckets_raw
        )

    out_dir = _resolve_path(
        project_root,
        args.out_dir or _get("out_dir") or "output/analysis/gt_edge_structure",
    )

    return GtEdgeStructureRunConfig(
        gt_paths=gt_paths,
        graph_pt=graph_pt,
        meta_json=meta_json,
        embeddings_json=_resolve_path(
            project_root, args.embeddings_json or _get("embeddings_json")
        ),
        pair_training_csv=_resolve_path(
            project_root, args.pair_training_csv or _get("pair_training_csv")
        ),
        candidate_union_csv=_resolve_path(
            project_root, args.candidate_union_csv or _get("candidate_union_csv")
        ),
        anchor_run_dir=_resolve_path(
            project_root, args.anchor_run_dir or _get("anchor_run_dir")
        ),
        anchor_graph_config=_resolve_path(
            project_root,
            args.anchor_graph_config
            or _get("anchor_graph_config")
            or "seed_candidate_workflow/configs/anchor_graph.default.json",
        ),
        anchor_seed_config=_resolve_path(
            project_root,
            args.anchor_seed_config
            or _get("anchor_seed_config")
            or "seed_candidate_workflow/configs/anchor_seed.default.json",
        ),
        anchor_candidate_config=_resolve_path(
            project_root,
            args.anchor_candidate_config
            or _get("anchor_candidate_config")
            or "seed_candidate_workflow/configs/anchor_candidate_generation.default.json",
        ),
        out_dir=out_dir or project_root / "output/analysis/gt_edge_structure",
        max_same_pairs=int(args.max_same_pairs or _get("max_same_pairs", 8000)),
        max_cross_pairs=int(args.max_cross_pairs or _get("max_cross_pairs", 8000)),
        seed=int(args.seed if args.seed is not None else _get("seed", 0)),
        min_support=int(args.min_support or _get("min_support", 30)),
        frontier_max_abs_diff=float(
            args.frontier_max_abs_diff or _get("frontier_max_abs_diff", 0.15)
        ),
        cosine_buckets=buckets,
        top_joint_combinations=int(
            args.top_joint_combinations or _get("top_joint_combinations", 25)
        ),
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=None, help="JSON config with defaults")
    ap.add_argument("--gt-json", type=Path, action="append", default=[])
    ap.add_argument("--gt-dir", type=Path, default=None)
    ap.add_argument("--gt-set", type=str, default=None, help="Key in gt_sets.json")
    ap.add_argument("--graph-pt", type=Path, default=None)
    ap.add_argument("--meta-json", type=Path, default=None)
    ap.add_argument("--embeddings-json", type=Path, default=None)
    ap.add_argument("--pair-training-csv", type=Path, default=None)
    ap.add_argument(
        "--candidate-union-csv",
        type=Path,
        default=None,
        help="candidate_union.csv for novelty vs current union (auto from bundle if omitted)",
    )
    ap.add_argument("--anchor-run-dir", type=Path, default=None)
    ap.add_argument("--anchor-graph-config", type=Path, default=None)
    ap.add_argument("--anchor-seed-config", type=Path, default=None)
    ap.add_argument("--anchor-candidate-config", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--max-same-pairs", type=int, default=None)
    ap.add_argument("--max-cross-pairs", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--min-support", type=int, default=None)
    ap.add_argument("--frontier-max-abs-diff", type=float, default=None)
    ap.add_argument("--top-joint-combinations", type=int, default=None)
    args = ap.parse_args()

    project_root = gh.find_project_root()
    cfg = _build_config_from_args(args, project_root)

    anchor_ok = (
        cfg.anchor_run_dir is not None
        and cfg.anchor_run_dir.is_dir()
        and (cfg.anchor_run_dir / "nodes.csv").is_file()
    )
    if not anchor_ok:
        try:
            import torch_geometric  # noqa: F401
        except ImportError:
            print(
                "ERROR: torch-geometric is not installed and no anchor nodes.csv was found.\n"
                "  Fix A: pip install -r requirements.txt  (includes torch-geometric)\n"
                "  Fix B: run anchor graph first, then set --anchor-run-dir to that run "
                "(must contain nodes.csv)",
                file=sys.stderr,
            )
            return 1

    result = run_gt_edge_structure_analysis(cfg)
    print(json.dumps(result.get("output_paths", {}), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
