"""
Candidate-family scorecard: screen rule templates before adding them to the seed-candidate graph.

Example (repo root):

  python analysis/scripts/run_candidate_family_scorecard.py \\
    --config analysis/configs/candidate_family_scorecard.dedup_task_identity_7.json
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
from seed_candidate_workflow.utils.candidate_family_scorecard import (
    CandidateFamilyScorecardRunConfig,
    CandidateFamilySpec,
    RecommendationThresholds,
    _parse_families,
    resolve_gt_paths,
    run_candidate_family_scorecard,
)


def _load_json_config(path: Path) -> dict:
    return json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))


def _resolve_path(project_root: Path, raw: str | None) -> Path | None:
    if not raw or not str(raw).strip():
        return None
    p = Path(str(raw).strip())
    return p if p.is_absolute() else (project_root / p).resolve()


def _build_config(args: argparse.Namespace, project_root: Path) -> CandidateFamilyScorecardRunConfig:
    cfg_json: dict = {}
    if args.config is not None:
        cfg_json = _load_json_config(args.config)

    def _get(key: str, default: Any = None):
        return cfg_json.get(key, default)

    graph_pt = _resolve_path(project_root, args.graph_pt or _get("graph_pt"))
    meta_json = _resolve_path(project_root, args.meta_json or _get("meta_json"))
    if graph_pt is None or meta_json is None:
        paths = gh.resolve_graph_analysis_paths(project_root)
        graph_pt = graph_pt or paths.graph_pt
        meta_json = meta_json or paths.meta_json

    gt_paths = resolve_gt_paths(
        gt_json=[Path(p) for p in (args.gt_json or [])] or None,
        gt_dir=Path(args.gt_dir) if args.gt_dir else None,
        gt_set=args.gt_set or _get("gt_set"),
        project_root=project_root,
    )

    th_raw = _get("recommendation_thresholds") or {}
    thresholds = RecommendationThresholds(
        min_new_same_pairs=int(th_raw.get("min_new_same_pairs", 5)),
        min_oracle_v_gain=float(th_raw.get("min_oracle_v_gain", 0.005)),
        min_precision_like_new=float(th_raw.get("min_precision_like_new", 0.65)),
        max_cross_new_capture_rate=float(th_raw.get("max_cross_new_capture_rate", 0.05)),
        max_graph_only_fraction_of_oracle=float(th_raw.get("max_graph_only_fraction_of_oracle", 0.85)),
        min_learnability_score=float(th_raw.get("min_learnability_score", 0.15)),
        weak_gain_max_new_same=int(th_raw.get("weak_gain_max_new_same", 3)),
    )

    families_raw = _get("families")
    families: list[CandidateFamilySpec] = []
    if isinstance(families_raw, list) and families_raw:
        families = _parse_families(families_raw, project_root)

    out_dir = _resolve_path(
        project_root,
        args.out_dir or _get("out_dir") or "output/analysis/candidate_family_scorecard",
    )

    return CandidateFamilyScorecardRunConfig(
        gt_paths=gt_paths,
        graph_pt=graph_pt,
        meta_json=meta_json,
        embeddings_json=_resolve_path(project_root, args.embeddings_json or _get("embeddings_json")),
        pair_training_csv=_resolve_path(
            project_root, args.pair_training_csv or _get("pair_training_csv")
        ),
        candidate_union_csv=_resolve_path(
            project_root, args.candidate_union_csv or _get("candidate_union_csv")
        ),
        seed_candidate_edges_csv=_resolve_path(
            project_root, args.seed_candidate_edges_csv or _get("seed_candidate_edges_csv")
        ),
        anchor_run_dir=_resolve_path(project_root, args.anchor_run_dir or _get("anchor_run_dir")),
        out_dir=out_dir or project_root / "output/analysis/candidate_family_scorecard",
        families=families,
        max_same_pairs=int(args.max_same_pairs or _get("max_same_pairs", 8000)),
        max_cross_pairs=int(args.max_cross_pairs or _get("max_cross_pairs", 8000)),
        seed=int(args.seed if args.seed is not None else _get("seed", 0)),
        min_support=int(args.min_support or _get("min_support", 10)),
        community_method=str(args.community_method or _get("community_method", "louvain")),
        community_resolution=float(
            args.community_resolution or _get("community_resolution", 1.0)
        ),
        community_seed=int(args.community_seed or _get("community_seed", 0)),
        thresholds=thresholds,
        family_catalog=_get("family_catalog"),
        misp_json=_resolve_path(project_root, _get("misp_json")),
        admitting_evidence_dir=_resolve_path(project_root, _get("admitting_evidence_dir")),
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=None, help="JSON config with defaults")
    ap.add_argument("--gt-json", type=Path, action="append", default=[])
    ap.add_argument("--gt-dir", type=Path, default=None)
    ap.add_argument("--gt-set", type=str, default=None)
    ap.add_argument("--graph-pt", type=Path, default=None)
    ap.add_argument("--meta-json", type=Path, default=None)
    ap.add_argument("--embeddings-json", type=Path, default=None)
    ap.add_argument("--pair-training-csv", type=Path, default=None)
    ap.add_argument("--candidate-union-csv", type=Path, default=None)
    ap.add_argument("--seed-candidate-edges-csv", type=Path, default=None)
    ap.add_argument("--anchor-run-dir", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--max-same-pairs", type=int, default=None)
    ap.add_argument("--max-cross-pairs", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--min-support", type=int, default=None)
    ap.add_argument("--community-method", type=str, default=None)
    ap.add_argument("--community-resolution", type=float, default=None)
    ap.add_argument("--community-seed", type=int, default=None)
    args = ap.parse_args()

    project_root = gh.find_project_root()
    cfg = _build_config(args, project_root)

    anchor_ok = (
        cfg.anchor_run_dir is not None
        and cfg.anchor_run_dir.is_dir()
        and (cfg.anchor_run_dir / "anchor_graph_nodes.csv").is_file()
    )
    if not anchor_ok:
        try:
            import torch_geometric  # noqa: F401
        except ImportError:
            print(
                "ERROR: torch-geometric is not installed and no anchor_graph_nodes.csv was found.\n"
                "  Fix A: pip install -r requirements.txt\n"
                "  Fix B: pass --anchor-run-dir to a completed anchor graph run",
                file=sys.stderr,
            )
            return 1

    result = run_candidate_family_scorecard(cfg)
    print(json.dumps(result.get("output_paths", {}), indent=2))
    print(json.dumps(result.get("global_shortlist", {}), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
