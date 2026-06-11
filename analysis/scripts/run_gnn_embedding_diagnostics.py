"""
GNN / encoder email embedding diagnostic analysis.

Example (repo root):

  python analysis/scripts/run_gnn_embedding_diagnostics.py ^
    --run-dir output/runs/main_gnn_pu_1_no_ts_dedup_task_identity_13 ^
    --graph-pt core/graph/output/main_gnn_pu_1_no_ts_dedup_task_identity_12_hetero.pt ^
    --gt-path data/groundtruth/ground_truth.dedup_task_identity.json
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
if str(_REPO / "core") not in sys.path:
    sys.path.insert(0, str(_REPO / "core"))
if str(_REPO / "core" / "GNN") not in sys.path:
    sys.path.insert(0, str(_REPO / "core" / "GNN"))

from analysis.utils.gnn_embedding_diagnostics import (  # noqa: E402
    GnnEmbeddingDiagConfig,
    run_gnn_embedding_diagnostics,
)


def _resolve_path(project_root: Path, raw: str | None) -> Path | None:
    if not raw or not str(raw).strip():
        return None
    p = Path(str(raw).strip())
    return p if p.is_absolute() else (project_root / p).resolve()


def main() -> int:
    p = argparse.ArgumentParser(description="GNN embedding diagnostic analysis for a PU run.")
    p.add_argument("--config", type=Path, default=None, help="Optional JSON config.")
    p.add_argument("--run-dir", type=Path, default=None)
    p.add_argument("--graph-pt", type=Path, default=None)
    p.add_argument("--gt-path", type=Path, action="append", default=[])
    p.add_argument("--pair-csv", type=Path, default=None)
    p.add_argument("--candidate-union-csv", type=Path, default=None)
    p.add_argument("--bridge-scores-csv", type=Path, default=None)
    p.add_argument("--output-subdir", type=str, default=None)
    p.add_argument("--checkpoint-name", type=str, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--max-pairs-per-relation", type=int, default=None)
    p.add_argument("--skip-plots", action="store_true")
    p.add_argument("--skip-html", action="store_true")
    p.add_argument("--skip-probe", action="store_true")
    args = p.parse_args()

    cfg_json: dict[str, Any] = {}
    if args.config is not None:
        cfg_json = json.loads(Path(args.config).read_text(encoding="utf-8"))

    def _get(key: str, default: Any = None):
        return cfg_json.get(key, default)

    project_root = _REPO
    run_dir = _resolve_path(project_root, args.run_dir or _get("run_dir"))
    graph_pt = _resolve_path(project_root, args.graph_pt or _get("graph_pt"))
    if run_dir is None or graph_pt is None:
        raise SystemExit("run-dir and graph-pt are required (CLI or config).")

    gt_paths: list[Path] = []
    for raw in list(args.gt_path or []) + list(_get("gt_paths") or []):
        rp = _resolve_path(project_root, raw)
        if rp is not None:
            gt_paths.append(rp)
    if not gt_paths:
        default_gt = project_root / "data" / "groundtruth" / "ground_truth.dedup_task_identity.json"
        if default_gt.is_file():
            gt_paths.append(default_gt)

    bridge_default = run_dir / "bridge_candidate_experiment" / "bridge_candidate_scores.csv"
    cfg = GnnEmbeddingDiagConfig(
        run_dir=run_dir,
        graph_pt=graph_pt,
        gt_paths=gt_paths,
        output_subdir=str(args.output_subdir or _get("output_subdir") or "gnn_embedding_diagnostics"),
        pair_csv=_resolve_path(project_root, args.pair_csv or _get("pair_csv")),
        candidate_union_csv=_resolve_path(
            project_root, args.candidate_union_csv or _get("candidate_union_csv")
        ),
        bridge_scores_csv=_resolve_path(
            project_root,
            args.bridge_scores_csv
            or _get("bridge_scores_csv")
            or (str(bridge_default) if bridge_default.is_file() else None),
        ),
        checkpoint_name=str(args.checkpoint_name or _get("checkpoint_name") or "best_model.pt"),
        device=str(args.device or _get("device") or "cpu"),
        to_undirected=bool(_get("to_undirected", True)),
        embeddings_json=_resolve_path(project_root, _get("embeddings_json")),
        max_pairs_per_relation=int(
            args.max_pairs_per_relation
            if args.max_pairs_per_relation is not None
            else _get("max_pairs_per_relation", 80_000)
        ),
        max_emails_for_retrieval=int(_get("max_emails_for_retrieval", 0)),
        skip_plots=bool(args.skip_plots or _get("skip_plots", False)),
        skip_html=bool(args.skip_html or _get("skip_html", False)),
        skip_probe=bool(args.skip_probe or _get("skip_probe", False)),
    )

    summary = run_gnn_embedding_diagnostics(cfg)
    print(json.dumps(summary.get("export_paths") or summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
