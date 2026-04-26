"""Experiment runner for two-phase setup + scoring pipeline."""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.utils.anchor_graph_community_helpers import run_anchor_multi_gt_community_sweep
from analysis.pipelines.graph_setup_pipeline import run_graph_setup
from analysis.utils.config_run_fields import resolve_graph_id, resolve_scoring_run_id
from analysis.utils.graph_scorer_registry import validate_scorer_target


DEFAULT_EXPERIMENT_CONFIG_PATH = (
    PROJECT_ROOT / "analysis" / "configs" / "experiments" / "seed_candidate.pu.default.json"
)
DEFAULT_GT_SETS_PATH = (
    PROJECT_ROOT / "analysis" / "configs" / "experiments" / "gt_sets.json"
)
DEFAULT_ANCHOR_COMMUNITY_CONFIG_PATH = (
    PROJECT_ROOT / "analysis" / "configs" / "anchor_community.default.json"
)
def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_path(raw: str | Path) -> Path:
    p = Path(str(raw)).expanduser()
    if not p.is_absolute():
        p = (PROJECT_ROOT / p).resolve()
    else:
        p = p.resolve()
    return p


def _resolve_bundle_dir(*, graph_bundle_root: Path, graph_id: str) -> Path:
    p = (graph_bundle_root / graph_id).resolve()
    if not p.is_dir():
        raise FileNotFoundError(f"Graph bundle not found for graph_id={graph_id}: {p}")
    return p


def _resolve_target_edges_csv(*, bundle_dir: Path, target: str) -> Path:
    target_l = str(target).strip().lower()
    if target_l == "seed_candidate":
        p = bundle_dir / "seed_candidate" / bundle_dir.name / "seed_candidate_pairgraph_unscored.csv"
    elif target_l == "candidate":
        cand_root = bundle_dir / "candidate" / bundle_dir.name
        dirs = [d for d in cand_root.iterdir() if d.is_dir() and d.name.startswith("candidate_generation_")]
        if not dirs:
            raise FileNotFoundError(f"No candidate stage dirs found under {cand_root}")
        p = max(dirs, key=lambda d: d.stat().st_mtime) / "candidate_union.csv"
    elif target_l == "seed":
        seed_root = bundle_dir / "seed" / bundle_dir.name
        dirs = [d for d in seed_root.iterdir() if d.is_dir() and d.name.startswith("seed_generation_")]
        if not dirs:
            raise FileNotFoundError(f"No seed stage dirs found under {seed_root}")
        p = max(dirs, key=lambda d: d.stat().st_mtime) / "seed_edges_all.csv"
    elif target_l == "anchor":
        p = bundle_dir / "anchor" / bundle_dir.name / "anchor_graph_edges_unscored.csv"
    else:
        raise ValueError(f"Unsupported score target: {target!r}")
    if not p.is_file():
        raise FileNotFoundError(f"Target edges file not found for target={target!r}: {p}")
    return p


def _resolve_gt_paths(*, gt_set_name: str, gt_sets_path: Path) -> list[str]:
    gt_sets = _read_json(gt_sets_path)
    if gt_set_name not in gt_sets:
        raise ValueError(
            f"Unknown gt_set {gt_set_name!r}. Available sets: {sorted(gt_sets)}"
        )
    vals = gt_sets[gt_set_name]
    if not isinstance(vals, list) or not vals:
        raise ValueError(f"gt_set {gt_set_name!r} must map to a non-empty list")
    return [str(v) for v in vals]


def _normalize_experiment_block(cfg: dict[str, Any]) -> None:
    exp = dict(cfg.get("experiment") or {})
    gid = str(exp.get("graph_id") or "").strip()
    legacy_gid = str(exp.get("graph_run_id") or "").strip()
    if not gid and legacy_gid:
        warnings.warn(
            "experiment.graph_run_id is deprecated; use experiment.graph_id instead.",
            DeprecationWarning,
            stacklevel=3,
        )
        exp["graph_id"] = legacy_gid
    sid = str(exp.get("scoring_run_id") or "").strip()
    legacy_sid = str(exp.get("run_id") or "").strip()
    if not sid and legacy_sid:
        warnings.warn(
            "experiment.run_id is deprecated; use experiment.scoring_run_id instead.",
            DeprecationWarning,
            stacklevel=3,
        )
        exp["scoring_run_id"] = legacy_sid
    cfg["experiment"] = exp


def run_experiment(config: dict[str, Any], *, dry_run: bool = False) -> dict[str, Any]:
    cfg = dict(config)
    _normalize_experiment_block(cfg)
    exp = dict(cfg.get("experiment") or {})
    artifacts = dict(cfg.get("artifacts") or {})
    setup = dict(cfg.get("setup") or {})
    selection = dict(cfg.get("selection") or {})
    scoring = dict(cfg.get("scoring") or {})
    community = dict(cfg.get("community") or {})

    scoring_run_id = resolve_scoring_run_id(exp)
    graph_id = resolve_graph_id(exp)
    mode = str(exp.get("mode") or "").strip().lower()
    if mode not in {"setup_only", "score_only", "setup_and_score"}:
        raise ValueError("experiment.mode must be one of: setup_only, score_only, setup_and_score")

    graph_bundle_root = _resolve_path(str(artifacts.get("graph_bundle_root") or "analysis/output/graph_bundles"))
    scoring_output_root = _resolve_path(str(artifacts.get("scoring_output_root") or "analysis/output/scoring_runs"))
    run_root = (scoring_output_root / scoring_run_id).resolve()
    run_root.mkdir(parents=True, exist_ok=True)

    gt_set = str(selection.get("gt_set") or "default_multi_gt").strip()
    gt_sets_path = _resolve_path(str(selection.get("gt_sets_path") or DEFAULT_GT_SETS_PATH))
    gt_paths = _resolve_gt_paths(gt_set_name=gt_set, gt_sets_path=gt_sets_path)

    score_targets = selection.get("score_targets") or ["seed_candidate"]
    if not isinstance(score_targets, list) or not score_targets:
        raise ValueError("selection.score_targets must be a non-empty list")
    score_targets = [str(t).strip().lower() for t in score_targets if str(t).strip()]

    score_mode_raw = str(scoring.get("score_mode") or "").strip()
    score_mode = "" if score_mode_raw.lower() == "none" else score_mode_raw
    score_params_root = dict(scoring.get("params") or {})
    if score_mode == "seed_candidate_handcrafted_v1":
        score_params = dict(score_params_root.get("handcrafted") or {})
    elif score_mode == "seed_candidate_pu_v1":
        score_params = dict(score_params_root.get("pu") or {})
    else:
        score_params = {}

    setup_result = None
    if mode in {"setup_only", "setup_and_score"} and not dry_run:
        setup_result = run_graph_setup(
            project_root=PROJECT_ROOT,
            graph_id=graph_id,
            graph_bundle_root=graph_bundle_root,
            setup_cfg=setup,
        )
    if dry_run and mode in {"setup_only", "setup_and_score"}:
        bundle_dir = (graph_bundle_root / graph_id).resolve()
    else:
        bundle_dir = _resolve_bundle_dir(graph_bundle_root=graph_bundle_root, graph_id=graph_id)
    anchor_output_root = bundle_dir / "anchor"
    if not dry_run and not (anchor_output_root / graph_id).is_dir():
        raise FileNotFoundError(f"Anchor graph bundle missing: {anchor_output_root / graph_id}")

    run_manifest: dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "scoring_run_id": scoring_run_id,
        "graph_id": graph_id,
        "mode": mode,
        "graph_bundle_root": str(graph_bundle_root),
        "scoring_output_root": str(scoring_output_root),
        "run_root": str(run_root),
        "gt_set": gt_set,
        "gt_sets_path": str(gt_sets_path),
        "gt_paths": gt_paths,
        "score_mode": score_mode or None,
        "score_targets": score_targets,
    }
    if setup_result is not None:
        run_manifest["setup"] = setup_result

    if mode == "setup_only":
        p_manifest = run_root / "run_manifest.json"
        p_manifest.write_text(json.dumps(run_manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        return {
            "dry_run": dry_run,
            "manifest": run_manifest,
            "manifest_json": str(p_manifest),
            "setup_result": setup_result,
            "community_results": [],
        }

    target_results: list[dict[str, Any]] = []
    for target in score_targets:
        if score_mode:
            validate_scorer_target(score_mode=score_mode, graph_kind=target)
        if dry_run and mode in {"setup_only", "setup_and_score"}:
            target_edges_csv = bundle_dir / f"dry_run_{target}.csv"
        else:
            target_edges_csv = _resolve_target_edges_csv(bundle_dir=bundle_dir, target=target)
        target_root = (run_root / target).resolve()
        target_root.mkdir(parents=True, exist_ok=True)

        base_community_cfg_path = _resolve_path(str(community.get("base_config") or DEFAULT_ANCHOR_COMMUNITY_CONFIG_PATH))
        comm_cfg = _read_json(base_community_cfg_path)
        comm_cfg.setdefault("run", {})
        comm_cfg.setdefault("sweep", {})
        comm_cfg.setdefault("ground_truth", {})
        comm_cfg.setdefault("output", {})

        comm_cfg["run"]["graph_id"] = graph_id
        comm_cfg["run"]["anchor_output_root"] = str(anchor_output_root)
        comm_cfg["run"]["custom_edges_csv"] = str(target_edges_csv)
        comm_cfg["run"]["community_bundle_out_dir"] = str((target_root / "community").resolve())
        comm_cfg["run"]["seed_output_root"] = str(bundle_dir / "seed")
        seed_root_dir = bundle_dir / "seed" / graph_id
        if seed_root_dir.is_dir():
            seed_dirs = [d for d in seed_root_dir.iterdir() if d.is_dir() and d.name.startswith("seed_generation_")]
            if seed_dirs:
                comm_cfg["run"]["seed_stage_dir"] = str(max(seed_dirs, key=lambda d: d.stat().st_mtime))
        comm_cfg["sweep"]["score_mode"] = score_mode
        comm_cfg["sweep"]["score_params"] = score_params
        comm_cfg["ground_truth"]["paths"] = gt_paths
        comm_cfg["output"]["output_root"] = str((target_root / "community_root").resolve())
        comm_cfg["output"]["stage_name"] = "community_sweep"
        comm_cfg["sweep"].update(dict(community.get("sweep") or {}))

        if dry_run:
            comm_res = {"dry_run": True, "target": target, "community_config": comm_cfg}
        else:
            comm_res = run_anchor_multi_gt_community_sweep(comm_cfg)
        target_results.append(
            {
                "target": target,
                "edges_csv": str(target_edges_csv),
                "community_result": comm_res,
            }
        )

    run_manifest["targets"] = target_results
    p_manifest = run_root / "run_manifest.json"
    p_manifest.write_text(json.dumps(run_manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return {
        "dry_run": dry_run,
        "manifest": run_manifest,
        "manifest_json": str(p_manifest),
        "setup_result": setup_result,
        "community_results": target_results,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_EXPERIMENT_CONFIG_PATH,
        help="Path to experiment JSON config.",
    )
    p.add_argument("--dry-run", action="store_true", help="Resolve and print run plan without executing.")
    p.add_argument(
        "--run-mode",
        type=str,
        default="",
        help="Override experiment mode: setup_only|score_only|setup_and_score",
    )
    p.add_argument(
        "--mode-override",
        type=str,
        default="",
        help="Override scoring.score_mode",
    )
    p.add_argument(
        "--scoring-run-id",
        "--run-id",
        type=str,
        default="",
        dest="scoring_run_id",
        help="Override experiment.scoring_run_id (legacy: experiment.run_id; --run-id alias).",
    )
    p.add_argument("--gt-set", type=str, default="", help="Override selection.gt_set")
    p.add_argument(
        "--graph-id",
        "--graph-run-id",
        type=str,
        default="",
        dest="graph_id",
        help="Override experiment.graph_id (legacy: experiment.graph_run_id).",
    )
    args = p.parse_args()

    cfg_path = args.config.expanduser().resolve()
    cfg = _read_json(cfg_path)
    if args.run_mode:
        cfg.setdefault("experiment", {})
        cfg["experiment"]["mode"] = str(args.run_mode)
    if args.mode_override:
        cfg.setdefault("scoring", {})
        cfg["scoring"]["score_mode"] = str(args.mode_override)
    if args.scoring_run_id:
        cfg.setdefault("experiment", {})
        cfg["experiment"]["scoring_run_id"] = str(args.scoring_run_id)
    if args.gt_set:
        cfg.setdefault("selection", {})
        cfg["selection"]["gt_set"] = str(args.gt_set)
    if args.graph_id:
        cfg.setdefault("experiment", {})
        cfg["experiment"]["graph_id"] = str(args.graph_id)

    out = run_experiment(cfg, dry_run=bool(args.dry_run))
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
