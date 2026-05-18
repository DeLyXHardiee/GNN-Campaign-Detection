from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from seed_candidate_workflow.utils.config_run_fields import resolve_graph_id, resolve_scoring_run_id
from seed_candidate_workflow.utils.graph_scorer_registry import SCORER_REGISTRY, resolve_score_params


@dataclass(frozen=True)
class RunContext:
    graph_id: str
    scoring_run_id: str
    mode: str
    graph_bundle_root: Path
    scoring_output_root: Path
    run_root: Path
    gt_set: str
    gt_sets_path: Path
    gt_paths: list[str]
    score_targets: list[str]
    score_mode: str
    score_params: dict[str, Any]
    diagnostics_cfg: dict[str, Any]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def resolve_path(project_root: Path, raw: str | Path) -> Path:
    p = Path(str(raw)).expanduser()
    if not p.is_absolute():
        return (project_root / p).resolve()
    return p.resolve()


def resolve_gt_paths(*, gt_set_name: str, gt_sets_path: Path) -> list[str]:
    gt_sets = read_json(gt_sets_path)
    if gt_set_name not in gt_sets:
        raise ValueError(f"Unknown gt_set {gt_set_name!r}. Available sets: {sorted(gt_sets)}")
    vals = gt_sets[gt_set_name]
    if not isinstance(vals, list) or not vals:
        raise ValueError(f"gt_set {gt_set_name!r} must map to a non-empty list")
    return [str(v) for v in vals]


def validate_experiment_config(cfg: dict[str, Any]) -> None:
    exp = dict(cfg.get("experiment") or {})
    mode = str(exp.get("mode") or "").strip().lower()
    if mode not in {"setup_only", "score_only", "setup_and_score"}:
        raise ValueError("experiment.mode must be one of: setup_only, score_only, setup_and_score")
    selection = dict(cfg.get("selection") or {})
    targets = selection.get("score_targets") or ["seed_candidate"]
    if not isinstance(targets, list) or not targets:
        raise ValueError("selection.score_targets must be a non-empty list")
    supported_targets = {"anchor", "seed", "candidate", "seed_candidate", "semantic_shard"}
    bad = sorted(set(str(t).strip().lower() for t in targets if str(t).strip()) - supported_targets)
    if bad:
        raise ValueError(f"Unsupported targets in selection.score_targets: {bad}")
    scoring = dict(cfg.get("scoring") or {})
    score_mode_raw = str(scoring.get("score_mode") or "").strip()
    score_mode = "" if score_mode_raw.lower() == "none" else score_mode_raw
    if score_mode and score_mode not in SCORER_REGISTRY:
        raise ValueError(f"Unknown scoring.score_mode {score_mode!r}. Available: {sorted(SCORER_REGISTRY)}")
    diagnostics_cfg = scoring.get("diagnostics")
    if diagnostics_cfg is not None and not isinstance(diagnostics_cfg, dict):
        raise ValueError("scoring.diagnostics must be an object when provided")


def build_run_context(
    *,
    cfg: dict[str, Any],
    project_root: Path,
    default_gt_sets_path: Path,
) -> RunContext:
    validate_experiment_config(cfg)
    exp = dict(cfg.get("experiment") or {})
    artifacts = dict(cfg.get("artifacts") or {})
    selection = dict(cfg.get("selection") or {})
    scoring = dict(cfg.get("scoring") or {})

    scoring_run_id = resolve_scoring_run_id(exp)
    graph_id = resolve_graph_id(exp)
    mode = str(exp.get("mode") or "").strip().lower()
    graph_bundle_root = resolve_path(project_root, str(artifacts.get("graph_bundle_root") or "seed_candidate_workflow/output/graph_bundles"))
    scoring_output_root = resolve_path(project_root, str(artifacts.get("scoring_output_root") or "seed_candidate_workflow/output/scoring_runs"))
    run_root = (scoring_output_root / scoring_run_id).resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    gt_set = str(selection.get("gt_set") or "default_multi_gt").strip()
    gt_sets_path = resolve_path(project_root, str(selection.get("gt_sets_path") or default_gt_sets_path))
    gt_paths = resolve_gt_paths(gt_set_name=gt_set, gt_sets_path=gt_sets_path)
    score_targets = [str(t).strip().lower() for t in (selection.get("score_targets") or ["seed_candidate"]) if str(t).strip()]
    score_mode_raw = str(scoring.get("score_mode") or "").strip()
    score_mode = "" if score_mode_raw.lower() == "none" else score_mode_raw
    score_params = resolve_score_params(score_mode, dict(scoring.get("params") or {})) if score_mode else {}
    diagnostics_cfg = dict(scoring.get("diagnostics") or {})
    return RunContext(
        graph_id=graph_id,
        scoring_run_id=scoring_run_id,
        mode=mode,
        graph_bundle_root=graph_bundle_root,
        scoring_output_root=scoring_output_root,
        run_root=run_root,
        gt_set=gt_set,
        gt_sets_path=gt_sets_path,
        gt_paths=gt_paths,
        score_targets=score_targets,
        score_mode=score_mode,
        score_params=score_params,
        diagnostics_cfg=diagnostics_cfg,
    )
