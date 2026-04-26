"""Experiment runner for two-phase setup + scoring pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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


def _print_cli_stage_done(stage_slug: str, detail: str) -> None:
    """One-line stage footer; keep tqdm on its own lines above."""
    d = str(detail or "").strip()
    if d:
        print(f"{stage_slug}: {d}", flush=True)
    else:
        print(f"{stage_slug}: complete", flush=True)


def _print_experiment_cli_summary(out: dict[str, Any]) -> None:
    """Short paths-only summary instead of dumping the full result dict."""
    m = out.get("manifest") or {}
    dry = bool(out.get("dry_run"))
    print("", flush=True)
    if dry:
        print(
            "experiment: dry run (paths resolved; graph setup and community not executed).",
            flush=True,
        )
    else:
        print("experiment: complete", flush=True)
    mj = str(out.get("manifest_json") or "").strip()
    if mj:
        print(f"  run_manifest: {mj}", flush=True)
    gbr = str(m.get("graph_bundle_root") or "").strip()
    gid = str(m.get("graph_id") or "").strip()
    if gbr and gid:
        print(f"  graph_bundle: {Path(gbr) / gid}", flush=True)
    rr = str(m.get("run_root") or "").strip()
    if rr:
        print(f"  scoring_run_dir: {rr}", flush=True)
    mode = str(m.get("mode") or "").strip()
    if mode:
        print(f"  mode: {mode}", flush=True)
    sm = m.get("score_mode")
    if sm:
        print(f"  score_mode: {sm}", flush=True)
    for row in out.get("community_results") or []:
        tgt = str(row.get("target") or "").strip() or "(target)"
        cr = row.get("community_result") or {}
        if not isinstance(cr, dict):
            continue
        if cr.get("dry_run"):
            print(f"  [{tgt}] community: (dry run — not executed)", flush=True)
            continue
        od = str(cr.get("output_dir") or "").strip()
        sj = str(cr.get("summary_json") or "").strip()
        if od:
            print(f"  [{tgt}] community_dir: {od}", flush=True)
        if sj:
            print(f"  [{tgt}] community_summary: {sj}", flush=True)


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


def _resolve_target_edges_csv(*, bundle_dir: Path, graph_id: str, target: str) -> Path:
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
    else:
        raise ValueError(f"Unsupported score target: {target!r}")
    if not p.is_file():
        raise FileNotFoundError(f"Target edges file not found for target={target!r}: {p}")
    return p


def _dry_run_planned_target_edges_csv(*, bundle_dir: Path, graph_id: str, target: str) -> Path:
    """Paths that match the bundle layout; files need not exist (``--dry-run``)."""
    target_l = str(target).strip().lower()
    if target_l == "seed_candidate":
        return bundle_dir / "seed_candidate" / graph_id / "seed_candidate_pairgraph_unscored.csv"
    if target_l == "candidate":
        return bundle_dir / "candidate" / graph_id / "candidate_generation_dryrun" / "candidate_union.csv"
    if target_l == "seed":
        return bundle_dir / "seed" / graph_id / "seed_generation_dryrun" / "seed_edges_all.csv"
    if target_l == "anchor":
        return bundle_dir / "anchor" / graph_id / "anchor_graph_edges_unscored.csv"
    raise ValueError(f"Unsupported score target: {target!r}")


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


def run_experiment(config: dict[str, Any], *, dry_run: bool = False) -> dict[str, Any]:
    cfg = dict(config)
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
        from analysis.pipelines.graph_setup_pipeline import run_graph_setup

        setup_result = run_graph_setup(
            project_root=PROJECT_ROOT,
            graph_id=graph_id,
            graph_bundle_root=graph_bundle_root,
            setup_cfg=setup,
        )
        if setup_result and not dry_run:
            br = str(setup_result.get("bundle_root") or "").strip()
            if br:
                _print_cli_stage_done("graph_setup", f"graph_bundle={br}")
    if dry_run:
        bundle_dir = (graph_bundle_root / graph_id).resolve()
    else:
        bundle_dir = _resolve_bundle_dir(graph_bundle_root=graph_bundle_root, graph_id=graph_id)
    # PU scorer: empty pair_dataset_csv would read pair_dataset_csv from GNN training_config.json,
    # which often points at an old anchor_candidates/... layout. Prefer this run's graph bundle.
    if score_mode == "seed_candidate_pu_v1":
        pu_run = dict(score_params.get("pu_run") or {})
        if not str(pu_run.get("pair_dataset_csv") or "").strip():
            bundle_pair_csv = (bundle_dir / "pair_training" / graph_id / "pair_training_dataset.csv").resolve()
            score_params = {**score_params, "pu_run": {**pu_run, "pair_dataset_csv": str(bundle_pair_csv)}}
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
        if dry_run:
            target_edges_csv = _dry_run_planned_target_edges_csv(
                bundle_dir=bundle_dir, graph_id=graph_id, target=target
            )
        else:
            target_edges_csv = _resolve_target_edges_csv(
                bundle_dir=bundle_dir, graph_id=graph_id, target=target
            )
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
            from analysis.utils.anchor_graph_community_helpers import run_anchor_multi_gt_community_sweep

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
        type=str,
        default="",
        dest="scoring_run_id",
        help="Override experiment.scoring_run_id.",
    )
    p.add_argument("--gt-set", type=str, default="", help="Override selection.gt_set")
    p.add_argument(
        "--graph-id",
        type=str,
        default="",
        dest="graph_id",
        help="Override experiment.graph_id.",
    )
    p.add_argument(
        "--dump-json",
        action="store_true",
        help="Print the full JSON result to stdout (default is a short paths-only summary).",
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
    if args.dump_json:
        print(json.dumps(out, indent=2, ensure_ascii=False))
    else:
        _print_experiment_cli_summary(out)


if __name__ == "__main__":
    main()
