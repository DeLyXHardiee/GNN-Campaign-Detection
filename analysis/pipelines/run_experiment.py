"""Experiment runner for two-phase setup + scoring pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - optional dependency
    tqdm = None

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.utils.graph_scorer_registry import SCORER_REGISTRY, apply_scorer, validate_scorer_target
from analysis.pipelines import runner_config as rcfg
from analysis.pipelines import runner_manifest as rman
from analysis.pipelines import runner_targets as rtgt


DEFAULT_EXPERIMENT_CONFIG_PATH = (
    PROJECT_ROOT / "analysis" / "configs" / "experiments" / "exp03.seedcand.setupscore.pu.json"
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
    rman.print_experiment_cli_summary(out)


def _read_json(path: Path) -> dict[str, Any]:
    return rcfg.read_json(path)


def _resolve_path(raw: str | Path) -> Path:
    p = Path(str(raw)).expanduser()
    if not p.is_absolute():
        p = (PROJECT_ROOT / p).resolve()
    else:
        p = p.resolve()
    return p


def _resolve_bundle_dir(*, graph_bundle_root: Path, graph_id: str) -> Path:
    return rtgt.resolve_bundle_dir(graph_bundle_root=graph_bundle_root, graph_id=graph_id)


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
    elif target_l == "semantic_shard":
        p = bundle_dir / "semantic_shard" / graph_id / "semantic_shard_pairgraph_unscored.csv"
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
    if target_l == "semantic_shard":
        return bundle_dir / "semantic_shard" / graph_id / "semantic_shard_pairgraph_unscored.csv"
    raise ValueError(f"Unsupported score target: {target!r}")


TARGET_EDGE_RESOLVERS = rtgt.TARGET_EDGE_RESOLVERS


def _gt_slug(path: str | Path) -> str:
    p = Path(path)
    stem = p.stem or "ground_truth"
    return "".join(ch if (ch.isalnum() or ch in {"_", "-"}) else "_" for ch in stem)


def _run_semantic_shard_multi_gt_community_sweep(
    *,
    bundle_dir: Path,
    graph_id: str,
    target_root: Path,
    score_mode: str,
    score_params: dict[str, Any],
    gt_paths: list[str],
    sweep_cfg: dict[str, Any],
) -> dict[str, Any]:
    from analysis.utils import raw_gnn_notebook as rn
    from analysis.utils import community_sweep_driver as csd
    from analysis.utils import semantic_shard_step3_helpers as s3

    shard_root = (bundle_dir / "semantic_shard" / graph_id).resolve()
    p_assign = shard_root / "semantic_shard_assignments.csv"
    p_nodes = shard_root / "semantic_shard_nodes.csv"
    p_edges = shard_root / "semantic_shard_edges_weighted.csv"
    if not p_assign.is_file():
        raise FileNotFoundError(f"semantic shard assignments missing: {p_assign}")
    if not p_nodes.is_file():
        raise FileNotFoundError(f"semantic shard nodes missing: {p_nodes}")
    if not p_edges.is_file():
        raise FileNotFoundError(f"semantic shard weighted edges missing: {p_edges}")

    assignments_df = pd.read_csv(p_assign, low_memory=False)
    nodes_df = pd.read_csv(p_nodes, low_memory=False)
    edges_df = pd.read_csv(p_edges, low_memory=False)
    assignments_df["external_id"] = assignments_df["external_id"].astype(str)
    assignments_df["shard_id"] = assignments_df["shard_id"].astype(str)
    nodes_df["shard_id"] = nodes_df["shard_id"].astype(str)
    edges_df["shard_a"] = edges_df["shard_a"].astype(str)
    edges_df["shard_b"] = edges_df["shard_b"].astype(str)

    if score_mode:
        sr = apply_scorer(
            score_mode=score_mode,
            graph_kind="semantic_shard",
            score_params=score_params,
            payload={"shard_edges_df": edges_df},
        )
        edges_df = sr.scored_all
        if "shard_a" not in edges_df.columns or "shard_b" not in edges_df.columns:
            raise ValueError("Semantic shard scorer output must include shard_a and shard_b columns")
    else:
        # Unweighted baseline: topology-only partitioning for semantic shard comparison.
        edges_df = edges_df.copy()
        edges_df["edge_weight"] = 1.0

    methods = [str(x).strip().lower() for x in (sweep_cfg.get("methods") or ["louvain", "leiden"]) if str(x).strip()]
    resolutions = [float(x) for x in (sweep_cfg.get("resolutions") or [1.0])]
    weight_thresholds = [float(x) for x in (sweep_cfg.get("weight_thresholds") or [0.0])]
    if not score_mode:
        weight_thresholds = [0.0]
    seed = int(sweep_cfg.get("seed", 0))
    sort_by = "v_measure"

    out_dir = (target_root / "community").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    def _per_gt(gt_path: str) -> tuple[pd.DataFrame, dict[str, Any]]:
        gt_label_map, _eid_row, _campaign_to_members = rn.load_ground_truth_structures(gt_path)
        sweep_frames: list[pd.DataFrame] = []
        preds_map: dict[str, pd.DataFrame] = {}
        infos_map: dict[str, dict[str, Any]] = {}
        method_iter = tqdm(methods, desc=f"Sweep methods [{_gt_slug(gt_path)}]", leave=False) if tqdm is not None else methods
        for method in method_iter:
            sweep_df, email_preds_by_key, info_by_key = s3.run_community_sweep(
                assignments_df=assignments_df,
                shard_ids=nodes_df["shard_id"].astype(str).tolist(),
                edges_df=edges_df,
                gt_label_map=gt_label_map,
                method=method,
                resolution_values=resolutions,
                min_edge_weight_values=weight_thresholds,
                weight_col="edge_weight",
                seed=seed,
            )
            if sweep_df.empty:
                continue
            sweep_df = sweep_df.copy()
            sweep_df["setting_key"] = [f"{method}:{k}" for k in sweep_df["setting_key"].astype(str).tolist()]
            sweep_frames.append(sweep_df)
            for k, v in email_preds_by_key.items():
                preds_map[f"{method}:{k}"] = v
            for k, v in info_by_key.items():
                infos_map[f"{method}:{k}"] = v

        sweep_all = pd.concat(sweep_frames, axis=0, ignore_index=True) if sweep_frames else pd.DataFrame()
        best_row = s3.best_sweep_metric_row(sweep_all, metric=sort_by)
        return sweep_all, {
            "preds_map": preds_map,
            "infos_map": infos_map,
            "best_row": best_row.to_dict() if not best_row.empty else {},
        }

    def _write_gt(gt_path: str, sweep_all: pd.DataFrame, info: dict[str, Any]) -> dict[str, Any]:
        gt_slug = _gt_slug(gt_path)
        p_sweep = out_dir / f"semantic_shard_community_sweep__{gt_slug}.csv"
        sweep_all.to_csv(p_sweep, index=False)
        best_payload: dict[str, Any] = {
            "graph_id": graph_id,
            "gt_path": str(gt_path),
            "gt_slug": gt_slug,
            "sort_by": sort_by,
            "best_row": dict(info.get("best_row") or {}),
        }
        p_best = out_dir / f"semantic_shard_community_best__{gt_slug}.json"
        p_best.write_text(json.dumps(best_payload, indent=2, ensure_ascii=False), encoding="utf-8")

        p_email = out_dir / f"semantic_shard_email_predictions_best__{gt_slug}.csv"
        p_shard = out_dir / f"semantic_shard_shard_communities_best__{gt_slug}.csv"
        best_row = dict(info.get("best_row") or {})
        preds_map = dict(info.get("preds_map") or {})
        infos_map = dict(info.get("infos_map") or {})
        if best_row:
            skey = str(best_row.get("setting_key") or "")
            best_email_df = preds_map.get(skey)
            if best_email_df is not None:
                best_email_df.to_csv(p_email, index=False)
                best_shard_df = best_email_df[["shard_id", "pred_community"]].drop_duplicates().reset_index(drop=True)
                best_shard_df.to_csv(p_shard, index=False)
                info = infos_map.get(skey) or {}
                best_row = {**best_row, **info}
        return {
            "gt_path": str(gt_path),
            "gt_slug": gt_slug,
            "sweep_csv": str(p_sweep),
            "best_json": str(p_best),
            "best_email_predictions_csv": str(p_email),
            "best_shard_communities_csv": str(p_shard),
            "n_rows": int(len(sweep_all)),
            "best_row": best_row,
        }

    per_gt_outputs, best_rows_by_gt = csd.run_multi_gt_sweep(
        gt_paths=gt_paths,
        per_gt_sweep=_per_gt,
        write_per_gt=_write_gt,
    )

    summary = {
        "target": "semantic_shard",
        "graph_id": graph_id,
        "score_mode": (score_mode or None),
        "methods": methods,
        "weight_thresholds": weight_thresholds,
        "resolutions": resolutions,
        "sort_by": sort_by,
        "semantic_shard_root": str(shard_root),
        "per_ground_truth_outputs": per_gt_outputs,
        "best_rows_by_gt": best_rows_by_gt,
    }
    p_summary = out_dir / "semantic_shard_community_multi_gt_summary.json"
    p_summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return {
        "output_dir": str(out_dir),
        "summary_json": str(p_summary),
        "per_ground_truth_outputs": per_gt_outputs,
        "target": "semantic_shard",
        "metrics": {"sort_by": sort_by},
        "artifacts": {"summary_json": str(p_summary), "output_dir": str(out_dir)},
    }


def _validate_experiment_config(cfg: dict[str, Any]) -> None:
    exp = dict(cfg.get("experiment") or {})
    mode = str(exp.get("mode") or "").strip().lower()
    if mode not in {"setup_only", "score_only", "setup_and_score"}:
        raise ValueError("experiment.mode must be one of: setup_only, score_only, setup_and_score")
    sel = dict(cfg.get("selection") or {})
    targets = sel.get("score_targets") or ["seed_candidate"]
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


def _seed_stage_dir(bundle_dir: Path, graph_id: str) -> str:
    seed_root_dir = bundle_dir / "seed" / graph_id
    if not seed_root_dir.is_dir():
        return ""
    seed_dirs = [d for d in seed_root_dir.iterdir() if d.is_dir() and d.name.startswith("seed_generation_")]
    if not seed_dirs:
        return ""
    return str(max(seed_dirs, key=lambda d: d.stat().st_mtime))


def _execute_anchor_like_target(
    *,
    dry_run: bool,
    target: str,
    graph_id: str,
    bundle_dir: Path,
    target_root: Path,
    target_edges_csv: Path,
    gt_paths: list[str],
    community_cfg: dict[str, Any],
    score_mode: str,
    score_params: dict[str, Any],
) -> dict[str, Any]:
    base_community_cfg_path = _resolve_path(str(community_cfg.get("base_config") or DEFAULT_ANCHOR_COMMUNITY_CONFIG_PATH))
    comm_cfg = _read_json(base_community_cfg_path)
    comm_cfg.setdefault("run", {})
    comm_cfg.setdefault("sweep", {})
    comm_cfg.setdefault("ground_truth", {})
    comm_cfg.setdefault("output", {})
    comm_cfg["run"]["graph_id"] = graph_id
    comm_cfg["run"]["anchor_output_root"] = str(bundle_dir / "anchor")
    comm_cfg["run"]["custom_edges_csv"] = str(target_edges_csv)
    comm_cfg["run"]["community_bundle_out_dir"] = str((target_root / "community").resolve())
    comm_cfg["run"]["seed_output_root"] = str(bundle_dir / "seed")
    sdir = _seed_stage_dir(bundle_dir, graph_id)
    if sdir:
        comm_cfg["run"]["seed_stage_dir"] = sdir
    comm_cfg["sweep"]["score_mode"] = score_mode
    comm_cfg["sweep"]["score_params"] = score_params
    comm_cfg["ground_truth"]["paths"] = gt_paths
    comm_cfg["output"]["output_root"] = str((target_root / "community_root").resolve())
    comm_cfg["output"]["stage_name"] = "community_sweep"
    comm_cfg["sweep"].update(dict(community_cfg.get("sweep") or {}))
    if dry_run:
        return {"dry_run": True, "target": target, "community_config": comm_cfg}
    from analysis.utils.anchor_graph_community_helpers import run_anchor_multi_gt_community_sweep
    out = run_anchor_multi_gt_community_sweep(comm_cfg)
    return {
        "target": target,
        "output_dir": out.get("output_dir"),
        "summary_json": out.get("summary_json"),
        "per_ground_truth_outputs": out.get("per_ground_truth_outputs") or [],
        "artifacts": {"summary_json": out.get("summary_json"), "output_dir": out.get("output_dir")},
        "metrics": {"sort_by": (comm_cfg.get("sweep") or {}).get("sort_by")},
    }


TARGET_REGISTRY: dict[str, dict[str, Any]] = {
    "anchor": {"executor": _execute_anchor_like_target},
    "seed": {"executor": _execute_anchor_like_target},
    "candidate": {"executor": _execute_anchor_like_target},
    "seed_candidate": {"executor": _execute_anchor_like_target},
    "semantic_shard": {"executor": _run_semantic_shard_multi_gt_community_sweep},
}


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
    ctx = rcfg.build_run_context(cfg=cfg, project_root=PROJECT_ROOT, default_gt_sets_path=DEFAULT_GT_SETS_PATH)
    exp = dict(cfg.get("experiment") or {})
    setup = dict(cfg.get("setup") or {})
    community = dict(cfg.get("community") or {})
    graph_id = ctx.graph_id
    scoring_run_id = ctx.scoring_run_id
    mode = ctx.mode
    graph_bundle_root = ctx.graph_bundle_root
    scoring_output_root = ctx.scoring_output_root
    run_root = ctx.run_root
    gt_set = ctx.gt_set
    gt_sets_path = ctx.gt_sets_path
    gt_paths = ctx.gt_paths
    score_targets = ctx.score_targets
    score_mode = ctx.score_mode
    score_params = dict(ctx.score_params)

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
    requires_anchor_bundle = any(t != "semantic_shard" for t in score_targets)
    if not dry_run and requires_anchor_bundle and not (anchor_output_root / graph_id).is_dir():
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
        "manifest_version": "2.0",
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
            "community_results_legacy": [],
        }

    target_results: list[dict[str, Any]] = []
    for target in score_targets:
        target_meta = TARGET_REGISTRY.get(target)
        if target_meta is None:
            raise ValueError(f"Unsupported target {target!r}")
        if score_mode:
            validate_scorer_target(score_mode=score_mode, graph_kind=target)
        edge_resolver = TARGET_EDGE_RESOLVERS["dry_run" if dry_run else "actual"]
        target_edges_csv = edge_resolver(bundle_dir=bundle_dir, graph_id=graph_id, target=target)
        target_root = (run_root / target).resolve()
        target_root.mkdir(parents=True, exist_ok=True)
        if target == "semantic_shard":
            if dry_run:
                comm_res = {"dry_run": True, "target": target}
            else:
                comm_res = target_meta["executor"](
                    bundle_dir=bundle_dir,
                    graph_id=graph_id,
                    target_root=target_root,
                    score_mode=score_mode,
                    score_params=score_params,
                    gt_paths=gt_paths,
                    sweep_cfg=dict(community.get("sweep") or {}),
                )
        else:
            comm_res = target_meta["executor"](
                dry_run=dry_run,
                target=target,
                graph_id=graph_id,
                bundle_dir=bundle_dir,
                target_root=target_root,
                target_edges_csv=target_edges_csv,
                gt_paths=gt_paths,
                community_cfg=community,
                score_mode=score_mode,
                score_params=score_params,
            )
        target_results.append(
            {
                "target": target,
                "inputs": {
                    "edges_csv": str(target_edges_csv),
                    "score_mode": score_mode or None,
                },
                "artifacts": {
                    "output_dir": comm_res.get("output_dir"),
                    "summary_json": comm_res.get("summary_json"),
                },
                "metrics": comm_res.get("metrics") or {},
                "community_result": comm_res,
            }
        )

    run_manifest["targets"] = target_results
    p_manifest = Path(rman.write_manifest(run_root, run_manifest))
    return {
        "dry_run": dry_run,
        "manifest": run_manifest,
        "manifest_json": str(p_manifest),
        "setup_result": setup_result,
        "community_results": target_results,
        "community_results_legacy": [rman.legacy_compatible_target_view(x) for x in target_results],
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
