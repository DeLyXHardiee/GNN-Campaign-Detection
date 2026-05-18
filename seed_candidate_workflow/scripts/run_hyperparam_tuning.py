"""Hyperparameter tuner for seed/candidate graph generation.

This is a thin orchestrator over ``seed_candidate_workflow/pipelines/run_experiment.py``.

Per trial:

1. Sample params via Optuna TPE (or replay them on resume).
2. Materialize a per-trial config bundle (anchor/seed/candidate/experiment JSONs).
3. Decide ``setup_gnn_score`` vs ``score_only`` based on whether the bundle's
   ``bundle_hash`` already has a built graph bundle + trained GNN model.
4. Invoke ``run_experiment.py`` as a subprocess with ``PIPELINE_RUN_OUTPUT_DIR``
   set so the GNN run output dir is unique per bundle.
5. Parse ``run_manifest.json``, extract the primary-GT ``v_measure`` as the
   objective, and append a JSONL row.
6. Prune non-top-K artifact directories.

The JSONL file doubles as the resume checkpoint: completed rows are replayed
into a fresh Optuna study on startup via ``study.add_trial``.

Usage:

    python seed_candidate_workflow/scripts/run_hyperparam_tuning.py \
        --study-name relax85_seedsem95_tpe

Stop with Ctrl+C; the in-flight trial finishes (or is marked failed), then
the loop exits cleanly. Run the same command again to resume.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import optuna  # noqa: E402

from seed_candidate_workflow.scripts.tuning.config_patcher import (  # noqa: E402
    MaterializedTrialConfig,
    bundle_hash as compute_bundle_hash,
    full_hash as compute_full_hash,
    materialize_trial_config,
    switch_to_score_only,
)
from seed_candidate_workflow.scripts.tuning.retention import (  # noqa: E402
    StudyPaths,
    apply_retention,
)
from seed_candidate_workflow.scripts.tuning.search_space import (  # noqa: E402
    PARAM_SPECS,
    applied_value,
    enabled_specs,
    spec_by_name,
)
from seed_candidate_workflow.scripts.tuning.store import (  # noqa: E402
    TrialRecord,
    append_record,
    best_record,
    iter_records,
    make_study,
    reseed_study_from_jsonl,
    utc_isoformat_now,
)


DEFAULT_BASE_CONFIG = (
    PROJECT_ROOT
    / "seed_candidate_workflow"
    / "configs"
    / "experiments"
    / "exp04.seedcand.relaxed_sem85.pu.json"
)
DEFAULT_TUNING_ROOT = PROJECT_ROOT / "seed_candidate_workflow" / "output" / "tuning"
DEFAULT_GRAPH_BUNDLE_ROOT = (
    PROJECT_ROOT / "seed_candidate_workflow" / "output" / "graph_bundles"
)
DEFAULT_SCORING_OUTPUT_ROOT = (
    PROJECT_ROOT / "seed_candidate_workflow" / "output" / "scoring_runs"
)
DEFAULT_GNN_RUNS_PARENT = PROJECT_ROOT / "output" / "runs"
DEFAULT_PRIMARY_GT_BASENAME = "ground_truth_merged.json"
DEFAULT_KEEP_TOP_K = 5
DEFAULT_WARMUP_TRIALS = 8
DEFAULT_SEED = 42


# ---------------------------------------------------------------------------
# Bundle / GNN-model cache detection
# ---------------------------------------------------------------------------


def _has_built_bundle(*, graph_bundle_root: Path, graph_id: str) -> bool:
    """A bundle is "built" when the pair_training CSV (the most downstream
    artifact ``setup_gnn_score`` produces in the setup phase) exists.
    """
    pair_csv = (
        graph_bundle_root
        / graph_id
        / "pair_training"
        / graph_id
        / "pair_training_dataset.csv"
    )
    return pair_csv.is_file()


def _has_trained_gnn(*, gnn_runs_parent: Path, gnn_run_id: str) -> bool:
    """A GNN run is "trained" when ``gnn/models/best_model.pt`` exists."""
    candidate = gnn_runs_parent / gnn_run_id / "gnn" / "models" / "best_model.pt"
    return candidate.is_file()


def decide_run_mode(
    *,
    graph_bundle_root: Path,
    gnn_runs_parent: Path,
    graph_id: str,
    gnn_run_id: str,
) -> str:
    """Return ``"score_only"`` if both bundle and trained GNN are reusable, else ``"setup_gnn_score"``."""
    if _has_built_bundle(graph_bundle_root=graph_bundle_root, graph_id=graph_id) and _has_trained_gnn(
        gnn_runs_parent=gnn_runs_parent, gnn_run_id=gnn_run_id
    ):
        return "score_only"
    return "setup_gnn_score"


# ---------------------------------------------------------------------------
# Subprocess runner
# ---------------------------------------------------------------------------


def _run_experiment_subprocess(
    *,
    experiment_config_path: Path,
    gnn_run_dir: Path,
    log_path: Path,
    trial_number: int,
    quiet: bool = False,
    env_overrides: dict[str, str] | None = None,
) -> int:
    """Invoke ``run_experiment.py`` as a child process.

    Output is teed: every line is appended to ``log_path`` and (unless
    ``quiet`` is set) also forwarded to the orchestrator's stdout with a
    ``[trial N]`` prefix so the user can follow progress in real time.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PIPELINE_RUN_OUTPUT_DIR"] = str(gnn_run_dir.expanduser().resolve())
    env["PYTHONUNBUFFERED"] = "1"
    # Force tqdm to render ASCII bars on the child side, since we're piping
    # output through a non-tty. Without this tqdm may stay silent on pipes.
    env.setdefault("TQDM_MININTERVAL", "1.0")
    # Suppress the PyG NeighborSampler/pyg-lib deprecation warning that
    # otherwise spams once per sampler construction (= once per training batch).
    # core/GNN/src/__init__.py also installs an in-process filter, but setting
    # the env var here covers DataLoader workers spawned with "spawn" start
    # method (which start a fresh interpreter that only inherits env).
    _pyg_warn_filter = (
        "ignore:Using 'NeighborSampler' without a 'pyg-lib' installation:UserWarning"
    )
    existing = env.get("PYTHONWARNINGS", "")
    if _pyg_warn_filter not in existing:
        env["PYTHONWARNINGS"] = (
            (existing + ",") if existing else ""
        ) + _pyg_warn_filter
    if env_overrides:
        env.update(env_overrides)
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "seed_candidate_workflow" / "pipelines" / "run_experiment.py"),
        "--config",
        str(experiment_config_path),
    ]
    prefix = f"[trial {trial_number}] "
    with open(log_path, "a", encoding="utf-8") as logf:
        logf.write(
            f"\n===== {utc_isoformat_now()} :: {' '.join(cmd)} =====\n"
            f"PIPELINE_RUN_OUTPUT_DIR={env['PIPELINE_RUN_OUTPUT_DIR']}\n"
        )
        logf.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=str(PROJECT_ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            text=True,
        )
        try:
            assert proc.stdout is not None
            for raw_line in proc.stdout:
                # Always write to log unchanged (preserves \r for tqdm reruns).
                logf.write(raw_line)
                logf.flush()
                if not quiet:
                    # For the terminal: turn any embedded \r into newlines so
                    # tqdm progress updates each get their own line, and add
                    # the trial prefix so interleaving is unambiguous.
                    for segment in raw_line.replace("\r", "\n").splitlines():
                        if not segment.strip():
                            continue
                        sys.stdout.write(prefix + segment + "\n")
                    sys.stdout.flush()
            returncode = proc.wait()
        except KeyboardInterrupt:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
            raise
    return int(returncode)


# ---------------------------------------------------------------------------
# Manifest parsing
# ---------------------------------------------------------------------------


def _normalize_gt_slug(value: str) -> str:
    p = Path(str(value))
    return p.stem


def parse_run_manifest(
    *,
    manifest_path: Path,
    primary_gt_basename: str,
) -> tuple[float | None, str, dict[str, dict[str, float]]]:
    """Return ``(objective, primary_gt_slug, per_gt_metrics)``.

    ``objective`` is the v_measure of the primary GT (by basename match) for
    the first target's community result; if missing, ``None``.

    ``per_gt_metrics`` is the full v_measure / homogeneity / completeness
    matrix across all targets and GTs (for the JSONL row).
    """
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    primary_slug = _normalize_gt_slug(primary_gt_basename)

    per_gt: dict[str, dict[str, float]] = {}
    objective: float | None = None

    for target in manifest.get("targets") or []:
        target_name = str(target.get("target") or "unknown")
        comm = target.get("community_result") or {}
        for per in comm.get("per_ground_truth_outputs") or []:
            slug = str(per.get("gt_slug") or _normalize_gt_slug(per.get("gt_path") or ""))
            best_row = per.get("best_row") or {}
            metrics = {
                "v_measure": _safe_float(best_row.get("v_measure")),
                "homogeneity": _safe_float(best_row.get("homogeneity")),
                "completeness": _safe_float(best_row.get("completeness")),
                "n_communities": _safe_float(best_row.get("n_communities")),
                "n_eval": _safe_float(best_row.get("n_eval")),
                "resolution": _safe_float(best_row.get("resolution")),
                "min_edge_weight": _safe_float(best_row.get("min_edge_weight")),
            }
            key = f"{target_name}::{slug}"
            per_gt[key] = metrics
            if objective is None and slug == primary_slug:
                objective = metrics["v_measure"]

    return objective, primary_slug, per_gt


def _safe_float(x: Any) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def _resolved_scoring_run_dir(
    *,
    scoring_output_root: Path,
    scoring_run_id: str,
) -> Path:
    return (scoring_output_root / scoring_run_id).resolve()


# ---------------------------------------------------------------------------
# Console formatting
# ---------------------------------------------------------------------------


def _format_param_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4g}"
    return repr(value)


def _print_trial_start_banner(
    *,
    trial_number: int,
    run_mode: str,
    bundle_hash: str,
    graph_id: str,
    scoring_run_id: str,
    gnn_run_dir: Path,
    log_path: Path,
    results_path: Path,
    params: dict[str, Any],
    best_record_so_far: TrialRecord | None,
) -> None:
    cache_note = " (bundle+GNN cache hit)" if run_mode == "score_only" else ""
    print("", flush=True)
    print(f"================ trial {trial_number} ================", flush=True)
    print(f"[trial {trial_number}] mode={run_mode}{cache_note}", flush=True)
    print(f"[trial {trial_number}] bundle_hash={bundle_hash}  graph_id={graph_id}", flush=True)
    print(f"[trial {trial_number}] scoring_run_id={scoring_run_id}", flush=True)
    print(f"[trial {trial_number}] gnn_run_dir={gnn_run_dir}", flush=True)
    print(f"[trial {trial_number}] log_file={log_path}", flush=True)
    print(f"[trial {trial_number}] jsonl={results_path}", flush=True)

    best_params = (best_record_so_far.applied_params or best_record_so_far.params) if best_record_so_far else {}
    diffs: list[tuple[str, Any, Any]] = []
    same: list[tuple[str, Any]] = []
    for name, value in params.items():
        if name in best_params and best_params[name] != value:
            diffs.append((name, best_params[name], value))
        elif name in best_params:
            same.append((name, value))
    if best_record_so_far is not None and diffs:
        print(
            f"[trial {trial_number}] {len(diffs)} param(s) differ from best so far "
            f"(trial={best_record_so_far.trial_number}, "
            f"v_measure={best_record_so_far.objective:.4f}):",
            flush=True,
        )
        for name, old, new in diffs:
            print(
                f"[trial {trial_number}]   {name}: "
                f"{_format_param_value(old)} -> {_format_param_value(new)}",
                flush=True,
            )
    else:
        print(f"[trial {trial_number}] sampled params:", flush=True)
        for name, value in params.items():
            print(f"[trial {trial_number}]   {name} = {_format_param_value(value)}", flush=True)


def _print_trial_end_banner(
    *,
    trial_number: int,
    status: str,
    objective: float | None,
    duration_seconds: float | None,
    error: str | None,
    primary_gt_slug: str,
    per_gt_metrics: dict[str, dict[str, float]],
) -> None:
    dur = f"{duration_seconds:.1f}s" if duration_seconds is not None else "?"
    if status == "completed" and objective is not None:
        head = (
            f"[trial {trial_number}] DONE in {dur}  "
            f"v_measure({primary_gt_slug}) = {objective:.4f}"
        )
    else:
        head = f"[trial {trial_number}] FAILED in {dur}  status={status}"
        if error:
            head += f"  ({error})"
    print(head, flush=True)
    if per_gt_metrics:
        for key, metrics in per_gt_metrics.items():
            vm = metrics.get("v_measure", float("nan"))
            hom = metrics.get("homogeneity", float("nan"))
            com = metrics.get("completeness", float("nan"))
            ncomm = metrics.get("n_communities", float("nan"))
            print(
                f"[trial {trial_number}]   {key}: "
                f"v={vm:.4f}  h={hom:.4f}  c={com:.4f}  n_communities={ncomm:.0f}",
                flush=True,
            )


# ---------------------------------------------------------------------------
# Trial executor
# ---------------------------------------------------------------------------


def execute_trial(
    *,
    trial: optuna.Trial,
    cfg: "TunerConfig",
) -> tuple[float | None, TrialRecord]:
    """Sample params, build configs, run a subprocess, parse metrics, return record."""
    from seed_candidate_workflow.scripts.tuning.search_space import suggest_params

    raw_params = suggest_params(trial)

    materialized = materialize_trial_config(
        project_root=PROJECT_ROOT,
        study_name=cfg.study_name,
        trial_number=trial.number,
        params=raw_params,
        base_experiment_config_path=cfg.base_experiment_config_path,
        tuning_root=cfg.tuning_root,
    )

    run_mode = decide_run_mode(
        graph_bundle_root=cfg.graph_bundle_root,
        gnn_runs_parent=cfg.gnn_runs_parent,
        graph_id=materialized.graph_id,
        gnn_run_id=materialized.gnn_run_id,
    )
    if run_mode == "score_only":
        switch_to_score_only(materialized.experiment_config_path)

    gnn_run_dir = (cfg.gnn_runs_parent / materialized.gnn_run_id).resolve()
    log_path = (cfg.tuning_root / "logs" / cfg.study_name / f"{Path(materialized.trial_dir).name}.log")

    applied_params: dict[str, Any] = {}
    for name, value in raw_params.items():
        spec = spec_by_name(name)
        applied_params[name] = applied_value(spec, value) if spec is not None else value

    record = TrialRecord(
        study_name=cfg.study_name,
        trial_number=trial.number,
        status="running",
        started_at=utc_isoformat_now(),
        finished_at="",
        objective=None,
        primary_gt_slug=_normalize_gt_slug(cfg.primary_gt),
        params=raw_params,
        applied_params=applied_params,
        bundle_hash=materialized.bundle_hash,
        full_hash=materialized.full_hash,
        graph_id=materialized.graph_id,
        scoring_run_id=materialized.scoring_run_id,
        gnn_run_id=materialized.gnn_run_id,
        run_mode=run_mode,
        per_gt_metrics={},
        duration_seconds=None,
        error=None,
        base_experiment_config=str(cfg.base_experiment_config_path),
        runner_returncode=None,
    )

    _print_trial_start_banner(
        trial_number=trial.number,
        run_mode=run_mode,
        bundle_hash=materialized.bundle_hash,
        graph_id=materialized.graph_id,
        scoring_run_id=materialized.scoring_run_id,
        gnn_run_dir=gnn_run_dir,
        log_path=log_path,
        results_path=cfg.results_path,
        params=applied_params,
        best_record_so_far=best_record(cfg.results_path),
    )

    t_start = time.monotonic()
    try:
        returncode = _run_experiment_subprocess(
            experiment_config_path=materialized.experiment_config_path,
            gnn_run_dir=gnn_run_dir,
            log_path=log_path,
            trial_number=trial.number,
            quiet=cfg.quiet,
        )
        record.runner_returncode = returncode
        if returncode != 0:
            raise RuntimeError(
                f"run_experiment.py exited with code {returncode}; see {log_path}"
            )

        manifest_path = (
            _resolved_scoring_run_dir(
                scoring_output_root=cfg.scoring_output_root,
                scoring_run_id=materialized.scoring_run_id,
            )
            / "run_manifest.json"
        )
        if not manifest_path.is_file():
            raise FileNotFoundError(f"run_manifest.json missing: {manifest_path}")
        objective, primary_slug, per_gt = parse_run_manifest(
            manifest_path=manifest_path,
            primary_gt_basename=cfg.primary_gt,
        )
        record.primary_gt_slug = primary_slug
        record.per_gt_metrics = per_gt
        record.objective = objective
        record.status = "completed" if (objective is not None and objective == objective) else "failed"
        if record.status == "failed":
            record.error = (
                f"primary GT {primary_slug!r} not found in run_manifest "
                f"or v_measure is NaN; per_gt keys: {list(per_gt)}"
            )
    except KeyboardInterrupt:
        record.status = "failed"
        record.error = "KeyboardInterrupt"
        raise
    except Exception as e:
        record.status = "failed"
        record.error = f"{type(e).__name__}: {e}"
    finally:
        record.duration_seconds = round(time.monotonic() - t_start, 3)
        record.finished_at = utc_isoformat_now()
        append_record(cfg.results_path, record)
        _print_trial_end_banner(
            trial_number=trial.number,
            status=record.status,
            objective=record.objective,
            duration_seconds=record.duration_seconds,
            error=record.error,
            primary_gt_slug=record.primary_gt_slug,
            per_gt_metrics=record.per_gt_metrics,
        )

    return (record.objective if record.status == "completed" else None, record)


# ---------------------------------------------------------------------------
# Tuner config + main loop
# ---------------------------------------------------------------------------


class TunerConfig:
    def __init__(
        self,
        *,
        study_name: str,
        base_experiment_config_path: Path,
        results_path: Path,
        tuning_root: Path,
        graph_bundle_root: Path,
        scoring_output_root: Path,
        gnn_runs_parent: Path,
        primary_gt: str,
        keep_top_k: int,
        warmup_trials: int,
        seed: int,
        quiet: bool = False,
    ) -> None:
        self.study_name = study_name
        self.base_experiment_config_path = base_experiment_config_path
        self.results_path = results_path
        self.tuning_root = tuning_root
        self.graph_bundle_root = graph_bundle_root
        self.scoring_output_root = scoring_output_root
        self.gnn_runs_parent = gnn_runs_parent
        self.primary_gt = primary_gt
        self.keep_top_k = keep_top_k
        self.warmup_trials = warmup_trials
        self.seed = seed
        self.quiet = quiet

    def study_paths(self) -> StudyPaths:
        return StudyPaths(
            project_root=PROJECT_ROOT,
            study_name=self.study_name,
            graph_bundle_root=self.graph_bundle_root,
            scoring_output_root=self.scoring_output_root,
            gnn_runs_parent=self.gnn_runs_parent,
            tuning_root=self.tuning_root,
        )


class _StopFlag:
    """Cooperative-stop flag toggled by SIGINT/SIGTERM."""

    def __init__(self) -> None:
        self.stop = False
        self._original_sigint = None
        self._original_sigterm = None

    def install(self) -> None:
        def _handler(signum, _frame):
            print(
                f"\n[tuner] signal {signum} received; stopping after current trial. "
                "Send the signal again to abort immediately.",
                flush=True,
            )
            if self.stop:
                # Second signal: hard exit.
                sys.exit(130)
            self.stop = True

        self._original_sigint = signal.signal(signal.SIGINT, _handler)
        try:
            self._original_sigterm = signal.signal(signal.SIGTERM, _handler)
        except (AttributeError, ValueError):
            self._original_sigterm = None

    def uninstall(self) -> None:
        if self._original_sigint is not None:
            signal.signal(signal.SIGINT, self._original_sigint)
        if self._original_sigterm is not None:
            try:
                signal.signal(signal.SIGTERM, self._original_sigterm)
            except (AttributeError, ValueError):
                pass


def _print_best_summary(jsonl_path: Path, primary_gt: str) -> None:
    rec = best_record(jsonl_path)
    if rec is None:
        print(f"[tuner] no completed trials yet in {jsonl_path}")
        return
    print(
        f"[tuner] best so far: v_measure({_normalize_gt_slug(primary_gt)}) = "
        f"{rec.objective:.4f}  trial={rec.trial_number}  "
        f"graph_id={rec.graph_id}  scoring_run_id={rec.scoring_run_id}"
    )


def _run_loop(
    *,
    cfg: TunerConfig,
    n_trials: int | None,
    max_seconds: float | None,
) -> None:
    cfg.results_path.parent.mkdir(parents=True, exist_ok=True)
    study = make_study(
        study_name=cfg.study_name,
        seed=cfg.seed,
        warmup_trials=cfg.warmup_trials,
    )
    n_added, n_skipped = reseed_study_from_jsonl(study=study, jsonl_path=cfg.results_path)
    print(
        f"[tuner] resumed {n_added} completed trials from "
        f"{cfg.results_path} (skipped {n_skipped})",
        flush=True,
    )
    _print_best_summary(cfg.results_path, cfg.primary_gt)

    stopper = _StopFlag()
    stopper.install()
    t_loop_start = time.monotonic()
    completed_in_session = 0
    try:
        while True:
            if stopper.stop:
                print("[tuner] stop requested; exiting outer loop.", flush=True)
                break
            if n_trials is not None and completed_in_session >= n_trials:
                print(f"[tuner] reached --n-trials={n_trials}; exiting.", flush=True)
                break
            if max_seconds is not None and (time.monotonic() - t_loop_start) >= max_seconds:
                print(f"[tuner] reached --max-seconds={max_seconds}; exiting.", flush=True)
                break

            trial = study.ask()
            try:
                objective, _record = execute_trial(trial=trial, cfg=cfg)
            except KeyboardInterrupt:
                stopper.stop = True
                study.tell(trial, state=optuna.trial.TrialState.FAIL)
                break
            if objective is None or objective != objective:  # NaN check
                study.tell(trial, state=optuna.trial.TrialState.FAIL)
            else:
                study.tell(trial, float(objective))

            apply_retention(
                paths=cfg.study_paths(),
                jsonl_path=cfg.results_path,
                keep_top_k=cfg.keep_top_k,
            )
            completed_in_session += 1
            _print_best_summary(cfg.results_path, cfg.primary_gt)
    finally:
        stopper.uninstall()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _print_param_table() -> None:
    print(f"{'name':<70} {'kind':<12} {'bundle':<8} range/choices")
    print("-" * 110)
    for spec in PARAM_SPECS:
        rng: str
        if spec.choices is not None:
            rng = repr(list(spec.choices))
        elif spec.low is not None and spec.high is not None:
            step = f", step={spec.step}" if spec.step is not None else ""
            log = ", log" if spec.log else ""
            rng = f"[{spec.low}, {spec.high}{step}{log}]"
        else:
            rng = "?"
        marker = "*" if spec.enabled else "(off)"
        print(f"{marker} {spec.name:<68} {spec.kind:<12} {str(spec.bundle_affecting):<8} {rng}")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_hyperparam_tuning",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--study-name",
        type=str,
        required=False,
        default="seedcand_tpe_default",
        help="Stable identifier; trial dirs and JSONL filename derive from this.",
    )
    p.add_argument(
        "--base-config",
        type=Path,
        default=DEFAULT_BASE_CONFIG,
        help="Base experiment JSON used as the template for every trial.",
    )
    p.add_argument(
        "--results-path",
        type=Path,
        default=None,
        help="Where to append per-trial JSONL records (default: <tuning-root>/<study>.jsonl).",
    )
    p.add_argument(
        "--tuning-root",
        type=Path,
        default=DEFAULT_TUNING_ROOT,
        help="Directory for per-trial configs / logs / JSONL.",
    )
    p.add_argument(
        "--graph-bundle-root",
        type=Path,
        default=DEFAULT_GRAPH_BUNDLE_ROOT,
        help="Where run_experiment writes graph bundles.",
    )
    p.add_argument(
        "--scoring-output-root",
        type=Path,
        default=DEFAULT_SCORING_OUTPUT_ROOT,
        help="Where run_experiment writes scoring-run dirs.",
    )
    p.add_argument(
        "--gnn-runs-parent",
        type=Path,
        default=DEFAULT_GNN_RUNS_PARENT,
        help="Where GNN training runs land (must match pipeline_config.output_runs_root).",
    )
    p.add_argument(
        "--primary-gt",
        type=str,
        default=DEFAULT_PRIMARY_GT_BASENAME,
        help="Basename of the ground-truth file whose v_measure is the objective.",
    )
    p.add_argument(
        "--keep-top-k",
        type=int,
        default=DEFAULT_KEEP_TOP_K,
        help="Retain only top-K trials' bundle/scoring/GNN dirs.",
    )
    p.add_argument(
        "--warmup-trials",
        type=int,
        default=DEFAULT_WARMUP_TRIALS,
        help="Number of random startup trials before TPE engages.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Sampler seed (reproducible given the same JSONL prefix).",
    )
    p.add_argument(
        "--n-trials",
        type=int,
        default=None,
        help="Stop after this many trials in this invocation (default: unlimited).",
    )
    p.add_argument(
        "--max-seconds",
        type=float,
        default=None,
        help="Stop after this wall-clock budget (default: unlimited).",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help=(
            "Do not stream the per-trial subprocess output to the terminal. "
            "Output still goes to the per-trial log file. The orchestrator's "
            "own start/end banners and best-so-far summary remain visible."
        ),
    )
    p.add_argument(
        "--list-params",
        action="store_true",
        help="Print the configured search space and exit.",
    )
    p.add_argument(
        "--show-best",
        action="store_true",
        help="Print the current best trial from the JSONL and exit.",
    )
    return p


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.list_params:
        _print_param_table()
        return 0

    results_path = (
        args.results_path
        if args.results_path is not None
        else (args.tuning_root / f"{args.study_name}.jsonl")
    ).resolve()
    base_cfg_path = args.base_config.expanduser().resolve()
    if not base_cfg_path.is_file():
        print(f"--base-config not found: {base_cfg_path}", file=sys.stderr)
        return 2

    cfg = TunerConfig(
        study_name=str(args.study_name),
        base_experiment_config_path=base_cfg_path,
        results_path=results_path,
        tuning_root=args.tuning_root.expanduser().resolve(),
        graph_bundle_root=args.graph_bundle_root.expanduser().resolve(),
        scoring_output_root=args.scoring_output_root.expanduser().resolve(),
        gnn_runs_parent=args.gnn_runs_parent.expanduser().resolve(),
        primary_gt=str(args.primary_gt),
        keep_top_k=int(args.keep_top_k),
        warmup_trials=int(args.warmup_trials),
        seed=int(args.seed),
        quiet=bool(args.quiet),
    )

    if args.show_best:
        _print_best_summary(cfg.results_path, cfg.primary_gt)
        return 0

    _run_loop(cfg=cfg, n_trials=args.n_trials, max_seconds=args.max_seconds)
    _print_best_summary(cfg.results_path, cfg.primary_gt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
