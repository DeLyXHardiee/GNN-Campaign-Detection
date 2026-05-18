"""JSONL-backed checkpoint and Optuna study reseed.

The single source of truth for tuning state is one JSONL file: each line is a
self-contained trial record. The file is append-only (one ``fsync`` per line)
so it survives crashes and Ctrl+C, and it can be opened in any editor for
inspection.

On startup, :func:`reseed_study_from_jsonl` reads every completed record and
calls :meth:`optuna.Study.add_trial` so TPE conditions on prior trials.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import optuna

from seed_candidate_workflow.scripts.tuning.search_space import (
    PARAM_SPECS,
    distribution_for,
    enabled_specs,
    spec_by_name,
)


@dataclass
class TrialRecord:
    """One JSONL row.

    ``params`` are the **raw** sampled values (Optuna-distribution friendly);
    ``applied_params`` are what actually landed in the config files.
    """

    study_name: str
    trial_number: int
    status: str  # "completed" | "failed" | "pruned"
    started_at: str
    finished_at: str
    objective: float | None
    primary_gt_slug: str
    params: dict[str, Any]
    applied_params: dict[str, Any]
    bundle_hash: str
    full_hash: str
    graph_id: str
    scoring_run_id: str
    gnn_run_id: str
    run_mode: str
    per_gt_metrics: dict[str, dict[str, float]] = field(default_factory=dict)
    duration_seconds: float | None = None
    error: str | None = None
    base_experiment_config: str = ""
    runner_returncode: int | None = None


def append_record(path: Path, record: TrialRecord) -> None:
    """Atomically append one JSON record line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(asdict(record), ensure_ascii=False) + "\n"
    with open(path, "a", encoding="utf-8") as f:
        f.write(line)
        f.flush()
        os.fsync(f.fileno())


def iter_records(path: Path) -> Iterable[TrialRecord]:
    if not path.is_file():
        return []
    out: list[TrialRecord] = []
    with open(path, "r", encoding="utf-8") as f:
        for lineno, raw in enumerate(f, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError as e:
                raise ValueError(f"Corrupted JSONL row {path}:{lineno}: {e}") from e
            try:
                out.append(TrialRecord(**payload))
            except TypeError as e:
                # Forward-compatible: ignore unknown extra fields.
                known = {k: payload[k] for k in TrialRecord.__dataclass_fields__ if k in payload}
                out.append(TrialRecord(**known))
    return out


def make_study(
    *,
    study_name: str,
    seed: int,
    warmup_trials: int,
) -> optuna.Study:
    """In-memory TPE study with a small random warmup."""
    sampler = optuna.samplers.TPESampler(
        n_startup_trials=int(warmup_trials),
        seed=int(seed),
        multivariate=True,
        group=True,
    )
    return optuna.create_study(
        study_name=study_name,
        direction="maximize",
        sampler=sampler,
    )


def reseed_study_from_jsonl(
    *,
    study: optuna.Study,
    jsonl_path: Path,
) -> tuple[int, int]:
    """Replay completed trials from the JSONL into ``study``.

    Returns ``(n_added, n_skipped)``. Skipped rows include failed/pruned trials
    and rows whose params reference disabled or removed specs (so renaming a
    spec doesn't crash resume, but those points won't influence TPE).
    """
    if not jsonl_path.is_file():
        return (0, 0)
    name_to_dist = {s.name: distribution_for(s) for s in enabled_specs()}
    n_added = 0
    n_skipped = 0
    for rec in iter_records(jsonl_path):
        if rec.status != "completed" or rec.objective is None:
            n_skipped += 1
            continue
        params_subset: dict[str, Any] = {}
        dists_subset: dict[str, optuna.distributions.BaseDistribution] = {}
        ok = True
        for name, value in rec.params.items():
            dist = name_to_dist.get(name)
            if dist is None:
                ok = False
                break
            params_subset[name] = value
            dists_subset[name] = dist
        if not ok or not params_subset:
            n_skipped += 1
            continue
        try:
            trial = optuna.trial.create_trial(
                params=params_subset,
                distributions=dists_subset,
                value=float(rec.objective),
            )
            study.add_trial(trial)
            n_added += 1
        except Exception:
            n_skipped += 1
    return (n_added, n_skipped)


def best_record(jsonl_path: Path) -> TrialRecord | None:
    """Return the highest-objective completed trial from the JSONL."""
    best: TrialRecord | None = None
    for rec in iter_records(jsonl_path):
        if rec.status != "completed" or rec.objective is None:
            continue
        if best is None or rec.objective > (best.objective or float("-inf")):
            best = rec
    return best


def top_k_records(jsonl_path: Path, k: int) -> list[TrialRecord]:
    """Top-K completed trials by objective, descending. Ties broken by recency."""
    recs = [r for r in iter_records(jsonl_path) if r.status == "completed" and r.objective is not None]
    recs.sort(key=lambda r: (float(r.objective or float("-inf")), r.trial_number), reverse=True)
    return recs[: max(0, int(k))]


def utc_isoformat_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat(timespec="seconds")
