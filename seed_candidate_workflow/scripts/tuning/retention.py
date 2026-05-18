"""Top-K retention sweep for per-trial graph bundles, GNN runs, scoring runs.

A study can generate many trial directories; without pruning, ``output/`` fills
up quickly. After every completed trial we keep only the top-K by objective.

What we delete (only paths that start with the study's ``tuner_<study>__``
prefix, so non-tuner artifacts are never touched):

- ``seed_candidate_workflow/output/graph_bundles/tuner_<study>__<bundle_hash>/``
  — but only when **no surviving top-K trial** still references that bundle.
- ``output/runs/tuner_<study>__<bundle_hash>/``
  — same survival rule (one GNN per bundle).
- ``seed_candidate_workflow/output/scoring_runs/tuner_<study>__t####__<hash>/``
  — per-trial; deleted unless this trial is in the top-K.

Per-trial config dirs under ``output/tuning/configs/<study>/`` and per-trial
log files under ``output/tuning/logs/<study>/`` are pruned for non-top-K
trials to keep disk usage bounded (they're small, but they accumulate).
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from seed_candidate_workflow.scripts.tuning.store import TrialRecord, iter_records, top_k_records


@dataclass(frozen=True)
class StudyPaths:
    """Filesystem roots used by the tuner for one study."""

    project_root: Path
    study_name: str
    graph_bundle_root: Path
    scoring_output_root: Path
    gnn_runs_parent: Path
    tuning_root: Path  # seed_candidate_workflow/output/tuning

    def bundle_prefix(self) -> str:
        return f"tuner_{self.study_name}__"

    def scoring_prefix(self) -> str:
        return f"tuner_{self.study_name}__"

    def gnn_run_prefix(self) -> str:
        return f"tuner_{self.study_name}__"

    def configs_dir(self) -> Path:
        return self.tuning_root / "configs" / self.study_name

    def logs_dir(self) -> Path:
        return self.tuning_root / "logs" / self.study_name


def _trial_dir_name(trial_number: int, full_hash: str) -> str:
    return f"t{trial_number:04d}__{full_hash}"


def _safe_rmtree(path: Path) -> bool:
    if not path.exists():
        return False
    shutil.rmtree(path, ignore_errors=True)
    return True


def apply_retention(
    *,
    paths: StudyPaths,
    jsonl_path: Path,
    keep_top_k: int,
) -> dict[str, list[str]]:
    """Apply top-K retention. Returns a dict of removed paths grouped by kind."""
    survivors = top_k_records(jsonl_path, keep_top_k)
    survivor_full_hashes = {r.full_hash for r in survivors}
    survivor_bundle_hashes = {r.bundle_hash for r in survivors}
    survivor_scoring_ids = {r.scoring_run_id for r in survivors}
    survivor_gnn_ids = {r.gnn_run_id for r in survivors}

    removed: dict[str, list[str]] = {
        "scoring_runs": [],
        "graph_bundles": [],
        "gnn_runs": [],
        "trial_config_dirs": [],
        "trial_logs": [],
    }

    sp = paths.scoring_prefix()
    if paths.scoring_output_root.is_dir():
        for d in paths.scoring_output_root.iterdir():
            if not d.is_dir() or not d.name.startswith(sp):
                continue
            if d.name in survivor_scoring_ids:
                continue
            if _safe_rmtree(d):
                removed["scoring_runs"].append(str(d))

    bp = paths.bundle_prefix()
    if paths.graph_bundle_root.is_dir():
        for d in paths.graph_bundle_root.iterdir():
            if not d.is_dir() or not d.name.startswith(bp):
                continue
            bundle_hash = d.name[len(bp):]
            if bundle_hash in survivor_bundle_hashes:
                continue
            if _safe_rmtree(d):
                removed["graph_bundles"].append(str(d))

    gp = paths.gnn_run_prefix()
    if paths.gnn_runs_parent.is_dir():
        for d in paths.gnn_runs_parent.iterdir():
            if not d.is_dir() or not d.name.startswith(gp):
                continue
            if d.name in survivor_gnn_ids:
                continue
            if _safe_rmtree(d):
                removed["gnn_runs"].append(str(d))

    cd = paths.configs_dir()
    if cd.is_dir():
        for d in cd.iterdir():
            if not d.is_dir():
                continue
            full_hash = d.name.rsplit("__", 1)[-1] if "__" in d.name else None
            if full_hash in survivor_full_hashes:
                continue
            if _safe_rmtree(d):
                removed["trial_config_dirs"].append(str(d))

    ld = paths.logs_dir()
    if ld.is_dir():
        for f in ld.iterdir():
            if not f.is_file():
                continue
            stem = f.stem
            full_hash = stem.rsplit("__", 1)[-1] if "__" in stem else None
            if full_hash in survivor_full_hashes:
                continue
            try:
                f.unlink()
                removed["trial_logs"].append(str(f))
            except OSError:
                pass

    return removed
