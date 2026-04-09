"""Resolve run output directories under ``output_runs_root``."""
from __future__ import annotations

from pathlib import Path

try:
    from config.pipeline_config import output_runs_parent_from_pipeline
except ModuleNotFoundError:
    from core.config.pipeline_config import output_runs_parent_from_pipeline


def resolve_run_dir_by_run_id(cfg: dict, run_id: str) -> Path:
    """
    Resolve ``run_id`` to an existing directory under the configured runs root.

    Accepts either a full folder name (e.g. ``my_experiment (1)``) or a logical
    prefix: if ``<runs_root>/<run_id>`` is missing, a unique subdirectory whose
    name starts with ``run_id`` is used; if several match, raises ``ValueError``.
    """
    rid = (run_id or "").strip()
    if not rid:
        raise ValueError("run_id must be non-empty.")

    runs_root = Path(output_runs_parent_from_pipeline(cfg)).expanduser().resolve()
    if not runs_root.is_dir():
        raise FileNotFoundError(f"Runs root does not exist: {runs_root}")

    direct = runs_root / rid
    if direct.is_dir():
        return direct.resolve()

    matches = sorted(
        p
        for p in runs_root.iterdir()
        if p.is_dir() and p.name.startswith(rid)
    )
    if len(matches) == 1:
        return matches[0].resolve()
    if len(matches) > 1:
        names = ", ".join(m.name for m in matches[:15])
        more = " …" if len(matches) > 15 else ""
        raise ValueError(
            f"Ambiguous run_id {rid!r}: multiple directories start with it: {names}{more}. "
            "Pass the full folder name (e.g. with ' (1)')."
        )

    raise FileNotFoundError(
        f"No run directory under {runs_root} for run_id {rid!r}. "
        f"Expected {direct} or a single subdirectory starting with {rid!r}."
    )
