"""
Unified run output directories under ``<output_runs_root>/<resolved_run_name>/``.

Allocation picks ``<sanitized run_id>``, then ``<sanitized run_id> (1)``, ``(2)``, …
if the directory already exists. The same resolved path is reused for the lifetime
of the process (session) unless ``PIPELINE_RUN_OUTPUT_DIR`` or an explicit path is set.

Override: set env ``PIPELINE_RUN_OUTPUT_DIR`` to an absolute or user-expanded path
to pin the run directory (e.g. CI or resuming a specific folder).
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config.pipeline_config import output_runs_parent_from_pipeline, sanitize_run_id

_SESSION_RUN_DIR: Path | None = None

_ENV_RUN_OUTPUT = "PIPELINE_RUN_OUTPUT_DIR"
_MANIFEST_NAME = "run_manifest.json"


def allocate_unique_run_dir(runs_root: Path, logical_run_id: str) -> Path:
    """
    Create and return ``runs_root/<name>`` where ``name`` is the first free name among
    ``sanitize_run_id(logical_run_id)``, then ``name (1)``, ``name (2)``, …
    Writes :file:`run_manifest.json` when the directory is created.
    """
    runs_root = runs_root.expanduser().resolve()
    runs_root.mkdir(parents=True, exist_ok=True)
    base = sanitize_run_id(logical_run_id)
    candidate = runs_root / base
    if not candidate.exists():
        candidate.mkdir(parents=True, exist_ok=True)
        _write_run_manifest(candidate, logical_run_id=logical_run_id, resolved_directory=base)
        return candidate
    n = 1
    while True:
        name = f"{base} ({n})"
        candidate = runs_root / name
        if not candidate.exists():
            candidate.mkdir(parents=True, exist_ok=True)
            _write_run_manifest(candidate, logical_run_id=logical_run_id, resolved_directory=name)
            return candidate
        n += 1


def _write_run_manifest(
    run_dir: Path,
    *,
    logical_run_id: str,
    resolved_directory: str,
) -> None:
    payload = {
        "logical_run_id": logical_run_id,
        "resolved_directory": resolved_directory,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (run_dir / _MANIFEST_NAME).write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


def resolve_session_run_output_dir(
    cfg: dict[str, Any],
    *,
    project_root: Path | None = None,
    explicit_run_dir: str | Path | None = None,
    runs_root: str | Path | None = None,
) -> Path:
    """
    Return the run output directory for this process.

    Precedence:
    1. ``PIPELINE_RUN_OUTPUT_DIR`` environment variable (directory is created if missing).
    2. ``explicit_run_dir`` when non-empty.
    3. Cached session directory from an earlier call in this process.
    4. Allocate under ``runs_root`` (if given) or ``output_runs_parent_from_pipeline(cfg)``.
    """
    global _SESSION_RUN_DIR

    env = os.environ.get(_ENV_RUN_OUTPUT, "").strip()
    if env:
        p = Path(env).expanduser().resolve()
        p.mkdir(parents=True, exist_ok=True)
        _SESSION_RUN_DIR = p
        return p

    if explicit_run_dir is not None and str(explicit_run_dir).strip():
        p = Path(explicit_run_dir).expanduser().resolve()
        p.mkdir(parents=True, exist_ok=True)
        _SESSION_RUN_DIR = p
        return p

    if _SESSION_RUN_DIR is not None:
        return _SESSION_RUN_DIR

    if runs_root is not None and str(runs_root).strip():
        root = Path(str(runs_root).strip()).expanduser().resolve()
    else:
        root_str = output_runs_parent_from_pipeline(cfg, project_root=project_root)
        root = Path(root_str).resolve()

    logical = str(cfg.get("run_id") or "")
    allocated = allocate_unique_run_dir(root, logical)
    _SESSION_RUN_DIR = allocated
    return allocated


def reset_session_run_output_dir_for_tests() -> None:
    """Clear the process session (intended for tests only)."""
    global _SESSION_RUN_DIR
    _SESSION_RUN_DIR = None
