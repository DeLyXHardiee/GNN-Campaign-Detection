"""Shared path rules for the GNN stage pipeline."""

from __future__ import annotations

import re
from pathlib import Path

_RUN_ID_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]{0,127}$")


def sanitize_run_id(run_id: str) -> str:
    """
    ``run_id`` is the folder name under RUNS_PARENT. Restricted so paths stay predictable
    and safe (no ``..`` or absolute paths sneaking in via config).
    """
    s = (run_id or "").strip()
    if not s:
        raise ValueError(
            "Set 'run_id' in pipeline_config.json to a unique experiment name "
            "(e.g. 'sage_email_v1'). All stages read/write <RUNS_PARENT>/<run_id>/."
        )
    if not _RUN_ID_RE.match(s):
        raise ValueError(
            f"Invalid run_id {s!r}: use only letters, digits, '.', '_', '-' "
            "(max 128 chars, must start with a letter or digit)."
        )
    return s


def run_dir_for(runs_parent: str | Path, run_id: str) -> Path:
    return Path(runs_parent).expanduser() / sanitize_run_id(run_id)
