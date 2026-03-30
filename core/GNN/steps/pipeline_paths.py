"""Shared path rules for the GNN stage pipeline."""

from __future__ import annotations

# Canonical definitions live in ``core/config/pipeline_config.py`` (importable as ``config.*``).
from config.pipeline_config import run_dir_for, sanitize_run_id

__all__ = ["sanitize_run_id", "run_dir_for"]
