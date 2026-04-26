"""Shared resolution of run-scoped config fields (graph identity, etc.)."""

from __future__ import annotations

from typing import Any, Mapping


def resolve_graph_id(run_cfg: Mapping[str, Any] | None) -> str:
    """Canonical graph bundle / anchor-run directory name (``run.graph_id`` or experiment block)."""
    r = dict(run_cfg or {})
    v = str(r.get("graph_id") or "").strip()
    if v:
        return v
    raise ValueError("graph_id is required (e.g. run.graph_id in stage JSON or experiment.graph_id)")


def resolve_scoring_run_id(exp_cfg: Mapping[str, Any] | None) -> str:
    """Folder name under ``artifacts.scoring_output_root`` (``experiment.scoring_run_id``)."""
    e = dict(exp_cfg or {})
    v = str(e.get("scoring_run_id") or "").strip()
    if v:
        return v
    raise ValueError("experiment.scoring_run_id is required")
