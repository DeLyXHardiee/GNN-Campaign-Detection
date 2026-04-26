"""Shared resolution of run-scoped config fields (graph identity, etc.)."""

from __future__ import annotations

from typing import Any, Mapping


def resolve_graph_id(
    run_cfg: Mapping[str, Any] | None,
    *,
    default_if_missing: str | None = None,
) -> str:
    """
    Canonical graph bundle / anchor-run directory name.

    Prefer ``run.graph_id``. ``run.graph_run_id`` is accepted for older configs only.
    """
    r = dict(run_cfg or {})
    v = str(r.get("graph_id") or r.get("graph_run_id") or "").strip()
    if v:
        return v
    if default_if_missing is not None:
        d = str(default_if_missing).strip()
        if d:
            return d
    raise ValueError("run.graph_id is required (legacy run.graph_run_id is also accepted)")


def resolve_scoring_run_id(
    exp_cfg: Mapping[str, Any] | None,
    *,
    default_if_missing: str | None = None,
) -> str:
    """
    Folder name under ``artifacts.scoring_output_root`` for this scoring/community run.

    Prefer ``experiment.scoring_run_id``. ``experiment.run_id`` is accepted for older configs only.
    """
    e = dict(exp_cfg or {})
    v = str(e.get("scoring_run_id") or e.get("run_id") or "").strip()
    if v:
        return v
    if default_if_missing is not None:
        d = str(default_if_missing).strip()
        if d:
            return d
    raise ValueError(
        "experiment.scoring_run_id is required (legacy experiment.run_id is also accepted)"
    )
