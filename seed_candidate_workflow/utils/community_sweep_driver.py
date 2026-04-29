from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd


def gt_slug(path: str | Path) -> str:
    p = Path(path)
    stem = p.stem or "ground_truth"
    return "".join(ch if (ch.isalnum() or ch in {"_", "-"}) else "_" for ch in stem)


def run_multi_gt_sweep(
    *,
    gt_paths: list[str],
    per_gt_sweep: Callable[[str], tuple[pd.DataFrame, dict[str, Any]]],
    write_per_gt: Callable[[str, pd.DataFrame, dict[str, Any]], dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    per_gt_outputs: list[dict[str, Any]] = []
    best_rows_by_gt: list[dict[str, Any]] = []
    for gt_path in gt_paths:
        sweep_df, best_info = per_gt_sweep(gt_path)
        out = write_per_gt(gt_path, sweep_df, best_info)
        per_gt_outputs.append(out)
        br = dict(best_info.get("best_row") or {})
        if br:
            best_rows_by_gt.append({"gt_path": str(gt_path), "gt_slug": gt_slug(gt_path), **br})
    return per_gt_outputs, best_rows_by_gt
