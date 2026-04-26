from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from analysis.utils import graph_structure_helpers as gh
from analysis.utils.config_run_fields import resolve_graph_id
from analysis.utils.anchor_graph_helpers import score_anchor_pairgraph_handcrafted


def run_anchor_graph_scoring_stage(config: dict[str, Any]) -> dict[str, Any]:
    """
    Score an existing unscored anchor graph edge table.
    """
    run_cfg = config.get("run") or {}
    scoring_cfg = config.get("scoring") or {}

    project_root = gh.find_project_root()
    graph_id = resolve_graph_id(run_cfg, default_if_missing="anchor_graph_run")
    anchor_output_root = Path(
        run_cfg.get("anchor_output_root")
        or (project_root / "analysis" / "output" / "anchor_graph")
    ).expanduser().resolve()
    run_dir = (anchor_output_root / graph_id).resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Anchor graph run directory not found: {run_dir}")

    p_unscored = run_dir / "anchor_graph_edges_unscored.csv"
    if not p_unscored.is_file():
        raise FileNotFoundError(f"Missing unscored anchor edges: {p_unscored}")
    unscored = pd.read_csv(p_unscored, low_memory=False)

    semantic_weight = float(scoring_cfg.get("semantic_weight", 0.45))
    infra_weight = float(scoring_cfg.get("infra_weight", 0.45))
    temporal_weight = float(scoring_cfg.get("temporal_weight", 0.1))
    score_mode = str(scoring_cfg.get("score_mode") or "anchor_handcrafted_v1")

    scored = score_anchor_pairgraph_handcrafted(
        unscored_df=unscored,
        semantic_weight=semantic_weight,
        infra_weight=infra_weight,
        temporal_weight=temporal_weight,
        score_mode=score_mode,
    )
    scored["email_a"] = scored["email_i"].astype(str)
    scored["email_b"] = scored["email_j"].astype(str)
    p_scored = run_dir / "anchor_graph_edges_weighted.csv"
    scored.to_csv(p_scored, index=False)

    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "graph_id": graph_id,
        "anchor_run_dir": str(run_dir),
        "input_unscored_csv": str(p_unscored),
        "output_scored_csv": str(p_scored),
        "score_mode": score_mode,
        "weights": {
            "semantic_weight": semantic_weight,
            "infra_weight": infra_weight,
            "temporal_weight": temporal_weight,
        },
        "n_edges": int(len(scored)),
    }
    p_summary = run_dir / "anchor_graph_scoring_summary.json"
    p_summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return {
        "output_dir": str(run_dir),
        "scored_edges_csv": str(p_scored),
        "summary_json": str(p_summary),
    }
