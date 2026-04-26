from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from analysis.utils import graph_structure_helpers as gh
from analysis.utils.config_run_fields import resolve_graph_id
from analysis.utils.anchor_seed_helpers import run_anchor_seed_generation
from analysis.utils.anchor_candidate_generation_helpers import run_anchor_candidate_generation
from analysis.utils.pair_graph_contract import (
    GRAPH_KIND_SEED_CANDIDATE,
    ensure_unscored_contract,
)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_config_path(project_root: Path, raw: str) -> Path:
    p = Path(str(raw)).expanduser()
    if not p.is_absolute():
        p = (project_root / p).resolve()
    else:
        p = p.resolve()
    if not p.is_file():
        raise FileNotFoundError(f"Config file not found: {p}")
    return p


def run_seed_candidate_graph_stage(config: dict[str, Any]) -> dict[str, Any]:
    """
    Unified stage: run seed generation and candidate generation, then emit a
    canonical unscored seed+candidate PairGraph.
    """
    run_cfg = config.get("run") or {}
    paths_cfg = config.get("paths") or {}
    out_cfg = config.get("output") or {}

    project_root = gh.find_project_root()
    graph_id = resolve_graph_id(run_cfg, default_if_missing="anchor_graph_run")

    p_seed_cfg = _resolve_config_path(project_root, str(paths_cfg.get("seed_config_path") or "analysis/configs/anchor_seed.default.json"))
    p_cand_cfg = _resolve_config_path(project_root, str(paths_cfg.get("candidate_config_path") or "analysis/configs/anchor_candidate_generation.default.json"))

    seed_cfg = _load_json(p_seed_cfg)
    cand_cfg = _load_json(p_cand_cfg)
    seed_cfg.setdefault("run", {})
    seed_cfg["run"]["graph_id"] = graph_id
    cand_cfg.setdefault("run", {})
    cand_cfg["run"]["graph_id"] = graph_id

    seed_res = run_anchor_seed_generation(seed_cfg)
    # Ensure candidate stage consumes exactly the seed stage produced in this unified run.
    cand_cfg["run"]["seed_stage_dir"] = str(seed_res["output_dir"])
    cand_res = run_anchor_candidate_generation(cand_cfg)

    p_union = Path(str(cand_res["candidate_union_csv"])).expanduser().resolve()
    union = pd.read_csv(p_union, low_memory=False)
    if union.empty:
        pair_df = pd.DataFrame(
            {
                "email_i": pd.Series(dtype=str),
                "email_j": pd.Series(dtype=str),
                "graph_kind": pd.Series(dtype=str),
                "graph_run_id": pd.Series(dtype=str),
                "from_seed": pd.Series(dtype=bool),
                "from_semantic": pd.Series(dtype=bool),
                "from_rare_artifact": pd.Series(dtype=bool),
                "from_component": pd.Series(dtype=bool),
                "from_2hop": pd.Series(dtype=bool),
                "source_count": pd.Series(dtype=int),
            }
        )
    else:
        pair_df = union.copy()
        pair_df["graph_kind"] = GRAPH_KIND_SEED_CANDIDATE
        pair_df["graph_run_id"] = graph_id
        # Canonical names.
        pair_df["from_seed"] = pair_df.get("from_seed", False)
        pair_df["from_semantic"] = pair_df.get("from_semantic", False)
        pair_df["from_rare_artifact"] = pair_df.get("from_rare_artifact", False)
        pair_df["from_component"] = pair_df.get("from_component", False)
        pair_df["from_2hop"] = pair_df.get("from_2hop", False)
    pair_df = ensure_unscored_contract(pair_df)

    out_parent = str(out_cfg.get("output_parent_dir") or "").strip()
    if out_parent:
        out_root = Path(out_parent).expanduser()
        if not out_root.is_absolute():
            out_root = (project_root / out_root).resolve()
        else:
            out_root = out_root.resolve()
    else:
        out_root = (project_root / "analysis" / "output" / "seed_candidate_graph").resolve()
    stage_name = str(out_cfg.get("stage_name") or "seed_candidate_graph")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = (out_root / graph_id / f"{stage_name}_{stamp}").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    p_pairgraph = out_dir / "seed_candidate_pairgraph_unscored.csv"
    pair_df.to_csv(p_pairgraph, index=False)

    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "graph_id": graph_id,
        "seed_config_path": str(p_seed_cfg),
        "candidate_config_path": str(p_cand_cfg),
        "seed_output_dir": seed_res.get("output_dir"),
        "candidate_output_dir": cand_res.get("output_dir"),
        "candidate_union_csv": str(p_union),
        "pairgraph_unscored_csv": str(p_pairgraph),
        "n_pairs": int(len(pair_df)),
    }
    p_summary = out_dir / "seed_candidate_graph_summary.json"
    p_summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return {
        "output_dir": str(out_dir),
        "pairgraph_unscored_csv": str(p_pairgraph),
        "summary_json": str(p_summary),
        "seed_stage_output_dir": seed_res.get("output_dir"),
        "candidate_stage_output_dir": cand_res.get("output_dir"),
    }
