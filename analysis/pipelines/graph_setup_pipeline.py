from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from analysis.utils.anchor_candidate_generation_helpers import run_anchor_candidate_generation
from analysis.utils.anchor_graph_helpers import build_anchor_graph
from analysis.utils.anchor_seed_helpers import run_anchor_seed_generation
from analysis.utils.pair_graph_contract import GRAPH_KIND_SEED_CANDIDATE, ensure_unscored_contract
from analysis.utils.pair_training_dataset_helpers import build_pair_training_dataset


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_path(project_root: Path, raw: str | Path) -> Path:
    p = Path(str(raw)).expanduser()
    if not p.is_absolute():
        p = (project_root / p).resolve()
    else:
        p = p.resolve()
    return p


def _resolve_latest_stage_dir(stage_root: Path, stage_prefix: str) -> Path:
    if not stage_root.is_dir():
        raise FileNotFoundError(f"Stage root not found: {stage_root}")
    dirs = [p for p in stage_root.iterdir() if p.is_dir() and p.name.startswith(stage_prefix)]
    if not dirs:
        raise FileNotFoundError(f"No stage directories with prefix {stage_prefix!r} under {stage_root}")
    return max(dirs, key=lambda p: p.stat().st_mtime)


def _resolve_bundle_paths(*, graph_bundle_root: Path, graph_id: str) -> dict[str, Path]:
    bundle_root = (graph_bundle_root / graph_id).resolve()
    return {
        "bundle_root": bundle_root,
        "anchor_root": bundle_root / "anchor",
        "seed_root": bundle_root / "seed",
        "candidate_root": bundle_root / "candidate",
        "seed_candidate_root": bundle_root / "seed_candidate",
        "pair_training_root": bundle_root / "pair_training",
    }


def _build_seed_candidate_pairgraph(
    *,
    candidate_union_csv: Path,
    graph_id: str,
    out_dir: Path,
) -> dict[str, str]:
    union = pd.read_csv(candidate_union_csv, low_memory=False)
    if union.empty:
        pair_df = pd.DataFrame(
            {
                "email_i": pd.Series(dtype=str),
                "email_j": pd.Series(dtype=str),
                "graph_kind": pd.Series(dtype=str),
                "graph_id": pd.Series(dtype=str),
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
        pair_df["graph_id"] = graph_id
        pair_df["from_seed"] = pair_df.get("from_seed", False)
        pair_df["from_semantic"] = pair_df.get("from_semantic", False)
        pair_df["from_rare_artifact"] = pair_df.get("from_rare_artifact", False)
        pair_df["from_component"] = pair_df.get("from_component", False)
        pair_df["from_2hop"] = pair_df.get("from_2hop", False)
    pair_df = ensure_unscored_contract(pair_df)

    out_dir.mkdir(parents=True, exist_ok=True)
    p_pair = out_dir / "seed_candidate_pairgraph_unscored.csv"
    p_summary = out_dir / "seed_candidate_graph_summary.json"
    pair_df.to_csv(p_pair, index=False)
    p_summary.write_text(
        json.dumps(
            {
                "graph_id": graph_id,
                "candidate_union_csv": str(candidate_union_csv),
                "pairgraph_unscored_csv": str(p_pair),
                "n_pairs": int(len(pair_df)),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return {
        "pairgraph_unscored_csv": str(p_pair),
        "summary_json": str(p_summary),
    }


def run_graph_setup(
    *,
    project_root: Path,
    graph_id: str,
    graph_bundle_root: Path,
    setup_cfg: dict[str, Any],
) -> dict[str, Any]:
    enable_cfg = dict(setup_cfg.get("enable") or {})
    policy_cfg = dict(setup_cfg.get("policy") or {})
    paths_cfg = dict(setup_cfg.get("paths") or {})
    pair_training_cfg = dict(paths_cfg.get("pair_training") or {})

    on_missing = str(policy_cfg.get("on_missing") or "build").strip().lower()
    on_present = str(policy_cfg.get("on_present") or "reuse").strip().lower()
    if on_missing not in {"build", "fail"}:
        raise ValueError(f"Unsupported setup.policy.on_missing={on_missing!r}")
    if on_present not in {"reuse", "rebuild"}:
        raise ValueError(f"Unsupported setup.policy.on_present={on_present!r}")

    bundle = _resolve_bundle_paths(graph_bundle_root=graph_bundle_root, graph_id=graph_id)
    for p in bundle.values():
        p.mkdir(parents=True, exist_ok=True)

    anchor_cfg_path = _resolve_path(project_root, paths_cfg.get("anchor_config") or "analysis/configs/anchor_graph.default.json")
    seed_cfg_path = _resolve_path(project_root, paths_cfg.get("seed_config") or "analysis/configs/anchor_seed.default.json")
    candidate_cfg_path = _resolve_path(project_root, paths_cfg.get("candidate_config") or "analysis/configs/anchor_candidate_generation.default.json")

    anchor_run_dir = bundle["anchor_root"] / graph_id
    anchor_edges_csv = anchor_run_dir / "anchor_graph_edges_unscored.csv"
    anchor_exists = anchor_edges_csv.is_file()

    seed_stage_prefix = "seed_generation_"
    candidate_stage_prefix = "candidate_generation_"
    seed_stage_dir = None
    candidate_stage_dir = None

    component_actions: dict[str, str] = {}
    outputs: dict[str, Any] = {}

    # Anchor
    anchor_enabled = bool(enable_cfg.get("anchor", True))
    if not anchor_enabled:
        if not anchor_exists:
            raise FileNotFoundError("setup.enable.anchor=false but anchor artifact is missing")
        component_actions["anchor"] = "reused_existing"
    else:
        should_build_anchor = (not anchor_exists and on_missing == "build") or (anchor_exists and on_present == "rebuild")
        if not anchor_exists and on_missing == "fail":
            raise FileNotFoundError(f"Anchor artifact missing and on_missing=fail: {anchor_edges_csv}")
        if should_build_anchor:
            anchor_cfg = _read_json(anchor_cfg_path)
            anchor_cfg.setdefault("run", {})
            anchor_cfg["run"]["graph_id"] = graph_id
            anchor_cfg.setdefault("persistence", {})
            anchor_cfg["persistence"]["output_dir"] = str(bundle["anchor_root"])
            outputs["anchor"] = build_anchor_graph(anchor_cfg)
            component_actions["anchor"] = "built"
        else:
            component_actions["anchor"] = "reused_existing"

    # Seed
    seed_enabled = bool(enable_cfg.get("seed", True))
    if seed_enabled:
        seed_cfg = _read_json(seed_cfg_path)
        seed_cfg.setdefault("run", {})
        seed_cfg["run"]["graph_id"] = graph_id
        seed_cfg["run"]["anchor_output_root"] = str(bundle["anchor_root"])
        seed_cfg.setdefault("output", {})
        seed_cfg["output"]["output_root"] = str(bundle["seed_root"])
        seed_stage_prefix = str(seed_cfg["output"].get("stage_name") or "seed_generation") + "_"
        existing_seed = None
        try:
            existing_seed = _resolve_latest_stage_dir(bundle["seed_root"] / graph_id, seed_stage_prefix)
        except FileNotFoundError:
            existing_seed = None
        if existing_seed is None and on_missing == "fail":
            raise FileNotFoundError("Seed stage missing and on_missing=fail")
        should_build_seed = (existing_seed is None and on_missing == "build") or (existing_seed is not None and on_present == "rebuild")
        if should_build_seed:
            outputs["seed"] = run_anchor_seed_generation(seed_cfg)
            seed_stage_dir = Path(str(outputs["seed"]["output_dir"])).resolve()
            component_actions["seed"] = "built"
        else:
            seed_stage_dir = existing_seed
            component_actions["seed"] = "reused_existing"
    else:
        seed_stage_dir = _resolve_latest_stage_dir(bundle["seed_root"] / graph_id, seed_stage_prefix)
        component_actions["seed"] = "reused_existing"

    # Candidate
    candidate_enabled = bool(enable_cfg.get("candidate", True))
    if candidate_enabled:
        candidate_cfg = _read_json(candidate_cfg_path)
        candidate_cfg.setdefault("run", {})
        candidate_cfg["run"]["graph_id"] = graph_id
        candidate_cfg["run"]["anchor_output_root"] = str(bundle["anchor_root"])
        candidate_cfg["run"]["seed_output_root"] = str(bundle["seed_root"])
        candidate_cfg["run"]["seed_stage_dir"] = str(seed_stage_dir)
        candidate_cfg.setdefault("output", {})
        candidate_cfg["output"]["output_root"] = str(bundle["candidate_root"])
        candidate_stage_prefix = str(candidate_cfg["output"].get("stage_name") or "candidate_generation") + "_"
        existing_cand = None
        try:
            existing_cand = _resolve_latest_stage_dir(bundle["candidate_root"] / graph_id, candidate_stage_prefix)
        except FileNotFoundError:
            existing_cand = None
        if existing_cand is None and on_missing == "fail":
            raise FileNotFoundError("Candidate stage missing and on_missing=fail")
        should_build_candidate = (existing_cand is None and on_missing == "build") or (existing_cand is not None and on_present == "rebuild")
        if should_build_candidate:
            outputs["candidate"] = run_anchor_candidate_generation(candidate_cfg)
            candidate_stage_dir = Path(str(outputs["candidate"]["output_dir"])).resolve()
            component_actions["candidate"] = "built"
        else:
            candidate_stage_dir = existing_cand
            component_actions["candidate"] = "reused_existing"
    else:
        candidate_stage_dir = _resolve_latest_stage_dir(bundle["candidate_root"] / graph_id, candidate_stage_prefix)
        component_actions["candidate"] = "reused_existing"

    p_candidate_union = candidate_stage_dir / "candidate_union.csv"
    p_seed_all = seed_stage_dir / "seed_edges_all.csv"
    if not p_candidate_union.is_file():
        raise FileNotFoundError(f"Missing candidate_union.csv: {p_candidate_union}")
    if not p_seed_all.is_file():
        raise FileNotFoundError(f"Missing seed_edges_all.csv: {p_seed_all}")

    # Seed-candidate pairgraph
    seed_candidate_dir = bundle["seed_candidate_root"] / graph_id
    p_seed_candidate = seed_candidate_dir / "seed_candidate_pairgraph_unscored.csv"
    seed_candidate_enabled = bool(enable_cfg.get("seed_candidate", True))
    if seed_candidate_enabled:
        if p_seed_candidate.is_file() and on_present == "reuse":
            component_actions["seed_candidate"] = "reused_existing"
        else:
            outputs["seed_candidate"] = _build_seed_candidate_pairgraph(
                candidate_union_csv=p_candidate_union,
                graph_id=graph_id,
                out_dir=seed_candidate_dir,
            )
            component_actions["seed_candidate"] = "built"
    else:
        if not p_seed_candidate.is_file():
            raise FileNotFoundError("setup.enable.seed_candidate=false but seed_candidate pairgraph is missing")
        component_actions["seed_candidate"] = "reused_existing"

    # Pair training
    pair_enabled = bool(enable_cfg.get("pair_training", True))
    p_pair_training = bundle["pair_training_root"] / graph_id / "pair_training_dataset.csv"
    if pair_enabled:
        if p_pair_training.is_file() and on_present == "reuse":
            component_actions["pair_training"] = "reused_existing"
        else:
            p_graph_meta = pair_training_cfg.get("graph_meta_json")
            graph_meta_path = _resolve_path(project_root, p_graph_meta) if p_graph_meta else None
            pt_out_dir = bundle["pair_training_root"] / graph_id
            pt_out_dir.mkdir(parents=True, exist_ok=True)
            outputs["pair_training"] = build_pair_training_dataset(
                seed_edges_all_csv=p_seed_all,
                candidate_union_csv=p_candidate_union,
                output_dir=pt_out_dir,
                graph_meta_json=graph_meta_path,
                graph_id=graph_id,
                project_root=project_root,
            )
            component_actions["pair_training"] = "built"
    else:
        if not p_pair_training.is_file():
            raise FileNotFoundError("setup.enable.pair_training=false but pair_training_dataset.csv is missing")
        component_actions["pair_training"] = "reused_existing"

    return {
        "graph_id": graph_id,
        "bundle_root": str(bundle["bundle_root"]),
        "paths": {
            "anchor_run_dir": str(anchor_run_dir),
            "seed_stage_dir": str(seed_stage_dir),
            "candidate_stage_dir": str(candidate_stage_dir),
            "seed_candidate_pairgraph_csv": str(p_seed_candidate),
            "pair_training_dataset_csv": str(p_pair_training),
            "candidate_union_csv": str(p_candidate_union),
            "seed_edges_all_csv": str(p_seed_all),
        },
        "actions": component_actions,
        "outputs": outputs,
    }

