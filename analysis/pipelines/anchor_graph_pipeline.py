"""Analysis-only pipeline for anchor graph stages."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.utils.anchor_graph_community_helpers import run_anchor_multi_gt_community_sweep
from analysis.utils.anchor_graph_helpers import build_anchor_graph
from analysis.utils.anchor_seed_helpers import run_anchor_seed_generation
from analysis.utils.anchor_candidate_generation_helpers import run_anchor_candidate_generation
from analysis.utils.anchor_scored_clustering_helpers import run_anchor_scored_clustering_stage
from analysis.utils.anchor_pu_scored_clustering_helpers import run_anchor_pu_scored_clustering_stage


STAGE_BUILD_ANCHOR_GRAPH = "build_anchor_graph"
STAGE_ANCHOR_SEED_GENERATION = "anchor_seed_generation"
STAGE_ANCHOR_COMMUNITY_SWEEP = "anchor_community_sweep"
STAGE_ANCHOR_CANDIDATE_GENERATION = "anchor_candidate_generation"
STAGE_ANCHOR_SCORED_CLUSTERING = "anchor_scored_clustering"
STAGE_ANCHOR_PU_SCORED_CLUSTERING = "anchor_pu_scored_clustering"
DEFAULT_BUILD_CONFIG_PATH = (
    PROJECT_ROOT / "analysis" / "configs" / "anchor_graph.default.json"
)
DEFAULT_SEED_CONFIG_PATH = (
    PROJECT_ROOT / "analysis" / "configs" / "anchor_seed.default.json"
)
DEFAULT_CANDIDATE_CONFIG_PATH = (
    PROJECT_ROOT / "analysis" / "configs" / "anchor_candidate_generation.default.json"
)
DEFAULT_COMMUNITY_CONFIG_PATH = (
    PROJECT_ROOT / "analysis" / "configs" / "anchor_community.default.json"
)
DEFAULT_SCORED_CLUSTERING_CONFIG_PATH = (
    PROJECT_ROOT / "analysis" / "configs" / "anchor_scored_clustering.default.json"
)
DEFAULT_PU_SCORED_CLUSTERING_CONFIG_PATH = (
    PROJECT_ROOT / "analysis" / "configs" / "anchor_pu_scored_clustering.default.json"
)
CONFIG_PRESETS: dict[str, Path] = {
    "default": DEFAULT_BUILD_CONFIG_PATH,
    "build_default": DEFAULT_BUILD_CONFIG_PATH,
    "seed_default": DEFAULT_SEED_CONFIG_PATH,
    "candidate_default": DEFAULT_CANDIDATE_CONFIG_PATH,
    "community_default": DEFAULT_COMMUNITY_CONFIG_PATH,
    "scored_clustering_default": DEFAULT_SCORED_CLUSTERING_CONFIG_PATH,
    "pu_scored_clustering_default": DEFAULT_PU_SCORED_CLUSTERING_CONFIG_PATH,
}


def _load_pipeline_config(config_path: Path | None = None) -> dict[str, object]:
    p = (config_path or DEFAULT_BUILD_CONFIG_PATH).expanduser().resolve()
    if not p.is_file():
        raise SystemExit(f"Config not found: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def run_stage_build_anchor_graph(
    *,
    config_path: Path | None = None,
) -> dict[str, object]:
    cfg = _load_pipeline_config(config_path)
    result = run_anchor_graph_pipeline(cfg, stages=[STAGE_BUILD_ANCHOR_GRAPH])
    stage_res = result.get(STAGE_BUILD_ANCHOR_GRAPH)
    if isinstance(stage_res, dict):
        print("Wrote:", stage_res.get("paths"))
        print("Validation:", (stage_res.get("summary") or {}).get("validation"))
    return result


def run_stage_anchor_community_sweep(
    *,
    config_path: Path | None = None,
) -> dict[str, object]:
    cfg = _load_pipeline_config(config_path or DEFAULT_COMMUNITY_CONFIG_PATH)
    result = run_anchor_graph_pipeline(cfg, stages=[STAGE_ANCHOR_COMMUNITY_SWEEP])
    stage_res = result.get(STAGE_ANCHOR_COMMUNITY_SWEEP)
    if isinstance(stage_res, dict):
        print("Output dir:", stage_res.get("output_dir"))
        print("Summary:", stage_res.get("summary_json"))
    return result


def run_stage_anchor_seed_generation(
    *,
    config_path: Path | None = None,
) -> dict[str, object]:
    cfg = _load_pipeline_config(config_path or DEFAULT_SEED_CONFIG_PATH)
    result = run_anchor_graph_pipeline(cfg, stages=[STAGE_ANCHOR_SEED_GENERATION])
    stage_res = result.get(STAGE_ANCHOR_SEED_GENERATION)
    if isinstance(stage_res, dict):
        print("Output dir:", stage_res.get("output_dir"))
        print("Seeds:", stage_res.get("seed_edges_hard_csv"))
        print("Summary:", stage_res.get("summary_json"))
    return result


def run_stage_anchor_candidate_generation(
    *,
    config_path: Path | None = None,
) -> dict[str, object]:
    cfg = _load_pipeline_config(config_path or DEFAULT_CANDIDATE_CONFIG_PATH)
    result = run_anchor_graph_pipeline(cfg, stages=[STAGE_ANCHOR_CANDIDATE_GENERATION])
    stage_res = result.get(STAGE_ANCHOR_CANDIDATE_GENERATION)
    if isinstance(stage_res, dict):
        print("Output dir:", stage_res.get("output_dir"))
        print("Summary:", stage_res.get("summary_json"))
    return result


def run_stage_anchor_pu_scored_clustering(
    *,
    config_path: Path | None = None,
) -> dict[str, object]:
    pu_path = (config_path or DEFAULT_PU_SCORED_CLUSTERING_CONFIG_PATH).expanduser().resolve()
    cfg = _load_pipeline_config(pu_path)
    cfg["_pipeline_config_path"] = str(pu_path)
    result = run_anchor_graph_pipeline(
        config_by_stage={STAGE_ANCHOR_PU_SCORED_CLUSTERING: cfg},
        stages=[STAGE_ANCHOR_PU_SCORED_CLUSTERING],
    )
    stage_res = result.get(STAGE_ANCHOR_PU_SCORED_CLUSTERING)
    if isinstance(stage_res, dict):
        print("Bundle dir:", stage_res.get("output_dir"))
        print("Artifact parent:", stage_res.get("artifact_parent_dir"))
        print("Graph summary:", stage_res.get("pu_scored_graph_summary_json"))
        print("Eval summary:", stage_res.get("pu_scored_graph_eval_summary_json"))
        print("Community sweep:", stage_res.get("community_sweep_output_dir"))
        print("PU threshold retention:", stage_res.get("pu_threshold_retention_summary_json"))
    return result


def run_stage_anchor_scored_clustering(
    *,
    config_path: Path | None = None,
) -> dict[str, object]:
    sc_path = (config_path or DEFAULT_SCORED_CLUSTERING_CONFIG_PATH).expanduser().resolve()
    cfg = _load_pipeline_config(sc_path)
    cfg["_pipeline_config_path"] = str(sc_path)
    result = run_anchor_graph_pipeline(config_by_stage={STAGE_ANCHOR_SCORED_CLUSTERING: cfg}, stages=[STAGE_ANCHOR_SCORED_CLUSTERING])
    stage_res = result.get(STAGE_ANCHOR_SCORED_CLUSTERING)
    if isinstance(stage_res, dict):
        print("Config:", stage_res.get("pipeline_config_path"))
        print("Scored edges:", stage_res.get("scored_clustering_edges_csv"))
        print("Graph summary:", stage_res.get("scored_clustering_graph_summary_json"))
        print("Community sweep:", stage_res.get("community_sweep_output_dir"))
    return result


def run_pipeline(
    *,
    build_config_path: Path | None = None,
    seed_config_path: Path | None = None,
    community_config_path: Path | None = None,
    candidate_config_path: Path | None = None,
) -> dict[str, object]:
    configs = {
        STAGE_BUILD_ANCHOR_GRAPH: _load_pipeline_config(
            build_config_path or DEFAULT_BUILD_CONFIG_PATH
        ),
        STAGE_ANCHOR_SEED_GENERATION: _load_pipeline_config(
            seed_config_path or DEFAULT_SEED_CONFIG_PATH
        ),
        STAGE_ANCHOR_COMMUNITY_SWEEP: _load_pipeline_config(
            community_config_path or DEFAULT_COMMUNITY_CONFIG_PATH
        ),
        STAGE_ANCHOR_CANDIDATE_GENERATION: _load_pipeline_config(
            candidate_config_path or DEFAULT_CANDIDATE_CONFIG_PATH
        ),
    }
    return run_anchor_graph_pipeline(
        config_by_stage=configs,
        stages=[STAGE_BUILD_ANCHOR_GRAPH, STAGE_ANCHOR_SEED_GENERATION],
    )


def run_anchor_graph_pipeline(
    config: dict | None = None,
    *,
    config_by_stage: dict[str, dict] | None = None,
    stages: list[str] | None = None,
) -> dict[str, object]:
    requested = stages or [STAGE_BUILD_ANCHOR_GRAPH]
    stage_cfg = dict(config_by_stage or {})
    if config is not None:
        stage_cfg.setdefault(STAGE_BUILD_ANCHOR_GRAPH, config)
    out: dict[str, object] = {"stages_run": []}
    for stage in requested:
        s = str(stage).strip().lower()
        if s == STAGE_BUILD_ANCHOR_GRAPH:
            cfg = stage_cfg.get(STAGE_BUILD_ANCHOR_GRAPH)
            if not isinstance(cfg, dict):
                raise ValueError("Missing build-anchor-graph config")
            res = build_anchor_graph(cfg)
            out[STAGE_BUILD_ANCHOR_GRAPH] = res
            out["stages_run"].append(STAGE_BUILD_ANCHOR_GRAPH)
        elif s == STAGE_ANCHOR_SEED_GENERATION:
            cfg = stage_cfg.get(STAGE_ANCHOR_SEED_GENERATION)
            if not isinstance(cfg, dict):
                raise ValueError("Missing anchor-seed-generation config")
            res = run_anchor_seed_generation(cfg)
            out[STAGE_ANCHOR_SEED_GENERATION] = res
            out["stages_run"].append(STAGE_ANCHOR_SEED_GENERATION)
        elif s == STAGE_ANCHOR_COMMUNITY_SWEEP:
            cfg = stage_cfg.get(STAGE_ANCHOR_COMMUNITY_SWEEP)
            if not isinstance(cfg, dict):
                raise ValueError("Missing anchor-community-sweep config")
            res = run_anchor_multi_gt_community_sweep(cfg)
            out[STAGE_ANCHOR_COMMUNITY_SWEEP] = res
            out["stages_run"].append(STAGE_ANCHOR_COMMUNITY_SWEEP)
        elif s == STAGE_ANCHOR_CANDIDATE_GENERATION:
            cfg = stage_cfg.get(STAGE_ANCHOR_CANDIDATE_GENERATION)
            if not isinstance(cfg, dict):
                raise ValueError("Missing anchor-candidate-generation config")
            res = run_anchor_candidate_generation(cfg)
            out[STAGE_ANCHOR_CANDIDATE_GENERATION] = res
            out["stages_run"].append(STAGE_ANCHOR_CANDIDATE_GENERATION)
        elif s == STAGE_ANCHOR_SCORED_CLUSTERING:
            cfg = stage_cfg.get(STAGE_ANCHOR_SCORED_CLUSTERING)
            if not isinstance(cfg, dict):
                raise ValueError("Missing anchor-scored-clustering config")
            res = run_anchor_scored_clustering_stage(cfg)
            out[STAGE_ANCHOR_SCORED_CLUSTERING] = res
            out["stages_run"].append(STAGE_ANCHOR_SCORED_CLUSTERING)
        elif s == STAGE_ANCHOR_PU_SCORED_CLUSTERING:
            cfg = stage_cfg.get(STAGE_ANCHOR_PU_SCORED_CLUSTERING)
            if not isinstance(cfg, dict):
                raise ValueError("Missing anchor-pu-scored-clustering config")
            res = run_anchor_pu_scored_clustering_stage(cfg)
            out[STAGE_ANCHOR_PU_SCORED_CLUSTERING] = res
            out["stages_run"].append(STAGE_ANCHOR_PU_SCORED_CLUSTERING)
        else:
            raise ValueError(
                f"Unsupported stage: {stage!r}. "
                f"Supported stages: {STAGE_BUILD_ANCHOR_GRAPH}, {STAGE_ANCHOR_SEED_GENERATION}, {STAGE_ANCHOR_CANDIDATE_GENERATION}, {STAGE_ANCHOR_COMMUNITY_SWEEP}, {STAGE_ANCHOR_SCORED_CLUSTERING}, {STAGE_ANCHOR_PU_SCORED_CLUSTERING}"
            )
    return out


def main_cli() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_BUILD_CONFIG_PATH,
        help="Backward-compatible single config path (used for build stage).",
    )
    p.add_argument(
        "--build-config",
        type=Path,
        default=DEFAULT_BUILD_CONFIG_PATH,
        help="Path to anchor graph build config JSON.",
    )
    p.add_argument(
        "--community-config",
        type=Path,
        default=DEFAULT_COMMUNITY_CONFIG_PATH,
        help="Path to anchor community sweep config JSON.",
    )
    p.add_argument(
        "--seed-config",
        type=Path,
        default=DEFAULT_SEED_CONFIG_PATH,
        help="Path to anchor seed generation config JSON.",
    )
    p.add_argument(
        "--candidate-config",
        type=Path,
        default=DEFAULT_CANDIDATE_CONFIG_PATH,
        help="Path to anchor candidate generation config JSON.",
    )
    p.add_argument(
        "--stages",
        type=str,
        default=STAGE_BUILD_ANCHOR_GRAPH,
        help="Comma-separated stages: build_anchor_graph,anchor_seed_generation,anchor_candidate_generation,anchor_community_sweep,anchor_scored_clustering,anchor_pu_scored_clustering",
    )
    p.add_argument(
        "--pu-scored-clustering-config",
        type=Path,
        default=DEFAULT_PU_SCORED_CLUSTERING_CONFIG_PATH,
        help="Path to PU-scored clustering + community bridge config JSON.",
    )
    p.add_argument(
        "--scored-clustering-config",
        type=Path,
        default=DEFAULT_SCORED_CLUSTERING_CONFIG_PATH,
        help=(
            "Path to scored clustering graph + community bridge config JSON "
            "(default: analysis/configs/anchor_scored_clustering.default.json under project root)."
        ),
    )
    p.add_argument(
        "--preset",
        type=str,
        default="",
        help="Optional config preset name (see CONFIG_PRESETS in anchor_graph_pipeline.py).",
    )
    args = p.parse_args()

    build_cfg_path = args.build_config
    seed_cfg_path = args.seed_config
    community_cfg_path = args.community_config
    candidate_cfg_path = getattr(args, "candidate_config", None)
    scored_clustering_cfg_path = args.scored_clustering_config
    pu_scored_clustering_cfg_path = args.pu_scored_clustering_config
    cfg_path = args.config
    if str(args.preset or "").strip():
        key = str(args.preset).strip().lower()
        if key not in CONFIG_PRESETS:
            raise SystemExit(
                f"Unknown preset {args.preset!r}. Available: {', '.join(sorted(CONFIG_PRESETS))}"
            )
        cfg_path = CONFIG_PRESETS[key]
        if key in {"default", "build_default"}:
            build_cfg_path = cfg_path
        elif key == "seed_default":
            seed_cfg_path = cfg_path
        elif key == "candidate_default":
            candidate_cfg_path = cfg_path
        elif key == "community_default":
            community_cfg_path = cfg_path
        elif key == "scored_clustering_default":
            scored_clustering_cfg_path = cfg_path
        elif key == "pu_scored_clustering_default":
            pu_scored_clustering_cfg_path = cfg_path

    stages = [x.strip() for x in str(args.stages).split(",") if x.strip()]
    configs: dict[str, dict] = {}
    if STAGE_BUILD_ANCHOR_GRAPH in stages:
        configs[STAGE_BUILD_ANCHOR_GRAPH] = _load_pipeline_config(build_cfg_path)
    if STAGE_ANCHOR_SEED_GENERATION in stages:
        configs[STAGE_ANCHOR_SEED_GENERATION] = _load_pipeline_config(seed_cfg_path)
    if STAGE_ANCHOR_CANDIDATE_GENERATION in stages:
        configs[STAGE_ANCHOR_CANDIDATE_GENERATION] = _load_pipeline_config(
            candidate_cfg_path or DEFAULT_CANDIDATE_CONFIG_PATH
        )
    if STAGE_ANCHOR_COMMUNITY_SWEEP in stages:
        configs[STAGE_ANCHOR_COMMUNITY_SWEEP] = _load_pipeline_config(community_cfg_path)
    if STAGE_ANCHOR_SCORED_CLUSTERING in stages:
        sc_cfg_path = (scored_clustering_cfg_path or DEFAULT_SCORED_CLUSTERING_CONFIG_PATH).expanduser().resolve()
        sc_cfg = _load_pipeline_config(sc_cfg_path)
        sc_cfg["_pipeline_config_path"] = str(sc_cfg_path)
        configs[STAGE_ANCHOR_SCORED_CLUSTERING] = sc_cfg
    if STAGE_ANCHOR_PU_SCORED_CLUSTERING in stages:
        pu_cfg_path = (pu_scored_clustering_cfg_path or DEFAULT_PU_SCORED_CLUSTERING_CONFIG_PATH).expanduser().resolve()
        pu_cfg = _load_pipeline_config(pu_cfg_path)
        pu_cfg["_pipeline_config_path"] = str(pu_cfg_path)
        configs[STAGE_ANCHOR_PU_SCORED_CLUSTERING] = pu_cfg
    # Preserve old behavior when users only pass --config and run build stage.
    if stages == [STAGE_BUILD_ANCHOR_GRAPH] and build_cfg_path == DEFAULT_BUILD_CONFIG_PATH:
        configs[STAGE_BUILD_ANCHOR_GRAPH] = _load_pipeline_config(cfg_path)
    result = run_anchor_graph_pipeline(config_by_stage=configs, stages=stages)
    stage_res = result.get(STAGE_BUILD_ANCHOR_GRAPH)
    if isinstance(stage_res, dict):
        print("Wrote:", stage_res.get("paths"))
        print("Validation:", (stage_res.get("summary") or {}).get("validation"))
    elif isinstance(result.get(STAGE_ANCHOR_SEED_GENERATION), dict):
        s = result[STAGE_ANCHOR_SEED_GENERATION]
        print("Output dir:", s.get("output_dir"))
        print("Seeds:", s.get("seed_edges_hard_csv"))
        print("Summary:", s.get("summary_json"))
    elif isinstance(result.get(STAGE_ANCHOR_COMMUNITY_SWEEP), dict):
        c = result[STAGE_ANCHOR_COMMUNITY_SWEEP]
        print("Output dir:", c.get("output_dir"))
        print("Summary:", c.get("summary_json"))
    elif isinstance(result.get(STAGE_ANCHOR_SCORED_CLUSTERING), dict):
        sc = result[STAGE_ANCHOR_SCORED_CLUSTERING]
        print("Config:", sc.get("pipeline_config_path"))
        print("Scored edges:", sc.get("scored_clustering_edges_csv"))
        print("Graph summary:", sc.get("scored_clustering_graph_summary_json"))
        print("Community sweep:", sc.get("community_sweep_output_dir"))
    elif isinstance(result.get(STAGE_ANCHOR_PU_SCORED_CLUSTERING), dict):
        pu = result[STAGE_ANCHOR_PU_SCORED_CLUSTERING]
        print("Config:", pu.get("pipeline_config_path"))
        print("Bundle dir:", pu.get("output_dir"))
        print("Artifact parent:", pu.get("artifact_parent_dir"))
        print("PU scored all CSV:", pu.get("pu_scored_candidate_edges_all_csv"))
        print("Clustering edges:", pu.get("pu_scored_clustering_edges_csv"))
        print("Graph summary:", pu.get("pu_scored_graph_summary_json"))
        print("Eval summary:", pu.get("pu_scored_graph_eval_summary_json"))
        print("Community sweep:", pu.get("community_sweep_output_dir"))
        print("PU threshold retention:", pu.get("pu_threshold_retention_summary_json"))
    else:
        print("Stages run:", result.get("stages_run"))


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Optional CLI mode for overrides.
        main_cli()
        raise SystemExit(0)

    # For individual pipeline stages, uncomment as needed:
    # run_stage_build_anchor_graph(config_path=DEFAULT_BUILD_CONFIG_PATH)
    # run_stage_anchor_seed_generation(config_path=DEFAULT_SEED_CONFIG_PATH)
    # run_stage_anchor_community_sweep(config_path=DEFAULT_COMMUNITY_CONFIG_PATH)
    #
    # To run graph build + strict seed generation:
    run_pipeline(
        build_config_path=DEFAULT_BUILD_CONFIG_PATH,
        seed_config_path=DEFAULT_SEED_CONFIG_PATH,
        community_config_path=DEFAULT_COMMUNITY_CONFIG_PATH,
        candidate_config_path=DEFAULT_CANDIDATE_CONFIG_PATH,
    )

