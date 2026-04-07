"""
Pipeline configuration: load pipeline_config.json and resolve project-relative paths.
Graph build settings are derived here; graph builders receive only explicit parameters.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def load_pipeline_config(*, project_root: Path | None = None) -> dict[str, Any]:
    root = project_root or _project_root()
    config_path = root / "pipeline_config.json"
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_project_path(path_value: str | None, *, project_root: Path | None = None) -> str | None:
    if not path_value:
        return None
    candidate = Path(path_value)
    if candidate.is_absolute():
        return str(candidate)
    root = project_root or _project_root()
    return str(root / candidate)


@dataclass(frozen=True)
class GnnPathLayout:
    """
    Output layout for core/GNN stages (train, eval, clustering).
    All directory names are single path segments unless documented otherwise.
    """

    runs_parent: str
    models_subdir: str = "models"
    metrics_csv: str = "metrics.csv"
    training_config_json: str = "training_config.json"
    eval_auroc_ap_subdir: str = "eval_auroc_ap"
    eval_recall_at_k_subdir: str = "eval_recall_at_k"
    clustering_subdir: str = "clustering"
    clustering_plots_subdir: str = "plots"
    stage_result_json: str = "stage_result.json"


def gnn_path_layout_from_pipeline(
    cfg: dict[str, Any],
    *,
    project_root: Path | None = None,
) -> GnnPathLayout:
    gnn = cfg.get("gnn") or {}
    runs_raw = gnn.get("runs_parent") or "core/outputs"
    runs_resolved = resolve_project_path(str(runs_raw), project_root=project_root)
    if not runs_resolved:
        raise ValueError("pipeline_config gnn.runs_parent resolved to an empty path.")

    def _s(key: str, default: str) -> str:
        v = gnn.get(key)
        return str(v).strip() if v is not None and str(v).strip() else default

    return GnnPathLayout(
        runs_parent=runs_resolved,
        models_subdir=_s("models_subdir", "models"),
        metrics_csv=_s("metrics_csv", "metrics.csv"),
        training_config_json=_s("training_config_json", "training_config.json"),
        eval_auroc_ap_subdir=_s("eval_auroc_ap_subdir", "eval_auroc_ap"),
        eval_recall_at_k_subdir=_s("eval_recall_at_k_subdir", "eval_recall_at_k"),
        clustering_subdir=_s("clustering_subdir", "clustering"),
        clustering_plots_subdir=_s("clustering_plots_subdir", "plots"),
        stage_result_json=_s("stage_result_json", "stage_result.json"),
    )


@dataclass(frozen=True)
class MemgraphSettings:
    enabled: bool = False
    uri: str = "bolt://localhost:7687"
    user: str | None = None
    password: str | None = None
    clear: bool = True
    create_indexes: bool = True


@dataclass(frozen=True)
class GraphBuildSettings:
    """Resolved paths and options for MISP → PyG / Memgraph graph build."""

    misp_json_path: str
    output_dir: str
    exclude_node_types: list[str]
    embeddings_output_dir: str | None
    memgraph: MemgraphSettings
    max_misp_events: int | None = None


def graph_build_settings_from_pipeline(
    cfg: dict[str, Any],
    *,
    project_root: Path | None = None,
) -> GraphBuildSettings:
    graph_cfg = cfg.get("graph") or {}
    datasets_cfg = cfg.get("datasets") or {}

    raw_misp = graph_cfg.get("misp_json_path")
    if raw_misp is None or raw_misp == "":
        raw_misp = datasets_cfg.get("misp_json_path")
    if not raw_misp:
        raise ValueError(
            "pipeline_config: set graph.misp_json_path or datasets.misp_json_path for graph build."
        )
    misp_json_path = resolve_project_path(str(raw_misp), project_root=project_root)
    if not misp_json_path:
        raise ValueError("pipeline_config: misp_json_path resolved to empty path.")

    out_dir = graph_cfg.get("output_dir") or "graph/output"
    output_dir = resolve_project_path(str(out_dir), project_root=project_root) or str(out_dir)

    exclude = graph_cfg.get("exclude_node_types")
    if exclude is None:
        exclude = []
    if not isinstance(exclude, list):
        raise TypeError("graph.exclude_node_types must be a list of strings.")
    exclude_node_types = [str(x) for x in exclude]

    emb_raw = graph_cfg.get("embeddings_output_dir")
    embeddings_output_dir = (
        resolve_project_path(str(emb_raw), project_root=project_root) if emb_raw else None
    )

    mg = graph_cfg.get("memgraph") or {}
    memgraph = MemgraphSettings(
        enabled=bool(mg.get("enabled", False)),
        uri=str(mg.get("uri") or "bolt://localhost:7687"),
        user=mg.get("user"),
        password=mg.get("password"),
        clear=bool(mg.get("clear", True)),
        create_indexes=bool(mg.get("create_indexes", True)),
    )

    raw_max = graph_cfg.get("max_misp_events")
    max_misp_events: int | None = None
    if raw_max is not None and not isinstance(raw_max, bool):
        if isinstance(raw_max, int) and raw_max > 0:
            max_misp_events = raw_max
        elif isinstance(raw_max, str) and raw_max.strip():
            try:
                v = int(raw_max.strip(), 10)
            except ValueError as e:
                raise ValueError(
                    "pipeline_config graph.max_misp_events must be a positive integer or null."
                ) from e
            if v > 0:
                max_misp_events = v

    return GraphBuildSettings(
        misp_json_path=misp_json_path,
        output_dir=output_dir,
        exclude_node_types=exclude_node_types,
        embeddings_output_dir=embeddings_output_dir,
        memgraph=memgraph,
        max_misp_events=max_misp_events,
    )


def default_hetero_graph_pt_path(*, project_root: Path | None = None) -> str:
    """
    Path to the hetero .pt file produced by build_graph for the current pipeline config
    (same basename rule: {misp_basename}_hetero.pt under graph.output_dir).
    """
    cfg = load_pipeline_config(project_root=project_root)
    graph_cfg = cfg.get("graph") or {}
    raw_override = graph_cfg.get("graph_pt_path_override")
    if raw_override is not None and str(raw_override).strip():
        resolved = resolve_project_path(str(raw_override), project_root=project_root)
        if resolved:
            return resolved
    s = graph_build_settings_from_pipeline(cfg, project_root=project_root)
    base, _ = os.path.splitext(os.path.basename(s.misp_json_path))
    return os.path.join(s.output_dir, f"{base}_hetero.pt")
