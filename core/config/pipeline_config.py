"""
Pipeline configuration: load pipeline_config.json and resolve project-relative paths.
Graph build settings are derived here; graph builders receive only explicit parameters.
"""
from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from types import MappingProxyType
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

_RUN_ID_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]{0,127}$")


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def load_pipeline_config(*, project_root: Path | None = None) -> dict[str, Any]:
    root = project_root or _project_root()
    config_path = root / "pipeline_config.json"
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def get_pipeline_config(*, project_root: Path | None = None) -> Mapping[str, Any]:
    """
    Return process-wide pipeline config loaded once and exposed as read-only mapping.

    The mapping itself cannot be reassigned/mutated via item assignment.
    """
    return MappingProxyType(load_pipeline_config(project_root=project_root))


# Single-process global config mapping loaded once at import time.
PIPELINE_CONFIG: Mapping[str, Any] = get_pipeline_config()


def resolve_project_path(path_value: str | None, *, project_root: Path | None = None) -> str | None:
    if not path_value:
        return None
    candidate = Path(path_value)
    if candidate.is_absolute():
        return str(candidate)
    root = project_root or _project_root()
    return str(root / candidate)


def sanitize_run_id(run_id: str) -> str:
    """
    ``run_id`` from config: safe folder basename before allocation suffixes like `` (1)``.
    """
    s = (run_id or "").strip()
    if not s:
        raise ValueError(
            "Set 'run_id' in pipeline_config.json to a unique experiment name "
            "(e.g. 'sage_email_v1'). All stages read/write under <output_runs_root>/<run_id>/."
        )
    if not _RUN_ID_RE.match(s):
        raise ValueError(
            f"Invalid run_id {s!r}: use only letters, digits, '.', '_', '-' "
            "(max 128 chars, must start with a letter or digit)."
        )
    return s


def run_dir_for(runs_parent: str | Path, run_folder_name: str) -> Path:
    """Join ``runs_parent`` with a run directory name (logical or allocated, e.g. ``my_run (1)``)."""
    return Path(runs_parent).expanduser() / run_folder_name


def output_runs_parent_from_pipeline(
    cfg: dict[str, Any],
    *,
    project_root: Path | None = None,
) -> str:
    """
    Parent directory for all runs: top-level ``output_runs_root`` if set, else ``gnn.runs_parent``.
    """
    top = cfg.get("output_runs_root")
    if top is not None and str(top).strip():
        resolved = resolve_project_path(str(top).strip(), project_root=project_root)
        if not resolved:
            raise ValueError("pipeline_config output_runs_root resolved to an empty path.")
        return resolved
    gnn = cfg.get("gnn") or {}
    runs_raw = gnn.get("runs_parent") or "core/outputs"
    resolved = resolve_project_path(str(runs_raw), project_root=project_root)
    if not resolved:
        raise ValueError("pipeline_config gnn.runs_parent resolved to an empty path.")
    return resolved


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
    runs_resolved = output_runs_parent_from_pipeline(cfg, project_root=project_root)

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
class EmailFeatureProjectionSettings:
    """
    Email node SBERT / structured feature projection (see ``graph.feature_projection``).

    - ``other_out_dim`` null: structured block is not passed through a linear layer (full width).
    - ``other_out_dim`` set: ``Linear(other_in_dim → other_out_dim)`` on the structured block.
    - ``bert_out_dim`` null: SBERT maps to the same width as the structured *output* (50/50 by channel count when both sides match).
    - ``bert_out_dim`` set: SBERT maps to that width explicitly.
    """

    seed: int = 42
    bert_out_dim: int | None = None
    other_out_dim: int | None = None


@dataclass(frozen=True)
class DegreeNodeFilterSettings:
    """
    Degree-based node pruning for graph construction.

    ``strength`` in [0, 1]: low values prune only top-degree hubs; higher values
    progressively lower the degree threshold and prune more nodes.
    """

    enabled: bool = False
    strength: float = 0.0
    target_node_types: list[str] | None = None
    min_degree: int = 2


@dataclass(frozen=True)
class GraphBuildSettings:
    """Resolved paths and options for MISP → PyG / Memgraph graph build."""

    misp_json_path: str
    output_dir: str
    exclude_node_types: list[str]
    embeddings_output_dir: str | None
    memgraph: MemgraphSettings
    max_misp_events: int | None = None
    email_feature_projection: EmailFeatureProjectionSettings | None = None
    degree_node_filter: DegreeNodeFilterSettings | None = None


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

    proj_raw = graph_cfg.get("email_feature_projection")
    email_feature_projection: EmailFeatureProjectionSettings | None = None
    if proj_raw is not None:
        if not isinstance(proj_raw, dict):
            raise TypeError("graph.email_feature_projection must be an object or omitted.")
        seed_raw = proj_raw.get("seed", 42)
        try:
            proj_seed = int(seed_raw)
        except (TypeError, ValueError) as e:
            raise ValueError(
                "graph.email_feature_projection.seed must be an integer."
            ) from e

        def _opt_pos_int(key: str) -> int | None:
            v = proj_raw.get(key)
            if v is None or v == "":
                return None
            if isinstance(v, bool):
                raise ValueError(f"graph.email_feature_projection.{key} must be a positive integer or null.")
            try:
                iv = int(v)
            except (TypeError, ValueError) as e:
                raise ValueError(
                    f"graph.email_feature_projection.{key} must be a positive integer or null."
                ) from e
            if iv <= 0:
                raise ValueError(
                    f"graph.email_feature_projection.{key} must be positive when set."
                )
            return iv

        email_feature_projection = EmailFeatureProjectionSettings(
            seed=proj_seed,
            bert_out_dim=_opt_pos_int("bert_out_dim"),
            other_out_dim=_opt_pos_int("other_out_dim"),
        )

    degree_raw = graph_cfg.get("degree_node_filter")
    degree_node_filter: DegreeNodeFilterSettings | None = None
    if degree_raw is not None:
        if not isinstance(degree_raw, dict):
            raise TypeError("graph.degree_node_filter must be an object or omitted.")
        enabled = bool(degree_raw.get("enabled", False))
        strength_raw = degree_raw.get("strength", 0.0)
        try:
            strength = float(strength_raw)
        except (TypeError, ValueError) as e:
            raise ValueError("graph.degree_node_filter.strength must be a number in [0, 1].") from e
        if strength < 0.0 or strength > 1.0:
            raise ValueError("graph.degree_node_filter.strength must be within [0, 1].")

        target_node_types = degree_raw.get("target_node_types")
        if target_node_types is not None:
            if not isinstance(target_node_types, list):
                raise TypeError("graph.degree_node_filter.target_node_types must be a list of strings or null.")
            target_node_types = [str(x) for x in target_node_types]

        min_degree_raw = degree_raw.get("min_degree", 2)
        if isinstance(min_degree_raw, bool):
            raise ValueError("graph.degree_node_filter.min_degree must be a non-negative integer.")
        try:
            min_degree = int(min_degree_raw)
        except (TypeError, ValueError) as e:
            raise ValueError("graph.degree_node_filter.min_degree must be a non-negative integer.") from e
        if min_degree < 0:
            raise ValueError("graph.degree_node_filter.min_degree must be >= 0.")

        degree_node_filter = DegreeNodeFilterSettings(
            enabled=enabled,
            strength=strength,
            target_node_types=target_node_types,
            min_degree=min_degree,
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
        email_feature_projection=email_feature_projection,
        degree_node_filter=degree_node_filter,
    )


def default_hetero_graph_pt_path(*, project_root: Path | None = None) -> str:
    """
    Path to the hetero .pt file produced by build_graph for the current pipeline config
    (same basename rule: {misp_basename}_hetero.pt under graph.output_dir).
    """
    cfg = get_pipeline_config(project_root=project_root)
    s = graph_build_settings_from_pipeline(cfg, project_root=project_root)
    base, _ = os.path.splitext(os.path.basename(s.misp_json_path))
    return os.path.join(s.output_dir, f"{base}_hetero.pt")


# ---------------------------------------------------------------------------
# Featureset clustering settings
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FeaturesetClusteringSettings:
    """Typed, coerced featureset-clustering settings derived from the pipeline config."""

    dataset_base: str
    ground_truth_json: str | None
    feature_sets: list[str]
    # DBSCAN
    eps_values: list[float]
    min_samples: int
    # Mean Shift
    quantile_values: list[float]
    n_samples: int
    # HDBSCAN
    hdbscan_enabled: bool
    min_cluster_size_values: list[int]
    hdbscan_min_samples: int | None
    # Shared embedding
    n_components_values: list[int]
    max_tfidf_features: int | None
    remove_outliers: bool
    outlier_contamination: float
    # Cluster-selection thresholds
    min_coverage_ground_truth: float
    min_coverage_all: float


def featureset_clustering_settings_from_pipeline(
    cfg: Mapping[str, Any],
    *,
    project_root: Path | None = None,
) -> FeaturesetClusteringSettings:
    """Return a fully-coerced, frozen :class:`FeaturesetClusteringSettings` from *cfg*."""
    datasets_cfg = cfg.get("datasets") or {}
    fs_cfg = cfg.get("featureset-clustering") or cfg.get("clustering") or {}
    dbscan_cfg = fs_cfg.get("dbscan") or {}
    meanshift_cfg = fs_cfg.get("meanshift") or {}
    hdbscan_cfg = fs_cfg.get("hdbscan") or {}
    outlier_cfg = fs_cfg.get("outlier_removal") or {}
    gnn_sel = (cfg.get("gnn_clustering") or {}).get("selection") or {}

    min_cov_gt = float(gnn_sel.get("min_coverage_ground_truth", 0.5))
    feature_sets = [str(v) for v in (fs_cfg.get("feature_sets") or [
        "FS1",
        "FS2",
        "FS3",
        "FS4",
        "FS5",
        "FS6",
        "FS7",
    ])]
    return FeaturesetClusteringSettings(
        dataset_base=str(datasets_cfg.get("featureset_base_name") or "synthetic_email_dataset_50"),
        ground_truth_json=resolve_project_path(
            datasets_cfg.get("ground_truth_json"), project_root=project_root
        ),
        feature_sets=feature_sets,
        eps_values=[float(v) for v in (dbscan_cfg.get("eps_values") or [])],
        min_samples=int(dbscan_cfg.get("min_samples", 5)),
        quantile_values=[float(v) for v in (meanshift_cfg.get("quantile_values") or [])],
        n_samples=int(meanshift_cfg.get("n_samples", 500)),
        hdbscan_enabled=bool(hdbscan_cfg.get("enabled", True)),
        min_cluster_size_values=[int(v) for v in (hdbscan_cfg.get("min_cluster_size_values") or [2])],
        hdbscan_min_samples=(
            int(hdbscan_cfg["min_samples"])
            if hdbscan_cfg.get("min_samples") is not None
            else None
        ),
        n_components_values=[int(v) for v in (fs_cfg.get("n_components_values") or [1000])],
        max_tfidf_features=(
            int(fs_cfg["max_tfidf_features"])
            if fs_cfg.get("max_tfidf_features") is not None
            else None
        ),
        remove_outliers=bool(outlier_cfg.get("enabled", True)),
        outlier_contamination=float(outlier_cfg.get("contamination", 0.05)),
        min_coverage_ground_truth=min_cov_gt,
        min_coverage_all=float(gnn_sel.get("min_coverage_all", min_cov_gt)),
    )


FEATURESET_CLUSTERING_CONFIG: FeaturesetClusteringSettings = (
    featureset_clustering_settings_from_pipeline(PIPELINE_CONFIG)
)
