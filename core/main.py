import config.blas_env  # noqa: F401 — before NumPy / sklearn

import json
import ast
import os
import sys
import warnings
from pathlib import Path
from typing import Any
from config.pipeline_config import (
    EmailFeatureProjectionSettings,
    PIPELINE_CONFIG,
    graph_build_settings_from_pipeline,
    output_runs_parent_from_pipeline,
    resolve_project_path,
)

from metric_comparison import run_metric_comparison_for_run
from preprocessing.data_parser import parse_incidents_with_email_bodies, parse_incidents_from_lake_stream
from preprocessing.misp_converter import incidents_to_misp_file

# Make `core/GNN/steps/*` and `core/GNN/src/*` importable from here.
_GNN_ROOT = Path(__file__).resolve().parent / "GNN"
if str(_GNN_ROOT) not in sys.path:
    sys.path.insert(0, str(_GNN_ROOT))

warnings.filterwarnings(
    "ignore",
    message=r".*NeighborSampler.*pyg-lib.*",
    category=UserWarning,
    module="torch_geometric.loader.neighbor_loader",
)

from steps.cluster_stage import run_clustering_stage  # noqa: E402
from steps.eval_auroc_ap_stage import run_auroc_ap_stage  # noqa: E402
from steps.eval_recall_at_k_stage import run_recall_at_k_stage  # noqa: E402
from steps.gnn_pipeline_helpers import load_gnn_cfg, resolve_gnn_paths  # noqa: E402
from steps.train_stage import run_train_stage  # noqa: E402
from steps.clustering_plot_stage import run_clustering_plot_stage  # noqa: E402


def _require_path(value: str | None, field_name: str) -> str:
    if not value:
        raise ValueError(f"Missing required path configuration for {field_name}.")
    return value


def _load_env_file(path: Path) -> None:
    if not path.exists() or not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = value


def _coerce_limit(value: object) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return int(text)
        except ValueError:
            return None
    return None


def run_gnn(
    *,
    graph_path: str | Path | None = None,
    run_dir: str | Path | None = None,
    runs_parent: str | Path | None = None,
    checkpoint_path: str | Path | None = None,  # unused for training but accepted for symmetry
    device_pref: str | None = None,
    pair_training_overrides: dict[str, Any] | None = None,
):
    cfg = PIPELINE_CONFIG
    g = load_gnn_cfg(cfg)
    run_dir_str, _checkpoint_path_str, graph_path_str, _gt_str = resolve_gnn_paths(
        cfg=cfg,
        run_dir=run_dir,
        runs_parent=runs_parent,
        checkpoint_path=checkpoint_path,
        graph_path=graph_path,
        ground_truth_path=None,
        require_ground_truth=False,
    )

    run_path = Path(run_dir_str)
    runs_parent_effective = run_path.parent
    run_folder_name = run_path.name

    return run_train_stage(
        graph_path=graph_path_str,
        runs_parent=runs_parent_effective,
        run_id=run_folder_name,
        training_cfg=g["training_cfg"],
        path_layout=g["path_layout"],
        device_pref=device_pref if device_pref is not None else g["device_pref"],
        to_undirected=g["to_undirected"],
        pair_training_overrides=pair_training_overrides,
    )


def run_gnn_evaluation(
    *,
    graph_path: str | Path | None = None,
    run_dir: str | Path | None = None,
    runs_parent: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
):
    cfg = PIPELINE_CONFIG
    g = load_gnn_cfg(cfg)
    run_dir_str, checkpoint_path_str, graph_path_str, _gt_str = resolve_gnn_paths(
        cfg=cfg,
        run_dir=run_dir,
        runs_parent=runs_parent,
        checkpoint_path=checkpoint_path,
        graph_path=graph_path,
        ground_truth_path=None,
        require_ground_truth=False,
    )
    evaluation_cfg_auroc = g["evaluation_auroc_cfg"]
    recall_cfg = g["recall_cfg"]
    layout = g["path_layout"]

    res_auroc = run_auroc_ap_stage(
        graph_path=graph_path_str,
        checkpoint_path=checkpoint_path_str,
        output_dir=run_dir_str,
        evaluation_cfg=evaluation_cfg_auroc,
        path_layout=layout,
        device_pref=g["device_pref"],
        to_undirected=g["to_undirected"],
    )
    res_recall = run_recall_at_k_stage(
        graph_path=graph_path_str,
        checkpoint_path=checkpoint_path_str,
        output_dir=run_dir_str,
        evaluation_cfg=recall_cfg,
        path_layout=layout,
        device_pref=g["device_pref"],
        to_undirected=g["to_undirected"],
    )
    return {
        "run_dir": run_dir_str,
        "checkpoint_path": checkpoint_path_str,
        "auroc_ap": res_auroc,
        "recall_at_k": res_recall,
    }


def run_gnn_clustering(
    *,
    graph_path: str | Path | None = None,
    ground_truth_path: str | Path | None = None,
    run_dir: str | Path | None = None,
    runs_parent: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
    make_plots: bool = True,
):
    cfg = PIPELINE_CONFIG
    g = load_gnn_cfg(cfg)
    run_dir_str, checkpoint_path_str, graph_path_str, ground_truth_path_str = resolve_gnn_paths(
        cfg=cfg,
        run_dir=run_dir,
        runs_parent=runs_parent,
        checkpoint_path=checkpoint_path,
        graph_path=graph_path,
        ground_truth_path=ground_truth_path,
        require_ground_truth=True,
    )
    print(
        "[clustering diag] resolve_gnn_paths OK | "
        f"run_dir={run_dir_str!r} graph={graph_path_str!r}",
        flush=True,
    )
    tf_cfg = (g["gnn_clustering_selection_cfg"].get("transformer_text_baseline", {}) or {})
    tf_enabled = bool(tf_cfg.get("enabled", False))
    tf_json_cfg = tf_cfg.get("embeddings_json_path")
    tf_json_resolved = resolve_project_path(tf_json_cfg) if tf_json_cfg else ""
    if not tf_json_resolved:
        tf_json_resolved = str(
            (_GNN_ROOT.parent / "utils" / "embeddings" / "output" / "embeddings.json").resolve()
        )

    res = run_clustering_stage(
        graph_path=graph_path_str,
        ground_truth_path=ground_truth_path_str,
        checkpoint_path=checkpoint_path_str,
        output_dir=run_dir_str,
        clustering_cfg=g["gnn_clustering_cfg"],
        baselines_cfg=g["gnn_clustering_baselines_cfg"],
        min_coverage_ground_truth=float(
            g["gnn_clustering_selection_cfg"].get("min_coverage_ground_truth", 0.5)
        ),
        min_coverage_all=float(
            g["gnn_clustering_selection_cfg"].get(
                "min_coverage_all",
                g["gnn_clustering_selection_cfg"].get("min_coverage_ground_truth", 0.5),
            )
        ),
        hybrid_embeddings=bool(
            g["gnn_clustering_selection_cfg"].get("hybrid_embeddings", False)
        ),
        hybrid_raw_weight=float(
            g["gnn_clustering_selection_cfg"].get("hybrid_raw_weight", 1.0)
        ),
        hybrid_gnn_weight=float(
            g["gnn_clustering_selection_cfg"].get("hybrid_gnn_weight", 1.0)
        ),
        model_save_name=g["training_cfg"]["model_save_name"],
        path_layout=g["path_layout"],
        device_pref=g["device_pref"],
        to_undirected=g["to_undirected"],
        transformer_baseline_enabled=tf_enabled,
        transformer_embeddings_json_path=tf_json_resolved,
    )

    if make_plots:
        plots_res = run_clustering_plot_stage(
            output_dir=run_dir_str,
            path_layout=g["path_layout"],
        )
        return res | {"clustering_plots": plots_res}

    return res


def run_preprocessing_trec(
    *,
    incidents_csv_path: str | Path | None = None,
    misp_json_path: str | Path | None = None,
    limit: int | None = None,
):
    """
    Parse a TREC-style CSV (columns: sender, receiver, date, subject, body, label, urls)
    and write MISP JSON.

    Paths default from ``pipeline_config.json`` ``preprocessing``:

    - CSV: ``trec_csv_path`` if set, else ``data/csv/TREC-07-only-phishing-6m.csv``
    - Output: ``misp_json_path`` (required in config unless passed explicitly; parent dirs are created)
    - Row cap: ``limit`` if > 0 (from config unless passed explicitly)

    For the same downstream behavior as ``run_preprocessing()``, keep ``datasets.misp_json_path``
    and ``graph.misp_json_path`` aligned with ``preprocessing.misp_json_path``, and set
    ``datasets.featureset_base_name`` to the basename of that MISP file without ``.json``.

    Pass keyword arguments to override config without editing this file.
    """
    cfg = PIPELINE_CONFIG
    prep_cfg = cfg.get("preprocessing", {})

    if incidents_csv_path is not None and str(incidents_csv_path).strip():
        csv_rel = str(incidents_csv_path).strip()
    else:
        csv_rel = prep_cfg.get("trec_csv_path") or "data/csv/TREC-07-only-phishing-6m.csv"
    incidents_csv_path_resolved = _require_path(
        resolve_project_path(csv_rel),
        "TREC-style CSV (preprocessing.trec_csv_path or incidents_csv_path argument)",
    )

    if misp_json_path is not None and str(misp_json_path).strip():
        misp_rel = str(misp_json_path).strip()
    else:
        misp_rel = prep_cfg.get("misp_json_path")
    misp_json_path_resolved = _require_path(
        resolve_project_path(misp_rel),
        "preprocessing misp_json_path",
    )

    if limit is None:
        limit = prep_cfg.get("limit")
    limit_eff = _coerce_limit(limit)
    if limit_eff is None:
        limit_eff = 0

    print(f"Parsing TREC-style CSV incidents from {incidents_csv_path_resolved}...")

    import pandas as pd
    if limit_eff > 0:
        df = pd.read_csv(incidents_csv_path_resolved, encoding="utf-8-sig", nrows=limit_eff)
    else:
        df = pd.read_csv(incidents_csv_path_resolved, encoding="utf-8-sig")
    df = df.fillna("")

    # Align source columns with the incident schema used by synthetic data preprocessing.
    column_rename_map = {
        "date": "date_sent",
        "body": "email_body",
        "urls": "email_urls",
    }
    df = df.rename(columns=column_rename_map)

    incidents = []
    for idx, row in enumerate(df.to_dict(orient="records")):
        raw_urls = row.get("email_urls", "")
        parsed_urls = []
        if isinstance(raw_urls, list):
            parsed_urls = [u for u in raw_urls if isinstance(u, str) and u]
        elif isinstance(raw_urls, str) and raw_urls.strip():
            raw_text = raw_urls.strip()
            if raw_text.startswith("[") and raw_text.endswith("]"):
                try:
                    literal = ast.literal_eval(raw_text)
                    if isinstance(literal, list):
                        parsed_urls = [u for u in literal if isinstance(u, str) and u]
                except Exception:
                    parsed_urls = [raw_text]
            else:
                parsed_urls = [raw_text]

        label_value = str(row.get("label", "")).strip()
        category_value = "phishing" if label_value == "1" else label_value

        incident = {
            "record_index": idx,
            "external_id": f"trec_{idx}",
            "subject": row.get("subject", ""),
            "date_sent": row.get("date_sent", ""),
            "email_body": row.get("email_body", ""),
            "email_urls": parsed_urls,
            "category": category_value,
            "rfc_defects": "false",
            "cyrillic_domain": "false",
            "contains_symbols": "false",
            "body_has_tracking_url": "false",
            "body_has_tracking_image": "false",
            "body_has_tracking_pixel": "false",
            "body_has_unsubscribe_link": "false",
            "domain_is_common_webprovided": "false",
            "email_attachments": [],
            "email_attachment_metadata": [],
            "email_html": {"tag_counts": {}, "tree_stats": {}, "structure_fingerprint": ""},
            "email_css": {"style_features": {}},
            "email_headers": {
                "From": row.get("sender", ""),
                "To": row.get("receiver", ""),
                "Received": [],
                "Return-Path": {"email": "", "domain": ""},
                "Content-Type": "",
                "Received-SPF": "",
                "List-Unsubscribe": "",
                "Authentication-Results": "",
                "X-Forefront-Antispam-Report": "",
                "X-MS-Exchange-Organization-SCL": "",
            },
        }
        incidents.append(incident)

    print(f"Parsed {len(incidents)} incidents from CSV using pandas.")

    print(f"Converting incidents to MISP and writing secure output at {misp_json_path_resolved}...")
    incidents_to_misp_file(incidents, misp_json_path_resolved)
    print("MISP conversion complete.")
    return misp_json_path_resolved

def run_preprocessing():
    """
    Parse incident metadata and email body files, then convert to MISP JSON.
    """
    cfg = PIPELINE_CONFIG
    prep_cfg = cfg.get("preprocessing", {})
    limit = prep_cfg.get("limit")

    incidents_csv_path = _require_path(
        resolve_project_path(prep_cfg.get("incidents_csv_path")),
        "preprocessing incidents_csv_path",
    )
    bodies_dir = _require_path(
        resolve_project_path(prep_cfg.get("bodies_dir")),
        "preprocessing bodies_dir",
    )
    misp_json_path = _require_path(
        resolve_project_path(prep_cfg.get("misp_json_path")),
        "preprocessing misp_json_path",
    )

    raw_category_allow = prep_cfg.get("category_allowlist")
    allowed_categories = None
    if isinstance(raw_category_allow, list) and raw_category_allow:
        allowed_categories = [str(x) for x in raw_category_allow]

    print(f"Parsing incidents from {incidents_csv_path}...")
    incidents = parse_incidents_with_email_bodies(
        incidents_csv_path,
        bodies_dir,
        limit=limit,
        allowed_categories=allowed_categories,
    )
    print(f"Parsed {len(incidents)} incidents with matched email body content.")

    print(f"Converting incidents to MISP and writing secure output at {misp_json_path}...")
    incidents_to_misp_file(incidents, misp_json_path)
    print("MISP conversion complete.")
    return misp_json_path


def run_preprocessing_lake():
    """
    Parse incidents by streaming joined rows from lake tables, then convert to MISP JSON.
    Secrets must come from environment variables.
    """
    cfg = PIPELINE_CONFIG
    prep_cfg = cfg.get("preprocessing", {})
    prep_lake_cfg = cfg.get("preprocessing_lake", {})

    # Allow running without manual export when credentials are in local .env files.
    _load_env_file(Path(__file__).resolve().parent / "lake" / ".env")
    _load_env_file(Path(__file__).resolve().parent.parent / ".env")

    limit = prep_cfg.get("limit")
    lake_limit = _coerce_limit(prep_lake_cfg.get("limit"))
    if lake_limit is not None:
        limit = lake_limit

    raw_category_allow = prep_cfg.get("category_allowlist")
    if isinstance(prep_lake_cfg.get("category_allowlist"), list):
        raw_category_allow = prep_lake_cfg.get("category_allowlist")

    allowed_categories = None
    if isinstance(raw_category_allow, list) and raw_category_allow:
        allowed_categories = [str(x) for x in raw_category_allow]

    base_url = os.getenv("LAKE_BASE_URL", "").strip()
    api_key = os.getenv("LAKE_API_KEY", "").strip()
    incidents_table = str(
        prep_lake_cfg.get("incidents_table")
        or os.getenv("LAKE_INCIDENTS_TABLE", "intellagent.public.incidents")
    ).strip()
    parsed_emails_table = str(
        prep_lake_cfg.get("parsed_emails_table")
        or os.getenv("LAKE_PARSED_EMAILS_TABLE", "parsed_emails")
    ).strip()

    if not base_url or not api_key:
        raise RuntimeError(
            "Missing required environment variables LAKE_BASE_URL and/or LAKE_API_KEY."
        )

    misp_json_path = _require_path(
        resolve_project_path(prep_cfg.get("misp_json_path")),
        "preprocessing misp_json_path",
    )
    start_date = prep_lake_cfg.get("start_date")
    end_date = prep_lake_cfg.get("end_date")

    timeframe_note = ""
    if start_date is not None or end_date is not None:
        timeframe_note = f", timeframe start_date={start_date!r} end_date={end_date!r}"

    print(
        "Parsing incidents from lake stream "
        f"({incidents_table} JOIN {parsed_emails_table}){timeframe_note}..."
    )
    incidents = parse_incidents_from_lake_stream(
        base_url=base_url,
        api_key=api_key,
        incidents_table=incidents_table,
        parsed_emails_table=parsed_emails_table,
        limit=limit,
        allowed_categories=allowed_categories,
        start_date=start_date,
        end_date=end_date,
    )
    print(f"Parsed {len(incidents)} incidents from lake stream.")

    print(f"Converting incidents to MISP and writing secure output at {misp_json_path}...")
    incidents_to_misp_file(incidents, misp_json_path)
    print("MISP conversion complete.")
    return misp_json_path

def run_graph_creation(
    misp_json_path: str | None = None,
    *,
    max_misp_events: int | None = None,
    to_memgraph: bool | None = None,
    mg_uri: str | None = None,
    mg_user: str | None = None,
    mg_password: str | None = None,
):
    """
    MISP JSON → PyTorch Geometric graph (and optionally Memgraph).
    Paths and exclusions come from pipeline_config.json ``graph`` (and ``datasets`` for MISP).
    Pass misp_json_path to override (e.g. output of run_preprocessing).
    Use graph.max_misp_events in config, or pass max_misp_events here, to use only the first N
    MISP events after loading the file (full file is still read from disk).
    """
    from graph.graph_builder_pytorch import build_graph
    from graph.graph_builder_memgraph import build_memgraph

    cfg = PIPELINE_CONFIG
    settings = graph_build_settings_from_pipeline(cfg)
    path = misp_json_path or settings.misp_json_path
    limit = (
        max_misp_events
        if max_misp_events is not None
        else settings.max_misp_events
    )
    limit_eff = limit if limit is not None and limit > 0 else None

    email_proj = settings.email_feature_projection or EmailFeatureProjectionSettings()
    graph, graph_path, meta_path = build_graph(
        misp_json_path=path,
        out_dir=settings.output_dir,
        exclude_nodes=settings.exclude_node_types,
        degree_node_filter=settings.degree_node_filter,
        embeddings_output_dir=settings.embeddings_output_dir,
        max_misp_events=limit_eff,
        email_feature_projection=email_proj,
    )
    print(f"Graph created: {graph}")
    print(f"Saved graph to: {graph_path}")
    print(f"Saved metadata to: {meta_path}")

    use_memgraph = settings.memgraph.enabled if to_memgraph is None else to_memgraph
    if use_memgraph:
        mg = settings.memgraph
        summary = build_memgraph(
            misp_json_path=path,
            mg_uri=mg_uri if mg_uri is not None else mg.uri,
            mg_user=mg_user if mg_user is not None else mg.user,
            mg_password=mg_password if mg_password is not None else mg.password,
            clear=mg.clear,
            create_indexes=mg.create_indexes,
            exclude_nodes=settings.exclude_node_types,
            max_misp_events=limit_eff,
        )
        print("Memgraph load summary:")
        print(json.dumps(summary, indent=2))

    return graph

def create_feature_sets():
    from feature_set_extraction.feature_set_extraction import run_featureset_extraction
    cfg = PIPELINE_CONFIG
    misp_path = resolve_project_path(cfg.get("datasets", {}).get("misp_json_path"))
    max_workers = cfg.get("featureset-creation", {}).get("max_workers", 2)
    run_featureset_extraction(misp_path=misp_path, max_workers=max_workers)

def visualize_clusters(
    *,
    run_dir: str | Path | None = None,
    run_id: str | None = None,
    misp_json_path: str | Path | None = None,
    include_attribute_similarity: bool | None = None,
) -> dict[str, Any]:
    """
    Build ``<run_dir>/visualization/data.json`` and optionally start the Docker Compose UI.

    Exactly one of ``run_dir`` or ``run_id`` should be set, or neither to use the current
    session run directory (see ``PIPELINE_RUN_OUTPUT_DIR`` / allocation).

    ``run_id`` is resolved under ``output_runs_root``: either an exact folder name
    (e.g. ``my_run`` or ``my_run (1)``) or a unique prefix among subdirectory names.

    Reads optional ``visualization`` block from ``pipeline_config.json``:
    - ``enabled`` (default True): run ``docker compose`` for :file:`docker-compose.visualization.yml`
    - ``port`` (default 8787): host port mapping
    - ``compose_file`` (default ``docker-compose.visualization.yml``): path relative to repo root
    - ``include_attribute_similarity`` (default True): SBERT similarity vs campaign peer average
    """
    from config.run_output_paths import resolve_session_run_output_dir

    cfg = PIPELINE_CONFIG
    runs_root = output_runs_parent_from_pipeline(cfg)

    if run_dir is not None and run_id is not None:
        raise ValueError("Pass at most one of run_dir or run_id.")

    if run_id is not None:
        try:
            from core.visualization.run_paths import resolve_run_dir_by_run_id
        except ModuleNotFoundError:
            from visualization.run_paths import resolve_run_dir_by_run_id
        run_path = resolve_run_dir_by_run_id(cfg, run_id)
    elif run_dir is None:
        run_path = resolve_session_run_output_dir(cfg, runs_root=runs_root)
    else:
        run_path = Path(run_dir).expanduser().resolve()

    if misp_json_path is None:
        misp_raw = graph_build_settings_from_pipeline(cfg).misp_json_path
        misp_resolved = misp_raw
    else:
        misp_resolved = resolve_project_path(str(misp_json_path)) or str(
            Path(misp_json_path).expanduser().resolve()
        )

    try:
        from core.visualization.data_builder import write_visualization_data_json
    except ModuleNotFoundError:
        from visualization.data_builder import write_visualization_data_json

    viz_cfg = cfg.get("visualization") or {}
    sim_default = bool(viz_cfg.get("include_attribute_similarity", True))
    include_sim = sim_default if include_attribute_similarity is None else bool(
        include_attribute_similarity
    )

    out_json = write_visualization_data_json(
        run_dir=run_path,
        misp_json_path=str(misp_resolved),
        include_attribute_similarity=include_sim,
    )
    print(f"Visualization data written to: {out_json}")

    if not viz_cfg.get("enabled", True):
        print("visualization.enabled is false; skipping docker compose.")
        return {"visualization_json": str(out_json), "compose": None}

    port = int(viz_cfg.get("port", 8787))
    compose_name = str(viz_cfg.get("compose_file", "docker-compose.visualization.yml")).strip()
    repo_root = Path(__file__).resolve().parent.parent
    compose_path = repo_root / compose_name
    if not compose_path.is_file():
        print(f"Warning: compose file not found at {compose_path}; skip docker compose.")
        return {"visualization_json": str(out_json), "compose": None}

    import subprocess

    env = os.environ.copy()
    env["RUN_DIR"] = str(run_path.resolve())
    env["VIZ_PORT"] = str(port)

    try:
        subprocess.run(
            ["docker", "compose", "-f", str(compose_path), "up", "-d", "--build"],
            cwd=str(repo_root),
            env=env,
            check=True,
        )
        print(f"Cluster visualization UI: http://localhost:{port}/")
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        print(f"Warning: could not start visualization stack ({exc}). Data file is still at {out_json}")

    return {
        "visualization_json": str(out_json),
        "url": f"http://localhost:{port}/",
        "compose": str(compose_path),
    }


def run_featureset_clustering():
    """
    Run DBSCAN, Mean Shift, and (when enabled) HDBSCAN clustering with grid search.
    Delegates to feature_set_extraction.clustering.featureset_clustering.
    """
    from feature_set_extraction.clustering.featureset_clustering import (
        run_featureset_clustering as _run,
    )

    _run()

def run_seed_candidate_pu_pipeline(
    experiment_config: str | Path | None = None,
    *,
    relaxed_semantics: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    """
    One-shot flow: seed/candidate graph setup (``setup_only``) → pair-supervised GNN training
    on the bundle pair CSV (same train stage as :func:`run_gnn`) → community scoring (``score_only``)
    with PU scorer paths filled from ``pipeline_config.json``.

    Graph setup always uses ``on_present: rebuild`` (fresh anchor/seed/candidate/pair artifacts).

    Equivalent to ``experiment.mode`` = ``setup_gnn_score`` in
    :func:`seed_candidate_workflow.pipelines.run_experiment.run_experiment`.

    Pass ``relaxed_semantics=True`` (and leave ``experiment_config`` unset) to use the relaxed
    semantic-threshold experiment (``exp04`` — lower min cosine for seeds, candidates, and anchor).
    """
    from seed_candidate_workflow.pipelines.run_experiment import (
        DEFAULT_EXPERIMENT_CONFIG_PATH,
        RELAXED_SEED_CANDIDATE_PU_EXPERIMENT_CONFIG_PATH,
        run_experiment,
    )

    if experiment_config is not None:
        cfg_path = Path(experiment_config).expanduser().resolve()
    elif relaxed_semantics:
        cfg_path = RELAXED_SEED_CANDIDATE_PU_EXPERIMENT_CONFIG_PATH
    else:
        cfg_path = DEFAULT_EXPERIMENT_CONFIG_PATH
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    cfg.setdefault("experiment", {})
    cfg["experiment"]["mode"] = "setup_gnn_score"
    return run_experiment(cfg, dry_run=dry_run)


def run_metric_comparison(
    *,
    run_dir: str | Path | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    """
    Compare feature-set clustering vs GNN campaign outputs for a run.

    Writes ``<run_dir>/metric_comparison/`` (plots + JSON + CSV).

    Exactly one of ``run_dir`` or ``run_id`` should be set, or neither to use the
    current session run directory (see ``PIPELINE_RUN_OUTPUT_DIR`` / allocation).
    """
    from config.run_output_paths import resolve_session_run_output_dir

    cfg = PIPELINE_CONFIG
    runs_root = output_runs_parent_from_pipeline(cfg)

    if run_dir is not None and run_id is not None:
        raise ValueError("Pass at most one of run_dir or run_id.")

    if run_id is not None:
        try:
            from core.visualization.run_paths import resolve_run_dir_by_run_id
        except ModuleNotFoundError:
            from visualization.run_paths import resolve_run_dir_by_run_id
        run_path = resolve_run_dir_by_run_id(cfg, run_id)
    elif run_dir is None:
        run_path = resolve_session_run_output_dir(cfg, runs_root=runs_root)
    else:
        run_path = Path(run_dir).expanduser().resolve()

    gt_rel = cfg.get("datasets", {}).get("ground_truth_json")
    gt_path = resolve_project_path(gt_rel) if gt_rel else None
    if not gt_path:
        raise ValueError(
            "pipeline_config datasets.ground_truth_json is required for metric comparison."
        )

    return run_metric_comparison_for_run(run_path, ground_truth_path=gt_path)

def run_pipeline():
    misp_json_path = run_preprocessing_lake()
    create_feature_sets()
    run_featureset_clustering()
    run_graph_creation(misp_json_path)
    run_gnn()
    run_gnn_evaluation()
    run_gnn_clustering()
    run_metric_comparison()
    visualize_clusters()

if __name__ == "__main__":
    
    # For individual stages of the pipeline, uncomment as needed:
    #misp_path = run_preprocessing_lake()
    #create_feature_sets()
    #run_featureset_clustering()
    
    #misp_path = "preprocessing/output/incidents-lake-misp.json"
    #run_graph_creation(misp_path, to_memgraph=False)
    run_seed_candidate_pu_pipeline(relaxed_semantics=True)   # uses exp04
    #run_gnn()
    #run_gnn_evaluation()
    #run_gnn_clustering()
    #run_metric_comparison()
    #visualize_clusters()
    
    # To run the entire pipeline, uncomment the line below:
    # run_pipeline()