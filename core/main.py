import config.blas_env  # noqa: F401 — before NumPy / sklearn

import json
import ast
import sys
from pathlib import Path
from config.pipeline_config import (
    graph_build_settings_from_pipeline,
    load_pipeline_config,
    resolve_project_path,
)
from preprocessing.data_parser import parse_incidents_with_email_bodies
from preprocessing.misp_converter import incidents_to_misp_file

# Make `core/GNN/steps/*` and `core/GNN/src/*` importable from here.
_GNN_ROOT = Path(__file__).resolve().parent / "GNN"
if str(_GNN_ROOT) not in sys.path:
    sys.path.insert(0, str(_GNN_ROOT))

from steps.cluster_stage import run_clustering_stage  # noqa: E402
from steps.eval_auroc_ap_stage import run_auroc_ap_stage  # noqa: E402
from steps.eval_recall_at_k_stage import run_recall_at_k_stage  # noqa: E402
from steps.gnn_pipeline_helpers import load_gnn_cfg, resolve_gnn_paths  # noqa: E402
from steps.pipeline_paths import run_dir_for, sanitize_run_id  # noqa: E402
from steps.train_stage import run_train_stage  # noqa: E402
from steps.clustering_plot_stage import run_clustering_plot_stage  # noqa: E402

def run_gnn(
    *,
    graph_path: str | Path | None = None,
    run_dir: str | Path | None = None,
    runs_parent: str | Path = "outputs",
    checkpoint_path: str | Path | None = None,  # unused for training but accepted for symmetry
    device_pref: str | None = None,
):
    cfg = load_pipeline_config()
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

    # Make sure the training output directory matches `run_dir` override (if provided).
    runs_parent_effective: str | Path = runs_parent
    if run_dir is not None and str(run_dir).strip() != "":
        expected = sanitize_run_id(str(g["run_id"]))
        if Path(run_dir_str).name != expected:
            raise ValueError(
                f"run_dir override basename {Path(run_dir_str).name!r} must match sanitize_run_id(run_id)={expected!r}"
            )
        runs_parent_effective = Path(run_dir_str).parent

    return run_train_stage(
        graph_path=graph_path_str,
        runs_parent=runs_parent_effective,
        run_id=g["run_id"],
        training_cfg=g["training_cfg"],
        device_pref=device_pref if device_pref is not None else g["device_pref"],
        to_undirected=g["to_undirected"],
    )


def run_gnn_evaluation(
    *,
    graph_path: str | Path | None = None,
    run_dir: str | Path | None = None,
    runs_parent: str | Path = "outputs",
    checkpoint_path: str | Path | None = None,
):
    cfg = load_pipeline_config()
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

    res_auroc = run_auroc_ap_stage(
        graph_path=graph_path_str,
        checkpoint_path=checkpoint_path_str,
        output_dir=run_dir_str,
        evaluation_cfg=evaluation_cfg_auroc,
        device_pref=g["device_pref"],
        to_undirected=g["to_undirected"],
    )
    res_recall = run_recall_at_k_stage(
        graph_path=graph_path_str,
        checkpoint_path=checkpoint_path_str,
        output_dir=run_dir_str,
        evaluation_cfg=recall_cfg,
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
    runs_parent: str | Path = "outputs",
    checkpoint_path: str | Path | None = None,
    make_plots: bool = True,
):
    cfg = load_pipeline_config()
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

    res = run_clustering_stage(
        graph_path=graph_path_str,
        ground_truth_path=ground_truth_path_str,
        checkpoint_path=checkpoint_path_str,
        output_dir=run_dir_str,
        clustering_cfg=g["gnn_clustering_cfg"],
        min_coverage_ground_truth=float(
            g["gnn_clustering_selection_cfg"].get("min_coverage_ground_truth", 0.5)
        ),
        min_coverage_all=float(
            g["gnn_clustering_selection_cfg"].get(
                "min_coverage_all",
                g["gnn_clustering_selection_cfg"].get("min_coverage_ground_truth", 0.5),
            )
        ),
        model_save_name=g["training_cfg"]["model_save_name"],
        device_pref=g["device_pref"],
        to_undirected=g["to_undirected"],
    )

    if make_plots:
        plots_res = run_clustering_plot_stage(output_dir=run_dir_str)
        return res | {"clustering_plots": plots_res}

    return res


def run_preprocessing_trec():
    """
    Loads the TREC-07-only-phishing-6m.csv path from config and creates a MISP JSON file from it.
    """
    cfg = load_pipeline_config()
    prep_cfg = cfg.get("preprocessing", {})
    # Force the incidents_csv_path to the TREC-07-only-phishing-6m.csv from config
    incidents_csv_path = resolve_project_path("data/csv/TREC-07-only-phishing-6m.csv")
    misp_json_path = resolve_project_path(prep_cfg.get("misp_json_path"))
    limit = prep_cfg.get("limit")

    print(f"Parsing TREC-07-only-phishing-6m incidents from {incidents_csv_path}...")

    import pandas as pd
    if limit > 0:
        df = pd.read_csv(incidents_csv_path, encoding="utf-8-sig", nrows=limit)
    else:
        df = pd.read_csv(incidents_csv_path, encoding="utf-8-sig")
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

    print(f"Converting incidents to MISP and writing secure output at {misp_json_path}...")
    incidents_to_misp_file(incidents, misp_json_path)
    print("MISP conversion complete.")
    return misp_json_path

def run_preprocessing():
    """
    Parse incident metadata and email body files, then convert to MISP JSON.
    """
    cfg = load_pipeline_config()
    prep_cfg = cfg.get("preprocessing", {})
    limit = prep_cfg.get("limit")

    incidents_csv_path = resolve_project_path(prep_cfg.get("incidents_csv_path"))
    bodies_dir = resolve_project_path(prep_cfg.get("bodies_dir"))
    misp_json_path = resolve_project_path(prep_cfg.get("misp_json_path"))

    print(f"Parsing incidents from {incidents_csv_path}...")
    incidents = parse_incidents_with_email_bodies(
        incidents_csv_path,
        bodies_dir,
        limit=limit,
    )
    print(f"Parsed {len(incidents)} incidents with matched email body content.")

    print(f"Converting incidents to MISP and writing secure output at {misp_json_path}...")
    incidents_to_misp_file(incidents, misp_json_path)
    print("MISP conversion complete.")
    return misp_json_path

def run_graph_creation(
    misp_json_path: str | None = None,
    *,
    to_memgraph: bool | None = None,
    mg_uri: str | None = None,
    mg_user: str | None = None,
    mg_password: str | None = None,
):
    """
    MISP JSON → PyTorch Geometric graph (and optionally Memgraph).
    Paths and exclusions come from pipeline_config.json ``graph`` (and ``datasets`` for MISP).
    Pass misp_json_path to override (e.g. output of run_preprocessing).
    """
    from graph.graph_builder_pytorch import build_graph
    from graph.graph_builder_memgraph import build_memgraph

    cfg = load_pipeline_config()
    settings = graph_build_settings_from_pipeline(cfg)
    path = misp_json_path or settings.misp_json_path

    graph, graph_path, meta_path = build_graph(
        misp_json_path=path,
        out_dir=settings.output_dir,
        exclude_nodes=settings.exclude_node_types,
        embeddings_output_dir=settings.embeddings_output_dir,
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
        )
        print("Memgraph load summary:")
        print(json.dumps(summary, indent=2))

    return graph

def create_feature_sets():
    from feature_set_extraction.feature_set_extraction import run_featureset_extraction
    cfg = load_pipeline_config()
    misp_path = resolve_project_path(cfg.get("datasets", {}).get("misp_json_path"))
    run_featureset_extraction(misp_path=misp_path)

def run_featureset_clustering():
    """
    Run DBSCAN and Mean Shift clustering with grid search over parameters.
    Delegates to feature_set_extraction.clustering.featureset_clustering.
    """
    from feature_set_extraction.clustering.featureset_clustering import (
        run_featureset_clustering as _run,
    )

    cfg = load_pipeline_config()
    graph_s = graph_build_settings_from_pipeline(cfg)
    clustering_cfg = cfg.get("featureset-clustering", cfg.get("clustering", {}))
    dbscan_cfg = clustering_cfg.get("dbscan", {})
    meanshift_cfg = clustering_cfg.get("meanshift", {})
    outlier_cfg = clustering_cfg.get("outlier_removal", {})

    _run(
        dataset_base=cfg.get("datasets", {}).get(
            "featureset_base_name", "synthetic_email_dataset_50"
        ),
        ground_truth_json=resolve_project_path(
            cfg.get("datasets", {}).get("ground_truth_json")
        ),
        eps_values=dbscan_cfg.get("eps_values", [1, 1.5, 2]),
        min_samples=dbscan_cfg.get("min_samples", 5),
        quantile_values=meanshift_cfg.get("quantile_values", [0.25]),
        n_samples=meanshift_cfg.get("n_samples", 500),
        n_components_values=clustering_cfg.get("n_components_values", [1000]),
        max_tfidf_features=clustering_cfg.get("max_tfidf_features"),
        remove_outliers=outlier_cfg.get("enabled", True),
        outlier_contamination=outlier_cfg.get("contamination", 0.05),
        embeddings_output_dir=graph_s.embeddings_output_dir,
    )

def run_metrics_evaluation():
    # input Clusters --> Evaluate clustering results using metrics -- > store metrics
    pass

def run_pipeline():
    misp_json_path = run_preprocessing()
    create_feature_sets()
    run_featureset_clustering()
    run_graph_creation(misp_json_path)
    run_gnn()
    run_gnn_evaluation()
    run_gnn_clustering()
    run_metrics_evaluation()

if __name__ == "__main__":
    # For individual stages of the pipeline, uncomment as needed:
    # misp_path = run_preprocessing()
    # create_feature_sets()
    # run_featureset_clustering()
    # misp_path = "preprocessing/output/incidents-20260211-misp.json"
    # run_graph_creation(misp_path, to_memgraph=False)
    # run_gnn()
    # run_gnn_evaluation()
    run_gnn_clustering()
    run_metrics_evaluation()
    
    # To run the entire pipeline, uncomment the line below:
    # run_pipeline()