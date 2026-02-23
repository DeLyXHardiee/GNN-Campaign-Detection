import json
from pathlib import Path
from preprocessing.data_parser import parse_incidents_with_email_bodies
from preprocessing.misp_converter import incidents_to_misp_file


def run_preprocessing(limit: int | None = None):
    """
    Parse incident metadata and email body files, then convert to MISP JSON.
    """
    project_root = Path(__file__).resolve().parent.parent
    incidents_csv_path = project_root / "data" / "incidents_202602111435.csv"
    bodies_dir = project_root / "data" / "POTENTIALLY MALICIOUS_phishing_emails_last_three_months_11_2_2026"
    misp_json_path = project_root / "data" / "misp" / "incidents-20260211-misp.json"

    print(f"Parsing incidents from {incidents_csv_path}...")
    incidents = parse_incidents_with_email_bodies(
        str(incidents_csv_path),
        str(bodies_dir),
        limit=limit,
    )
    print(f"Parsed {len(incidents)} incidents with matched email body content.")

    print(f"Converting incidents to MISP and writing secure output at {misp_json_path}...")
    incidents_to_misp_file(incidents, str(misp_json_path))
    print("MISP conversion complete.")
    return str(misp_json_path)

def run_graph_creation(misp_json_path="../data/misp/incidents-20260211-misp.json", *, to_memgraph: bool = False,
                       mg_uri: str = "bolt://localhost:7687",
                       mg_user: str | None = None, mg_password: str | None = None):
    # input MISP JSON file --> Run graph creation --> output PyTorch Geometric graph
    # Also optionally mirror it into Memgraph for visualization
    from graph.graph_builder_pytorch import build_graph
    from graph.graph_builder_memgraph import build_memgraph
    from graph.graph_filter import NodeType

    # Filter out selected node types (omit to include everything)
    # Toggle the list below to control which node types to EXCLUDE.
    # e.g., exclude URL + DOMAIN related nodes to build a smaller graph:
    excluded: list[NodeType] = [NodeType.WEEK]  # [NodeType.URL, NodeType.DOMAIN, NodeType.STEM]

    graph, graph_path, meta_path = build_graph(
        misp_json_path=misp_json_path,
        out_dir="graph/output",
        exclude_nodes=excluded,
    )
    print(f"Graph created: {graph}")
    print(f"Saved graph to: {graph_path}")
    print(f"Saved metadata to: {meta_path}")

    if to_memgraph:
        summary = build_memgraph(
            misp_json_path=misp_json_path,
            mg_uri=mg_uri,
            mg_user=mg_user,
            mg_password=mg_password,
            clear=True,
            create_indexes=True,
            exclude_nodes=excluded,
        )
        print("Memgraph load summary:")
        print(json.dumps(summary, indent=2))

    return graph

def create_feature_sets():
    from feature_set_extraction.feature_set_extraction import run_featureset_extraction
    run_featureset_extraction()

def run_featureset_clustering():
    """
    Run DBSCAN and Mean Shift clustering with grid search over parameters.
    Tests different parameter combinations to find optimal homogeneity scores.
    """
    from feature_set_extraction.cluster_comparison.dbScanComparison import dbscan_cluster_all
    from feature_set_extraction.cluster_comparison.meanshiftComparison import meanshift_cluster_all

    ground_truth_csv = "data/groundtruths/campaigns.csv"
    
    eps_values = [0.5, 1, 1.5, 2, 2.5]
    tfidf_values = [500,1000,5000]
    min_samples = 5
    n_components = 200
    
    #eps_values = [1.5]
    #tfidf_values = [5000]
    #min_samples = 2
    
    print(f"{'='*80}")
    print(f"DBSCAN Parameter Grid Search")
    print(f"{'='*80}")
    print(f"Testing {len(eps_values)} eps values: {eps_values}")
    print(f"Testing {len(tfidf_values)} TF-IDF features: {tfidf_values}")
    print(f"Total configurations: {len(eps_values) * len(tfidf_values)}")
    print(f"{'='*80}\n")
    
    for eps in eps_values:
        for max_tfidf in tfidf_values:
            print(f"\n{'='*80}")
            print(f"Testing: eps={eps}, max_tfidf_features={max_tfidf}, min_samples={min_samples}")
            print(f"{'='*80}")
            
            dbscan_cluster_all(
                eps=eps, 
                min_samples=min_samples, 
                max_tfidf_features=max_tfidf, 
                #n_components=n_components,
                ground_truth_csv=ground_truth_csv
            )
    
    print(f"\n{'='*80}")
    print(f"DBSCAN grid search complete!")
    print(f"Results saved to data/fsclusters/dbscan_*_scores.txt")
    print(f"{'='*80}\n")
    
    quantile_values = [0.2,0.25,0.3]
    tfidf_values_ms = [500,1000,5000]
    #n_components = 200
    n_samples = 500 
    
    print(f"\n{'='*80}")
    print(f"Mean Shift Parameter Grid Search")
    print(f"{'='*80}")
    print(f"Testing {len(quantile_values)} bandwidth values: {quantile_values}")
    print(f"Testing {len(tfidf_values_ms)} TF-IDF features: {tfidf_values_ms}")
    print(f"Total configurations: {len(quantile_values) * len(tfidf_values_ms)}")
    print(f"{'='*80}\n")
    
    for quantile in quantile_values:
        for max_tfidf in tfidf_values_ms:
            print(f"\n{'='*80}")
            print(f"Testing: quantile={quantile}, max_tfidf_features={max_tfidf}, n_samples={n_samples}")
            print(f"{'='*80}")
            
            meanshift_cluster_all(
                quantile=quantile,
                n_samples=n_samples,
                max_tfidf_features=max_tfidf,
                #n_components=n_components,
                ground_truth_csv=ground_truth_csv
            )
    
    print(f"\n{'='*80}")
    print(f"Mean Shift grid search complete!")
    print(f"Results saved to data/fsclusters/meanshift_*_scores.txt")
    print(f"{'='*80}\n")
    
    print(f"\n{'='*80}")
    print(f"All grid searches complete!")
    print(f"Review homogeneity scores to find optimal parameters:")
    print(f"  - DBSCAN: data/fsclusters/dbscan_homogeneity_scores.txt")
    print(f"  - Mean Shift: data/fsclusters/meanshift_homogeneity_scores.txt")
    print(f"{'='*80}")

def run_GNN():
    # input PyTorch Geometric graph --> Run GNN model on the graph --> output embeddings
    pass

def run_clustering():
    # input GNN Embeddings --> Run clustering on embeddings --> output clusters
    pass

def run_metrics_evaluation():
    # input Clusters --> Evaluate clustering results using metrics -- > store metrics
    pass

def run_pipeline():
    misp_json_path = run_preprocessing()
    run_graph_creation(misp_json_path)
    run_GNN()
    run_clustering()
    run_metrics_evaluation()

if __name__ == "__main__":
    # For individual stages of the pipeline, uncomment as needed:
    # misp_path = run_preprocessing()
    # create_feature_sets()
    # run_featureset_clustering()
    misp_path = "../data/misp/incidents-20260211-misp.json"
    run_graph_creation(misp_path, to_memgraph=False)
    # run_GNN()
    # run_clustering()
    # run_metrics_evaluation()
    
    # To run the entire pipeline, uncomment the line below:
    # run_pipeline()