import json
from misp.trec_to_misp import csv_to_misp
from graph.graph_builder_pytorch import build_graph
from graph.graph_builder_memgraph import build_memgraph
from graph.graph_filter import NodeType
from featureNormalization.featureNormalizationTREC import run_featureset_extraction
from clusteringComparison.dbScanComparison import dbscan_cluster_all


def run_preprocessing():
    """
    Run preprocessing steps to filter and prepare data.
    Currently removes non-phishing emails from TREC-07 dataset.
    """
    from utils.filter_csv import filter_phishing_emails
    
    input_csv = "../data/csv/TREC-07.csv"
    output_csv = "../data/csv/TREC-07-only-phishing-6m.csv"
    
    print(f"Filtering phishing emails from {input_csv}...")
    phishing_count = filter_phishing_emails(input_csv, months=6)
    print(f"Filtered dataset saved to {output_csv} ({phishing_count} phishing emails)")

def run_trec_misp_converter(csv_path="../data/csv/TREC-07-only-phishing-6m.csv", misp_json_path="../data/misp/TREC-07-misp.json"):
    # input csv file --> Run MISP converter --> output MISP JSON file
    csv_to_misp(csv_path, misp_json_path)

def run_graph_creation(misp_json_path="../data/misp/TREC-07-misp.json", *, to_memgraph: bool = False,
                       mg_uri: str = "bolt://localhost:7687",
                       mg_user: str | None = None, mg_password: str | None = None):
    # input MISP JSON file --> Run graph creation --> output PyTorch Geometric graph
    # Also optionally mirror it into Memgraph for visualization

    # Filter out selected node types (omit to include everything)
    # Toggle the list below to control which node types to EXCLUDE.
    # e.g., exclude URL + DOMAIN related nodes to build a smaller graph:
    excluded: list[NodeType] = [NodeType.WEEK]  # [NodeType.URL, NodeType.DOMAIN, NodeType.STEM]

    graph, graph_path, meta_path = build_graph(
        misp_json_path=misp_json_path,
        out_dir="results",
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
    # run all of the get_FS functions from feaureNormalizationTrec.py to create feature sets
    run_featureset_extraction()

def run_featureset_clustering():
    dbscan_cluster_all(eps=2, min_samples=5, max_tfidf_features=500, ground_truth_csv="data/groundtruths/pair_votes_lukas.csv")

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
    run_preprocessing()
    run_trec_misp_converter()
    run_graph_creation()
    run_GNN()
    run_clustering()
    run_metrics_evaluation()

if __name__ == "__main__":
    # For individual stages of the pipeline, uncomment as needed:
    run_preprocessing()
    run_trec_misp_converter()
    #run_featureset_extraction()
    #run_featureset_clustering()
    #run_graph_creation(to_memgraph=True)
    # run_GNN()
    # run_clustering()
    # run_metrics_evaluation()
    
    # To run the entire pipeline, uncomment the line below:
    # run_pipeline()