import json
from misp.trec_to_misp import csv_to_misp
from graph.graph_builder_pytorch import build_graph
from graph.graph_builder_memgraph import build_memgraph
from graph.graph_filter import NodeType

def run_preprocessing():
    # Placeholder for any preprocessing steps if needed
    pass

def run_trec_misp_converter(csv_path="../data/csv/TREC-07.csv", misp_json_path="data/misp/trec07_misp.json"):
    # input csv file --> Run MISP converter --> output MISP JSON file
    csv_to_misp(csv_path, misp_json_path)

def run_graph_creation(misp_json_path="data/misp/trec07_misp.json", *, to_memgraph: bool = False,
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
    # run_preprocessing()
    run_trec_misp_converter()
    run_graph_creation(to_memgraph=False)
    # run_GNN()
    # run_clustering()
    # run_metrics_evaluation()
    
    # To run the entire pipeline, uncomment the line below:
    # run_pipeline()