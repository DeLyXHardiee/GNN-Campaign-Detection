from TRECtoMISP import csv_to_misp

def run_trec_misp_converter(csv_path="data/TREC-07.csv", misp_json_path="data/misp/trec07_misp.json"):
    # input csv file --> Run MISP converter --> output MISP JSON file
    csv_to_misp(csv_path, misp_json_path)

def run_graph_creation(misp_json_path="data/misp/trec07_misp.json"):
    # input MISP JSON file --> Run graph creation --> output PyTorch Geometric graph
    pass

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
    run_trec_misp_converter()
    run_graph_creation()
    run_GNN()
    run_clustering()
    run_metrics_evaluation()

if __name__ == "__main__":
    # For individual stages of the pipeline, uncomment as needed:
    run_trec_misp_converter()
    # run_graph_creation()
    # run_GNN()
    # run_clustering()
    # run_metrics_evaluation()
    
    # To run the entire pipeline, uncomment the line below:
    # run_pipeline()