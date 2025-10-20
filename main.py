from TRECtoMISP import csv_to_misp

def run_trec_misp_converter(csv_path="data/TREC-07.csv", misp_json_path="data/misp/trec07_misp.json"):
    # Retrieve csv file --> Run MISP converter --> output MISP JSON file
    csv_to_misp(csv_path, misp_json_path)

def run_graph_creation(misp_json_path="data/misp/trec07_misp.json"):
    # MISP JSON file --> Run graph creation --> output PyTorch Geometric graph
    pass

def run_GNN():
    # PyTorch Geometric graph --> Run GNN model on the graph --> output embeddings
    pass

def run_clustering():
    # GNN Embeddings --> Run clustering on embeddings --> output clusters
    pass

def run_metrics_evaluation():
    # Clusters --> Evaluate clustering results using metrics -- > store metrics
    pass

if __name__ == "__main__":
    run_trec_misp_converter()
    # run_graph_creation()
    # run_GNN()
    # run_clustering()
    # run_metrics_evaluation()