from TRECtoMISP import csv_to_misp

def run_trec_misp_converter():
    # Run MISP converter --> output MISP JSON file
    csv_to_misp("data/TREC-07.csv", "data/misp/trec07_misp.json")

def run_graph_creation():
    # Run graph creation --> output PyTorch Geometric graph
    pass

def run_GNN():
    # Run GNN model on the graph --> output embeddings
    pass

def run_clustering():
    # Run clustering on embeddings --> store clusters
    pass

if __name__ == "__main__":
    run_trec_misp_converter()
    # run_graph_creation()
    # run_GNN()
    # run_clustering()