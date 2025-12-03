import json
import sys
import os
from sklearn.cluster import DBSCAN
from collections import defaultdict
from clusteringCommonFunctions import preprocess_for_clustering, save_clusters_to_json

def cluster_with_ids(records, eps=1, min_samples=5):
    # Extract IDs
    idxs = [r["email_index"] for r in records]

    # Preprocess to numeric features (common function)
    X, feature_names = preprocess_for_clustering(records)

    # Run DBSCAN with configurable parameters
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X)

    # Build cluster → ID list map
    clusters = defaultdict(list)
    
    for record_id, label in zip(idxs, labels):
        clusters[label].append(record_id)

    return clusters, labels


# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # Use pre-computed feature set instead of extracting from CSV/MISP
    feature_set_path = "../../data/featuresets/TREC-07-misp-FS4.json"

    records = []

    with open(feature_set_path, 'r', encoding='utf-8') as f:
        records = json.load(f)[:10000]
        
    # Run clustering
    clusters, labels = cluster_with_ids(records)
    
    
    # Save clusters to JSON file (use feature set path for output naming)
    output_path = save_clusters_to_json(clusters, records, feature_set_path, algorithm_name="dbscan")
    
    # Print summary
    print(f"\nCluster Summary:")
    for cluster_id in sorted(clusters.keys()):
        cluster_name = "noise" if cluster_id == -1 else f"cluster_{cluster_id}"
        print(f"{cluster_name}: {len(clusters[cluster_id])} emails")
    '''
    '''