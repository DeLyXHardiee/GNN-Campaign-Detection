import json
import os
import sys
import numpy as np
from collections import defaultdict
from sklearn.cluster import MeanShift, estimate_bandwidth

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from clusteringCommonFunctions import preprocess_for_clustering, save_clusters_to_json

def cluster_with_ids(records, quantile=0.3, n_samples=500):
    """
    Perform Mean Shift clustering on email records.
    Follows repo's schema-driven pattern with outlier-robust preprocessing.
    
    Args:
        records: List of email feature dictionaries
        quantile: Bandwidth estimation quantile (0.2-0.35, default 0.3)
        n_samples: Number of samples for bandwidth estimation (default 500)
    
    Returns:
        clusters: Dict mapping cluster_id -> list of email_indices
        labels: Array of cluster labels for each email
    """
    # Extract IDs
    idxs = [r["email_index"] for r in records]

    # Preprocess with outlier clipping (following "small numeric features" principle)
    X, feature_names = preprocess_for_clustering(records)

    # DIAGNOSTIC: Check feature statistics
    print(f"\n--- Feature Statistics ---")
    print(f"Feature matrix shape: {X.shape}")
    print(f"Mean: {X.mean():.4f}, Std: {X.std():.4f}")
    print(f"Range: [{X.min():.2f}, {X.max():.2f}]")

    # Estimate bandwidth
    bandwidth = estimate_bandwidth(X, quantile=quantile, n_samples=min(n_samples, len(X)))
    print(f"Estimated bandwidth: {bandwidth:.4f}")

    # Run Mean Shift
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    labels = ms.fit_predict(X)

    # Build cluster → ID list map
    clusters = defaultdict(list)
    
    for record_id, label in zip(idxs, labels):
        clusters[label].append(record_id)

    return clusters, labels


# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # Use pre-computed feature set (schema-driven from Graph IR pipeline)
    feature_set_path = "../../data/featuresets/TREC-07-misp-FS4.json"

    with open(feature_set_path, 'r', encoding='utf-8') as f:
        records = json.load(f)[:1000]
    
    # Experiment with different quantiles
    print("--- Testing different quantile values ---")
    for q in [0.25, 0.3, 0.35]:
        print(f"\n=== Quantile: {q} ===")
        clusters, labels = cluster_with_ids(records, quantile=q)
        
        # Analyze cluster distribution
        cluster_sizes = sorted([len(v) for v in clusters.values()], reverse=True)
        print(f"\nCluster sizes: {cluster_sizes[:10]}")
        print(f"Total clusters: {len(clusters)}")
        print(f"Singletons: {sum(1 for size in cluster_sizes if size == 1)}")
    
    # Use best quantile for final output
    print("\n=== Final Clustering (quantile=0.3) ===")
    clusters, labels = cluster_with_ids(records, quantile=0.3)
    output_path = save_clusters_to_json(clusters, records, feature_set_path, algorithm_name="meanshift")
    
    # Print summary
    print(f"\nCluster Summary:")
    for cluster_id in sorted(clusters.keys()):
        cluster_name = f"cluster_{cluster_id}"
        print(f"{cluster_name}: {len(clusters[cluster_id])} emails")