import json
import os
import sys
import numpy as np
from datetime import datetime
from collections import defaultdict
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.metrics import silhouette_score

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from clusteringCommonFunctions import (
    preprocess_for_clustering, 
    save_clusters_to_json,
    load_ground_truth_from_csv,
    compute_homogeneity_from_clusters
)

def cluster_with_ids(records, quantile, n_samples, max_tfidf_features, n_components=None):
    """
    Perform Mean Shift clustering on email records.
    Follows repo's schema-driven pattern with outlier-robust preprocessing.
    
    Args:
        records: List of email feature dictionaries
        bandwidth: Bandwidth parameter for Mean Shift
        n_samples: Number of samples for bandwidth estimation (default 500)
        max_tfidf_features: Maximum TF-IDF features per text field
        n_components: Number of SVD components for dimensionality reduction (None = no reduction)
    
    Returns:
        clusters: Dict mapping cluster_id -> list of email_indices
        labels: Array of cluster labels for each email
        X: Feature matrix (for silhouette score computation)
    """
    # Extract IDs
    idxs = [r["email_index"] for r in records]

    # Preprocess with outlier clipping (following "small numeric features" principle)
    X, feature_names = preprocess_for_clustering(records, max_tfidf_features, n_components=n_components)

    # Diagnostic: Check feature statistics
    print(f"Feature matrix shape: {X.shape}")
    print(f"Feature range: [{X.min():.4f}, {X.max():.4f}]")
    print(f"Feature mean: {X.mean():.4f}, std: {X.std():.4f}")
    
    # Use provided bandwidth
    bandwidth = estimate_bandwidth(X, quantile=quantile, n_samples=min(n_samples, len(X)))
    print(f"Using bandwidth: {bandwidth:.6f}")
    
    # Handle zero bandwidth (data too similar after scaling)
    if bandwidth <= 0.0001:
        print(f"WARNING: Bandwidth too small ({bandwidth:.6f}), using manual bandwidth")
        # Use a fraction of the feature space range or standard deviation
        bandwidth = max(X.std() * 0.5, 0.1)
        print(f"Using manual bandwidth: {bandwidth:.4f}")

    # Run Mean Shift
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    labels = ms.fit_predict(X)

    # Build cluster → ID list map
    clusters = defaultdict(list)
    
    for record_id, label in zip(idxs, labels):
        clusters[label].append(record_id)

    return clusters, labels, X


def compute_silhouette_score(X, labels):
    """
    Compute silhouette score for clustering results.
    
    Args:
        X: Feature matrix (n_samples, n_features)
        labels: Cluster labels for each sample
    
    Returns:
        silhouette_avg: Average silhouette score (float) or None if not computable
    
    Notes:
        - Requires at least 2 clusters
        - Returns None if score cannot be computed
    """
    n_clusters = len(set(labels))
    
    if n_clusters >= 2 and len(labels) > 0:
        silhouette_avg = silhouette_score(X, labels)
        return silhouette_avg
    else:
        return None


def meanshift_cluster_all(quantile=0.3, n_samples=500, max_tfidf_features=500, ground_truth_csv=None, n_components=None):
    """
    Run Mean Shift clustering on all feature sets with automatic metrics computation.
    
    Args:
        quantile: quantile parameter for Mean Shift
        n_samples: Number of samples for quantile estimation
        max_tfidf_features: Maximum TF-IDF features per text field
        ground_truth_csv: Optional path to ground truth CSV for homogeneity computation
        n_components: Number of SVD components for dimensionality reduction (None = no reduction)
    """
    # Get project root (two levels up from this file: core/clusteringComparison -> core -> project)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    featuresets_dir = os.path.join(project_root, 'data', 'featuresets')
    
    # Create output directory for results
    results_dir = os.path.join(project_root, 'data', 'fsclusters')
    os.makedirs(results_dir, exist_ok=True)
    
    # Load ground truth if provided
    ground_truth = None
    if ground_truth_csv:
        if not os.path.isabs(ground_truth_csv):
            ground_truth_csv = os.path.join(project_root, ground_truth_csv)
        if os.path.exists(ground_truth_csv):
            print(f"Loading ground truth from: {ground_truth_csv}")
            ground_truth = load_ground_truth_from_csv(ground_truth_csv)
            print(f"Ground truth loaded: {len(ground_truth)} emails in {len(set(ground_truth.values()))} clusters")
        else:
            print(f"Warning: Ground truth file not found: {ground_truth_csv}")
    
    # Create output files (append mode to preserve history)
    silhouette_file = os.path.join(results_dir, 'meanshift_silhouette_scores.txt')
    homogeneity_file = os.path.join(results_dir, 'meanshift_homogeneity_scores.txt') if ground_truth else None
    
    # Define all feature sets to cluster
    feature_sets = ['FS1', 'FS2', 'FS3', 'FS4', 'FS5', 'FS6', 'FS7']
    
    print(f"{'='*80}")
    print(f"Starting Mean Shift clustering on {len(feature_sets)} feature sets...")
    print(f"Parameters: quantile={quantile}, n_samples={n_samples}, max_tfidf_features={max_tfidf_features}, n_components={n_components}")
    print(f"{'='*80}")
    
    # Open output files in append mode
    with open(silhouette_file, 'a', encoding='utf-8') as sil_f:
        # Write run header with timestamp and parameters
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        sil_f.write("\n" + "="*80 + "\n")
        sil_f.write(f"Mean Shift Run - {timestamp}\n")
        sil_f.write(f"Parameters: quantile={quantile}, n_samples={n_samples}, max_tfidf_features={max_tfidf_features}, n_components={n_components}\n")
        sil_f.write("="*80 + "\n\n")
        
        # Open homogeneity file if ground truth is available
        hom_f = None
        if homogeneity_file:
            hom_f = open(homogeneity_file, 'a', encoding='utf-8')
            hom_f.write("\n" + "="*80 + "\n")
            hom_f.write(f"Mean Shift Run - {timestamp}\n")
            hom_f.write(f"Parameters: quantile={quantile}, n_samples={n_samples}, max_tfidf_features={max_tfidf_features}, n_components={n_components}\n")
            hom_f.write("="*80 + "\n\n")
    
        for fs_name in feature_sets:
            feature_set_path = os.path.join(featuresets_dir, f"TREC-07-misp-{fs_name}.json")
            
            # Check if file exists
            if not os.path.exists(feature_set_path):
                print(f"\n✗ Skipping {fs_name}: File not found at {feature_set_path}")
                sil_f.write(f"{fs_name}: SKIPPED (file not found)\n")
                continue
            
            print(f"\n{'='*80}")
            print(f"Clustering {fs_name}...")
            print(f"{'='*80}")
            
            # Load records
            with open(feature_set_path, 'r', encoding='utf-8') as f:
                records = json.load(f)
            
            print(f"Loaded {len(records)} records")
            
            # Run clustering
            clusters, labels, X = cluster_with_ids(records, quantile, n_samples, max_tfidf_features, n_components)
            
            # Compute silhouette score
            silhouette_avg = compute_silhouette_score(X, labels)
            n_clusters = len(set(labels))
            
            if silhouette_avg is not None:
                print(f"Silhouette Score: {silhouette_avg:.4f}")
                sil_f.write(f"{fs_name}: {silhouette_avg:.4f} (clusters: {n_clusters})\n")
            else:
                if n_clusters < 2:
                    print(f"Silhouette Score: N/A (only {n_clusters} cluster(s) found)")
                    sil_f.write(f"{fs_name}: N/A (only {n_clusters} cluster(s) found)\n")
                else:
                    print(f"Silhouette Score: N/A")
                    sil_f.write(f"{fs_name}: N/A\n")
            
            # Compute homogeneity score if ground truth available
            if ground_truth and hom_f:
                homogeneity_scores = compute_homogeneity_from_clusters(clusters, ground_truth)
                print(f"Homogeneity: {homogeneity_scores['homogeneity']:.4f}, "
                      f"Completeness: {homogeneity_scores['completeness']:.4f}, "
                      f"V-measure: {homogeneity_scores['v_measure']:.4f} "
                      f"({homogeneity_scores['n_samples']} samples)")
                hom_f.write(f"{fs_name}: H={homogeneity_scores['homogeneity']:.4f}, "
                           f"C={homogeneity_scores['completeness']:.4f}, "
                           f"V={homogeneity_scores['v_measure']:.4f} "
                           f"(n={homogeneity_scores['n_samples']}, clusters={n_clusters})\n")
            
            # Save clusters to JSON file
            output_path = save_clusters_to_json(clusters, records, feature_set_path, algorithm_name="meanshift")
            
            # Print summary
            print(f"\nCluster Summary for {fs_name}:")
            for cluster_id in sorted(clusters.keys()):
                cluster_name = f"cluster_{cluster_id}"
                print(f"  {cluster_name}: {len(clusters[cluster_id])} emails")
            
            print(f"✓ {fs_name} clustering complete")
        
        # Close homogeneity file if opened
        if hom_f:
            hom_f.close()
    
    print(f"\n{'='*80}")
    print("All feature sets clustered successfully!")
    print(f"Silhouette scores saved to: {silhouette_file}")
    if homogeneity_file:
        print(f"Homogeneity scores saved to: {homogeneity_file}")
    print(f"{'='*80}")
