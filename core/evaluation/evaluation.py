"""
Evaluation script for clustering results using homogeneity score.
Compares ground truth labels from manual voting against clustering algorithm results.
"""

import json
import pandas as pd
import numpy as np
from sklearn.metrics import homogeneity_score, completeness_score, v_measure_score
from pathlib import Path


def load_cluster_results(cluster_file_path):
    """
    Load clustering results from JSON file.
    
    Args:
        cluster_file_path: Path to clustering results JSON file
        
    Returns:
        dict: cluster_id -> list of email_indices
    """
    with open(cluster_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    clusters = {}
    for cluster_name, cluster_info in data['clusters'].items():
        # Extract cluster_id (convert "cluster_0" -> 0, "noise" -> -1)
        if cluster_name == "noise":
            cluster_id = -1
        else:
            cluster_id = int(cluster_name.split('_')[1])
        
        clusters[cluster_id] = cluster_info['email_indices']
    
    return clusters


def load_ground_truth(gt_file_path):
    """
    Load ground truth cluster assignments from CSV file.
    
    Args:
        gt_file_path: Path to ground truth CSV file
        
    Returns:
        dict: email_id -> cluster_id (from ground truth)
    """
    df = pd.read_csv(gt_file_path)
    
    # Build mapping: email_id -> ground truth cluster
    gt_labels = {}
    
    # Process each pair vote
    for _, row in df.iterrows():
        email_a = row['email_a_id']
        email_b = row['email_b_id']
        vote = row['vote']  # 'same' or 'different'
        cluster_a = row.get('cluster_a', np.nan)
        cluster_b = row.get('cluster_b', np.nan)
        
        # Use the cluster assignments from the ground truth file
        if not pd.isna(cluster_a):
            gt_labels[email_a] = int(cluster_a)
        if not pd.isna(cluster_b):
            gt_labels[email_b] = int(cluster_b)
    
    return gt_labels


def clusters_to_labels(clusters, email_ids):
    """
    Convert cluster dictionary to label array for sklearn metrics.
    
    Args:
        clusters: dict of cluster_id -> list of email_indices
        email_ids: list of email IDs to get labels for
        
    Returns:
        numpy array of cluster labels for each email_id
    """
    # Create reverse mapping: email_id -> cluster_id
    email_to_cluster = {}
    for cluster_id, email_list in clusters.items():
        for email_id in email_list:
            email_to_cluster[email_id] = cluster_id
    
    # Build label array
    labels = []
    for email_id in email_ids:
        labels.append(email_to_cluster.get(email_id, -1))  # -1 if not found
    
    return np.array(labels)


def compute_homogeneity(ground_truth_labels, predicted_labels):
    """
    Compute homogeneity score and related metrics.
    
    Args:
        ground_truth_labels: dict of email_id -> true cluster
        predicted_labels: dict of cluster_id -> list of email_ids
        
    Returns:
        dict with homogeneity, completeness, and v_measure scores
    """
    # Get common email IDs
    common_email_ids = sorted(set(ground_truth_labels.keys()))
    
    # Convert to label arrays
    true_labels = [ground_truth_labels[eid] for eid in common_email_ids]
    pred_labels_dict = {}
    for cluster_id, email_list in predicted_labels.items():
        for email_id in email_list:
            pred_labels_dict[email_id] = cluster_id
    
    pred_labels = [pred_labels_dict.get(eid, -1) for eid in common_email_ids]
    
    # Compute metrics
    homogeneity = homogeneity_score(true_labels, pred_labels)
    completeness = completeness_score(true_labels, pred_labels)
    v_measure = v_measure_score(true_labels, pred_labels)
    
    return {
        'homogeneity': homogeneity,
        'completeness': completeness,
        'v_measure': v_measure,
        'n_samples': len(common_email_ids)
    }


def evaluate_clustering(cluster_file, ground_truth_file):
    """
    Evaluate clustering results against ground truth.
    
    Args:
        cluster_file: Path to clustering results JSON
        ground_truth_file: Path to ground truth CSV
        
    Returns:
        dict with evaluation metrics
    """
    print(f"\nEvaluating: {Path(cluster_file).name}")
    print("="*80)
    
    # Load data
    clusters = load_cluster_results(cluster_file)
    gt_labels = load_ground_truth(ground_truth_file)
    
    print(f"Loaded {len(clusters)} clusters")
    print(f"Loaded ground truth for {len(gt_labels)} emails")
    
    # Compute metrics
    metrics = compute_homogeneity(gt_labels, clusters)
    
    print(f"\nResults:")
    print(f"  Homogeneity Score:  {metrics['homogeneity']:.4f}")
    print(f"  Completeness Score: {metrics['completeness']:.4f}")
    print(f"  V-Measure Score:    {metrics['v_measure']:.4f}")
    print(f"  Common Samples:     {metrics['n_samples']}")
    
    return metrics


if __name__ == "__main__":
    # Paths
    data_dir = Path("../../data")
    gt_file = data_dir / "groundtruths" / "pair_votes_lukas.csv"
    
    # Find all clustering result files
    featureset_dir = data_dir / "featuresets"
    cluster_files = list(featureset_dir.glob("*_clusters.json"))
    
    print(f"Found {len(cluster_files)} clustering result files")
    
    # Evaluate each clustering result
    results = {}
    for cluster_file in sorted(cluster_files):
        try:
            metrics = evaluate_clustering(cluster_file, gt_file)
            results[cluster_file.name] = metrics
        except Exception as e:
            print(f"Error evaluating {cluster_file.name}: {e}")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"{'File':<50} {'Homogeneity':>12} {'Completeness':>12} {'V-Measure':>12}")
    print("-"*80)
    for filename, metrics in sorted(results.items()):
        print(f"{filename:<50} {metrics['homogeneity']:>12.4f} {metrics['completeness']:>12.4f} {metrics['v_measure']:>12.4f}")
