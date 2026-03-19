import json
import os
import numpy as np
from datetime import datetime
from collections import defaultdict
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.metrics import silhouette_score

from feature_set_extraction.cluster_comparison.clusteringCommonFunctions import (
    preprocess_for_clustering,
    record_cluster_id,
    save_clusters_to_json,
    load_ground_truth_from_json,
    remove_outliers_from_matrix,
)

from clustering.clusteringMetrics import compute_silhouette_score, compute_homogeneity_from_clusters

def cluster_with_ids(
    records,
    quantile,
    n_samples,
    max_tfidf_features=None,
    n_components=None,
    remove_outliers=False,
    outlier_contamination=0.05,
):
    idxs = [record_cluster_id(r) for r in records]

    X, feature_names = preprocess_for_clustering(records, max_tfidf_features, n_components=n_components)

    if remove_outliers:
        X, keep_mask, removed = remove_outliers_from_matrix(X, contamination=outlier_contamination)
        idxs = [idx for idx, keep in zip(idxs, keep_mask) if keep]
        print(f"Removed {removed} outliers before Mean Shift (contamination={outlier_contamination})")

    print(f"Feature matrix shape: {X.shape}")
    print(f"Feature range: [{X.min():.4f}, {X.max():.4f}]")
    print(f"Feature mean: {X.mean():.4f}, std: {X.std():.4f}")
    
    bandwidth = estimate_bandwidth(X, quantile=quantile, n_samples=min(n_samples, len(X)))
    print(f"Using bandwidth: {bandwidth:.6f}")
    
    if bandwidth <= 0.0001:
        print(f"WARNING: Bandwidth too small ({bandwidth:.6f}), using manual bandwidth")
        bandwidth = max(X.std() * 0.5, 0.1)
        print(f"Using manual bandwidth: {bandwidth:.4f}")

    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    labels = ms.fit_predict(X)

    clusters = defaultdict(list)
    
    for record_id, label in zip(idxs, labels):
        clusters[label].append(record_id)

    return clusters, labels, X


def compute_silhouette_score(X, labels):
    n_clusters = len(set(labels))
    
    if n_clusters >= 2 and len(labels) > 0:
        silhouette_avg = silhouette_score(X, labels)
        return silhouette_avg
    else:
        return None


def meanshift_cluster_all(
    quantile=0.3,
    n_samples=500,
    max_tfidf_features=None,
    ground_truth_json=None,
    n_components=None,
    remove_outliers=False,
    outlier_contamination=0.05,
    dataset_base="synthetic_email_dataset_50",
):
    # read feature sets from package-local output/featuresets inside core/feature_set_extraction
    package_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    featuresets_dir = os.path.join(package_dir, 'output', 'featuresets')
    
    results_dir = os.path.join(package_dir, 'output', 'fsclusters', 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    ground_truth = None
    if ground_truth_json:
        if not os.path.isabs(ground_truth_json):
            ground_truth_json = os.path.join(package_dir, ground_truth_json)
        if os.path.exists(ground_truth_json):
            print(f"Loading ground truth from: {ground_truth_json}")
            ground_truth = load_ground_truth_from_json(ground_truth_json)
            print(f"Ground truth loaded: {len(ground_truth)} emails in {len(set(ground_truth.values()))} clusters")
        else:
            print(f"Warning: Ground truth file not found: {ground_truth_json}")
    
    scores_file = os.path.join(results_dir, 'meanshift_scores.txt')
    
    feature_sets = ['FS1', 'FS2', 'FS3', 'FS4', 'FS5', 'FS6', 'FS7']#['FS4', 'FS5']
    
    print(f"{'='*80}")
    print(f"Starting Mean Shift clustering on {len(feature_sets)} feature sets...")
    print(
        f"Parameters: quantile={quantile}, n_samples={n_samples}, max_tfidf_features=uncapped, "
        f"n_components={n_components}, remove_outliers={remove_outliers}, "
        f"outlier_contamination={outlier_contamination}"
    )
    print(f"{'='*80}")
    
    with open(scores_file, 'a', encoding='utf-8') as score_f:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        score_f.write("\n" + "="*80 + "\n")
        score_f.write(f"Mean Shift Run - {timestamp}\n")
        score_f.write(
            f"Parameters: quantile={quantile}, n_samples={n_samples}, max_tfidf_features=uncapped, "
            f"n_components={n_components}, remove_outliers={remove_outliers}, "
            f"outlier_contamination={outlier_contamination}\n"
        )
        score_f.write("="*80 + "\n\n")
    
        for fs_name in feature_sets:
            feature_set_path = os.path.join(featuresets_dir, f"{dataset_base}-{fs_name}.json")
            
            if not os.path.exists(feature_set_path):
                print(f"\n✗ Skipping {fs_name}: File not found at {feature_set_path}")
                score_f.write(f"{fs_name}: SKIPPED (file not found)\n")
                continue
            
            print(f"\n{'='*80}")
            print(f"Clustering {fs_name}...")
            print(f"{'='*80}")
            
            with open(feature_set_path, 'r', encoding='utf-8') as f:
                records = json.load(f)
            
            print(f"Loaded {len(records)} records")
            
            clusters, labels, X = cluster_with_ids(
                records,
                quantile,
                n_samples,
                max_tfidf_features,
                n_components,
                remove_outliers=remove_outliers,
                outlier_contamination=outlier_contamination,
            )
            
            silhouette_avg = compute_silhouette_score(X, labels)
            n_clusters = len(set(labels))
            
            if silhouette_avg is not None:
                print(f"Silhouette Score: {silhouette_avg:.4f}")
                silhouette_text = f"{silhouette_avg:.4f}"
            else:
                if n_clusters < 2:
                    print(f"Silhouette Score: N/A (only {n_clusters} cluster(s) found)")
                    silhouette_text = f"N/A (only {n_clusters} cluster(s) found)"
                else:
                    print(f"Silhouette Score: N/A")
                    silhouette_text = "N/A"

            metric_text = f"{fs_name}: silhouette={silhouette_text}, clusters={n_clusters}"
            if ground_truth:
                homogeneity_scores = compute_homogeneity_from_clusters(clusters, ground_truth)
                print(f"Homogeneity: {homogeneity_scores['homogeneity']:.4f}, "
                      f"Completeness: {homogeneity_scores['completeness']:.4f}, "
                      f"V-measure: {homogeneity_scores['v_measure']:.4f} "
                      f"({homogeneity_scores['n_samples']} samples)")
                metric_text += (
                    f", H={homogeneity_scores['homogeneity']:.4f}, "
                    f"C={homogeneity_scores['completeness']:.4f}, "
                    f"V={homogeneity_scores['v_measure']:.4f}, "
                    f"n={homogeneity_scores['n_samples']}"
                )
            score_f.write(metric_text + "\n")
            
            output_path = save_clusters_to_json(clusters, records, feature_set_path, algorithm_name="meanshift")
            
            print(f"\nCluster Summary for {fs_name}:")
            for cluster_id in sorted(clusters.keys()):
                cluster_name = f"cluster_{cluster_id}"
                print(f"  {cluster_name}: {len(clusters[cluster_id])} emails")
            
            print(f"✓ {fs_name} clustering complete")
        

    print(f"\n{'='*80}")
    print("All feature sets clustered successfully!")
    print(f"Combined scores saved to: {scores_file}")
    print(f"{'='*80}")
