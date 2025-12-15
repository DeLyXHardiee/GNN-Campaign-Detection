import json
import sys
import os
from datetime import datetime
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from collections import defaultdict
from clusteringComparison.clusteringCommonFunctions import (
    preprocess_for_clustering, 
    save_clusters_to_json,
    load_ground_truth_from_csv,
    compute_homogeneity_from_clusters
)

def cluster_with_ids(records, eps, min_samples, max_tfidf_features, n_components=None):
    idxs = [r["email_index"] for r in records]

    X, feature_names = preprocess_for_clustering(records, max_tfidf_features, n_components=n_components)

    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X)

    clusters = defaultdict(list)
    
    for record_id, label in zip(idxs, labels):
        clusters[label].append(record_id)

    return clusters, labels, X


def compute_silhouette_score(X, labels):
    non_noise_mask = labels != -1
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    
    if n_clusters >= 2 and non_noise_mask.sum() > 0:
        silhouette_avg = silhouette_score(X[non_noise_mask], labels[non_noise_mask])
        return silhouette_avg
    else:
        return None

def dbscan_cluster_all(eps=2, min_samples=5, max_tfidf_features=500, ground_truth_csv=None, n_components=None):
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    featuresets_dir = os.path.join(project_root, 'data', 'featuresets')
    
    results_dir = os.path.join(project_root, 'data', 'fsclusters')
    os.makedirs(results_dir, exist_ok=True)
    
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
    
    silhouette_file = os.path.join(results_dir, 'dbscan_silhouette_scores.txt')
    homogeneity_file = os.path.join(results_dir, 'dbscan_homogeneity_scores.txt') if ground_truth else None
    
    feature_sets = ['FS1', 'FS2', 'FS3', 'FS4', 'FS5', 'FS6', 'FS7']
    
    print(f"{'='*80}")
    print(f"Starting DBSCAN clustering on {len(feature_sets)} feature sets...")
    print(f"Parameters: eps={eps}, min_samples={min_samples}, max_tfidf_features={max_tfidf_features}, n_components={n_components}")
    print(f"{'='*80}")
    
    with open(silhouette_file, 'a', encoding='utf-8') as sil_f:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        sil_f.write("\n" + "="*80 + "\n")
        sil_f.write(f"DBSCAN Run - {timestamp}\n")
        sil_f.write(f"Parameters: eps={eps}, min_samples={min_samples}, max_tfidf_features={max_tfidf_features}, n_components={n_components}\n")
        sil_f.write("="*80 + "\n\n")
        
        hom_f = None
        if homogeneity_file:
            hom_f = open(homogeneity_file, 'a', encoding='utf-8')
            hom_f.write("\n" + "="*80 + "\n")
            hom_f.write(f"DBSCAN Run - {timestamp}\n")
            hom_f.write(f"Parameters: eps={eps}, min_samples={min_samples}, max_tfidf_features={max_tfidf_features}, n_components={n_components}\n")
            hom_f.write("="*80 + "\n\n")
    
        for fs_name in feature_sets:
            feature_set_path = os.path.join(featuresets_dir, f"TREC-07-misp-{fs_name}.json")
            
            if not os.path.exists(feature_set_path):
                print(f"\n✗ Skipping {fs_name}: File not found at {feature_set_path}")
                sil_f.write(f"{fs_name}: SKIPPED (file not found)\n")
                continue
            
            print(f"\n{'='*80}")
            print(f"Clustering {fs_name}...")
            print(f"{'='*80}")
            
            with open(feature_set_path, 'r', encoding='utf-8') as f:
                records = json.load(f)
            
            print(f"Loaded {len(records)} records")
            
            clusters, labels, X = cluster_with_ids(records, eps, min_samples, max_tfidf_features, n_components)
            
            silhouette_avg = compute_silhouette_score(X, labels)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            
            if silhouette_avg is not None:
                print(f"Silhouette Score (excluding noise): {silhouette_avg:.4f}")
                sil_f.write(f"{fs_name}: {silhouette_avg:.4f} (clusters: {n_clusters})\n")
            else:
                if n_clusters < 2:
                    print(f"Silhouette Score: N/A (only {n_clusters} cluster(s) found)")
                    sil_f.write(f"{fs_name}: N/A (only {n_clusters} cluster(s) found)\n")
                else:
                    print(f"Silhouette Score: N/A (all points are noise)")
                    sil_f.write(f"{fs_name}: N/A (all points are noise)\n")
            
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
            
            output_path = save_clusters_to_json(clusters, records, feature_set_path, algorithm_name="dbscan")
            
            print(f"\nCluster Summary for {fs_name}:")
            for cluster_id in sorted(clusters.keys()):
                cluster_name = "noise" if cluster_id == -1 else f"cluster_{cluster_id}"
                print(f"  {cluster_name}: {len(clusters[cluster_id])} emails")
                if cluster_id > 5:
                    break
            
            print(f"✓ {fs_name} clustering complete")
        
        if hom_f:
            hom_f.close()
    
    print(f"\n{'='*80}")
    print("All feature sets clustered successfully!")
    print(f"Silhouette scores saved to: {silhouette_file}")
    if homogeneity_file:
        print(f"Homogeneity scores saved to: {homogeneity_file}")
    print(f"{'='*80}")