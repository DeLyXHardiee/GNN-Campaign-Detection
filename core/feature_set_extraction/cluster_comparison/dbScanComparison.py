import json
import os
from datetime import datetime
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from collections import defaultdict
from feature_set_extraction.cluster_comparison.clusteringCommonFunctions import (
    preprocess_for_clustering, 
    save_clusters_to_json,
    load_ground_truth_from_csv,
    remove_outliers_from_matrix,
)

from clustering.clusteringMetrics import compute_silhouette_score, compute_homogeneity_from_clusters

def cluster_with_ids(
    records,
    eps,
    min_samples,
    max_tfidf_features=10000,
    n_components=None,
    remove_outliers=False,
    outlier_contamination=0.05,
):
    idxs = [r["email_index"] for r in records]

    X, feature_names = preprocess_for_clustering(records, max_tfidf_features, n_components=n_components)

    if remove_outliers:
        X, keep_mask, removed = remove_outliers_from_matrix(X, contamination=outlier_contamination)
        idxs = [idx for idx, keep in zip(idxs, keep_mask) if keep]
        print(f"Removed {removed} outliers before DBSCAN (contamination={outlier_contamination})")

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

def dbscan_cluster_all(
    eps=2,
    min_samples=5,
    max_tfidf_features=None,
    ground_truth_csv=None,
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
    if ground_truth_csv:
        if not os.path.isabs(ground_truth_csv):
            ground_truth_csv = os.path.join(package_dir, ground_truth_csv)
        if os.path.exists(ground_truth_csv):
            print(f"Loading ground truth from: {ground_truth_csv}")
            ground_truth = load_ground_truth_from_csv(ground_truth_csv)
            print(f"Ground truth loaded: {len(ground_truth)} emails in {len(set(ground_truth.values()))} clusters")
        else:
            print(f"Warning: Ground truth file not found: {ground_truth_csv}")
    
    scores_file = os.path.join(results_dir, 'dbscan_scores.txt')
    
    feature_sets = ['FS4', 'FS5']#['FS1', 'FS2', 'FS3', 'FS4', 'FS5', 'FS6', 'FS7']
    
    print(f"{'='*80}")
    print(f"Starting DBSCAN clustering on {len(feature_sets)} feature sets...")
    print(
        f"Parameters: eps={eps}, min_samples={min_samples}, max_tfidf_features=uncapped, "
        f"n_components={n_components}, remove_outliers={remove_outliers}, "
        f"outlier_contamination={outlier_contamination}"
    )
    print(f"{'='*80}")
    
    with open(scores_file, 'a', encoding='utf-8') as score_f:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        score_f.write("\n" + "="*80 + "\n")
        score_f.write(f"DBSCAN Run - {timestamp}\n")
        score_f.write(
            f"Parameters: eps={eps}, min_samples={min_samples}, max_tfidf_features=uncapped, "
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
                #cap at 5000 records
                records = json.load(f)[:5000]
            
            print(f"Loaded {len(records)} records")
            
            clusters, labels, X = cluster_with_ids(
                records,
                eps,
                min_samples,
                max_tfidf_features,
                n_components,
                remove_outliers=remove_outliers,
                outlier_contamination=outlier_contamination,
            )
            
            silhouette_avg = compute_silhouette_score(X, labels)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            
            if silhouette_avg is not None:
                print(f"Silhouette Score (excluding noise): {silhouette_avg:.4f}")
                silhouette_text = f"{silhouette_avg:.4f}"
            else:
                if n_clusters < 2:
                    print(f"Silhouette Score: N/A (only {n_clusters} cluster(s) found)")
                    silhouette_text = f"N/A (only {n_clusters} cluster(s) found)"
                else:
                    print(f"Silhouette Score: N/A (all points are noise)")
                    silhouette_text = "N/A (all points are noise)"

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
            
            output_path = save_clusters_to_json(clusters, records, feature_set_path, algorithm_name="dbscan")
            
            print(f"\nCluster Summary for {fs_name}:")
            for cluster_id in sorted(clusters.keys()):
                cluster_name = "noise" if cluster_id == -1 else f"cluster_{cluster_id}"
                print(f"  {cluster_name}: {len(clusters[cluster_id])} emails")
                if cluster_id > 5:
                    break
            
            print(f"✓ {fs_name} clustering complete")
        

    print(f"\n{'='*80}")
    print("All feature sets clustered successfully!")
    print(f"Combined scores saved to: {scores_file}")
    print(f"{'='*80}")