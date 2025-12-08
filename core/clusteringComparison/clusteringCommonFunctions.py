"""
Common preprocessing and output functions for clustering algorithms.
Following repo's schema-driven pattern: single source of truth for feature preprocessing.
"""

import json
import os
import csv
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import homogeneity_score, completeness_score, v_measure_score


def preprocess_for_clustering(records, max_tfidf_features, text_fields=None, exclude_fields=None):
    """
    Preprocess email records for clustering algorithms (DBSCAN, Mean Shift, etc.).
    Automatically detects numeric and text features, applies TF-IDF to text fields separately.
    
    Args:
        records: List of email feature dictionaries (can have varying schemas)
        max_tfidf_features: int - maximum number of TF-IDF features per text field
        text_fields: list of str - specific text field names to vectorize (auto-detect if None)
        exclude_fields: list of str - fields to exclude from feature extraction
    
    Returns:
        X: numpy array of shape (n_samples, n_features) - feature matrix
        feature_names: list of feature names corresponding to columns
    
    Notes:
        - Auto-detects numeric fields (int, float) and text fields (str)
        - Applies TF-IDF separately to each text field for better granularity
        - Returns standardized features using RobustScaler (robust to outliers)
        - Compatible with sklearn clustering algorithms
    """
    if not records:
        raise ValueError("Empty records list")
    
    # Default exclude fields (identifiers, dates that shouldn't be clustered on)
    if exclude_fields is None:
        exclude_fields = ['email_index']
    
    # Analyze first record to determine field types
    sample_record = records[0]
    numeric_fields = []
    detected_text_fields = []
    
    for key, value in sample_record.items():
        if key in exclude_fields:
            continue
            
        # Check field type
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            numeric_fields.append(key)
        elif isinstance(value, str) and len(value) > 0:
            # Only consider non-empty strings as potential text fields
            detected_text_fields.append(key)
    
    # Use provided text_fields or auto-detected ones
    if text_fields is None:
        text_fields = detected_text_fields
    else:
        # Filter to only text fields that exist in records
        text_fields = [f for f in text_fields if f in detected_text_fields]
    
    print(f"Detected {len(numeric_fields)} numeric fields: {numeric_fields[:5]}...")
    print(f"Using {len(text_fields)} text fields for TF-IDF: {text_fields}")
    
    # Extract numeric features
    X_numeric = []
    for record in records:
        features = []
        for fname in numeric_fields:
            # Get feature value, default to 0.0 if missing
            features.append(float(record.get(fname, 0.0)))
        X_numeric.append(features)
    
    X_numeric = np.array(X_numeric)
    
    # Start with numeric features
    feature_parts = [X_numeric]
    feature_names = numeric_fields.copy()
    
    # Add TF-IDF features for each text field separately
    for text_field in text_fields:
        texts = [str(record.get(text_field, '')) for record in records]
        
        # Skip if all texts are empty
        if all(len(t.strip()) == 0 for t in texts):
            print(f"  Skipping '{text_field}': all empty")
            continue
        
        try:
            tfidf = TfidfVectorizer(
                max_features=max_tfidf_features,
                stop_words='english',
                min_df=2,          # Ignore terms appearing in < 2 documents
                max_df=0.8,        # Ignore terms appearing in > 80% of documents
                ngram_range=(1, 2) # Unigrams and bigrams
            )
            X_text = tfidf.fit_transform(texts).toarray()
            
            if X_text.shape[1] > 0:
                feature_parts.append(X_text)
                feature_names.extend([f"{text_field}_tfidf_{i}" for i in range(X_text.shape[1])])
                print(f"  {text_field}: extracted {X_text.shape[1]} TF-IDF features")
            else:
                print(f"  Skipping '{text_field}': no features extracted")
        except Exception as e:
            print(f"  Error processing '{text_field}': {e}")
    
    # Combine all features
    X = np.hstack(feature_parts)
    
    print(f"\nTotal features: {X.shape[1]} ({len(numeric_fields)} numeric + {X.shape[1] - len(numeric_fields)} TF-IDF)")
    
    # Standardize features using RobustScaler (robust to outliers in phishing data)
    scaler = RobustScaler()
    X = scaler.fit_transform(X)
    
    return X, feature_names


def save_clusters_to_json(clusters, records, feature_set_path, algorithm_name="dbscan"):
    """
    Save cluster results to JSON file with full record details.
    Common function for DBSCAN, Mean Shift, and other clustering algorithms.
    
    Args:
        clusters: Dict mapping cluster_id -> list of email_indices
        records: List of email feature dictionaries
        feature_set_path: Path to input feature set JSON
        algorithm_name: Name of clustering algorithm (e.g., "dbscan", "meanshift")
    
    Returns:
        output_path: Path to saved JSON file
    
    Notes:
        - DBSCAN: cluster_id -1 is treated as "noise"
        - Mean Shift: all points assigned to clusters (no noise)
        - Output saved to data/fsclusters/ directory with _{algorithm}_clusters.json suffix
    """
    # Create output path in fsclusters folder
    # Get project root (going up from core/clusteringComparison)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    output_dir = os.path.join(project_root, 'data', 'fsclusters')
    
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    input_base = os.path.splitext(os.path.basename(feature_set_path))[0]
    output_path = os.path.join(output_dir, f"{input_base}_{algorithm_name}_clusters.json")
    
    # Create lookup dict for email_index -> record
    record_lookup = {r["email_index"]: r for r in records}
    
    # Build cluster data with full record details
    has_noise = -1 in clusters  # DBSCAN has noise cluster, Mean Shift doesn't
    
    cluster_data = {
        "metadata": {
            "total_emails": len(records),
            "num_clusters": len([c for c in clusters.keys() if c != -1]),
            "algorithm": algorithm_name,
            "feature_set_source": feature_set_path
        },
        "clusters": {}
    }
    
    # Add noise point count if applicable (DBSCAN)
    if has_noise:
        cluster_data["metadata"]["noise_points"] = len(clusters.get(-1, []))
    
    for cluster_id, email_indices in clusters.items():
        cluster_name = "noise" if cluster_id == -1 else f"cluster_{cluster_id}"
        
        cluster_data["clusters"][cluster_name] = {
            "size": len(email_indices),
            "email_indices": email_indices
        }
        
        # Only add email details for actual clusters (not noise)
        if cluster_id != -1:
            cluster_data["clusters"][cluster_name]["emails"] = []
            
            # Add full record details for each email in cluster
            for email_idx in email_indices:
                if email_idx in record_lookup:
                    email_record = record_lookup[email_idx].copy()
                    cluster_data["clusters"][cluster_name]["emails"].append(email_record)
    
    # Save to JSON
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(cluster_data, f, indent=2, ensure_ascii=False)
    
    print(f"Saved cluster results to: {output_path}")
    return output_path


def load_ground_truth_from_csv(csv_path):
    """
    Load ground truth cluster assignments from pairwise voting CSV.
    
    Args:
        csv_path: Path to CSV file with columns: timestamp, uuid, email_a_id, email_b_id, vote, ...
                  Vote can be 'same_campaign' or 'unrelated'
    
    Returns:
        dict: Mapping of email_id (int) -> cluster_id (int)
              Emails voted as 'same_campaign' get the same cluster_id
    
    Notes:
        - Uses union-find algorithm to group emails into clusters based on votes
        - Only considers 'same_campaign' votes (unrelated votes are ignored)
    """
    # Read pairwise votes
    pairs_same_campaign = []
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 5:
                continue
            
            try:
                email_a = int(row[2])
                email_b = int(row[3])
                vote = row[4].strip().lower()
                
                if vote == 'same_campaign':
                    pairs_same_campaign.append((email_a, email_b))
            except (ValueError, IndexError):
                continue
    
    # Build clusters using union-find
    parent = {}
    
    def find(x):
        if x not in parent:
            parent[x] = x
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(x, y):
        root_x = find(x)
        root_y = find(y)
        if root_x != root_y:
            parent[root_y] = root_x
    
    # Union all pairs marked as same_campaign
    for email_a, email_b in pairs_same_campaign:
        union(email_a, email_b)
    
    # Assign cluster IDs
    # Is this correct? The cluster ID's will be random and not correspond to the cluster ID's from the DBSCAN
    email_to_cluster = {}
    cluster_id_map = {}
    next_cluster_id = 0
    
    for email_id in parent.keys():
        root = find(email_id)
        if root not in cluster_id_map:
            cluster_id_map[root] = next_cluster_id
            next_cluster_id += 1
        email_to_cluster[email_id] = cluster_id_map[root]
    
    return email_to_cluster


def compute_homogeneity_from_clusters(clusters, ground_truth):
    """
    Compute homogeneity, completeness, and V-measure scores.
    
    Args:
        clusters: Dict mapping cluster_id -> list of email_indices (from clustering algorithm)
        ground_truth: Dict mapping email_id -> true_cluster_id (from ground truth CSV)
    
    Returns:
        dict: {'homogeneity': float, 'completeness': float, 'v_measure': float}
              All scores range from 0 to 1 (higher is better)
    
    Notes:
        - Only evaluates emails that appear in both clustering results AND ground truth
        - Homogeneity: Each cluster contains only members of a single class
        - Completeness: All members of a given class are assigned to the same cluster
        - V-measure: Harmonic mean of homogeneity and completeness
    """
    # Build predicted labels array
    email_to_predicted_cluster = {}
    for cluster_id, email_indices in clusters.items():
        for email_idx in email_indices:
            email_to_predicted_cluster[email_idx] = cluster_id
    
    # Find common emails between predictions and ground truth
    common_emails = set(email_to_predicted_cluster.keys()) & set(ground_truth.keys())
    
    if len(common_emails) < 2:
        return {'homogeneity': 0.0, 'completeness': 0.0, 'v_measure': 0.0, 'n_samples': len(common_emails)}
    
    # Create aligned label arrays
    common_emails = sorted(common_emails)
    predicted_labels = [email_to_predicted_cluster[e] for e in common_emails]
    true_labels = [ground_truth[e] for e in common_emails]
    
    # Compute scores
    homogeneity = homogeneity_score(true_labels, predicted_labels)
    completeness = completeness_score(true_labels, predicted_labels)
    v_measure = v_measure_score(true_labels, predicted_labels)
    
    return {
        'homogeneity': homogeneity,
        'completeness': completeness,
        'v_measure': v_measure,
        'n_samples': len(common_emails)
    }
