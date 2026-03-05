import json
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from preprocessing.utils.defang import sanitize_for_json


def preprocess_for_clustering(records, max_tfidf_features, text_fields=None, exclude_fields=None, n_components=None):

    if not records:
        raise ValueError("Empty records list")
    
    if exclude_fields is None:
        exclude_fields = ['email_index']
    
    sample_record = records[0]
    numeric_fields = []
    detected_text_fields = []
    
    for key, value in sample_record.items():
        if key in exclude_fields:
            continue
            
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            numeric_fields.append(key)
        elif isinstance(value, str) and len(value) > 0:
            detected_text_fields.append(key)
    
    if text_fields is None:
        text_fields = detected_text_fields
    else:
        text_fields = [f for f in text_fields if f in detected_text_fields]
    
    print(f"Detected {len(numeric_fields)} numeric fields: {numeric_fields[:5]}...")
    print(f"Using {len(text_fields)} text fields for TF-IDF: {text_fields}")
    
    X_numeric = []
    for record in records:
        features = []
        for fname in numeric_fields:
            features.append(float(record.get(fname, 0.0)))
        X_numeric.append(features)
    
    X_numeric = np.array(X_numeric)
    
    feature_parts = [X_numeric]
    feature_names = numeric_fields.copy()
    
    for text_field in text_fields:
        texts = [str(record.get(text_field, '')) for record in records]
        
        if all(len(t.strip()) == 0 for t in texts):
            print(f"  Skipping '{text_field}': all empty")
            continue
        
        try:
            tfidf = TfidfVectorizer(
                max_features=max_tfidf_features,
                stop_words='english',
                min_df=2,
                max_df=0.8,
                ngram_range=(1, 2)
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
    
    # combine
    X = np.hstack(feature_parts)
    
    if n_components is not None and n_components < X.shape[1]:
        print(f"Applying SVD dimensionality reduction: {X.shape[1]} -> {n_components} components")
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        X = svd.fit_transform(X)
        
        feature_names = [f"svd_component_{i}" for i in range(n_components)]
        
        explained_variance = svd.explained_variance_ratio_.sum()
        print(f"  Explained variance ratio: {explained_variance:.4f} ({explained_variance*100:.2f}%)")
        print(f"  Reduced to {X.shape[1]} features")
    
    scaler = RobustScaler()
    X = scaler.fit_transform(X)
    
    return X, feature_names


def save_clusters_to_json(clusters, records, feature_set_path, algorithm_name="dbscan"):
    # write cluster outputs to package-local `core/feature_set_extraction/output/fsclusters`
    package_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(package_dir, 'output', 'fsclusters')
    os.makedirs(output_dir, exist_ok=True)
    
    input_base = os.path.splitext(os.path.basename(feature_set_path))[0]
    output_path = os.path.join(output_dir, f"{input_base}_{algorithm_name}_clusters.json")
    
    record_lookup = {r["email_index"]: r for r in records}
    
    has_noise = -1 in clusters
    
    cluster_data = {
        "metadata": {
            "total_emails": len(records),
            "num_clusters": len([c for c in clusters.keys() if c != -1]),
            "algorithm": algorithm_name,
            "feature_set_source": feature_set_path
        },
        "clusters": {}
    }
    
    if has_noise:
        cluster_data["metadata"]["noise_points"] = len(clusters.get(-1, []))
    
    for cluster_id, email_indices in clusters.items():
        cluster_name = "noise" if cluster_id == -1 else f"cluster_{cluster_id}"
        
        cluster_data["clusters"][cluster_name] = {
            "size": len(email_indices),
            "email_indices": email_indices
        }
        
        if cluster_id != -1:
            cluster_data["clusters"][cluster_name]["emails"] = []
            
            for email_idx in email_indices:
                if email_idx in record_lookup:
                    email_record = record_lookup[email_idx].copy()
                    cluster_data["clusters"][cluster_name]["emails"].append(email_record)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(sanitize_for_json(cluster_data), f, indent=2, ensure_ascii=False)
    
    print(f"Saved cluster results to: {output_path}")
    return output_path

def load_ground_truth_from_csv(path):
    """
        email_id (int) -> true_cluster_id
    """
    df = pd.read_csv(path)

    mapping = {}
    for idx, row in df.iterrows():
        email_ids = str(row["email_ids"]).split(",")
        for e in email_ids:
            e = e.strip()
            if e:
                mapping[int(e)] = idx

    return mapping
