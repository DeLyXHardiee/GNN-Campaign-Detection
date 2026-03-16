import json
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from preprocessing.utils.defang import sanitize_for_json


def _normalize_token_list(value):
    """Normalize list-like or whitespace-joined token fields into clean token lists."""
    if value is None:
        return []

    if isinstance(value, (list, tuple, set)):
        return [str(v).strip().lower() for v in value if str(v).strip()]

    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return []

        # Some pipelines store list-like content as JSON strings.
        if raw.startswith("[") and raw.endswith("]"):
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, list):
                    return [str(v).strip().lower() for v in parsed if str(v).strip()]
            except Exception:
                pass

        return [tok.strip().lower() for tok in raw.split() if tok.strip()]

    return [str(value).strip().lower()] if str(value).strip() else []


def preprocess_for_clustering(
    records,
    max_tfidf_features,
    text_fields=None,
    exclude_fields=None,
    n_components=None,
    token_list_fields=None,
    token_svd_components=32,
):

    if not records:
        raise ValueError("Empty records list")
    
    if exclude_fields is None:
        exclude_fields = ['email_index']

    if token_list_fields is None:
        # Common URL list-like fields represented as token strings or arrays.
        token_list_fields = ['hostnames', 'domains']
    
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

    # Route list-like/tokenized URL fields into dedicated processing.
    token_list_fields = [
        f for f in token_list_fields
        if f in sample_record and f not in exclude_fields
    ]
    text_fields = [f for f in text_fields if f not in token_list_fields]
    
    print(f"Detected {len(numeric_fields)} numeric fields: {numeric_fields[:5]}...")
    print(f"Using {len(text_fields)} text fields for TF-IDF: {text_fields}")
    print(f"Using {len(token_list_fields)} token-list fields: {token_list_fields}")
    
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

    # Dedicated processing for token-list style fields (e.g., multiple hostnames per email).
    for token_field in token_list_fields:
        token_docs = [" ".join(_normalize_token_list(record.get(token_field, ""))) for record in records]

        if all(len(doc.strip()) == 0 for doc in token_docs):
            print(f"  Skipping '{token_field}': all empty")
            continue

        try:
            vectorizer = TfidfVectorizer(
                max_features=max_tfidf_features,
                min_df=1,
                max_df=1.0,
                token_pattern=r"(?u)\b[\w\.-]+\b",
                lowercase=True,
            )
            X_tokens_sparse = vectorizer.fit_transform(token_docs)

            # Compress very wide sparse hostname/domain spaces to stable dense features.
            if token_svd_components is not None and X_tokens_sparse.shape[1] > 1:
                max_components = min(
                    int(token_svd_components),
                    X_tokens_sparse.shape[0] - 1,
                    X_tokens_sparse.shape[1] - 1,
                )
                if max_components >= 1 and X_tokens_sparse.shape[1] > max_components:
                    token_svd = TruncatedSVD(n_components=max_components, random_state=42)
                    X_tokens = token_svd.fit_transform(X_tokens_sparse)
                    feature_parts.append(X_tokens)
                    feature_names.extend([f"{token_field}_svd_{i}" for i in range(X_tokens.shape[1])])
                    explained = token_svd.explained_variance_ratio_.sum()
                    print(
                        f"  {token_field}: {X_tokens_sparse.shape[1]} TF-IDF -> {X_tokens.shape[1]} SVD "
                        f"({explained*100:.2f}% explained)"
                    )
                    continue

            X_tokens = X_tokens_sparse.toarray()
            if X_tokens.shape[1] > 0:
                feature_parts.append(X_tokens)
                feature_names.extend([f"{token_field}_tfidf_{i}" for i in range(X_tokens.shape[1])])
                print(f"  {token_field}: extracted {X_tokens.shape[1]} TF-IDF token features")
            else:
                print(f"  Skipping '{token_field}': no features extracted")
        except Exception as e:
            print(f"  Error processing token-list field '{token_field}': {e}")
    
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
