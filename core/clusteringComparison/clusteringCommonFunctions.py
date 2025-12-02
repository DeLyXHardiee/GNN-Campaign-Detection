"""
Common preprocessing and output functions for clustering algorithms.
Following repo's schema-driven pattern: single source of truth for feature preprocessing.
"""

import json
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder


def preprocess_for_clustering(records, max_categories=20, clip_outliers=False, use_robust_scaler=False):
    """
    Preprocess features for clustering algorithms (DBSCAN, Mean Shift, etc.).
    Schema-driven: automatically detects numeric vs categorical fields from data.
    
    Args:
        records: List of email feature dictionaries (must include 'email_index')
        max_categories: Max unique values for one-hot encoding (reduces dimensionality)
        clip_outliers: Clip standardized features to [-5, +5] range
        use_robust_scaler: Use RobustScaler instead of StandardScaler (better for outliers)
    
    Returns:
        X_scaled: Standardized feature matrix (numpy array)
        feature_names: List of feature names corresponding to columns in X_scaled
    
    Notes:
        - Follows repo's "small numeric features" principle
        - One-hot encodes categorical (string) features with cardinality limit
        - Standardizes all features to mean=0, std=1
        - Clips extreme outliers to prevent bandwidth estimation issues
        - Excludes identifier fields (email_index, *_id, id, guid)
    """
    # 1. Create DataFrame from records (excluding email_index)
    records_wo_id = [{k: v for k, v in r.items() if k != "email_index"} for r in records]
    df = pd.DataFrame(records_wo_id)
    
    # 2. Drop identifier fields
    drop_cols = [c for c in df.columns if c.endswith("_id") or c in ["id", "guid"]]
    df = df.drop(columns=drop_cols, errors='ignore')
    
    # 3. Split categorical vs numeric (pandas auto-detection)
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    num_cols = df.select_dtypes(exclude=["object"]).columns.tolist()
    
    '''
    # 4. Limit categorical cardinality (following "small numeric features" principle)
    if len(cat_cols) > 0:
        for col in cat_cols:
            # Keep only top N categories, rest become 'other'
            value_counts = df[col].value_counts()
            if len(value_counts) > max_categories:
                top_categories = value_counts.nlargest(max_categories).index
                df[col] = df[col].where(df[col].isin(top_categories), 'other')
                print(f"Limited '{col}' from {len(value_counts)} to {max_categories} categories")
    '''
    # 5. One-hot encode categoricals
    if len(cat_cols) > 0:
        enc = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        encoded = enc.fit_transform(df[cat_cols])
        cat_df = pd.DataFrame(
            encoded, 
            columns=enc.get_feature_names_out(cat_cols),
            index=df.index
        )
        df_final = pd.concat(
            [df[num_cols].reset_index(drop=True), 
             cat_df.reset_index(drop=True)], 
            axis=1
        )
    else:
        df_final = df[num_cols].reset_index(drop=True)
    
    # 6. Drop zero-variance columns (prevent NaN in standardization)
    zero_var_cols = df_final.columns[df_final.std() == 0]
    if len(zero_var_cols) > 0:
        print(f"Dropping {len(zero_var_cols)} zero-variance columns")
        df_final = df_final.drop(columns=zero_var_cols)
    
    # 7. Fill NaNs with 0
    df_final = df_final.fillna(0)
    
    # 8. Replace infinities
    df_final = df_final.replace([np.inf, -np.inf], 0)
    
    # 9. Standardize all features
    if use_robust_scaler:
        # RobustScaler uses median/IQR, less sensitive to outliers
        scaler = RobustScaler()
    else:
        scaler = StandardScaler()
    
    X_scaled = scaler.fit_transform(df_final)
    
    # 10. Clip extreme outliers (prevents bandwidth explosion in Mean Shift)
    if clip_outliers:
        X_scaled = np.clip(X_scaled, -5, 5)
        print(f"Clipped outliers to [-5, +5] range")
    
    print(f"Final feature matrix: {X_scaled.shape}")
    print(f"After scaling - Mean: {X_scaled.mean():.4f}, Std: {X_scaled.std():.4f}")
    print(f"Range: [{X_scaled.min():.2f}, {X_scaled.max():.2f}]")
    
    return X_scaled, df_final.columns.tolist()


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
        - Output saved to same directory as input with _{algorithm}_clusters.json suffix
    """
    # Create output path
    input_dir = os.path.dirname(feature_set_path)
    input_base = os.path.splitext(os.path.basename(feature_set_path))[0]
    output_path = os.path.join(input_dir, f"{input_base}_{algorithm_name}_clusters.json")
    
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
