import sys
import os
import csv
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import OneHotEncoder
import pandas as pd


# --- IMPORT get_FS7 ---
sys.path.append(os.path.join(os.path.dirname(__file__), "../featureNormalization"))
from featureNormalizationTREC import get_FS7


from sklearn.cluster import DBSCAN
import numpy as np

def flatten_dict(d, parent_key="", sep="."):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def flatten_records(records):
    return [flatten_dict(rec) for rec in records]

def align_features(records):
    all_keys = sorted({key for rec in records for key in rec.keys()})
    aligned = []
    for rec in records:
        aligned.append({k: float(rec.get(k, 0)) for k in all_keys})
    return aligned, all_keys

def dict_list_to_matrix(dict_list):
    feature_keys = list(dict_list[0].keys())
    matrix = [[float(row[k]) for k in feature_keys] for row in dict_list]
    return np.array(matrix), feature_keys

def run_dbscan(records, eps=0.5, min_samples=5):
    flat = flatten_records(records)
    aligned, feature_names = align_features(flat)
    matrix, feature_names = dict_list_to_matrix(aligned)

    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(matrix)
    return labels, feature_names

def preprocess_for_dbscan(records):
    # 1. Flatten nested dictionaries
    flat = [flatten_dict(r) for r in records]

    # 2. Create DataFrame
    df = pd.DataFrame(flat)

    # 3. Drop string fields that are identifiers
    drop_cols = [c for c in df.columns if c.endswith("_id") or c in ["id", "guid"]]
    df = df.drop(columns=drop_cols)

    # 4. Split categorical vs numeric
    cat_cols = df.select_dtypes(include=["object"]).columns
    num_cols = df.select_dtypes(exclude=["object"]).columns

    # 5. One-hot encode categoricals
    enc = OneHotEncoder(handle_unknown='ignore')
    if len(cat_cols) > 0:
        encoded = enc.fit_transform(df[cat_cols])
        cat_df = pd.DataFrame(encoded, columns=enc.get_feature_names_out(cat_cols))
        df_final = pd.concat([df[num_cols].reset_index(drop=True), cat_df], axis=1)
    else:
        df_final = df[num_cols]

    return df_final.to_numpy(), df_final.columns


def cluster_with_ids(records):
    # Extract IDs
    idxs = [r["email_index"] for r in records]

    # Remove ID from features
    records_wo_id = [{k: v for k, v in r.items() if k != "email_index"} for r in records]

    # Preprocess to numeric features
    X, feature_names = preprocess_for_dbscan(records_wo_id)

    # Run DBSCAN
    labels = DBSCAN(eps=0.5, min_samples=5).fit_predict(X)

    # Build cluster → ID list map
    from collections import defaultdict
    clusters = defaultdict(list)
    
    for record_id, label in zip(idxs, labels):
        clusters[label].append(record_id)

    return clusters, labels

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    csv_path = "../../data/csv/TREC-07-only-phishing.csv"

    records = get_FS7(csv_path)
    clusters, labels = cluster_with_ids(records)

    print(clusters[0])   # IDs in cluster 0
    print(clusters[1])   # IDs in cluster 1
    print(clusters[-1])  # Noise points

    