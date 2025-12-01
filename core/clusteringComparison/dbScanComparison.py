import json
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
from featureNormalizationTREC import get_test_set


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
'''
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
'''

def preprocess_for_dbscan(records):
    # 1. Flatten nested dictionaries
    flat = [flatten_dict(r) for r in records]

    # 2. Create DataFrame
    df = pd.DataFrame(flat)
    print(df.isna().sum())  # Debug: print number of NaNs per column

    # 3. Drop string fields that are identifiers
    drop_cols = [c for c in df.columns if c.endswith("_id") or c in ["id", "guid"]]
    df = df.drop(columns=drop_cols, errors='ignore')

    # 4. Split categorical vs numeric
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    num_cols = df.select_dtypes(exclude=["object"]).columns.tolist()

    # 5. One-hot encode categoricals
    if len(cat_cols) > 0:
        enc = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        encoded = enc.fit_transform(df[cat_cols])
        cat_df = pd.DataFrame(
            encoded, 
            columns=enc.get_feature_names_out(cat_cols),
            index=df.index  # Preserve index alignment
        )
        df_final = pd.concat(
            [df[num_cols].reset_index(drop=True), 
             cat_df.reset_index(drop=True)], 
            axis=1
        )
    else:
        df_final = df[num_cols].reset_index(drop=True)

    #df_final = df_final.fillna(0)

    return df_final.to_numpy(), df_final.columns.tolist()


def cluster_with_ids(records):
    # Extract IDs
    idxs = [r["email_index"] for r in records]

    # Remove ID from features
    records_wo_id = [{k: v for k, v in r.items() if k != "email_index"} for r in records]

    # Preprocess to numeric features
    X, feature_names = preprocess_for_dbscan(records_wo_id)

    # Run DBSCAN
    labels = DBSCAN(eps=2, min_samples=5).fit_predict(X)

    # Build cluster → ID list map
    from collections import defaultdict
    clusters = defaultdict(list)
    
    for record_id, label in zip(idxs, labels):
        clusters[label].append(record_id)

    return clusters, labels

def save_clusters_to_json(clusters, records, csv_path):
    """Save cluster results to JSON file with full record details"""
    # Create output path
    input_dir = os.path.dirname(csv_path)
    input_base = os.path.splitext(os.path.basename(csv_path))[0]
    output_path = os.path.join(input_dir, f"{input_base}_clusters.json")
    
    # Create lookup dict for email_index -> record
    record_lookup = {r["email_index"]: r for r in records}
    
    # Build cluster data with full record details
    cluster_data = {
        "metadata": {
            "total_emails": len(records),
            "num_clusters": len([c for c in clusters.keys() if c != -1]),
            "noise_points": len(clusters.get(-1, [])),
            "csv_source": csv_path
        },
        "clusters": {}
    }
    
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



def read_misp_data(misp_path):
    """
    Read MISP JSON data and extract email records with features.
    Returns list of email dictionaries with normalized structure.
    """
    with open(misp_path, 'r', encoding='utf-8') as f:
        misp_data = json.load(f)
    
    records = []
    
    # Extract events from MISP format
    events = misp_data.get('response', {}).get('Event', [])
    if not isinstance(events, list):
        events = [events]  # Handle single event case
    
    for event_idx, event in enumerate(events):
        # Extract email attributes from event
        attributes = event.get('Attribute', [])
        
        email_record = {
            'email_index': event_idx,
            'event_id': event.get('id', ''),
            'event_uuid': event.get('uuid', ''),
        }
        
        # Parse attributes to extract email features
        for attr in attributes:
            attr_type = attr.get('type', '')
            attr_value = attr.get('value', '')
            
            # Map MISP attribute types to feature names
            if attr_type == 'email-subject':
                email_record['subject'] = attr_value
                email_record['subject_length'] = len(attr_value)
                email_record['subject_whitespace'] = attr_value.count(' ')
            
            elif attr_type == 'email-body':
                email_record['body'] = attr_value
                email_record['body_length'] = len(attr_value)
            
            elif attr_type == 'email-src':
                email_record['sender_email'] = attr_value
                # Extract sender name/domain if needed
                if '@' in attr_value:
                    email_record['sender_domain'] = attr_value.split('@')[1]
            
            elif attr_type == 'email-dst':
                email_record['recipient_email'] = attr_value
                if '@' in attr_value:
                    email_record['recipient_domain'] = attr_value.split('@')[1]
            
            elif attr_type == 'email-date':
                email_record['date'] = attr_value
            
            elif attr_type == 'url':
                # Collect URLs (can be multiple)
                if 'urls' not in email_record:
                    email_record['urls'] = []
                email_record['urls'].append(attr_value)
        
        # Compute derived URL features
        urls = email_record.get('urls', [])
        email_record['num_urls'] = len(urls)
        email_record['has_urls'] = len(urls) > 0
        
        # Set defaults for missing fields
        for field in ['subject', 'body', 'sender_email', 'recipient_email', 'date']:
            if field not in email_record:
                email_record[field] = ''
        
        for field in ['subject_length', 'body_length', 'subject_whitespace']:
            if field not in email_record:
                email_record[field] = 0
        
        records.append(email_record)
    
    return records

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # Use pre-computed feature set instead of extracting from CSV/MISP
    feature_set_path = "../../data/featuresets/TREC-07-misp-FSTest.json"

    with open(feature_set_path, 'r', encoding='utf-8') as f:
        records = json.load(f)#[:5000]
        
    # Run clustering
    clusters, labels = cluster_with_ids(records)

    # Save clusters to JSON file (use feature set path for output naming)
    output_path = save_clusters_to_json(clusters, records, feature_set_path)
    
    # Print summary
    print(f"\nCluster Summary:")
    for cluster_id in sorted(clusters.keys()):
        cluster_name = "noise" if cluster_id == -1 else f"cluster_{cluster_id}"
        print(f"{cluster_name}: {len(clusters[cluster_id])} emails")