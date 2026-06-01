import json
import os
from pathlib import Path

import config.blas_env  # noqa: F401 — before pandas / NumPy

import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix, hstack as sparse_hstack
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, normalize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import IsolationForest
from sentence_transformers import SentenceTransformer
from preprocessing.utils.defang import sanitize_for_json
from utils.embeddings import DEFAULT_OUTPUT_DIR, MODEL_NAME, get_embeddings


_SBERT_MODEL = None


def _get_sbert_model(model_name="intfloat/multilingual-e5-large"):
    global _SBERT_MODEL
    if _SBERT_MODEL is None:
        _SBERT_MODEL = SentenceTransformer(model_name)
    return _SBERT_MODEL


def _encode_texts_with_sbert(texts, model_name="intfloat/multilingual-e5-large", batch_size=64):
    model = _get_sbert_model(model_name=model_name)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )
    return np.asarray(embeddings, dtype=float)


def _records_as_emails_for_embedder(records):
    """Shape records into the list[dict] expected by ``utils.embeddings.get_embeddings``."""
    emails = []
    for i, record in enumerate(records):
        ext = record.get("external_id")
        emails.append(
            {
                "external_id": str(ext).strip() if ext is not None else "",
                "subject": str(record.get("subject") or ""),
                "body": str(record.get("body") or ""),
                "email_index": i,
            }
        )
    return emails


def _pad_sbert_row(vec, dim: int) -> np.ndarray:
    """Fixed-width row for stacking (embedder may return [] for empty text)."""
    if dim <= 0:
        return np.zeros(0, dtype=float)
    v = np.asarray(vec, dtype=float)
    if v.size == 0:
        return np.zeros(dim, dtype=float)
    if v.size == dim:
        return v
    if v.size > dim:
        return v[:dim]
    out = np.zeros(dim, dtype=float)
    out[: v.size] = v
    return out


def _subject_body_sbert_from_embedder_cache(records, embeddings_output_dir):
    """
    Load or compute subject/body SBERT via the shared embeddings cache (same as graph build).
    Returns dict with optional keys ``subject``, ``body`` -> (n, dim) float arrays.
    """
    out: dict[str, np.ndarray] = {}
    emails = _records_as_emails_for_embedder(records)
    out_dir = Path(embeddings_output_dir) if embeddings_output_dir else DEFAULT_OUTPUT_DIR
    subj_vecs, body_vecs, subj_dim, body_dim = get_embeddings(emails, output_dir=out_dir)
    if subj_dim > 0:
        out["subject"] = np.stack([_pad_sbert_row(v, subj_dim) for v in subj_vecs])
    if body_dim > 0:
        out["body"] = np.stack([_pad_sbert_row(v, body_dim) for v in body_vecs])
    return out


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

        if raw.startswith("[") and raw.endswith("]"):
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, list):
                    return [str(v).strip().lower() for v in parsed if str(v).strip()]
            except Exception:
                pass

        return [tok.strip().lower() for tok in raw.split() if tok.strip()]

    return [str(value).strip().lower()] if str(value).strip() else []


def _normalize_numeric_dict(value):
    """Normalize dict-like features into str->float mapping for vectorization."""
    if value is None:
        return {}

    raw_dict = None
    if isinstance(value, dict):
        raw_dict = value
    elif isinstance(value, str):
        s = value.strip()
        if not s:
            return {}
        try:
            parsed = json.loads(s)
            if isinstance(parsed, dict):
                raw_dict = parsed
        except Exception:
            return {}
    else:
        return {}

    norm = {}
    for key, val in raw_dict.items():
        if key is None:
            continue
        k = str(key).strip().lower()
        if not k:
            continue
        try:
            norm[k] = float(val)
        except Exception:
            continue
    return norm


def _dense_block_to_csr(arr: np.ndarray) -> csr_matrix:
    """CSR view of a 2-D dense block for sparse horizontal stacking."""
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("Expected 2-D array for feature block")
    if arr.shape[1] == 0:
        return csr_matrix((arr.shape[0], 0))
    return csr_matrix(arr)


def scale_and_normalize_matrix(X, scaler_type="robust", l2_normalize=True):
    """Apply feature scaling and optional row-wise L2 normalization."""
    if scaler_type == "robust":
        scaler = RobustScaler()
    elif scaler_type == "standard":
        scaler = StandardScaler()
    elif scaler_type == "minmax":
        scaler = MinMaxScaler()
    elif scaler_type in (None, "none"):
        scaler = None
    else:
        raise ValueError(f"Unsupported scaler_type: {scaler_type}")

    if scaler is not None:
        X = scaler.fit_transform(X)

    if l2_normalize:
        X = normalize(X, norm="l2", axis=1)

    return X


def remove_outliers_from_matrix(X, contamination=0.05, random_state=42):
    """Remove outlier rows from feature matrix using IsolationForest."""
    if X is None or len(X) == 0:
        return X, np.array([], dtype=bool), 0

    if contamination is None or contamination <= 0:
        keep_mask = np.ones(X.shape[0], dtype=bool)
        return X, keep_mask, 0

    contamination = min(float(contamination), 0.5)
    model = IsolationForest(
        contamination=contamination,
        random_state=random_state,
        n_estimators=200,
    )
    pred = model.fit_predict(X)
    keep_mask = pred == 1
    removed = int((~keep_mask).sum())

    if keep_mask.sum() == 0:
        keep_mask = np.ones(X.shape[0], dtype=bool)
        return X, keep_mask, 0

    return X[keep_mask], keep_mask, removed


def preprocess_for_clustering(
    records,
    max_tfidf_features,
    text_fields=None,
    exclude_fields=None,
    n_components=None,
    token_list_fields=None,
    token_svd_components=32,
    dict_feature_fields=None,
    scaler_type="robust",
    l2_normalize=True,
    sbert_model_name="intfloat/multilingual-e5-large",
    embeddings_output_dir=None,
):

    if not records:
        raise ValueError("Empty records list")
    if exclude_fields is None:
        exclude_fields = ["external_id"]

    if token_list_fields is None:
        token_list_fields = ['hostnames', 'domains']

    if dict_feature_fields is None:
        dict_feature_fields = ['subject_term_frequency']
    sample_record = records[0]
    numeric_fields = []
    detected_text_fields = []
    detected_dict_fields = []
    for key, value in sample_record.items():
        if key in exclude_fields:
            continue
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            numeric_fields.append(key)
        elif isinstance(value, str) and len(value) > 0:
            detected_text_fields.append(key)
        elif isinstance(value, dict):
            detected_dict_fields.append(key)
    if text_fields is None:
        text_fields = detected_text_fields
    else:
        text_fields = [f for f in text_fields if f in detected_text_fields]

    token_list_fields = [
        f for f in token_list_fields
        if f in sample_record and f not in exclude_fields
    ]

    dict_feature_fields = [
        f for f in dict_feature_fields
        if f in sample_record and f not in exclude_fields
    ]
    text_fields = [f for f in text_fields if f not in token_list_fields]
    text_fields = [f for f in text_fields if f not in dict_feature_fields]
    print(f"Detected {len(numeric_fields)} numeric fields: {numeric_fields[:5]}...")
    print(f"Using {len(text_fields)} text fields: {text_fields}")
    print(f"Using {len(token_list_fields)} token-list fields: {token_list_fields}")
    print(f"Using {len(dict_feature_fields)} dict feature fields: {dict_feature_fields}")

    precomputed_sbert = {}
    if sbert_model_name == MODEL_NAME and any(f in text_fields for f in ("subject", "body")):
        emb_dir = Path(embeddings_output_dir) if embeddings_output_dir else DEFAULT_OUTPUT_DIR
        try:
            precomputed_sbert = _subject_body_sbert_from_embedder_cache(
                records, embeddings_output_dir
            )
            if precomputed_sbert:
                print(
                    f"  SBERT subject/body: using shared embedder cache at {emb_dir} "
                    f"(fields: {list(precomputed_sbert.keys())})"
                )
        except Exception as e:
            print(f"  SBERT shared cache/embedder failed ({e}); using local SentenceTransformer encode.")
            precomputed_sbert = {}

    X_numeric = []
    for record in records:
        features = []
        for fname in numeric_fields:
            features.append(float(record.get(fname, 0.0)))
        X_numeric.append(features)
    X_numeric = np.asarray(X_numeric, dtype=np.float64)

    feature_parts_sparse = [_dense_block_to_csr(X_numeric)]                          
    feature_names = numeric_fields.copy()
    for text_field in text_fields:
        texts = [str(record.get(text_field, '')) for record in records]
        if all(len(t.strip()) == 0 for t in texts):
            print(f"  Skipping '{text_field}': all empty")
            continue
        try:
            if text_field in {"subject", "body"}:
                if text_field in precomputed_sbert:
                    X_text = precomputed_sbert[text_field]
                else:
                    X_text = _encode_texts_with_sbert(texts, model_name=sbert_model_name)
                feature_parts_sparse.append(_dense_block_to_csr(X_text))                       
                feature_names.extend([f"{text_field}_sbert_{i}" for i in range(X_text.shape[1])])
                print(f"  {text_field}: extracted {X_text.shape[1]} SBERT features -> shape {X_text.shape}")
                continue

            tfidf = TfidfVectorizer(
                max_features=max_tfidf_features,
                stop_words='english',
                min_df=2,
                max_df=0.8,
                ngram_range=(1, 2)
            )
            X_text = tfidf.fit_transform(texts)

            if X_text.shape[1] > 0:
                feature_parts_sparse.append(X_text.tocsr())                       
                feature_names.extend([f"{text_field}_tfidf_{i}" for i in range(X_text.shape[1])])
                print(f"  {text_field}: extracted {X_text.shape[1]} TF-IDF features -> shape {X_text.shape}")
            else:
                print(f"  Skipping '{text_field}': no features extracted")
        except Exception as e:
            print(f"  Error processing '{text_field}': {e}")

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
            if X_tokens_sparse.shape[1] > 0:
                feature_parts_sparse.append(X_tokens_sparse.tocsr())                        
                feature_names.extend(
                    [f"{token_field}_tfidf_{i}" for i in range(X_tokens_sparse.shape[1])]
                )
                print(f"  {token_field}: extracted {X_tokens_sparse.shape[1]} TF-IDF token features -> shape {X_tokens_sparse.shape}")
            else:
                print(f"  Skipping '{token_field}': no features extracted")
        except Exception as e:
            print(f"  Error processing token-list field '{token_field}': {e}")

    for dict_field in dict_feature_fields:
        dict_docs = [_normalize_numeric_dict(record.get(dict_field, {})) for record in records]

        if all(len(d) == 0 for d in dict_docs):
            print(f"  Skipping '{dict_field}': all empty")
            continue

        try:
            vectorizer = DictVectorizer(sparse=True)
            X_dict_sparse = vectorizer.fit_transform(dict_docs)
            if X_dict_sparse.shape[1] > 0:
                feature_parts_sparse.append(X_dict_sparse.tocsr())                       
                feature_names.extend(
                    [f"{dict_field}_dict_{i}" for i in range(X_dict_sparse.shape[1])]
                )
                print(f"  {dict_field}: extracted {X_dict_sparse.shape[1]} dict features -> shape {X_dict_sparse.shape}")
            else:
                print(f"  Skipping '{dict_field}': no features extracted")
        except Exception as e:
            print(f"  Error processing dict field '{dict_field}': {e}")
    X = sparse_hstack(feature_parts_sparse, format="csr")
    '''
    print("feature_parts_sparse block shapes:")
    for i, part in enumerate(feature_parts_sparse):
        print(f"  block {i}: shape {part.shape}")
    total_width = sum(part.shape[1] for part in feature_parts_sparse)
    print(f"feature_parts_sparse summary: {len(feature_parts_sparse)} blocks, total width {total_width}")
    print(f"Combined feature matrix before SVD: shape {X.shape}")
    '''
    if n_components is not None and n_components < X.shape[1]:
        print(f"Applying SVD dimensionality reduction: {X.shape[1]} -> {n_components} components")
        svd = TruncatedSVD(
            n_components=n_components,
            random_state=42,
            algorithm="randomized",
        )
        X = svd.fit_transform(X)
        feature_names = [f"svd_component_{i}" for i in range(n_components)]
        explained_variance = svd.explained_variance_ratio_.sum()
        print(f"  Explained variance ratio: {explained_variance:.4f} ({explained_variance*100:.2f}%)")
        print(f"  Reduced to {X.shape[1]} features")
    else:
        X = X.toarray()

    print(f"Applying scaler: {scaler_type}, l2_normalize={l2_normalize}")
    X = scale_and_normalize_matrix(X, scaler_type=scaler_type, l2_normalize=l2_normalize)
    return X, feature_names


def record_cluster_id(record):
    """
    Stable id for clustering outputs and ground-truth alignment (matches
    :func:`load_ground_truth_from_json` keys).
    """
    ext = record.get("external_id")
    if ext is None:
        raise ValueError("Feature record missing external_id")
    s = str(ext).strip()
    if not s:
        raise ValueError("Feature record has empty external_id")
    return s


def save_clusters_to_json(clusters, records, feature_set_path, algorithm_name="dbscan"):
    package_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(package_dir, 'output', 'fsclusters')
    os.makedirs(output_dir, exist_ok=True)
    input_base = os.path.splitext(os.path.basename(feature_set_path))[0]
    output_path = os.path.join(output_dir, f"{input_base}_{algorithm_name}_clusters.json")
    record_lookup = {}
    for r in records:
        rid = record_cluster_id(r)
        record_lookup.setdefault(rid, r)

    has_noise = -1 in clusters

    cluster_data = {
        "metadata": {
            "total_emails": len(records),
            "num_clusters": len([c for c in clusters.keys() if c != -1]),
            "algorithm": algorithm_name,
            "feature_set_source": feature_set_path,
            "member_id_key": "external_id",
        },
        "clusters": {},
    }

    if has_noise:
        cluster_data["metadata"]["noise_points"] = len(clusters.get(-1, []))

    for cluster_id, member_ids in clusters.items():
        cluster_name = "noise" if cluster_id == -1 else f"cluster_{cluster_id}"

        cluster_data["clusters"][cluster_name] = {
            "size": len(member_ids),
            "external_ids": member_ids,
        }

        if cluster_id != -1:
            cluster_data["clusters"][cluster_name]["emails"] = []

            for mid in member_ids:
                if mid in record_lookup:
                    email_record = record_lookup[mid].copy()
                    cluster_data["clusters"][cluster_name]["emails"].append(
                        email_record
                    )
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(sanitize_for_json(cluster_data), f, indent=2, ensure_ascii=False)
    print(f"Saved cluster results to: {output_path}")
    return output_path


def _ground_truth_cluster_id_from_key(raw_key):
    """Derive cluster/campaign id from a JSON cluster key (e.g. label_store_1/49)."""
    if raw_key == "noise":
        return -1
    tail = raw_key.split("/")[-1] if isinstance(raw_key, str) and "/" in raw_key else raw_key
    try:
        return int(tail)
    except (ValueError, TypeError):
        return raw_key


def load_ground_truth_from_json(path):
    """
    Load ground truth from a JSON file.

    Expects a top-level ``clusters`` object. Each entry is either a list of
    records with ``external_id``, or a dict containing ``emails`` or ``records``
    (same shape as outputs from MISP / :func:`save_clusters_to_json`).

    Returns:
        dict: ``external_id`` (int if numeric, else str) -> true cluster id
        (int when parsable from the cluster key, else str; noise -> -1).
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    mapping = {}
    clusters = data.get("clusters", {})
    for raw_key, payload in clusters.items():
        cluster_id = _ground_truth_cluster_id_from_key(raw_key)
        records = None
        if isinstance(payload, list):
            records = payload
        elif isinstance(payload, dict):
            records = payload.get("emails") or payload.get("records")
            if not records and payload.get("external_ids"):
                records = [
                    {"external_id": str(e).strip()}
                    for e in payload["external_ids"]
                    if str(e).strip()
                ]
            if not records:
                records = []

        if not records:
            print(f"Warning: No records found for cluster {raw_key}")
            continue

        for rec in records:
            if not isinstance(rec, dict):
                print(f"Warning: Record {rec} is not a dictionary")
                continue
            ext = rec.get("external_id")
            if ext is None:
                print(f"Warning: Record {rec} has no external_id")
                continue
            ext_s = str(ext).strip()
            if not ext_s:
                continue
            mapping[ext_s] = cluster_id

    return mapping
