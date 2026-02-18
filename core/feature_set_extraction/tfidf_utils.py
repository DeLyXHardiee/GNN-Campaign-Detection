import os
import json
try:
    import joblib
    _HAS_JOBLIB = True
except Exception:
    import pickle
    _HAS_JOBLIB = False

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def build_vectorizer(texts, max_features=2000, stop_words='english'):
    """Fit and return a TfidfVectorizer with sensible defaults."""
    vec = TfidfVectorizer(
        max_features=max_features,
        stop_words=stop_words,
        min_df=2,
        max_df=0.8,
        ngram_range=(1, 2)
    )
    vec.fit(texts)
    return vec


def transform_texts(vectorizer, texts):
    return vectorizer.transform(texts).toarray()


def save_vectorizer(path, vectorizer):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    tmp = path + '.tmp'
    if _HAS_JOBLIB:
        joblib.dump(vectorizer, tmp)
    else:
        with open(tmp, 'wb') as f:
            pickle.dump(vectorizer, f)
    os.replace(tmp, path)


def load_vectorizer(path):
    if _HAS_JOBLIB:
        return joblib.load(path)
    else:
        with open(path, 'rb') as f:
            return pickle.load(f)


def save_idf_csv(path, vectorizer):
    """Save idf values to CSV using the same format as the original code (term,idf)."""
    terms = vectorizer.get_feature_names_out()
    idfs = vectorizer.idf_
    df = pd.DataFrame({'term': terms, 'idf': idfs}).sort_values('idf', ascending=False)
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    tmp = path + '.tmp'
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def precompute_subject_idf(misp_path):
    """Compute and save subject IDF CSV for a given MISP JSON path.

    Returns the path to the idf CSV. Does nothing if the CSV already exists.
    """
    # compute project root relative to this file (core/feature_set_extraction)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    dir_name = os.path.dirname(misp_path)
    base_name = os.path.splitext(os.path.basename(misp_path))[0]
    if base_name.endswith('-misp'):
        csv_base = base_name.replace('-misp', '-only-phishing')
    else:
        csv_base = base_name + '-only-phishing'

    idf_path = os.path.join(project_root, 'data', 'csv', f"{csv_base}_subject_idf.csv")
    if os.path.exists(idf_path):
        return idf_path

    # try to load MISP JSON and extract all subjects
    try:
        with open(misp_path, 'r', encoding='utf-8') as f:
            misp_data = json.load(f)
    except Exception:
        return idf_path

    # extract events
    if isinstance(misp_data, list):
        events = misp_data
    elif isinstance(misp_data, dict):
        events = misp_data.get('response', {}).get('Event', [])
        if not isinstance(events, list):
            events = [events]
    else:
        events = []

    subjects = []
    for evt in events:
        event = evt.get('Event', {}) if isinstance(evt, dict) else {}
        # find attribute entries
        attrs = event.get('Attribute', [])
        subj = ''
        for a in attrs:
            if a.get('type') == 'email-subject':
                subj = a.get('value', '')
                break
        subjects.append(subj if isinstance(subj, str) else "")

    if not any(s.strip() for s in subjects):
        return idf_path

    try:
        vec = build_vectorizer(subjects)
        save_idf_csv(idf_path, vec)
    except Exception:
        # ignore errors; caller will fallback to worker-side computation
        pass

    return idf_path
