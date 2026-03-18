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
from graph.common import parse_misp_events


def build_vectorizer(texts, max_features=None, stop_words='english', ngram_range=(1, 2)):
    """Fit and return a TfidfVectorizer with sensible defaults.

    `ngram_range` can be overridden by callers. For subject IDF CSVs we
    prefer unigrams (1,1) to avoid multi-word terms like "word1 word2".
    """
    vec = TfidfVectorizer(
        max_features=max_features,
        stop_words=stop_words,
        min_df=1,
        max_df=1.0,
        ngram_range=ngram_range
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

    Returns the path to the idf CSV.
    Always recomputes to ensure full-term coverage from the current input data.
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
    # try to load MISP JSON and extract all subjects
    try:
        with open(misp_path, 'r', encoding='utf-8') as f:
            misp_data = json.load(f)
    except Exception:
        return idf_path

    # Extract raw events using the same envelope handling as graph loading.
    if isinstance(misp_data, list):
        raw_events = misp_data
    elif isinstance(misp_data, dict):
        if isinstance(misp_data.get('Events'), list):
            raw_events = misp_data.get('Events', [])
        else:
            raw_events = misp_data.get('response', {}).get('Event', [])
            if isinstance(raw_events, dict):
                raw_events = [raw_events]
            elif not isinstance(raw_events, list):
                raw_events = []
    else:
        raw_events = []

    events = parse_misp_events(raw_events)

    subjects = []
    for evt in events:
        subj = evt.get('subject', '') if isinstance(evt, dict) else ''
        subjects.append(subj if isinstance(subj, str) else "")

    if not any(s.strip() for s in subjects):
        return idf_path

    try:
        # Use unigrams only and include all terms:
        # - max_features=None (no cap)
        # - stop_words=None (do not filter vocabulary)
        vec = build_vectorizer(subjects, max_features=None, stop_words=None, ngram_range=(1, 1))
        save_idf_csv(idf_path, vec)
    except Exception:
        # ignore errors; caller will fallback to worker-side computation
        pass

    return idf_path
