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


def precompute_subject_idf(misp_path, events):
    """Compute and save subject IDF JSON for a given MISP JSON path.

    Returns the path to the idf JSON.
    Always recomputes to ensure full-term coverage from the current input data.
    """
    base_name = os.path.splitext(os.path.basename(misp_path))[0]
    package_dir = os.path.dirname(os.path.abspath(__file__))
    helpers_dir = os.path.join(package_dir, 'output', 'helpers')
    idf_path = os.path.join(helpers_dir, f"{base_name}_subject_idf.json")

    subjects = []
    for evt in events:
        subj = evt.get('subject', '') if isinstance(evt, dict) else ''
        subjects.append(subj if isinstance(subj, str) else "")

    if not any(s.strip() for s in subjects):
        return idf_path

    try:
        vec = build_vectorizer(subjects, max_features=None, stop_words=None, ngram_range=(1, 1))
        terms = vec.get_feature_names_out()
        idfs = vec.idf_
        idf_dict = {str(term): float(idf) for term, idf in zip(terms, idfs)}

        os.makedirs(os.path.dirname(idf_path) or '.', exist_ok=True)
        tmp = idf_path + '.tmp'
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(idf_dict, f, indent=2, ensure_ascii=False)
        os.replace(tmp, idf_path)
    except Exception:
        # ignore errors; caller will fallback to worker-side computation
        pass

    return idf_path
