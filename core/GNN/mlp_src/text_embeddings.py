from sentence_transformers import SentenceTransformer
from typing import List, Tuple, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize


# Load once (do NOT reload inside functions)
curr_model_name = "intfloat/multilingual-e5-large"
model = SentenceTransformer(curr_model_name)


# Semantic embedding of the email-body (NumPy output)
def get_semantic_embeddings(
    texts: List[str],
    batch_size: int = 32,
    device: Optional[str] = None,
    l2_normalize: bool = True,
    model_name: str = "intfloat/multilingual-e5-large"
) -> np.ndarray:
    if curr_model_name != model_name:
        raise ValueError(
            f"Model mismatch: {model_name} provided as argument, "
            f"but {curr_model_name} used in text_embeddings.py."
        )

    if device is not None:
        model.to(device)

    texts = ["passage: " + t for t in texts]
    embs = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_tensor=False,      # <-- NumPy
        normalize_embeddings=l2_normalize,    # good for cosine similarity
        show_progress_bar=True,
    )
    # sentence-transformers returns a numpy.ndarray when convert_to_tensor=False
    return np.asarray(embs, dtype=np.float32)  # shape: (N, dim)


def train_char_tfidf_model(
    train_texts: List[str],
    *,
    analyzer: str = "char_wb",
    ngram_range: Tuple[int, int] = (3, 5),
    min_df: int = 3,
    max_features: int = 200_000,
):
    """
    Fits a char n-gram TF-IDF model on ALL training texts.
    Returns the fitted vectorizer and the TF-IDF matrix for the training texts.
    """
    tfidf = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=ngram_range,
        min_df=min_df,
        max_features=max_features,
        lowercase=False,  # keep consistent with your cleaning
    )
    X_train = tfidf.fit_transform(train_texts)  # sparse (N_train, V)
    return tfidf, X_train


def train_tfidf_svd_reducer(
    tfidf_train_matrix,
    *,
    out_dim: int = 256,
    seed: int = 42,
) -> TruncatedSVD:
    """
    Fits a TruncatedSVD reducer on the TF-IDF matrix from the TRAIN split.
    Returns the fitted reducer.
    """
    svd = TruncatedSVD(n_components=out_dim, random_state=seed)
    svd.fit(tfidf_train_matrix)
    return svd


def get_char_embeddings(
    texts: List[str],
    *,
    tfidf_model: TfidfVectorizer,
    svd_reducer: TruncatedSVD,
    l2_normalize: bool = True,
) -> np.ndarray:
    """
    Produces reduced char-style embeddings for `texts` using a fitted TF-IDF model + fitted SVD reducer.
    Output shape: (N, out_dim)
    """
    X = tfidf_model.transform(texts)  # sparse (N, V)
    Z = svd_reducer.transform(X)      # dense (N, out_dim)

    if l2_normalize:
        Z = normalize(Z)

    return np.asarray(Z, dtype=np.float32)


def fuse_text_embeddings(semantic_embeddings: np.ndarray, char_style_embeddings: np.ndarray, l2_normalize: bool = True) -> np.ndarray:
    """
    Concatenates semantic + char-style embeddings.
    Output shape: (N, d_sem + d_char)
    """
    if semantic_embeddings.shape[0] != char_style_embeddings.shape[0]:
        raise ValueError(
            f"Row mismatch: semantic has {semantic_embeddings.shape[0]} rows "
            f"but char has {char_style_embeddings.shape[0]} rows."
        )
    fused_embeddings = np.concatenate([semantic_embeddings, char_style_embeddings], axis=1).astype(np.float32)
    if l2_normalize:
        fused_embeddings = normalize(fused_embeddings)
    return fused_embeddings


