import json
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer


def compute_body_lsa(body, lsa_model, vectorizer):
    """
    Compute LSA topic vector for a single email body using pre-fitted model.
    This is used during feature extraction for individual emails.
    
    Args:
        body: Email body text
        lsa_model: Fitted TruncatedSVD model
        vectorizer: Fitted TfidfVectorizer
    
    Returns:
        Dict with LSA topic weights (topic_0, topic_1, ..., topic_9)
    """
    if not isinstance(body, str):
        body = ""
    
    tfidf_vec = vectorizer.transform([body])
    
    topic_vec = lsa_model.transform(tfidf_vec)[0]
    
    lsa_features = {f"lsa_topic_{i}": round(float(val), 4) for i, val in enumerate(topic_vec)}
    
    return lsa_features

def get_lsa_features(bodies):
    """
    Extract LSA features from MISP email bodies.
    Fits LSA model once on entire corpus, then extracts topic vectors per email.
    """
    
    print(f"Fitting LSA model on {len(bodies)} email bodies...")
    
    vectorizer = TfidfVectorizer(
        max_features=1000, 
        stop_words='english', 
        min_df=2, 
        max_df=0.8
    )
    tfidf_matrix = vectorizer.fit_transform(bodies)
    
    lsa_model = TruncatedSVD(n_components=10, random_state=42)
    lsa_model.fit(tfidf_matrix)
    
    print("Extracting LSA features from email bodies...")
    
    lsa_features_list = []
    
    for body in bodies:
        lsa_features = compute_body_lsa(body, lsa_model, vectorizer)
        lsa_features_list.append(lsa_features)

    print(f"Returning {len(lsa_features_list)} LSA feature vectors...")

    return lsa_features_list
