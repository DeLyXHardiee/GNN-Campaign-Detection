import json
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer


def compute_body_lsa(body, lsa_model, vectorizer):
    if not isinstance(body, str):
        body = ""
    
    tfidf_vec = vectorizer.transform([body])
    
    topic_vec = lsa_model.transform(tfidf_vec)[0]
    
    lsa_features = {f"lsa_topic_{i}": round(float(val), 4) for i, val in enumerate(topic_vec)}
    
    return lsa_features

def get_lsa_features(bodies):
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
    
    lsa_features_list = []
    
    for body in bodies:
        lsa_features = compute_body_lsa(body, lsa_model, vectorizer)
        lsa_features_list.append(lsa_features)

    return lsa_features_list
