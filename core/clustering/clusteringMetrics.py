from sklearn.metrics import homogeneity_score, completeness_score, silhouette_score, v_measure_score


def compute_homogeneity_from_clusters(clusters, ground_truth):
    email_to_predicted_cluster = {}
    for cluster_id, email_indices in clusters.items():
        for email_idx in email_indices:
            email_to_predicted_cluster[email_idx] = cluster_id
    

    common_emails = set(email_to_predicted_cluster.keys()) & set(ground_truth.keys())
    
    if len(common_emails) < 2:
        return {'homogeneity': 0.0, 'completeness': 0.0, 'v_measure': 0.0, 'n_samples': len(common_emails)}
    
    common_emails = sorted(common_emails)
    predicted_labels = [email_to_predicted_cluster[e] for e in common_emails]
    true_labels = [ground_truth[e] for e in common_emails]
    
    homogeneity = homogeneity_score(true_labels, predicted_labels)
    completeness = completeness_score(true_labels, predicted_labels)
    v_measure = v_measure_score(true_labels, predicted_labels)
    
    return {
        'homogeneity': homogeneity,
        'completeness': completeness,
        'v_measure': v_measure,
        'n_samples': len(common_emails)
    }


def compute_silhouette_score(X, labels):
    n_clusters = len(set(labels))
    
    if n_clusters >= 2 and len(labels) > 0:
        silhouette_avg = silhouette_score(X, labels)
        return silhouette_avg
    else:
        return None