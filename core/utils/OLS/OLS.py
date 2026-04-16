import json
import argparse
import itertools
from collections import defaultdict
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

from config.pipeline_config import (
    EmailFeatureProjectionSettings,
    PIPELINE_CONFIG,
    graph_build_settings_from_pipeline,
    output_runs_parent_from_pipeline,
    resolve_project_path,
)


# -------------------------
# Helpers
# -------------------------

def load_emails(path):
    with open(path, "r") as f:
        data = json.load(f)

    emails = {}
    for item in data:
        event = item["Event"]
        external_id = event["external_id"]

        attr_map = {}
        for attr in event["Attribute"]:
            attr_map[attr["type"]] = attr["value"]

        emails[external_id] = attr_map

    return emails


def load_ground_truth(path):
    with open(path, "r") as f:
        data = json.load(f)

    clusters = defaultdict(list)

    for cluster_name, items in data["clusters"].items():
        for item in items:
            clusters[item["cluster_id"]].append(item["external_id"])

    return clusters


# -------------------------
# Feature extraction
# -------------------------

def safe_bool(x):
    if isinstance(x, bool):
        return int(x)
    if isinstance(x, str):
        return int(x.lower() == "true")
    return 0


def extract_numeric_features(email):
    features = {}

    html = email.get("html", {})
    css = email.get("css", {})

    if isinstance(html, dict):
        tree = html.get("tree_stats", {})
        for k, v in tree.items():
            if isinstance(v, (int, float)):
                features[f"html_{k}"] = v

    if isinstance(css, dict):
        style = css.get("style_features", {})
        for k, v in style.items():
            if isinstance(v, (int, float)):
                features[f"css_{k}"] = v

    # Boolean flags
    bool_keys = [
        "rfc_defects",
        "cyrillic_domain",
        "contains_symbols",
        "body_has_tracking_url",
        "body_has_tracking_image",
        "body_has_tracking_pixel",
        "body_has_unsubscribe_link",
        "domain_is_common_webprovided",
    ]

    for k in bool_keys:
        features[k] = safe_bool(email.get(k, False))

    return features


def extract_text(email):
    subject = email.get("subject", "") or ""
    body = email.get("body", "") or ""
    return subject + " " + body


# -------------------------
# Build dataset
# -------------------------

def build_pairs(emails, clusters, max_pairs=20000):
    pairs = []
    labels = []

    cluster_lookup = {}
    for cid, ids in clusters.items():
        for eid in ids:
            cluster_lookup[eid] = cid

    email_ids = list(emails.keys())

    # Generate pairs
    for i, j in itertools.combinations(email_ids, 2):
        same = int(cluster_lookup.get(i) == cluster_lookup.get(j))
        pairs.append((i, j))
        labels.append(same)

        if len(pairs) >= max_pairs:
            break

    return pairs, np.array(labels)


# -------------------------
# Pairwise feature builder
# -------------------------

def build_pair_features(pairs, emails):
    numeric_keys = set()

    email_numeric = {}
    email_text = {}

    # Precompute
    for eid, email in emails.items():
        num = extract_numeric_features(email)
        email_numeric[eid] = num
        numeric_keys.update(num.keys())

        email_text[eid] = extract_text(email)

    numeric_keys = sorted(list(numeric_keys))

    # Build matrices
    X_num = []
    texts = []

    for e1, e2 in pairs:
        f1 = email_numeric[e1]
        f2 = email_numeric[e2]

        row = []
        for k in numeric_keys:
            v1 = f1.get(k, 0)
            v2 = f2.get(k, 0)

            # Use absolute difference
            row.append(abs(v1 - v2))

        X_num.append(row)

        texts.append((email_text[e1], email_text[e2]))

    return np.array(X_num), texts, numeric_keys


# -------------------------
# Text similarity features
# -------------------------

def compute_text_similarity(text_pairs):
    texts = [t for pair in text_pairs for t in pair]

    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(texts)

    sims = []

    for i in range(0, len(texts), 2):
        v1 = X[i]
        v2 = X[i + 1]

        sim = cosine_similarity(v1, v2)[0][0]
        sims.append(sim)

    return np.array(sims).reshape(-1, 1), ["text_cosine_similarity"]


# -------------------------
# Main
# -------------------------

def main(email_path, gt_path):
    print("Loading data...")
    emails = load_emails(email_path)
    clusters = load_ground_truth(gt_path)

    print("Building pairs...")
    pairs, y = build_pairs(emails, clusters)

    print(f"Pairs: {len(pairs)}")

    print("Extracting features...")
    X_num, text_pairs, num_feature_names = build_pair_features(pairs, emails)
    X_text, text_feature_names = compute_text_similarity(text_pairs)

    # Combine
    X = np.hstack([X_num, X_text])
    feature_names = num_feature_names + text_feature_names

    # Scale
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    print("Training model...")
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)

    coefs = model.coef_[0]

    # Rank features
    ranked = sorted(
        zip(feature_names, coefs),
        key=lambda x: abs(x[1]),
        reverse=True
    )

    print("\nTop 20 most important features:\n")
    for name, coef in ranked[:20]:
        print(f"{name:40s} {coef:.4f}")


# -------------------------
# CLI
# -------------------------

if __name__ == "__main__":
    cfg = PIPELINE_CONFIG.get("datasets", {})

    email_path = cfg.get("misp_json_path")
    gt_path = cfg.get("ground_truth_json")

    if not email_path or not gt_path:
        raise ValueError(
            "Missing dataset paths in PIPELINE_CONFIG['datasets'] "
            "(misp_json_path, ground_truth_json)"
        )

    # Resolve paths
    email_path = resolve_project_path(email_path)
    gt_path = resolve_project_path(gt_path)

    print(f"Using email data: {email_path}")
    print(f"Using ground truth: {gt_path}")

    main(email_path, gt_path)