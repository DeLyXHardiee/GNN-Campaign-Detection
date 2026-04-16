import json
import argparse
import itertools
import sys
import csv
import re
from collections import defaultdict
from pathlib import Path
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

try:
    from core.config.pipeline_config import (
        PIPELINE_CONFIG,
        resolve_project_path,
    )
except ModuleNotFoundError:
    # Allow running this file directly (python core/utils/OLS/OLS.py).
    project_root = Path(__file__).resolve().parents[3]
    root_str = str(project_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    from core.config.pipeline_config import (
        PIPELINE_CONFIG,
        resolve_project_path,
    )



# -------------------------
# Helpers
# -------------------------

def _normalize_external_id(value):
    if value is None:
        return ""
    return str(value).strip()


def _iter_misp_events(payload):
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict):
                if "Event" in item and isinstance(item["Event"], dict):
                    yield item["Event"]
                elif "Attribute" in item:
                    yield item
        return

    if isinstance(payload, dict):
        if "Event" in payload:
            event = payload["Event"]
            if isinstance(event, list):
                for e in event:
                    if isinstance(e, dict):
                        yield e
            elif isinstance(event, dict):
                yield event
            return

        for key in ("response", "events"):
            items = payload.get(key)
            if isinstance(items, list):
                for item in items:
                    if isinstance(item, dict):
                        if "Event" in item and isinstance(item["Event"], dict):
                            yield item["Event"]
                        elif "Attribute" in item:
                            yield item


def load_emails(path):
    """Load all emails from one MISP JSON source file as external_id -> attribute map."""
    misp_path = Path(path)
    if not misp_path.exists() or not misp_path.is_file():
        raise FileNotFoundError(f"MISP JSON file not found: {path}")

    with open(misp_path, "r", encoding="utf-8-sig") as f:
        payload = json.load(f)

    emails = {}
    for event in _iter_misp_events(payload):
        external_id = _normalize_external_id(event.get("external_id"))
        if not external_id or external_id in emails:
            continue

        attr_map = {}
        for attr in event.get("Attribute", []):
            if not isinstance(attr, dict):
                continue
            attr_type = attr.get("type")
            if not attr_type:
                continue
            attr_map[attr_type] = attr.get("value")

        emails[external_id] = attr_map

    return emails


def merge_campaigns_with_emails(clusters, emails):
    """Keep only campaign members that exist in the loaded MISP email map."""
    merged_clusters = defaultdict(list)
    merged_emails = {}
    missing = 0

    for cid, ids in clusters.items():
        for eid in ids:
            rec = emails.get(eid)
            if rec is None:
                missing += 1
                continue
            merged_clusters[cid].append(eid)
            merged_emails[eid] = rec

    return merged_clusters, merged_emails, missing


def load_ground_truth(path):
    clusters = defaultdict(list)
    path_obj = Path(path)

    if path_obj.suffix.lower() == ".csv":
        with open(path_obj, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                raw_label = str(row.get("campaign_label") or "").strip()
                m = re.search(r"(\d+)$", raw_label)
                cluster_id = int(m.group(1)) if m else (raw_label or "unknown")

                raw_ids = str(row.get("email_ids") or "").strip()
                if not raw_ids:
                    continue
                for token in raw_ids.split(","):
                    eid = _normalize_external_id(token)
                    if eid:
                        clusters[cluster_id].append(eid)
        return clusters

    with open(path_obj, "r", encoding="utf-8-sig") as f:
        data = json.load(f)

    for cluster_name, items in (data.get("clusters") or {}).items():
        m = re.search(r"(\d+)$", str(cluster_name))
        cluster_id = int(m.group(1)) if m else cluster_name

        records = []
        if isinstance(items, list):
            records = items
        elif isinstance(items, dict):
            records = items.get("emails") or items.get("records") or []
            if not records and items.get("external_ids"):
                records = [{"external_id": x} for x in items.get("external_ids", [])]

        for rec in records:
            if not isinstance(rec, dict):
                continue
            eid = _normalize_external_id(rec.get("external_id"))
            if eid:
                clusters[cluster_id].append(eid)

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


def load_text_embeddings(path):
    """Load external_id keyed subject/body embeddings from embeddings.json."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Embeddings file not found: {p}")

    with p.open("r", encoding="utf-8-sig") as f:
        payload = json.load(f)

    by_key = payload.get("by_key") or {}
    if not isinstance(by_key, dict):
        raise ValueError("Invalid embeddings format: expected object at key 'by_key'")

    vectors = {}
    for key, entry in by_key.items():
        if not isinstance(entry, dict):
            continue
        subj = entry.get("subj") or []
        body = entry.get("body") or []
        combined = np.asarray(list(subj) + list(body), dtype=np.float64)
        if combined.size == 0:
            continue
        ext_id = _normalize_external_id(entry.get("external_id") or key)
        if ext_id:
            vectors[ext_id] = combined
    return vectors


# -------------------------
# Build dataset
# -------------------------

def build_pairs(emails, clusters, max_pairs=None):
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

def _cosine_sim(a, b):
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def compute_embedding_similarity(pairs, embedding_by_id):
    sims = []
    missing = 0

    for e1, e2 in pairs:
        v1 = embedding_by_id.get(e1)
        v2 = embedding_by_id.get(e2)
        if v1 is None or v2 is None:
            sims.append(0.0)
            missing += 1
            continue
        sims.append(_cosine_sim(v1, v2))

    if missing:
        print(f"Embedding similarity fallback to 0.0 for {missing} pairs with missing vectors")

    return np.asarray(sims, dtype=np.float64).reshape(-1, 1), ["text_embedding_cosine_similarity"]


# -------------------------
# Main
# -------------------------

def main(email_path, gt_path):
    print("Loading data...")
    clusters = load_ground_truth(gt_path)
    print(f"Ground truth clusters: {len(clusters)}")

    all_misp_emails = load_emails(email_path)
    print(f"Loaded all emails from MISP: {len(all_misp_emails)}")

    if str(gt_path).lower().endswith(".csv"):
        clusters, emails, missing = merge_campaigns_with_emails(clusters, all_misp_emails)
        print(f"Merged campaign emails from MISP: {len(emails)}")
        if missing:
            print(f"Warning: {missing} campaign external IDs not found in MISP file")
    else:
        emails = all_misp_emails

    if not emails:
        raise ValueError("No emails available after joining ground truth with MISP data")

    print("Building pairs...")
    pairs, y = build_pairs(emails, clusters)

    if len(pairs) == 0:
        raise ValueError("No training pairs could be built from the joined campaign/email data")

    print(f"Pairs: {len(pairs)}")

    embeddings_path = Path(__file__).resolve().parents[1] / "embeddings" / "output" / "embeddings.json"
    print(f"Loading text embeddings: {embeddings_path}")
    embedding_by_id = load_text_embeddings(embeddings_path)

    print("Extracting features...")
    X_num, _, num_feature_names = build_pair_features(pairs, emails)
    X_text, text_feature_names = compute_embedding_similarity(pairs, embedding_by_id)

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