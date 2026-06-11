import json
import argparse
import itertools
import sys
import csv
import re
import ast
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
    project_root = Path(__file__).resolve().parents[3]
    root_str = str(project_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    from core.config.pipeline_config import (
        PIPELINE_CONFIG,
        resolve_project_path,
    )




def _normalize_external_id(value):
    if value is None:
        return ""
    return str(value).strip()


def _sanitize_feature_key(key):
    return re.sub(r"[^0-9a-zA-Z_]+", "_", str(key)).strip("_") or "field"


def _flatten_numeric_dict(data, prefix="", out=None):
    """Recursively flatten nested dictionaries/lists into numeric features.

    Includes numeric/bool leaves and list length signals.
    Repeated paths are summarized with mean/min/max.
    """
    if out is None:
        out = {}

    if not isinstance(data, dict):
        return out

    bucket = defaultdict(list)

    def _collect_numeric_values(value, key_prefix):
        if isinstance(value, dict):
            for raw_k, raw_v in value.items():
                k = _sanitize_feature_key(raw_k)
                child = f"{key_prefix}_{k}" if key_prefix else k
                _collect_numeric_values(raw_v, child)
            return

        if isinstance(value, list):
            if key_prefix:
                bucket[f"{key_prefix}_len"].append(float(len(value)))
            for item in value:
                _collect_numeric_values(item, key_prefix)
            return

        if isinstance(value, bool):
            if key_prefix:
                bucket[key_prefix].append(float(int(value)))
            return

        if isinstance(value, (int, float, np.integer, np.floating)):
            if key_prefix:
                bucket[key_prefix].append(float(value))

    _collect_numeric_values(data, prefix)

    for key, vals in bucket.items():
        if not vals:
            continue
        if len(vals) == 1:
            out[key] = float(vals[0])
        else:
            arr = np.asarray(vals, dtype=np.float64)
            out[f"{key}_mean"] = float(np.mean(arr))
            out[f"{key}_min"] = float(np.min(arr))
            out[f"{key}_max"] = float(np.max(arr))
            out[f"{key}_sum"] = float(np.sum(arr))

    return out


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

    if isinstance(payload, list):
        direct_records = [x for x in payload if isinstance(x, dict) and x.get("external_id")]
        if direct_records:
            emails = {}
            for record in direct_records:
                external_id = _normalize_external_id(record.get("external_id"))
                if not external_id or external_id in emails:
                    continue
                emails[external_id] = record
            return emails

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



def safe_bool(x):
    if isinstance(x, bool):
        return int(x)
    if isinstance(x, str):
        return int(x.lower() == "true")
    return 0


def _maybe_parse_json_like(value):
    if not isinstance(value, str):
        return value

    stripped = value.strip()
    if not stripped:
        return ""

    if stripped[0] not in "[{":
        return value

    try:
        return json.loads(stripped)
    except Exception:
        try:
            return ast.literal_eval(stripped)
        except Exception:
            return value


def _textify(value):
    value = _maybe_parse_json_like(value)

    if value is None:
        return ""

    if isinstance(value, str):
        return value

    if isinstance(value, bool):
        return "true" if value else "false"

    if isinstance(value, (int, float, np.integer, np.floating)):
        return str(value)

    if isinstance(value, list):
        parts = [_textify(item) for item in value]
        return " ".join([p for p in parts if p])

    if isinstance(value, dict):
        parts = []
        for k, v in value.items():
            k_txt = _textify(k)
            v_txt = _textify(v)
            if k_txt:
                parts.append(k_txt)
            if v_txt:
                parts.append(v_txt)
        return " ".join(parts)

    return str(value)


def _tokenize_text(value):
    text = _textify(value).lower()
    return re.findall(r"[a-z0-9@._:/+-]+", text)


def _jaccard_similarity(tokens_a, tokens_b):
    set_a = set(tokens_a)
    set_b = set(tokens_b)
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    return float(len(set_a & set_b) / len(set_a | set_b))


def _extract_attachment_meta_text(email):
    meta = email.get("attachments_meta", [])
    if not isinstance(meta, list):
        return ""

    parts = []
    keep_keys = {
        "filename",
        "content_type",
        "content_disposition",
        "content_id",
        "scan_status",
    }
    for item in meta:
        if not isinstance(item, dict):
            continue
        for key in keep_keys:
            if key in item:
                parts.append(_textify(item.get(key)))
    return " ".join([p for p in parts if p])


def _extract_header_text(email):
    parts = []
    for key, value in email.items():
        if str(key).startswith("header_"):
            parts.append(_textify(value))
    return " ".join([p for p in parts if p])


def _extract_header_text_by_type(email):
    headers = {}
    for key, value in email.items():
        key_str = str(key)
        if not key_str.startswith("header_"):
            continue
        header_name = key_str[len("header_"):].strip() or "unknown"
        headers[header_name] = _textify(value)
    return headers


def extract_text_fields(email):
    headers_by_type = _extract_header_text_by_type(email)
    fields = {
        "subject": _textify(email.get("subject", "")),
        "body": _textify(email.get("body", "")),
        "from": _textify(email.get("from", "")),
        "to": _textify(email.get("to", "")),
        "url": _textify(email.get("url", "")),
        "category": _textify(email.get("category", "")),
        "date": _textify(email.get("date", "")),
        "headers": " ".join([v for v in headers_by_type.values() if v]),
        "headers_by_type": headers_by_type,
        "attachments_meta": _extract_attachment_meta_text(email),
        "attachments": _textify(email.get("attachments", "")),
    }
    fields["all_text"] = " ".join(
        [v for k, v in fields.items() if isinstance(v, str) and k != "headers_by_type" and v]
    )

    dynamic_text_by_key = {}

    def _accumulate_text(value, key_prefix):
        value = _maybe_parse_json_like(value)

        if value is None:
            return

        if isinstance(value, dict):
            for raw_k, raw_v in value.items():
                k = _sanitize_feature_key(raw_k)
                child = f"{key_prefix}_{k}" if key_prefix else k
                _accumulate_text(raw_v, child)
            return

        if isinstance(value, list):
            for item in value:
                _accumulate_text(item, key_prefix)
            return

        if isinstance(value, (bool, int, float, np.integer, np.floating)):
            return

        txt = _textify(value).strip()
        if not txt or not key_prefix:
            return

        existing = dynamic_text_by_key.get(key_prefix, "")
        dynamic_text_by_key[key_prefix] = (existing + " " + txt).strip() if existing else txt

    for raw_key, raw_value in email.items():
        key = str(raw_key)
        if key.startswith("header_"):
            continue
        _accumulate_text(raw_value, _sanitize_feature_key(key))

    fields["dynamic_text_by_key"] = dynamic_text_by_key
    return fields


def build_text_pair_features(pairs, email_text_fields):
    header_types = sorted(
        {
            header_type
            for fields in email_text_fields.values()
            for header_type in fields.get("headers_by_type", {}).keys()
        }
    )

    feature_names = [
        "subject_token_jaccard",
        "body_token_jaccard",
        "from_token_jaccard",
        "to_token_jaccard",
        "url_token_jaccard",
        "attachments_meta_token_jaccard",
        "subject_length_abs_diff",
        "body_length_abs_diff",
        "category_exact_match",
        "date_exact_match",
    ]
    feature_names.extend(
        [f"header_{_sanitize_feature_key(h)}_token_jaccard" for h in header_types]
    )

    dynamic_text_keys = sorted(
        {
            k
            for fields in email_text_fields.values()
            for k in fields.get("dynamic_text_by_key", {}).keys()
        }
    )
    feature_names.extend([f"attr_{k}_token_jaccard" for k in dynamic_text_keys])

    X = []
    for e1, e2 in pairs:
        f1 = email_text_fields[e1]
        f2 = email_text_fields[e2]

        subj1 = _tokenize_text(f1["subject"])
        subj2 = _tokenize_text(f2["subject"])
        body1 = _tokenize_text(f1["body"])
        body2 = _tokenize_text(f2["body"])
        from1 = _tokenize_text(f1["from"])
        from2 = _tokenize_text(f2["from"])
        to1 = _tokenize_text(f1["to"])
        to2 = _tokenize_text(f2["to"])
        url1 = _tokenize_text(f1["url"])
        url2 = _tokenize_text(f2["url"])
        att1 = _tokenize_text(f1["attachments_meta"])
        att2 = _tokenize_text(f2["attachments_meta"])

        row = [
            _jaccard_similarity(subj1, subj2),
            _jaccard_similarity(body1, body2),
            _jaccard_similarity(from1, from2),
            _jaccard_similarity(to1, to2),
            _jaccard_similarity(url1, url2),
            _jaccard_similarity(att1, att2),
            abs(len(f1["subject"]) - len(f2["subject"])),
            abs(len(f1["body"]) - len(f2["body"])),
            float(f1["category"].strip().lower() == f2["category"].strip().lower()),
            float(f1["date"].strip() == f2["date"].strip()),
        ]

        headers1 = f1.get("headers_by_type", {})
        headers2 = f2.get("headers_by_type", {})
        for header_type in header_types:
            h1 = _tokenize_text(headers1.get(header_type, ""))
            h2 = _tokenize_text(headers2.get(header_type, ""))
            row.append(_jaccard_similarity(h1, h2))

        dynamic1 = f1.get("dynamic_text_by_key", {})
        dynamic2 = f2.get("dynamic_text_by_key", {})
        for key in dynamic_text_keys:
            t1 = _tokenize_text(dynamic1.get(key, ""))
            t2 = _tokenize_text(dynamic2.get(key, ""))
            row.append(_jaccard_similarity(t1, t2))

        X.append(row)

    return np.asarray(X, dtype=np.float64), feature_names


def extract_numeric_features(email):
    features = _flatten_numeric_dict(email)

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
    fields = extract_text_fields(email)
    return fields["all_text"]


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



def build_pairs(emails, clusters, max_pairs=float("inf")):
    pairs = []
    labels = []

    cluster_lookup = {}
    for cid, ids in clusters.items():
        for eid in ids:
            cluster_lookup[eid] = cid

    email_ids = list(emails.keys())

    for i, j in itertools.combinations(email_ids, 2):
        same = int(cluster_lookup.get(i) == cluster_lookup.get(j))
        pairs.append((i, j))
        labels.append(same)

        if len(pairs) >= max_pairs:
            break

    return pairs, np.array(labels)


def build_global_labeled_pairs(clusters, emails, max_pairs=float("inf")):
    """Build globally labeled pairs using only emails present in ground truth clusters.

    Positive pairs: same campaign.
    Negative pairs: different campaigns.
    """
    cluster_lookup = {}
    for cid, ids in clusters.items():
        for eid in ids:
            if eid in emails:
                cluster_lookup[eid] = cid

    labeled_ids = sorted(cluster_lookup.keys())
    pairs = []
    labels = []

    for e1, e2 in itertools.combinations(labeled_ids, 2):
        same = int(cluster_lookup[e1] == cluster_lookup[e2])
        pairs.append((e1, e2))
        labels.append(same)

        if len(pairs) >= max_pairs:
            break

    return pairs, np.asarray(labels, dtype=np.int32)


def build_campaign_pairs(campaign_email_ids, all_email_ids, negative_multiplier=3, seed=42):
    """Build one-vs-rest pair set for a single campaign.

    Positive pairs: both emails in campaign.
    Negative pairs: one email in campaign, one outside campaign.
    """
    campaign_set = set(campaign_email_ids)
    outside_ids = [eid for eid in all_email_ids if eid not in campaign_set]

    positives = list(itertools.combinations(campaign_email_ids, 2))
    negatives = [(e1, e2) for e1 in campaign_email_ids for e2 in outside_ids]

    if not positives or not negatives:
        return [], np.array([])

    max_negatives = min(len(negatives), len(positives) * max(1, int(negative_multiplier)))
    if len(negatives) > max_negatives:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(negatives), size=max_negatives, replace=False)
        negatives = [negatives[i] for i in idx]

    pairs = positives + negatives
    labels = np.array([1] * len(positives) + [0] * len(negatives), dtype=np.int32)
    return pairs, labels



def build_pair_features(pairs, emails):
    numeric_keys = set()

    email_numeric = {}
    email_text = {}
    email_text_fields = {}

    for eid, email in emails.items():
        num = extract_numeric_features(email)
        email_numeric[eid] = num
        numeric_keys.update(num.keys())

        email_text[eid] = extract_text(email)
        email_text_fields[eid] = extract_text_fields(email)

    numeric_keys = sorted(list(numeric_keys))

    X_num = []
    texts = []

    for e1, e2 in pairs:
        f1 = email_numeric[e1]
        f2 = email_numeric[e2]

        row = []
        for k in numeric_keys:
            v1 = f1.get(k, 0)
            v2 = f2.get(k, 0)

            row.append(abs(v1 - v2))

        X_num.append(row)

        texts.append((email_text[e1], email_text[e2]))

    X_textual, textual_feature_names = build_text_pair_features(pairs, email_text_fields)
    return np.array(X_num), X_textual, texts, numeric_keys, textual_feature_names



def main(email_path, gt_path):
    print("Loading data...")
    clusters = load_ground_truth(gt_path)
    print(f"Ground truth clusters: {len(clusters)}")

    hardcoded_email_path = resolve_project_path(
        "core/feature_set_extraction/output/featuresets/TREC-07-only-phishing-6m-FS1.json"
    )
    hardcoded_email_path = resolve_project_path("core/preprocessing/output/incidents-lake-misp-url-fixed.json")
    print(f"Using hardcoded feature set input: {hardcoded_email_path}")

    all_misp_emails = load_emails(hardcoded_email_path)
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

    print("Computing global signal impact...")
    pairs, y = build_global_labeled_pairs(clusters, emails)
    if len(pairs) == 0 or len(np.unique(y)) < 2:
        raise ValueError("No global model could be trained (insufficient labeled positive/negative pairs)")

    X_num, X_textual, _, num_feature_names, textual_feature_names = build_pair_features(pairs, emails)

    X = np.hstack([X_num, X_textual])
    feature_names = num_feature_names + textual_feature_names

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)

    coefs = model.coef_[0]
    ranked = sorted(
        zip(feature_names, coefs),
        key=lambda x: abs(x[1]),
        reverse=True,
    )

    campaign_impacts = {
        "global": {
            name: float(abs(coef))
            for name, coef in ranked
        }
    }

    output_path = Path.cwd() / "campaign_signal_impacts.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(campaign_impacts, f, indent=2)

    print(f"Wrote campaign signal impact JSON: {output_path}")

    attribute_totals = defaultdict(float)
    attribute_counts = defaultdict(int)
    for impact_map in campaign_impacts.values():
        for attr, score in impact_map.items():
            attribute_totals[attr] += float(score)
            attribute_counts[attr] += 1

    attribute_averages = {
        attr: attribute_totals[attr] / attribute_counts[attr]
        for attr in attribute_totals
        if attribute_counts[attr] > 0
    }

    sorted_attribute_averages = dict(
        sorted(attribute_averages.items(), key=lambda x: x[1], reverse=True)
    )

    avg_output_path = Path.cwd() / "attribute_average_impacts.json"
    with avg_output_path.open("w", encoding="utf-8") as f:
        json.dump(sorted_attribute_averages, f, indent=2)

    print(f"Wrote attribute average impact JSON: {avg_output_path}")



if __name__ == "__main__":
    cfg = PIPELINE_CONFIG.get("datasets", {})

    email_path = cfg.get("misp_json_path")
    gt_path = cfg.get("ground_truth_json")

    if not email_path or not gt_path:
        raise ValueError(
            "Missing dataset paths in PIPELINE_CONFIG['datasets'] "
            "(misp_json_path, ground_truth_json)"
        )

    email_path = resolve_project_path(email_path)
    gt_path = resolve_project_path(gt_path)

    print(f"Using email data: {email_path}")
    print(f"Using ground truth: {gt_path}")

    main(email_path, gt_path)