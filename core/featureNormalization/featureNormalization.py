from email.utils import parsedate_to_datetime
import pandas as pd
import os
import sys
from collections import Counter
import json
import re
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.url_extractor import extract_urls_from_text

def extract_time_features(date_str):
    try:
        dt = parsedate_to_datetime(date_str)
        # ensure integer values (no decimals)
        return pd.Series({
            'date_sent': dt.date().isoformat() if hasattr(dt, "date") else None,
            'time_sent': dt.time().isoformat() if hasattr(dt, "time") else None,
            'day': int(dt.day),
            'month': int(dt.month),
            'year': int(dt.year),
            'weekday': dt.strftime("%A"),
            'workday': int(1 if dt.weekday() < 5 else 0)
        })
    except:
        return pd.Series({
            'date_sent': None, 'time_sent': None, 'day': pd.NA,
            'month': pd.NA, 'year': pd.NA, 'weekday': None, 'workday': pd.NA
        })


def extract_subject_features(subj):
    if not isinstance(subj, str):
        subj = ""

    num_chars = len(subj)
    num_whitespace = sum(c.isspace() for c in subj)

    return pd.Series({
        "subject_length": num_chars,
        "subject_whitespace_count": num_whitespace
    })

def extract_greeting_features(body):
    """
    Simplified greeting extractor returning a single key 'greeting'.
    Examples: {'greeting': 'hi, name'}, {'greeting': 'hello, email'}, {'greeting': None}
    """
    if not isinstance(body, str) or not body.strip():
        return {"greeting": None}

    text = body.lstrip()
    first_line = text.splitlines()[0].strip()

    greetings = [
        "good morning", "good afternoon", "good evening",
        "dearest", "greetings", "hello", "hi", "hey", "dear"
    ]

    greet_regex = r'^(?:' + '|'.join(re.escape(g) for g in greetings) + r')\b[,\s:!-]*([^\n,!?]*)'
    m = re.search(greet_regex, first_line, flags=re.I)
    if not m:
        return {"greeting": None}

    greet_found = re.match(r'^(?:' + '|'.join(re.escape(g) for g in greetings) + r')\b', first_line, flags=re.I)
    greeting_token = greet_found.group(0).strip().lower() if greet_found else None

    follow_text = (m.group(1) or "").strip()
    if not follow_text:
        return {"greeting": greeting_token}

    # detect email
    email_re = re.compile(r'[\w\.-]+@[\w\.-]+\.\w+')
    if email_re.search(follow_text):
        return {"greeting": f"{greeting_token}, email"}

    # tokenise following text and apply simple heuristics
    tokens = re.findall(r"[^\s,;:()<>\"']+", follow_text)
    first_tok = tokens[0] if tokens else ""

    # name heuristic: capitalized word(s)
    if re.match(r'^[A-Z][a-z\'-]+$', first_tok):
        return {"greeting": f"{greeting_token}, name"}

    # username heuristic: lowercase or contains digits/dots/underscores
    if re.match(r'^[\w\.-]{2,}$', first_tok):
        return {"greeting": f"{greeting_token}, username"}

    return {"greeting": f"{greeting_token}, other"}

def extract_attachment_features(attachments):
    return

def extract_origin_based_features(sender):
    return

def extract_recipient_based_features(recipients):
    return

def extract_url_based_features(urls):
    return

def compute_and_save_idf(subjects, output_path, max_features=2000):
    """Compute and save IDF values for subject terms to a separate CSV"""
    vectorizer = TfidfVectorizer(max_features=max_features) 
    vectorizer.fit(subjects) 
    
    # Get IDF values and terms
    terms = vectorizer.get_feature_names_out()
    idfs = vectorizer.idf_
    
    # Create and save IDF DataFrame
    idf_df = pd.DataFrame({
        'term': terms,
        'idf': idfs
    }).sort_values('idf', ascending=False)
    
    idf_df.to_csv(output_path, index=False)
    print(f"Saved IDF values to: {output_path}")
    return vectorizer

def compute_and_save_term_frequencies(subjects, output_path):
    """Compute and save term frequencies as JSON objects (subjects)"""
    email_frequencies = []
    
    for idx, subject in enumerate(subjects):
        if not isinstance(subject, str):
            subject = ""
            
        # Simple word splitting to keep all words
        words = subject.lower().split()
        
        # Get word frequencies for this subject
        word_freq = dict(Counter(words))
        
        # Only store if we found words
        if word_freq:
            email_frequencies.append({
                "email_id": idx,
                "subject": subject,  # store original subject for reference
                "frequencies": word_freq
            })
    
    # Save as JSON
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(email_frequencies, f, indent=2, ensure_ascii=False)
    
    print(f"Saved subject term frequencies to: {output_path}")

def compute_and_save_body_bow(bodies, output_path):
    """Compute and save bag-of-words (term frequencies) for each email body as JSON objects"""
    email_bows = []
    
    for idx, body in enumerate(bodies):
        if not isinstance(body, str):
            body = ""
        
        # Use regex word tokenizer to keep common words and punctuation-free tokens
        words = re.findall(r"\w+", body.lower())
        word_freq = dict(Counter(words))
        
        if word_freq:
            email_bows.append({
                "email_id": idx,
                # omit storing full body to keep file smaller; add if you need it
                "frequencies": word_freq
            })
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(email_bows, f, indent=2, ensure_ascii=False)
    
    print(f"Saved body bag-of-words to: {output_path}")

def run_feature_normalization(csv_path, max_features=2000):
    # Read CSV
    df = pd.read_csv(csv_path)
    
    # Filter rows where label == 1 and cap to 1000 rows
    df = df[df.get("label", 0) == 1]#.head(1000)

    features_list = []

    for idx, row in df.iterrows():
        if row.get("label", 0) != 1:
            continue

        time_features = extract_time_features(row.get("date"))
        subject_features = extract_subject_features(row.get("subject"))

        body = row.get("body", "") if row.get("body", "") is not None else ""
        
        # Extract URLs from the email body (using utils/url_extractor.py)
        extracted_urls = extract_urls_from_text(body) if body else []

        # Extract greeting features from body
        greeting_features = extract_greeting_features(body)

        # Build feature dict for this row
        feat = {
            "email_index": idx,  # Add email index to features
            "num_urls": len(extracted_urls),
            "has_urls": 1 if len(extracted_urls) > 0 else 0,
            "first_url": extracted_urls[0] if len(extracted_urls) > 0 else None,
            "body_length": len(body),
            "body_word_count": len(re.findall(r"\w+", body.lower())),
            "body_unique_word_count": len(set(re.findall(r"\w+", body.lower())))
        }

        # merge time, subject and greeting features (Series -> dict)
        feat.update(time_features.to_dict())
        feat.update(subject_features.to_dict())
        feat.update(greeting_features)

        features_list.append(feat)

    # Create dataframe and write to CSV next to input file
    features_df = pd.DataFrame(features_list)

    # Compute TF-IDF on subjects (aligned with the filtered df order)
    subjects = df[df.get("label", 0) == 1].get("subject", pd.Series([""])).fillna("").astype(str).tolist()
    bodies = df[df.get("label", 0) == 1].get("body", pd.Series([""])).fillna("").astype(str).tolist()

    # ensure integer columns are integer dtype without decimals (nullable Int64)
    int_cols = [c for c in ("day", "month", "year", "workday") if c in features_df.columns]
    if int_cols:
        features_df[int_cols] = features_df[int_cols].astype("Int64")

    input_dir = os.path.dirname(csv_path)
    input_base = os.path.splitext(os.path.basename(csv_path))[0]
    output_path = os.path.join(input_dir, f"{input_base}_normalized.csv")

    features_df.to_csv(output_path, index=False)
    print(f"Saved normalized features to: {output_path}")

    # Save IDF values to separate CSV (subjects)
    idf_path = os.path.join(input_dir, f"{input_base}_subject_idf.csv")
    compute_and_save_idf(subjects, idf_path, max_features)

    # Save subject term frequencies to JSON
    term_freq_path = os.path.join(input_dir, f"{input_base}_term_frequencies.json")
    compute_and_save_term_frequencies(subjects, term_freq_path)

    # Save body bag-of-words (per-email JSON)
    body_bow_path = os.path.join(input_dir, f"{input_base}_body_bow.json")
    compute_and_save_body_bow(bodies, body_bow_path)
    

run_feature_normalization("../data/csv/TREC-07.csv")
