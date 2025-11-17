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
        return {
            'date_sent': dt.date().isoformat() if hasattr(dt, "date") else None,
            'time_sent': dt.time().isoformat() if hasattr(dt, "time") else None,
            'day': int(dt.day),
            'month': int(dt.month),
            'year': int(dt.year),
            'weekday': dt.strftime("%A"),
            'workday': int(1 if dt.weekday() < 5 else 0)
        }
    except:
        return {
            'date_sent': None, 'time_sent': None, 'day': None,
            'month': None, 'year': None, 'weekday': None, 'workday': None
        }

def extract_subject_features(subject):
    if not isinstance(subject, str):
        subject = ""

    num_chars = len(subject)
    num_whitespace = sum(c.isspace() for c in subject)

    return {
        "subject_length": num_chars,
        "subject_whitespace_count": num_whitespace,
        "subject_term_frequency": get_term_frequency(subject)
    }

def get_term_frequency(subject):
    
    if not isinstance(subject, str):
        subject = ""
        
    terms = subject.lower().split()
    
    return dict(Counter(terms))
    
def save_term_frequencies(subjects, output_path):
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
    

def get_idf(subjects, output_path, max_features=2000):
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

def extract_body_based_features(body):
    # Extract URLs from the email body (using utils/url_extractor.py)
    extracted_urls = extract_urls_from_text(body) if body else []
    
    # Calculate number of lines
    num_lines = len(body.splitlines())
    
    # Calculate number of words
    words = re.findall(r"\w+", body.lower())
    num_words = len(words)
    
    # Calculate average word length
    avg_word_length = round(sum(len(word) for word in words) / num_words,1) if num_words > 0 else 0

    # Extract greeting features from body
    greeting_features = extract_greeting_features(body)

    bow = compute_body_bow(body)

    # Build feature dict for this row
    return {
        "num_urls": len(extracted_urls),
        "has_urls": 1 if len(extracted_urls) > 0 else 0,
        "body_word_count": num_words,
        "num_lines": num_lines,
        "avg_word_length": avg_word_length,
        "greeting": greeting_features.get("greeting", None),
        "bow": bow
    }

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
    """
    Extract sender name and email from two possible formats:
    1. email@domain.com -> name is 'email'
    2. Name <email@domain.com> -> name is 'Name'
    Removes surrounding quotes from names (e.g., "name lastname" -> name lastname)
    """
    if not isinstance(sender, str) or not sender.strip():
        return {"sender_name": None, "sender_email": None}
    
    sender = sender.strip()
    
    # Format 2: Name <email@domain.com>
    angle_match = re.search(r'^(.+?)\s*<(.+?)>', sender)
    if angle_match:
        name = angle_match.group(1).strip()
        email = angle_match.group(2).strip()
        # Remove surrounding quotes from name
        name = name.strip('"')
        return {"sender_name": name if name else None, "sender_email": email}
    
    # Format 1: email@domain.com
    email_match = re.match(r'^([\w\.-]+)@[\w\.-]+\.\w+$', sender)
    if email_match:
        name = email_match.group(1)
        return {"sender_name": name, "sender_email": sender}
    
    # Fallback: couldn't parse
    return {"sender_name": None, "sender_email": None}

def extract_recipient_based_features(recipient):
    """
    Extract recipient name and email from two possible formats:
    1. email@domain.com -> name is 'email'
    2. Name <email@domain.com> -> name is 'Name'
    """
    if not isinstance(recipient, str) or not recipient.strip():
        return {"recipient_name": None, "recipient_email": None}
    
    recipient = recipient.strip()
    
    # Format 2: Name <email@domain.com>
    angle_match = re.search(r'^(.+?)\s*<(.+?)>', recipient)
    if angle_match:
        name = angle_match.group(1).strip()
        email = angle_match.group(2).strip()
        return {"recipient_name": name if name else None, "recipient_email": email}
    
    # Format 1: email@domain.com
    email_match = re.match(r'^([\w\.-]+)@[\w\.-]+\.\w+$', recipient)
    if email_match:
        name = email_match.group(1)
        return {"recipient_name": name, "recipient_email": recipient}
    
    # Fallback: couldn't parse
    return {"recipient_name": None, "recipient_email": None}

def extract_url_based_features(urls):
    """
    Extract URL-based features for phishing detection.
    
    Features include:
    - Domain and hostname information
    - Subdomain and hyphen counts
    - Binary features (EV cert, @symbol, non-ASCII, IP address, etc.)
    - URL counts (short URLs, blacklisted, different domains)
    - Hyperlink properties
    """
    if not urls or not isinstance(urls, list):
        return {
            "num_urls": 0,
            "num_different_domains": 0,
            "num_urls_with_ip": 0,
            "num_short_urls": 0,
            "has_at_symbol": 0,
            "has_non_ascii_chars": 0,
            "has_extra_http": 0,
            "avg_subdomain_count": 0,
            "avg_hyphen_count": 0,
            
            #"has_ev_certificate": 0,
            #"has_webhost_domain": 0,
            #"num_blacklisted_urls": 0,
            #"has_phishtank_domain": 0,
            #"has_domain_typos": 0,
            #"has_click_here_link": 0,
            #"visual_url_mismatch": 0
        }
    
    domains = set()
    urls_with_ip = 0
    short_urls = 0
    at_symbol_count = 0
    non_ascii_count = 0
    extra_http_count = 0
    total_subdomains = 0
    total_hyphens = 0
    blacklisted_urls = 0
    ev_cert_count = 0
    webhost_count = 0
    phishtank_count = 0
    typo_count = 0
    
    for url in urls:
        if not isinstance(url, str):
            continue
        
        # Extract domain from URL
        try:
            # Remove protocol
            url_clean = url.replace("https://", "").replace("http://", "")
            domain = url_clean.split("/")[0].split("?")[0]
            domains.add(domain)
        except:
            domain = ""
        
        # Check for IP address (simple pattern)
        if re.match(r'\d+\.\d+\.\d+\.\d+', domain):
            urls_with_ip += 1
        
        # Check for short URL services (bit.ly, tinyurl, etc.)
        short_url_services = ['bit.ly', 'tinyurl.com', 'short.link', 'ow.ly', 'goo.gl']
        if any(service in url.lower() for service in short_url_services):
            short_urls += 1
        
        # Check for @ symbol (redirect technique)
        if '@' in url:
            at_symbol_count += 1
        
        # Check for non-ASCII characters
        try:
            url.encode('ascii')
        except UnicodeEncodeError:
            non_ascii_count += 1
        
        # Check for extra http/https
        if url.count('http') > 1 or url.count('https') > 1:
            extra_http_count += 1
        
        # Count subdomains (simplified: count dots minus 1)
        if domain:
            subdomain_count = domain.count('.') - 1
            total_subdomains += max(0, subdomain_count)
            
            # Count hyphens in domain
            hyphen_count = domain.count('-')
            total_hyphens += hyphen_count
        
        # ev_cert_count += check_ev_certificate(url)
        # blacklisted_urls += check_blacklist(url)
        # webhost_count += check_webhost(domain)
        # phishtank_count += check_phishtank(domain)
        # typo_count += check_typos(domain)
    
    # Calculate averages
    avg_subdomains = total_subdomains / len(urls) if urls else 0
    avg_hyphens = total_hyphens / len(urls) if urls else 0
    
    return {
        "num_urls": len(urls),
        "num_different_domains": len(domains),
        "num_urls_with_ip": urls_with_ip,
        "num_short_urls": short_urls,
        "has_at_symbol": 1 if at_symbol_count > 0 else 0,
        "has_non_ascii_chars": 1 if non_ascii_count > 0 else 0,
        "has_extra_http": 1 if extra_http_count > 0 else 0,
        "avg_subdomain_count": round(avg_subdomains, 2),
        "avg_hyphen_count": round(avg_hyphens, 2),
        #"has_ev_certificate": 1 if ev_cert_count > 0 else 0,
        #"num_blacklisted_urls": blacklisted_urls,
        #"has_webhost_domain": 1 if webhost_count > 0 else 0,
        #"has_phishtank_domain": 1 if phishtank_count > 0 else 0,
        #"has_domain_typos": 1 if typo_count > 0 else 0,
        #"has_click_here_link": 0,  # Would need body text analysis
        #"visual_url_mismatch": 0   # Would need hyperlink/display text comparison
    }

def compute_body_bow(body):
    """Compute and return bag-of-words (term frequencies) for a single email body"""
    if not isinstance(body, str):
        body = ""
    
    # Use regex word tokenizer to keep common words and punctuation-free tokens
    words = re.findall(r"\w+", body.lower())
    word_freq = dict(Counter(words))
    
    return word_freq

def compute_and_save_body_bow(bodies, output_path):
    """Compute and save bag-of-words (term frequencies) for each email body as JSON objects"""
    # should also include latent semantic analysis at some point #TODO
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


def get_FS6(csv_path):
    """
    Extract FS6 features: subject (length + spaces), attachments (count + size), 
    date, body, origin (sender_name only), and URL features.
    """
    features_list = extract_features(csv_path, ["subject", "time", "body", "origin", "urls"])
    #features_list = extract_features(csv_path, ["subject", "attachments", "time", "body", "origin", "urls"])
    
    # Filter out unwanted keys from each feature dictionary
    filtered_features = []
    for feat in features_list:
        filtered_feat = {k: v for k, v in feat.items() 
                        if k not in ["subject_term_frequency", "bow", "sender_email", "greeting"]
                        }
        filtered_features.append(filtered_feat)
    
    return filtered_features

def get_FS7(csv_path):
    """
    Extract FS7 features: subject, body, origin (sender_name only), and urls.
    Excludes subject/body term frequencies and only includes sender_name from origin.
    """
    features_list = extract_features(csv_path, ["subject", "body", "origin", "urls"])
    
    # Filter out unwanted keys from each feature dictionary
    filtered_features = []
    for feat in features_list:
        filtered_feat = {k: v for k, v in feat.items() 
                        if not (k == "bow")  # Exclude body bag-of-words
                        and not (k == "sender_email")}  # Only keep sender_name, not sender_email
        
        filtered_features.append(filtered_feat)
    
    return filtered_features

def extract_features(csv_path, features):
    """
    Extract baseline features for specified types.
    
    Args:
        csv_path: Path to the CSV file
        types: List of feature types to extract (e.g., ["time", "subject", "body", "attachments", "origin", "receiver", "urls"])
    
    Returns:
        List of feature dictionaries, one per row
    """
    df = pd.read_csv(csv_path)
    features_list = []
    
    for idx, row in df.iterrows():
        feat = {'email_index': idx}
        
        # Extract features for each requested type
        for feature_type in features:
            if feature_type == "time":
                feat.update(extract_time_features(row.get("date")))
            elif feature_type == "subject":
                feat.update(extract_subject_features(row.get("subject")))
            elif feature_type == "body":
                feat.update(extract_body_based_features(row.get("body")))
            elif feature_type == "attachments":
                continue
                #Not implemented yet
                feat.update(extract_attachment_features(row.get("attachments")))
            elif feature_type == "origin":
                feat.update(extract_origin_based_features(row.get("sender")))
            elif feature_type == "receiver":
                feat.update(extract_recipient_based_features(row.get("receiver")))
            elif feature_type == "urls":
                body = row.get("body", "") if row.get("body", "") is not None else ""
                extracted_urls = extract_urls_from_text(body) if body else []
                feat.update(extract_url_based_features(extracted_urls))
        
        features_list.append(feat)
    
    return features_list

'''
def run_feature_normalization(csv_path, max_features=2000):
    # Read CSV
    df = pd.read_csv(csv_path)
    
    # Filter rows where label == 1 and cap to 1000 rows
    #df = df.head(1000)

    features_list = []

    for idx, row in df.iterrows():

        time_features = extract_time_features(row.get("date"))
        subject_features = extract_subject_features(row.get("subject"))

        body = row.get("body", "") if row.get("body", "") is not None else ""
        
        # Extract URLs from the email body (using utils/url_extractor.py)
        extracted_urls = extract_urls_from_text(body) if body else []

        # Extract greeting features from body
        greeting_features = extract_greeting_features(body)

        # Build feature dict for this row
        feat = {
            "email_index": idx+1,  # Add email index to features
            "num_urls": len(extracted_urls),
            "has_urls": 1 if len(extracted_urls) > 0 else 0,
            "first_url": extracted_urls[0] if len(extracted_urls) > 0 else None,
            "body_length": len(body),
            "body_word_count": len(re.findall(r"\w+", body.lower())),
            "body_unique_word_count": len(set(re.findall(r"\w+", body.lower())))
        }

        # merge time, subject and greeting features (Series -> dict)
        feat.update(time_features)
        feat.update(subject_features)
        feat.update(greeting_features)

        features_list.append(feat)

    # Create dataframe and write to CSV next to input file
    features_df = pd.DataFrame(features_list)

    # Compute TF-IDF on subjects (aligned with the filtered df order)
    subjects = df.get("subject", pd.Series([""])).fillna("").astype(str).tolist()
    bodies = df.get("body", pd.Series([""])).fillna("").astype(str).tolist()

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
    get_idf(subjects, idf_path, max_features)

    # Save subject term frequencies to JSON
    term_freq_path = os.path.join(input_dir, f"{input_base}_term_frequencies.json")
    save_term_frequencies(subjects, term_freq_path)

    # Save body bag-of-words (per-email JSON)
    body_bow_path = os.path.join(input_dir, f"{input_base}_body_bow.json")
    compute_and_save_body_bow(bodies, body_bow_path)

run_feature_normalization("../../data/csv/TREC-07-only-phishing.csv")
    
'''



if __name__ == "__main__":
    csv_path = "../../data/csv/TREC-07-only-phishing.csv"
    
    # Extract FS features
    fs_features = get_FS6(csv_path)
    
    # Save to JSON file
    input_dir = os.path.dirname(csv_path)
    input_base = os.path.splitext(os.path.basename(csv_path))[0]
    output_path = os.path.join(input_dir, f"{input_base}_FS6.json")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(fs_features, f, indent=2, ensure_ascii=False)
    
    print(f"Saved FS6 features to: {output_path}")
    print(f"Total emails processed: {len(fs_features)}")
    if fs_features:
        print(f"Sample feature keys: {list(fs_features[0].keys())}")