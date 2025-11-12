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

def extract_body_based_features(body):
    return

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
    """
    if not isinstance(sender, str) or not sender.strip():
        return {"sender_name": None, "sender_email": None}
    
    sender = sender.strip()
    
    # Format 2: Name <email@domain.com>
    angle_match = re.search(r'^(.+?)\s*<(.+?)>', sender)
    if angle_match:
        name = angle_match.group(1).strip()
        email = angle_match.group(2).strip()
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
        
        # Placeholder: would require external data sources
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
        feat.update(time_features.to_dict())
        feat.update(subject_features.to_dict())
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
    compute_and_save_idf(subjects, idf_path, max_features)

    # Save subject term frequencies to JSON
    term_freq_path = os.path.join(input_dir, f"{input_base}_term_frequencies.json")
    compute_and_save_term_frequencies(subjects, term_freq_path)

    # Save body bag-of-words (per-email JSON)
    body_bow_path = os.path.join(input_dir, f"{input_base}_body_bow.json")
    compute_and_save_body_bow(bodies, body_bow_path)
    

run_feature_normalization("../../data/csv/TREC-07-only-phishing.csv")
