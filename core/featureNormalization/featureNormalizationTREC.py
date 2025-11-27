from email.utils import parsedate_to_datetime
import pandas as pd
import os
import sys
import csv
from collections import Counter
import json
import re
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.url_extractor import extract_urls_from_text


'''
1) TIME-BASED FEATURES
This feature category covers the time in which the phishing email was received. Phishing campaigns tend to be sent
to organisation email addresses in batches within a short
time frame [1], [16] making time-based features valuable
for identification. Features are taken from the DATE header
of the email. They are: date sent [38], time, day, month,
year, weekday, and a derived binary feature (work day / nonwork day). 
We added this feature since phishers might target
working days as it is likely victims read the message before it is deleted [38]

This function is able to extract all of the above features from the DATE header.
'''

def extract_time_features(date_str):
    try:
        dt = parsedate_to_datetime(date_str)
        data = {
            'day': int(dt.day),
            'month': int(dt.month),
            'year': int(dt.year),
            'weekday': dt.strftime("%A"),
            'workday': int(1 if dt.weekday() < 5 else 0)
        }
        if hasattr(dt, "date"):
            data["date_sent"] = dt.date().isoformat()

        if hasattr(dt, "time"):
            data["time_sent"] = dt.time().isoformat()
        # ensure integer values (no decimals)
        return data
    except:
        return {}

'''
This feature category is extracted from the email SUBJECT
header. It covers number of characters [38], number of white
spaces, and the vector of Term Frequency - Inverse Document
Frequency (TF-IDF) values of all words in the subject.

This function extracts all of the features above 
but collapses the TF-IDF vector into three summary statistics:
    - average idf of subject terms
    - highest idf among subject terms  
    - number of terms in the subject

'''

def extract_subject_features(subject, idf_dict):
    if not isinstance(subject, str):
        subject = ""

    num_chars = len(subject)
    num_whitespace = sum(c.isspace() for c in subject)

    dict = {
        "subject_length": num_chars,
        "subject_whitespace_count": num_whitespace,
    }

    dict.update(get_term_frequency_information(subject, idf_dict))

    return dict

def get_term_frequency_information(subject, idf_dict=None):
    """
    Returns a dictionary with:
      - average idf of subject terms
      - highest idf among subject terms
      - number of terms in the subject
    If idf_dict is not provided, uses term frequency (count) as a fallback.
    """
    if not isinstance(subject, str):
        subject = ""
    terms = subject.lower().split()
    n_terms = len(terms)
    if n_terms == 0:
        return {
            "subject_avg_idf": 0.0,
            "subject_max_idf": 0.0,
            "subject_n_terms": 0
        }
    # If no idf_dict provided, just use term frequency as fallback
    if idf_dict is None:
        idfs = [1.0 for _ in terms]
    else:
        idfs = [idf_dict.get(term, 0.0) for term in terms]
    avg_idf = sum(idfs) / n_terms
    max_idf = max(idfs)
    return {
        "subject_avg_idf": avg_idf,
        "subject_max_idf": max_idf,
        "subject_n_terms": n_terms
    }

def load_idf_dict(csv_path):
    """
    Load IDF values from a CSV file into a dictionary.
    Assumes columns: term,idf
    """
    idf_dict = {}
    with open(csv_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            term = row['term']
            try:
                idf = round(float(row['idf']), 3)
            except Exception:
                idf = 0.0
            idf_dict[term] = idf
    return idf_dict

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


def get_idf_path(csv_path):
    """
    Given a csv_path, returns the path for the corresponding subject IDF CSV.
    Example: '../../data/csv/TREC-07-only-phishing.csv' ->
             '../../data/csv/TREC-07-only-phishing_subject_idf.csv'
    """
    dir_name = os.path.dirname(csv_path)
    base, ext = os.path.splitext(os.path.basename(csv_path))
    return os.path.join(dir_name, f"{base}_subject_idf{ext}")

'''
This feature category was derived from the plain text part and
the HTML part of the email object. In order to check the
web technology used, we computed the types and numbers
of email elements, presence and number of images, presence
and number of URLs, and presence of HTML tags, scripts,
and CSS specifications [45], [46], [47]. We then removed all
HTML tags and other scripts as well as links to obtain the
pure body text. The text was converted into a bag of words.
We used Latent Semantic Analysis to extract the top ten terms
describing the emails content [38], [45]. We also computed
the number of lines, number of words, and average word
length [38]. While prior research focused on whether an email
contain a greeting line or not [47], from our observations
we found that several campaigns follow the same greeting
type. Therefore, we added a feature describing the greeting
type (style of greeting, such as hi, hello, and dear; checking
whether greeting is followed by recipient name, username or
email address).

This function extracts the body based features, however since the TREC data is much simpler
the amount of features has been reduced.
There are no HTML tags or attachments in the TREC data. 
It currently also does not do LSA on the body text.
Bag of word should probably be reduced to some summary statistics instead of full vector
or flattened to a single feature somehow, as otherwise the feature vector gets too impactful
and reduces clustering performance.

'''

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
        "greeting": greeting_features.get("greeting", ""),
        "bow": bow
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

def extract_greeting_features(body):
    """
    Simplified greeting extractor returning a single key 'greeting'.
    Examples: {'greeting': 'hi, name'}, {'greeting': 'hello, email'}, {'greeting': None}
    """
    if not isinstance(body, str) or not body.strip():
        return {"greeting": ""}

    text = body.lstrip()
    first_line = text.splitlines()[0].strip()

    greetings = [
        "good morning", "good afternoon", "good evening",
        "dearest", "greetings", "hello", "hi", "hey", "dear"
    ]

    greet_regex = r'^(?:' + '|'.join(re.escape(g) for g in greetings) + r')\b[,\s:!-]*([^\n,!?]*)'
    m = re.search(greet_regex, first_line, flags=re.I)
    if not m:
        return {"greeting": ""}

    greet_found = re.match(r'^(?:' + '|'.join(re.escape(g) for g in greetings) + r')\b', first_line, flags=re.I)
    greeting_token = greet_found.group(0).strip().lower() if greet_found else ""

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

'''
This feature category concerns the email attachments.
We determined whether the email has an attachment, how
many attachments the email has [38], [46], and attachment
size and type [38]. This information indicates if the attacker
distributes the same files within a campaign.

For the TREC dataset this is irrelevant so not implemented

'''

def extract_attachment_features(attachments):
    return

'''
The origin feature category is mostly about the sender of
the email. This can be either the attacker themselves or the
compromised accounts. We extracted name and email address
from both the FROM header and the RECEIVED header.
We also checked whether the email from the RECEIVED
header matches the one in the FROM header in order to
detect spoofed FROM addresses. This information can indicate the impersonated identity and details on the origin of
phishing campaign. We extracted the sender IP [38] and relevant domain information such as the domain from both
headers [38], domain registrar, domain registration date and
the registrar location [38]. This provides information about
the attacker origin and whether they used a public service or
compromised accounts to send the email.

This function extracts only the sender name and email from the FROM header.
The sender IP, domain information etc. is unavailable in the TREC dataset.
'''

def extract_origin_based_features(sender):
    """
    Extract sender name and email from two possible formats:
    1. email@domain.com -> name is 'email'
    2. Name <email@domain.com> -> name is 'Name'
    Removes surrounding quotes from names (e.g., "name lastname" -> name lastname)
    """
    if not isinstance(sender, str) or not sender.strip():
        return {"sender_name": "", "sender_email": ""}
    
    sender = sender.strip()
    
    # Format 2: Name <email@domain.com>
    angle_match = re.search(r'^(.+?)\s*<(.+?)>', sender)
    if angle_match:
        name = angle_match.group(1).strip()
        email = angle_match.group(2).strip()
        # Remove surrounding quotes from name
        name = name.strip('"')
        return {"sender_name": name if name else "", "sender_email": email}
    
    # Format 1: email@domain.com
    email_match = re.match(r'^([\w\.-]+)@[\w\.-]+\.\w+$', sender)
    if email_match:
        name = email_match.group(1)
        return {"sender_name": name, "sender_email": sender}
    
    # Fallback: couldn't parse
    return {"sender_name": "", "sender_email": ""}

'''
Recipient features concern the target users which only
includes recipient names and recipient counts. Other information that has been shown to be effective at 
identifying the target characteristics [38] was excluded as most of the 
information we have about recipients was redacted for anonymity reasons

This function extracts only the recipient name and email from the RECEIVER header.
No idea what recipient counts means.
'''

def extract_recipient_based_features(recipient):
    """
    Extract recipient name and email from two possible formats:
    1. email@domain.com -> name is 'email'
    2. Name <email@domain.com> -> name is 'Name'
    """
    if not isinstance(recipient, str) or not recipient.strip():
        return {"recipient_name": "", "recipient_email": ""}
    
    recipient = recipient.strip()
    
    # Format 2: Name <email@domain.com>
    angle_match = re.search(r'^(.+?)\s*<(.+?)>', recipient)
    if angle_match:
        name = angle_match.group(1).strip()
        email = angle_match.group(2).strip()
        return {"recipient_name": name if name else "", "recipient_email": email}
    
    # Format 1: email@domain.com
    email_match = re.match(r'^([\w\.-]+)@[\w\.-]+\.\w+$', recipient)
    if email_match:
        name = email_match.group(1)
        return {"recipient_name": name, "recipient_email": recipient}
    
    # Fallback: couldn't parse
    return {"recipient_name": "", "recipient_email": ""}

'''
URL-based features are one of the most important features in
phishing detection [48], [49], [50]. In this work we excluded
any feature that requires visiting the link, because it takes
a long time, and for older emails, the links probably were
taken down or changed. Features in the URL category include
the domain names, hostnames, domain categories, location
of domain registrar, subdomain count, and hyphen count.
We also computed binary features that reflect whether at least
one URL in the email has an Extended Validation Certificate (EV) that validates 
the owner of the domain, an extra http
and Top-Level Domain (TLD), a web-host domain, a @
symbol, non-ASCII characters, whether it has typos comparing to top 10,000 popular domains, 
whether it is similar to
top targeted domains on PhishTank and whether one of the
subdomains contains a popular domain on PhishTank [51].
For emails with several URLs, we counted the number of
URLs with an IP address, the number of different domains,
number of short URLs, and number of blacklisted links.
In the case of hyperlinks, we checked whether the visual link
presented in the email directed to the same URL [46], [47] and
checked whether there was a link under a text such as click
here. For the domain information, we collect the registration
dates of the oldest and the most recent domains, the minimum
PageRank and popularity, and the maximum PageRank and
Popularity for the list of URLs.

This function currently extracts only a subset of the above URL features.
Included are:
This function currently extracts the following URL features:
- URL counts: total URLs, unique domains, URLs with IP addresses, short URLs
- Binary indicators: @symbol presence, non-ASCII characters, extra http/https
- Aggregate statistics: average subdomain count, average hyphen count per URL
'''

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
    #print(features_list[0])
    return filtered_features

def get_test_set(misp_path):
    """
    Extract features: origin, receiver, url, and subject (without term frequency).
    Excludes subject term frequencies and only includes sender_name from origin.
    """
    features_list = extract_features(misp_path, ["subject", "origin", "receiver", "urls"])
    
    filtered_features = []
    for feat in features_list:
        filtered_feat = {k: v for k, v in feat.items()
                         if k not in ["subject_term_frequency", "bow",]}
        filtered_features.append(filtered_feat)
    return filtered_features

'''
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
                feat.update(extract_subject_features(row.get("subject"), load_idf_dict(get_idf_path(csv_path))))
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

def extract_features(misp_path, features):
    """
    Extract baseline features for specified types from MISP JSON format.
    
    Args:
        misp_path: Path to the MISP JSON file
        features: List of feature types to extract (e.g., ["time", "subject", "body", "origin", "receiver", "urls"])
    
    Returns:
        List of feature dictionaries, one per event
    """
    with open(misp_path, 'r', encoding='utf-8') as f:
        misp_data = json.load(f)
    events = []
    # Handle different MISP JSON structures
    if isinstance(misp_data, list):
        # Direct list of events
        events = misp_data
    elif isinstance(misp_data, dict):
        # Nested structure with 'response' key
        events = misp_data.get('response', {}).get('Event', [])
        if not isinstance(events, list):
            events = [events]  # Handle single event case
    else:
        events = []
    

    features_list = []
    
    print("extracting features from MISP events...")

    for event_idx, event in enumerate(events):
        feat = {'email_index': event_idx}

        # Parse MISP attributes into email fields
        email_fields = parse_misp_event_attributes(event.get('Event', {}))
        
        # Extract features for each requested type
        for feature_type in features:
            if feature_type == "time":
                feat.update(extract_time_features(email_fields.get("date")))
            elif feature_type == "subject":
                # Get IDF path relative to MISP file
                idf_path = get_idf_path_for_misp(misp_path)
                idf_dict = load_idf_dict(idf_path) if os.path.exists(idf_path) else None
                feat.update(extract_subject_features(email_fields.get("subject"), idf_dict))
            elif feature_type == "body":
                feat.update(extract_body_based_features(email_fields.get("body")))
            elif feature_type == "attachments":
                # Not implemented yet
                continue
            elif feature_type == "origin":
                feat.update(extract_origin_based_features(email_fields.get("sender")))
            elif feature_type == "receiver":
                feat.update(extract_recipient_based_features(email_fields.get("receiver")))
            elif feature_type == "urls":
                body = email_fields.get("body", "") if email_fields.get("body", "") is not None else ""
                extracted_urls = extract_urls_from_text(body) if body else []
                feat.update(extract_url_based_features(extracted_urls))
        
        features_list.append(feat)

    print("returning extracted features...")

    return features_list

def parse_misp_event_attributes(event):
    """
    Parse MISP event attributes into normalized email field dictionary.
    
    Returns:
        Dict with keys: subject, body, sender, receiver, date, etc.
    """
    email_fields = {
        'subject': '',
        'body': '',
        'sender': '',
        'receiver': '',
        'date': ''
    }
    attributes = event.get('Attribute', [])
    for attr in attributes:
        attr_type = attr.get('type', '')
        attr_value = attr.get('value', '')
        
        # Map MISP attribute types to email fields
        if attr_type == 'email-subject':
            email_fields['subject'] = attr_value
        elif attr_type == 'email-body':
            email_fields['body'] = attr_value
        elif attr_type == 'email-src':
            email_fields['sender'] = attr_value
        elif attr_type == 'email-dst':
            email_fields['receiver'] = attr_value
        elif attr_type == 'email-date':
            email_fields['date'] = attr_value
    
    return email_fields

def get_idf_path_for_misp(misp_path):
    """
    Given a MISP JSON path, returns the path for the corresponding subject IDF CSV.
    Example: '../../data/misp/TREC-07-misp.json' ->
             '../../data/csv/TREC-07-only-phishing_subject_idf.csv'
    """
    # Convert MISP path to corresponding CSV IDF path
    dir_name = os.path.dirname(misp_path)
    base_name = os.path.splitext(os.path.basename(misp_path))[0]
    
    # Map from MISP naming to CSV naming (adjust as needed)
    if base_name.endswith('-misp'):
        csv_base = base_name.replace('-misp', '-only-phishing')
    else:
        csv_base = base_name + '-only-phishing'
    
    # Assume CSV files are in ../csv/ relative to MISP files
    csv_dir = os.path.join(os.path.dirname(dir_name), 'csv')
    #return os.path.join(csv_dir, f"{csv_base}_subject_idf.csv")
    return "../../data/csv/TREC-07-only-phishing_subject_idf.csv"
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
            "email_index": idx,  # Add email index to features
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
    misp_path = "../../data/misp/TREC-07-misp.json"
    
    # Extract FS features
    fs_features = get_test_set(misp_path)
    
    # Save to JSON file
    input_dir = os.path.dirname(misp_path)
    input_base = os.path.splitext(os.path.basename(misp_path))[0]
    output_path = f"../../data/featuresets/{input_base}-FSTest.json"
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(fs_features, f, indent=2, ensure_ascii=False)
    
    print(f"Saved FS7 features to: {output_path}")
    print(f"Total emails processed: {len(fs_features)}")
    if fs_features:
        print(f"Sample feature keys: {list(fs_features[0].keys())}")