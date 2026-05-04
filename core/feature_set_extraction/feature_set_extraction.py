from email.utils import parsedate_to_datetime
from datetime import datetime, timezone
import pandas as pd
import os
from collections import Counter
import json
import re
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from feature_set_extraction.tfidf_utils import build_vectorizer, precompute_subject_idf
from concurrent.futures import ProcessPoolExecutor, as_completed
from feature_set_extraction.lsa import get_lsa_features
from preprocessing.utils.url_extractor import extract_urls_from_text
from preprocessing.RDAP_processor import load_cache as load_rdap_cache
from preprocessing.RDAP_processor import ensure_rdap_cache as ensure_rdap_cache
from preprocessing.utils.defang import sanitize_for_json
from feature_set_extraction.url_extraction_utils import extract_url_features as extract_url_features_utils
from feature_set_extraction.domain_lists_loader import load_url_intelligence_sets
from feature_set_extraction.fsols_extractor import extract_fsols_features
from graph.common import parse_misp_events


LSA_TOPIC_KEYS = [f"lsa_topic_{index}" for index in range(10)]

FS1_FEATURE_TYPES = ["time", "subject", "body", "origin", "receiver", "urls", "attachments"]

FS2_FEATURE_TYPES = ["time", "subject", "body", "urls", "origin", "attachments"]
FS2_OMIT_KEYS = frozenset(["sender_email"])

FS3_FEATURE_TYPES = ["body", "urls", "origin"]
FS3_OMIT_KEYS = frozenset([
    "body_word_count",
    "num_lines",
    "avg_word_length",
    "greeting",
    "body",
    "bow",
    "has_html",
    "num_html_tags",
    "num_images",
    "num_urls_in_body",
    "has_script",
    "has_css",
    "num_css_rules",
    "has_forms",
    "image_text_ratio",
    *LSA_TOPIC_KEYS,
])

FS4_FEATURE_TYPES = ["subject", "body"]
FS4_OMIT_KEYS = frozenset([
    "num_urls_in_body",
    "has_urls_in_body",
    "body_word_count",
    "num_lines",
    "avg_word_length",
    "greeting",
    "subject_length",
    "subject_whitespace_count",
    "subject_avg_idf",
    "subject_max_idf",
    "subject_n_terms",
])

FS5_FEATURE_TYPES = ["subject", "body", "receiver", "origin", "urls", "attachments"]
FS5_OMIT_KEYS = frozenset([
    "subject_length",
    "subject_whitespace_count",
    "subject_avg_idf",
    "subject_max_idf",
    "subject_n_terms",
    "num_urls_in_body",
    "has_urls_in_body",
    "body_word_count",
    "num_lines",
    "avg_word_length",
    "greeting",
    "recipient_email",
    "domain_categories",
    "registrar_locations",
    "subdomain_counts",
    "hyphen_counts",
    "any_ev_cert",
    "any_has_extra_http",
    "any_multi_part_tld",
    "any_www_host",
    "any_has_at_symbol",
    "any_has_non_ascii",
    "any_typo_popular_domains",
    "any_similar_phish_targets",
    "any_popular_domain_in_subdomain",
    "num_ip_urls",
    "num_distinct_domains",
    "num_short_urls",
    "num_blacklisted",
    "has_attachments",
    "num_attachments",
    "attachment_sizes_bytes",
])

FS6_FEATURE_TYPES = ["subject", "time", "body", "origin", "urls", "attachments"]
FS6_OMIT_KEYS = frozenset([
    "subject_term_frequency",
    "bow",
    "sender_email",
    "greeting",
    "body",
    "subject",
    *LSA_TOPIC_KEYS,
    "has_attachments",
])

FS7_FEATURE_TYPES = ["subject", "body", "origin", "urls"]
FS7_OMIT_KEYS = frozenset(["bow", "sender_email", "body"])

# FSOLS: Top OLS features extracted directly from event HTML/CSS/URL attrs.
FSOLS_FEATURE_TYPES = ["body", "urls"]
FSOLS_OMIT_KEYS = frozenset([
    # Text features (not included per OLS analysis)
    "body_word_count",
    "num_lines",
    "avg_word_length",
    "greeting",
    "body",
    "bow",
    # LSA topics (not in top 20)
    *LSA_TOPIC_KEYS,
    # HTML features not in top 20
    "has_html_tags",
    "has_images",
    "has_script",
    "has_css_specs",
    "image_text_ratio",
    "num_images",
    "num_urls_in_body",
    "has_urls_in_body",
    # CSS features not in top 20
    "css_primary_color",
    # Other URL/domain features not in top 20
    "num_distinct_domains",
    "num_blacklisted",
    "num_ip_urls",
    "num_short_urls",
    "any_ev_cert",
    "any_has_extra_http",
    "any_multi_part_tld",
    "any_www_host",
    "any_has_at_symbol",
    "any_has_non_ascii",
    "any_typo_popular_domains",
    "any_similar_phish_targets",
    "any_popular_domain_in_subdomain",
    "domain_categories",
    "registrar_locations",
    "subdomain_counts",
    "hyphen_counts",
    "domains_failed",
])

TEST_SET_FEATURE_TYPES = ["subject", "origin", "receiver", "urls"]
TEST_SET_OMIT_KEYS = frozenset(["subject_term_frequency", "bow"])


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
        date_str = datetime.fromtimestamp(float(date_str), tz=timezone.utc).strftime("%a, %d %b %Y %H:%M:%S %z")
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
        return data
    except:
        return {}

'''
This feature category is extracted

from the email SUBJECT header. It covers number of char-
acters [39], number of white spaces, and the vector of Term
Frequency - Inverse Document Frequency (TF-IDF) values
of all words in the subject.

'''


def extract_subject_features(subject, idf_dict):
    if not isinstance(subject, str):
        subject = ""

    num_chars = len(subject)
    num_whitespace = sum(c.isspace() for c in subject)

    # Match sklearn's default tokenization behavior used during IDF precompute
    # so punctuation-suffixed tokens (e.g., "pills?") become "pills".
    terms = re.findall(r"(?u)\b\w\w+\b", subject.lower())
    n_terms = len(terms)
    term_counts = Counter(terms)
    tfidf_vector = {}

    for term, count in term_counts.items():
        tf = count / n_terms if n_terms > 0 else 0.0
        idf = idf_dict.get(term, 0.0) if isinstance(idf_dict, dict) else 1.0
        tfidf_vector[term] = round(float(tf * idf), 6)

    return {
        "subject_length": num_chars,
        "subject_whitespace_count": num_whitespace,
        "subject": subject,
        "subject_term_frequency": tfidf_vector,
    }

def load_idf_dict(idf_path):
    with open(idf_path, 'r', encoding='utf-8') as f:
        raw = json.load(f)

    if not isinstance(raw, dict):
        return {}

    idf_dict = {}
    for term, value in raw.items():
        try:
            idf_dict[str(term)] = float(value)
        except Exception:
            idf_dict[str(term)] = 0.0
    return idf_dict
 
def get_idf(subjects, output_path):
    vectorizer = build_vectorizer(subjects)
    terms = vectorizer.get_feature_names_out()
    idfs = vectorizer.idf_
    idf_dict = {str(term): float(idf) for term, idf in zip(terms, idfs)}

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(idf_dict, f, indent=2, ensure_ascii=False)

    print(f"Saved IDF values to: {output_path}")
    return vectorizer


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

'''

def extract_body_markup_presence_features(body, html=None, css=None):
    if not isinstance(body, str):
        body = ""

    tag_counts = {}
    tree_stats = {}
    style_features = {}
    if isinstance(html, dict):
        if isinstance(html.get("tag_counts"), dict):
            tag_counts = html.get("tag_counts")
        if isinstance(html.get("tree_stats"), dict):
            tree_stats = html.get("tree_stats")
    if isinstance(css, dict) and isinstance(css.get("style_features"), dict):
        style_features = css.get("style_features")

    has_structured_markup = bool(tag_counts or tree_stats or style_features)

    if has_structured_markup:
        html_tag_total = 0
        for v in tag_counts.values():
            try:
                iv = int(v)
            except Exception:
                iv = 0
            if iv > 0:
                html_tag_total += iv

        num_images = 0
        try:
            num_images = int(tag_counts.get("img", 0))
        except Exception:
            num_images = 0
        if num_images <= 0:
            try:
                num_images = int(tree_stats.get("images", 0))
            except Exception:
                num_images = 0

        script_count = 0
        try:
            script_count = int(tag_counts.get("script", 0))
        except Exception:
            script_count = 0
        if script_count <= 0:
            try:
                script_count = int(tree_stats.get("external_scripts", 0))
            except Exception:
                script_count = 0

        style_tag_count = 0
        try:
            style_tag_count = int(tag_counts.get("style", 0))
        except Exception:
            style_tag_count = 0

        has_html_tags = int(html_tag_total > 0)
        has_script = int(script_count > 0)
        has_css_specs = int(style_tag_count > 0 or any(bool(v) for v in style_features.values()))

    else:
        has_html_tags = int(bool(re.search(r"<[^>]+>", body)))
        num_images = 0
        has_script = 0
        has_css_specs = 0

        if has_html_tags:
            soup = BeautifulSoup(body, "html.parser")
            num_images = len(soup.find_all("img"))
            has_script = int(bool(soup.find_all("script")))
            has_css_specs = int(bool(soup.find_all("style") or soup.find_all(style=True)))

    return {
        "has_html_tags": has_html_tags,
        "has_images": int(num_images > 0),
        "num_images": num_images,
        "has_script": has_script,
        "has_css_specs": has_css_specs,
    }


def extract_body_based_features(body, html=None, css=None):
    if not isinstance(body, str):
        body = ""
    extracted_urls = extract_urls_from_text(body) if body else []
    
    num_lines = len(body.splitlines())
    
    words = re.findall(r"\w+", body.lower())
    num_words = len(words)
    
    avg_word_length = round(sum(len(word) for word in words) / num_words,1) if num_words > 0 else 0

    greeting_features = extract_greeting_features(body)
    bow = compute_body_bow(body)

    return {
        "num_urls_in_body": len(extracted_urls),
        "has_urls_in_body": 1 if len(extracted_urls) > 0 else 0,
        "body_word_count": num_words,
        "num_lines": num_lines,
        "avg_word_length": avg_word_length,
        "greeting": greeting_features.get("greeting", ""),
        "body": body,
        "bow": bow,
        **extract_body_markup_presence_features(body, html=html, css=css),
    }

def compute_body_bow(body):
    if not isinstance(body, str):
        body = ""
    
    words = re.findall(r"\w+", body.lower())
    word_freq = dict(Counter(words))
    
    return word_freq


def extract_greeting_features(body):
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

    email_re = re.compile(r'[\w\.-]+@[\w\.-]+\.\w+')
    if email_re.search(follow_text):
        return {"greeting": f"{greeting_token}, email"}

    tokens = re.findall(r"[^\s,;:()<>\"']+", follow_text)
    first_tok = tokens[0] if tokens else ""

    if re.match(r'^[A-Z][a-z\'-]+$', first_tok):
        return {"greeting": f"{greeting_token}, name"}

    if re.match(r'^[\w\.-]{2,}$', first_tok):
        return {"greeting": f"{greeting_token}, username"}

    return {"greeting": f"{greeting_token}, other"}

def extract_body_features(body: str) -> dict:
    """
    Extract structural features from an email body.

    Returns a dictionary with:
        has_html
        num_html_tags
        num_images
        num_urls_in_body
        has_script
        has_css
        num_css_rules
        has_forms
        image_text_ratio
    """

    # -------- Detect HTML --------
    has_html = int(bool(re.search(r"<[^>]+>", body)))

    if has_html:
        soup = BeautifulSoup(body, "html.parser")

        # -------- HTML tags --------
        all_tags = soup.find_all(True)
        num_html_tags = len(all_tags)

        # -------- Images --------
        images = soup.find_all("img")
        num_images = len(images)

        # -------- URLs --------
        urls = set()

        # links
        for tag in soup.find_all(href=True):
            urls.add(tag["href"])

        # media / src attributes
        for tag in soup.find_all(src=True):
            urls.add(tag["src"])

        # also catch raw URLs in text
        raw_urls = re.findall(r'https?://\S+', body)
        urls.update(raw_urls)

        num_urls_in_body = len(urls)

        # -------- Scripts --------
        has_script = int(bool(soup.find_all("script")))

        # -------- CSS --------
        style_tags = soup.find_all("style")
        inline_styles = soup.find_all(style=True)
        has_css = int(bool(style_tags or inline_styles))

        # Count CSS rules (simple heuristic: count "{")
        num_css_rules = 0
        for style in style_tags:
            if style.string:
                num_css_rules += style.string.count("{")

        # -------- Forms --------
        has_forms = int(bool(soup.find_all("form")))

        # -------- Text content --------
        visible_text = soup.get_text(separator=" ", strip=True)

    else:
        # Plain text handling
        num_html_tags = 0
        num_images = 0
        num_urls_in_body = len(re.findall(r'https?://\S+', body))
        has_script = 0
        has_css = 0
        num_css_rules = 0
        has_forms = 0
        visible_text = body

    # -------- Image-text ratio --------
    # Defined as images per word (common simple heuristic)
    words = re.findall(r"\b\w+\b", visible_text)
    word_count = len(words)

    image_text_ratio = num_images / word_count if word_count > 0 else 0.0

    return {
        "has_html": has_html,
        "num_html_tags": num_html_tags,
        "num_images": num_images,
        "num_urls_in_body": num_urls_in_body,
        "has_script": has_script,
        "has_css": has_css,
        "num_css_rules": num_css_rules,
        "has_forms": has_forms,
        "image_text_ratio": round(image_text_ratio, 4),
    }


'''
This feature category concerns the email attachments.
We determined whether the email has an attachment, how
many attachments the email has [38], [46], and attachment
size and type [38]. This information indicates if the attacker
distributes the same files within a campaign.

'''

def extract_attachment_features(attachments, attachment_metadata=None):
    if isinstance(attachments, list):
        cleaned = [a for a in attachments if isinstance(a, str) and a]
    elif isinstance(attachments, str) and attachments:
        cleaned = [attachments]
    else:
        cleaned = []

    if isinstance(attachment_metadata, dict):
        metadata_items = [attachment_metadata]
    elif isinstance(attachment_metadata, list):
        metadata_items = [item for item in attachment_metadata if isinstance(item, dict)]
    else:
        metadata_items = []

    sizes = []
    content_types = []
    top_level_types = []
    for item in metadata_items:
        raw_size = item.get("size_bytes")
        try:
            size_bytes = int(raw_size)
            if size_bytes >= 0:
                sizes.append(size_bytes)
        except Exception:
            pass

        raw_ct = item.get("content_type", "")
        content_type = str(raw_ct).strip().lower()
        if content_type:
            content_types.append(content_type)
            top_level_types.append(content_type.split("/", 1)[0])

    unique_content_types = sorted(set(content_types))
    unique_top_level_types = sorted(set(top_level_types))

    return {
        #"attachments": cleaned,
        "has_attachments": int(len(cleaned) > 0 or len(metadata_items) > 0),
        "num_attachments": int(max(len(cleaned), len(metadata_items))),
        "attachment_sizes_bytes": sizes,
        "attachment_types": unique_content_types,
        #"attachment_top_level_types": " ".join(unique_top_level_types),
    }

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

'''

def extract_origin_based_features(sender, auth_spf=None):
    def _parse_sender_entry(sender_entry):
        if not isinstance(sender_entry, str) or not sender_entry.strip():
            return "", ""

        sender_entry = sender_entry.strip()

        angle_match = re.search(r'^(.+?)\s*<(.+?)>', sender_entry)
        if angle_match:
            name = angle_match.group(1).strip().strip('"')
            email = angle_match.group(2).strip()
            return (name if name else ""), email

        email_match = re.match(r'^([\w\.-]+)@[\w\.-]+\.\w+$', sender_entry)
        if email_match:
            return email_match.group(1), sender_entry

        return "", ""

    if isinstance(sender, list):
        sender_entries = sender
    elif isinstance(sender, str):
        sender_entries = [sender]
    else:
        sender_entries = []

    sender_names = []
    sender_emails = []
    for sender_entry in sender_entries:
        name, email = _parse_sender_entry(sender_entry)
        if email:
            sender_names.append(name)
            sender_emails.append(email)

    spf_pass = isinstance(auth_spf, str) and auth_spf.strip().lower() == "pass"

    # RDAP lookup on the domain portion of each sender email
    sender_domains = []
    for email in sender_emails:
        at_idx = email.rfind("@")
        if at_idx != -1:
            sender_domains.append(email[at_idx + 1:].lower())

    domain_registrars = []
    domain_registration_dates = []
    registrar_locations = []

    if sender_domains:
        try:
            cache = load_rdap_cache()
            for domain in sender_domains:
                item = cache.get(domain, {})
                domain_registrars.append(item.get("registrar") or "")
                domain_registration_dates.append(item.get("registration_date") or "")
                registrar_locations.append(item.get("registrar_location") or "")
        except Exception:
            domain_registrars = [""] * len(sender_domains)
            domain_registration_dates = [""] * len(sender_domains)
            registrar_locations = [""] * len(sender_domains)

    return {
        "sender_name": sender_names,
        "sender_email": sender_emails,
        "spf_pass": spf_pass,
        "domain_registrars": domain_registrars,
        "domain_registration_dates": domain_registration_dates,
        "registrar_locations": registrar_locations,
    }

'''
Recipient features concern the target users which only
includes recipient names and recipient counts. Other information that has been shown to be effective at 
identifying the target characteristics [38] was excluded as most of the 
information we have about recipients was redacted for anonymity reasons
'''

def extract_recipient_based_features(recipient):
    def _parse_recipient_entry(recipient_entry):
        if not isinstance(recipient_entry, str) or not recipient_entry.strip():
            return "", ""

        recipient_entry = recipient_entry.strip()

        angle_match = re.search(r'^(.+?)\s*<(.+?)>', recipient_entry)
        if angle_match:
            name = angle_match.group(1).strip().strip('"')
            email = angle_match.group(2).strip()
            return (name if name else ""), email

        email_match = re.match(r'^([\w\.-]+)@[\w\.-]+\.\w+$', recipient_entry)
        if email_match:
            return email_match.group(1), recipient_entry

        return "", ""

    if isinstance(recipient, list):
        recipient_entries = recipient
    elif isinstance(recipient, str):
        recipient_entries = [recipient]
    else:
        recipient_entries = []

    recipient_names = []
    recipient_emails = []
    for recipient_entry in recipient_entries:
        name, email = _parse_recipient_entry(recipient_entry)
        if email:
            recipient_names.append(name)
            recipient_emails.append(email)

    return {"recipient_name": recipient_names, "recipient_email": recipient_emails, "recipient_count": len(recipient_emails)}

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

The top targets on phishtank is also used to compute num blacklist links.
PageRank and popularity are excluded. Suggested online tools appear to cost money and not entirely sure what these metrics mean anyway.

'''

def extract_url_based_features(urls):
    # Delegate to shared extractor and return its full feature dict
    try:
        url_intel_sets = load_url_intelligence_sets()
        return extract_url_features_utils(
        urls,
        popular_domains=url_intel_sets.get("popular_domains", set()),
        webhost_domains=url_intel_sets.get("webhost_domains", set()),
        phishing_target_domains=url_intel_sets.get("phishing_target_domains", set()),
        blacklist=url_intel_sets.get("blacklist", set()),
        domain_metadata=None,
        anchor_pairs=None
    )
    except Exception as e:
        return {
            "domains_failed": " ".join(urls) if urls else "",}


'''

Main extraction program

'''


def _load_raw_misp_events(misp_path):
    """Load raw MISP events in the same shape expected by graph parsing."""
    with open(misp_path, 'r', encoding='utf-8') as f:
        misp_data = json.load(f)

    if isinstance(misp_data, list):
        return misp_data

    if isinstance(misp_data, dict):
        events = misp_data.get('Events')
        if isinstance(events, list):
            return events

        events = misp_data.get('response', {}).get('Event', [])
        if isinstance(events, list):
            return events
        if isinstance(events, dict):
            return [events]

        if isinstance(misp_data.get('Event'), dict):
            return [misp_data]

    return []


def ensure_subject_idf(misp_path, events):
    """Ensure subject IDF JSON exists on disk for this MISP input."""
    idf_path = get_idf_path_for_misp(misp_path)
    if os.path.exists(idf_path):
        return idf_path

    try:
        precompute_subject_idf(misp_path, events)
        return idf_path
    except Exception:
        return idf_path


def get_subject_idf_for_misp(misp_path):
    """Load cached subject IDF dict for a MISP input."""
    idf_path = get_idf_path_for_misp(misp_path)
    if not os.path.exists(idf_path):
        return {}

    try:
        loaded = load_idf_dict(idf_path)
        return loaded if isinstance(loaded, dict) else {}
    except Exception:
        return {}


def ensure_lsa_features(events):
    """Return precomputed LSA feature dicts for all event bodies."""
    bodies = []
    for email_fields in events:
        body = email_fields.get("body", "") if isinstance(email_fields, dict) else ""
        bodies.append(body if isinstance(body, str) else "")

    try:
        return get_lsa_features(bodies)
    except Exception:
        return []


def get_helpers_output_dir():
    package_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(package_dir, 'output', 'helpers')


def get_lsa_path_for_misp(misp_path):
    base_name = os.path.splitext(os.path.basename(misp_path))[0]
    return os.path.join(get_helpers_output_dir(), f"{base_name}_lsa.json")


def ensure_lsa_features_for_misp(misp_path, events):
    """Ensure LSA features JSON exists on disk for this MISP input."""
    lsa_path = get_lsa_path_for_misp(misp_path)

    if os.path.exists(lsa_path):
        return lsa_path

    lsa_features = ensure_lsa_features(events)
    try:
        os.makedirs(os.path.dirname(lsa_path) or '.', exist_ok=True)
        with open(lsa_path, 'w', encoding='utf-8') as f:
            json.dump(sanitize_for_json(lsa_features), f, indent=2, ensure_ascii=False)
        print(f"Saved LSA features to: {lsa_path}")
    except Exception:
        pass

    return lsa_path


def get_lsa_features_for_misp(misp_path, events_count=None):
    """Load cached LSA feature dicts for a MISP input."""
    lsa_path = get_lsa_path_for_misp(misp_path)
    if not os.path.exists(lsa_path):
        return []

    try:
        with open(lsa_path, 'r', encoding='utf-8') as f:
            cached = json.load(f)
        if not isinstance(cached, list) or not all(isinstance(item, dict) for item in cached):
            return []
        if events_count is not None and len(cached) != events_count:
            return []
        return cached
    except Exception:
        return []


def extract_features(misp_path, features, events=None):
    if events is None:
        events = parse_misp_events(_load_raw_misp_events(misp_path))

    subject_idf_dict = get_subject_idf_for_misp(misp_path) if "subject" in features else {}
    lsa_features_list = get_lsa_features_for_misp(misp_path, events_count=len(events)) if "body" in features else []

    features_list = []

    for event_idx, email_fields in enumerate(events):
        ext = email_fields.get("external_id")
        ext_s = str(ext).strip() if ext is not None else ""
        if not ext_s:
            raise ValueError(
                f"Event at index {event_idx} has no external_id; required for feature rows"
            )
        feat = {"external_id": ext_s}

        for feature_type in features:
            if feature_type == "time":
                feat.update(extract_time_features(email_fields.get("date")))

            elif feature_type == "subject":
                feat.update(extract_subject_features(email_fields.get("subject"), subject_idf_dict))

            elif feature_type == "body":
                feat.update(
                    extract_body_based_features(
                        email_fields.get("body"),
                        email_fields.get("html", {}),
                        email_fields.get("css", {}),
                    )
                )
                if event_idx < len(lsa_features_list) and isinstance(lsa_features_list[event_idx], dict):
                    feat.update(lsa_features_list[event_idx])

            elif feature_type == "attachments":
                feat.update(
                    extract_attachment_features(
                        email_fields.get("attachments"),
                        email_fields.get("attachment_metadata", []),
                    )
                )

            elif feature_type == "origin":
                feat.update(
                    extract_origin_based_features(
                        email_fields.get("senders"),
                        email_fields.get("auth_spf", ""),
                    )
                )

            elif feature_type == "receiver":
                feat.update(extract_recipient_based_features(email_fields.get("receivers")))

            elif feature_type == "urls":
                explicit_urls = email_fields.get("urls", [])
                extracted_urls = []
                if isinstance(explicit_urls, list):
                    extracted_urls.extend([u for u in explicit_urls if isinstance(u, str) and u])
                elif isinstance(explicit_urls, str) and explicit_urls:
                    extracted_urls.append(explicit_urls)
                extracted_urls = list(dict.fromkeys(extracted_urls))
                feat.update(extract_url_based_features(extracted_urls))

        features_list.append(feat)
    return features_list

def omit_feature_keys(features_list, omitted_keys):
    return [
        {key: value for key, value in feat.items() if key not in omitted_keys}
        for feat in features_list
    ]

def parse_misp_event_attributes(event):
    """Backward-compatible shim that delegates to graph/common schema parser."""
    normalized = parse_misp_events([{"Event": event}] if isinstance(event, dict) else [])
    if not normalized:
        return {
            'subject': '',
            'body': '',
            'sender': '',
            'receiver': '',
            'date': '',
            'urls': []
        }

    parsed = normalized[0]
    senders = parsed.get('senders', [])
    receivers = parsed.get('receivers', [])
    return {
        'subject': parsed.get('subject', ''),
        'body': parsed.get('body', ''),
        'sender': senders[0] if senders else '',
        'receiver': receivers[0] if receivers else '',
        'date': parsed.get('date', ''),
        'urls': parsed.get('urls', []),
        'attachments': parsed.get('attachments', []),
        'attachment_metadata': parsed.get('attachment_metadata', []),
        'html': parsed.get('html', {}),
        'css': parsed.get('css', {}),
        'received_hops': parsed.get('received_hops', []),
        'return_path': parsed.get('return_path', {}),
        'auth_spf': parsed.get('auth_spf', ''),
        'auth_dkim': parsed.get('auth_dkim', ''),
        'auth_dmarc': parsed.get('auth_dmarc', ''),
        'cyrillic_domain': parsed.get('cyrillic_domain', ''),
        'contains_symbols': parsed.get('contains_symbols', ''),
        'body_has_tracking_url': parsed.get('body_has_tracking_url', ''),
        'body_has_tracking_image': parsed.get('body_has_tracking_image', ''),
        'body_has_tracking_pixel': parsed.get('body_has_tracking_pixel', ''),
        'body_has_unsubscribe_link': parsed.get('body_has_unsubscribe_link', ''),
        'domain_is_common_webprovided': parsed.get('domain_is_common_webprovided', ''),
    }


def get_idf_path_for_misp(misp_path):
    base_name = os.path.splitext(os.path.basename(misp_path))[0]
    return os.path.join(get_helpers_output_dir(), f"{base_name}_subject_idf.json")


def get_FS1(misp_path, events):
    return extract_features(misp_path, FS1_FEATURE_TYPES, events=events)


def get_FS2(misp_path, events):
    features_list = extract_features(misp_path, FS2_FEATURE_TYPES, events=events)
    return omit_feature_keys(features_list, FS2_OMIT_KEYS)


def get_FS3(misp_path, events):
    features_list = extract_features(misp_path, FS3_FEATURE_TYPES, events=events)
    return omit_feature_keys(features_list, FS3_OMIT_KEYS)


def get_FS4(misp_path, events):
    features_list = extract_features(misp_path, FS4_FEATURE_TYPES, events=events)
    return omit_feature_keys(features_list, FS4_OMIT_KEYS)


def get_FS5(misp_path, events):
    features_list = extract_features(misp_path, FS5_FEATURE_TYPES, events=events)
    return omit_feature_keys(features_list, FS5_OMIT_KEYS)


#Maybe should not include some of the body features, unsure based on description
def get_FS6(misp_path, events):
    features_list = extract_features(misp_path, FS6_FEATURE_TYPES, events=events)
    return omit_feature_keys(features_list, FS6_OMIT_KEYS)


def get_FS7(misp_path, events):
    features_list = extract_features(misp_path, FS7_FEATURE_TYPES, events=events)
    return omit_feature_keys(features_list, FS7_OMIT_KEYS)


def get_FSOLS(misp_path, events):
    """FSOLS uses direct event attributes for OLS-selected fields."""
    return extract_fsols_features(events, FSOLS_FEATURE_TYPES, FSOLS_OMIT_KEYS)


def get_test_set(misp_path, events):
    features_list = extract_features(misp_path, TEST_SET_FEATURE_TYPES, events=events)
    return omit_feature_keys(features_list, TEST_SET_OMIT_KEYS)


def _extract_and_save_featureset(args):
    fs_name, fs_function, misp_path, events, output_path = args

    try:
        fs_features = fs_function(misp_path, events)
        # ensure output directory exists (worker processes may run before directory created)
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(sanitize_for_json(fs_features), f, indent=2, ensure_ascii=False)

        sample_keys = list(fs_features[0].keys()) if fs_features else []
        return (fs_name, output_path, len(fs_features), sample_keys, True, None)

    except Exception as e:
        return (fs_name, output_path, 0, [], False, str(e))


def run_featureset_extraction(misp_path=None, parallel=True, max_workers=None):
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    if misp_path is None:
        misp_path = os.path.join(project_root, 'data', 'misp', 'TREC-07-misp.json')

    # Ensure expensive/cached artifacts are materialized before workers start.
    events_for_precompute = parse_misp_events(_load_raw_misp_events(misp_path))

    try:
        # Single upfront RDAP cache ensure from parsed events.
        # This attempts all domains from event URLs and received hop hosts before workers start.
        ensure_rdap_cache(events_for_precompute)
    except Exception:
        print("Warning: RDAP cache ensure failed")

    try:
        ensure_subject_idf(misp_path, events_for_precompute)
    except Exception:
        # If ensure fails, workers will compute missing artifacts on demand.
        pass

    try:
        ensure_lsa_features_for_misp(misp_path, events_for_precompute)
    except Exception:
        # If ensure fails, workers will compute missing artifacts on demand.
        pass

    fs_extractors = {
        'FS1': get_FS1,
        'FS2': get_FS2,
        'FS3': get_FS3,
        'FS4': get_FS4,
        'FS5': get_FS5,
        'FS6': get_FS6,
        'FS7': get_FS7,
        'FSOLS': get_FSOLS,
    }

    input_base = os.path.splitext(os.path.basename(misp_path))[0]

    extraction_args = []
    package_dir = os.path.dirname(os.path.abspath(__file__))
    for fs_name, fs_function in fs_extractors.items():
        output_path = os.path.join(package_dir, 'output', 'featuresets', f"{input_base}-{fs_name}.json")
        extraction_args.append((fs_name, fs_function, misp_path, events_for_precompute, output_path))

    if parallel:
        cpu_count = os.cpu_count() or 1
        if max_workers is None:
            effective_max_workers = min(2, cpu_count)
        else:
            try:
                requested_workers = int(max_workers)
            except Exception:
                requested_workers = 2
            effective_max_workers = max(1, min(requested_workers, cpu_count, 8))

        print(f"\n{'='*80}")
        print(f"Starting parallel feature extraction ({len(fs_extractors)} feature sets)...")
        print(f"Max workers: {effective_max_workers}")
        print(f"{'='*80}")

        results = []
        with ProcessPoolExecutor(max_workers=effective_max_workers) as executor:
            future_to_fs = {executor.submit(_extract_and_save_featureset, args): args[0] 
                           for args in extraction_args}

            for future in as_completed(future_to_fs):
                fs_name = future_to_fs[future]
                try:
                    result = future.result()
                    results.append(result)

                    if result[4]:
                        print(f"✔ {result[0]} completed ({result[2]} emails)")
                    else:
                        print(f"✖ {result[0]} failed: {result[5]}")

                except Exception as e:
                    print(f"✖ {fs_name} raised exception: {e}")
                    results.append((fs_name, "", 0, [], False, str(e)))

        print(f"\n{'='*80}")
        print("Parallel extraction complete!")
        print(f"{'='*80}")

        successful = [r for r in results if r[4]]
        failed = [r for r in results if not r[4]]

        if successful:
            print(f"\nSuccessful extractions ({len(successful)}):")
            for fs_name, output_path, num_emails, sample_keys, _, _ in successful:
                print(f"  {fs_name}: {output_path}")

        if failed:
            print(f"\nFailed extractions ({len(failed)}):")
            for fs_name, _, _, _, _, error_msg in failed:
                print(f"  {fs_name}: {error_msg}")

    else:
        print(f"\n{'='*80}")
        print(f"Starting sequential feature extraction ({len(fs_extractors)} feature sets)...")
        print(f"{'='*80}")

        for args in extraction_args:
            fs_name, fs_function, misp_path, events_for_precompute, output_path = args

            print(f"\n{'='*80}")
            print(f"Extracting {fs_name}...")
            print(f"{'='*80}")

            result = _extract_and_save_featureset(args)

            if result[4]:  # success
                print(f"Saved {result[0]} features to: {result[1]}")
                print(f"Total emails processed: {result[2]}")
                if result[3]:
                    print(f"Sample feature keys: {result[3]}")
            else:
                print(f"Error extracting {result[0]}: {result[5]}")

        print(f"\n{'='*80}")
        print("All feature sets extracted successfully!")
        print(f"{'='*80}")
