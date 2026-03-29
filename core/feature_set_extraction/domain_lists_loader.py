import csv
import os
from functools import lru_cache
from urllib.parse import urlparse

import tldextract


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_CACHE_DIR = os.path.join(_THIS_DIR, "caches")


def _clean_token(value):
    token = str(value or "").strip().lower()
    if not token or token.startswith("#"):
        return ""
    return token


def load_domain_set_from_txt(file_name):
    """Load a domain set from a TXT file in the local caches folder."""
    path = os.path.join(_CACHE_DIR, file_name)
    if not os.path.exists(path):
        return set()

    values = set()
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            token = _clean_token(raw)
            if token:
                values.add(token)
    return values


def _extract_registrable_domain(candidate):
    value = _clean_token(candidate)
    if not value:
        return ""

    parsed = urlparse(value)
    host = (parsed.hostname or "").lower() if parsed.scheme else value
    if not host:
        return ""

    ext = tldextract.extract(host)
    if not ext.domain or not ext.suffix:
        return ""
    return f"{ext.domain}.{ext.suffix}"


def load_phishtank_domains_from_urls(file_name):
    path = os.path.join(_CACHE_DIR, file_name)
    if not os.path.exists(path):
        return set()

    domains = set()
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            domain = _extract_registrable_domain(raw)
            if domain:
                domains.add(domain)
    return domains


@lru_cache(maxsize=1)
def load_url_intelligence_sets():
    os.makedirs(_CACHE_DIR, exist_ok=True)
    return {
        "popular_domains": load_domain_set_from_txt("popular_domains.txt"),
        "webhost_domains": load_domain_set_from_txt("web_hosting_domains.txt"),
        "phishing_target_domains": load_domain_set_from_txt("phishing_target_domains.txt"),
        #"blacklist": load_domain_set_from_txt("url_blacklist.txt"),
    }
