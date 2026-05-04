import json
import os
import time
from datetime import datetime
from typing import Dict, List, Optional

import requests
import tldextract

_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
RDAP_CACHE_DIR = os.path.normpath(os.path.join(_MODULE_DIR, "..", "feature_set_extraction", "caches"))
RDAP_CACHE_FILE = os.path.join(RDAP_CACHE_DIR, "rdap_cache.json")
RDAP_CACHE_LOCK_FILE = f"{RDAP_CACHE_FILE}.lock"
LEGACY_RDAP_CACHE_FILE = os.path.normpath(os.path.join(_MODULE_DIR, "..", "rdap_cache.json"))
RDAP_BOOTSTRAP_DNS_URL = "https://data.iana.org/rdap/dns.json"
RDAP_TIMEOUT_SECONDS = 12
RDAP_LOCK_TIMEOUT_SECONDS = 3
RDAP_LOCK_POLL_INTERVAL_SECONDS = 0.1
_RDAP_BOOTSTRAP_CACHE: Optional[Dict[str, List[str]]] = None
_REQUEST_HEADERS = {
    "User-Agent": "GNN-Campaign-Detection/1.0 (+https://github.com)",
    "Accept": "application/rdap+json, application/json;q=0.9",
}

# -----------------------------
# Utility: Extract root domain
# -----------------------------
def extract_domain(hostname: str) -> str:
    if not hostname:
        return None
    ext = tldextract.extract(hostname)
    if ext.domain and ext.suffix:
        return f"{ext.domain}.{ext.suffix}"
    return None


def _extract_tld(domain: str) -> str:
    ext = tldextract.extract(domain or "")
    return (ext.suffix or "").lower()


def _load_rdap_bootstrap() -> Dict[str, List[str]]:
    global _RDAP_BOOTSTRAP_CACHE
    if _RDAP_BOOTSTRAP_CACHE is not None:
        return _RDAP_BOOTSTRAP_CACHE

    mapping: Dict[str, List[str]] = {}
    try:
        response = requests.get(RDAP_BOOTSTRAP_DNS_URL, timeout=RDAP_TIMEOUT_SECONDS, headers=_REQUEST_HEADERS)
        response.raise_for_status()
        payload = response.json() if response.content else {}
        services = payload.get("services", []) if isinstance(payload, dict) else []
        for row in services:
            if not isinstance(row, list) or len(row) != 2:
                continue
            tlds_raw, urls_raw = row
            if not isinstance(tlds_raw, list) or not isinstance(urls_raw, list):
                continue
            urls = [str(u).strip().rstrip("/") for u in urls_raw if str(u).strip()]
            if not urls:
                continue
            for tld in tlds_raw:
                key = str(tld).strip().lower()
                if key:
                    mapping[key] = urls
    except Exception:
        # Network/bootstrap failures should not crash lookup flow.
        mapping = {}

    _RDAP_BOOTSTRAP_CACHE = mapping
    return mapping


def _candidate_rdap_urls_for_domain(domain: str) -> List[str]:
    urls: List[str] = []

    tld = _extract_tld(domain)
    bootstrap = _load_rdap_bootstrap()
    if tld and tld in bootstrap:
        for base in bootstrap[tld]:
            urls.append(f"{base}/domain/{domain}")

    # Keep rdap.org as a fallback only.
    urls.append(f"https://rdap.org/domain/{domain}")

    # Deduplicate while preserving order.
    seen = set()
    ordered: List[str] = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            ordered.append(u)
    return ordered


def _extract_registrar_fields(data: dict) -> Dict[str, Optional[str]]:
    registrar = None
    creation_date = None
    registrar_location = None

    entities = data.get("entities", []) if isinstance(data, dict) else []
    for ent in entities:
        if not isinstance(ent, dict):
            continue
        roles = ent.get("roles", [])
        if not isinstance(roles, list) or "registrar" not in roles:
            continue
        vcard = ent.get("vcardArray", [])
        if len(vcard) > 1 and isinstance(vcard[1], list):
            for item in vcard[1]:
                if not isinstance(item, list) or len(item) < 4:
                    continue
                if item[0] == "fn":
                    registrar = item[3]
                if item[0] == "adr" and isinstance(item[3], list):
                    registrar_location = " ".join(str(x) for x in item[3] if str(x).strip())

    for event in data.get("events", []) if isinstance(data, dict) else []:
        if isinstance(event, dict) and event.get("eventAction") == "registration":
            creation_date = event.get("eventDate")

    return {
        "registrar": registrar,
        "registration_date": creation_date,
        "registrar_location": registrar_location,
    }


# -----------------------------
# Step 1: Extract domains
# -----------------------------
def extract_domains_from_received(value_list):
    domains = set()

    for entry in value_list:
        for key in ["helo_host", "by_host"]:
            host = entry.get(key)
            domain = extract_domain(host)
            if domain:
                domains.add(domain)

    return list(domains)


# -----------------------------
# RDAP Fetch
# -----------------------------
def fetch_rdap(domain):
    last_error = None
    for url in _candidate_rdap_urls_for_domain(domain):
        try:
            response = requests.get(url, timeout=RDAP_TIMEOUT_SECONDS, headers=_REQUEST_HEADERS)
            response.raise_for_status()
            data = response.json() if response.content else {}
            fields = _extract_registrar_fields(data if isinstance(data, dict) else {})
            return {
                "domain": domain,
                "registrar": fields.get("registrar"),
                "registration_date": fields.get("registration_date"),
                "registrar_location": fields.get("registrar_location"),
                "fetched_at": datetime.utcnow().isoformat(),
                "rdap_url": url,
            }
        except Exception as e:
            last_error = f"{type(e).__name__}: {e}"

    return {
        "domain": domain,
        "error": last_error or "RDAP lookup failed",
        "fetched_at": datetime.utcnow().isoformat(),
    }


# -----------------------------
# Cache handling
# -----------------------------
def _acquire_cache_lock(timeout_seconds: int = RDAP_LOCK_TIMEOUT_SECONDS):
    os.makedirs(RDAP_CACHE_DIR, exist_ok=True)
    deadline = time.time() + timeout_seconds

    while True:
        try:
            # O_EXCL makes lock creation atomic across processes.
            fd = os.open(RDAP_CACHE_LOCK_FILE, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            try:
                os.write(fd, str(os.getpid()).encode("ascii", "ignore"))
            finally:
                os.close(fd)
            return
        except FileExistsError:
            if time.time() >= deadline:
                raise TimeoutError(
                    f"Timed out acquiring RDAP cache lock after {timeout_seconds} seconds"
                )
            time.sleep(RDAP_LOCK_POLL_INTERVAL_SECONDS)


def _release_cache_lock():
    try:
        os.remove(RDAP_CACHE_LOCK_FILE)
    except FileNotFoundError:
        pass


def _load_cache_unlocked():
    if os.path.exists(RDAP_CACHE_FILE):
        with open(RDAP_CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)

    # Backward-compatible one-time fallback from legacy cache location.
    if os.path.exists(LEGACY_RDAP_CACHE_FILE):
        with open(LEGACY_RDAP_CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)

    return {}


def load_cache():
    _acquire_cache_lock()
    try:
        return _load_cache_unlocked()
    finally:
        _release_cache_lock()


def _save_cache_unlocked(cache):
    os.makedirs(RDAP_CACHE_DIR, exist_ok=True)
    tmp_path = f"{RDAP_CACHE_FILE}.tmp.{os.getpid()}"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    # Atomic replace prevents partially written cache files.
    os.replace(tmp_path, RDAP_CACHE_FILE)


def save_cache(cache):
    _acquire_cache_lock()
    try:
        _save_cache_unlocked(cache)
    finally:
        _release_cache_lock()


def _is_retryable_cached_error(entry):
    if not isinstance(entry, dict):
        return False

    error_text = str(entry.get("error", "")).lower()
    if not error_text:
        return False

    # Retry cached rate-limit errors so they are not treated as permanently done.
    return "429" in error_text or "too many requests" in error_text


# -----------------------------
# Ensure cache exists / populate
# -----------------------------
def ensure_rdap_cache(domains):
    _acquire_cache_lock()
    try:
        cache = _load_cache_unlocked()
        updated = False

        for domain in domains:
            cached_entry = cache.get(domain)
            should_fetch = domain not in cache or _is_retryable_cached_error(cached_entry)
            if should_fetch:
                print(f"Fetching RDAP for {domain}...")
                cache[domain] = fetch_rdap(domain)
                updated = True

        if updated:
            _save_cache_unlocked(cache)

        return cache
    finally:
        _release_cache_lock()


# -----------------------------
# Main function (your use case)
# -----------------------------
def process_received_headers(value_list):
    domains = extract_domains_from_received(value_list)
    cache = ensure_rdap_cache(domains)

    results = []
    for domain in domains:
        results.append(cache.get(domain, {"domain": domain}))

    return results


# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    sample_input = [
        {
            "origin_ip": "203.0.113.8",
            "helo_host": "mail.mailservice.net",
            "by_host": "mx.mailservice.net",
            "timestamp": "Mon, 16 Mar 2026 10:27:18 GMT"
        },
        {
            "origin_ip": "203.0.113.149",
            "helo_host": "mx.mailservice.net",
            "by_host": "mailbox.outlook.com",
            "timestamp": "Mon, 16 Mar 2026 10:27:18 GMT"
        }
    ]

    output = process_received_headers(sample_input)
    print(json.dumps(output, indent=2))
