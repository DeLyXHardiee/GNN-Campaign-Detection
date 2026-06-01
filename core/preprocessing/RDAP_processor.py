import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from email.utils import parsedate_to_datetime
from urllib.parse import urlsplit
from typing import Dict, List, Optional

import requests
import tldextract

_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
RDAP_CACHE_DIR = os.path.normpath(os.path.join(_MODULE_DIR, "..", "feature_set_extraction", "caches"))
RDAP_CACHE_FILE = os.path.join(RDAP_CACHE_DIR, "rdap_cache.json")
RDAP_CACHE_LOCK_FILE = f"{RDAP_CACHE_FILE}.lock"
LEGACY_RDAP_CACHE_FILE = os.path.normpath(os.path.join(_MODULE_DIR, "..", "rdap_cache.json"))
RDAP_BOOTSTRAP_DNS_URL = "https://data.iana.org/rdap/dns.json"
RDAP_TIMEOUT_SECONDS = 2
RDAP_LOCK_TIMEOUT_SECONDS = 3
RDAP_LOCK_POLL_INTERVAL_SECONDS = 0.1
RDAP_PREFETCH_MAX_WORKERS = 2
RDAP_MAX_RETRIES_PER_URL = 5
RDAP_BACKOFF_BASE_SECONDS = 1.0
RDAP_BACKOFF_MAX_SECONDS = 8.0
_RDAP_BOOTSTRAP_CACHE: Optional[Dict[str, List[str]]] = None
_REQUEST_HEADERS = {
    "User-Agent": "GNN-Campaign-Detection/1.0 (+https://github.com)",
    "Accept": "application/rdap+json, application/json;q=0.9",
}

def extract_domain(hostname: str) -> str:
    if not hostname:
        return None
    ext = tldextract.extract(hostname)
    if ext.domain and ext.suffix:
        return f"{ext.domain}.{ext.suffix}"
    return None


def normalize_domain_input(value: str) -> Optional[str]:
    if not value:
        return None

    text = str(value).strip()
    if not text:
        return None

    if "://" in text:
        parsed = urlsplit(text)
        text = parsed.netloc or parsed.path

    text = text.split("/", 1)[0].split("?", 1)[0].split("#", 1)[0]
    text = text.split("@")[-1]
    text = text.split(":", 1)[0]
    return extract_domain(text)


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

    urls.append(f"https://rdap.org/domain/{domain}")

    seen = set()
    ordered: List[str] = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            ordered.append(u)
    return ordered


def _is_retryable_status(status_code: int) -> bool:
    return status_code == 429 or status_code >= 500


def _compute_retry_delay_seconds(response, attempt_index: int) -> float:
    retry_after = response.headers.get("Retry-After", "") if response is not None else ""
    if retry_after:
        try:
            return max(0.0, float(retry_after))
        except Exception:
            try:
                retry_dt = parsedate_to_datetime(retry_after)
                now = datetime.now(retry_dt.tzinfo) if retry_dt.tzinfo else datetime.utcnow()
                return max(0.0, (retry_dt - now).total_seconds())
            except Exception:
                pass

    backoff = RDAP_BACKOFF_BASE_SECONDS * (2 ** attempt_index)
    return min(float(backoff), RDAP_BACKOFF_MAX_SECONDS)


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


def extract_domains_from_received(value_list):
    domains = set()

    for entry in value_list:
        for key in ["helo_host", "by_host"]:
            host = entry.get(key)
            domain = extract_domain(host)
            if domain:
                domains.add(domain)

    return list(domains)


def _extract_domains_from_urls(url_values):
    domains = set()
    if isinstance(url_values, str):
        url_values = [url_values]

    if not isinstance(url_values, list):
        return domains

    for value in url_values:
        domain = normalize_domain_input(value)
        if domain:
            domains.add(domain)
    return domains


def extract_domains_from_records(records):
    """Extract registrable domains from parsed events or received-hop records.

    Supports:
    - Parsed events with `urls` and `received_hops`
    - Raw received hop dicts containing `helo_host`/`by_host`
    """
    if not isinstance(records, list):
        return []

    domains = set()
    for record in records:
        if not isinstance(record, dict):
            continue

        domains.update(_extract_domains_from_urls(record.get("urls", [])))

        received_hops = record.get("received_hops")
        if isinstance(received_hops, list):
            for hop in received_hops:
                if not isinstance(hop, dict):
                    continue
                for key in ("helo_host", "by_host"):
                    domain = normalize_domain_input(hop.get(key))
                    if domain:
                        domains.add(domain)

        for key in ("helo_host", "by_host"):
            domain = normalize_domain_input(record.get(key))
            if domain:
                domains.add(domain)

    return list(domains)


def fetch_rdap(domain):
    last_error = None
    for url in _candidate_rdap_urls_for_domain(domain):
        for attempt_index in range(RDAP_MAX_RETRIES_PER_URL + 1):
            try:
                response = requests.get(url, timeout=RDAP_TIMEOUT_SECONDS, headers=_REQUEST_HEADERS)
                status_code = int(response.status_code)

                if _is_retryable_status(status_code):
                    if attempt_index < RDAP_MAX_RETRIES_PER_URL:
                        time.sleep(_compute_retry_delay_seconds(response, attempt_index))
                        continue

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
            except requests.RequestException as e:
                response = getattr(e, "response", None)
                status_code = int(response.status_code) if response is not None else 0
                if _is_retryable_status(status_code) and attempt_index < RDAP_MAX_RETRIES_PER_URL:
                    time.sleep(_compute_retry_delay_seconds(response, attempt_index))
                    continue
                last_error = f"{type(e).__name__}: {e}"
                break
            except Exception as e:
                last_error = f"{type(e).__name__}: {e}"
                break

    return {
        "domain": domain,
        "error": last_error or "RDAP lookup failed",
        "fetched_at": datetime.utcnow().isoformat(),
    }


def _acquire_cache_lock(timeout_seconds: int = RDAP_LOCK_TIMEOUT_SECONDS):
    os.makedirs(RDAP_CACHE_DIR, exist_ok=True)
    deadline = time.time() + timeout_seconds

    while True:
        try:
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

    if os.path.exists(LEGACY_RDAP_CACHE_FILE):
        with open(LEGACY_RDAP_CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)

    return {}


def load_cache():
    return _load_cache_unlocked()


def _save_cache_unlocked(cache):
    os.makedirs(RDAP_CACHE_DIR, exist_ok=True)
    tmp_path = f"{RDAP_CACHE_FILE}.tmp.{os.getpid()}"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
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

    return "429" in error_text or "too many requests" in error_text


def ensure_rdap_cache(records):
    domains = extract_domains_from_records(records)
    cache_snapshot = load_cache()
    requested_domains = set()
    domains_to_fetch = []

    for domain in domains:
        if domain in requested_domains:
            continue
        cached_entry = cache_snapshot.get(domain)
        should_fetch = domain not in cache_snapshot or _is_retryable_cached_error(cached_entry)
        if should_fetch:
            domains_to_fetch.append(domain)
        requested_domains.add(domain)

    if not domains_to_fetch:
        return cache_snapshot

    total_domains = len(domains_to_fetch)
    print(f"RDAP cache ensure: fetching {total_domains} domains...")

    fetched_entries = {}
    completed = 0
    max_workers = min(RDAP_PREFETCH_MAX_WORKERS, len(domains_to_fetch))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_domain = {
            executor.submit(fetch_rdap, domain): domain
            for domain in domains_to_fetch
        }
        for future in as_completed(future_to_domain):
            domain = future_to_domain[future]
            try:
                fetched_entries[domain] = future.result()
            except Exception as exc:
                fetched_entries[domain] = {
                    "domain": domain,
                    "error": f"{type(exc).__name__}: {exc}",
                    "fetched_at": datetime.utcnow().isoformat(),
                }
            completed += 1
            print(f"RDAP cache ensure progress: {completed}/{total_domains}")

    _acquire_cache_lock()
    try:
        cache = _load_cache_unlocked()
        updated = False
        for domain, entry in fetched_entries.items():
            cached_entry = cache.get(domain)
            should_store = domain not in cache or _is_retryable_cached_error(cached_entry)
            if should_store:
                cache[domain] = entry
                updated = True

        if updated:
            _save_cache_unlocked(cache)

        print(f"RDAP cache ensure done: {len(fetched_entries)} fetched, updated={updated}")
        return cache
    finally:
        _release_cache_lock()
