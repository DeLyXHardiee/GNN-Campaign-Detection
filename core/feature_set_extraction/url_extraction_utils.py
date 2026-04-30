import re
import math
import tldextract
import idna
from urllib.parse import urlparse
from datetime import datetime
from collections import defaultdict, Counter
from preprocessing.RDAP_processor import ensure_rdap_cache

# WHOIS import disabled — domain lookups are outcommented per request
# try:
#     import whois
# except Exception:
#     whois = None
whois = None

try:
    import Levenshtein
    def edit_distance(a, b):
        return Levenshtein.distance(a, b)
except ImportError:
    def edit_distance(a, b):
        # fallback simple implementation
        if len(a) < len(b):
            return edit_distance(b, a)
        if len(b) == 0:
            return len(a)
        prev = range(len(b) + 1)
        for i, c1 in enumerate(a):
            curr = [i + 1]
            for j, c2 in enumerate(b):
                ins = prev[j + 1] + 1
                dele = curr[j] + 1
                sub = prev[j] + (c1 != c2)
                curr.append(min(ins, dele, sub))
            prev = curr
        return prev[-1]

SHORTENER_DOMAINS = {
    "bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly",
    "is.gd", "buff.ly", "adf.ly", "cutt.ly", "rb.gy"
}

IP_PATTERN = re.compile(r"^\d{1,3}(\.\d{1,3}){3}$")


def _denoise_url_string(url: str) -> str:
    """Strip noise so ``urlparse`` / heuristics see a single URL token where possible."""
    if url is None:
        return ""
    s = str(url).strip()
    if not s:
        return ""
    s = s.replace("\x00", "")
    s = "".join(c for c in s if c == "\t" or ord(c) >= 32)
    parts = s.split()
    s = parts[0] if parts else ""
    return s.strip("<>\"'")


def _urlparse_safe(url: str):
    if not url:
        return None
    try:
        return urlparse(url)
    except ValueError:
        return None


def _fallback_hostname(url: str) -> str:
    """
    Hostname when ``urlparse`` fails (e.g. ``Invalid IPv6 URL`` on broken ``[``/``:``).

    Does not validate IPv6; returns empty for unclosed ``[``.
    """
    s = (url or "").strip()
    if not s:
        return ""
    if "://" in s:
        authority = s.split("://", 1)[1]
    else:
        authority = s
    authority = authority.split("/")[0].split("?")[0].split("#")[0]
    if "@" in authority:
        authority = authority.rsplit("@", 1)[-1]
    authority = authority.strip()
    if not authority:
        return ""
    if authority.startswith("["):
        end = authority.find("]")
        if end == -1:
            return ""
        return authority[1:end].strip().lower()
    if authority.count(":") == 1:
        host, _, port = authority.partition(":")
        if port.isdigit():
            authority = host.strip().lower()
        else:
            authority = authority.lower()
    else:
        authority = authority.lower()
    return authority


def parse_url_host_and_registrable_domain(url: str) -> tuple[str, str, bool]:
    """
    Robust host + eTLD+1 for cache / policy matching.

    Returns:
        ``(hostname_lower, registrable_domain_lower, host_ok)``
        ``host_ok`` is False only if no hostname could be recovered after denoise + fallbacks.
    """
    raw = _denoise_url_string(url)
    if not raw:
        return "", "", False
    norm = _normalize_url_for_hostname_extraction(raw)
    host = ""
    p = _urlparse_safe(norm)
    if p is not None:
        try:
            h = p.hostname
            host = (h or "").strip().lower() if h else ""
        except ValueError:
            host = ""
    if not host:
        host = _fallback_hostname(norm) or _fallback_hostname(raw)
    if not host:
        return "", "", False
    ext = tldextract.extract(host)
    reg = ".".join(p for p in [ext.domain, ext.suffix] if p).lower()
    return host, reg, True


def shard_url_infra_classify(url: str, popular_domains: frozenset[str]) -> tuple[str, str]:
    """
    Returns ``(kind, registrable_domain)`` where ``kind`` is
    ``"malformed"`` (no host), ``"benign"`` (reg domain in list), or ``"kept"``.
    """
    _, reg, ok = parse_url_host_and_registrable_domain(url)
    if not ok:
        return "malformed", ""
    if reg and reg in popular_domains:
        return "benign", reg
    return "kept", reg or ""


def _normalize_url_for_hostname_extraction(url: str) -> str:
    """
    Ensure ``urlparse`` can see a network location.

    Graph and email URLs are often stored **without** a scheme (e.g. ``www.example.com/a``).
    Bare ``urlparse`` puts that in ``path`` and leaves ``hostname`` empty. Prepending
    ``http://`` matches ``url_extractor.parse_url_components`` and yields a real host for
    tldextract / popular-domain lists (which contain registrable domains only, no scheme).
    """
    s = (url or "").strip()
    if not s:
        return s
    low = s.lower()
    if low.startswith(("http://", "https://")):
        return s
    if s.startswith("//"):
        return "http:" + s
    return "http://" + s


def extract_domain_info(url):
    raw = "" if url is None else str(url)
    host, domain, host_ok = parse_url_host_and_registrable_domain(raw)
    if not host_ok:
        host = ""
        domain = ""
        ext = tldextract.extract("")
    else:
        ext = tldextract.extract(host)

    domain = domain or ".".join(p for p in [ext.domain, ext.suffix] if p)
    subdomain = ext.subdomain

    subdomain_count = 0 if not subdomain else len(subdomain.split("."))

    hyphen_count = host.count("-")
    has_at_symbol = "@" in raw
    has_extra_http = raw.lower().count("http") > 1

    try:
        host.encode("ascii")
        has_non_ascii = False
    except UnicodeEncodeError:
        has_non_ascii = True

    is_ip = bool(IP_PATTERN.match(host))
    is_shortener = domain in SHORTENER_DOMAINS

    return {
        "domain": domain,
        "hostname": host,
        "subdomain_count": subdomain_count,
        "hyphen_count": hyphen_count,
        "has_at_symbol": has_at_symbol,
        "has_extra_http": has_extra_http,
        "has_non_ascii": has_non_ascii,
        "is_ip": is_ip,
        "is_shortener": is_shortener,
        "tld": ext.suffix
    }


def is_typo_of_popular(domain, popular_domains, max_distance=2):
    base = domain.split(".")[0]
    for pop in popular_domains:
        if edit_distance(base, pop.split(".")[0]) <= max_distance:
            return True
    return False


def contains_popular_in_subdomain(url, popular_domains):
    host, _, ok_h = parse_url_host_and_registrable_domain("" if url is None else str(url))
    host = host if ok_h else ""
    for pop in popular_domains:
        if pop in host and not host.endswith(pop):
            return True
    return False

def extract_url_features(
    urls,
    popular_domains=None,
    webhost_domains=None,
    phishing_target_domains=None,
    blacklist=None,
    domain_metadata=None,
    anchor_pairs=None   # list of (visible_text, actual_url)
):
    """
    urls: list[str]
    popular_domains: set[str] (top 10k etc.)
    webhost_domains: set[str] (known web-hosting-like domains)
    phishing_target_domains: set[str]
    blacklist: set[str]
    domain_metadata: dict[domain] -> {
        'created': datetime,            # registration/creation date (from WHOIS)
        'registrar': str,               # registrar name (from WHOIS)
        'category': str,                # optional category/label for the domain
        'registrar_location': str       # optional registrar country/location
    }
    anchor_pairs: [(visible_text, actual_url)]
    """

    popular_domains = popular_domains or set()
    webhost_domains = webhost_domains or set()
    phishing_target_domains = phishing_target_domains or set()
    blacklist = blacklist or set()
    domain_metadata = domain_metadata or {}

    # Domain metadata lookups disabled (WHOIS/RDAP/TLS). Keep `domain_metadata` if supplied,
    # otherwise proceed without attempting lookups.
    per_url = [extract_domain_info(u) for u in urls]

    # ---------- aggregate ----------
    domains = [d["domain"] for d in per_url if d["domain"]]
    unique_domains = set(domains)

    num_ip_urls = sum(d["is_ip"] for d in per_url)
    num_short_urls = sum(d["is_shortener"] for d in per_url)
    num_blacklisted_domains = sum(1 for d in unique_domains if d in phishing_target_domains)


    # ---------- per-URL boolean aggregations ----------
    any_has_extra_http = any(d.get("has_extra_http") for d in per_url)
    any_has_at_symbol = any(d.get("has_at_symbol") for d in per_url)
    any_has_non_ascii = any(d.get("has_non_ascii") for d in per_url)

    # ---------- domain stats (creation dates via RDAP cache) ----------
    creation_dates = []
    if unique_domains:
        try:
            cache = ensure_rdap_cache(unique_domains)
            for d in unique_domains:
                item = cache.get(d, {})
                registration_date = item.get("registration_date")
                if registration_date:
                    creation_dates.append(registration_date)
        except Exception:
            pass

    # ---------- domain categories / registrar locations ----------
    domain_category_map = {}
    registrar_location_map = {}
    for d in unique_domains:
        meta = domain_metadata.get(d, {})
        cat = meta.get("category")
        if cat:
            domain_category_map[d] = cat
        reg_loc = meta.get("registrar_location")
        if reg_loc:
            registrar_location_map[d] = reg_loc

    # ---------- accumulated subdomain / hyphen counts ----------
    subdomain_counts = sum(d.get("subdomain_count", 0) for d in per_url)
    hyphen_counts = sum(d.get("hyphen_count", 0) for d in per_url)

    # ---------- EV certs and web-host heuristics ----------
    ev_domains = {d for d in unique_domains if domain_metadata.get(d, {}).get("ev")}
    any_ev_cert = len(ev_domains) > 0

    # Match against caller-provided web-host domain list.
    any_is_web_hosting_domain = any((d.get("domain") or "") in webhost_domains for d in per_url)

    # heuristic for multi-part TLDs (e.g., co.uk)
    any_multi_part_tld = any((d.get("tld") or "").count(".") >= 1 for d in per_url)

    # aggregated binary flags for typo/similarity/popular-subdomain
    any_typo_popular = any(is_typo_of_popular(d["domain"], popular_domains) for d in per_url if d.get("domain"))
    any_similar_phish_target = any(d["domain"] in phishing_target_domains for d in per_url)
    any_popular_in_subdomain = any(contains_popular_in_subdomain(u, popular_domains) for u in urls)

    oldest_domain = min(creation_dates) if creation_dates else None
    newest_domain = max(creation_dates) if creation_dates else None

    # ---------- hyperlink analysis ----------
    mismatch_count = 0
    click_here_links = 0

    if anchor_pairs:
        for text, actual in anchor_pairs:
            if text.strip().lower() in {"click here", "here", "link"}:
                click_here_links += 1
            if text.startswith("http") and text != actual:
                mismatch_count += 1

    # keep domain/hostname values as lists for downstream set/text processing
    domain_list = sorted(unique_domains)
    hostname_list = [h for h in (d.get("hostname") for d in per_url) if h]

    return {
        "domains": domain_list,
        "hostnames": hostname_list,
        "domain_categories": domain_category_map,
        "registrar_locations": registrar_location_map,
        "subdomain_counts": subdomain_counts,
        "hyphen_counts": hyphen_counts,
        
        #outcommented EV cert because its unavailable, is part of WHOIS but this is not provided and getting EV cert requires connecting to host
        #"any_ev_cert": any_ev_cert,
        "any_has_extra_http": any_has_extra_http,
        "any_multi_part_tld": any_multi_part_tld,
        "any_is_web_hosting_domain": any_is_web_hosting_domain,
        "any_has_at_symbol": any_has_at_symbol,
        "any_has_non_ascii": any_has_non_ascii,
        "any_typo_popular_domains": any_typo_popular,
        "any_similar_phish_targets": any_similar_phish_target,
        "any_popular_domain_in_subdomain": any_popular_in_subdomain,

        "num_ip_urls": num_ip_urls,
        "num_distinct_domains": len(unique_domains),
        "num_short_urls": num_short_urls,
        "num_blacklisted_domains": num_blacklisted_domains,

        "oldest_domain_registration": oldest_domain,
        "newest_domain_registration": newest_domain,
    }



'''
urls = [
    "http://secure-paypal-login.com/login",   # hyphen-heavy phishing-like domain
    "https://bit.ly/3abc",                    # shortener
    "http://192.168.1.1/update",              # raw IP
    "http://www.example.co.uk/path",          # www + multi-part TLD
    "http://goggle.com",                      # typo of google.com
    "http://paypal.com/login",                # known target domain
    "http://google.com.evil.com/page",        # popular domain inside subdomain
    "http://user@example.com/",               # contains an '@' in authority
    "http://exämple.test/",                   # non-ascii host (unicode)
    "http://a.b.c.example.com/page",          # multiple subdomains
    "https://tinyurl.com/xyz",                # another shortener
    "http://httphttp.com/httphttp",           # contains repeated 'http' token
    "https://secure-login-paypal.co.uk/login",# hyphens + alternate TLD
    "http://sub.www.google.com/path",         # 'www' in middle of subdomain
]

popular_domains = {"google.com", "paypal.com", "amazon.com", "facebook.com", "microsoft.com"}
phish_targets = {"paypal.com", "amazon.com"}
blacklist = {"http://secure-paypal-login.com/login", "https://malicious.example.com/bad"}

features = extract_url_features(
    urls,
    popular_domains=popular_domains,
    phishing_target_domains=phish_targets,
    blacklist=blacklist,
    domain_metadata=None
)

print(features)'''