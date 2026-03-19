import json
import os
from datetime import datetime
import requests
import tldextract

RDAP_CACHE_FILE = "rdap_cache.json"

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
    try:
        url = f"https://rdap.org/domain/{domain}"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()

        registrar = None
        creation_date = None
        registrar_location = None

        # Extract registrar
        entities = data.get("entities", [])
        for ent in entities:
            roles = ent.get("roles", [])
            if "registrar" in roles:
                vcard = ent.get("vcardArray", [])
                if len(vcard) > 1:
                    for item in vcard[1]:
                        if item[0] == "fn":
                            registrar = item[3]
                        if item[0] == "adr":
                            registrar_location = " ".join(item[3])

        # Extract creation date
        for event in data.get("events", []):
            if event.get("eventAction") == "registration":
                creation_date = event.get("eventDate")

        return {
            "domain": domain,
            "registrar": registrar,
            "registration_date": creation_date,
            "registrar_location": registrar_location,
            "fetched_at": datetime.utcnow().isoformat()
        }

    except Exception as e:
        return {
            "domain": domain,
            "error": str(e)
        }


# -----------------------------
# Cache handling
# -----------------------------
def load_cache():
    if not os.path.exists(RDAP_CACHE_FILE):
        return {}

    with open(RDAP_CACHE_FILE, "r") as f:
        return json.load(f)


def save_cache(cache):
    with open(RDAP_CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2)


# -----------------------------
# Ensure cache exists / populate
# -----------------------------
def ensure_rdap_cache(domains):
    cache = load_cache()
    updated = False

    for domain in domains:
        if domain not in cache:
            print(f"Fetching RDAP for {domain}...")
            cache[domain] = fetch_rdap(domain)
            updated = True

    if updated:
        save_cache(cache)

    return cache


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
