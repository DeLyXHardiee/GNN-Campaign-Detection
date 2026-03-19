import hashlib
import json
import os
import random
import string
from email.utils import formatdate
from urllib.parse import urlparse


categories = ["legitimate", "marketing", "spam", "phishing"]
default_domains = ["example.com", "store.com", "mailservice.net", "secure-bank.com", "company.dk"]
providers = ["gmail.com", "yahoo.com", "outlook.com", "proton.me"]
colors = ["#000000", "#1a73e8", "#ff6600", "#0078d4", "#2a9d8f"]
body_rng = random.SystemRandom()

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_CORE_DIR = os.path.dirname(os.path.dirname(_THIS_DIR))
_REPO_ROOT = os.path.dirname(_CORE_DIR)
_CACHE_DIR = os.path.join(_CORE_DIR, "feature_set_extraction", "caches")
_OUTPUT_PATH = os.path.join(_REPO_ROOT, "data", "misp", "synthetic_email_dataset_50.json")

english_words = [
    "ability", "access", "account", "accuracy", "action", "adapter", "address", "adjustment", "admin", "alert",
    "analysis", "analyst", "anchor", "answer", "api", "appeal", "application", "approve", "archive", "argument",
    "array", "article", "asset", "assignment", "audit", "authority", "automation", "backup", "balance", "banner",
    "base", "behavior", "benefit", "billing", "binary", "board", "bookmark", "bot", "branch", "brand",
    "browser", "budget", "buffer", "build", "button", "cache", "calendar", "camera", "campaign", "candidate",
    "capacity", "capture", "card", "carrier", "catalog", "category", "cell", "center", "certificate", "change",
    "channel", "chart", "check", "cipher", "claim", "class", "client", "clock", "cloud", "cluster",
    "code", "column", "command", "comment", "commerce", "communication", "company", "compare", "component", "compute",
    "concept", "condition", "config", "confirmation", "connect", "connection", "console", "constraint", "contact", "container",
    "content", "context", "contract", "control", "cookie", "copy", "core", "correction", "cost", "count",
    "course", "coverage", "create", "credential", "credit", "criteria", "customer", "cycle", "dashboard", "data",
    "dataset", "date", "debug", "decision", "default", "delay", "delivery", "delta", "density", "department",
    "dependency", "deployment", "description", "design", "desktop", "detail", "device", "difference", "digest", "dimension",
    "directory", "discount", "discovery", "display", "distribution", "document", "domain", "download", "driver", "edition",
    "editor", "element", "email", "employee", "encoding", "endpoint", "engine", "entry", "environment", "episode",
    "error", "estimate", "event", "evidence", "example", "exchange", "execution", "expense", "experiment", "exposure",
    "extension", "factor", "failure", "feature", "feedback", "field", "file", "filter", "finance", "fingerprint",
    "flag", "flow", "folder", "format", "form", "framework", "function", "gateway", "general", "graph",
    "group", "guard", "guide", "handler", "hash", "header", "health", "help", "history", "home",
    "host", "hour", "html", "hyperlink", "idea", "identifier", "image", "import", "incident", "index",
    "indicator", "inference", "information", "infrastructure", "input", "inspection", "instance", "instruction", "integrity", "interface",
    "inventory", "invoice", "item", "job", "journey", "kernel", "key", "keyword", "knowledge", "label",
    "language", "latency", "layer", "layout", "ledger", "length", "letter", "level", "library", "license",
    "limit", "link", "listener", "load", "local", "location", "logic", "login", "lookup", "mail",
    "maintenance", "management", "manager", "map", "market", "matrix", "measure", "member", "memory", "message",
    "metadata", "metric", "model", "module", "monitor", "month", "name", "namespace", "navigation", "network",
    "node", "normalization", "notice", "notification", "number", "object", "observation", "office", "offset", "online",
    "operation", "operator", "option", "order", "organization", "output", "overview", "owner", "package", "page",
    "panel", "parameter", "parser", "partner", "password", "path", "pattern", "payment", "payload", "people",
    "performance", "permission", "person", "phase", "phone", "pipeline", "pixel", "plan", "platform", "policy",
    "portal", "position", "preference", "preparation", "presence", "preview", "price", "priority", "process", "processor",
    "product", "profile", "project", "projection", "proof", "property", "proposal", "protect", "protocol", "provider",
    "proxy", "quality", "queue", "quota", "range", "rate", "reader", "reason", "receipt", "record",
    "recovery", "reference", "registry", "relationship", "release", "reminder", "report", "repository", "request", "requirement",
    "research", "resolution", "resource", "response", "result", "retention", "review", "revision", "risk", "route",
    "rule", "sample", "schedule", "schema", "scope", "screen", "script", "search", "season", "section",
    "secure", "security", "segment", "selection", "sender", "sequence", "server", "service", "session", "setting",
    "share", "sheet", "signal", "signature", "site", "size", "snapshot", "software", "solution", "source",
    "specification", "speed", "stack", "stage", "standard", "state", "statement", "status", "storage", "store",
    "strategy", "stream", "string", "structure", "style", "subject", "submission", "subscription", "success", "summary",
    "support", "surface", "switch", "symbol", "system", "table", "tag", "target", "task", "team",
    "template", "tenant", "terminal", "test", "theme", "thread", "ticket", "time", "token", "topic",
    "trace", "tracking", "traffic", "training", "transaction", "transfer", "transform", "transport", "tree", "trend",
    "trigger", "trust", "type", "update", "upload", "url", "usage", "user", "utility", "validation",
    "value", "variable", "vector", "vendor", "verification", "verify", "version", "view", "visibility", "vision",
    "volume", "warning", "web", "website", "window", "wire", "workflow", "workspace", "writer", "zone"
]


def _load_txt_values(file_name):
    path = os.path.join(_CACHE_DIR, file_name)
    if not os.path.exists(path):
        return []

    values = []
    seen = set()
    with open(path, "r", encoding="utf-8") as handle:
        for raw in handle:
            value = str(raw or "").strip()
            if not value or value.startswith("#"):
                continue
            if value not in seen:
                seen.add(value)
                values.append(value)
    return values


POPULAR_DOMAINS = _load_txt_values("popular_domains.txt")
WEB_HOSTING_DOMAINS = _load_txt_values("web_hosting_domains.txt")
PHISHTANK_URLS = _load_txt_values("phishtank_urls.txt")


def rand_hash():
    return hashlib.sha256(str(random.random()).encode()).hexdigest()


def to_str(value):
    if isinstance(value, str):
        return value
    return json.dumps(value)


def rfc_timestamp():
    return formatdate(usegmt=True)


def random_body(word_count=100, rng=None):
    rng = rng or body_rng
    if word_count <= len(english_words):
        return " ".join(rng.sample(english_words, k=word_count))
    unique_words = rng.sample(english_words, k=len(english_words))
    remaining_words = rng.choices(english_words, k=word_count - len(english_words))
    return " ".join(unique_words + remaining_words)


def random_subject(min_words=3, max_words=7, rng=None):
    rng = rng or body_rng
    pick_count = min(rng.randint(min_words, max_words), len(english_words))
    picked = rng.sample(english_words, k=pick_count)
    return " ".join(picked)


def _pick_one(values, fallback):
    if values:
        return body_rng.choice(values)
    return fallback


def _mutate_single_letter(label):
    letters = [index for index, char in enumerate(label) if char.isalpha()]
    if not letters:
        return label

    index = body_rng.choice(letters)
    original = label[index].lower()
    replacements = [char for char in string.ascii_lowercase if char != original]
    replacement = body_rng.choice(replacements)
    return f"{label[:index]}{replacement}{label[index + 1:]}"


def make_typo_domain(domain):
    parts = str(domain or "").split(".")
    if len(parts) < 2:
        return domain

    label_index = -2 if len(parts) >= 2 else 0
    if len(parts[label_index]) < 3:
        return domain

    parts[label_index] = _mutate_single_letter(parts[label_index])
    return ".".join(parts)


def _extract_host(url):
    try:
        return (urlparse(url).hostname or "").lower()
    except Exception:
        return ""


def _build_hosted_url(domain, email_index, link_index, category):
    prefix = body_rng.choice(["cdn", "assets", "login", "portal", "static", "files"])
    path = {
        "marketing": "offer",
        "spam": "promo",
        "phishing": "verify",
        "legitimate": "page",
    }.get(category, "page")
    return f"https://{prefix}.{domain}/{path}"


def _build_typo_phish_url(email_index, link_index):
    source_domain = _pick_one(POPULAR_DOMAINS, _pick_one(default_domains, "example.com"))
    typo_domain = make_typo_domain(source_domain)
    subdomain = body_rng.choice(["login", "secure", "verify", "account", "update"])
    path = body_rng.choice(["login", "verify", "auth", "secure", "review"])
    return f"https://{subdomain}.{typo_domain}/{path}"


def _build_urls(category, email_index, link_count):
    urls = []

    phishtank_url = _pick_one(PHISHTANK_URLS, _build_typo_phish_url(email_index, 0))
    hosted_domain = _pick_one(WEB_HOSTING_DOMAINS, "pages.dev")
    hosted_url = _build_hosted_url(hosted_domain, email_index, 1, category)
    typo_url = _build_typo_phish_url(email_index, 2)

    urls.append(phishtank_url)
    urls.append(hosted_url)
    urls.append(typo_url)
    return urls


def _choose_sender_domain(category, urls):
    if category == "phishing" and POPULAR_DOMAINS:
        return make_typo_domain(_pick_one(POPULAR_DOMAINS, "google.com"))

    if category in {"marketing", "spam"} and WEB_HOSTING_DOMAINS and body_rng.random() < 0.5:
        return _pick_one(WEB_HOSTING_DOMAINS, "pages.dev")

    if urls and category == "phishing":
        host = _extract_host(urls[0])
        if host:
            return host

    return _pick_one(default_domains, "example.com")


def _make_body_with_urls(category, urls):
    body_text = random_body(20, rng=body_rng)
    if not urls:
        return body_text

    intro = {
        "legitimate": "Reference links:",
        "marketing": "View offers here:",
        "spam": "Open these links now:",
        "phishing": "Review your account immediately:",
    }.get(category, "Links:")
    return f"{body_text}\n\n{intro}\n" + "\n".join(urls)


def _is_hosting_backed(domain, urls):
    if domain in WEB_HOSTING_DOMAINS:
        return True
    for url in urls:
        host = _extract_host(url)
        if any(host == provider or host.endswith(f".{provider}") for provider in WEB_HOSTING_DOMAINS):
            return True
    return False


events = []
base_epoch = 1773100000

for index in range(1, 1001):
    category = random.choices(categories, weights=[0.55, 0.25, 0.15, 0.05])[0]
    to_domain = random.choice(providers)
    links = 3
    images = random.randint(0, 2)

    attachments = []
    if random.random() < 0.25:
        attachments = [rand_hash() for _ in range(random.randint(1, 2))]

    urls = _build_urls(category, index, links)
    sender_domain = _choose_sender_domain(category, urls)
    from_email = f"sender{index}@{sender_domain}"
    to_email = f"user{index}@{to_domain}"

    auth = {
        "legitimate": f"spf=pass; dkim=pass; dmarc=pass; header.from={sender_domain}",
        "marketing": f"spf=pass; dkim=pass; dmarc=pass; header.from={sender_domain}",
        "spam": f"spf=fail; dkim=none; dmarc=fail; header.from={sender_domain}",
        "phishing": f"spf=fail; dkim=none; dmarc=fail; header.from={sender_domain}",
    }[category]

    scl = {"legitimate": -1, "marketing": 1, "spam": 8, "phishing": 7}[category]
    body_text = _make_body_with_urls(category, urls)
    subject_text = random_subject(rng=body_rng)
    uses_hosting_domain = _is_hosting_backed(sender_domain, urls)

    event = {
        "Event": {
            "info": "Synthetic dataset email",
            "email_index": index,
            "external_id": f"evt_{index:04d}",
            "Attribute": [
                {"type": "from", "value": [from_email]},
                {"type": "to", "value": [to_email]},
                {"type": "subject", "value": subject_text},
                {"type": "date", "value": to_str(base_epoch + index)},
                {"type": "body", "value": body_text},
                {"type": "html", "value": {
                    "tag_counts": {"html": 1, "head": 1, "meta": 1, "style": random.randint(0, 1), "body": 1, "table": random.randint(0, 2),
                                   "tbody": random.randint(0, 2), "tr": random.randint(0, 4), "td": random.randint(0, 8), "div": random.randint(0, 3),
                                   "a": links, "p": random.randint(1, 3), "o:p": 0, "u": 0, "ul": random.randint(0, 1), "li": random.randint(0, 3),
                                   "b": random.randint(0, 2), "span": random.randint(0, 2), "br": random.randint(0, 3), "img": images},
                    "tree_stats": {"total_elements": random.randint(8, 50), "max_depth": random.randint(3, 7),
                                  "avg_depth": round(random.uniform(2.0, 3.8), 2), "forms": 0,
                                  "password_fields": 1 if category == "phishing" and random.random() < 0.5 else 0,
                                  "hidden_elements": random.randint(0, 1), "external_scripts": 0,
                                  "links": links, "images": images,
                                  "link_ratio": round(random.uniform(0.02, 0.15), 3),
                                  "image_ratio": round(random.uniform(0.0, 0.08), 3)},
                    "structure_fingerprint": hashlib.md5(f"fp{index}".encode()).hexdigest(),
                }},
                {"type": "css", "value": {"style_features": {
                    "unique_color_count": random.randint(1, 5),
                    "primary_color": random.choice(colors),
                    "uses_position_absolute": False,
                    "uses_z_index": False,
                    "uses_media_queries": random.random() < 0.4,
                    "unique_class_count": random.randint(0, 6),
                    "class_entropy": round(random.uniform(0.0, 2.5), 2),
                }}},
                {"type": "attachments", "value": attachments},
                {"type": "url", "value": urls},
                {"type": "category", "value": category},
                {"type": "rfc_defects", "value": []},
                {"type": "cyrillic_domain", "value": to_str(False)},
                {"type": "contains_symbols", "value": to_str(random.random() < 0.3)},
                {"type": "body_has_tracking_url", "value": to_str(links > 0)},
                {"type": "body_has_tracking_image", "value": to_str(images > 0)},
                {"type": "body_has_tracking_pixel", "value": to_str(images > 0 and random.random() < 0.5)},
                {"type": "body_has_unsubscribe_link", "value": to_str(category == "marketing")},
                {"type": "domain_is_common_webprovided", "value": to_str(uses_hosting_domain)},
                {"type": "header_Received", "value": [
                    {"origin_ip": "203.0.113." + str(random.randint(1, 200)), "helo_host": "mail." + sender_domain,
                     "by_host": "mx." + sender_domain, "timestamp": rfc_timestamp()},
                    {"origin_ip": "203.0.113." + str(random.randint(1, 200)), "helo_host": "mx." + sender_domain,
                     "by_host": "mailbox." + to_domain, "timestamp": rfc_timestamp()},
                ]},
                {"type": "header_Return-Path", "value": {"email": from_email, "domain": sender_domain}},
                {"type": "header_Content-Type", "value": ["multipart/alternative", "text/html"]},
                {"type": "header_Received-SPF", "value": f"domain={sender_domain}; helo=mail.{sender_domain}"},
                {"type": "header_List-Unsubscribe", "value": f"<https://unsubscribe.{sender_domain}>" if category == "marketing" else ""},
                {"type": "header_Authentication-Results", "value": auth},
                {"type": "header_X-Forefront-Antispam-Report", "value": f"CTRY:US; LANG:en; SCL:{scl}; SFV:NSPM; CAT:{category.upper()}"},
                {"type": "header_X-MS-Exchange-Organization-SCL", "value": [to_str(scl)]},
            ]
        }
    }

    events.append(event)

dataset = {"Events": events}
os.makedirs(os.path.dirname(_OUTPUT_PATH), exist_ok=True)
with open(_OUTPUT_PATH, "w", encoding="utf-8") as handle:
    json.dump(dataset, handle, indent=2)

print(f"Wrote synthetic dataset to {_OUTPUT_PATH}")