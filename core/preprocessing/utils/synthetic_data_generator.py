# Generate a synthetic dataset of 50 emails following the user's schema and save as JSON

import json, random, hashlib, time
from email.utils import formatdate

categories = ["legitimate", "marketing", "spam", "phishing"]
domains = ["example.com","store.com","mailservice.net","secure-bank.com","company.dk"]
providers = ["gmail.com","yahoo.com","outlook.com","proton.me"]
colors = ["#000000","#1a73e8","#ff6600","#0078d4","#2a9d8f"]
body_rng = random.SystemRandom()
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

def rand_hash():
    return hashlib.sha256(str(random.random()).encode()).hexdigest()

def to_str(v):
    if isinstance(v, str):
        return v
    return json.dumps(v)

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
    """Create a subject by sampling from the shared english_words dictionary."""
    rng = rng or body_rng
    pick_count = min(rng.randint(min_words, max_words), len(english_words))
    picked = rng.sample(english_words, k=pick_count)
    return " ".join(picked)

events = []

base_epoch = 1773100000

for i in range(1, 1001):
    cat = random.choices(categories, weights=[0.55,0.25,0.15,0.05])[0]
    domain = random.choice(domains)
    to_domain = random.choice(providers)
    from_email = f"sender{i}@{domain}"
    to_email = f"user{i}@{to_domain}"
    
    links = random.randint(0,3)
    images = random.randint(0,2)
    
    attachments = []
    if random.random() < 0.25:
        attachments = [rand_hash() for _ in range(random.randint(1,2))]
    
    auth = {
        "legitimate": "spf=pass; dkim=pass; dmarc=pass; header.from="+domain,
        "marketing": "spf=pass; dkim=pass; dmarc=pass; header.from="+domain,
        "spam": "spf=fail; dkim=none; dmarc=fail; header.from="+domain,
        "phishing": "spf=fail; dkim=none; dmarc=fail; header.from="+domain
    }[cat]
    
    scl = {"legitimate":-1,"marketing":1,"spam":8,"phishing":7}[cat]
    body_text = random_body(20, rng=body_rng)
    subject_text = random_subject(rng=body_rng)
    
    event = {
        "Event": {
            "info": "Synthetic dataset email",
            "email_index": i,
            "external_id": f"evt_{i:04d}",
            "Attribute": [
                {"type":"from","value":[from_email]},
                {"type":"to","value":[to_email]},
                {"type":"subject","value":subject_text},
                {"type":"date","value":to_str(base_epoch + i)},
                {"type":"body","value":body_text},
                
                {"type":"html","value":{
                    "tag_counts":{"html":1,"head":1,"meta":1,"style":random.randint(0,1),"body":1,"table":random.randint(0,2),
                                  "tbody":random.randint(0,2),"tr":random.randint(0,4),"td":random.randint(0,8),"div":random.randint(0,3),
                                  "a":links,"p":random.randint(1,3),"o:p":0,"u":0,"ul":random.randint(0,1),"li":random.randint(0,3),
                                  "b":random.randint(0,2),"span":random.randint(0,2),"br":random.randint(0,3),"img":images},
                    "tree_stats":{"total_elements":random.randint(8,50),"max_depth":random.randint(3,7),
                                  "avg_depth":round(random.uniform(2.0,3.8),2),"forms":0,
                                  "password_fields":1 if cat=="phishing" and random.random()<0.5 else 0,
                                  "hidden_elements":random.randint(0,1),"external_scripts":0,
                                  "links":links,"images":images,
                                  "link_ratio":round(random.uniform(0.02,0.15),3),
                                  "image_ratio":round(random.uniform(0.0,0.08),3)},
                    "structure_fingerprint": hashlib.md5(f"fp{i}".encode()).hexdigest()
                }},
                
                {"type":"css","value":{"style_features":{
                    "unique_color_count":random.randint(1,5),
                    "primary_color":random.choice(colors),
                    "uses_position_absolute":False,
                    "uses_z_index":False,
                    "uses_media_queries":random.random()<0.4,
                    "unique_class_count":random.randint(0,6),
                    "class_entropy":round(random.uniform(0.0,2.5),2)
                }}},
                
                {"type":"attachments","value":attachments},
                {"type":"url","value":[f"https://link{i}-{j}.{domain}/page" for j in range(links)]},
                {"type":"category","value":cat},
                {"type":"rfc_defects","value":[]},
                {"type":"cyrillic_domain","value":to_str(False)},
                {"type":"contains_symbols","value":to_str(random.random()<0.3)},
                {"type":"body_has_tracking_url","value":to_str(links>0)},
                {"type":"body_has_tracking_image","value":to_str(images>0)},
                {"type":"body_has_tracking_pixel","value":to_str(images>0 and random.random()<0.5)},
                {"type":"body_has_unsubscribe_link","value":to_str(cat=="marketing")},
                {"type":"domain_is_common_webprovided","value":to_str(False)},
                
                {"type":"header_Received","value":[
                    {"origin_ip":"203.0.113."+str(random.randint(1,200)),"helo_host":"mail."+domain,
                     "by_host":"mx."+domain,"timestamp": rfc_timestamp()},
                    {"origin_ip":"203.0.113."+str(random.randint(1,200)),"helo_host":"mx."+domain,
                     "by_host":"mailbox."+to_domain,"timestamp": rfc_timestamp()}
                ]},
                
                {"type":"header_Return-Path","value":{"email":from_email,"domain":domain}},
                {"type":"header_Content-Type","value":["multipart/alternative","text/html"]},
                {"type":"header_Received-SPF","value":f"domain={domain}; helo=mail.{domain}"},
                {"type":"header_List-Unsubscribe","value":"<https://unsubscribe."+domain+">" if cat=="marketing" else ""},
                {"type":"header_Authentication-Results","value":auth},
                {"type":"header_X-Forefront-Antispam-Report","value":f"CTRY:US; LANG:en; SCL:{scl}; SFV:NSPM; CAT:{cat.upper()}"},
                {"type":"header_X-MS-Exchange-Organization-SCL","value":[to_str(scl)]}
            ]
        }
    }
    
    events.append(event)

dataset = {"Events": events}

path = "../../../data/misp/synthetic_email_dataset_50.json"
with open(path, "w") as f:
    json.dump(dataset, f, indent=2)

path