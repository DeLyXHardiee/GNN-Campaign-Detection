import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import torch
import os

def save_model(model, optimizer, epoch, config, path="checkpoint.pt"):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
        "config": config
    }, path)

    print(f"Model saved to {path}")


def load_model(path, model_class):
    checkpoint = torch.load(path, map_location="cuda")

    config = checkpoint["config"]
    model = model_class(**config)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.cuda()

    optimizer = None
    if checkpoint["optimizer_state_dict"] is not None:
        optimizer = torch.optim.AdamW(model.parameters())
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    print(f"Loaded model from epoch {checkpoint['epoch']}")

    return model, optimizer, checkpoint["epoch"], config

from transformers import RobertaTokenizer

tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

MAX_LEN = 128

def tokenize(batch_urls):
    return tokenizer(
        batch_urls,
        padding="max_length",
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="pt"
    )

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaConfig, RobertaModel

class URLEncoder(nn.Module):
    def __init__(self,
                 embed_dim=256,
                 hidden_size=384,
                 num_hidden_layers=6,
                 num_attention_heads=6,
                 intermediate_size=768,
                 max_len=128):

        super().__init__()

        config = RobertaConfig(
            vocab_size=len(tokenizer),
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            max_position_embeddings=max_len + 2
        )

        self.encoder = RobertaModel(config)

        self.projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, embed_dim)
        )
    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        cls = outputs.last_hidden_state[:, 0]
        z = self.projection(cls)
        z = F.normalize(z, dim=1)
        return z
    
import random
import re
from urllib.parse import urlparse, parse_qsl, urlencode, urlunparse, unquote

# -----------------------------
# Tier A: Canonicalization
# -----------------------------

TRACKING_KEYS = {
    "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
    "gclid", "fbclid", "msclkid", "igshid", "mc_cid", "mc_eid", "ref", "referrer",
}

DEFAULT_PORTS = {"80", "443"}

# Detect "random-looking" segments (IDs) we can safely replace with same-shape noise
HEX_RE = re.compile(r"^[0-9a-f]{8,}$", re.IGNORECASE)
B64ISH_RE = re.compile(r"^[A-Za-z0-9_-]{12,}$")  # url-safe base64-ish / tokens
UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE
)

def safe_urlparse(url: str):
    try:
        return urlparse(url)
    except ValueError:
        return None

def _canonicalize_url(url: str) -> str:
    url = (url or "").strip()
    if not url:
        return ""

    try:
        parsed = urlparse(url if "://" in url else "http://" + url)
    except ValueError:
        # Malformed URL (often bad IPv6). Fall back to a safe, non-crashing form.
        return url.lower()

    scheme = (parsed.scheme or "http").lower()
    netloc = parsed.netloc
    path = parsed.path or ""
    query = parsed.query or ""
    fragment = ""

    # Lowercase host + remove default port + optionally strip leading www.
    if netloc:
        host = netloc
        userinfo = ""
        port = ""

        # split userinfo if present
        if "@" in host:
            userinfo, host = host.rsplit("@", 1)

        # split port if present
        if ":" in host:
            h, p = host.rsplit(":", 1)
            if p.isdigit() and p in DEFAULT_PORTS:
                host = h
            else:
                host = h
                port = p

        host = host.lower()
        if host.startswith("www."):
            host = host[4:]

        netloc = (userinfo + "@" if userinfo else "") + host + (f":{port}" if port else "")

    # Normalize repeated slashes in path
    path = re.sub(r"/{2,}", "/", path)

    # Normalize trailing slash (keep single root slash)
    if path != "/" and path.endswith("/"):
        path = path[:-1]

    # Percent-decode *safely* (unquote leaves reserved chars decoded too; acceptable for clustering)
    # You can comment this out if you're worried about changing semantics.
    path = unquote(path)
    query = unquote(query)

    # Sort query params by key and drop tracking params
    if query:
        params = parse_qsl(query, keep_blank_values=True)
        params = [(k, v) for (k, v) in params if k.lower() not in TRACKING_KEYS]
        params.sort(key=lambda kv: kv[0])
        query = urlencode(params, doseq=True)

    # Collapse repeated separators in whole URL-ish parts (conservative)
    # e.g. "__" -> "_" and "--" -> "-"
    path = re.sub(r"_{2,}", "_", path)
    path = re.sub(r"-{2,}", "-", path)

    return urlunparse((scheme, netloc, path, "", query, fragment))


# -----------------------------
# Tier B: Strong "campaign-preserving" noise
# -----------------------------

def _random_digits_same_len(s: str) -> str:
    return "".join(str(random.randint(0, 9)) for _ in range(len(s)))

def _random_alnum(n: int) -> str:
    alphabet = "abcdefghijklmnopqrstuvwxyz0123456789"
    return "".join(random.choices(alphabet, k=n))

def _random_urlsafe(n: int) -> str:
    alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
    return "".join(random.choices(alphabet, k=n))

def random_digit_replace(s: str, p: float = 0.35) -> str:
    """Replace individual digits with random digits (same length overall)."""
    return "".join(
        str(random.randint(0, 9)) if c.isdigit() and random.random() < p else c
        for c in s
    )

def shuffle_query_params(url: str) -> str:
    parsed = urlparse(url)
    params = parse_qsl(parsed.query, keep_blank_values=True)
    random.shuffle(params)
    new_query = urlencode(params, doseq=True)
    return urlunparse(parsed._replace(query=new_query))

def maybe_add_tracking_params(url: str, max_add: int = 3) -> str:
    """Add 1..max_add synthetic tracking-ish params."""
    parsed = urlparse(url)
    params = parse_qsl(parsed.query, keep_blank_values=True)

    add_n = random.randint(1, max_add)
    choices = ["utm_source", "utm_medium", "utm_campaign", "ref", "sid"]
    for _ in range(add_n):
        k = random.choice(choices)
        v = _random_urlsafe(random.randint(4, 12))
        params.append((k, v))

    new_query = urlencode(params, doseq=True)
    return urlunparse(parsed._replace(query=new_query))

def perturb_query_values(url: str, p_numeric: float = 0.7, p_longtoken: float = 0.5) -> str:
    """Replace numeric / random-looking query values with same-shape noise."""
    parsed = urlparse(url)
    params = parse_qsl(parsed.query, keep_blank_values=True)

    new_params = []
    for k, v in params:
        if v and v.isdigit() and random.random() < p_numeric:
            new_params.append((k, _random_digits_same_len(v)))
        elif v and (UUID_RE.match(v) or HEX_RE.match(v) or B64ISH_RE.match(v)) and random.random() < p_longtoken:
            # Replace with same "class" and length
            if UUID_RE.match(v):
                # Keep UUID shape
                repl = f"{_random_urlsafe(8)}-{_random_urlsafe(4)}-{_random_urlsafe(4)}-{_random_urlsafe(4)}-{_random_urlsafe(12)}"
                new_params.append((k, repl))
            else:
                new_params.append((k, _random_urlsafe(len(v))))
        else:
            new_params.append((k, v))

    new_query = urlencode(new_params, doseq=True)
    return urlunparse(parsed._replace(query=new_query))

def random_subdomain_injection(url: str, token_len: int = 6) -> str:
    """Add random subdomain prefix (keeps eTLD+1 intact)."""
    parsed = urlparse(url)
    if not parsed.netloc:
        return url

    netloc = parsed.netloc
    # remove userinfo if any
    userinfo = ""
    hostport = netloc
    if "@" in hostport:
        userinfo, hostport = hostport.rsplit("@", 1)

    host = hostport
    port = ""
    if ":" in hostport:
        host, port = hostport.rsplit(":", 1)

    parts = host.split(".")
    if len(parts) >= 2:
        token = "".join(random.choices("abcdefghijklmnopqrstuvwxyz", k=token_len))
        parts.insert(0, token)
        host = ".".join(parts)

    new_netloc = (userinfo + "@" if userinfo else "") + host + (f":{port}" if port else "")
    return urlunparse(parsed._replace(netloc=new_netloc))

def path_replace_random_segments(url: str, p_replace: float = 0.8) -> str:
    """Replace random-looking path segments (IDs) with same-length noise."""
    parsed = urlparse(url)
    path = parsed.path or ""
    if not path or path == "/":
        return url

    segments = path.split("/")
    new_segments = []
    for seg in segments:
        if not seg:
            new_segments.append(seg)
            continue

        if random.random() < p_replace and (UUID_RE.match(seg) or HEX_RE.match(seg) or B64ISH_RE.match(seg)):
            if UUID_RE.match(seg):
                seg = f"{_random_urlsafe(8)}-{_random_urlsafe(4)}-{_random_urlsafe(4)}-{_random_urlsafe(4)}-{_random_urlsafe(12)}"
            else:
                seg = _random_urlsafe(len(seg))
        new_segments.append(seg)

    new_path = "/".join(new_segments)
    return urlunparse(parsed._replace(path=new_path))

def path_separator_jitter(url: str, p: float = 0.4) -> str:
    """Swap separators inside tokens: - <-> _ <-> . (not across path boundaries)."""
    parsed = urlparse(url)
    path = parsed.path or ""
    if not path:
        return url

    def jitter_token(tok: str) -> str:
        if not tok:
            return tok
        repl = tok
        # Only change a little bit
        if random.random() < p:
            # pick one separator change
            if "-" in repl and random.random() < 0.5:
                repl = repl.replace("-", "_", 1)
            elif "_" in repl and random.random() < 0.5:
                repl = repl.replace("_", "-", 1)
            elif "." in repl and random.random() < 0.5:
                repl = repl.replace(".", "-", 1)
        return repl

    segments = path.split("/")
    segments = [jitter_token(seg) for seg in segments]
    return urlunparse(parsed._replace(path="/".join(segments)))

def homoglyph_replace_light(s: str, p: float = 0.05) -> str:
    """Very light homoglyph simulation (keep low!)."""
    mapping = {"o": "0", "l": "1", "i": "1", "e": "3", "a": "4"}
    out = []
    for c in s:
        if c in mapping and random.random() < p:
            out.append(mapping[c])
        else:
            out.append(c)
    return "".join(out)


# -----------------------------
# Tier C: Rare "hard" transforms (stress-test)
# -----------------------------

def maybe_drop_query(url: str) -> str:
    parsed = urlparse(url)
    return urlunparse(parsed._replace(query=""))

def maybe_token_reorder_in_query(url: str) -> str:
    """Rare: permute query param list (beyond simple shuffle) — same as shuffle, kept for clarity."""
    return shuffle_query_params(url)


# -----------------------------
# Main augmentation pipeline
# -----------------------------

def augment_url(url: str) -> str:
    """
    Aggressive-but-sane URL augmentation:
      - Tier A always
      - Multiple Tier B ops (query-heavy)
      - Rare Tier C ops
    """
    aug = _canonicalize_url(url)

    # Tier B: apply 3..6 ops; bias towards query + random-looking segments
    ops = []

    # Always include some digit jitter with decent probability
    if random.random() < 0.60:
        ops.append(lambda u: random_digit_replace(u, p=0.40))

    # Query-focused ops
    if random.random() < 0.80:
        ops.append(shuffle_query_params)
    if random.random() < 0.70:
        ops.append(perturb_query_values)
    if random.random() < 0.40:
        ops.append(maybe_add_tracking_params)

    # Host/subdomain
    if random.random() < 0.35:
        ops.append(random_subdomain_injection)

    # Path ops
    if random.random() < 0.70:
        ops.append(path_replace_random_segments)
    if random.random() < 0.35:
        ops.append(path_separator_jitter)

    # Light homoglyph (very low)
    if random.random() < 0.10:
        ops.append(lambda u: homoglyph_replace_light(u, p=0.04))

    # Choose 3..6 ops from the pool without exploding
    # (If pool is small, it will just use what's available.)
    random.shuffle(ops)
    k = min(len(ops), random.randint(3, 6))
    for op in ops[:k]:
        aug = op(aug)

    # Tier C: rare stress tests (keep low!)
    if random.random() < 0.05:
        aug = maybe_drop_query(aug)
    if random.random() < 0.03:
        aug = maybe_token_reorder_in_query(aug)

    return aug
    
from torch.utils.data import Dataset

class URLDataset(Dataset):
    def __init__(self, urls):
        self.urls = urls

    def __len__(self):
        return len(self.urls)

    def __getitem__(self, idx):
        url = self.urls[idx]
        aug1 = augment_url(url)
        aug2 = augment_url(url)
        return aug1, aug2


def contrastive_loss(z1, z2, temperature=0.1):
    batch_size = z1.size(0)

    z = torch.cat([z1, z2], dim=0)
    similarity = torch.matmul(z, z.T)

    # remove self-similarity
    mask = torch.eye(2*batch_size, dtype=torch.bool).to(z.device)
    similarity = similarity.masked_fill(mask, -1e4)

    similarity /= temperature

    labels = torch.arange(batch_size).to(z.device)
    labels = torch.cat([labels + batch_size, labels])

    return F.cross_entropy(similarity, labels)

from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

# ---------- 1) Reliable split ----------
def train_val_split(urls, val_frac=0.05, seed=42, shuffle=True):
    urls = list(urls)
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(urls)
    n_val = max(1, int(len(urls) * val_frac))
    val_urls = urls[:n_val]
    train_urls = urls[n_val:]
    return train_urls, val_urls
  

# ---------- 2) Eval function (contrastive val loss) ----------
@torch.no_grad()
def evaluate(model, loader, device, temperature=0.1):
    model.eval()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        urls1, urls2 = batch

        inputs1 = tokenize(urls1)
        inputs2 = tokenize(urls2)

        # Keep IDs in-range; also ensure correct dtype
        input_ids1 = inputs1["input_ids"].to(device).long()
        mask1 = inputs1["attention_mask"].to(device)

        input_ids2 = inputs2["input_ids"].to(device).long()
        mask2 = inputs2["attention_mask"].to(device)

        # AMP not needed for eval; but it's fine to keep it consistent
        with autocast(enabled=(device.type == "cuda")):
            z1 = model(input_ids1, mask1)
            z2 = model(input_ids2, mask2)
            loss = contrastive_loss(z1, z2, temperature=temperature)

        total_loss += float(loss.item())
        n_batches += 1

    return total_loss / max(1, n_batches)


# ---------- 3) Strong CPU-only preflight (kept) ----------
def _preflight_token_ids(urls, n=256):
    sample = list(urls)[:n]
    a1 = [augment_url(u) for u in sample]
    a2 = [augment_url(u) for u in sample]
    t1 = tokenize(a1)["input_ids"]
    t2 = tokenize(a2)["input_ids"]

    print("len(tokenizer) =", len(tokenizer))
    print("tokenizer.vocab_size =", tokenizer.vocab_size)
    print("t1 dtype =", t1.dtype, "min/max =", int(t1.min()), int(t1.max()))
    print("t2 dtype =", t2.dtype, "min/max =", int(t2.min()), int(t2.max()))

    assert t1.dtype == torch.long and t2.dtype == torch.long, "input_ids must be torch.long"
    assert int(t1.min()) >= 0 and int(t2.min()) >= 0, "negative token id found"
    assert int(t1.max()) < len(tokenizer) and int(t2.max()) < len(tokenizer), "token id out of range"



# ---------- 4) Train with val split + best-only saving ----------
def train(model,
          urls,
          config,
          epochs=5,
          batch_size=128,
          lr=3e-4,
          save_dir="checkpoints",
          resume_checkpoint=None,
          val_frac=0.05,
          seed=42,
          temperature=0.1):

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(dev)

    # ✅ Assert model/tokenizer vocab match
    assert model.encoder.config.vocab_size == len(tokenizer), (
        f"Model vocab_size={model.encoder.config.vocab_size} but len(tokenizer)={len(tokenizer)}. "
        "Use vocab_size=len(tokenizer) when building the model."
    )

    # ✅ Split
    train_urls, val_urls = train_val_split(urls, val_frac=val_frac, seed=seed, shuffle=True)
    print(f"Split: train={len(train_urls):,}, val={len(val_urls):,} (val_frac={val_frac})")

    # ✅ Preflight on train split (and optionally val)
    _preflight_token_ids(train_urls, n=min(256, len(train_urls)))
    _preflight_token_ids(val_urls, n=min(256, len(val_urls)))

    train_ds = URLDataset(train_urls)
    val_ds = URLDataset(val_urls)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scaler = GradScaler(enabled=(dev.type == "cuda"))

    start_epoch = 0
    best_val = float("inf")

    # ✅ Resume if checkpoint provided
    if resume_checkpoint is not None:
        checkpoint = torch.load(resume_checkpoint, map_location=dev)
        model.load_state_dict(checkpoint["model_state_dict"])
        if checkpoint.get("optimizer_state_dict") is not None:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_val = checkpoint.get("best_val_loss", float("inf"))
        print(f"Resuming from epoch {start_epoch} (best_val={best_val:.4f})")

    for epoch in range(start_epoch, epochs):

        model.train()
        total_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Train epoch {epoch}"):

            urls1, urls2 = batch

            inputs1 = tokenize(urls1)
            inputs2 = tokenize(urls2)

            # Optional extra guard (cheap)
            max1 = int(inputs1["input_ids"].max())
            max2 = int(inputs2["input_ids"].max())
            if max1 >= len(tokenizer) or max2 >= len(tokenizer):
                raise ValueError(f"Out-of-range token id in batch: max1={max1}, max2={max2}, len(tokenizer)={len(tokenizer)}")

            input_ids1 = inputs1["input_ids"].to(dev).long()
            mask1 = inputs1["attention_mask"].to(dev)

            input_ids2 = inputs2["input_ids"].to(dev).long()
            mask2 = inputs2["attention_mask"].to(dev)

            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=(dev.type == "cuda")):
                z1 = model(input_ids1, mask1)
                z2 = model(input_ids2, mask2)
                loss = contrastive_loss(z1, z2, temperature=temperature)

            if dev.type == "cuda":
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            total_loss += float(loss.item())

        train_loss = total_loss / max(1, len(train_loader))

        # ✅ Validation loss
        val_loss = evaluate(model, val_loader, dev, temperature=temperature)

        print(f"Epoch {epoch}: train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

        # ✅ Best-only saving by validation loss
        if val_loss < best_val:
            best_val = val_loss
            save_model(
                model,
                optimizer,
                epoch,
                {**config, "val_frac": val_frac, "seed": seed, "temperature": temperature},
                path=f"{save_dir}/best_model.pt"
            )
            # Store best val loss in the checkpoint too (useful if resuming)
            # (We just re-save with an extra field by patching the file)
            ckpt = torch.load(f"{save_dir}/best_model.pt", map_location="cpu")
            ckpt["best_val_loss"] = best_val
            torch.save(ckpt, f"{save_dir}/best_model.pt")
            print(f"✅ New best val_loss={best_val:.4f} (saved best_model.pt)")

    print(f"Done. Best val_loss={best_val:.4f}. Best checkpoint: {save_dir}/best_model.pt")

