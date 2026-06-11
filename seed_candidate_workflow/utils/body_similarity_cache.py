"""
Persistent per-email body token / char-4gram caches for candidate generation and analysis.

Semantics match ``pair_similarity_features`` (normalization, tokenization, char-4grams).

Layout (v2): caches are keyed by **content** (MISP fingerprint + sorted email-id set +
tokenization params), not by ``graph_id``, so multiple graph-bundle ids that share the
same hetero email universe reuse one on-disk store. Legacy v1 caches under
``<cache_root>/<graph_id>/<hash>/`` are still discoverable via scan when the v2 path misses.
"""

from __future__ import annotations

import hashlib
import json
import logging
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable

if TYPE_CHECKING:
    from seed_candidate_workflow.utils.body_similarity_progress import BodySimilarityProgress

from seed_candidate_workflow.utils.pair_similarity_features import (
    char_ngrams_text,
    jaccard_similarity,
    tokenize_text,
)

logger = logging.getLogger(__name__)

BODY_SIMILARITY_CACHE_VERSION = "body_sim_v2"
LEGACY_BODY_SIMILARITY_CACHE_VERSIONS = frozenset({"body_sim_v1", "body_sim_v2"})
DEFAULT_CACHE_ROOT = Path("seed_candidate_workflow/output/cache/body_similarity")

_MAX_LEGACY_CACHE_DIRS_TO_SCAN = 800


def normalize_body_text(body: str) -> str:
    """Whitespace-collapsed lowercase body (char-ngram input normalization)."""
    import re

    return re.sub(r"\s+", " ", str(body or "").lower()).strip()


def body_token_set_from_normalized(normalized_body: str, *, min_len: int = 2) -> frozenset[str]:
    return frozenset(tokenize_text(normalized_body, min_len=min_len))


def body_char4_set_from_normalized(normalized_body: str, *, n: int = 4) -> frozenset[str]:
    return frozenset(char_ngrams_text(normalized_body, n))


@dataclass
class EmailBodyFeatureStore:
    """Per-email precomputed body representations (immutable sets for safe sharing)."""

    token_sets: dict[str, frozenset[str]] = field(default_factory=dict)
    char4_sets: dict[str, frozenset[str]] = field(default_factory=dict)
    normalized_bodies: dict[str, str] = field(default_factory=dict)

    def token_jaccard(self, email_i: str, email_j: str) -> float:
        return jaccard_similarity(
            self.token_sets.get(email_i) or frozenset(),
            self.token_sets.get(email_j) or frozenset(),
        )

    def char4_jaccard(self, email_i: str, email_j: str) -> float:
        return jaccard_similarity(
            self.char4_sets.get(email_i) or frozenset(),
            self.char4_sets.get(email_j) or frozenset(),
        )

    def token_jaccard_if_above(
        self,
        email_i: str,
        email_j: str,
        *,
        min_jaccard: float,
    ) -> float | None:
        """Return token Jaccard when >= min_jaccard, else None (size-bound prune)."""
        a = self.token_sets.get(email_i) or frozenset()
        b = self.token_sets.get(email_j) or frozenset()
        if not _jaccard_may_reach_threshold(len(a), len(b), min_jaccard):
            return None
        j = jaccard_similarity(a, b)
        return j if j >= min_jaccard else None

    def char4_jaccard_if_above(
        self,
        email_i: str,
        email_j: str,
        *,
        min_jaccard: float,
    ) -> float | None:
        """Return char-4 Jaccard when >= min_jaccard, else None (size-bound prune)."""
        a = self.char4_sets.get(email_i) or frozenset()
        b = self.char4_sets.get(email_j) or frozenset()
        if not _jaccard_may_reach_threshold(len(a), len(b), min_jaccard):
            return None
        j = jaccard_similarity(a, b)
        return j if j >= min_jaccard else None


def _jaccard_may_reach_threshold(len_a: int, len_b: int, min_jaccard: float) -> bool:
    """True if Jaccard could still be >= min_jaccard given set sizes only."""
    if len_a == 0 and len_b == 0:
        return min_jaccard <= 0.0
    if len_a == 0 or len_b == 0:
        return False
    upper = min(len_a, len_b) / max(len_a, len_b)
    return upper >= min_jaccard


def _file_fingerprint(path: Path) -> dict[str, Any]:
    p = path.resolve()
    if not p.is_file():
        return {"path": str(p), "exists": False}
    st = p.stat()
    return {
        "path": str(p),
        "exists": True,
        "size_bytes": int(st.st_size),
        "mtime_ns": int(st.st_mtime_ns),
    }


def build_body_similarity_content_fingerprint(
    *,
    misp_json_path: Path,
    email_ids: Iterable[str],
    min_token_len: int,
    char_n: int,
    cache_salt: str | None = None,
) -> dict[str, Any]:
    """
    Fingerprint for cache identity (no graph_id).

    Optional ``cache_salt`` splits the namespace when you intentionally want a separate
    cache for the same MISP file + email set (advanced; default None).
    """
    ids = sorted({str(x) for x in email_ids})
    misp_path = Path(misp_json_path).resolve()
    fp: dict[str, Any] = {
        "cache_version": BODY_SIMILARITY_CACHE_VERSION,
        "misp_source": _file_fingerprint(misp_path),
        "n_emails": int(len(ids)),
        "email_id_hash": hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()[:16],
        "min_token_len": int(min_token_len),
        "char_ngram_n": int(char_n),
    }
    s = (cache_salt or "").strip()
    if s:
        fp["cache_salt"] = s
    return fp


def cache_manifest_payload(
    *,
    graph_id: str,
    misp_json_path: Path,
    email_ids: Iterable[str],
    min_token_len: int,
    char_n: int,
    cache_salt: str | None = None,
) -> dict[str, Any]:
    """Full manifest written to disk (includes ``graph_id`` for provenance; not used in v2 path)."""
    fp = build_body_similarity_content_fingerprint(
        misp_json_path=misp_json_path,
        email_ids=email_ids,
        min_token_len=min_token_len,
        char_n=char_n,
        cache_salt=cache_salt,
    )
    return {**fp, "graph_id": str(graph_id)}


def cache_dir_for_content_fingerprint(*, cache_root: Path, content_fp: dict[str, Any]) -> Path:
    """Stable directory for v2 content fingerprint (under ``by_content/``)."""
    key = hashlib.sha256(
        json.dumps(content_fp, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:24]
    return (cache_root / "by_content" / key).resolve()


def _misp_fingerprint_match(got: dict[str, Any], exp: dict[str, Any]) -> bool:
    for k in ("path", "size_bytes", "mtime_ns"):
        if (got or {}).get(k) != (exp or {}).get(k):
            return False
    if not bool((got or {}).get("exists")) or not bool((exp or {}).get("exists")):
        return False
    return True


def body_cache_content_matches(manifest: dict[str, Any], *, expected_content: dict[str, Any]) -> bool:
    """True when ``manifest`` matches ``expected_content`` on all body-cache identity fields."""
    if str(manifest.get("cache_version") or "") not in LEGACY_BODY_SIMILARITY_CACHE_VERSIONS:
        return False
    if int(manifest.get("min_token_len", -1)) != int(expected_content.get("min_token_len", -2)):
        return False
    if int(manifest.get("char_ngram_n", -1)) != int(expected_content.get("char_ngram_n", -2)):
        return False
    if str(manifest.get("email_id_hash") or "") != str(expected_content.get("email_id_hash") or ""):
        return False
    if str(manifest.get("cache_salt") or "") != str(expected_content.get("cache_salt") or ""):
        return False
    return _misp_fingerprint_match(
        manifest.get("misp_source") or {},
        expected_content.get("misp_source") or {},
    )


def manifest_is_fresh(manifest: dict[str, Any], *, expected: dict[str, Any]) -> bool:
    """
    Backward-compatible name: true when loaded manifest matches ``expected`` on cache identity.

    ``expected`` may be a full manifest (with ``graph_id``); comparison ignores ``graph_id``.
    """
    exp_fp = {k: v for k, v in expected.items() if k != "graph_id"}
    return body_cache_content_matches(manifest, expected_content=exp_fp)


def _iter_legacy_body_cache_leaf_dirs(cache_root: Path) -> Iterable[Path]:
    root = Path(cache_root).resolve()
    if not root.is_dir():
        return
    n = 0
    by_content = root / "by_content"
    if by_content.is_dir():
        for hdir in sorted(by_content.iterdir()):
            if not hdir.is_dir():
                continue
            if (hdir / "manifest.json").is_file():
                yield hdir
                n += 1
                if n >= _MAX_LEGACY_CACHE_DIRS_TO_SCAN:
                    return
    for gid_dir in sorted(root.iterdir()):
        if not gid_dir.is_dir() or gid_dir.name == "by_content":
            continue
        for hdir in sorted(gid_dir.iterdir()):
            if not hdir.is_dir():
                continue
            if (hdir / "manifest.json").is_file():
                yield hdir
                n += 1
                if n >= _MAX_LEGACY_CACHE_DIRS_TO_SCAN:
                    return


def _find_legacy_compatible_cache_dir(
    cache_root: Path,
    *,
    expected_content: dict[str, Any],
    exclude: Path | None = None,
) -> Path | None:
    excl = exclude.resolve() if exclude is not None else None
    for leaf in _iter_legacy_body_cache_leaf_dirs(cache_root):
        try:
            if excl is not None and leaf.resolve() == excl:
                continue
            mp = leaf / "manifest.json"
            loaded = json.loads(mp.read_text(encoding="utf-8"))
            if body_cache_content_matches(loaded, expected_content=expected_content):
                return leaf
        except Exception:
            continue
    return None


def build_email_body_feature_store(
    *,
    email_ids: Iterable[str],
    text_catalog: dict[str, dict[str, str]],
    min_token_len: int = 2,
    char_n: int = 4,
    store_normalized_body: bool = False,
    progress: BodySimilarityProgress | None = None,
) -> tuple[EmailBodyFeatureStore, dict[str, Any]]:
    """Precompute token and char-4gram sets once per email."""
    t0 = time.perf_counter()
    store = EmailBodyFeatureStore()
    n_empty = 0
    id_list = [str(x) for x in email_ids]
    if progress is not None:
        progress.phase_start("precompute email body features", detail=f"{len(id_list):,} emails")
        progress.loop_start("precompute emails", len(id_list))
    tick_every = max(1, len(id_list) // 20) if id_list else 1
    since_tick = 0
    for eid in id_list:
        since_tick += 1
        body = str((text_catalog.get(eid) or {}).get("body") or "")
        norm = normalize_body_text(body)
        if store_normalized_body:
            store.normalized_bodies[eid] = norm
        store.token_sets[eid] = body_token_set_from_normalized(norm, min_len=min_token_len)
        store.char4_sets[eid] = body_char4_set_from_normalized(norm, n=char_n)
        if not store.token_sets[eid] and not store.char4_sets[eid]:
            n_empty += 1
        if progress is not None and since_tick >= tick_every:
            progress.loop_tick(since_tick)
            since_tick = 0
    if progress is not None:
        if since_tick:
            progress.loop_tick(since_tick)
        progress.phase_done(
            "precompute email body features",
            n_emails=len(store.token_sets),
            empty=n_empty,
        )
    elapsed = time.perf_counter() - t0
    meta = {
        "status": "built",
        "n_emails": int(len(store.token_sets)),
        "n_emails_empty_body_features": int(n_empty),
        "preprocess_seconds": float(elapsed),
        "min_token_len": int(min_token_len),
        "char_ngram_n": int(char_n),
    }
    return store, meta


def save_email_body_feature_store(
    store: EmailBodyFeatureStore,
    *,
    cache_dir: Path,
    manifest: dict[str, Any],
) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    pkl_path = cache_dir / "email_body_features.pkl"
    manifest_path = cache_dir / "manifest.json"
    with open(pkl_path, "wb") as f:
        pickle.dump(
            {
                "token_sets": store.token_sets,
                "char4_sets": store.char4_sets,
                "normalized_bodies": store.normalized_bodies,
            },
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return pkl_path


def load_email_body_feature_store(cache_dir: Path) -> tuple[EmailBodyFeatureStore, dict[str, Any]]:
    manifest_path = cache_dir / "manifest.json"
    pkl_path = cache_dir / "email_body_features.pkl"
    if not manifest_path.is_file() or not pkl_path.is_file():
        raise FileNotFoundError(f"Body similarity cache incomplete under {cache_dir}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    with open(pkl_path, "rb") as f:
        blob = pickle.load(f)
    store = EmailBodyFeatureStore(
        token_sets=dict(blob.get("token_sets") or {}),
        char4_sets=dict(blob.get("char4_sets") or {}),
        normalized_bodies=dict(blob.get("normalized_bodies") or {}),
    )
    return store, manifest


def build_or_load_email_body_feature_store(
    *,
    email_ids: Iterable[str],
    text_catalog: dict[str, dict[str, str]],
    graph_id: str,
    misp_json_path: Path,
    cache_root: Path | None = None,
    min_token_len: int = 2,
    char_n: int = 4,
    force_rebuild: bool = False,
    store_normalized_body: bool = False,
    progress: BodySimilarityProgress | None = None,
    cache_salt: str | None = None,
) -> tuple[EmailBodyFeatureStore, dict[str, Any]]:
    """
    Load persistent cache when fresh; otherwise build, save, and return store.

    v2 stores under ``<cache_root>/by_content/<hash>/``. Older v1/v2 layouts under
    ``<cache_root>/<graph_id>/...`` are still discovered when the primary path misses.
    """
    ids = sorted({str(x) for x in email_ids})
    root = (cache_root or DEFAULT_CACHE_ROOT).resolve()
    misp_path = Path(misp_json_path).resolve()
    content_fp = build_body_similarity_content_fingerprint(
        misp_json_path=misp_path,
        email_ids=ids,
        min_token_len=min_token_len,
        char_n=char_n,
        cache_salt=cache_salt,
    )
    full_manifest = {**content_fp, "graph_id": str(graph_id)}
    primary_dir = cache_dir_for_content_fingerprint(cache_root=root, content_fp=content_fp)
    diag: dict[str, Any] = {
        "cache_root": str(root),
        "cache_dir": str(primary_dir),
        "content_fingerprint": content_fp,
        "expected_manifest": full_manifest,
        "force_rebuild": bool(force_rebuild),
    }

    def _try_dir(cache_dir: Path, *, status_hit: str) -> tuple[EmailBodyFeatureStore, dict[str, Any]] | None:
        if not cache_dir.is_dir():
            return None
        try:
            t0 = time.perf_counter()
            store, loaded_manifest = load_email_body_feature_store(cache_dir)
            load_s = time.perf_counter() - t0
            if not body_cache_content_matches(loaded_manifest, expected_content=content_fp):
                diag["cache_status"] = "stale_manifest"
                return None
            diag.update(
                {
                    "cache_status": status_hit,
                    "cache_load_seconds": float(load_s),
                    "loaded_manifest": loaded_manifest,
                    "n_emails_loaded": int(len(store.token_sets)),
                    "cache_dir_resolved": str(cache_dir.resolve()),
                }
            )
            if progress is not None:
                progress.message(
                    f"cache {status_hit.upper()} loaded {len(store.token_sets):,} emails in {load_s:.1f}s"
                )
            logger.info(
                "body_similarity_cache %s graph_id=%s dir=%s load_s=%.2f n_emails=%d",
                status_hit,
                graph_id,
                cache_dir,
                load_s,
                len(store.token_sets),
            )
            return store, diag
        except Exception as exc:
            diag["cache_status"] = f"load_failed:{type(exc).__name__}"
            return None

    if not force_rebuild:
        hit = _try_dir(primary_dir, status_hit="hit")
        if hit is not None:
            return hit
        legacy_dir = _find_legacy_compatible_cache_dir(
            root, expected_content=content_fp, exclude=primary_dir
        )
        if legacy_dir is not None:
            hit2 = _try_dir(legacy_dir, status_hit="hit_legacy")
            if hit2 is not None:
                return hit2

    if progress is not None:
        progress.message("cache MISS — building email body feature store")
    store, build_meta = build_email_body_feature_store(
        email_ids=ids,
        text_catalog=text_catalog,
        min_token_len=min_token_len,
        char_n=char_n,
        store_normalized_body=store_normalized_body,
        progress=progress,
    )
    t_save = time.perf_counter()
    pkl_path = save_email_body_feature_store(store, cache_dir=primary_dir, manifest=full_manifest)
    save_s = time.perf_counter() - t_save
    diag.update(
        {
            "cache_status": "rebuilt",
            "cache_save_seconds": float(save_s),
            "cache_pkl_path": str(pkl_path),
            "build": build_meta,
            "cache_dir_resolved": str(primary_dir.resolve()),
        }
    )
    logger.info(
        "body_similarity_cache rebuilt graph_id=%s dir=%s preprocess_s=%.2f save_s=%.2f n_emails=%d",
        graph_id,
        primary_dir,
        float(build_meta.get("preprocess_seconds") or 0.0),
        save_s,
        len(store.token_sets),
    )
    return store, diag
