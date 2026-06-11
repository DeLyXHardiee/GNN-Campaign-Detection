"""
Cache full body Jaccard generator outputs (candidate pair tables).

Email-level token/char4 sets are cached in ``body_similarity_cache``; this module caches
the expensive **pair mining** result (prior pool + inverted index) so reruns with the
same MISP/email universe, generator settings, and evaluation pools skip hours of char4
bucket scanning.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any

import pandas as pd

from seed_candidate_workflow.utils.body_similarity_cache import (
    BODY_SIMILARITY_CACHE_VERSION,
    build_body_similarity_content_fingerprint,
    body_cache_content_matches,
)

logger = logging.getLogger(__name__)

BODY_JACCARD_GENERATOR_CACHE_VERSION = "body_jaccard_gen_v1"
DEFAULT_GENERATOR_CACHE_ROOT = Path(
    "seed_candidate_workflow/output/cache/body_jaccard_generators"
)


def _pair_set_hash(pairs: set[tuple[str, str]] | None) -> str:
    if not pairs:
        return hashlib.sha256(b"").hexdigest()[:16]
    lines = "\n".join(f"{a}\t{b}" for a, b in sorted(pairs))
    return hashlib.sha256(lines.encode("utf-8")).hexdigest()[:16]


def generator_output_manifest(
    *,
    email_content_fp: dict[str, Any],
    generator_name: str,
    mode: str,
    min_jaccard: float,
    max_candidate_rows: int,
    use_filtered_inverted_index: bool,
    max_token_document_frequency: int,
    max_char4gram_document_frequency: int,
    prior_pair_pool_hash: str,
    semantic_band_pool_hash: str,
    cache_salt: str | None = None,
) -> dict[str, Any]:
    m: dict[str, Any] = {
        "cache_version": BODY_JACCARD_GENERATOR_CACHE_VERSION,
        "email_cache_version": BODY_SIMILARITY_CACHE_VERSION,
        "email_content": dict(email_content_fp),
        "generator_name": str(generator_name),
        "mode": str(mode),
        "min_jaccard": float(min_jaccard),
        "max_candidate_rows": int(max_candidate_rows),
        "use_filtered_inverted_index": bool(use_filtered_inverted_index),
        "max_token_document_frequency": int(max_token_document_frequency),
        "max_char4gram_document_frequency": int(max_char4gram_document_frequency),
        "prior_pair_pool_hash": str(prior_pair_pool_hash),
        "semantic_band_pool_hash": str(semantic_band_pool_hash),
        "n_prior_pairs": None,
        "n_semantic_pairs": None,
    }
    s = (cache_salt or "").strip()
    if s:
        m["cache_salt"] = s
    return m


def _manifest_identity(manifest: dict[str, Any]) -> dict[str, Any]:
    """Fields that must match for a generator-output cache hit."""
    return {
        "cache_version": manifest.get("cache_version"),
        "email_cache_version": manifest.get("email_cache_version"),
        "email_content": manifest.get("email_content"),
        "generator_name": manifest.get("generator_name"),
        "mode": manifest.get("mode"),
        "min_jaccard": manifest.get("min_jaccard"),
        "max_candidate_rows": manifest.get("max_candidate_rows"),
        "use_filtered_inverted_index": manifest.get("use_filtered_inverted_index"),
        "max_token_document_frequency": manifest.get("max_token_document_frequency"),
        "max_char4gram_document_frequency": manifest.get("max_char4gram_document_frequency"),
        "prior_pair_pool_hash": manifest.get("prior_pair_pool_hash"),
        "semantic_band_pool_hash": manifest.get("semantic_band_pool_hash"),
        "cache_salt": manifest.get("cache_salt"),
    }


def generator_manifest_matches(loaded: dict[str, Any], *, expected: dict[str, Any]) -> bool:
    if str(loaded.get("cache_version")) != str(expected.get("cache_version")):
        return False
    exp_email = expected.get("email_content") or {}
    got_email = loaded.get("email_content") or {}
    if not body_cache_content_matches(got_email, expected_content=exp_email):
        return False
    for k, v in _manifest_identity(expected).items():
        if k == "email_content":
            continue
        if loaded.get(k) != v:
            return False
    return True


def cache_dir_for_generator_manifest(
    *,
    cache_root: Path,
    email_content_fp: dict[str, Any],
    generator_manifest: dict[str, Any],
) -> Path:
    email_key = hashlib.sha256(
        json.dumps(email_content_fp, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:24]
    gen_key = hashlib.sha256(
        json.dumps(_manifest_identity(generator_manifest), sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()[:20]
    gen_name = str(generator_manifest.get("generator_name") or "unknown")
    return (cache_root / "by_content" / email_key / gen_name / gen_key).resolve()


def try_load_cached_generator_output(
    cache_dir: Path,
    *,
    expected_manifest: dict[str, Any],
) -> tuple[pd.DataFrame | None, dict[str, Any]]:
    diag: dict[str, Any] = {"cache_dir": str(cache_dir)}
    manifest_path = cache_dir / "manifest.json"
    csv_path = cache_dir / "candidates.csv"
    if not manifest_path.is_file() or not csv_path.is_file():
        diag["cache_status"] = "miss_missing_files"
        return None, diag
    try:
        loaded_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not generator_manifest_matches(loaded_manifest, expected=expected_manifest):
            diag["cache_status"] = "miss_stale_manifest"
            return None, diag
        t0 = time.perf_counter()
        df = pd.read_csv(csv_path)
        load_s = time.perf_counter() - t0
        diag.update(
            {
                "cache_status": "hit",
                "cache_load_seconds": float(load_s),
                "n_rows_loaded": int(len(df)),
                "loaded_manifest": loaded_manifest,
            }
        )
        return df, diag
    except Exception as exc:
        diag["cache_status"] = f"miss_load_error:{type(exc).__name__}"
        diag["error"] = str(exc)
        return None, diag


def save_cached_generator_output(
    df: pd.DataFrame,
    *,
    cache_dir: Path,
    manifest: dict[str, Any],
) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    csv_path = cache_dir / "candidates.csv"
    manifest_path = cache_dir / "manifest.json"
    df.to_csv(csv_path, index=False)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return csv_path


def resolve_generator_cache_root(generator_cfg: dict[str, Any]) -> Path:
    raw = generator_cfg.get("body_jaccard_generator_cache_root")
    if raw:
        return Path(str(raw)).expanduser().resolve()
    return DEFAULT_GENERATOR_CACHE_ROOT.resolve()


def use_generator_output_cache_enabled(generator_cfg: dict[str, Any]) -> bool:
    if "use_body_jaccard_generator_cache" in generator_cfg:
        return bool(generator_cfg.get("use_body_jaccard_generator_cache"))
    return bool(generator_cfg.get("use_body_similarity_cache", True))


def try_load_body_jaccard_generator_from_cache(
    *,
    nodes_df: pd.DataFrame,
    generator_cfg: dict[str, Any],
    project_root: Path,
    graph_id: str,
    misp_json_path: Path,
    generator_name: str,
    mode: str,
    min_jaccard: float,
    prior_pair_pool: set[tuple[str, str]] | None,
    semantic_band_pool: set[tuple[str, str]] | None,
    force_rebuild: bool,
) -> tuple[pd.DataFrame | None, dict[str, Any]]:
    """Return cached candidate DataFrame when manifest matches; else ``(None, diag)``."""
    if force_rebuild or not use_generator_output_cache_enabled(generator_cfg):
        return None, {"cache_status": "disabled"}

    email_ids = sorted({str(x) for x in nodes_df["external_id"].astype(str).tolist()})
    email_fp = build_body_similarity_content_fingerprint(
        misp_json_path=misp_json_path,
        email_ids=email_ids,
        min_token_len=2,
        char_n=4,
        cache_salt=generator_cfg.get("body_similarity_cache_salt"),
    )
    pool_cfg = {
        "use_filtered_inverted_index": bool(
            generator_cfg.get("use_filtered_inverted_index", True)
        ),
        "max_token_document_frequency": int(
            generator_cfg.get("max_token_document_frequency", 40)
        ),
        "max_char4gram_document_frequency": int(
            generator_cfg.get("max_char4gram_document_frequency", 60)
        ),
    }
    prior_h = _pair_set_hash(prior_pair_pool)
    sem_h = _pair_set_hash(semantic_band_pool)
    manifest = generator_output_manifest(
        email_content_fp=email_fp,
        generator_name=generator_name,
        mode=mode,
        min_jaccard=min_jaccard,
        max_candidate_rows=int(generator_cfg.get("max_candidate_rows", 500_000)),
        prior_pair_pool_hash=prior_h,
        semantic_band_pool_hash=sem_h,
        cache_salt=generator_cfg.get("body_similarity_cache_salt"),
        **pool_cfg,
    )
    manifest["n_prior_pairs"] = int(len(prior_pair_pool or set()))
    manifest["n_semantic_pairs"] = int(len(semantic_band_pool or set()))
    manifest["graph_id"] = str(graph_id)

    cache_root = resolve_generator_cache_root(generator_cfg)
    cache_dir = cache_dir_for_generator_manifest(
        cache_root=cache_root,
        email_content_fp=email_fp,
        generator_manifest=manifest,
    )
    df, diag = try_load_cached_generator_output(cache_dir, expected_manifest=manifest)
    diag["expected_manifest"] = manifest
    if df is not None:
        logger.info(
            "body_jaccard_generator_cache hit name=%s mode=%s dir=%s n_rows=%d load_s=%.2f",
            generator_name,
            mode,
            cache_dir,
            len(df),
            float(diag.get("cache_load_seconds") or 0.0),
        )
    return df, diag


def save_body_jaccard_generator_to_cache(
    df: pd.DataFrame,
    *,
    nodes_df: pd.DataFrame,
    generator_cfg: dict[str, Any],
    misp_json_path: Path,
    graph_id: str,
    generator_name: str,
    mode: str,
    min_jaccard: float,
    prior_pair_pool: set[tuple[str, str]] | None,
    semantic_band_pool: set[tuple[str, str]] | None,
) -> dict[str, Any]:
    if not use_generator_output_cache_enabled(generator_cfg):
        return {"cache_status": "save_skipped_disabled"}

    email_ids = sorted({str(x) for x in nodes_df["external_id"].astype(str).tolist()})
    email_fp = build_body_similarity_content_fingerprint(
        misp_json_path=misp_json_path,
        email_ids=email_ids,
        min_token_len=2,
        char_n=4,
        cache_salt=generator_cfg.get("body_similarity_cache_salt"),
    )
    prior_h = _pair_set_hash(prior_pair_pool)
    sem_h = _pair_set_hash(semantic_band_pool)
    manifest = generator_output_manifest(
        email_content_fp=email_fp,
        generator_name=generator_name,
        mode=mode,
        min_jaccard=min_jaccard,
        max_candidate_rows=int(generator_cfg.get("max_candidate_rows", 500_000)),
        use_filtered_inverted_index=bool(generator_cfg.get("use_filtered_inverted_index", True)),
        max_token_document_frequency=int(generator_cfg.get("max_token_document_frequency", 40)),
        max_char4gram_document_frequency=int(
            generator_cfg.get("max_char4gram_document_frequency", 60)
        ),
        prior_pair_pool_hash=prior_h,
        semantic_band_pool_hash=sem_h,
        cache_salt=generator_cfg.get("body_similarity_cache_salt"),
    )
    manifest["n_prior_pairs"] = int(len(prior_pair_pool or set()))
    manifest["n_semantic_pairs"] = int(len(semantic_band_pool or set()))
    manifest["graph_id"] = str(graph_id)

    cache_root = resolve_generator_cache_root(generator_cfg)
    cache_dir = cache_dir_for_generator_manifest(
        cache_root=cache_root,
        email_content_fp=email_fp,
        generator_manifest=manifest,
    )
    t0 = time.perf_counter()
    path = save_cached_generator_output(df, cache_dir=cache_dir, manifest=manifest)
    save_s = time.perf_counter() - t0
    logger.info(
        "body_jaccard_generator_cache saved name=%s mode=%s path=%s n_rows=%d save_s=%.2f",
        generator_name,
        mode,
        path,
        len(df),
        save_s,
    )
    return {
        "cache_status": "saved",
        "cache_dir": str(cache_dir),
        "cache_csv_path": str(path),
        "cache_save_seconds": float(save_s),
    }
