"""
Embedding component: load-or-compute subject/body SBERT embeddings per email.

Storage: output_dir/embeddings.json with by-key entries so that existing
embeddings are loaded and only missing ones are computed. Can run independently
or be used by the graph assembler.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

MODEL_NAME = "intfloat/multilingual-e5-large"
_CACHE_FILENAME = "embeddings.json"

# Default output folder for this component (utils/embeddings/output)
DEFAULT_OUTPUT_DIR: Path = Path(__file__).resolve().parent / "output"


def _email_key(em: Dict[str, Any], index: int) -> str:
    """Stable key for an email (external_id or email_index)."""
    key = (em.get("external_id") or "").strip()
    if key:
        return key
    return str(em.get("email_index", index))


def _cache_entry_has_vectors(entry: Any) -> bool:
    """True if cached subject and/or body lists are non-empty (usable for graph / clustering)."""
    if not isinstance(entry, dict):
        return False
    sj = entry.get("subj")
    bd = entry.get("body")
    sj_ok = isinstance(sj, list) and len(sj) > 0
    bd_ok = isinstance(bd, list) and len(bd) > 0
    return sj_ok or bd_ok


def _load_cache(output_dir: Path) -> Tuple[Dict[str, Dict[str, List[float]]], int, int]:
    """Load cache from output_dir/embeddings.json. Returns (by_key, subj_dim, body_dim) or (empty dict, 0, 0)."""
    path = output_dir / _CACHE_FILENAME
    if not path.exists():
        return {}, 0, 0
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return {}, 0, 0
    by_key = data.get("by_key") or {}
    if not isinstance(by_key, dict):
        return {}, 0, 0
    subj_dim = int(data.get("subj_dim") or 0)
    body_dim = int(data.get("body_dim") or 0)
    return by_key, subj_dim, body_dim


def _save_cache(
    output_dir: Path,
    by_key: Dict[str, Dict[str, List[float]]],
    subj_dim: int,
    body_dim: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / _CACHE_FILENAME
    # Ground-truth join key (same as cache key: external_id or str(email_index))
    serializable = {}
    for k, v in by_key.items():
        if isinstance(v, dict):
            e = {**v, "external_id": v.get("external_id") or k}
            serializable[k] = e
        else:
            serializable[k] = v
    payload = {
        "model": MODEL_NAME,
        "subj_dim": subj_dim,
        "body_dim": body_dim,
        "by_key": serializable,
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=0, separators=(",", ":"))


def _compute_batch(
    emails: List[Dict[str, Any]],
) -> Tuple[List[List[float]], List[List[float]], int, int]:
    """Compute SBERT subject/body embeddings for the given emails. Returns (subj_vecs, body_vecs, subj_dim, body_dim)."""
    subj_vecs: List[List[float]] = []
    body_vecs: List[List[float]] = []
    subj_dim = 0
    body_dim = 0
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore

        model = SentenceTransformer(MODEL_NAME)
        subj_corpus = [(em.get("subject") or "").strip() for em in emails]
        body_corpus = [(em.get("body") or "").strip() for em in emails]
        if any(bool(t) for t in subj_corpus):
            subj_inputs = [f"passage: {text}" for text in subj_corpus]
            subj_vec = model.encode(subj_inputs, show_progress_bar=True, convert_to_numpy=True)
            subj_dim = int(subj_vec.shape[1]) if len(subj_vec.shape) > 1 else 0
            if subj_dim > 0:
                subj_vecs = subj_vec.astype("float32").tolist()
        if any(bool(t) for t in body_corpus):
            body_inputs = [f"passage: {text}" for text in body_corpus]
            body_vec = model.encode(body_inputs, show_progress_bar=True, convert_to_numpy=True)
            body_dim = int(body_vec.shape[1]) if len(body_vec.shape) > 1 else 0
            if body_dim > 0:
                body_vecs = body_vec.astype("float32").tolist()
    except Exception:
        pass
    return subj_vecs, body_vecs, subj_dim, body_dim


def get_embeddings(
    emails: List[Dict[str, Any]],
    output_dir: Optional[str | Path] = None,
) -> Tuple[List[List[float]], List[List[float]], int, int]:
    """Load existing embeddings from the embedder output dir and compute any missing ones per email.

    Each email is identified by external_id (or email_index). Cached embeddings are loaded;
    for emails without a cache entry, embeddings are computed and then saved to the cache.
    Returns (subj_vecs, body_vecs, subj_dim, body_dim) with vectors in the same order as `emails`.

    To align with ground truth, same order as `external_ids_for_email_order(emails)`;
    each ``embeddings.json`` ``by_key`` entry includes ``external_id`` (join key).
    """
    out = Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR
    keys = [_email_key(em, i) for i, em in enumerate(emails)]
    cache, cache_subj_dim, cache_body_dim = _load_cache(out)

    subj_vecs: List[List[float]] = [None] * len(emails)  # type: ignore[list-item]
    body_vecs: List[List[float]] = [None] * len(emails)  # type: ignore[list-item]
    missing_indices: List[int] = []

    for i in tqdm(range(len(emails)), total=len(emails), desc="Checking embedding cache"):
        k = keys[i]
        entry = cache.get(k) if isinstance(cache.get(k), dict) else None
        if entry and _cache_entry_has_vectors(entry):
            subj_vecs[i] = list(entry.get("subj") or [])
            body_vecs[i] = list(entry.get("body") or [])
        else:
            missing_indices.append(i)

    subj_dim = cache_subj_dim
    body_dim = cache_body_dim

    if missing_indices:
        missing_emails = [emails[j] for j in missing_indices]
        new_subj, new_body, new_subj_dim, new_body_dim = _compute_batch(missing_emails)
        any_missing_text = any(
            str(emails[j].get("subject") or "").strip() or str(emails[j].get("body") or "").strip()
            for j in missing_indices
        )
        if any_missing_text and new_subj_dim == 0 and new_body_dim == 0:
            raise RuntimeError(
                "SBERT embedding computation returned no vectors (model import/encode failed or "
                "sentence-transformers not installed) while some emails have non-empty subject/body. "
                "Fix the environment (e.g. `pip install sentence-transformers`), then retry. "
                "The embeddings cache was not updated to avoid overwriting a good cache with empty rows."
            )
        if cache_subj_dim or cache_body_dim:
            if (new_subj_dim and new_subj_dim != cache_subj_dim) or (new_body_dim and new_body_dim != cache_body_dim):
                raise ValueError(
                    f"Computed embedding dims ({new_subj_dim}, {new_body_dim}) do not match cache ({cache_subj_dim}, {cache_body_dim}). "
                    "Clear the cache or use the same model."
                )
        else:
            subj_dim = new_subj_dim
            body_dim = new_body_dim
        for idx, pos in enumerate(missing_indices):
            entry = {
                "subj": new_subj[idx] if idx < len(new_subj) else [],
                "body": new_body[idx] if idx < len(new_body) else [],
                "external_id": keys[pos],
            }
            cache[keys[pos]] = entry
            subj_vecs[pos] = entry["subj"]
            body_vecs[pos] = entry["body"]
        _save_cache(out, cache, subj_dim, body_dim)

    # Fill any remaining None with empty lists
    for i in range(len(emails)):
        if subj_vecs[i] is None:
            subj_vecs[i] = []
        if body_vecs[i] is None:
            body_vecs[i] = []

    return subj_vecs, body_vecs, subj_dim, body_dim


def run_standalone(
    misp_events_or_path: List[dict] | str | Path,
    output_dir: Optional[str | Path] = None,
) -> Tuple[List[List[float]], List[List[float]], int, int]:
    """Run the embedding component independently: load emails from MISP, then load-or-compute embeddings.

    Args:
        misp_events_or_path: List of MISP event dicts, or path to a MISP JSON file.
        output_dir: Where to read/write the cache. Defaults to DEFAULT_OUTPUT_DIR.

    Returns:
        (subj_vecs, body_vecs, subj_dim, body_dim) in email order.
    """
    if isinstance(misp_events_or_path, (str, Path)):
        path = Path(misp_events_or_path)
        if not path.exists():
            raise FileNotFoundError(f"MISP path not found: {path}")
        with path.open("r", encoding="utf-8") as f:
            misp_events = json.load(f)
    else:
        misp_events = misp_events_or_path

    from ..common import parse_misp_events

    emails = parse_misp_events(misp_events)
    if not emails:
        return [], [], 0, 0
    return get_embeddings(emails, output_dir=output_dir)


def external_ids_for_email_order(emails: List[Dict[str, Any]]) -> List[str]:
    """Stable id per row, matching ``get_embeddings`` order and ``embeddings.json`` cache keys."""
    return [_email_key(em, i) for i, em in enumerate(emails)]
