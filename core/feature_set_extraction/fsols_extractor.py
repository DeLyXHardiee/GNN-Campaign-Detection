from __future__ import annotations

import json
from pathlib import Path


_EMBEDDINGS_CACHE: dict[str, list[float]] | None = None
_EMBEDDING_DIM: int = 0


def _safe_bool(value) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value != 0)
    if isinstance(value, str):
        return int(value.strip().lower() in {"1", "true", "yes", "y"})
    return 0


def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _load_embeddings() -> tuple[dict[str, list[float]], int]:
    global _EMBEDDINGS_CACHE
    global _EMBEDDING_DIM

    if _EMBEDDINGS_CACHE is not None:
        return _EMBEDDINGS_CACHE, _EMBEDDING_DIM

    embeddings_path = (
        Path(__file__).resolve().parents[1]
        / "utils"
        / "embeddings"
        / "output"
        / "embeddings.json"
    )

    if not embeddings_path.exists():
        _EMBEDDINGS_CACHE = {}
        _EMBEDDING_DIM = 0
        return _EMBEDDINGS_CACHE, _EMBEDDING_DIM

    with open(embeddings_path, "r", encoding="utf-8-sig") as f:
        payload = json.load(f)

    by_key = payload.get("by_key") if isinstance(payload, dict) else None
    if not isinstance(by_key, dict):
        _EMBEDDINGS_CACHE = {}
        _EMBEDDING_DIM = 0
        return _EMBEDDINGS_CACHE, _EMBEDDING_DIM

    cache: dict[str, list[float]] = {}
    embedding_dim = 0

    for key, entry in by_key.items():
        if not isinstance(entry, dict):
            continue

        ext = entry.get("external_id") or key
        external_id = str(ext).strip()
        if not external_id:
            continue

        subj = entry.get("subj") or []
        body = entry.get("body") or []
        if not isinstance(subj, list) or not isinstance(body, list):
            continue

        combined: list[float] = []
        for value in subj + body:
            try:
                combined.append(float(value))
            except Exception:
                combined.append(0.0)

        if not combined:
            continue

        cache[external_id] = combined
        if embedding_dim == 0:
            embedding_dim = len(combined)

    _EMBEDDINGS_CACHE = cache
    _EMBEDDING_DIM = embedding_dim
    return _EMBEDDINGS_CACHE, _EMBEDDING_DIM


def _pick_event_features(email_fields: dict, feature_types: list[str]) -> dict:
    include_body = "body" in feature_types
    include_urls = "urls" in feature_types

    features: dict[str, float | int] = {}

    if include_body:
        html = email_fields.get("html") if isinstance(email_fields.get("html"), dict) else {}
        css = email_fields.get("css") if isinstance(email_fields.get("css"), dict) else {}
        tree_stats = html.get("tree_stats") if isinstance(html.get("tree_stats"), dict) else {}
        style_features = css.get("style_features") if isinstance(css.get("style_features"), dict) else {}

        features.update(
            {
                "html_total_elements": _safe_float(tree_stats.get("total_elements", 0.0)),
                "html_hidden_elements": _safe_float(tree_stats.get("hidden_elements", 0.0)),
                "html_links": _safe_float(tree_stats.get("links", 0.0)),
                "html_link_ratio": _safe_float(tree_stats.get("link_ratio", 0.0)),
                "html_max_depth": _safe_float(tree_stats.get("max_depth", 0.0)),
                "html_avg_depth": _safe_float(tree_stats.get("avg_depth", 0.0)),
                "css_class_entropy": _safe_float(style_features.get("class_entropy", 0.0)),
                "css_uses_media_queries": _safe_bool(style_features.get("uses_media_queries", 0)),
                "css_unique_color_count": _safe_float(style_features.get("unique_color_count", 0.0)),
                "css_uses_z_index": _safe_bool(style_features.get("uses_z_index", 0)),
                "css_uses_position_absolute": _safe_bool(style_features.get("uses_position_absolute", 0)),
                "css_unique_class_count": _safe_float(style_features.get("unique_class_count", 0.0)),
                "contains_symbols": _safe_bool(email_fields.get("contains_symbols", 0)),
                "body_has_tracking_url": _safe_bool(email_fields.get("body_has_tracking_url", 0)),
                "body_has_tracking_pixel": _safe_bool(email_fields.get("body_has_tracking_pixel", 0)),
                "body_has_tracking_image": _safe_bool(email_fields.get("body_has_tracking_image", 0)),
                "body_has_unsubscribe_link": _safe_bool(email_fields.get("body_has_unsubscribe_link", 0)),
            }
        )

    if include_urls:
        features["domain_is_common_webprovided"] = _safe_bool(
            email_fields.get("domain_is_common_webprovided", 0)
        )

    return features


def extract_fsols_features(
    events: list[dict],
    feature_types: list[str],
    omitted_keys: frozenset[str],
) -> list[dict]:
    """Build FSOLS from direct event attributes plus per-email text embedding dimensions."""
    #embedding_map, embedding_dim = _load_embeddings()

    rows: list[dict] = []
    for event_idx, email_fields in enumerate(events):
        if not isinstance(email_fields, dict):
            continue

        ext = email_fields.get("external_id")
        external_id = str(ext).strip() if ext is not None else ""
        if not external_id:
            raise ValueError(
                f"Event at index {event_idx} has no external_id; required for feature rows"
            )

        row = {"external_id": external_id}
        row.update(_pick_event_features(email_fields, feature_types))
        '''
        embedding = embedding_map.get(external_id)
        if embedding is None and embedding_dim > 0:
            embedding = [0.0] * embedding_dim

        if embedding:
            for idx, value in enumerate(embedding):
                row[f"text_embedding_{idx}"] = float(value)
        '''

        if omitted_keys:
            row = {k: v for k, v in row.items() if k not in omitted_keys}

        rows.append(row)

    return rows
