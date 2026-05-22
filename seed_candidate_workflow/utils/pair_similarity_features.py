"""
Shared pair similarity features for training and low-band feature discovery.

Definitions match ``pair_low_band_feature_discovery`` analysis (body token/char Jaccard,
normalized sender local-part similarity, path-token Jaccard).
"""

from __future__ import annotations

import difflib
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_TOKEN_RE = re.compile(r"[a-z0-9]+", re.I)
_DIGITS_RE = re.compile(r"\d+")
_SENDER_DISPLAY_RE = re.compile(r"^(.+?)\s*<([^>]+)>$")
_URL_IN_BODY_RE = re.compile(r"https?://[^\s\"'<>]+|www\.[^\s\"'<>]+", re.IGNORECASE)

TEXT_SIMILARITY_PAIR_FEATURE_COLS = (
    "body_token_jaccard",
    "body_char4gram_jaccard",
    "sender_localpart_norm_jaccard",
)

BODY_ONLY_PAIR_FEATURE_COLS = (
    "body_only_token_jaccard",
    "body_only_char4gram_jaccard",
)

RESCUE_ALIGNED_SCORER_FEATURE_COLS = (
    "body_only_token_jaccard",
    "body_only_char4gram_jaccard",
    "path_token_jaccard_combined",
)

SCORER_PAIR_NUMERIC_FEATURE_COLS = TEXT_SIMILARITY_PAIR_FEATURE_COLS + RESCUE_ALIGNED_SCORER_FEATURE_COLS


def jaccard_similarity(a: set[str], b: set[str]) -> float:
    """Jaccard index; returns 0.0 when both sets are empty."""
    if not a and not b:
        return 0.0
    u = a | b
    if not u:
        return 0.0
    return float(len(a & b) / len(u))


def tokenize_text(text: str, *, min_len: int = 2) -> set[str]:
    """Word tokens from lowercased alphanumeric runs (analysis ``_tokenize``)."""
    return {t.lower() for t in _TOKEN_RE.findall(str(text or "")) if len(t) >= min_len}


def char_ngrams_text(text: str, n: int) -> set[str]:
    """Character n-grams after whitespace collapse (analysis ``_char_ngrams``)."""
    s = re.sub(r"\s+", " ", str(text or "").lower()).strip()
    if len(s) < n:
        return set()
    return {s[i : i + n] for i in range(len(s) - n + 1)}


def body_token_jaccard_from_bodies(body_i: str, body_j: str) -> float:
    """Token Jaccard on raw bodies (min token length 2)."""
    return jaccard_similarity(
        tokenize_text(body_i, min_len=2),
        tokenize_text(body_j, min_len=2),
    )


def body_char4gram_jaccard_from_bodies(body_i: str, body_j: str) -> float:
    """Character 4-gram Jaccard on normalized bodies."""
    return jaccard_similarity(char_ngrams_text(body_i, 4), char_ngrams_text(body_j, 4))


def strip_url_like_tokens_from_body(body: str) -> str:
    """Remove URL-like substrings before body-only similarity (analysis + scorer alignment)."""
    t = _URL_IN_BODY_RE.sub(" ", str(body or ""))
    return re.sub(r"\s+", " ", t).strip()


def body_only_token_jaccard_from_bodies(body_i: str, body_j: str) -> float:
    """Token Jaccard on bodies after stripping URL-like tokens."""
    return body_token_jaccard_from_bodies(
        strip_url_like_tokens_from_body(body_i),
        strip_url_like_tokens_from_body(body_j),
    )


def body_only_char4gram_jaccard_from_bodies(body_i: str, body_j: str) -> float:
    """Char-4gram Jaccard on bodies after stripping URL-like tokens."""
    return body_char4gram_jaccard_from_bodies(
        strip_url_like_tokens_from_body(body_i),
        strip_url_like_tokens_from_body(body_j),
    )


def parse_sender_parts(sender: str) -> tuple[str, str, str]:
    """Return (local_part, domain, display_name)."""
    s = str(sender or "").strip()
    m = _SENDER_DISPLAY_RE.match(s)
    if m:
        display = m.group(1).strip().strip('"')
        addr = m.group(2).strip().lower()
    else:
        display = ""
        addr = s.lower()
    if "@" in addr:
        local, dom = addr.split("@", 1)
    else:
        local, dom = addr, ""
    return local, dom, display


def normalize_sender_localpart(local: str) -> str:
    return _DIGITS_RE.sub("", str(local or "").lower())


def first_sender_string(node: dict[str, Any]) -> str:
    ss = node.get("sender_set") or set()
    return str(next(iter(ss), "")) if ss else ""


def sender_localpart_norm_similarity(local_a: str, local_b: str) -> float:
    """
    SequenceMatcher ratio on digit-stripped lowercased local-parts.

    Same definition as feature-discovery ``sender_localpart_norm_jaccard``.
    Returns 0.0 when either normalized local-part is empty.
    """
    na = normalize_sender_localpart(local_a)
    nb = normalize_sender_localpart(local_b)
    if not na and not nb:
        return 0.0
    if not na or not nb:
        return 0.0
    return float(difflib.SequenceMatcher(None, na, nb).ratio())


def sender_localpart_norm_jaccard_for_nodes(
    node_a: dict[str, Any],
    node_b: dict[str, Any],
) -> float:
    """Normalized local-part similarity using the first sender on each email."""
    la, _, _ = parse_sender_parts(first_sender_string(node_a))
    lb, _, _ = parse_sender_parts(first_sender_string(node_b))
    return sender_localpart_norm_similarity(la, lb)


def compute_text_similarity_pair_features(
    *,
    email_i: str,
    email_j: str,
    text_catalog: dict[str, dict[str, str]],
    nodes_by_email: dict[str, dict[str, Any]] | None,
) -> dict[str, float]:
    """Compute raw body + sender similarity features for one pair."""
    ti = text_catalog.get(str(email_i), {}) or {}
    tj = text_catalog.get(str(email_j), {}) or {}
    bi = str(ti.get("body") or "")
    bj = str(tj.get("body") or "")
    na = (nodes_by_email or {}).get(str(email_i)) or {}
    nb = (nodes_by_email or {}).get(str(email_j)) or {}
    return {
        "body_token_jaccard": body_token_jaccard_from_bodies(bi, bj),
        "body_char4gram_jaccard": body_char4gram_jaccard_from_bodies(bi, bj),
        "sender_localpart_norm_jaccard": sender_localpart_norm_jaccard_for_nodes(na, nb),
    }


def compute_body_only_pair_features(
    *,
    email_i: str,
    email_j: str,
    text_catalog: dict[str, dict[str, str]],
) -> dict[str, float]:
    """Body-only (URL-stripped) similarity for one pair."""
    ti = text_catalog.get(str(email_i), {}) or {}
    tj = text_catalog.get(str(email_j), {}) or {}
    bi = str(ti.get("body") or "")
    bj = str(tj.get("body") or "")
    return {
        "body_only_token_jaccard": body_only_token_jaccard_from_bodies(bi, bj),
        "body_only_char4gram_jaccard": body_only_char4gram_jaccard_from_bodies(bi, bj),
    }


def add_text_similarity_pair_features_to_dataframe(
    df: pd.DataFrame,
    *,
    text_catalog: dict[str, dict[str, str]],
    nodes_by_email: dict[str, dict[str, Any]] | None,
    body_feature_store: Any | None = None,
) -> pd.DataFrame:
    """Add (or overwrite) text-similarity pair feature columns on ``df``.

    When ``body_feature_store`` is set, ``body_token_jaccard`` and ``body_char4gram_jaccard``
    reuse the same per-email token/char4 sets as candidate-generation caches (fast path).
    """
    from seed_candidate_workflow.utils.body_similarity_cache import EmailBodyFeatureStore

    if df.empty:
        out = df.copy()
        for c in TEXT_SIMILARITY_PAIR_FEATURE_COLS:
            out[c] = np.nan
        return out

    out = df.copy()
    nodes = nodes_by_email or {}
    if body_feature_store is not None:
        if not isinstance(body_feature_store, EmailBodyFeatureStore):
            raise TypeError("body_feature_store must be an EmailBodyFeatureStore or None")
        body_tok: list[float] = []
        body_c4: list[float] = []
        sender_lp: list[float] = []
        for _, r in out.iterrows():
            ei = str(r["email_i"])
            ej = str(r["email_j"])
            body_tok.append(float(body_feature_store.token_jaccard(ei, ej)))
            body_c4.append(float(body_feature_store.char4_jaccard(ei, ej)))
            na = nodes.get(ei) or {}
            nb = nodes.get(ej) or {}
            sender_lp.append(float(sender_localpart_norm_jaccard_for_nodes(na, nb)))
        out["body_token_jaccard"] = body_tok
        out["body_char4gram_jaccard"] = body_c4
        out["sender_localpart_norm_jaccard"] = sender_lp
        return out

    body_tok = []
    body_c4 = []
    sender_lp = []
    for _, r in out.iterrows():
        feats = compute_text_similarity_pair_features(
            email_i=str(r["email_i"]),
            email_j=str(r["email_j"]),
            text_catalog=text_catalog,
            nodes_by_email=nodes_by_email,
        )
        body_tok.append(feats["body_token_jaccard"])
        body_c4.append(feats["body_char4gram_jaccard"])
        sender_lp.append(feats["sender_localpart_norm_jaccard"])
    out["body_token_jaccard"] = body_tok
    out["body_char4gram_jaccard"] = body_c4
    out["sender_localpart_norm_jaccard"] = sender_lp
    return out


def text_similarity_pair_features_present(df: pd.DataFrame) -> bool:
    return all(c in df.columns for c in TEXT_SIMILARITY_PAIR_FEATURE_COLS)


def pair_scorer_similarity_features_present(df: pd.DataFrame) -> bool:
    """True when all numeric columns consumed by the pair MLP scorer exist."""
    return all(c in df.columns for c in SCORER_PAIR_NUMERIC_FEATURE_COLS)


def add_body_only_pair_features_to_dataframe(
    df: pd.DataFrame,
    *,
    text_catalog: dict[str, dict[str, str]],
) -> pd.DataFrame:
    """Add URL-stripped body-only Jaccard columns."""
    if df.empty:
        out = df.copy()
        for c in BODY_ONLY_PAIR_FEATURE_COLS:
            out[c] = np.nan
        return out
    out = df.copy()
    tok: list[float] = []
    c4: list[float] = []
    for _, r in out.iterrows():
        feats = compute_body_only_pair_features(
            email_i=str(r["email_i"]),
            email_j=str(r["email_j"]),
            text_catalog=text_catalog,
        )
        tok.append(feats["body_only_token_jaccard"])
        c4.append(feats["body_only_char4gram_jaccard"])
    out["body_only_token_jaccard"] = tok
    out["body_only_char4gram_jaccard"] = c4
    return out


def load_misp_text_catalog_for_pairs(
    *,
    project_root: Path | None = None,
    misp_json_path: Path | None = None,
) -> tuple[dict[str, dict[str, str]], dict[str, Any]]:
    """Load MISP subject/body catalog (lazy import of pair_score_separation helpers)."""
    from seed_candidate_workflow.utils import graph_structure_helpers as gh
    from seed_candidate_workflow.utils.pair_score_separation import (
        _load_email_text_catalog,
        _resolve_default_misp_json_path,
    )

    root = project_root or gh.find_project_root()
    misp_path = misp_json_path
    if misp_path is None:
        try:
            misp_path = _resolve_default_misp_json_path(root)
        except Exception:
            misp_path = None
    if misp_path is None or not Path(misp_path).is_file():
        return {}, {"status": "skipped", "reason": f"misp_json_not_found:{misp_path}"}
    return _load_email_text_catalog(
        project_root=root,
        misp_json_path=Path(misp_path),
        misp_translated_json_path=None,
    )


def _infer_graph_id_from_pair_csv(csv_path: Path) -> str | None:
    parts = list(csv_path.resolve().parts)
    try:
        i = [p.lower() for p in parts].index("graph_bundles")
        if i + 1 < len(parts):
            gid = str(parts[i + 1]).strip()
            return gid or None
    except ValueError:
        return None
    return None


def _resolve_candidate_union_csv_for_pair_csv(csv_path: Path, graph_id: str | None) -> Path | None:
    """Locate candidate_union.csv adjacent to a graph bundle pair_training artifact."""
    gid = graph_id or _infer_graph_id_from_pair_csv(csv_path)
    if gid:
        bundle_root = csv_path.parent.parent.parent
        cand = bundle_root / "candidate" / gid / "candidate_union.csv"
        if cand.is_file():
            return cand
    for p in csv_path.parents:
        if p.name == "candidate_union.csv":
            return p
    return None


def _load_nodes_by_email_for_pair_csv(
    *,
    csv_path: Path,
    graph_id: str | None,
    project_root: Path | None,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    from seed_candidate_workflow.utils.pair_training_dataset_helpers import _load_anchor_node_sets_by_email

    root = project_root
    if root is None:
        from seed_candidate_workflow.utils import graph_structure_helpers as gh

        root = gh.find_project_root()
    gid = graph_id or _infer_graph_id_from_pair_csv(csv_path)
    cand_union = _resolve_candidate_union_csv_for_pair_csv(csv_path, gid)
    if cand_union is None:
        return {}, {"status": "skipped", "reason": "candidate_union_csv_not_found"}
    return _load_anchor_node_sets_by_email(
        candidate_union_csv=cand_union,
        graph_id=gid,
        project_root=root,
    )


def ensure_pair_scorer_similarity_features_in_dataframe(
    df: pd.DataFrame,
    *,
    csv_path: Path | None = None,
    project_root: Path | None = None,
    graph_id: str | None = None,
    misp_json_path: Path | None = None,
    text_catalog: dict[str, dict[str, str]] | None = None,
    nodes_by_email: dict[str, dict[str, Any]] | None = None,
    force_recompute: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Ensure all pair-scorer numeric similarity columns exist (raw body, body-only, path, sender).

    When columns are missing (or ``force_recompute``), loads MISP text + anchor node sets.
    """
    meta: dict[str, Any] = {"enriched": False}
    if not force_recompute and pair_scorer_similarity_features_present(df):
        meta["enriched"] = False
        meta["reason"] = "columns_already_present"
        return df, meta

    root = project_root
    if root is None:
        from seed_candidate_workflow.utils import graph_structure_helpers as gh

        root = gh.find_project_root()

    catalog = text_catalog
    catalog_meta: dict[str, Any] = {}
    if catalog is None:
        catalog, catalog_meta = load_misp_text_catalog_for_pairs(
            project_root=root,
            misp_json_path=misp_json_path,
        )
    meta["text_catalog"] = catalog_meta

    nodes = nodes_by_email
    nodes_meta: dict[str, Any] = {}
    if nodes is None and csv_path is not None:
        nodes, nodes_meta = _load_nodes_by_email_for_pair_csv(
            csv_path=Path(csv_path),
            graph_id=graph_id,
            project_root=root,
        )
    meta["anchor_nodes"] = nodes_meta

    out = df.copy()
    if not catalog:
        for c in SCORER_PAIR_NUMERIC_FEATURE_COLS:
            out[c] = 0.0
        meta["enriched"] = True
        meta["reason"] = "empty_text_catalog_defaults_zero"
        return out, meta

    if force_recompute or not text_similarity_pair_features_present(out):
        body_store = None
        body_cache_meta: dict[str, Any] = {"status": "skipped"}
        if csv_path is not None:
            try:
                from seed_candidate_workflow.utils.body_similarity_cache import (
                    build_or_load_email_body_feature_store,
                )
                from seed_candidate_workflow.utils.pair_training_dataset_helpers import (
                    infer_graph_id_from_graph_bundles_path,
                    load_sorted_anchor_external_ids_for_candidate_union,
                )
                from seed_candidate_workflow.utils.pair_score_separation import (
                    _resolve_default_misp_json_path,
                )

                p_csv = Path(csv_path)
                gid = graph_id or _infer_graph_id_from_pair_csv(p_csv)
                cand_path = _resolve_candidate_union_csv_for_pair_csv(p_csv, gid)
                mpath = misp_json_path
                if mpath is None:
                    try:
                        mpath = _resolve_default_misp_json_path(root)
                    except Exception:
                        mpath = None
                if cand_path is not None and mpath is not None and Path(mpath).is_file():
                    anchor_ids, aid_m = load_sorted_anchor_external_ids_for_candidate_union(
                        candidate_union_csv=cand_path,
                        graph_id=gid,
                        project_root=root,
                    )
                    gidr = gid or infer_graph_id_from_graph_bundles_path(cand_path) or "unknown_graph"
                    if anchor_ids:
                        body_store, body_cache_meta = build_or_load_email_body_feature_store(
                            email_ids=anchor_ids,
                            text_catalog=catalog,
                            graph_id=str(gidr),
                            misp_json_path=Path(mpath),
                            force_rebuild=False,
                        )
                        body_cache_meta["anchor_external_ids_meta"] = aid_m
                    else:
                        body_cache_meta = {
                            "status": "skipped",
                            "reason": "anchor_external_ids_unavailable",
                            "anchor_external_ids_meta": aid_m,
                        }
                else:
                    body_cache_meta = {"status": "skipped", "reason": "candidate_union_or_misp_missing"}
            except Exception as exc:
                body_cache_meta = {"status": "skipped", "error": str(exc)}
        meta["body_email_feature_cache"] = body_cache_meta
        out = add_text_similarity_pair_features_to_dataframe(
            out,
            text_catalog=catalog,
            nodes_by_email=nodes or {},
            body_feature_store=body_store,
        )
    if force_recompute or not all(c in out.columns for c in BODY_ONLY_PAIR_FEATURE_COLS):
        out = add_body_only_pair_features_to_dataframe(out, text_catalog=catalog)
    if force_recompute or "path_token_jaccard_combined" not in out.columns:
        if nodes:
            out = attach_path_jaccard_features_to_dataframe(
                out,
                nodes_by_email=nodes,
                prefer_existing=not force_recompute,
            )
        else:
            out["path_token_jaccard_combined"] = 0.0
    out["path_token_jaccard_combined"] = (
        pd.to_numeric(out["path_token_jaccard_combined"], errors="coerce").fillna(0.0)
    )

    meta["enriched"] = True
    meta["reason"] = "computed_from_misp_and_anchor_nodes"
    meta["columns_added"] = [c for c in SCORER_PAIR_NUMERIC_FEATURE_COLS if c in out.columns]
    return out, meta


def ensure_text_similarity_pair_features_in_dataframe(
    df: pd.DataFrame,
    *,
    csv_path: Path | None = None,
    project_root: Path | None = None,
    graph_id: str | None = None,
    misp_json_path: Path | None = None,
    text_catalog: dict[str, dict[str, str]] | None = None,
    nodes_by_email: dict[str, dict[str, Any]] | None = None,
    force_recompute: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Backward-compatible alias: ensures full scorer similarity feature set."""
    return ensure_pair_scorer_similarity_features_in_dataframe(
        df,
        csv_path=csv_path,
        project_root=project_root,
        graph_id=graph_id,
        misp_json_path=misp_json_path,
        text_catalog=text_catalog,
        nodes_by_email=nodes_by_email,
        force_recompute=force_recompute,
    )


def build_scorer_input_feature_sanity(df: pd.DataFrame) -> dict[str, Any]:
    """Non-null counts / presence for features passed to the pair MLP."""
    expected = list(SCORER_PAIR_NUMERIC_FEATURE_COLS)
    present = [c for c in expected if c in df.columns]
    missing = [c for c in expected if c not in df.columns]
    non_null: dict[str, int] = {}
    for c in present:
        non_null[c] = int(pd.to_numeric(df[c], errors="coerce").notna().sum())
    return {
        "expected_numeric_features": expected,
        "present_in_dataset": present,
        "missing_from_dataset": missing,
        "non_null_counts": non_null,
        "n_rows": int(len(df)),
        "rescue_aligned_features": {
            "body_only_token_jaccard": "body_only_token_jaccard" in present,
            "body_only_char4gram_jaccard": "body_only_char4gram_jaccard" in present,
            "path_token_jaccard_combined": "path_token_jaccard_combined" in present,
        },
        "raw_body_features_retained": [
            c for c in ("body_token_jaccard", "body_char4gram_jaccard") if c in present
        ],
    }


# --- URL / path token features (unchanged) ---

_ROOT_STEM = "/"


def parse_url_path_tokens(url: str) -> tuple[str, list[str], int]:
    """Return (registrable_domain, path_tokens, path_depth)."""
    from core.feature_set_extraction.url_extraction_utils import parse_url_host_and_registrable_domain
    from core.preprocessing.utils.url_extractor import parse_url_components

    u = str(url or "").strip()
    if not u:
        return "", [], 0
    _host, reg, ok = parse_url_host_and_registrable_domain(u)
    reg = reg.lower() if ok else ""
    comp = parse_url_components(u)
    stem = str(comp.get("stem") or "").strip()
    parts = [p for p in stem.split("/") if p and p != _ROOT_STEM]
    depth = len(parts)
    tokens: list[str] = []
    for p in parts:
        p_norm = re.sub(r"\d{4,}", "<id>", p.lower())
        for t in re.split(r"[/_.-]+", p_norm):
            t = t.strip()
            if t and t not in ("<id>",):
                tokens.append(t)
    return reg, tokens, depth


def nontrivial_stems(stems: set[str]) -> set[str]:
    return {s for s in stems if s and s != _ROOT_STEM}


def path_token_set_for_node(node: dict[str, Any]) -> set[str]:
    """Union of URL path tokens and nontrivial stem tokens for one email."""
    url_tokens: list[str] = []
    for u in node.get("url_set") or set():
        _reg, toks, _dep = parse_url_path_tokens(str(u))
        url_tokens.extend(toks)
    path_tokens = set(url_tokens)
    stem_tokens: set[str] = set()
    for st in nontrivial_stems(node.get("stem_set") or set()):
        for t in re.split(r"[/_.-]+", str(st).lower()):
            if t:
                stem_tokens.add(t)
    return path_tokens | stem_tokens


def path_token_jaccard_combined_for_nodes(
    node_a: dict[str, Any],
    node_b: dict[str, Any],
) -> float:
    """
    Jaccard similarity of combined URL path + nontrivial stem path tokens.

    Same definition as feature-discovery ``path_token_jaccard_combined``.
    """
    return jaccard_similarity(path_token_set_for_node(node_a), path_token_set_for_node(node_b))


PATH_JACCARD_FEATURE_COLS: tuple[str, ...] = (
    "path_token_jaccard_combined",
    "url_path_token_jaccard",
    "stem_path_token_jaccard",
)


def url_path_token_set_for_node(node: dict[str, Any]) -> set[str]:
    """Path tokens from URL sets only (excludes stem-set tokens)."""
    tokens: list[str] = []
    for u in node.get("url_set") or set():
        _reg, toks, _dep = parse_url_path_tokens(str(u))
        tokens.extend(toks)
    return set(tokens)


def stem_path_token_set_for_node(node: dict[str, Any]) -> set[str]:
    """Path tokens from nontrivial stem sets only."""
    stem_tokens: set[str] = set()
    for st in nontrivial_stems(node.get("stem_set") or set()):
        for t in re.split(r"[/_.-]+", str(st).lower()):
            if t:
                stem_tokens.add(t)
    return stem_tokens


def path_jaccard_features_for_node_pair(
    node_a: dict[str, Any] | None,
    node_b: dict[str, Any] | None,
) -> dict[str, float | None]:
    """Per-pair URL/stem/combined path Jaccard (None when anchor context missing)."""
    if not node_a or not node_b:
        return {c: None for c in PATH_JACCARD_FEATURE_COLS}
    return {
        "url_path_token_jaccard": jaccard_similarity(
            url_path_token_set_for_node(node_a), url_path_token_set_for_node(node_b)
        ),
        "stem_path_token_jaccard": jaccard_similarity(
            stem_path_token_set_for_node(node_a), stem_path_token_set_for_node(node_b)
        ),
        "path_token_jaccard_combined": path_token_jaccard_combined_for_nodes(node_a, node_b),
    }


def attach_path_jaccard_features_to_dataframe(
    df: pd.DataFrame,
    *,
    nodes_by_email: dict[str, dict[str, Any]],
    prefer_existing: bool = True,
) -> pd.DataFrame:
    """
    Attach path Jaccard columns from anchor-graph node context.

    Path features are not stored on ``pair_training_dataset.csv``; they are derived from
    ``url_set`` / ``stem_set`` on anchor nodes (same as low-band feature discovery).
    """
    if df.empty:
        return df
    if not nodes_by_email:
        out = df.copy()
        out["path_token_jaccard_combined"] = 0.0
        return out
    out = df.copy()
    computed: dict[str, list[float | None]] = {c: [] for c in PATH_JACCARD_FEATURE_COLS}
    for _, row in out.iterrows():
        feats = path_jaccard_features_for_node_pair(
            nodes_by_email.get(str(row["email_i"])),
            nodes_by_email.get(str(row["email_j"])),
        )
        for col in PATH_JACCARD_FEATURE_COLS:
            computed[col].append(feats.get(col))
    for col in PATH_JACCARD_FEATURE_COLS:
        new_s = pd.Series(computed[col], index=out.index, dtype=float)
        if prefer_existing and col in out.columns:
            old_s = pd.to_numeric(out[col], errors="coerce")
            out[col] = old_s.where(old_s.notna(), new_s)
        else:
            out[col] = new_s
    return out
