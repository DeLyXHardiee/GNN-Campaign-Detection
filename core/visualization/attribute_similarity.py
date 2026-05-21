"""
Per-campaign SBERT similarity for visualization (email fields and URLs).

Uses the same model as ``utils.embeddings.embedder`` (multilingual-e5-large) and
``passage:`` prefix for encoding consistency with graph build.

Within each campaign, scores measure cosine similarity of each embedding to the
mean of the other members' embeddings (same field or URL set). Values are
min–max normalized per campaign so the UI red→green scale uses the full range.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

_CORE_ROOT = Path(__file__).resolve().parents[1]
if str(_CORE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CORE_ROOT))

try:
    from utils.embeddings.embedder import MODEL_NAME  # noqa: E402
    from preprocessing.utils.defang import defang_url_string, refang_url_string  # noqa: E402
except ModuleNotFoundError:
    from core.utils.embeddings.embedder import MODEL_NAME  # noqa: E402
    from core.preprocessing.utils.defang import defang_url_string, refang_url_string  # noqa: E402

ATTR_KEYS = ("subject", "body", "senders", "receivers", "date")


def _text_for_attr(email: dict[str, Any], attr: str) -> str:
    if attr == "subject":
        t = (email.get("subject") or "").strip()
    elif attr == "body":
        t = (email.get("body") or "").strip()
        if len(t) > 12000:
            t = t[:12000]
    elif attr == "senders":
        s = email.get("senders") or []
        t = ", ".join(str(x) for x in s) if isinstance(s, list) else str(s)
    elif attr == "receivers":
        s = email.get("receivers") or []
        t = ", ".join(str(x) for x in s) if isinstance(s, list) else str(s)
    elif attr == "date":
        t = (email.get("date") or "").strip()
    else:
        t = ""
    t = t.strip()
    return t if t else "(empty)"


def _canonical_url_key(url: str) -> str:
    return defang_url_string(refang_url_string(str(url).strip()))


def _urls_for_email_row(email: dict[str, Any]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for raw in email.get("urls") or []:
        key = _canonical_url_key(str(raw))
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def _l2_normalize_rows(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return x / n


def _peer_mean_cosine_similarity(v_unit: np.ndarray) -> np.ndarray:
    """
    For each i, cosine between v_i and L2-normalized mean(v_j for j!=i).
    """
    n = v_unit.shape[0]
    if n == 0:
        return np.array([])
    if n == 1:
        return np.ones(1, dtype=np.float64)
    out = np.zeros(n, dtype=np.float64)
    s = v_unit.sum(axis=0)
    for i in range(n):
        mo = (s - v_unit[i]) / float(n - 1)
        mn = float(np.linalg.norm(mo))
        if mn < 1e-12:
            out[i] = 0.0
        else:
            out[i] = float(np.dot(v_unit[i], mo / mn))
    return out


def _cosine_to_unit_interval(c: float) -> float:
    return float(max(0.0, min(1.0, (c + 1.0) / 2.0)))


def _min_max_normalize_campaign_scores(
    by_key: dict[str, float],
) -> None:
    """In-place: scale scores in one campaign bucket to [0, 1]; all equal → 0.5."""
    if not by_key:
        return
    vals = list(by_key.values())
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-12:
        for k in by_key:
            by_key[k] = 0.5
    else:
        for k, v in by_key.items():
            by_key[k] = float((v - lo) / (hi - lo))


def _min_max_normalize_email_attrs(
    sol_out: dict[str, dict[str, dict[str, float]]],
) -> None:
    for _cid, by_eid in sol_out.items():
        if not by_eid:
            continue
        eids = list(by_eid.keys())
        for attr in ATTR_KEYS:
            vals = [by_eid[e].get(attr, 0.5) for e in eids]
            if not vals:
                continue
            lo, hi = min(vals), max(vals)
            if hi - lo < 1e-12:
                for e in eids:
                    by_eid[e][attr] = 0.5
            else:
                for e in eids:
                    v = by_eid[e].get(attr, lo)
                    by_eid[e][attr] = float((v - lo) / (hi - lo))


def _encode_texts(
    model: Any,
    texts: list[str],
    *,
    show_progress: bool = True,
) -> np.ndarray:
    if not texts:
        return np.zeros((0, 0), dtype=np.float64)
    inputs = [f"passage: {t}" for t in texts]
    arr = model.encode(
        inputs,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )
    return np.asarray(arr, dtype=np.float64)


def _load_sentence_model() -> Any:
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "sentence-transformers is required for attribute similarity. "
            f"Install with: pip install sentence-transformers ({exc})"
        ) from exc
    return SentenceTransformer(MODEL_NAME)


def _member_eids_from_solutions(
    solutions: dict[str, dict[str, Any]],
) -> set[str]:
    eid_set: set[str] = set()
    for payload in (solutions or {}).values():
        if not payload:
            continue
        for c in payload.get("campaigns") or []:
            for eid in c.get("member_external_ids") or []:
                eid_set.add(str(eid))
    return eid_set


def build_attribute_similarity_sidecar(
    *,
    solutions: dict[str, dict[str, Any]],
    emails: dict[str, dict[str, Any]],
    model: Any | None = None,
) -> dict[str, Any]:
    """
    Per solution / campaign / email / attribute: peer-mean cosine vs other members,
    min–max normalized per campaign and attribute to [0, 1] for red→green UI.

    Returns ``{ <solution_key>: { <campaign_id>: { <eid>: { attr: score } } } }``.
    """
    eid_set = _member_eids_from_solutions(solutions)
    if not eid_set:
        return {}

    eids_sorted = sorted(eid_set, key=str)
    eid_to_row = {eid: i for i, eid in enumerate(eids_sorted)}

    if model is None:
        print(
            f"[viz] Loading SBERT ({MODEL_NAME}) for {len(eids_sorted)} campaign emails…",
            flush=True,
        )
        model = _load_sentence_model()

    emb_by_attr: dict[str, np.ndarray] = {}
    for attr in ATTR_KEYS:
        texts = [_text_for_attr(emails.get(eid, {}), attr) for eid in eids_sorted]
        print(f"[viz] Encoding attribute '{attr}' ({len(texts)} texts)…", flush=True)
        raw = _encode_texts(model, texts)
        emb_by_attr[attr] = _l2_normalize_rows(raw)

    out: dict[str, Any] = {}
    for sol_key, payload in (solutions or {}).items():
        if not payload or not payload.get("campaigns"):
            continue
        sol_out: dict[str, Any] = {}
        for camp in payload["campaigns"]:
            cid = camp.get("id")
            members = [str(x) for x in (camp.get("member_external_ids") or [])]
            members_ok = [eid for eid in members if eid in eid_to_row]
            if not members_ok:
                continue
            idxs = [eid_to_row[eid] for eid in members_ok]
            cid_str = str(cid)
            sol_out[cid_str] = {eid: {} for eid in members_ok}

            for attr in ATTR_KEYS:
                sub = emb_by_attr[attr][idxs]
                sims = _peer_mean_cosine_similarity(sub)
                for row_i, eid in enumerate(members_ok):
                    sol_out[cid_str][eid][attr] = _cosine_to_unit_interval(float(sims[row_i]))

        if sol_out:
            _min_max_normalize_email_attrs(sol_out)
            out[sol_key] = sol_out

    return out


def build_url_similarity_sidecar(
    *,
    solutions: dict[str, dict[str, Any]],
    emails: dict[str, dict[str, Any]],
    model: Any | None = None,
) -> dict[str, Any]:
    """
    Per solution / campaign / canonical URL: peer-mean cosine vs other URLs in
    that campaign, min–max normalized per campaign to [0, 1].

    Embeds each distinct URL once for the whole run, then scores per campaign.

    Returns ``{ <solution_key>: { <campaign_id>: { <url>: score } } }``.
    """
    if not solutions:
        return {}

    url_to_row: dict[str, int] = {}
    urls_sorted: list[str] = []
    plan: dict[str, dict[str, list[str]]] = {}

    for sol_key, payload in (solutions or {}).items():
        if not payload or not payload.get("campaigns"):
            continue
        sol_plan: dict[str, list[str]] = {}
        for camp in payload["campaigns"]:
            members = [str(x) for x in (camp.get("member_external_ids") or [])]
            camp_urls: list[str] = []
            seen: set[str] = set()
            for eid in members:
                for u in _urls_for_email_row(emails.get(eid) or {}):
                    if u in seen:
                        continue
                    seen.add(u)
                    camp_urls.append(u)
            if not camp_urls:
                continue
            camp_urls.sort(key=str.casefold)
            sol_plan[str(camp.get("id"))] = camp_urls
            for u in camp_urls:
                if u not in url_to_row:
                    url_to_row[u] = len(urls_sorted)
                    urls_sorted.append(u)
        if sol_plan:
            plan[sol_key] = sol_plan

    if not urls_sorted:
        return {}

    if model is None:
        print(f"[viz] Loading SBERT ({MODEL_NAME}) for URL similarity…", flush=True)
        model = _load_sentence_model()

    print(f"[viz] Encoding {len(urls_sorted)} unique URLs (one batch)…", flush=True)
    unit = _l2_normalize_rows(
        _encode_texts(model, urls_sorted),
    )

    out: dict[str, Any] = {}
    for sol_key, sol_plan in plan.items():
        sol_out: dict[str, Any] = {}
        for cid, camp_urls in sol_plan.items():
            idxs = [url_to_row[u] for u in camp_urls]
            sub = unit[idxs]
            sims = _peer_mean_cosine_similarity(sub)
            by_url: dict[str, float] = {}
            for i, url in enumerate(camp_urls):
                by_url[url] = _cosine_to_unit_interval(float(sims[i]))
            _min_max_normalize_campaign_scores(by_url)
            sol_out[cid] = by_url
        if sol_out:
            out[sol_key] = sol_out

    return out


def build_campaign_similarity_sidecars(
    *,
    solutions: dict[str, dict[str, Any]],
    emails: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Attribute and URL sidecars (per campaign) with one SBERT model load."""
    if not solutions:
        return {}, {}
    print("[viz] Computing per-campaign SBERT similarity for visualization…", flush=True)
    model = _load_sentence_model()
    attr = build_attribute_similarity_sidecar(
        solutions=solutions, emails=emails, model=model
    )
    url = build_url_similarity_sidecar(
        solutions=solutions, emails=emails, model=model
    )
    print("[viz] Similarity sidecars done.", flush=True)
    return attr, url
