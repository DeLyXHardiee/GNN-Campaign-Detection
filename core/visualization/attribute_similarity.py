"""
Per-attribute SBERT similarity vs peer average within a campaign (for visualization).

Uses the same model as ``utils.embeddings.embedder`` (multilingual-e5-large) and
``passage:`` prefix for encoding consistency with graph build.
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
except ModuleNotFoundError:
    from core.utils.embeddings.embedder import MODEL_NAME  # noqa: E402

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


def _l2_normalize_rows(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return x / n


def _peer_mean_cosine_similarity(v_unit: np.ndarray) -> np.ndarray:
    """
    v_unit: (n, d) row-wise L2 unit vectors.
    For each i, cosine similarity between v_i and mean(v_j for j!=i), L2-normalized.
    Returns length n in [-1, 1].
    """
    n = v_unit.shape[0]
    if n == 0:
        return np.array([])
    if n == 1:
        return np.ones(1, dtype=np.float64)
    out = np.zeros(n, dtype=np.float64)
    s = v_unit.sum(axis=0)
    for i in range(n):
        if n == 1:
            mo = v_unit[i]
        else:
            mo = (s - v_unit[i]) / float(n - 1)
        mn = float(np.linalg.norm(mo))
        if mn < 1e-12:
            out[i] = 0.0
        else:
            mo_u = mo / mn
            out[i] = float(np.dot(v_unit[i], mo_u))
    return out


def _cosine_to_unit_interval(c: float) -> float:
    return float(max(0.0, min(1.0, (c + 1.0) / 2.0)))


def _min_max_normalize_campaign_attrs(
    sol_out: dict[str, dict[str, dict[str, float]]],
) -> None:
    """
    In-place: for each campaign, scale each attribute's raw scores to [0, 1]
    across emails in that campaign.

    Raw peer-mean cosines in a tight cluster often sit in a narrow band (e.g.
    0.95–0.99), which mapped to hue all looks "green". Normalizing per campaign
    and attribute recovers a full red→green spread for comparison *within* the
    campaign. If all values are equal, use 0.5 (neutral).
    """
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


def _encode_texts(model: Any, texts: list[str]) -> np.ndarray:
    inputs = [f"passage: {t}" for t in texts]
    arr = model.encode(
        inputs,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )
    return np.asarray(arr, dtype=np.float64)


def build_attribute_similarity_sidecar(
    *,
    gnn: dict[str, Any] | None,
    featureset: dict[str, Any] | None,
    emails: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """
    For each solution (gnn / featureset) and each campaign, compute per-email
    scores for how similar each attribute embedding is to the average of the
    other members' embeddings in that campaign. Values are **min–max normalized
    per campaign and attribute** to [0, 1] so the UI red→green scale uses the
    full range when comparing emails *within the same campaign* (raw cosines in
    a tight cluster are often all ~0.97+ and would otherwise look uniformly green).
    """
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "sentence-transformers is required for attribute similarity. "
            f"Install with: pip install sentence-transformers ({exc})"
        ) from exc

    # Unique external ids that appear in any campaign
    eid_set: set[str] = set()

    def collect(camps: list[dict[str, Any]] | None) -> None:
        if not camps:
            return
        for c in camps:
            for eid in c.get("member_external_ids") or []:
                eid_set.add(str(eid))

    if gnn and gnn.get("campaigns"):
        collect(gnn["campaigns"])
    if featureset and featureset.get("campaigns"):
        collect(featureset["campaigns"])

    if not eid_set:
        return {}

    eids_sorted = sorted(eid_set, key=str)
    eid_to_row = {eid: i for i, eid in enumerate(eids_sorted)}

    model = SentenceTransformer(MODEL_NAME)

    # Precompute embeddings per attribute for all needed emails (global order)
    emb_by_attr: dict[str, np.ndarray] = {}
    for attr in ATTR_KEYS:
        texts = [_text_for_attr(emails.get(eid, {}), attr) for eid in eids_sorted]
        raw = _encode_texts(model, texts)
        emb_by_attr[attr] = _l2_normalize_rows(raw)

    out: dict[str, Any] = {"gnn": {}, "featureset": {}}

    def fill_solution(sol_key: str, payload: dict[str, Any] | None) -> None:
        if not payload or not payload.get("campaigns"):
            return
        camps = payload["campaigns"]
        sol_out: dict[str, Any] = {}
        for camp in camps:
            cid = camp.get("id")
            members = [str(x) for x in (camp.get("member_external_ids") or [])]
            members_ok = [eid for eid in members if eid in eid_to_row]
            if not members_ok:
                continue
            idxs = [eid_to_row[eid] for eid in members_ok]
            cid_str = str(cid)
            sol_out[cid_str] = {}
            for eid in members_ok:
                sol_out[cid_str][eid] = {}

            for attr in ATTR_KEYS:
                v_full = emb_by_attr[attr]
                sub = v_full[idxs]
                sims = _peer_mean_cosine_similarity(sub)
                for row_i, eid in enumerate(members_ok):
                    sol_out[cid_str][eid][attr] = _cosine_to_unit_interval(float(sims[row_i]))

        if sol_out:
            _min_max_normalize_campaign_attrs(sol_out)
            out[sol_key] = sol_out

    fill_solution("gnn", gnn)
    fill_solution("featureset", featureset)

    return {k: v for k, v in out.items() if v}
