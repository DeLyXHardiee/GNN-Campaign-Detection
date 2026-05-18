"""
Shared MISP email parsing and strict/near signature helpers.

Shared by ``collapse_misp_lake_strict_duplicates.py`` (and optional duplicate analysis
under ``data/misp/``) so collapse exports and diagnostics cannot drift.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

try:
    from sklearn.metrics import homogeneity_completeness_v_measure
except ImportError:  # pragma: no cover
    homogeneity_completeness_v_measure = None  # type: ignore[misc, assignment]

URL_RE = re.compile(r"https?://[^\s\"'<>]+", re.IGNORECASE)
WS_RE = re.compile(r"\s+")
HEX_TOKEN_RE = re.compile(r"\b[a-f0-9]{16,}\b", re.IGNORECASE)
LONG_INT_RE = re.compile(r"\b\d{6,}\b")
TIME_LIKE_RE = re.compile(
    r"\b(?:\d{4}[-/]\d{2}[-/]\d{2}"
    r"(?:[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d{1,6})?)?(?:Z|[+-]\d{2}:\d{2})?)?"
    r"|\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class EmailRecord:
    external_id: str
    subject: str
    body: str
    senders: tuple[str, ...]
    attachments: tuple[str, ...]


def _to_str(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, str):
        return v
    return str(v)


def _normalize_space(text: str) -> str:
    return WS_RE.sub(" ", text).strip()


def _basic_norm(text: str) -> str:
    t = unicodedata.normalize("NFKC", _to_str(text)).lower()
    return _normalize_space(t)


def _canonicalize_url(u: str) -> str:
    s = _to_str(u).strip()
    if not s:
        return ""
    s = s.lower()
    s = re.sub(r"^https?://", "", s)
    s = s.split("#", 1)[0]
    s = s.split("?", 1)[0]
    s = s.strip("/")
    return s


def _aggressive_norm(text: str) -> str:
    t = _basic_norm(text)
    if not t:
        return t
    t = URL_RE.sub(lambda m: f" url:{_canonicalize_url(m.group(0))} ", t)
    t = TIME_LIKE_RE.sub(" ", t)
    t = HEX_TOKEN_RE.sub(" ", t)
    t = LONG_INT_RE.sub(" ", t)
    return _normalize_space(t)


def _to_list(v: Any) -> list[Any]:
    if v is None:
        return []
    if isinstance(v, list):
        return v
    if isinstance(v, tuple):
        return list(v)
    if isinstance(v, str):
        s = v.strip()
        if not s:
            return []
        if s.startswith("[") and s.endswith("]"):
            try:
                x = json.loads(s)
                if isinstance(x, list):
                    return x
            except Exception:
                pass
        if "|" in s:
            return [p.strip() for p in s.split("|") if p.strip()]
        if "," in s:
            return [p.strip() for p in s.split(",") if p.strip()]
        return [s]
    return [v]


def _pick_first_nonempty(d: dict[str, Any], keys: list[str]) -> str:
    for k in keys:
        if k in d:
            v = _to_str(d.get(k)).strip()
            if v:
                return v
    return ""


def _extract_event_dict(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return {}
    ev = raw.get("Event")
    if isinstance(ev, dict):
        return ev
    return raw


def _attribute_lookup(event_dict: dict[str, Any]) -> dict[str, list[Any]]:
    out: dict[str, list[Any]] = defaultdict(list)
    attrs = event_dict.get("Attribute")
    if not isinstance(attrs, list):
        return out
    for a in attrs:
        if not isinstance(a, dict):
            continue
        t = _to_str(a.get("type")).strip().lower()
        if not t:
            continue
        out[t].append(a.get("value"))
    return out


def _first_attr(attrs: dict[str, list[Any]], keys: list[str]) -> Any:
    for k in keys:
        vals = attrs.get(k.lower(), [])
        for v in vals:
            s = _to_str(v).strip()
            if s:
                return v
    return ""


def _extract_email_record(raw: Any, idx: int) -> EmailRecord:
    ev = _extract_event_dict(raw)
    attrs = _attribute_lookup(ev)
    external_id = (
        _to_str(ev.get("external_id")).strip()
        or _to_str(ev.get("email_index")).strip()
        or str(idx)
    )
    subject = (
        _to_str(_first_attr(attrs, ["subject"]))
        or _pick_first_nonempty(ev, ["subject", "email_subject", "title"])
    )
    body = (
        _to_str(_first_attr(attrs, ["body"]))
        or _pick_first_nonempty(ev, ["body", "email_body", "content", "message"])
    )

    senders_raw = _first_attr(attrs, ["from", "sender", "senders", "from_email"])
    if not senders_raw:
        senders_raw = ev.get("senders")
    if not senders_raw:
        senders_raw = [ev.get("sender") or ev.get("from") or ev.get("from_email")]
    senders = sorted({_basic_norm(_to_str(x)) for x in _to_list(senders_raw) if _to_str(x).strip()})

    attachments_raw = _first_attr(attrs, ["attachments", "attachment_hashes", "attachment_set"])
    if not attachments_raw:
        attachments_raw = (
            ev.get("attachments")
            or ev.get("attachment_hashes")
            or ev.get("attachment_set")
            or []
        )
    attachments = sorted({_basic_norm(_to_str(x)) for x in _to_list(attachments_raw) if _to_str(x).strip()})
    return EmailRecord(
        external_id=external_id,
        subject=subject,
        body=body,
        senders=tuple(senders),
        attachments=tuple(attachments),
    )


def _sig_strict(rec: EmailRecord) -> str:
    payload = {
        "subject": _basic_norm(rec.subject),
        "body": _basic_norm(rec.body),
        "senders": list(rec.senders),
        "attachments": list(rec.attachments),
    }
    return json.dumps(payload, sort_keys=True, ensure_ascii=False)


def _sig_content(rec: EmailRecord) -> str:
    payload = {
        "subject": _basic_norm(rec.subject),
        "body": _basic_norm(rec.body),
    }
    return json.dumps(payload, sort_keys=True, ensure_ascii=False)


def _sig_near(rec: EmailRecord) -> str:
    payload = {
        "subject": _aggressive_norm(rec.subject),
        "body": _aggressive_norm(rec.body),
        "senders": list(rec.senders),
    }
    return json.dumps(payload, sort_keys=True, ensure_ascii=False)


def _url_tokens_from_text(*texts: str) -> list[str]:
    """Canonical URL tokens from raw subject/body (cheap extraction, no fetching)."""
    found: set[str] = set()
    for text in texts:
        for m in URL_RE.finditer(_to_str(text)):
            c = _canonicalize_url(m.group(0))
            if c:
                found.add(c)
    return sorted(found)


def _sig_strict_task_message_identity(rec: EmailRecord) -> str:
    """
    Task-grounded collapse identity: same message/template for campaign detection.

    Includes (deterministic canonical JSON):
      - aggressively normalized subject/body (strips embedded timestamps, long hex/ints,
        normalizes inline URLs to url: tokens)
      - sorted basic-normalized senders
      - sorted basic-normalized attachment hashes/names
      - sorted canonical URL tokens from subject+body

    Explicitly excludes from the signature (never read for identity):
      - receiver / recipient fields
      - routing / transit metadata attributes
      - event timestamps as scalar fields
    """
    payload = {
        "subject": _aggressive_norm(rec.subject),
        "body": _aggressive_norm(rec.body),
        "senders": list(rec.senders),
        "attachments": list(rec.attachments),
        "url_tokens": _url_tokens_from_text(rec.subject, rec.body),
    }
    return json.dumps(payload, sort_keys=True, ensure_ascii=False)


SIGNATURE_STRICT_FULL_EMAIL = "strict_full_email"
SIGNATURE_CONTENT_SUBJECT_BODY = "content_subject_body"
SIGNATURE_NEAR_TEMPLATE = "near_template_subject_body_sender"
SIGNATURE_STRICT_TASK_MESSAGE = "strict_task_message_identity"

ANALYSIS_SIGNATURE_ORDER: tuple[str, ...] = (
    SIGNATURE_STRICT_FULL_EMAIL,
    SIGNATURE_STRICT_TASK_MESSAGE,
    SIGNATURE_CONTENT_SUBJECT_BODY,
    SIGNATURE_NEAR_TEMPLATE,
)

COLLAPSE_SIGNATURE_CHOICES: tuple[str, ...] = (
    SIGNATURE_STRICT_FULL_EMAIL,
    SIGNATURE_STRICT_TASK_MESSAGE,
)

SIGNATURE_FN_BY_NAME: dict[str, Callable[[EmailRecord], str]] = {
    SIGNATURE_STRICT_FULL_EMAIL: _sig_strict,
    SIGNATURE_STRICT_TASK_MESSAGE: _sig_strict_task_message_identity,
    SIGNATURE_CONTENT_SUBJECT_BODY: _sig_content,
    SIGNATURE_NEAR_TEMPLATE: _sig_near,
}

SIGNATURE_DESCRIPTIONS: dict[str, str] = {
    SIGNATURE_STRICT_FULL_EMAIL: (
        "Basic-normalized subject, body, senders, attachments (byte-exact strict)."
    ),
    SIGNATURE_STRICT_TASK_MESSAGE: (
        "Aggressive subject/body norm + senders + attachments + URL tokens; "
        "ignores delivery-instance noise (timestamps/receivers/routing not in signature)."
    ),
    SIGNATURE_CONTENT_SUBJECT_BODY: "Basic subject + body only.",
    SIGNATURE_NEAR_TEMPLATE: "Aggressive subject + body + senders.",
}


def resolve_signature_fn(name: str) -> Callable[[EmailRecord], str]:
    key = str(name).strip().lower()
    fn = SIGNATURE_FN_BY_NAME.get(key)
    if fn is None:
        raise ValueError(
            f"Unknown signature type: {name!r}; choices: {sorted(SIGNATURE_FN_BY_NAME)}"
        )
    return fn


def resolve_collapse_signature_type(name: str) -> str:
    key = str(name).strip().lower()
    if key not in COLLAPSE_SIGNATURE_CHOICES:
        raise ValueError(
            f"Unknown collapse_signature_type: {name!r}; "
            f"choices: {list(COLLAPSE_SIGNATURE_CHOICES)}"
        )
    return key


def default_collapse_paths(
    signature_type: str,
    *,
    project_root: Path | None = None,
) -> dict[str, Path]:
    """Default output paths per collapse signature (avoid overwriting strict artifacts)."""
    root = project_root or Path(__file__).resolve().parents[2]
    misp = root / "data" / "misp"
    sig = resolve_collapse_signature_type(signature_type)
    if sig == SIGNATURE_STRICT_TASK_MESSAGE:
        stem = "dedup_task_identity"
        dir_name = "misp_lake_dedup_task_identity"
    else:
        stem = "dedup_strict"
        dir_name = "misp_lake_dedup_strict"
    return {
        "out_json": misp / f"incidents-lake-misp.{stem}.json",
        "out_dir": misp / dir_name,
        "ground_truth_out_default": root / "data" / "groundtruth" / f"ground_truth.{stem}.json",
    }


def _percentiles(values: list[int]) -> dict[str, float]:
    if not values:
        return {"p50": 0.0, "p90": 0.0, "p95": 0.0}
    arr = sorted(int(v) for v in values)
    n = len(arr)

    def _pct(p: float) -> float:
        if n == 1:
            return float(arr[0])
        idx = min(n - 1, max(0, int(round(p * (n - 1)))))
        return float(arr[idx])

    return {"p50": _pct(0.50), "p90": _pct(0.90), "p95": _pct(0.95)}


def analyze_signature_duplicate_burden(
    records: list[EmailRecord],
    sig_fn: Callable[[EmailRecord], str],
    *,
    signature_name: str,
    top_k: int = 25,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Duplicate-group summary + per-group rows (size >= 2), shared by analyze/collapse."""
    groups: dict[str, list[EmailRecord]] = defaultdict(list)
    for rec in records:
        groups[sig_fn(rec)].append(rec)

    group_sizes = [len(v) for v in groups.values()]
    duplicate_sizes = [s for s in group_sizes if s >= 2]
    duplicate_groups = [(k, groups[k]) for k in groups.keys() if len(groups[k]) >= 2]
    duplicate_groups.sort(key=lambda kv: len(kv[1]), reverse=True)

    total_emails = len(records)
    all_possible_pairs = _safe_edges_from_size(total_emails)
    easy_edges_total = sum(_safe_edges_from_size(len(v)) for _, v in duplicate_groups)
    emails_in_dup = int(sum(duplicate_sizes))
    n_non_singleton_clusters = len(duplicate_groups)

    hist: dict[str, int] = defaultdict(int)
    for s in group_sizes:
        hist[_histogram_bucket(s)] += 1
    hist_dup: dict[str, int] = defaultdict(int)
    for s in duplicate_sizes:
        hist_dup[_histogram_bucket(s)] += 1

    top_preview: list[dict[str, Any]] = []
    for rank, (sig, members) in enumerate(duplicate_groups[:top_k], start=1):
        top_preview.append(
            {
                "signature_type": signature_name,
                "rank": rank,
                "signature_hash12": _sha12(sig),
                "group_size": len(members),
                "easy_edges_if_all_paired": _safe_edges_from_size(len(members)),
                "sample_external_ids": [m.external_id for m in members[:8]],
                "sample_subject": _normalize_space(_to_str(members[0].subject))[:180],
            }
        )

    summary = {
        "signature_type": signature_name,
        "n_emails_total": total_emails,
        "n_groups_total": int(len(groups)),
        "n_duplicate_groups_size_ge_2": int(n_non_singleton_clusters),
        "n_emails_in_duplicate_groups": emails_in_dup,
        "fraction_emails_in_duplicate_groups": (
            float(emails_in_dup / total_emails) if total_emails else 0.0
        ),
        "fraction_emails_in_non_singleton_clusters": (
            float(emails_in_dup / total_emails) if total_emails else 0.0
        ),
        "max_group_size": int(max(group_sizes) if group_sizes else 0),
        "median_group_size": float(
            sorted(group_sizes)[len(group_sizes) // 2] if group_sizes else 0.0
        ),
        "cluster_size_percentiles_duplicate_groups_only": _percentiles(duplicate_sizes),
        "all_possible_pairs_n_choose_2": int(all_possible_pairs),
        "estimated_easy_edges_from_duplicate_groups": int(easy_edges_total),
        "easy_edge_fraction_of_all_possible_pairs": (
            float(easy_edges_total / all_possible_pairs) if all_possible_pairs else 0.0
        ),
        "group_size_histogram_all_groups": dict(sorted(hist.items())),
        "group_size_histogram_duplicate_groups_only": dict(sorted(hist_dup.items())),
        "top_duplicate_groups_preview": top_preview,
    }

    rows: list[dict[str, Any]] = []
    for rank, (sig, members) in enumerate(duplicate_groups, start=1):
        rows.append(
            {
                "signature_type": signature_name,
                "rank_by_size": rank,
                "signature_hash12": _sha12(sig),
                "group_size": len(members),
                "easy_edges_if_all_paired": _safe_edges_from_size(len(members)),
                "sample_external_ids": "|".join(m.external_id for m in members[:12]),
                "sample_subject": _normalize_space(_to_str(members[0].subject))[:220],
            }
        )
    return summary, rows


def load_external_id_to_gt_label(ground_truth_path: Path) -> dict[str, str]:
    """Map external_id -> GT campaign label (first cluster key containing the id)."""
    data = json.loads(ground_truth_path.expanduser().resolve().read_text(encoding="utf-8"))
    clusters = data.get("clusters")
    if not isinstance(clusters, dict):
        raise TypeError("ground_truth.json: expected top-level 'clusters' object")
    out: dict[str, str] = {}
    for label, rows in clusters.items():
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, dict):
                continue
            eid = str(row.get("external_id", "")).strip()
            if eid and eid not in out:
                out[eid] = str(label)
    return out


def evaluate_collapse_clusters_against_ground_truth(
    collapsed_clusters: list[dict[str, Any]],
    *,
    ground_truth_path: Path,
    signature_type: str,
) -> dict[str, Any]:
    """
    Purity-first GT diagnostics on duplicate collapse clusters (size >= 2).

    Uses sklearn H/C/V when available; dominant-label purity per collapse cluster.
    """
    gt_path = ground_truth_path.expanduser().resolve()
    label_map = load_external_id_to_gt_label(gt_path)

    true_labels: list[str] = []
    pred_labels: list[str] = []
    cluster_purities: list[float] = []
    n_clusters_eval = 0
    n_members_with_gt = 0
    n_members_total = 0

    for cluster in collapsed_clusters:
        members = list(cluster.get("member_external_ids") or [])
        if len(members) < 2:
            continue
        cid = str(cluster.get("cluster_id") or cluster.get("signature_hash12") or "")
        gt_counts: Counter[str] = Counter()
        for eid in members:
            n_members_total += 1
            gt = label_map.get(str(eid))
            if gt is None:
                continue
            n_members_with_gt += 1
            gt_counts[gt] += 1
            true_labels.append(gt)
            pred_labels.append(cid)

        if gt_counts:
            n_clusters_eval += 1
            cluster_purities.append(gt_counts.most_common(1)[0][1] / sum(gt_counts.values()))

    out: dict[str, Any] = {
        "ground_truth_file": str(gt_path),
        "collapse_signature_type": signature_type,
        "n_collapse_clusters_size_ge_2": int(len(collapsed_clusters)),
        "n_collapse_clusters_with_any_gt_member": n_clusters_eval,
        "n_member_rows_in_duplicate_clusters": n_members_total,
        "n_member_rows_with_gt_label": n_members_with_gt,
        "gt_coverage_fraction_in_duplicate_clusters": (
            float(n_members_with_gt / n_members_total) if n_members_total else 0.0
        ),
    }

    if len(true_labels) < 2:
        out["note"] = "Too few GT-labeled members in duplicate collapse clusters for H/C/V."
        return out

    if homogeneity_completeness_v_measure is None:
        out["note"] = "sklearn not installed; purity stats only."
    else:
        import numpy as np

        pred_arr = np.array(pred_labels, dtype=object)
        true_arr = np.array(true_labels, dtype=object)
        uniq = sorted(set(pred_labels))
        remap = {p: i for i, p in enumerate(uniq)}
        pred_int = np.array([remap[p] for p in pred_labels], dtype=np.int64)
        h, co, vm = homogeneity_completeness_v_measure(true_arr, pred_int)
        out["homogeneity"] = float(h)
        out["completeness"] = float(co)
        out["v_measure"] = float(vm)
        out["interpretation"] = (
            "Conservative dedup clusters should be high-purity (homogeneity); "
            "lower completeness is acceptable when compressing delivery-instance redundancy."
        )

    if cluster_purities:
        import numpy as np

        pur = np.array(cluster_purities, dtype=np.float64)
        out["dominant_gt_purity"] = {
            "mean": float(pur.mean()),
            "median": float(np.median(pur)),
            "fraction_clusters_purity_ge_0.95": float(np.mean(pur >= 0.95)),
            "fraction_clusters_purity_ge_0.99": float(np.mean(pur >= 0.99)),
            "n_clusters_with_gt_purity_computed": int(len(cluster_purities)),
        }
    return out


def _safe_edges_from_size(n: int) -> int:
    if n < 2:
        return 0
    return (n * (n - 1)) // 2


def _histogram_bucket(size: int) -> str:
    if size == 1:
        return "1"
    if size == 2:
        return "2"
    if 3 <= size <= 5:
        return "3-5"
    if 6 <= size <= 10:
        return "6-10"
    if 11 <= size <= 25:
        return "11-25"
    return "26+"


def _sha12(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:12]


def _cluster_id_full(sig: str) -> str:
    """Collision-safe id for the canonical signature string (same sig string => same cluster)."""
    return hashlib.sha256(sig.encode("utf-8", errors="replace")).hexdigest()


def _scan_top_level_object_slices(text: str) -> list[tuple[int, int]]:
    """
    Return [start,end) slices for top-level objects inside a JSON array.
    Works even if some objects are malformed, as long as braces/quotes are balanced enough.
    """
    out: list[tuple[int, int]] = []
    n = len(text)
    i = text.find("[")
    if i < 0:
        return out
    depth = 0
    in_str = False
    esc = False
    obj_start = -1
    while i < n:
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == "\"":
                in_str = False
            i += 1
            continue
        if ch == "\"":
            in_str = True
            i += 1
            continue
        if ch == "{":
            if depth == 1 and obj_start < 0:
                obj_start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 1 and obj_start >= 0:
                out.append((obj_start, i + 1))
                obj_start = -1
        elif ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth <= 0:
                break
        i += 1
    return out


def _load_records(input_json: Path, max_events: int | None) -> list[EmailRecord]:
    _, recs = _load_raw_events_and_records(input_json, max_events=max_events)
    return recs


def _load_raw_events_and_records(
    input_json: Path, max_events: int | None
) -> tuple[list[dict[str, Any]], list[EmailRecord]]:
    """
    Load top-level JSON array elements as raw dicts plus parsed EmailRecords (same order).
    Non-dict elements are skipped (robustness). Index i for _extract_email_record matches
    enumerate(payload) in the strict JSON path (same as historical _load_records behavior
    when all entries are dicts).
    """
    text = input_json.read_text(encoding="utf-8")
    raw_out: list[dict[str, Any]] = []
    rec_out: list[EmailRecord] = []

    try:
        payload = json.loads(text)
        if not isinstance(payload, list):
            raise TypeError(f"Expected top-level JSON array; got {type(payload).__name__}")
        for i, raw in enumerate(payload):
            if max_events is not None and i >= max_events:
                break
            if not isinstance(raw, dict):
                continue
            raw_out.append(raw)
            rec_out.append(_extract_email_record(raw, i))
        return raw_out, rec_out
    except json.JSONDecodeError:
        slices = _scan_top_level_object_slices(text)
        for i, (s, e) in enumerate(slices):
            if max_events is not None and len(raw_out) >= max_events:
                break
            frag = text[s:e]
            try:
                raw = json.loads(frag)
            except Exception:
                continue
            if not isinstance(raw, dict):
                continue
            raw_out.append(raw)
            rec_out.append(_extract_email_record(raw, i))
        if not raw_out:
            raise
        return raw_out, rec_out
