"""
Build ``visualization/data.json`` for the cluster inspection web UI.

Discovers every ``campaigns*.json`` artifact in the run directory (recursively),
joins them with email content from MISP JSON, and emits one entry per file under
``solutions`` so the webapp can render a tab per discovered solution.
"""
from __future__ import annotations

import ast
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

_CORE_ROOT = Path(__file__).resolve().parents[1]
if str(_CORE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CORE_ROOT))

from graph.common import parse_misp_events  # noqa: E402
from preprocessing.utils.defang import defang_url_string, refang_url_string  # noqa: E402
from preprocessing.utils.url_extractor import deduplicate_urls, extract_urls_from_text  # noqa: E402

_VIZ_DIRNAME = "visualization"
_CAMPAIGNS_GLOB = "campaigns*.json"


def _load_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_misp_events_list(misp_json_path: str) -> list[dict[str, Any]]:
    with open(misp_json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict):
        ev = raw.get("Events") or raw.get("response", {}).get("Event", [])
        if isinstance(ev, list):
            return ev
        if isinstance(ev, dict):
            return [ev]
    return []


def _expand_url_value(raw: Any) -> list[str]:
    if isinstance(raw, list):
        out: list[str] = []
        for item in raw:
            out.extend(_expand_url_value(item))
        return out
    text = str(raw).strip()
    if not text:
        return []
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, (list, tuple)):
                out = []
                for item in parsed:
                    out.extend(_expand_url_value(item))
                return out
        except (ValueError, SyntaxError):
            pass
    return extract_urls_from_text(text)


def _urls_for_email_row(ev: dict[str, Any]) -> list[str]:
    """Merge MISP url attributes with body/html extraction; refang defanged schemes."""
    seen: set[str] = set()
    out: list[str] = []

    def add(raw: Any) -> None:
        for candidate in _expand_url_value(raw):
            u = refang_url_string(candidate.strip())
            if not u or u in seen:
                continue
            seen.add(u)
            out.append(u)

    for u in ev.get("urls") or []:
        add(u)

    body = ev.get("body") or ""
    if isinstance(body, str) and body.strip():
        for u in extract_urls_from_text(body):
            add(u)

    html = ev.get("html")
    if isinstance(html, str) and html.strip():
        for u in extract_urls_from_text(html):
            add(u)
    elif isinstance(html, dict):
        for part in html.values():
            if isinstance(part, str) and part.strip():
                for u in extract_urls_from_text(part):
                    add(u)

    deduped = deduplicate_urls(out)
    return sorted((defang_url_string(u) for u in deduped), key=str.casefold)


def build_email_catalog_from_misp(misp_json_path: str) -> dict[str, dict[str, Any]]:
    """external_id -> display fields for the UI."""
    events = _load_misp_events_list(misp_json_path)
    parsed = parse_misp_events(events)
    out: dict[str, dict[str, Any]] = {}
    for ev in parsed:
        eid = str(ev.get("external_id") or "").strip()
        if not eid:
            continue
        out[eid] = {
            "external_id": eid,
            "subject": ev.get("subject") or "",
            "date": ev.get("date") or "",
            "senders": list(ev.get("senders") or []),
            "receivers": list(ev.get("receivers") or []),
            "body": ev.get("body") or "",
            "urls": _urls_for_email_row(ev),
            "email_info": ev.get("email_info") or "",
        }
    return out


def _campaign_email_count(campaign: dict[str, Any]) -> int:
    size = campaign.get("size")
    if size is not None:
        try:
            return int(size)
        except (TypeError, ValueError):
            pass
    return len(campaign.get("member_external_ids") or [])


def _sort_campaigns_by_size(campaigns: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(campaigns, key=_campaign_email_count, reverse=True)


def _enrich_campaign_noise_metadata(payload: dict[str, Any]) -> None:
    """
    Mark singleton campaigns as noise (PU and similar pipelines emit one-email
    clusters instead of ``n_noise``). Sets per-campaign ``is_noise`` and totals
    for the visualization summary.
    """
    camps = payload.get("campaigns") or []
    try:
        n_labeled_noise = int(payload.get("n_noise") or 0)
    except (TypeError, ValueError):
        n_labeled_noise = 0
    n_labeled_noise = max(0, n_labeled_noise)

    n_singleton = 0
    n_non_noise = 0
    for camp in camps:
        sz = _campaign_email_count(camp)
        is_singleton = sz == 1
        camp["is_noise"] = is_singleton
        if is_singleton:
            n_singleton += 1
        else:
            n_non_noise += 1

    payload["n_singleton_campaigns"] = n_singleton
    payload["n_noise_labeled"] = n_labeled_noise
    payload["n_noise_total"] = n_labeled_noise + n_singleton
    payload["n_campaigns_non_noise"] = n_non_noise


def _strip_campaigns_payload(raw: dict[str, Any] | None) -> dict[str, Any] | None:
    if not raw:
        return None
    camps = _sort_campaigns_by_size(list(raw.get("campaigns") or []))
    if not camps:
        return None
    payload: dict[str, Any] = {
        "solution": raw.get("solution"),
        "algorithm": raw.get("algorithm"),
        "model": raw.get("model"),
        "feature_set": raw.get("feature_set"),
        "n_components": raw.get("n_components"),
        "params": raw.get("params") or {},
        "metrics": raw.get("metrics") or {},
        "n_campaigns": raw.get("n_campaigns", len(camps)),
        "n_noise": raw.get("n_noise"),
        "campaigns": camps,
    }
    _enrich_campaign_noise_metadata(payload)
    return payload


def _discover_campaigns_files(run_dir: Path) -> list[Path]:
    """All ``campaigns*.json`` files under ``run_dir`` excluding the visualization output."""
    out: list[Path] = []
    for p in sorted(run_dir.rglob(_CAMPAIGNS_GLOB)):
        if not p.is_file():
            continue
        try:
            rel_parts = p.relative_to(run_dir).parts
        except ValueError:
            rel_parts = ()
        if rel_parts and rel_parts[0] == _VIZ_DIRNAME:
            continue
        out.append(p)
    return out


def _solutions_from_run_dir(run_dir: Path) -> dict[str, dict[str, Any]]:
    """
    Map ``<rel-path-from-run-dir>`` -> stripped campaigns payload, with a
    UI ``label`` field set to the JSON file name (collisions disambiguated by
    appending the parent directory name).
    """
    solutions: dict[str, dict[str, Any]] = {}
    label_counts: dict[str, int] = defaultdict(int)

    for path in _discover_campaigns_files(run_dir):
        raw = _load_json_if_exists(path)
        payload = _strip_campaigns_payload(raw)
        if payload is None:
            continue
        rel = path.relative_to(run_dir).as_posix()
        label = path.name
        payload["label"] = label
        payload["file"] = rel
        solutions[rel] = payload
        label_counts[label] += 1

    for rel, payload in solutions.items():
        if label_counts[payload["label"]] > 1:
            parent = Path(rel).parent.name or "root"
            payload["label"] = f"{payload['label']} ({parent})"

    return solutions


def _load_ground_truth_labels(ground_truth_path: str | Path | None) -> dict[str, Any] | None:
    if not ground_truth_path:
        return None
    path = Path(ground_truth_path)
    if not path.is_file():
        return None
    try:
        from clustering.clusteringMetrics import extract_ground_truth_labels
    except ModuleNotFoundError:
        from core.clustering.clusteringMetrics import extract_ground_truth_labels
    return extract_ground_truth_labels(str(path.resolve()))


def _attach_ground_truth_to_emails(
    emails: dict[str, dict[str, Any]],
    ground_truth: dict[str, Any],
) -> list[str]:
    """Set ``ground_truth_label`` on catalog rows; return sorted external ids with GT."""
    gt_ids: list[str] = []
    for eid, label in ground_truth.items():
        key = str(eid)
        gt_ids.append(key)
        row = emails.get(key)
        if row is None and eid in emails:
            row = emails[eid]
        if isinstance(row, dict):
            row["ground_truth_label"] = label
    return sorted(gt_ids, key=str)


def _enrich_solutions_with_campaign_eval(
    solutions: dict[str, dict[str, Any]],
    ground_truth: dict[str, Any],
) -> None:
    try:
        from visualization.campaign_eval_metrics import enrich_campaigns_with_eval_metrics
    except ModuleNotFoundError:
        from core.visualization.campaign_eval_metrics import enrich_campaigns_with_eval_metrics

    for payload in solutions.values():
        camps = payload.get("campaigns")
        if isinstance(camps, list) and camps:
            enrich_campaigns_with_eval_metrics(camps, ground_truth)


def build_visualization_data(
    *,
    run_dir: str | Path,
    misp_json_path: str,
    include_attribute_similarity: bool = True,
    ground_truth_path: str | Path | None = None,
) -> dict[str, Any]:
    """
    Assemble the webapp data file by discovering every ``campaigns*.json``
    artifact under ``run_dir`` and joining members with email content.

    Output shape:
    - ``solutions``: { <rel_path>: { label, file, campaigns, ... } }
    - ``emails``: external_id -> email fields (includes per-email ``urls``)
    - ``attribute_similarity``: optional { <solution>: { <cid>: { <eid>: { attr: score } } } }
    - ``url_similarity``: optional { <solution>: { <cid>: { <url>: score } } }
    - per-campaign ``homogeneity``, ``completeness``, ``v_measure`` when ``ground_truth_path`` resolves
    """
    run_dir = Path(run_dir)
    print(f"[viz] Building email catalog from MISP: {misp_json_path}", flush=True)
    emails = build_email_catalog_from_misp(misp_json_path)
    print(f"[viz] Loaded {len(emails)} emails; discovering campaigns under {run_dir}", flush=True)
    solutions = _solutions_from_run_dir(run_dir)
    print(f"[viz] Found {len(solutions)} solution file(s)", flush=True)

    gt_resolved: str | None = None
    ground_truth: dict[str, Any] | None = None
    if ground_truth_path:
        gt_path = Path(ground_truth_path)
        if gt_path.is_file():
            gt_resolved = str(gt_path.resolve())
            ground_truth = _load_ground_truth_labels(gt_path)

    payload: dict[str, Any] = {
        "run_dir": str(run_dir.resolve()),
        "misp_json_path": str(Path(misp_json_path).resolve()),
        "solutions": solutions,
        "emails": emails,
        "campaign_eval_metrics_available": bool(ground_truth),
    }
    if gt_resolved:
        payload["ground_truth_path"] = gt_resolved
    if ground_truth:
        payload["ground_truth_ids"] = _attach_ground_truth_to_emails(emails, ground_truth)
        _enrich_solutions_with_campaign_eval(solutions, ground_truth)

    if include_attribute_similarity and solutions:
        try:
            from visualization.attribute_similarity import build_campaign_similarity_sidecars
        except ModuleNotFoundError:
            from core.visualization.attribute_similarity import (
                build_campaign_similarity_sidecars,
            )

        try:
            attr_sim, url_sim = build_campaign_similarity_sidecars(
                solutions=solutions,
                emails=emails,
            )
            if attr_sim:
                payload["attribute_similarity"] = attr_sim
            if url_sim:
                payload["url_similarity"] = url_sim
        except Exception as exc:
            payload["attribute_similarity_error"] = str(exc)

    return payload


def write_visualization_data_json(
    *,
    run_dir: str | Path,
    misp_json_path: str,
    out_name: str = "data.json",
    include_attribute_similarity: bool = True,
    ground_truth_path: str | Path | None = None,
) -> Path:
    run_dir = Path(run_dir)
    payload = build_visualization_data(
        run_dir=run_dir,
        misp_json_path=misp_json_path,
        include_attribute_similarity=include_attribute_similarity,
        ground_truth_path=ground_truth_path,
    )
    viz_dir = run_dir / _VIZ_DIRNAME
    viz_dir.mkdir(parents=True, exist_ok=True)
    out_path = viz_dir / out_name
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return out_path
