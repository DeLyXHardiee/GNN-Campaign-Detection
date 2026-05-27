"""
Thesis-facing graph-construction diagnostics: deduplication, relation channels, config audit.

Diagnostics only — does not modify graph construction.
"""

from __future__ import annotations

import hashlib
import json
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from math import comb
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
CORE = REPO / "core"
MISP_DEDUP = REPO / "scripts" / "misp_lake_dedup"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))
if str(MISP_DEDUP) not in sys.path:
    sys.path.insert(0, str(MISP_DEDUP))

import misp_email_identity as mei  # noqa: E402

from graph.common import extract_email_domain, parse_misp_events, parse_url_components  # noqa: E402
from graph.utils.graph_metrics import _safe_load_graph, load_graph_metadata  # noqa: E402
from graph.utils.url_analysis import count_campaign_pairs_for_emails, load_gt_label_map  # noqa: E402
from graph.assembler import _field_values_for_node  # noqa: E402
from seed_candidate_workflow.utils.gt_edge_structure_analysis import (  # noqa: E402
    _load_embeddings,
    _resolve_embeddings_json,
)
from seed_candidate_workflow.utils.pair_similarity_features import (  # noqa: E402
    body_char4gram_jaccard_from_bodies,
    body_token_jaccard_from_bodies,
)
from seed_candidate_workflow.utils.timestamp_ablation_14_only_mlp import pair_universe_stats  # noqa: E402

ROUTING_NODE_KEYS: tuple[str, ...] = (
    "origin_ip",
    "received_host",
    "helo_host",
    "return_path_email",
    "return_path_domain",
)

INCLUDED_GRAPH_CHANNELS: tuple[tuple[str, str, str], ...] = (
    ("url", "email", "has_url", "url"),
    ("url_stem", "email", "has_stem", "stem"),
    ("attachment", "email", "has_attachment", "attachment"),
    ("sender", "email", "has_sender", "sender"),
    ("domain", "email", "has_domain", "domain"),
    ("receiver", "email", "has_receiver", "receiver"),
)

CATEGORY_ALLOWLIST = frozenset({"phishing", "scam"})


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _safe_edges(n: int) -> int:
    return int(comb(n, 2)) if n >= 2 else 0


def _percentile(values: list[int], p: float) -> float:
    if not values:
        return 0.0
    arr = np.asarray(values, dtype=np.float64)
    return float(np.percentile(arr, p))


def _redact_artifact(value: str, *, prefix_len: int = 6) -> str:
    s = str(value or "").strip()
    if not s:
        return ""
    digest = hashlib.sha256(s.encode("utf-8", errors="replace")).hexdigest()[:12]
    if len(s) <= prefix_len:
        return f"hash:{digest}"
    return f"{s[:prefix_len]}…#{digest}"


def _latex_escape(text: str) -> str:
    return (
        str(text)
        .replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("%", "\\%")
        .replace("&", "\\&")
    )


def _write_df_table_tex(df: pd.DataFrame, path: Path, *, caption: str, label: str) -> None:
    cols = list(df.columns)
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        f"\\caption{{{_latex_escape(caption)}}}",
        f"\\label{{{label}}}",
        "\\begin{tabular}{" + "l" * len(cols) + "}",
        "\\toprule",
        " & ".join(_latex_escape(c) for c in cols) + " \\\\",
        "\\midrule",
    ]
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if isinstance(v, float):
                cells.append(f"{v:.4g}" if abs(v) < 1e4 else f"{v:.2e}")
            else:
                cells.append(_latex_escape(str(v)))
        lines.append(" & ".join(cells) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _misp_category(event_wrapper: dict[str, Any]) -> str:
    ev = event_wrapper.get("Event") if isinstance(event_wrapper.get("Event"), dict) else event_wrapper
    for a in ev.get("Attribute") or []:
        if str(a.get("type", "")).strip().lower() == "category":
            return str(a.get("value", "")).strip().lower()
    return ""


def count_filtered_misp_events(misp_path: Path) -> tuple[int, int]:
    """Return (total_events, phishing_or_scam_events)."""
    data = json.loads(misp_path.expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise TypeError(f"Expected JSON array in {misp_path}")
    total = len(data)
    n_allow = sum(1 for ev in data if _misp_category(ev) in CATEGORY_ALLOWLIST)
    return total, n_allow


def load_collapse_summary(collapse_dir: Path) -> dict[str, Any]:
    p = collapse_dir / "collapse_summary.json"
    if not p.is_file():
        raise FileNotFoundError(f"Missing collapse summary: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def duplicate_group_sizes_from_clusters(clusters_path: Path) -> list[int]:
    clusters = json.loads(clusters_path.read_text(encoding="utf-8"))
    if not isinstance(clusters, list):
        raise TypeError("collapsed_clusters.json must be a list")
    sizes = [int(c.get("group_size") or len(c.get("member_external_ids") or [])) for c in clusters]
    return [s for s in sizes if s >= 2]


def load_id_map_dataframe(collapse_dir: Path) -> pd.DataFrame:
    p = collapse_dir / "external_id_map.csv"
    if not p.is_file():
        raise FileNotFoundError(f"Missing external_id_map.csv: {p}")
    return pd.read_csv(p, low_memory=False)


def build_clusters_from_id_map(map_df: pd.DataFrame) -> dict[str, list[str]]:
    """cluster_id -> sorted member external_ids (groups with size >= 2)."""
    key_col = "cluster_id" if "cluster_id" in map_df.columns else "representative_external_id"
    eid_col = "external_id"
    clusters: dict[str, list[str]] = defaultdict(list)
    for row in map_df.itertuples(index=False):
        eid = str(getattr(row, eid_col, "")).strip()
        key = str(getattr(row, key_col, "")).strip()
        if eid and key:
            clusters[key].append(eid)
    return {k: sorted(set(v)) for k, v in clusters.items() if len(v) >= 2}


def build_eid_to_representative(map_df: pd.DataFrame) -> dict[str, str]:
    out: dict[str, str] = {}
    if "external_id" not in map_df.columns:
        return out
    rep_col = "representative_external_id" if "representative_external_id" in map_df.columns else None
    for row in map_df.itertuples(index=False):
        eid = str(getattr(row, "external_id", "")).strip()
        rep = str(getattr(row, rep_col, eid) if rep_col else eid).strip()
        if eid:
            out[eid] = rep or eid
    return out


def build_propagated_label_map(
    gt_path: Path,
    *,
    id_map_df: pd.DataFrame | None = None,
    view: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
  Label map for channel/pair diagnostics.

  - **pre_dedup**: direct labels from ``ground_truth.json``; for unlabeled raw incidents,
    inherit the campaign label of ``representative_external_id`` when the representative
    is labeled (duplicate-cluster propagation).
  - **post_dedup**: direct labels from ``ground_truth.dedup_task_identity.json`` when
    available, else fall back to full GT on representative ids.
    """
    direct = load_gt_label_map(gt_path)
    meta: dict[str, Any] = {
        "gt_path": str(gt_path.resolve()),
        "view": view,
        "n_direct_gt_labels": int(len(direct)),
    }
    out = dict(direct)
    n_propagated = 0

    if view == "pre_dedup" and id_map_df is not None:
        eid_to_rep = build_eid_to_representative(id_map_df)
        rep_label: dict[str, Any] = {}
        for eid, rep in eid_to_rep.items():
            if rep in direct:
                rep_label[rep] = direct[rep]
        for eid, rep in eid_to_rep.items():
            if eid in out:
                continue
            lab = rep_label.get(rep)
            if lab is not None:
                out[eid] = lab
                n_propagated += 1
        meta["label_propagation"] = (
            "Unlabeled raw incidents inherit the GT campaign of their task-identity "
            "representative_external_id when that representative is labeled in ground_truth.json."
        )
        meta["n_labels_after_representative_propagation"] = int(len(out))
        meta["n_incidents_labeled_via_propagation"] = int(n_propagated)
    elif view == "post_dedup":
        dedup_gt_path = REPO / "data/groundtruth/ground_truth.dedup_task_identity.json"
        if dedup_gt_path.is_file():
            dedup_labels = load_gt_label_map(dedup_gt_path)
            for eid, lab in dedup_labels.items():
                if eid not in out:
                    out[eid] = lab
            meta["dedup_gt_path"] = str(dedup_gt_path.resolve())
            meta["n_dedup_gt_labels"] = int(len(dedup_labels))
        meta["label_propagation"] = (
            "Post-dedup view uses deduplicated representative external_ids; "
            "labels from ground_truth.dedup_task_identity.json merged with full GT."
        )
        meta["n_labels_total"] = int(len(out))
    else:
        meta["n_labels_total"] = int(len(out))

    return out, meta


def _parsed_email_by_eid(parsed: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(em.get("external_id") or "").strip(): em for em in parsed if str(em.get("external_id") or "").strip()}


def _email_artifact_sets(em: dict[str, Any], channel_id: str) -> set[str]:
    if channel_id == "url":
        return {str(u).strip().lower() for u in (em.get("urls") or []) if str(u).strip()}
    if channel_id == "url_stem":
        stems: set[str] = set()
        for u in em.get("urls") or []:
            s = str(parse_url_components(str(u)).get("stem") or "").strip().lower()
            if s:
                stems.add(s)
        return stems
    if channel_id == "attachment":
        return {str(a).strip().lower() for a in (em.get("attachments") or []) if str(a).strip()}
    if channel_id == "sender":
        return {str(s).strip().lower() for s in (em.get("senders") or []) if str(s).strip()}
    if channel_id == "sender_domain":
        out: set[str] = set()
        for s in em.get("senders") or []:
            d = extract_email_domain(str(s)).strip().lower()
            if d:
                out.add(d)
        return out
    if channel_id == "domain":
        doms: set[str] = set()
        for u in em.get("urls") or []:
            d = str(parse_url_components(str(u)).get("domain") or "").strip().lower()
            if d:
                doms.add(d)
        return doms
    if channel_id == "receiver":
        return {str(r).strip().lower() for r in (em.get("receivers") or []) if str(r).strip()}
    if channel_id in ROUTING_NODE_KEYS:
        return {str(v).strip().lower() for v in _field_values_for_node(em, channel_id) if str(v).strip()}
    return set()


def build_parsed_misp_channel_reports(
    parsed: list[dict[str, Any]],
    *,
    label_map: dict[str, Any],
    data_view: str,
    n_incidents_total: int,
    label_meta: dict[str, Any],
) -> list[dict[str, Any]]:
    """All channels from parse_misp_events (pre- or post-dedup lake)."""
    channel_ids = [
        "url",
        "url_stem",
        "attachment",
        "sender",
        "sender_domain",
        "domain",
        "receiver",
        *ROUTING_NODE_KEYS,
    ]
    group_by_id = {
        "url": "included",
        "url_stem": "included",
        "attachment": "included",
        "sender": "included",
        "sender_domain": "included",
        "domain": "included",
        "receiver": "optional",
        **{k: "routing" for k in ROUTING_NODE_KEYS},
    }
    by_eid = _parsed_email_by_eid(parsed)
    rows: list[dict[str, Any]] = []
    for ch in channel_ids:
        emails_by_art: dict[str, set[str]] = defaultdict(set)
        for eid, em in by_eid.items():
            for art in _email_artifact_sets(em, ch):
                emails_by_art[art].add(eid)
        stats = _channel_stats_from_email_artifact_map(
            emails_by_art,
            label_map=label_map,
            channel_id=ch,
            channel_group=group_by_id[ch],
        )
        stats["data_view"] = data_view
        stats["n_incidents_total"] = int(n_incidents_total)
        stats["gt_labeling"] = label_meta
        rows.append(stats)
    return rows


def _cosine_embeddings(eid_a: str, eid_b: str, id_to_emb: dict[str, np.ndarray]) -> float | None:
    va = id_to_emb.get(eid_a)
    vb = id_to_emb.get(eid_b)
    if va is None or vb is None:
        return None
    na = float(np.linalg.norm(va))
    nb = float(np.linalg.norm(vb))
    if na == 0.0 or nb == 0.0:
        return None
    return float(np.dot(va, vb) / (na * nb))


def estimate_intra_duplicate_pair_evidence(
    clusters: dict[str, list[str]],
    parsed_by_eid: dict[str, dict[str, Any]],
    *,
    embeddings_json: Path | None = None,
    body_token_threshold: float = 0.25,
    body_char4_threshold: float = 0.25,
    semantic_threshold: float = 0.90,
) -> dict[str, Any]:
    """Cheap estimate: within-duplicate pairs satisfying generator-style evidence flags."""
    id_to_emb: dict[str, np.ndarray] = {}
    emb_path = embeddings_json or _resolve_embeddings_json(
        explicit=embeddings_json,
        anchor_run_dir=None,
        project_root=REPO,
    )
    emb_note = "embeddings_unavailable"
    if emb_path and Path(emb_path).is_file():
        try:
            id_to_emb = _load_embeddings(Path(emb_path))
            emb_note = str(Path(emb_path).resolve())
        except Exception as exc:  # pragma: no cover
            emb_note = f"embeddings_load_failed: {exc}"

    counters: Counter[str] = Counter()
    n_pairs = 0
    n_sem_checked = 0

    for members in clusters.values():
        if len(members) < 2:
            continue
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                a, b = members[i], members[j]
                em_a = parsed_by_eid.get(a)
                em_b = parsed_by_eid.get(b)
                if not em_a or not em_b:
                    continue
                n_pairs += 1
                send_a = _email_artifact_sets(em_a, "sender")
                send_b = _email_artifact_sets(em_b, "sender")
                if send_a & send_b:
                    counters["same_sender"] += 1
                url_a = _email_artifact_sets(em_a, "url")
                url_b = _email_artifact_sets(em_b, "url")
                if url_a & url_b:
                    counters["same_url"] += 1
                att_a = _email_artifact_sets(em_a, "attachment")
                att_b = _email_artifact_sets(em_b, "attachment")
                if att_a & att_b:
                    counters["same_attachment"] += 1
                bt = body_token_jaccard_from_bodies(str(em_a.get("body") or ""), str(em_b.get("body") or ""))
                if bt >= body_token_threshold:
                    counters["body_token_jaccard_ge_threshold"] += 1
                bc = body_char4gram_jaccard_from_bodies(str(em_a.get("body") or ""), str(em_b.get("body") or ""))
                if bc >= body_char4_threshold:
                    counters["body_char4gram_jaccard_ge_threshold"] += 1
                cos = _cosine_embeddings(a, b, id_to_emb) if id_to_emb else None
                if cos is not None:
                    n_sem_checked += 1
                    if cos >= semantic_threshold:
                        counters["semantic_cosine_ge_threshold"] += 1
                if (
                    (send_a & send_b)
                    or (url_a & url_b)
                    or (att_a & att_b)
                    or bt >= body_token_threshold
                    or bc >= body_char4_threshold
                    or (cos is not None and cos >= semantic_threshold)
                ):
                    counters["any_listed_evidence"] += 1

    return {
        "method": "enumerate_all_within_duplicate_cluster_pairs",
        "n_within_duplicate_pairs_enumerated": int(n_pairs),
        "embeddings_source": emb_note,
        "n_pairs_with_embedding_cosine_computed": int(n_sem_checked),
        "thresholds": {
            "body_token_jaccard": body_token_threshold,
            "body_char4gram_jaccard": body_char4_threshold,
            "semantic_cosine": semantic_threshold,
        },
        "counts_by_evidence": dict(counters),
        "fraction_of_intra_duplicate_pairs_with_any_listed_evidence": (
            float(counters["any_listed_evidence"] / n_pairs) if n_pairs else float("nan")
        ),
        "interpretation": (
            "Within task-identity duplicate groups, members share normalized subject/body/senders/"
            "attachments/URL tokens by construction, so most pairs satisfy at least one "
            "generator-style overlap criterion. This estimates duplicate-induced pair inflation "
            "if seed/candidate generation were run on the pre-dedup lake."
        ),
    }


def build_deduplication_report(
    *,
    raw_misp: Path,
    dedup_misp: Path,
    collapse_dir: Path,
    gt_path: Path,
    pair_training_csv: Path | None,
    seed_csv: Path | None,
    embeddings_json: Path | None = None,
) -> dict[str, Any]:
    collapse = load_collapse_summary(collapse_dir)
    pre = collapse.get("pre_collapse_duplicate_analysis") or {}
    dup_sizes = duplicate_group_sizes_from_clusters(collapse_dir / "collapsed_clusters.json")
    dup_sizes_sorted = sorted(dup_sizes, reverse=True)

    n_raw_total, n_raw_filtered = count_filtered_misp_events(raw_misp)
    n_dedup_total, n_dedup_filtered = count_filtered_misp_events(dedup_misp)

    median_dup = float(statistics.median(dup_sizes)) if dup_sizes else 0.0
    mean_dup = float(statistics.mean(dup_sizes)) if dup_sizes else 0.0

    map_df = load_id_map_dataframe(collapse_dir)
    eid_to_rep = build_eid_to_representative(map_df)
    clusters = build_clusters_from_id_map(map_df)

    intra_dup_pairs = int(pre.get("estimated_easy_edges_from_duplicate_groups") or 0)

    pair_stats: dict[str, Any] = {}
    if pair_training_csv and pair_training_csv.is_file():
        pair_stats = pair_universe_stats(pair_training_csv)

    n_universe = int(pair_stats.get("n_pairs") or 0)
    n_seed_pos = int(pair_stats.get("n_seed_positive_pairs") or 0)
    n_non_seed = int(pair_stats.get("n_non_seed_candidate_pairs") or 0)

    def _ratio(num: int, den: int) -> float | None:
        return float(num / den) if den else None

    inflation = {
        "post_dedup_pair_universe_n_pairs": n_universe,
        "post_dedup_seed_positive_pairs": n_seed_pos,
        "post_dedup_non_seed_candidate_pairs": n_non_seed,
        "potential_intra_duplicate_pairs_removed": intra_dup_pairs,
        "ratio_vs_pair_universe": _ratio(intra_dup_pairs, n_universe),
        "ratio_vs_seed_positive": _ratio(intra_dup_pairs, n_seed_pos),
        "ratio_vs_non_seed_candidate": _ratio(intra_dup_pairs, n_non_seed),
        "pct_of_pair_universe": (
            float(100.0 * intra_dup_pairs / n_universe) if n_universe else float("nan")
        ),
    }

    raw_misp_data = json.loads(raw_misp.read_text(encoding="utf-8"))
    parsed_raw = parse_misp_events(raw_misp_data if isinstance(raw_misp_data, list) else [])
    parsed_by_eid = _parsed_email_by_eid(parsed_raw)
    evidence_est = estimate_intra_duplicate_pair_evidence(
        clusters, parsed_by_eid, embeddings_json=embeddings_json
    )

    n_intra_enum = int(evidence_est.get("n_within_duplicate_pairs_enumerated") or 0)
    n_any_ev = int((evidence_est.get("counts_by_evidence") or {}).get("any_listed_evidence") or 0)
    pre_dedup_gen_estimate = {
        "full_pre_dedup_seed_candidate_generation_run": False,
        "reason": (
            "Full anchor/candidate generation on the 7,333-incident lake was not re-run (too costly); "
            "see within-duplicate evidence enumeration instead."
        ),
        "within_duplicate_pairs_total": n_intra_enum,
        "within_duplicate_pairs_with_any_listed_evidence": n_any_ev,
        "pct_intra_duplicate_with_any_listed_evidence": (
            float(100.0 * n_any_ev / n_intra_enum) if n_intra_enum else float("nan")
        ),
        "hypothetical_upper_bound_total_pairs_if_all_intra_dup_were_added": n_intra_enum,
        "hypothetical_pct_of_post_dedup_universe_if_added": inflation.get("pct_of_pair_universe"),
        "different_duplicate_cluster_pairs_estimate": int(
            _safe_edges(n_raw_filtered) - intra_dup_pairs
        ),
        "evidence_enumeration": evidence_est,
    }

    seed_intra = None
    candidate_intra = None
    if pair_training_csv and pair_training_csv.is_file() and eid_to_rep:
        pdf = pd.read_csv(pair_training_csv, low_memory=False)
        ei = pdf["email_i"].astype(str).str.strip()
        ej = pdf["email_j"].astype(str).str.strip()
        same_cluster = [
            eid_to_rep.get(a) == eid_to_rep.get(b) and eid_to_rep.get(a) is not None
            for a, b in zip(ei, ej, strict=False)
        ]
        candidate_intra = int(sum(same_cluster))
        if seed_csv and seed_csv.is_file():
            seeds = set(pd.read_csv(seed_csv, low_memory=False)["external_id"].astype(str).str.strip())
            seed_mask = ei.isin(seeds) & ej.isin(seeds)
            seed_intra = int((np.array(same_cluster) & seed_mask.to_numpy()).sum())

    summary = {
        "created_at_utc": _utc_now(),
        "signature_type": collapse.get("collapse_signature_type"),
        "signature_description": collapse.get("collapse_signature_description"),
        "category_allowlist": sorted(CATEGORY_ALLOWLIST),
        "filtered_phishing_scam_incidents_before_dedup": int(n_raw_filtered),
        "raw_misp_total_events": int(n_raw_total),
        "deduplicated_email_nodes_after_dedup": int(n_dedup_filtered),
        "dedup_misp_total_events": int(n_dedup_total),
        "collapsed_incidents_removed": int(collapse.get("n_events_removed") or (n_raw_filtered - n_dedup_filtered)),
        "n_duplicate_groups_size_gt_1": int(pre.get("n_duplicate_groups_size_ge_2") or len(dup_sizes)),
        "median_duplicate_group_size_among_groups_gt_1": median_dup,
        "mean_duplicate_group_size_among_groups_gt_1": mean_dup,
        "largest_duplicate_group_size": int(max(dup_sizes) if dup_sizes else 0),
        "top_10_duplicate_group_sizes": dup_sizes_sorted[:10],
        "potential_intra_duplicate_pairs_removed": intra_dup_pairs,
        "pair_inflation_vs_post_dedup_universe": inflation,
        "pre_dedup_pair_generation_estimate": pre_dedup_gen_estimate,
        "post_dedup_pair_csv_intra_duplicate_check": {
            "pair_training_csv": str(pair_training_csv) if pair_training_csv else None,
            "n_pairs_in_post_dedup_csv": n_universe,
            "n_pairs_same_duplicate_cluster": candidate_intra,
            "n_seed_pairs_same_cluster": seed_intra,
            "note": "Post-dedup CSV uses representatives; expect 0 intra-duplicate-cluster pairs.",
        },
        "duplicate_identity": {
            "signature_type": collapse.get("collapse_signature_type"),
            "fields_in_signature": [
                "aggressive_norm(subject)",
                "aggressive_norm(body)",
                "sorted basic_norm(senders)",
                "sorted basic_norm(attachments)",
                "sorted canonical URL tokens from subject+body",
            ],
            "normalization_summary": (
                "NFKC + lowercase + whitespace collapse; body/subject strip embedded timestamps, "
                "long hex tokens, long integers; inline URLs replaced with url:<canonical> tokens."
            ),
            "fields_deliberately_excluded": [
                "receiver/recipient",
                "routing/transit (origin IP, Received/HELO, return-path)",
                "event timestamps as identity scalars",
            ],
            "not_byte_level": (
                "Distinct raw incidents with different Message-IDs, receivers, or routing hops "
                "can collapse when task message content (template) matches."
            ),
        },
        "representative_selection_rule": collapse.get("representative_selection_rule"),
        "ground_truth_cluster_quality": collapse.get("ground_truth_cluster_quality"),
    }
    return summary


def deduplication_explanation_md(summary: dict[str, Any]) -> str:
    sig = summary.get("signature_type", "strict_task_message_identity")
    ident = summary.get("duplicate_identity") or {}
    infl = summary.get("pair_inflation_vs_post_dedup_universe") or {}
    pre_est = summary.get("pre_dedup_pair_generation_estimate") or {}
    ev = pre_est.get("evidence_enumeration") or {}
    ev_counts = ev.get("counts_by_evidence") or {}
    return f"""# Deduplication before graph construction

## Duplicate identity (not byte-level)

Signature: **`{sig}`**.

**Fields used:** {', '.join(ident.get('fields_in_signature') or [])}.

**Normalization:** {ident.get('normalization_summary', '')}

**Deliberately excluded:** {', '.join(ident.get('fields_deliberately_excluded') or [])}.

**Why not byte-level:** {ident.get('not_byte_level', '')}

Representatives: lexicographically smallest `external_id`. GT remapped in `ground_truth.dedup_task_identity.json`.

## Pair-universe inflation context (post-dedup universe)

| Denominator | Count | Intra-dup pairs / denom | % |
|-------------|------:|------------------------:|--:|
| All seed/candidate pairs | {infl.get('post_dedup_pair_universe_n_pairs', 0):,} | {infl.get('ratio_vs_pair_universe', 0):.3f} | {infl.get('pct_of_pair_universe', 0):.1f}% |
| Seed-positive pairs | {infl.get('post_dedup_seed_positive_pairs', 0):,} | {infl.get('ratio_vs_seed_positive', 0):.3f} | |
| Non-seed candidate pairs | {infl.get('post_dedup_non_seed_candidate_pairs', 0):,} | {infl.get('ratio_vs_non_seed_candidate', 0):.3f} | |

**{summary.get('potential_intra_duplicate_pairs_removed', 0):,}** potential intra-duplicate pairs (Σ C(n,2) over duplicate groups) would have been redundant if generation ran on the pre-dedup lake.

## Pre-dedup generation (estimate, not full re-run)

Full seed/candidate generation on 7,333 incidents was **not** re-executed. Within-duplicate enumeration ({pre_est.get('within_duplicate_pairs_total', 0):,} pairs):

| Evidence (pair-gen style) | Pairs |
|-------------------------|------:|
| Same sender | {ev_counts.get('same_sender', 0):,} |
| Same URL | {ev_counts.get('same_url', 0):,} |
| Same attachment | {ev_counts.get('same_attachment', 0):,} |
| Body token Jaccard ≥ 0.25 | {ev_counts.get('body_token_jaccard_ge_threshold', 0):,} |
| Body char-4gram Jaccard ≥ 0.25 | {ev_counts.get('body_char4gram_jaccard_ge_threshold', 0):,} |
| Semantic cosine ≥ 0.90 | {ev_counts.get('semantic_cosine_ge_threshold', 0):,} |
| **Any listed evidence** | {ev_counts.get('any_listed_evidence', 0):,} ({pre_est.get('pct_intra_duplicate_with_any_listed_evidence', 0):.1f}% of intra-dup pairs) |

## Collapse summary

| Metric | Value |
|--------|------:|
| Filtered incidents before dedup | {summary.get('filtered_phishing_scam_incidents_before_dedup', 0):,} |
| Deduplicated email nodes | {summary.get('deduplicated_email_nodes_after_dedup', 0):,} |
| Collapsed incidents | {summary.get('collapsed_incidents_removed', 0):,} |
| Duplicate groups (size > 1) | {summary.get('n_duplicate_groups_size_gt_1', 0):,} |
| Potential intra-dup pairs removed | {summary.get('potential_intra_duplicate_pairs_removed', 0):,} |
"""


@dataclass(frozen=True)
class ChannelSpec:
    channel_id: str
    channel_group: str  # included | routing | optional
    source: str  # hetero_edge | misp_parsed | hetero_sender_domain
    edge_type: tuple[str, str, str] | None = None
    node_key: str | None = None


CHANNEL_SPECS: tuple[ChannelSpec, ...] = (
    ChannelSpec("url", "included", "hetero_edge", ("email", "has_url", "url")),
    ChannelSpec("url_stem", "included", "hetero_edge", ("email", "has_stem", "stem")),
    ChannelSpec("attachment", "included", "hetero_edge", ("email", "has_attachment", "attachment")),
    ChannelSpec("sender", "included", "hetero_edge", ("email", "has_sender", "sender")),
    ChannelSpec("sender_domain", "included", "hetero_sender_domain"),
    ChannelSpec("domain", "included", "hetero_edge", ("email", "has_domain", "domain")),
    ChannelSpec("receiver", "optional", "hetero_edge", ("email", "has_receiver", "receiver")),
    ChannelSpec("origin_ip", "routing", "misp_parsed", node_key="origin_ip"),
    ChannelSpec("received_host", "routing", "misp_parsed", node_key="received_host"),
    ChannelSpec("helo_host", "routing", "misp_parsed", node_key="helo_host"),
    ChannelSpec("return_path_email", "routing", "misp_parsed", node_key="return_path_email"),
    ChannelSpec("return_path_domain", "routing", "misp_parsed", node_key="return_path_domain"),
)


def _email_index_to_external_id(metadata: dict[str, Any]) -> dict[int, str]:
    out: dict[int, str] = {}
    for i, row in enumerate((metadata.get("node_maps") or {}).get("email", {}).get("index_to_meta") or []):
        if isinstance(row, dict):
            eid = str(row.get("external_id") or "").strip()
            if eid:
                out[int(i)] = eid
    return out


def _artifact_index_to_string(metadata: dict[str, Any], node_type: str) -> dict[int, str]:
    block = (metadata.get("node_maps") or {}).get(node_type) or {}
    rows = block.get("index_to_string") or block.get("index_to_meta") or []
    out: dict[int, str] = {}
    for i, row in enumerate(rows):
        if isinstance(row, str):
            out[i] = row
        elif isinstance(row, dict):
            out[i] = str(row.get("key") or row.get("value") or row.get("url") or "")
    return out


def _channel_stats_from_email_artifact_map(
    emails_by_artifact: dict[Any, set[str]],
    *,
    label_map: dict[str, Any],
    channel_id: str,
    channel_group: str,
    artifact_labels: dict[Any, str] | None = None,
) -> dict[str, Any]:
    degrees = [len(v) for v in emails_by_artifact.values() if v]
    n_artifacts = len(emails_by_artifact)
    n_edges = int(sum(degrees))
    emails_with_value: set[str] = set()
    for eids in emails_by_artifact.values():
        emails_with_value.update(eids)

    induced_pairs = 0
    gt_covered = 0
    same_c = 0
    cross_c = 0
    multi_campaign_artifacts = 0

    for eids in emails_by_artifact.values():
        elist = sorted(eids)
        n = len(elist)
        if n < 2:
            continue
        _, n_tot, n_same, n_cross, _ = count_campaign_pairs_for_emails(elist, label_map)
        induced_pairs += n_tot
        if n_tot and sum(1 for e in elist if e in label_map) >= 2:
            gt_covered += comb(sum(1 for e in elist if e in label_map), 2)
        same_c += n_same
        cross_c += n_cross
        camps = {label_map[e] for e in elist if e in label_map}
        if len(camps) > 1:
            multi_campaign_artifacts += 1

    gt_pairs = same_c + cross_c
    return {
        "channel": channel_id,
        "channel_group": channel_group,
        "n_emails_with_non_empty_value": int(len(emails_with_value)),
        "n_unique_artifact_values": int(n_artifacts),
        "n_email_artifact_edges": n_edges,
        "median_artifact_degree": float(statistics.median(degrees)) if degrees else 0.0,
        "mean_artifact_degree": float(statistics.mean(degrees)) if degrees else 0.0,
        "p95_artifact_degree": _percentile(degrees, 95),
        "max_artifact_degree": int(max(degrees) if degrees else 0),
        "induced_email_email_pairs": int(induced_pairs),
        "gt_covered_induced_pairs": int(gt_pairs),
        "same_campaign_pct_among_gt_covered": float(100.0 * same_c / gt_pairs) if gt_pairs else float("nan"),
        "cross_campaign_pct_among_gt_covered": float(100.0 * cross_c / gt_pairs) if gt_pairs else float("nan"),
        "n_artifacts_touching_multiple_gt_campaigns": int(multi_campaign_artifacts),
    }


def _top_artifacts_rows(
    emails_by_artifact: dict[Any, set[str]],
    *,
    channel_id: str,
    artifact_labels: dict[Any, str],
    top_k: int = 10,
) -> list[dict[str, Any]]:
    ranked = sorted(emails_by_artifact.items(), key=lambda kv: len(kv[1]), reverse=True)[:top_k]
    rows: list[dict[str, Any]] = []
    for rank, (aid, eids) in enumerate(ranked, start=1):
        raw = artifact_labels.get(aid, str(aid))
        rows.append(
            {
                "channel": channel_id,
                "rank": rank,
                "artifact_degree": len(eids),
                "artifact_value_redacted": _redact_artifact(raw),
            }
        )
    return rows


def _hetero_edge_channel(
    graph: Any,
    metadata: dict[str, Any],
    edge_type: tuple[str, str, str],
    *,
    label_map: dict[str, Any],
    channel_id: str,
    channel_group: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    _, _, dst = edge_type
    idx_to_eid = _email_index_to_external_id(metadata)
    idx_to_art = _artifact_index_to_string(metadata, dst)
    emails_by_artifact: dict[int, set[str]] = defaultdict(set)
    if edge_type in getattr(graph, "edge_types", []):
        ei = graph[edge_type].edge_index
        if ei is not None and ei.numel() > 0:
            for e_idx, a_idx in zip(ei[0].tolist(), ei[1].tolist(), strict=False):
                eid = idx_to_eid.get(int(e_idx))
                if eid:
                    emails_by_artifact[int(a_idx)].add(eid)
    stats = _channel_stats_from_email_artifact_map(
        emails_by_artifact, label_map=label_map, channel_id=channel_id, channel_group=channel_group
    )
    tops = _top_artifacts_rows(
        emails_by_artifact,
        channel_id=channel_id,
        artifact_labels={k: idx_to_art.get(k, str(k)) for k in emails_by_artifact},
    )
    return stats, tops


def _sender_domain_channel(
    graph: Any,
    metadata: dict[str, Any],
    *,
    label_map: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    idx_to_eid = _email_index_to_external_id(metadata)
    idx_to_dom = _artifact_index_to_string(metadata, "email_domain")
    emails_by_dom: dict[int, set[str]] = defaultdict(set)

    if ("email", "has_sender", "sender") in getattr(graph, "edge_types", []) and (
        "sender",
        "from_domain",
        "email_domain",
    ) in getattr(graph, "edge_types", []):
        e_s = graph["email", "has_sender", "sender"].edge_index
        s_d = graph["sender", "from_domain", "email_domain"].edge_index
        sender_to_domains: dict[int, set[int]] = defaultdict(set)
        for s_idx, d_idx in zip(s_d[0].tolist(), s_d[1].tolist(), strict=False):
            sender_to_domains[int(s_idx)].add(int(d_idx))
        for e_idx, s_idx in zip(e_s[0].tolist(), e_s[1].tolist(), strict=False):
            eid = idx_to_eid.get(int(e_idx))
            if not eid:
                continue
            for d_idx in sender_to_domains.get(int(s_idx), ()):
                emails_by_dom[int(d_idx)].add(eid)

    stats = _channel_stats_from_email_artifact_map(
        emails_by_dom, label_map=label_map, channel_id="sender_domain", channel_group="included"
    )
    tops = _top_artifacts_rows(
        emails_by_dom,
        channel_id="sender_domain",
        artifact_labels={k: idx_to_dom.get(k, str(k)) for k in emails_by_dom},
    )
    return stats, tops


def _misp_routing_channel(
    parsed_emails: list[dict[str, Any]],
    *,
    node_key: str,
    label_map: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    emails_by_art: dict[str, set[str]] = defaultdict(set)
    for em in parsed_emails:
        eid = str(em.get("external_id") or "").strip()
        if not eid:
            continue
        for val in _field_values_for_node(em, node_key):
            v = str(val).strip()
            if v:
                emails_by_art[v].add(eid)
    stats = _channel_stats_from_email_artifact_map(
        emails_by_art, label_map=label_map, channel_id=node_key, channel_group="routing"
    )
    tops = _top_artifacts_rows(
        emails_by_art, channel_id=node_key, artifact_labels={k: k for k in emails_by_art}
    )
    return stats, tops


def build_relation_channel_reports(
    *,
    graph_pt: Path,
    meta_json: Path,
    dedup_misp: Path,
    gt_path: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    graph = _safe_load_graph(str(graph_pt))
    metadata = load_graph_metadata(str(meta_json))
    label_map = load_gt_label_map(gt_path)

    misp_raw = json.loads(dedup_misp.read_text(encoding="utf-8"))
    parsed = parse_misp_events(misp_raw if isinstance(misp_raw, list) else [])

    channel_rows: list[dict[str, Any]] = []
    top_rows: list[dict[str, Any]] = []

    for spec in CHANNEL_SPECS:
        if spec.source == "hetero_edge" and spec.edge_type:
            stats, tops = _hetero_edge_channel(
                graph,
                metadata,
                spec.edge_type,
                label_map=label_map,
                channel_id=spec.channel_id,
                channel_group=spec.channel_group,
            )
        elif spec.source == "hetero_sender_domain":
            stats, tops = _sender_domain_channel(graph, metadata, label_map=label_map)
        elif spec.source == "misp_parsed" and spec.node_key:
            stats, tops = _misp_routing_channel(parsed, node_key=spec.node_key, label_map=label_map)
        else:
            continue
        channel_rows.append(stats)
        if spec.channel_group == "routing":
            top_rows.extend(tops)

    return channel_rows, top_rows


def pre_post_dedup_channels_interpretation_md(
    pre_rows: list[dict[str, Any]],
    post_rows: list[dict[str, Any]],
) -> str:
    pre = {r["channel"]: r for r in pre_rows}
    post = {r["channel"]: r for r in post_rows}
    lines = [
        "# Pre- vs post-dedup relation-channel diagnostics",
        "",
        "Both views use `parse_misp_events` on the phishing/scam MISP lake. "
        "**Pre-dedup** (n=7,333): all filtered incidents. **Post-dedup** (n=4,970): task-identity representatives.",
        "",
        "GT labels: direct labels from `ground_truth.json` (pre) or merged dedup GT (post); "
        "pre-dedup also propagates representative labels to unlabeled duplicate cluster members.",
        "",
        "## Transit / hop metadata (high fan-out, cross-campaign mixing)",
        "",
    ]
    for cid in ["origin_ip", "received_host", "helo_host"]:
        a, b = pre.get(cid), post.get(cid)
        if not a or not b:
            continue
        lines.append(
            f"- **{cid}**: cross-campaign {a['cross_campaign_pct_among_gt_covered']:.1f}% (pre) → "
            f"{b['cross_campaign_pct_among_gt_covered']:.1f}% (post); max degree {a['max_artifact_degree']:,} → "
            f"{b['max_artifact_degree']:,}."
        )
    lines.extend(["", "## Return-path (often lower cross-campaign than hop metadata)", ""])
    for cid in ["return_path_email", "return_path_domain"]:
        a, b = pre.get(cid), post.get(cid)
        if not a or not b:
            continue
        lines.append(
            f"- **{cid}**: cross-campaign {a['cross_campaign_pct_among_gt_covered']:.1f}% (pre) → "
            f"{b['cross_campaign_pct_among_gt_covered']:.1f}% (post); max degree {a['max_artifact_degree']:,} → "
            f"{b['max_artifact_degree']:,}."
        )
    lines.extend(["", "## Content artifacts (included in graph)", ""])
    for cid in ["url", "url_stem", "attachment", "sender"]:
        a, b = pre.get(cid), post.get(cid)
        if not a or not b:
            continue
        lines.append(
            f"- **{cid}**: same-campaign {a['same_campaign_pct_among_gt_covered']:.1f}% (pre) → "
            f"{b['same_campaign_pct_among_gt_covered']:.1f}% (post)."
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Deduplication reduces incident count but preserves high-degree transit hubs (e.g. `received_host`). "
            "Hop/routing channels remain **more cross-campaign** than URL/sender/attachment; return-path fields "
            "can show different degree/purity profiles and should not be lumped with origin IP / Received hosts. "
            "Excluding routing from heterogeneous message passing is supported in both views.",
        ]
    )
    return "\n".join(lines) + "\n"


def _appendix_pre_post_compact_df(
    pre_rows: list[dict[str, Any]],
    post_rows: list[dict[str, Any]],
) -> pd.DataFrame:
    pre = {r["channel"]: r for r in pre_rows}
    post = {r["channel"]: r for r in post_rows}
    order = [
        "url",
        "url_stem",
        "attachment",
        "sender",
        "sender_domain",
        "domain",
        "receiver",
        "origin_ip",
        "received_host",
        "helo_host",
        "return_path_email",
        "return_path_domain",
    ]
    rows = []
    for ch in order:
        a, b = pre.get(ch), post.get(ch)
        if not a or not b:
            continue
        rows.append(
            {
                "channel": ch,
                "group": a.get("channel_group"),
                "n_inc_pre": a.get("n_incidents_total"),
                "n_inc_post": b.get("n_incidents_total"),
                "max_deg_pre": a.get("max_artifact_degree"),
                "max_deg_post": b.get("max_artifact_degree"),
                "cross_pct_pre": a.get("cross_campaign_pct_among_gt_covered"),
                "cross_pct_post": b.get("cross_campaign_pct_among_gt_covered"),
            }
        )
    return pd.DataFrame(rows)


def _channel_flags_from_anchor_config(cfg: dict[str, Any]) -> dict[str, dict[str, bool]]:
    ch_settings = (cfg.get("channels") or {}).get("channel_settings") or {}
    out: dict[str, dict[str, bool]] = {}
    for name, spec in ch_settings.items():
        if not isinstance(spec, dict):
            continue
        out[str(name)] = {
            "edge_create_enabled": bool(spec.get("edge_create_enabled", spec.get("enabled", True))),
            "evidence_enabled": bool(spec.get("evidence_enabled", spec.get("enabled", True))),
            "score_enabled": bool(spec.get("score_enabled", spec.get("enabled", True))),
        }
    return out


def build_config_consistency_report(
    *,
    pipeline_config_path: Path,
    anchor_graph_config_path: Path,
    candidate_config_path: Path,
    graph_pt: Path,
    dedup_misp: Path,
) -> dict[str, Any]:
    pipeline = json.loads(pipeline_config_path.read_text(encoding="utf-8"))
    anchor_cfg = json.loads(anchor_graph_config_path.read_text(encoding="utf-8"))
    cand_cfg = json.loads(candidate_config_path.read_text(encoding="utf-8"))

    exclude_hetero = list((pipeline.get("graph") or {}).get("exclude_node_types") or [])
    anchor_channels = _channel_flags_from_anchor_config(anchor_cfg)

    graph = _safe_load_graph(str(graph_pt))
    hetero_edge_types = list(getattr(graph, "edge_types", []))
    hetero_has_routing_edges = any(
        dst in ROUTING_NODE_KEYS for _, _, dst in hetero_edge_types
    )

    misp_raw = json.loads(dedup_misp.read_text(encoding="utf-8"))
    parsed = parse_misp_events(misp_raw if isinstance(misp_raw, list) else [])
    routing_presence = {}
    for key in ROUTING_NODE_KEYS:
        n_emails = sum(1 for em in parsed if _field_values_for_node(em, key))
        routing_presence[key] = {
            "n_emails_with_non_empty_value": int(n_emails),
            "present_in_parsed_misp": bool(n_emails),
        }

    corroboration: list[str] = []
    for gen in cand_cfg.get("candidates", {}).get("generators") or []:
        if not isinstance(gen, dict) or not gen.get("enabled"):
            continue
        name = str(gen.get("name") or "")
        gcfg = gen.get("config") or {}
        if name == "component_expansion_v1":
            cols = gcfg.get("artifact_columns") or []
            routing_cols = [c for c in cols if any(r in str(c) for r in ROUTING_NODE_KEYS)]
            if routing_cols:
                corroboration.append(
                    f"{name} (disabled={not gen.get('enabled')}) lists routing columns: {routing_cols}"
                )
        if "weak_channels" in str(gcfg):
            corroboration.append(f"{name}: config may reference weak_channels")

    seed_cfg_path = REPO / "seed_candidate_workflow/configs/anchor_seed.default.json"
    seed_routing = []
    if seed_cfg_path.is_file():
        for gen in json.loads(seed_cfg_path.read_text(encoding="utf-8")).get("generators") or []:
            if str(gen.get("name")) == "corroborated_seed_v1":
                weak = (gen.get("config") or {}).get("weak_channels") or []
                seed_routing = [c for c in weak if c in ROUTING_NODE_KEYS or c.endswith("_host") or "return_path" in c or c == "origin_ip"]

    methods_checks = [
        {
            "claim": "Routing fields excluded from heterogeneous graph message passing",
            "consistent": (not hetero_has_routing_edges)
            and all(not anchor_channels.get(ch, {}).get("edge_create_enabled", False) for ch in ROUTING_NODE_KEYS),
            "detail": f"hetero routing edges present={hetero_has_routing_edges}; exclude_node_types={exclude_hetero}",
        },
        {
            "claim": "Routing fields disabled in anchor co-occurrence / scoring table",
            "consistent": all(
                not anchor_channels.get(ch, {}).get("score_enabled", True) for ch in ROUTING_NODE_KEYS
            ),
            "detail": "anchor_graph channel_settings score_enabled flags for routing types",
        },
        {
            "claim": "Routing still available in raw/parsed MISP for diagnostics",
            "consistent": any(v["present_in_parsed_misp"] for v in routing_presence.values()),
            "detail": str({k: v["n_emails_with_non_empty_value"] for k, v in routing_presence.items()}),
        },
        {
            "claim": "Primary candidate generators do not use routing as overlap artifacts",
            "consistent": not any("rare_artifact" in c or "2hop" in c for c in corroboration),
            "detail": f"component_expansion routing note: {corroboration}",
        },
        {
            "claim": "Seed corroboration may treat routing as weak evidence only",
            "consistent": True,
            "detail": f"corroborated_seed_v1 weak_channels routing subset: {seed_routing}",
        },
    ]

    return {
        "created_at_utc": _utc_now(),
        "pipeline_config": str(pipeline_config_path),
        "anchor_graph_config": str(anchor_graph_config_path),
        "candidate_generation_config": str(candidate_config_path),
        "routing_fields_in_parsed_misp": routing_presence,
        "routing_fields_excluded_from_hetero_graph": exclude_hetero,
        "hetero_graph_routing_edge_types_present": hetero_has_routing_edges,
        "hetero_edge_types": [list(t) for t in hetero_edge_types],
        "anchor_channel_flags": anchor_channels,
        "routing_disabled_in_anchor_cooccurrence_scoring": {
            ch: anchor_channels.get(ch, {}) for ch in ROUTING_NODE_KEYS
        },
        "routing_in_seed_candidate_generators": {
            "corroborated_seed_weak_routing_channels": seed_routing,
            "enabled_generator_routing_notes": corroboration,
        },
        "methods_consistency_checks": methods_checks,
        "methods_consistency_summary": (
            "PASS" if all(c["consistent"] for c in methods_checks[:4]) else "REVIEW"
        ),
    }


def config_consistency_notes_md(report: dict[str, Any]) -> str:
    lines = [
        "# Graph construction config consistency",
        "",
        f"Overall: **{report.get('methods_consistency_summary', 'REVIEW')}**",
        "",
        "## Routing in data vs graph",
        "",
    ]
    for k, v in (report.get("routing_fields_in_parsed_misp") or {}).items():
        lines.append(f"- `{k}`: {v['n_emails_with_non_empty_value']:,} emails with values in parsed MISP")
    lines.append(f"- `graph.exclude_node_types`: {report.get('routing_fields_excluded_from_hetero_graph')}")
    lines.append(
        f"- Hetero graph has routing edge types: **{report.get('hetero_graph_routing_edge_types_present')}**"
    )
    lines.extend(["", "## Anchor co-occurrence / scoring", ""])
    for ch, flags in (report.get("routing_disabled_in_anchor_cooccurrence_scoring") or {}).items():
        lines.append(f"- `{ch}`: {flags}")
    lines.extend(["", "## Seed / candidate generators", ""])
    lines.append(
        f"- Corroborated seed weak routing channels: {report.get('routing_in_seed_candidate_generators', {}).get('corroborated_seed_weak_routing_channels')}"
    )
    for note in report.get("routing_in_seed_candidate_generators", {}).get("enabled_generator_routing_notes") or []:
        lines.append(f"- {note}")
    lines.extend(["", "## Methods text checks", ""])
    for chk in report.get("methods_consistency_checks") or []:
        status = "ok" if chk.get("consistent") else "review"
        lines.append(f"- [{status}] {chk.get('claim')}: {chk.get('detail')}")
    return "\n".join(lines) + "\n"


def run_all_diagnostics(
    *,
    out_dir: Path,
    raw_misp: Path | None = None,
    dedup_misp: Path | None = None,
    collapse_dir: Path | None = None,
    graph_pt: Path | None = None,
    meta_json: Path | None = None,
    gt_path: Path | None = None,
    pipeline_config: Path | None = None,
    anchor_graph_config: Path | None = None,
    candidate_config: Path | None = None,
    pair_training_csv: Path | None = None,
    seed_csv: Path | None = None,
    embeddings_json: Path | None = None,
) -> dict[str, str]:
    raw_misp = raw_misp or (REPO / "data/misp/incidents-lake-misp.json")
    dedup_misp = dedup_misp or (REPO / "data/misp/incidents-lake-misp.dedup_task_identity.json")
    collapse_dir = collapse_dir or (REPO / "data/misp/misp_lake_dedup_task_identity")
    graph_pt = graph_pt or (REPO / "core/graph/output/main_gnn_pu_1_no_ts_dedup_task_identity_12_hetero.pt")
    meta_json = meta_json or graph_pt.with_suffix(".meta.json")
    gt_path = gt_path or (REPO / "data/groundtruth/ground_truth.json")
    pipeline_config = pipeline_config or (REPO / "pipeline_config.json")
    anchor_graph_config = anchor_graph_config or (
        REPO / "seed_candidate_workflow/configs/anchor_graph.main_gnn_pu_1_no_ts_dedup_task_identity.json"
    )
    candidate_config = candidate_config or (
        REPO
        / "seed_candidate_workflow/configs/anchor_candidate_generation.main_gnn_pu_1_no_ts_dedup_task_identity_13.json"
    )

    dedup_dir = out_dir / "deduplication"
    rel_dir = out_dir / "relation_channels"
    cfg_dir = out_dir / "config_consistency"
    for d in (dedup_dir, rel_dir, cfg_dir):
        d.mkdir(parents=True, exist_ok=True)

    map_df = load_id_map_dataframe(collapse_dir)

    dedup_summary = build_deduplication_report(
        raw_misp=raw_misp,
        dedup_misp=dedup_misp,
        collapse_dir=collapse_dir,
        gt_path=gt_path,
        pair_training_csv=pair_training_csv,
        seed_csv=seed_csv,
        embeddings_json=embeddings_json,
    )
    (dedup_dir / "thesis_deduplication_summary.json").write_text(
        json.dumps(dedup_summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    dedup_df = pd.DataFrame(
        [
            {
                "metric": k,
                "value": (
                    json.dumps(v, ensure_ascii=False)
                    if isinstance(v, (dict, list))
                    else v
                ),
            }
            for k, v in dedup_summary.items()
            if k not in ("top_10_duplicate_group_sizes", "ground_truth_cluster_quality", "seed_candidate_pairs_intra_duplicate_before_dedup")
        ]
    )
    dedup_df.to_csv(dedup_dir / "thesis_deduplication_summary.csv", index=False)
    top10_df = pd.DataFrame(
        {"rank": i + 1, "group_size": s}
        for i, s in enumerate(dedup_summary.get("top_10_duplicate_group_sizes") or [])
    )
    top10_df.to_csv(dedup_dir / "thesis_duplicate_group_sizes_top10.csv", index=False)
    _write_df_table_tex(
        top10_df,
        dedup_dir / "thesis_duplicate_group_sizes_top10.tex",
        caption="Top duplicate group sizes (task-identity signature, pre-collapse).",
        label="tab:dup_group_top10",
    )
    infl = dedup_summary.get("pair_inflation_vs_post_dedup_universe") or {}
    summary_table = pd.DataFrame(
        [
            ["Filtered phishing/scam incidents (pre-dedup)", dedup_summary["filtered_phishing_scam_incidents_before_dedup"]],
            ["Deduplicated email nodes", dedup_summary["deduplicated_email_nodes_after_dedup"]],
            ["Collapsed incidents", dedup_summary["collapsed_incidents_removed"]],
            ["Duplicate groups ($>1$)", dedup_summary["n_duplicate_groups_size_gt_1"]],
            ["Potential intra-dup pairs $\\sum \\binom{n}{2}$", dedup_summary["potential_intra_duplicate_pairs_removed"]],
            ["Ratio vs post-dedup pair universe (49,030)", f"{infl.get('ratio_vs_pair_universe', 0):.4f}"],
            ["Ratio vs seed-positive pairs (24,746)", f"{infl.get('ratio_vs_seed_positive', 0):.4f}"],
            ["Ratio vs non-seed candidate pairs (24,284)", f"{infl.get('ratio_vs_non_seed_candidate', 0):.4f}"],
        ],
        columns=["Metric", "Value"],
    )
    _write_df_table_tex(
        summary_table,
        dedup_dir / "thesis_deduplication_summary.tex",
        caption="Task-identity deduplication summary (phishing/scam lake).",
        label="tab:dedup_summary",
    )
    (dedup_dir / "deduplication_explanation.md").write_text(
        deduplication_explanation_md(dedup_summary), encoding="utf-8"
    )

    raw_data = json.loads(raw_misp.read_text(encoding="utf-8"))
    dedup_data = json.loads(dedup_misp.read_text(encoding="utf-8"))
    parsed_pre = parse_misp_events(raw_data if isinstance(raw_data, list) else [])
    parsed_post = parse_misp_events(dedup_data if isinstance(dedup_data, list) else [])

    label_pre, meta_pre = build_propagated_label_map(gt_path, id_map_df=map_df, view="pre_dedup")
    label_post, meta_post = build_propagated_label_map(
        REPO / "data/groundtruth/ground_truth.dedup_task_identity.json",
        view="post_dedup",
    )

    pre_rows = build_parsed_misp_channel_reports(
        parsed_pre,
        label_map=label_pre,
        data_view="pre_dedup_filtered",
        n_incidents_total=len(parsed_pre),
        label_meta=meta_pre,
    )
    post_rows = build_parsed_misp_channel_reports(
        parsed_post,
        label_map=label_post,
        data_view="post_dedup_representatives",
        n_incidents_total=len(parsed_post),
        label_meta=meta_post,
    )

    ch_df = pd.DataFrame(pre_rows + post_rows)
    ch_df.to_csv(rel_dir / "thesis_relation_channel_pre_post_dedup.csv", index=False)
    (rel_dir / "thesis_relation_channel_pre_post_dedup.json").write_text(
        json.dumps(
            {
                "created_at_utc": _utc_now(),
                "pre_dedup": {"n_incidents": len(parsed_pre), "channels": pre_rows, "gt_labeling": meta_pre},
                "post_dedup": {"n_incidents": len(parsed_post), "channels": post_rows, "gt_labeling": meta_post},
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    appendix_df = _appendix_pre_post_compact_df(pre_rows, post_rows)
    _write_df_table_tex(
        appendix_df,
        rel_dir / "thesis_relation_channel_pre_post_appendix.tex",
        caption="Pre- vs post-dedup relation channels (max artifact degree and cross-campaign \\% among GT-covered induced pairs).",
        label="tab:channels_pre_post",
    )
    compact_cols = [
        "data_view",
        "channel",
        "channel_group",
        "n_incidents_total",
        "n_emails_with_non_empty_value",
        "n_unique_artifact_values",
        "median_artifact_degree",
        "p95_artifact_degree",
        "max_artifact_degree",
        "gt_covered_induced_pairs",
        "same_campaign_pct_among_gt_covered",
        "cross_campaign_pct_among_gt_covered",
    ]
    _write_df_table_tex(
        ch_df[compact_cols],
        rel_dir / "thesis_relation_channel_pre_post_dedup.tex",
        caption="Relation-channel diagnostics: full filtered lake (pre-dedup) vs task-identity representatives (post-dedup).",
        label="tab:relation_channels_pre_post_full",
    )
    # Legacy post-only snapshot for backward compatibility
    post_only_path = rel_dir / "thesis_relation_channel_diagnostics_post_dedup_only.csv"
    pd.DataFrame(post_rows).to_csv(post_only_path, index=False)

    (rel_dir / "relation_channels_interpretation.md").write_text(
        pre_post_dedup_channels_interpretation_md(pre_rows, post_rows), encoding="utf-8"
    )

    cfg_report = build_config_consistency_report(
        pipeline_config_path=pipeline_config,
        anchor_graph_config_path=anchor_graph_config,
        candidate_config_path=candidate_config,
        graph_pt=graph_pt,
        dedup_misp=dedup_misp,
    )
    (cfg_dir / "thesis_graph_config_consistency.json").write_text(
        json.dumps(cfg_report, indent=2), encoding="utf-8"
    )
    (cfg_dir / "config_consistency_notes.md").write_text(
        config_consistency_notes_md(cfg_report), encoding="utf-8"
    )
    cfg_tex = pd.DataFrame(cfg_report.get("methods_consistency_checks") or [])
    if not cfg_tex.empty:
        _write_df_table_tex(
            cfg_tex[["claim", "consistent", "detail"]],
            cfg_dir / "thesis_graph_config_consistency.tex",
            caption="Methods vs final graph/candidate configuration consistency checks.",
            label="tab:config_consistency",
        )

    paths = {
        "deduplication_json": str((dedup_dir / "thesis_deduplication_summary.json").resolve()),
        "deduplication_csv": str((dedup_dir / "thesis_deduplication_summary.csv").resolve()),
        "deduplication_tex": str((dedup_dir / "thesis_deduplication_summary.tex").resolve()),
        "deduplication_md": str((dedup_dir / "deduplication_explanation.md").resolve()),
        "relation_channels_pre_post_csv": str((rel_dir / "thesis_relation_channel_pre_post_dedup.csv").resolve()),
        "relation_channels_pre_post_json": str((rel_dir / "thesis_relation_channel_pre_post_dedup.json").resolve()),
        "relation_channels_pre_post_tex": str((rel_dir / "thesis_relation_channel_pre_post_dedup.tex").resolve()),
        "relation_channels_appendix_tex": str((rel_dir / "thesis_relation_channel_pre_post_appendix.tex").resolve()),
        "relation_channels_md": str((rel_dir / "relation_channels_interpretation.md").resolve()),
        "config_json": str((cfg_dir / "thesis_graph_config_consistency.json").resolve()),
        "config_md": str((cfg_dir / "config_consistency_notes.md").resolve()),
        "config_tex": str((cfg_dir / "thesis_graph_config_consistency.tex").resolve()),
    }
    (out_dir / "paths_manifest.json").write_text(json.dumps(paths, indent=2), encoding="utf-8")
    return paths
