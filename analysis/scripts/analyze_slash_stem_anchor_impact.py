"""
Quantify ``\"/\"`` URL-template stems in anchor email ``stem_set`` and their effect on overlaps.

Context
-------
Anchor ``stem_set`` values come from graph URL→stem edges and from ``parse_url_components``
fallback in ``graph_structure_helpers.stem_strings_for_url``. A stem of ``\"/\"`` is a
degenerate "root path" token: many unrelated emails can share it alongside the same
registrable domain, so ``has_stem_overlap`` can fire for weak-channel corroboration even when
no meaningful path template is shared.

This script does **not** modify graphs or bundles. It reports:

- How many emails carry ``\"/\"`` in ``stem_set`` (same-email / internal artifact).
- Among **cross-email** anchor candidate edges (``email_a`` ≠ ``email_b``), how often stem
  intersection includes ``\"/\"``, and how often ``\"/\"`` is the **only** shared stem value.
- For edges matching the default **corroborated_v1** weak-channel list, how many **lose** the
  weak-multi trigger (``n_weak_channels >= 2``) if stem overlap is counted only when
  ``(stem_a ∩ stem_b) - {\"/\"}`` is non-empty (ablated stem channel).
- Optional: with ``--gt-json``, stratify **cross-email** edges whose stem intersection includes
  ``\"/\"`` (and separately **slash-only** intersections) by **same-campaign** vs **cross-campaign**
  vs partial / missing GT labels.
- Optional: with ``--gt-json`` + ``--bundle-dir`` + ``--seed-impact``, simulate slash/stem changes on
  ``seed_edges_all.csv``: row removals plus **canonical pair union** (pair survives if any seed row
  survives); headline same- vs cross-campaign **pair** losses under GT labels.
- Optional: hard-tier seed rows whose ``evidence_value`` is ``\"/\"`` for ``rare_exact_url_template``.

Community / scoring impact (read carefully)
-------------------------------------------
Downstream **PU scoring + Louvain/Leiden sweeps** read the **seed_candidate** weighted graph
built from candidates ∪ seeds. This script **does not** re-run partitioning; it gives **upper
bounds / structural proxies**:

- **Stem-only slash bridges**: cross-email edges whose *only* shared stem string is ``\"/\"``.
  Removing ``\"/\"`` from ``stem_set`` eliminates ``has_stem_overlap`` on those pairs unless
  other stems match — shrinking the channel overlap graph on that dimension.
- **Corroborated trigger risk**: pairs that currently hit ``weak_multi`` only because ``stem``
  counts as a weak channel thanks to ``\"/\"``. Those seeds can disappear after filtering,
  shrinking seed mass and changing which candidate edges receive seed edge weight boosts.

For definitive community metric deltas, re-run ``seed_candidate`` bundle generation (or patch
``stem_strings_for_url`` / stem_set construction) and compare ``run_manifest.json`` / sweep
summaries before vs after.

Examples::

  python analysis/scripts/analyze_slash_stem_anchor_impact.py \\
    --bundle-dir seed_candidate_workflow/output/graph_bundles/main_gnn_pu_1_no_ts_dedup_task_identity_4/anchor/main_gnn_pu_1_no_ts_dedup_task_identity_4 \\
    --gt-json data/groundtruth/ground_truth.dedup_task_identity.json \\
    --seed-impact \\
    --out-json output/analysis/stem_slash_impact_run4_with_seed_impact.json

  python analysis/scripts/analyze_slash_stem_anchor_impact.py \\
    --nodes-csv path/to/anchor_graph_nodes.csv \\
    --edges-csv path/to/anchor_graph_edges_unscored.csv \\
    --gt-json data/groundtruth/ground_truth.dedup_task_identity.json \\
    --out-json output/analysis/stem_slash_impact.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import numpy as np

_DEFAULT_WEAK_BASES = (
    "sender",
    "sender_email_domain",
    "domain",
    "stem",
    "return_path_domain",
    "received_host",
    "origin_ip",
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _to_set_cell(v: Any) -> set[str]:
    """Match anchor/seed helpers: pipe-split, JSON list, or Python-ish set literals."""
    if isinstance(v, set):
        return {str(x) for x in v if x is not None and str(x).strip()}
    if isinstance(v, (list, tuple)):
        return {str(x) for x in v if x is not None and str(x).strip()}
    if v is None:
        return set()
    s = str(v).strip()
    if not s:
        return set()
    try:
        obj = json.loads(s)
        if isinstance(obj, list):
            return {str(x) for x in obj if x is not None and str(x).strip()}
    except Exception:
        pass
    return {x.strip() for x in s.split("|") if x.strip()}


def _stem_set_for_row(row: pd.Series) -> set[str]:
    if "stem_set" not in row.index:
        return set()
    return _to_set_cell(row.get("stem_set"))


def _weak_overlap_cols(edges_df: pd.DataFrame) -> list[str]:
    cols: list[str] = []
    for base in _DEFAULT_WEAK_BASES:
        c = f"has_{base}_overlap"
        if c in edges_df.columns:
            cols.append(c)
    return cols


def _n_weak_hits(row: pd.Series, weak_cols: Iterable[str]) -> int:
    n = 0
    for c in weak_cols:
        try:
            if bool(row.get(c)):
                n += 1
        except Exception:
            pass
    return n


def _n_weak_hits_ablate_stem(
    row: pd.Series,
    weak_cols: list[str],
    *,
    stem_overlap_without_slash: bool,
) -> int:
    n = 0
    for c in weak_cols:
        if c == "has_stem_overlap":
            if stem_overlap_without_slash:
                n += 1
            continue
        try:
            if bool(row.get(c)):
                n += 1
        except Exception:
            pass
    return n


def _find_latest_seed_dir(seed_root: Path, graph_id: str) -> Path | None:
    base = seed_root / graph_id
    if not base.is_dir():
        return None
    dirs = [d for d in base.iterdir() if d.is_dir() and d.name.startswith("seed_generation_")]
    if not dirs:
        return None
    return max(dirs, key=lambda d: d.stat().st_mtime)


def _load_optional_seed_hard(seed_dir: Path | None) -> pd.DataFrame:
    if seed_dir is None:
        return pd.DataFrame()
    p = seed_dir / "seed_edges_hard.csv"
    if not p.is_file():
        return pd.DataFrame()
    return pd.read_csv(p, low_memory=False)


def _load_optional_seed_all(seed_dir: Path | None) -> pd.DataFrame:
    if seed_dir is None:
        return pd.DataFrame()
    p = seed_dir / "seed_edges_all.csv"
    if not p.is_file():
        return pd.DataFrame()
    return pd.read_csv(p, low_memory=False)


def _parse_campaign_key(raw: str) -> Any:
    s = str(raw).strip()
    if "/" in s:
        s = s.rsplit("/", 1)[-1]
    try:
        return int(s)
    except ValueError:
        return s


def load_gt_label_map(gt_path: Path) -> dict[str, Any]:
    """external_id -> campaign id (first GT occurrence wins)."""
    data = json.loads(gt_path.read_text(encoding="utf-8"))
    label_map: dict[str, Any] = {}
    for raw_key, emails in (data.get("clusters") or {}).items():
        cid = _parse_campaign_key(str(raw_key))
        if not isinstance(emails, list):
            continue
        for em in emails:
            if not isinstance(em, dict):
                continue
            eid = em.get("external_id")
            if eid is None:
                continue
            eid = str(eid).strip()
            if not eid or eid in label_map:
                continue
            label_map[eid] = cid
    return label_map


def _gt_pair_bucket(a: str, b: str, label_map: dict[str, Any]) -> str:
    ca = label_map.get(a)
    cb = label_map.get(b)
    ha = ca is not None
    hb = cb is not None
    if not ha and not hb:
        return "no_gt"
    if ha != hb:
        return "partial_gt"
    if ca == cb:
        return "same_campaign"
    return "cross_campaign"


def _infer_graph_bundle_root_from_anchor_run_dir(anchor_run_dir: Path) -> Path:
    """
    ``.../graph_bundles/<graph_id>/anchor/<graph_id>`` -> ``.../graph_bundles/<graph_id>``.
    """
    p = anchor_run_dir.resolve()
    return p.parent.parent


def _resolve_seed_edges_all_csv(anchor_run_dir: Path) -> Path | None:
    graph_id = anchor_run_dir.name
    root = _infer_graph_bundle_root_from_anchor_run_dir(anchor_run_dir)
    seed_base = root / "seed" / graph_id
    if not seed_base.is_dir():
        return None
    dirs = [d for d in seed_base.iterdir() if d.is_dir() and d.name.startswith("seed_generation_")]
    if not dirs:
        return None
    latest = max(dirs, key=lambda d: d.stat().st_mtime)
    p = latest / "seed_edges_all.csv"
    return p if p.is_file() else None


def _canonical_pair(a: str, b: str) -> tuple[str, str]:
    x, y = str(a).strip(), str(b).strip()
    return (x, y) if x <= y else (y, x)


def _build_anchor_edge_index(edges_df: pd.DataFrame) -> dict[tuple[str, str], pd.Series]:
    out: dict[tuple[str, str], pd.Series] = {}
    for _, row in edges_df.iterrows():
        a = str(row.get("email_a", "")).strip()
        b = str(row.get("email_b", "")).strip()
        if not a or not b or a == b:
            continue
        key = _canonical_pair(a, b)
        out[key] = row
    return out


def _corroborated_passes(
    *,
    n_support: int,
    sem_score: float,
    min_sem: float,
    req_weak: int,
    req_non_sem: bool,
    min_non_sem: int,
) -> tuple[bool, bool, bool]:
    """Returns (rule_by_weak, rule_by_sem, passes)."""
    sem_hit = bool(np.isfinite(sem_score) and sem_score >= min_sem)
    rule_by_weak = n_support >= req_weak
    rule_by_sem = sem_hit and (n_support >= min_non_sem if req_non_sem else True)
    return rule_by_weak, rule_by_sem, bool(rule_by_weak or rule_by_sem)


def _slash_sim_seed_row_removed(
    r: pd.Series,
    *,
    anchor_by_pair: dict[tuple[str, str], pd.Series],
    weak_cols: list[str],
    stem_by_eid: dict[str, set[str]],
    corro_min_sem: float,
    corro_req_weak: int,
    corro_req_non_sem: bool,
    corro_min_non_sem: int,
) -> tuple[bool, str]:
    """
    Whether this ``seed_edges_all`` row would disappear under the slash-stem simulation.

    Only **hard** ``rare_exact_url_template`` with evidence ``/`` and **corroborated** rows that
    flip from passing to failing corroborated_v1-style rules after stem ablation are removed.
    """
    tier = str(r.get("seed_tier", "") or "").strip().lower()
    ei = str(r.get("email_i", "")).strip()
    ej = str(r.get("email_j", "")).strip()
    rule_id = str(r.get("rule_id", "") or "").strip()
    ev_val = str(r.get("evidence_value", "") or "").strip()

    if tier == "hard":
        if rule_id == "rare_exact_url_template" and ev_val == "/":
            return True, "hard_rare_exact_url_template_evidence_is_slash"
        return False, "hard_row_not_slash_rare_exact_template_unchanged"
    if tier == "corroborated":
        key = _canonical_pair(ei, ej)
        arow = anchor_by_pair.get(key)
        if arow is None:
            return False, "corroborated_no_matching_anchor_row_unchanged"
        sem = float(pd.to_numeric(arow.get("semantic_score"), errors="coerce"))
        n_o = _n_support_channels_for_anchor_row(
            arow, weak_cols, stem_ablate_slash=False, stem_by_eid=stem_by_eid, ei=ei, ej=ej
        )
        n_a = _n_support_channels_for_anchor_row(
            arow, weak_cols, stem_ablate_slash=True, stem_by_eid=stem_by_eid, ei=ei, ej=ej
        )
        _, _, pass_o = _corroborated_passes(
            n_support=n_o,
            sem_score=sem,
            min_sem=corro_min_sem,
            req_weak=corro_req_weak,
            req_non_sem=corro_req_non_sem,
            min_non_sem=corro_min_non_sem,
        )
        _, _, pass_a = _corroborated_passes(
            n_support=n_a,
            sem_score=sem,
            min_sem=corro_min_sem,
            req_weak=corro_req_weak,
            req_non_sem=corro_req_non_sem,
            min_non_sem=corro_min_non_sem,
        )
        if pass_o and not pass_a:
            return True, "corroborated_rules_fail_after_stem_slash_ablation"
        return False, "corroborated_row_passes_before_and_after_or_never_passed_unchanged"
    return False, "non_hard_corroborated_tier_unchanged"


def _n_support_channels_for_anchor_row(
    edge_row: pd.Series,
    weak_cols: list[str],
    *,
    stem_ablate_slash: bool,
    stem_by_eid: dict[str, set[str]],
    ei: str,
    ej: str,
) -> int:
    n = 0
    for col in weak_cols:
        if col == "has_stem_overlap":
            if stem_ablate_slash:
                sa = stem_by_eid.get(ei, set())
                sb = stem_by_eid.get(ej, set())
                if (sa & sb) - {"/"}:
                    n += 1
            else:
                if bool(edge_row.get(col, False)):
                    n += 1
            continue
        try:
            if bool(edge_row.get(col, False)):
                n += 1
        except Exception:
            pass
    return n


def analyze_seed_positive_slash_removals(
    *,
    seed_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    label_map: dict[str, Any],
    weak_cols: list[str],
    stem_by_eid: dict[str, set[str]],
    seed_csv_path: str,
    corro_min_sem: float = 0.97,
    corro_req_weak: int = 2,
    corro_req_non_sem: bool = True,
    corro_min_non_sem: int = 1,
) -> dict[str, Any]:
    """
    Simulate slash-stem effects on ``seed_edges_all.csv``.

    Row removals follow ``_slash_sim_seed_row_removed``. **Canonical seed pairs** (unordered
    ``email_i/email_j``) are counted as still present after the change if **any** CSV row for that
    pair survives; otherwise the pair is **fully lost** from the seed union.
    """
    if seed_df.empty or "email_i" not in seed_df.columns or "email_j" not in seed_df.columns:
        return {
            "seed_edges_csv": seed_csv_path,
            "error": "seed_edges_all.csv missing or empty",
        }

    if "seed_tier" in seed_df.columns:
        tier_series = seed_df["seed_tier"].astype(str).str.strip().str.lower()
    else:
        tier_series = pd.Series("", index=seed_df.index, dtype=str)
    n_total_seed_rows = int(len(seed_df))
    tier_counts = dict(Counter(tier_series.tolist()))

    anchor_by_pair = _build_anchor_edge_index(edges_df)

    pair_any: set[tuple[str, str]] = set()
    pair_rep: dict[tuple[str, str], tuple[str, str]] = {}
    tiers_on_pair: defaultdict[tuple[str, str], set[str]] = defaultdict(set)
    pair_survives: set[tuple[str, str]] = set()
    pair_had_removed_hard_or_corro_row: set[tuple[str, str]] = set()

    n_rows_removed = 0
    removed_reason_counts: Counter[str] = Counter()
    removed_row_by_tier_and_rule: Counter[tuple[str, str]] = Counter()
    hc_row_by_bucket: defaultdict[str, dict[str, int]] = defaultdict(lambda: {"n_rows": 0, "n_removed": 0})

    for _, r in seed_df.iterrows():
        ei = str(r.get("email_i", "")).strip()
        ej = str(r.get("email_j", "")).strip()
        if not ei or not ej or ei == ej:
            continue
        key = _canonical_pair(ei, ej)
        pair_any.add(key)
        if key not in pair_rep:
            pair_rep[key] = (ei, ej)
        tier = str(r.get("seed_tier", "") or "").strip().lower()
        tiers_on_pair[key].add(tier or "_unknown_tier")
        rule_id = str(r.get("rule_id", "") or "").strip()

        removed, reason = _slash_sim_seed_row_removed(
            r,
            anchor_by_pair=anchor_by_pair,
            weak_cols=weak_cols,
            stem_by_eid=stem_by_eid,
            corro_min_sem=corro_min_sem,
            corro_req_weak=corro_req_weak,
            corro_req_non_sem=corro_req_non_sem,
            corro_min_non_sem=corro_min_non_sem,
        )
        if not removed:
            pair_survives.add(key)
        else:
            n_rows_removed += 1
            removed_reason_counts[reason] += 1
            removed_row_by_tier_and_rule[(tier, rule_id or "_")] += 1
            if tier in ("hard", "corroborated"):
                pair_had_removed_hard_or_corro_row.add(key)

        if tier in ("hard", "corroborated"):
            b = _gt_pair_bucket(ei, ej, label_map)
            hc_row_by_bucket[b]["n_rows"] += 1
            if removed:
                hc_row_by_bucket[b]["n_removed"] += 1

    if not pair_any:
        return {
            "seed_edges_csv": seed_csv_path,
            "error": "no_valid_cross_email_seed_pairs",
            "n_seed_edges_all_rows_total": n_total_seed_rows,
            "seed_tier_value_counts_all_rows": tier_counts,
        }

    pair_lost = pair_any - pair_survives
    rescued_pairs = pair_had_removed_hard_or_corro_row & pair_survives
    only_hard_corro_pairs = {
        k for k, ts in tiers_on_pair.items() if ts <= {"hard", "corroborated"}
    }

    def _bucket_counts_for_pair_keys(keys: Iterable[tuple[str, str]]) -> dict[str, int]:
        c: Counter[str] = Counter()
        for k in keys:
            ei, ej = pair_rep[k]
            c[_gt_pair_bucket(ei, ej, label_map)] += 1
        return {str(bk): int(c[bk]) for bk in sorted(c.keys())}

    lost_by_bucket = _bucket_counts_for_pair_keys(pair_lost)
    rescued_by_bucket = _bucket_counts_for_pair_keys(rescued_pairs)

    hc_row_by_bucket_out = {bk: dict(v) for bk, v in sorted(hc_row_by_bucket.items())}
    for bk, row in hc_row_by_bucket_out.items():
        row["n_kept"] = int(row["n_rows"] - row["n_removed"])

    removed_rows_flat = {
        f"{tier}|{rid}": int(removed_row_by_tier_and_rule[(tier, rid)])
        for tier, rid in sorted(removed_row_by_tier_and_rule.keys())
    }

    return {
        "seed_edges_csv": seed_csv_path,
        "seed_rules_and_channels_notes": {
            "hard_v1_default_rules_other_than_rare_exact_url_template": [
                "exact_attachment_hash (attachment channel)",
                "exact_html_fingerprint (html_structure_fingerprint channel)",
                "exact_normalized_url (url channel)",
            ],
            "slash_stem_simulation_targets": (
                "Hard: only rows with rule_id rare_exact_url_template on stem channel where the shared "
                "stem token is literally '/'. Corroborated: rows that pass corroborated_v1-style weak/semantic "
                "rules before ablation but fail after treating stem overlap as OFF unless "
                "(stem_set_i intersect stem_set_j) - {'/'} is non-empty."
            ),
            "not_targeted_by_this_simulation": (
                "semantic_strong and semantic_sender rows are never removed here; they can keep a canonical "
                "pair in the seed union even when all hard/corroborated rows for that pair are removed."
            ),
        },
        "n_seed_edges_all_rows_total": n_total_seed_rows,
        "seed_tier_value_counts_all_rows": tier_counts,
        "n_seed_rows_removed_by_slash_stem_simulation": int(n_rows_removed),
        "n_seed_rows_unchanged": int(n_total_seed_rows - n_rows_removed),
        "removed_row_reason_counts": dict(sorted(removed_reason_counts.items(), key=lambda x: (-x[1], x[0]))),
        "removed_rows_by_seed_tier_and_rule_id": removed_rows_flat,
        "canonical_seed_pair_union": {
            "definition": (
                "Distinct unordered email pairs appearing in seed_edges_all. After simulation, a pair "
                "counts as present if at least one CSV row for that pair is not removed. "
                "Fully lost = no row survives for that pair."
            ),
            "n_distinct_pairs_before": int(len(pair_any)),
            "n_distinct_pairs_after": int(len(pair_survives)),
            "n_distinct_pairs_fully_lost": int(len(pair_lost)),
            "distinct_pairs_fully_lost_by_gt_pair_bucket": lost_by_bucket,
            "headline_pair_losses_using_gt_campaign_labels": {
                "n_pairs_fully_lost_gt_same_campaign": int(lost_by_bucket.get("same_campaign", 0)),
                "n_pairs_fully_lost_gt_cross_campaign": int(lost_by_bucket.get("cross_campaign", 0)),
                "n_pairs_fully_lost_gt_partial_or_unlabeled_combined": int(
                    lost_by_bucket.get("partial_gt", 0) + lost_by_bucket.get("no_gt", 0)
                ),
            },
            "n_distinct_pairs_had_removed_hard_or_corro_row_but_union_survives_other_seed_rule": int(
                len(rescued_pairs)
            ),
            "rescued_pairs_by_gt_pair_bucket": rescued_by_bucket,
            "n_distinct_pairs_where_every_row_tier_is_hard_or_corroborated_only": int(len(only_hard_corro_pairs)),
            "n_distinct_pairs_fully_lost_and_only_ever_hard_or_corroborated_tiers": int(
                len(pair_lost & only_hard_corro_pairs)
            ),
        },
        "hard_corroborated_seed_row_counts_by_gt_pair_bucket": hc_row_by_bucket_out,
        "interpretation": (
            "Primary headline metrics are under canonical_seed_pair_union.headline_pair_losses_using_gt_campaign_labels: "
            "same_campaign pairs fully lost are GT-coherent edges removed from the seed union when no other seed row "
            "covers the pair; cross_campaign are GT-inconsistent pairs fully lost. "
            "hard_corroborated_seed_row_counts_* is row-level (multiple rows can map to one pair)."
        ),
    }


def analyze(
    *,
    nodes_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    seed_hard_df: pd.DataFrame,
    seed_all_df: pd.DataFrame,
    require_min_weak: int,
    sample_edges: int,
    label_map: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if "external_id" not in nodes_df.columns:
        raise ValueError("anchor_graph_nodes.csv must have external_id")
    if "stem_set" not in nodes_df.columns:
        raise ValueError("anchor_graph_nodes.csv must have stem_set")

    stem_by_eid: dict[str, set[str]] = {}
    slash_emails: list[str] = []
    for _, r in nodes_df.iterrows():
        eid = str(r["external_id"]).strip()
        st = _stem_set_for_row(r)
        stem_by_eid[eid] = st
        if "/" in st:
            slash_emails.append(eid)

    n_nodes = int(len(nodes_df))
    n_slash_email = len(slash_emails)
    pct_slash_email = float(n_slash_email / max(1, n_nodes))

    weak_cols = _weak_overlap_cols(edges_df)
    if not weak_cols:
        raise ValueError(
            "anchor_graph_edges_unscored.csv has no has_<channel>_overlap columns "
            f"for default weak list {_DEFAULT_WEAK_BASES!r}"
        )

    req = int(require_min_weak)

    rows_cross = 0
    rows_self = 0
    stem_inter_has_slash = 0
    stem_inter_only_slash = 0
    stem_inter_slash_and_more = 0
    corro_orig_weak = 0
    corro_ablated_weak = 0
    corro_lost_weak_only = 0
    edge_has_stem_col_true = 0
    edge_stem_mismatch_recomputed = 0

    gt_slash: Counter[str] = Counter()
    gt_slash_only: Counter[str] = Counter()

    flagged_sample: list[dict[str, Any]] = []

    for _, e in edges_df.iterrows():
        a = str(e.get("email_a", "")).strip()
        b = str(e.get("email_b", "")).strip()
        if not a or not b:
            continue
        if a == b:
            rows_self += 1
            continue
        rows_cross += 1

        sa = stem_by_eid.get(a, set())
        sb = stem_by_eid.get(b, set())
        inter = sa & sb
        inter_no = inter - {"/"}

        col_stem = "has_stem_overlap" in edges_df.columns and bool(e.get("has_stem_overlap"))
        re_stem = bool(inter)
        re_stem_no = bool(inter_no)
        if col_stem:
            edge_has_stem_col_true += 1
        if col_stem != re_stem:
            edge_stem_mismatch_recomputed += 1

        if "/" in inter:
            stem_inter_has_slash += 1
            if inter <= {"/"}:
                stem_inter_only_slash += 1
            else:
                stem_inter_slash_and_more += 1
            if label_map is not None:
                bucket = _gt_pair_bucket(a, b, label_map)
                gt_slash[bucket] += 1
                if inter <= {"/"}:
                    gt_slash_only[bucket] += 1
            if len(flagged_sample) < sample_edges:
                sample_row: dict[str, Any] = {
                    "email_a": a,
                    "email_b": b,
                    "stem_intersection": sorted(inter),
                    "stem_intersection_without_slash": sorted(inter_no),
                    "has_stem_overlap_column": col_stem,
                }
                if label_map is not None:
                    sample_row["gt_pair_bucket"] = _gt_pair_bucket(a, b, label_map)
                    sample_row["gt_campaign_a"] = label_map.get(a)
                    sample_row["gt_campaign_b"] = label_map.get(b)
                flagged_sample.append(sample_row)

        n_o = _n_weak_hits(e, weak_cols)
        n_a = _n_weak_hits_ablate_stem(e, weak_cols, stem_overlap_without_slash=re_stem_no)
        if n_o >= req:
            corro_orig_weak += 1
        if n_a >= req:
            corro_ablated_weak += 1
        if n_o >= req and n_a < req:
            corro_lost_weak_only += 1

    if (
        seed_hard_df.empty
        or "rule_id" not in seed_hard_df.columns
        or "evidence_value" not in seed_hard_df.columns
    ):
        hard_slash = seed_hard_df.iloc[0:0].copy()
    else:
        hard_slash = seed_hard_df[
            (seed_hard_df["rule_id"].astype(str) == "rare_exact_url_template")
            & (seed_hard_df["evidence_value"].astype(str).str.strip() == "/")
        ]
    n_hard_slash = int(len(hard_slash))

    if seed_all_df.empty or "seed_tier" not in seed_all_df.columns:
        corr_all = seed_all_df.iloc[0:0].copy()
    else:
        corr_all = seed_all_df[seed_all_df["seed_tier"].astype(str) == "corroborated"]
    n_corr_total = int(len(corr_all))

    out: dict[str, Any] = {
        "nodes": {
            "n_anchor_nodes": n_nodes,
            "n_emails_with_slash_in_stem_set": n_slash_email,
            "pct_emails_with_slash_in_stem_set": round(pct_slash_email * 100.0, 4),
        },
        "edges_unscored": {
            "n_rows": int(len(edges_df)),
            "n_cross_email_rows": rows_cross,
            "n_self_email_rows": rows_self,
            "n_cross_email_with_slash_in_stem_intersection": stem_inter_has_slash,
            "n_cross_email_stem_intersection_is_only_slash": stem_inter_only_slash,
            "n_cross_email_stem_intersection_has_slash_and_other_stems": stem_inter_slash_and_more,
            "pct_cross_email_with_slash_in_stem_intersection": round(
                stem_inter_has_slash / max(1, rows_cross) * 100.0, 4
            ),
            "n_rows_has_stem_overlap_true": edge_has_stem_col_true,
            "n_rows_has_stem_overlap_mismatch_vs_recomputed": edge_stem_mismatch_recomputed,
        },
        "same_vs_cross_campaign_edges": None
        if label_map is None
        else {
            "_comment": "Cross-email anchor edges only. Requires --gt-json.",
            "stem_intersection_contains_slash": {
                "n_same_campaign": int(gt_slash["same_campaign"]),
                "n_cross_campaign": int(gt_slash["cross_campaign"]),
                "n_partial_gt": int(gt_slash["partial_gt"]),
                "n_no_gt": int(gt_slash["no_gt"]),
                "n_fully_labeled": int(gt_slash["same_campaign"] + gt_slash["cross_campaign"]),
                "pct_cross_campaign_of_fully_labeled": round(
                    float(gt_slash["cross_campaign"])
                    / max(1, int(gt_slash["same_campaign"] + gt_slash["cross_campaign"]))
                    * 100.0,
                    4,
                ),
            },
            "stem_intersection_is_only_slash": {
                "n_same_campaign": int(gt_slash_only["same_campaign"]),
                "n_cross_campaign": int(gt_slash_only["cross_campaign"]),
                "n_partial_gt": int(gt_slash_only["partial_gt"]),
                "n_no_gt": int(gt_slash_only["no_gt"]),
                "n_fully_labeled": int(gt_slash_only["same_campaign"] + gt_slash_only["cross_campaign"]),
                "pct_cross_campaign_of_fully_labeled": round(
                    float(gt_slash_only["cross_campaign"])
                    / max(1, int(gt_slash_only["same_campaign"] + gt_slash_only["cross_campaign"]))
                    * 100.0,
                    4,
                ),
            },
        },
        "gt_stratified_cross_email_slash_in_stem_intersection": None
        if label_map is None
        else {
            "n_same_campaign": int(gt_slash["same_campaign"]),
            "n_cross_campaign": int(gt_slash["cross_campaign"]),
            "n_partial_gt": int(gt_slash["partial_gt"]),
            "n_no_gt": int(gt_slash["no_gt"]),
            "n_total_with_slash_in_intersection": int(sum(gt_slash.values())),
            "pct_cross_campaign_among_fully_labeled": round(
                float(gt_slash["cross_campaign"])
                / max(1, int(gt_slash["same_campaign"] + gt_slash["cross_campaign"]))
                * 100.0,
                4,
            ),
        },
        "gt_stratified_cross_email_slash_only_stem_intersection": None
        if label_map is None
        else {
            "n_same_campaign": int(gt_slash_only["same_campaign"]),
            "n_cross_campaign": int(gt_slash_only["cross_campaign"]),
            "n_partial_gt": int(gt_slash_only["partial_gt"]),
            "n_no_gt": int(gt_slash_only["no_gt"]),
            "n_total_slash_only_intersection": int(sum(gt_slash_only.values())),
            "pct_cross_campaign_among_fully_labeled": round(
                float(gt_slash_only["cross_campaign"])
                / max(1, int(gt_slash_only["same_campaign"] + gt_slash_only["cross_campaign"]))
                * 100.0,
                4,
            ),
        },
        "corroborated_weak_multi_proxy": {
            "require_min_support_channels": req,
            "weak_overlap_columns_used": weak_cols,
            "n_cross_email_edges_weak_multi_original": corro_orig_weak,
            "n_cross_email_edges_weak_multi_ablate_slash_stem": corro_ablated_weak,
            "n_cross_email_edges_weak_multi_lost_after_ablation": corro_lost_weak_only,
            "pct_cross_email_edges_lost_weak_multi": round(
                corro_lost_weak_only / max(1, rows_cross) * 100.0, 4
            ),
        },
        "seeds_optional": {
            "n_hard_seed_edges_rare_exact_url_template_evidence_slash": n_hard_slash,
            "n_corroborated_seed_rows_total": n_corr_total,
        },
        "sample_edges_slash_in_stem_intersection": flagged_sample,
        "interpretation_notes": [
            "Look for same_vs_cross_campaign_edges (with --gt-json) for plain-language same vs cross campaign counts.",
            "Self-email rows (email_a == email_b) are counted separately; corroborated rules skip self-pairs in seed generation.",
            "Ablated stem overlap: treat stem weak channel as ON only when (stem_a intersect stem_b) - {'/'} is non-empty.",
            "With --gt-json: gt_stratified_* counts only cross-email edges where '/' is in stem intersection; "
            "slash_only_stem_intersection means (stem_a intersect stem_b) is non-empty and is a subset of {'/'}.",
            "With --seed-impact + --bundle-dir + --gt-json: seed_positive_slash_stem_removal_simulation reports "
            "canonical pair union losses (see canonical_seed_pair_union) and row-level aggregates only.",
            "partial_gt: exactly one endpoint has a GT campaign label; no_gt: neither labeled.",
        ],
    }
    return out


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--bundle-dir",
        type=Path,
        default=None,
        help="Directory containing anchor_graph_nodes.csv and anchor_graph_edges_unscored.csv",
    )
    ap.add_argument("--nodes-csv", type=Path, default=None)
    ap.add_argument("--edges-csv", type=Path, default=None)
    ap.add_argument(
        "--seed-root",
        type=Path,
        default=None,
        help="e.g. seed_candidate_workflow/output/graph_bundles/<graph_id>/seed — latest seed_generation_* used if --graph-id set",
    )
    ap.add_argument(
        "--graph-id",
        type=str,
        default="",
        help="With --seed-root, pick latest .../seed/<graph_id>/seed_generation_*/",
    )
    ap.add_argument(
        "--gt-json",
        type=Path,
        default=None,
        help="Ground-truth clusters JSON (e.g. ground_truth.dedup_task_identity.json) to stratify "
        "slash stem edges by same-campaign vs cross-campaign vs partial/missing labels.",
    )
    ap.add_argument(
        "--seed-impact",
        action="store_true",
        help="Resolve seed_edges_all.csv from the graph bundle and append hard+corroborated slash-removal "
        "simulation (requires --bundle-dir and --gt-json).",
    )
    ap.add_argument("--require-min-weak", type=int, default=2, help="Corroborated weak_multi threshold (default 2).")
    ap.add_argument("--sample-edges", type=int, default=50, help="Max example edges in JSON output.")
    ap.add_argument("--out-json", type=Path, default=None, help="Write full summary JSON.")
    ap.add_argument(
        "--out-flagged-csv",
        type=Path,
        default=None,
        help="Optional CSV of sample cross-email pairs with slash in stem intersection.",
    )
    args = ap.parse_args()

    root = _repo_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    bundle = args.bundle_dir.expanduser().resolve() if args.bundle_dir else None
    if bundle is not None:
        nodes_p = bundle / "anchor_graph_nodes.csv"
        edges_p = bundle / "anchor_graph_edges_unscored.csv"
    else:
        if not args.nodes_csv or not args.edges_csv:
            ap.error("Pass --bundle-dir or both --nodes-csv and --edges-csv")
        nodes_p = args.nodes_csv.expanduser().resolve()
        edges_p = args.edges_csv.expanduser().resolve()

    if not nodes_p.is_file():
        raise SystemExit(f"Missing nodes CSV: {nodes_p}")
    if not edges_p.is_file():
        raise SystemExit(f"Missing edges CSV: {edges_p}")

    nodes_df = pd.read_csv(nodes_p, low_memory=False)
    edges_df = pd.read_csv(edges_p, low_memory=False)

    seed_dir: Path | None = None
    if args.seed_root and args.graph_id:
        seed_dir = _find_latest_seed_dir(args.seed_root.expanduser().resolve(), str(args.graph_id).strip())
    seed_hard = _load_optional_seed_hard(seed_dir)
    seed_all = _load_optional_seed_all(seed_dir)

    label_map: dict[str, Any] | None = None
    if args.gt_json:
        gp = args.gt_json.expanduser().resolve()
        if not gp.is_file():
            raise SystemExit(f"Missing --gt-json: {gp}")
        label_map = load_gt_label_map(gp)

    summary = analyze(
        nodes_df=nodes_df,
        edges_df=edges_df,
        seed_hard_df=seed_hard,
        seed_all_df=seed_all,
        require_min_weak=int(args.require_min_weak),
        sample_edges=int(args.sample_edges),
        label_map=label_map,
    )
    summary["inputs"] = {
        "nodes_csv": str(nodes_p),
        "edges_csv": str(edges_p),
        "seed_dir_used": str(seed_dir) if seed_dir is not None else None,
        "gt_json": str(args.gt_json.expanduser().resolve()) if args.gt_json else None,
        "seed_impact": bool(args.seed_impact),
    }

    if args.seed_impact:
        if label_map is None:
            ap.error("--seed-impact requires --gt-json")
        if bundle is None:
            ap.error("--seed-impact requires --bundle-dir (anchor dir: .../graph_bundles/<graph_id>/anchor/<graph_id>)")
        seed_csv = _resolve_seed_edges_all_csv(bundle)
        if seed_csv is None or not seed_csv.is_file():
            summary["seed_positive_slash_stem_removal_simulation"] = {
                "error": "Could not resolve seed_edges_all.csv under bundle seed/<graph_id>/seed_generation_*",
                "anchor_dir": str(bundle),
            }
        else:
            stem_by_eid_seed: dict[str, set[str]] = {}
            for _, r in nodes_df.iterrows():
                eid = str(r["external_id"]).strip()
                stem_by_eid_seed[eid] = _stem_set_for_row(r)
            weak_cols_seed = _weak_overlap_cols(edges_df)
            if not weak_cols_seed:
                summary["seed_positive_slash_stem_removal_simulation"] = {
                    "error": "anchor_graph_edges_unscored.csv has no weak overlap columns for corroborated simulation",
                }
            else:
                seed_df_imp = pd.read_csv(seed_csv, low_memory=False)
                summary["seed_positive_slash_stem_removal_simulation"] = analyze_seed_positive_slash_removals(
                    seed_df=seed_df_imp,
                    edges_df=edges_df,
                    label_map=label_map,
                    weak_cols=weak_cols_seed,
                    stem_by_eid=stem_by_eid_seed,
                    seed_csv_path=str(seed_csv),
                )

    print(json.dumps(summary, indent=2, ensure_ascii=False))

    if args.out_json:
        outp = args.out_json.expanduser().resolve()
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nWrote: {outp}", file=sys.stderr)

    if args.out_flagged_csv and summary.get("sample_edges_slash_in_stem_intersection"):
        outc = args.out_flagged_csv.expanduser().resolve()
        outc.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(summary["sample_edges_slash_in_stem_intersection"]).to_csv(outc, index=False)
        print(f"Wrote sample edges: {outc}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
