from __future__ import annotations

import json
import math
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.metrics import homogeneity_score

from analysis.utils import graph_structure_helpers as gh
from analysis.utils.config_run_fields import resolve_graph_id
from analysis.utils import raw_gnn_notebook as rn
from analysis.utils.anchor_graph_helpers import load_anchor_graph_artifacts

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None


def _slugify(s: str) -> str:
    t = re.sub(r"[^A-Za-z0-9._-]+", "_", str(s).strip())
    t = re.sub(r"_+", "_", t).strip("_.-")
    return t or "unknown"


def _to_set_cell(v: Any) -> set[str]:
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


def _ensure_node_set_columns(nodes_df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = nodes_df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = out[c].map(_to_set_cell)
    return out


def _channel_col(channel: str) -> str:
    c = str(channel).strip().lower()
    if c.endswith("_set"):
        return c
    return f"{c}_set"


def _overlap_col(channel: str) -> str:
    base = str(channel).strip().lower()
    if base.endswith("_set"):
        base = base[:-4]
    return f"has_{base}_overlap"


def _artifact_df(nodes_df: pd.DataFrame, col: str) -> Counter[str]:
    c = Counter()
    if col not in nodes_df.columns:
        return c
    for s in nodes_df[col].tolist():
        vals = _to_set_cell(s)
        for x in vals:
            c[str(x)] += 1
    return c


def _artifact_idf(df_val: int, n_docs: int) -> float:
    return float(math.log((1.0 + n_docs) / (1.0 + max(1, int(df_val)))))


def _collect_edge_evidence_rows(
    *,
    edges_df: pd.DataFrame,
    node_sets: dict[str, dict[str, set[str]]],
    df_by_channel: dict[str, Counter[str]],
    n_docs: int,
    rules: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for _, e in edges_df.iterrows():
        a = str(e["email_a"])
        b = str(e["email_b"])
        if a == b:
            continue
        for r in rules:
            channel = str(r.get("channel", "")).strip()
            if not channel:
                continue
            evidence_type = str(r.get("evidence_type", channel))
            overlap_col = _overlap_col(channel)
            if overlap_col in edges_df.columns and not bool(e.get(overlap_col, False)):
                continue
            col = _channel_col(channel)
            aset = node_sets.get(a, {}).get(col, set())
            bset = node_sets.get(b, {}).get(col, set())
            shared = sorted(aset & bset)
            if not shared:
                continue

            max_df = r.get("max_artifact_df")
            max_idf = r.get("min_artifact_idf")
            for val in shared:
                df_val = int(df_by_channel.get(col, Counter()).get(str(val), 0))
                if max_df is not None and df_val > int(max_df):
                    continue
                idf = _artifact_idf(df_val=df_val, n_docs=n_docs)
                if max_idf is not None and idf < float(max_idf):
                    continue
                rows.append(
                    {
                        "email_i": min(a, b),
                        "email_j": max(a, b),
                        "evidence_type": evidence_type,
                        "evidence_value": str(val),
                        "evidence_rarity": float(idf),
                        "artifact_df": int(df_val),
                        "seed_tier": "hard",
                        "seed_generator": "hard_v1",
                        "rule_id": str(r.get("rule_id", evidence_type)),
                    }
                )
    return rows


def _make_node_sets_map(nodes_df: pd.DataFrame, cols: list[str]) -> dict[str, dict[str, set[str]]]:
    out: dict[str, dict[str, set[str]]] = {}
    for _, r in nodes_df.iterrows():
        eid = str(r["external_id"])
        d: dict[str, set[str]] = {}
        for c in cols:
            if c in nodes_df.columns:
                d[c] = _to_set_cell(r.get(c))
        out[eid] = d
    return out


def _hard_v1_default_rules() -> list[dict[str, Any]]:
    return [
        {
            "rule_id": "exact_attachment_hash",
            "channel": "attachment",
            "evidence_type": "exact_attachment_hash",
            "max_artifact_df": 30,
        },
        {
            "rule_id": "exact_html_fingerprint",
            "channel": "html_structure_fingerprint",
            "evidence_type": "exact_html_fingerprint",
            "max_artifact_df": 40,
        },
        {
            "rule_id": "exact_normalized_url",
            "channel": "url",
            "evidence_type": "exact_normalized_url",
            "max_artifact_df": 20,
        },
        {
            "rule_id": "rare_exact_url_template",
            "channel": "stem",
            "evidence_type": "rare_exact_url_template",
            "max_artifact_df": 8,
        },
    ]


def generate_hard_seed_edges_v1(
    *,
    nodes_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    generator_cfg: dict[str, Any],
) -> pd.DataFrame:
    rules = generator_cfg.get("rules")
    if not isinstance(rules, list) or not rules:
        rules = _hard_v1_default_rules()
    set_cols = sorted({_channel_col(str(r.get("channel", ""))) for r in rules if str(r.get("channel", "")).strip()})
    nodes = _ensure_node_set_columns(nodes_df, set_cols)
    node_sets = _make_node_sets_map(nodes, set_cols)
    df_by_channel = {c: _artifact_df(nodes, c) for c in set_cols}
    n_docs = int(len(nodes))
    rows = _collect_edge_evidence_rows(
        edges_df=edges_df,
        node_sets=node_sets,
        df_by_channel=df_by_channel,
        n_docs=n_docs,
        rules=rules,
    )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.drop_duplicates(
        subset=["email_i", "email_j", "evidence_type", "evidence_value", "rule_id"]
    ).reset_index(drop=True)
    out = out.sort_values(
        ["evidence_rarity", "artifact_df", "email_i", "email_j"],
        ascending=[False, True, True, True],
    ).reset_index(drop=True)
    return out


def _corroborated_v1_default_cfg() -> dict[str, Any]:
    return {
        "weak_channels": ["sender", "sender_email_domain", "domain", "stem", "return_path_domain", "received_host", "origin_ip"],
        "require_min_support_channels": 2,
        "semantic_support": {
            "enabled": True,
            "min_semantic_score": 0.97,
            "require_non_semantic_support": True,
            "min_non_semantic_support_channels": 1,
        },
    }


def generate_corroborated_seed_edges_v1(
    *,
    nodes_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    generator_cfg: dict[str, Any],
) -> pd.DataFrame:
    cfg = {**_corroborated_v1_default_cfg(), **(generator_cfg or {})}
    weak_channels = [str(x).strip() for x in (cfg.get("weak_channels") or []) if str(x).strip()]
    weak_overlap_cols = [_overlap_col(ch) for ch in weak_channels]
    req_min_support = int(cfg.get("require_min_support_channels", 2))

    sem_cfg = cfg.get("semantic_support") or {}
    sem_enabled = bool(sem_cfg.get("enabled", True))
    min_sem = float(sem_cfg.get("min_semantic_score", 0.97))
    req_non_sem = bool(sem_cfg.get("require_non_semantic_support", True))
    min_non_sem = int(sem_cfg.get("min_non_semantic_support_channels", 1))

    if edges_df.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for _, e in edges_df.iterrows():
        a = str(e["email_a"])
        b = str(e["email_b"])
        if a == b:
            continue
        support_channels: list[str] = []
        for col, ch in zip(weak_overlap_cols, weak_channels, strict=False):
            if col in edges_df.columns and bool(e.get(col, False)):
                support_channels.append(ch)
        n_support = len(support_channels)
        sem_score = float(pd.to_numeric(e.get("semantic_score"), errors="coerce"))
        sem_hit = bool(np.isfinite(sem_score) and sem_score >= min_sem) if sem_enabled else False

        rule_by_weak = n_support >= req_min_support
        rule_by_sem = sem_hit and (n_support >= min_non_sem if req_non_sem else True)
        if not (rule_by_weak or rule_by_sem):
            continue

        # Aggregate rarity from available infra idf sums for supporting channels.
        rarity_parts: list[float] = []
        for ch in support_channels:
            base = _channel_col(ch).replace("_set", "")
            idf_col = f"shared_{base}_idf_sum"
            if idf_col in edges_df.columns:
                v = pd.to_numeric(e.get(idf_col), errors="coerce")
                if pd.notna(v):
                    rarity_parts.append(float(v))
        if sem_hit:
            rarity_parts.append(float(max(0.0, sem_score)))
        evidence_rarity = float(np.mean(rarity_parts)) if rarity_parts else float("nan")

        evidence_fields = {
            "support_channels": sorted(set(support_channels)),
            "n_support_channels": int(n_support),
            "semantic_score": float(sem_score) if np.isfinite(sem_score) else None,
            "semantic_support": bool(sem_hit),
            "rule_triggered": "weak_multi" if rule_by_weak else "semantic_plus_support",
        }
        rows.append(
            {
                "email_i": min(a, b),
                "email_j": max(a, b),
                "evidence_type": "corroborated_multi_signal",
                "evidence_value": json.dumps(evidence_fields, ensure_ascii=False, sort_keys=True),
                "evidence_rarity": evidence_rarity,
                "artifact_df": np.nan,
                "seed_tier": "corroborated",
                "seed_generator": "corroborated_v1",
                "rule_id": str(evidence_fields["rule_triggered"]),
                "n_support_channels": int(n_support),
                "semantic_support": bool(sem_hit),
                "evidence_fields_json": json.dumps(evidence_fields, ensure_ascii=False),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.drop_duplicates(
        subset=["email_i", "email_j", "seed_generator", "rule_id"]
    ).reset_index(drop=True)
    out = out.sort_values(
        ["n_support_channels", "semantic_support", "evidence_rarity", "email_i", "email_j"],
        ascending=[False, False, False, True, True],
    ).reset_index(drop=True)
    return out


def _component_homogeneity_on_gt(
    *,
    seed_edges_df: pd.DataFrame,
    gt_label_map: dict[str, Any],
) -> dict[str, float]:
    g = nx.Graph()
    if not seed_edges_df.empty:
        g.add_edges_from(
            list(
                zip(
                    seed_edges_df["email_i"].astype(str).tolist(),
                    seed_edges_df["email_j"].astype(str).tolist(),
                    strict=False,
                )
            )
        )
    gt = {str(k): v for k, v in gt_label_map.items()}
    comps = list(nx.connected_components(g))
    comp_scores: list[float] = []
    covered = 0
    for comp in comps:
        labels = [gt[e] for e in comp if e in gt]
        if not labels:
            continue
        cnt = Counter(labels)
        covered += len(labels)
        comp_scores.append(max(cnt.values()) / max(1, len(labels)))
    if not comp_scores:
        return {
            "components_with_gt": 0.0,
            "component_homogeneity_mean": float("nan"),
            "component_homogeneity_weighted": float("nan"),
        }
    # Weighted by number of GT-covered emails in each component.
    weighted_num = 0.0
    weighted_den = 0.0
    for comp in comps:
        labels = [gt[e] for e in comp if e in gt]
        if not labels:
            continue
        cnt = Counter(labels)
        score = max(cnt.values()) / max(1, len(labels))
        w = float(len(labels))
        weighted_num += score * w
        weighted_den += w
    return {
        "components_with_gt": float(len(comp_scores)),
        "component_homogeneity_mean": float(np.mean(comp_scores)),
        "component_homogeneity_weighted": float(weighted_num / max(1.0, weighted_den)),
    }


def _labeled_pair_precision(
    *,
    seed_edges_df: pd.DataFrame,
    gt_label_map: dict[str, Any],
) -> dict[str, float]:
    if seed_edges_df.empty:
        return {
            "n_labeled_pairs": 0.0,
            "n_labeled_positive_pairs": 0.0,
            "labeled_pair_precision": float("nan"),
        }
    gt = {str(k): v for k, v in gt_label_map.items()}
    n_lab = 0
    n_pos = 0
    for _, r in seed_edges_df.iterrows():
        a = str(r["email_i"])
        b = str(r["email_j"])
        if a not in gt or b not in gt:
            continue
        n_lab += 1
        if gt[a] == gt[b]:
            n_pos += 1
    return {
        "n_labeled_pairs": float(n_lab),
        "n_labeled_positive_pairs": float(n_pos),
        "labeled_pair_precision": float(n_pos / max(1, n_lab)) if n_lab > 0 else float("nan"),
    }


def _seed_graph_metrics(seed_edges_df: pd.DataFrame) -> dict[str, Any]:
    g = nx.Graph()
    if not seed_edges_df.empty:
        g.add_edges_from(
            list(
                zip(
                    seed_edges_df["email_i"].astype(str).tolist(),
                    seed_edges_df["email_j"].astype(str).tolist(),
                    strict=False,
                )
            )
        )
    comp_sizes = sorted((len(c) for c in nx.connected_components(g)), reverse=True)
    return {
        "n_seed_edges": int(len(seed_edges_df)),
        "n_emails_touched": int(g.number_of_nodes()),
        "n_components": int(len(comp_sizes)),
        "component_size_distribution_top50": comp_sizes[:50],
    }


def _pair_set(df: pd.DataFrame) -> set[tuple[str, str]]:
    if df.empty:
        return set()
    return set(
        zip(
            df["email_i"].astype(str).tolist(),
            df["email_j"].astype(str).tolist(),
            strict=False,
        )
    )


def _corroborated_specific_metrics(
    corroborated_df: pd.DataFrame,
    hard_df: pd.DataFrame,
) -> dict[str, float]:
    if corroborated_df.empty:
        return {
            "fraction_supported_by_2plus_channels": float("nan"),
            "fraction_semantic_involving": float("nan"),
            "pair_overlap_with_hard": float("nan"),
        }
    n = float(len(corroborated_df))
    n_two_plus = float(
        (pd.to_numeric(corroborated_df.get("n_support_channels"), errors="coerce").fillna(0) >= 2).sum()
    )
    n_sem = float(corroborated_df.get("semantic_support", False).astype(bool).sum())
    corr_pairs = _pair_set(corroborated_df)
    hard_pairs = _pair_set(hard_df)
    overlap = len(corr_pairs & hard_pairs)
    return {
        "fraction_supported_by_2plus_channels": float(n_two_plus / max(1.0, n)),
        "fraction_semantic_involving": float(n_sem / max(1.0, n)),
        "pair_overlap_with_hard": float(overlap / max(1, len(corr_pairs))),
    }


def _build_union_components(
    *,
    all_node_ids: list[str],
    seed_edges_all: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    g = nx.Graph()
    g.add_nodes_from([str(x) for x in all_node_ids])
    if not seed_edges_all.empty:
        g.add_edges_from(
            list(
                zip(
                    seed_edges_all["email_i"].astype(str).tolist(),
                    seed_edges_all["email_j"].astype(str).tolist(),
                    strict=False,
                )
            )
        )
    rows: list[dict[str, Any]] = []
    comp_rows: list[dict[str, Any]] = []
    comp_sizes: list[int] = []
    for comp_id, members in enumerate(nx.connected_components(g)):
        m = sorted(str(x) for x in members)
        s = int(len(m))
        comp_sizes.append(s)
        comp_rows.append(
            {
                "component_id": int(comp_id),
                "size": int(s),
                "n_edges_internal": int(g.subgraph(m).number_of_edges()),
            }
        )
        for eid in m:
            rows.append(
                {
                    "external_id": str(eid),
                    "component_id": int(comp_id),
                    "component_size": int(s),
                    "is_singleton": bool(s == 1),
                }
            )
    members_df = pd.DataFrame(rows)
    components_df = pd.DataFrame(comp_rows).sort_values(
        ["size", "component_id"], ascending=[False, True]
    ).reset_index(drop=True)
    n_comp = int(len(comp_sizes))
    n_singleton = int(sum(1 for x in comp_sizes if x == 1))
    n_size2 = int(sum(1 for x in comp_sizes if x == 2))
    n_size3_5 = int(sum(1 for x in comp_sizes if 3 <= x <= 5))
    n_size6_plus = int(sum(1 for x in comp_sizes if x >= 6))
    summary = {
        "n_components": n_comp,
        "singleton_rate": float(n_singleton / max(1, n_comp)),
        "n_components_size1": n_singleton,
        "n_components_size2": n_size2,
        "n_components_size3_5": n_size3_5,
        "n_components_size6_plus": n_size6_plus,
        "pct_components_size2": float(n_size2 / max(1, n_comp)),
        "pct_components_size3_5": float(n_size3_5 / max(1, n_comp)),
        "pct_components_size6_plus": float(n_size6_plus / max(1, n_comp)),
    }
    return members_df, components_df, summary


def _b_cubed_precision(
    *,
    members_df: pd.DataFrame,
    gt_label_map: dict[str, Any],
) -> dict[str, float]:
    if members_df.empty:
        return {"n_eval": 0.0, "b_cubed_precision": float("nan"), "homogeneity": float("nan")}
    gt = {str(k): v for k, v in gt_label_map.items()}
    d = members_df.copy()
    d["external_id"] = d["external_id"].astype(str)
    d["gt_label"] = d["external_id"].map(gt)
    d = d[d["gt_label"].notna()].copy()
    if d.empty:
        return {"n_eval": 0.0, "b_cubed_precision": float("nan"), "homogeneity": float("nan")}
    d["component_id"] = d["component_id"].astype(int)
    comp_sizes = d.groupby("component_id", dropna=False).size().rename("comp_n")
    cross = (
        d.groupby(["component_id", "gt_label"], dropna=False)
        .size()
        .rename("n")
        .reset_index()
    )
    cross = cross.merge(comp_sizes.reset_index(), on="component_id", how="left")
    cross["prec"] = pd.to_numeric(cross["n"], errors="coerce") / pd.to_numeric(
        cross["comp_n"], errors="coerce"
    )
    # B-cubed precision averaged over emails (weight by n in contingency cell).
    weighted = (cross["prec"] * cross["n"]).sum()
    n_eval = float(len(d))
    y_true = d["gt_label"].tolist()
    y_pred = d["component_id"].tolist()
    return {
        "n_eval": n_eval,
        "b_cubed_precision": float(weighted / max(1.0, n_eval)),
        "homogeneity": float(homogeneity_score(y_true, y_pred)),
    }


def _labeled_campaign_touch_rate(
    *,
    touched_email_ids: set[str],
    gt_label_map: dict[str, Any],
) -> dict[str, float]:
    if not gt_label_map:
        return {"n_labeled_campaigns": 0.0, "n_labeled_campaigns_touched": 0.0, "pct_labeled_campaigns_touched": float("nan")}
    camp_to_ids: dict[str, set[str]] = {}
    for eid, camp in gt_label_map.items():
        k = str(camp)
        camp_to_ids.setdefault(k, set()).add(str(eid))
    n_c = int(len(camp_to_ids))
    n_t = int(sum(1 for ids in camp_to_ids.values() if ids & touched_email_ids))
    return {
        "n_labeled_campaigns": float(n_c),
        "n_labeled_campaigns_touched": float(n_t),
        "pct_labeled_campaigns_touched": float(n_t / max(1, n_c)),
    }


def _coverage_diagnostics(
    *,
    all_emails: list[str],
    hard_edges: pd.DataFrame,
    corroborated_edges: pd.DataFrame,
    union_edges: pd.DataFrame,
) -> dict[str, Any]:
    emails_total = int(len(all_emails))
    touched_hard = set(hard_edges["email_i"].astype(str)).union(set(hard_edges["email_j"].astype(str))) if not hard_edges.empty else set()
    touched_corr = set(corroborated_edges["email_i"].astype(str)).union(set(corroborated_edges["email_j"].astype(str))) if not corroborated_edges.empty else set()
    touched_union = set(union_edges["email_i"].astype(str)).union(set(union_edges["email_j"].astype(str))) if not union_edges.empty else set()
    return {
        "emails_total": emails_total,
        "emails_touched_by_hard": int(len(touched_hard)),
        "emails_touched_by_corroborated": int(len(touched_corr)),
        "emails_touched_by_union": int(len(touched_union)),
        "pct_emails_touched_by_hard": float(len(touched_hard) / max(1, emails_total)),
        "pct_emails_touched_by_corroborated": float(len(touched_corr) / max(1, emails_total)),
        "pct_emails_touched_by_union": float(len(touched_union) / max(1, emails_total)),
        "_touched_hard": touched_hard,
        "_touched_corroborated": touched_corr,
        "_touched_union": touched_union,
    }


def _component_concentration_diagnostics(
    *,
    union_components_df: pd.DataFrame,
    seeded_emails: set[str],
) -> dict[str, Any]:
    if union_components_df.empty:
        return {
            "largest_component_size": 0,
            "largest_component_pct_of_seeded_emails": float("nan"),
            "top_10_component_sizes": [],
            "n_components_size_ge_10": 0,
            "pct_seeded_emails_in_components_size_ge_10": float("nan"),
        }
    sizes = pd.to_numeric(union_components_df["size"], errors="coerce").fillna(0).astype(int).tolist()
    sizes_sorted = sorted(sizes, reverse=True)
    largest = int(sizes_sorted[0]) if sizes_sorted else 0
    seeded_n = int(len(seeded_emails))
    n_ge_10 = int(sum(1 for x in sizes if int(x) >= 10))
    seeded_in_ge_10 = int(sum(int(x) for x in sizes if int(x) >= 10))
    return {
        "largest_component_size": largest,
        "largest_component_pct_of_seeded_emails": float(largest / max(1, seeded_n)),
        "top_10_component_sizes": [int(x) for x in sizes_sorted[:10]],
        "n_components_size_ge_10": n_ge_10,
        "pct_seeded_emails_in_components_size_ge_10": float(seeded_in_ge_10 / max(1, seeded_n)),
    }


def _union_component_purity_spread(
    *,
    union_members_df: pd.DataFrame,
    gt_label_map: dict[str, Any],
    mixed_threshold: float = 0.90,
) -> dict[str, Any]:
    if union_members_df.empty or not gt_label_map:
        return {
            "n_gt_covered_components": 0.0,
            "median_component_purity": float("nan"),
            "p25_component_purity": float("nan"),
            "p10_component_purity": float("nan"),
            "n_mixed_components": 0.0,
            "fraction_gt_covered_seeded_emails_in_mixed_components": float("nan"),
            "mixed_definition": f"mixed if max_campaign_fraction < {float(mixed_threshold):.2f}",
        }
    gt = {str(k): v for k, v in gt_label_map.items()}
    d = union_members_df.copy()
    d["external_id"] = d["external_id"].astype(str)
    d["gt_label"] = d["external_id"].map(gt)
    d = d[d["gt_label"].notna()].copy()
    if d.empty:
        return {
            "n_gt_covered_components": 0.0,
            "median_component_purity": float("nan"),
            "p25_component_purity": float("nan"),
            "p10_component_purity": float("nan"),
            "n_mixed_components": 0.0,
            "fraction_gt_covered_seeded_emails_in_mixed_components": float("nan"),
            "mixed_definition": f"mixed if max_campaign_fraction < {float(mixed_threshold):.2f}",
        }

    purities: list[float] = []
    mixed_comp_ids: set[int] = set()
    gt_cov_by_comp: dict[int, int] = {}
    for comp_id, sub in d.groupby("component_id", dropna=False):
        labels = sub["gt_label"].tolist()
        if not labels:
            continue
        cnt = Counter(labels)
        purity = float(max(cnt.values()) / max(1, len(labels)))
        purities.append(purity)
        cid = int(comp_id)
        gt_cov_by_comp[cid] = int(len(labels))
        if purity < float(mixed_threshold):
            mixed_comp_ids.add(cid)

    if not purities:
        return {
            "n_gt_covered_components": 0.0,
            "median_component_purity": float("nan"),
            "p25_component_purity": float("nan"),
            "p10_component_purity": float("nan"),
            "n_mixed_components": 0.0,
            "fraction_gt_covered_seeded_emails_in_mixed_components": float("nan"),
            "mixed_definition": f"mixed if max_campaign_fraction < {float(mixed_threshold):.2f}",
        }
    pur_arr = np.asarray(purities, dtype=float)
    n_gt_covered = int(len(purities))
    n_mixed = int(len(mixed_comp_ids))
    mixed_gt_cov = int(sum(gt_cov_by_comp.get(cid, 0) for cid in mixed_comp_ids))
    total_gt_cov = int(sum(gt_cov_by_comp.values()))
    return {
        "n_gt_covered_components": float(n_gt_covered),
        "median_component_purity": float(np.nanmedian(pur_arr)),
        "p25_component_purity": float(np.nanpercentile(pur_arr, 25)),
        "p10_component_purity": float(np.nanpercentile(pur_arr, 10)),
        "n_mixed_components": float(n_mixed),
        "fraction_gt_covered_seeded_emails_in_mixed_components": float(mixed_gt_cov / max(1, total_gt_cov)),
        "mixed_definition": f"mixed if max_campaign_fraction < {float(mixed_threshold):.2f}",
    }


def _campaign_touch_distribution(
    *,
    union_members_df: pd.DataFrame,
    union_edges_df: pd.DataFrame,
    gt_label_map: dict[str, Any],
    touched_union: set[str],
) -> dict[str, Any]:
    if not gt_label_map:
        return {}
    gt = {str(k): v for k, v in gt_label_map.items()}
    campaign_to_ids: dict[str, set[str]] = {}
    for eid, camp in gt.items():
        campaign_to_ids.setdefault(str(camp), set()).add(str(eid))
    n_campaigns = int(len(campaign_to_ids))
    if n_campaigns == 0:
        return {}

    # touch>=1 and touch>=2
    touched_counts = []
    has_non_singleton = 0
    for camp, ids in campaign_to_ids.items():
        t = len(ids & touched_union)
        touched_counts.append(t)
        if t > 0:
            # non-singleton component among touched
            sub = union_members_df[
                union_members_df["external_id"].astype(str).isin(list(ids & touched_union))
            ]
            if not sub.empty and (pd.to_numeric(sub["component_size"], errors="coerce").fillna(1) > 1).any():
                has_non_singleton += 1

    n_touch1 = int(sum(1 for x in touched_counts if x >= 1))
    n_touch2 = int(sum(1 for x in touched_counts if x >= 2))

    # internal seed edge per campaign
    internal_campaigns: set[str] = set()
    if not union_edges_df.empty:
        for _, r in union_edges_df.iterrows():
            a = str(r["email_i"])
            b = str(r["email_j"])
            ca = gt.get(a)
            cb = gt.get(b)
            if ca is None or cb is None:
                continue
            if ca == cb:
                internal_campaigns.add(str(ca))
    return {
        "pct_labeled_campaigns_touched_by_at_least_1_seeded_email": float(n_touch1 / max(1, n_campaigns)),
        "pct_labeled_campaigns_touched_by_at_least_2_seeded_emails": float(n_touch2 / max(1, n_campaigns)),
        "pct_labeled_campaigns_with_at_least_1_non_singleton_seed_component": float(has_non_singleton / max(1, n_campaigns)),
        "pct_labeled_campaigns_with_at_least_1_internal_seed_edge": float(len(internal_campaigns) / max(1, n_campaigns)),
        "n_labeled_campaigns": float(n_campaigns),
    }


def _corroborated_redundancy_diagnostics(
    *,
    hard_edges: pd.DataFrame,
    corroborated_edges: pd.DataFrame,
    all_node_ids: list[str],
) -> dict[str, Any]:
    if corroborated_edges.empty:
        return {
            "n_corroborated_edges": 0,
            "n_connect_within_hard_components": 0,
            "pct_connect_within_hard_components": float("nan"),
            "n_expand_existing_hard_component": 0,
            "pct_expand_existing_hard_component": float("nan"),
            "n_create_new_structure_outside_hard": 0,
            "pct_create_new_structure_outside_hard": float("nan"),
            "n_merge_two_hard_components": 0,
            "pct_merge_two_hard_components": float("nan"),
        }
    # Build hard components (using same node universe used for union component build).
    hard_members_df, _hard_components_df, _hard_summary = _build_union_components(
        all_node_ids=[str(x) for x in all_node_ids],
        seed_edges_all=hard_edges.rename(columns={"email_i": "email_i", "email_j": "email_j"}) if not hard_edges.empty else hard_edges,
    )
    hard_comp = {
        str(r["external_id"]): int(r["component_id"])
        for _, r in hard_members_df.iterrows()
    } if not hard_members_df.empty else {}
    touched_by_hard = set(hard_edges["email_i"].astype(str)).union(set(hard_edges["email_j"].astype(str))) if not hard_edges.empty else set()
    n = int(len(corroborated_edges))
    within = 0
    expand = 0
    new_struct = 0
    merge = 0
    for _, r in corroborated_edges.iterrows():
        a = str(r["email_i"])
        b = str(r["email_j"])
        a_t = a in touched_by_hard
        b_t = b in touched_by_hard
        if not a_t and not b_t:
            new_struct += 1
            continue
        if a_t and b_t:
            ca = hard_comp.get(a)
            cb = hard_comp.get(b)
            if ca is not None and cb is not None and ca == cb:
                within += 1
            else:
                merge += 1
            continue
        expand += 1
    return {
        "n_corroborated_edges": n,
        "n_connect_within_hard_components": within,
        "pct_connect_within_hard_components": float(within / max(1, n)),
        "n_expand_existing_hard_component": expand,
        "pct_expand_existing_hard_component": float(expand / max(1, n)),
        "n_create_new_structure_outside_hard": new_struct,
        "pct_create_new_structure_outside_hard": float(new_struct / max(1, n)),
        "n_merge_two_hard_components": merge,
        "pct_merge_two_hard_components": float(merge / max(1, n)),
    }


def _manual_review_sample(
    *,
    out_dir: Path,
    seed_edges_all: pd.DataFrame,
    hard_edges: pd.DataFrame,
    corroborated_edges: pd.DataFrame,
    union_members_df: pd.DataFrame,
    gt_label_map: dict[str, Any] | None,
    random_seed: int = 1337,
) -> str:
    rng = np.random.default_rng(int(random_seed))
    if seed_edges_all.empty:
        p = out_dir / "seed_manual_review_sample.csv"
        pd.DataFrame().to_csv(p, index=False)
        return str(p)

    comp_map = {
        str(r["external_id"]): (int(r["component_id"]), int(r["component_size"]))
        for _, r in union_members_df.iterrows()
    } if not union_members_df.empty else {}
    gt = {str(k): v for k, v in (gt_label_map or {}).items()}

    hard_pairs = _pair_set(hard_edges) if not hard_edges.empty else set()
    corr_only = corroborated_edges.copy()
    if not corroborated_edges.empty:
        mask = []
        for a, b in zip(
            corroborated_edges["email_i"].astype(str),
            corroborated_edges["email_j"].astype(str),
            strict=False,
        ):
            mask.append((str(a), str(b)) not in hard_pairs)
        corr_only = corroborated_edges.loc[mask].copy()

    sem_corr = corroborated_edges[
        corroborated_edges.get("semantic_support", False).astype(bool)
    ].copy() if not corroborated_edges.empty else corroborated_edges

    # edges from largest union components: pick 25 edges with largest component size
    edges_with_comp = seed_edges_all.copy()
    edges_with_comp["component_id_after_union"] = edges_with_comp["email_i"].astype(str).map(lambda x: comp_map.get(str(x), (None, None))[0])
    edges_with_comp["component_size_after_union"] = edges_with_comp["email_i"].astype(str).map(lambda x: comp_map.get(str(x), (None, None))[1])
    edges_largest = edges_with_comp.sort_values(
        ["component_size_after_union"], ascending=[False]
    ).dropna(subset=["component_size_after_union"]).head(25)

    def _sample(df: pd.DataFrame, n: int) -> pd.DataFrame:
        if df.empty:
            return df
        if len(df) <= n:
            return df.copy()
        idx = rng.choice(np.arange(len(df)), size=int(n), replace=False)
        return df.iloc[np.sort(idx)].copy()

    s_hard = _sample(hard_edges, 25)
    s_corr_only = _sample(corr_only, 25)
    s_largest = edges_largest.copy()
    s_sem = _sample(sem_corr, 25)

    def _decorate(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        d = df.copy()
        d["email_i"] = d["email_i"].astype(str)
        d["email_j"] = d["email_j"].astype(str)
        d["semantic_involved"] = d.get("semantic_support", False).astype(bool) if "semantic_support" in d.columns else False
        d["component_id_after_union"] = d["email_i"].map(lambda x: comp_map.get(str(x), (None, None))[0])
        d["component_size_after_union"] = d["email_i"].map(lambda x: comp_map.get(str(x), (None, None))[1])
        d["gt_campaign_i_if_available"] = d["email_i"].map(lambda x: gt.get(str(x)))
        d["gt_campaign_j_if_available"] = d["email_j"].map(lambda x: gt.get(str(x)))
        d["gt_same_campaign_if_available"] = d.apply(
            lambda r: (r["gt_campaign_i_if_available"] == r["gt_campaign_j_if_available"])
            if (r["gt_campaign_i_if_available"] is not None and r["gt_campaign_j_if_available"] is not None)
            else None,
            axis=1,
        )
        # Keep required cols
        keep = [
            "email_i",
            "email_j",
            "seed_tier",
            "evidence_type",
            "semantic_involved",
            "component_id_after_union",
            "component_size_after_union",
            "gt_same_campaign_if_available",
            "gt_campaign_i_if_available",
            "gt_campaign_j_if_available",
        ]
        # evidence_types: include evidence_fields_json if present for corroborated
        if "evidence_fields_json" in d.columns:
            d["evidence_types"] = d["evidence_fields_json"]
            if "evidence_types" not in keep:
                keep.insert(4, "evidence_types")
        return d[[c for c in keep if c in d.columns]]

    out = pd.concat(
        [
            _decorate(s_hard),
            _decorate(s_corr_only),
            _decorate(s_largest),
            _decorate(s_sem),
        ],
        axis=0,
        ignore_index=True,
    ).drop_duplicates(subset=["email_i", "email_j", "seed_tier", "evidence_type"], keep="first")

    p = out_dir / "seed_manual_review_sample.csv"
    out.to_csv(p, index=False)
    return str(p)


def _channel_ablation(seed_edges_df: pd.DataFrame) -> list[dict[str, Any]]:
    if seed_edges_df.empty:
        return []
    rows: list[dict[str, Any]] = []
    for ch, d in seed_edges_df.groupby("evidence_type", dropna=False):
        rows.append(
            {
                "evidence_type": str(ch),
                "n_edges": int(len(d)),
                "n_emails_touched": int(
                    len(set(d["email_i"].astype(str)).union(set(d["email_j"].astype(str))))
                ),
            }
        )
    rows = sorted(rows, key=lambda x: x["n_edges"], reverse=True)
    return rows


def _load_gt_maps(gt_paths: list[Path]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for p in gt_paths:
        lm, _eid_row, _camp = rn.load_ground_truth_structures(p)
        out[str(p)] = {str(k): v for k, v in lm.items()}
    return out


def _resolve_gt_paths(project_root: Path, gt_cfg: dict[str, Any]) -> list[Path]:
    raw = gt_cfg.get("paths") or []
    if not raw:
        return []
    if not isinstance(raw, list):
        raise ValueError("ground_truth.paths must be a list when provided")
    out: list[Path] = []
    for x in raw:
        p = Path(str(x)).expanduser()
        if not p.is_absolute():
            p = project_root / p
        p = p.resolve()
        if not p.is_file():
            raise FileNotFoundError(f"Ground truth file not found: {p}")
        out.append(p)
    return out


GeneratorFn = Callable[[pd.DataFrame, pd.DataFrame, dict[str, Any]], pd.DataFrame]


def _generator_registry() -> dict[str, GeneratorFn]:
    return {
        "hard_v1": lambda nodes_df, edges_df, cfg: generate_hard_seed_edges_v1(
            nodes_df=nodes_df,
            edges_df=edges_df,
            generator_cfg=cfg,
        ),
        "corroborated_v1": lambda nodes_df, edges_df, cfg: generate_corroborated_seed_edges_v1(
            nodes_df=nodes_df,
            edges_df=edges_df,
            generator_cfg=cfg,
        ),
    }


def run_anchor_seed_generation(config: dict[str, Any]) -> dict[str, Any]:
    run_cfg = config.get("run") or {}
    input_cfg = config.get("input") or {}
    seed_cfg = config.get("seeds") or {}
    component_cfg = config.get("seed_components") or {}
    out_cfg = config.get("output") or {}
    gt_cfg = config.get("ground_truth") or {}

    project_root = gh.find_project_root()
    graph_id = resolve_graph_id(run_cfg)
    anchor_output_root = Path(
        run_cfg.get("anchor_output_root")
        or (project_root / "analysis" / "output" / "anchor_graph")
    ).expanduser().resolve()
    anchor_run_dir = anchor_output_root / graph_id
    if not anchor_run_dir.is_dir():
        raise FileNotFoundError(f"Anchor graph run directory not found: {anchor_run_dir}")

    generators = seed_cfg.get("generators") or []
    if not isinstance(generators, list) or not generators:
        raise ValueError("seeds.generators must be a non-empty list")

    pbar_total = 6 + int(len(generators))
    pbar = tqdm(total=pbar_total, desc=f"Anchor seed generation [{graph_id}]") if tqdm is not None else None
    try:
        nodes_df, edges_df, _cand, _summary, _g = load_anchor_graph_artifacts(
            anchor_run_dir, load_graph_pickle=False
        )
        nodes_df["external_id"] = nodes_df["external_id"].astype(str)
        edges_df["email_a"] = edges_df["email_a"].astype(str)
        edges_df["email_b"] = edges_df["email_b"].astype(str)
        if pbar is not None:
            pbar.update(1)

        min_edge_weight = input_cfg.get("min_edge_weight")
        if min_edge_weight is not None and "edge_weight" in edges_df.columns:
            edges_df = edges_df[
                pd.to_numeric(edges_df["edge_weight"], errors="coerce") >= float(min_edge_weight)
            ].copy()
        if pbar is not None:
            pbar.update(1)

        registry = _generator_registry()
        seed_frames: list[pd.DataFrame] = []
        for gcfg in generators:
            if not isinstance(gcfg, dict):
                if pbar is not None:
                    pbar.update(1)
                continue
            name = str(gcfg.get("name") or "").strip().lower()
            if not name:
                if pbar is not None:
                    pbar.update(1)
                continue
            if name not in registry:
                raise ValueError(f"Unknown seed generator {name!r}. Available: {sorted(registry)}")
            if bool(gcfg.get("enabled", True)):
                sdf = registry[name](nodes_df, edges_df, gcfg)
                if not sdf.empty:
                    seed_frames.append(sdf)
            if pbar is not None:
                pbar.update(1)

        seed_edges_all = (
            pd.concat(seed_frames, axis=0, ignore_index=True)
            if seed_frames
            else pd.DataFrame(
                columns=[
                    "email_i",
                    "email_j",
                    "evidence_type",
                    "evidence_value",
                    "evidence_rarity",
                    "artifact_df",
                    "seed_tier",
                    "seed_generator",
                    "rule_id",
                    "n_support_channels",
                    "semantic_support",
                    "evidence_fields_json",
                ]
            )
        )
        if not seed_edges_all.empty:
            seed_edges_all = seed_edges_all.drop_duplicates(
                subset=["email_i", "email_j", "evidence_type", "evidence_value", "seed_generator", "rule_id"]
            ).reset_index(drop=True)
        hard_edges = seed_edges_all[seed_edges_all.get("seed_tier", "").astype(str) == "hard"].copy()
        corroborated_edges = seed_edges_all[
            seed_edges_all.get("seed_tier", "").astype(str) == "corroborated"
        ].copy()
        if pbar is not None:
            pbar.update(1)

        out_root = Path(
            out_cfg.get("output_root")
            or (project_root / "analysis" / "output" / "anchor_seeds")
        ).expanduser().resolve()
        stage_name = str(out_cfg.get("stage_name") or "seed_generation")
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_dir = out_root / graph_id / f"{stage_name}_{stamp}"
        out_dir.mkdir(parents=True, exist_ok=True)

        p_seed_all = out_dir / "seed_edges_all.csv"
        p_seed_hard = out_dir / "seed_edges_hard.csv"
        p_seed_corr = out_dir / "seed_edges_corroborated.csv"
        seed_edges_all.to_csv(p_seed_all, index=False)
        hard_edges.to_csv(p_seed_hard, index=False)
        corroborated_edges.to_csv(p_seed_corr, index=False)

        include_all_nodes = bool(component_cfg.get("include_all_nodes", True))
        if include_all_nodes:
            all_node_ids = nodes_df["external_id"].astype(str).tolist()
        else:
            all_node_ids = sorted(
                set(seed_edges_all["email_i"].astype(str)).union(set(seed_edges_all["email_j"].astype(str)))
            ) if not seed_edges_all.empty else []
        union_members_df, union_components_df, union_component_summary = _build_union_components(
            all_node_ids=all_node_ids,
            seed_edges_all=seed_edges_all,
        )
        p_union_members = out_dir / "seed_union_component_members.csv"
        p_union_components = out_dir / "seed_union_components.csv"
        union_members_df.to_csv(p_union_members, index=False)
        union_components_df.to_csv(p_union_components, index=False)

        gt_paths = _resolve_gt_paths(project_root, gt_cfg)
        gt_maps = _load_gt_maps(gt_paths)
        all_emails = nodes_df["external_id"].astype(str).tolist()
        coverage = _coverage_diagnostics(
            all_emails=all_emails,
            hard_edges=hard_edges,
            corroborated_edges=corroborated_edges,
            union_edges=seed_edges_all,
        )
        touched_union = set(coverage["_touched_union"])
        concentration = _component_concentration_diagnostics(
            union_components_df=union_components_df,
            seeded_emails=touched_union,
        )
        redundancy = _corroborated_redundancy_diagnostics(
            hard_edges=hard_edges,
            corroborated_edges=corroborated_edges,
            all_node_ids=all_node_ids,
        )

        gt_metrics: list[dict[str, Any]] = []
        for p, label_map in gt_maps.items():
            hard_pair_prec = _labeled_pair_precision(seed_edges_df=hard_edges, gt_label_map=label_map)
            hard_comp_h = _component_homogeneity_on_gt(seed_edges_df=hard_edges, gt_label_map=label_map)
            all_pair_prec = _labeled_pair_precision(seed_edges_df=seed_edges_all, gt_label_map=label_map)
            all_comp_h = _component_homogeneity_on_gt(seed_edges_df=seed_edges_all, gt_label_map=label_map)
            gt_ids = set(str(k) for k in label_map.keys())
            gt_touched_union = gt_ids & touched_union
            campaign_dist = _campaign_touch_distribution(
                union_members_df=union_members_df,
                union_edges_df=seed_edges_all,
                gt_label_map=label_map,
                touched_union=touched_union,
            )
            purity_spread = _union_component_purity_spread(
                union_members_df=union_members_df,
                gt_label_map=label_map,
                mixed_threshold=float(component_cfg.get("mixed_component_threshold", 0.90)),
            )
            gt_metrics.append(
                {
                    "gt_path": p,
                    "gt_labeled_emails_total": int(len(gt_ids)),
                    "gt_labeled_emails_touched_by_union": int(len(gt_touched_union)),
                    "pct_gt_labeled_emails_touched_by_union": float(len(gt_touched_union) / max(1, len(gt_ids))),
                    "hard": {**hard_pair_prec, **hard_comp_h},
                    "union_edges": {**all_pair_prec, **all_comp_h},
                    "union_components": {
                        **_b_cubed_precision(members_df=union_members_df, gt_label_map=label_map),
                        **_labeled_campaign_touch_rate(
                            touched_email_ids=touched_union,
                            gt_label_map=label_map,
                        ),
                        "purity_spread": purity_spread,
                        "campaign_touch_distribution": campaign_dist,
                    },
                }
            )
        if pbar is not None:
            pbar.update(1)

        manual_seed = int(component_cfg.get("manual_review_random_seed", 1337))
        manual_gt_map = next(iter(gt_maps.values()), None)
        manual_review_csv = _manual_review_sample(
            out_dir=out_dir,
            seed_edges_all=seed_edges_all,
            hard_edges=hard_edges,
            corroborated_edges=corroborated_edges,
            union_members_df=union_members_df,
            gt_label_map=manual_gt_map,
            random_seed=manual_seed,
        )

        coverage_public = {k: v for k, v in coverage.items() if not str(k).startswith("_")}
        if gt_maps:
            gt_union_ids: set[str] = set()
            for lm in gt_maps.values():
                gt_union_ids |= {str(k) for k in lm.keys()}
            gt_touched = gt_union_ids & touched_union
            coverage_public.update(
                {
                    "gt_labeled_emails_total": int(len(gt_union_ids)),
                    "gt_labeled_emails_touched_by_union": int(len(gt_touched)),
                    "pct_gt_labeled_emails_touched_by_union": float(len(gt_touched) / max(1, len(gt_union_ids))),
                }
            )

        summary = {
            "metadata": {
                "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "graph_id": graph_id,
                "anchor_run_dir": str(anchor_run_dir),
                "seed_output_dir": str(out_dir),
                "include_all_nodes_in_components": bool(include_all_nodes),
            },
            "hard": {
                "metrics": _seed_graph_metrics(hard_edges),
                "per_channel_ablation": _channel_ablation(hard_edges),
            },
            "corroborated": {
                "metrics": _seed_graph_metrics(corroborated_edges),
                "per_channel_ablation": _channel_ablation(corroborated_edges),
                "corroborated_specific_metrics": _corroborated_specific_metrics(corroborated_edges, hard_edges),
            },
            "union_edges": {
                "metrics": _seed_graph_metrics(seed_edges_all),
            },
            "union_components": {
                **union_component_summary,
                "concentration_diagnostics": concentration,
            },
            "gt_eval": gt_metrics,
            "diagnostics": {
                "seed_coverage": coverage_public,
                "corroborated_redundancy": redundancy,
                "manual_review_sample_csv": str(manual_review_csv),
            },
        }
        p_summary = out_dir / "anchor_seed_summary.json"
        p_summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        if pbar is not None:
            pbar.update(1)
        return {
            "output_dir": str(out_dir),
            "seed_edges_all_csv": str(p_seed_all),
            "seed_edges_hard_csv": str(p_seed_hard),
            "seed_edges_corroborated_csv": str(p_seed_corr),
            "seed_union_component_members_csv": str(p_union_members),
            "seed_union_components_csv": str(p_union_components),
            "summary_json": str(p_summary),
        }
    finally:
        if pbar is not None:
            pbar.close()

