from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


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
    if s.startswith("[") and s.endswith("]"):
        try:
            obj = json.loads(s)
            if isinstance(obj, list):
                return {str(x) for x in obj if x is not None and str(x).strip()}
        except Exception:
            pass
    return {x.strip() for x in s.split("|") if x.strip()}


def _artifact_idf(df_val: int, n_docs: int) -> float:
    return float(math.log((1.0 + n_docs) / (1.0 + max(1, int(df_val))))) if n_docs > 0 else float("nan")


def _pair(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a <= b else (b, a)


def _load_seed_members(seed_dir: Path) -> pd.DataFrame:
    p = seed_dir / "seed_union_component_members.csv"
    if not p.is_file():
        return pd.DataFrame(columns=["external_id", "component_id", "component_size", "is_singleton"])
    df = pd.read_csv(p)
    if df.empty:
        return pd.DataFrame(columns=["external_id", "component_id", "component_size", "is_singleton"])
    df["external_id"] = df["external_id"].astype(str)
    df["component_id"] = pd.to_numeric(df.get("component_id"), errors="coerce").fillna(-1).astype(int)
    df["component_size"] = pd.to_numeric(df.get("component_size"), errors="coerce").fillna(1).astype(int)
    if "is_singleton" not in df.columns:
        df["is_singleton"] = df["component_size"].eq(1)
    else:
        df["is_singleton"] = df["is_singleton"].astype(bool)
    return df


def generate_candidates_2hop_bounded_v1(
    *,
    nodes_df: pd.DataFrame,
    id_to_vec: dict[str, np.ndarray],
    ts_map: dict[str, float],
    seed_pairs: set[tuple[str, str]],
    seed_dir: Path,
    generator_cfg: dict[str, Any],
    out_dir: Path,
) -> dict[str, Any]:
    cfg = dict(generator_cfg or {})
    artifact_specs = cfg.get("artifact_specs") or []
    if not isinstance(artifact_specs, list) or not artifact_specs:
        raise ValueError("2hop_bounded_v1 requires non-empty config.artifact_specs")

    excluded_path_types = {str(x).strip() for x in (cfg.get("excluded_path_types") or []) if str(x).strip()}
    max_pairs_per_artifact_value = int(cfg.get("max_pairs_per_artifact_value", 200))
    max_total_pairs = int(cfg.get("max_total_pairs", 100000))
    drop_if_seed_pair = bool(cfg.get("drop_if_seed_pair", True))
    time_gating_enabled = bool(cfg.get("time_gating_enabled", False))
    max_time_gap_seconds = cfg.get("max_time_gap_seconds")
    if time_gating_enabled and max_time_gap_seconds is None:
        max_time_gap_seconds = 2592000.0
    semantic_contradiction_enabled = bool(cfg.get("semantic_contradiction_enabled", True))
    semantic_min_cos = float(cfg.get("semantic_min_cos", 0.3))
    seed_adjacent_only = bool(cfg.get("seed_adjacent_only", False))

    n_docs = int(len(nodes_df))
    node_sets: dict[str, dict[str, set[str]]] = {}
    for _, r in nodes_df.iterrows():
        eid = str(r["external_id"])
        node_sets[eid] = {}
        for spec in artifact_specs:
            col = str(spec.get("node_set_col") or "").strip()
            if not col:
                continue
            if col in nodes_df.columns:
                node_sets[eid][col] = _to_set_cell(r.get(col))

    members_df = _load_seed_members(seed_dir)
    comp_id_map = dict(zip(members_df["external_id"].astype(str).tolist(), members_df["component_id"].tolist(), strict=False))
    comp_size_map = dict(zip(members_df["external_id"].astype(str).tolist(), members_df["component_size"].tolist(), strict=False))
    in_non_singleton = {eid for eid, size in comp_size_map.items() if int(size) >= 2}

    value_emails: dict[tuple[str, str, str], set[str]] = defaultdict(set)                                             
    value_df: dict[tuple[str, str, str], int] = {}
    value_idf: dict[tuple[str, str, str], float] = {}
    drop_counts = Counter()
    kept_by_path_type = Counter()
    rows_before_filters = 0

    for spec in artifact_specs:
        artifact_type = str(spec.get("artifact_type") or "").strip()
        node_set_col = str(spec.get("node_set_col") or "").strip()
        path_type = str(spec.get("path_type") or f"email_{artifact_type}_email").strip()
        max_degree = spec.get("max_degree")
        max_degree = None if max_degree is None else int(max_degree)
        min_idf = float(spec.get("min_idf", 0.0))
        if not artifact_type or not node_set_col or not path_type:
            continue
        if path_type in excluded_path_types:
            continue
        inv: dict[str, set[str]] = defaultdict(set)
        for eid, cols in node_sets.items():
            vals = cols.get(node_set_col, set())
            if not vals:
                continue
            for v in vals:
                inv[v].add(eid)
        for val, emails in inv.items():
            deg = int(len(emails))
            if deg < 2:
                continue
            rows_before_filters += int((deg * (deg - 1)) // 2)
            if max_degree is not None and deg > max_degree:
                drop_counts["too_common"] += int((deg * (deg - 1)) // 2)
                continue
            idf = _artifact_idf(df_val=deg, n_docs=n_docs)
            if not np.isfinite(idf) or idf < min_idf:
                drop_counts["too_common"] += int((deg * (deg - 1)) // 2)
                continue
            key_prefix = (artifact_type, path_type, str(val))
            value_emails[key_prefix] = set(emails)
            value_df[key_prefix] = deg
            value_idf[key_prefix] = float(idf)
    scanned_by_type = Counter()
    passing_by_type = Counter()
    for (artifact_type, _path_type, _val), emails in value_emails.items():
        scanned_by_type[artifact_type] += 1
        if len(emails) >= 2:
            passing_by_type[artifact_type] += 1

    candidate_rows: list[dict[str, Any]] = []
    seen_rows: set[tuple[str, str, str, str, str]] = set()

    for (artifact_type, path_type, val), emails in value_emails.items():
        email_list = sorted(str(x) for x in emails)
        deg = int(value_df[(artifact_type, path_type, val)])
        idf = float(value_idf[(artifact_type, path_type, val)])
        emitted_for_value = 0
        for i in range(len(email_list)):
            a = email_list[i]
            for j in range(i + 1, len(email_list)):
                b = email_list[j]
                email_i, email_j = _pair(a, b)
                if email_i == email_j:
                    continue
                if drop_if_seed_pair and (email_i, email_j) in seed_pairs:
                    drop_counts["already_seed"] += 1
                    continue
                if path_type in excluded_path_types:
                    drop_counts["blocked_type"] += 1
                    continue

                t_i = float(ts_map.get(email_i, float("nan")))
                t_j = float(ts_map.get(email_j, float("nan")))
                time_gap = float(abs(t_i - t_j)) if np.isfinite(t_i) and np.isfinite(t_j) else float("nan")
                if time_gating_enabled and np.isfinite(time_gap) and time_gap > float(max_time_gap_seconds):
                    drop_counts["time_gap"] += 1
                    continue

                if semantic_contradiction_enabled:
                    v_i = id_to_vec.get(email_i)
                    v_j = id_to_vec.get(email_j)
                    if v_i is not None and v_j is not None:
                        cos = float(np.dot(v_i, v_j) / (np.linalg.norm(v_i) * np.linalg.norm(v_j) + 1e-12))
                        if np.isfinite(cos) and cos < semantic_min_cos:
                            drop_counts["semantic_contradiction"] += 1
                            continue

                comp_i = int(comp_id_map.get(email_i, -1))
                comp_j = int(comp_id_map.get(email_j, -1))
                in_seed_i = email_i in in_non_singleton
                in_seed_j = email_j in in_non_singleton
                seed_adjacent_flag = bool(in_seed_i or in_seed_j)
                both_in_seed_components = bool(in_seed_i and in_seed_j)
                if seed_adjacent_only and not seed_adjacent_flag:
                    drop_counts["not_seed_adjacent"] += 1
                    continue

                reason_code = "C1"
                if seed_adjacent_flag:
                    reason_code = "C1|C3"
                row_key = (email_i, email_j, path_type, artifact_type, str(val))
                if row_key in seen_rows:
                    continue
                seen_rows.add(row_key)
                candidate_rows.append(
                    {
                        "email_i": email_i,
                        "email_j": email_j,
                        "source": "2hop",
                        "path_type": path_type,
                        "intermediary_artifact_type": artifact_type,
                        "intermediary_artifact_value": str(val),
                        "intermediary_degree": int(deg),
                        "rarity_score": float(idf),
                        "time_gap_seconds": time_gap,
                        "seed_adjacent_flag": seed_adjacent_flag,
                        "email_i_component_id": comp_i,
                        "email_j_component_id": comp_j,
                        "both_in_seed_components": both_in_seed_components,
                        "reason_code": reason_code,
                    }
                )
                kept_by_path_type[path_type] += 1
                emitted_for_value += 1
                if emitted_for_value >= max(1, max_pairs_per_artifact_value):
                    break
                if len(candidate_rows) >= max(1, max_total_pairs):
                    break
            if emitted_for_value >= max(1, max_pairs_per_artifact_value) or len(candidate_rows) >= max(1, max_total_pairs):
                break
        if len(candidate_rows) >= max(1, max_total_pairs):
            break

    out_df = pd.DataFrame(candidate_rows)
    required_cols = [
        "email_i",
        "email_j",
        "source",
        "path_type",
        "intermediary_artifact_type",
        "intermediary_artifact_value",
        "intermediary_degree",
        "rarity_score",
        "time_gap_seconds",
        "seed_adjacent_flag",
        "email_i_component_id",
        "email_j_component_id",
        "both_in_seed_components",
        "reason_code",
    ]
    if out_df.empty:
        out_df = pd.DataFrame(columns=required_cols)
    else:
        out_df = out_df.drop_duplicates(
            subset=["email_i", "email_j", "path_type", "intermediary_artifact_type", "intermediary_artifact_value"]
        ).reset_index(drop=True)
        out_df = out_df.sort_values(
            ["path_type", "rarity_score", "intermediary_degree", "email_i", "email_j"],
            ascending=[True, False, True, True, True],
        ).reset_index(drop=True)

    p_csv = out_dir / "candidates_2hop.csv"
    out_df.to_csv(p_csv, index=False)
    pairs_set = set()
    if not out_df.empty:
        pairs_set = set(zip(out_df["email_i"].astype(str).tolist(), out_df["email_j"].astype(str).tolist(), strict=False))

    diagnostics = {
        "rows_before_filters": int(rows_before_filters),
        "rows_after_filters": int(len(out_df)),
        "kept_by_path_type": dict(sorted(kept_by_path_type.items())),
        "dropped_by_reason": dict(sorted(drop_counts.items())),
    }
    return {
        "candidates_2hop_csv": str(p_csv),
        "n_rows": int(len(out_df)),
        "n_pairs": int(len(pairs_set)),
        "pairs_set": pairs_set,
        "diagnostics": diagnostics,
    }

