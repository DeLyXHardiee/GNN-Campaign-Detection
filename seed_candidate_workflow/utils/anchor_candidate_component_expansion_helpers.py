from __future__ import annotations

import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

def _to_set_cell(x: Any) -> set[str]:
    if isinstance(x, set):
        return {str(v) for v in x if v is not None and str(v).strip()}
    if isinstance(x, (list, tuple)):
        return {str(v) for v in x if v is not None and str(v).strip()}
    if x is None:
        return set()
    s = str(x).strip()
    if not s:
        return set()
    if s.startswith("[") and s.endswith("]"):
        try:
            parsed = pd.read_json(s, typ="series")
            return {str(v) for v in parsed.tolist() if v is not None and str(v).strip()}
        except Exception:
            return set()
    return {v.strip() for v in s.split("|") if v.strip()}


def _pair_key(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a <= b else (b, a)


def _safe_float(x: Any, default: float = float("nan")) -> float:
    v = pd.to_numeric(x, errors="coerce")
    return float(v) if pd.notna(v) else float(default)


def _l2(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n <= 0:
        return v
    return (v / n).astype(np.float32, copy=False)


def _idf(df_val: int, n_docs: int) -> float:
    return float(math.log((1.0 + n_docs) / (1.0 + max(1, int(df_val))))) if n_docs > 0 else float("nan")


def _extract_component_members(seed_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    members_path = seed_dir / "seed_union_component_members.csv"
    comps_path = seed_dir / "seed_union_components.csv"
    if not members_path.is_file():
        raise FileNotFoundError(f"Missing seed union members file: {members_path}")
    if not comps_path.is_file():
        raise FileNotFoundError(f"Missing seed union components file: {comps_path}")
    members_df = pd.read_csv(members_path)
    comps_df = pd.read_csv(comps_path)
    if not {"external_id", "component_id", "component_size"}.issubset(members_df.columns):
        raise ValueError("seed_union_component_members.csv missing required columns")
    members_df["external_id"] = members_df["external_id"].astype(str)
    members_df["component_id"] = members_df["component_id"].astype(int)
    members_df["component_size"] = members_df["component_size"].astype(int)
    if "is_singleton" in members_df.columns:
        members_df["is_singleton"] = members_df["is_singleton"].astype(bool)
    else:
        members_df["is_singleton"] = members_df["component_size"].eq(1)
    if "component_id" in comps_df.columns:
        comps_df["component_id"] = comps_df["component_id"].astype(int)
    return members_df, comps_df


def _build_artifact_df(nodes_df: pd.DataFrame, artifact_cols: list[str]) -> dict[str, Counter[str]]:
    df_map: dict[str, Counter[str]] = {}
    for col in artifact_cols:
        c = Counter()
        if col not in nodes_df.columns:
            df_map[col] = c
            continue
        for vals in nodes_df[col].tolist():
            for v in _to_set_cell(vals):
                c[v] += 1
        df_map[col] = c
    return df_map


def _component_signatures(
    *,
    nodes_df: pd.DataFrame,
    id_to_vec: dict[str, np.ndarray],
    members_df: pd.DataFrame,
    artifact_cols: list[str],
    min_artifact_idf: float,
    max_artifact_df: int | None,
    max_artifacts_per_type: int,
) -> tuple[dict[int, dict[str, Any]], dict[str, dict[str, set[str]]], dict[str, float]]:
    n_docs = int(len(nodes_df))
    node_sets: dict[str, dict[str, set[str]]] = {}
    ts_map: dict[str, float] = {}
    ext_ids = nodes_df["external_id"].astype(str).tolist()
    for _, row in nodes_df.iterrows():
        eid = str(row["external_id"])
        ts_map[eid] = _safe_float(row.get("ts"), float("nan"))
        node_sets[eid] = {c: _to_set_cell(row.get(c)) for c in artifact_cols if c in nodes_df.columns}

    df_map = _build_artifact_df(nodes_df, artifact_cols)
    comp_to_members = (
        members_df.groupby("component_id", dropna=False)["external_id"].apply(lambda s: sorted(set(s.astype(str)))).to_dict()
    )
    signatures: dict[int, dict[str, Any]] = {}
    for comp_id, member_ids in comp_to_members.items():
        vecs = [id_to_vec[eid] for eid in member_ids if eid in id_to_vec]
        centroid = None
        if vecs:
            centroid = _l2(np.mean(np.stack(vecs).astype(np.float32), axis=0))
        ts_vals = [ts_map.get(eid, float("nan")) for eid in member_ids]
        ts_clean = [t for t in ts_vals if np.isfinite(t)]
        artifact_summary: dict[str, list[dict[str, Any]]] = {}
        dominant_type = "mixed"
        all_strengths: list[tuple[str, float]] = []
        for col in artifact_cols:
            value_count = Counter()
            for eid in member_ids:
                value_count.update(node_sets.get(eid, {}).get(col, set()))
            scored: list[tuple[float, str, int]] = []
            for val, _ct in value_count.items():
                df_val = int(df_map.get(col, Counter()).get(val, 0))
                idf = _idf(df_val, n_docs)
                if not np.isfinite(idf) or idf < float(min_artifact_idf):
                    continue
                if max_artifact_df is not None and df_val > int(max_artifact_df):
                    continue
                scored.append((idf, val, df_val))
            scored.sort(key=lambda x: x[0], reverse=True)
            top_rows = [
                {"artifact_value": val, "artifact_idf": float(idf), "artifact_df": int(df_val)}
                for idf, val, df_val in scored[: max(1, int(max_artifacts_per_type))]
            ]
            artifact_summary[col] = top_rows
            total_strength = float(sum(r["artifact_idf"] for r in top_rows))
            all_strengths.append((col, total_strength))
        if all_strengths:
            all_strengths.sort(key=lambda x: x[1], reverse=True)
            top = all_strengths[0][0]
            if "attachment" in top:
                dominant_type = "attachment_heavy"
            elif "url" in top or "domain" in top or "stem" in top:
                dominant_type = "url_heavy"
            elif "html" in top:
                dominant_type = "html_heavy"
            elif top in {"sender_set", "sender_email_domain_set"}:
                dominant_type = "semantic_heavy"
        signatures[int(comp_id)] = {
            "component_id": int(comp_id),
            "size": int(len(member_ids)),
            "members": member_ids,
            "centroid": centroid,
            "ts_min": float(min(ts_clean)) if ts_clean else float("nan"),
            "ts_max": float(max(ts_clean)) if ts_clean else float("nan"),
            "ts_median": float(np.median(ts_clean)) if ts_clean else float("nan"),
            "ts_span_seconds": float(max(ts_clean) - min(ts_clean)) if ts_clean else float("nan"),
            "artifact_summary": artifact_summary,
            "evidence_profile": dominant_type,
        }
    return signatures, node_sets, ts_map


def _overlap_artifacts(sig_a: dict[str, Any], sig_b: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for col, vals_a in (sig_a.get("artifact_summary") or {}).items():
        vals_b = (sig_b.get("artifact_summary") or {}).get(col) or []
        map_a = {str(r["artifact_value"]): r for r in vals_a}
        map_b = {str(r["artifact_value"]): r for r in vals_b}
        shared = sorted(set(map_a) & set(map_b))
        for v in shared:
            rows.append(
                {
                    "artifact_col": col,
                    "artifact_value": v,
                    "artifact_idf": float(max(map_a[v]["artifact_idf"], map_b[v]["artifact_idf"])),
                    "artifact_df": int(min(map_a[v]["artifact_df"], map_b[v]["artifact_df"])),
                }
            )
    rows.sort(key=lambda r: r["artifact_idf"], reverse=True)
    return rows


def _boundary_semantic_pairs(
    members_a: list[str],
    members_b: list[str],
    id_to_vec: dict[str, np.ndarray],
    top_k: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for a in members_a:
        va = id_to_vec.get(a)
        if va is None:
            continue
        for b in members_b:
            vb = id_to_vec.get(b)
            if vb is None:
                continue
            cos = float(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-12))
            rows.append({"email_i": min(a, b), "email_j": max(a, b), "cosine": cos, "a": a, "b": b})
    rows.sort(key=lambda r: r["cosine"], reverse=True)
    uniq: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for r in rows:
        k = (r["email_i"], r["email_j"])
        if k in seen:
            continue
        seen.add(k)
        uniq.append(r)
        if len(uniq) >= max(1, int(top_k)):
            break
    return uniq


def _generate_component_links(
    *,
    signatures: dict[int, dict[str, Any]],
    id_to_vec: dict[str, np.ndarray],
    semantic_component_top_k_neighbors: int,
    semantic_centroid_min_cos: float,
    singleton_semantic_min_cos: float,
    max_time_gap_seconds: float,
    max_component_size_for_expansion: int,
    max_component_links_total: int,
    semantic_email_cross_top_k: int,
    min_cross_retrieval_hits: int,
) -> pd.DataFrame:
    non_singletons = [s for s in signatures.values() if int(s["size"]) >= 2]
    singletons = [s for s in signatures.values() if int(s["size"]) == 1]
    rows: list[dict[str, Any]] = []

    def centroid_cos(sa: dict[str, Any], sb: dict[str, Any]) -> float:
        ca, cb = sa.get("centroid"), sb.get("centroid")
        if ca is None or cb is None:
            return float("nan")
        return float(np.dot(ca, cb))

    # Component-to-component.
    for sa in non_singletons:
        if len(rows) >= int(max_component_links_total):
            break
        ca = sa.get("centroid")
        if ca is None:
            continue
        cands: list[tuple[float, dict[str, Any]]] = []
        for sb in non_singletons:
            if int(sb["component_id"]) <= int(sa["component_id"]):
                continue
            cb = sb.get("centroid")
            if cb is None:
                continue
            cands.append((float(np.dot(ca, cb)), sb))
        cands.sort(key=lambda x: x[0], reverse=True)
        for cos, sb in cands[: max(1, int(semantic_component_top_k_neighbors))]:
            reasons: list[str] = []
            if int(sa["size"]) > int(max_component_size_for_expansion) or int(sb["size"]) > int(max_component_size_for_expansion):
                continue
            art_overlap = _overlap_artifacts(sa, sb)
            support_non_sem = bool(art_overlap)
            med_gap = abs(_safe_float(sa.get("ts_median")) - _safe_float(sb.get("ts_median")))
            if np.isfinite(med_gap) and med_gap > float(max_time_gap_seconds):
                continue
            boundary = _boundary_semantic_pairs(sa["members"], sb["members"], id_to_vec, semantic_email_cross_top_k)
            if cos >= float(semantic_centroid_min_cos) and (support_non_sem or len(boundary) >= int(min_cross_retrieval_hits)):
                reasons.append("A1")
            if support_non_sem:
                reasons.append("A2")
            if len(boundary) >= int(min_cross_retrieval_hits):
                reasons.append("A3")
            boundary_a = len({r["a"] for r in boundary})
            boundary_b = len({r["b"] for r in boundary})
            if boundary and boundary_a <= 5 and boundary_b <= 5:
                reasons.append("A4")
            if not reasons:
                continue
            rows.append(
                {
                    "component_a": int(sa["component_id"]),
                    "component_b": int(sb["component_id"]),
                    "proposal_type": "component_to_component",
                    "reason_codes": "|".join(sorted(set(reasons))),
                    "centroid_cos": float(cos),
                    "time_gap_seconds": float(med_gap) if np.isfinite(med_gap) else float("nan"),
                    "size_a": int(sa["size"]),
                    "size_b": int(sb["size"]),
                    "n_overlapped_rare_artifacts": int(len(art_overlap)),
                    "top_artifact_overlap": str(art_overlap[0]["artifact_value"]) if art_overlap else "",
                    "n_boundary_emails_a": int(boundary_a),
                    "n_boundary_emails_b": int(boundary_b),
                    "evidence_profile_a": str(sa.get("evidence_profile", "mixed")),
                    "evidence_profile_b": str(sb.get("evidence_profile", "mixed")),
                }
            )
            if len(rows) >= int(max_component_links_total):
                break

    # Singleton-to-component.
    for s1 in singletons:
        if len(rows) >= int(max_component_links_total):
            break
        s_vec = s1.get("centroid")
        if s_vec is None:
            continue
        cands: list[tuple[float, dict[str, Any]]] = []
        for sb in non_singletons:
            cb = sb.get("centroid")
            if cb is None:
                continue
            cands.append((float(np.dot(s_vec, cb)), sb))
        cands.sort(key=lambda x: x[0], reverse=True)
        for cos, sb in cands[: max(1, int(semantic_component_top_k_neighbors))]:
            if int(sb["size"]) > int(max_component_size_for_expansion):
                continue
            reasons: list[str] = []
            art_overlap = _overlap_artifacts(s1, sb)
            if cos >= float(singleton_semantic_min_cos):
                reasons.append("B1")
            if art_overlap:
                reasons.append("B2")
            boundary = _boundary_semantic_pairs(s1["members"], sb["members"], id_to_vec, semantic_email_cross_top_k)
            if len(boundary) >= int(min_cross_retrieval_hits):
                reasons.append("B3")
            med_gap = abs(_safe_float(s1.get("ts_median")) - _safe_float(sb.get("ts_median")))
            if np.isfinite(med_gap) and med_gap <= float(max_time_gap_seconds):
                reasons.append("B4")
            if not reasons:
                continue
            rows.append(
                {
                    "component_a": int(s1["component_id"]),
                    "component_b": int(sb["component_id"]),
                    "proposal_type": "singleton_to_component",
                    "reason_codes": "|".join(sorted(set(reasons))),
                    "centroid_cos": float(cos),
                    "time_gap_seconds": float(med_gap) if np.isfinite(med_gap) else float("nan"),
                    "size_a": int(s1["size"]),
                    "size_b": int(sb["size"]),
                    "n_overlapped_rare_artifacts": int(len(art_overlap)),
                    "top_artifact_overlap": str(art_overlap[0]["artifact_value"]) if art_overlap else "",
                    "n_boundary_emails_a": int(1),
                    "n_boundary_emails_b": int(len({r["b"] for r in boundary})),
                    "evidence_profile_a": str(s1.get("evidence_profile", "mixed")),
                    "evidence_profile_b": str(sb.get("evidence_profile", "mixed")),
                }
            )
            if len(rows) >= int(max_component_links_total):
                break

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.drop_duplicates(subset=["component_a", "component_b", "proposal_type"]).sort_values(
        ["proposal_type", "centroid_cos", "n_overlapped_rare_artifacts"],
        ascending=[True, False, False],
    ).reset_index(drop=True)
    return out


def _artifact_supported_pairs(
    members_a: list[str],
    members_b: list[str],
    node_sets: dict[str, dict[str, set[str]]],
    artifact_cols: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for col in artifact_cols:
        for a in members_a:
            aset = node_sets.get(a, {}).get(col, set())
            if not aset:
                continue
            for b in members_b:
                bset = node_sets.get(b, {}).get(col, set())
                shared = aset & bset
                if not shared:
                    continue
                val = sorted(shared)[0]
                rows.append(
                    {
                        "email_i": min(a, b),
                        "email_j": max(a, b),
                        "support_type": "rare_artifact",
                        "artifact_col": col,
                        "artifact_value": val,
                    }
                )
    uniq = {(r["email_i"], r["email_j"], r["artifact_col"], r["artifact_value"]): r for r in rows}
    return list(uniq.values())


def _unfold_component_links(
    *,
    links_df: pd.DataFrame,
    signatures: dict[int, dict[str, Any]],
    id_to_vec: dict[str, np.ndarray],
    node_sets: dict[str, dict[str, set[str]]],
    artifact_cols: list[str],
    ts_map: dict[str, float],
    max_pairs_per_component_link: int,
    max_singleton_pairs_per_proposal: int,
    semantic_email_cross_top_k: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if links_df.empty:
        return pd.DataFrame(rows)
    for _, link in links_df.iterrows():
        ca = int(link["component_a"])
        cb = int(link["component_b"])
        sa = signatures.get(ca)
        sb = signatures.get(cb)
        if sa is None or sb is None:
            continue
        proposal_type = str(link["proposal_type"])
        reason_codes = str(link.get("reason_codes", ""))
        members_a = list(sa["members"])
        members_b = list(sb["members"])

        sem_pairs = _boundary_semantic_pairs(members_a, members_b, id_to_vec, semantic_email_cross_top_k)
        art_pairs = _artifact_supported_pairs(members_a, members_b, node_sets, artifact_cols)
        merged: list[dict[str, Any]] = []
        for r in sem_pairs:
            merged.append(
                {
                    "email_i": r["email_i"],
                    "email_j": r["email_j"],
                    "support_type": "semantic_boundary",
                    "cosine": float(r["cosine"]),
                    "artifact_col": "",
                    "artifact_value": "",
                }
            )
        for r in art_pairs:
            merged.append(
                {
                    "email_i": r["email_i"],
                    "email_j": r["email_j"],
                    "support_type": "rare_artifact",
                    "cosine": float("nan"),
                    "artifact_col": str(r["artifact_col"]),
                    "artifact_value": str(r["artifact_value"]),
                }
            )
        merged.sort(key=lambda r: (_safe_float(r.get("cosine"), -1.0), r["email_i"], r["email_j"]), reverse=True)

        cap = int(max_singleton_pairs_per_proposal) if proposal_type == "singleton_to_component" else int(max_pairs_per_component_link)
        emitted = 0
        seen_pairs: set[tuple[str, str, str, str]] = set()
        for r in merged:
            k = (r["email_i"], r["email_j"], r["support_type"], r["artifact_value"])
            if k in seen_pairs:
                continue
            seen_pairs.add(k)
            time_gap = abs(ts_map.get(r["email_i"], float("nan")) - ts_map.get(r["email_j"], float("nan")))
            rows.append(
                {
                    "email_i": r["email_i"],
                    "email_j": r["email_j"],
                    "source": "component_expansion",
                    "proposal_type": proposal_type,
                    "component_a": ca,
                    "component_b": cb,
                    "reason_code": reason_codes,
                    "support_type": r["support_type"],
                    "cosine": r.get("cosine", float("nan")),
                    "artifact_col": r.get("artifact_col", ""),
                    "artifact_value": r.get("artifact_value", ""),
                    "time_gap_seconds": float(time_gap) if np.isfinite(time_gap) else float("nan"),
                }
            )
            emitted += 1
            if emitted >= max(1, cap):
                break
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.drop_duplicates(
        subset=["email_i", "email_j", "proposal_type", "component_a", "component_b", "support_type", "artifact_col", "artifact_value"]
    ).reset_index(drop=True)
    return out


def generate_component_expansion_v1(
    *,
    nodes_df: pd.DataFrame,
    id_to_vec: dict[str, np.ndarray],
    seed_dir: Path,
    generator_cfg: dict[str, Any],
    out_dir: Path,
) -> dict[str, Any]:
    cfg = dict(generator_cfg or {})
    artifact_cols = [str(x).strip() for x in cfg.get("artifact_columns", []) if str(x).strip()]
    if not artifact_cols:
        artifact_cols = [
            "url_set",
            "stem_set",
            "domain_set",
            "attachment_set",
            "html_structure_fingerprint_set",
            "sender_set",
            "sender_email_domain_set",
            "received_host_set",
            "origin_ip_set",
            "return_path_domain_set",
        ]

    members_df, _comps_df = _extract_component_members(seed_dir)
    signatures, node_sets, ts_map = _component_signatures(
        nodes_df=nodes_df,
        id_to_vec=id_to_vec,
        members_df=members_df,
        artifact_cols=artifact_cols,
        min_artifact_idf=float(cfg.get("min_artifact_idf", 0.8)),
        max_artifact_df=cfg.get("max_artifact_df"),
        max_artifacts_per_type=int(cfg.get("max_artifacts_per_type", 25)),
    )
    links_df = _generate_component_links(
        signatures=signatures,
        id_to_vec=id_to_vec,
        semantic_component_top_k_neighbors=int(cfg.get("semantic_component_top_k_neighbors", 25)),
        semantic_centroid_min_cos=float(cfg.get("semantic_centroid_min_cos", 0.92)),
        singleton_semantic_min_cos=float(cfg.get("singleton_semantic_min_cos", 0.94)),
        max_time_gap_seconds=float(cfg.get("max_time_gap_seconds", 2592000)),
        max_component_size_for_expansion=int(cfg.get("max_component_size_for_expansion", 200)),
        max_component_links_total=int(cfg.get("max_component_links_total", 10000)),
        semantic_email_cross_top_k=int(cfg.get("semantic_email_cross_top_k", 50)),
        min_cross_retrieval_hits=int(cfg.get("min_cross_retrieval_hits", 2)),
    )
    pairs_df = _unfold_component_links(
        links_df=links_df,
        signatures=signatures,
        id_to_vec=id_to_vec,
        node_sets=node_sets,
        artifact_cols=artifact_cols,
        ts_map=ts_map,
        max_pairs_per_component_link=int(cfg.get("max_pairs_per_component_link", 200)),
        max_singleton_pairs_per_proposal=int(cfg.get("max_singleton_pairs_per_proposal", 25)),
        semantic_email_cross_top_k=int(cfg.get("semantic_email_cross_top_k", 50)),
    )

    links_path = out_dir / "component_candidate_links.csv"
    pairs_path = out_dir / "candidates_component_expanded.csv"
    links_df.to_csv(links_path, index=False)
    pairs_df.to_csv(pairs_path, index=False)
    pairs_set = set()
    if not pairs_df.empty:
        pairs_set = set(zip(pairs_df["email_i"].astype(str).tolist(), pairs_df["email_j"].astype(str).tolist(), strict=False))
    return {
        "component_links_csv": str(links_path),
        "candidates_component_expanded_csv": str(pairs_path),
        "n_component_links": int(len(links_df)),
        "n_candidate_rows": int(len(pairs_df)),
        "n_candidate_pairs": int(len(pairs_set)),
        "pairs_set": pairs_set,
    }

