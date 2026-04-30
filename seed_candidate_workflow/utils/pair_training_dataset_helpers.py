"""
Build the email-email pair training dataset artifact (PU pipeline substep 1).

Produces a supervision table: seed pairs as positives, remaining candidate union pairs
as unlabeled by default, and optionally a capped pool of ``reliable_negative`` rows derived
from conservative rules on existing pair metadata.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from seed_candidate_workflow.utils import graph_structure_helpers as gh


def _pair_key(a: str, b: str) -> tuple[str, str]:
    aa, bb = str(a).strip(), str(b).strip()
    return (aa, bb) if aa <= bb else (bb, aa)


def _feature_availability(df: pd.DataFrame, cols: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    n = int(len(df))
    for c in cols:
        if c not in df.columns:
            out[c] = {"present": False, "non_null_count": 0, "non_null_fraction": None}
            continue
        s = df[c]
        nn = int(s.notna().sum())
        out[c] = {
            "present": True,
            "non_null_count": nn,
            "non_null_fraction": float(nn / n) if n else None,
        }
    return out


def _reliable_negative_eligible_mask(
    df: pd.DataFrame,
    both_mapped: pd.Series,
    cfg: dict[str, Any],
) -> pd.Series:
    """
    Conservative high-precision pool: cross-seed-component, single-source, no rare artifact,
    and one of three weak single-source patterns (semantic / 2hop / component only).
    """
    if not bool(cfg.get("enabled", False)):
        return pd.Series(False, index=df.index, dtype=bool)

    not_seed = ~df["is_seed_pair"].astype(bool)
    sc = pd.to_numeric(df["source_count"], errors="coerce")
    if bool(cfg.get("require_source_count_eq_1", True)):
        sc_ok = sc == 1
    else:
        sc_ok = pd.Series(True, index=df.index, dtype=bool)

    if bool(cfg.get("require_cross_seed_component", True)):
        cross_ok = df["cross_seed_component_flag"].astype(bool)
    else:
        cross_ok = pd.Series(True, index=df.index, dtype=bool)

    if bool(cfg.get("exclude_same_seed_component", True)):
        no_same = ~df["same_seed_component_flag"].astype(bool)
    else:
        no_same = pd.Series(True, index=df.index, dtype=bool)

    if bool(cfg.get("exclude_from_rare_artifact", True)):
        no_rare = ~df["from_rare_artifact"].astype(bool)
    else:
        no_rare = pd.Series(True, index=df.index, dtype=bool)

    sem_max = float(cfg.get("semantic_max_cosine", 0.91))
    cos = pd.to_numeric(df["semantic_cosine_max"], errors="coerce")
    fs = df["from_semantic"].astype(bool)
    f2 = df["from_2hop"].astype(bool)
    fc = df["from_component"].astype(bool)
    fr = df["from_rare_artifact"].astype(bool)

    # Route 1 — weak semantic-only
    route1 = (
        fs
        & cos.le(sem_max).fillna(False)
        & ~fr
        & ~f2
        & ~fc
    )

    include_twohop_route = bool(cfg.get("include_twohop_only_route", False))
    include_component_route = bool(cfg.get("include_component_only_route", False))

    # Route 2 — weak 2hop-only bridge (disabled by default for conservative rollback)
    if include_twohop_route:
        th_thr = float(cfg.get("twohop_rarity_max_le", 6.0))
        th = pd.to_numeric(df["twohop_rarity_max"], errors="coerce")
        route2 = (
            f2
            & ~fs
            & ~fr
            & ~fc
            & th.notna()
            & th.le(th_thr)
        )
    else:
        route2 = pd.Series(False, index=df.index, dtype=bool)

    # Route 3 — weak component-only bridge (disabled by default for conservative rollback)
    if include_component_route:
        route3 = fc & ~fs & ~fr & ~f2
    else:
        route3 = pd.Series(False, index=df.index, dtype=bool)

    weak_pattern = route1 | route2 | route3

    return (
        not_seed
        & sc_ok
        & cross_ok
        & no_same
        & no_rare
        & weak_pattern
        & both_mapped.astype(bool)
    )


def _apply_reliable_negative_pool_inplace(
    df: pd.DataFrame,
    *,
    both_mapped: pd.Series,
    pool_cfg: dict[str, Any] | None,
) -> dict[str, Any]:
    """Mutate ``df['pair_status']`` for a subset of current ``unlabeled`` rows. Returns summary dict."""
    cfg = pool_cfg or {}
    if not bool(cfg.get("enabled", False)):
        return {
            "enabled": False,
            "n_candidates_matching_rules": 0,
            "n_reliable_negative_pairs_selected": 0,
            "max_pairs_total": int(cfg.get("max_pairs_total", 0) or 0),
            "random_seed": int(cfg.get("random_seed", 42)),
        }

    mask = _reliable_negative_eligible_mask(df, both_mapped, cfg)
    n_cand = int(mask.sum())
    max_n = max(0, int(cfg.get("max_pairs_total", 15_000)))
    seed = int(cfg.get("random_seed", 42))

    eligible_idx = df.index[mask].tolist()
    if n_cand <= max_n or max_n == 0:
        chosen = eligible_idx
    else:
        rng = np.random.default_rng(seed)
        pos = rng.choice(len(eligible_idx), size=max_n, replace=False)
        chosen = [eligible_idx[i] for i in pos]

    if chosen:
        df.loc[chosen, "pair_status"] = "reliable_negative"

    return {
        "enabled": True,
        "n_candidates_matching_rules": n_cand,
        "n_reliable_negative_pairs_selected": int(len(chosen)),
        "max_pairs_total": max_n,
        "random_seed": seed,
        "semantic_max_cosine": float(cfg.get("semantic_max_cosine", 0.90)),
        "include_twohop_only_route": bool(cfg.get("include_twohop_only_route", False)),
        "include_component_only_route": bool(cfg.get("include_component_only_route", False)),
        "twohop_rarity_max_le": float(cfg.get("twohop_rarity_max_le", 6.0)),
        "require_cross_seed_component": bool(cfg.get("require_cross_seed_component", True)),
        "require_source_count_eq_1": bool(cfg.get("require_source_count_eq_1", True)),
        "exclude_from_rare_artifact": bool(cfg.get("exclude_from_rare_artifact", True)),
        "exclude_same_seed_component": bool(cfg.get("exclude_same_seed_component", True)),
    }


def build_pair_training_dataset(
    *,
    seed_edges_all_csv: Path,
    candidate_union_csv: Path,
    output_dir: Path,
    graph_meta_json: Path | None = None,
    graph_id: str | None = None,
    write_parquet: bool = True,
    write_rejects_csv: bool = True,
    project_root: Path | None = None,
    reliable_negative_pool: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Write pair_training_dataset.csv, pair_training_dataset_summary.json, and optionally
    parquet + rejects CSV.

    Rows are one per unique canonical pair in (candidate_union ∪ seed_edges), with
    features taken from candidate_union when present; seed-only extras get empty feature cells.

    Optional ``reliable_negative_pool``: when ``enabled``, relabels a capped subset of
    non-seed rows from ``unlabeled`` to ``reliable_negative`` using conservative rules.
    """
    seed_edges_all_csv = seed_edges_all_csv.expanduser().resolve()
    candidate_union_csv = candidate_union_csv.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    seed_df = pd.read_csv(seed_edges_all_csv)
    n_seed_rows_read = int(len(seed_df))
    if seed_df.empty or not {"email_i", "email_j"}.issubset(seed_df.columns):
        raise ValueError(f"seed_edges_all.csv missing email_i/email_j or empty: {seed_edges_all_csv}")
    seed_pairs: set[tuple[str, str]] = set()
    for a, b in zip(seed_df["email_i"].astype(str), seed_df["email_j"].astype(str), strict=False):
        if str(a) == str(b):
            continue
        seed_pairs.add(_pair_key(a, b))

    cand = pd.read_csv(candidate_union_csv)
    n_candidate_rows_read = int(len(cand))
    if cand.empty or not {"email_i", "email_j"}.issubset(cand.columns):
        raise ValueError(f"candidate_union.csv missing required columns or empty: {candidate_union_csv}")

    # Canonicalize candidate rows; detect duplicates (keep first, deterministic sort).
    cand = cand.copy()
    cand["_pk"] = [
        _pair_key(a, b) for a, b in zip(cand["email_i"].astype(str), cand["email_j"].astype(str), strict=False)
    ]
    n_dup = int(cand["_pk"].duplicated().sum())
    cand = cand.drop_duplicates(subset=["_pk"], keep="first").sort_values("_pk").reset_index(drop=True)

    cand_pairs = set(cand["_pk"].tolist())
    seed_only = sorted(seed_pairs - cand_pairs)

    # Base rows from candidate union
    rows: list[dict[str, Any]] = []
    for _, r in cand.iterrows():
        pk = r["_pk"]
        is_seed = pk in seed_pairs
        rows.append(
            {
                "email_i": pk[0],
                "email_j": pk[1],
                "from_seed": bool(r.get("from_seed", False)),
                "from_rare_artifact": bool(r.get("from_rare_artifact", False)),
                "from_semantic": bool(r.get("from_semantic", False)),
                "from_component": bool(r.get("from_component", False)),
                "from_2hop": bool(r.get("from_2hop", False)),
                "source_count": r.get("source_count"),
                "semantic_cosine_max": r.get("semantic_cosine_max"),
                "rare_artifact_rarity_max": r.get("rare_artifact_rarity_max"),
                "twohop_rarity_max": r.get("twohop_rarity_max"),
                "component_cosine_max": r.get("component_cosine_max"),
                "time_gap_seconds_min": r.get("time_gap_seconds_min"),
                "email_i_seed_component_id": r.get("email_i_seed_component_id", np.nan),
                "email_j_seed_component_id": r.get("email_j_seed_component_id", np.nan),
                "same_seed_component": r.get("same_seed_component", np.nan),
                "_is_seed_pair": is_seed,
                "_from_candidate": True,
            }
        )

    # Seed-only pairs (should normally be empty when candidate generation invariant holds)
    for pk in seed_only:
        rows.append(
            {
                "email_i": pk[0],
                "email_j": pk[1],
                "from_seed": True,
                "from_rare_artifact": False,
                "from_semantic": False,
                "from_component": False,
                "from_2hop": False,
                "source_count": 1,
                "semantic_cosine_max": np.nan,
                "rare_artifact_rarity_max": np.nan,
                "twohop_rarity_max": np.nan,
                "component_cosine_max": np.nan,
                "time_gap_seconds_min": np.nan,
                "email_i_seed_component_id": np.nan,
                "email_j_seed_component_id": np.nan,
                "same_seed_component": np.nan,
                "_is_seed_pair": True,
                "_from_candidate": False,
            }
        )

    df = pd.DataFrame(rows)
    df = df.sort_values(["email_i", "email_j"]).reset_index(drop=True)

    # Every row is part of the candidate-universe handoff; seed-only extras are still
    # "candidate stage" outputs for training contract purposes.
    df["is_candidate_pair"] = True
    df["is_seed_pair"] = df["_is_seed_pair"].astype(bool)

    # Component context from union columns (needed before reliable-negative rules)
    sci = pd.to_numeric(df["email_i_seed_component_id"], errors="coerce")
    scj = pd.to_numeric(df["email_j_seed_component_id"], errors="coerce")
    df["seed_component_i"] = sci
    df["seed_component_j"] = scj
    same_b = sci.notna() & scj.notna() & (sci >= 0) & (scj >= 0) & (sci == scj)
    if "same_seed_component" in df.columns and df["same_seed_component"].notna().any():
        df["same_seed_component_flag"] = df["same_seed_component"].fillna(False).astype(bool) | same_b.fillna(False)
    else:
        df["same_seed_component_flag"] = same_b.fillna(False)
    both_comp = sci.notna() & scj.notna() & (sci >= 0) & (scj >= 0)
    df["cross_seed_component_flag"] = both_comp & (sci != scj)

    # Graph email indices
    ext_to_idx: dict[str, int] | None = None
    meta_path_resolved: str | None = None
    graph_meta_missing_reason: str | None = None  # why explicit path was not used
    if graph_meta_json is not None:
        gpath = graph_meta_json.expanduser().resolve()
        if gpath.is_file():
            meta = gh.load_meta(gpath)
            ext_to_idx = gh.external_id_to_row(meta)
            meta_path_resolved = str(gpath)
        else:
            graph_meta_missing_reason = f"graph_meta_json_not_found: {gpath}"
    if ext_to_idx is None and graph_meta_json is None:
        try:
            root = project_root or gh.find_project_root()
            paths = gh.resolve_graph_analysis_paths(root)
            if paths.meta_json.is_file():
                meta = gh.load_meta(paths.meta_json)
                ext_to_idx = gh.external_id_to_row(meta)
                meta_path_resolved = str(paths.meta_json)
        except Exception:
            ext_to_idx = None

    if ext_to_idx is not None:
        df["graph_email_idx_i"] = df["email_i"].astype(str).map(ext_to_idx)
        df["graph_email_idx_j"] = df["email_j"].astype(str).map(ext_to_idx)
    else:
        df["graph_email_idx_i"] = np.nan
        df["graph_email_idx_j"] = np.nan

    # Nullable integer indices for CSV
    df["graph_email_idx_i"] = pd.to_numeric(df["graph_email_idx_i"], errors="coerce").astype("Int64")
    df["graph_email_idx_j"] = pd.to_numeric(df["graph_email_idx_j"], errors="coerce").astype("Int64")

    miss_i = df["graph_email_idx_i"].isna()
    miss_j = df["graph_email_idx_j"].isna()
    both_mapped = (~miss_i) & (~miss_j)

    # Supervision labels (after structural flags exist)
    df["pair_status"] = np.where(df["is_seed_pair"], "positive", "unlabeled")
    df.loc[df["is_seed_pair"], "from_seed"] = True

    rn_pool = reliable_negative_pool or {}
    rn_summary = _apply_reliable_negative_pool_inplace(df, both_mapped=both_mapped, pool_cfg=rn_pool)

    # Rejects CSV (optional): mapping failures (main dataset still retains all pairs).
    rejects: list[dict[str, Any]] = []
    if write_rejects_csv:
        bad = df.loc[miss_i | miss_j]
        for _, r in bad.iterrows():
            rs: list[str] = []
            if pd.isna(r["graph_email_idx_i"]):
                rs.append("missing_graph_email_idx_i")
            if pd.isna(r["graph_email_idx_j"]):
                rs.append("missing_graph_email_idx_j")
            rejects.append(
                {
                    "email_i": str(r["email_i"]),
                    "email_j": str(r["email_j"]),
                    "reject_reason": "|".join(rs),
                }
            )

    # Final output columns (exact contract)
    df["split"] = np.nan
    out_df = pd.DataFrame(
        {
            "email_i": df["email_i"].astype(str),
            "email_j": df["email_j"].astype(str),
            "graph_email_idx_i": df["graph_email_idx_i"],
            "graph_email_idx_j": df["graph_email_idx_j"],
            "pair_status": df["pair_status"].astype(str),
            "is_seed_pair": df["is_seed_pair"].astype(bool),
            "is_candidate_pair": df["is_candidate_pair"].astype(bool),
            "from_seed": df["from_seed"].astype(bool),
            "from_rare_artifact": df["from_rare_artifact"].astype(bool),
            "from_semantic": df["from_semantic"].astype(bool),
            "from_component": df["from_component"].astype(bool),
            "from_2hop": df["from_2hop"].astype(bool),
            "source_count": pd.to_numeric(df["source_count"], errors="coerce").astype("Int64"),
            "semantic_cosine_max": pd.to_numeric(df["semantic_cosine_max"], errors="coerce"),
            "rare_artifact_rarity_max": pd.to_numeric(df["rare_artifact_rarity_max"], errors="coerce"),
            "twohop_rarity_max": pd.to_numeric(df["twohop_rarity_max"], errors="coerce"),
            "component_cosine_max": pd.to_numeric(df["component_cosine_max"], errors="coerce"),
            "time_gap_seconds_min": pd.to_numeric(df["time_gap_seconds_min"], errors="coerce"),
            "seed_component_i": pd.to_numeric(df["seed_component_i"], errors="coerce"),
            "seed_component_j": pd.to_numeric(df["seed_component_j"], errors="coerce"),
            "same_seed_component_flag": df["same_seed_component_flag"].astype(bool),
            "cross_seed_component_flag": df["cross_seed_component_flag"].astype(bool),
            "split": df["split"],
        }
    )

    p_csv = output_dir / "pair_training_dataset.csv"
    out_df.to_csv(p_csv, index=False)

    p_sidecar: Path | None = None
    if rn_summary.get("enabled") and bool(rn_pool.get("write_sidecar_csv", True)):
        p_sidecar = output_dir / "reliable_negative_pairs.csv"
        rn_rows = out_df[out_df["pair_status"].astype(str).str.lower() == "reliable_negative"]
        rn_rows.to_csv(p_sidecar, index=False)

    p_parquet = output_dir / "pair_training_dataset.parquet"
    parquet_written = False
    parquet_note: str | None = None
    if write_parquet:
        try:
            out_df.to_parquet(p_parquet, index=False)
            parquet_written = True
        except Exception as exc:  # pragma: no cover - optional dep
            parquet_note = f"parquet_skipped: {type(exc).__name__}: {exc}"

    p_rejects = output_dir / "pair_training_dataset_rejects.csv"
    if write_rejects_csv and rejects:
        pd.DataFrame(rejects).to_csv(p_rejects, index=False)
    elif write_rejects_csv:
        pd.DataFrame(columns=["email_i", "email_j", "reject_reason"]).to_csv(p_rejects, index=False)

    st = out_df["pair_status"].astype(str).str.lower()
    n_pos = int((st == "positive").sum())
    n_unl = int((st == "unlabeled").sum())
    n_rn = int((st == "reliable_negative").sum())
    overlap = int((df["is_seed_pair"] & df["_from_candidate"]).sum())
    unique_emails = int(
        len(set(out_df["email_i"].astype(str).tolist()) | set(out_df["email_j"].astype(str).tolist()))
    )

    feat_cols = [
        "semantic_cosine_max",
        "rare_artifact_rarity_max",
        "twohop_rarity_max",
        "component_cosine_max",
        "time_gap_seconds_min",
        "source_count",
    ]

    both_ids = (
        out_df["seed_component_i"].notna()
        & out_df["seed_component_j"].notna()
        & (out_df["seed_component_i"] >= 0)
        & (out_df["seed_component_j"] >= 0)
    )
    comp_ctx = {
        "seed_component_columns_present": bool(
            "seed_component_i" in out_df.columns and "seed_component_j" in out_df.columns
        ),
        "n_pairs_with_both_seed_component_ids": int(both_ids.sum()),
        "n_pairs_same_seed_component": int((out_df["same_seed_component_flag"] & both_ids).sum()),
        "n_pairs_cross_seed_component": int((out_df["cross_seed_component_flag"] & both_ids).sum()),
        "n_pairs_missing_seed_component_context": int((~both_ids).sum()),
    }

    members_hint = seed_edges_all_csv.parent / "seed_union_component_members.csv"
    comps_hint = seed_edges_all_csv.parent / "seed_union_components.csv"

    summary: dict[str, Any] = {
        "metadata": {
            "created_at_utc": created_at,
            "graph_id": graph_id,
            "seed_edges_all_csv": str(seed_edges_all_csv),
            "candidate_union_csv": str(candidate_union_csv),
            "graph_meta_json": meta_path_resolved,
            "output_dir": str(output_dir),
            "pair_training_dataset_csv": str(p_csv),
            "pair_training_dataset_summary_json": str(output_dir / "pair_training_dataset_summary.json"),
            "pair_training_dataset_parquet": str(p_parquet) if parquet_written else None,
            "pair_training_dataset_rejects_csv": str(p_rejects) if write_rejects_csv else None,
            "reliable_negative_pairs_csv": str(p_sidecar) if p_sidecar is not None else None,
        },
        "inputs": {
            "n_seed_rows_read": n_seed_rows_read,
            "n_candidate_rows_read": n_candidate_rows_read,
            "n_duplicate_candidate_pairs_dropped": n_dup,
            "n_seed_pairs_canonical": int(len(seed_pairs)),
            "n_seed_only_pairs_not_in_candidate_union": int(len(seed_only)),
            "seed_union_component_members_available": bool(members_hint.is_file()),
            "seed_union_components_available": bool(comps_hint.is_file()),
            "graph_email_index_mapping_available": ext_to_idx is not None,
            "graph_meta_path_issue": graph_meta_missing_reason,
        },
        "pair_counts": {
            "n_unique_pairs_final": int(len(out_df)),
            "n_positive_pairs": n_pos,
            "n_unlabeled_pairs": n_unl,
            "n_reliable_negative_pairs": n_rn,
            "n_seed_cap_candidate_overlap_pairs": overlap,
            "unlabeled_to_positive_ratio": float(n_unl / max(1, n_pos)),
            "n_unique_emails_in_pair_dataset": unique_emails,
        },
        "reliable_negative_pool": rn_summary,
        "mapping_quality": {
            "n_rows_total": int(len(out_df)),
            "n_rows_both_graph_indices_present": int(both_mapped.sum()),
            "n_rows_missing_graph_index_i": int(miss_i.sum()),
            "n_rows_missing_graph_index_j": int(miss_j.sum()),
            "n_rows_missing_either_graph_index": int((miss_i | miss_j).sum()),
            "n_rows_dropped_from_output": 0,
            "final_usable_row_count_for_training_both_indices": int(both_mapped.sum()),
        },
        "feature_availability": _feature_availability(out_df, feat_cols),
        "component_context": comp_ctx,
        "split_strategy": "not_constructed_in_substep_1",
        "notes": [
            "Pair supervision: seed pairs => positive; other candidate_union rows => unlabeled by default.",
            "Optional reliable_negative_pool relabels a capped subset of mapped non-seed rows to reliable_negative.",
            "No ground-truth labels for campaign class; train/val/test split happens in GNN training.",
        ],
    }
    if n_dup:
        summary["notes"].append(f"Dropped {n_dup} duplicate candidate_union row(s) on canonical pair key (kept first).")
    if seed_only:
        summary["notes"].append(
            f"Added {len(seed_only)} seed-only pair(s) not present in candidate_union.csv (sparse feature columns)."
        )
    if ext_to_idx is None:
        msg = (
            "Graph email index mapping unavailable; graph_email_idx_* are null. "
            "Pass --graph-meta-json or ensure default pipeline meta resolves."
        )
        if graph_meta_missing_reason:
            msg = f"{msg} ({graph_meta_missing_reason})"
        summary["notes"].append(msg)
    if parquet_note:
        summary["notes"].append(parquet_note)
    summary["notes"].append(
        "Split policy for negatives: reliable_negative rows are written into the same CSV as P/U; "
        "GNN training uses a single random shuffle split across all statuses (see pair_training_setup_summary)."
    )

    p_summary = output_dir / "pair_training_dataset_summary.json"
    p_summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    return {
        "output_dir": str(output_dir),
        "pair_training_dataset_csv": str(p_csv),
        "pair_training_dataset_summary_json": str(p_summary),
        "pair_training_dataset_parquet": str(p_parquet) if parquet_written else None,
        "pair_training_dataset_rejects_csv": str(p_rejects) if write_rejects_csv else None,
        "reliable_negative_pairs_csv": str(p_sidecar) if p_sidecar is not None else None,
        "summary": summary,
    }
