from __future__ import annotations

import json
import math
import shutil
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from seed_candidate_workflow.utils.anchor_candidate_eval_helpers import _pair, _pairs_from_df

DEFAULT_SILVER_EVIDENCE_TYPES = (
    "exact_attachment_hash",
    "exact_html_fingerprint",
    "exact_normalized_url",
    "rare_exact_url_template",
)


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(x) for x in obj]
    return obj


def _load_hard_silver_pool(
    *,
    seed_dir: Path,
    evidence_types: frozenset[str],
) -> pd.DataFrame:
    p_hard = seed_dir / "seed_edges_hard.csv"
    if not p_hard.is_file():
        return pd.DataFrame()
    df = pd.read_csv(p_hard, low_memory=False)
    if df.empty or "evidence_type" not in df.columns:
        return pd.DataFrame()
    if not {"email_i", "email_j"}.issubset(df.columns):
        return pd.DataFrame()
    tier = df.get("seed_tier", pd.Series(["hard"] * len(df))).astype(str)
    gen = df.get("seed_generator", pd.Series(["hard_v1"] * len(df))).astype(str)
    ev = df["evidence_type"].astype(str)
    mask = ev.isin(evidence_types) & ((tier == "hard") | (gen.str.contains("hard", case=False, na=False)))
    out = df.loc[mask].copy()
    if out.empty:
        return out
    out["email_i"] = out["email_i"].astype(str)
    out["email_j"] = out["email_j"].astype(str)
    out = out[out["email_i"] != out["email_j"]]
    out["pair_key"] = [_pair(str(a), str(b)) for a, b in zip(out["email_i"], out["email_j"], strict=False)]
    dedupe_cols = ["pair_key", "evidence_type"]
    if "evidence_value" in out.columns:
        dedupe_cols.append("evidence_value")
    out = out.drop_duplicates(subset=dedupe_cols, keep="first").reset_index(drop=True)
    return out


def _stratified_holdout_row_indices(
    silver_df: pd.DataFrame,
    *,
    random_seed: int,
    holdout_fraction: float,
    min_per_type: int,
) -> set[int]:
    if silver_df.empty:
        return set()
    rng = np.random.default_rng(int(random_seed))
    held: set[int] = set()
    for _et, idx in silver_df.groupby("evidence_type", dropna=False).groups.items():
        idx_list = [int(i) for i in idx]
        n = len(idx_list)
        if n == 0:
            continue
        k = max(int(min_per_type), int(math.ceil(float(holdout_fraction) * n)))
        k = min(k, n)
        chosen = rng.choice(idx_list, size=k, replace=False).tolist()
        held.update(int(i) for i in chosen)
    return held


def _prepare_benchmark_seed_dir_remove_pairs(
    *,
    original_seed_dir: Path,
    benchmark_seed_dir: Path,
    held_out_pair_keys: set[tuple[str, str]],
) -> None:
    if benchmark_seed_dir.exists():
        shutil.rmtree(benchmark_seed_dir)
    benchmark_seed_dir.mkdir(parents=True, exist_ok=True)
    for p in original_seed_dir.iterdir():
        if p.is_file():
            shutil.copy2(p, benchmark_seed_dir / p.name)
    p_all = benchmark_seed_dir / "seed_edges_all.csv"
    if not p_all.is_file():
        return
    df = pd.read_csv(p_all, low_memory=False)
    if df.empty or not {"email_i", "email_j"}.issubset(df.columns):
        return
    df["email_i"] = df["email_i"].astype(str)
    df["email_j"] = df["email_j"].astype(str)
    pk = [_pair(a, b) for a, b in zip(df["email_i"], df["email_j"], strict=False)]
    df = df.loc[[p not in held_out_pair_keys for p in pk]].reset_index(drop=True)
    df.to_csv(p_all, index=False)


def _pairs_in_csv(path: Path) -> set[tuple[str, str]]:
    if not path.is_file():
        return set()
    df = pd.read_csv(path, usecols=["email_i", "email_j"], low_memory=False)
    if df.empty:
        return set()
    return {_pair(str(a), str(b)) for a, b in zip(df["email_i"], df["email_j"], strict=False)}


def _load_component_map(seed_dir: Path) -> dict[str, int]:
    members_path = seed_dir / "seed_union_component_members.csv"
    comp_map: dict[str, int] = {}
    if not members_path.is_file():
        return comp_map
    mdf = pd.read_csv(members_path, low_memory=False)
    if mdf.empty or not {"external_id", "component_id"}.issubset(mdf.columns):
        return comp_map
    for eid, cid in zip(mdf["external_id"].astype(str), mdf["component_id"], strict=False):
        v = pd.to_numeric(cid, errors="coerce")
        comp_map[str(eid)] = int(v) if pd.notna(v) else -1
    return comp_map


def _semantic_cosine_map(sem_df: pd.DataFrame) -> dict[tuple[str, str], float]:
    if sem_df.empty or "cosine" not in sem_df.columns:
        return {}
    out: dict[tuple[str, str], float] = {}
    for a, b, c in zip(
        sem_df["email_i"].astype(str),
        sem_df["email_j"].astype(str),
        sem_df["cosine"].tolist(),
        strict=False,
    ):
        k = _pair(a, b)
        v = float(pd.to_numeric(c, errors="coerce"))
        if pd.isna(v):
            continue
        out[k] = max(out.get(k, float("-inf")), v)
    return out


def _median_semantic_rank_for_pairs(
    sem_df: pd.DataFrame,
    held_recovered: set[tuple[str, str]],
) -> float | None:
    if sem_df.empty or not held_recovered or "cosine" not in sem_df.columns:
        return None
    ranks: list[int] = []
    for anchor in sorted(set(a for p in held_recovered for a in p)):
        sub = sem_df[(sem_df["email_i"].astype(str) == anchor) | (sem_df["email_j"].astype(str) == anchor)].copy()
        if sub.empty:
            continue
        sub["partner"] = np.where(sub["email_i"].astype(str) == anchor, sub["email_j"].astype(str), sub["email_i"].astype(str))
        sub["cos"] = pd.to_numeric(sub["cosine"], errors="coerce")
        sub = sub.dropna(subset=["cos"]).sort_values("cos", ascending=False).reset_index(drop=True)
        for i, partner in enumerate(sub["partner"].tolist()):
            pk = _pair(anchor, partner)
            if pk in held_recovered:
                ranks.append(int(i) + 1)
    if not ranks:
        return None
    return float(np.median(ranks))


def _score_summary_recovered(
    df: pd.DataFrame,
    pair_col_ok: bool,
    held_recovered: set[tuple[str, str]],
    score_col: str,
) -> dict[str, Any] | None:
    if df.empty or not pair_col_ok or score_col not in df.columns:
        return None
    pk_list = [_pair(str(a), str(b)) for a, b in zip(df["email_i"], df["email_j"], strict=False)]
    vals = [float(pd.to_numeric(v, errors="coerce")) for v in df[score_col].tolist()]
    xs = [v for pk, v in zip(pk_list, vals, strict=False) if pk in held_recovered and pd.notna(v)]
    if not xs:
        return None
    arr = np.asarray(xs, dtype=np.float64)
    return {"mean": float(np.mean(arr)), "median": float(np.median(arr)), "p90": float(np.quantile(arr, 0.9)), "n": int(len(xs))}


def run_silver_hidden_link_benchmark(
    *,
    project_root: Path,
    graph_id: str,
    main_out_dir: Path,
    original_seed_dir: Path,
    full_generation_config: dict[str, Any],
    main_candidate_union_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    eval_block = full_generation_config.get("evaluation") or {}
    cfg = (eval_block.get("silver_hidden_link_benchmark") or {})
    if not bool(cfg.get("enabled", False)):
        return {"silver_hidden_link_eval": {"enabled": False}}

    rerun = bool(cfg.get("rerun_candidate_generation", True))
    random_seed = int(cfg.get("random_seed", 42))
    holdout_fraction = float(cfg.get("holdout_fraction", 0.2))
    min_per_type = int(cfg.get("min_holdout_per_evidence_type", 1))
    evidence_allow = frozenset(str(x) for x in (cfg.get("evidence_types") or list(DEFAULT_SILVER_EVIDENCE_TYPES)))
    stage_name = str(cfg.get("benchmark_stage_name") or "candidate_generation_silver_benchmark")
    split_note = str(
        cfg.get(
            "split_note",
            "Stratified random holdout by evidence_type on hard silver rows; all seed rows for held-out pairs are removed from benchmark seed_edges_all.csv.",
        )
    )

    semantic_threshold = 0.9
    for g in (full_generation_config.get("candidates") or {}).get("generators") or []:
        if isinstance(g, dict) and str(g.get("name") or "") == "semantic_reciprocal_v1":
            gc = g.get("config") or {}
            semantic_threshold = float(gc.get("semantic_min_cos", semantic_threshold))
            break

    silver_df = _load_hard_silver_pool(seed_dir=original_seed_dir, evidence_types=evidence_allow)
    pool_counts = silver_df.groupby("evidence_type").size().to_dict() if not silver_df.empty else {}
    n_silver_total = int(len(silver_df))
    if n_silver_total == 0:
        block = _json_safe(
            {
                "enabled": True,
                "random_seed": random_seed,
                "holdout_fraction": holdout_fraction,
                "n_silver_edges_total": 0,
                "n_silver_edges_eligible": 0,
                "n_silver_edges_held_out": 0,
                "n_unique_held_out_pairs": 0,
                "held_out_by_evidence_type": {},
                "benchmark_mode": "no_silver_pool",
                "notes": "No hard-tier silver edges found in seed_edges_hard.csv for the configured evidence_types.",
                "silver_pool_counts_by_evidence_type": {},
                "benchmark_invalid": False,
            }
        )
        p_summary = main_out_dir / "silver_hidden_link_summary.json"
        p_summary.write_text(json.dumps(block, indent=2, ensure_ascii=False), encoding="utf-8")
        pd.DataFrame().to_csv(main_out_dir / "silver_hidden_link_per_source.csv", index=False)
        pd.DataFrame().to_csv(main_out_dir / "silver_hidden_link_manual_review.csv", index=False)
        return {
            "silver_hidden_link_eval": block,
            "silver_hidden_link_summary_json": str(p_summary),
            "silver_hidden_link_per_source_csv": str(main_out_dir / "silver_hidden_link_per_source.csv"),
            "silver_hidden_link_manual_review_csv": str(main_out_dir / "silver_hidden_link_manual_review.csv"),
        }

    held_idx = _stratified_holdout_row_indices(
        silver_df,
        random_seed=random_seed,
        holdout_fraction=holdout_fraction,
        min_per_type=min_per_type,
    )
    held_rows = silver_df.loc[sorted(held_idx)].copy() if held_idx else pd.DataFrame()
    held_pairs = set(held_rows["pair_key"].tolist()) if not held_rows.empty else set()
    held_by_et = held_rows.groupby("evidence_type").size().to_dict() if not held_rows.empty else {}
    if held_rows.empty or not held_pairs:
        block = _json_safe(
            {
                "enabled": True,
                "random_seed": random_seed,
                "holdout_fraction": holdout_fraction,
                "n_silver_edges_total": n_silver_total,
                "n_silver_edges_eligible": n_silver_total,
                "n_silver_edges_held_out": 0,
                "n_unique_held_out_pairs": 0,
                "held_out_by_evidence_type": {},
                "benchmark_mode": "no_holdout_sample",
                "notes": "Stratified holdout produced zero rows (check holdout_fraction and pool size).",
                "silver_pool_counts_by_evidence_type": {str(k): int(v) for k, v in pool_counts.items()},
                "benchmark_invalid": False,
            }
        )
        p_summary = main_out_dir / "silver_hidden_link_summary.json"
        p_summary.write_text(json.dumps(block, indent=2, ensure_ascii=False), encoding="utf-8")
        pd.DataFrame().to_csv(main_out_dir / "silver_hidden_link_per_source.csv", index=False)
        pd.DataFrame().to_csv(main_out_dir / "silver_hidden_link_manual_review.csv", index=False)
        return {
            "silver_hidden_link_eval": block,
            "silver_hidden_link_summary_json": str(p_summary),
            "silver_hidden_link_per_source_csv": str(main_out_dir / "silver_hidden_link_per_source.csv"),
            "silver_hidden_link_manual_review_csv": str(main_out_dir / "silver_hidden_link_manual_review.csv"),
        }

    comp_map_orig = _load_component_map(original_seed_dir)

    main_union = main_candidate_union_df if main_candidate_union_df is not None else pd.DataFrame()
    main_sem_max: dict[tuple[str, str], float] = {}
    if not main_union.empty and "semantic_cosine_max" in main_union.columns:
        for _, r in main_union.iterrows():
            pk = _pair(str(r["email_i"]), str(r["email_j"]))
            v = float(pd.to_numeric(r.get("semantic_cosine_max"), errors="coerce"))
            if pd.notna(v):
                main_sem_max[pk] = max(main_sem_max.get(pk, float("-inf")), v)

    if not rerun:
        block: dict[str, Any] = _json_safe(
            {
                "enabled": True,
                "random_seed": random_seed,
                "holdout_fraction": holdout_fraction,
                "n_silver_edges_total": n_silver_total,
                "n_silver_edges_eligible": n_silver_total,
                "n_silver_edges_held_out": int(len(held_rows)),
                "n_unique_held_out_pairs": int(len(held_pairs)),
                "held_out_by_evidence_type": {str(k): int(v) for k, v in held_by_et.items()},
                "benchmark_mode": "skipped_rerun_candidate_generation",
                "notes": split_note + " rerun_candidate_generation=false; no benchmark candidate run executed.",
                "silver_pool_counts_by_evidence_type": {str(k): int(v) for k, v in pool_counts.items()},
                "benchmark_candidate_output_dir": None,
                "benchmark_seed_dir": None,
                "leak_checks": None,
                "universe_recovery": None,
                "per_source_recovery": [],
                "recovery_overlap": {},
                "per_evidence_type_recovery": [],
                "difficulty_aware_slices": [],
            }
        )
        p_summary = main_out_dir / "silver_hidden_link_summary.json"
        p_summary.write_text(json.dumps(block, indent=2, ensure_ascii=False), encoding="utf-8")
        pd.DataFrame().to_csv(main_out_dir / "silver_hidden_link_per_source.csv", index=False)
        pd.DataFrame().to_csv(main_out_dir / "silver_hidden_link_manual_review.csv", index=False)
        return {
            "silver_hidden_link_eval": block,
            "silver_hidden_link_summary_json": str(p_summary),
            "silver_hidden_link_per_source_csv": str(main_out_dir / "silver_hidden_link_per_source.csv"),
            "silver_hidden_link_manual_review_csv": str(main_out_dir / "silver_hidden_link_manual_review.csv"),
        }

    bench_seed_dir = (main_out_dir / "silver_benchmark_seed").resolve()
    _prepare_benchmark_seed_dir_remove_pairs(
        original_seed_dir=original_seed_dir,
        benchmark_seed_dir=bench_seed_dir,
        held_out_pair_keys=held_pairs,
    )

    leak_pairs = _pairs_in_csv(bench_seed_dir / "seed_edges_all.csv") & held_pairs
    held_out_seed_leak_count = int(len(leak_pairs))

    bench_cfg = deepcopy(full_generation_config)
    bench_cfg.setdefault("evaluation", {})
    bench_cfg["evaluation"] = {**bench_cfg["evaluation"], "silver_hidden_link_benchmark": {"enabled": False}}
    bench_cfg.setdefault("run", {})
    bench_cfg["run"]["seed_stage_dir"] = str(bench_seed_dir)
    bench_cfg.setdefault("output", {})
    bench_cfg["output"]["stage_name"] = stage_name

    from seed_candidate_workflow.utils.anchor_candidate_generation_helpers import run_anchor_candidate_generation

    bench_result = run_anchor_candidate_generation(bench_cfg)
    bench_out = Path(str(bench_result["output_dir"]))

    union_path = bench_out / "candidate_union.csv"
    union_df = pd.read_csv(union_path, low_memory=False) if union_path.is_file() else pd.DataFrame()
    union_pair_set = _pairs_from_df(union_df)
    n_held_edges = int(len(held_rows))
    n_held_pairs = int(len(held_pairs))
    union_recovered_pairs = held_pairs & union_pair_set
    union_edge_recovered_count = int(sum(1 for _, row in held_rows.iterrows() if row["pair_key"] in union_pair_set))
    uni_metrics = {
        "union_recovered_edge_count": union_edge_recovered_count,
        "union_missed_edge_count": int(n_held_edges - union_edge_recovered_count),
        "union_recall_on_held_out_silver": float(union_edge_recovered_count / n_held_edges) if n_held_edges else None,
        "n_unique_held_out_pairs_recovered_in_union": int(len(union_recovered_pairs)),
        "n_unique_held_out_pairs": n_held_pairs,
    }

    held_emails: set[str] = set()
    for a, b in held_pairs:
        held_emails.add(a)
        held_emails.add(b)
    touched: set[str] = set()
    if not union_df.empty and {"email_i", "email_j"}.issubset(union_df.columns):
        for a, b in zip(union_df["email_i"].astype(str), union_df["email_j"].astype(str), strict=False):
            if a in held_emails or b in held_emails:
                touched.add(a)
                touched.add(b)
    n_held_em = len(held_emails)
    pct_touched = float(len(touched & held_emails) / n_held_em) if n_held_em else None

    paths = {
        "seed": bench_out / "candidates_seed_backbone.csv",
        "rare_artifact": bench_out / "candidates_rare_artifact.csv",
        "semantic": bench_out / "candidates_semantic.csv",
        "component": bench_out / "candidates_component_expanded.csv",
        "2hop": bench_out / "candidates_2hop.csv",
    }
    source_pairs: dict[str, set[tuple[str, str]]] = {k: _pairs_in_csv(p) for k, p in paths.items()}

    sem_df = pd.read_csv(paths["semantic"], low_memory=False) if paths["semantic"].is_file() else pd.DataFrame()
    cos_map = _semantic_cosine_map(sem_df)

    held_recovered_by_source: dict[str, set[tuple[str, str]]] = {
        src: held_pairs & pairs for src, pairs in source_pairs.items()
    }
    seed_recovered = held_recovered_by_source.get("seed", set())
    held_out_recovered_via_seed_source_count = int(len(seed_recovered))
    seed_source_leak_warning = held_out_recovered_via_seed_source_count > 0

    overlap_counts: dict[str, int] = {
        "held_out_recovered_by_exactly_1_non_seed_source": 0,
        "held_out_recovered_by_exactly_2_non_seed_sources": 0,
        "held_out_recovered_by_3_plus_non_seed_sources": 0,
        "held_out_recovered_semantic_only": 0,
        "held_out_recovered_infra_only": 0,
        "held_out_recovered_semantic_plus_non_semantic": 0,
    }
    for _, row in held_rows.iterrows():
        pk = row["pair_key"]
        if pk not in union_pair_set:
            continue
        sem = pk in source_pairs.get("semantic", set())
        rare = pk in source_pairs.get("rare_artifact", set())
        comp = pk in source_pairs.get("component", set())
        hop = pk in source_pairs.get("2hop", set())
        n_ns = sum([rare, sem, comp, hop])
        if n_ns == 1 and sem:
            overlap_counts["held_out_recovered_semantic_only"] += 1
        if n_ns >= 1 and not sem and (rare or comp or hop):
            overlap_counts["held_out_recovered_infra_only"] += 1
        if sem and (rare or comp or hop):
            overlap_counts["held_out_recovered_semantic_plus_non_semantic"] += 1
        if n_ns == 1:
            overlap_counts["held_out_recovered_by_exactly_1_non_seed_source"] += 1
        elif n_ns == 2:
            overlap_counts["held_out_recovered_by_exactly_2_non_seed_sources"] += 1
        elif n_ns >= 3:
            overlap_counts["held_out_recovered_by_3_plus_non_seed_sources"] += 1

    all_src_order = ["seed", "rare_artifact", "semantic", "component", "2hop"]
    overlap_counts["held_out_recovered_by_exactly_1_any_source"] = 0
    overlap_counts["held_out_recovered_by_exactly_2_any_source"] = 0
    overlap_counts["held_out_recovered_by_3_plus_any_source"] = 0
    for _, row in held_rows.iterrows():
        pk = row["pair_key"]
        if pk not in union_pair_set:
            continue
        n_all = sum(1 for s in all_src_order if pk in source_pairs.get(s, set()))
        if n_all == 1:
            overlap_counts["held_out_recovered_by_exactly_1_any_source"] += 1
        elif n_all == 2:
            overlap_counts["held_out_recovered_by_exactly_2_any_source"] += 1
        elif n_all >= 3:
            overlap_counts["held_out_recovered_by_3_plus_any_source"] += 1

    rare_df = pd.read_csv(paths["rare_artifact"], low_memory=False) if paths["rare_artifact"].is_file() else pd.DataFrame()
    comp_df = pd.read_csv(paths["component"], low_memory=False) if paths["component"].is_file() else pd.DataFrame()
    hop_df = pd.read_csv(paths["2hop"], low_memory=False) if paths["2hop"].is_file() else pd.DataFrame()

    sem_held_rec = held_recovered_by_source.get("semantic", set())
    median_rank_sem = _median_semantic_rank_for_pairs(sem_df, sem_held_rec)

    per_src_rows: list[dict[str, Any]] = []
    for src, p in paths.items():
        rec = held_recovered_by_source.get(src, set())
        by_et: dict[str, int] = defaultdict(int)
        n_edge_rec = 0
        for _, row in held_rows.iterrows():
            pk = row["pair_key"]
            et = str(row["evidence_type"])
            if pk in rec:
                by_et[et] += 1
                n_edge_rec += 1
        row_out: dict[str, Any] = {
            "source_name": src,
            "held_out_silver_recall": float(n_edge_rec / n_held_edges) if n_held_edges else None,
            "held_out_silver_recovered_edge_count": int(n_edge_rec),
            "held_out_silver_recovered_unique_pair_count": int(len(rec)),
            "recovered_by_evidence_type_json": json.dumps({str(k): int(v) for k, v in sorted(by_et.items())}),
            "output_path": str(p) if p.is_file() else None,
            "median_rank_recovered_held_out_within_source": median_rank_sem if src == "semantic" and sem_held_rec else None,
        }
        ss: dict[str, Any] | None = None
        if src == "rare_artifact":
            ss = _score_summary_recovered(rare_df, not rare_df.empty, rec, "rarity_score")
        elif src == "semantic":
            ss = _score_summary_recovered(sem_df, not sem_df.empty, rec, "cosine")
        elif src == "2hop":
            ss = _score_summary_recovered(hop_df, not hop_df.empty, rec, "rarity_score")
        elif src == "component":
            ss = _score_summary_recovered(comp_df, not comp_df.empty and "cosine" in comp_df.columns, rec, "cosine")
        if ss:
            row_out["score_on_recovered_held_out_mean"] = ss.get("mean")
            row_out["score_on_recovered_held_out_median"] = ss.get("median")
            row_out["score_on_recovered_held_out_n"] = ss.get("n")
        per_src_rows.append(row_out)

    by_evidence: list[dict[str, Any]] = []
    for et in sorted(evidence_allow):
        sub = held_rows[held_rows["evidence_type"].astype(str) == et] if not held_rows.empty else pd.DataFrame()
        n_h_edges = int(len(sub))
        urec_edges = int(sum(1 for _, row in sub.iterrows() if row["pair_key"] in union_pair_set)) if n_h_edges else 0
        src_rec: dict[str, Any] = {}
        for sname in ["seed", "rare_artifact", "semantic", "component", "2hop"]:
            sp = source_pairs.get(sname, set())
            c = int(sum(1 for _, row in sub.iterrows() if row["pair_key"] in sp)) if n_h_edges else 0
            src_rec[sname] = float(c / n_h_edges) if n_h_edges else None
        by_evidence.append(
            {
                "evidence_type": et,
                "held_out_edge_count": n_h_edges,
                "held_out_unique_pair_count": int(len(set(sub["pair_key"].tolist()))) if n_h_edges else 0,
                "union_recall_on_edges": float(urec_edges / n_h_edges) if n_h_edges else None,
                "per_source_recall_on_edges": src_rec,
            }
        )

    pair_n_evidence = silver_df.groupby("pair_key").size().to_dict() if not silver_df.empty else {}

    def _difficulty_slice(name: str, mask: pd.Series) -> dict[str, Any] | None:
        if held_rows.empty or int(mask.sum()) == 0:
            return None
        sub = held_rows.loc[mask]
        hp_sub = set(sub["pair_key"].tolist())
        rec_edges = int(sum(1 for _, row in sub.iterrows() if row["pair_key"] in union_pair_set))
        n_e = int(len(sub))
        return {
            "slice": name,
            "n_held_out_edges": n_e,
            "n_unique_held_out_pairs": int(len(hp_sub)),
            "union_recovered_edge_count": rec_edges,
            "union_recall_on_edges": float(rec_edges / n_e) if n_e else None,
        }

    difficulty_slices: list[dict[str, Any]] = []
    if not held_rows.empty:
        same_comp_flags = []
        high_sem_flags = []
        multi_ev_flags = []
        for _, row in held_rows.iterrows():
            pk = row["pair_key"]
            a, b = pk
            ci, cj = comp_map_orig.get(a, -1), comp_map_orig.get(b, -1)
            same_comp_flags.append(bool(ci >= 0 and cj >= 0 and ci == cj))
            ms = main_sem_max.get(pk)
            high_sem_flags.append(bool(ms is not None and ms >= semantic_threshold))
            multi_ev_flags.append(int(pair_n_evidence.get(pk, 1)) > 1)
        hr = held_rows.copy()
        hr["_same"] = same_comp_flags
        hr["_high_sem"] = high_sem_flags
        hr["_multi"] = multi_ev_flags
        for slc_name, m in [
            ("same_original_seed_component", hr["_same"]),
            ("different_original_seed_component", ~hr["_same"]),
            ("high_semantic_similarity_in_main_union", hr["_high_sem"]),
            ("low_semantic_similarity_in_main_union", ~hr["_high_sem"]),
            ("multi_evidence_silver_context_per_pair", hr["_multi"]),
            ("single_evidence_silver_context_per_pair", ~hr["_multi"]),
        ]:
            d = _difficulty_slice(slc_name, m)
            if d:
                difficulty_slices.append(d)

    recall_val = uni_metrics.get("union_recall_on_held_out_silver")
    strength = "weak"
    if recall_val is not None:
        if recall_val >= 0.7:
            strength = "strong"
        elif recall_val >= 0.4:
            strength = "moderate"

    manual_rows: list[dict[str, Any]] = []
    if not held_rows.empty:
        for _, row in held_rows.iterrows():
            pk = row["pair_key"]
            a, b = pk
            et = str(row["evidence_type"])
            ci, cj = comp_map_orig.get(a, -1), comp_map_orig.get(b, -1)
            same_comp = bool(ci >= 0 and cj >= 0 and ci == cj)
            cos = cos_map.get(pk)
            ms = main_sem_max.get(pk)
            manual_rows.append(
                {
                    "email_i": a,
                    "email_j": b,
                    "silver_evidence_type": et,
                    "was_recovered_in_union": bool(pk in union_pair_set),
                    "recovered_by_seed": bool(pk in source_pairs.get("seed", set())),
                    "recovered_by_rare_artifact": bool(pk in source_pairs.get("rare_artifact", set())),
                    "recovered_by_semantic": bool(pk in source_pairs.get("semantic", set())),
                    "recovered_by_component": bool(pk in source_pairs.get("component", set())),
                    "recovered_by_2hop": bool(pk in source_pairs.get("2hop", set())),
                    "semantic_score_if_available": cos if cos is not None else None,
                    "semantic_cosine_max_main_union_if_available": ms if ms is not None else None,
                    "original_seed_component_i": int(ci),
                    "original_seed_component_j": int(cj),
                    "original_same_seed_component_flag": same_comp,
                    "recovery_source_count": int(
                        sum(
                            [
                                pk in source_pairs.get("seed", set()),
                                pk in source_pairs.get("rare_artifact", set()),
                                pk in source_pairs.get("semantic", set()),
                                pk in source_pairs.get("component", set()),
                                pk in source_pairs.get("2hop", set()),
                            ]
                        )
                    ),
                    "high_semantic_similarity_main_union": bool(ms is not None and ms >= semantic_threshold),
                    "multi_silver_evidence_on_pair_in_pool": bool(int(pair_n_evidence.get(pk, 1)) > 1),
                }
            )
    manual_df = pd.DataFrame(manual_rows)
    n_samp = int(cfg.get("manual_review_sample_per_arm", 25))
    rng = np.random.default_rng(random_seed + 991)
    picks: list[pd.DataFrame] = []
    if not manual_df.empty:
        for arm, mask_col in [("recovered", manual_df["was_recovered_in_union"]), ("missed", ~manual_df["was_recovered_in_union"])]:
            base = manual_df[mask_col]
            if base.empty:
                continue
            by_et = base.groupby("silver_evidence_type")
            parts: list[pd.DataFrame] = []
            n_types = max(1, len(by_et))
            per_type = max(1, n_samp // n_types)
            for et, sub in by_et:
                k = min(per_type, len(sub))
                if k > 0:
                    parts.append(sub.sample(n=k, random_state=int(rng.integers(0, 2**31 - 1))).assign(sample_arm=arm, stratify_evidence_type=str(et)))
            if parts:
                cat = pd.concat(parts, ignore_index=True)
                if len(cat) > n_samp:
                    cat = cat.sample(n=n_samp, random_state=int(rng.integers(0, 2**31 - 1)))
                picks.append(cat)
    manual_sample = pd.concat(picks, ignore_index=True) if picks else pd.DataFrame()

    benchmark_invalid = bool(held_out_seed_leak_count > 0 or seed_source_leak_warning)

    summary_block: dict[str, Any] = {
        "enabled": True,
        "random_seed": random_seed,
        "holdout_fraction": holdout_fraction,
        "n_silver_edges_total": n_silver_total,
        "n_silver_edges_eligible": n_silver_total,
        "n_silver_edges_held_out": int(len(held_rows)),
        "n_unique_held_out_pairs": int(len(held_pairs)),
        "held_out_by_evidence_type": {str(k): int(v) for k, v in held_by_et.items()},
        "benchmark_mode": "mode_a_rerun_candidate_generation_seed_minus_held_out_pairs",
        "notes": split_note,
        "silver_pool_counts_by_evidence_type": {str(k): int(v) for k, v in pool_counts.items()},
        "benchmark_candidate_output_dir": str(bench_out),
        "benchmark_seed_dir": str(bench_seed_dir),
        "leak_checks": {
            "held_out_seed_leak_count": held_out_seed_leak_count,
            "held_out_recovered_via_seed_source_count": held_out_recovered_via_seed_source_count,
            "seed_source_leak_warning": seed_source_leak_warning,
        },
        "universe_recovery": {
            **uni_metrics,
            "held_out_emails_total": int(n_held_em),
            "held_out_emails_touched_by_any_candidate": int(len(touched & held_emails)),
            "pct_held_out_emails_touched_by_any_candidate": pct_touched,
        },
        "per_source_recovery": per_src_rows,
        "recovery_overlap": overlap_counts,
        "recovery_overlap_counting_basis": "Each held-out silver edge (row) whose pair appears in the benchmark candidate union is counted once in overlap metrics; source flags are per pair within that benchmark run.",
        "per_evidence_type_recovery": by_evidence,
        "difficulty_aware_slices": difficulty_slices,
        "mode_b_note": "Per-source recovery for semantic, rare_artifact, component, and 2hop is in per_source_recovery and recovery_overlap (held_out_recovered_semantic_only, infra_only, semantic_plus_non_semantic).",
        "pct_recovered_held_out_semantic_only_of_union_recovered_edges": (
            float(overlap_counts["held_out_recovered_semantic_only"] / max(1, union_edge_recovered_count))
            if union_edge_recovered_count
            else None
        ),
        "recovery_strength_label": strength,
        "benchmark_invalid": benchmark_invalid,
        "semantic_threshold_used": semantic_threshold,
    }

    summary_block = _json_safe(summary_block)
    p_summary = main_out_dir / "silver_hidden_link_summary.json"
    p_summary.write_text(json.dumps(summary_block, indent=2, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame(per_src_rows).to_csv(main_out_dir / "silver_hidden_link_per_source.csv", index=False)
    manual_sample.to_csv(main_out_dir / "silver_hidden_link_manual_review.csv", index=False)

    return {
        "silver_hidden_link_eval": summary_block,
        "silver_hidden_link_summary_json": str(p_summary),
        "silver_hidden_link_per_source_csv": str(main_out_dir / "silver_hidden_link_per_source.csv"),
        "silver_hidden_link_manual_review_csv": str(main_out_dir / "silver_hidden_link_manual_review.csv"),
    }
