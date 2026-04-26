from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from analysis.utils import graph_structure_helpers as gh
from analysis.utils.config_run_fields import resolve_graph_id
from analysis.utils.anchor_graph_helpers import load_anchor_graph_artifacts


def _to_set_cell(x: Any) -> set[str]:
    # Fallback if nodes_df didn't deserialize *_set columns.
    if isinstance(x, set):
        return {str(v) for v in x if v is not None and str(v).strip()}
    if isinstance(x, (list, tuple)):
        return {str(v) for v in x if v is not None and str(v).strip()}
    if x is None:
        return set()
    s = str(x).strip()
    if not s:
        return set()
    try:
        obj = json.loads(s)
        if isinstance(obj, list):
            return {str(v) for v in obj if v is not None and str(v).strip()}
    except Exception:
        pass
    # Some cached formats may use "|" separators.
    return {v.strip() for v in s.split("|") if v.strip()}


def _artifact_idf(df_val: int, n_docs: int) -> float:
    # Same spirit as seed-stage helper: higher idf => rarer.
    return float(math.log((1.0 + n_docs) / (1.0 + max(1, int(df_val))))) if n_docs > 0 else float("nan")


def _resolve_latest_seed_dir(
    *,
    seed_output_root: Path,
    graph_run_id: str,
    seed_stage_name_prefix: str,
) -> Path:
    base = (seed_output_root / graph_run_id).expanduser().resolve()
    if not base.is_dir():
        raise FileNotFoundError(f"Seed output root missing: {base}")

    dirs = [d for d in base.iterdir() if d.is_dir() and d.name.startswith(seed_stage_name_prefix)]
    if not dirs:
        raise FileNotFoundError(
            f"No seed generation dirs starting with {seed_stage_name_prefix!r} under {base}"
        )

    # Names include ISO-ish timestamps; lexicographic sort is stable for that format.
    dirs_sorted = sorted(dirs, key=lambda p: p.name)
    return dirs_sorted[-1]


def _load_seed_pairs(seed_edges_all_csv: Path) -> set[tuple[str, str]]:
    df = pd.read_csv(seed_edges_all_csv, low_memory=False)
    if df.empty:
        return set()
    if not {"email_i", "email_j"}.issubset(df.columns):
        raise ValueError(f"seed_edges_all.csv missing email_i/email_j: {seed_edges_all_csv}")
    pairs = set(zip(df["email_i"].astype(str).tolist(), df["email_j"].astype(str).tolist(), strict=False))
    return pairs


def generate_candidates_rare_artifact_v1(
    *,
    nodes_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    seed_pairs: set[tuple[str, str]] | None,
    generator_cfg: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Rare-artifact candidates generated from anchor graph edges + node artifact sets.

    This generator does NOT inject seed-backbone rows. The global seed-universe
    invariant is enforced at the candidate-union level in the unified stage.
    """
    n_docs = int(len(nodes_df))
    include_time_gap_seconds = bool(generator_cfg.get("include_time_gap_seconds", True))

    # Artifact specs are configured with both the node set column and edge overlap flag base name.
    artifact_specs = generator_cfg.get("artifact_specs") or []
    if not isinstance(artifact_specs, list) or not artifact_specs:
        raise ValueError("rare_artifact_v1 requires artifact_specs: non-empty list")

    # Precompute document frequencies per spec node-set column.
    # doc freq = number of emails containing the artifact value at least once.
    df_counts_by_col: dict[str, Counter[str]] = {}
    ts_map: dict[str, float] = {}
    for _, r in nodes_df.iterrows():
        eid = str(r["external_id"])
        if "ts" in nodes_df.columns:
            try:
                ts_map[eid] = float(pd.to_numeric(r.get("ts"), errors="coerce"))
            except Exception:
                ts_map[eid] = float("nan")

    # Build value counters.
    cols_needed = [str(s.get("node_set_col")) for s in artifact_specs if str(s.get("node_set_col", "")).strip()]
    for col in cols_needed:
        if col not in nodes_df.columns:
            continue
        c = Counter()
        for vals in nodes_df[col].tolist():
            sset = _to_set_cell(vals)
            for v in sset:
                c[str(v)] += 1
        df_counts_by_col[col] = c

    # Build a fast node->set mapping for required cols.
    node_sets: dict[str, dict[str, set[str]]] = defaultdict(dict)
    for _, r in nodes_df.iterrows():
        eid = str(r["external_id"])
        for col in cols_needed:
            if col in nodes_df.columns:
                node_sets[eid][col] = _to_set_cell(r.get(col))

    candidates_rows: list[dict[str, Any]] = []

    # Helper: deduplicate within run
    max_rows = generator_cfg.get("max_candidate_rows")
    max_total_rows = int(max_rows) if max_rows is not None else None

    min_idf = float(generator_cfg.get("min_artifact_idf", 0.0))
    max_df = generator_cfg.get("max_artifact_df")
    max_df = None if max_df is None else int(max_df)
    max_shared_per_edge_per_artifact = int(generator_cfg.get("max_shared_values_per_edge_per_artifact", 25))

    # Iterate anchor graph edges (much narrower than all-pairs).
    for _, e in edges_df.iterrows():
        a = str(e["email_a"])
        b = str(e["email_b"])
        if a == b:
            continue

        # Ensure deterministic ordering.
        email_i, email_j = (a, b) if a < b else (b, a)

        # Skip if we already reached global cap (best-effort).
        if max_total_rows is not None and len(candidates_rows) >= max_total_rows:
            break

        for spec in artifact_specs:
            artifact_type = str(spec.get("artifact_type") or "").strip()
            node_set_col = str(spec.get("node_set_col") or "").strip()
            overlap_base = str(spec.get("overlap_base") or artifact_type).strip()
            if not artifact_type or not node_set_col:
                continue
            if node_set_col not in node_sets.get(a, {}) or node_set_col not in node_sets.get(b, {}):
                continue

            overlap_col = f"has_{overlap_base}_overlap"
            if overlap_col in edges_df.columns and not bool(e.get(overlap_col, False)):
                continue

            shared_vals = node_sets[a].get(node_set_col, set()) & node_sets[b].get(node_set_col, set())
            if not shared_vals:
                continue

            # Filter shared values by rarity.
            df_counter = df_counts_by_col.get(node_set_col) or Counter()
            scored: list[tuple[float, str, int]] = []
            for v in shared_vals:
                df_val = int(df_counter.get(str(v), 0))
                idf = _artifact_idf(df_val=df_val, n_docs=n_docs)
                if not np.isfinite(idf):
                    continue
                if idf < min_idf:
                    continue
                if max_df is not None and df_val > max_df:
                    continue
                scored.append((idf, str(v), df_val))

            if not scored:
                continue

            scored.sort(key=lambda x: x[0], reverse=True)  # highest idf first
            for idf, val, df_val in scored[:max_shared_per_edge_per_artifact]:
                row: dict[str, Any] = {
                    "email_i": email_i,
                    "email_j": email_j,
                    "source": "rare_artifact",
                    "artifact_type": artifact_type,
                    "artifact_value": val,
                    "artifact_df": int(df_val),
                    "rarity_score": float(idf),
                }
                if include_time_gap_seconds:
                    ts_a = ts_map.get(a, float("nan"))
                    ts_b = ts_map.get(b, float("nan"))
                    if np.isfinite(ts_a) and np.isfinite(ts_b):
                        row["time_gap_seconds"] = float(abs(ts_a - ts_b))
                    else:
                        row["time_gap_seconds"] = float("nan")
                candidates_rows.append(row)

    candidates_df = pd.DataFrame(candidates_rows)

    # Deduplicate rows.
    if not candidates_df.empty:
        candidates_df = candidates_df.drop_duplicates(
            subset=["email_i", "email_j", "source", "artifact_type", "artifact_value"]
        ).reset_index(drop=True)

    # Coverage diagnostics (for reporting only).
    candidate_pairs = set(zip(candidates_df["email_i"].astype(str).tolist(), candidates_df["email_j"].astype(str).tolist(), strict=False))
    seed_pairs_set = set(seed_pairs or set())
    missing = seed_pairs_set - candidate_pairs
    sup = {
        "n_seed_pairs": int(len(seed_pairs_set)),
        "n_candidate_pairs": int(len(candidate_pairs)),
        "n_candidate_pairs_that_are_seed": int(len(candidate_pairs & seed_pairs_set)),
        "n_missing_seed_pairs": int(len(missing)),
    }

    return candidates_df, sup


def run_anchor_rare_artifact_candidate_generation(config: dict[str, Any]) -> dict[str, Any]:
    run_cfg = config.get("run") or {}
    anchor_cfg = config.get("anchor_output_root") or {}
    candidate_cfg = config.get("candidate_generation") or {}
    out_cfg = config.get("output") or {}
    seed_cfg = config.get("seed") or {}

    graph_id = resolve_graph_id(run_cfg)

    project_root = gh.find_project_root()

    anchor_output_root = Path(
        run_cfg.get("anchor_output_root") or (project_root / "analysis" / "output" / "anchor_graph")
    ).expanduser().resolve()
    anchor_run_dir = anchor_output_root / graph_id
    if not anchor_run_dir.is_dir():
        raise FileNotFoundError(f"Anchor graph run directory not found: {anchor_run_dir}")

    nodes_df, edges_df, _candidates, _summary, _g = load_anchor_graph_artifacts(anchor_run_dir, load_graph_pickle=False)

    # Resolve seed stage latest dir.
    seed_output_root = Path(seed_cfg.get("seed_output_root") or (project_root / "analysis" / "output" / "anchor_seeds")).expanduser().resolve()
    seed_stage_prefix = str(seed_cfg.get("seed_stage_name_prefix") or "seed_generation_")
    seed_dir = _resolve_latest_seed_dir(
        seed_output_root=seed_output_root,
        graph_run_id=graph_id,
        seed_stage_name_prefix=seed_stage_prefix,
    )
    seed_edges_all_csv = seed_dir / "seed_edges_all.csv"
    if not seed_edges_all_csv.is_file():
        raise FileNotFoundError(f"Missing seed_edges_all.csv: {seed_edges_all_csv}")
    seed_pairs = _load_seed_pairs(seed_edges_all_csv)

    # Output dir.
    out_root = Path(out_cfg.get("output_root") or (project_root / "analysis" / "output" / "anchor_candidates")).expanduser().resolve()
    stage_name = str(out_cfg.get("stage_name") or "candidate_generation")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = out_root / graph_id / f"{stage_name}_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    gen_name = str(candidate_cfg.get("generator") or "rare_artifact_v1").strip().lower()
    if gen_name != "rare_artifact_v1":
        raise ValueError(f"Unsupported candidate generator: {gen_name!r}")
    generator_cfg = candidate_cfg.get("rare_artifact_v1") or candidate_cfg.get("generator_cfg") or {}

    # Ensure overlap columns are present; otherwise candidate might be empty (fine).
    candidates_df, sup = generate_candidates_rare_artifact_v1(
        nodes_df=nodes_df,
        edges_df=edges_df,
        seed_pairs=seed_pairs,
        generator_cfg=generator_cfg,
    )

    p_candidates = out_dir / "candidates_rare_artifact.csv"
    candidates_df.to_csv(p_candidates, index=False)

    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "graph_id": graph_id,
        "anchor_run_dir": str(anchor_run_dir),
        "seed_stage_dir": str(seed_dir),
        "generator": gen_name,
        "n_candidates_rows": int(len(candidates_df)),
        "n_seed_pairs": sup["n_seed_pairs"],
        "n_candidate_pairs": sup["n_candidate_pairs"],
        "n_candidate_pairs_that_are_seed": sup["n_candidate_pairs_that_are_seed"],
        "candidate_universe_superset_check": {
            "n_missing_seed_pairs": sup["n_missing_seed_pairs"],
        },
    }
    p_summary = out_dir / "anchor_candidates_summary.json"
    p_summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    return {
        "output_dir": str(out_dir),
        "candidates_csv": str(p_candidates),
        "summary_json": str(p_summary),
    }

