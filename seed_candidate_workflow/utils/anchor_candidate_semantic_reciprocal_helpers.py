from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from seed_candidate_workflow.utils import graph_structure_helpers as gh
from seed_candidate_workflow.utils.config_run_fields import resolve_graph_id
from seed_candidate_workflow.utils.anchor_graph_helpers import load_anchor_graph_artifacts, load_embedding_vectors

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None


def _resolve_latest_seed_dir(
    *,
    seed_output_root: Path,
    graph_id: str,
    seed_stage_name_prefix: str,
) -> Path:
    base = (seed_output_root / graph_id).expanduser().resolve()
    if not base.is_dir():
        raise FileNotFoundError(f"Seed output root missing: {base}")
    dirs = [d for d in base.iterdir() if d.is_dir() and d.name.startswith(seed_stage_name_prefix)]
    if not dirs:
        raise FileNotFoundError(
            f"No seed generation dirs starting with {seed_stage_name_prefix!r} under {base}"
        )
    dirs_sorted = sorted(dirs, key=lambda p: p.name)
    return dirs_sorted[-1]


def _load_seed_pairs(seed_edges_all_csv: Path) -> set[tuple[str, str]]:
    df = pd.read_csv(seed_edges_all_csv)
    if df.empty:
        return set()
    if not {"email_i", "email_j"}.issubset(df.columns):
        raise ValueError(f"seed_edges_all.csv missing email_i/email_j: {seed_edges_all_csv}")
    pairs = set()
    for a, b in zip(df["email_i"].astype(str).tolist(), df["email_j"].astype(str).tolist(), strict=False):
        i, j = (a, b) if a <= b else (b, a)
        if i == j:
            continue
        pairs.add((i, j))
    return pairs


def _pair_time_gap_seconds(ts_map: dict[str, float], a: str, b: str) -> float:
    ta = ts_map.get(a, float("nan"))
    tb = ts_map.get(b, float("nan"))
    if not np.isfinite(ta) or not np.isfinite(tb):
        return float("nan")
    return float(abs(float(ta) - float(tb)))


@dataclass(frozen=True)
class MutualNeighborInfo:
    rank: int  # 1-based
    cosine: float


def _compute_mutual_topk_cosine_candidates(
    *,
    node_ids: list[str],
    id_to_vec: dict[str, np.ndarray],
    semantic_top_k: int,
    semantic_min_cos: float,
) -> tuple[pd.DataFrame, dict[tuple[str, str], MutualNeighborInfo]]:
    """
    Returns:
      - candidates_df with mutual top-k pairs
      - neighbor_info dict for direction (i,j) where i in node_ids, j in neighbors(i)
        keyed by (i,j) with i and j external_id.
    """
    semantic_node_ids = [eid for eid in node_ids if eid in id_to_vec]
    if len(semantic_node_ids) < 2 or semantic_top_k <= 0:
        empty = pd.DataFrame(
            columns=[
                "email_i",
                "email_j",
                "source",
                "cosine",
                "rank_i_to_j",
                "rank_j_to_i",
                "mutual_topk",
                "time_gap_seconds",
            ]
        )
        return empty, {}

    emb = np.stack([id_to_vec[eid] for eid in semantic_node_ids]).astype(np.float32)
    n = emb.shape[0]
    k_plus = min(int(semantic_top_k) + 1, n)  # +1 to include self

    nn = NearestNeighbors(n_neighbors=k_plus, metric="cosine", algorithm="brute")
    nn.fit(emb)
    dists, neigh = nn.kneighbors(emb, return_distance=True)

    # Build directed neighbor lists (thresholded) with ranks.
    neighbor_info: dict[tuple[str, str], MutualNeighborInfo] = {}
    # Also keep outgoing neighbor sets for mutual computation.
    outgoing: dict[str, list[str]] = {eid: [] for eid in semantic_node_ids}

    thr = float(semantic_min_cos)
    for local_i in range(n):
        i_eid = semantic_node_ids[local_i]
        # Iterate neighbors in returned order: smaller cosine distance => higher cosine similarity.
        rank = 0
        for local_j, dist in zip(neigh[local_i], dists[local_i], strict=False):
            j_eid = semantic_node_ids[int(local_j)]
            if j_eid == i_eid:
                continue
            cs = float(1.0 - float(dist))  # cosine similarity
            if cs < thr:
                continue
            rank += 1
            if rank > int(semantic_top_k):
                break
            outgoing[i_eid].append(j_eid)
            neighbor_info[(i_eid, j_eid)] = MutualNeighborInfo(rank=rank, cosine=cs)

    mutual_pairs: set[tuple[str, str]] = set()
    for i_eid, out_list in outgoing.items():
        for j_eid in out_list:
            if (j_eid, i_eid) in neighbor_info:
                a, b = (i_eid, j_eid) if i_eid <= j_eid else (j_eid, i_eid)
                mutual_pairs.add((a, b))

    # Caller will fill time_gap_seconds; compute later.
    # For cosine/ranks, take from i->j direction (a is <= b lexicographically, but that might not match rank_i_to_j direction).
    # We'll define rank_i_to_j relative to the row's email_i/email_j ordering: i = email_i, j = email_j.
    rows: list[dict[str, Any]] = []
    for a, b in mutual_pairs:
        info_ab = neighbor_info.get((a, b))
        info_ba = neighbor_info.get((b, a))
        if info_ab is None or info_ba is None:
            continue
        rows.append(
            {
                "email_i": a,
                "email_j": b,
                "source": "semantic",
                "cosine": float(info_ab.cosine),
                "rank_i_to_j": int(info_ab.rank),
                "rank_j_to_i": int(info_ba.rank),
                "mutual_topk": True,
                "time_gap_seconds": float("nan"),
            }
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["cosine", "rank_i_to_j", "rank_j_to_i", "email_i", "email_j"], ascending=[False, True, True, True, True]).reset_index(drop=True)
    return df, neighbor_info


def run_anchor_semantic_reciprocal_candidate_generation(config: dict[str, Any]) -> dict[str, Any]:
    run_cfg = config.get("run") or {}
    anchor_cfg = config.get("anchor_output_root") or {}
    candidate_cfg = config.get("candidate_generation") or {}
    output_cfg = config.get("output") or {}
    seed_cfg = config.get("seed") or {}

    project_root = gh.find_project_root()
    graph_id = resolve_graph_id(run_cfg)

    anchor_output_root_raw = str(run_cfg.get("anchor_output_root") or "").strip()
    if anchor_output_root_raw:
        anchor_output_root = Path(anchor_output_root_raw).expanduser().resolve()
        anchor_run_dir = anchor_output_root / graph_id
    else:
        anchor_run_dir = (
            project_root / "seed_candidate_workflow" / "output" / "graph_bundles" / graph_id / "anchor" / graph_id
        ).resolve()
    if not anchor_run_dir.is_dir():
        raise FileNotFoundError(f"Anchor graph run directory not found: {anchor_run_dir}")

    # Load anchor nodes (for timestamps + embedding loading).
    nodes_df, _edges_df, _candidates, _summary, _g = load_anchor_graph_artifacts(
        anchor_run_dir, load_graph_pickle=False
    )
    nodes_df["external_id"] = nodes_df["external_id"].astype(str)

    # Load embed config from anchor graph run config snapshot (so this stage doesn't need paths in its own config).
    p_anchor_run_cfg = anchor_run_dir / "anchor_graph_run_config.json"
    if not p_anchor_run_cfg.is_file():
        raise FileNotFoundError(f"Missing anchor_graph_run_config.json: {p_anchor_run_cfg}")
    run_meta = json.loads(p_anchor_run_cfg.read_text(encoding="utf-8"))
    build_cfg = run_meta.get("config") or {}
    inputs = build_cfg.get("inputs") or {}

    embeddings_json_raw = inputs.get("embeddings_json")
    embeddings_json = Path(str(embeddings_json_raw)).expanduser() if embeddings_json_raw not in (None, "") else None
    if embeddings_json is not None and not embeddings_json.is_absolute():
        embeddings_json = (project_root / embeddings_json).resolve()

    embedding_source = str(inputs.get("embedding_source") or "cache_or_compute")
    prefer_translated_for_compute = bool(inputs.get("prefer_translated_for_compute", True))
    tfidf_max_features = int(inputs.get("tfidf_max_features", 4096))

    node_ids = nodes_df["external_id"].astype(str).tolist()
    id_to_vec, emb_meta = load_embedding_vectors(
        nodes_df=nodes_df,
        embeddings_json=embeddings_json,
        embedding_source=embedding_source,
        prefer_translated_for_compute=prefer_translated_for_compute,
        tfidf_max_features=tfidf_max_features,
    )

    # Candidate parameters.
    semantic_cfg = candidate_cfg.get("semantic_reciprocal_v1") or {}
    semantic_top_k = int(semantic_cfg.get("semantic_top_k", 50))
    semantic_min_cos = float(semantic_cfg.get("semantic_min_cos", 0.9))
    time_gating_enabled = bool(semantic_cfg.get("time_gating_enabled", False))
    max_time_gap_seconds = semantic_cfg.get("max_time_gap_seconds")
    if time_gating_enabled and max_time_gap_seconds is None:
        max_time_gap_seconds = 86400.0

    # Timestamp map.
    ts_map: dict[str, float] = {}
    if "ts" in nodes_df.columns:
        for eid, ts in zip(nodes_df["external_id"].astype(str).tolist(), nodes_df["ts"].tolist(), strict=False):
            v = pd.to_numeric(ts, errors="coerce")
            ts_map[eid] = float(v) if pd.notna(v) else float("nan")

    # Mutual semantic candidates.
    mutual_df, _neighbor_info = _compute_mutual_topk_cosine_candidates(
        node_ids=node_ids,
        id_to_vec=id_to_vec,
        semantic_top_k=semantic_top_k,
        semantic_min_cos=semantic_min_cos,
    )

    # Fill time gaps + optional time gating.
    if not mutual_df.empty:
        mutual_df["time_gap_seconds"] = [
            _pair_time_gap_seconds(ts_map, a, b) for a, b in zip(
                mutual_df["email_i"].astype(str).tolist(),
                mutual_df["email_j"].astype(str).tolist(),
                strict=False,
            )
        ]
        if time_gating_enabled:
            mutual_df = mutual_df[
                pd.to_numeric(mutual_df["time_gap_seconds"], errors="coerce").fillna(float("inf")).le(
                    float(max_time_gap_seconds)
                )
            ].copy()

    # Note: seed backbone inclusion is enforced at the candidate-union level
    # in the unified candidate stage, not in this generator.
    candidates_df = mutual_df
    if not candidates_df.empty:
        candidates_df = candidates_df.drop_duplicates(
            subset=["email_i", "email_j", "source"], keep="first"
        ).reset_index(drop=True)

    out_root_raw = str(output_cfg.get("output_root") or "").strip()
    out_root = (
        Path(out_root_raw).expanduser().resolve()
        if out_root_raw
        else (project_root / "seed_candidate_workflow" / "output" / "graph_bundles" / graph_id / "candidate").resolve()
    )
    stage_name = str(output_cfg.get("stage_name") or "candidate_generation")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = out_root / graph_id / f"{stage_name}_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    p_candidates = out_dir / "candidates_semantic.csv"
    candidates_df.to_csv(p_candidates, index=False)

    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "graph_id": graph_id,
        "anchor_run_dir": str(anchor_run_dir),
        "generator": "semantic_reciprocal_v1",
        "semantic_top_k": semantic_top_k,
        "semantic_min_cos": semantic_min_cos,
        "time_gating_enabled": time_gating_enabled,
        "max_time_gap_seconds": max_time_gap_seconds,
        "n_candidates_rows": int(len(candidates_df)),
        "embedding_meta": emb_meta,
    }
    p_summary = out_dir / "anchor_candidates_summary.json"
    p_summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    return {
        "output_dir": str(out_dir),
        "candidates_csv": str(p_candidates),
        "summary_json": str(p_summary),
    }

