"""
Bridge-candidate retrieval + scoring experiment (analysis-only).

Broad generic retrieval of missing (non-edge) pairs, then trained pair-scorer scoring.
"""

from __future__ import annotations

import json
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None  # type: ignore[assignment,misc]


def _tqdm(it: Iterable[Any], *, desc: str, total: int | None = None, **kw: Any) -> Any:
    if tqdm is None:
        return it
    return tqdm(it, desc=desc, total=total, **kw)


RETRIEVAL_CHANNELS: tuple[str, ...] = (
    "semantic",
    "body_only",
    "path",
    "sender_localpart",
    "html_fp",
    "registrable_domain",
)


def canonical_pair(email_a: str, email_b: str) -> tuple[str, str] | None:
    a, b = str(email_a).strip(), str(email_b).strip()
    if not a or not b or a == b:
        return None
    return (a, b) if a <= b else (b, a)


def load_connected_pair_keys(
    *,
    candidate_union_csv: Path | None,
    seed_edges_csv: Path | None,
    extra_edge_csvs: list[Path] | None = None,
) -> set[tuple[str, str]]:
    """All unordered pairs already present in the candidate / seed graph."""
    connected: set[tuple[str, str]] = set()
    paths = list(extra_edge_csvs or [])
    if candidate_union_csv is not None:
        paths.append(candidate_union_csv)
    if seed_edges_csv is not None:
        paths.append(seed_edges_csv)

    for p in paths:
        p = Path(p).resolve()
        if not p.is_file():
            continue
        df = pd.read_csv(p, low_memory=False)
        if not {"email_i", "email_j"}.issubset(df.columns):
            continue
        for a, b in zip(df["email_i"].astype(str), df["email_j"].astype(str), strict=False):
            pk = canonical_pair(a, b)
            if pk is not None:
                connected.add(pk)
    return connected


def load_email_universe_from_meta(meta_json: Path) -> list[str]:
    from seed_candidate_workflow.utils import graph_structure_helpers as gh

    meta = gh.load_meta(meta_json)
    ext = (meta.get("email_attrs") or {}).get("external_id")
    if not isinstance(ext, list):
        return []
    return [str(x).strip() for x in ext if str(x).strip()]


def _load_embeddings_by_external_id(embeddings_json: Path) -> dict[str, np.ndarray]:
    with open(embeddings_json, encoding="utf-8") as f:
        payload = json.load(f)
    by_key = payload.get("by_key")
    if not isinstance(by_key, dict):
        return {}
    out: dict[str, np.ndarray] = {}
    for k, v in by_key.items():
        if not isinstance(v, dict):
            continue
        subj = np.asarray(v.get("subj") or [], dtype=np.float64).reshape(-1)
        body = np.asarray(v.get("body") or [], dtype=np.float64).reshape(-1)
        if subj.size == 0 and body.size == 0:
            continue
        eid = str(v.get("external_id") or k).strip()
        if eid:
            out[eid] = np.concatenate([subj, body], axis=0)
    return out


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na <= 0 or nb <= 0:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def _minhash_signature(token_set: frozenset[str], num_perm: int = 64) -> tuple[int, ...]:
    if not token_set:
        return tuple(0 for _ in range(num_perm))
    return tuple(
        min(hash((seed, tok)) & 0xFFFFFFFFFFFFFFFF for tok in token_set) for seed in range(num_perm)
    )


def _lsh_retrieve_topk(
    *,
    email_ids: list[str],
    id_to_tokens: dict[str, frozenset[str]],
    top_k: int,
    connected: set[tuple[str, str]],
    channel: str,
    num_perm: int = 64,
    bands: int = 16,
    desc: str | None = None,
) -> dict[tuple[str, str], dict[str, Any]]:
    """
    MinHash LSH retrieval: for each email, propose up to top_k missing neighbors.
    Returns pair_key -> {channels, body_only_jaccard_est, ...}
    """
    rows_per_band = max(1, num_perm // bands)
    n = len(email_ids)
    if n < 2 or top_k <= 0:
        return {}

    local_to_eid = list(email_ids)
    eid_to_local = {e: i for i, e in enumerate(local_to_eid)}
    sigs = [_minhash_signature(id_to_tokens.get(e, frozenset()), num_perm) for e in local_to_eid]

    buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
    for li, sig in enumerate(sigs):
        for b in range(bands):
            start = b * rows_per_band
            band_sig = sig[start : start + rows_per_band]
            buckets[(b, hash(band_sig))].append(li)

    hits: dict[tuple[str, str], dict[str, Any]] = {}
    progress_desc = desc or f"{channel} retrieval"
    iterator = _tqdm(range(n), desc=progress_desc, total=n)
    for li in iterator:
        ei = local_to_eid[li]
        cand_scores: Counter[int] = Counter()
        for b in range(bands):
            start = b * rows_per_band
            band_sig = sigs[li][start : start + rows_per_band]
            for lj in buckets.get((b, hash(band_sig)), []):
                if lj != li:
                    cand_scores[lj] += 1

        if not cand_scores:
            continue
        tok_i = id_to_tokens.get(ei) or frozenset()
        ranked: list[tuple[int, float]] = []
        for lj, band_hits in cand_scores.most_common(min(top_k * 8, len(cand_scores))):
            ej = local_to_eid[lj]
            pk = canonical_pair(ei, ej)
            if pk is None or pk in connected:
                continue
            tok_j = id_to_tokens.get(ej) or frozenset()
            if not tok_i and not tok_j:
                jac = 0.0
            else:
                inter = len(tok_i & tok_j)
                union = len(tok_i | tok_j)
                jac = float(inter / union) if union else 0.0
            ranked.append((lj, jac + 0.01 * band_hits))
        ranked.sort(key=lambda x: x[1], reverse=True)
        seen_local: set[int] = set()
        added = 0
        for lj, score in ranked:
            if lj in seen_local:
                continue
            seen_local.add(lj)
            ej = local_to_eid[lj]
            pk = canonical_pair(ei, ej)
            if pk is None or pk in connected:
                continue
            rec = hits.setdefault(
                pk,
                {
                    "retrieval_channels": set(),
                    "retrieval_semantic_cosine": None,
                    "retrieval_body_only_token_jaccard": None,
                    "retrieval_path_token_jaccard": None,
                    "retrieval_semantic_rank": None,
                    "retrieval_body_only_rank": None,
                    "retrieval_path_rank": None,
                },
            )
            rec["retrieval_channels"].add(channel)
            if channel == "body_only":
                rec["retrieval_body_only_token_jaccard"] = max(
                    rec.get("retrieval_body_only_token_jaccard") or 0.0, score
                )
                prev_r = rec.get("retrieval_body_only_rank")
                rec["retrieval_body_only_rank"] = added if prev_r is None else min(int(prev_r), added)
            elif channel == "path":
                rec["retrieval_path_token_jaccard"] = max(
                    rec.get("retrieval_path_token_jaccard") or 0.0, score
                )
                prev_r = rec.get("retrieval_path_rank")
                rec["retrieval_path_rank"] = added if prev_r is None else min(int(prev_r), added)
            added += 1
            if added >= top_k:
                break
    return hits


def retrieve_semantic_missing_pairs(
    *,
    email_ids: list[str],
    id_to_emb: dict[str, np.ndarray],
    top_k: int,
    connected: set[tuple[str, str]],
) -> dict[tuple[str, str], dict[str, Any]]:
    from sklearn.neighbors import NearestNeighbors

    semantic_ids = [e for e in email_ids if e in id_to_emb]
    if len(semantic_ids) < 2 or top_k <= 0:
        return {}

    emb = np.stack([id_to_emb[e] for e in semantic_ids]).astype(np.float32)
    n = emb.shape[0]
    k_query = min(int(top_k) + 1, n)
    nn = NearestNeighbors(n_neighbors=k_query, metric="cosine", algorithm="brute")
    nn.fit(emb)
    dists, neigh = nn.kneighbors(emb, return_distance=True)

    hits: dict[tuple[str, str], dict[str, Any]] = {}
    for i in _tqdm(range(n), desc="semantic retrieval", total=n):
        ei = semantic_ids[i]
        rank = 0
        for j_idx, dist in zip(neigh[i], dists[i], strict=False):
            ej = semantic_ids[int(j_idx)]
            if ei == ej:
                continue
            rank += 1
            if rank > top_k:
                break
            cos = 1.0 - float(dist)
            pk = canonical_pair(ei, ej)
            if pk is None or pk in connected:
                continue
            rec = hits.setdefault(
                pk,
                {
                    "retrieval_channels": set(),
                    "retrieval_semantic_cosine": None,
                    "retrieval_body_only_token_jaccard": None,
                    "retrieval_path_token_jaccard": None,
                    "retrieval_semantic_rank": None,
                    "retrieval_body_only_rank": None,
                    "retrieval_path_rank": None,
                },
            )
            rec["retrieval_channels"].add("semantic")
            prev = rec.get("retrieval_semantic_cosine")
            rec["retrieval_semantic_cosine"] = cos if prev is None else max(float(prev), cos)
            prev_r = rec.get("retrieval_semantic_rank")
            rec["retrieval_semantic_rank"] = rank if prev_r is None else min(int(prev_r), rank)
    return hits


def _blocking_retrieval(
    *,
    email_ids: list[str],
    id_to_keys: dict[str, list[str]],
    channel: str,
    top_k: int,
    connected: set[tuple[str, str]],
) -> dict[tuple[str, str], dict[str, Any]]:
    """Broad blocking: same key -> all pairs (capped per email)."""
    key_to_emails: dict[str, list[str]] = defaultdict(list)
    for e in email_ids:
        for k in id_to_keys.get(e) or []:
            if k:
                key_to_emails[k].append(e)

    hits: dict[tuple[str, str], dict[str, Any]] = {}
    for e in _tqdm(email_ids, desc=f"{channel} retrieval", total=len(email_ids)):
        cand_scores: Counter[str] = Counter()
        for k in id_to_keys.get(e) or []:
            for other in key_to_emails.get(k, []):
                if other != e:
                    cand_scores[other] += 1
        if not cand_scores:
            continue
        for ej, _cnt in cand_scores.most_common(top_k):
            pk = canonical_pair(e, ej)
            if pk is None or pk in connected:
                continue
            rec = hits.setdefault(
                pk,
                {
                    "retrieval_channels": set(),
                    "retrieval_semantic_cosine": None,
                    "retrieval_body_only_token_jaccard": None,
                    "retrieval_path_token_jaccard": None,
                },
            )
            rec["retrieval_channels"].add(channel)
    return hits


def _html_fingerprint_blocking_keys(
    *,
    email_universe: list[str],
    candidate_union_csv: Path | None,
    graph_id: str | None,
    project_root: Path,
) -> dict[str, list[str]]:
    from seed_candidate_workflow.utils.pair_training_dataset_helpers import _infer_anchor_run_dir
    from seed_candidate_workflow.utils.anchor_graph_helpers import load_anchor_graph_artifacts

    if candidate_union_csv is None:
        return {}
    run_dir = _infer_anchor_run_dir(
        candidate_union_csv=candidate_union_csv,
        graph_id=graph_id,
        project_root=project_root,
    )
    if run_dir is None:
        return {}
    nodes_df, _, _, _, _ = load_anchor_graph_artifacts(run_dir, load_graph_pickle=False)
    if "external_id" not in nodes_df.columns or "html_structure_fingerprint_set" not in nodes_df.columns:
        return {}

    def _to_set_cell(v: Any) -> set[str]:
        if isinstance(v, set):
            return {str(x) for x in v if str(x).strip()}
        if isinstance(v, list):
            return {str(x) for x in v if str(x).strip()}
        if isinstance(v, str) and v.strip():
            if v.startswith("[") and v.endswith("]"):
                try:
                    xs = json.loads(v)
                    if isinstance(xs, list):
                        return {str(x) for x in xs if str(x).strip()}
                except Exception:
                    pass
            if "|" in v:
                return {p.strip() for p in v.split("|") if p.strip()}
        return set()

    out: dict[str, list[str]] = {}
    universe = set(email_universe)
    for _, r in nodes_df[["external_id", "html_structure_fingerprint_set"]].iterrows():
        eid = str(r["external_id"])
        if eid not in universe:
            continue
        fps = _to_set_cell(r.get("html_structure_fingerprint_set"))
        if fps:
            out[eid] = [f"fp:{x}" for x in fps]
    return out


def merge_retrieval_hits(
    *hit_maps: dict[tuple[str, str], dict[str, Any]],
) -> dict[tuple[str, str], dict[str, Any]]:
    merged: dict[tuple[str, str], dict[str, Any]] = {}
    for hm in hit_maps:
        for pk, rec in hm.items():
            if pk not in merged:
                merged[pk] = {
                    "retrieval_channels": set(),
                    "retrieval_semantic_cosine": None,
                    "retrieval_body_only_token_jaccard": None,
                    "retrieval_path_token_jaccard": None,
                    "retrieval_semantic_rank": None,
                    "retrieval_body_only_rank": None,
                    "retrieval_path_rank": None,
                }
            m = merged[pk]
            m["retrieval_channels"] |= set(rec.get("retrieval_channels") or set())
            for col in (
                "retrieval_semantic_cosine",
                "retrieval_body_only_token_jaccard",
                "retrieval_path_token_jaccard",
            ):
                v = rec.get(col)
                if v is not None and np.isfinite(v):
                    prev = m.get(col)
                    m[col] = float(v) if prev is None else max(float(prev), float(v))
            for col in (
                "retrieval_semantic_rank",
                "retrieval_body_only_rank",
                "retrieval_path_rank",
            ):
                v = rec.get(col)
                if v is not None:
                    prev = m.get(col)
                    m[col] = int(v) if prev is None else min(int(prev), int(v))
    return merged


def retrieval_hits_to_dataframe(
    hits: dict[tuple[str, str], dict[str, Any]],
    *,
    max_candidates: int | None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (ei, ej), rec in hits.items():
        chans = sorted(rec.get("retrieval_channels") or [])
        rows.append(
            {
                "email_i": ei,
                "email_j": ej,
                "retrieval_channels": "|".join(chans),
                "n_retrieval_channels": len(chans),
                "retrieval_semantic_cosine": rec.get("retrieval_semantic_cosine"),
                "retrieval_body_only_token_jaccard": rec.get("retrieval_body_only_token_jaccard"),
                "retrieval_path_token_jaccard": rec.get("retrieval_path_token_jaccard"),
                "retrieval_semantic_rank": rec.get("retrieval_semantic_rank"),
                "retrieval_body_only_rank": rec.get("retrieval_body_only_rank"),
                "retrieval_path_rank": rec.get("retrieval_path_rank"),
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.sort_values(
        ["n_retrieval_channels", "retrieval_semantic_cosine"],
        ascending=[False, False],
        na_position="last",
    ).reset_index(drop=True)
    if max_candidates is not None and len(df) > int(max_candidates):
        df = df.head(int(max_candidates)).copy()
    return df


@dataclass
class BridgeCandidateConfig:
    run_dir: Path
    graph_pt: Path
    output_subdir: str = "bridge_candidate_experiment"
    candidate_union_csv: Path | None = None
    seed_edges_csv: Path | None = None
    pair_csv: Path | None = None
    graph_meta_json: Path | None = None
    embeddings_json: Path | None = None
    gt_path: Path | None = None
    semantic_top_k_missing: int = 50
    body_only_top_k_missing: int = 50
    path_top_k_missing: int = 50
    enable_sender_localpart: bool = True
    enable_html_fp: bool = True
    enable_registrable_domain: bool = True
    max_bridge_candidates: int = 500_000
    score_thresholds: tuple[float, ...] = (0.8, 0.9)
    score_batch_size: int = 256
    skip_gt_diagnostics: bool = False
    device: str = "cpu"
    checkpoint_name: str = "best_model.pt"
    to_undirected: bool = True
    medium_score_review_min: float = 0.5
    medium_score_review_max: float = 0.8
    low_score_review_max: float = 0.2
    high_score_review_max_rows: int = 500
    mid_score_review_max_rows: int = 300
    low_score_review_max_rows: int = 200
    review_gnn_latent_max_rows: int = 5000
    skip_review_enrichment: bool = False


def _resolve_embeddings_json(project_root: Path) -> Path | None:
    p = (project_root / "core" / "utils" / "embeddings" / "output" / "embeddings.json").resolve()
    return p if p.is_file() else None


def _attach_graph_indices(df: pd.DataFrame, meta_json: Path) -> pd.DataFrame:
    from seed_candidate_workflow.utils import graph_structure_helpers as gh

    meta = gh.load_meta(meta_json)
    ext_to_idx = gh.external_id_to_row(meta)
    out = df.copy()
    out["graph_email_idx_i"] = out["email_i"].astype(str).map(ext_to_idx)
    out["graph_email_idx_j"] = out["email_j"].astype(str).map(ext_to_idx)
    out["graph_email_idx_i"] = pd.to_numeric(out["graph_email_idx_i"], errors="coerce").astype("Int64")
    out["graph_email_idx_j"] = pd.to_numeric(out["graph_email_idx_j"], errors="coerce").astype("Int64")
    out["pair_status"] = "unlabeled"
    return out


def _enrich_for_scorer(
    df: pd.DataFrame,
    *,
    project_root: Path,
    candidate_union_csv: Path | None,
    graph_id: str | None,
    nodes_by_email: dict[str, dict[str, Any]],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    from seed_candidate_workflow.utils.pair_similarity_features import (
        ensure_pair_scorer_similarity_features_in_dataframe,
    )
    from seed_candidate_workflow.utils.pair_training_dataset_helpers import (
        _add_shared_attribute_pair_features,
    )

    out = df.copy()
    for col in (
        "from_seed",
        "from_semantic",
        "from_rare_artifact",
        "from_component",
        "from_2hop",
        "same_seed_component_flag",
        "cross_seed_component_flag",
    ):
        if col not in out.columns:
            out[col] = False
    if "source_count" not in out.columns:
        out["source_count"] = 0

    if nodes_by_email:
        out = _add_shared_attribute_pair_features(df=out, nodes_by_email=nodes_by_email)

    out, enrich_meta = ensure_pair_scorer_similarity_features_in_dataframe(
        out,
        csv_path=candidate_union_csv,
        project_root=project_root,
        graph_id=graph_id,
        nodes_by_email=nodes_by_email,
        force_recompute=True,
    )
    return out, enrich_meta


def score_bridge_dataframe(
    df: pd.DataFrame,
    *,
    run_dir: Path,
    graph_pt: Path,
    device: str,
    checkpoint_name: str,
    to_undirected: bool,
    score_batch_size: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    from seed_candidate_workflow.utils.pair_model_inference import (
        load_pair_supervision_for_inference,
        score_pair_rows,
    )

    gi = pd.to_numeric(df["graph_email_idx_i"], errors="coerce")
    gj = pd.to_numeric(df["graph_email_idx_j"], errors="coerce")
    ok = gi.notna() & gj.notna()
    work = df.loc[ok].copy()
    n_skip = int((~ok).sum())
    if work.empty:
        out = df.copy()
        out["score"] = np.nan
        return out, {"n_scored": 0, "n_skipped_no_graph_index": n_skip}

    work = work.reset_index(drop=True)
    work["_row"] = np.arange(len(work), dtype=np.int64)
    n_work = len(work)

    ctx = load_pair_supervision_for_inference(
        run_dir=run_dir,
        graph_pt=graph_pt,
        checkpoint_name=checkpoint_name,
        device=device,
        to_undirected=to_undirected,
    )
    scores = np.full(n_work, np.nan, dtype=np.float64)
    bs = max(1, int(score_batch_size))
    n_batches = (n_work + bs - 1) // bs
    for start in _tqdm(range(0, n_work, bs), desc="bridge scoring batches", total=n_batches):
        end = min(start + bs, n_work)
        chunk = work.iloc[start:end].copy()
        chunk["_row"] = np.arange(len(chunk), dtype=np.int64)
        chunk_scores = score_pair_rows(
            model=ctx["model"],
            pair_scorer=ctx["pair_scorer"],
            data_cpu=ctx["data_cpu"],
            df_work=chunk,
            device=ctx["device"],
            fanout=ctx["fanout"],
            pair_batch_size=min(bs, int(ctx["pair_batch_size"])),
            max_unique_emails=int(ctx["max_unique_emails"]),
            pair_feature_columns=ctx.get("pair_feature_columns"),
        )
        scores[start:end] = np.asarray(chunk_scores, dtype=np.float64)

    out = df.copy()
    out["score"] = np.nan
    pair_key = out["email_i"].astype(str) + "\0" + out["email_j"].astype(str)
    work_key = work["email_i"].astype(str) + "\0" + work["email_j"].astype(str)
    score_by_key = dict(zip(work_key.tolist(), scores.tolist(), strict=False))
    out["score"] = pair_key.map(score_by_key)
    n_scored = int(pd.to_numeric(out["score"], errors="coerce").notna().sum())
    return out, {
        "n_scored": n_scored,
        "n_skipped_no_graph_index": n_skip,
        "checkpoint_path": ctx.get("checkpoint_path"),
    }


def _gt_labels_for_pairs(
    df: pd.DataFrame,
    label_map: dict[str, Any],
) -> pd.Series:
    rel: list[str | None] = []
    for _, r in df.iterrows():
        ci = label_map.get(str(r["email_i"]))
        cj = label_map.get(str(r["email_j"]))
        if ci is None or cj is None:
            rel.append(None)
        elif ci == cj:
            rel.append("same_campaign")
        else:
            rel.append("cross_campaign")
    return pd.Series(rel, index=df.index, dtype=object)


def build_gt_bridge_diagnostics(
    df: pd.DataFrame,
    *,
    gt_path: Path,
    label_map: dict[str, Any],
) -> dict[str, Any]:
    sub = df.copy()
    sub["gt_relation"] = _gt_labels_for_pairs(sub, label_map)
    covered = sub["gt_relation"].notna()
    n_cov = int(covered.sum())
    same = int((sub.loc[covered, "gt_relation"] == "same_campaign").sum())
    cross = int((sub.loc[covered, "gt_relation"] == "cross_campaign").sum())
    scores = pd.to_numeric(sub.loc[covered, "score"], errors="coerce")

    def _frac_high(thr: float, relation: str) -> float | None:
        m = covered & (sub["gt_relation"] == relation)
        if not m.any():
            return None
        s = pd.to_numeric(sub.loc[m, "score"], errors="coerce")
        return float((s >= thr).mean()) if s.notna().any() else None

    return {
        "gt_path": str(gt_path.resolve()),
        "n_bridge_candidates_gt_covered": n_cov,
        "n_gt_same_campaign": same,
        "n_gt_cross_campaign": cross,
        "gt_same_fraction": float(same / n_cov) if n_cov else None,
        "score_mean_gt_covered": float(scores.mean()) if scores.notna().any() else None,
        "fraction_ge_0_80_same": _frac_high(0.8, "same_campaign"),
        "fraction_ge_0_80_cross": _frac_high(0.8, "cross_campaign"),
        "fraction_ge_0_90_same": _frac_high(0.9, "same_campaign"),
        "fraction_ge_0_90_cross": _frac_high(0.9, "cross_campaign"),
        "note": "Diagnostic only — GT was not used for retrieval.",
    }


def _write_score_plots(
    df: pd.DataFrame,
    plots_dir: Path,
    *,
    gt_diag: dict[str, Any] | None,
) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    scores = pd.to_numeric(df["score"], errors="coerce").dropna()
    if scores.empty:
        return written

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(scores, bins=50, color="#4c72b0", edgecolor="white")
    ax.set_xlabel("Bridge candidate score")
    ax.set_ylabel("Count")
    ax.set_title("All bridge candidate scores")
    ax.grid(True, alpha=0.3)
    p0 = plots_dir / "bridge_candidate_score_histogram.png"
    fig.tight_layout()
    fig.savefig(p0, dpi=120)
    plt.close(fig)
    written.append(str(p0.name))

    if "retrieval_channels" in df.columns and df["retrieval_channels"].notna().any():
        top_ch = df["retrieval_channels"].astype(str).value_counts().head(12).index.tolist()
        fig, ax = plt.subplots(figsize=(9, 5))
        for ch in top_ch:
            sub = df.loc[df["retrieval_channels"].astype(str).str.contains(ch, regex=False)]
            s = pd.to_numeric(sub["score"], errors="coerce").dropna()
            if len(s) > 5:
                ax.hist(s, bins=40, alpha=0.45, label=ch[:40], density=True)
        ax.set_xlabel("Score")
        ax.set_title("Score distribution by retrieval channel (subset)")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        p1 = plots_dir / "bridge_candidate_score_by_channel.png"
        fig.tight_layout()
        fig.savefig(p1, dpi=120)
        plt.close(fig)
        written.append(str(p1.name))

    if "n_retrieval_channels" in df.columns:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for nch in sorted(df["n_retrieval_channels"].dropna().unique()):
            sub = df.loc[df["n_retrieval_channels"] == nch]
            s = pd.to_numeric(sub["score"], errors="coerce").dropna()
            if len(s) > 3:
                ax.hist(s, bins=35, alpha=0.5, label=f"{int(nch)} channels", density=True)
        ax.set_xlabel("Score")
        ax.set_title("Score vs number of retrieval channels")
        ax.legend()
        ax.grid(True, alpha=0.3)
        p2 = plots_dir / "bridge_candidate_score_by_n_channels.png"
        fig.tight_layout()
        fig.savefig(p2, dpi=120)
        plt.close(fig)
        written.append(str(p2.name))

    if gt_diag and "gt_relation" in df.columns:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for rel, color in (("same_campaign", "#ff7f0e"), ("cross_campaign", "#1f77b4")):
            s = pd.to_numeric(df.loc[df["gt_relation"] == rel, "score"], errors="coerce").dropna()
            if len(s) > 3:
                ax.hist(s, bins=35, alpha=0.55, label=rel, color=color, density=True)
        ax.set_xlabel("Score")
        ax.set_title("GT-covered bridge candidates by relation")
        ax.legend()
        ax.grid(True, alpha=0.3)
        p3 = plots_dir / "bridge_candidate_score_gt_same_vs_cross.png"
        fig.tight_layout()
        fig.savefig(p3, dpi=120)
        plt.close(fig)
        written.append(str(p3.name))

    return written


def run_bridge_candidate_experiment(cfg: BridgeCandidateConfig) -> dict[str, Any]:
    """Full pipeline: retrieve missing pairs, score, export artifacts."""
    t0 = time.perf_counter()
    timing: dict[str, float] = {}

    project_root = Path(__file__).resolve().parents[2]
    run_dir = Path(cfg.run_dir).resolve()
    graph_pt = Path(cfg.graph_pt).resolve()
    out_root = (run_dir / cfg.output_subdir).resolve()
    plots_dir = out_root / "plots"
    debug_csv = out_root / "debug_csv"
    debug_json = out_root / "debug_json"
    for d in (out_root, plots_dir, debug_csv, debug_json):
        d.mkdir(parents=True, exist_ok=True)

    from seed_candidate_workflow.utils.pair_model_inference import resolve_pair_dataset_csv_path

    pair_csv = (
        Path(cfg.pair_csv).resolve()
        if cfg.pair_csv
        else resolve_pair_dataset_csv_path(run_dir, project_root=project_root)
    )
    graph_id = run_dir.name
    cand_csv = cfg.candidate_union_csv
    if cand_csv is None:
        hint = (
            project_root
            / "seed_candidate_workflow"
            / "output"
            / "graph_bundles"
            / graph_id
            / "candidate"
            / graph_id
            / "candidate_union.csv"
        )
        cand_csv = hint if hint.is_file() else None
    seed_csv = cfg.seed_edges_csv
    if seed_csv is None and cand_csv is not None:
        seed_hint = cand_csv.parent.parent.parent / "seed" / graph_id / "seed_edges_all.csv"
        seed_csv = seed_hint if seed_hint.is_file() else None

    meta_json = cfg.graph_meta_json
    if meta_json is None:
        meta_hint = graph_pt.with_suffix(".meta.json")
        meta_json = meta_hint if meta_hint.is_file() else None
    if meta_json is None or not Path(meta_json).is_file():
        raise FileNotFoundError(f"graph meta JSON required (e.g. {graph_pt.with_suffix('.meta.json')})")

    meta_json = Path(meta_json).resolve()
    t_retrieval = time.perf_counter()
    connected = load_connected_pair_keys(
        candidate_union_csv=cand_csv,
        seed_edges_csv=seed_csv,
        extra_edge_csvs=[pair_csv],
    )
    email_universe = load_email_universe_from_meta(meta_json)
    timing["load_connected_pairs_sec"] = time.perf_counter() - t_retrieval

    emb_path = cfg.embeddings_json or _resolve_embeddings_json(project_root)
    id_to_emb = _load_embeddings_by_external_id(emb_path) if emb_path and emb_path.is_file() else {}

    from seed_candidate_workflow.utils.pair_training_dataset_helpers import (
        _load_anchor_node_sets_by_email,
    )
    from seed_candidate_workflow.utils.pair_similarity_features import path_token_set_for_node

    nodes_by_email, nodes_meta = (
        _load_anchor_node_sets_by_email(
            candidate_union_csv=cand_csv or pair_csv,
            graph_id=graph_id,
            project_root=project_root,
        )
        if cand_csv is not None
        else ({}, {"available": False})
    )

    hit_maps: list[dict[tuple[str, str], dict[str, Any]]] = []
    channel_counts: dict[str, int] = {}

    if id_to_emb:
        sem_hits = retrieve_semantic_missing_pairs(
            email_ids=email_universe,
            id_to_emb=id_to_emb,
            top_k=int(cfg.semantic_top_k_missing),
            connected=connected,
        )
        hit_maps.append(sem_hits)
        channel_counts["semantic"] = len(sem_hits)

    body_store = None
    try:
        from seed_candidate_workflow.utils.body_similarity_cache import (
            build_or_load_email_body_feature_store,
        )
        from seed_candidate_workflow.utils.pair_similarity_features import (
            load_misp_text_catalog_for_pairs,
        )
        from seed_candidate_workflow.utils.pair_score_separation import _resolve_default_misp_json_path

        catalog, _ = load_misp_text_catalog_for_pairs(project_root=project_root)
        misp_path = _resolve_default_misp_json_path(project_root)
        if catalog and misp_path.is_file():
            body_store, _ = build_or_load_email_body_feature_store(
                email_ids=email_universe,
                text_catalog=catalog,
                graph_id=graph_id,
                misp_json_path=misp_path,
            )
    except Exception:
        body_store = None

    if body_store is not None:
        id_to_body_tok = {
            e: body_store.token_sets.get(e) or frozenset() for e in email_universe if e in body_store.token_sets
        }
        body_hits = _lsh_retrieve_topk(
            email_ids=[e for e in email_universe if e in id_to_body_tok],
            id_to_tokens=id_to_body_tok,
            top_k=int(cfg.body_only_top_k_missing),
            connected=connected,
            channel="body_only",
        )
        hit_maps.append(body_hits)
        channel_counts["body_only"] = len(body_hits)

    if nodes_by_email:
        id_to_path = {e: frozenset(path_token_set_for_node(nodes_by_email[e])) for e in email_universe if e in nodes_by_email}
        path_hits = _lsh_retrieve_topk(
            email_ids=list(id_to_path.keys()),
            id_to_tokens=id_to_path,
            top_k=int(cfg.path_top_k_missing),
            connected=connected,
            channel="path",
        )
        hit_maps.append(path_hits)
        channel_counts["path"] = len(path_hits)

        if cfg.enable_sender_localpart:

            def _sender_keys(eid: str) -> list[str]:
                keys: list[str] = []
                for s in nodes_by_email.get(eid, {}).get("sender_set") or set():
                    s = str(s)
                    if "@" in s:
                        lp = s.split("@", 1)[0].strip().lower()
                        if lp:
                            keys.append(f"lp:{lp}")
                return keys

            id_to_lp = {e: _sender_keys(e) for e in email_universe if _sender_keys(e)}
            lp_hits = _blocking_retrieval(
                email_ids=list(id_to_lp.keys()),
                id_to_keys=id_to_lp,
                channel="sender_localpart",
                top_k=int(cfg.path_top_k_missing),
                connected=connected,
            )
            hit_maps.append(lp_hits)
            channel_counts["sender_localpart"] = len(lp_hits)

        if cfg.enable_html_fp:
            id_to_fp = _html_fingerprint_blocking_keys(
                email_universe=email_universe,
                candidate_union_csv=cand_csv,
                graph_id=graph_id,
                project_root=project_root,
            )
            fp_hits = _blocking_retrieval(
                email_ids=list(id_to_fp.keys()),
                id_to_keys=id_to_fp,
                channel="html_fp",
                top_k=min(30, int(cfg.path_top_k_missing)),
                connected=connected,
            )
            hit_maps.append(fp_hits)
            channel_counts["html_fp"] = len(fp_hits)

        if cfg.enable_registrable_domain:

            def _dom_keys(eid: str) -> list[str]:
                doms = nodes_by_email.get(eid, {}).get("domain_set") or set()
                return [f"dom:{d}" for d in doms if str(d).strip()]

            id_to_dom = {e: _dom_keys(e) for e in email_universe if _dom_keys(e)}
            dom_hits = _blocking_retrieval(
                email_ids=list(id_to_dom.keys()),
                id_to_keys=id_to_dom,
                channel="registrable_domain",
                top_k=min(40, int(cfg.path_top_k_missing)),
                connected=connected,
            )
            hit_maps.append(dom_hits)
            channel_counts["registrable_domain"] = len(dom_hits)

    timing["retrieval_sec"] = time.perf_counter() - t_retrieval

    t_dedup = time.perf_counter()
    merged = merge_retrieval_hits(*hit_maps)
    df_bridge = retrieval_hits_to_dataframe(
        merged, max_candidates=int(cfg.max_bridge_candidates) if cfg.max_bridge_candidates else None
    )
    timing["dedup_sec"] = time.perf_counter() - t_dedup

    t_score = time.perf_counter()
    df_bridge = _attach_graph_indices(df_bridge, meta_json)
    df_bridge, enrich_meta = _enrich_for_scorer(
        df_bridge,
        project_root=project_root,
        candidate_union_csv=cand_csv,
        graph_id=graph_id,
        nodes_by_email=nodes_by_email,
    )
    df_bridge, score_meta = score_bridge_dataframe(
        df_bridge,
        run_dir=run_dir,
        graph_pt=graph_pt,
        device=cfg.device,
        checkpoint_name=cfg.checkpoint_name,
        to_undirected=cfg.to_undirected,
        score_batch_size=cfg.score_batch_size,
    )
    timing["scoring_sec"] = time.perf_counter() - t_score

    label_map: dict[str, Any] | None = None
    gt_diag: dict[str, Any] | None = None
    if cfg.gt_path and not cfg.skip_gt_diagnostics:
        from seed_candidate_workflow.utils.raw_gnn_notebook import load_ground_truth_structures

        label_map, _, _ = load_ground_truth_structures(Path(cfg.gt_path))
        label_map = {str(k): v for k, v in label_map.items()}
        df_bridge["gt_relation"] = _gt_labels_for_pairs(df_bridge, label_map)
        gt_diag = build_gt_bridge_diagnostics(df_bridge, gt_path=Path(cfg.gt_path), label_map=label_map)

    review_meta: dict[str, Any] = {}
    pop_diag: dict[str, Any] = {}
    band_analysis: dict[str, Any] = {}
    trust_rec: dict[str, Any] = {}
    review_export: dict[str, Any] = {}
    if not cfg.skip_review_enrichment:
        from seed_candidate_workflow.utils.bridge_candidate_review import (
            enrich_bridge_dataframe_for_review,
            export_bridge_review_artifacts,
            _attach_gt_campaign_columns,
        )
        from seed_candidate_workflow.utils.pair_score_separation import _resolve_default_misp_json_path

        try:
            misp_for_nodes = _resolve_default_misp_json_path(project_root)
        except Exception:
            misp_for_nodes = None

        df_bridge, review_meta = enrich_bridge_dataframe_for_review(
            df_bridge,
            project_root=project_root,
            run_dir=run_dir,
            graph_pt=graph_pt,
            connected=connected,
            candidate_union_csv=cand_csv,
            pair_csv=pair_csv,
            to_undirected=cfg.to_undirected,
            compute_gnn_latent_max_rows=int(cfg.review_gnn_latent_max_rows),
            device=cfg.device,
            checkpoint_name=cfg.checkpoint_name,
            misp_json_path=misp_for_nodes,
        )
        if label_map:
            df_bridge = _attach_gt_campaign_columns(df_bridge, label_map=label_map)

    t_export = time.perf_counter()
    scores = pd.to_numeric(df_bridge["score"], errors="coerce")
    summary: dict[str, Any] = {
        "experiment": "bridge_candidate_retrieval_scoring",
        "run_dir": str(run_dir),
        "graph_pt": str(graph_pt),
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(cfg).items()},
        "n_emails_in_universe": len(email_universe),
        "n_connected_pairs_excluded": len(connected),
        "n_bridge_candidates_after_dedup": int(len(df_bridge)),
        "retrieval_channel_pair_counts": channel_counts,
        "n_retrieval_channels_histogram": (
            df_bridge["n_retrieval_channels"].value_counts().sort_index().to_dict()
            if not df_bridge.empty and "n_retrieval_channels" in df_bridge.columns
            else {}
        ),
        "score_distribution": {
            "n_scored": int(scores.notna().sum()),
            "mean": float(scores.mean()) if scores.notna().any() else None,
            "median": float(scores.median()) if scores.notna().any() else None,
            "p90": float(scores.quantile(0.9)) if scores.notna().any() else None,
            "p95": float(scores.quantile(0.95)) if scores.notna().any() else None,
        },
        "score_threshold_counts": {
            str(thr): int((scores >= float(thr)).sum()) for thr in cfg.score_thresholds
        },
        "top_retrieval_channel_combos": (
            df_bridge["retrieval_channels"].astype(str).value_counts().head(20).to_dict()
            if not df_bridge.empty
            else {}
        ),
        "timing_seconds": timing,
        "enrichment_meta": enrich_meta,
        "scoring_meta": score_meta,
        "nodes_meta": nodes_meta,
        "embeddings_json": str(emb_path) if emb_path else None,
    }
    if gt_diag:
        summary["gt_bridge_diagnostics"] = gt_diag
    timing["total_sec"] = time.perf_counter() - t0
    summary["timing_seconds"]["total_sec"] = timing["total_sec"]

    main_csv = out_root / "bridge_candidate_scores.csv"

    export_paths: dict[str, str] = {"bridge_candidate_scores_csv": str(main_csv)}
    for thr in cfg.score_thresholds:
        sub = df_bridge.loc[scores >= float(thr)].copy()
        p = out_root / f"bridge_candidate_scores_ge_{str(thr).replace('.', '_')}.csv"
        sub.to_csv(p, index=False)
        export_paths[f"scores_ge_{thr}"] = str(p)

    thr_hi = max(cfg.score_thresholds) if cfg.score_thresholds else 0.9
    proposed = df_bridge.loc[scores >= float(thr_hi)].copy()
    prop_path = out_root / f"bridge_edges_proposed_ge_{str(thr_hi).replace('.', '_')}.csv"
    proposed[["email_i", "email_j", "score", "retrieval_channels", "n_retrieval_channels"]].to_csv(
        prop_path, index=False
    )
    export_paths["bridge_edges_proposed"] = str(prop_path)

    plot_files = _write_score_plots(df_bridge, plots_dir, gt_diag=gt_diag)
    summary["plot_files"] = plot_files

    email_catalog: dict[str, dict[str, str]] = {}
    try:
        from seed_candidate_workflow.utils.pair_score_separation import (
            _load_email_text_catalog,
            _resolve_default_misp_json_path,
        )

        misp_path = _resolve_default_misp_json_path(project_root)
        email_catalog, _meta = _load_email_text_catalog(
            project_root=project_root,
            misp_json_path=misp_path,
            misp_translated_json_path=None,
        )
    except Exception:
        email_catalog = {}

    if not cfg.skip_review_enrichment:
        from seed_candidate_workflow.utils.bridge_candidate_review import export_bridge_review_artifacts

        review_export = export_bridge_review_artifacts(
            df_bridge,
            out_root=out_root,
            email_catalog=email_catalog,
            label_map=label_map,
            review_meta=review_meta if not cfg.skip_review_enrichment else None,
            score_threshold_high=float(thr_hi),
            high_score_max_rows=int(cfg.high_score_review_max_rows),
            mid_score_max_rows=int(cfg.mid_score_review_max_rows),
            low_score_max_rows=int(cfg.low_score_review_max_rows),
        )
        export_paths.update(review_export.get("export_paths") or {})
        summary["bridge_review_meta"] = review_meta
        summary["bridge_feature_population_diagnostics"] = review_export.get(
            "bridge_feature_population_diagnostics"
        )
        summary["bridge_band_analysis"] = review_export.get("bridge_band_analysis")
        summary["bridge_suspicious_high_score_analysis"] = review_export.get(
            "bridge_suspicious_high_score_analysis"
        )
        summary["bridge_trustworthiness_recommendation"] = review_export.get(
            "bridge_trustworthiness_recommendation"
        )
    else:
        df_bridge.to_csv(main_csv, index=False)

    summary_path = out_root / "bridge_candidate_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    export_paths["summary_json"] = str(summary_path)

    if gt_diag:
        gt_path_out = out_root / "bridge_candidate_gt_diagnostic_summary.json"
        with open(gt_path_out, "w", encoding="utf-8") as f:
            json.dump(gt_diag, f, indent=2, default=str)
        export_paths["gt_diagnostic_json"] = str(gt_path_out)

    return {
        "output_dir": str(out_root),
        "summary_path": str(summary_path),
        "main_csv": str(main_csv),
        "export_paths": export_paths,
        "summary": summary,
        "n_bridge_candidates": int(len(df_bridge)),
    }
