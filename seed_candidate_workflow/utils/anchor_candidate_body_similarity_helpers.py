"""
Body-text similarity candidate families (token / char-4gram Jaccard).

Uses cached per-email token/char-4gram sets and evaluates pairs on a restricted pool
(prior generator union + optional semantic band + filtered inverted index).
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator, Literal

import pandas as pd

from seed_candidate_workflow.utils.body_similarity_cache import (
    EmailBodyFeatureStore,
    _jaccard_may_reach_threshold,
    build_or_load_email_body_feature_store,
)
from seed_candidate_workflow.utils.body_similarity_progress import (
    BodySimilarityProgress,
    progress_from_cfg,
)
from seed_candidate_workflow.utils.pair_similarity_features import (
    load_misp_text_catalog_for_pairs,
)

logger = logging.getLogger(__name__)

PairMode = Literal["token", "char4"]
PairRow = tuple[str, str, float, float]  # email_i, email_j, primary_jaccard, secondary_jaccard


def _show_progress(generator_cfg: dict[str, Any]) -> bool:
    return bool(generator_cfg.get("show_progress", True))


def _jaccard_primary_if_above(
    store: EmailBodyFeatureStore,
    a: str,
    b: str,
    mode: PairMode,
    min_jaccard: float,
) -> float | None:
    if mode == "token":
        return store.token_jaccard_if_above(a, b, min_jaccard=min_jaccard)
    return store.char4_jaccard_if_above(a, b, min_jaccard=min_jaccard)


def _jaccard_secondary(store: EmailBodyFeatureStore, a: str, b: str, mode: PairMode) -> float:
    if mode == "token":
        return store.char4_jaccard(a, b)
    return store.token_jaccard(a, b)


def _element_document_frequencies(
    email_sets: dict[str, frozenset[str]],
    *,
    max_df: int | None,
) -> dict[str, int]:
    df: dict[str, int] = defaultdict(int)
    for tokset in email_sets.values():
        for el in tokset:
            df[el] += 1
    if max_df is None or max_df <= 0:
        return dict(df)
    return dict(df)


def _iter_pairs_from_prior_pool(
    store: EmailBodyFeatureStore,
    prior_pair_pool: set[tuple[str, str]],
    *,
    mode: PairMode,
    min_jaccard: float,
    max_candidate_rows: int,
    progress: BodySimilarityProgress,
    phase_label: str,
) -> Iterator[PairRow]:
    n_emitted = 0
    pool_list = list(prior_pair_pool)
    progress.loop_start(phase_label, len(pool_list))
    tick_every = max(1, len(pool_list) // 200) if pool_list else 1
    since_tick = 0
    for a, b in pool_list:
        if n_emitted >= max_candidate_rows:
            break
        since_tick += 1
        jac_p = _jaccard_primary_if_above(store, a, b, mode, min_jaccard)
        if jac_p is None:
            if since_tick >= tick_every:
                progress.loop_tick(since_tick, hits=n_emitted)
                since_tick = 0
            continue
        jac_s = _jaccard_secondary(store, a, b, mode)
        n_emitted += 1
        yield a, b, float(jac_p), float(jac_s)
        if since_tick >= tick_every:
            progress.loop_tick(since_tick, hits=n_emitted)
            since_tick = 0
    if since_tick:
        progress.loop_tick(since_tick, hits=n_emitted)
    progress.loop_done(hits=n_emitted)


def _iter_pairs_from_filtered_inverted_index(
    store: EmailBodyFeatureStore,
    *,
    mode: PairMode,
    min_jaccard: float,
    max_candidate_rows: int,
    max_element_df: int,
    email_id_subset: set[str] | None = None,
    progress: BodySimilarityProgress,
    phase_label: str,
) -> Iterator[PairRow]:
    """Inverted index with hub filtering (max document frequency)."""
    if mode == "token":
        email_sets = store.token_sets
    else:
        email_sets = store.char4_sets

    if email_id_subset is not None:
        email_sets = {eid: s for eid, s in email_sets.items() if eid in email_id_subset}

    if not email_sets or max_candidate_rows <= 0:
        return

    progress.phase_start(f"{phase_label}: document frequencies")
    el_df = _element_document_frequencies(email_sets, max_df=max_element_df)
    progress.phase_done(f"{phase_label}: document frequencies", elements=len(el_df))

    progress.phase_start(f"{phase_label}: build inverted index")
    inv: dict[str, list[str]] = defaultdict(list)
    email_items = list(email_sets.items())
    for eid, tokset in email_items:
        if not tokset:
            continue
        for t in tokset:
            if el_df.get(t, 0) <= max_element_df:
                inv[t].append(eid)
    progress.phase_done(f"{phase_label}: build inverted index", terms=len(inv))

    buckets = [sorted(set(emails)) for emails in inv.values() if len(emails) >= 2]
    seen: set[tuple[str, str]] = set()
    n_emitted = 0
    n_compared = 0
    n_pruned = 0

    progress.loop_start(f"{phase_label}: scan buckets", len(buckets))
    tick_every = max(1, len(buckets) // 100) if buckets else 1
    since_tick = 0
    for uniq in buckets:
        if n_emitted >= max_candidate_rows:
            break
        since_tick += 1
        bucket_size = len(uniq)
        for i in range(bucket_size):
            if n_emitted >= max_candidate_rows:
                break
            ei = uniq[i]
            len_i = len(email_sets.get(ei) or ())
            for j in range(i + 1, bucket_size):
                if n_emitted >= max_candidate_rows:
                    break
                ej = uniq[j]
                pk = (ei, ej) if ei <= ej else (ej, ei)
                if pk in seen:
                    continue
                n_compared += 1
                len_j = len(email_sets.get(ej) or ())
                if not _jaccard_may_reach_threshold(len_i, len_j, min_jaccard):
                    n_pruned += 1
                    continue
                jac_p = _jaccard_primary_if_above(store, ei, ej, mode, min_jaccard)
                if jac_p is None:
                    continue
                seen.add(pk)
                n_emitted += 1
                jac_s = _jaccard_secondary(store, ei, ej, mode)
                yield pk[0], pk[1], float(jac_p), float(jac_s)
        if since_tick >= tick_every:
            progress.loop_tick(
                since_tick,
                hits=n_emitted,
                compared=n_compared,
                pruned=n_pruned,
            )
            since_tick = 0
    if since_tick:
        progress.loop_tick(since_tick, hits=n_emitted, compared=n_compared, pruned=n_pruned)
    progress.loop_done(hits=n_emitted, compared=n_compared, pruned=n_pruned)


def _collect_pairs_for_mode(
    store: EmailBodyFeatureStore,
    *,
    mode: PairMode,
    min_jaccard: float,
    max_candidate_rows: int,
    prior_pair_pool: set[tuple[str, str]] | None,
    semantic_band_pool: set[tuple[str, str]] | None,
    use_filtered_inverted_index: bool,
    max_element_df: int,
    email_id_subset: set[str] | None,
    progress: BodySimilarityProgress,
    progress_label: str,
    skip_inverted_index_if_prior_hits_at_least: int | None = None,
) -> tuple[list[PairRow], dict[str, Any]]:
    timing: dict[str, Any] = {}
    seen: set[tuple[str, str]] = set()
    rows: list[PairRow] = []
    mode_label = "token" if mode == "token" else "char4"

    def _consume(it: Iterator[PairRow], label: str) -> None:
        nonlocal rows
        t0 = time.perf_counter()
        n_before = len(rows)
        n_seen = 0
        for a, b, jac_p, jac_s in it:
            n_seen += 1
            if len(rows) >= max_candidate_rows:
                break
            pk = (a, b) if a <= b else (b, a)
            if pk in seen:
                continue
            seen.add(pk)
            rows.append((a, b, jac_p, jac_s))
        timing[f"{label}_seconds"] = float(time.perf_counter() - t0)
        timing[f"{label}_pairs_added"] = int(len(rows) - n_before)
        timing[f"{label}_pairs_scanned"] = int(n_seen)

    eval_pool: set[tuple[str, str]] = set()
    if prior_pair_pool:
        eval_pool |= prior_pair_pool
    if semantic_band_pool:
        eval_pool |= semantic_band_pool

    if eval_pool:
        progress.phase_start(
            f"{progress_label} ({mode_label}): prior pool",
            detail=f"{len(eval_pool):,} pairs",
        )
        _consume(
            _iter_pairs_from_prior_pool(
                store,
                eval_pool,
                mode=mode,
                min_jaccard=min_jaccard,
                max_candidate_rows=max_candidate_rows,
                progress=progress,
                phase_label=f"{progress_label} ({mode_label}): prior pool",
            ),
            "prior_pool_jaccard",
        )
        progress.phase_done(
            f"{progress_label} ({mode_label}): prior pool",
            added=timing.get("prior_pool_jaccard_pairs_added", 0),
        )
        timing["n_pairs_in_eval_pool"] = int(len(eval_pool))

    if use_filtered_inverted_index and len(rows) < max_candidate_rows:
        prior_added = int(timing.get("prior_pool_jaccard_pairs_added") or 0)
        if (
            skip_inverted_index_if_prior_hits_at_least is not None
            and prior_added >= int(skip_inverted_index_if_prior_hits_at_least)
        ):
            timing["filtered_inverted_index_skipped"] = True
            timing["filtered_inverted_index_skip_reason"] = (
                f"prior_pool_hits>={skip_inverted_index_if_prior_hits_at_least}"
            )
            timing["filtered_inverted_index_seconds"] = 0.0
            timing["filtered_inverted_index_pairs_added"] = 0
            progress.message(
                f"{progress_label} ({mode_label}): inverted index SKIPPED "
                f"(prior pool already added {prior_added:,} hits)"
            )
        else:
            progress.phase_start(f"{progress_label} ({mode_label}): inverted index")
            _consume(
                _iter_pairs_from_filtered_inverted_index(
                    store,
                    mode=mode,
                    min_jaccard=min_jaccard,
                    max_candidate_rows=max_candidate_rows - len(rows),
                    max_element_df=max_element_df,
                    email_id_subset=email_id_subset,
                    progress=progress,
                    phase_label=f"{progress_label} ({mode_label}): inverted index",
                ),
                "filtered_inverted_index",
            )
            progress.phase_done(
                f"{progress_label} ({mode_label}): inverted index",
                added=timing.get("filtered_inverted_index_pairs_added", 0),
            )

    timing["n_pairs_emitted_total"] = int(len(rows))
    timing["n_pairs_unique_seen"] = int(len(seen))
    return rows, timing


def _resolve_misp_path(project_root: Path, generator_cfg: dict[str, Any]) -> Path:
    raw = generator_cfg.get("misp_json_path")
    if raw:
        p = Path(str(raw))
        return p if p.is_absolute() else (project_root / p).resolve()
    from seed_candidate_workflow.utils.pair_score_separation import _resolve_default_misp_json_path

    resolved = _resolve_default_misp_json_path(project_root)
    if resolved is None:
        raise FileNotFoundError("MISP JSON path not found for body similarity cache")
    return Path(resolved).resolve()


def prepare_body_feature_store(
    *,
    nodes_df: pd.DataFrame,
    generator_cfg: dict[str, Any],
    project_root: Path,
    graph_id: str,
    text_catalog: dict[str, dict[str, str]] | None,
    progress: BodySimilarityProgress | None = None,
) -> tuple[EmailBodyFeatureStore, dict[str, Any], dict[str, dict[str, str]]]:
    catalog = text_catalog
    catalog_meta: dict[str, Any] = {}
    if catalog is None:
        catalog, catalog_meta = load_misp_text_catalog_for_pairs(project_root=project_root)

    node_ids = sorted(set(nodes_df["external_id"].astype(str).tolist()))
    if not catalog:
        raise ValueError("empty_text_catalog")

    prog = progress or progress_from_cfg(generator_cfg, graph_id=graph_id)
    use_cache = bool(generator_cfg.get("use_body_similarity_cache", True))
    force_rebuild = bool(generator_cfg.get("force_rebuild_body_cache", False))
    cache_root_raw = generator_cfg.get("body_similarity_cache_root")
    cache_root = (
        Path(str(cache_root_raw)).expanduser().resolve()
        if cache_root_raw
        else None
    )

    t0 = time.perf_counter()
    if use_cache:
        misp_path = _resolve_misp_path(project_root, generator_cfg)
        raw_salt = generator_cfg.get("body_similarity_cache_salt")
        cache_salt = str(raw_salt).strip() if raw_salt is not None and str(raw_salt).strip() else None
        store, cache_diag = build_or_load_email_body_feature_store(
            email_ids=node_ids,
            text_catalog=catalog,
            graph_id=graph_id,
            misp_json_path=misp_path,
            cache_root=cache_root,
            force_rebuild=force_rebuild,
            progress=prog,
            cache_salt=cache_salt,
        )
    else:
        from seed_candidate_workflow.utils.body_similarity_cache import build_email_body_feature_store

        store, build_meta = build_email_body_feature_store(
            email_ids=node_ids,
            text_catalog=catalog,
            progress=prog,
        )
        cache_diag = {"cache_status": "disabled", "build": build_meta}

    cache_diag["total_prepare_seconds"] = float(time.perf_counter() - t0)
    cache_diag["text_catalog"] = catalog_meta
    return store, cache_diag, catalog


def _parse_pool_cfg(generator_cfg: dict[str, Any]) -> dict[str, Any]:
    return {
        "use_filtered_inverted_index": bool(
            generator_cfg.get("use_filtered_inverted_index", True)
        ),
        "max_token_document_frequency": int(
            generator_cfg.get("max_token_document_frequency", 40)
        ),
        "max_char4gram_document_frequency": int(
            generator_cfg.get("max_char4gram_document_frequency", 60)
        ),
        "show_progress": _show_progress(generator_cfg),
    }


def _generate_body_jaccard_highconf(
    *,
    nodes_df: pd.DataFrame,
    generator_cfg: dict[str, Any],
    project_root: Any,
    graph_id: str,
    mode: PairMode,
    min_jaccard_key: str,
    default_min_j: float,
    source_label: str,
    primary_col: str,
    secondary_col: str,
    text_catalog: dict[str, dict[str, str]] | None,
    prior_pair_pool: set[tuple[str, str]] | None,
    semantic_band_pool: set[tuple[str, str]] | None,
    body_feature_store: EmailBodyFeatureStore | None,
    cache_diag_preload: dict[str, Any] | None,
    progress: BodySimilarityProgress | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    min_j = float(generator_cfg.get(min_jaccard_key, default_min_j))
    max_rows = int(generator_cfg.get("max_candidate_rows", 500_000))
    pool_cfg = _parse_pool_cfg(generator_cfg)
    prog = progress or progress_from_cfg(generator_cfg, graph_id=graph_id)
    root = Path(project_root) if project_root is not None else Path(".")
    max_element_df = (
        pool_cfg["max_token_document_frequency"]
        if mode == "token"
        else pool_cfg["max_char4gram_document_frequency"]
    )

    t_all = time.perf_counter()
    prog.message(f"--- {source_label} ---")
    if body_feature_store is None:
        store, cache_diag, _catalog = prepare_body_feature_store(
            nodes_df=nodes_df,
            generator_cfg=generator_cfg,
            project_root=root,
            graph_id=graph_id,
            text_catalog=text_catalog,
            progress=prog,
        )
    else:
        store = body_feature_store
        cache_diag = dict(cache_diag_preload or {})

    misp_path = _resolve_misp_path(root, generator_cfg)
    force_gen_cache = bool(
        generator_cfg.get("force_rebuild_body_jaccard_generator_cache", False)
        or generator_cfg.get("force_rebuild_body_cache", False)
    )
    from seed_candidate_workflow.utils.body_jaccard_generator_cache import (
        save_body_jaccard_generator_to_cache,
        try_load_body_jaccard_generator_from_cache,
    )

    cached_df, gen_cache_diag = try_load_body_jaccard_generator_from_cache(
        nodes_df=nodes_df,
        generator_cfg=generator_cfg,
        project_root=root,
        graph_id=graph_id,
        misp_json_path=misp_path,
        generator_name=source_label,
        mode=mode,
        min_jaccard=min_j,
        prior_pair_pool=prior_pair_pool,
        semantic_band_pool=semantic_band_pool,
        force_rebuild=force_gen_cache,
    )
    if cached_df is not None:
        prog.message(
            f"generator output cache HIT — skipped pair mining "
            f"({len(cached_df):,} rows, {float(gen_cache_diag.get('cache_load_seconds') or 0):.1f}s load)"
        )
        diag = {
            "status": "ok",
            "from_generator_output_cache": True,
            min_jaccard_key: min_j,
            "max_candidate_rows": max_rows,
            "n_pairs_emitted": int(len(cached_df)),
            "truncated": bool(len(cached_df) >= max_rows),
            "cache": cache_diag,
            "generator_output_cache": gen_cache_diag,
            "pool_config": pool_cfg,
            "n_prior_pool_pairs": int(len(prior_pair_pool or set())),
            "n_semantic_band_pool_pairs": int(len(semantic_band_pool or set())),
            "generator_total_seconds": float(time.perf_counter() - t_all),
        }
        prog.phase_done(source_label, emitted=len(cached_df), cached=True)
        return cached_df, diag

    email_subset: set[str] | None = None
    if prior_pair_pool or semantic_band_pool:
        email_subset = set()
        for a, b in (prior_pair_pool or set()) | (semantic_band_pool or set()):
            email_subset.add(a)
            email_subset.add(b)

    skip_index_prior = generator_cfg.get("skip_inverted_index_if_prior_hits_at_least")
    skip_index_prior_int: int | None = None
    if skip_index_prior is not None and str(skip_index_prior).strip() != "":
        skip_index_prior_int = int(skip_index_prior)

    pair_rows, pair_timing = _collect_pairs_for_mode(
        store,
        mode=mode,
        min_jaccard=min_j,
        max_candidate_rows=max_rows,
        prior_pair_pool=prior_pair_pool,
        semantic_band_pool=semantic_band_pool,
        use_filtered_inverted_index=pool_cfg["use_filtered_inverted_index"],
        max_element_df=max_element_df,
        email_id_subset=email_subset,
        progress=prog,
        progress_label=source_label,
        skip_inverted_index_if_prior_hits_at_least=skip_index_prior_int,
    )

    out_rows: list[dict[str, Any]] = []
    for a, b, jac_p, jac_s in pair_rows:
        out_rows.append(
            {
                "email_i": a,
                "email_j": b,
                "source": source_label,
                primary_col: float(jac_p),
                secondary_col: float(jac_s),
            }
        )

    df = pd.DataFrame(out_rows)
    gen_save_diag = save_body_jaccard_generator_to_cache(
        df,
        nodes_df=nodes_df,
        generator_cfg=generator_cfg,
        misp_json_path=misp_path,
        graph_id=graph_id,
        generator_name=source_label,
        mode=mode,
        min_jaccard=min_j,
        prior_pair_pool=prior_pair_pool,
        semantic_band_pool=semantic_band_pool,
    )
    diag = {
        "status": "ok",
        "from_generator_output_cache": False,
        min_jaccard_key: min_j,
        "max_candidate_rows": max_rows,
        "n_pairs_emitted": int(len(df)),
        "truncated": bool(len(df) >= max_rows),
        "cache": cache_diag,
        "generator_output_cache": gen_save_diag,
        "pair_evaluation": pair_timing,
        "pool_config": pool_cfg,
        "n_prior_pool_pairs": int(len(prior_pair_pool or set())),
        "n_semantic_band_pool_pairs": int(len(semantic_band_pool or set())),
        "generator_total_seconds": float(time.perf_counter() - t_all),
    }
    prog.phase_done(
        source_label,
        emitted=len(df),
        pool_s=float(pair_timing.get("prior_pool_jaccard_seconds") or 0.0),
        index_s=float(pair_timing.get("filtered_inverted_index_seconds") or 0.0),
    )
    logger.info(
        "%s graph_id=%s emitted=%d pool_s=%.2f index_s=%.2f total_s=%.2f cache=%s",
        source_label,
        graph_id,
        len(df),
        float(pair_timing.get("prior_pool_jaccard_seconds") or 0.0),
        float(pair_timing.get("filtered_inverted_index_seconds") or 0.0),
        diag["generator_total_seconds"],
        cache_diag.get("cache_status"),
    )
    return df, diag


def generate_body_token_jaccard_highconf_v1(
    *,
    nodes_df: pd.DataFrame,
    generator_cfg: dict[str, Any],
    project_root: Any,
    graph_id: str = "",
    text_catalog: dict[str, dict[str, str]] | None = None,
    prior_pair_pool: set[tuple[str, str]] | None = None,
    semantic_band_pool: set[tuple[str, str]] | None = None,
    body_feature_store: EmailBodyFeatureStore | None = None,
    cache_diag_preload: dict[str, Any] | None = None,
    progress: BodySimilarityProgress | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Candidate pairs with ``body_token_jaccard >= min_body_token_jaccard`` (default 0.25)."""
    return _generate_body_jaccard_highconf(
        nodes_df=nodes_df,
        generator_cfg=generator_cfg,
        project_root=project_root,
        graph_id=graph_id,
        mode="token",
        min_jaccard_key="min_body_token_jaccard",
        default_min_j=0.25,
        source_label="body_token_jaccard_highconf_v1",
        primary_col="body_token_jaccard",
        secondary_col="body_char4gram_jaccard",
        text_catalog=text_catalog,
        prior_pair_pool=prior_pair_pool,
        semantic_band_pool=semantic_band_pool,
        body_feature_store=body_feature_store,
        cache_diag_preload=cache_diag_preload,
        progress=progress,
    )


def generate_body_char4gram_jaccard_highconf_v1(
    *,
    nodes_df: pd.DataFrame,
    generator_cfg: dict[str, Any],
    project_root: Any,
    graph_id: str = "",
    text_catalog: dict[str, dict[str, str]] | None = None,
    prior_pair_pool: set[tuple[str, str]] | None = None,
    semantic_band_pool: set[tuple[str, str]] | None = None,
    body_feature_store: EmailBodyFeatureStore | None = None,
    cache_diag_preload: dict[str, Any] | None = None,
    progress: BodySimilarityProgress | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Candidate pairs with ``body_char4gram_jaccard >= min`` (default 0.25)."""
    return _generate_body_jaccard_highconf(
        nodes_df=nodes_df,
        generator_cfg=generator_cfg,
        project_root=project_root,
        graph_id=graph_id,
        mode="char4",
        min_jaccard_key="min_body_char4gram_jaccard",
        default_min_j=0.25,
        source_label="body_char4gram_jaccard_highconf_v1",
        primary_col="body_char4gram_jaccard",
        secondary_col="body_token_jaccard",
        text_catalog=text_catalog,
        prior_pair_pool=prior_pair_pool,
        semantic_band_pool=semantic_band_pool,
        body_feature_store=body_feature_store,
        cache_diag_preload=cache_diag_preload,
        progress=progress,
    )


BODY_GENERATOR_NAMES = frozenset(
    {"body_token_jaccard_highconf_v1", "body_char4gram_jaccard_highconf_v1"}
)


def build_semantic_band_pool_for_body_generators(
    *,
    nodes_df: pd.DataFrame,
    id_to_vec: dict[str, Any],
    generator_cfg: dict[str, Any],
    progress: BodySimilarityProgress | None = None,
) -> set[tuple[str, str]]:
    """Optional semantic mid-band pairs (0.85 <= cos < 0.90) to widen body Jaccard evaluation."""
    if not bool(generator_cfg.get("include_semantic_band_pool", True)):
        return set()
    from seed_candidate_workflow.utils.anchor_candidate_semantic_mid_support_helpers import (
        _compute_direct_cosine_band_pairs,
    )

    prog = progress or progress_from_cfg(generator_cfg)
    prog.phase_start("semantic mid-band pool (kNN)")
    node_ids = nodes_df["external_id"].astype(str).tolist()
    band_pairs = _compute_direct_cosine_band_pairs(
        node_ids=node_ids,
        id_to_vec=id_to_vec,
        semantic_top_k=int(generator_cfg.get("semantic_top_k", 50)),
        semantic_min_cos=float(generator_cfg.get("semantic_min_cos", 0.85)),
        semantic_max_cos_exclusive=float(generator_cfg.get("semantic_max_cos_exclusive", 0.90)),
    )
    prog.phase_done("semantic mid-band pool (kNN)", pairs=len(band_pairs))
    return set(band_pairs.keys())
