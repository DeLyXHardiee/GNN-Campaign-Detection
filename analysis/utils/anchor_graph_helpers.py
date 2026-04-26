from __future__ import annotations

from datetime import datetime, timezone
import json
import math
import pickle
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

from analysis.utils import graph_structure_helpers as gh
from analysis.utils import semantic_shard_helpers as ssh
from analysis.utils.config_run_fields import resolve_graph_id
from analysis.utils.pair_graph_contract import migrate_unscored_graph_id_column

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - optional dependency
    tqdm = None


DEFAULT_INFRA_CHANNELS: tuple[str, ...] = (
    "url_set",
    "sender_set",
    "attachment_set",
    "sender_email_domain_set",
    "domain_set",
    "stem_set",
    "html_structure_fingerprint_set",
    "return_path_email_set",
    "return_path_domain_set",
    "helo_host_set",
    "origin_ip_set",
    "received_host_set",
)

DEFAULT_CANDIDATE_CHANNELS: tuple[str, ...] = (
    "url_set",
    "sender_set",
    "attachment_set",
    "sender_email_domain_set",
    "domain_set",
    "stem_set",
)

DEFAULT_NODE_COLUMNS: tuple[str, ...] = (
    "external_id",
    "ts",
    "subject",
    "body",
    "url_set",
    "sender_set",
    "attachment_set",
    "sender_email_domain_set",
    "domain_set",
    "stem_set",
    "html_structure_fingerprint_set",
    "return_path_email_set",
    "return_path_domain_set",
    "helo_host_set",
    "origin_ip_set",
    "received_host_set",
)


@dataclass
class AnchorChannelScoring:
    weight: float
    scoring_mode: str = "idf_saturated"
    idf_exponent: float = 1.0
    idf_scale: float = 1.0
    max_email_df: int | None = None
    contribution_cap: float | None = None

    @classmethod
    def from_mapping(cls, d: dict[str, Any]) -> AnchorChannelScoring:
        return cls(
            weight=float(d.get("weight", 0.5)),
            scoring_mode=str(d.get("scoring_mode", "idf_saturated")).strip().lower(),
            idf_exponent=float(d.get("idf_exponent", 1.0)),
            idf_scale=float(d.get("idf_scale", 1.0)),
            max_email_df=(
                None if d.get("max_email_df") is None else int(d["max_email_df"])
            ),
            contribution_cap=(
                None
                if d.get("contribution_cap") is None
                else float(d["contribution_cap"])
            ),
        )


def _l2_rows(x: np.ndarray) -> np.ndarray:
    a = np.asarray(x, dtype=np.float32)
    n = np.linalg.norm(a, axis=1, keepdims=True)
    n[n == 0.0] = 1.0
    return a / n


def _normalize_channel_to_set_col(channel: str) -> str:
    c = str(channel).strip()
    if not c:
        return c
    aliases = {
        "sender_domain": "sender_email_domain_set",
        "sender_email_domain": "sender_email_domain_set",
        "sender_email_domains": "sender_email_domain_set",
    }
    c = aliases.get(c, c)
    if c.endswith("_set"):
        return c
    return f"{c}_set"


def _base_channel_name(set_col: str) -> str:
    s = str(set_col)
    if s.endswith("_set"):
        return s[:-4]
    return s


def _load_domain_list(path: Path | None) -> frozenset[str]:
    if path is None:
        return frozenset()
    p = path.expanduser().resolve()
    if not p.is_file():
        return frozenset()
    vals: set[str] = set()
    for raw in p.read_text(encoding="utf-8", errors="replace").splitlines():
        t = str(raw).strip().lower()
        if not t or t.startswith("#"):
            continue
        vals.add(t)
    return frozenset(vals)


def _to_set_cell(v: Any) -> set[str]:
    if isinstance(v, set):
        return {str(x) for x in v if str(x)}
    if isinstance(v, list):
        return {str(x) for x in v if str(x)}
    if isinstance(v, str):
        t = v.strip()
        if not t:
            return set()
        if t.startswith("[") and t.endswith("]"):
            try:
                xs = json.loads(t)
                if isinstance(xs, list):
                    return {str(x) for x in xs if str(x)}
            except Exception:
                return set()
    return set()


def _serialize_set_col(v: Any) -> Any:
    if isinstance(v, set):
        return json.dumps(sorted(v), ensure_ascii=False)
    return v


def _index_to_str(meta: dict[str, Any], node_type: str) -> list[str]:
    xs = (
        (meta.get("node_maps") or {})
        .get(node_type, {})
        .get("index_to_string")
    ) or []
    return [str(x) for x in xs]


def _build_email_node_table(
    *,
    graph_pt: Path,
    meta_json: Path,
    to_undirected: bool,
    filter_popular_domains: bool,
    popular_domains_path: Path | None,
    filter_web_hosting_domains: bool,
    web_hosting_domains_path: Path | None,
    include_text_fields: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    meta = gh.load_meta(meta_json)
    data = gh.load_hetero(graph_pt, to_undirected=to_undirected)
    external_ids = gh.email_external_id_list(meta)
    n_email = len(external_ids)
    ts = (meta.get("email_attrs") or {}).get("ts") or [None] * n_email

    root = gh.find_project_root()
    if popular_domains_path is None:
        popular_domains_path = gh.default_popular_domains_path(root)
    pop_domains = _load_domain_list(popular_domains_path)
    if not filter_popular_domains:
        pop_domains = frozenset()

    web_domains = _load_domain_list(web_hosting_domains_path)
    if not filter_web_hosting_domains:
        web_domains = frozenset()

    combined_benign = frozenset(set(pop_domains) | set(web_domains))
    url_sets, domain_sets, stem_sets, benign_diag = gh.build_email_url_derived_infra_sets(
        data,
        meta,
        popular_domains=combined_benign,
        popular_domains_source=popular_domains_path
        if filter_popular_domains
        else (web_hosting_domains_path if filter_web_hosting_domains else ""),
    )

    email_sets = gh.build_email_artifact_sets(data)
    skip_direct = frozenset({"url", "domain", "stem"})
    email_sets = {k: v for k, v in email_sets.items() if k not in skip_direct}
    index_to_str = {node_type: _index_to_str(meta, node_type) for node_type in email_sets}

    # Sender->domain union (semantic shard step2 convention).
    sender_map = index_to_str.get("sender", _index_to_str(meta, "sender"))
    email_domain_map = index_to_str.get(
        "email_domain", _index_to_str(meta, "email_domain")
    )
    sender_to_domain: dict[int, set[str]] = defaultdict(set)
    if ("sender", "from_domain", "email_domain") in data.edge_types:
        ei = data["sender", "from_domain", "email_domain"].edge_index
        if ei is not None and ei.numel() > 0:
            s_idx = ei[0].detach().cpu().numpy().astype(np.int64)
            d_idx = ei[1].detach().cpu().numpy().astype(np.int64)
            for s, d in zip(s_idx, d_idx, strict=False):
                if 0 <= int(d) < len(email_domain_map):
                    sender_to_domain[int(s)].add(email_domain_map[int(d)])

    sender_domain_sets: list[set[str]] = [set() for _ in range(n_email)]
    if ("email", "has_sender", "sender") in data.edge_types:
        ei = data["email", "has_sender", "sender"].edge_index
        if ei is not None and ei.numel() > 0:
            e_idx = ei[0].detach().cpu().numpy().astype(np.int64)
            s_idx = ei[1].detach().cpu().numpy().astype(np.int64)
            for e, s in zip(e_idx, s_idx, strict=False):
                if 0 <= int(e) < n_email:
                    sender_domain_sets[int(e)].update(sender_to_domain.get(int(s), set()))

    attrs = meta.get("email_attrs") or {}
    subject = attrs.get("subject") or [None] * n_email
    body = attrs.get("body") or [None] * n_email
    subject_tr = attrs.get("subject_translated") or [None] * n_email
    body_tr = attrs.get("body_translated") or [None] * n_email

    rows: list[dict[str, Any]] = []
    for i, eid in enumerate(external_ids):
        rec: dict[str, Any] = {
            "external_id": str(eid),
            "ts": float(ts[i]) if i < len(ts) and ts[i] is not None else np.nan,
            "url_set": set(url_sets[i]) if i < len(url_sets) else set(),
            "domain_set": set(domain_sets[i]) if i < len(domain_sets) else set(),
            "stem_set": set(stem_sets[i]) if i < len(stem_sets) else set(),
            "sender_email_domain_set": set(sender_domain_sets[i])
            if i < len(sender_domain_sets)
            else set(),
        }
        for node_type, idx_sets in email_sets.items():
            idxs = idx_sets[i] if i < len(idx_sets) else set()
            labels = index_to_str.get(node_type, [])
            rec[f"{node_type}_set"] = set(
                labels[int(j)] if 0 <= int(j) < len(labels) else str(int(j))
                for j in idxs
                if j is not None
            )
        if include_text_fields:
            s0 = str(subject[i]) if i < len(subject) and subject[i] is not None else ""
            b0 = str(body[i]) if i < len(body) and body[i] is not None else ""
            st = (
                str(subject_tr[i])
                if i < len(subject_tr) and subject_tr[i] is not None
                else ""
            )
            bt = (
                str(body_tr[i]) if i < len(body_tr) and body_tr[i] is not None else ""
            )
            rec["subject"] = s0
            rec["body"] = b0
            rec["subject_translated"] = st
            rec["body_translated"] = bt
        rows.append(rec)

    node_df = pd.DataFrame(rows)
    benign_diag = {
        **(benign_diag or {}),
        "filter_popular_domains": bool(filter_popular_domains),
        "filter_web_hosting_domains": bool(filter_web_hosting_domains),
        "n_web_hosting_domains_loaded": int(len(web_domains)),
        "n_popular_domains_loaded_effective": int(len(pop_domains)),
    }
    return node_df, benign_diag


def _compute_tfidf_embeddings(
    nodes_df: pd.DataFrame,
    *,
    prefer_translated: bool = True,
    max_features: int = 4096,
) -> dict[str, np.ndarray]:
    subj_col = "subject_translated" if prefer_translated else "subject"
    body_col = "body_translated" if prefer_translated else "body"
    if subj_col not in nodes_df.columns:
        subj_col = "subject"
    if body_col not in nodes_df.columns:
        body_col = "body"
    texts = []
    ids = []
    for _, r in nodes_df.iterrows():
        sid = str(r.get("external_id") or "")
        if not sid:
            continue
        subj = str(r.get(subj_col) or "").strip()
        body = str(r.get(body_col) or "").strip()
        txt = f"{subj}\n{body}".strip()
        ids.append(sid)
        texts.append(txt)
    if not texts:
        return {}
    vec = TfidfVectorizer(max_features=int(max_features))
    x = vec.fit_transform(texts)
    x = _l2_rows(x.toarray().astype(np.float32))
    out: dict[str, np.ndarray] = {}
    for i, eid in enumerate(ids):
        out[eid] = x[i]
    return out


def load_embedding_vectors(
    *,
    nodes_df: pd.DataFrame,
    embeddings_json: Path | None,
    embedding_source: str = "cache_or_compute",
    prefer_translated_for_compute: bool = True,
    tfidf_max_features: int = 4096,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    source = str(embedding_source).strip().lower()
    use_cache = source in {"cache", "cache_or_compute", "cache_or_skip"}
    allow_compute = source in {"compute", "cache_or_compute"}
    allow_skip = source in {"cache_or_skip"}

    id_to_vec: dict[str, np.ndarray] = {}
    cache_summary: dict[str, Any] = {}
    if use_cache and embeddings_json is not None and embeddings_json.is_file():
        payload, id_to_vec_raw, summ = ssh.load_transformer_cache(embeddings_json)
        id_to_vec = {str(k): np.asarray(v, dtype=np.float32) for k, v in id_to_vec_raw.items()}
        cache_summary = {
            "embedding_file_model": str(summ.model),
            "embedding_file_subj_dim": int(summ.subj_dim),
            "embedding_file_body_dim": int(summ.body_dim),
            "embedding_entries_in_by_key": int(summ.n_entries_in_by_key),
            "embedding_entries_with_any_vector": int(summ.n_entries_with_any_vector),
            "embedding_entry_fields": list(summ.entry_fields),
            "embedding_payload_keys": sorted(payload.keys()),
        }

    node_ids = [str(x) for x in nodes_df["external_id"].astype(str).tolist()]
    miss = [eid for eid in node_ids if eid not in id_to_vec]
    if miss and allow_compute:
        computed = _compute_tfidf_embeddings(
            nodes_df,
            prefer_translated=prefer_translated_for_compute,
            max_features=tfidf_max_features,
        )
        for eid in miss:
            if eid in computed:
                id_to_vec[eid] = computed[eid]

    if miss and not allow_compute and not allow_skip:
        raise ValueError(
            "Missing embeddings for some emails and compute fallback disabled. "
            "Use embedding_source=cache_or_compute or cache_or_skip."
        )

    n_cov = sum(1 for eid in node_ids if eid in id_to_vec)
    meta = {
        **cache_summary,
        "embedding_source": source,
        "n_node_ids": int(len(node_ids)),
        "n_node_ids_with_vector": int(n_cov),
        "n_node_ids_missing_vector": int(len(node_ids) - n_cov),
        "compute_fallback_used": bool(allow_compute and len(miss) > 0),
    }
    return id_to_vec, meta


def _semantic_candidate_pairs(
    *,
    node_ids: list[str],
    id_to_vec: dict[str, np.ndarray],
    semantic_top_k: int,
    semantic_min_cos: float,
) -> tuple[set[tuple[int, int]], dict[tuple[int, int], float]]:
    idx_ids = [eid for eid in node_ids if eid in id_to_vec]
    if not idx_ids:
        return set(), {}

    emb = _l2_rows(np.stack([id_to_vec[eid] for eid in idx_ids]).astype(np.float32))
    index_map = {eid: i for i, eid in enumerate(node_ids)}
    dense_idx = [index_map[eid] for eid in idx_ids]
    n = emb.shape[0]
    if n <= 1 or semantic_top_k <= 0:
        return set(), {}

    k = min(int(semantic_top_k) + 1, n)
    nn = NearestNeighbors(n_neighbors=k, metric="cosine", algorithm="brute")
    nn.fit(emb)
    dists, neigh = nn.kneighbors(emb, return_distance=True)

    pairs: set[tuple[int, int]] = set()
    cos_map: dict[tuple[int, int], float] = {}
    thr = float(semantic_min_cos)
    for local_i in range(n):
        i = dense_idx[local_i]
        for local_j, dist in zip(neigh[local_i], dists[local_i], strict=False):
            if int(local_j) == int(local_i):
                continue
            j = dense_idx[int(local_j)]
            a, b = (i, j) if i < j else (j, i)
            cs = float(1.0 - float(dist))
            if cs < thr:
                continue
            pairs.add((a, b))
            prev = cos_map.get((a, b))
            cos_map[(a, b)] = cs if prev is None else max(prev, cs)
    return pairs, cos_map


def _infra_candidate_pairs(
    *,
    nodes_df: pd.DataFrame,
    candidate_channels: tuple[str, ...],
    max_pairs_per_artifact: int | None,
) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    n = len(nodes_df)
    for ch in candidate_channels:
        if ch not in nodes_df.columns:
            continue
        inv: dict[str, list[int]] = defaultdict(list)
        for i, s in enumerate(nodes_df[ch].tolist()):
            vals = _to_set_cell(s)
            if not vals:
                continue
            for art in vals:
                inv[str(art)].append(i)
        for idxs in inv.values():
            uniq = sorted(set(int(x) for x in idxs if 0 <= int(x) < n))
            if len(uniq) < 2:
                continue
            total_pairs = len(uniq) * (len(uniq) - 1) // 2
            if max_pairs_per_artifact is not None and total_pairs > int(max_pairs_per_artifact):
                continue
            for a_i in range(len(uniq)):
                ia = uniq[a_i]
                for b_i in range(a_i + 1, len(uniq)):
                    ib = uniq[b_i]
                    out.add((ia, ib))
    return out


def build_anchor_candidates(
    *,
    nodes_df: pd.DataFrame,
    id_to_vec: dict[str, np.ndarray],
    semantic_top_k: int = 20,
    semantic_min_cos: float = 0.80,
    candidate_channels: tuple[str, ...] = DEFAULT_CANDIDATE_CHANNELS,
    max_pairs_per_artifact: int | None = 10000,
) -> tuple[pd.DataFrame, dict[tuple[int, int], float], dict[str, Any]]:
    node_ids = nodes_df["external_id"].astype(str).tolist()
    sem_pairs, sem_cos = _semantic_candidate_pairs(
        node_ids=node_ids,
        id_to_vec=id_to_vec,
        semantic_top_k=int(semantic_top_k),
        semantic_min_cos=float(semantic_min_cos),
    )
    infra_pairs = _infra_candidate_pairs(
        nodes_df=nodes_df,
        candidate_channels=tuple(candidate_channels),
        max_pairs_per_artifact=max_pairs_per_artifact,
    )
    all_pairs = sorted(sem_pairs | infra_pairs)
    rows = [
        {
            "idx_a": int(i),
            "idx_b": int(j),
            "email_a": str(node_ids[i]),
            "email_b": str(node_ids[j]),
            "from_semantic_candidate": bool((i, j) in sem_pairs),
            "from_infra_candidate": bool((i, j) in infra_pairs),
        }
        for i, j in all_pairs
    ]
    summary = {
        "n_nodes": int(len(node_ids)),
        "n_semantic_candidate_pairs": int(len(sem_pairs)),
        "n_infra_candidate_pairs": int(len(infra_pairs)),
        "n_candidate_pairs_union": int(len(all_pairs)),
    }
    return pd.DataFrame(rows), sem_cos, summary


def _idf_raw(n_docs: int, df: int) -> float:
    return float(math.log((1.0 + n_docs) / (1.0 + max(df, 0))))


def _temporal_score(
    ts_a: float,
    ts_b: float,
    *,
    decay_days: float,
    overlap_seconds: float = 3600.0,
) -> tuple[float, float, float]:
    if pd.isna(ts_a) or pd.isna(ts_b):
        return 0.0, float("nan"), 0.0
    gap = abs(float(ts_a) - float(ts_b))
    if gap <= float(overlap_seconds):
        return 1.0, 0.0, 1.0
    gap_days = float(gap / 86400.0)
    if decay_days <= 0:
        return 0.0, gap_days, 0.0
    score = float(math.exp(-gap_days / float(decay_days)))
    return 0.0, gap_days, score


def _resolve_channel_scoring(
    cfg_channels: dict[str, Any],
    *,
    scoring_channels: tuple[str, ...],
) -> dict[str, AnchorChannelScoring]:
    out: dict[str, AnchorChannelScoring] = {}
    default_weight = float(cfg_channels.get("default_weight", 0.5))
    specs = cfg_channels.get("scoring") or {}
    for raw in scoring_channels:
        ch = _normalize_channel_to_set_col(raw)
        d = specs.get(ch, specs.get(_base_channel_name(ch), {})) or {}
        if d.get("enabled") is False:
            continue
        if "weight" not in d:
            d = {**d, "weight": default_weight}
        out[ch] = AnchorChannelScoring.from_mapping(d)
    return out


def _resolve_unified_channel_configuration(
    config: dict[str, Any],
) -> tuple[tuple[str, ...], tuple[str, ...], dict[str, AnchorChannelScoring], dict[str, Any]]:
    """
    Resolve channel behavior from unified `channels.channel_settings`.

    Expected shape:
      channels:
        channel_settings:
          semantic:
            candidate_enabled: bool
            score_enabled: bool
            top_k: int
            min_cos: float
            weight: float
          url: / url_set:
            enabled: bool
            candidate_enabled: bool
            score_enabled: bool
            weight: float
            ...
    """
    channels_cfg = config.get("channels") or {}
    cand_cfg = config.get("candidate_generation") or {}

    settings = channels_cfg.get("channel_settings")
    if not isinstance(settings, dict) or not settings:
        raise ValueError(
            "Invalid config: channels.channel_settings is required and must be a non-empty object."
        )

    default_weight = float(channels_cfg.get("default_weight", 0.5))
    candidate_channels_list: list[str] = []
    scoring_channels_list: list[str] = []
    scoring_specs: dict[str, AnchorChannelScoring] = {}

    sem_raw = settings.get("semantic", {}) if isinstance(settings.get("semantic"), dict) else {}
    semantic = {
        "candidate_enabled": bool(sem_raw.get("edge_create_enabled", sem_raw.get("candidate_enabled", True))),
        "score_enabled": bool(sem_raw.get("score_enabled", True)),
        "top_k": int(sem_raw.get("top_k", cand_cfg.get("semantic_top_k", 20))),
        "min_cos": float(sem_raw.get("min_cos", cand_cfg.get("semantic_min_cos", 0.80))),
    }

    for raw_key, raw_cfg in settings.items():
        if str(raw_key).strip().lower() == "semantic":
            continue
        if not isinstance(raw_cfg, dict):
            raw_cfg = {}
        enabled = bool(raw_cfg.get("enabled", True))
        # New explicit names are aliases; preserve old behavior/values.
        candidate_enabled = bool(
            raw_cfg.get(
                "edge_create_enabled",
                raw_cfg.get("candidate_enabled", enabled),
            )
        )
        score_enabled = bool(raw_cfg.get("score_enabled", enabled))
        ch = _normalize_channel_to_set_col(str(raw_key))

        if candidate_enabled:
            candidate_channels_list.append(ch)
        if score_enabled:
            scoring_channels_list.append(ch)
            d = {k: v for k, v in raw_cfg.items() if k not in {"enabled", "candidate_enabled", "score_enabled"}}
            if "weight" not in d:
                d["weight"] = default_weight
            scoring_specs[ch] = AnchorChannelScoring.from_mapping(d)

    candidate_channels = tuple(sorted(set(candidate_channels_list)))
    scoring_channels = tuple(sorted(set(scoring_channels_list)))
    return candidate_channels, scoring_channels, scoring_specs, semantic


def build_anchor_weighted_edges(
    *,
    nodes_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    semantic_cosine_map: dict[tuple[int, int], float],
    scoring_channels: tuple[str, ...],
    channel_scoring: dict[str, AnchorChannelScoring],
    temporal_decay_days: float = 30.0,
    temporal_overlap_seconds: float = 3600.0,
) -> pd.DataFrame:
    if candidate_df.empty:
        return pd.DataFrame()

    active_channels = [
        _normalize_channel_to_set_col(ch)
        for ch in scoring_channels
        if _normalize_channel_to_set_col(ch) in nodes_df.columns
        and _normalize_channel_to_set_col(ch) in channel_scoring
    ]
    n_docs = max(1, len(nodes_df))
    email_df_maps: dict[str, dict[str, int]] = {}
    for ch in active_channels:
        c = Counter[str]()
        for vals in nodes_df[ch].tolist():
            s = _to_set_cell(vals)
            if s:
                c.update(set(s))
        email_df_maps[ch] = dict(c)

    node_recs = list(nodes_df.to_dict("records"))
    rows: list[dict[str, Any]] = []
    for _, e in candidate_df.iterrows():
        i, j = int(e["idx_a"]), int(e["idx_b"])
        a, b = node_recs[i], node_recs[j]
        pair = (i, j) if i < j else (j, i)
        sem = float(semantic_cosine_map.get(pair, 0.0))

        rec: dict[str, Any] = {
            "email_a": str(a["external_id"]),
            "email_b": str(b["external_id"]),
            "idx_a": int(i),
            "idx_b": int(j),
            "semantic_score": sem,
        }
        active_bases: list[str] = []
        for ch in active_channels:
            base = _base_channel_name(ch)
            spec = channel_scoring[ch]
            sa = _to_set_cell(a.get(ch))
            sb = _to_set_cell(b.get(ch))
            inter = sa & sb
            uni = sa | sb
            cnt = int(len(inter))
            jac = float(cnt / max(1, len(uni)))
            idf_sum = 0.0
            n_cut = 0
            for art in inter:
                df = int(email_df_maps[ch].get(art, 0))
                if spec.max_email_df is not None and df > int(spec.max_email_df):
                    n_cut += 1
                    continue
                raw = _idf_raw(n_docs, df)
                eff = float(spec.idf_scale) * (max(0.0, raw) ** float(spec.idf_exponent))
                idf_sum += eff
            has_overlap = bool(cnt > 0)
            if has_overlap:
                active_bases.append(base)
            rec[f"has_{base}_overlap"] = has_overlap
            rec[f"shared_{base}_count"] = cnt
            rec[f"{base}_jaccard"] = jac
            rec[f"shared_{base}_idf_sum"] = float(idf_sum)
            rec[f"shared_{base}_n_cutoff_filtered"] = int(n_cut)

        t_overlap, t_gap_days, t_score = _temporal_score(
            float(a.get("ts", np.nan)),
            float(b.get("ts", np.nan)),
            decay_days=float(temporal_decay_days),
            overlap_seconds=float(temporal_overlap_seconds),
        )
        rec["temporal_overlap"] = float(t_overlap)
        rec["temporal_gap_days"] = float(t_gap_days) if pd.notna(t_gap_days) else np.nan
        rec["temporal_score"] = float(t_score)
        rec["active_channel_list"] = json.dumps(sorted(active_bases), ensure_ascii=False)
        rec["n_active_channels"] = int(len(active_bases))
        rows.append(rec)
    return pd.DataFrame(rows)


def build_anchor_graph_view(
    edges_df: pd.DataFrame,
    *,
    include_channels: list[str] | None = None,
    exclude_channels: list[str] | None = None,
    require_all_include: bool = False,
    min_active_channels: int = 1,
    min_edge_weight: float | None = None,
) -> pd.DataFrame:
    if edges_df.empty:
        return edges_df.copy()

    out = edges_df.copy()
    m = np.ones(len(out), dtype=bool)
    if min_active_channels is not None:
        m &= pd.to_numeric(out.get("n_active_channels"), errors="coerce").fillna(0).astype(int) >= int(min_active_channels)
    if min_edge_weight is not None and "edge_weight" in out.columns:
        m &= pd.to_numeric(out["edge_weight"], errors="coerce").fillna(0.0) >= float(min_edge_weight)

    inc_cols = []
    for ch in include_channels or []:
        base = _base_channel_name(_normalize_channel_to_set_col(ch))
        col = f"has_{base}_overlap"
        if col in out.columns:
            inc_cols.append(col)
    if inc_cols:
        if require_all_include:
            inc_mask = np.ones(len(out), dtype=bool)
            for c in inc_cols:
                inc_mask &= out[c].astype(bool).to_numpy()
            m &= inc_mask
        else:
            inc_mask = np.zeros(len(out), dtype=bool)
            for c in inc_cols:
                inc_mask |= out[c].astype(bool).to_numpy()
            m &= inc_mask

    exc_cols = []
    for ch in exclude_channels or []:
        base = _base_channel_name(_normalize_channel_to_set_col(ch))
        col = f"has_{base}_overlap"
        if col in out.columns:
            exc_cols.append(col)
    for c in exc_cols:
        m &= ~out[c].astype(bool).to_numpy()

    return out.loc[m].reset_index(drop=True)


def build_anchor_networkx_graph(
    *,
    nodes_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    node_columns: list[str] | None = None,
) -> nx.Graph:
    g = nx.Graph()
    keep_node_cols = list(node_columns or [])
    if not keep_node_cols:
        keep_node_cols = [c for c in nodes_df.columns if c != "external_id"]
    for _, r in nodes_df.iterrows():
        eid = str(r["external_id"])
        attrs: dict[str, Any] = {}
        for c in keep_node_cols:
            v = r.get(c)
            attrs[c] = sorted(v) if isinstance(v, set) else v
        g.add_node(eid, **attrs)
    if not edges_df.empty:
        edge_cols = [c for c in edges_df.columns if c not in {"email_a", "email_b", "idx_a", "idx_b"}]
        for _, r in edges_df.iterrows():
            a = str(r["email_a"])
            b = str(r["email_b"])
            attrs = {c: r.get(c) for c in edge_cols}
            g.add_edge(a, b, **attrs)
    return g


def _channel_prevalence(edges_df: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if edges_df.empty:
        return rows
    has_cols = [c for c in edges_df.columns if c.startswith("has_") and c.endswith("_overlap")]
    for c in sorted(has_cols):
        x = edges_df[c].astype(bool)
        rows.append(
            {
                "channel": c.replace("has_", "").replace("_overlap", ""),
                "n_edges_with_channel": int(x.sum()),
                "frac_edges_with_channel": float(x.mean()),
            }
        )
    return rows


def validate_anchor_graph_artifacts(
    *,
    edges_df: pd.DataFrame,
    required_channels: tuple[str, ...],
    require_edge_weight: bool = True,
) -> dict[str, Any]:
    req = {"email_a", "email_b", "active_channel_list", "n_active_channels"}
    if require_edge_weight:
        req.add("edge_weight")
    missing = sorted(c for c in req if c not in edges_df.columns)
    has_mismatch = 0
    if not edges_df.empty and "active_channel_list" in edges_df.columns and "n_active_channels" in edges_df.columns:
        for _, r in edges_df.iterrows():
            try:
                xs = json.loads(str(r["active_channel_list"]))
            except Exception:
                xs = []
            n = int(r["n_active_channels"]) if pd.notna(r["n_active_channels"]) else 0
            if len(xs) != n:
                has_mismatch += 1
    ch_missing: list[str] = []
    for ch in required_channels:
        base = _base_channel_name(_normalize_channel_to_set_col(ch))
        col = f"has_{base}_overlap"
        if col not in edges_df.columns:
            ch_missing.append(col)
    return {
        "required_columns_missing": missing,
        "required_channel_flags_missing": ch_missing,
        "active_channel_count_mismatch_rows": int(has_mismatch),
        "ok": bool(not missing and not ch_missing and has_mismatch == 0),
    }


def save_anchor_graph_artifacts(
    *,
    output_dir: Path,
    nodes_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    candidates_df: pd.DataFrame,
    summary: dict[str, Any],
    write_candidates: bool = True,
    write_serialized_graph: bool = True,
    graph_obj: nx.Graph | None = None,
) -> dict[str, str]:
    out = output_dir.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    p_nodes = out / "anchor_graph_nodes.csv"
    p_edges = out / "anchor_graph_edges_unscored.csv"
    p_cand = out / "anchor_graph_candidates.csv"
    p_summary = out / "anchor_graph_summary.json"
    p_graph = out / "anchor_graph.graph.pkl"

    node_out = nodes_df.copy()
    for c in node_out.columns:
        if c.endswith("_set"):
            node_out[c] = node_out[c].map(_serialize_set_col)
    node_out.to_csv(p_nodes, index=False)
    edges_df.to_csv(p_edges, index=False)
    if write_candidates:
        candidates_df.to_csv(p_cand, index=False)
    p_summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    paths: dict[str, str] = {
        "nodes_csv": str(p_nodes),
        "edges_unscored_csv": str(p_edges),
        "summary_json": str(p_summary),
    }
    if write_candidates:
        paths["candidates_csv"] = str(p_cand)
    if write_serialized_graph:
        g = graph_obj if graph_obj is not None else build_anchor_networkx_graph(nodes_df=nodes_df, edges_df=edges_df)
        with p_graph.open("wb") as f:
            pickle.dump(g, f)
        paths["graph_pickle"] = str(p_graph)
    return paths


def score_anchor_pairgraph_handcrafted(
    *,
    unscored_df: pd.DataFrame,
    semantic_weight: float,
    infra_weight: float,
    temporal_weight: float,
    score_mode: str = "anchor_handcrafted_v1",
) -> pd.DataFrame:
    """
    Score an unscored anchor PairGraph using the existing handcrafted rule.
    """
    if unscored_df.empty:
        out = unscored_df.copy()
        out["edge_weight"] = pd.Series(dtype=float)
        out["score_mode"] = score_mode
        return out
    out = unscored_df.copy()
    sem = pd.to_numeric(out.get("semantic_score"), errors="coerce").fillna(0.0).clip(lower=0.0, upper=1.0)
    infra = pd.to_numeric(out.get("infra_score"), errors="coerce").fillna(0.0)
    temp = pd.to_numeric(out.get("temporal_score"), errors="coerce").fillna(0.0)
    out["edge_weight"] = (
        float(semantic_weight) * sem
        + float(infra_weight) * infra
        + float(temporal_weight) * temp
    ).astype(float)
    out["score_mode"] = str(score_mode)
    return out


def load_anchor_graph_artifacts(
    output_dir: str | Path,
    *,
    load_graph_pickle: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None, dict[str, Any], nx.Graph | None]:
    d = Path(output_dir).expanduser().resolve()
    nodes = pd.read_csv(d / "anchor_graph_nodes.csv")
    p_edges_unscored = d / "anchor_graph_edges_unscored.csv"
    if not p_edges_unscored.is_file():
        raise FileNotFoundError(
            f"Missing anchor edge table in {d}; expected anchor_graph_edges_unscored.csv"
        )
    edges = pd.read_csv(p_edges_unscored)
    edges = migrate_unscored_graph_id_column(edges)
    p_cand = d / "anchor_graph_candidates.csv"
    candidates = pd.read_csv(p_cand) if p_cand.is_file() else None
    summary = json.loads((d / "anchor_graph_summary.json").read_text(encoding="utf-8"))
    if isinstance(summary, dict) and "graph_run_id" in summary and "graph_id" not in summary:
        summary = dict(summary)
        summary["graph_id"] = summary.pop("graph_run_id")

    for c in nodes.columns:
        if c.endswith("_set"):
            nodes[c] = nodes[c].map(_to_set_cell)

    g: nx.Graph | None = None
    if load_graph_pickle:
        p_graph = d / "anchor_graph.graph.pkl"
        if p_graph.is_file():
            with p_graph.open("rb") as f:
                g = pickle.load(f)
    return nodes, edges, candidates, summary, g


def materialize_anchor_graph_views(
    *,
    base_output_dir: Path,
    nodes_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    views_cfg: list[dict[str, Any]],
    write_serialized_graph: bool = True,
) -> dict[str, Any]:
    views_dir = base_output_dir / "views"
    views_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, Any]] = []
    for v in views_cfg:
        name = str(v.get("name") or "").strip()
        if not name:
            continue
        sub = build_anchor_graph_view(
            edges_df,
            include_channels=list(v.get("include_channels") or []),
            exclude_channels=list(v.get("exclude_channels") or []),
            require_all_include=bool(v.get("require_all_include", False)),
            min_active_channels=int(v.get("min_active_channels", 1)),
            min_edge_weight=(
                None
                if v.get("min_edge_weight") is None
                else float(v.get("min_edge_weight"))
            ),
        )
        p_edges = views_dir / f"{name}_edges.csv"
        sub.to_csv(p_edges, index=False)
        view_paths: dict[str, str] = {"edges_csv": str(p_edges)}
        if write_serialized_graph:
            g = build_anchor_networkx_graph(
                nodes_df=nodes_df,
                edges_df=sub,
                node_columns=["ts"],
            )
            p_graph = views_dir / f"{name}.graph.pkl"
            with p_graph.open("wb") as f:
                pickle.dump(g, f)
            view_paths["graph_pickle"] = str(p_graph)
        summary_rows.append(
            {
                "name": name,
                "n_edges": int(len(sub)),
                "n_nodes_with_edges": int(
                    len(set(sub["email_a"].astype(str)).union(set(sub["email_b"].astype(str))))
                )
                if not sub.empty
                else 0,
                "paths": view_paths,
            }
        )
    p_summary = views_dir / "views_summary.json"
    p_summary.write_text(json.dumps(summary_rows, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"views_summary_json": str(p_summary), "views": summary_rows}


def build_anchor_graph(config: dict[str, Any]) -> dict[str, Any]:
    run_cfg = config.get("run") or {}
    inputs = config.get("inputs") or {}
    filters = config.get("filters") or {}
    cand_cfg = config.get("candidate_generation") or {}
    persistence_cfg = config.get("persistence") or {}
    node_fields_cfg = config.get("node_fields") or {}

    graph_id = resolve_graph_id(run_cfg)
    pbar = tqdm(total=7, desc=f"Building anchor graph [{graph_id}]") if tqdm is not None else None
    try:
        default_paths = gh.resolve_graph_analysis_paths()
        project_root = gh.find_project_root()

        def _resolve_input_path(raw: Any, default_value: Path | None = None) -> Path | None:
            if raw is None or str(raw).strip() == "":
                return default_value
            p = Path(str(raw)).expanduser()
            if not p.is_absolute():
                p = project_root / p
            return p.resolve()

        graph_pt = _resolve_input_path(inputs.get("graph_pt"), default_paths.graph_pt)
        if graph_pt is None:
            raise ValueError("graph_pt must be set in config inputs or resolvable by default.")
        meta_json = Path(
            inputs.get("meta_json")
            or graph_pt.with_suffix(".meta.json")
        ).expanduser().resolve()
        embeddings_json = _resolve_input_path(inputs.get("embeddings_json"), None)
        popular_domains_path = _resolve_input_path(inputs.get("popular_domains_path"), None)
        web_hosting_domains_path = _resolve_input_path(inputs.get("web_hosting_domains_path"), None)

        nodes_df, benign_diag = _build_email_node_table(
            graph_pt=graph_pt,
            meta_json=meta_json,
            to_undirected=bool(inputs.get("to_undirected", True)),
            filter_popular_domains=bool(filters.get("filter_popular_domains", True)),
            popular_domains_path=popular_domains_path,
            filter_web_hosting_domains=bool(filters.get("filter_web_hosting_domains", False)),
            web_hosting_domains_path=web_hosting_domains_path,
            include_text_fields=bool(node_fields_cfg.get("include_text_fields", True)),
        )
        if pbar is not None:
            pbar.update(1)

        embed_source = str(inputs.get("embedding_source", "cache_or_compute"))
        id_to_vec, embedding_meta = load_embedding_vectors(
            nodes_df=nodes_df,
            embeddings_json=embeddings_json,
            embedding_source=embed_source,
            prefer_translated_for_compute=bool(inputs.get("prefer_translated_for_compute", True)),
            tfidf_max_features=int(inputs.get("tfidf_max_features", 4096)),
        )
        if pbar is not None:
            pbar.update(1)

        candidate_channels, scoring_channels, channel_scoring, semantic_cfg = _resolve_unified_channel_configuration(config)
        if pbar is not None:
            pbar.update(1)

        candidates_df, sem_cos_map, candidate_summary = build_anchor_candidates(
            nodes_df=nodes_df,
            id_to_vec=id_to_vec,
            semantic_top_k=(
                int(semantic_cfg["top_k"]) if bool(semantic_cfg["candidate_enabled"]) else 0
            ),
            semantic_min_cos=float(semantic_cfg["min_cos"]),
            candidate_channels=candidate_channels,
            max_pairs_per_artifact=(
                None
                if cand_cfg.get("max_pairs_per_artifact") is None
                else int(cand_cfg.get("max_pairs_per_artifact"))
            ),
        )
        if pbar is not None:
            pbar.update(1)

        edges_df = build_anchor_weighted_edges(
            nodes_df=nodes_df,
            candidate_df=candidates_df,
            semantic_cosine_map=sem_cos_map,
            scoring_channels=scoring_channels,
            channel_scoring=channel_scoring,
            temporal_decay_days=30.0,
            temporal_overlap_seconds=3600.0,
        )
        edges_df["email_i"] = edges_df["email_a"].astype(str)
        edges_df["email_j"] = edges_df["email_b"].astype(str)
        edges_df["graph_kind"] = "anchor"
        edges_df["graph_id"] = graph_id
        # Anchor graph does not use seed/candidate generator provenance; semantic candidate is exposed.
        edges_df["from_seed"] = False
        if "from_semantic_candidate" in edges_df.columns:
            edges_df["from_semantic"] = edges_df["from_semantic_candidate"].fillna(False).astype(bool)
        else:
            edges_df["from_semantic"] = False
        edges_df["from_rare_artifact"] = False
        edges_df["from_component"] = False
        edges_df["from_2hop"] = False
        edges_df["source_count"] = pd.to_numeric(
            edges_df.get("n_active_channels"), errors="coerce"
        ).fillna(0).astype(int)
        if pbar is not None:
            pbar.update(1)

        required_channels = tuple(scoring_channels)
        validation = validate_anchor_graph_artifacts(
            edges_df=edges_df,
            required_channels=required_channels,
            require_edge_weight=False,
        )

        out_base = Path(
            persistence_cfg.get("output_dir")
            or (project_root / "analysis" / "output" / "anchor_graph")
        ).expanduser().resolve()
        out_dir = (out_base / graph_id).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        created_at_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
        run_meta = {
            "created_at_utc": created_at_utc,
            "graph_id": graph_id,
            "output_dir": str(out_dir),
            "config": config,
        }
        p_run_cfg = out_dir / "anchor_graph_run_config.json"
        p_run_cfg.write_text(json.dumps(run_meta, indent=2, ensure_ascii=False), encoding="utf-8")
        write_candidates = bool(persistence_cfg.get("write_candidates", True))
        write_graph = bool(persistence_cfg.get("write_serialized_graph", True))

        node_cols_for_graph = [
            c for c in (node_fields_cfg.get("graph_node_columns") or list(DEFAULT_NODE_COLUMNS))
            if c in nodes_df.columns and c != "external_id"
        ]
        graph_obj = build_anchor_networkx_graph(
            nodes_df=nodes_df,
            edges_df=edges_df,
            node_columns=node_cols_for_graph,
        )
        summary = {
            "graph_id": graph_id,
            "created_at_utc": created_at_utc,
            "output_dir": str(out_dir),
            "run_config_json": str(p_run_cfg),
            "n_nodes": int(len(nodes_df)),
            "n_candidates": int(len(candidates_df)),
            "n_edges": int(len(edges_df)),
            "channel_prevalence": _channel_prevalence(edges_df),
            "candidate_summary": candidate_summary,
            "semantic_config_resolved": semantic_cfg,
            "candidate_channels_resolved": list(candidate_channels),
            "scoring_channels_resolved": list(scoring_channels),
            "url_filter_diagnostics": benign_diag,
            "embedding_meta": embedding_meta,
            "validation": validation,
            "config_snapshot": config,
        }
        paths = save_anchor_graph_artifacts(
            output_dir=out_dir,
            nodes_df=nodes_df,
            edges_df=edges_df,
            candidates_df=candidates_df,
            summary=summary,
            write_candidates=write_candidates,
            write_serialized_graph=write_graph,
            graph_obj=graph_obj,
        )
        if pbar is not None:
            pbar.update(1)

        views_out = {}
        views_cfg = config.get("views") or []
        if views_cfg:
            views_out = materialize_anchor_graph_views(
                base_output_dir=out_dir,
                nodes_df=nodes_df,
                edges_df=edges_df,
                views_cfg=list(views_cfg),
                write_serialized_graph=write_graph,
            )
            paths.update({k: str(v) for k, v in views_out.items() if isinstance(v, str)})
        paths["run_config_json"] = str(p_run_cfg)
        if pbar is not None:
            pbar.update(1)

        return {
            "paths": paths,
            "summary": summary,
            "views": views_out,
        }
    finally:
        if pbar is not None:
            pbar.close()

