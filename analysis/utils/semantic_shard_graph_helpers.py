from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from analysis.utils import graph_structure_helpers as gh
from analysis.utils import semantic_shard_helpers as ssh


FREQUENCY_STAT_NOTE = (
    "Shard-edge infra uses **shard document frequency**: for each artifact value, "
    "`shard_df` counts how many *shards* include that value in their union set "
    "(the same counting scheme as the existing IDF map: one increment per shard per distinct artifact). "
    "Email-level `email_df` counts how many *email rows* contain the value — diagnostic only. "
    "**URL / domain / stem** columns follow **email → url → {domain, stem}** only; popular/benign "
    "domains from `core/feature_set_extraction/caches/popular_domains.txt` drop the whole URL."
)


def is_noise_stem(stem: str) -> bool:
    """
    URL path tokens that should not drive shard–shard stem overlap in analysis.

    These can still exist on the core hetero graph; we drop them only when building
    the semantic shard graph so noisy paths like ``/*`` do not create false bridges.

    Note: many junk tokens are not plain ``/*`` but pasted markup, e.g. ``/*]`` (see
    ``semantic_shard_step2_nodes.csv`` stem_set), so we match glob-like roots and
    bracket-drag variants—not only exact ``/*``.
    """
    s = (stem or "").strip().strip("'\"")
    if not s:
        return True
    if s == "/":
        return True
    if s == "*":
        return True
    # Root-level globs and pasted markup: /*, /**, /*], /**], /], etc. (no real path segment)
    if re.fullmatch(r"/[\*\]]+\s*$", s):
        return True
    return False


def filter_noise_stems_from_set(stems: set[str]) -> set[str]:
    """Return a copy of ``stems`` with noise stems removed."""
    out: set[str] = set()
    for x in stems:
        sx = str(x).strip()
        if sx and not is_noise_stem(sx):
            out.add(sx)
    return out


@dataclass
class ShardEdgeChannelScoring:
    """Per-infrastructure-column scoring controls for shard–shard edge weights."""

    weight: float
    scoring_mode: str = "legacy"  # "legacy" | "routed"
    idf_exponent: float = 1.0
    idf_scale: float = 1.0
    max_shard_df: int | None = None
    contribution_cap: float | None = None

    @classmethod
    def from_mapping(cls, d: dict[str, Any]) -> ShardEdgeChannelScoring:
        cap = d.get("contribution_cap")
        msd = d.get("max_shard_df")
        return cls(
            weight=float(d["weight"]),
            scoring_mode=str(d.get("scoring_mode", "legacy")),
            idf_exponent=float(d.get("idf_exponent", 1.0)),
            idf_scale=float(d.get("idf_scale", 1.0)),
            max_shard_df=None if msd is None else int(msd),
            contribution_cap=None if cap is None else float(cap),
        )

    def to_jsonable(self) -> dict[str, Any]:
        return asdict(self)


def scoring_specs_from_weights_legacy(channel_weights: dict[str, float]) -> dict[str, ShardEdgeChannelScoring]:
    return {
        ch: ShardEdgeChannelScoring(weight=float(w), scoring_mode="legacy")
        for ch, w in channel_weights.items()
    }


def _artifact_shard_df_counter(shard_nodes_df: pd.DataFrame, ch: str) -> dict[str, int]:
    """#shards containing each artifact (distinct values per shard counted once)."""
    dfc: Counter[str] = Counter()
    if ch not in shard_nodes_df.columns:
        return {}
    for s in shard_nodes_df[ch].tolist():
        if isinstance(s, set) and s:
            dfc.update(set(str(x) for x in s))
    return dict(dfc)


def _artifact_email_df_counter(email_df: pd.DataFrame, ch: str) -> dict[str, int]:
    """#email rows containing each artifact."""
    ec: Counter[str] = Counter()
    if ch not in email_df.columns:
        return {}
    for s in email_df[ch].tolist():
        if isinstance(s, set) and s:
            ec.update(set(str(x) for x in s))
    return dict(ec)


def routing_channel_frequency_report(
    shard_nodes_df: pd.DataFrame,
    email_df: pd.DataFrame,
    infra_col: str,
    *,
    max_shard_df: int | None,
    top_k: int = 20,
) -> dict[str, Any]:
    """
    Compact diagnostics for a single infra column: shard vs email frequency, cutoff, top values.
    """
    shard_c = _artifact_shard_df_counter(shard_nodes_df, infra_col)
    email_c = _artifact_email_df_counter(email_df, infra_col)
    if not shard_c:
        return {
            "infra_col": infra_col,
            "n_distinct_artifacts": 0,
            "frac_values_above_max_shard_df": float("nan"),
            "top_by_shard_df": pd.DataFrame(),
            "shard_df_values": np.array([], dtype=np.int64),
        }
    n_art = len(shard_c)
    dropped = sum(1 for c in shard_c.values() if max_shard_df is not None and c > int(max_shard_df))
    rows = []
    for a, sd in shard_c.items():
        rows.append(
            {
                "artifact": a[:120] + ("…" if len(str(a)) > 120 else ""),
                "shard_df": int(sd),
                "email_df": int(email_c.get(a, 0)),
                "would_score": bool(max_shard_df is None or sd <= int(max_shard_df)),
            }
        )
    top_df = pd.DataFrame(rows).sort_values(["shard_df", "email_df"], ascending=[False, False]).head(int(top_k))

    return {
        "infra_col": infra_col,
        "n_distinct_artifacts": int(n_art),
        "max_shard_df_cutoff": max_shard_df,
        "frac_values_above_max_shard_df": float(dropped / max(1, n_art)),
        "top_by_shard_df": top_df,
        "shard_df_values": np.fromiter(shard_c.values(), dtype=np.int64, count=len(shard_c)),
    }


def resolve_shard_edge_channel_scoring(
    *,
    scoring_channels_logical: list[str],
    scoring_spec_by_logical: dict[str, dict[str, Any]],
    logical_to_col: Callable[[str], str],
    available_infra_cols: set[str],
    default_legacy_weight: float = 0.55,
) -> dict[str, ShardEdgeChannelScoring]:
    """
    Build `channel_scoring` keys (infra *_set columns) from notebook logical names.

    If `enabled` is False for a logical channel, it is omitted even when listed in
    `scoring_channels_logical`. Missing specs are filled with a conservative legacy default.
    """
    out: dict[str, ShardEdgeChannelScoring] = {}
    for logical in scoring_channels_logical:
        col = logical_to_col(logical)
        if col not in available_infra_cols:
            continue
        raw = scoring_spec_by_logical.get(logical)
        if raw is not None and raw.get("enabled") is False:
            continue
        if raw is None:
            spec = ShardEdgeChannelScoring(weight=float(default_legacy_weight), scoring_mode="legacy")
        else:
            d = {k: v for k, v in raw.items() if k != "enabled"}
            if "weight" not in d:
                d["weight"] = float(default_legacy_weight)
            spec = ShardEdgeChannelScoring.from_mapping(d)
        out[col] = spec
    return out


def risky_channel_edge_adjustment_table(edges_df: pd.DataFrame, bases: tuple[str, ...]) -> pd.DataFrame:
    """Mean pre-cap vs post-cap infra contribution on edges that have overlap on that channel."""
    rows: list[dict[str, Any]] = []
    for b in bases:
        pre = f"infra_contrib_{b}_pre_cap"
        post = f"infra_contrib_{b}"
        sc = f"shared_{b}_count"
        if post not in edges_df.columns:
            continue
        mask = (edges_df[sc] > 0) if sc in edges_df.columns else (edges_df[post] > 0)
        n = int(mask.sum())
        mpre = float(edges_df.loc[mask, pre].mean()) if n else float("nan")
        mpost = float(edges_df.loc[mask, post].mean()) if n else float("nan")
        rows.append(
            {
                "channel": b,
                "edges_with_overlap": n,
                "mean_pre_cap": mpre,
                "mean_post_cap": mpost,
                "mean_cap_shrink": float(mpre - mpost) if n and (mpre == mpre) and (mpost == mpost) else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def edge_risky_channel_infra_fractions(
    edges_df: pd.DataFrame,
    *,
    risky_bases: tuple[str, ...] = ("origin_ip", "received_host"),
    infra_col: str = "infra_score",
) -> dict[str, float]:
    """Fraction of summed infra_score across edges attributable to listed channels (by infra_contrib_*)."""
    denom = float(np.clip(edges_df[infra_col], 0.0, None).sum()) if infra_col in edges_df.columns else 0.0
    out: dict[str, float] = {"sum_infra_score": denom}
    if denom <= 0.0:
        for b in risky_bases:
            out[f"frac_infra_from_{b}"] = float("nan")
        out["frac_infra_from_risky_listed"] = float("nan")
        return out
    listed = 0.0
    for b in risky_bases:
        c = f"infra_contrib_{b}"
        if c not in edges_df.columns:
            out[f"frac_infra_from_{b}"] = float("nan")
            continue
        s = float(np.clip(edges_df[c], 0.0, None).sum())
        listed += s
        out[f"frac_infra_from_{b}"] = float(s / denom)
    out["frac_infra_from_risky_listed"] = float(listed / denom)
    return out


def _base_channel_name(ch: str) -> str:
    return ch.replace("_set", "")


def _idf_raw(n_shards: int, shard_df: int) -> float:
    return math.log((1.0 + float(n_shards)) / (1.0 + float(shard_df))) + 1.0


def _l2_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0.0] = 1.0
    return x / n


def load_step1_assignments(assignments_csv: str | Path) -> pd.DataFrame:
    p = Path(assignments_csv).expanduser().resolve()
    df = pd.read_csv(p)
    need = {"external_id", "shard_id"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"Step-1 assignments missing required columns: {sorted(missing)}")
    df["external_id"] = df["external_id"].astype(str)
    df["shard_id"] = df["shard_id"].astype(str)
    return df


def _index_to_str(meta: dict[str, Any], node_type: str) -> list[str]:
    xs = (meta.get("node_maps") or {}).get(node_type, {}).get("index_to_string") or []
    return [str(x) for x in xs]


def _build_sender_email_domain_sets(
    data,
    *,
    sender_map: list[str],
    email_domain_map: list[str],
    n_email: int,
) -> list[set[str]]:
    out = [set() for _ in range(n_email)]
    if ("email", "has_sender", "sender") not in data.edge_types:
        return out
    if ("sender", "from_domain", "email_domain") not in data.edge_types:
        return out

    s_to_dom: dict[int, set[str]] = defaultdict(set)
    ei_sd = data["sender", "from_domain", "email_domain"].edge_index
    if ei_sd is not None and ei_sd.numel() > 0:
        for s, d in zip(
            ei_sd[0].detach().cpu().numpy().astype(np.int64),
            ei_sd[1].detach().cpu().numpy().astype(np.int64),
        ):
            if 0 <= int(d) < len(email_domain_map):
                s_to_dom[int(s)].add(email_domain_map[int(d)])

    ei_es = data["email", "has_sender", "sender"].edge_index
    if ei_es is not None and ei_es.numel() > 0:
        for e, s in zip(
            ei_es[0].detach().cpu().numpy().astype(np.int64),
            ei_es[1].detach().cpu().numpy().astype(np.int64),
        ):
            if 0 <= int(e) < n_email:
                if 0 <= int(s) < len(sender_map):
                    out[int(e)].add(sender_map[int(s)])
                out[int(e)].update(s_to_dom.get(int(s), set()))
    return out


def load_email_level_inputs(
    *,
    graph_pt: str | Path,
    meta_json: str | Path,
    to_undirected: bool,
    popular_domains_path: str | Path | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Load per-email infrastructure sets from a heteroGraph ``.pt`` + ``.meta.json``.

    **URL / domain / stem (shard graph):** derived only via ``email → url → {domain, stem}``.
    URLs whose registrable domain is in ``popular_domains.txt`` (tldextract, same as
    ``url_extraction_utils.extract_domain_info``) are **omitted** with their URL/domain/stem
    children for shard-graph purposes. Direct ``email → domain`` / ``email → stem`` edges are
    ignored for those three channels so infra aligns with URL-node provenance.

    Returns ``(email_df, benign_url_diagnostics)``.
    """
    meta = gh.load_meta(meta_json)
    data = gh.load_hetero(graph_pt, to_undirected=to_undirected)
    external_ids = gh.email_external_id_list(meta)
    n_email = len(external_ids)
    ts = (meta.get("email_attrs") or {}).get("ts") or [None] * n_email

    root = gh.find_project_root()
    pop_path = Path(popular_domains_path).expanduser().resolve() if popular_domains_path else gh.default_popular_domains_path(root)
    popular_domains = gh.load_popular_domains_file(pop_path)
    url_sets, domain_sets, stem_sets, benign_diag = gh.build_email_url_derived_infra_sets(
        data,
        meta,
        popular_domains=popular_domains,
        popular_domains_source=pop_path,
    )

    email_sets = gh.build_email_artifact_sets(data)
    skip_email_adjacent = frozenset({"url", "domain", "stem"})
    email_sets = {k: v for k, v in email_sets.items() if k not in skip_email_adjacent}

    rows: list[dict[str, Any]] = []
    # Preload index->string maps for any node types we actually have.
    index_to_str_map: dict[str, list[str]] = {
        node_type: _index_to_str(meta, node_type) for node_type in email_sets.keys()
    }
    # Derived "sender_email_domain_set" depends on these maps even if
    # the corresponding node types are absent from `email_sets`.
    index_to_str_map.setdefault("sender", _index_to_str(meta, "sender"))
    index_to_str_map.setdefault("email_domain", _index_to_str(meta, "email_domain"))

    sender_domain_sets = _build_sender_email_domain_sets(
        data,
        sender_map=index_to_str_map.get("sender") or [],
        email_domain_map=index_to_str_map.get("email_domain") or [],
        n_email=n_email,
    )
    for i, eid in enumerate(external_ids):
        rec: dict[str, Any] = {
            "external_id": str(eid),
            "ts": float(ts[i]) if i < len(ts) and ts[i] is not None else np.nan,
        }

        # URL / domain / stem: only via url-node traversal (+ popular-domain filter).
        rec["url_set"] = set(url_sets[i]) if i < len(url_sets) else set()
        rec["domain_set"] = set(domain_sets[i]) if i < len(domain_sets) else set()
        rec["stem_set"] = filter_noise_stems_from_set(set(stem_sets[i]) if i < len(stem_sets) else set())

        # Other infrastructure: direct email-adjacent artifact nodes (unchanged).
        for node_type, idx_sets in email_sets.items():
            idxs = idx_sets[i] if i < len(idx_sets) else set()
            xs = index_to_str_map.get(node_type) or []
            rec[f"{node_type}_set"] = set(
                xs[int(j)] if 0 <= int(j) < len(xs) else str(int(j))
                for j in idxs
                if j is not None
            )

        # Derived "sender-side email_domain" set (often referenced as sender_email_domain_set).
        rec["sender_email_domain_set"] = set(sender_domain_sets[i])
        rows.append(rec)
    return pd.DataFrame(rows), benign_diag


def _sampled_within_cos(member_vecs: np.ndarray, *, max_pairs: int = 3000) -> tuple[float, float]:
    n = member_vecs.shape[0]
    if n < 2:
        return float("nan"), float("nan")
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    if len(pairs) > max_pairs:
        rng = np.random.default_rng(0)
        choose = rng.choice(len(pairs), size=max_pairs, replace=False)
        pairs = [pairs[int(i)] for i in choose]
    vals = [float(np.dot(member_vecs[i], member_vecs[j])) for i, j in pairs]
    return float(np.mean(vals)), float(np.median(vals))


def build_shard_nodes(
    *,
    assignments_df: pd.DataFrame,
    id_to_semantic: dict[str, np.ndarray],
    email_df: pd.DataFrame,
    gt_label_map: dict[str, Any] | None = None,
    infra_channels: tuple[str, ...] = (
        "sender_set",
        "sender_email_domain_set",
        "url_set",
        "domain_set",
        "stem_set",
        "attachment_set",
    ),
) -> tuple[pd.DataFrame, np.ndarray]:
    gt_map = {str(k): v for k, v in (gt_label_map or {}).items()}
    email_lookup = email_df.set_index("external_id")

    shard_rows: list[dict[str, Any]] = []
    centroids: list[np.ndarray] = []
    for shard_id, g in assignments_df.groupby("shard_id", sort=False):
        members = [str(x) for x in g["external_id"].tolist() if str(x) in id_to_semantic]
        if not members:
            continue
        x = _l2_rows(np.stack([id_to_semantic[e] for e in members]).astype(np.float32))
        centroid = _l2_rows(np.mean(x, axis=0, keepdims=True))[0]
        centroids.append(centroid)

        mean_cos, med_cos = _sampled_within_cos(x)
        dist_centroid = 1.0 - np.clip(x @ centroid, -1.0, 1.0)
        comp_mean = float(np.mean(dist_centroid))
        comp_med = float(np.median(dist_centroid))

        member_ts = email_lookup.reindex(members)["ts"].astype(float).dropna()
        ts_min = float(member_ts.min()) if not member_ts.empty else np.nan
        ts_max = float(member_ts.max()) if not member_ts.empty else np.nan
        ts_span = float(ts_max - ts_min) if pd.notna(ts_min) and pd.notna(ts_max) else np.nan

        def _u(col: str) -> set[str]:
            if col not in email_lookup.columns:
                return set()
            vals = email_lookup.reindex(members)[col].dropna().tolist()
            out: set[str] = set()
            for v in vals:
                if isinstance(v, set):
                    out.update(str(x) for x in v if str(x))
            return out

        # Compute requested infra set-columns for this shard.
        shard_infra_sets: dict[str, set[str]] = {ch: _u(ch) for ch in infra_channels}
        if "stem_set" in shard_infra_sets:
            shard_infra_sets["stem_set"] = filter_noise_stems_from_set(shard_infra_sets["stem_set"])

        # Keep the original core summary columns for interpretability.
        sender_set = shard_infra_sets.get("sender_set", set())
        sender_dom_set = shard_infra_sets.get("sender_email_domain_set", set())
        url_set = shard_infra_sets.get("url_set", set())
        domain_set = shard_infra_sets.get("domain_set", set())
        stem_set = shard_infra_sets.get("stem_set", set())
        att_set = shard_infra_sets.get("attachment_set", set())

        gt_members = [e for e in members if e in gt_map]
        if gt_members:
            ct = Counter(gt_map[e] for e in gt_members)
            dom_campaign, dom_n = ct.most_common(1)[0]
            dom_frac = float(dom_n / len(gt_members))
            n_gt_campaigns = int(len(ct))
        else:
            dom_campaign, dom_frac, n_gt_campaigns = None, np.nan, 0

        shard_rows.append(
            {
                "shard_id": str(shard_id),
                "size": int(len(members)),
                "member_external_ids": members,
                "within_cos_mean": mean_cos,
                "within_cos_median": med_cos,
                "centroid_dist_mean": comp_mean,
                "centroid_dist_median": comp_med,
                "n_unique_senders": int(len(sender_set)),
                "n_unique_sender_email_domains": int(len(sender_dom_set)),
                "n_unique_urls": int(len(url_set)),
                "n_unique_domains": int(len(domain_set)),
                "n_unique_stems": int(len(stem_set)),
                "n_unique_attachments": int(len(att_set)),
                "sender_set": sender_set,
                "sender_email_domain_set": sender_dom_set,
                "url_set": url_set,
                "domain_set": domain_set,
                "stem_set": stem_set,
                "attachment_set": att_set,
                # Also include any extra requested infrastructure channels (as set columns)
                # so candidate generation and scoring can use them.
                **{ch: shard_infra_sets[ch] for ch in infra_channels if ch not in {"sender_set", "sender_email_domain_set", "url_set", "domain_set", "stem_set", "attachment_set"}},
                "ts_min": ts_min,
                "ts_max": ts_max,
                "ts_span_seconds": ts_span,
                "n_members_with_gt": int(len(gt_members)),
                "n_gt_campaigns_touched": n_gt_campaigns,
                "dominant_campaign": dom_campaign,
                "dominant_campaign_fraction": dom_frac,
            }
        )

    node_df = pd.DataFrame(shard_rows)
    centroid_mat = np.stack(centroids).astype(np.float32) if centroids else np.empty((0, 0), dtype=np.float32)
    return node_df, centroid_mat


def build_candidate_edges(
    shard_nodes_df: pd.DataFrame,
    centroid_mat: np.ndarray,
    *,
    semantic_top_k: int = 8,
    semantic_min_cos: float = 0.72,
    candidate_infra_channels: tuple[str, ...] | None = None,
    infra_channels: tuple[str, ...] = (
        "url_set",
        "sender_email_domain_set",
        "domain_set",
        "stem_set",
        "sender_set",
    ),
    show_progress: bool = False,
) -> pd.DataFrame:
    # candidate_infra_channels cleanly separates "which channels create candidates" from
    # "which channels score edges later".
    if candidate_infra_channels is None:
        candidate_infra_channels = infra_channels
    n = len(shard_nodes_df)
    sid = shard_nodes_df["shard_id"].tolist()
    cand: set[tuple[int, int]] = set()
    sims = centroid_mat @ centroid_mat.T if n > 0 else np.empty((0, 0), dtype=np.float32)
    if n > 0:
        np.fill_diagonal(sims, -np.inf)

    # Optional progress bars (tqdm is optional).
    def _iter_with_progress(iterable, *, total: int | None = None, desc: str = ""):
        if not show_progress:
            return iterable
        try:
            from tqdm.auto import tqdm  # type: ignore

            return tqdm(iterable, total=total, desc=desc)
        except Exception:
            return iterable

    # Semantic candidates: top-k neighbors above threshold.
    topk = max(int(semantic_top_k), 0)
    for i in _iter_with_progress(range(n), total=n, desc="Semantic candidates"):
        if topk <= 0:
            continue
        row = sims[i]
        if topk < n:
            part = np.argpartition(-row, topk)[:topk]
            order = part[np.argsort(-row[part])]
        else:
            order = np.argsort(-row)
        for j in order:
            if j <= i:
                continue
            if float(sims[i, j]) >= float(semantic_min_cos):
                cand.add((i, int(j)))

    # Infra candidates (optimized):
    # Build inverted index artifact -> shard indices and connect shard pairs that share any artifact.
    for ch in _iter_with_progress(candidate_infra_channels, total=len(candidate_infra_channels), desc="Infra channels"):
        inv: dict[str, list[int]] = defaultdict(list)
        vals = shard_nodes_df[ch].tolist() if ch in shard_nodes_df.columns else []
        for i, s in enumerate(vals):
            if not isinstance(s, set) or not s:
                continue
            for art in s:
                inv[str(art)].append(i)
        for idxs in _iter_with_progress(inv.values(), total=len(inv), desc=f"Infra pairs {ch}"):
            m = len(idxs)
            if m < 2:
                continue
            # Deduplicate indices in case input sets contain repeated converted values.
            uniq = sorted(set(int(x) for x in idxs))
            for a_i in range(len(uniq)):
                ia = uniq[a_i]
                for b_i in range(a_i + 1, len(uniq)):
                    ib = uniq[b_i]
                    cand.add((ia, ib))

    rows = [
        {"idx_a": i, "idx_b": j, "shard_a": sid[i], "shard_b": sid[j]}
        for i, j in sorted(cand)
    ]
    return pd.DataFrame(rows)


def _temporal_features(a_min: float, a_max: float, b_min: float, b_max: float) -> tuple[float, float]:
    if any(pd.isna(v) for v in (a_min, a_max, b_min, b_max)):
        return 0.0, float("nan")
    overlap = 1.0 if (max(a_min, b_min) <= min(a_max, b_max)) else 0.0
    if overlap > 0:
        return overlap, 0.0
    gap = min(abs(a_min - b_max), abs(b_min - a_max))
    gap_days = gap / 86400.0
    return overlap, float(gap_days)


def build_weighted_edges(
    *,
    shard_nodes_df: pd.DataFrame,
    centroid_mat: np.ndarray,
    candidate_df: pd.DataFrame,
    semantic_weight: float = 0.45,
    infra_weight: float = 0.45,
    temporal_weight: float = 0.10,
    scoring_infra_channels: tuple[str, ...] = (
        "url_set",
        "sender_email_domain_set",
        "domain_set",
        "stem_set",
        "sender_set",
    ),
    channel_weights: dict[str, float] | None = None,
    channel_scoring: dict[str, ShardEdgeChannelScoring] | None = None,
) -> pd.DataFrame:
    if channel_scoring is None:
        if channel_weights is None:
            channel_weights = {
                "url_set": 1.00,
                "sender_email_domain_set": 0.85,
                "domain_set": 0.60,
                "stem_set": 0.55,
                "sender_set": 0.50,
            }
        channel_scoring = scoring_specs_from_weights_legacy(channel_weights)

    # Score only channels present in shard_nodes with explicit scoring specs.
    active_channels = [
        ch
        for ch in scoring_infra_channels
        if ch in shard_nodes_df.columns and ch in channel_scoring
    ]
    n_shards = max(1, len(shard_nodes_df))
    shard_df_maps: dict[str, dict[str, int]] = {
        ch: _artifact_shard_df_counter(shard_nodes_df, ch) for ch in active_channels
    }

    idx_to_row = {i: r for i, r in enumerate(shard_nodes_df.to_dict("records"))}
    sims = centroid_mat @ centroid_mat.T

    rows: list[dict[str, Any]] = []
    for _, e in candidate_df.iterrows():
        i, j = int(e["idx_a"]), int(e["idx_b"])
        ai, bj = idx_to_row[i], idx_to_row[j]
        sem = float(np.clip(sims[i, j], -1.0, 1.0))
        sem_pos = max(0.0, sem)

        # IMPORTANT: scoring uses ONLY exact/shared overlap + frequency-aware weighting.
        # Jaccard is computed as a diagnostic output column and is NOT used in infra_score.
        infra_score = 0.0
        rec: dict[str, Any] = {
            "shard_a": ai["shard_id"],
            "shard_b": bj["shard_id"],
            "centroid_cosine": sem,
        }
        for ch in active_channels:
            spec = channel_scoring[ch]
            w = float(spec.weight)
            mode = str(spec.scoring_mode).lower().strip()
            if mode not in {"legacy", "routed"}:
                raise ValueError(f"Unknown scoring_mode {spec.scoring_mode!r} for channel {ch!r}")

            sa = ai[ch] if ch in ai and isinstance(ai[ch], set) else set()
            sb = bj[ch] if ch in bj and isinstance(bj[ch], set) else set()
            inter = sa & sb
            uni = sa | sb
            cnt = int(len(inter))
            jac = float(cnt / max(1, len(uni)))  # diagnostic only (not used in score)
            shard_c = shard_df_maps[ch]

            n_cut = 0
            # legacy: sum of raw log-idf weights on kept overlap; routed: sum of effective_idf = scale * raw^exp
            idf_diag_sum = 0.0
            contrib_pre_cap = 0.0

            if cnt > 0:
                if mode == "legacy":
                    idf_sum = 0.0
                    for x in inter:
                        df = int(shard_c.get(x, 0))
                        if spec.max_shard_df is not None and df > int(spec.max_shard_df):
                            n_cut += 1
                            continue
                        idf_sum += _idf_raw(n_shards, df)
                    idf_diag_sum = float(idf_sum)
                    contrib_pre_cap = w * (1.0 - math.exp(-idf_sum)) if idf_sum > 0 else 0.0
                else:
                    # routed: per-artifact saturated pieces, stronger transform, then optional cap
                    acc = 0.0
                    for x in inter:
                        df = int(shard_c.get(x, 0))
                        if spec.max_shard_df is not None and df > int(spec.max_shard_df):
                            n_cut += 1
                            continue
                        raw = _idf_raw(n_shards, df)
                        eff = float(spec.idf_scale) * (max(0.0, raw) ** float(spec.idf_exponent))
                        idf_diag_sum += eff
                        acc += w * (1.0 - math.exp(-eff))
                    contrib_pre_cap = float(acc)

                contrib = (
                    min(contrib_pre_cap, float(spec.contribution_cap))
                    if spec.contribution_cap is not None
                    else contrib_pre_cap
                )
            else:
                contrib = 0.0
                contrib_pre_cap = 0.0

            infra_score += contrib
            base = _base_channel_name(ch)
            rec[f"shared_{base}_count"] = cnt
            rec[f"{base}_jaccard"] = jac
            rec[f"shared_{base}_idf_sum"] = float(idf_diag_sum)
            rec[f"shared_{base}_n_cutoff_filtered"] = int(n_cut)
            rec[f"infra_contrib_{base}_pre_cap"] = float(contrib_pre_cap)
            rec[f"infra_contrib_{base}"] = float(contrib)

        t_overlap, t_gap_days = _temporal_features(ai["ts_min"], ai["ts_max"], bj["ts_min"], bj["ts_max"])
        temporal_score = (
            1.0 if t_overlap > 0 else (math.exp(-t_gap_days / 30.0) if pd.notna(t_gap_days) else 0.0)
        )

        # Explicit interpretable weighted formula.
        # edge_weight = 0.45 * semantic + 0.45 * infra + 0.10 * temporal
        edge_weight = (float(semantic_weight) * sem_pos) + (float(infra_weight) * infra_score) + (float(temporal_weight) * temporal_score)
        rec.update(
            {
                "infra_score": float(infra_score),
                "temporal_overlap": float(t_overlap),
                "temporal_gap_days": float(t_gap_days) if pd.notna(t_gap_days) else np.nan,
                "temporal_score": float(temporal_score),
                "edge_weight": float(edge_weight),
            }
        )
        rows.append(rec)
    return pd.DataFrame(rows)


def graph_component_sizes(node_ids: list[str], edges_df: pd.DataFrame) -> list[int]:
    idx = {s: i for i, s in enumerate(node_ids)}
    parent = list(range(len(node_ids)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for _, r in edges_df.iterrows():
        a = idx.get(str(r["shard_a"]))
        b = idx.get(str(r["shard_b"]))
        if a is None or b is None:
            continue
        union(a, b)

    comp = Counter(find(i) for i in range(len(node_ids)))
    return sorted([int(v) for v in comp.values()], reverse=True)


def campaign_bridgeability_from_shard_graph(
    assignments_df: pd.DataFrame,
    gt_label_map: dict[str, Any],
    edges_df: pd.DataFrame,
) -> pd.DataFrame:
    gt = {str(k): v for k, v in gt_label_map.items()}
    adf = assignments_df.copy()
    adf["external_id"] = adf["external_id"].astype(str)
    adf = adf[adf["external_id"].isin(set(gt.keys()))].copy()
    adf["campaign_id"] = adf["external_id"].map(gt)
    edge_pairs = {
        tuple(sorted((str(r["shard_a"]), str(r["shard_b"]))))
        for _, r in edges_df.iterrows()
    }

    rows: list[dict[str, Any]] = []
    for cid, g in adf.groupby("campaign_id", sort=False):
        shards = sorted(set(g["shard_id"].astype(str)))
        if len(shards) < 2:
            continue
        pairs = list(combinations(shards, 2))
        bridged = sum(1 for a, b in pairs if tuple(sorted((a, b))) in edge_pairs)
        rows.append(
            {
                "campaign_id": cid,
                "campaign_size": int(len(g)),
                "n_shards_touched": int(len(shards)),
                "n_shard_pairs": int(len(pairs)),
                "n_bridged_pairs": int(bridged),
                "frac_bridged_pairs": float(bridged / max(1, len(pairs))),
                "has_any_bridge": bool(bridged > 0),
                "all_pairs_bridged": bool(bridged == len(pairs)),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["frac_bridged_pairs", "campaign_size"],
        ascending=[True, False],
    ).reset_index(drop=True)


def save_step2_graph_artifacts(
    *,
    output_dir: str | Path,
    shard_nodes_df: pd.DataFrame,
    centroid_mat: np.ndarray,
    edges_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    graph_config_summary: dict[str, Any] | None = None,
) -> dict[str, str]:
    out = Path(output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    p_nodes = out / "semantic_shard_step2_nodes.csv"
    p_centroids = out / "semantic_shard_step2_centroids.npy"
    p_edges = out / "semantic_shard_step2_edges_weighted.csv"
    p_candidates = out / "semantic_shard_step2_candidates.csv"
    p_summary = out / "semantic_shard_step2_graph_summary.json"

    node_out = shard_nodes_df.copy()
    # Serialize set-valued infrastructure columns so the nodes CSV remains readable.
    # Any column ending with "_set" is treated as set-like in this prototype.
    for c in [c for c in node_out.columns if c.endswith("_set") or c == "sender_email_domain_set"]:
        if c not in node_out.columns:
            continue
        node_out[c] = node_out[c].map(
            lambda v: json.dumps(sorted(list(v))) if isinstance(v, set) else v
        )
    node_out.to_csv(p_nodes, index=False)
    np.save(p_centroids, centroid_mat)
    edges_df.to_csv(p_edges, index=False)
    candidate_df.to_csv(p_candidates, index=False)

    summary = {
        "n_nodes": int(len(shard_nodes_df)),
        "n_candidates": int(len(candidate_df)),
        "n_edges": int(len(edges_df)),
    }
    if graph_config_summary:
        summary["graph_config_summary"] = graph_config_summary
    p_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {
        "nodes_csv": str(p_nodes),
        "centroids_npy": str(p_centroids),
        "edges_csv": str(p_edges),
        "candidates_csv": str(p_candidates),
        "graph_summary_json": str(p_summary),
    }
