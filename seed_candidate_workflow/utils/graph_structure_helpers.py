"""
Exploratory helpers for heterogeneous email graphs (PyG HeteroData + companion .meta.json).

Alignment (canonical for this repo):
- Row i of `data["email"]` matches `meta["email_attrs"]["external_id"][i]` (same order as graph build).
- `meta["node_maps"][node_type]["index_to_string"]` lists human-readable keys for non-email nodes
  (where applicable).
"""
from __future__ import annotations

import json
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

# Optional heavy deps imported lazily where needed

# Conceptual groups for campaign-focused interpretation (not mutually exhaustive).
CORE_INFRA_ARTIFACT_TYPES: frozenset[str] = frozenset(
    {"sender", "url", "domain", "stem", "email_domain"}
)
ROUTING_NOISY_ARTIFACT_TYPES: frozenset[str] = frozenset(
    {"receiver", "origin_ip", "received_host", "return_path_email", "return_path_domain"}
)


def find_project_root(start: Path | None = None) -> Path:
    p = (start or Path.cwd()).resolve()
    for d in (p, *p.parents):
        if (d / "pipeline_config.json").is_file():
            return d
    raise FileNotFoundError("pipeline_config.json not found; run from repo root or seed_candidate_workflow/.")


def ensure_core_gnn_on_path(project_root: Path) -> None:
    core = project_root / "core"
    gnn = core / "GNN"
    for x in (core, gnn):
        s = str(x.resolve())
        if s not in sys.path:
            sys.path.insert(0, s)


@dataclass
class GraphAnalysisPaths:
    project_root: Path
    graph_pt: Path
    meta_json: Path
    ground_truth_json: Path
    to_undirected: bool


def resolve_graph_analysis_paths(project_root: Path | None = None) -> GraphAnalysisPaths:
    root = project_root or find_project_root()
    ensure_core_gnn_on_path(root)
    from config.pipeline_config import default_hetero_graph_pt_path, load_pipeline_config

    cfg = load_pipeline_config(project_root=root)
    graph_pt = Path(default_hetero_graph_pt_path(project_root=root))
    # Campaign analyses use deduplicated GT by default (first label per external_id).
    dedup = root / "data" / "groundtruth" / "ground_truth_dedup.json"
    if dedup.is_file():
        gt = dedup
    else:
        gt_raw = (cfg.get("datasets") or {}).get("ground_truth_json")
        if not gt_raw:
            raise ValueError(
                "No ground_truth_dedup.json under data/groundtruth/ and "
                "pipeline_config.datasets.ground_truth_json missing."
            )
        gt = Path(gt_raw)
        if not gt.is_absolute():
            gt = root / gt
    return GraphAnalysisPaths(
        project_root=root,
        graph_pt=graph_pt,
        meta_json=graph_pt.with_suffix(".meta.json"),
        ground_truth_json=gt,
        to_undirected=bool(cfg.get("to_undirected", True)),
    )


def load_meta(meta_path: Path | str) -> dict[str, Any]:
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_hetero(graph_pt: Path | str, *, to_undirected: bool):
    ensure_core_gnn_on_path(find_project_root())
    from src.load_graph_data import load_hetero_pt

    return load_hetero_pt(str(graph_pt), to_undirected=to_undirected)


def email_external_id_list(meta: dict[str, Any]) -> list[str]:
    xs = meta.get("email_attrs", {}).get("external_id")
    if xs is None:
        raise ValueError("meta missing email_attrs.external_id")
    out = []
    for x in xs:
        if hasattr(x, "item"):
            out.append(str(x.item()))
        else:
            out.append(str(x))
    return out


def external_id_to_row(meta: dict[str, Any]) -> dict[str, int]:
    ids = email_external_id_list(meta)
    return {eid: i for i, eid in enumerate(ids)}


def infer_num_nodes(data, node_type: str) -> int:
    if hasattr(data[node_type], "num_nodes") and data[node_type].num_nodes is not None:
        return int(data[node_type].num_nodes)
    if "x" in data[node_type] and data[node_type].x is not None:
        return int(data[node_type].x.size(0))
    return 0


def graph_overview_counts(data) -> tuple[dict[str, Any], list[str], list]:
    """Return summary dict, node_types, edge_types."""
    ntypes = list(data.node_types)
    etypes = list(data.edge_types)
    nn = sum(infer_num_nodes(data, nt) for nt in ntypes)
    ne = 0
    for et in etypes:
        ei = data[et].edge_index
        if ei is not None and ei.numel() > 0:
            ne += int(ei.size(1))
    summary = {
        "n_node_types": len(ntypes),
        "n_edge_types": len(etypes),
        "total_nodes": nn,
        "total_undirected_edge_stores": ne,
    }
    return summary, ntypes, etypes


def edge_counts_per_type(data) -> list[tuple[tuple[str, str, str], int]]:
    rows = []
    for et in data.edge_types:
        ei = data[et].edge_index
        n = int(ei.size(1)) if ei is not None and ei.numel() > 0 else 0
        rows.append((et, n))
    return sorted(rows, key=lambda x: -x[1])


def total_degree_per_node(data, node_type: str) -> np.ndarray:
    """Undirected total degree: count each incident edge end once per edge store."""
    n = infer_num_nodes(data, node_type)
    deg = np.zeros(n, dtype=np.int64)
    for (src_t, rel, dst_t) in data.edge_types:
        ei = data[src_t, rel, dst_t].edge_index
        if ei is None or ei.numel() == 0:
            continue
        src = ei[0].numpy()
        dst = ei[1].numpy()
        if src_t == node_type:
            np.add.at(deg, src, 1)
        if dst_t == node_type:
            np.add.at(deg, dst, 1)
    return deg


def degree_summary_stats(deg: np.ndarray) -> dict[str, float]:
    if deg.size == 0:
        return {
            "n_nodes": 0,
            "mean_deg": float("nan"),
            "median_deg": float("nan"),
            "pct_deg_eq_1": float("nan"),
            "pct_deg_gt_1": float("nan"),
        }
    return {
        "n_nodes": int(deg.size),
        "mean_deg": float(deg.mean()),
        "median_deg": float(np.median(deg)),
        "pct_deg_eq_1": float((deg == 1).mean()),
        "pct_deg_gt_1": float((deg > 1).mean()),
    }


def _edge_neighbors_email_to(
    data, dst_type: str
) -> list[tuple[str, str]]:
    """Return list of (rel, dst) for edges (email, rel, dst) where dst == dst_type."""
    out = []
    for (src_t, rel, dst_t) in data.edge_types:
        if src_t == "email" and dst_t == dst_type:
            out.append((rel, dst_type))
    return out


def build_email_artifact_sets(
    data,
) -> dict[str, list[set[int]]]:
    """
    For each artifact node type T connected directly from `email`, build a list of sets:
    email_idx -> set of T node indices.
    Special case `email_domain`: union of domains reachable via sender or receiver
    (edges sender/receiver --from_domain--> email_domain).
    """
    n_email = infer_num_nodes(data, "email")
    out: dict[str, list[set[int]]] = defaultdict(lambda: [set() for _ in range(n_email)])

    for (src_t, rel, dst_t) in data.edge_types:
        ei = data[src_t, rel, dst_t].edge_index
        if ei is None or ei.numel() == 0:
            continue
        if src_t == "email" and dst_t != "email":
            emails = ei[0].numpy().astype(np.int64)
            arts = ei[1].numpy().astype(np.int64)
            buckets = out[dst_t]
            for e, a in zip(emails, arts):
                buckets[int(e)].add(int(a))
        elif dst_t == "email" and src_t != "email":
            arts = ei[0].numpy().astype(np.int64)
            emails = ei[1].numpy().astype(np.int64)
            buckets = out[src_t]
            for a, e in zip(arts, emails):
                buckets[int(e)].add(int(a))

    s_to_d: dict[int, set[int]] = defaultdict(set)
    if ("sender", "from_domain", "email_domain") in data.edge_types:
        ei_sd = data["sender", "from_domain", "email_domain"].edge_index
        if ei_sd is not None and ei_sd.numel() > 0:
            for s, d in zip(
                ei_sd[0].numpy().astype(np.int64),
                ei_sd[1].numpy().astype(np.int64),
            ):
                s_to_d[int(s)].add(int(d))
    if s_to_d and ("email", "has_sender", "sender") in data.edge_types:
        ei_es = data["email", "has_sender", "sender"].edge_index
        if ei_es is not None and ei_es.numel() > 0:
            buckets_ed = out["email_domain"]
            for e, s in zip(
                ei_es[0].numpy().astype(np.int64),
                ei_es[1].numpy().astype(np.int64),
            ):
                for dom in s_to_d.get(int(s), ()):
                    buckets_ed[int(e)].add(dom)

    r_to_d: dict[int, set[int]] = defaultdict(set)
    if ("receiver", "from_domain", "email_domain") in data.edge_types:
        ei_rd = data["receiver", "from_domain", "email_domain"].edge_index
        if ei_rd is not None and ei_rd.numel() > 0:
            for r, d in zip(
                ei_rd[0].numpy().astype(np.int64),
                ei_rd[1].numpy().astype(np.int64),
            ):
                r_to_d[int(r)].add(int(d))
    if r_to_d and ("email", "has_receiver", "receiver") in data.edge_types:
        ei_er = data["email", "has_receiver", "receiver"].edge_index
        if ei_er is not None and ei_er.numel() > 0:
            buckets_ed = out["email_domain"]
            for e, r in zip(
                ei_er[0].numpy().astype(np.int64),
                ei_er[1].numpy().astype(np.int64),
            ):
                for dom in r_to_d.get(int(r), ()):
                    buckets_ed[int(e)].add(dom)

    return dict(out)


def default_popular_domains_path(project_root: Path | None = None) -> Path:
    root = project_root if project_root is not None else find_project_root()
    return (root / "core" / "feature_set_extraction" / "caches" / "popular_domains.txt").resolve()


def load_popular_domains_file(path: Path | str) -> frozenset[str]:
    """Lowercase registrable-domain strings, one per line (comments/empty skipped)."""
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(f"popular domains file not found: {p}")
    out: set[str] = set()
    for raw in p.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        out.add(line.lower())
    return frozenset(out)


def _ensure_repo_root_importable() -> Path:
    root = find_project_root()
    s = str(root)
    if s not in sys.path:
        sys.path.insert(0, s)
    return root


def url_matches_popular_domain(url_string: str, popular_domains: frozenset[str]) -> tuple[bool, str]:
    """
    True if the URL's registrable domain (eTLD+1 via tldextract) is listed in ``popular_domains``.

    Uses the same robust parsing as ``shard_url_infra_classify`` (no crash on noisy URLs).
    """
    if not url_string or not popular_domains:
        return False, ""
    _ensure_repo_root_importable()
    from core.feature_set_extraction.url_extraction_utils import shard_url_infra_classify

    kind, reg = shard_url_infra_classify(str(url_string).strip(), popular_domains)
    if kind == "benign":
        return True, reg
    return False, reg or ""


def node_index_to_strings(meta: dict[str, Any], node_type: str) -> list[str]:
    xs = (meta.get("node_maps") or {}).get(node_type, {}).get("index_to_string") or []
    return [str(x) for x in xs]


def iter_email_url_index_pairs(data) -> list[tuple[int, int]]:
    """All (email_idx, url_idx) for ``has_url`` in either direction (handles ToUndirected)."""
    pairs: list[tuple[int, int]] = []
    for (src_t, rel, dst_t) in data.edge_types:
        if rel != "has_url":
            continue
        ei = data[src_t, rel, dst_t].edge_index
        if ei is None or ei.numel() == 0:
            continue
        e0 = ei[0].detach().cpu().numpy().astype(np.int64)
        e1 = ei[1].detach().cpu().numpy().astype(np.int64)
        if src_t == "email" and dst_t == "url":
            for e, u in zip(e0, e1, strict=False):
                pairs.append((int(e), int(u)))
        elif src_t == "url" and dst_t == "email":
            for u, e in zip(e0, e1, strict=False):
                pairs.append((int(e), int(u)))
    return pairs


def _url_to_neighbor_indices(data, src_t: str, rel: str, dst_t: str) -> dict[int, set[int]]:
    out: dict[int, set[int]] = defaultdict(set)
    if (src_t, rel, dst_t) not in data.edge_types:
        return {}
    ei = data[src_t, rel, dst_t].edge_index
    if ei is None or ei.numel() == 0:
        return {}
    a = ei[0].detach().cpu().numpy().astype(np.int64)
    b = ei[1].detach().cpu().numpy().astype(np.int64)
    for x, y in zip(a, b, strict=False):
        out[int(x)].add(int(y))
    return dict(out)


def stem_strings_for_url(
    url_idx: int,
    url_str: str,
    url_to_stems: Mapping[int, set[int]],
    stem_map: list[str],
) -> list[str]:
    """Stem labels from ``url -> stem`` edges, else ``parse_url_components`` fallback."""
    stems: list[str] = []
    for sidx in url_to_stems.get(int(url_idx), ()):
        if 0 <= int(sidx) < len(stem_map):
            t = str(stem_map[int(sidx)]).strip()
            if t:
                stems.append(t)
    if stems:
        return stems
    if not url_str:
        return []
    _ensure_repo_root_importable()
    from core.preprocessing.utils.url_extractor import parse_url_components

    fb = (parse_url_components(str(url_str).strip()).get("stem") or "").strip()
    return [fb] if fb else []


def domain_strings_for_url(
    url_idx: int,
    url_str: str,
    url_to_doms: Mapping[int, set[int]],
    domain_map: list[str],
) -> list[str]:
    """Domain labels from ``url -> domain`` edges, else ``parse_url_components`` fallback (host)."""
    doms: list[str] = []
    for didx in url_to_doms.get(int(url_idx), ()):
        if 0 <= int(didx) < len(domain_map):
            t = str(domain_map[int(didx)]).strip()
            if t:
                doms.append(t)
    if doms:
        return doms
    if not url_str:
        return []
    _ensure_repo_root_importable()
    from core.preprocessing.utils.url_extractor import parse_url_components

    fb = (parse_url_components(str(url_str).strip()).get("domain") or "").strip()
    return [fb.lower()] if fb else []


def build_email_url_derived_infra_sets(
    data,
    meta: dict[str, Any],
    *,
    popular_domains: frozenset[str],
    popular_domains_source: str | Path | None = None,
) -> tuple[list[set[str]], list[set[str]], list[set[str]], dict[str, Any]]:
    """
    Per-email ``url_set``, ``domain_set``, and ``stem_set`` using only **url nodes**:
    ``email -> url -> {domain, stem}``.

    URLs whose registrable domain matches ``popular_domains`` are **skipped** entirely (no URL,
    domain, or stem from that URL contributes). Uses robust host parsing in
    ``url_extraction_utils.parse_url_host_and_registrable_domain`` (denoise + safe urlparse +
    fallback; never raises). URLs that still have **no recoverable hostname** are **skipped**
    (malformed) and counted in diagnostics — not included as benign or kept.

    Returns:
        url_sets, domain_sets, stem_sets (each length = n_email), and a diagnostics dict.
    """
    n_email = infer_num_nodes(data, "email")
    url_map = node_index_to_strings(meta, "url")
    domain_map = node_index_to_strings(meta, "domain")
    stem_map = node_index_to_strings(meta, "stem")

    url_to_doms_raw = _url_to_neighbor_indices(data, "url", "has_domain", "domain")
    url_to_stems_raw = _url_to_neighbor_indices(data, "url", "has_stem", "stem")

    email_to_urls: dict[int, set[int]] = defaultdict(set)
    for e_idx, u_idx in iter_email_url_index_pairs(data):
        if 0 <= int(e_idx) < n_email:
            email_to_urls[int(e_idx)].add(int(u_idx))

    all_url_nodes: set[int] = set()
    for s in email_to_urls.values():
        all_url_nodes |= s

    benign_url_nodes: set[int] = set()
    malformed_url_nodes: set[int] = set()
    benign_hits: Counter[str] = Counter()
    _ensure_repo_root_importable()
    from core.feature_set_extraction.url_extraction_utils import shard_url_infra_classify

    for u in all_url_nodes:
        u_str = url_map[int(u)] if 0 <= int(u) < len(url_map) else ""
        kind, reg = shard_url_infra_classify(u_str, popular_domains)
        if kind == "benign":
            benign_url_nodes.add(int(u))
            if reg:
                benign_hits[reg] += 1
        elif kind == "malformed":
            malformed_url_nodes.add(int(u))

    url_sets = [set() for _ in range(n_email)]
    domain_sets = [set() for _ in range(n_email)]
    stem_sets = [set() for _ in range(n_email)]

    pairs_total = 0
    pairs_benign = 0
    pairs_malformed = 0

    doms_from_benign_urls: set[str] = set()
    doms_from_kept_urls: set[str] = set()
    stems_from_benign_urls: set[str] = set()
    stems_from_kept_urls: set[str] = set()

    for e_idx, url_idxs in email_to_urls.items():
        if e_idx < 0 or e_idx >= n_email:
            continue
        for u in url_idxs:
            pairs_total += 1
            u_str = url_map[int(u)] if 0 <= int(u) < len(url_map) else ""
            if int(u) in malformed_url_nodes:
                pairs_malformed += 1
                continue
            if int(u) in benign_url_nodes:
                pairs_benign += 1
                for d in domain_strings_for_url(u, u_str, url_to_doms_raw, domain_map):
                    if d:
                        doms_from_benign_urls.add(d.lower())
                for st in stem_strings_for_url(u, u_str, url_to_stems_raw, stem_map):
                    if st:
                        stems_from_benign_urls.add(st)
                continue

            if u_str:
                url_sets[e_idx].add(str(u_str).strip())
            for d in domain_strings_for_url(u, u_str, url_to_doms_raw, domain_map):
                if not d:
                    continue
                dl = d.lower()
                domain_sets[e_idx].add(dl)
                doms_from_kept_urls.add(dl)
            for st in stem_strings_for_url(u, u_str, url_to_stems_raw, stem_map):
                if not st:
                    continue
                stem_sets[e_idx].add(st)
                stems_from_kept_urls.add(st)

    n_urls_kept_distinct = len(all_url_nodes - benign_url_nodes - malformed_url_nodes)

    diag: dict[str, Any] = {
        "popular_domains_source": str(Path(popular_domains_source).resolve())
        if popular_domains_source
        else "",
        "n_popular_domains_loaded": len(popular_domains),
        "n_emails": int(n_email),
        "n_distinct_url_nodes_reachable_from_emails": len(all_url_nodes),
        "n_distinct_url_nodes_flagged_benign_popular": len(benign_url_nodes),
        "n_distinct_url_nodes_malformed_unparsed": len(malformed_url_nodes),
        "n_distinct_url_nodes_used_for_shard_graph": int(n_urls_kept_distinct),
        "n_email_url_incidence_pairs": int(pairs_total),
        "n_email_url_pairs_filtered_benign": int(pairs_benign),
        "n_email_url_pairs_skipped_malformed": int(pairs_malformed),
        "n_email_url_pairs_used": int(pairs_total - pairs_benign - pairs_malformed),
        "n_unique_domain_strings_only_from_benign_urls": len(doms_from_benign_urls - doms_from_kept_urls),
        "n_unique_stem_strings_only_from_benign_urls": len(stems_from_benign_urls - stems_from_kept_urls),
        "top_registrable_domains_among_benign_urls": benign_hits.most_common(15),
    }

    return url_sets, domain_sets, stem_sets, diag


def artifact_inverse_email_sets(
    email_sets: dict[str, list[set[int]]],
    artifact_type: str,
) -> dict[int, set[int]]:
    inv: dict[int, set[int]] = defaultdict(set)
    if artifact_type not in email_sets:
        return {}
    for eid, arts in enumerate(email_sets[artifact_type]):
        for a in arts:
            inv[a].add(eid)
    return dict(inv)


def artifact_unique_email_attachment_summary(
    email_sets: dict[str, list[set[int]]],
    artifact_type: str,
) -> dict[str, Any]:
    """
    For one artifact type: distribution of **unique email indices** per artifact node
    (undirected-edge doubling on raw degree does not change these sets).
    """
    inv = artifact_inverse_email_sets(email_sets, artifact_type)
    if not inv:
        return {
            "artifact_type": artifact_type,
            "n_artifact_nodes": 0,
            "mean_unique_emails_per_artifact": float("nan"),
            "median_unique_emails_per_artifact": float("nan"),
            "pct_artifacts_one_email": float("nan"),
            "pct_artifacts_gt_one_email": float("nan"),
        }
    sizes = [len(s) for s in inv.values()]
    p1 = float(np.mean([1 if s == 1 else 0 for s in sizes]))
    return {
        "artifact_type": artifact_type,
        "n_artifact_nodes": len(inv),
        "mean_unique_emails_per_artifact": float(np.mean(sizes)),
        "median_unique_emails_per_artifact": float(np.median(sizes)),
        "pct_artifacts_one_email": p1,
        "pct_artifacts_gt_one_email": float(1.0 - p1),
    }


def sample_campaign_email_pairs(
    campaign_to_members: dict[Any, list[str]],
    eid_row: dict[str, int],
    *,
    max_same_pairs: int = 4000,
    max_diff_pairs: int = 4000,
    seed: int = 0,
    min_camp_size: int = 2,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Labeled same-campaign vs different-campaign email index pairs (aligned to eid_row)."""
    rng = random.Random(seed)
    camp_lists: dict[Any, list[int]] = {}
    for cid, members in campaign_to_members.items():
        idxs = [eid_row[e] for e in members if e in eid_row]
        if len(idxs) >= min_camp_size:
            camp_lists[cid] = idxs
    same_pairs: list[tuple[int, int]] = []
    for idxs in camp_lists.values():
        if len(idxs) < 2:
            continue
        pairs_c = [(idxs[a], idxs[b]) for a in range(len(idxs)) for b in range(a + 1, len(idxs))]
        rng.shuffle(pairs_c)
        same_pairs.extend(pairs_c[:200])
    rng.shuffle(same_pairs)
    same_pairs = same_pairs[:max_same_pairs]

    camps = list(camp_lists.keys())
    diff_pairs: list[tuple[int, int]] = []
    attempts = 0
    while len(diff_pairs) < max_diff_pairs and attempts < max_diff_pairs * 25:
        attempts += 1
        if len(camps) < 2:
            break
        c1, c2 = rng.sample(camps, 2)
        if c1 == c2:
            continue
        i = rng.choice(camp_lists[c1])
        j = rng.choice(camp_lists[c2])
        if i != j:
            diff_pairs.append((i, j) if i < j else (j, i))
    diff_pairs = list(dict.fromkeys(diff_pairs))[:max_diff_pairs]
    return same_pairs, diff_pairs


def summarize_pair_overlap(
    pairs: list[tuple[int, int]],
    email_sets: dict[str, list[set[int]]],
    compare_types: list[str],
) -> dict[str, Any]:
    if not pairs:
        return {t: 0.0 for t in compare_types} | {"n_pairs": 0, "multi_share_mean": 0.0}
    hit = {t: 0 for t in compare_types}
    multi: list[int] = []
    for i, j in pairs:
        d = pair_shares_artifact(i, j, email_sets, compare_types)
        for t in compare_types:
            if d[t]:
                hit[t] += 1
        multi.append(sum(1 for t in compare_types if d[t]))
    n = len(pairs)
    return {t: hit[t] / n for t in compare_types} | {
        "n_pairs": n,
        "multi_share_mean": float(np.mean(multi)) if multi else 0.0,
    }


def lift_rows_from_summaries(
    same_stats: dict[str, Any],
    diff_stats: dict[str, Any],
    channels: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for t in channels:
        sr = float(same_stats.get(t, 0))
        dr = float(diff_stats.get(t, 0))
        lift = (sr / dr) if dr > 1e-12 else float("nan")
        rows.append(
            {
                "channel": t,
                "same_campaign_rate": sr,
                "random_pair_rate": dr,
                "lift": lift,
            }
        )
    return rows


def shared_channel_counts_for_pairs(
    pairs: list[tuple[int, int]],
    email_sets: dict[str, list[set[int]]],
    compare_types: list[str],
) -> list[int]:
    return [n_shared_types(i, j, email_sets, compare_types) for i, j in pairs]


def artifact_bridge_stats(
    email_sets: dict[str, list[set[int]]],
    *,
    artifact_type: str,
    n_email: int,
) -> dict[str, Any]:
    """Email–email connectivity induced by sharing an artifact of this type."""
    if artifact_type not in email_sets:
        return {
            "artifact_type": artifact_type,
            "n_artifact_nodes": 0,
            "distinct_email_pairs_sharing": 0,
            "sum_within_artifact_pairs": 0,
            "mean_emails_per_artifact": float("nan"),
            "median_emails_per_artifact": float("nan"),
            "pct_artifact_nodes_single_email": float("nan"),
            "pairs_per_artifact_node": float("nan"),
            "pairs_per_multiemail_artifact": float("nan"),
        }
    # invert: artifact_idx -> set of emails
    inv: dict[int, set[int]] = defaultdict(set)
    buckets = email_sets[artifact_type]
    for eid, arts in enumerate(buckets):
        for a in arts:
            inv[a].add(eid)
    if not inv:
        sizes = []
    else:
        sizes = [len(s) for s in inv.values()]
    # distinct unordered email pairs sharing at least one artifact
    pair_set: set[tuple[int, int]] = set()
    sum_choose2 = 0
    for _a, emails in inv.items():
        arr = sorted(emails)
        n = len(arr)
        sum_choose2 += n * (n - 1) // 2
        for i in range(n):
            for j in range(i + 1, n):
                pair_set.add((arr[i], arr[j]))
    mean_sz = float(np.mean(sizes)) if sizes else float("nan")
    med_sz = float(np.median(sizes)) if sizes else float("nan")
    pct_1 = float(np.mean([1 if s == 1 else 0 for s in sizes])) if sizes else float("nan")
    n_art = len(inv)
    n_pairs = len(pair_set)
    n_multi = int(sum(1 for s in sizes if s >= 2))
    pairs_per_node = float(n_pairs) / max(1, n_art)
    pairs_per_multiemail = float(n_pairs) / max(1, n_multi)
    return {
        "artifact_type": artifact_type,
        "n_artifact_nodes": n_art,
        "distinct_email_pairs_sharing": n_pairs,
        "sum_within_artifact_pairs": int(sum_choose2),
        "mean_emails_per_artifact": mean_sz,
        "median_emails_per_artifact": med_sz,
        "pct_artifact_nodes_single_email": pct_1,
        "pairs_per_artifact_node": pairs_per_node,
        "pairs_per_multiemail_artifact": pairs_per_multiemail,
    }


def pair_shares_artifact(
    i: int,
    j: int,
    email_sets: dict[str, list[set[int]]],
    types: Iterable[str],
) -> dict[str, bool]:
    out = {}
    for t in types:
        if t not in email_sets:
            out[t] = False
            continue
        si = email_sets[t][i]
        sj = email_sets[t][j]
        out[t] = len(si & sj) > 0
    return out


def n_shared_types(
    i: int,
    j: int,
    email_sets: dict[str, list[set[int]]],
    types: list[str],
) -> int:
    return sum(1 for t in types if pair_shares_artifact(i, j, email_sets, [t])[t])


def campaign_overlap_analysis(
    campaign_to_members: dict[Any, list[str]],
    eid_row: dict[str, int],
    email_sets: dict[str, list[set[int]]],
    compare_types: list[str],
    *,
    max_same_pairs: int = 4000,
    max_diff_pairs: int = 4000,
    seed: int = 0,
    min_camp_size: int = 2,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Sample same-campaign and different-campaign email pairs (graph-aligned ids only).
    Returns (same_stats, diff_stats) each mapping channel -> fraction of pairs with share.
    """
    sp, dp = sample_campaign_email_pairs(
        campaign_to_members,
        eid_row,
        max_same_pairs=max_same_pairs,
        max_diff_pairs=max_diff_pairs,
        seed=seed,
        min_camp_size=min_camp_size,
    )
    return summarize_pair_overlap(sp, email_sets, compare_types), summarize_pair_overlap(
        dp, email_sets, compare_types
    )


def hetero_connected_components_global(data) -> tuple[np.ndarray, list[int]]:
    """Map each unified global node id -> component id; return sorted component sizes descending."""
    node_types = list(data.node_types)
    counts = {nt: infer_num_nodes(data, nt) for nt in node_types}
    offsets: dict[str, int] = {}
    off = 0
    for nt in node_types:
        offsets[nt] = off
        off += counts[nt]
    total = off
    parent = np.arange(total, dtype=np.int64)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for src_t, rel, dst_t in data.edge_types:
        ei = data[src_t, rel, dst_t].edge_index
        if ei is None or ei.numel() == 0:
            continue
        bs, bd = offsets[src_t], offsets[dst_t]
        for s, d in zip(
            ei[0].numpy().astype(np.int64),
            ei[1].numpy().astype(np.int64),
        ):
            union(bs + int(s), bd + int(d))

    comp_of = np.array([find(i) for i in range(total)], dtype=np.int64)
    sizes = Counter(comp_of.tolist())
    size_list = sorted(sizes.values(), reverse=True)
    return comp_of, size_list


def email_component_labels(
    data,
    comp_of: np.ndarray,
    *,
    offsets: dict[str, int],
) -> np.ndarray:
    n_email = infer_num_nodes(data, "email")
    base = offsets["email"]
    return np.array([comp_of[base + i] for i in range(n_email)], dtype=np.int64)


def build_global_offsets(data) -> dict[str, int]:
    node_types = list(data.node_types)
    counts = {nt: infer_num_nodes(data, nt) for nt in node_types}
    offsets: dict[str, int] = {}
    off = 0
    for nt in node_types:
        offsets[nt] = off
        off += counts[nt]
    return offsets


def stem_url_analysis(
    data,
    meta: dict[str, Any],
) -> dict[str, Any]:
    """
    URL vs stem: use graph edge types if present; fallback to meta strings only.
    """
    out: dict[str, Any] = {}
    url_strings = (meta.get("node_maps") or {}).get("url", {}).get("index_to_string") or []
    stem_strings = (meta.get("node_maps") or {}).get("stem", {}).get("index_to_string") or []
    out["n_url_nodes"] = len(url_strings)
    out["n_stem_nodes"] = len(stem_strings)

    url_to_stems: dict[int, set[int]] = defaultdict(set)
    if ("url", "has_stem", "stem") in data.edge_types:
        ei = data["url", "has_stem", "stem"].edge_index
        if ei is not None and ei.numel() > 0:
            for u, s in zip(
                ei[0].numpy().astype(np.int64),
                ei[1].numpy().astype(np.int64),
            ):
                url_to_stems[int(u)].add(int(s))
    elif ("email", "has_url", "url") in data.edge_types and (
        "email", "has_stem", "stem"
    ) in data.edge_types:
        ei_u = data["email", "has_url", "url"].edge_index
        ei_s = data["email", "has_stem", "stem"].edge_index
        if (
            ei_u is not None
            and ei_s is not None
            and ei_u.numel() > 0
            and ei_s.numel() > 0
        ):
            em_urls: dict[int, list[int]] = defaultdict(list)
            em_stems: dict[int, list[int]] = defaultdict(list)
            for e, u in zip(
                ei_u[0].numpy().astype(np.int64),
                ei_u[1].numpy().astype(np.int64),
            ):
                em_urls[int(e)].append(int(u))
            for e, s in zip(
                ei_s[0].numpy().astype(np.int64),
                ei_s[1].numpy().astype(np.int64),
            ):
                em_stems[int(e)].append(int(s))
            for e in em_urls:
                for u in em_urls[e]:
                    for s in em_stems.get(e, ()):
                        url_to_stems[u].add(s)

    stem_n_urls = [len(v) for v in url_to_stems.values()] if url_to_stems else []
    if stem_n_urls:
        url_per_stem = Counter()
        for _u, stems in url_to_stems.items():
            for s in stems:
                url_per_stem[s] += 1
        counts = list(url_per_stem.values())
        out["urls_per_stem_dist"] = counts
        out["mean_urls_per_stem"] = float(np.mean(counts))
        out["median_urls_per_stem"] = float(np.median(counts))
    else:
        out["urls_per_stem_dist"] = []
        out["mean_urls_per_stem"] = float("nan")
        out["median_urls_per_stem"] = float("nan")

    # stem -> domains via emails
    stem_to_domains: dict[int, set[int]] = defaultdict(set)
    if ("email", "has_stem", "stem") in data.edge_types and (
        "email", "has_domain", "domain"
    ) in data.edge_types:
        ei_s = data["email", "has_stem", "stem"].edge_index
        ei_d = data["email", "has_domain", "domain"].edge_index
        if ei_s is not None and ei_d is not None and ei_s.numel() and ei_d.numel() > 0:
            em_stem: dict[int, set[int]] = defaultdict(set)
            em_dom: dict[int, set[int]] = defaultdict(set)
            for e, s in zip(
                ei_s[0].numpy().astype(np.int64),
                ei_s[1].numpy().astype(np.int64),
            ):
                em_stem[int(e)].add(int(s))
            for e, d in zip(
                ei_d[0].numpy().astype(np.int64),
                ei_d[1].numpy().astype(np.int64),
            ):
                em_dom[int(e)].add(int(d))
            for e, stems in em_stem.items():
                for d in em_dom.get(e, ()):
                    for s in stems:
                        stem_to_domains[s].add(d)
    n_multi_dom = sum(1 for _s, ds in stem_to_domains.items() if len(ds) > 1)
    out["n_stems_spanning_multiple_domains"] = int(n_multi_dom)
    out["pct_stems_multi_domain"] = (
        float(n_multi_dom) / max(1, len(stem_to_domains)) if stem_to_domains else 0.0
    )
    if stem_to_domains:
        dom_sz = [len(v) for v in stem_to_domains.values()]
        out["domains_per_stem_dist"] = dom_sz
        out["mean_domains_per_stem"] = float(np.mean(dom_sz))
        out["median_domains_per_stem"] = float(np.median(dom_sz))
        out["n_stems_with_domain_annotations"] = len(dom_sz)
    else:
        out["domains_per_stem_dist"] = []
        out["mean_domains_per_stem"] = float("nan")
        out["median_domains_per_stem"] = float("nan")
        out["n_stems_with_domain_annotations"] = 0
    return out
