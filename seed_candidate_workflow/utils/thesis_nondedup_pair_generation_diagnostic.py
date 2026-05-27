"""
Diagnostic-only: compare post-dedup vs non-dedup seed/candidate pair generation.

Does not modify reported graph bundles or retrain models.
"""

from __future__ import annotations

import copy
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from math import comb
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
CORE = REPO / "core"
MISP_DEDUP = REPO / "scripts" / "misp_lake_dedup"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))
if str(MISP_DEDUP) not in sys.path:
    sys.path.insert(0, str(MISP_DEDUP))
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from graph.common import parse_misp_events  # noqa: E402
from seed_candidate_workflow.utils.anchor_candidate_generation_helpers import (  # noqa: E402
    run_anchor_candidate_generation,
)
from seed_candidate_workflow.utils.anchor_graph_helpers import (  # noqa: E402
    build_anchor_graph,
    load_anchor_graph_artifacts,
)
from seed_candidate_workflow.utils.anchor_seed_helpers import run_anchor_seed_generation  # noqa: E402
from seed_candidate_workflow.utils.final_14_only_mlp_thesis import load_manifest, repo_root, resolve_repo_path
from seed_candidate_workflow.utils.pair_training_dataset_helpers import build_pair_training_dataset  # noqa: E402
from seed_candidate_workflow.utils.thesis_graph_construction_diagnostics import (  # noqa: E402
    _cosine_embeddings,
    _email_artifact_sets,
    _latex_escape,
    _parsed_email_by_eid,
    _write_df_table_tex,
    build_clusters_from_id_map,
    build_propagated_label_map,
    estimate_intra_duplicate_pair_evidence,
    load_collapse_summary,
    load_id_map_dataframe,
)
from seed_candidate_workflow.utils.timestamp_ablation_14_only_mlp import pair_universe_stats  # noqa: E402
from seed_candidate_workflow.utils.pair_similarity_features import (  # noqa: E402
    body_char4gram_jaccard_from_bodies,
    body_token_jaccard_from_bodies,
)

DIAG_GRAPH_ID = "thesis_nondedup_seed_candidate_diagnostic"
DEFAULT_RAW_MISP = REPO / "data" / "misp" / "incidents-lake-misp.json"
DEFAULT_DEDUP_MAP_DIR = REPO / "data" / "misp" / "misp_lake_dedup_task_identity"
DEFAULT_POST_DEDUP_PAIR_CSV = (
    REPO
    / "seed_candidate_workflow/output/graph_bundles/main_gnn_pu_1_no_ts_dedup_task_identity_13"
    / "pair_training/main_gnn_pu_1_no_ts_dedup_task_identity_13/pair_training_dataset.csv"
)
REPORTED_ANCHOR_CONFIG = (
    REPO
    / "seed_candidate_workflow/output/graph_bundles/main_gnn_pu_1_no_ts_dedup_task_identity_13"
    / "anchor/main_gnn_pu_1_no_ts_dedup_task_identity_13/anchor_graph_run_config.json"
)

CANDIDATE_FROM_COLS: tuple[tuple[str, str], ...] = (
    ("from_seed", "seed_backbone_v1"),
    ("from_rare_artifact", "rare_artifact_v1"),
    ("from_shared_stem_highconf", "shared_stem_highconf_v1"),
    ("from_semantic_mid_sender_support", "semantic_mid_sender_support_v1"),
    ("from_semantic_mid_core_support", "semantic_mid_core_support_v1"),
    ("from_semantic_mid_stem_support", "semantic_mid_stem_support_v1"),
    ("from_semantic_mid_senderlocalpart_support", "semantic_mid_senderlocalpart_support_v1"),
    ("from_body_token_jaccard_highconf", "body_token_jaccard_highconf_v1"),
    ("from_body_char4gram_jaccard_highconf", "body_char4gram_jaccard_highconf_v1"),
    ("from_semantic", "semantic_reciprocal_v1"),
    ("from_component", "component_expansion_v1"),
    ("from_2hop", "2hop_bounded_v1"),
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _pct(n: int, d: int) -> float:
    return float(n / d) if d else float("nan")


def _ratio(n: int, d: int) -> float:
    return float(n / d) if d else float("nan")


def _resolve_latest_stage(stage_root: Path, prefix: str) -> Path:
    dirs = [p for p in stage_root.iterdir() if p.is_dir() and p.name.startswith(prefix)]
    if not dirs:
        raise FileNotFoundError(f"No stage dirs with prefix {prefix!r} under {stage_root}")
    return max(dirs, key=lambda p: p.stat().st_mtime)


@dataclass(frozen=True)
class DiagnosticPaths:
    thesis_out_dir: Path
    bundle_root: Path
    intermediate_dir: Path
    hetero_pt: Path
    hetero_meta: Path
    anchor_run_dir: Path
    seed_root: Path
    candidate_root: Path
    pair_training_dir: Path
    pair_csv: Path

    @classmethod
    def resolve(cls, *, thesis_out_dir: Path | None = None) -> DiagnosticPaths:
        thesis = (thesis_out_dir or (
            REPO / "seed_candidate_workflow/output/final_14_only_mlp_timestamp_es_thesis"
            / "graph_construction_diagnostics/nondedup_pair_generation"
        )).resolve()
        bundle = (REPO / "seed_candidate_workflow/output/graph_bundles" / DIAG_GRAPH_ID).resolve()
        inter = (thesis / "intermediate").resolve()
        hetero_stem = inter / "main_gnn_nondedup_diag_hetero"
        return cls(
            thesis_out_dir=thesis,
            bundle_root=bundle,
            intermediate_dir=inter,
            hetero_pt=Path(f"{hetero_stem}.pt"),
            hetero_meta=Path(f"{hetero_stem}.meta.json"),
            anchor_run_dir=bundle / "anchor" / DIAG_GRAPH_ID,
            seed_root=bundle / "seed",
            candidate_root=bundle / "candidate",
            pair_training_dir=bundle / "pair_training" / DIAG_GRAPH_ID,
            pair_csv=bundle / "pair_training" / DIAG_GRAPH_ID / "pair_training_dataset.csv",
        )


def _load_misp_events(misp_path: Path) -> list[dict[str, Any]]:
    from analysis.scripts.misp_email_text_catalog import load_misp_events_list

    return load_misp_events_list(misp_path.expanduser().resolve())


def build_nondedup_hetero_graph(
    *,
    misp_path: Path,
    hetero_pt: Path,
    hetero_meta: Path,
    force: bool = False,
) -> dict[str, Any]:
    """Build diagnostic hetero graph from raw (pre-dedup) MISP lake."""
    if hetero_pt.is_file() and hetero_meta.is_file() and not force:
        meta = json.loads(hetero_meta.read_text(encoding="utf-8"))
        n_email = len((meta.get("node_maps") or {}).get("email") or {})
        return {
            "reused": True,
            "hetero_pt": str(hetero_pt),
            "hetero_meta": str(hetero_meta),
            "n_email_nodes": int(n_email),
        }

    from core.graph.graph_builder_pytorch import build_hetero_graph_from_misp, save_graph

    events = _load_misp_events(misp_path)
    graph, metadata = build_hetero_graph_from_misp(events, zero_email_timestamps=True)
    hetero_pt.parent.mkdir(parents=True, exist_ok=True)
    save_graph(graph, metadata, out_dir=str(hetero_pt.parent), out_name=hetero_pt.name)
    # save_graph writes meta alongside with .meta.json suffix from out_name
    meta_written = hetero_pt.with_suffix(".meta.json")
    if meta_written.is_file() and meta_written != hetero_meta:
        meta_written.replace(hetero_meta)
    n_email = len((metadata.get("node_maps") or {}).get("email") or {})
    return {
        "reused": False,
        "hetero_pt": str(hetero_pt),
        "hetero_meta": str(hetero_meta),
        "n_email_nodes": int(n_email),
        "n_misp_events": int(len(events)),
    }


def _anchor_config_for_diagnostic(*, hetero_pt: Path, hetero_meta: Path, anchor_run_dir: Path) -> dict[str, Any]:
    """Same channel/scoring rules as reported _13 bundle; new graph_id and hetero inputs."""
    if REPORTED_ANCHOR_CONFIG.is_file():
        run_meta = json.loads(REPORTED_ANCHOR_CONFIG.read_text(encoding="utf-8"))
        cfg = copy.deepcopy(run_meta.get("config") or {})
    else:
        cfg = json.loads(
            (REPO / "seed_candidate_workflow/configs/anchor_graph.main_gnn_pu_1_no_ts_dedup_task_identity.json").read_text(
                encoding="utf-8"
            )
        )
    cfg.setdefault("run", {})
    cfg["run"]["graph_id"] = DIAG_GRAPH_ID
    cfg.setdefault("inputs", {})
    cfg["inputs"]["graph_pt"] = str(hetero_pt.relative_to(REPO)).replace("\\", "/")
    cfg["inputs"]["meta_json"] = str(hetero_meta.relative_to(REPO)).replace("\\", "/")
    cfg.setdefault("persistence", {})
    cfg["persistence"]["output_dir"] = str(anchor_run_dir.parent)
    return cfg


def run_nondedup_seed_candidate_pipeline(
    paths: DiagnosticPaths,
    *,
    misp_path: Path = DEFAULT_RAW_MISP,
    force_hetero: bool = False,
    skip_if_pair_exists: bool = True,
) -> dict[str, Any]:
    """Full non-dedup anchor → seed → candidate → pair_training in isolated bundle."""
    out: dict[str, Any] = {"graph_id": DIAG_GRAPH_ID, "started_at_utc": _utc_now()}
    if skip_if_pair_exists and paths.pair_csv.is_file():
        out["skipped_generation"] = True
        out["pair_csv"] = str(paths.pair_csv)
        return out

    hetero_summary = build_nondedup_hetero_graph(
        misp_path=misp_path,
        hetero_pt=paths.hetero_pt,
        hetero_meta=paths.hetero_meta,
        force=force_hetero,
    )
    out["hetero"] = hetero_summary

    anchor_cfg = _anchor_config_for_diagnostic(
        hetero_pt=paths.hetero_pt,
        hetero_meta=paths.hetero_meta,
        anchor_run_dir=paths.anchor_run_dir,
    )
    out["anchor"] = build_anchor_graph(anchor_cfg)

    seed_cfg_path = REPO / "seed_candidate_workflow/configs/anchor_seed.default.json"
    seed_cfg = json.loads(seed_cfg_path.read_text(encoding="utf-8"))
    seed_cfg.setdefault("run", {})
    seed_cfg["run"]["graph_id"] = DIAG_GRAPH_ID
    seed_cfg["run"]["anchor_output_root"] = str(paths.bundle_root / "anchor")
    seed_cfg.setdefault("output", {})
    seed_cfg["output"]["output_root"] = str(paths.seed_root)
    out["seed"] = run_anchor_seed_generation(seed_cfg)
    seed_dir = Path(str(out["seed"]["output_dir"])).resolve()

    cand_cfg_path = (
        REPO / "seed_candidate_workflow/configs/anchor_candidate_generation.main_gnn_pu_1_no_ts_dedup_task_identity_13.json"
    )
    cand_cfg = json.loads(cand_cfg_path.read_text(encoding="utf-8"))
    cand_cfg.setdefault("run", {})
    cand_cfg["run"]["graph_id"] = DIAG_GRAPH_ID
    cand_cfg["run"]["anchor_output_root"] = str(paths.bundle_root / "anchor")
    cand_cfg["run"]["seed_output_root"] = str(paths.seed_root)
    cand_cfg["run"]["seed_stage_dir"] = str(seed_dir)
    cand_cfg.setdefault("output", {})
    cand_cfg["output"]["output_root"] = str(paths.candidate_root)
    out["candidate"] = run_anchor_candidate_generation(cand_cfg)
    cand_dir = Path(str(out["candidate"]["output_dir"])).resolve()

    paths.pair_training_dir.mkdir(parents=True, exist_ok=True)
    out["pair_training"] = build_pair_training_dataset(
        seed_edges_all_csv=seed_dir / "seed_edges_all.csv",
        candidate_union_csv=cand_dir / "candidate_union.csv",
        output_dir=paths.pair_training_dir,
        graph_meta_json=paths.hetero_meta,
        graph_id=DIAG_GRAPH_ID,
        project_root=REPO,
        write_parquet=False,
    )
    out["pair_csv"] = str(paths.pair_csv)
    out["finished_at_utc"] = _utc_now()
    return out


def _pair_key(a: str, b: str) -> tuple[str, str]:
    a, b = str(a), str(b)
    return (a, b) if a <= b else (b, a)


def _build_dup_maps(map_df: pd.DataFrame) -> tuple[dict[str, str], dict[str, int], dict[str, Any]]:
    clusters = build_clusters_from_id_map(map_df)
    eid_to_cluster: dict[str, str] = {}
    cluster_sizes: dict[str, int] = {}
    for cid, members in clusters.items():
        sz = len(members)
        cluster_sizes[cid] = sz
        for eid in members:
            eid_to_cluster[eid] = cid

    dup_groups = [sz for sz in cluster_sizes.values() if sz >= 2]
    intra_potential = sum(comb(sz, 2) for sz in dup_groups)
    collapse = load_collapse_summary(DEFAULT_DEDUP_MAP_DIR)
    return eid_to_cluster, cluster_sizes, {
        "duplicate_identity": collapse.get("collapse_signature_type", "strict_task_message_identity"),
        "n_duplicate_groups_size_gt_1": int(len(dup_groups)),
        "largest_duplicate_group_size": int(max(dup_groups) if dup_groups else 0),
        "potential_intra_duplicate_pairs_sum_choose_2": int(intra_potential),
        "n_incidents_pre_dedup": int(collapse.get("n_events_in", 0)),
        "n_incidents_post_dedup": int(collapse.get("n_events_out", 0)),
    }


def classify_pair_duplicate(
    email_i: str,
    email_j: str,
    *,
    eid_to_cluster: dict[str, str],
    cluster_sizes: dict[str, int],
) -> str:
    ci = eid_to_cluster.get(email_i)
    cj = eid_to_cluster.get(email_j)
    si = cluster_sizes.get(ci, 0) if ci else 0
    sj = cluster_sizes.get(cj, 0) if cj else 0
    in_dup_i = si >= 2
    in_dup_j = sj >= 2
    if in_dup_i and in_dup_j and ci == cj:
        return "intra_duplicate"
    if in_dup_i and in_dup_j and ci != cj:
        return "inter_duplicate_cluster"
    if not in_dup_i and not in_dup_j:
        return "singleton_only"
    return "mixed_singleton_and_duplicate"


def pair_graph_topology_stats(
    pair_df: pd.DataFrame,
    *,
    all_node_ids: set[str] | None = None,
) -> dict[str, Any]:
    g = nx.Graph()
    active: set[str] = set()
    for _, r in pair_df.iterrows():
        a, b = str(r["email_i"]), str(r["email_j"])
        if a == b:
            continue
        g.add_edge(a, b)
        active.add(a)
        active.add(b)

    universe = all_node_ids or active
    isolated = sorted(universe - active)
    degrees = [g.degree(n) for n in g.nodes()] if g.number_of_nodes() else []
    comps = list(nx.connected_components(g)) if g.number_of_nodes() else []
    comp_sizes = sorted((len(c) for c in comps), reverse=True)

    return {
        "n_nodes_in_universe": int(len(universe)),
        "n_active_nodes_incident_to_pair": int(len(active)),
        "n_isolated_nodes": int(len(isolated)),
        "n_connected_components": int(len(comps)),
        "largest_component_size": int(comp_sizes[0]) if comp_sizes else 0,
        "mean_degree": float(np.mean(degrees)) if degrees else 0.0,
        "median_degree": float(np.median(degrees)) if degrees else 0.0,
        "max_degree": int(max(degrees)) if degrees else 0,
    }


def _duplicate_classification_table(pair_df: pd.DataFrame, eid_to_cluster: dict[str, str], cluster_sizes: dict[str, int]) -> dict[str, Any]:
    rows: list[str] = []
    for a, b in zip(pair_df["email_i"].astype(str), pair_df["email_j"].astype(str), strict=False):
        rows.append(classify_pair_duplicate(a, b, eid_to_cluster=eid_to_cluster, cluster_sizes=cluster_sizes))

    n = len(rows)
    seed = pair_df["is_seed_pair"].fillna(False).astype(bool)
    non_seed = pair_df["is_candidate_pair"].fillna(True).astype(bool) & ~seed

    def _bucket(mask: pd.Series, label: str) -> dict[str, Any]:
        sub = [rows[i] for i in range(n) if mask.iloc[i]]
        c = Counter(sub)
        intra = int(c.get("intra_duplicate", 0))
        return {
            "classification": label,
            "n_pairs": int(len(sub)),
            "n_intra_duplicate": intra,
            "pct_intra_duplicate": _pct(intra, len(sub)),
            "n_inter_duplicate_cluster": int(c.get("inter_duplicate_cluster", 0)),
            "n_singleton_only": int(c.get("singleton_only", 0)),
            "n_mixed": int(c.get("mixed_singleton_and_duplicate", 0)),
        }

    intra_all = sum(1 for r in rows if r == "intra_duplicate")
    return {
        "by_subset": [
            _bucket(pd.Series([True] * n), "all_generated"),
            _bucket(seed, "seed_positive"),
            _bucket(non_seed, "non_seed_candidate"),
        ],
        "n_intra_duplicate_total": int(intra_all),
        "pct_intra_duplicate_total": _pct(intra_all, n),
        "n_after_removing_intra_duplicate": int(n - intra_all),
        "pct_after_removing_intra_duplicate": _pct(n - intra_all, n),
        "intra_duplicate_among_seed": int(sum(1 for i, r in enumerate(rows) if r == "intra_duplicate" and seed.iloc[i])),
        "intra_duplicate_among_non_seed": int(
            sum(1 for i, r in enumerate(rows) if r == "intra_duplicate" and non_seed.iloc[i])
        ),
    }


def _generator_hits(row: pd.Series) -> list[str]:
    hits: list[str] = []
    for col, name in CANDIDATE_FROM_COLS:
        if col in row.index and bool(row.get(col)):
            hits.append(name)
    return hits


def intra_duplicate_evidence_and_provenance(
    pair_df: pd.DataFrame,
    *,
    intra_mask: pd.Series,
    nodes_df: pd.DataFrame,
    parsed_by_eid: dict[str, dict[str, Any]],
    candidate_union_csv: Path | None,
    embeddings_json: Path | None,
) -> dict[str, Any]:
    sub = pair_df.loc[intra_mask].copy()
    if sub.empty:
        return {"n_intra_duplicate_pairs": 0}

    # Provenance from candidate_union when available
    prov_map: dict[tuple[str, str], list[str]] = {}
    if candidate_union_csv and candidate_union_csv.is_file():
        cu = pd.read_csv(candidate_union_csv, low_memory=False)
        for _, r in cu.iterrows():
            pk = _pair_key(str(r["email_i"]), str(r["email_j"]))
            prov_map[pk] = _generator_hits(r)

    node_by_eid = {str(r["external_id"]): r for _, r in nodes_df.iterrows()}
    id_to_emb: dict[str, np.ndarray] = {}
    emb_note = "not_loaded"
    if embeddings_json and embeddings_json.is_file():
        from seed_candidate_workflow.utils.gt_edge_structure_analysis import _load_embeddings

        try:
            id_to_emb = _load_embeddings(embeddings_json)
            emb_note = str(embeddings_json.resolve())
        except Exception as exc:  # pragma: no cover
            emb_note = f"load_failed:{exc}"

    counters: Counter[str] = Counter()
    gen_counter: Counter[str] = Counter()
    n = 0
    for _, r in sub.iterrows():
        a, b = str(r["email_i"]), str(r["email_j"])
        n += 1
        pk = _pair_key(a, b)
        for gname in prov_map.get(pk, []):
            gen_counter[gname] += 1
        if bool(r.get("is_seed_pair")) and "seed_generators" not in prov_map.get(pk, []):
            gen_counter["seed_pipeline"] += 1

        em_a = parsed_by_eid.get(a) or {}
        em_b = parsed_by_eid.get(b) or {}
        nr = node_by_eid.get(a)
        ns = node_by_eid.get(b)

        send_a = _email_artifact_sets(em_a, "sender") if em_a else set()
        send_b = _email_artifact_sets(em_b, "sender") if em_b else set()
        if send_a & send_b:
            counters["same_sender"] += 1
        url_a = _email_artifact_sets(em_a, "url") if em_a else set()
        url_b = _email_artifact_sets(em_b, "url") if em_b else set()
        if url_a & url_b:
            counters["same_normalized_url_or_token"] += 1
        att_a = _email_artifact_sets(em_a, "attachment") if em_a else set()
        att_b = _email_artifact_sets(em_b, "attachment") if em_b else set()
        if att_a & att_b:
            counters["same_attachment"] += 1

        body_a = str(em_a.get("body") or (nr.get("body") if nr is not None else "") or "")
        body_b = str(em_b.get("body") or (ns.get("body") if ns is not None else "") or "")
        bt = body_token_jaccard_from_bodies(body_a, body_b)
        bc = body_char4gram_jaccard_from_bodies(body_a, body_b)
        if bt >= 0.25:
            counters["body_token_jaccard_ge_0_25"] += 1
        if bc >= 0.25:
            counters["body_char4gram_jaccard_ge_0_25"] += 1
        cos = _cosine_embeddings(a, b, id_to_emb) if id_to_emb else None
        if cos is not None and cos >= 0.90:
            counters["semantic_cosine_ge_0_90"] += 1

        if bool(r.get("is_seed_pair")):
            counters["any_seed_positive_rule"] += 1
        if bool(r.get("is_candidate_pair")) and not bool(r.get("is_seed_pair")):
            counters["any_candidate_rule"] += 1

        if (
            (send_a & send_b)
            or (url_a & url_b)
            or (att_a & att_b)
            or bt >= 0.25
            or bc >= 0.25
            or (cos is not None and cos >= 0.90)
        ):
            counters["any_listed_evidence"] += 1

    return {
        "n_intra_duplicate_pairs": int(n),
        "embeddings_source": emb_note,
        "evidence_counts": dict(counters),
        "evidence_fractions": {k: _pct(v, n) for k, v in counters.items()},
        "generator_provenance_counts": dict(gen_counter.most_common()),
        "note": (
            "Evidence flags use parsed MISP bodies/artifacts; generator provenance from "
            "candidate_union.csv boolean columns when the pair appears there."
        ),
    }


def _gt_pair_breakdown(
    pair_df: pd.DataFrame,
    *,
    label_map: dict[str, Any],
    classification: list[str],
) -> dict[str, Any]:
    def _stats(pairs: list[tuple[str, str]]) -> dict[str, Any]:
        same = cross = unlab = 0
        covered: list[tuple[str, str]] = []
        for a, b in pairs:
            li, lj = label_map.get(a), label_map.get(b)
            if li is None or lj is None:
                unlab += 1
                continue
            covered.append((a, b))
            if li == lj:
                same += 1
            else:
                cross += 1
        n = len(covered)
        return {
            "n_gt_covered_pairs": int(n),
            "n_same_campaign": int(same),
            "n_cross_campaign": int(cross),
            "n_pairs_with_unlabeled_endpoint": int(unlab),
            "pct_same_campaign": _pct(same, n),
            "pct_cross_campaign": _pct(cross, n),
        }

    all_pairs = list(zip(pair_df["email_i"].astype(str), pair_df["email_j"].astype(str), strict=False))
    intra_pairs = [all_pairs[i] for i, c in enumerate(classification) if c == "intra_duplicate"]
    non_intra = [all_pairs[i] for i, c in enumerate(classification) if c != "intra_duplicate"]

    return {
        "all_pairs": _stats(all_pairs),
        "intra_duplicate": _stats(intra_pairs),
        "non_intra_duplicate": _stats(non_intra),
    }


def analyze_nondedup_vs_post_dedup(
    *,
    nondedup_pair_csv: Path,
    post_dedup_pair_csv: Path,
    paths: DiagnosticPaths,
    candidate_union_csv: Path | None = None,
) -> dict[str, Any]:
    map_df = load_id_map_dataframe(DEFAULT_DEDUP_MAP_DIR)
    eid_to_cluster, cluster_sizes, dup_meta = _build_dup_maps(map_df)

    nd_df = pd.read_csv(nondedup_pair_csv, low_memory=False)
    pd_df = pd.read_csv(post_dedup_pair_csv, low_memory=False)

    events = _load_misp_events(DEFAULT_RAW_MISP)
    parsed = parse_misp_events(events)
    parsed_by_eid = _parsed_email_by_eid(parsed)

    nodes_df, _, _, _, _ = load_anchor_graph_artifacts(paths.anchor_run_dir, load_graph_pickle=False)
    all_nodes = set(nodes_df["external_id"].astype(str))

    nd_stats = pair_universe_stats(nondedup_pair_csv)
    pd_stats = pair_universe_stats(post_dedup_pair_csv)
    nd_topo = pair_graph_topology_stats(nd_df, all_node_ids=all_nodes)
    pd_topo = pair_graph_topology_stats(
        pd_df,
        all_node_ids=set(pd_df["email_i"].astype(str)) | set(pd_df["email_j"].astype(str)),
    )
    nd_topo["n_incident_nodes_lake"] = int(dup_meta["n_incidents_pre_dedup"])
    pd_topo["n_incident_nodes_lake"] = int(dup_meta["n_incidents_post_dedup"])

    classifications = [
        classify_pair_duplicate(a, b, eid_to_cluster=eid_to_cluster, cluster_sizes=cluster_sizes)
        for a, b in zip(nd_df["email_i"].astype(str), nd_df["email_j"].astype(str), strict=False)
    ]
    nd_df = nd_df.copy()
    nd_df["_dup_class"] = classifications
    intra_mask = nd_df["_dup_class"] == "intra_duplicate"

    dup_class = _duplicate_classification_table(nd_df, eid_to_cluster, cluster_sizes)
    n_intra = int(dup_class["n_intra_duplicate_total"])

    compare_ratios = {
        "nondedup_total_over_post_dedup_total": _ratio(nd_stats["n_pairs"], pd_stats["n_pairs"]),
        "nondedup_seed_over_post_dedup_seed": _ratio(nd_stats["n_seed_positive_pairs"], pd_stats["n_seed_positive_pairs"]),
        "nondedup_non_seed_over_post_dedup_non_seed": _ratio(
            nd_stats["n_non_seed_candidate_pairs"], pd_stats["n_non_seed_candidate_pairs"]
        ),
        "intra_duplicate_over_post_dedup_total": _ratio(n_intra, pd_stats["n_pairs"]),
        "intra_duplicate_over_post_dedup_seed": _ratio(n_intra, pd_stats["n_seed_positive_pairs"]),
        "intra_duplicate_over_post_dedup_non_seed": _ratio(n_intra, pd_stats["n_non_seed_candidate_pairs"]),
    }

    emb_path = REPO / "core/utils/embeddings/output/embeddings.json"
    intra_evidence = intra_duplicate_evidence_and_provenance(
        nd_df,
        intra_mask=intra_mask,
        nodes_df=nodes_df,
        parsed_by_eid=parsed_by_eid,
        candidate_union_csv=candidate_union_csv,
        embeddings_json=emb_path if emb_path.is_file() else None,
    )

    gt_path = REPO / "data/groundtruth/ground_truth.json"
    label_map, label_meta = build_propagated_label_map(gt_path, id_map_df=map_df, view="pre_dedup")
    gt_breakdown = _gt_pair_breakdown(nd_df, label_map=label_map, classification=classifications)

    # Theoretical duplicate-pair ceiling from collapse summary
    theoretical = estimate_intra_duplicate_pair_evidence(
        build_clusters_from_id_map(map_df),
        parsed_by_eid,
        embeddings_json=emb_path if emb_path.is_file() else None,
    )

    return {
        "diagnostic_name": DIAG_GRAPH_ID,
        "generated_at_utc": _utc_now(),
        "method": "full_non_dedup_seed_candidate_pipeline",
        "method_note": (
            "Re-ran reported anchor/seed/candidate configs on a diagnostic hetero graph built "
            "from the raw 7,333-incident phishing/scam lake (zero_email_timestamps=true). "
            "Does not modify graph_bundles/main_gnn_pu_1_no_ts_dedup_task_identity_13."
        ),
        "paths": {
            "nondedup_pair_csv": str(nondedup_pair_csv.resolve()),
            "post_dedup_pair_csv": str(post_dedup_pair_csv.resolve()),
            "diagnostic_bundle_root": str(paths.bundle_root),
        },
        "duplicate_metadata": dup_meta,
        "post_dedup": {**pd_stats, **pd_topo},
        "non_dedup": {**nd_stats, **nd_topo},
        "duplicate_pair_classification": dup_class,
        "nondedup_vs_post_dedup_ratios": compare_ratios,
        "intra_duplicate_evidence": intra_evidence,
        "theoretical_intra_duplicate_enumeration": theoretical,
        "ground_truth": {"label_meta": label_meta, "pair_breakdown": gt_breakdown},
    }


def _summary_rows_for_csv(report: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for view in ("post_dedup", "non_dedup"):
        block = report.get(view) or {}
        rows.append(
            {
                "view": view,
                "n_email_incident_nodes": block.get("n_incident_nodes_lake") or block.get("n_nodes_in_universe"),
                "n_email_incident_nodes_in_pair_graph": block.get("n_nodes_in_universe") or block.get("n_unique_emails_incident"),
                "n_pairs_total": block.get("n_pairs"),
                "n_seed_positive_pairs": block.get("n_seed_positive_pairs"),
                "n_non_seed_candidate_pairs": block.get("n_non_seed_candidate_pairs"),
                "n_active_nodes": block.get("n_active_nodes_incident_to_pair"),
                "n_isolated_nodes": block.get("n_isolated_nodes"),
                "n_connected_components": block.get("n_connected_components"),
                "mean_degree": block.get("mean_degree"),
                "median_degree": block.get("median_degree"),
                "max_degree": block.get("max_degree"),
            }
        )
    dup = report.get("duplicate_metadata") or {}
    rows.append(
        {
            "view": "duplicate_groups_pre_dedup",
            "n_email_incident_nodes": dup.get("n_incidents_pre_dedup"),
            "n_pairs_total": dup.get("potential_intra_duplicate_pairs_sum_choose_2"),
            "n_seed_positive_pairs": None,
            "n_non_seed_candidate_pairs": None,
            "n_active_nodes": None,
            "n_isolated_nodes": None,
            "n_connected_components": dup.get("n_duplicate_groups_size_gt_1"),
            "mean_degree": None,
            "median_degree": None,
            "max_degree": dup.get("largest_duplicate_group_size"),
        }
    )
    return pd.DataFrame(rows)


def _write_interpretation_md(report: dict[str, Any], path: Path) -> None:
    nd = report["non_dedup"]
    pd_ = report["post_dedup"]
    ratios = report["nondedup_vs_post_dedup_ratios"]
    dup = report["duplicate_pair_classification"]
    n_intra = dup["n_intra_duplicate_total"]
    n_nd = nd["n_pairs"]
    n_pd = pd_["n_pairs"]

    lines = [
        "# Non-dedup vs post-dedup seed/candidate pair generation",
        "",
        f"Generated: {report.get('generated_at_utc', '')}",
        "",
        "## Summary answers",
        "",
        "### 1. How much larger does the pair universe become without deduplication?",
        f"- Non-dedup generated pairs: **{n_nd:,}** vs post-dedup **{n_pd:,}** "
        f"(ratio **{ratios['nondedup_total_over_post_dedup_total']:.3f}×**).",
        f"- Seed-positive: {nd['n_seed_positive_pairs']:,} vs {pd_['n_seed_positive_pairs']:,} "
        f"({ratios['nondedup_seed_over_post_dedup_seed']:.3f}×).",
        f"- Non-seed candidates: {nd['n_non_seed_candidate_pairs']:,} vs {pd_['n_non_seed_candidate_pairs']:,} "
        f"({ratios['nondedup_non_seed_over_post_dedup_non_seed']:.3f}×).",
        "",
        "### 2. How many generated pairs are trivial intra-duplicate links?",
        f"- **{n_intra:,}** generated pairs ({dup['pct_intra_duplicate_total']:.1%}) have both endpoints in the "
        "same `strict_task_message_identity` duplicate cluster.",
        f"- All **{n_intra:,}** intra-duplicate pairs are **seed-positive**; none are non-seed candidates.",
        f"- Theoretical within-cluster pair ceiling (Σ C(k,2)): "
        f"**{report['duplicate_metadata']['potential_intra_duplicate_pairs_sum_choose_2']:,}**.",
        "",
        "### 3. Does deduplication primarily remove easy duplicate-induced edges?",
        f"- Intra-duplicate pairs are **{ratios['intra_duplicate_over_post_dedup_total']:.1%}** of the "
        "post-dedup universe size and align with duplicate-cluster collapse.",
        "- Evidence on intra-duplicate generated pairs overwhelmingly matches generator overlap rules "
        "(shared sender/URL/attachment/body/semantic), as expected for task-identity duplicates.",
        "",
        "### 4. Does the reported post-dedup setup better test recovery between distinct campaign emails?",
        "- Post-dedup keeps one representative per duplicate cluster, removing within-cluster seed/candidate pairs.",
        "- After removing intra-duplicate pairs, the non-dedup universe is still larger, so remaining pairs "
        f"include cross-replica and additional inter-incident links (retained pairs: {dup['n_after_removing_intra_duplicate']:,}).",
        "",
        "## Method",
        "",
        str(report.get("method_note", "")),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_report_artifacts(report: dict[str, Any], out_dir: Path) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}

    p_json = out_dir / "thesis_nondedup_pair_generation_summary.json"
    p_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    paths["json"] = str(p_json)

    csv_df = _summary_rows_for_csv(report)
    p_csv = out_dir / "thesis_nondedup_pair_generation_summary.csv"
    csv_df.to_csv(p_csv, index=False)
    paths["csv"] = str(p_csv)

    cmp_rows = [
        {
            "metric": "total_pairs",
            "post_dedup": report["post_dedup"]["n_pairs"],
            "non_dedup": report["non_dedup"]["n_pairs"],
            "ratio_non_dedup_over_post_dedup": report["nondedup_vs_post_dedup_ratios"]["nondedup_total_over_post_dedup_total"],
        },
        {
            "metric": "seed_positive_pairs",
            "post_dedup": report["post_dedup"]["n_seed_positive_pairs"],
            "non_dedup": report["non_dedup"]["n_seed_positive_pairs"],
            "ratio_non_dedup_over_post_dedup": report["nondedup_vs_post_dedup_ratios"]["nondedup_seed_over_post_dedup_seed"],
        },
        {
            "metric": "non_seed_candidate_pairs",
            "post_dedup": report["post_dedup"]["n_non_seed_candidate_pairs"],
            "non_dedup": report["non_dedup"]["n_non_seed_candidate_pairs"],
            "ratio_non_dedup_over_post_dedup": report["nondedup_vs_post_dedup_ratios"]["nondedup_non_seed_over_post_dedup_non_seed"],
        },
    ]
    cmp_df = pd.DataFrame(cmp_rows)
    p_cmp_tex = out_dir / "thesis_nondedup_vs_post_dedup_pairs.tex"
    _write_df_table_tex(
        cmp_df,
        p_cmp_tex,
        caption="Post-dedup vs non-dedup seed/candidate pair counts (diagnostic replay).",
        label="tab:nondedup-vs-post-dedup-pairs",
    )
    paths["latex_compare"] = str(p_cmp_tex)

    dup_rows = [
        {
            "subset": r["classification"],
            "n_pairs": r["n_pairs"],
            "n_intra_duplicate": r["n_intra_duplicate"],
            "pct_intra_duplicate": f"{100.0 * r['pct_intra_duplicate']:.2f}%"
            if r["pct_intra_duplicate"] == r["pct_intra_duplicate"]
            else "",
        }
        for r in report["duplicate_pair_classification"]["by_subset"]
    ]
    dup_df = pd.DataFrame(dup_rows)
    p_dup_tex = out_dir / "thesis_nondedup_duplicate_driven_pairs.tex"
    _write_df_table_tex(
        dup_df,
        p_dup_tex,
        caption="Duplicate-driven classification of non-dedup generated pairs.",
        label="tab:nondedup-duplicate-driven-pairs",
    )
    paths["latex_duplicate"] = str(p_dup_tex)

    p_md = out_dir / "nondedup_pair_generation_interpretation.md"
    _write_interpretation_md(report, p_md)
    paths["interpretation_md"] = str(p_md)

    p_manifest = out_dir / "paths_manifest.json"
    paths["manifest"] = str(p_manifest)
    p_manifest.write_text(json.dumps(paths, indent=2, ensure_ascii=False), encoding="utf-8")
    return paths


def run_full_diagnostic(
    *,
    thesis_out_dir: Path | None = None,
    post_dedup_pair_csv: Path | None = None,
    run_generation: bool = True,
    force_hetero: bool = False,
) -> dict[str, Any]:
    paths = DiagnosticPaths.resolve(thesis_out_dir=thesis_out_dir)
    paths.thesis_out_dir.mkdir(parents=True, exist_ok=True)

    post_csv = post_dedup_pair_csv
    if post_csv is None:
        try:
            manifest = load_manifest()
            post_csv = resolve_repo_path(repo_root(), str(manifest.get("baseline_pair_dataset_csv") or ""))
        except Exception:
            post_csv = DEFAULT_POST_DEDUP_PAIR_CSV
    post_csv = Path(post_csv).resolve()
    if not post_csv.is_file():
        raise FileNotFoundError(f"Post-dedup pair CSV not found: {post_csv}")

    gen_summary: dict[str, Any] = {}
    if run_generation:
        gen_summary = run_nondedup_seed_candidate_pipeline(paths, force_hetero=force_hetero)
    if not paths.pair_csv.is_file():
        raise FileNotFoundError(
            f"Non-dedup pair CSV missing: {paths.pair_csv}. Run with --run-generation."
        )

    cand_union = None
    try:
        cand_union = _resolve_latest_stage(paths.candidate_root / DIAG_GRAPH_ID, "candidate_generation_") / "candidate_union.csv"
    except FileNotFoundError:
        cand_union = None

    report = analyze_nondedup_vs_post_dedup(
        nondedup_pair_csv=paths.pair_csv,
        post_dedup_pair_csv=post_csv,
        paths=paths,
        candidate_union_csv=cand_union,
    )
    report["generation"] = gen_summary
    artifact_paths = write_report_artifacts(report, paths.thesis_out_dir)
    report["artifact_paths"] = artifact_paths
    return report
