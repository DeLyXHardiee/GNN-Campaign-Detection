"""
Optional HDBSCAN sweeps **before** GNN training:

- **raw_graph_hdbscan**: cluster on ``data['email'].x`` (input features to the GNN).
- **bert_embeddings_hdbscan**: cluster on concatenated SBERT subject + body vectors
  (same source as graph assembly; uses the embedder cache under ``graph.embeddings_output_dir``).

Both require ``datasets.ground_truth_json`` and ``email_attrs.external_id`` in the graph ``.meta.json``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

_CORE = Path(__file__).resolve().parents[1]
if str(_CORE) not in sys.path:
    sys.path.insert(0, str(_CORE))

from config.pipeline_config import (  # noqa: E402
    graph_build_settings_from_pipeline,
    load_pipeline_config,
    resolve_project_path,
)
try:
    from core.clustering.clusteringMetrics import extract_ground_truth_labels  # noqa: E402
except ModuleNotFoundError:
    from clustering.clusteringMetrics import extract_ground_truth_labels  # noqa: E402

from src.clustering.clustering_helpers import (  # noqa: E402
    extract_raw_email_feature_map,
    run_hdbscan_sweep_from_embedding_map,
)


def _load_misp_events_list(misp_json_path: str) -> list[dict[str, Any]]:
    with open(misp_json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict):
        ev = raw.get("Events") or raw.get("response", {}).get("Event", [])
        if isinstance(ev, list):
            return ev
        if isinstance(ev, dict):
            return [ev]
    return []


def _build_bert_concat_embedding_map(
    *,
    emails: list[dict[str, Any]],
    email_external_ids: list[Any],
    embeddings_output_dir: str | None,
) -> dict[str, Any]:
    """
    Returns ``{"map": id_to_vec, "error": str|None}``.
    Vectors are ``concat(SBERT(subject), SBERT(body))`` per email row, aligned by index
    with ``email_external_ids`` (same convention as graph assembly).
    """
    from graph.common import parse_misp_events  # noqa: E402
    from utils.embeddings import DEFAULT_OUTPUT_DIR, get_embeddings  # noqa: E402

    parsed = parse_misp_events(emails)
    out_dir = embeddings_output_dir if embeddings_output_dir else str(DEFAULT_OUTPUT_DIR)
    subj_vecs, body_vecs, subj_dim, body_dim = get_embeddings(parsed, output_dir=out_dir)

    if (not subj_vecs and not body_vecs) or (subj_dim <= 0 and body_dim <= 0):
        return {
            "map": {},
            "error": "No SBERT vectors (build graph with include_semantic_embeddings or warm cache).",
        }

    n = min(len(email_external_ids), len(parsed))
    id_to_emb: dict[str, Any] = {}
    for i in range(n):
        eid = str(email_external_ids[i])
        parts: list[float] = []
        if subj_vecs and i < len(subj_vecs):
            parts.extend(float(v) for v in subj_vecs[i])
        if body_vecs and i < len(body_vecs):
            parts.extend(float(v) for v in body_vecs[i])
        if not parts:
            continue
        id_to_emb[eid] = np.asarray(parts, dtype=np.float64)

    if not id_to_emb:
        return {"map": {}, "error": "SBERT dims present but no per-email vectors aligned."}
    return {"map": id_to_emb, "error": None}


def run_pretrain_embedding_clustering(
    *,
    data: Any,
    graph_path: str | Path,
    run_dir: str | Path,
    pipeline_cfg: dict[str, Any] | None = None,
    project_root: Path | None = None,
) -> dict[str, Any]:
    """
    Run enabled optional HDBSCAN sweeps under ``run_dir`` (before training).

    Config under ``pipeline_cfg["gnn_clustering"]``:

    - ``raw_graph_hdbscan``: ``{ "enabled": bool, "min_cluster_size_values": [...], "min_samples": null, "subdir": "clustering_raw_graph", "file_prefix": "raw_graph" }``
    - ``bert_embeddings_hdbscan``: same shape; defaults ``subdir``: ``clustering_bert_embeddings``, ``file_prefix``: ``bert_embeddings``.
    """
    pipeline_cfg = pipeline_cfg or load_pipeline_config(project_root=project_root)
    graph_path = Path(graph_path).expanduser().resolve()
    run_dir = Path(run_dir).expanduser().resolve()

    gc = pipeline_cfg.get("gnn_clustering") or {}
    raw_cfg = gc.get("raw_graph_hdbscan")
    bert_cfg = gc.get("bert_embeddings_hdbscan")
    raw_on = isinstance(raw_cfg, dict) and raw_cfg.get("enabled")
    bert_on = isinstance(bert_cfg, dict) and bert_cfg.get("enabled")
    if not raw_on and not bert_on:
        return {"skipped": True}

    meta_path = graph_path.with_suffix(".meta.json")
    if not meta_path.is_file():
        return {"error": f"Missing graph metadata: {meta_path}"}

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    email_external_ids = meta.get("email_attrs", {}).get("external_id")
    if not email_external_ids:
        return {"error": "email_attrs.external_id missing in metadata."}

    gt_raw = pipeline_cfg.get("datasets", {}).get("ground_truth_json")
    gt_path = resolve_project_path(str(gt_raw), project_root=project_root) if gt_raw else None
    if not gt_path or not Path(gt_path).is_file():
        return {
            "error": "datasets.ground_truth_json not set or file missing; pretrain HDBSCAN requires ground truth.",
        }
    ground_truth = extract_ground_truth_labels(str(gt_path))

    out: dict[str, Any] = {"ground_truth_path": gt_path}

    graph_s = graph_build_settings_from_pipeline(pipeline_cfg, project_root=project_root)
    emb_dir = graph_s.embeddings_output_dir

    if raw_on:
        rc = raw_cfg if isinstance(raw_cfg, dict) else {}
        subdir = str(rc.get("subdir") or "clustering_raw_graph").strip() or "clustering_raw_graph"
        prefix = str(rc.get("file_prefix") or "raw_graph").strip() or "raw_graph"
        mcs = rc.get("min_cluster_size_values") or [2]
        ms = rc.get("min_samples")
        ms = None if ms is None else int(ms)

        id_map = extract_raw_email_feature_map(data, email_external_ids)
        hdb_out = run_dir / subdir / "hdbscan"
        hdb_out.mkdir(parents=True, exist_ok=True)
        csv_p = hdb_out / f"{prefix}_hdbscan_sweep.csv"
        out["raw_graph_hdbscan"] = run_hdbscan_sweep_from_embedding_map(
            id_map,
            ground_truth,
            min_cluster_size_values=list(mcs),
            min_samples=ms,
            output_csv=csv_p,
            model_column_name=prefix,
        )

    if bert_on:
        bc = bert_cfg if isinstance(bert_cfg, dict) else {}
        subdir = str(bc.get("subdir") or "clustering_bert_embeddings").strip() or "clustering_bert_embeddings"
        prefix = str(bc.get("file_prefix") or "bert_embeddings").strip() or "bert_embeddings"
        mcs = bc.get("min_cluster_size_values") or [2]
        ms = bc.get("min_samples")
        ms = None if ms is None else int(ms)

        misp_raw = pipeline_cfg.get("datasets", {}).get("misp_json_path") or (
            pipeline_cfg.get("graph") or {}
        ).get("misp_json_path")
        misp_path = resolve_project_path(str(misp_raw), project_root=project_root) if misp_raw else None
        if not misp_path or not Path(misp_path).is_file():
            out["bert_embeddings_hdbscan"] = {
                "error": "MISP JSON path missing for SBERT clustering.",
            }
        else:
            events = _load_misp_events_list(misp_path)
            built = _build_bert_concat_embedding_map(
                emails=events,
                email_external_ids=email_external_ids,
                embeddings_output_dir=emb_dir,
            )
            if built.get("error"):
                out["bert_embeddings_hdbscan"] = {"error": built["error"]}
            else:
                id_map = built["map"]
                hdb_out = run_dir / subdir / "hdbscan"
                hdb_out.mkdir(parents=True, exist_ok=True)
                csv_p = hdb_out / f"{prefix}_hdbscan_sweep.csv"
                out["bert_embeddings_hdbscan"] = run_hdbscan_sweep_from_embedding_map(
                    id_map,
                    ground_truth,
                    min_cluster_size_values=list(mcs),
                    min_samples=ms,
                    output_csv=csv_p,
                    model_column_name=prefix,
                )

    result_path = run_dir / "pretrain_embedding_clustering.json"
    result_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    out["result_path"] = str(result_path)
    return out
