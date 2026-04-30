from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.decomposition import TruncatedSVD

from config.pipeline_config import (
    GnnPathLayout,
    gnn_path_layout_from_pipeline,
    load_pipeline_config,
    resolve_project_path,
)
try:
    from core.clustering.clusteringMetrics import fit_predict_labels
    from core.visualization.campaign_utils import build_campaign_artifact_payload
except ModuleNotFoundError:
    from clustering.clusteringMetrics import fit_predict_labels
    from visualization.campaign_utils import build_campaign_artifact_payload
from src.clustering.clustering_helpers import (
    extract_email_embeddings,
    extract_ground_truth_labels,
    run_locked_param_across_checkpoints,
    sweep_clustering_for_embedding_map,
)
from config.pipeline_config import (  # noqa: E402
    GnnPathLayout,
    gnn_path_layout_from_pipeline,
    load_pipeline_config,
)
from src.load_graph_data import load_hetero_pt
from src.model_io import load_model_checkpoint, select_device


def _best_param_key_for_algo(algo_name: str) -> str | None:
    if algo_name == "dbscan":
        return "epsilon"
    if algo_name == "meanshift":
        return "quantile"
    if algo_name == "hdbscan":
        return "min_cluster_size"
    return None


def _build_id_to_embedding_from_matrix(
    *,
    external_ids: list[str],
    matrix: np.ndarray,
) -> dict[str, np.ndarray]:
    if matrix.ndim != 2:
        raise ValueError(f"Expected a 2D embedding matrix, got shape={tuple(matrix.shape)}")
    if len(external_ids) != int(matrix.shape[0]):
        raise ValueError(
            f"external_id count ({len(external_ids)}) does not match embedding row count ({int(matrix.shape[0])})."
        )
    return {
        str(eid): np.asarray(matrix[i], dtype=np.float64).copy()
        for i, eid in enumerate(external_ids)
    }


def _load_bert_embedding_map(
    *,
    embeddings_json_path: str | Path,
    external_ids: list[str],
) -> dict[str, np.ndarray]:
    with open(embeddings_json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    by_key = payload.get("by_key")
    if not isinstance(by_key, dict):
        raise ValueError(
            f"Expected a JSON object `by_key` in BERT embeddings file: {embeddings_json_path}"
        )

    id_to_embedding: dict[str, np.ndarray] = {}
    ext_ids = [str(eid) for eid in external_ids]

    for eid in ext_ids:
        entry = by_key.get(eid)
        if not isinstance(entry, dict):
            continue
        subj = entry.get("subj") or []
        body = entry.get("body") or []
        vec = np.asarray([*subj, *body], dtype=np.float64)
        if vec.size > 0:
            id_to_embedding[eid] = vec

    if not id_to_embedding:
        reverse_index: dict[str, dict[str, Any]] = {}
        for _key, value in by_key.items():
            if not isinstance(value, dict):
                continue
            eid = value.get("external_id")
            if eid is None:
                continue
            reverse_index[str(eid)] = value
        for eid in ext_ids:
            entry = reverse_index.get(eid)
            if not entry:
                continue
            subj = entry.get("subj") or []
            body = entry.get("body") or []
            vec = np.asarray([*subj, *body], dtype=np.float64)
            if vec.size > 0:
                id_to_embedding[eid] = vec

    if not id_to_embedding:
        raise ValueError(
            f"No overlapping external_ids found between graph metadata and BERT embeddings at {embeddings_json_path}."
        )

    expected_dim = len(next(iter(id_to_embedding.values())))
    invalid_dims = [eid for eid, vec in id_to_embedding.items() if len(vec) != expected_dim]
    if invalid_dims:
        sample = invalid_dims[:10]
        suffix = "..." if len(invalid_dims) > 10 else ""
        raise ValueError(
            f"Inconsistent BERT embedding dimensions in {embeddings_json_path}. "
            f"Expected dim={expected_dim}; bad external_ids: {sample}{suffix}"
        )

    return id_to_embedding


def _prepare_embedding_map_for_clustering(
    *,
    id_to_embedding_map: dict[str, np.ndarray],
    l2_normalize: bool,
    max_components: int | None,
    random_state: int = 42,
) -> dict[str, np.ndarray]:
    if not id_to_embedding_map:
        return {}

    ordered_ids = sorted(id_to_embedding_map.keys(), key=str)
    X = np.stack(
        [np.asarray(id_to_embedding_map[eid], dtype=np.float32) for eid in ordered_ids],
        axis=0,
    )

    if l2_normalize:
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms = np.where(norms > 0.0, norms, 1.0)
        X = X / norms

    if max_components is not None:
        max_components = int(max_components)
        if max_components > 1 and X.shape[1] > max_components:
            n_components = min(max_components, int(X.shape[1] - 1))
            if n_components > 1:
                svd = TruncatedSVD(n_components=n_components, random_state=random_state)
                X = svd.fit_transform(X).astype(np.float32, copy=False)

    return {
        eid: np.asarray(X[i], dtype=np.float64).copy()
        for i, eid in enumerate(ordered_ids)
    }


def _resolve_bert_algorithms_cfg(
    *,
    default_algorithms_cfg: dict[str, Any],
    bert_cfg: dict[str, Any],
    n_embeddings: int,
) -> tuple[dict[str, Any], list[str]]:
    cfg_raw = bert_cfg.get("config")
    if isinstance(cfg_raw, dict) and cfg_raw:
        algorithms_cfg: dict[str, Any] = {
            str(k): (dict(v) if isinstance(v, dict) else v)
            for k, v in cfg_raw.items()
        }
    else:
        algorithms_cfg = {
            str(k): (dict(v) if isinstance(v, dict) else v)
            for k, v in default_algorithms_cfg.items()
        }

    optimization = bert_cfg.get("optimization")
    if not isinstance(optimization, dict):
        optimization = {}
    force_expensive = bool(optimization.get("force_expensive_algorithms", False))
    max_ms = int(optimization.get("max_embeddings_for_meanshift", 3000))
    max_hdb = int(optimization.get("max_embeddings_for_hdbscan", 5000))

    notes: list[str] = []
    if not force_expensive:
        ms_cfg = algorithms_cfg.get("meanshift")
        if isinstance(ms_cfg, dict) and bool(ms_cfg.get("enabled", False)) and n_embeddings > max_ms:
            ms_cfg["enabled"] = False
            notes.append(
                f"Disabled meanshift for BERT baseline (n_embeddings={n_embeddings} exceeds max_embeddings_for_meanshift={max_ms})."
            )
        hdb_cfg = algorithms_cfg.get("hdbscan")
        if isinstance(hdb_cfg, dict) and bool(hdb_cfg.get("enabled", False)) and n_embeddings > max_hdb:
            hdb_cfg["enabled"] = False
            notes.append(
                f"Disabled hdbscan for BERT baseline (n_embeddings={n_embeddings} exceeds max_embeddings_for_hdbscan={max_hdb})."
            )

    return algorithms_cfg, notes


def _run_embedding_clustering_suite(
    *,
    algorithms_cfg: dict[str, Any],
    id_to_embedding_map: dict[str, np.ndarray],
    ground_truth: dict[str, Any],
    clustering_root_dir: Path,
    model_stem: str,
    min_coverage_ground_truth: float,
    min_coverage_all: float,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    outputs: dict[str, dict[str, Any]] = {}
    best_locked_params: dict[str, dict[str, Any]] = {}

    for algo_name, algo_cfg in algorithms_cfg.items():
        if not isinstance(algo_cfg, dict) or not algo_cfg.get("enabled", False):
            continue
        algo_name = str(algo_name).lower().strip()
        param_key = _best_param_key_for_algo(algo_name)
        if param_key is None:
            continue

        cfg_for_sweep = {k: v for k, v in algo_cfg.items() if k != "enabled"}
        cfg_for_sweep["cluster_algorithm"] = algo_name

        algo_out = clustering_root_dir / algo_name
        print(
            f"[clustering] start algo={algo_name} model={model_stem} "
            f"n_embeddings={len(id_to_embedding_map)} out={algo_out}"
        )
        sweep_res = sweep_clustering_for_embedding_map(
            id_to_embedding_map=id_to_embedding_map,
            ground_truth_labels=ground_truth,
            clustering_config=cfg_for_sweep,
            output_dir=algo_out,
            model_column_name=model_stem,
        )
        print(
            f"[clustering] done  algo={algo_name} model={model_stem} "
            f"rows={len(sweep_res.get('rows') or [])}"
        )
        algo_entry: dict[str, Any] = {
            "csv_path": str(sweep_res["csv_path"]),
            "output_dir": str(algo_out),
        }

        rows = sweep_res.get("rows") or []
        if rows:
            candidates = [
                r
                for r in rows
                if float(r.get("coverage_ground_truth", 0.0)) >= min_coverage_ground_truth
                and float(r.get("coverage_all", 0.0)) >= min_coverage_all
            ]
            pool = candidates if candidates else list(rows)
            best_row = max(pool, key=lambda r: float(r.get("v_measure", 0.0)))
            best_locked_params[algo_name] = {
                param_key: float(best_row.get(param_key)),
                "v_measure": float(best_row.get("v_measure", 0.0)),
                "coverage_ground_truth": float(best_row.get("coverage_ground_truth", 0.0)),
                "coverage_all": float(best_row.get("coverage_all", 0.0)),
                "min_coverage_ground_truth": float(min_coverage_ground_truth),
                "min_coverage_all": float(min_coverage_all),
            }

        clustering_errors = [
            str(r["clustering_error"])
            for r in sweep_res["rows"]
            if r.get("clustering_error")
        ]
        if clustering_errors:
            algo_entry["clustering_errors"] = clustering_errors
        outputs[algo_name] = algo_entry

    return outputs, best_locked_params


def _write_campaigns_for_best_algorithm(
    *,
    algorithms_cfg: dict[str, Any],
    id_to_embedding_map: dict[str, np.ndarray],
    best_locked_params: dict[str, dict[str, Any]],
    output_dir: Path,
    model_stem: str,
    solution_name: str,
    out_filename: str,
) -> str | None:
    if not best_locked_params:
        return None

    best_algo_name, best_info = max(
        best_locked_params.items(),
        key=lambda kv: float(kv[1].get("v_measure", 0.0)),
    )
    algo_cfg_best = algorithms_cfg.get(best_algo_name)
    if not isinstance(algo_cfg_best, dict):
        algo_cfg_best = {}

    if best_algo_name == "dbscan":
        sorted_ids, labels = fit_predict_labels(
            id_to_embedding_map,
            "dbscan",
            epsilon=float(best_info["epsilon"]),
            min_samples=int(algo_cfg_best.get("min_samples", 5)),
        )
        params_out: dict[str, Any] = {
            "epsilon": float(best_info["epsilon"]),
            "min_samples": int(algo_cfg_best.get("min_samples", 5)),
        }
    elif best_algo_name == "meanshift":
        sorted_ids, labels = fit_predict_labels(
            id_to_embedding_map,
            "meanshift",
            quantile=float(best_info["quantile"]),
            n_samples=algo_cfg_best.get("n_samples"),
        )
        params_out = {
            "quantile": float(best_info["quantile"]),
            "n_samples": algo_cfg_best.get("n_samples"),
        }
    elif best_algo_name == "hdbscan":
        sorted_ids, labels = fit_predict_labels(
            id_to_embedding_map,
            "hdbscan",
            min_cluster_size=int(best_info["min_cluster_size"]),
            hdbscan_min_samples=algo_cfg_best.get("min_samples"),
            hdbscan_metric=str(algo_cfg_best.get("metric") or "cosine"),
        )
        params_out = {
            "min_cluster_size": int(best_info["min_cluster_size"]),
            "min_samples": algo_cfg_best.get("min_samples"),
            "metric": str(algo_cfg_best.get("metric") or "cosine"),
        }
    else:
        return None

    payload = build_campaign_artifact_payload(
        solution=solution_name,
        algorithm=best_algo_name,
        sorted_ids=sorted_ids,
        labels=labels,
        params=params_out,
        metrics={
            "v_measure": float(best_info.get("v_measure", 0.0)),
            "coverage_ground_truth": float(best_info.get("coverage_ground_truth", 0.0)),
            "coverage_all": float(best_info.get("coverage_all", 0.0)),
        },
        model_name=model_stem,
    )
    out_p = output_dir / out_filename
    out_p.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return str(out_p)


def run_clustering_stage(
    *,
    graph_path: str | Path,
    ground_truth_path: str | Path,
    checkpoint_path: str | Path,
    output_dir: str | Path,
    clustering_cfg: dict[str, Any],
    cluster_only_ground_truth: bool = False,
    min_coverage_ground_truth: float = 0.5,
    min_coverage_all: float | None = None,
    hybrid_embeddings: bool = False,
    hybrid_raw_weight: float = 1.0,
    hybrid_gnn_weight: float = 1.0,
    model_save_name: str,
    device_pref: str | None,
    to_undirected: bool,
    baselines_cfg: dict[str, Any] | None = None,
    path_layout: GnnPathLayout | None = None,
    transformer_baseline_enabled: bool = False,
    transformer_embeddings_json_path: str | Path | None = None,
) -> dict[str, Any]:
    graph_path = str(graph_path)
    ground_truth_path = str(ground_truth_path)
    checkpoint_path = str(checkpoint_path)

    if not graph_path:
        raise ValueError("GRAPH_PATH is empty in core/GNN/run_pipeline.py.")
    if not ground_truth_path:
        raise ValueError("GROUND_TRUTH_PATH is empty in core/GNN/run_pipeline.py (required for clustering).")
    if not checkpoint_path:
        raise ValueError("CHECKPOINT_PATH is empty in core/GNN/run_pipeline.py (required for clustering).")

    print(
        "[clustering diag] run_clustering_stage: starting (before graph load)",
        flush=True,
    )

    layout = path_layout or gnn_path_layout_from_pipeline(load_pipeline_config())

    output_dir = Path(output_dir)
    clustering_out = output_dir / layout.clustering_subdir
    clustering_out.mkdir(parents=True, exist_ok=True)

    pref = torch.device(device_pref) if isinstance(device_pref, str) and device_pref else None
    device = select_device(pref)

    data = load_hetero_pt(
        path=str(Path(graph_path).expanduser()),
        to_undirected=bool(to_undirected),
    )
    # Load metadata for email external_id (graph does not store it; PyG loaders require tensor-only node stores)
    meta_path = Path(graph_path).expanduser().with_suffix(".meta.json")
    if not meta_path.exists():
        raise FileNotFoundError(
            f"Graph metadata not found: {meta_path}. "
            "Clustering requires email_attrs.external_id from the companion .meta.json."
        )
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    email_external_ids = meta.get("email_attrs", {}).get("external_id")
    if not email_external_ids:
        raise ValueError(
            f"Metadata at {meta_path} has no email_attrs.external_id. "
            "Clustering requires external_id per email node."
        )
    print(
        f"[clustering diag] loading ground truth JSON: {ground_truth_path!r}",
        flush=True,
    )
    ground_truth = extract_ground_truth_labels(ground_truth_path)
    print(
        f"[clustering diag] ground truth labels loaded: {len(ground_truth)} ids",
        flush=True,
    )

    # --- Diagnostic prints (ground truth vs graph identity overlap) ---
    _graph_id_list = [str(x) for x in email_external_ids]
    _graph_ids = set(_graph_id_list)
    _gt_ids = set(map(str, ground_truth.keys()))
    _overlap = _graph_ids & _gt_ids
    try:
        _n_email_tensor = int(data["email"].num_nodes)
    except Exception:
        _n_email_tensor = -1
    _enabled_algos = [
        str(k)
        for k, v in (clustering_cfg or {}).items()
        if isinstance(v, dict) and v.get("enabled", False)
    ]
    print("\n[clustering diag] ==========", flush=True)
    print(f"[clustering diag] graph_path:\n    {graph_path}", flush=True)
    print(f"[clustering diag] ground_truth_path:\n    {ground_truth_path}", flush=True)
    print(f"[clustering diag] checkpoint_path:\n    {checkpoint_path}", flush=True)
    print(
        f"[clustering diag] checkpoint exists: {Path(checkpoint_path).expanduser().exists()}",
        flush=True,
    )
    print(f"[clustering diag] clustering output dir:\n    {clustering_out}", flush=True)
    print(
        f"[clustering diag] cluster_only_ground_truth: {cluster_only_ground_truth}",
        flush=True,
    )
    print(
        f"[clustering diag] min_coverage_ground_truth={min_coverage_ground_truth}, "
        f"min_coverage_all={min_coverage_all}",
        flush=True,
    )
    print(
        f"[clustering diag] graph email num_nodes: {_n_email_tensor}, "
        f"meta external_id list length: {len(_graph_id_list)}, "
        f"unique meta external_id: {len(_graph_ids)}",
        flush=True,
    )
    print(
        f"[clustering diag] ground_truth labeled emails (unique): {len(_gt_ids)}",
        flush=True,
    )
    print(
        f"[clustering diag] overlap (graph ∩ ground_truth external_id): {len(_overlap)}",
        flush=True,
    )
    if not _overlap and _gt_ids and _graph_ids:
        print(
            "[clustering diag] ZERO overlap — clustering will fail when "
            "cluster_only_ground_truth=True (no embeddings after restriction).",
            flush=True,
        )
        _sgt = sorted(_gt_ids, key=str)[:5]
        _sgr = sorted(_graph_ids, key=str)[:5]
        print(
            f"[clustering diag] sample ground_truth external_ids: {_sgt}",
            flush=True,
        )
        print(
            f"[clustering diag] sample graph external_ids: {_sgr}",
            flush=True,
        )
    _only_gt = sorted(_gt_ids - _graph_ids, key=str)
    _only_graph = sorted(_graph_ids - _gt_ids, key=str)
    print(
        f"[clustering diag] only_in_ground_truth (not on graph): {len(_only_gt)}",
        flush=True,
    )
    if _only_gt and len(_only_gt) <= 8:
        print(f"[clustering diag] only_in_ground_truth ids: {_only_gt}", flush=True)
    elif _only_gt:
        print(
            f"[clustering diag] only_in_ground_truth (first 5): {_only_gt[:5]}",
            flush=True,
        )
    print(
        f"[clustering diag] only_on_graph (not in ground_truth): {len(_only_graph)}",
        flush=True,
    )
    print(
        f"[clustering diag] enabled clustering algorithms: "
        f"{_enabled_algos if _enabled_algos else '(none — no sweeps will run)'}",
        flush=True,
    )
    print(
        f"[clustering diag] hybrid_embeddings={hybrid_embeddings} "
        f"(raw_w={hybrid_raw_weight}, gnn_w={hybrid_gnn_weight})",
        flush=True,
    )
    print(
        f"[clustering diag] transformer_text_baseline={transformer_baseline_enabled} "
        f"(embeddings_json={transformer_embeddings_json_path})",
        flush=True,
    )
    print("[clustering diag] ==========\n", flush=True)
    # --- end diagnostic prints ---

    model, predictor, checkpoint = load_model_checkpoint(
        device=device, metadata=data.metadata(), filename=checkpoint_path
    )
    _ = predictor, checkpoint

    baselines_cfg = baselines_cfg if isinstance(baselines_cfg, dict) else {}

    # clustering_cfg is a dict: algo_name -> { "enabled": bool, ...params }.
    # Model name comes from training.model_save_name (stem) so it stays consistent when not running training.
    outputs: dict[str, dict[str, str]] = {}
    model_stem = Path(model_save_name).stem
    raw_model_stem = "raw_email_embeddings"
    transformer_model_stem = "transformer_text_embeddings"

    # Default `min_coverage_all` to the same threshold as ground truth coverage.
    if min_coverage_all is None:
        min_coverage_all = float(min_coverage_ground_truth)

    gnn_id_to_emb = extract_email_embeddings(
        model, data, device, external_ids=email_external_ids
    )
    outputs, best_locked_params = _run_embedding_clustering_suite(
        algorithms_cfg=clustering_cfg,
        id_to_embedding_map=gnn_id_to_emb,
        ground_truth=ground_truth,
        clustering_root_dir=clustering_out,
        model_stem=model_stem,
        min_coverage_ground_truth=float(min_coverage_ground_truth),
        min_coverage_all=float(min_coverage_all),
    )

    raw_graph_id_to_emb = _build_id_to_embedding_from_matrix(
        external_ids=[str(x) for x in email_external_ids],
        matrix=data["email"].x.detach().cpu().numpy(),
    )

    # Run locked-parameter clustering across epoch checkpoints so we can plot metrics vs epoch
    # without doing a full epsilon/quantile sweep for every checkpoint.
    models_dir = Path(checkpoint_path).parent
    epoch_ckpts: list[Path] = []
    for p in models_dir.glob("model_epoch_*.pt"):
        m = re.search(r"(\d+)", p.stem)
        if not m:
            continue
        epoch_ckpts.append(p)
    def _epoch_num(p: Path) -> int:
        em = re.search(r"(\d+)", p.stem)
        return int(em.group(1)) if em else 0

    epoch_ckpts = sorted(epoch_ckpts, key=_epoch_num)

    if epoch_ckpts and best_locked_params:
        for algo_name, best in best_locked_params.items():
            algo_cfg = clustering_cfg.get(algo_name) if isinstance(clustering_cfg, dict) else None
            if not isinstance(algo_cfg, dict) or not algo_cfg.get("enabled", False):
                continue

            if algo_name == "dbscan":
                locked_param_value = float(best["epsilon"])
            elif algo_name == "meanshift":
                locked_param_value = float(best["quantile"])
            elif algo_name == "hdbscan":
                locked_param_value = float(best["min_cluster_size"])
            else:
                continue

            cfg_for_sweep = {k: v for k, v in algo_cfg.items() if k != "enabled"}
            cfg_for_sweep["cluster_algorithm"] = algo_name

            algo_out = clustering_out / algo_name
            run_locked_param_across_checkpoints(
                data=data,
                device=device,
                checkpoints=[str(p) for p in epoch_ckpts],
                ground_truth_labels=ground_truth,
                clustering_config=cfg_for_sweep,
                locked_param_value=locked_param_value,
                output_dir=algo_out,
                email_external_ids=email_external_ids,
                cluster_only_ground_truth=bool(cluster_only_ground_truth),
                use_hybrid_embeddings=bool(hybrid_embeddings),
                hybrid_raw_weight=float(hybrid_raw_weight),
                hybrid_gnn_weight=float(hybrid_gnn_weight),
            )

    campaigns_gnn_path = _write_campaigns_for_best_algorithm(
        algorithms_cfg=clustering_cfg,
        id_to_embedding_map=gnn_id_to_emb,
        best_locked_params=best_locked_params,
        output_dir=clustering_out,
        model_stem=model_stem,
        solution_name="gnn",
        out_filename="campaigns_gnn.json",
    )

    baseline_results: dict[str, Any] = {}

    raw_graph_cfg = baselines_cfg.get("raw_graph")
    if isinstance(raw_graph_cfg, dict) and bool(raw_graph_cfg.get("enabled", False)):
        raw_root = clustering_out / "baseline_raw_graph"
        raw_outputs, raw_best = _run_embedding_clustering_suite(
            algorithms_cfg=clustering_cfg,
            id_to_embedding_map=raw_graph_id_to_emb,
            ground_truth=ground_truth,
            clustering_root_dir=raw_root,
            model_stem=model_stem,
            min_coverage_ground_truth=float(min_coverage_ground_truth),
            min_coverage_all=float(min_coverage_all),
        )
        raw_campaigns_path = _write_campaigns_for_best_algorithm(
            algorithms_cfg=clustering_cfg,
            id_to_embedding_map=raw_graph_id_to_emb,
            best_locked_params=raw_best,
            output_dir=raw_root,
            model_stem=model_stem,
            solution_name="raw_graph",
            out_filename="campaigns_raw_graph.json",
        )
        baseline_results["raw_graph"] = {
            "output_dir": str(raw_root),
            "algorithms": raw_outputs,
            "best_locked_params": raw_best,
            "campaigns_path": raw_campaigns_path,
        }

    bert_cfg = baselines_cfg.get("bert_embeddings")
    if isinstance(bert_cfg, dict) and bool(bert_cfg.get("enabled", False)):
        bert_path_raw = (
            bert_cfg.get("embeddings_json_path")
            or "core/utils/embeddings/output/embeddings.json"
        )
        bert_path_resolved = resolve_project_path(str(bert_path_raw)) or str(
            Path(str(bert_path_raw)).expanduser().resolve()
        )
        bert_id_to_emb = _load_bert_embedding_map(
            embeddings_json_path=bert_path_resolved,
            external_ids=[str(x) for x in email_external_ids],
        )
        optimization_cfg = bert_cfg.get("optimization")
        if not isinstance(optimization_cfg, dict):
            optimization_cfg = {}
        bert_id_to_emb_prepared = _prepare_embedding_map_for_clustering(
            id_to_embedding_map=bert_id_to_emb,
            l2_normalize=bool(optimization_cfg.get("l2_normalize", True)),
            max_components=(
                int(optimization_cfg["max_components"])
                if optimization_cfg.get("max_components") is not None
                else 256
            ),
            random_state=int(optimization_cfg.get("random_state", 42)),
        )
        bert_algorithms_cfg, bert_optimization_notes = _resolve_bert_algorithms_cfg(
            default_algorithms_cfg=clustering_cfg,
            bert_cfg=bert_cfg,
            n_embeddings=int(len(bert_id_to_emb_prepared)),
        )
        bert_root = clustering_out / "baseline_bert_embeddings"
        bert_outputs, bert_best = _run_embedding_clustering_suite(
            algorithms_cfg=bert_algorithms_cfg,
            id_to_embedding_map=bert_id_to_emb_prepared,
            ground_truth=ground_truth,
            clustering_root_dir=bert_root,
            model_stem=model_stem,
            min_coverage_ground_truth=float(min_coverage_ground_truth),
            min_coverage_all=float(min_coverage_all),
        )
        bert_campaigns_path = _write_campaigns_for_best_algorithm(
            algorithms_cfg=bert_algorithms_cfg,
            id_to_embedding_map=bert_id_to_emb_prepared,
            best_locked_params=bert_best,
            output_dir=bert_root,
            model_stem=model_stem,
            solution_name="bert_embeddings",
            out_filename="campaigns_bert_embeddings.json",
        )
        baseline_results["bert_embeddings"] = {
            "output_dir": str(bert_root),
            "algorithms": bert_outputs,
            "best_locked_params": bert_best,
            "campaigns_path": bert_campaigns_path,
            "embeddings_json_path": str(bert_path_resolved),
            "n_embeddings_used": int(len(bert_id_to_emb)),
            "n_embeddings_clustered": int(len(bert_id_to_emb_prepared)),
            "embedding_dim_clustered": int(
                len(next(iter(bert_id_to_emb_prepared.values())))
            ) if bert_id_to_emb_prepared else 0,
            "optimization_notes": bert_optimization_notes,
        }

    result = {
        "output_dir": str(clustering_out),
        "model_column_name": f"{model_stem}_hybrid" if hybrid_embeddings else model_stem,
        "hybrid_embeddings": bool(hybrid_embeddings),
        "hybrid_raw_weight": float(hybrid_raw_weight),
        "hybrid_gnn_weight": float(hybrid_gnn_weight),
        "cluster_only_ground_truth": bool(cluster_only_ground_truth),
        "transformer_baseline_enabled": bool(transformer_baseline_enabled),
        "transformer_embeddings_json_path": (
            str(Path(transformer_embeddings_json_path).expanduser().resolve())
            if transformer_embeddings_json_path
            else None
        ),
        "algorithms": outputs,
        "best_locked_params": best_locked_params,
        "locked_param_min_coverage_ground_truth": float(min_coverage_ground_truth),
        "locked_param_min_coverage_all": float(min_coverage_all),
        "locked_param_epoch_checkpoints": [str(p) for p in epoch_ckpts],
        "campaigns_gnn_path": campaigns_gnn_path,
        "baselines": baseline_results,
    }
    (clustering_out / layout.stage_result_json).write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    return result

