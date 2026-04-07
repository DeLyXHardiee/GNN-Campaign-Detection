from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import torch

from src.clustering.clustering_helpers import (
    extract_ground_truth_labels,
    run_locked_param_across_checkpoints,
    sweep_clustering_for_raw_email_embeddings,
    sweep_clustering_for_transformer_text_embeddings,
    sweep_clustering_for_one_model,
)
from config.pipeline_config import (  # noqa: E402
    GnnPathLayout,
    gnn_path_layout_from_pipeline,
    load_pipeline_config,
)
from src.load_graph_data import load_hetero_pt
from src.model_io import load_model_checkpoint, select_device


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

    # clustering_cfg is a dict: algo_name -> { "enabled": bool, ...params }.
    # Model name comes from training.model_save_name (stem) so it stays consistent when not running training.
    outputs: dict[str, dict[str, str]] = {}
    model_stem = Path(model_save_name).stem
    raw_model_stem = "raw_email_embeddings"
    transformer_model_stem = "transformer_text_embeddings"

    # Default `min_coverage_all` to the same threshold as ground truth coverage.
    if min_coverage_all is None:
        min_coverage_all = float(min_coverage_ground_truth)

    # Select a single locked parameter from the best-model sweep.
    # Criterion: maximize v_measure, with minimum thresholds on BOTH:
    # - coverage_ground_truth
    # - coverage_all
    best_locked_params: dict[str, dict[str, Any]] = {}

    for algo_name, algo_cfg in clustering_cfg.items():
        if not algo_cfg.get("enabled", False):
            continue
        algo_name = str(algo_name).lower().strip()
        cfg_for_sweep = {k: v for k, v in algo_cfg.items() if k != "enabled"}
        cfg_for_sweep["cluster_algorithm"] = algo_name

        algo_out = clustering_out / algo_name
        sweep_res = sweep_clustering_for_one_model(
            model=model,
            data=data,
            device=device,
            ground_truth_labels=ground_truth,
            clustering_config=cfg_for_sweep,
            output_dir=algo_out,
            model_column_name=model_stem,
            email_external_ids=email_external_ids,
            cluster_only_ground_truth=bool(cluster_only_ground_truth),
            use_hybrid_embeddings=bool(hybrid_embeddings),
            hybrid_raw_weight=float(hybrid_raw_weight),
            hybrid_gnn_weight=float(hybrid_gnn_weight),
        )
        algo_entry: dict[str, Any] = {
            "csv_path": str(sweep_res["csv_path"]),
            "output_dir": str(algo_out),
        }

        # Baseline on raw graph email embeddings (pre-training representation).
        raw_sweep_res = sweep_clustering_for_raw_email_embeddings(
            data=data,
            ground_truth_labels=ground_truth,
            clustering_config=cfg_for_sweep,
            output_dir=algo_out,
            email_external_ids=email_external_ids,
            model_column_name=raw_model_stem,
            cluster_only_ground_truth=bool(cluster_only_ground_truth),
        )
        algo_entry["raw_email_embeddings_csv_path"] = str(raw_sweep_res["csv_path"])

        if transformer_baseline_enabled:
            if not transformer_embeddings_json_path:
                raise ValueError(
                    "transformer_baseline_enabled=True but transformer_embeddings_json_path is empty."
                )
            tf_sweep_res = sweep_clustering_for_transformer_text_embeddings(
                ground_truth_labels=ground_truth,
                clustering_config=cfg_for_sweep,
                output_dir=algo_out,
                embeddings_json_path=transformer_embeddings_json_path,
                model_column_name=transformer_model_stem,
                cluster_only_ground_truth=bool(cluster_only_ground_truth),
            )
            algo_entry["transformer_text_embeddings_csv_path"] = str(tf_sweep_res["csv_path"])

        # Choose best epsilon/quantile from the best-model sweep.
        rows = sweep_res.get("rows") or []
        if algo_name == "dbscan":
            param_key = "epsilon"
        elif algo_name == "meanshift":
            param_key = "quantile"
        else:
            param_key = None

        if param_key is not None and rows:
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

    # Run locked-parameter clustering across epoch checkpoints so we can plot metrics vs epoch
    # without doing a full epsilon/quantile sweep for every checkpoint.
    models_dir = Path(checkpoint_path).parent
    epoch_ckpts: list[Path] = []
    for p in models_dir.glob("model_epoch_*.pt"):
        m = re.search(r"(\d+)", p.stem)
        if not m:
            continue
        epoch_ckpts.append(p)
    epoch_ckpts = sorted(epoch_ckpts, key=lambda p: int(re.search(r"(\d+)", p.stem).group(1)))

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
    }
    (clustering_out / layout.stage_result_json).write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    return result

