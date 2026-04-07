from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from config.pipeline_config import GnnPathLayout, gnn_path_layout_from_pipeline, load_pipeline_config
from src.load_graph_data import load_hetero_pt
from src.train import run_training
from src.train_vicreg import run_vicreg_training
from src.train_contrastive import run_contrastive_training
from src.model_io import select_device

from steps.pipeline_paths import run_dir_for


def run_train_stage(
    *,
    graph_path: str | Path,
    runs_parent: str | Path,
    run_id: str,
    training_cfg: dict[str, Any],
    device_pref: str | None,
    to_undirected: bool,
    path_layout: GnnPathLayout | None = None,
) -> dict[str, Any]:
    """
    Train into ``<runs_parent>/<run_id>/`` (``run_id`` from config). Subpaths for
    checkpoints and artifacts come from ``pipeline_config.json`` ``gnn`` (via ``path_layout``).
    """
    graph_path = str(graph_path)
    if not graph_path:
        raise ValueError("GRAPH_PATH is empty in run_pipeline.py.")

    layout = path_layout or gnn_path_layout_from_pipeline(load_pipeline_config())

    pref = torch.device(device_pref) if isinstance(device_pref, str) and device_pref else None
    device = select_device(pref)

    data = load_hetero_pt(
        path=str(Path(graph_path).expanduser()),
        to_undirected=bool(to_undirected),
    )

    run_dir = run_dir_for(runs_parent, run_id)

    objective = str(training_cfg.get("training_objective", "link_prediction")).lower().strip()

    if objective == "vicreg":
        edge_drop = training_cfg.get("vicreg_edge_drop_probs")
        if isinstance(edge_drop, dict):
            edge_drop = {str(k): float(v) for k, v in edge_drop.items()}
        else:
            edge_drop = None
        ph = training_cfg.get("vicreg_projector_hidden_dim")
        po = training_cfg.get("vicreg_projector_out_dim")
        raw_sem_block = training_cfg.get("vicreg_email_semantic_block")
        sem_block = None
        if raw_sem_block is not None:
            if not isinstance(raw_sem_block, (list, tuple)) or len(raw_sem_block) != 2:
                raise ValueError(
                    "training.vicreg_email_semantic_block must be [start_idx, end_idx] or null."
                )
            sem_block = (int(raw_sem_block[0]), int(raw_sem_block[1]))
        raw_anchor = training_cfg.get("anchor_batch_size")
        anchor_bs = (
            int(raw_anchor)
            if raw_anchor is not None and int(raw_anchor) > 0
            else None
        )
        run_vicreg_training(
            DEVICE=device,
            TORCH_SEED=int(training_cfg["torch_seed"]),
            data=data,
            primary_ntype=str(training_cfg.get("primary_ntype", "email")),
            hidden=int(training_cfg["hidden"]),
            out_dim=int(training_cfg["out_dim"]),
            layers=int(training_cfg["layers"]),
            dropout=float(training_cfg["dropout"]),
            fanout=training_cfg["fanout"],
            val_ratio=float(training_cfg["val_ratio"]),
            test_ratio=float(training_cfg["test_ratio"]),
            epochs=int(training_cfg["epochs"]),
            lr=float(training_cfg["lr"]),
            wd=float(training_cfg["wd"]),
            anchor_batch_size=int(anchor_bs) if anchor_bs is not None else None,
            batch_size=int(training_cfg["batch_size"]),
            model_save_name=str(training_cfg["model_save_name"]),
            early_stopping_patience=int(training_cfg["early_stopping_patience"]),
            lr_reduce_patience=int(training_cfg["lr_reduce_patience"]),
            lr_reduce_factor=float(training_cfg["lr_reduce_factor"]),
            lr_reduce_min=float(training_cfg["lr_reduce_min"]),
            save_epoch_checkpoints=bool(
                training_cfg.get("vicreg_save_epoch_checkpoints", True)
            ),
            run_dir=str(run_dir),
            runs_parent=runs_parent,
            models_subdir=layout.models_subdir,
            metrics_csv=layout.metrics_csv,
            training_config_json=layout.training_config_json,
            w_inv=float(training_cfg.get("vicreg_weight_invariance", 25.0)),
            w_var=float(training_cfg.get("vicreg_weight_variance", 25.0)),
            w_cov=float(training_cfg.get("vicreg_weight_covariance", 1.0)),
            feat_mask_prob=float(training_cfg.get("vicreg_feat_mask_prob", 0.05)),
            edge_drop_probs=edge_drop,
            email_full_zero_prob=float(
                training_cfg.get("vicreg_email_full_zero_prob", 0.15)
            ),
            email_full_zero_apply_to=str(
                training_cfg.get("vicreg_email_full_zero_apply_to", "train_only")
            ),
            email_semantic_mask_prob=float(
                training_cfg.get("vicreg_email_semantic_mask_prob", 0.0)
            ),
            email_semantic_mask_mode=str(
                training_cfg.get("vicreg_email_semantic_mask_mode", "none")
            ),
            email_semantic_apply_to=str(
                training_cfg.get("vicreg_email_semantic_apply_to", "train_only")
            ),
            email_semantic_block=sem_block,
            projector_hidden_dim=int(ph) if ph is not None else None,
            projector_out_dim=int(po) if po is not None else None,
            vicreg_debug_anchor_matching=bool(
                training_cfg.get("vicreg_debug_anchor_matching", False)
            ),
            vicreg_debug_num_batches=int(training_cfg.get("vicreg_debug_num_batches", 3)),
        )
    elif objective == "contrastive":
        edge_drop = training_cfg.get("contrastive_edge_drop_probs")
        if edge_drop is None:
            edge_drop = training_cfg.get("vicreg_edge_drop_probs")
        if isinstance(edge_drop, dict):
            edge_drop = {str(k): float(v) for k, v in edge_drop.items()}
        else:
            edge_drop = None
        ph = training_cfg.get("contrastive_projector_hidden_dim")
        po = training_cfg.get("contrastive_projector_out_dim")
        raw_sem_block = training_cfg.get("contrastive_email_semantic_block")
        sem_block = None
        if raw_sem_block is not None:
            if not isinstance(raw_sem_block, (list, tuple)) or len(raw_sem_block) != 2:
                raise ValueError(
                    "training.contrastive_email_semantic_block must be [start_idx, end_idx] or null."
                )
            sem_block = (int(raw_sem_block[0]), int(raw_sem_block[1]))
        raw_anchor = training_cfg.get("anchor_batch_size")
        anchor_bs = (
            int(raw_anchor)
            if raw_anchor is not None and int(raw_anchor) > 0
            else None
        )
        run_contrastive_training(
            DEVICE=device,
            TORCH_SEED=int(training_cfg["torch_seed"]),
            data=data,
            primary_ntype=str(training_cfg.get("primary_ntype", "email")),
            hidden=int(training_cfg["hidden"]),
            out_dim=int(training_cfg["out_dim"]),
            layers=int(training_cfg["layers"]),
            dropout=float(training_cfg["dropout"]),
            fanout=training_cfg["fanout"],
            val_ratio=float(training_cfg["val_ratio"]),
            test_ratio=float(training_cfg["test_ratio"]),
            epochs=int(training_cfg["epochs"]),
            lr=float(training_cfg["lr"]),
            wd=float(training_cfg["wd"]),
            anchor_batch_size=int(anchor_bs) if anchor_bs is not None else None,
            batch_size=int(training_cfg["batch_size"]),
            model_save_name=str(training_cfg["model_save_name"]),
            early_stopping_patience=int(training_cfg["early_stopping_patience"]),
            lr_reduce_patience=int(training_cfg["lr_reduce_patience"]),
            lr_reduce_factor=float(training_cfg["lr_reduce_factor"]),
            lr_reduce_min=float(training_cfg["lr_reduce_min"]),
            save_epoch_checkpoints=bool(
                training_cfg.get("contrastive_save_epoch_checkpoints", False)
            ),
            run_dir=str(run_dir),
            runs_parent=runs_parent,
            models_subdir=layout.models_subdir,
            metrics_csv=layout.metrics_csv,
            training_config_json=layout.training_config_json,
            feat_mask_prob=float(training_cfg.get("contrastive_feat_mask_prob", 0.08)),
            edge_drop_probs=edge_drop,
            email_semantic_mask_prob=float(
                training_cfg.get("contrastive_email_semantic_mask_prob", 0.05)
            ),
            email_semantic_mask_mode=str(
                training_cfg.get("contrastive_email_semantic_mask_mode", "block_zero")
            ),
            email_semantic_block=sem_block,
            projector_hidden_dim=int(ph) if ph is not None else None,
            projector_out_dim=int(po) if po is not None else None,
            contrastive_temperature=float(training_cfg.get("contrastive_temperature", 0.07)),
            contrastive_raw_cosine_threshold=float(
                training_cfg.get("contrastive_raw_cosine_threshold", 0.30)
            ),
            contrastive_max_negatives_per_anchor=int(
                training_cfg.get("contrastive_max_negatives_per_anchor", 16)
            ),
            contrastive_use_negative_channels=(
                training_cfg.get("contrastive_use_negative_channels")
                or training_cfg.get("contrastive_use_channels")
            ),
            contrastive_use_positive_rules=training_cfg.get(
                "contrastive_use_positive_rules"
            ),
            contrastive_max_cross_positives_per_anchor=int(
                training_cfg.get("contrastive_max_cross_positives_per_anchor", 4)
            ),
            contrastive_cross_positive_raw_cosine_min=float(
                training_cfg.get("contrastive_cross_positive_raw_cosine_min", 0.20)
            ),
            contrastive_debug_anchor_matching=bool(
                training_cfg.get("contrastive_debug_anchor_matching", False)
            ),
            contrastive_debug_num_batches=int(
                training_cfg.get("contrastive_debug_num_batches", 3)
            ),
        )
    else:
        run_training(
            DEVICE=device,
            TORCH_SEED=int(training_cfg["torch_seed"]),
            data=data,
            primary_ntype=training_cfg["primary_ntype"],
            hidden=int(training_cfg["hidden"]),
            out_dim=int(training_cfg["out_dim"]),
            layers=int(training_cfg["layers"]),
            dropout=float(training_cfg["dropout"]),
            neg_ratio=float(training_cfg["neg_ratio"]),
            batch_size=int(training_cfg["batch_size"]),
            fanout=training_cfg["fanout"],
            val_ratio=float(training_cfg["val_ratio"]),
            test_ratio=float(training_cfg["test_ratio"]),
            epochs=int(training_cfg["epochs"]),
            lr=float(training_cfg["lr"]),
            wd=float(training_cfg["wd"]),
            score_head=training_cfg["score_head"],
            early_stopping_patience=int(training_cfg["early_stopping_patience"]),
            lr_reduce_patience=int(training_cfg["lr_reduce_patience"]),
            lr_reduce_factor=float(training_cfg["lr_reduce_factor"]),
            lr_reduce_min=float(training_cfg["lr_reduce_min"]),
            supervised_edge_types=training_cfg["supervised_edge_types"],
            model_save_name=training_cfg["model_save_name"],
            run_dir=str(run_dir),
            runs_parent=runs_parent,
            models_subdir=layout.models_subdir,
            metrics_csv=layout.metrics_csv,
            training_config_json=layout.training_config_json,
        )

    model_save = training_cfg["model_save_name"]
    best_ckpt = run_dir / layout.models_subdir / model_save
    result = {
        "run_dir": str(run_dir),
        "models_dir": str(run_dir / layout.models_subdir),
        "best_checkpoint_path": str(best_ckpt),
        "metrics_csv_path": str(run_dir / layout.metrics_csv),
        "training_config_path": str(run_dir / layout.training_config_json),
    }
    (run_dir / layout.stage_result_json).write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    return result
