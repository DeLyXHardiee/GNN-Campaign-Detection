"""Training loop for Edge-GNN (candidate-edge line graph + nnPU)."""

from __future__ import annotations

import csv
import json
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from .edge_candidate_line_graph import build_candidate_edge_line_graph
from .edge_pair_gnn import (
    LOCAL_HEAD_EMAIL_PAIR_MLP_COMPATIBLE,
    EdgePairGnnModel,
    build_edge_pair_gnn_model,
    edge_gnn_config_from_training_cfg,
)
from .pu_loss import aggregate_epoch_pu_stats, compute_pair_loss, resolve_pair_loss_type
from .pair_train import (
    PAIR_ENCODER_EDGE_GNN,
    PAIR_FEATURE_COLUMNS,
    PAIR_METRICS_HEADER,
    _build_train_df_epoch_emphasis,
    _easy_positive_mask,
    _hard_positive_mask,
    _hard_unlabeled_mask,
    _safe_bool_series,
    build_pair_feature_matrix,
    metrics_row_pair_training,
    reliable_negative_supervision_active,
)
from .pair_semantic_cluster_sampling import build_train_epoch_cluster_aware


def _ensure_edge_node_ids_on_splits(
    df: pd.DataFrame,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stable edge-node ids (0..N-1) on full df and split frames for mask construction."""
    out_df = df.copy() if "_edge_node_id" not in df.columns else df
    if "_edge_node_id" not in out_df.columns:
        out_df["_edge_node_id"] = np.arange(len(out_df), dtype=np.int64)
    lookup = out_df[["email_i", "email_j", "_edge_node_id"]].copy()
    lookup["email_i"] = lookup["email_i"].astype(str)
    lookup["email_j"] = lookup["email_j"].astype(str)
    if lookup.duplicated(subset=["email_i", "email_j"]).any():
        raise ValueError("pair dataset has duplicate (email_i, email_j) rows; cannot map edge_node_id")

    def _attach(frame: pd.DataFrame) -> pd.DataFrame:
        if "_edge_node_id" in frame.columns:
            return frame
        merged = frame.copy()
        merged["email_i"] = merged["email_i"].astype(str)
        merged["email_j"] = merged["email_j"].astype(str)
        merged = merged.merge(lookup, on=["email_i", "email_j"], how="left")
        if merged["_edge_node_id"].isna().any():
            raise ValueError(
                "split frame row(s) could not be mapped to full-dataset _edge_node_id; "
                "assign _edge_node_id on df before train/val/test split"
            )
        return merged

    return out_df, _attach(train_df), _attach(val_df), _attach(test_df)


def _split_masks_from_subframes(
    n: int,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if "_edge_node_id" not in train_df.columns:
        raise ValueError("train_df missing _edge_node_id; assign on full df before split")
    train_mask = torch.zeros(n, dtype=torch.bool)
    val_mask = torch.zeros(n, dtype=torch.bool)
    test_mask = torch.zeros(n, dtype=torch.bool)
    for frame, mask in (
        (train_df, train_mask),
        (val_df, val_mask),
        (test_df, test_mask),
    ):
        ids = frame["_edge_node_id"].to_numpy(dtype=np.int64, copy=True)
        mask[torch.as_tensor(ids, dtype=torch.long)] = True
    return train_mask, val_mask, test_mask


def _fmt_prob(v: float) -> str:
    return f"{v:.4f}" if math.isfinite(v) else "n/a"


def _chunked_index_list(indices: torch.Tensor, batch_size: int) -> list[torch.Tensor]:
    if indices.numel() == 0:
        return []
    bs = max(1, int(batch_size))
    return [indices[i : i + bs] for i in range(0, int(indices.numel()), bs)]


def _parse_epoch_sampling_flags(training_cfg: dict[str, Any]) -> dict[str, Any]:
    """Mirror ``run_pair_training`` epoch-sampling switches (same as explicit-only MLP / _14)."""
    sc_cfg = dict(training_cfg.get("semantic_cluster_sampling") or {})
    sc_enabled = bool(sc_cfg.get("enabled", False))
    redundancy_cfg = dict(training_cfg.get("cluster_redundancy_control") or {})
    balance_cfg = dict(training_cfg.get("train_balance") or {})
    redundancy_enabled = bool(redundancy_cfg.get("enabled", False)) and sc_enabled
    balance_enabled = bool(balance_cfg.get("enabled", False)) and sc_enabled
    cluster_epoch_sampling = redundancy_enabled or balance_enabled

    hpe_cfg = dict(training_cfg.get("hard_positive_emphasis") or {})
    hue_cfg = dict(training_cfg.get("hard_unlabeled_emphasis") or {})
    epc_cfg = dict(training_cfg.get("easy_positive_capping") or {})
    rne_cfg = dict(training_cfg.get("reliable_negative_emphasis") or {})

    def _optional_float(raw: Any) -> float | None:
        if raw is None or str(raw).strip().lower() in ("", "none", "null"):
            return None
        return float(raw)

    def _optional_int(raw: Any) -> int | None:
        if raw is None or str(raw).strip().lower() in ("", "none", "null"):
            return None
        return int(raw)

    shuffle_train_epoch = (
        (bool(hpe_cfg.get("enabled", False)) and bool(hpe_cfg.get("shuffle_each_epoch", True)))
        or (bool(hue_cfg.get("enabled", False)) and bool(hue_cfg.get("shuffle_each_epoch", True)))
        or (bool(epc_cfg.get("enabled", False)) and bool(epc_cfg.get("shuffle_each_epoch", True)))
        or (bool(rne_cfg.get("enabled", False)) and bool(rne_cfg.get("shuffle_each_epoch", True)))
        or (
            cluster_epoch_sampling
            and bool(redundancy_cfg.get("shuffle_each_epoch", True) or balance_cfg.get("shuffle_each_epoch", True))
        )
    )

    return {
        "sc_enabled": sc_enabled,
        "redundancy_cfg": redundancy_cfg,
        "balance_cfg": balance_cfg,
        "redundancy_enabled": redundancy_enabled,
        "balance_enabled": balance_enabled,
        "cluster_epoch_sampling": cluster_epoch_sampling,
        "hpe_enabled": bool(hpe_cfg.get("enabled", False)),
        "hpe_oversample_factor": float(hpe_cfg.get("oversample_factor", 1.0)),
        "hpe_cross_seed_component_only": bool(hpe_cfg.get("cross_seed_component_only", True)),
        "hpe_require_from_2hop": bool(hpe_cfg.get("require_from_2hop", True)),
        "hpe_max_source_count": _optional_int(hpe_cfg.get("max_source_count")),
        "hpe_exclude_from_rare_artifact": bool(hpe_cfg.get("exclude_from_rare_artifact", False)),
        "hpe_require_not_same_seed_component": bool(hpe_cfg.get("require_not_same_seed_component", True)),
        "hue_enabled": bool(hue_cfg.get("enabled", False)),
        "hue_oversample_factor": float(hue_cfg.get("oversample_factor", 1.0)),
        "hue_cross_seed_component_only": bool(hue_cfg.get("cross_seed_component_only", True)),
        "hue_require_from_2hop": bool(hue_cfg.get("require_from_2hop", True)),
        "hue_max_source_count": _optional_int(hue_cfg.get("max_source_count")),
        "hue_exclude_from_rare_artifact": bool(hue_cfg.get("exclude_from_rare_artifact", True)),
        "hue_require_not_same_seed_component": bool(hue_cfg.get("require_not_same_seed_component", False)),
        "hue_require_from_semantic_false": bool(hue_cfg.get("require_from_semantic_false", False)),
        "epc_enabled": bool(epc_cfg.get("enabled", False)),
        "epc_downsample_fraction": float(epc_cfg.get("downsample_fraction", 1.0)),
        "epc_same_seed_component_only": bool(epc_cfg.get("same_seed_component_only", True)),
        "epc_min_semantic_cosine": _optional_float(epc_cfg.get("min_semantic_cosine")),
        "epc_min_source_count": _optional_int(epc_cfg.get("min_source_count")),
        "epc_or_rule_across_conditions": bool(epc_cfg.get("or_rule_across_conditions", True)),
        "rne_enabled": bool(rne_cfg.get("enabled", False)),
        "rne_oversample_factor": float(rne_cfg.get("oversample_factor", 1.0)),
        "shuffle_train_epoch": shuffle_train_epoch,
    }


def _build_epoch_train_edge_indices(
    *,
    train_df: pd.DataFrame,
    edge_id_lookup: pd.DataFrame,
    training_cfg: dict[str, Any],
    split_seed: int,
    epoch: int,
    epoch_sampling: dict[str, Any],
    hard_pos_mask_train: pd.Series,
    hard_unl_mask_train: pd.Series,
    easy_pos_mask_train: pd.Series,
    reliable_neg_mask_train: pd.Series,
    rn_supervision_active: bool,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Per-epoch supervised edge-node index list (with repetition when MLP oversampling duplicates rows).
    """
    train_df_for_epoch = train_df
    cluster_epoch_diag: dict[str, Any] = {}
    if epoch_sampling["cluster_epoch_sampling"]:
        train_df_for_epoch, cluster_epoch_diag = build_train_epoch_cluster_aware(
            train_df,
            redundancy_cfg=epoch_sampling["redundancy_cfg"],
            balance_cfg=epoch_sampling["balance_cfg"],
            epoch_seed=int(split_seed) + int(epoch),
            include_reliable_negative_in_epoch=rn_supervision_active,
        )

    train_df_epoch, emphasis_epoch_diag = _build_train_df_epoch_emphasis(
        train_df_for_epoch,
        easy_pos_mask=easy_pos_mask_train,
        epc_enabled=epoch_sampling["epc_enabled"],
        epc_downsample_fraction=epoch_sampling["epc_downsample_fraction"],
        hard_pos_mask=hard_pos_mask_train,
        hpe_enabled=epoch_sampling["hpe_enabled"],
        hpe_oversample_factor=epoch_sampling["hpe_oversample_factor"],
        hard_unl_mask=hard_unl_mask_train,
        hue_enabled=epoch_sampling["hue_enabled"],
        hue_oversample_factor=epoch_sampling["hue_oversample_factor"],
        reliable_neg_mask=reliable_neg_mask_train,
        rne_enabled=epoch_sampling["rne_enabled"],
        rne_oversample_factor=epoch_sampling["rne_oversample_factor"],
        shuffle_each_epoch=epoch_sampling["shuffle_train_epoch"],
        epoch_seed=int(split_seed) + int(epoch),
    )

    lookup = edge_id_lookup[["email_i", "email_j", "_edge_node_id"]].copy()
    lookup["email_i"] = lookup["email_i"].astype(str)
    lookup["email_j"] = lookup["email_j"].astype(str)
    ep = train_df_epoch.copy()
    if "_edge_node_id" in ep.columns:
        ep = ep.drop(columns=["_edge_node_id"])
    ep["email_i"] = ep["email_i"].astype(str)
    ep["email_j"] = ep["email_j"].astype(str)
    merged = ep.merge(lookup, on=["email_i", "email_j"], how="left", validate="many_to_one")
    if merged["_edge_node_id"].isna().any():
        n_miss = int(merged["_edge_node_id"].isna().sum())
        raise ValueError(
            f"{n_miss} epoch train pair rows could not map to edge_node_id "
            "(check pair key alignment with full dataset)"
        )

    indices = torch.as_tensor(merged["_edge_node_id"].to_numpy(dtype=np.int64, copy=True), dtype=torch.long)
    diag: dict[str, Any] = {
        "epoch": int(epoch),
        "cluster_epoch_sampling": cluster_epoch_diag,
        "emphasis": emphasis_epoch_diag,
        "n_epoch_train_pair_rows": int(len(train_df_epoch)),
        "n_epoch_train_edge_steps": int(indices.numel()),
        "n_unique_edge_nodes_in_epoch": int(indices.unique().numel()),
    }
    if cluster_epoch_diag.get("train_balance"):
        tb = dict(cluster_epoch_diag.get("train_balance") or {})
        diag["train_balance"] = tb
        diag["effective_pos_to_unl_ratio"] = cluster_epoch_diag.get("effective_pos_to_unl_ratio")
        n_pos_b = int(tb.get("n_pos_before", 0))
        n_pos_a = int(tb.get("n_pos_after", 0))
        n_unl_b = int(tb.get("n_unl_before", 0))
        n_unl_a = int(tb.get("n_unl_after", 0))
        if n_pos_a < n_pos_b:
            diag["balance_cap_applied_to"] = "positive"
        elif n_unl_a < n_unl_b:
            diag["balance_cap_applied_to"] = "unlabeled"
        else:
            diag["balance_cap_applied_to"] = "none"
    elif emphasis_epoch_diag.get("n_pos_effective_epoch") is not None:
        n_pos = int(emphasis_epoch_diag.get("n_pos_effective_epoch", 0))
        n_unl = int(emphasis_epoch_diag.get("n_unl_effective_epoch", 0))
        diag["effective_pos_to_unl_ratio"] = float(n_pos / max(1, n_unl))
    return indices, diag


def _unique_edge_mask_from_indices(n: int, indices: torch.Tensor, device: torch.device) -> torch.Tensor:
    mask = torch.zeros(n, dtype=torch.bool, device=device)
    if indices.numel() > 0:
        mask[indices.unique()] = True
    return mask


def _pu_tensors_from_df(df: pd.DataFrame, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    is_pos = torch.as_tensor(df["is_positive"].to_numpy(dtype=bool, copy=True), device=device)
    is_unl = torch.as_tensor(df["is_unlabeled"].to_numpy(dtype=bool, copy=True), device=device)
    is_neg = torch.as_tensor(df["is_reliable_negative"].to_numpy(dtype=bool, copy=True), device=device)
    return is_pos, is_unl, is_neg


@torch.no_grad()
def _pu_separation_extras(
    logits: torch.Tensor,
    mask: torch.Tensor,
    is_pos: torch.Tensor,
    is_unl: torch.Tensor,
) -> dict[str, float]:
    """Diagnostic separation on masked rows (positive vs unlabeled only)."""
    if not mask.any():
        return {}
    lg = logits[mask].view(-1)
    pos_m = is_pos[mask]
    unl_m = is_unl[mask]
    out: dict[str, float] = {}
    if pos_m.any():
        lp = lg[pos_m]
        out["epoch_mean_logit_pos"] = float(lp.mean().item())
        out["epoch_median_logit_pos"] = float(lp.median().item())
    if unl_m.any():
        lu = lg[unl_m]
        out["epoch_mean_logit_unl"] = float(lu.mean().item())
        out["epoch_median_logit_unl"] = float(lu.median().item())
    pu_m = pos_m | unl_m
    if int(pos_m.sum()) > 0 and int(unl_m.sum()) > 0:
        try:
            from sklearn.metrics import average_precision_score, roc_auc_score

            labels = pos_m[pu_m].detach().cpu().numpy().astype(int)
            scores = torch.sigmoid(lg[pu_m]).detach().cpu().numpy()
            out["epoch_auc_pos_vs_unl"] = float(roc_auc_score(labels, scores))
            out["epoch_ap_pos_vs_unl"] = float(average_precision_score(labels, scores))
        except Exception:
            pass
    return out


@torch.no_grad()
def _eval_split_loss(
    model: EdgePairGnnModel,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    mask: torch.Tensor,
    is_pos: torch.Tensor,
    is_unl: torch.Tensor,
    is_neg: torch.Tensor,
    *,
    pair_loss_type: str,
    pi_p: float,
    pu_non_negative: bool,
    reliable_negative_loss_weight: float,
) -> tuple[float, dict[str, Any]]:
    if not mask.any():
        return float("nan"), {}
    model.eval()
    logits = model(x, edge_index)
    loss, diag = compute_pair_loss(
        logits[mask],
        is_pos[mask],
        is_unl[mask],
        pair_loss_type,
        pi_p=pi_p,
        pu_non_negative=pu_non_negative,
        is_reliable_negative=is_neg[mask] if is_neg is not None else None,
        reliable_negative_loss_weight=reliable_negative_loss_weight,
    )
    pu_epoch = {
        "epoch_mean_pos_prob": diag.get("mean_pos_prob", float("nan")),
        "epoch_mean_unl_prob": diag.get("mean_unl_prob", float("nan")),
        "epoch_score_separation": diag.get("score_separation", float("nan")),
        "epoch_sum_n_positive": diag.get("n_positive", 0),
        "epoch_sum_n_unlabeled": diag.get("n_unlabeled", 0),
        "epoch_mean_r_p_pos": diag.get("r_p_pos", float("nan")),
        "epoch_mean_r_p_neg": diag.get("r_p_neg", float("nan")),
        "epoch_mean_r_u_neg": diag.get("r_u_neg", float("nan")),
        "epoch_mean_neg_risk_raw": diag.get("neg_risk_raw", float("nan")),
        "epoch_mean_neg_risk_after_nn": diag.get("neg_risk_after_nn", float("nan")),
        **_pu_separation_extras(logits, mask, is_pos, is_unl),
    }
    return float(loss.item()), pu_epoch


def _should_log_epoch_detail(epoch: int, epochs: int) -> bool:
    if epoch == 1 or epoch == epochs:
        return True
    return epoch % 5 == 0


def _write_epoch_detail(
    *,
    epoch: int,
    epochs: int,
    tr_pu: dict[str, Any],
    va_pu: dict[str, Any],
    write: Any,
) -> None:
    if not _should_log_epoch_detail(epoch, epochs):
        return

    def _line(split: str, pu: dict[str, Any]) -> str:
        np_ = int(pu.get("epoch_sum_n_positive", 0))
        nu_ = int(pu.get("epoch_sum_n_unlabeled", 0))
        mp = float(pu.get("epoch_mean_pos_prob", float("nan")))
        mu = float(pu.get("epoch_mean_unl_prob", float("nan")))
        sep = float(pu.get("epoch_score_separation", float("nan")))
        auc = pu.get("epoch_auc_pos_vs_unl", float("nan"))
        ap = pu.get("epoch_ap_pos_vs_unl", float("nan"))
        auc_s = f"{auc:.4f}" if isinstance(auc, float) and auc == auc else "n/a"
        ap_s = f"{ap:.4f}" if isinstance(ap, float) and ap == ap else "n/a"
        mlp = pu.get("epoch_mean_logit_pos", float("nan"))
        mlu = pu.get("epoch_mean_logit_unl", float("nan"))
        return (
            f"   [{split}] n_pos={np_} n_unl={nu_} "
            f"P(pos)={_fmt_prob(mp)} P(unl)={_fmt_prob(mu)} sep={sep:+.4f} | "
            f"logit_pos={mlp:.3f} logit_unl={mlu:.3f} | AUC={auc_s} AP={ap_s} | "
            f"r_p_pos={pu.get('epoch_mean_r_p_pos', float('nan')):.4f} "
            f"r_p_neg={pu.get('epoch_mean_r_p_neg', float('nan')):.4f} "
            f"r_u_neg={pu.get('epoch_mean_r_u_neg', float('nan')):.4f} "
            f"neg_raw={pu.get('epoch_mean_neg_risk_raw', float('nan')):.4f} "
            f"neg_nn={pu.get('epoch_mean_neg_risk_after_nn', float('nan')):.4f}"
        )

    write(f"   📋 PU detail (epoch {epoch})")
    write(_line("train", tr_pu))
    write(_line("val", va_pu))


def save_edge_gnn_checkpoint(
    *,
    save_dir: Path,
    filename: str,
    model: EdgePairGnnModel,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_loss: float,
    payload: dict[str, Any],
) -> Path:
    save_dir.mkdir(parents=True, exist_ok=True)
    path = save_dir / filename
    ckpt = {
        **payload,
        "epoch": int(epoch),
        "val_loss": float(val_loss),
        "edge_gnn_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    torch.save(ckpt, path)
    return path


def _export_edge_gnn_pair_scores(
    *,
    run_dir: Path,
    model: EdgePairGnnModel,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    df: pd.DataFrame,
    edge_node_meta: pd.DataFrame,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    test_mask: torch.Tensor,
    device: torch.device,
) -> Path:
    model.eval()
    with torch.no_grad():
        logits = model(x, edge_index).detach().cpu().numpy()
    pu_score = 1.0 / (1.0 + np.exp(-logits.astype(np.float64)))

    split_col = np.full(len(df), "", dtype=object)
    tr = train_mask.cpu().numpy()
    va = val_mask.cpu().numpy()
    te = test_mask.cpu().numpy()
    split_col[tr] = "train"
    split_col[va] = "val"
    split_col[te] = "test"
    split_col[(tr | va | te) == False] = ""  # noqa: E712

    out = edge_node_meta.copy()
    out["logit"] = logits
    out["pu_score"] = pu_score
    if "pair_status" in df.columns:
        out["pair_status"] = df["pair_status"].astype(str).tolist()
    out["split"] = split_col.tolist()

    optional_cols = [
        "source_count",
        "from_seed",
        "from_semantic",
        "from_rare_artifact",
        "from_component",
        "from_2hop",
        "semantic_cosine_max",
        "time_gap_seconds_min",
        "same_seed_component_flag",
    ]
    for c in optional_cols:
        if c in df.columns:
            out[c] = df[c].tolist()

    out_path = run_dir / "edge_gnn_pair_scores.csv"
    out.to_csv(out_path, index=False)
    return out_path


def run_edge_gnn_pair_training(
    *,
    DEVICE: torch.device,
    TORCH_SEED: int,
    training_cfg: dict[str, Any],
    run_dir: str | Path,
    models_subdir: str,
    metrics_csv: str,
    training_config_json: str,
    df: pd.DataFrame,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    load_stats: dict[str, Any],
    pair_split_note: str,
    cluster_split_meta: dict[str, Any] | None,
    csv_path: Path,
    project_root: Path | None = None,
) -> dict[str, Any]:
    """Full-batch Edge-GNN training; returns same keys as ``run_pair_training``."""
    del project_root
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = run_dir / models_subdir
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    setup_parent = (run_dir / Path(training_config_json).parent).resolve()
    setup_parent.mkdir(parents=True, exist_ok=True)
    setup_summary_path = setup_parent / "pair_training_setup_summary.json"

    edge_cfg = edge_gnn_config_from_training_cfg(training_cfg)
    pair_loss_type = resolve_pair_loss_type(training_cfg)
    pi_p = float(training_cfg.get("pu_class_prior", training_cfg.get("pi_p", 0.1)))
    pu_non_negative = bool(training_cfg.get("pu_non_negative", True))
    reliable_negative_loss_weight = float(training_cfg.get("reliable_negative_loss_weight", 1.0))
    rn_active = reliable_negative_supervision_active(training_cfg, pair_loss_type)

    epochs = int(training_cfg["epochs"])
    lr = float(training_cfg["lr"])
    wd = float(training_cfg["wd"])
    early_stopping_patience = int(training_cfg["early_stopping_patience"])
    lr_reduce_patience = int(training_cfg["lr_reduce_patience"])
    lr_reduce_factor = float(training_cfg["lr_reduce_factor"])
    lr_reduce_min = float(training_cfg["lr_reduce_min"])
    model_save_name = str(training_cfg["model_save_name"])

    df, train_df, val_df, test_df = _ensure_edge_node_ids_on_splits(df, train_df, val_df, test_df)
    n = int(len(df))
    train_mask, val_mask, test_mask = _split_masks_from_subframes(n, train_df, val_df, test_df)

    feat_np = build_pair_feature_matrix(df)
    in_dim = int(feat_np.shape[1])
    x_cpu = torch.from_numpy(feat_np)
    num_gnn_layers = int(edge_cfg["num_gnn_layers"])
    local_head = str(edge_cfg.get("local_head", "edge_gnn_default"))
    combine_mode = str(edge_cfg.get("combine_mode", "concat_local_graph"))
    if num_gnn_layers <= 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_node_meta = pd.DataFrame(
            {
                "edge_node_id": np.arange(n, dtype=np.int64),
                "email_i": df["email_i"].astype(str).tolist(),
                "email_j": df["email_j"].astype(str).tolist(),
                "row_index": np.arange(n, dtype=np.int64),
            }
        )
        line_stats = {
            "line_graph_built": False,
            "message_passing": False,
            "num_gnn_layers": 0,
            "local_head": local_head,
            "num_edge_nodes": n,
            "num_line_edges": 0,
            "note": "num_gnn_layers=0; line graph skipped (local head only)",
        }
        print("[edge_gnn] num_gnn_layers=0 → skipping line graph construction")
        print(f"[edge_gnn] local_head={local_head}")
    else:
        edge_index, edge_node_meta, line_stats = build_candidate_edge_line_graph(
            df,
            max_neighbors_per_endpoint=edge_cfg.get("max_neighbors_per_endpoint"),
            rank_column=str(edge_cfg["rank_column"]),
        )
        line_stats = {
            **line_stats,
            "line_graph_built": True,
            "message_passing": True,
            "num_gnn_layers": num_gnn_layers,
            "local_head": local_head,
            "combine_mode": combine_mode,
        }
        print(f"[edge_gnn] local_head={local_head}")
        print(f"[edge_gnn] combine_mode={combine_mode}")

    print("[edge_gnn] line graph stats:", json.dumps(line_stats, indent=2))
    for k, v in line_stats.items():
        print(f"[edge_gnn] {k}={v}")

    try:
        device = DEVICE
        x = x_cpu.to(device)
        edge_index_dev = edge_index.to(device)
        is_pos, is_unl, is_neg = _pu_tensors_from_df(df, device)
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            raise RuntimeError(
                f"Edge-GNN OOM during tensor materialization: num_edge_nodes={n}, "
                f"num_line_edges={line_stats['num_line_edges']}, in_dim={in_dim}"
            ) from exc
        raise

    model = build_edge_pair_gnn_model(in_dim, training_cfg, edge_cfg=edge_cfg).to(device)
    if local_head == LOCAL_HEAD_EMAIL_PAIR_MLP_COMPATIBLE:
        if num_gnn_layers <= 0:
            print(
                "[edge_gnn] using EmailPairMLPScorer (explicit-only) as local head — "
                f"hidden={edge_cfg['hidden_dim']} dropout={edge_cfg['dropout']}"
            )
        else:
            print(
                "[edge_gnn] MLP-compatible local encoder + "
                f"GraphSAGE×{num_gnn_layers} combine_mode={combine_mode} — "
                f"hidden={edge_cfg['hidden_dim']} dropout={edge_cfg['dropout']}"
            )

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt,
        mode="min",
        factor=lr_reduce_factor,
        patience=lr_reduce_patience,
        min_lr=lr_reduce_min,
    )

    line_graph_path = setup_parent / "edge_line_graph.pt"
    torch.save(
        {
            "edge_index": edge_index.cpu(),
            "edge_node_meta": edge_node_meta,
            "line_graph_stats": line_stats,
            "line_graph_config": edge_cfg,
        },
        line_graph_path,
    )

    metrics_csv_path = os.path.join(run_dir, metrics_csv)
    with open(metrics_csv_path, mode="w", newline="") as f:
        csv.writer(f).writerow(PAIR_METRICS_HEADER)

    best_val = float("inf")
    patience_counter = 0
    best_state: dict[str, torch.Tensor] | None = None

    ckpt_payload_base = {
        "pair_encoder_backend": PAIR_ENCODER_EDGE_GNN,
        "model_config": {
            "in_dim": in_dim,
            **edge_cfg,
        },
        "pair_feature_columns": list(PAIR_FEATURE_COLUMNS),
        "line_graph_config": edge_cfg,
        "line_graph_stats": line_stats,
        "pair_loss_type": pair_loss_type,
        "pu_class_prior": pi_p,
        "pu_non_negative": pu_non_negative,
        "edge_index_shape": list(edge_index.shape),
        "torch_seed": int(TORCH_SEED),
    }

    training_config_path = run_dir / training_config_json
    training_config_path.parent.mkdir(parents=True, exist_ok=True)
    training_config_path.write_text(
        json.dumps(
            {
                **training_cfg,
                "pair_encoder_backend": PAIR_ENCODER_EDGE_GNN,
                "pair_dataset_csv": str(csv_path),
                "edge_gnn": edge_cfg,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    train_batch_size = int(
        edge_cfg.get("train_batch_size")
        or training_cfg.get("edge_gnn_train_batch_size")
        or training_cfg.get("pair_batch_size", 4096)
    )
    if num_gnn_layers <= 0:
        print(
            f"[edge_gnn] training loop: per-batch forward/backward/step "
            f"(pair_batch_size={train_batch_size}, same as _14 explicit MLP)"
        )
    else:
        print(
            f"[edge_gnn] training loop: full-graph forward per batch + backward/step "
            f"(pair_batch_size={train_batch_size}, num_gnn_layers={num_gnn_layers})"
        )
    split_seed = int(training_cfg.get("pair_split_seed", training_cfg.get("torch_seed", 42)))
    epoch_sampling = _parse_epoch_sampling_flags(training_cfg)
    edge_id_lookup = df[["email_i", "email_j", "_edge_node_id"]].copy()

    hard_pos_mask_train = _hard_positive_mask(
        train_df,
        cross_seed_component_only=epoch_sampling["hpe_cross_seed_component_only"],
        require_from_2hop=epoch_sampling["hpe_require_from_2hop"],
        max_source_count=epoch_sampling["hpe_max_source_count"],
        exclude_from_rare_artifact=epoch_sampling["hpe_exclude_from_rare_artifact"],
        require_not_same_seed_component=epoch_sampling["hpe_require_not_same_seed_component"],
    )
    hard_unl_mask_train = _hard_unlabeled_mask(
        train_df,
        cross_seed_component_only=epoch_sampling["hue_cross_seed_component_only"],
        require_from_2hop=epoch_sampling["hue_require_from_2hop"],
        max_source_count=epoch_sampling["hue_max_source_count"],
        exclude_from_rare_artifact=epoch_sampling["hue_exclude_from_rare_artifact"],
        require_not_same_seed_component=epoch_sampling["hue_require_not_same_seed_component"],
        require_from_semantic_false=epoch_sampling["hue_require_from_semantic_false"],
    )
    easy_pos_mask_train = _easy_positive_mask(
        train_df,
        same_seed_component_only=epoch_sampling["epc_same_seed_component_only"],
        min_semantic_cosine=epoch_sampling["epc_min_semantic_cosine"],
        min_source_count=epoch_sampling["epc_min_source_count"],
        or_rule_across_conditions=epoch_sampling["epc_or_rule_across_conditions"],
    )
    reliable_neg_mask_train = _safe_bool_series(train_df, "is_reliable_negative", default=False)

    n_train_split = int(train_mask.sum().item())
    n_val = int(val_mask.sum().item())
    n_test = int(test_mask.sum().item())
    epoch1_idx, epoch1_diag = _build_epoch_train_edge_indices(
        train_df=train_df,
        edge_id_lookup=edge_id_lookup,
        training_cfg=training_cfg,
        split_seed=split_seed,
        epoch=1,
        epoch_sampling=epoch_sampling,
        hard_pos_mask_train=hard_pos_mask_train,
        hard_unl_mask_train=hard_unl_mask_train,
        easy_pos_mask_train=easy_pos_mask_train,
        reliable_neg_mask_train=reliable_neg_mask_train,
        rn_supervision_active=rn_active,
    )
    n_epoch1_steps = int(epoch1_idx.numel())

    mp_label = "no-MP (local MLP)" if num_gnn_layers <= 0 else f"GraphSAGE×{num_gnn_layers}"
    print(
        f"\n🧠 Edge-GNN training | nodes={n:,} | line_edges={line_stats['num_line_edges']:,} "
        f"| {mp_label} | in_dim={in_dim} | loss={pair_loss_type} | π_p={pi_p}"
    )
    print(
        f"   splits: train={n_train_split:,} | val={n_val:,} | test={n_test:,} "
        f"| epoch-1 supervised steps={n_epoch1_steps:,} "
        f"(batch_size={train_batch_size}, batches/epoch≈{max(1, (n_epoch1_steps + train_batch_size - 1) // train_batch_size)})"
    )
    if epoch_sampling["sc_enabled"]:
        print(
            f"   🧩 semantic_cluster_sampling | train_balance={epoch_sampling['balance_enabled']} "
            f"| redundancy_cap={epoch_sampling['redundancy_enabled']} "
            f"| easy_positive_cap={epoch_sampling['epc_enabled']} "
            f"| hard_pos={epoch_sampling['hpe_enabled']} | hard_unl={epoch_sampling['hue_enabled']} "
            f"| reliable_neg_emphasis={epoch_sampling['rne_enabled']}"
        )
    if epoch1_diag.get("effective_pos_to_unl_ratio") is not None:
        cap_msg = ""
        if epoch1_diag.get("balance_cap_applied_to") == "positive":
            tb = dict(epoch1_diag.get("train_balance") or {})
            cap_msg = (
                f" | capped positives {tb.get('n_pos_before')}→{tb.get('n_pos_after')} "
                f"(keep unl {tb.get('n_unl_after')})"
            )
        elif epoch1_diag.get("balance_cap_applied_to") == "unlabeled":
            tb = dict(epoch1_diag.get("train_balance") or {})
            cap_msg = (
                f" | capped unlabeled {tb.get('n_unl_before')}→{tb.get('n_unl_after')} "
                f"(keep pos {tb.get('n_pos_after')})"
            )
        print(
            f"   ⚖️  epoch-1 effective pos:unl ratio={float(epoch1_diag['effective_pos_to_unl_ratio']):.3f} "
            f"(target={epoch_sampling['balance_cfg'].get('target_pos_to_unl_ratio', 'n/a')}){cap_msg}"
        )
    print(
        f"   ⚙️  early_stopping_patience={early_stopping_patience} | "
        f"ReduceLROnPlateau patience={lr_reduce_patience} factor={lr_reduce_factor} min_lr={lr_reduce_min}"
    )

    epoch_iter = tqdm(range(1, epochs + 1), desc="Edge-GNN epochs", unit="epoch")
    last_lr = float(opt.param_groups[0]["lr"])
    last_epoch_sampling_diag: dict[str, Any] = epoch1_diag

    for epoch in epoch_iter:
        train_idx, epoch_sampling_diag = _build_epoch_train_edge_indices(
            train_df=train_df,
            edge_id_lookup=edge_id_lookup,
            training_cfg=training_cfg,
            split_seed=split_seed,
            epoch=epoch,
            epoch_sampling=epoch_sampling,
            hard_pos_mask_train=hard_pos_mask_train,
            hard_unl_mask_train=hard_unl_mask_train,
            easy_pos_mask_train=easy_pos_mask_train,
            reliable_neg_mask_train=reliable_neg_mask_train,
            rn_supervision_active=rn_active,
        )
        last_epoch_sampling_diag = epoch_sampling_diag
        train_eval_mask = _unique_edge_mask_from_indices(n, train_idx, device)

        model.train()
        train_chunks = _chunked_index_list(train_idx, train_batch_size)
        batch_pu_diags: list[dict[str, Any]] = []
        loss_sum = 0.0
        n_batches = 0
        chunk_iter = tqdm(
            train_chunks,
            desc=f"  epoch {epoch}/{epochs} train batches",
            leave=False,
            unit="batch",
        )

        if num_gnn_layers <= 0:
            for chunk in chunk_iter:
                opt.zero_grad(set_to_none=True)
                try:
                    logits_chunk = model(x[chunk], edge_index_dev)
                    loss, batch_diag = compute_pair_loss(
                        logits_chunk,
                        is_pos[chunk],
                        is_unl[chunk],
                        pair_loss_type,
                        pi_p=pi_p,
                        pu_non_negative=pu_non_negative,
                        is_reliable_negative=is_neg[chunk] if rn_active else None,
                        reliable_negative_loss_weight=reliable_negative_loss_weight,
                    )
                except RuntimeError as exc:
                    if "out of memory" in str(exc).lower():
                        raise RuntimeError(
                            f"Edge-GNN OOM in train batch epoch={epoch} chunk_size={chunk.numel()}"
                        ) from exc
                    raise
                loss.backward()
                opt.step()
                loss_sum += float(loss.item())
                n_batches += 1
                batch_pu_diags.append(batch_diag)
                chunk_iter.set_postfix(loss=f"{float(loss.item()):.4f}", refresh=False)
            tr_loss = loss_sum / max(n_batches, 1)
            tr_pu = aggregate_epoch_pu_stats(batch_pu_diags, pair_loss_type)
        else:
            for chunk in chunk_iter:
                opt.zero_grad(set_to_none=True)
                try:
                    logits = model(x, edge_index_dev)
                    loss, batch_diag = compute_pair_loss(
                        logits[chunk],
                        is_pos[chunk],
                        is_unl[chunk],
                        pair_loss_type,
                        pi_p=pi_p,
                        pu_non_negative=pu_non_negative,
                        is_reliable_negative=is_neg[chunk] if rn_active else None,
                        reliable_negative_loss_weight=reliable_negative_loss_weight,
                    )
                except RuntimeError as exc:
                    if "out of memory" in str(exc).lower():
                        raise RuntimeError(
                            f"Edge-GNN OOM in train batch epoch={epoch} chunk_size={chunk.numel()} "
                            f"num_edge_nodes={n} num_line_edges={line_stats['num_line_edges']}"
                        ) from exc
                    raise
                loss.backward()
                opt.step()
                loss_sum += float(loss.item())
                n_batches += 1
                batch_pu_diags.append(batch_diag)
                chunk_iter.set_postfix(loss=f"{float(loss.item()):.4f}", refresh=False)
            tr_loss = loss_sum / max(n_batches, 1)
            tr_pu = aggregate_epoch_pu_stats(batch_pu_diags, pair_loss_type)

        tr_loss_eval, _tr_pu_eval_mask = _eval_split_loss(
            model,
            x,
            edge_index_dev,
            train_eval_mask,
            is_pos,
            is_unl,
            is_neg,
            pair_loss_type=pair_loss_type,
            pi_p=pi_p,
            pu_non_negative=pu_non_negative,
            reliable_negative_loss_weight=reliable_negative_loss_weight,
        )
        va_loss, va_pu = _eval_split_loss(
            model,
            x,
            edge_index_dev,
            val_mask,
            is_pos,
            is_unl,
            is_neg,
            pair_loss_type=pair_loss_type,
            pi_p=pi_p,
            pu_non_negative=pu_non_negative,
            reliable_negative_loss_weight=reliable_negative_loss_weight,
        )
        del tr_loss_eval, _tr_pu_eval_mask

        cur_lr = float(opt.param_groups[0]["lr"])
        epoch_iter.write(
            f"📊 epoch {epoch}/{epochs} | "
            f"train_loss={tr_loss:.4f} | val_loss={va_loss:.4f} | lr={cur_lr:.2e}"
        )
        tr_sep = float(tr_pu.get("epoch_score_separation", float("nan")))
        va_sep = float(va_pu.get("epoch_score_separation", float("nan")))
        va_auc = va_pu.get("epoch_auc_pos_vs_unl", float("nan"))
        va_auc_s = f"{va_auc:.4f}" if isinstance(va_auc, float) and va_auc == va_auc else "n/a"
        epoch_iter.write(
            f"   ✅ train  P(pos)={_fmt_prob(float(tr_pu.get('epoch_mean_pos_prob', float('nan'))))} "
            f"P(unl)={_fmt_prob(float(tr_pu.get('epoch_mean_unl_prob', float('nan'))))} "
            f"sep={tr_sep:+.4f}"
        )
        epoch_iter.write(
            f"   🔍 val    P(pos)={_fmt_prob(float(va_pu.get('epoch_mean_pos_prob', float('nan'))))} "
            f"P(unl)={_fmt_prob(float(va_pu.get('epoch_mean_unl_prob', float('nan'))))} "
            f"sep={va_sep:+.4f} AUC={va_auc_s}"
        )
        _write_epoch_detail(
            epoch=epoch,
            epochs=epochs,
            tr_pu=tr_pu,
            va_pu=va_pu,
            write=epoch_iter.write,
        )
        if _should_log_epoch_detail(epoch, epochs) and num_gnn_layers > 0 and model.uses_mlp_compatible_local_graph:
            graph_stats = model.graph_representation_stats(x, edge_index_dev, val_mask)
            if graph_stats:
                epoch_iter.write(
                    "   📐 graph_repr (val) "
                    f"h_local μ/σ={graph_stats['h_local_norm_mean']:.3f}/{graph_stats['h_local_norm_std']:.3f} "
                    f"h_graph μ/σ={graph_stats['h_graph_norm_mean']:.3f}/{graph_stats['h_graph_norm_std']:.3f} "
                    f"Δnorm μ={graph_stats['graph_delta_norm_mean']:.3f}"
                )
        epoch_iter.set_postfix(
            train_loss=f"{tr_loss:.4f}",
            val_loss=f"{va_loss:.4f}",
            lr=f"{cur_lr:.2e}",
            refresh=False,
        )

        with open(metrics_csv_path, mode="a", newline="") as f:
            csv.writer(f).writerow(
                metrics_row_pair_training(
                    epoch,
                    pair_loss_type=pair_loss_type,
                    pi_p=pi_p,
                    train_loss=tr_loss,
                    val_loss=va_loss,
                    train_pu=tr_pu,
                    val_pu=va_pu,
                    train_agg={},
                    val_agg={},
                )
            )

        improved = math.isfinite(va_loss) and va_loss < best_val
        if improved:
            best_val = va_loss
            patience_counter = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            save_edge_gnn_checkpoint(
                save_dir=ckpt_dir,
                filename=model_save_name,
                model=model,
                optimizer=opt,
                epoch=epoch,
                val_loss=va_loss,
                payload=ckpt_payload_base,
            )
            epoch_iter.write(f"   💾 new best val_loss={va_loss:.4f} → {ckpt_dir / model_save_name}")
        else:
            patience_counter += 1
            epoch_iter.write(
                f"   ⏳ no improvement ({patience_counter}/{early_stopping_patience}) | best_val={best_val:.4f}"
            )

        scheduler.step(va_loss if math.isfinite(va_loss) else tr_loss)
        new_lr = float(opt.param_groups[0]["lr"])
        if new_lr < last_lr - 1e-12:
            epoch_iter.write(f"   📉 ReduceLROnPlateau: lr {last_lr:.2e} → {new_lr:.2e}")
            last_lr = new_lr
        else:
            last_lr = new_lr

        if patience_counter >= early_stopping_patience:
            epoch_iter.write(f"🛑 early stopping at epoch {epoch}/{epochs} (patience={early_stopping_patience})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    te_loss, te_pu = _eval_split_loss(
        model,
        x,
        edge_index_dev,
        test_mask,
        is_pos,
        is_unl,
        is_neg,
        pair_loss_type=pair_loss_type,
        pi_p=pi_p,
        pu_non_negative=pu_non_negative,
        reliable_negative_loss_weight=reliable_negative_loss_weight,
    )
    te_sep = float(te_pu.get("epoch_score_separation", float("nan")))
    te_auc = te_pu.get("epoch_auc_pos_vs_unl", float("nan"))
    te_auc_s = f"{te_auc:.4f}" if isinstance(te_auc, float) and te_auc == te_auc else "n/a"
    print(
        f"\n🏁 Edge-GNN done | test_loss={te_loss:.4f} | "
        f"test P(pos)={_fmt_prob(float(te_pu.get('epoch_mean_pos_prob', float('nan'))))} "
        f"P(unl)={_fmt_prob(float(te_pu.get('epoch_mean_unl_prob', float('nan'))))} "
        f"sep={te_sep:+.4f} AUC={te_auc_s}"
    )

    scores_path = _export_edge_gnn_pair_scores(
        run_dir=run_dir,
        model=model,
        x=x,
        edge_index=edge_index_dev,
        df=df,
        edge_node_meta=edge_node_meta,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        device=device,
    )
    print(f"[edge_gnn] wrote scores -> {scores_path}")

    setup_summary = {
        "metadata": {
            "created_at_utc": datetime.now().isoformat(timespec="seconds"),
            "pair_dataset_csv": str(csv_path),
            "load_stats": load_stats,
            "pair_encoder_backend": PAIR_ENCODER_EDGE_GNN,
        },
        "pair_feature_columns": list(PAIR_FEATURE_COLUMNS),
        "num_pair_features": in_dim,
        "line_graph_stats": line_stats,
        "line_graph_config": edge_cfg,
        "line_graph_artifact": str(line_graph_path),
        "model_config": ckpt_payload_base["model_config"],
        "loss_config": {
            "pair_loss_type": pair_loss_type,
            "pu_class_prior": pi_p,
            "pu_non_negative": pu_non_negative,
        },
        "split_protocol": pair_split_note,
        "line_graph_leakage_note": (
            "Line graph is built on all candidate-edge nodes (full-graph forward). "
            "Supervised loss uses the same per-epoch train row sampling as explicit-only MLP "
            "(semantic cluster balance, easy-positive cap, hard-pos/unl/RN emphasis) mapped to edge_node_id."
        ),
        "epoch_sampling": {
            **epoch_sampling,
            "pair_split_seed": split_seed,
            "last_epoch_diagnostics": last_epoch_sampling_diag,
        },
        "cluster_split_hygiene": cluster_split_meta,
        "final_test": {"test_loss": te_loss, "test_pu_metrics": te_pu},
        "edge_gnn_pair_scores_csv": str(scores_path),
    }
    setup_summary_path.write_text(
        json.dumps(setup_summary, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    return {
        "model": None,
        "pair_scorer": None,
        "run_dir": str(run_dir),
        "best_checkpoint_path": str(ckpt_dir / model_save_name),
        "setup_summary_path": str(setup_summary_path),
    }
