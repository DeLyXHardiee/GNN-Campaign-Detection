"""
Training and full-graph scoring for Method 1 V2 edge plausibility MLP.

Supports validation split, per-epoch batch tqdm progress, best-checkpoint saving, LR reduction
on plateau, and early stopping.
"""

from __future__ import annotations

import copy
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from analysis.utils.semantic_shard_edge_plausibility_v2_config import EdgePlausibilityV2Config
from analysis.utils.semantic_shard_edge_plausibility_v2_features import build_v2_edge_feature_table
from analysis.utils.semantic_shard_edge_plausibility_v2_io import save_v2_model_checkpoint, save_v2_run_bundle
from analysis.utils.semantic_shard_edge_plausibility_v2_model import EdgePlausibilityMLP
from analysis.utils.semantic_shard_edge_plausibility_v2_buckets import (
    compute_ranking_bucket_masks,
    pools_from_masks,
    sample_ranking_pairs_hybrid,
    split_regime_plausibility_stats,
)
from analysis.utils.semantic_shard_edge_plausibility_v2_perturb import perturb_features
from analysis.utils.semantic_shard_edge_plausibility_v2_gt_diagnostics import (
    attach_edge_taxonomy,
    build_same_cross_hsli_masks,
    compact_gaps_from_scores,
    full_same_cross_separation_report,
    write_gt_separation_artifacts,
)
from analysis.utils.semantic_shard_edge_plausibility_v2_views import (
    build_view_scores_df,
    compute_agreement_scalar,
)
from analysis.utils.raw_gnn_notebook import load_ground_truth_structures


def _resolve_default_gt_json(project_root: Path) -> Path | None:
    """
    Full (non-dedup) ground truth only — same-campaign / cross-campaign diagnostics should align
    with the canonical labels file, not ``ground_truth_dedup.json``.
    """
    try:
        cfg_p = project_root / "pipeline_config.json"
        if cfg_p.is_file():
            cfg = json.loads(cfg_p.read_text(encoding="utf-8"))
            raw = (cfg.get("datasets") or {}).get("ground_truth_json")
            if isinstance(raw, str) and raw.strip():
                gp = Path(raw.strip())
                if not gp.is_absolute():
                    gp = (project_root / gp).resolve()
                if gp.is_file():
                    return gp
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        pass
    legacy = project_root / "data" / "groundtruth" / "ground_truth.json"
    if legacy.is_file():
        return legacy
    return None


def _resolve_default_assignments_csv(
    project_root: Path,
    step2_dir: Path | None,
) -> Path | None:
    ordered: list[Path] = []
    if step2_dir is not None:
        s2 = step2_dir.expanduser().resolve()
        par = s2.parent
        ordered.extend(
            [
                par / "semantic_shard_step1" / "semantic_shard_step1_assignments.csv",
                par / "semantic_shard_step1_graph" / "semantic_shard_step1_assignments.csv",
            ]
        )
    ordered.extend(
        [
            project_root
            / "analysis"
            / "output"
            / "semantic_shard_step1"
            / "semantic_shard_step1_assignments.csv",
            project_root
            / "analysis"
            / "output"
            / "semantic_shard_step1_graph"
            / "semantic_shard_step1_assignments.csv",
        ]
    )
    for p in ordered:
        if p.is_file():
            return p
    ao = project_root / "analysis" / "output"
    if not ao.is_dir():
        return None
    hits = [p for p in ao.rglob("semantic_shard_step1_assignments.csv") if p.is_file()]
    if not hits:
        return None

    def sort_key(p: Path) -> tuple[int, float]:
        parent = p.parent.name.lower()
        if parent == "semantic_shard_step1":
            tier = 0
        elif re.search(r"step1", parent):
            tier = 1
        else:
            tier = 2
        try:
            mtime = -p.stat().st_mtime
        except OSError:
            mtime = 0.0
        return tier, mtime

    hits.sort(key=sort_key)
    return hits[0]


def _resolve_gt_separation_inputs(
    cfg: EdgePlausibilityV2Config,
    gt_label_map: dict[str, Any] | None,
    assignments_df: pd.DataFrame | None,
) -> tuple[dict[str, Any] | None, pd.DataFrame | None]:
    """
    Merge caller-provided GT / assignments with config paths and repo defaults when
    ``cfg.log_gt_separation`` is True.
    """
    if not cfg.log_gt_separation:
        return gt_label_map, assignments_df

    gt_out = gt_label_map
    as_out = assignments_df

    explicit_gt = (
        Path(cfg.gt_separation_gt_json).expanduser().resolve()
        if cfg.gt_separation_gt_json
        else None
    )
    explicit_as = (
        Path(cfg.gt_separation_assignments_csv).expanduser().resolve()
        if cfg.gt_separation_assignments_csv
        else None
    )

    root: Path | None = None
    try:
        from analysis.utils.graph_structure_helpers import find_project_root

        root = find_project_root()
    except FileNotFoundError:
        pass

    step2: Path | None = None
    if cfg.gt_separation_step2_dir:
        step2 = Path(cfg.gt_separation_step2_dir).expanduser().resolve()

    path_gt_used: Path | None = None
    path_as_used: Path | None = None

    if gt_out is None:
        path_gt = explicit_gt
        if (path_gt is None or not path_gt.is_file()) and root is not None:
            path_gt = _resolve_default_gt_json(root)
        if path_gt is not None and path_gt.is_file():
            path_gt_used = path_gt
            gt_out, _, _ = load_ground_truth_structures(path_gt)

    if as_out is None:
        path_as = explicit_as
        if (path_as is None or not path_as.is_file()) and root is not None:
            path_as = _resolve_default_assignments_csv(root, step2)
        if path_as is not None and path_as.is_file():
            path_as_used = path_as
            as_out = pd.read_csv(path_as)

    if gt_out is None or as_out is None:
        print(
            "  [GT separation] log_gt_separation=True but GT JSON and/or assignments CSV "
            "were not found (and not passed in). Skipping diag gaps / v2_gt_score_separation artifacts.\n"
            "  Fix: set cfg.gt_separation_gt_json and cfg.gt_separation_assignments_csv (see STEP1 / GT paths "
            "in your Step 2 notebook), and/or cfg.gt_separation_step2_dir to the Step 2 graph folder "
            "(assignments are resolved next to its parent as semantic_shard_step1/...)."
        )
        return None, None

    if path_gt_used is not None and path_as_used is not None:
        print(
            f"  [GT separation] logging enabled — GT: {path_gt_used} | assignments: {path_as_used}"
        )

    return gt_out, as_out


def _pick_device(explicit: str | None) -> torch.device:
    if explicit:
        return torch.device(explicit)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _weighted_total(
    loss_rank: torch.Tensor,
    loss_stab: torch.Tensor,
    loss_hub: torch.Tensor,
    loss_aux: torch.Tensor,
    loss_anti_compress: torch.Tensor,
    cfg: EdgePlausibilityV2Config,
) -> torch.Tensor:
    return (
        cfg.loss_weight_ranking * loss_rank
        + cfg.loss_weight_stability * loss_stab
        + cfg.loss_weight_hub * loss_hub
        + cfg.loss_weight_agreement_aux * loss_aux
        + cfg.loss_weight_anti_compress * loss_anti_compress
    )


def _anti_compress_loss(
    p: torch.Tensor,
    cfg: EdgePlausibilityV2Config,
) -> tuple[torch.Tensor, float]:
    """Penalty when batch marginal std of scores is below ``anti_compress_target_std``."""
    if p.numel() < 2:
        z = torch.tensor(0.0, device=p.device, dtype=p.dtype)
        return z, float("nan")
    std = p.std(unbiased=False)
    bs = float(std.detach().cpu())
    if not cfg.anti_compress_enabled:
        z = torch.tensor(0.0, device=p.device, dtype=p.dtype)
        return z, bs
    target = torch.tensor(
        float(cfg.anti_compress_target_std),
        device=p.device,
        dtype=p.dtype,
    )
    short = F.relu(target - std.clamp_min(float(cfg.anti_compress_eps)))
    return short * short, bs


def _forward_batch_losses(
    model: EdgePlausibilityMLP,
    idx: np.ndarray,
    Xn: np.ndarray,
    agreement: np.ndarray,
    hub_raw: np.ndarray,
    edges_reset: pd.DataFrame,
    feature_names: list[str],
    perturb_groups: dict[str, list[str]],
    rng: np.random.Generator,
    cfg: EdgePlausibilityV2Config,
    dev: torch.device,
    *,
    index_pool_for_pairs: np.ndarray | None,
    ranking_pools: dict[str, np.ndarray],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Single training-style forward for indices ``idx``; returns component losses, total, pair meta."""
    xb = torch.tensor(Xn[idx], dtype=torch.float32, device=dev)
    ag_b = torch.tensor(agreement[idx], dtype=torch.float32, device=dev)
    p = model(xb)
    loss_ac, bs = _anti_compress_loss(p, cfg)

    x_np = Xn[idx]
    x_pert = perturb_features(
        x_np,
        rng,
        feature_names=feature_names,
        manifest_groups=perturb_groups,
        dropout_prob=cfg.feature_dropout_prob,
        noise_std=cfg.feature_noise_std,
        view_dropout_prob=cfg.view_dropout_prob,
        use_view_dropout=cfg.use_view_dropout_in_stability,
    )
    x_pert_t = torch.tensor(x_pert, dtype=torch.float32, device=dev)
    p_pert = model(x_pert_t)
    loss_stab = F.mse_loss(p, p_pert)
    loss_aux = F.mse_loss(p, ag_b)

    hub_batch = torch.tensor(hub_raw[idx], dtype=torch.float32, device=dev)
    mask = hub_batch > cfg.hub_dominance_threshold
    if mask.any():
        loss_hub = (hub_batch[mask] * p[mask]).mean()
    else:
        loss_hub = torch.tensor(0.0, device=dev)

    ih, il, pmeta, margins_np = sample_ranking_pairs_hybrid(
        edges_reset=edges_reset,
        agreement=agreement,
        pools=ranking_pools,
        rng=rng,
        cfg=cfg,
        index_pool=index_pool_for_pairs,
    )
    x_hi = torch.tensor(Xn[ih], dtype=torch.float32, device=dev)
    x_lo = torch.tensor(Xn[il], dtype=torch.float32, device=dev)
    p_hi = model(x_hi)
    p_lo = model(x_lo)
    mar_t = torch.as_tensor(margins_np, dtype=torch.float32, device=dev)
    loss_rank = F.relu(mar_t - (p_hi - p_lo)).mean()

    total = _weighted_total(loss_rank, loss_stab, loss_hub, loss_aux, loss_ac, cfg)
    pmeta["batch_score_std"] = bs
    return loss_rank, loss_stab, loss_hub, loss_aux, loss_ac, total, pmeta


def _run_epoch_train(
    model: EdgePlausibilityMLP,
    opt: torch.optim.Optimizer,
    train_idx: np.ndarray,
    Xn: np.ndarray,
    agreement: np.ndarray,
    hub_raw: np.ndarray,
    edges_reset: pd.DataFrame,
    feature_names: list[str],
    perturb_groups: dict[str, list[str]],
    rng: np.random.Generator,
    cfg: EdgePlausibilityV2Config,
    dev: torch.device,
    epoch: int,
    n_epochs: int,
    show_batch_bar: bool,
    ranking_pools: dict[str, np.ndarray],
) -> tuple[float, float, float, float, float, float, float, dict[str, Any]]:
    model.train()
    perm = rng.permutation(train_idx)
    sum_rank = sum_stab = sum_hub = sum_aux = sum_ac = sum_tot = 0.0
    sum_batch_std = 0.0
    n_batch_std = 0
    n_steps = 0
    pair_mode_ct: dict[str, int] = defaultdict(int)
    pair_type_ct: dict[str, int] = defaultdict(int)
    batch_starts = list(range(0, len(perm), cfg.batch_size))
    inner = tqdm(
        batch_starts,
        desc=f"train ep {epoch + 1}/{n_epochs}",
        unit="batch",
        leave=False,
        dynamic_ncols=True,
        disable=not show_batch_bar,
    )
    for start in inner:
        idx = perm[start : start + cfg.batch_size]
        if len(idx) < 2:
            continue
        opt.zero_grad()
        loss_rank, loss_stab, loss_hub, loss_aux, loss_ac, loss, pmeta = _forward_batch_losses(
            model,
            idx,
            Xn,
            agreement,
            hub_raw,
            edges_reset,
            feature_names,
            perturb_groups,
            rng,
            cfg,
            dev,
            index_pool_for_pairs=train_idx,
            ranking_pools=ranking_pools,
        )
        loss.backward()
        opt.step()
        sum_rank += float(loss_rank.detach().cpu())
        sum_stab += float(loss_stab.detach().cpu())
        sum_hub += float(loss_hub.detach().cpu())
        sum_aux += float(loss_aux.detach().cpu())
        sum_ac += float(loss_ac.detach().cpu())
        sum_tot += float(loss.detach().cpu())
        n_steps += 1
        bstd = pmeta.get("batch_score_std")
        if isinstance(bstd, float) and np.isfinite(bstd):
            sum_batch_std += bstd
            n_batch_std += 1
        pair_mode_ct[str(pmeta.get("mode", "?"))] += 1
        for k, v in pmeta.items():
            if isinstance(v, int) and k not in ("n_legacy", "n_fallback", "n_fallback_partial_bucket"):
                pair_type_ct[k] += int(v)
        inner.set_postfix(loss=f"{sum_tot / n_steps:.4f}", refresh=False)
    if n_steps == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, float("nan"), {}
    inv = 1.0 / n_steps
    mean_bs = float(sum_batch_std / max(1, n_batch_std))
    pair_stats: dict[str, Any] = {
        "pair_mode_batches": dict(pair_mode_ct),
        "pair_type_counts_in_pairs": dict(pair_type_ct),
        "n_batches": int(n_steps),
        "mean_batch_score_std": mean_bs,
    }
    return (
        sum_rank * inv,
        sum_stab * inv,
        sum_hub * inv,
        sum_aux * inv,
        sum_ac * inv,
        sum_tot * inv,
        mean_bs,
        pair_stats,
    )


@torch.no_grad()
def _run_epoch_eval(
    model: EdgePlausibilityMLP,
    val_idx: np.ndarray,
    Xn: np.ndarray,
    agreement: np.ndarray,
    hub_raw: np.ndarray,
    edges_reset: pd.DataFrame,
    feature_names: list[str],
    perturb_groups: dict[str, list[str]],
    rng: np.random.Generator,
    cfg: EdgePlausibilityV2Config,
    dev: torch.device,
    epoch: int,
    n_epochs: int,
    show_batch_bar: bool,
    ranking_pools: dict[str, np.ndarray],
) -> tuple[float, float, float, float, float, float, float, dict[str, Any]]:
    model.eval()
    perm = rng.permutation(val_idx)
    sum_rank = sum_stab = sum_hub = sum_aux = sum_ac = sum_tot = 0.0
    sum_batch_std = 0.0
    n_batch_std = 0
    n_steps = 0
    batch_starts = list(range(0, len(perm), cfg.batch_size))
    inner = tqdm(
        batch_starts,
        desc=f"val ep {epoch + 1}/{n_epochs}",
        unit="batch",
        leave=False,
        dynamic_ncols=True,
        disable=not show_batch_bar,
    )
    for start in inner:
        idx = perm[start : start + cfg.batch_size]
        if len(idx) < 2:
            continue
        loss_rank, loss_stab, loss_hub, loss_aux, loss_ac, loss, pmeta = _forward_batch_losses(
            model,
            idx,
            Xn,
            agreement,
            hub_raw,
            edges_reset,
            feature_names,
            perturb_groups,
            rng,
            cfg,
            dev,
            index_pool_for_pairs=val_idx,
            ranking_pools=ranking_pools,
        )
        sum_rank += float(loss_rank.cpu())
        sum_stab += float(loss_stab.cpu())
        sum_hub += float(loss_hub.cpu())
        sum_aux += float(loss_aux.cpu())
        sum_ac += float(loss_ac.cpu())
        sum_tot += float(loss.cpu())
        n_steps += 1
        bstd = pmeta.get("batch_score_std")
        if isinstance(bstd, float) and np.isfinite(bstd):
            sum_batch_std += bstd
            n_batch_std += 1
        inner.set_postfix(loss=f"{sum_tot / n_steps:.4f}", refresh=False)
    if n_steps == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, float("nan"), {}
    inv = 1.0 / n_steps
    mean_bs = float(sum_batch_std / max(1, n_batch_std))
    return (
        sum_rank * inv,
        sum_stab * inv,
        sum_hub * inv,
        sum_aux * inv,
        sum_ac * inv,
        sum_tot * inv,
        mean_bs,
        {},
    )


def train_and_score_edge_plausibility(
    edges_df: pd.DataFrame,
    nodes_df: pd.DataFrame,
    cfg: EdgePlausibilityV2Config,
    *,
    device: str | None = None,
    save_views_debug: bool = True,
    gt_label_map: dict[str, Any] | None = None,
    assignments_df: pd.DataFrame | None = None,
    gt_min_dominant_fraction: float = 0.7,
) -> dict[str, Any]:
    """
    Train MLP with ranking + stability (+ hub/aux + optional anti-compress), track val loss, LR decay.

    Loads **best validation** weights for final full-graph ``edge_plausibility`` scoring when a
    validation split exists; otherwise tracks best **train** loss.

    Optional **GT diagnostics only** (no supervision): when ``cfg.log_gt_separation`` is True (default),
    loads ``gt_label_map`` and ``assignments_df`` from ``cfg.gt_separation_*`` paths or repo defaults
    if not passed explicitly. Logs same-vs-cross and HS-LI gaps each epoch in ``training_history.json``
    and writes ``v2_gt_score_separation*.json/csv``. Set ``cfg.log_gt_separation=False`` to disable.
    """
    rng = np.random.default_rng(cfg.random_seed)
    torch.manual_seed(cfg.random_seed)
    dev = _pick_device(device)

    gt_label_map, assignments_df = _resolve_gt_separation_inputs(cfg, gt_label_map, assignments_df)

    features_df, feature_names, manifest = build_v2_edge_feature_table(edges_df, nodes_df, cfg)
    views_df = build_view_scores_df(edges_df)
    agreement = compute_agreement_scalar(views_df)

    n = len(edges_df)
    X = features_df.to_numpy(dtype=np.float64)
    scaler = StandardScaler()
    Xn = scaler.fit_transform(X).astype(np.float64)

    scaler_state = {
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "feature_names": feature_names,
    }

    hub_raw = (
        features_df["v2_infra_dominance"].to_numpy(dtype=np.float64)
        if "v2_infra_dominance" in features_df.columns
        else np.zeros(n, dtype=np.float64)
    )

    edges_reset = edges_df.reset_index(drop=True)
    perturb_groups = manifest["perturb_groups"]
    x_all_t = torch.tensor(Xn, dtype=torch.float32, device=dev)

    # Train / val split (unsupervised holdout)
    idx_all = np.arange(n, dtype=np.int64)
    rng.shuffle(idx_all)
    n_val = int(round(n * float(cfg.validation_fraction)))
    n_val = max(0, min(n - 2, n_val))  # keep at least 2 train
    if n_val < 2:
        n_val = 0
    val_idx = idx_all[:n_val] if n_val > 0 else np.array([], dtype=np.int64)
    train_idx = idx_all[n_val:] if n_val > 0 else idx_all
    use_val = len(val_idx) >= 2 and len(train_idx) >= 2

    masks, bucket_meta = compute_ranking_bucket_masks(
        features_df,
        views_df,
        agreement,
        hub_raw,
        cfg,
        train_idx=train_idx,
        val_idx=val_idx,
    )
    train_pools = pools_from_masks(masks, train_idx)
    val_pools = pools_from_masks(masks, val_idx)

    gt_ctx: dict[str, Any] | None = None
    if gt_label_map is not None and assignments_df is not None:
        th0 = bucket_meta.get("thresholds") or {}
        tsem0 = float(th0.get("thr_semantic_high", 0.0))
        tinf0 = float(th0.get("thr_infra_false_bridge_max", 0.0))
        sem_pre = pd.to_numeric(views_df["view_semantic"], errors="coerce").fillna(0.0).to_numpy(
            dtype=np.float64
        )
        inf_pre = pd.to_numeric(views_df["view_infra"], errors="coerce").fillna(0.0).to_numpy(
            dtype=np.float64
        )
        e_tax = attach_edge_taxonomy(
            edges_reset,
            assignments_df,
            gt_label_map,
            min_dominant_fraction=gt_min_dominant_fraction,
        )
        sm, cr, hsl = build_same_cross_hsli_masks(e_tax, sem_pre, inf_pre, tsem0, tinf0)
        gt_ctx = {"same": sm, "cross": cr, "hsli": hsl}

    ranking_supervision_meta: dict[str, Any] = {
        **bucket_meta,
        "anti_compress_config": {
            "enabled": bool(cfg.anti_compress_enabled),
            "loss_weight": float(cfg.loss_weight_anti_compress),
            "target_std": float(cfg.anti_compress_target_std),
            "eps": float(cfg.anti_compress_eps),
        },
        "ranking_supervision_mode_config": str(getattr(cfg, "ranking_supervision_mode", "buckets")),
        "pools_train_sizes": {k: int(len(train_pools[k])) for k in train_pools},
        "pools_val_sizes": {k: int(len(val_pools[k])) for k in val_pools},
        "pair_target_fractions": {
            "pos_vs_hard_neg_hsli": float(cfg.ranking_frac_pos_vs_hard_neg_hsli),
            "pos_vs_hard_neg_other": float(cfg.ranking_frac_pos_vs_hard_neg_other),
            "pos_vs_strong_neg": float(cfg.ranking_frac_pos_vs_strong_neg),
            "hard_neg_hsli_vs_strong_neg": float(cfg.ranking_frac_hard_neg_hsli_vs_strong_neg),
            "ranking_margin": float(cfg.ranking_margin),
            "ranking_margin_hsli": float(cfg.ranking_margin_hsli),
            "note": "Dominant pair: safe_pos > hard_neg_hsli with larger ranking_margin_hsli.",
        },
        "bucket_definitions_summary": (
            "V2.1+HS-LI split: strong_neg unchanged. hard_neg_hsli = high sem + clearly low infra (stricter quantile) "
            "+ ≥1 risk cue (weak local, hub, infra dominance, generic URL, weak mv_min, or high view spread). "
            "hard_neg_other = remaining false-bridge negatives in the old hard_neg union. "
            "strong_pos tightened (higher infra floor, local support, mv floor, lower hub cap, tighter spread). "
            "hard_neg = union(hsli, other) for diagnostics."
        ),
        "per_epoch_pair_supervision": [],
    }
    out_dir = cfg.output_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / "ranking_supervision_meta.json").write_text(
        json.dumps(ranking_supervision_meta, indent=2), encoding="utf-8"
    )

    in_dim = Xn.shape[1]
    model = EdgePlausibilityMLP(
        in_dim,
        hidden_dim=cfg.hidden_dim,
        hidden_dim2=cfg.hidden_dim2,
        activation=cfg.activation,
    ).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    current_lr = cfg.learning_rate

    history: list[dict[str, float]] = []
    best_metric_total = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    epochs_no_improve = 0
    epochs_no_improve_lr = 0
    best_ckpt_path = out_dir / "model_best.pt"

    last_model_copy: EdgePlausibilityMLP | None = None

    final_epoch = 0
    n_ep = max(1, int(cfg.epochs))
    show_batches = cfg.show_progress
    for epoch in range(cfg.epochs):
        final_epoch = epoch

        tr_r, tr_s, tr_h, tr_a, tr_ac, tr_tot, tr_bs, tr_pair = _run_epoch_train(
            model,
            opt,
            train_idx,
            Xn,
            agreement,
            hub_raw,
            edges_reset,
            feature_names,
            perturb_groups,
            rng,
            cfg,
            dev,
            epoch,
            n_ep,
            show_batches,
            train_pools,
        )

        if use_val:
            va_r, va_s, va_h, va_a, va_ac, va_tot, va_bs, _ = _run_epoch_eval(
                model,
                val_idx,
                Xn,
                agreement,
                hub_raw,
                edges_reset,
                feature_names,
                perturb_groups,
                rng,
                cfg,
                dev,
                epoch,
                n_ep,
                show_batches,
                val_pools,
            )
        else:
            va_tot = float("nan")
            va_r = va_s = va_h = va_a = va_ac = va_bs = float("nan")

        model.eval()
        with torch.no_grad():
            scores_full_ep = model(x_all_t).detach().cpu().numpy().astype(np.float64).ravel()
        scores_full_ep = np.clip(scores_full_ep, 0.0, 1.0)
        full_std_ep = float(np.std(scores_full_ep))

        row = {
            "epoch": float(epoch),
            "lr": float(current_lr),
            "train_loss_total": float(tr_tot),
            "train_loss_ranking": float(tr_r),
            "train_loss_stability": float(tr_s),
            "train_loss_hub": float(tr_h),
            "train_loss_agreement_aux": float(tr_a),
            "train_loss_anti_compress": float(tr_ac),
            "val_loss_total": float(va_tot),
            "val_loss_ranking": float(va_r),
            "val_loss_stability": float(va_s),
            "val_loss_hub": float(va_h),
            "val_loss_agreement_aux": float(va_a),
            "val_loss_anti_compress": float(va_ac),
            "train_mean_batch_score_std": float(tr_bs) if np.isfinite(tr_bs) else float("nan"),
            "val_mean_batch_score_std": float(va_bs) if np.isfinite(va_bs) else float("nan"),
            "score_std_full_graph": full_std_ep,
        }
        if gt_ctx is not None:
            cg = compact_gaps_from_scores(
                scores_full_ep,
                gt_ctx["same"],
                gt_ctx["cross"],
                gt_ctx["hsli"],
            )
            row["diag_gt_mean_gap_same_minus_cross"] = float(
                cg["all_labeled_mean_gap_same_minus_cross"]
            )
            row["diag_hsli_mean_gap_same_minus_cross"] = float(cg["hsli_mean_gap_same_minus_cross"])
        model.train()
        pm = tr_pair.get("pair_mode_batches", {}) if tr_pair else {}
        pt = tr_pair.get("pair_type_counts_in_pairs", {}) if tr_pair else {}
        row["pair_batches_buckets"] = float(pm.get("buckets", 0))
        row["pair_batches_fallback_teacher"] = float(pm.get("fallback_teacher", 0))
        row["pair_batches_fallback_partial"] = float(pm.get("fallback_teacher_partial_bucket", 0))
        row["pair_batches_legacy_teacher"] = float(pm.get("legacy_teacher", 0))
        row["pairs_drawn_pos_vs_hard_neg_hsli"] = float(pt.get("pos_vs_hard_neg_hsli", 0))
        row["pairs_drawn_pos_vs_hard_neg_other"] = float(pt.get("pos_vs_hard_neg_other", 0))
        row["pairs_drawn_pos_vs_strong_neg"] = float(pt.get("pos_vs_strong_neg", 0))
        row["pairs_drawn_hard_neg_hsli_vs_strong_neg"] = float(pt.get("hard_neg_hsli_vs_strong_neg", 0))
        row["pairs_drawn_hard_neg_other_vs_strong_neg"] = float(pt.get("hard_neg_other_vs_strong_neg", 0))
        row["pairs_drawn_pos_vs_hard_neg"] = float(
            pt.get("pos_vs_hard_neg_hsli", 0) + pt.get("pos_vs_hard_neg_other", 0)
        )
        row["pairs_drawn_hard_vs_strong_neg"] = float(
            pt.get("hard_neg_hsli_vs_strong_neg", 0) + pt.get("hard_neg_other_vs_strong_neg", 0)
        )
        history.append(row)
        ranking_supervision_meta["per_epoch_pair_supervision"].append(
            {"epoch": int(epoch), **tr_pair} if tr_pair else {"epoch": int(epoch)}
        )

        metric = va_tot if use_val else tr_tot
        if use_val:
            msg = (
                f"epoch {epoch}  train_loss={tr_tot:.5f}  val_loss={va_tot:.5f}  "
                f"lr={current_lr:.2e}"
            )
        else:
            msg = (
                f"epoch {epoch}  train_loss={tr_tot:.5f}  val_loss=n/a  lr={current_lr:.2e}"
            )
        print(msg)

        improved = metric < best_metric_total - 1e-7
        if improved:
            best_metric_total = metric
            best_state = copy.deepcopy(model.state_dict())
            extra_best: dict[str, Any] = {"epoch": epoch, "role": "best"}
            if use_val:
                extra_best["val_loss_total"] = float(va_tot)
            else:
                extra_best["train_loss_total"] = float(tr_tot)
            save_v2_model_checkpoint(best_ckpt_path, model, cfg, extra=extra_best)
            label = "val_loss" if use_val else "train_loss"
            print(f"  *** New best model ({label}={metric:.6f}) — saved {best_ckpt_path}")
            if use_val:
                epochs_no_improve = 0
                epochs_no_improve_lr = 0
        elif use_val:
            epochs_no_improve += 1
            epochs_no_improve_lr += 1

        if (
            use_val
            and epochs_no_improve_lr >= cfg.lr_reduce_patience
            and current_lr > cfg.min_learning_rate
        ):
            new_lr = max(cfg.min_learning_rate, current_lr * cfg.lr_reduce_factor)
            print(
                f"  No val improvement for {cfg.lr_reduce_patience} epochs — "
                f"reducing LR {current_lr:.2e} -> {new_lr:.2e}"
            )
            current_lr = new_lr
            for g in opt.param_groups:
                g["lr"] = current_lr
            epochs_no_improve_lr = 0

        if use_val and epochs_no_improve >= cfg.early_stop_patience:
            print(f"  Early stopping: no val improvement for {cfg.early_stop_patience} epochs.")
            break

        last_model_copy = copy.deepcopy(model).cpu()

        if cfg.save_every_epoch_checkpoint:
            ck_dir = out_dir / "checkpoints"
            ck_dir.mkdir(parents=True, exist_ok=True)
            cpu_m = model.cpu()
            save_v2_model_checkpoint(
                ck_dir / f"epoch_{epoch + 1:04d}.pt",
                cpu_m,
                cfg,
                extra={"epoch": int(epoch), "role": "epoch_end"},
            )
            model.to(dev)

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        scores = model(x_all_t).cpu().numpy().astype(np.float64)
    scores = np.clip(scores, 0.0, 1.0)

    out_edges = edges_df.copy()
    out_edges["edge_plausibility"] = scores

    gt_extra_paths: dict[str, str] = {}
    if gt_ctx is not None:
        sep_report = full_same_cross_separation_report(
            scores,
            gt_ctx["same"],
            gt_ctx["cross"],
            gt_ctx["hsli"],
        )
        gt_extra_paths = write_gt_separation_artifacts(out_dir, sep_report)
        ranking_supervision_meta["gt_score_separation_compact"] = {
            "mean_gap_same_minus_cross": sep_report.get("mean_gap_same_minus_cross"),
            "median_gap_same_minus_cross": sep_report.get("median_gap_same_minus_cross"),
            "hsli_mean_gap_same_minus_cross": sep_report.get("hsli", {}).get(
                "mean_gap_same_minus_cross"
            ),
            "hsli_median_gap_same_minus_cross": sep_report.get("hsli", {}).get(
                "median_gap_same_minus_cross"
            ),
        }

    th = bucket_meta.get("thresholds") or {}
    tsem = float(th.get("thr_semantic_high", 0.0))
    tinf = float(th.get("thr_infra_false_bridge_max", 0.0))
    sem_arr = pd.to_numeric(views_df["view_semantic"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    inf_arr = pd.to_numeric(views_df["view_infra"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    ranking_supervision_meta["high_sem_low_infra_regime_after_fit"] = {
        "thresholds_ref": {"thr_semantic_high": tsem, "thr_infra_false_bridge_max": tinf},
        "full": split_regime_plausibility_stats(sem_arr, inf_arr, scores, tsem, tinf, None),
        "train": split_regime_plausibility_stats(sem_arr, inf_arr, scores, tsem, tinf, train_idx),
        "val": split_regime_plausibility_stats(sem_arr, inf_arr, scores, tsem, tinf, val_idx),
    }

    model_cpu = model.cpu()
    (Path(out_dir) / "ranking_supervision_meta.json").write_text(
        json.dumps(ranking_supervision_meta, indent=2), encoding="utf-8"
    )
    paths = save_v2_run_bundle(
        output_dir=out_dir,
        scored_edges_df=out_edges,
        cfg=cfg,
        feature_manifest=manifest,
        scaler_state=scaler_state,
        model=model_cpu,
        training_history=history,
        views_debug_df=views_df if save_views_debug else None,
        last_epoch_model=last_model_copy,
        ranking_supervision_meta=ranking_supervision_meta,
    )
    paths = {**paths, **gt_extra_paths}
    return {
        "output_dir": str(out_dir),
        "paths": paths,
        "training_history": history,
        "scored_edges_df": out_edges,
        "feature_names": feature_names,
        "best_val_loss": float(best_metric_total) if use_val and best_state is not None else None,
        "best_train_loss": float(best_metric_total) if (not use_val) and best_state is not None else None,
        "epochs_run": final_epoch + 1,
    }
