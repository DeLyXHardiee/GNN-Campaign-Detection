"""
Save / load Method 1 V2 run bundles (config, manifest, model, history, scored edges).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from analysis.utils.semantic_shard_edge_plausibility_v2_config import EdgePlausibilityV2Config
from analysis.utils.semantic_shard_edge_plausibility_v2_model import EdgePlausibilityMLP


def save_v2_model_checkpoint(
    path: str | Path,
    model: EdgePlausibilityMLP,
    cfg: EdgePlausibilityV2Config,
    *,
    extra: dict[str, Any] | None = None,
) -> None:
    """Write ``model.pt``-compatible checkpoint (``state_dict`` + dims)."""
    payload: dict[str, Any] = {
        "state_dict": model.state_dict(),
        "in_dim": model.net[0].in_features,
        "hidden_dim": cfg.hidden_dim,
        "hidden_dim2": cfg.hidden_dim2,
        "activation": cfg.activation,
    }
    if extra:
        payload["extra"] = extra
    torch.save(payload, Path(path).expanduser().resolve())


def save_v2_run_bundle(
    *,
    output_dir: str | Path,
    scored_edges_df: pd.DataFrame,
    cfg: EdgePlausibilityV2Config,
    feature_manifest: dict[str, Any],
    scaler_state: dict[str, Any],
    model: EdgePlausibilityMLP,
    training_history: list[dict[str, float]],
    views_debug_df: pd.DataFrame | None = None,
    last_epoch_model: EdgePlausibilityMLP | None = None,
    ranking_supervision_meta: dict[str, Any] | None = None,
) -> dict[str, str]:
    out = Path(output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    p_edges = out / "semantic_shard_step2_edges_scored.csv"
    p_cfg = out / "method2_config.json"
    p_fm = out / "feature_manifest.json"
    p_hist = out / "training_history.json"
    p_model = out / "model.pt"
    p_best = out / "model_best.pt"
    p_last = out / "model_last.pt"

    scored_edges_df.to_csv(p_edges, index=False)
    p_cfg.write_text(json.dumps(cfg.to_dict(), indent=2), encoding="utf-8")
    feature_manifest["scaler"] = scaler_state
    p_fm.write_text(json.dumps(feature_manifest, indent=2), encoding="utf-8")
    p_hist.write_text(json.dumps(training_history, indent=2), encoding="utf-8")
    save_v2_model_checkpoint(p_best, model, cfg, extra={"role": "best_val_loss"})
    save_v2_model_checkpoint(p_model, model, cfg, extra={"role": "best_val_loss_primary"})
    if last_epoch_model is not None:
        save_v2_model_checkpoint(p_last, last_epoch_model, cfg, extra={"role": "last_epoch"})
    else:
        save_v2_model_checkpoint(p_last, model, cfg, extra={"role": "last_epoch_same_as_best"})
    if views_debug_df is not None:
        p_dbg = out / "view_scores_debug.csv"
        views_debug_df.to_csv(p_dbg, index=False)
    if ranking_supervision_meta is not None:
        (out / "ranking_supervision_meta.json").write_text(
            json.dumps(ranking_supervision_meta, indent=2), encoding="utf-8"
        )

    # Fit summary (plausibility stats)
    pl = pd.to_numeric(scored_edges_df["edge_plausibility"], errors="coerce")
    fit_summary = {
        "edge_plausibility_mean": float(pl.mean()),
        "edge_plausibility_median": float(pl.median()),
        "edge_plausibility_p10": float(pl.quantile(0.1)),
        "edge_plausibility_p90": float(pl.quantile(0.9)),
        "n_edges": int(len(scored_edges_df)),
    }
    (out / "fit_summary.json").write_text(json.dumps(fit_summary, indent=2), encoding="utf-8")

    paths_out: dict[str, str] = {
        "scored_edges_csv": str(p_edges),
        "config_json": str(p_cfg),
        "feature_manifest_json": str(p_fm),
        "model_pt": str(p_model),
        "model_best_pt": str(p_best),
        "model_last_pt": str(p_last),
        "training_history_json": str(p_hist),
    }
    if ranking_supervision_meta is not None:
        paths_out["ranking_supervision_meta_json"] = str(out / "ranking_supervision_meta.json")
    return paths_out


def load_v2_model_checkpoint(path: str | Path, device: str = "cpu") -> tuple[EdgePlausibilityMLP, dict[str, Any]]:
    try:
        ck = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        ck = torch.load(path, map_location=device)
    m = EdgePlausibilityMLP(
        int(ck["in_dim"]),
        hidden_dim=int(ck["hidden_dim"]),
        hidden_dim2=int(ck["hidden_dim2"]),
        activation=str(ck.get("activation", "gelu")),
    )
    m.load_state_dict(ck["state_dict"])
    m.to(device)
    m.eval()
    return m, ck
