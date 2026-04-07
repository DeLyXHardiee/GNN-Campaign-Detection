"""
Score shard edges with a saved Method 1 V2 MLP checkpoint and training-time scaler.

Used for post-training diagnostics and per-epoch checkpoint evaluation (no re-training).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from analysis.utils.semantic_shard_edge_plausibility_v2_config import EdgePlausibilityV2Config
from analysis.utils.semantic_shard_edge_plausibility_v2_features import build_v2_edge_feature_table
from analysis.utils.semantic_shard_edge_plausibility_v2_io import load_v2_model_checkpoint


def load_v2_feature_manifest(run_dir: str | Path) -> dict[str, Any]:
    p = Path(run_dir).expanduser().resolve() / "feature_manifest.json"
    if not p.is_file():
        raise FileNotFoundError(f"Missing feature manifest: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def load_v2_training_config(run_dir: str | Path) -> EdgePlausibilityV2Config:
    p = Path(run_dir).expanduser().resolve() / "method2_config.json"
    if not p.is_file():
        raise FileNotFoundError(f"Missing V2 config: {p}")
    return EdgePlausibilityV2Config.from_dict(json.loads(p.read_text(encoding="utf-8")))


def _apply_saved_scaler(
    X: np.ndarray,
    scaler_block: dict[str, Any],
    expected_names: list[str],
    actual_names: list[str],
) -> np.ndarray:
    if list(actual_names) != list(expected_names):
        raise ValueError(
            "Feature name order mismatch between current build and saved scaler.\n"
            f"  saved ({len(expected_names)}): {expected_names[:8]!r}...\n"
            f"  now   ({len(actual_names)}): {actual_names[:8]!r}...\n"
            "Rebuild Step 2 / use the same code version as training."
        )
    mean = np.asarray(scaler_block["mean"], dtype=np.float64)
    scale = np.asarray(scaler_block["scale"], dtype=np.float64)
    scale = np.where(scale == 0.0, 1.0, scale)
    return ((X - mean) / scale).astype(np.float64)


def build_normalized_v2_features(
    edges_df: pd.DataFrame,
    nodes_df: pd.DataFrame,
    cfg: EdgePlausibilityV2Config,
    feature_manifest: dict[str, Any],
) -> tuple[np.ndarray, list[str]]:
    features_df, feature_names, _manifest = build_v2_edge_feature_table(edges_df, nodes_df, cfg)
    scaler_block = feature_manifest.get("scaler") or {}
    expected = list(scaler_block.get("feature_names") or [])
    X = features_df.to_numpy(dtype=np.float64)
    Xn = _apply_saved_scaler(X, scaler_block, expected, feature_names)
    return Xn, feature_names


def score_edges_v2_checkpoint(
    edges_df: pd.DataFrame,
    nodes_df: pd.DataFrame,
    checkpoint_path: str | Path,
    *,
    run_dir: str | Path | None = None,
    cfg: EdgePlausibilityV2Config | None = None,
    feature_manifest: dict[str, Any] | None = None,
    device: str = "cpu",
) -> np.ndarray:
    """
    Return ``edge_plausibility`` scores in [0, 1], same length as ``edges_df``.

    ``run_dir`` defaults to ``checkpoint_path`` parent if it is ``.../checkpoints/epoch_XXXX.pt``,
    else the checkpoint's parent directory. Loads ``feature_manifest.json`` and ``method2_config.json``
    from that directory unless overridden.
    """
    ck_path = Path(checkpoint_path).expanduser().resolve()
    if run_dir is None:
        run_dir = ck_path.parent.parent if ck_path.parent.name == "checkpoints" else ck_path.parent
    run_dir = Path(run_dir).resolve()
    if cfg is None:
        cfg = load_v2_training_config(run_dir)
    if feature_manifest is None:
        feature_manifest = load_v2_feature_manifest(run_dir)

    Xn, _names = build_normalized_v2_features(edges_df, nodes_df, cfg, feature_manifest)
    model, _ck = load_v2_model_checkpoint(ck_path, device=device)
    dev = torch.device(device)
    with torch.no_grad():
        t = torch.tensor(Xn, dtype=torch.float32, device=dev)
        out = model(t).detach().float().cpu().numpy().astype(np.float64)
    return np.clip(out, 0.0, 1.0)


def list_epoch_checkpoints(run_dir: str | Path) -> list[tuple[int, Path]]:
    """Sorted list of (1-based epoch index, path) for ``checkpoints/epoch_XXXX.pt`` files."""
    d = Path(run_dir).expanduser().resolve() / "checkpoints"
    if not d.is_dir():
        return []
    out: list[tuple[int, Path]] = []
    for p in sorted(d.glob("epoch_*.pt")):
        stem = p.stem  # epoch_0001
        try:
            ep = int(stem.split("_", 1)[1])
        except (IndexError, ValueError):
            continue
        out.append((ep, p))
    out.sort(key=lambda x: x[0])
    return out
