from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


def _resolve_inference_device(device: str | torch.device) -> torch.device:
    """Map config device to a load/run device; fall back to CPU if backend is unavailable."""
    dev = torch.device(device) if isinstance(device, str) else device
    if dev.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    if dev.type == "mps":
        mps_ok = getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
        if not mps_ok:
            return torch.device("cpu")
    return dev


def load_pair_supervision_for_inference(
    *,
    run_dir: Path,
    graph_pt: Path,
    checkpoint_name: str = "best_model.pt",
    device: str = "cpu",
    to_undirected: bool = True,
) -> dict[str, Any]:
    from src.load_graph_data import load_hetero_pt
    from src.model import HeteroSAGE
    from src.pair_scorer import build_email_pair_mlp_scorer
    from src.pair_train import PAIR_ENCODER_MLP_RAW_EMAIL_X, PAIR_FEATURE_COLUMNS

    run_dir = Path(run_dir).resolve()
    graph_pt = Path(graph_pt).resolve()
    ckpt_path = run_dir / "models" / checkpoint_name
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    cfg_path = run_dir / "training_config.json"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"training_config.json not found under run_dir: {cfg_path}")
    with open(cfg_path, encoding="utf-8") as f:
        train_cfg = json.load(f)
    fanout = list(train_cfg.get("pair_fanout") or train_cfg.get("fanout") or [25, 15])
    pair_batch_size = int(train_cfg.get("pair_batch_size", 64))
    max_unique = int(train_cfg.get("pair_max_unique_emails_per_graph_batch", 2048))

    dev = _resolve_inference_device(device)
    data = load_hetero_pt(str(graph_pt), to_undirected=to_undirected)
    data_cpu = data.to("cpu")
    metadata = data_cpu.metadata()

    # map_location must be a device PyTorch can deserialize to (CPU if CUDA checkpoint on CPU-only host)
    ckpt = torch.load(str(ckpt_path), map_location=dev, weights_only=False)  # nosemgrep
    enc = train_cfg
    pair_backend = str(ckpt.get("pair_encoder_backend") or enc.get("pair_encoder_backend") or "").strip().lower()
    is_mlp_raw = pair_backend == PAIR_ENCODER_MLP_RAW_EMAIL_X

    hidden = int(enc.get("hidden", 128))
    out_dim = int(enc.get("out_dim", 128))
    layers = int(enc.get("layers", 2))
    dropout = float(enc.get("dropout", 0.0))

    model: HeteroSAGE | None
    if is_mlp_raw:
        model = None
    else:
        gnn_model = HeteroSAGE(metadata=metadata, hidden=hidden, out=out_dim, layers=layers, dropout=dropout).to(dev)
        gnn_model.load_state_dict(ckpt["model_state_dict"], strict=True)
        model = gnn_model

    pair_feat_dim = int(enc.get("pair_feature_dim_passed_to_scorer") or len(PAIR_FEATURE_COLUMNS))
    use_exp = bool(enc.get("pair_scorer_use_explicit_features", True))
    if not use_exp:
        pair_feat_dim = 0
    scorer_embed_dim = int(enc.get("raw_email_feature_dim") or out_dim)
    pair_scorer = build_email_pair_mlp_scorer(scorer_embed_dim, pair_feat_dim, train_cfg).to(dev)
    pair_scorer.load_state_dict(ckpt["pair_scorer_state_dict"], strict=True)

    return {
        "train_cfg": train_cfg,
        "model": model,
        "pair_scorer": pair_scorer,
        "data_cpu": data_cpu,
        "fanout": fanout,
        "pair_batch_size": pair_batch_size,
        "max_unique_emails": max_unique,
        "device": dev,
        "checkpoint_path": str(ckpt_path),
        "training_config_path": str(cfg_path),
        "pair_encoder_backend": PAIR_ENCODER_MLP_RAW_EMAIL_X if is_mlp_raw else "gnn",
    }


@torch.no_grad()
def score_pair_rows(
    *,
    model: Any,
    pair_scorer: torch.nn.Module,
    data_cpu: Any,
    df_work: pd.DataFrame,
    device: torch.device,
    fanout: list[int],
    pair_batch_size: int,
    max_unique_emails: int,
    with_logits: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    from src.pair_train import (
        build_pair_feature_matrix,
        forward_encoder_and_pair_logits,
        forward_raw_email_pair_logits,
        iter_pair_batches,
    )

    if model is not None:
        from src.pair_graph_sampling import sample_hetero_around_pair_endpoints

        model.eval()
    pair_scorer.eval()
    n = len(df_work)
    scores = np.full(n, np.nan, dtype=np.float64)
    logits_out = np.full(n, np.nan, dtype=np.float64) if with_logits else None
    for chunk, gi, gj in iter_pair_batches(df_work, pair_batch_size, max_unique_emails):
        feats: torch.Tensor | None = None
        if pair_scorer.use_explicit_pair_features:
            feats = torch.from_numpy(build_pair_feature_matrix(chunk))
        if model is not None:
            sample = sample_hetero_around_pair_endpoints(data_cpu, gi, gj, fanout)
            logits, ok_m, _, _ = forward_encoder_and_pair_logits(
                model, pair_scorer, sample, feats, device
            )
        else:
            logits, ok_m, _, _ = forward_raw_email_pair_logits(
                pair_scorer, data_cpu, gi, gj, feats, device
            )
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        log_np = logits.detach().cpu().numpy().reshape(-1)
        ok_np = ok_m.cpu().numpy().astype(bool)
        row_ids = chunk["_row"].to_numpy(dtype=np.int64, copy=False)
        for i in range(len(row_ids)):
            if ok_np[i]:
                ri = int(row_ids[i])
                scores[ri] = float(probs[i])
                if logits_out is not None:
                    logits_out[ri] = float(log_np[i])
    if with_logits:
        return scores, logits_out  # type: ignore[return-value]
    return scores
