from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

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
    from src.pair_train import PAIR_FEATURE_COLUMNS

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

    dev = torch.device(device)
    data = load_hetero_pt(str(graph_pt), to_undirected=to_undirected)
    data_cpu = data.to("cpu")
    metadata = data_cpu.metadata()

    ckpt = torch.load(str(ckpt_path), map_location=dev, weights_only=False)  # nosemgrep
    enc = train_cfg
    hidden = int(enc.get("hidden", 128))
    out_dim = int(enc.get("out_dim", 128))
    layers = int(enc.get("layers", 2))
    dropout = float(enc.get("dropout", 0.0))

    model = HeteroSAGE(metadata=metadata, hidden=hidden, out=out_dim, layers=layers, dropout=dropout).to(dev)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)

    pair_feat_dim = int(enc.get("pair_feature_dim_passed_to_scorer") or len(PAIR_FEATURE_COLUMNS))
    use_exp = bool(enc.get("pair_scorer_use_explicit_features", True))
    if not use_exp:
        pair_feat_dim = 0
    pair_scorer = build_email_pair_mlp_scorer(out_dim, pair_feat_dim, train_cfg).to(dev)
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
    }


@torch.no_grad()
def score_pair_rows(
    *,
    model: HeteroSAGE,
    pair_scorer: torch.nn.Module,
    data_cpu: Any,
    df_work: pd.DataFrame,
    device: torch.device,
    fanout: list[int],
    pair_batch_size: int,
    max_unique_emails: int,
    with_logits: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    from src.pair_graph_sampling import sample_hetero_around_pair_endpoints
    from src.pair_train import (
        build_pair_feature_matrix,
        forward_encoder_and_pair_logits,
        iter_pair_batches,
    )

    model.eval()
    pair_scorer.eval()
    n = len(df_work)
    scores = np.full(n, np.nan, dtype=np.float64)
    logits_out = np.full(n, np.nan, dtype=np.float64) if with_logits else None
    for chunk, gi, gj in iter_pair_batches(df_work, pair_batch_size, max_unique_emails):
        sample = sample_hetero_around_pair_endpoints(data_cpu, gi, gj, fanout)
        feats: torch.Tensor | None = None
        if pair_scorer.use_explicit_pair_features:
            feats = torch.from_numpy(build_pair_feature_matrix(chunk))
        logits, ok_m, _, _ = forward_encoder_and_pair_logits(
            model, pair_scorer, sample, feats, device
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
