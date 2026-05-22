from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


def _load_pair_feature_columns_from_setup_summary(training_config_path: Path) -> list[str] | None:
    """Match training-time pair MLP input (see pair_training_setup_summary.json)."""
    summary_path = training_config_path.parent / "pair_training_setup_summary.json"
    if not summary_path.is_file():
        return None
    try:
        raw = json.loads(summary_path.read_text(encoding="utf-8"))
        cols = raw.get("pair_feature_columns_ordered")
        if isinstance(cols, list) and cols:
            return [str(c) for c in cols]
    except Exception:
        return None
    return None


def _discover_training_config_paths(run_dir: Path) -> list[Path]:
    """Ordered training_config.json paths under a PU run directory."""
    run_dir = Path(run_dir).resolve()
    out: list[Path] = []
    seen: set[str] = set()

    def _add(p: Path) -> None:
        key = str(p.resolve())
        if key not in seen and p.is_file():
            seen.add(key)
            out.append(p.resolve())

    for rel in (
        "gnn/training_config.json",
        "training_config.json",
        "mlp/training_config.json",
    ):
        _add(run_dir / rel)
    for stage_rel in ("gnn/stage_result.json", "stage_result.json"):
        stage_path = run_dir / stage_rel
        if not stage_path.is_file():
            continue
        try:
            stage = json.loads(stage_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        for key in ("training_config_path",):
            raw = stage.get(key)
            if raw:
                _add(Path(raw))
        per_be = stage.get("pair_training_per_backend")
        if isinstance(per_be, dict):
            for info in per_be.values():
                if isinstance(info, dict) and info.get("training_config_path"):
                    _add(Path(str(info["training_config_path"])))
    for p in sorted(run_dir.rglob("training_config.json")):
        _add(p)
    return out


def _discover_checkpoint_paths(run_dir: Path, checkpoint_name: str) -> list[Path]:
    run_dir = Path(run_dir).resolve()
    out: list[Path] = []
    seen: set[str] = set()

    def _add(p: Path) -> None:
        key = str(p.resolve())
        if key not in seen and p.is_file():
            seen.add(key)
            out.append(p.resolve())

    for rel in (
        f"gnn/models/{checkpoint_name}",
        f"models/{checkpoint_name}",
        f"mlp/models/{checkpoint_name}",
    ):
        _add(run_dir / rel)
    for p in sorted(run_dir.rglob(checkpoint_name)):
        if p.parent.name == "models":
            _add(p)
    return out


def _pair_csv_hint_for_run_id(run_id: str, *, project_root: Path) -> str | None:
    """Default pair_training_dataset.csv from pipeline fragment / graph bundle layout."""
    frag_dir = project_root / "seed_candidate_workflow" / "configs" / "experiments"
    if frag_dir.is_dir():
        for frag in sorted(frag_dir.glob("pipeline_fragment*.json")):
            try:
                raw = json.loads(frag.read_text(encoding="utf-8"))
            except Exception:
                continue
            if str(raw.get("run_id") or "") == run_id:
                pt = (raw.get("pair_training") or {}).get("pair_dataset_csv")
                if pt:
                    return str(pt)
    bundle = (
        project_root
        / "seed_candidate_workflow"
        / "output"
        / "graph_bundles"
        / run_id
        / "pair_training"
        / run_id
        / "pair_training_dataset.csv"
    )
    if bundle.is_file():
        return str(bundle)
    return None


def resolve_pair_dataset_csv_path(
    run_dir: Path,
    *,
    pair_csv: Path | None = None,
    project_root: Path | None = None,
) -> Path:
    """Resolve pair_training_dataset.csv from explicit arg, training_config, or pipeline defaults."""
    if pair_csv is not None:
        p = Path(pair_csv).expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(f"pair CSV not found: {p}")
        return p

    run_dir = Path(run_dir).resolve()
    repo = (project_root or Path(__file__).resolve().parents[2]).resolve()

    for cfg_path in _discover_training_config_paths(run_dir):
        try:
            raw = json.loads(cfg_path.read_text(encoding="utf-8")).get("pair_dataset_csv")
        except Exception:
            continue
        if not raw:
            continue
        p = Path(str(raw))
        if not p.is_absolute():
            p = (repo / p).resolve()
        if p.is_file():
            return p

    hint = _pair_csv_hint_for_run_id(run_dir.name, project_root=repo)
    if hint:
        p = Path(hint)
        if not p.is_absolute():
            p = (repo / p).resolve()
        if p.is_file():
            return p

    raise FileNotFoundError(
        f"Could not resolve pair_training_dataset.csv for run_dir={run_dir}.\n"
        f"Pass --pair-csv explicitly, or train pair supervision first (e.g. python core/main.py gnn).\n"
        f"Expected bundle path (if built): {_pair_csv_hint_for_run_id(run_dir.name, project_root=repo)}"
    )


def resolve_pair_supervision_run_artifacts(
    run_dir: Path,
    *,
    checkpoint_name: str = "best_model.pt",
    project_root: Path | None = None,
) -> tuple[Path, Path]:
    """
    Resolve training_config.json and checkpoint for a pair-supervision run.

    Supports legacy layout (``<run_dir>/training_config.json``, ``<run_dir>/models/``)
    and per-backend layout (``<run_dir>/gnn/training_config.json``, ``<run_dir>/gnn/models/``).
    """
    run_dir = Path(run_dir).resolve()
    repo = (project_root or Path(__file__).resolve().parents[2]).resolve()

    if not run_dir.is_dir():
        hint = _pair_csv_hint_for_run_id(run_dir.name, project_root=repo)
        msg = (
            f"run_dir does not exist: {run_dir}\n"
            "Train the PU pair model first, for example:\n"
            "  python core/main.py gnn\n"
            "Use pipeline_config.json run_id / output_runs_root to confirm the run folder name."
        )
        if hint:
            msg += f"\nAfter training, pair CSV is typically at: {hint}"
        raise FileNotFoundError(msg)

    configs = _discover_training_config_paths(run_dir)
    ckpts = _discover_checkpoint_paths(run_dir, checkpoint_name)

    def _pair_score(cfg: Path, ckpt: Path) -> int:
        score = 0
        if "gnn" in str(cfg).replace("\\", "/").lower():
            score += 2
        if "gnn" in str(ckpt).replace("\\", "/").lower():
            score += 2
        return score

    if configs and ckpts:
        best: tuple[Path, Path] | None = None
        best_score = -1
        for cfg in configs:
            for ckpt in ckpts:
                sc = _pair_score(cfg, ckpt)
                if sc > best_score:
                    best_score = sc
                    best = (cfg, ckpt)
        assert best is not None
        return best

    if configs and not ckpts:
        tried = [str(run_dir / f"gnn/models/{checkpoint_name}"), str(run_dir / f"models/{checkpoint_name}")]
        raise FileNotFoundError(
            f"Found training_config at {configs[0]} but no checkpoint named {checkpoint_name!r}.\n"
            f"Tried: {', '.join(tried)}"
        )

    if ckpts and not configs:
        raise FileNotFoundError(
            f"Found checkpoint {ckpts[0]} but no training_config.json under {run_dir}.\n"
            "Re-run pair-supervision training or pass a run_dir that contains gnn/training_config.json."
        )

    tried_cfg = [str(run_dir / "gnn/training_config.json"), str(run_dir / "training_config.json")]
    hint = _pair_csv_hint_for_run_id(run_dir.name, project_root=repo)
    msg = (
        f"training_config.json not found under run_dir: {run_dir}\n"
        f"Checked: {', '.join(tried_cfg)} and stage_result.json references.\n"
        "Train pair supervision first (python core/main.py gnn) or verify --run-dir matches pipeline run_id."
    )
    if hint:
        msg += f"\nExpected pair dataset (graph bundle): {hint}"
    raise FileNotFoundError(msg)


def _load_train_cfg_for_inference(
    cfg_path: Path,
    ckpt_path: Path,
    *,
    device: torch.device,
) -> dict[str, Any]:
    if cfg_path.is_file():
        with open(cfg_path, encoding="utf-8") as f:
            return json.load(f)
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)  # nosemgrep
    enc = ckpt.get("encoder_config") or ckpt.get("pair_training_config")
    if isinstance(enc, dict) and enc:
        return enc
    raise FileNotFoundError(
        f"training_config.json missing at {cfg_path} and checkpoint {ckpt_path} has no encoder_config."
    )


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
    from src.model import build_pair_gnn_encoder
    from src.pair_scorer import build_email_pair_mlp_scorer
    from src.pair_train import (
        PAIR_ENCODER_EXPLICIT_ONLY,
        PAIR_ENCODER_MLP_RAW_EMAIL_X,
        PAIR_FEATURE_COLUMNS,
    )

    run_dir = Path(run_dir).resolve()
    graph_pt = Path(graph_pt).resolve()
    cfg_path, ckpt_path = resolve_pair_supervision_run_artifacts(
        run_dir, checkpoint_name=checkpoint_name
    )
    dev = _resolve_inference_device(device)
    train_cfg = _load_train_cfg_for_inference(cfg_path, ckpt_path, device=dev)
    fanout = list(train_cfg.get("pair_fanout") or train_cfg.get("fanout") or [25, 15])
    pair_batch_size = int(train_cfg.get("pair_batch_size", 64))
    max_unique = int(train_cfg.get("pair_max_unique_emails_per_graph_batch", 2048))

    data = load_hetero_pt(str(graph_pt), to_undirected=to_undirected)
    data_cpu = data.to("cpu")
    metadata = data_cpu.metadata()

    # map_location must be a device PyTorch can deserialize to (CPU if CUDA checkpoint on CPU-only host)
    ckpt = torch.load(str(ckpt_path), map_location=dev, weights_only=False)  # nosemgrep
    enc = train_cfg
    pair_backend = str(ckpt.get("pair_encoder_backend") or enc.get("pair_encoder_backend") or "").strip().lower()
    is_explicit_only = pair_backend == PAIR_ENCODER_EXPLICIT_ONLY
    is_mlp_raw = pair_backend == PAIR_ENCODER_MLP_RAW_EMAIL_X

    hidden = int(enc.get("hidden", 128))
    out_dim = int(enc.get("out_dim", 128))
    layers = int(enc.get("layers", 2))
    dropout = float(enc.get("dropout", 0.0))

    model: Any | None
    if is_mlp_raw or is_explicit_only:
        model = None
    else:
        model = build_pair_gnn_encoder(
            metadata,
            train_cfg,
            device=dev,
            state_dict=ckpt["model_state_dict"],
        )
        model.eval()

    pair_feat_dim = int(enc.get("pair_feature_dim_passed_to_scorer") or len(PAIR_FEATURE_COLUMNS))
    use_exp = bool(enc.get("pair_scorer_use_explicit_features", True))
    if not use_exp:
        pair_feat_dim = 0
    use_emb = bool(enc.get("pair_scorer_use_embedding_features", True))
    if is_explicit_only:
        use_emb = False
        train_cfg = {**train_cfg, "pair_scorer_use_embedding_features": False}
    if use_emb:
        scorer_embed_dim = int(enc.get("raw_email_feature_dim") or out_dim)
    else:
        scorer_embed_dim = 1
    pair_scorer = build_email_pair_mlp_scorer(scorer_embed_dim, pair_feat_dim, train_cfg).to(dev)
    pair_scorer.load_state_dict(ckpt["pair_scorer_state_dict"], strict=True)

    pair_feature_columns: list[str] | None = None
    if use_exp and pair_feat_dim > 0:
        pair_feature_columns = _load_pair_feature_columns_from_setup_summary(cfg_path)
        if pair_feature_columns is not None and len(pair_feature_columns) != pair_feat_dim:
            pair_feature_columns = None

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
        "pair_encoder_backend": (
            PAIR_ENCODER_EXPLICIT_ONLY
            if is_explicit_only
            else (PAIR_ENCODER_MLP_RAW_EMAIL_X if is_mlp_raw else "gnn")
        ),
        "pair_feature_columns": pair_feature_columns,
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
    pair_feature_columns: list[str] | None = None,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    from src.pair_train import (
        build_pair_feature_matrix,
        forward_encoder_and_pair_logits,
        forward_explicit_only_pair_logits,
        forward_raw_email_pair_logits,
        iter_pair_batches,
    )

    if model is not None:
        from src.pair_graph_sampling import sample_hetero_around_pair_endpoints

        model.eval()
    pair_scorer.eval()
    exp_dim = int(getattr(pair_scorer, "pair_feat_dim", 0)) if pair_scorer.use_explicit_pair_features else 0
    n = len(df_work)
    scores = np.full(n, np.nan, dtype=np.float64)
    logits_out = np.full(n, np.nan, dtype=np.float64) if with_logits else None
    for chunk, gi, gj in iter_pair_batches(df_work, pair_batch_size, max_unique_emails):
        feats: torch.Tensor | None = None
        if pair_scorer.use_explicit_pair_features:
            feats_np = build_pair_feature_matrix(chunk, pair_feature_columns)
            if exp_dim > 0 and feats_np.shape[1] != exp_dim:
                if feats_np.shape[1] > exp_dim:
                    feats_np = np.ascontiguousarray(feats_np[:, :exp_dim], dtype=np.float32)
                else:
                    pad = np.zeros((feats_np.shape[0], exp_dim - feats_np.shape[1]), dtype=np.float32)
                    feats_np = np.hstack([feats_np, pad])
            feats = torch.from_numpy(feats_np)
        if model is not None:
            sample = sample_hetero_around_pair_endpoints(data_cpu, gi, gj, fanout)
            logits, ok_m, _, _ = forward_encoder_and_pair_logits(
                model, pair_scorer, sample, feats, device
            )
        elif not getattr(pair_scorer, "use_embedding_features", True):
            logits, ok_m, _, _ = forward_explicit_only_pair_logits(
                pair_scorer, data_cpu, gi, gj, feats, device
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
