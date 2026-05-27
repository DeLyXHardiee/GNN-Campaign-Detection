"""Thesis GNN pair scoring (timestamp heterograph + ES100 + final pair universe)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any


def load_manifest(manifest_path: Path | None = None) -> dict[str, Any]:
    repo = Path(__file__).resolve().parents[2]
    path = manifest_path or (
        repo / "seed_candidate_workflow/configs/final_gnn_timestamp_es_thesis/final_gnn_timestamp_es_thesis.manifest.json"
    )
    return json.loads(Path(path).resolve().read_text(encoding="utf-8-sig"))


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_repo_path(repo: Path, rel: str) -> Path:
    p = Path(rel)
    return p if p.is_absolute() else (repo / p).resolve()


def training_run_dir(repo: Path, run_id: str) -> Path:
    return (repo / "output/runs" / run_id).resolve()


def scoring_run_dir(repo: Path, scoring_run_id: str) -> Path:
    return (repo / "seed_candidate_workflow/output/scoring_runs" / scoring_run_id).resolve()


def steps_dir(repo: Path, manifest: dict[str, Any]) -> Path:
    d = resolve_repo_path(repo, str(manifest.get("steps_output_dir") or "seed_candidate_workflow/output/final_gnn_timestamp_es_steps"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def thesis_dir(repo: Path, manifest: dict[str, Any]) -> Path:
    d = resolve_repo_path(
        repo, str(manifest.get("thesis_output_dir") or "seed_candidate_workflow/output/final_gnn_pair_scoring_timestamp_es_thesis")
    )
    d.mkdir(parents=True, exist_ok=True)
    return d


def community_sweep_csv(repo: Path, scoring_run_id: str, *, gt_slug: str = "ground_truth") -> Path:
    return (
        scoring_run_dir(repo, scoring_run_id)
        / "seed_candidate"
        / "community"
        / f"anchor_community_sweep__{gt_slug}.csv"
    )


def community_sweep_in_run_dir(run_dir: Path, *, gt_slug: str = "ground_truth") -> Path:
    return run_dir / "community" / f"anchor_community_sweep__{gt_slug}.csv"


def build_gnn_training_cfg(
    *,
    pair_dataset_csv: str,
    reference_training_config: Path,
    gnn_only: bool,
    project_root: Path | None = None,
    pi: float = 0.1,
    epochs: int = 100,
    early_stopping_patience: int = 10,
    save_best_val_checkpoint_history: bool = True,
) -> dict[str, Any]:
    """GNN pair supervision config: same hyperparams as reference _13/_15 except pair CSV + ES100."""
    repo = Path(project_root or repo_root())
    ref = json.loads(Path(reference_training_config).read_text(encoding="utf-8-sig"))

    pipeline_training: dict[str, Any] = {}
    pcfg_path = repo / "pipeline_config.json"
    if pcfg_path.is_file():
        pipeline_training = dict(json.loads(pcfg_path.read_text(encoding="utf-8-sig")).get("training") or {})

    cfg: dict[str, Any] = {**pipeline_training, **ref}
    cfg.update(
        {
            "pu_class_prior": float(pi),
            "pi_p": float(pi),
            "pair_dataset_csv": str(pair_dataset_csv),
            "training_objective": "pair_supervision",
            "pair_encoder_backend": "gnn",
            "epochs": int(epochs),
            "early_stopping_patience": int(early_stopping_patience),
            "save_best_val_checkpoint_history": bool(save_best_val_checkpoint_history),
        }
    )
    if gnn_only:
        cfg["pair_scorer_use_explicit_features"] = False
        cfg["pair_scorer_use_embedding_features"] = True
        cfg["pair_feature_dim_passed_to_scorer"] = 0
    else:
        cfg["pair_scorer_use_explicit_features"] = True
        cfg.setdefault("pair_scorer_use_embedding_features", True)
    for key, value in {
        "torch_seed": 42,
        "lr": 2e-4,
        "wd": 2e-5,
        "lr_reduce_patience": 4,
        "lr_reduce_factor": 0.5,
        "lr_reduce_min": 1e-6,
        "model_save_name": "best_model.pt",
        "hidden": 128,
        "out_dim": 128,
        "layers": 2,
        "dropout": 0.0,
    }.items():
        cfg.setdefault(key, value)
    return cfg


def read_training_stability(run_dir: Path, *, target_epochs: int | None = None) -> dict[str, Any]:
    from seed_candidate_workflow.utils.early_stopping_sanity_14_only_mlp import read_early_stopping_training_metrics

    return read_early_stopping_training_metrics(run_dir, target_epochs=target_epochs)


def resolve_best_community_from_sweep(sweep_csv: Path) -> dict[str, Any]:
    import pandas as pd

    df = pd.read_csv(sweep_csv, low_memory=False)
    df["v_measure"] = pd.to_numeric(df["v_measure"], errors="coerce")
    best = df.sort_values("v_measure", ascending=False).iloc[0]
    return {
        "algorithm": str(best.get("method") or ""),
        "threshold": float(best["min_edge_weight"]) if pd.notna(best.get("min_edge_weight")) else None,
        "resolution": float(best["resolution"]) if pd.notna(best.get("resolution")) else None,
        "homogeneity": float(best["homogeneity"]) if pd.notna(best.get("homogeneity")) else None,
        "completeness": float(best["completeness"]) if pd.notna(best.get("completeness")) else None,
        "v_measure": float(best["v_measure"]) if pd.notna(best.get("v_measure")) else None,
        "n_communities": int(float(best["n_communities"])) if pd.notna(best.get("n_communities")) else None,
    }


def write_graph_timestamp_summary(
    *,
    graph_pt: Path,
    meta_json: Path,
    out_path: Path,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    import torch

    meta = json.loads(meta_json.read_text(encoding="utf-8"))
    data = torch.load(graph_pt, map_location="cpu", weights_only=False)
    n_email = int(data["email"].num_nodes) if "email" in data.node_types else 0
    email_x_dim = int(data["email"].x.shape[1]) if "email" in data.node_types and hasattr(data["email"], "x") else None

    ts_raw = (meta.get("email_attrs") or {}).get("ts") or []
    ts_finite = [float(t) for t in ts_raw if t is not None and float(t) != 0.0]

    summary: dict[str, Any] = {
        "graph_stem": manifest.get("graph_stem"),
        "graph_pt": str(graph_pt.resolve()),
        "meta_json": str(meta_json.resolve()),
        "misp_json_path": manifest.get("misp_json_path"),
        "n_email_nodes": n_email,
        "node_types": list(data.node_types),
        "edge_types": [str(et) for et in data.edge_types],
        "email_feature_dim_after_projection": email_x_dim,
        "timestamp_feature_enabled": True,
        "zero_email_timestamps": False,
        "timestamp_representation": {
            "raw_scalar": "Unix seconds from MISP event date (email_attrs.ts in meta.json)",
            "in_email_feature_matrix": "First scalar column of pre-projection email feature matrix",
            "graph_normalization": (
                "Per email node-type: IQR outlier replacement to column median, then zero-mean unit-variance "
                "(core/graph/normalizer.py normalize_graph). Raw Unix timestamps are NOT fed unscaled to the GNN."
            ),
        },
        "n_emails_with_nonzero_raw_ts": int(len(ts_finite)),
        "raw_ts_unix_min": float(min(ts_finite)) if ts_finite else None,
        "raw_ts_unix_max": float(max(ts_finite)) if ts_finite else None,
        "seed_candidate_time_gating": "disabled (candidate generators unchanged; timestamps only in heterograph node features)",
        "deduplication": "incidents-lake-misp.dedup_task_identity.json (strict_task_message_identity representatives)",
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.is_file():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True
