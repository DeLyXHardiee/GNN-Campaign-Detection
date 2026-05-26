#!/usr/bin/env python3
"""
Investigate Edge-GNN training collapse vs explicit-only MLP (_14).

Usage (repo root):
  python core/GNN/debug/debug_edge_gnn_training.py
  python core/GNN/debug/debug_edge_gnn_training.py --run-dir output/runs/main_gnn_pu_1_no_ts_dedup_task_identity_19_edge_gnn_v1
  python core/GNN/debug/debug_edge_gnn_training.py --compare-mlp-dir output/runs/main_gnn_pu_1_no_ts_dedup_task_identity_14_only_mlp
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

_REPO = Path(__file__).resolve().parents[3]
_GNN = _REPO / "core" / "GNN"
for p in (str(_REPO), str(_REPO / "core"), str(_GNN)):
    if p not in sys.path:
        sys.path.insert(0, p)

from src.edge_candidate_line_graph import build_candidate_edge_line_graph  # noqa: E402
from src.edge_pair_gnn import EdgePairGnnModel, build_edge_pair_gnn_model, edge_gnn_config_from_training_cfg  # noqa: E402
from src.pair_train import (  # noqa: E402
    PAIR_FEATURE_COLUMNS,
    build_pair_feature_matrix,
    load_pair_training_dataframe,
    split_pairs_train_val_test,
)
from src.pu_loss import nnpu_binary_loss  # noqa: E402


def _load_pipeline_cfg() -> dict[str, Any]:
    p = _REPO / "pipeline_config.json"
    return json.loads(p.read_text(encoding="utf-8-sig"))


def _split_counts(df: pd.DataFrame, mask: np.ndarray, name: str) -> dict[str, int]:
    sub = df.loc[mask]
    return {
        f"{name}_rows": int(len(sub)),
        f"{name}_positive": int(sub["is_positive"].sum()),
        f"{name}_unlabeled": int(sub["is_unlabeled"].sum()),
        f"{name}_reliable_negative": int(sub.get("is_reliable_negative", pd.Series(False, index=sub.index)).sum()),
    }


def _feature_diagnostics(feat: np.ndarray) -> dict[str, Any]:
    cols = list(PAIR_FEATURE_COLUMNS)
    out: dict[str, Any] = {"shape": list(feat.shape), "columns": cols}
    per_col: list[dict[str, Any]] = []
    zero_cols = []
    for j, c in enumerate(cols):
        col = feat[:, j]
        per_col.append(
            {
                "col": c,
                "mean": float(np.mean(col)),
                "std": float(np.std(col)),
                "min": float(np.min(col)),
                "max": float(np.max(col)),
            }
        )
        if np.all(col == 0):
            zero_cols.append(c)
    out["per_column"] = per_col
    out["all_zero_columns"] = zero_cols
    out["nan_count"] = int(np.isnan(feat).sum())
    return out


def _line_graph_degree_stats(
    edge_index: torch.Tensor,
    n_nodes: int,
    is_pos: np.ndarray,
    is_unl: np.ndarray,
) -> dict[str, Any]:
    if edge_index.numel() == 0:
        deg = np.zeros(n_nodes, dtype=np.int64)
    else:
        ei = edge_index.cpu().numpy()
        deg = np.bincount(ei[0], minlength=n_nodes) + np.bincount(ei[1], minlength=n_nodes)
        deg = deg // 2 if edge_index.size(1) else deg
    pos_deg = deg[is_pos] if is_pos.any() else np.array([])
    unl_deg = deg[is_unl] if is_unl.any() else np.array([])
    pct = [50, 90, 95, 99]
    return {
        "num_edge_nodes": int(n_nodes),
        "num_line_edges_undirected_approx": int(edge_index.size(1) // 2),
        "mean_degree": float(deg.mean()),
        "max_degree": int(deg.max()) if len(deg) else 0,
        "degree_percentiles": {f"p{p}": float(np.percentile(deg, p)) for p in pct},
        "isolated_nodes": int((deg == 0).sum()),
        "mean_degree_positive": float(pos_deg.mean()) if len(pos_deg) else float("nan"),
        "mean_degree_unlabeled": float(unl_deg.mean()) if len(unl_deg) else float("nan"),
        "median_degree_positive": float(np.median(pos_deg)) if len(pos_deg) else float("nan"),
        "median_degree_unlabeled": float(np.median(unl_deg)) if len(unl_deg) else float("nan"),
    }


def _prob_logit_stats(
    logits: torch.Tensor,
    is_pos: torch.Tensor,
    is_unl: torch.Tensor,
    *,
    label: str,
) -> dict[str, Any]:
    with torch.no_grad():
        probs = torch.sigmoid(logits.view(-1))
        lp = logits[is_pos]
        lu = logits[is_unl]
        pp = probs[is_pos]
        pu = probs[is_unl]

    def _t(x: torch.Tensor) -> dict[str, float]:
        if x.numel() == 0:
            return {"mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan")}
        x = x.detach().cpu().float()
        return {
            "mean": float(x.mean()),
            "std": float(x.std()),
            "min": float(x.min()),
            "max": float(x.max()),
            "median": float(x.median()),
        }

    out = {
        "label": label,
        "logits_positive": _t(lp),
        "logits_unlabeled": _t(lu),
        "prob_positive": _t(pp),
        "prob_unlabeled": _t(pu),
        "frac_positive_above_0.5": float((pp > 0.5).float().mean()) if pp.numel() else float("nan"),
        "frac_unlabeled_above_0.5": float((pu > 0.5).float().mean()) if pu.numel() else float("nan"),
    }
    try:
        from sklearn.metrics import average_precision_score, roc_auc_score

        y = torch.zeros(int(is_pos.numel() + is_unl.numel()), dtype=torch.long)
        # rebuild on combined mask subset only
        mask = is_pos | is_unl
        y = is_pos[mask].long()
        scores = probs[mask].cpu().numpy()
        labels = is_pos[mask].cpu().numpy().astype(int)
        if len(np.unique(labels)) == 2:
            out["auc_pos_vs_unl"] = float(roc_auc_score(labels, scores))
            out["ap_pos_vs_unl"] = float(average_precision_score(labels, scores))
    except Exception as exc:
        out["sklearn_metrics_error"] = str(exc)
    return out


def _nnpu_at_constant_logit(logit_val: float, n_pos: int, n_unl: int, pi_p: float = 0.1) -> dict[str, float]:
    """Synthetic nnPU when all logits equal logit_val."""
    logits = torch.full((n_pos + n_unl,), float(logit_val))
    is_pos = torch.zeros(n_pos + n_unl, dtype=torch.bool)
    is_unl = torch.zeros(n_pos + n_unl, dtype=torch.bool)
    is_pos[:n_pos] = True
    is_unl[n_pos:] = True
    loss, diag = nnpu_binary_loss(logits, is_pos, is_unl, pi_p=pi_p, non_negative=True)
    return {**{k: float(v) for k, v in diag.items() if isinstance(v, (int, float))}, "loss": float(loss.item())}


def _compare_mlp_metrics(mlp_dir: Path) -> None:
    p = mlp_dir / "mlp" / "metrics.csv"
    if not p.is_file():
        p = mlp_dir / "metrics.csv"
    if not p.is_file():
        print(f"[compare] no metrics.csv under {mlp_dir}")
        return
    df = pd.read_csv(p)
    print(f"\n=== _14 explicit MLP metrics: {p} ===")
    for ep in (1, 5, 10, 20, 30):
        row = df[df["epoch"] == ep]
        if row.empty:
            continue
        r = row.iloc[0]
        print(
            f"epoch {ep}: train_loss={r['train_loss']:.4f} val_loss={r['val_loss']:.4f} "
            f"P(pos)={r['train_epoch_mean_pos_prob']:.4f} P(unl)={r['train_epoch_mean_unl_prob']:.4f} "
            f"sep={r['train_epoch_score_separation']:.4f} | "
            f"r_p_pos={r.get('train_epoch_mean_r_p_pos', float('nan'))} "
            f"r_u_neg={r.get('train_epoch_mean_r_u_neg', float('nan'))} "
            f"neg_raw={r.get('train_epoch_mean_neg_risk_raw', float('nan'))}"
        )


def _run_checkpoint_diagnostics(
    *,
    run_dir: Path,
    df: pd.DataFrame,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    is_pos: torch.Tensor,
    is_unl: torch.Tensor,
    training_cfg: dict[str, Any],
    device: torch.device,
) -> None:
    ckpt_path = run_dir / "edge_gnn" / "models" / "best_model.pt"
    if not ckpt_path.is_file():
        print(f"[checkpoint] missing {ckpt_path}")
        return
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    edge_cfg = edge_gnn_config_from_training_cfg(training_cfg)
    model = build_edge_pair_gnn_model(int(x.size(1)), training_cfg, edge_cfg=edge_cfg).to(device)
    model.load_state_dict(ckpt["edge_gnn_state_dict"])
    model.eval()
    x_dev = x.to(device)
    ei = edge_index.to(device)
    ip = is_pos.to(device)
    iu = is_unl.to(device)

    with torch.no_grad():
        h0 = model.input_mlp(x_dev)
        logits_full = model(x_dev, ei)
        # layer-wise
        h = h0
        stages = [("after_input_mlp", h0)]
        for i, conv in enumerate(model.convs):
            h = conv(h, ei)
            if i < len(model.convs) - 1:
                h = F.relu(h)
            stages.append((f"after_sage_{i + 1}", h))

    print(f"\n=== Checkpoint diagnostics ({ckpt_path.name}) ===")
    for name, h in stages:
        # proxy logits: linear probe from hidden to scalar using output layer only on that stage
        proxy = model.output_mlp(h).squeeze(-1)
        print(f"\n--- stage: {name} ---")
        print(_prob_logit_stats(proxy, ip, iu, label=name))

    print("\n--- final logits (full model) train/val ---")
    print(_prob_logit_stats(logits_full, ip, iu, label="all_nodes"))
    print(_prob_logit_stats(logits_full[train_mask], ip[train_mask], iu[train_mask], label="train_split"))
    print(_prob_logit_stats(logits_full[val_mask], ip[val_mask], iu[val_mask], label="val_split"))

    pi_p = float(training_cfg.get("pu_class_prior", 0.1))
    loss_tr, diag_tr = nnpu_binary_loss(logits_full[train_mask], ip[train_mask], iu[train_mask], pi_p=pi_p)
    print(f"\nTrain-mask nnPU loss={float(loss_tr.item()):.4f} terms={json.dumps(diag_tr, indent=2)}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path, default=None, help="Edge-GNN run dir with edge_gnn/models/best_model.pt")
    ap.add_argument("--compare-mlp-dir", type=Path, default=_REPO / "output/runs/main_gnn_pu_1_no_ts_dedup_task_identity_14_only_mlp")
    ap.add_argument("--pair-csv", type=Path, default=None)
    args = ap.parse_args()

    cfg = _load_pipeline_cfg()
    pt = dict(cfg.get("pair_training") or {})
    pair_csv = args.pair_csv or Path(str(pt.get("pair_dataset_csv", "")))
    if not pair_csv.is_absolute():
        pair_csv = (_REPO / pair_csv).resolve()

    print("=== Edge-GNN training investigation ===\n")
    df, load_stats = load_pair_training_dataframe(pair_csv)
    print("load_stats:", json.dumps(load_stats, indent=2))

    # Invariants
    df = df.copy()
    df["_edge_node_id"] = np.arange(len(df), dtype=np.int64)
    split_seed = int(pt.get("pair_split_seed", pt.get("torch_seed", 42)))
    val_ratio = float(pt.get("pair_val_ratio", 0.1))
    test_ratio = float(pt.get("pair_test_ratio", 0.1))
    train_df, val_df, test_df = split_pairs_train_val_test(df, val_ratio=val_ratio, test_ratio=test_ratio, split_seed=split_seed)

    n = len(df)
    assert len(df) == len(df["_edge_node_id"])
    assert (df["_edge_node_id"].to_numpy() == np.arange(n)).all()

    feat = build_pair_feature_matrix(df)
    assert feat.shape == (n, len(PAIR_FEATURE_COLUMNS))
    print("\n=== Feature matrix ===")
    print(json.dumps(_feature_diagnostics(feat), indent=2))

    edge_index, edge_meta, lg_stats = build_candidate_edge_line_graph(
        df,
        max_neighbors_per_endpoint=(pt.get("edge_gnn") or {}).get("max_neighbors_per_endpoint", 64),
        rank_column=str((pt.get("edge_gnn") or {}).get("rank_column", "semantic_cosine_max")),
    )
    assert len(edge_meta) == n
    assert (edge_meta["edge_node_id"].to_numpy() == np.arange(n)).all()
    assert edge_meta["email_i"].astype(str).tolist() == df["email_i"].astype(str).tolist()
    assert edge_meta["email_j"].astype(str).tolist() == df["email_j"].astype(str).tolist()
    print("\n=== Line graph ===")
    print(json.dumps(lg_stats, indent=2))

    is_pos_np = df["is_positive"].to_numpy(dtype=bool)
    is_unl_np = df["is_unlabeled"].to_numpy(dtype=bool)
    print("\n=== Line-graph degree by label ===")
    print(json.dumps(_line_graph_degree_stats(edge_index, n, is_pos_np, is_unl_np), indent=2))

    # Split masks
    train_ids = set(train_df["_edge_node_id"].astype(int).tolist())
    val_ids = set(val_df["_edge_node_id"].astype(int).tolist())
    test_ids = set(test_df["_edge_node_id"].astype(int).tolist())
    train_mask = torch.zeros(n, dtype=torch.bool)
    val_mask = torch.zeros(n, dtype=torch.bool)
    test_mask = torch.zeros(n, dtype=torch.bool)
    for i in train_ids:
        train_mask[i] = True
    for i in val_ids:
        val_mask[i] = True
    for i in test_ids:
        test_mask[i] = True

    print("\n=== Split counts (full df masks) ===")
    print(_split_counts(df, train_mask.numpy(), "train"))
    print(_split_counts(df, val_mask.numpy(), "val"))
    print(_split_counts(df, test_mask.numpy(), "test"))

    print("\n=== pair_status sample ===")
    cols = ["email_i", "email_j", "pair_status", "is_positive", "is_unlabeled", "_edge_node_id"]
    print(df[cols].head(8).to_string(index=False))
    bad = df.groupby("pair_status").agg(
        n=("pair_status", "size"),
        is_pos_sum=("is_positive", "sum"),
        is_unl_sum=("is_unlabeled", "sum"),
    )
    print("\n=== pair_status vs flags ===")
    print(bad.to_string())

    pi_p = float(pt.get("pu_class_prior", pt.get("pi_p", 0.1)))
    n_pos = int(df.loc[train_mask.numpy(), "is_positive"].sum())
    n_unl = int(df.loc[train_mask.numpy(), "is_unlabeled"].sum())
    print("\n=== nnPU synthetic (train counts, pi_p=0.1) ===")
    for lv in (-4.0, -2.0, -1.0, 0.0, 1.0, 2.0):
        print(f"logit={lv:+.1f}: {_nnpu_at_constant_logit(lv, n_pos, n_unl, pi_p=pi_p)}")

    print("\n=== Hypothesis: global negative collapse incentive ===")
    print(
        "When all logits are very negative, r_u_neg and r_p_neg are small; "
        "r_p_pos is large but weighted only by pi_p. neg_risk_raw = r_u_neg - pi*r_p_neg can stay small."
    )

    if args.compare_mlp_dir:
        _compare_mlp_metrics(args.compare_mlp_dir.resolve())

    device = torch.device("cpu")
    x_cpu = torch.from_numpy(feat)
    is_pos = torch.as_tensor(is_pos_np, dtype=torch.bool)
    is_unl = torch.as_tensor(is_unl_np, dtype=torch.bool)

    if args.run_dir:
        _run_checkpoint_diagnostics(
            run_dir=args.run_dir.resolve(),
            df=df,
            train_mask=train_mask,
            val_mask=val_mask,
            x=x_cpu,
            edge_index=edge_index,
            is_pos=is_pos,
            is_unl=is_unl,
            training_cfg=pt,
            device=device,
        )

    print("\n=== Investigation summary (see report in chat / docs) ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
