"""Writes analysis/email_teacher_contrastive_experiment_2.ipynb (training). Run from repo root."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "analysis" / "email_teacher_contrastive_experiment_2.ipynb"

MD = """# Email teacher–student (stage 2): train supervised contrastive encoder

Uses **teacher pseudo-communities** only (no GT). Loads fixed features from ``email.x`` (projected semantic block or full projected vector). Saves checkpoints, history, and embeddings under ``analysis/output/email_teacher_contrastive/<RUN_ID>/``.

**What changed in this pass:** earlier batches gave most anchors only *same-shard* positives, so the student could score well without learning **cross-shard** campaign structure. Training now **oversamples multi-shard teacher communities**, **biases within-community draws toward many distinct shards**, and optionally **drops single-shard-only communities** from training (ablation). **Semantic hard negatives** (raw-feature neighbors from *different* teacher communities) are mixed with random batch negatives. The student is a **residual** on the projected raw embedding (``proj(x) + α·MLP(x)``), not a full rewrite—smaller defaults encourage gentle refinement.

**Prerequisite:** run stage 1 notebook (or equivalent) so ``teacher_pseudo_labels.csv`` and ``train_val_split_summary.json`` exist in the same run directory you point to below. **Validation** still uses all val communities; **train-only multi-shard** affects training rows only."""


def _lines(s: str) -> list[str]:
    if not s.endswith("\n"):
        s += "\n"
    return [line + "\n" if not line.endswith("\n") else line for line in s.splitlines()]


def main() -> None:
    cells = [
        {"cell_type": "markdown", "metadata": {}, "id": "md0", "source": _lines(MD)},
        {
            "cell_type": "code",
            "metadata": {},
            "id": "cfg",
            "execution_count": None,
            "outputs": [],
            "source": _lines(
                r"""# ========== config ==========
from pathlib import Path

_here = Path.cwd().resolve()
PROJECT_ROOT = _here if (_here / "pipeline_config.json").is_file() else _here.parent

# New training run id (writes under email_teacher_contrastive/<RUN_ID>/)
RUN_ID = "student_train_001"

# Stage-1 artifacts directory (CSV + split summary + optional config)
STAGE1_RUN_DIR = PROJECT_ROOT / "analysis" / "output" / "email_teacher_contrastive" / "dev_run_001"

RUN_DIR = PROJECT_ROOT / "analysis" / "output" / "email_teacher_contrastive" / RUN_ID
RUN_DIR.mkdir(parents=True, exist_ok=True)

RNG_SEED = 42

# Graph features (same hetero checkpoint as shard pipeline)
GRAPH_PT = PROJECT_ROOT / "core" / "graph" / "output" / "incidents-lake-misp-large_hetero.pt"
GRAPH_META = GRAPH_PT.with_suffix(".meta.json")
FEATURE_MODE = "auto"  # auto | projected_bert128 | projected_full | raw_subject_body

# If train_val_split_summary.json has no cluster id lists (old runs), re-split with:
VAL_FRACTION_FALLBACK = 0.15
VAL_N_COMMUNITIES_FALLBACK = None

# Batch construction (CommunityAwareBatchIterator; cross-shard-aware when prefer_* is True)
N_COMMUNITIES_PER_BATCH = 8
N_EMAILS_PER_COMMUNITY = 4
PREFER_CROSS_SHARD_POSITIVES = True
MULTI_SHARD_OVERSAMPLE_FACTOR = 2.0
# When set, multi-shard communities get at least this many distinct shards in a draw (if sizes allow)
MIN_DISTINCT_SHARDS_PER_MULTI_SHARD_COMMUNITY = 2
STEPS_PER_EPOCH = None  # e.g. 200; None -> heuristic from train size
VAL_STEPS = None

# Ablate single-shard-only teacher communities from *training* (val unchanged)
TRAIN_ONLY_MULTI_SHARD_COMMUNITIES = False

# Semantic hard negatives (raw features, different teacher community)
USE_HARD_NEGATIVES = True
HARD_NEGATIVE_FRACTION = 0.35
HARD_NEGATIVE_TOPK = 48
HARD_NEGATIVE_NN_POOL = 96  # sklearn NN n_neighbors cap (>= TOPK)

# Student architecture: "residual" (default) or legacy "mlp"
STUDENT_ARCH = "residual"
HIDDEN_DIM = 256
NUM_MLP_HIDDEN_LAYERS = 1
HIDDEN_DIM_2 = None  # second hidden width if NUM_MLP_HIDDEN_LAYERS == 2
OUTPUT_DIM = 128
DROPOUT = 0.1
RESIDUAL_ALPHA = 0.2
RESIDUAL_ALPHA_LEARNABLE = False
# Legacy MLP-only widths (used only if STUDENT_ARCH == "mlp")
HIDDEN_DIM_1 = 512
HIDDEN_DIM_2_MLP = 256

# Train
LR = 1e-3
WEIGHT_DECAY = 1e-4
TEMPERATURE = 0.07
MAX_EPOCHS = 100
SCHEDULER_PATIENCE = 5
EARLY_STOP_PATIENCE = 10
CHECKPOINT_EVERY_EPOCH = False

print("RUN_DIR:", RUN_DIR.resolve())"""
            ),
        },
        {
            "cell_type": "code",
            "metadata": {},
            "id": "imports",
            "execution_count": None,
            "outputs": [],
            "source": _lines(
                r"""from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

assert (PROJECT_ROOT / "pipeline_config.json").is_file(), PROJECT_ROOT
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

with open(PROJECT_ROOT / "pipeline_config.json", "r", encoding="utf-8") as _f:
    TO_UNDIRECTED = bool(json.load(_f).get("to_undirected", True))

from analysis.utils.email_teacher_contrastive_data import (
    CommunityAwareBatchIterator,
    build_hard_negative_index_from_features,
    split_communities_train_val,
)
from analysis.utils.email_teacher_contrastive_features import (
    load_graph_email_features_for_external_ids,
)
from analysis.utils.email_teacher_contrastive_train import (
    build_student_from_train_config,
    eval_epoch,
    export_embeddings,
    save_checkpoint,
    save_training_history_csv,
    teacher_cluster_to_labels,
    train_one_epoch,
)

torch.manual_seed(RNG_SEED)
np_rng = np.random.default_rng(RNG_SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)"""
            ),
        },
        {
            "cell_type": "code",
            "metadata": {},
            "id": "data",
            "execution_count": None,
            "outputs": [],
            "source": _lines(
                r"""csv_path = STAGE1_RUN_DIR / "teacher_pseudo_labels.csv"
split_path = STAGE1_RUN_DIR / "train_val_split_summary.json"
cfg1_path = STAGE1_RUN_DIR / "config.json"

df_full = pd.read_csv(csv_path)
df_full["external_id"] = df_full["external_id"].astype(str)
df_full["teacher_cluster_id"] = df_full["teacher_cluster_id"].astype(str)
df_full["shard_id"] = df_full["shard_id"].astype(str)

if not split_path.is_file():
    raise FileNotFoundError(split_path)

split_doc = json.loads(split_path.read_text(encoding="utf-8"))
train_comm = split_doc.get("train_teacher_clusters")
val_comm = split_doc.get("val_teacher_clusters")

if train_comm is None or val_comm is None:
    c1 = json.loads(cfg1_path.read_text(encoding="utf-8")) if cfg1_path.is_file() else {}
    vf = c1.get("VAL_FRACTION", VAL_FRACTION_FALLBACK)
    vn = c1.get("VAL_N_COMMUNITIES", VAL_N_COMMUNITIES_FALLBACK)
    seed = int(c1.get("RNG_SEED", RNG_SEED))
    rng = np.random.default_rng(seed)
    sp = split_communities_train_val(df_full, rng, val_fraction=vf, val_n_communities=vn)
    train_df, val_df = sp.train_df, sp.val_df
    print("Recomputed split (old summary without cluster lists).")
else:
    train_df = df_full[df_full["teacher_cluster_id"].isin(train_comm)].copy()
    val_df = df_full[df_full["teacher_cluster_id"].isin(val_comm)].copy()
    print("Split from train_val_split_summary.json")

train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)

if TRAIN_ONLY_MULTI_SHARD_COMMUNITIES:
    _n_sh = train_df.groupby("teacher_cluster_id")["shard_id"].transform("nunique")
    train_df = train_df[_n_sh > 1].reset_index(drop=True)
    print("TRAIN_ONLY_MULTI_SHARD_COMMUNITIES: kept", len(train_df), "train rows")

print("train", len(train_df), "val", len(val_df))

# Global teacher id -> int (stable across train/val for metadata export)
all_tc = pd.concat([train_df["teacher_cluster_id"], val_df["teacher_cluster_id"]]).astype(str)
_, cluster_to_int = teacher_cluster_to_labels(all_tc.values)
y_train = np.array([cluster_to_int[c] for c in train_df["teacher_cluster_id"].astype(str)], dtype=np.int64)
y_val = np.array([cluster_to_int[c] for c in val_df["teacher_cluster_id"].astype(str)], dtype=np.int64)

train_eids = train_df["external_id"].astype(str).tolist()
val_eids = val_df["external_id"].astype(str).tolist()

X_tr, m_tr, info_tr = load_graph_email_features_for_external_ids(
    GRAPH_PT, GRAPH_META, train_eids, feature_mode=FEATURE_MODE, to_undirected=TO_UNDIRECTED
)
X_va, m_va, info_va = load_graph_email_features_for_external_ids(
    GRAPH_PT, GRAPH_META, val_eids, feature_mode=FEATURE_MODE, to_undirected=TO_UNDIRECTED
)
print("feature info train:", info_tr)
print("feature info val:", info_va)

train_df = train_df[m_tr].reset_index(drop=True)
y_train = y_train[m_tr]
X_tr = X_tr[m_tr]
train_eids = train_df["external_id"].astype(str).tolist()

val_df = val_df[m_va].reset_index(drop=True)
y_val = y_val[m_va]
X_va = X_va[m_va]
val_eids = val_df["external_id"].astype(str).tolist()

shard_train_np = train_df["shard_id"].astype(str).values
shard_val_np = val_df["shard_id"].astype(str).values

x_train = torch.from_numpy(X_tr).float()
y_train_t = torch.from_numpy(y_train).long()
x_val = torch.from_numpy(X_va).float()
y_val_t = torch.from_numpy(y_val).long()

feat_dim = x_train.shape[1]
print("feat_dim", feat_dim, "n_train", len(train_df), "n_val", len(val_df))

hard_neg_index = None
if USE_HARD_NEGATIVES:
    hard_neg_index = build_hard_negative_index_from_features(
        train_df,
        X_tr,
        max_neighbors_query=int(max(HARD_NEGATIVE_NN_POOL, HARD_NEGATIVE_TOPK + 8)),
    )
    print("hard negative index: train emails", len(train_df))

with open(RUN_DIR / "feature_load_info.json", "w", encoding="utf-8") as f:
    json.dump({"train": info_tr, "val": info_va, "n_train": len(train_df), "n_val": len(val_df)}, f, indent=2)"""
            ),
        },
        {
            "cell_type": "code",
            "metadata": {},
            "id": "trainloop",
            "execution_count": None,
            "outputs": [],
            "source": _lines(
                r"""_cfg_model = {
    "STUDENT_ARCH": STUDENT_ARCH,
    "HIDDEN_DIM": HIDDEN_DIM,
    "NUM_MLP_HIDDEN_LAYERS": NUM_MLP_HIDDEN_LAYERS,
    "HIDDEN_DIM_2": HIDDEN_DIM_2,
    "HIDDEN_DIM_1": HIDDEN_DIM_1,
    "HIDDEN_DIM_2": HIDDEN_DIM_2_MLP,
    "OUTPUT_DIM": OUTPUT_DIM,
    "DROPOUT": DROPOUT,
    "RESIDUAL_ALPHA": RESIDUAL_ALPHA,
    "RESIDUAL_ALPHA_LEARNABLE": RESIDUAL_ALPHA_LEARNABLE,
}
model = build_student_from_train_config(_cfg_model, feat_dim).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
    opt, mode="min", factor=0.5, patience=SCHEDULER_PATIENCE, min_lr=1e-6
)

est_batch = max(1, N_COMMUNITIES_PER_BATCH * N_EMAILS_PER_COMMUNITY)
steps_tr = STEPS_PER_EPOCH or max(30, len(train_df) // max(1, est_batch // 2))
steps_va = VAL_STEPS or max(15, len(val_df) // max(1, est_batch // 2))
print("steps_tr", steps_tr, "steps_va", steps_va)

history = []
best_val = float("inf")
best_epoch = 0
no_improve = 0
last_epoch = 0

for epoch in range(1, MAX_EPOCHS + 1):
    last_epoch = epoch
    rng_te = np.random.default_rng(RNG_SEED + 1_000 * epoch + 7)
    train_it = CommunityAwareBatchIterator(
        train_df,
        rng_te,
        n_communities_per_batch=N_COMMUNITIES_PER_BATCH,
        n_emails_per_community=N_EMAILS_PER_COMMUNITY,
        prefer_cross_shard_positives=PREFER_CROSS_SHARD_POSITIVES,
        multi_shard_oversample_factor=MULTI_SHARD_OVERSAMPLE_FACTOR,
        min_distinct_shards_per_multi_shard_community_in_batch=MIN_DISTINCT_SHARDS_PER_MULTI_SHARD_COMMUNITY,
    )
    rng_step = np.random.default_rng(RNG_SEED + 10_000 * epoch + 11)
    tr_metrics = train_one_epoch(
        model,
        opt,
        device,
        x_train,
        y_train_t,
        shard_train_np,
        train_eids,
        train_it,
        rng_step=rng_step,
        steps=steps_tr,
        temperature=TEMPERATURE,
        desc=f"train ep{epoch}",
        hard_neg_index=hard_neg_index,
        use_hard_negatives=USE_HARD_NEGATIVES,
        hard_negative_fraction=HARD_NEGATIVE_FRACTION,
        hard_negative_topk=HARD_NEGATIVE_TOPK,
    )

    rng_ve = np.random.default_rng(RNG_SEED + 5_000 + epoch + 3)
    val_it = CommunityAwareBatchIterator(
        val_df,
        rng_ve,
        n_communities_per_batch=N_COMMUNITIES_PER_BATCH,
        n_emails_per_community=N_EMAILS_PER_COMMUNITY,
        prefer_cross_shard_positives=PREFER_CROSS_SHARD_POSITIVES,
        multi_shard_oversample_factor=MULTI_SHARD_OVERSAMPLE_FACTOR,
        min_distinct_shards_per_multi_shard_community_in_batch=MIN_DISTINCT_SHARDS_PER_MULTI_SHARD_COMMUNITY,
    )
    va_metrics = eval_epoch(
        model,
        device,
        x_val,
        y_val_t,
        shard_val_np,
        val_eids,
        val_it,
        steps=steps_va,
        temperature=TEMPERATURE,
        desc=f"val ep{epoch}",
    )

    sched.step(va_metrics["loss"])
    lr = float(opt.param_groups[0]["lr"])
    row = {
        "epoch": epoch,
        "train_loss": tr_metrics["loss"],
        "val_loss": va_metrics["loss"],
        "lr": lr,
        "train_batch_cross_shard_pos_frac": tr_metrics.get("batch_cross_shard_positive_frac"),
        "train_batch_cross_shard_pos_mean": tr_metrics.get("batch_cross_shard_pos_mean"),
        "val_batch_cross_shard_pos_frac": va_metrics.get("val_batch_cross_shard_positive_frac"),
        "val_batch_cross_shard_pos_mean": va_metrics.get("val_batch_cross_shard_pos_mean"),
        "val_pos_cos_mean": va_metrics.get("val_pos_cos_mean"),
        "val_neg_cos_mean": va_metrics.get("val_neg_cos_mean"),
        "train_mean_cos_hard_neg": tr_metrics.get("train_mean_cos_hard_neg"),
        "train_mean_cos_rand_neg": tr_metrics.get("train_mean_cos_rand_neg"),
    }
    history.append(row)
    print(
        f"Epoch {epoch:03d}  train_loss={row['train_loss']:.4f}  val_loss={row['val_loss']:.4f}  lr={lr:.2e}"
    )

    meta_ckpt = {
        "RUN_ID": RUN_ID,
        "epoch": epoch,
        "feat_dim": feat_dim,
        "cluster_map_size": len(cluster_to_int),
    }
    if CHECKPOINT_EVERY_EPOCH:
        save_checkpoint(
            RUN_DIR / f"checkpoint_epoch_{epoch:04d}.pt",
            epoch=epoch,
            model=model,
            optimizer=opt,
            scheduler_state=sched.state_dict(),
            meta=meta_ckpt,
        )

    if va_metrics["loss"] < best_val - 1e-9:
        best_val = va_metrics["loss"]
        best_epoch = epoch
        no_improve = 0
        save_checkpoint(
            RUN_DIR / "checkpoint_best.pt",
            epoch=epoch,
            model=model,
            optimizer=opt,
            scheduler_state=sched.state_dict(),
            meta={**meta_ckpt, "best_val_loss": best_val},
        )
    else:
        no_improve += 1

    if no_improve >= EARLY_STOP_PATIENCE:
        print(f"Early stopping at epoch {epoch} (no val improvement for {EARLY_STOP_PATIENCE} epochs). Best: {best_epoch}")
        break

save_checkpoint(
    RUN_DIR / "checkpoint_final.pt",
    epoch=last_epoch,
    model=model,
    optimizer=opt,
    scheduler_state=sched.state_dict(),
    meta={**meta_ckpt, "best_val_loss": best_val, "best_epoch": best_epoch},
)

cfg_out = {
    "RUN_ID": RUN_ID,
    "STAGE1_RUN_DIR": str(STAGE1_RUN_DIR.resolve()),
    "GRAPH_PT": str(GRAPH_PT.resolve()),
    "FEATURE_MODE": FEATURE_MODE,
    "RNG_SEED": RNG_SEED,
    "N_COMMUNITIES_PER_BATCH": N_COMMUNITIES_PER_BATCH,
    "N_EMAILS_PER_COMMUNITY": N_EMAILS_PER_COMMUNITY,
    "PREFER_CROSS_SHARD_POSITIVES": PREFER_CROSS_SHARD_POSITIVES,
    "MULTI_SHARD_OVERSAMPLE_FACTOR": MULTI_SHARD_OVERSAMPLE_FACTOR,
    "MIN_DISTINCT_SHARDS_PER_MULTI_SHARD_COMMUNITY": MIN_DISTINCT_SHARDS_PER_MULTI_SHARD_COMMUNITY,
    "TRAIN_ONLY_MULTI_SHARD_COMMUNITIES": TRAIN_ONLY_MULTI_SHARD_COMMUNITIES,
    "USE_HARD_NEGATIVES": USE_HARD_NEGATIVES,
    "HARD_NEGATIVE_FRACTION": HARD_NEGATIVE_FRACTION,
    "HARD_NEGATIVE_TOPK": HARD_NEGATIVE_TOPK,
    "HARD_NEGATIVE_NN_POOL": HARD_NEGATIVE_NN_POOL,
    "STUDENT_ARCH": STUDENT_ARCH,
    "HIDDEN_DIM": HIDDEN_DIM,
    "NUM_MLP_HIDDEN_LAYERS": NUM_MLP_HIDDEN_LAYERS,
    "HIDDEN_DIM_2": HIDDEN_DIM_2,
    "HIDDEN_DIM_1": HIDDEN_DIM_1,
    "HIDDEN_DIM_2_MLP": HIDDEN_DIM_2_MLP,
    "OUTPUT_DIM": OUTPUT_DIM,
    "DROPOUT": DROPOUT,
    "RESIDUAL_ALPHA": RESIDUAL_ALPHA,
    "RESIDUAL_ALPHA_LEARNABLE": RESIDUAL_ALPHA_LEARNABLE,
    "LR": LR,
    "WEIGHT_DECAY": WEIGHT_DECAY,
    "TEMPERATURE": TEMPERATURE,
    "MAX_EPOCHS": MAX_EPOCHS,
    "SCHEDULER_PATIENCE": SCHEDULER_PATIENCE,
    "EARLY_STOP_PATIENCE": EARLY_STOP_PATIENCE,
    "best_epoch": best_epoch,
    "best_val_loss": best_val,
}
RUN_DIR.joinpath("config_train.json").write_text(json.dumps(cfg_out, indent=2), encoding="utf-8")
RUN_DIR.joinpath("training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
save_training_history_csv(RUN_DIR / "training_history.csv", history)
print("Saved history + checkpoints to", RUN_DIR)"""
            ),
        },
        {
            "cell_type": "code",
            "metadata": {},
            "id": "embed",
            "execution_count": None,
            "outputs": [],
            "source": _lines(
                r"""# Reload best weights for embedding export
def _load_ckpt(p):
    try:
        return torch.load(p, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(p, map_location=device)

ckpt_best = _load_ckpt(RUN_DIR / "checkpoint_best.pt")
model.load_state_dict(ckpt_best["model_state_dict"])

x_all = torch.cat([x_train, x_val], dim=0)
emb_best = export_embeddings(model, device, x_all, batch_size=4096)

eids_all = train_eids + val_eids
split_flag = np.array(["train"] * len(train_eids) + ["val"] * len(val_eids), dtype=object)
tc_all = np.concatenate([train_df["teacher_cluster_id"].astype(str).values, val_df["teacher_cluster_id"].astype(str).values])
sh_all = np.concatenate([shard_train_np, shard_val_np])
y_all = np.concatenate([y_train, y_val])

np.save(RUN_DIR / "embeddings_best_val.npy", emb_best)
meta_rows = [
    {
        "row": i,
        "external_id": eids_all[i],
        "split": str(split_flag[i]),
        "teacher_cluster_id": str(tc_all[i]),
        "teacher_label_int": int(y_all[i]),
        "shard_id": str(sh_all[i]),
    }
    for i in range(len(eids_all))
]
RUN_DIR.joinpath("embedding_meta.json").write_text(json.dumps(meta_rows, indent=2), encoding="utf-8")
print("emb shape", emb_best.shape, "wrote embeddings_best_val.npy + embedding_meta.json")

# Final weights embedding (optional duplicate if same as last epoch)
ckpt_final = _load_ckpt(RUN_DIR / "checkpoint_final.pt")
model.load_state_dict(ckpt_final["model_state_dict"])
emb_final = export_embeddings(model, device, x_all, batch_size=4096)
np.save(RUN_DIR / "embeddings_final.npy", emb_final)
print("wrote embeddings_final.npy")"""
            ),
        },
        {
            "cell_type": "code",
            "metadata": {},
            "id": "plots",
            "execution_count": None,
            "outputs": [],
            "source": _lines(
                r"""fig_dir = RUN_DIR / "figures"
fig_dir.mkdir(exist_ok=True)

epochs = [h["epoch"] for h in history]
plt.figure(figsize=(8, 5))
plt.plot(epochs, [h["train_loss"] for h in history], label="train")
plt.plot(epochs, [h["val_loss"] for h in history], label="val")
plt.xlabel("epoch")
plt.ylabel("SupCon loss")
plt.legend()
plt.tight_layout()
plt.savefig(fig_dir / "loss_curves.png", dpi=120)
plt.show()
plt.close()

plt.figure(figsize=(8, 4))
plt.plot(epochs, [h["lr"] for h in history], color="green")
plt.xlabel("epoch")
plt.ylabel("learning rate")
plt.tight_layout()
plt.savefig(fig_dir / "lr_curve.png", dpi=120)
plt.show()
plt.close()

plt.figure(figsize=(8, 4))
plt.plot(epochs, [h.get("train_batch_cross_shard_pos_frac") for h in history], label="train")
plt.plot(epochs, [h.get("val_batch_cross_shard_pos_frac") for h in history], label="val")
plt.xlabel("epoch")
plt.ylabel("frac anchors w/ cross-shard positive in batch")
plt.legend()
plt.tight_layout()
plt.savefig(fig_dir / "cross_shard_positive_frac.png", dpi=120)
plt.show()
plt.close()

plt.figure(figsize=(8, 4))
plt.plot(epochs, [h.get("train_batch_cross_shard_pos_mean") for h in history], label="train mean count")
plt.plot(epochs, [h.get("val_batch_cross_shard_pos_mean") for h in history], label="val mean count")
plt.xlabel("epoch")
plt.ylabel("avg cross-shard positives / anchor")
plt.legend()
plt.tight_layout()
plt.savefig(fig_dir / "cross_shard_positive_counts.png", dpi=120)
plt.show()
plt.close()

if USE_HARD_NEGATIVES:
    plt.figure(figsize=(8, 4))
    plt.plot(epochs, [h.get("train_mean_cos_hard_neg") for h in history], label="hard neg cos")
    plt.plot(epochs, [h.get("train_mean_cos_rand_neg") for h in history], label="rand neg cos")
    plt.xlabel("epoch")
    plt.ylabel("mean cosine (train batch)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "hard_vs_random_neg_cosine.png", dpi=120)
    plt.show()
    plt.close()

plt.figure(figsize=(8, 5))
plt.plot(epochs, [h.get("val_pos_cos_mean") for h in history], label="val same-cluster cos")
plt.plot(epochs, [h.get("val_neg_cos_mean") for h in history], label="val diff-cluster cos")
plt.xlabel("epoch")
plt.ylabel("mean cosine")
plt.legend()
plt.tight_layout()
plt.savefig(fig_dir / "val_cosine_separation.png", dpi=120)
plt.show()
plt.close()

print("saved plots to", fig_dir)"""
            ),
        },
    ]

    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "cells": cells,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        f.write("\n")
    print("Wrote", OUT)


if __name__ == "__main__":
    main()
