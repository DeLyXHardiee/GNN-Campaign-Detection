"""Writes analysis/email_teacher_contrastive_experiment.ipynb. Run from repo root."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "analysis" / "email_teacher_contrastive_experiment.ipynb"

CELL_MD = """# Email-level supervised contrastive (stage 1): teacher pseudo-labels

Loads **teacher communities** from the shard-graph JSON (top-level `clusters` = Louvain community id). Builds **train/val splits over communities** (not over individual emails), positive/negative sampling utilities, and **community-aware batches** for a later student encoder.

**Ground truth** is not used to construct the training set here."""


def _lines(s: str) -> list[str]:
    if not s.endswith("\n"):
        s += "\n"
    return [line + "\n" if not line.endswith("\n") else line for line in s.splitlines()]


def main() -> None:
    cells: list[dict] = [
        {"cell_type": "markdown", "id": "md0", "metadata": {}, "source": _lines(CELL_MD)},
        {
            "cell_type": "code",
            "id": "cfg",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": _lines(
                r"""# --- config ---
from pathlib import Path

_here = Path.cwd().resolve()
PROJECT_ROOT = _here if (_here / "pipeline_config.json").is_file() else _here.parent

RUN_ID = "dev_run_001"

TEACHER_COMMUNITY_JSON = PROJECT_ROOT / "data" / "pseudo_ground_truth_no_gt_shard_graph.json"

OUT_ROOT = PROJECT_ROOT / "analysis" / "output" / "email_teacher_contrastive"
RUN_DIR = OUT_ROOT / RUN_ID

RNG_SEED = 42

# Train/val split over teacher communities: set exactly one of these to None
VAL_FRACTION = 0.15
VAL_N_COMMUNITIES = None

# For the upcoming loss: prefer positives from another shard within the same community
PREFERRED_POSITIVE_TYPE_A_WEIGHT = 0.7  # doc only for the training stage

# Negative smoke test
N_EASY_NEGATIVES_SMOKE_TEST = 1000

# Optional: sklearn cosine NN on graph embeddings for hard negatives
USE_GRAPH_FOR_HARD_NEGATIVES = True
GRAPH_PT = PROJECT_ROOT / "core" / "graph" / "output" / "incidents-lake-misp-large_hetero.pt"
GRAPH_META_JSON = GRAPH_PT.with_suffix(".meta.json")
HARD_NEG_EMB_DIM = 128
HARD_NEG_MAX_NN_QUERY = 96
HARD_NEG_SAMPLE_K = 4

# Community-aware batches (later DataLoader / training loop)
N_COMMUNITIES_PER_BATCH = 8
N_EMAILS_PER_COMMUNITY = 4
N_BATCH_EXAMPLES = 3

RUN_DIR.mkdir(parents=True, exist_ok=True)
print("RUN_DIR:", RUN_DIR.resolve())"""
            ),
        },
        {
            "cell_type": "code",
            "id": "imports",
            "metadata": {},
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

assert (PROJECT_ROOT / "pipeline_config.json").is_file(), PROJECT_ROOT
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.utils.email_teacher_contrastive_data import (
    CommunityAwareBatchIterator,
    EasyNegativeSampler,
    TrainValSplit,
    build_lookup_tables,
    compute_easy_negative_diagnostics,
    compute_positive_sampling_diagnostics,
    parse_teacher_community_json,
    positive_counts_per_email,
    split_communities_train_val,
    split_summary_dict,
    try_build_hard_negative_index_from_graph,
)

pd.set_option("display.max_columns", 50)
rng = np.random.default_rng(RNG_SEED)"""
            ),
        },
        {
            "cell_type": "code",
            "id": "load",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": _lines(
                r"""df, parse_stats = parse_teacher_community_json(TEACHER_COMMUNITY_JSON)
print("parse_stats:", parse_stats)
print(df.head())
print("shape:", df.shape)

look = build_lookup_tables(df)
with open(RUN_DIR / "lookup_tables.json", "w", encoding="utf-8") as f:
    json.dump(look.to_json_dict(), f, indent=2)

df.to_csv(RUN_DIR / "teacher_pseudo_labels.csv", index=False)
try:
    df.to_parquet(RUN_DIR / "teacher_pseudo_labels.parquet", index=False)
except Exception as e:
    print("(skip parquet)", e)"""
            ),
        },
        {
            "cell_type": "code",
            "id": "split",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": _lines(
                r"""if VAL_FRACTION is not None and VAL_N_COMMUNITIES is not None:
    raise ValueError("Set only one of VAL_FRACTION or VAL_N_COMMUNITIES (the other None)")
if VAL_FRACTION is None and VAL_N_COMMUNITIES is None:
    raise ValueError("Set VAL_FRACTION or VAL_N_COMMUNITIES")

split: TrainValSplit = split_communities_train_val(
    df,
    rng,
    val_fraction=VAL_FRACTION,
    val_n_communities=VAL_N_COMMUNITIES,
)

split_summary = split_summary_dict(split)
split_summary["parse_stats"] = parse_stats
split_summary["val_fraction"] = VAL_FRACTION
split_summary["val_n_communities"] = VAL_N_COMMUNITIES

with open(RUN_DIR / "train_val_split_summary.json", "w", encoding="utf-8") as f:
    json.dump(split_summary, f, indent=2)

print(
    "train communities:", split_summary["n_train_teacher_communities"],
    "val communities:", split_summary["n_val_teacher_communities"],
)
print("train emails:", split_summary["n_train_emails"], "val emails:", split_summary["n_val_emails"])

train_df = split.train_df
val_df = split.val_df"""
            ),
        },
        {
            "cell_type": "code",
            "id": "posneg",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": _lines(
                r"""pos_train = compute_positive_sampling_diagnostics(train_df)
pos_full = compute_positive_sampling_diagnostics(df)
pos_val = compute_positive_sampling_diagnostics(val_df)

easy_train = compute_easy_negative_diagnostics(train_df)
easy_full = compute_easy_negative_diagnostics(df)

diag = {
    "positive_train": pos_train,
    "positive_val": pos_val,
    "positive_full": pos_full,
    "easy_negative_train": easy_train,
    "easy_negative_full": easy_full,
    "preferred_positive_type_a_weight_note": PREFERRED_POSITIVE_TYPE_A_WEIGHT,
}

hard_idx = None
if USE_GRAPH_FOR_HARD_NEGATIVES:
    hard_idx, hard_meta = try_build_hard_negative_index_from_graph(
        train_df,
        GRAPH_PT,
        GRAPH_META_JSON,
        rng,
        embedding_dim=HARD_NEG_EMB_DIM,
        max_neighbors_query=HARD_NEG_MAX_NN_QUERY,
        sample_k_test=HARD_NEG_SAMPLE_K,
    )
    diag["hard_negative_index"] = hard_meta
else:
    diag["hard_negative_index"] = {"built": False, "reason": "disabled"}

with open(RUN_DIR / "positive_negative_diagnostics.json", "w", encoding="utf-8") as f:
    json.dump(diag, f, indent=2)

print("positive (train):")
for k, v in pos_train.items():
    print(f"  {k}: {v}")"""
            ),
        },
    ]

    cells.append(
        {
            "cell_type": "code",
            "id": "sampler",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": _lines(
                r"""# Easy negatives are *within the frame you pass* (here: train only).
# If train holds a single teacher community, every anchor has an empty pool — see easy_train.
easy_sampler = EasyNegativeSampler(train_df, rng)
_train_ids = train_df["external_id"].astype(str).to_numpy()
_sz = train_df.groupby("teacher_cluster_id").size().to_dict()
_tot = len(train_df)
miss = 0
att = 0
for _ in range(N_EASY_NEGATIVES_SMOKE_TEST):
    a = str(_train_ids[int(rng.integers(0, len(_train_ids)))])
    tc = str(train_df.loc[train_df["external_id"].astype(str) == a, "teacher_cluster_id"].iloc[0])
    if _tot - _sz[tc] <= 0:
        continue
    att += 1
    try:
        easy_sampler.sample(a)
    except Exception:
        miss += 1
print("easy neg smoke: attempts", att, "failures", miss)

if hard_idx is not None:
    a0 = str(train_df["external_id"].iloc[0])
    got = hard_idx.sample_hard(a0, HARD_NEG_SAMPLE_K, rng)
    print("hard negative example for", a0, "->", got)

batch_iter = CommunityAwareBatchIterator(
    train_df,
    rng,
    n_communities_per_batch=N_COMMUNITIES_PER_BATCH,
    n_emails_per_community=N_EMAILS_PER_COMMUNITY,
)
for i, batch in zip(range(N_BATCH_EXAMPLES), batch_iter):
    print(f"example batch {i} size {len(batch)}:", batch[:8])"""
            ),
        }
    )

    cells.append(
        {
            "cell_type": "code",
            "id": "plots",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": _lines(
                r"""fig_dir = RUN_DIR / "figures"
fig_dir.mkdir(exist_ok=True)


def _hist(arr, title, path, bins=40):
    plt.figure(figsize=(7, 4))
    plt.hist(arr, bins=bins, color="steelblue", edgecolor="white")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


sizes_all = df.groupby("teacher_cluster_id").size().astype(int).values
_hist(sizes_all, "Teacher community sizes (emails)", fig_dir / "hist_community_sizes.png")

shards_per_comm = df.groupby("teacher_cluster_id")["shard_id"].nunique().astype(int).values
_hist(shards_per_comm, "Shards per teacher community", fig_dir / "hist_shards_per_community.png")

by_comm = df.groupby("teacher_cluster_id")["shard_id"].nunique()
print("fraction single-shard communities:", float((by_comm <= 1).mean()))
print("fraction multi-shard communities:", float((by_comm > 1).mean()))

s_diff, s_same = positive_counts_per_email(df)
idx = df["external_id"].astype(str)
_hist(
    idx.map(s_diff).fillna(0).astype(int).values,
    "Cross-shard positives per anchor (same community)",
    fig_dir / "hist_pos_diff_shard_counts.png",
)
_hist(
    idx.map(s_same).fillna(0).astype(int).values,
    "Same-shard positives per anchor (same community)",
    fig_dir / "hist_pos_same_shard_counts.png",
)

_hist(
    np.array(split_summary["train_community_sizes"]),
    "Train community sizes",
    fig_dir / "hist_train_community_sizes.png",
)
if split_summary["val_community_sizes"]:
    _hist(
        np.array(split_summary["val_community_sizes"]),
        "Val community sizes",
        fig_dir / "hist_val_community_sizes.png",
    )

print("saved figures under", fig_dir)"""
            ),
        }
    )

    cells.append(
        {
            "cell_type": "code",
            "id": "config_save",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": _lines(
                r"""cfg_doc = {
    "RUN_ID": RUN_ID,
    "TEACHER_COMMUNITY_JSON": str(TEACHER_COMMUNITY_JSON.resolve()),
    "OUT_DIR": str(RUN_DIR.resolve()),
    "RNG_SEED": RNG_SEED,
    "VAL_FRACTION": VAL_FRACTION,
    "VAL_N_COMMUNITIES": VAL_N_COMMUNITIES,
    "N_COMMUNITIES_PER_BATCH": N_COMMUNITIES_PER_BATCH,
    "N_EMAILS_PER_COMMUNITY": N_EMAILS_PER_COMMUNITY,
    "USE_GRAPH_FOR_HARD_NEGATIVES": USE_GRAPH_FOR_HARD_NEGATIVES,
    "HARD_NEG_EMB_DIM": HARD_NEG_EMB_DIM,
}
with open(RUN_DIR / "config.json", "w", encoding="utf-8") as f:
    json.dump(cfg_doc, f, indent=2)

print("Artifacts:", list(RUN_DIR.iterdir()))"""
            ),
        }
    )

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
