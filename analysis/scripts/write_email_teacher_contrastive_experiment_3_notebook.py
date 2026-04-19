"""Writes analysis/email_teacher_contrastive_experiment_3_eval.ipynb (stage 3). Run from repo root."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "analysis" / "email_teacher_contrastive_experiment_3_eval.ipynb"

MD = """# Email teacher–student (stage 3): cluster & evaluate embeddings

**Evaluation only:** HDBSCAN on fixed representations, external metrics vs **real GT** (GT not used to fit clustering or the student).

**Graph vs shard:** Stage-2 ``feature_load_info.json`` resolves the **incidents hetero** ``*.pt`` / ``*.meta.json`` (email node features in ``data[\"email\"].x``). That is **not** the semantic-shard Step 2 CSV graph.

**Raw baseline — full graph:** For GT metrics, the primary raw baseline loads **every** ``external_id`` listed in that hetero ``GRAPH_META`` (same order as graph rows, typically ~all email nodes). Training ``embedding_meta.json`` is often built from a **no–ground-truth** train/val split, so its IDs can be **disjoint** from ``ground_truth.json`` by design; zero GT overlap on the **train_val_subset** baseline usually means GT emails were held out of training, not that the graph “excludes” them.

**Student — full graph (GT eval):** Exported ``embeddings_*.npy`` usually cover only the no–GT train/val split. For **generalization to held-out GT**, this notebook **re-runs the saved checkpoint** (``checkpoint_best.pt`` / ``checkpoint_final.pt``) on **all** graph email features (same slice as stage 2). Metrics named ``*_full_graph`` use those inferred vectors; ``*_train_subset`` rows keep the frozen export for comparison (often **no GT overlap** by design).

**Outputs** (under ``OUTPUT_SUBDIR``): ``clustering_metrics.csv``, ``clustering_metrics.json``, figures for metric bars and 2D embedding views.

**HDBSCAN noise:** A large ``n_noise`` is often **not a bug**. Density-based methods label sparse / border points as ``-1``. Raw projected email features are frequently **less modality-separated** than trained student embeddings, so the baseline can look noisier even when IDs and metrics code are correct. Try ``HDBSCAN_CLUSTER_SELECTION_METHOD = \"leaf\"``, lower ``MIN_SAMPLES``, raise ``CLUSTER_SELECTION_EPSILON``, or set ``STANDARDIZE_COLUMNS_BEFORE_CLUSTER = True`` (same for all variants for a fair comparison). If **all** points are noise, lower ``MIN_CLUSTER_SIZE`` and check ``HDBSCAN_METRIC`` / L2 normalization."""


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

# Stage-2 run directory (must contain embedding_meta.json, feature_load_info.json, embeddings_*.npy)
RUN_ID = "student_train_001"
RUN_DIR = PROJECT_ROOT / "analysis" / "output" / "email_teacher_contrastive" / RUN_ID

GT_JSON = PROJECT_ROOT / "data" / "groundtruth" / "ground_truth.json"

# HDBSCAN (shared across all variants)
# Euclidean in ~128-d often collapses to all-noise; cosine matches contrastive / SBERT-style geometry.
MIN_CLUSTER_SIZE = 10
MIN_SAMPLES = 2
# "cosine" → L2-normalize + HDBSCAN euclidean (same geometry; avoids sklearn Unrecognized metric 'cosine')
HDBSCAN_METRIC = "cosine"
# Extra L2 step when using euclidean / other metrics (usually False if HDBSCAN_METRIC is cosine)
L2_NORMALIZE_BEFORE_CLUSTER = True
# If clusters are still too fragmented, try 0.05–0.25 (merges flat regions per hdbscan docs)
CLUSTER_SELECTION_EPSILON = 0.0
ALLOW_SINGLE_CLUSTER = False
# "eom" = HDBSCAN default (Excess of Mass). "leaf" often reduces noise (-1) at the cost of more/smaller clusters.
HDBSCAN_CLUSTER_SELECTION_METHOD = "leaf"
# Per-dimension z-score before clustering (applied to all variants if True). Can help raw features with uneven column scales.
STANDARDIZE_COLUMNS_BEFORE_CLUSTER = False

# 2D projection: try UMAP if installed, else PCA
UMAP_N_NEIGHBORS = 30
UMAP_MIN_DIST = 0.1
UMAP_RANDOM_STATE = 42

# Optional: flat JSON object with metric columns for an extra table row, e.g. shard-graph baseline
# Example: {"variant": "reference_shard_graph_baseline", "homogeneity": 0.4, "v_measure": 0.35, ...}
REFERENCE_METRICS_JSON = None

RNG_SEED = 42
COSINE_DIAG_MAX_POINTS = 2500

# Full-graph student inference (forward pass on all graph emails for held-out GT eval)
EVAL_INFER_DEVICE = None  # None → cuda if available else cpu; or e.g. "cuda", "cpu"
STUDENT_INFER_BATCH_SIZE = 4096

OUTPUT_SUBDIR = RUN_DIR  # e.g. RUN_DIR / "eval_stage3"
FIG_DIR = OUTPUT_SUBDIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_SUBDIR.mkdir(parents=True, exist_ok=True)

print("RUN_DIR:", RUN_DIR.resolve())
print("FIG_DIR:", FIG_DIR.resolve())"""
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

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.clustering.clusteringMetrics import _emb_matrix_from_id_to_embedding

from analysis.utils.email_teacher_contrastive_eval import (
    cluster_size_counts,
    cosine_same_vs_diff_campaign,
    gt_id_set_overlap,
    hdbscan_evaluate,
    infer_student_embeddings_full_graph,
    load_embedding_meta_and_array,
    load_graph_paths_from_feature_info,
    matrix_clustering_sanity,
    meta_matrix_to_id_to_emb,
)
from analysis.utils.email_teacher_contrastive_features import load_graph_email_features_for_external_ids
from analysis.utils.raw_gnn_notebook import load_email_external_ids, load_ground_truth_structures

rng = np.random.default_rng(RNG_SEED)


def try_umap_2d(X: np.ndarray) -> tuple[np.ndarray, str]:
    try:
        import umap  # type: ignore

        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=int(UMAP_N_NEIGHBORS),
            min_dist=float(UMAP_MIN_DIST),
            random_state=int(UMAP_RANDOM_STATE),
            metric=HDBSCAN_METRIC if HDBSCAN_METRIC in ("euclidean", "cosine") else "euclidean",
        )
        return np.asarray(reducer.fit_transform(X), dtype=np.float64), "UMAP"
    except ImportError:
        from sklearn.decomposition import PCA

        z = PCA(n_components=2, random_state=UMAP_RANDOM_STATE).fit_transform(X)
        return np.asarray(z, dtype=np.float64), "PCA" """
            ),
        },
        {
            "cell_type": "code",
            "metadata": {},
            "id": "load",
            "execution_count": None,
            "outputs": [],
            "source": _lines(
                r"""label_map, _eid_row, _c2m = load_ground_truth_structures(GT_JSON)

loaded_best = load_embedding_meta_and_array(RUN_DIR, "embeddings_best_val")
loaded_final = load_embedding_meta_and_array(RUN_DIR, "embeddings_final")
if loaded_best is None:
    raise FileNotFoundError(f"Missing embeddings_best_val under {RUN_DIR}")

meta, emb_best = loaded_best
emb_final = None
if loaded_final is not None:
    _, emb_final = loaded_final

GRAPH_PT, GRAPH_META, FEATURE_MODE_RESOLVED = load_graph_paths_from_feature_info(RUN_DIR)

# All email node IDs from the hetero graph meta (same checkpoint as stage 2 — not shard CSVs).
all_graph_eids = load_email_external_ids(GRAPH_META)
meta_full = [{"external_id": eid} for eid in all_graph_eids]
X_raw_full, mask_full, raw_info_full = load_graph_email_features_for_external_ids(
    GRAPH_PT,
    GRAPH_META,
    all_graph_eids,
    feature_mode=FEATURE_MODE_RESOLVED,
    to_undirected=True,
)
id_raw_full = meta_matrix_to_id_to_emb(meta_full, X_raw_full, mask_full)
print("raw features (full graph order):", raw_info_full)
print(
    "graph meta emails:",
    len(all_graph_eids),
    "present in tensor:",
    int(mask_full.sum()),
    "/",
    len(mask_full),
)

# Train/val export order (often no-GT subset) — for student + apples-to-apples raw slice.
eids_train_val = [str(r["external_id"]) for r in meta]
X_raw_train, mask_train, raw_info_train = load_graph_email_features_for_external_ids(
    GRAPH_PT,
    GRAPH_META,
    eids_train_val,
    feature_mode=FEATURE_MODE_RESOLVED,
    to_undirected=True,
)
id_raw_train = meta_matrix_to_id_to_emb(meta, X_raw_train, mask_train)
print("raw features (train_val list from embedding_meta):", raw_info_train)
print("n embedding_meta rows:", len(meta), "present in graph:", int(mask_train.sum()), "/", len(mask_train))

id_best = meta_matrix_to_id_to_emb(meta, emb_best, mask_train)
if emb_final is not None:
    id_final = meta_matrix_to_id_to_emb(meta, emb_final, mask_train)
else:
    id_final = None

final_is_dup = False
if id_final is not None and id_best.keys() == id_final.keys():
    a = np.stack([id_best[k] for k in sorted(id_best, key=str)])
    b = np.stack([id_final[k] for k in sorted(id_final, key=str)])
    final_is_dup = a.shape == b.shape and np.allclose(a, b, rtol=1e-5, atol=1e-8)
if final_is_dup:
    print("Note: exported embeddings_final matches embeddings_best_val — student_final_train_subset omitted.")

# --- Student on ALL graph emails (same features as stage 2) for held-out GT evaluation ---
id_student_best_full, infer_info_best = infer_student_embeddings_full_graph(
    RUN_DIR,
    graph_pt=GRAPH_PT,
    meta_json=GRAPH_META,
    external_ids=all_graph_eids,
    feature_mode=FEATURE_MODE_RESOLVED,
    checkpoint="best",
    batch_size=STUDENT_INFER_BATCH_SIZE,
    device=EVAL_INFER_DEVICE,
    to_undirected=True,
)
print("student infer (best checkpoint, full graph):", infer_info_best)

id_student_final_full = None
infer_info_final = None
final_full_is_dup = False
if (RUN_DIR / "checkpoint_final.pt").is_file():
    id_student_final_full, infer_info_final = infer_student_embeddings_full_graph(
        RUN_DIR,
        graph_pt=GRAPH_PT,
        meta_json=GRAPH_META,
        external_ids=all_graph_eids,
        feature_mode=FEATURE_MODE_RESOLVED,
        checkpoint="final",
        batch_size=STUDENT_INFER_BATCH_SIZE,
        device=EVAL_INFER_DEVICE,
        to_undirected=True,
    )
    print("student infer (final checkpoint, full graph):", infer_info_final)
    if id_student_best_full.keys() == id_student_final_full.keys():
        ab = np.stack([id_student_best_full[k] for k in sorted(id_student_best_full, key=str)])
        af = np.stack([id_student_final_full[k] for k in sorted(id_student_final_full, key=str)])
        final_full_is_dup = ab.shape == af.shape and np.allclose(ab, af, rtol=1e-5, atol=1e-8)
    if final_full_is_dup:
        print("Note: full-graph final embeddings match best — student_final_full_graph omitted from metrics.")
else:
    print("No checkpoint_final.pt — skipping full-graph final student inference.")

variants = [
    ("baseline_raw_full_graph", id_raw_full),
    ("baseline_raw_train_val_subset", id_raw_train),
    ("student_best_val_full_graph", id_student_best_full),
]
if id_student_final_full is not None and not final_full_is_dup:
    variants.append(("student_final_full_graph", id_student_final_full))
variants.append(("student_best_val_train_subset", id_best))
if id_final is not None and not final_is_dup:
    variants.append(("student_final_train_subset", id_final))

_ov_full = gt_id_set_overlap(id_raw_full, label_map)
_ov_train = gt_id_set_overlap(id_raw_train, label_map)
_ov_st = gt_id_set_overlap(id_student_best_full, label_map)
print("GT ↔ IDs (full hetero graph baseline):", _ov_full)
print("GT ↔ IDs (train_val subset baseline / exported student keys):", _ov_train)
print("GT ↔ IDs (student best, full-graph inference):", _ov_st)
if _ov_full["n_gt_labeled_emails"] == 0:
    print("WARNING: ground truth has zero labeled emails — check GT_JSON / UTF-8 BOM / top-level 'clusters' key.")
elif _ov_full["n_intersection"] == 0:
    print(
        "WARNING: zero GT overlap even on full graph — GT external_id strings do not match this graph's meta. "
        "Confirm GT_JSON labels the same incident universe as GRAPH_PT / GRAPH_META."
    )
elif _ov_train["n_intersection"] == 0 and _ov_full["n_intersection"] > 0:
    print(
        "Note: train_val subset has zero GT overlap but full graph does — expected if stage-2 training used "
        "emails with ground truth removed (held-out GT set)."
    )

_, _Xraw_f = _emb_matrix_from_id_to_embedding(id_raw_full)
_, _Xraw_t = _emb_matrix_from_id_to_embedding(id_raw_train)
_, _Xsbf = _emb_matrix_from_id_to_embedding(id_student_best_full)
_, _Xsb0 = _emb_matrix_from_id_to_embedding(id_best)
print("Pre-HDBSCAN matrix sanity (raw full graph):", matrix_clustering_sanity(_Xraw_f, tag="raw_full_graph"))
print("Pre-HDBSCAN matrix sanity (raw train_val):", matrix_clustering_sanity(_Xraw_t, tag="raw_train_val_subset"))
print(
    "Pre-HDBSCAN matrix sanity (student best full graph):",
    matrix_clustering_sanity(_Xsbf, tag="student_best_full_graph"),
)
print(
    "Pre-HDBSCAN matrix sanity (student best train_subset export):",
    matrix_clustering_sanity(_Xsb0, tag="student_best_train_subset"),
)"""
            ),
        },
        {
            "cell_type": "code",
            "metadata": {},
            "id": "cluster",
            "execution_count": None,
            "outputs": [],
            "source": _lines(
                r"""rows = []
labels_by_variant: dict[str, tuple[np.ndarray, list[str]]] = {}

for name, id_map in variants:
    m, labels, sorted_ids = hdbscan_evaluate(
        id_map,
        label_map,
        min_cluster_size=MIN_CLUSTER_SIZE,
        min_samples=MIN_SAMPLES,
        variant=name,
        metric=HDBSCAN_METRIC,
        l2_normalize_rows=L2_NORMALIZE_BEFORE_CLUSTER,
        standardize_columns=STANDARDIZE_COLUMNS_BEFORE_CLUSTER,
        cluster_selection_epsilon=CLUSTER_SELECTION_EPSILON,
        allow_single_cluster=ALLOW_SINGLE_CLUSTER,
        cluster_selection_method=HDBSCAN_CLUSTER_SELECTION_METHOD,
    )
    labels_by_variant[name] = (labels, sorted_ids)
    row = {
        k: m[k]
        for k in m
        if k
        not in (
            "silhouette",
            "db_index",
            "ch_index",
            "coverage_ground_truth",
            "coverage_all",
            "n_samples",
        )
    }
    rows.append(row)

if REFERENCE_METRICS_JSON is not None:
    rp = Path(REFERENCE_METRICS_JSON)
    if rp.is_file():
        ref = json.loads(rp.read_text(encoding="utf-8"))
        if isinstance(ref, dict):
            if ref.get("variant") is None:
                ref = {"variant": "reference_shard_graph_baseline", **ref}
            rows.append(ref)
        print("Appended reference row from", rp)
    else:
        print("REFERENCE_METRICS_JSON not found:", rp)

metrics_df = pd.json_normalize(rows)
cols = [
    "variant",
    "homogeneity",
    "completeness",
    "v_measure",
    "n_eval",
    "coverage_gt",
    "coverage_assignments",
    "n_clusters",
    "n_noise",
    "n_embeddings",
    "min_cluster_size",
    "min_samples",
    "hdbscan_metric",
    "hdbscan_metric_used",
    "l2_normalize_rows",
    "standardize_columns",
    "cluster_selection_epsilon",
    "allow_single_cluster",
    "cluster_selection_method",
]
cols = [c for c in cols if c in metrics_df.columns] + [c for c in metrics_df.columns if c not in cols]
metrics_df = metrics_df[cols]
print(metrics_df)

csv_path = OUTPUT_SUBDIR / "clustering_metrics.csv"
json_path = OUTPUT_SUBDIR / "clustering_metrics.json"
metrics_df.to_csv(csv_path, index=False)
json_path.write_text(metrics_df.to_json(orient="records", indent=2), encoding="utf-8")
print("Wrote", csv_path)
print("Wrote", json_path)"""
            ),
        },
        {
            "cell_type": "code",
            "metadata": {},
            "id": "bars",
            "execution_count": None,
            "outputs": [],
            "source": _lines(
                r"""plot_df = metrics_df[metrics_df["variant"].str.startswith(("baseline", "student"))].copy()
if plot_df.empty:
    plot_df = metrics_df.copy()

x = np.arange(len(plot_df))
w = 0.25
plt.figure(figsize=(max(6, 1.2 * len(plot_df)), 5))
plt.bar(x - w, plot_df["homogeneity"], width=w, label="homogeneity")
plt.bar(x, plot_df["completeness"], width=w, label="completeness")
plt.bar(x + w, plot_df["v_measure"], width=w, label="v_measure")
plt.xticks(x, plot_df["variant"], rotation=25, ha="right")
plt.ylim(0, 1.05)
plt.legend()
plt.ylabel("score")
plt.title("External clustering metrics (GT eval)")
plt.tight_layout()
p = FIG_DIR / "cluster_metric_comparison.png"
plt.savefig(p, dpi=120, bbox_inches="tight")
plt.show()
plt.close()
print("Saved", p)

plt.figure(figsize=(max(6, 1.2 * len(plot_df)), 4))
plt.bar(x - 0.2, plot_df["n_clusters"], width=0.4, label="n_clusters")
plt.bar(x + 0.2, plot_df["n_noise"], width=0.4, label="n_noise")
plt.xticks(x, plot_df["variant"], rotation=25, ha="right")
plt.legend()
plt.title("Predicted clusters vs HDBSCAN noise")
plt.tight_layout()
p2 = FIG_DIR / "cluster_count_comparison.png"
plt.savefig(p2, dpi=120, bbox_inches="tight")
plt.show()
plt.close()
print("Saved", p2)"""
            ),
        },
        {
            "cell_type": "code",
            "metadata": {},
            "id": "2dfns",
            "execution_count": None,
            "outputs": [],
            "source": _lines(
                r"""def _tab20_colors(n: int):
    try:
        cmap = plt.colormaps["tab20"]
    except AttributeError:
        cmap = plt.cm.get_cmap("tab20")
    return [cmap(i % 20) for i in range(max(n, 1))]


def scatter_2d_cluster(Z, sorted_ids, labels, title, out_png, alpha=0.6):
    lab_map = {eid: int(la) for eid, la in zip(sorted_ids, labels)}
    y = np.array([lab_map[eid] for eid in sorted_ids], dtype=np.int64)
    noise = y == -1
    core = ~noise
    fig, ax = plt.subplots(figsize=(8, 6))
    if core.any():
        yu = np.unique(y[core])
        colors = _tab20_colors(len(yu))
        yu_list = list(yu)
        for j, c in enumerate(yu_list):
            m = y == c
            ax.scatter(Z[m, 0], Z[m, 1], s=6, alpha=alpha, color=colors[j], label=None)
    if noise.any():
        ax.scatter(Z[noise, 0], Z[noise, 1], s=8, alpha=0.35, c="0.45", marker="x", label="noise (-1)")
    ax.set_title(title)
    ax.legend(markerscale=1.5, loc="best")
    fig.tight_layout()
    fig.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    print("Saved", out_png)


def scatter_2d_gt(Z, sorted_ids, title, out_png, alpha=0.45):
    idx_keep = []
    gts = []
    for j, eid in enumerate(sorted_ids):
        g = label_map.get(eid)
        if g is not None:
            gts.append(g)
            idx_keep.append(j)
    if not idx_keep:
        print("No labeled points for GT plot:", title)
        return
    idx_keep = np.asarray(idx_keep, dtype=int)
    gts = np.asarray(gts)
    uniq = np.unique(gts)
    code = {u: i for i, u in enumerate(uniq)}
    cidx = np.array([code[g] for g in gts])
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(Z[idx_keep, 0], Z[idx_keep, 1], c=cidx, cmap="gist_ncar", s=7, alpha=alpha)
    ax.set_title(title + f" (n_labeled={len(idx_keep)})")
    fig.tight_layout()
    fig.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    print("Saved", out_png)


def plot_variant_2d(name: str, id_map: dict, labels: np.ndarray, sorted_ids: list[str], tag: str):
    _, Xs = _emb_matrix_from_id_to_embedding(id_map)
    Z, kind = try_umap_2d(Xs)
    scatter_2d_cluster(
        Z,
        sorted_ids,
        labels,
        f"{name} — by predicted cluster ({kind})",
        FIG_DIR / f"umap_{tag}_by_cluster.png",
    )
    scatter_2d_gt(
        Z,
        sorted_ids,
        f"{name} — by GT campaign ({kind})",
        FIG_DIR / f"umap_{tag}_by_gt.png",
    ) """
            ),
        },
        {
            "cell_type": "code",
            "metadata": {},
            "id": "run2d",
            "execution_count": None,
            "outputs": [],
            "source": _lines(
                r"""lb, sid_b = labels_by_variant["baseline_raw_full_graph"]
ls, sid_s = labels_by_variant["student_best_val_full_graph"]

plot_variant_2d("Raw baseline (full hetero graph)", id_raw_full, lb, sid_b, "baseline_full_graph")
plot_variant_2d(
    "Student best val (full graph inferred)",
    id_student_best_full,
    ls,
    sid_s,
    "student_best_full_graph",
)

if "student_final_full_graph" in labels_by_variant:
    lf, sid_f = labels_by_variant["student_final_full_graph"]
    plot_variant_2d(
        "Student final (full graph inferred)",
        id_student_final_full,
        lf,
        sid_f,
        "student_final_full_graph",
    ) """
            ),
        },
        {
            "cell_type": "code",
            "metadata": {},
            "id": "sizes",
            "execution_count": None,
            "outputs": [],
            "source": _lines(
                r"""fig, axes = plt.subplots(1, len(variants), figsize=(5 * len(variants), 4), sharey=False)
if len(variants) == 1:
    axes = [axes]
for ax, (name, _idm) in zip(axes, variants):
    labels, _sids = labels_by_variant[name]
    _cid, cnt = cluster_size_counts(labels)
    ax.bar(range(len(cnt)), cnt, color="steelblue")
    n_noise = int((labels == -1).sum())
    ax.set_title(f"{name}\nclusters={len(cnt)} noise={n_noise}")
    ax.set_xlabel("cluster rank (by size)")
    ax.set_ylabel("size")
fig.tight_layout()
p3 = FIG_DIR / "cluster_size_distribution.png"
fig.savefig(p3, dpi=120, bbox_inches="tight")
plt.show()
plt.close(fig)
print("Saved", p3)"""
            ),
        },
        {
            "cell_type": "code",
            "metadata": {},
            "id": "cosine",
            "execution_count": None,
            "outputs": [],
            "source": _lines(
                r"""try:
    from IPython.display import display
except ImportError:
    display = print

cos_rows = []
cos_rows.append(
    {"space": "baseline_raw_full_graph", **cosine_same_vs_diff_campaign(id_raw_full, label_map, rng=rng, max_points=COSINE_DIAG_MAX_POINTS)}
)
cos_rows.append(
    {
        "space": "baseline_raw_train_val_subset",
        **cosine_same_vs_diff_campaign(id_raw_train, label_map, rng=rng, max_points=COSINE_DIAG_MAX_POINTS),
    }
)
cos_rows.append(
    {
        "space": "student_best_full_graph",
        **cosine_same_vs_diff_campaign(
            id_student_best_full, label_map, rng=rng, max_points=COSINE_DIAG_MAX_POINTS
        ),
    }
)
if id_student_final_full is not None and not final_full_is_dup:
    cos_rows.append(
        {
            "space": "student_final_full_graph",
            **cosine_same_vs_diff_campaign(
                id_student_final_full, label_map, rng=rng, max_points=COSINE_DIAG_MAX_POINTS
            ),
        }
    )
cos_rows.append(
    {
        "space": "student_best_train_subset",
        **cosine_same_vs_diff_campaign(id_best, label_map, rng=rng, max_points=COSINE_DIAG_MAX_POINTS),
    }
)
if id_final is not None and not final_is_dup:
    cos_rows.append(
        {
            "space": "student_final_train_subset",
            **cosine_same_vs_diff_campaign(id_final, label_map, rng=rng, max_points=COSINE_DIAG_MAX_POINTS),
        }
    )
cos_df = pd.DataFrame(cos_rows)
display(cos_df)

if len(cos_df) >= 2:
    plt.figure(figsize=(6, 4))
    x = np.arange(len(cos_df))
    plt.bar(x - 0.2, cos_df["mean_cos_same"], 0.4, label="same GT campaign")
    plt.bar(x + 0.2, cos_df["mean_cos_diff"], 0.4, label="different GT")
    plt.xticks(x, cos_df["space"], rotation=20, ha="right")
    plt.ylabel("mean cosine")
    plt.legend()
    plt.title("Pairwise cosine (subsampled)")
    plt.tight_layout()
    pc = FIG_DIR / "cosine_same_vs_diff.png"
    plt.savefig(pc, dpi=120, bbox_inches="tight")
    plt.show()
    plt.close()
    print("Saved", pc)"""
            ),
        },
        {
            "cell_type": "code",
            "metadata": {},
            "id": "bestfinal",
            "execution_count": None,
            "outputs": [],
            "source": _lines(
                r"""try:
    from IPython.display import display
except ImportError:
    display = print

sub = metrics_df[
    metrics_df["variant"].isin(["student_best_val_full_graph", "student_final_full_graph"])
].copy()
if len(sub) == 2:
    display(sub.set_index("variant").T)
    diff = sub.set_index("variant").diff().iloc[-1]
    plt.figure(figsize=(6, 3))
    diff[["homogeneity", "completeness", "v_measure"]].plot(kind="bar", ax=plt.gca())
    plt.axhline(0, color="k", lw=0.5)
    plt.title("student_final_full_graph minus student_best_val_full_graph")
    plt.tight_layout()
    bf = FIG_DIR / "best_vs_final_metric_delta.png"
    plt.savefig(bf, dpi=120, bbox_inches="tight")
    plt.show()
    plt.close()
    print("Saved", bf)
else:
    print("Skipping best-vs-final (need both non-duplicate full-graph student rows)")"""
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
