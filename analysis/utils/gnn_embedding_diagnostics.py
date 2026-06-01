"""
GNN / encoder email embedding diagnostic analysis (analysis-only).

Measures whether learned email embeddings encode campaign-relevant structure vs generic similarity.
"""

from __future__ import annotations

import html
import json
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None  # type: ignore[assignment,misc]


def _tqdm(it: Iterable[Any], *, desc: str, total: int | None = None) -> Any:
    if tqdm is None:
        return it
    return tqdm(it, desc=desc, total=total)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _ensure_sys_path() -> None:
    root = _repo_root()
    import sys

    for p in (root, root / "core", root / "core" / "GNN"):
        s = str(p)
        if s not in sys.path:
            sys.path.insert(0, s)


@dataclass
class GnnEmbeddingDiagConfig:
    run_dir: Path
    graph_pt: Path
    gt_paths: list[Path] = field(default_factory=list)
    output_subdir: str = "gnn_embedding_diagnostics"
    pair_csv: Path | None = None
    candidate_union_csv: Path | None = None
    bridge_scores_csv: Path | None = None
    checkpoint_name: str = "best_model.pt"
    device: str = "cpu"
    to_undirected: bool = True
    embeddings_json: Path | None = None
    max_pairs_per_relation: int = 80_000
    max_emails_for_retrieval: int = 0
    retrieval_k_values: tuple[int, ...] = (5, 10, 25, 50)
    high_cosine_threshold: float = 0.85
    max_high_cosine_nonedges: int = 5000
    max_review_pairs: int = 400
    random_state: int = 42
    skip_plots: bool = False
    skip_html: bool = False
    skip_probe: bool = False


COSINE_CALIBRATION_BUCKETS: tuple[tuple[str, float | None, float | None], ...] = (
    ("lt_0.5", None, 0.5),
    ("0.5_0.7", 0.5, 0.7),
    ("0.7_0.85", 0.7, 0.85),
    ("0.85_0.95", 0.85, 0.95),
    ("gt_0.95", 0.95, None),
)

FRONTIER_BANDS: tuple[tuple[str, float, float | None], ...] = (
    ("low_same", 0.0, 0.15),
    ("mid_same", 0.15, 0.50),
    ("high_same", 0.80, 1.01),
    ("mid_cross", 0.15, 0.50),
    ("low_cross", 0.0, 0.15),
)


def _cosine_l2(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na <= 0 or nb <= 0:
        return float("nan"), float("nan")
    return float(np.dot(a, b) / (na * nb)), float(np.linalg.norm(a - b))


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return x / n


def _auroc_auprc(y_true: np.ndarray, scores: np.ndarray) -> dict[str, float | None]:
    try:
        from sklearn.metrics import average_precision_score, roc_auc_score
    except ImportError:
        return {"auroc": None, "auprc": None}
    y = np.asarray(y_true, dtype=np.int8)
    s = np.asarray(scores, dtype=np.float64)
    m = np.isfinite(s)
    if m.sum() < 10 or len(np.unique(y[m])) < 2:
        return {"auroc": None, "auprc": None}
    return {
        "auroc": float(roc_auc_score(y[m], s[m])),
        "auprc": float(average_precision_score(y[m], s[m])),
    }


def _relation_from_gt(
    email_i: str,
    email_j: str,
    label_map: dict[str, Any],
) -> str | None:
    ci = label_map.get(str(email_i))
    cj = label_map.get(str(email_j))
    if ci is None or cj is None:
        return None
    return "same_campaign" if ci == cj else "cross_campaign"


def _load_label_map(gt_paths: list[Path]) -> dict[str, Any]:
    from seed_candidate_workflow.utils.raw_gnn_notebook import load_ground_truth_structures

    label_map: dict[str, Any] = {}
    for p in gt_paths:
        if not Path(p).is_file():
            continue
        lm, _, _ = load_ground_truth_structures(Path(p))
        for k, v in lm.items():
            if k not in label_map:
                label_map[str(k)] = v
    return label_map


def load_embedding_sources(
    *,
    run_dir: Path,
    graph_pt: Path,
    checkpoint_name: str,
    device: str,
    to_undirected: bool,
    embeddings_json: Path | None,
    project_root: Path,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Any], list[str]]:
    """Return id->vector per source key and metadata."""
    _ensure_sys_path()
    from seed_candidate_workflow.utils.pair_model_inference import load_pair_supervision_for_inference
    from seed_candidate_workflow.utils.raw_gnn_notebook import load_email_external_ids

    meta_path = Path(graph_pt).with_suffix(".meta.json")
    external_ids = [str(x) for x in load_email_external_ids(meta_path)]
    bundle = load_pair_supervision_for_inference(
        run_dir=Path(run_dir),
        graph_pt=Path(graph_pt),
        checkpoint_name=checkpoint_name,
        device=device,
        to_undirected=to_undirected,
    )
    meta: dict[str, Any] = {
        "pair_encoder_backend": bundle.get("pair_encoder_backend"),
        "checkpoint_path": bundle.get("checkpoint_path"),
        "n_emails": len(external_ids),
    }
    sources: dict[str, dict[str, np.ndarray]] = {}

    from src.clustering.clustering_helpers import (
        extract_email_embeddings,
        extract_raw_email_embeddings,
        load_transformer_subject_body_embeddings_from_cache,
    )

    raw_map = extract_raw_email_embeddings(bundle["data_cpu"], external_ids)
    sources["raw_email_x"] = raw_map
    meta["raw_email_x_dim"] = int(next(iter(raw_map.values())).shape[0]) if raw_map else 0

    model = bundle.get("model")
    if model is not None:
        sources["gnn_encoder"] = extract_email_embeddings(
            model, bundle["data_cpu"], bundle["device"], external_ids
        )
        meta["gnn_encoder_dim"] = int(next(iter(sources["gnn_encoder"].values())).shape[0])
        meta["primary_scorer_embedding_source"] = "gnn_encoder"
    else:
        sources["scorer_input_email_x"] = raw_map
        meta["primary_scorer_embedding_source"] = "scorer_input_email_x"
        meta["note_mlp_raw_backend"] = (
            "Training used mlp_raw_email_x: no message-passing encoder. "
            "scorer_input_email_x equals raw email.x passed to the pair MLP."
        )

    emb_path = embeddings_json
    if emb_path is None:
        cand = project_root / "core" / "utils" / "embeddings" / "output" / "embeddings.json"
        emb_path = cand if cand.is_file() else None
    if emb_path and Path(emb_path).is_file():
        try:
            sources["static_subj_body"] = load_transformer_subject_body_embeddings_from_cache(
                embeddings_json_path=emb_path
            )
            meta["static_subj_body_path"] = str(Path(emb_path).resolve())
            meta["static_subj_body_dim"] = int(
                next(iter(sources["static_subj_body"].values())).shape[0]
            )
        except Exception as exc:
            meta["static_subj_body_error"] = str(exc)

    return sources, meta, external_ids


def _build_email_matrix(
    id_to_vec: dict[str, np.ndarray],
    external_ids: list[str],
) -> tuple[np.ndarray, list[str], dict[str, int]]:
    rows: list[np.ndarray] = []
    ids: list[str] = []
    for eid in external_ids:
        v = id_to_vec.get(eid)
        if v is None:
            continue
        rows.append(np.asarray(v, dtype=np.float64).reshape(-1))
        ids.append(eid)
    if not rows:
        return np.zeros((0, 1)), [], {}
    mat = np.stack(rows, axis=0)
    idx = {eid: i for i, eid in enumerate(ids)}
    return mat, ids, idx


def build_pair_sample_dataframe(
    *,
    pair_csv: Path,
    label_map: dict[str, Any],
    max_pairs_per_relation: int,
    random_state: int,
) -> pd.DataFrame:
    from src.pair_train import load_pair_training_dataframe

    df, _ = load_pair_training_dataframe(Path(pair_csv))
    if df.empty:
        return df
    work = df.copy()
    rels: list[str | None] = []
    for _, r in work.iterrows():
        rels.append(_relation_from_gt(str(r["email_i"]), str(r["email_j"]), label_map))
    work["gt_relation"] = rels
    work = work.loc[work["gt_relation"].notna()].copy()
    if work.empty:
        return work

    rng = np.random.default_rng(int(random_state))
    parts: list[pd.DataFrame] = []
    for rel in ("same_campaign", "cross_campaign"):
        sub = work.loc[work["gt_relation"] == rel]
        if sub.empty:
            continue
        if len(sub) > int(max_pairs_per_relation):
            sub = sub.sample(n=int(max_pairs_per_relation), random_state=rng)
        parts.append(sub)
    out = pd.concat(parts, ignore_index=True) if parts else work.iloc[0:0]
    if "pair_status" not in out.columns:
        out["pair_status"] = "unlabeled"
    return out.reset_index(drop=True)


def attach_embedding_similarities(
    df: pd.DataFrame,
    *,
    sources: dict[str, dict[str, np.ndarray]],
    external_ids: list[str],
) -> pd.DataFrame:
    """Add {source}_cosine and {source}_l2 columns."""
    if df.empty:
        return df
    out = df.copy()
    matrices: dict[str, tuple[np.ndarray, dict[str, int]]] = {}
    for name, id_map in sources.items():
        mat, ids, idx = _build_email_matrix(id_map, external_ids)
        if mat.shape[0] == 0:
            continue
        matrices[name] = (mat, idx)

    cos_cols: dict[str, list[float | None]] = {n: [] for n in matrices}
    l2_cols: dict[str, list[float | None]] = {n: [] for n in matrices}

    for _, r in _tqdm(out.iterrows(), desc="pair embedding similarities", total=len(out)):
        ei, ej = str(r["email_i"]), str(r["email_j"])
        for name, (mat, idx) in matrices.items():
            ii, jj = idx.get(ei), idx.get(ej)
            if ii is None or jj is None:
                cos_cols[name].append(None)
                l2_cols[name].append(None)
            else:
                c, l2 = _cosine_l2(mat[ii], mat[jj])
                cos_cols[name].append(c)
                l2_cols[name].append(l2)
    for name in matrices:
        out[f"{name}_cosine"] = cos_cols[name]
        out[f"{name}_l2"] = l2_cols[name]
    return out


def summarize_pairwise_by_relation(
    df: pd.DataFrame,
    *,
    source_names: list[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    block: dict[str, Any] = {"by_source": {}, "by_relation": {}}
    for src in source_names:
        col = f"{src}_cosine"
        if col not in df.columns:
            continue
        src_block: dict[str, Any] = {}
        for rel in ("same_campaign", "cross_campaign"):
            sub = df.loc[df["gt_relation"] == rel]
            s = pd.to_numeric(sub[col], errors="coerce").dropna()
            if s.empty:
                continue
            y = (sub["gt_relation"] == "same_campaign").astype(int).to_numpy()
            sc = pd.to_numeric(sub[col], errors="coerce").to_numpy()
            metrics = _auroc_auprc(y, sc)
            rec = {
                "embedding_source": src,
                "gt_relation": rel,
                "n_pairs": int(len(s)),
                "cosine_mean": float(s.mean()),
                "cosine_median": float(s.median()),
                "cosine_std": float(s.std()) if len(s) > 1 else 0.0,
                "cosine_p10": float(s.quantile(0.1)),
                "cosine_p90": float(s.quantile(0.9)),
                **metrics,
            }
            rows.append(rec)
            src_block[rel] = rec
        if src_block:
            same_m = src_block.get("same_campaign", {}).get("cosine_mean")
            cross_m = src_block.get("cross_campaign", {}).get("cosine_mean")
            if same_m is not None and cross_m is not None:
                src_block["same_minus_cross_mean_cosine"] = float(same_m) - float(cross_m)
            all_y = (df["gt_relation"] == "same_campaign").astype(int).to_numpy()
            all_sc = pd.to_numeric(df[f"{src}_cosine"], errors="coerce").to_numpy()
            src_block["separation"] = _auroc_auprc(all_y, all_sc)
            block["by_source"][src] = src_block
    return pd.DataFrame(rows), block


def compute_retrieval_metrics(
    *,
    id_to_vec: dict[str, np.ndarray],
    external_ids: list[str],
    label_map: dict[str, Any],
    k_values: tuple[int, ...],
    max_emails: int,
    random_state: int,
) -> pd.DataFrame:
    from sklearn.neighbors import NearestNeighbors

    mat, ids, _ = _build_email_matrix(id_to_vec, external_ids)
    if mat.shape[0] < 3:
        return pd.DataFrame()
    if max_emails > 0 and len(ids) > max_emails:
        rng = np.random.default_rng(random_state)
        pick = rng.choice(len(ids), size=int(max_emails), replace=False)
        ids = [ids[int(i)] for i in sorted(pick)]
        mat = mat[[int(i) for i in pick]]

    mat_n = _normalize_rows(mat.astype(np.float64))
    max_k = min(max(k_values), mat_n.shape[0] - 1)
    nn = NearestNeighbors(n_neighbors=max_k + 1, metric="cosine", algorithm="brute")
    nn.fit(mat_n)

    camp = np.array([label_map.get(eid) for eid in ids], dtype=object)
    rows: list[dict[str, Any]] = []
    for ki in k_values:
        k = min(int(ki), max_k)
        distances, indices = nn.kneighbors(mat_n, n_neighbors=k + 1)
        recall_hits = []
        prec_hits = []
        hit_any = []
        rr = []
        for i in range(len(ids)):
            neigh = [int(indices[i, j]) for j in range(1, k + 1)]
            same = [j for j in neigh if camp[j] is not None and camp[j] == camp[i]]
            n_same_gt = int(
                sum(1 for j in range(len(ids)) if j != i and camp[j] == camp[i])
            )
            recall_hits.append(float(len(same) / n_same_gt) if n_same_gt > 0 else np.nan)
            prec_hits.append(float(len(same) / k) if k else np.nan)
            hit_any.append(float(len(same) > 0))
            rank = next((r for r, j in enumerate(neigh, start=1) if camp[j] == camp[i]), None)
            rr.append(1.0 / float(rank) if rank else 0.0)
        rows.append(
            {
                "k": k,
                "n_emails": len(ids),
                "recall_at_k_mean": float(np.nanmean(recall_hits)),
                "precision_at_k_mean": float(np.nanmean(prec_hits)),
                "hit_at_k_mean": float(np.mean(hit_any)),
                "mrr_mean": float(np.mean(rr)),
            }
        )
    return pd.DataFrame(rows)


def _load_connected_pairs(
    *,
    project_root: Path,
    run_dir: Path,
    candidate_union_csv: Path | None,
) -> set[tuple[str, str]]:
    from seed_candidate_workflow.utils.bridge_candidate_experiment import (
        canonical_pair,
        load_connected_pair_keys,
    )

    cand = candidate_union_csv
    if cand is None:
        gid = run_dir.name
        hint = (
            project_root
            / "seed_candidate_workflow"
            / "output"
            / "graph_bundles"
            / gid
            / "candidate"
            / gid
            / "candidate_union.csv"
        )
        cand = hint if hint.is_file() else None
    return load_connected_pair_keys(candidate_union_csv=cand, seed_edges_csv=None)


def build_high_cosine_nonedge_pairs(
    *,
    primary_source: str,
    id_to_vec: dict[str, np.ndarray],
    external_ids: list[str],
    connected: set[tuple[str, str]],
    label_map: dict[str, Any],
    cosine_threshold: float,
    max_pairs: int,
    random_state: int,
) -> pd.DataFrame:
    from seed_candidate_workflow.utils.bridge_candidate_experiment import canonical_pair
    mat, ids, _ = _build_email_matrix(id_to_vec, external_ids)
    if mat.shape[0] < 2:
        return pd.DataFrame()
    mat_n = _normalize_rows(mat.astype(np.float64))
    n = mat_n.shape[0]
    rng = np.random.default_rng(random_state)
    max_anchors = min(n, max(500, max_pairs // 10))
    anchor_idx = rng.choice(n, size=max_anchors, replace=False) if n > max_anchors else np.arange(n)
    rows: list[dict[str, Any]] = []
    for i in _tqdm(anchor_idx, desc="high-cosine non-edge scan", total=len(anchor_idx)):
        v = mat_n[int(i)]
        sims = mat_n @ v
        sims[int(i)] = -1.0
        order = np.argsort(-sims)
        ei = ids[int(i)]
        for j in order[:50]:
            if float(sims[j]) < float(cosine_threshold):
                break
            ej = ids[int(j)]
            pk = canonical_pair(ei, ej)
            if pk is None or pk in connected:
                continue
            rows.append(
                {
                    "email_i": pk[0],
                    "email_j": pk[1],
                    f"{primary_source}_cosine": float(sims[j]),
                    "gt_relation": _relation_from_gt(pk[0], pk[1], label_map),
                    "in_candidate_graph": False,
                }
            )
            if len(rows) >= int(max_pairs):
                break
        if len(rows) >= int(max_pairs):
            break
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows).sort_values(f"{primary_source}_cosine", ascending=False)
    return out.head(int(max_pairs)).reset_index(drop=True)


def _email_artifact_neighbors(data: Any) -> dict[int, set[int]]:
    """Map global email node index -> set of 1-hop artifact node global indices."""
    out: dict[int, set[int]] = defaultdict(set)
    for et in data.edge_types:
        src, _, dst = et
        if src != "email":
            continue
        ei = data[et].edge_index
        if ei is None or ei.numel() == 0:
            continue
        src_idx = ei[0].cpu().numpy()
        dst_idx = ei[1].cpu().numpy()
        for s, d in zip(src_idx, dst_idx, strict=False):
            out[int(s)].add(int(d))
    return out


def attach_graph_context_light(
    df: pd.DataFrame,
    *,
    graph_pt: Path,
    to_undirected: bool,
    id_to_row: dict[str, int],
) -> pd.DataFrame:
    if df.empty:
        return df
    from seed_candidate_workflow.utils import graph_structure_helpers as gh

    data = gh.load_hetero(Path(graph_pt), to_undirected=to_undirected)
    neigh = _email_artifact_neighbors(data)
    cn: list[int] = []
    jacc: list[float] = []
    for _, r in df.iterrows():
        ii = id_to_row.get(str(r["email_i"]))
        jj = id_to_row.get(str(r["email_j"]))
        if ii is None or jj is None:
            cn.append(0)
            jacc.append(0.0)
            continue
        ni, nj = neigh.get(ii, set()), neigh.get(jj, set())
        inter = len(ni & nj)
        union = len(ni | nj)
        cn.append(int(inter))
        jacc.append(float(inter / union) if union else 0.0)
    out = df.copy()
    out["shared_1hop_artifact_count"] = cn
    out["shared_1hop_artifact_jaccard"] = jacc
    return out


def run_embedding_probe(
    df: pd.DataFrame,
    *,
    primary_source: str,
    explicit_cols: list[str],
) -> dict[str, Any]:
    if df.empty or "gt_relation" not in df.columns:
        return {"status": "skipped"}
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        return {"status": "skipped", "reason": "sklearn_missing"}

    y = (df["gt_relation"] == "same_campaign").astype(int).to_numpy()
    cos_col = f"{primary_source}_cosine"
    X_emb = pd.to_numeric(df[cos_col], errors="coerce").fillna(0.0).to_numpy().reshape(-1, 1)
    exp_avail = [c for c in explicit_cols if c in df.columns]
    X_exp = df[exp_avail].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy() if exp_avail else np.zeros((len(df), 0))
    X_comb = np.hstack([X_emb, X_exp]) if X_exp.shape[1] else X_emb

    def _fit_report(X: np.ndarray, name: str) -> dict[str, Any]:
        if X.shape[0] < 50 or len(np.unique(y)) < 2:
            return {"status": "insufficient_data"}
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )
        scaler = StandardScaler()
        Xtr_s = scaler.fit_transform(Xtr)
        Xte_s = scaler.transform(Xte)
        clf = LogisticRegression(max_iter=500, class_weight="balanced")
        clf.fit(Xtr_s, ytr)
        prob = clf.predict_proba(Xte_s)[:, 1]
        return {"status": "ok", "name": name, **_auroc_auprc(yte, prob)}

    return {
        "explicit_only": _fit_report(X_exp, "explicit_only"),
        "embedding_only": _fit_report(X_emb, "embedding_only"),
        "combined": _fit_report(X_comb, "combined"),
        "explicit_feature_columns": exp_avail,
        "embedding_feature": cos_col,
    }


def summarize_cosine_calibration(
    df: pd.DataFrame,
    *,
    primary_source: str,
) -> dict[str, Any]:
    col = f"{primary_source}_cosine"
    if col not in df.columns:
        return {}
    out: dict[str, Any] = {}
    s = pd.to_numeric(df[col], errors="coerce")
    for label, lo, hi in COSINE_CALIBRATION_BUCKETS:
        m = s.notna()
        if lo is not None:
            m &= s >= float(lo)
        if hi is not None:
            m &= s < float(hi)
        sub = df.loc[m]
        block: dict[str, Any] = {"n_pairs": int(len(sub))}
        if "gt_relation" in sub.columns and sub["gt_relation"].notna().any():
            cov = sub["gt_relation"].notna()
            block["gt_same_fraction"] = float((sub.loc[cov, "gt_relation"] == "same_campaign").mean())
            block["gt_cross_fraction"] = float((sub.loc[cov, "gt_relation"] == "cross_campaign").mean())
        for ec in ("body_token_jaccard", "path_token_jaccard_combined", "semantic_cosine_max"):
            if ec in sub.columns:
                block[f"mean_{ec}"] = float(pd.to_numeric(sub[ec], errors="coerce").mean())
        out[label] = block
    return out


def summarize_frontier_bands(
    df: pd.DataFrame,
    *,
    primary_source: str,
    score_col: str = "score",
) -> dict[str, Any]:
    if score_col not in df.columns or f"{primary_source}_cosine" not in df.columns:
        return {}
    out: dict[str, Any] = {}
    scores = pd.to_numeric(df[score_col], errors="coerce")
    cos = pd.to_numeric(df[f"{primary_source}_cosine"], errors="coerce")
    for name, lo, hi in FRONTIER_BANDS:
        m = scores.notna() & (scores >= lo)
        if hi is not None:
            m &= scores < hi
        if "gt_relation" in df.columns:
            if name.endswith("_same"):
                m &= df["gt_relation"] == "same_campaign"
            elif name.endswith("_cross"):
                m &= df["gt_relation"] == "cross_campaign"
        sub_cos = cos.loc[m].dropna()
        out[name] = {
            "n_pairs": int(m.sum()),
            f"mean_{primary_source}_cosine": float(sub_cos.mean()) if not sub_cos.empty else None,
        }
    return out


def build_recommendations(
    *,
    pairwise_block: dict[str, Any],
    retrieval_df: pd.DataFrame,
    primary_source: str,
    encoder_meta: dict[str, Any],
    suspicious_nonedge: dict[str, Any],
    probe_summary: dict[str, Any],
) -> dict[str, Any]:
    by_src = pairwise_block.get("by_source") or {}
    primary = by_src.get(primary_source) or {}
    sep = (primary.get("separation") or {}) if isinstance(primary, dict) else {}
    auroc = sep.get("auroc")
    same_minus = primary.get("same_minus_cross_mean_cosine")
    retr_primary = retrieval_df.loc[retrieval_df.index[:0]]
    if not retrieval_df.empty and "embedding_source" in retrieval_df.columns:
        retr_primary = retrieval_df                                

    useful = auroc is not None and float(auroc) >= 0.65
    return {
        "A_are_gnn_embeddings_useful": (
            f"Primary source '{primary_source}' AUROC(same vs cross)={auroc}; "
            f"mean cosine gap same-cross={same_minus}. "
            + ("Embeddings show campaign signal." if useful else "Weak same/cross separation — interpret latent scores cautiously.")
        ),
        "B_vs_baselines": (
            "Compare AUROC and same-minus-cross across sources in pairwise_similarity_summary.csv."
        ),
        "C_high_cosine_nonedges_trustworthy": suspicious_nonedge,
        "D_bridge_recovery": (
            "Favor bridge addition when high cosine non-edges are GT-same and have graph-context overlap; "
            "hold when high-cosine + weak context + cross GT."
        ),
        "E_next_intervention": (
            "Reduce reliance on latent-only bridges" if not useful else "Keep encoder; tighten bridge thresholds using scorer_input cosine + shared artifacts"
        ),
        "pair_encoder_backend": encoder_meta.get("pair_encoder_backend"),
        "primary_embedding_source": primary_source,
        "probe_summary": probe_summary,
    }


def _write_similarity_plots(
    df: pd.DataFrame,
    *,
    source_names: list[str],
    plots_dir: Path,
) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    for src in source_names:
        col = f"{src}_cosine"
        if col not in df.columns or "gt_relation" not in df.columns:
            continue
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for rel, color in (("same_campaign", "#ff7f0e"), ("cross_campaign", "#1f77b4")):
            s = pd.to_numeric(df.loc[df["gt_relation"] == rel, col], errors="coerce").dropna()
            if len(s) > 5:
                ax.hist(s, bins=40, alpha=0.55, density=True, label=rel, color=color)
        ax.set_xlabel(f"{src} cosine")
        ax.set_title(f"Pairwise {src} cosine by GT relation")
        ax.legend()
        ax.grid(True, alpha=0.3)
        p = plots_dir / f"pairwise_cosine_{src}.png"
        fig.tight_layout()
        fig.savefig(p, dpi=120)
        plt.close(fig)
        written.append(p.name)
    return written


def write_embedding_review_html(
    df: pd.DataFrame,
    *,
    out_path: Path,
    title: str,
    primary_source: str,
) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cards: list[str] = []
    for i, (_, r) in enumerate(df.iterrows()):
        lines = [
            f"GT: {r.get('gt_relation', 'unknown')}",
            f"score: {r.get('score', '—')}",
            f"{primary_source}_cosine: {r.get(f'{primary_source}_cosine', '—')}",
            f"shared_1hop_artifact_jaccard: {r.get('shared_1hop_artifact_jaccard', '—')}",
            f"body_token_jaccard: {r.get('body_token_jaccard', '—')}",
            f"retrieval: {r.get('retrieval_channels', '—')}",
        ]
        cards.append(
            f'<section class="card" id="p{i}"><h3>Pair {i+1}</h3><ul>'
            + "".join(f"<li>{html.escape(ln)}</li>" for ln in lines)
            + f"<li>{html.escape(str(r.get('email_i')))} ↔ {html.escape(str(r.get('email_j')))}</li></ul></section>"
        )
    doc = f"""<!DOCTYPE html><html><head><meta charset="utf-8"/><title>{html.escape(title)}</title>
    <style>body{{font-family:system-ui;background:#111;color:#eee;padding:1rem}}
    .card{{border:1px solid #444;padding:0.75rem;margin-bottom:1rem;border-radius:6px}}</style></head>
    <body><h1>{html.escape(title)}</h1>{"".join(cards)}</body></html>"""
    out_path.write_text(doc, encoding="utf-8")


def run_gnn_embedding_diagnostics(cfg: GnnEmbeddingDiagConfig) -> dict[str, Any]:
    """Main entry: write all artifacts under run_dir / output_subdir."""
    t0 = time.perf_counter()
    timing: dict[str, float] = {}
    _ensure_sys_path()
    project_root = _repo_root()
    run_dir = Path(cfg.run_dir).resolve()
    graph_pt = Path(cfg.graph_pt).resolve()
    out_root = (run_dir / cfg.output_subdir).resolve()
    plots_dir = out_root / "plots"
    debug_json = out_root / "debug_json"
    debug_csv = out_root / "debug_csv"
    for d in (out_root, plots_dir, debug_json, debug_csv):
        d.mkdir(parents=True, exist_ok=True)

    gt_paths = [Path(p).resolve() for p in cfg.gt_paths if Path(p).is_file()]
    label_map = _load_label_map(gt_paths)

    t_emb = time.perf_counter()
    sources, encoder_meta, external_ids = load_embedding_sources(
        run_dir=run_dir,
        graph_pt=graph_pt,
        checkpoint_name=cfg.checkpoint_name,
        device=cfg.device,
        to_undirected=cfg.to_undirected,
        embeddings_json=cfg.embeddings_json,
        project_root=project_root,
    )
    timing["load_embeddings_sec"] = time.perf_counter() - t_emb

    primary = str(encoder_meta.get("primary_scorer_embedding_source") or "gnn_encoder")
    if primary not in sources and "scorer_input_email_x" in sources:
        primary = "scorer_input_email_x"

    from seed_candidate_workflow.utils.pair_model_inference import resolve_pair_dataset_csv_path

    pair_csv = (
        Path(cfg.pair_csv).resolve()
        if cfg.pair_csv
        else resolve_pair_dataset_csv_path(run_dir, project_root=project_root)
    )

    t_pairs = time.perf_counter()
    pair_df = build_pair_sample_dataframe(
        pair_csv=pair_csv,
        label_map=label_map,
        max_pairs_per_relation=int(cfg.max_pairs_per_relation),
        random_state=int(cfg.random_state),
    )
    pair_df = attach_embedding_similarities(
        pair_df, sources=sources, external_ids=external_ids
    )
    timing["pairwise_similarity_sec"] = time.perf_counter() - t_pairs

    source_names = list(sources.keys())
    pairwise_csv_df, pairwise_block = summarize_pairwise_by_relation(
        pair_df, source_names=source_names
    )
    pairwise_path = out_root / "gnn_embedding_pairwise_similarity_summary.csv"
    pairwise_csv_df.to_csv(pairwise_path, index=False)

    t_retr = time.perf_counter()
    retr_rows: list[pd.DataFrame] = []
    for src in source_names:
        sub = compute_retrieval_metrics(
            id_to_vec=sources[src],
            external_ids=external_ids,
            label_map=label_map,
            k_values=cfg.retrieval_k_values,
            max_emails=int(cfg.max_emails_for_retrieval),
            random_state=int(cfg.random_state),
        )
        if not sub.empty:
            sub = sub.copy()
            sub["embedding_source"] = src
            retr_rows.append(sub)
    retrieval_df = pd.concat(retr_rows, ignore_index=True) if retr_rows else pd.DataFrame()
    retr_path = out_root / "gnn_embedding_retrieval_metrics.csv"
    retrieval_df.to_csv(retr_path, index=False)
    timing["retrieval_sec"] = time.perf_counter() - t_retr

    connected = _load_connected_pairs(
        project_root=project_root,
        run_dir=run_dir,
        candidate_union_csv=cfg.candidate_union_csv,
    )
    t_non = time.perf_counter()
    nonedge_df = build_high_cosine_nonedge_pairs(
        primary_source=primary,
        id_to_vec=sources[primary],
        external_ids=external_ids,
        connected=connected,
        label_map=label_map,
        cosine_threshold=float(cfg.high_cosine_threshold),
        max_pairs=int(cfg.max_high_cosine_nonedges),
        random_state=int(cfg.random_state),
    )
    _, _, id_to_row = _build_email_matrix(sources[primary], external_ids)
    nonedge_df = attach_graph_context_light(
        nonedge_df,
        graph_pt=graph_pt,
        to_undirected=cfg.to_undirected,
        id_to_row=id_to_row,
    )
    if cfg.bridge_scores_csv and Path(cfg.bridge_scores_csv).is_file():
        br = pd.read_csv(cfg.bridge_scores_csv, usecols=["email_i", "email_j", "score", "retrieval_channels"], low_memory=False)
        br["_pk"] = br["email_i"].astype(str) + "\0" + br["email_j"].astype(str)
        nonedge_df["_pk"] = nonedge_df["email_i"].astype(str) + "\0" + nonedge_df["email_j"].astype(str)
        nonedge_df = nonedge_df.merge(br.drop_duplicates("_pk"), on="_pk", how="left", suffixes=("", "_br"))
    nonedge_summary = {
        "n_high_cosine_nonedges": int(len(nonedge_df)),
        "cosine_threshold": float(cfg.high_cosine_threshold),
        "gt_same_fraction": float((nonedge_df["gt_relation"] == "same_campaign").mean())
        if nonedge_df["gt_relation"].notna().any()
        else None,
        "gt_cross_fraction": float((nonedge_df["gt_relation"] == "cross_campaign").mean())
        if nonedge_df["gt_relation"].notna().any()
        else None,
        "mean_shared_1hop_jaccard": float(
            pd.to_numeric(nonedge_df.get("shared_1hop_artifact_jaccard"), errors="coerce").mean()
        )
        if "shared_1hop_artifact_jaccard" in nonedge_df.columns
        else None,
    }
    with open(out_root / "gnn_embedding_high_cosine_nonedge_summary.json", "w", encoding="utf-8") as f:
        json.dump(nonedge_summary, f, indent=2, default=str)
    nonedge_df.to_csv(debug_csv / "gnn_embedding_high_cosine_nonedges.csv", index=False)
    timing["nonedge_sec"] = time.perf_counter() - t_non

    explicit_cols = [
        "body_token_jaccard",
        "body_only_token_jaccard",
        "path_token_jaccard_combined",
        "semantic_cosine_max",
        "sender_localpart_norm_jaccard",
    ]
    probe_summary = (
        {}
        if cfg.skip_probe
        else run_embedding_probe(pair_df, primary_source=primary, explicit_cols=explicit_cols)
    )
    if probe_summary:
        with open(out_root / "gnn_embedding_probe_summary.json", "w", encoding="utf-8") as f:
            json.dump(probe_summary, f, indent=2, default=str)

    calibration = summarize_cosine_calibration(pair_df, primary_source=primary)
    frontier = summarize_frontier_bands(pair_df, primary_source=primary, score_col="score")
    if "score" not in pair_df.columns:
        sep_path = run_dir / "pair_score_separation" / "core_csv" / "pair_score_separation_table.csv"
        if sep_path.is_file():
            sep = pd.read_csv(sep_path, usecols=["email_i", "email_j", "score"], low_memory=False)
            merged = pair_df.merge(sep, on=["email_i", "email_j"], how="left", suffixes=("", "_sep"))
            frontier = summarize_frontier_bands(merged, primary_source=primary, score_col="score")

    plot_files: list[str] = []
    if not cfg.skip_plots:
        plot_files = _write_similarity_plots(pair_df, source_names=source_names, plots_dir=plots_dir)

    if not cfg.skip_html and not nonedge_df.empty:
        hi_same = nonedge_df.loc[nonedge_df["gt_relation"] == "same_campaign"].head(cfg.max_review_pairs // 3)
        hi_cross = nonedge_df.loc[nonedge_df["gt_relation"] == "cross_campaign"].head(cfg.max_review_pairs // 3)
        hi_unk = nonedge_df.loc[nonedge_df["gt_relation"].isna()].head(cfg.max_review_pairs // 3)
        write_embedding_review_html(
            hi_same,
            out_path=out_root / "gnn_embedding_high_cosine_gt_same_for_review.html",
            title="High-cosine non-edges — GT same",
            primary_source=primary,
        )
        write_embedding_review_html(
            hi_cross,
            out_path=out_root / "gnn_embedding_high_cosine_gt_cross_for_review.html",
            title="High-cosine non-edges — GT cross",
            primary_source=primary,
        )
        write_embedding_review_html(
            nonedge_df.head(int(cfg.max_review_pairs)),
            out_path=out_root / "gnn_embedding_high_cosine_nonedges_for_review.html",
            title="High-cosine non-edges",
            primary_source=primary,
        )

    recommendations = build_recommendations(
        pairwise_block=pairwise_block,
        retrieval_df=retrieval_df,
        primary_source=primary,
        encoder_meta=encoder_meta,
        suspicious_nonedge=nonedge_summary,
        probe_summary=probe_summary,
    )

    timing["total_sec"] = time.perf_counter() - t0
    summary = {
        "run_dir": str(run_dir),
        "graph_pt": str(graph_pt),
        "output_dir": str(out_root),
        "gt_paths": [str(p) for p in gt_paths],
        "pair_csv": str(pair_csv),
        "embedding_sources": source_names,
        "encoder_meta": encoder_meta,
        "primary_embedding_source": primary,
        "pairwise_similarity": pairwise_block,
        "cosine_calibration_by_bucket": calibration,
        "frontier_embedding_bands": frontier,
        "high_cosine_nonedge": nonedge_summary,
        "gnn_embedding_recommendations": recommendations,
        "plot_files": plot_files,
        "timing_seconds": timing,
        "export_paths": {
            "summary_json": str(out_root / "gnn_embedding_diagnostic_summary.json"),
            "pairwise_csv": str(pairwise_path),
            "retrieval_csv": str(retr_path),
            "high_cosine_nonedge_json": str(out_root / "gnn_embedding_high_cosine_nonedge_summary.json"),
        },
    }
    with open(out_root / "gnn_embedding_diagnostic_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    return summary
