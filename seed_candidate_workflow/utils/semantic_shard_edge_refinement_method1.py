"""
Method 1: unsupervised shard-graph edge refinement + optional downstream Leiden.

Operates only on Step-2 shard graph artifacts (nodes, weighted edges). No labels,
no email-level GNN, no link-prediction training. Edge-list centric.

Refinement rules (after optional trust remap):

- **Trust remap (unsupervised):** ``edge_trust_calibrated = clip(edge_trust, 0, 1) ** trust_gamma``
  (``trust_gamma = 1`` leaves trust unchanged; ``gamma < 1`` raises low trusts and reduces over-suppression).

- **multiplicative:** ``edge_weight_refined = edge_weight_orig * edge_trust_calibrated``

- **softened:** ``edge_weight_refined = edge_weight_orig * (blend_floor + (1 - blend_floor) * edge_trust_calibrated)``

- **convex:** ``edge_weight_refined = convex_alpha * norm(edge_weight_orig) + (1 - convex_alpha) * edge_trust_calibrated``
  where ``norm`` is percentile min–max (5th–95th percentile of ``edge_weight_orig`` on this edge list),
  clipped to ``[0, 1]``. Both terms are in ``[0, 1]``, so refined weights are in ``[0, 1]``.

Raw ``edge_trust`` (pre-power) is always stored; calibrated trust is stored when ``trust_gamma != 1`` or for clarity when ``gamma == 1``.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Config (ablation-friendly)
# ---------------------------------------------------------------------------


@dataclass
class Method1RefinementConfig:
    """Hyperparameters for Method 1 edge trust. All unsupervised."""

    random_seed: int = 0
    weight_col: str = "edge_weight"
    use_semantic_view: bool = True
    use_infra_view: bool = True
    use_temporal_view: bool = True
    use_local_structure: bool = True
    use_perturbation_stability: bool = True
    n_perturb_passes: int = 12
    perturb_scale_low: float = 0.65
    perturb_scale_high: float = 1.0
    perturb_drop_one_view_prob: float = 0.25
    stability_mix: float = 0.55
    trust_epsilon: float = 1e-6
    clip_trust: tuple[float, float] = (0.0, 1.0)
    # Monotonic trust remap: trust_cal = clip(trust_raw, lo, hi) ** trust_gamma (gamma > 0)
    trust_gamma: float = 1.0
    # ``multiplicative``: refined = orig * trust_cal
    # ``softened``: refined = orig * (blend_floor + (1 - blend_floor) * trust_cal)
    # ``convex``: refined = convex_alpha * norm(orig) + (1 - convex_alpha) * trust_cal
    blend_rule: str = "multiplicative"
    blend_floor: float = 0.0
    # Used only when blend_rule == ``convex``; weight on normalized original edge weight
    convex_alpha: float = 0.5

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["clip_trust"] = list(self.clip_trust)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Method1RefinementConfig:
        d = dict(d)
        if "clip_trust" in d and isinstance(d["clip_trust"], list):
            d["clip_trust"] = tuple(float(x) for x in d["clip_trust"])
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# Features and views
# ---------------------------------------------------------------------------


def build_method1_edge_feature_frame(
    edges_df: pd.DataFrame,
    *,
    weight_col: str = "edge_weight",
) -> dict[str, Any]:
    """
    Parse Step-2 edge table into grouped column metadata used by Method 1.

    Returns a dict with ``edges_df`` (copy), ``weight_col``, and infra column lists.
    """
    e = edges_df.copy()
    e["shard_a"] = e["shard_a"].astype(str)
    e["shard_b"] = e["shard_b"].astype(str)
    if weight_col not in e.columns:
        raise KeyError(f"Missing weight column {weight_col!r} on edges dataframe.")

    infra_count_cols = sorted(
        c for c in e.columns if c.startswith("shared_") and c.endswith("_count")
    )
    infra_idf_cols = sorted(c for c in e.columns if c.startswith("shared_") and c.endswith("_idf_sum"))
    infra_contrib_cols = sorted(
        c for c in e.columns if c.startswith("infra_contrib_") and not c.endswith("_pre_cap")
    )

    return {
        "edges_df": e,
        "weight_col": weight_col,
        "infra_count_cols": infra_count_cols,
        "infra_idf_cols": infra_idf_cols,
        "infra_contrib_cols": infra_contrib_cols,
    }


def _rank01(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    if n == 0:
        return x
    order = np.argsort(x)
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.arange(n, dtype=np.float64)
    return ranks / max(1.0, float(n - 1))


def compute_method1_view_scores(
    feat: dict[str, Any],
    *,
    cfg: Method1RefinementConfig,
) -> pd.DataFrame:
    """
    Per-edge normalized view scores in [0, 1]: semantic, infra, temporal.
    """
    e = feat["edges_df"]
    n = len(e)
    out = pd.DataFrame({"shard_a": e["shard_a"].values, "shard_b": e["shard_b"].values})

    if cfg.use_semantic_view and "centroid_cosine" in e.columns:
        sem_raw = pd.to_numeric(e["centroid_cosine"], errors="coerce").fillna(-1.0).to_numpy()
        sem_pos = np.maximum(0.0, np.clip(sem_raw, -1.0, 1.0))
        out["view_semantic"] = _rank01(sem_pos)
    else:
        out["view_semantic"] = np.full(n, 0.5, dtype=np.float64)

    count_cols: list[str] = feat["infra_count_cols"]
    idf_cols: list[str] = feat["infra_idf_cols"]
    contrib_cols: list[str] = feat["infra_contrib_cols"]

    if cfg.use_infra_view and "infra_score" in e.columns:
        infra_strength = pd.to_numeric(e["infra_score"], errors="coerce").fillna(0.0).to_numpy()
        strength_n = _rank01(infra_strength)

        if count_cols:
            cnt_mat = e[count_cols].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy()
            active = (cnt_mat > 0).astype(np.float64)
            n_ch = float(max(1, cnt_mat.shape[1]))
            diversity = active.sum(axis=1) / n_ch
        else:
            diversity = np.full(n, 0.5, dtype=np.float64)

        if idf_cols:
            idf_mat = e[idf_cols].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy()
            idf_sum = idf_mat.sum(axis=1)
            rarity_n = _rank01(idf_sum)
        else:
            rarity_n = np.full(n, 0.5, dtype=np.float64)

        if contrib_cols:
            c_mat = e[contrib_cols].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy()
            s = np.maximum(c_mat.sum(axis=1), cfg.trust_epsilon)
            mx = np.max(c_mat, axis=1)
            dominance = mx / s
            hub_factor = 1.0 - np.clip(dominance, 0.0, 1.0)
        else:
            hub_factor = np.ones(n, dtype=np.float64)

        div_n = _rank01(diversity)
        hub_n = _rank01(hub_factor)
        out["view_infra"] = np.clip(
            0.34 * strength_n + 0.22 * rarity_n + 0.22 * div_n + 0.22 * hub_n, 0.0, 1.0
        )
    else:
        out["view_infra"] = np.full(n, 0.5, dtype=np.float64)

    if cfg.use_temporal_view:
        if "temporal_score" in e.columns:
            t = pd.to_numeric(e["temporal_score"], errors="coerce").fillna(0.0).to_numpy()
            out["view_temporal"] = _rank01(np.clip(t, 0.0, 1.0))
        elif "temporal_overlap" in e.columns:
            o = pd.to_numeric(e["temporal_overlap"], errors="coerce").fillna(0.0).to_numpy()
            out["view_temporal"] = _rank01(np.clip(o, 0.0, 1.0))
        else:
            out["view_temporal"] = np.full(n, 0.5, dtype=np.float64)
    else:
        out["view_temporal"] = np.full(n, 0.5, dtype=np.float64)

    return out


def _neighbor_weight_map(edges_df: pd.DataFrame, weight_col: str) -> dict[str, dict[str, float]]:
    nw: dict[str, dict[str, float]] = defaultdict(dict)
    for _, r in edges_df.iterrows():
        a, b = str(r["shard_a"]), str(r["shard_b"])
        w = float(r[weight_col])
        prev = nw[a].get(b)
        if prev is None or w > prev:
            nw[a][b] = w
            nw[b][a] = w
    return nw


def compute_method1_local_structure_features(
    feat: dict[str, Any],
    *,
    cfg: Method1RefinementConfig,
) -> np.ndarray:
    """
    Local support per edge: weighted triangle-like support + normalized embeddedness.
    """
    e = feat["edges_df"]
    wcol = feat["weight_col"]
    n = len(e)
    if not cfg.use_local_structure or n == 0:
        return np.full(n, 0.5, dtype=np.float64)

    nw = _neighbor_weight_map(e, wcol)
    tri_list: list[float] = []
    emb_list: list[float] = []

    for _, r in e.iterrows():
        a, b = str(r["shard_a"]), str(r["shard_b"])
        w_ab = max(float(r[wcol]), cfg.trust_epsilon)
        na = nw.get(a, {})
        nb = nw.get(b, {})
        common = set(na.keys()) & set(nb.keys())
        common.discard(a)
        common.discard(b)
        tri_support = 0.0
        for k in common:
            tri_support += min(float(na[k]), float(nb[k]))
        deg_a = max(1, len(na))
        deg_b = max(1, len(nb))
        emb = len(common) / float(min(deg_a, deg_b))
        tri_list.append(tri_support / (w_ab + cfg.trust_epsilon))
        emb_list.append(emb)

    tri_arr = np.asarray(tri_list, dtype=np.float64)
    emb_arr = np.asarray(emb_list, dtype=np.float64)
    tri_n = _rank01(tri_arr)
    emb_n = _rank01(emb_arr)
    local = 0.55 * tri_n + 0.45 * emb_n
    return np.clip(local, 0.0, 1.0)


def _geom_mean_views(mat: np.ndarray, eps: float) -> np.ndarray:
    """mat: (n_edges, n_views), values in [0,1]."""
    mat = np.maximum(mat, eps)
    return np.exp(np.mean(np.log(mat), axis=1))


def compute_method1_perturbation_stability(
    view_block: np.ndarray,
    *,
    cfg: Method1RefinementConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Recompute geometric-mean trust under random per-view scaling and occasional view drop.

    Returns per-edge stability in [0, 1] (higher = less sensitive to perturbation).
    """
    if not cfg.use_perturbation_stability or view_block.shape[0] == 0:
        n = view_block.shape[0]
        ones = np.ones(n, dtype=np.float64)
        return ones, {"n_passes": 0, "scheme": "disabled"}

    rng = np.random.default_rng(int(cfg.random_seed))
    n_passes = max(1, int(cfg.n_perturb_passes))
    scores = np.empty((view_block.shape[0], n_passes), dtype=np.float64)
    lo, hi = float(cfg.perturb_scale_low), float(cfg.perturb_scale_high)
    drop_p = float(cfg.perturb_drop_one_view_prob)

    for p in range(n_passes):
        m = view_block.copy()
        scales = rng.uniform(lo, hi, size=m.shape)
        m *= scales
        if drop_p > 0 and m.shape[1] > 1:
            mask_drop = rng.random(m.shape[0]) < drop_p
            if mask_drop.any():
                j = rng.integers(0, m.shape[1], size=int(mask_drop.sum()))
                m[mask_drop, j] *= 0.05
        scores[:, p] = _geom_mean_views(m, cfg.trust_epsilon)

    mean_s = scores.mean(axis=1)
    std_s = scores.std(axis=1)
    cv = std_s / (mean_s + cfg.trust_epsilon)
    stability = 1.0 / (1.0 + cv)
    stab_n = _rank01(stability)
    meta = {
        "n_passes": n_passes,
        "scheme": "per_view_uniform_scale_and_random_dropout",
        "perturb_scale_range": [lo, hi],
        "drop_one_view_prob": drop_p,
    }
    return stab_n, meta


def fit_or_compute_method1_edge_trust(
    views_df: pd.DataFrame,
    local_support: np.ndarray,
    *,
    cfg: Method1RefinementConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Combine views + local support + perturbation stability into ``edge_trust`` [0,1].

    Core logic:
    - base = geometric mean of semantic, infra, temporal, local (bottleneck / agreement).
    - agreement boost: penalize edges whose minimum view is far below the geometric mean.
    - multiply by stability-derived factor when perturbations enabled.
    """
    n = len(views_df)
    sem = views_df["view_semantic"].to_numpy(dtype=np.float64)
    inf = views_df["view_infra"].to_numpy(dtype=np.float64)
    tim = views_df["view_temporal"].to_numpy(dtype=np.float64)
    if cfg.use_local_structure:
        loc = np.clip(np.asarray(local_support, dtype=np.float64), 0.0, 1.0)
        block = np.column_stack([sem, inf, tim, loc])
        views_used = ["semantic", "infra", "temporal", "local_structure"]
    else:
        block = np.column_stack([sem, inf, tim])
        views_used = ["semantic", "infra", "temporal"]

    base = _geom_mean_views(np.maximum(block, cfg.trust_epsilon), cfg.trust_epsilon)
    vmin = np.min(block, axis=1)
    agree = np.sqrt(np.clip(vmin, 0.0, 1.0) * base)

    stab, pmeta = compute_method1_perturbation_stability(block, cfg=cfg)
    mix = float(np.clip(cfg.stability_mix, 0.0, 1.0))
    if cfg.use_perturbation_stability and mix > 0:
        combined = (1.0 - mix) * agree + mix * (agree * (0.55 + 0.45 * stab))
    else:
        combined = agree

    lo, hi = cfg.clip_trust
    trust = np.clip(combined, lo, hi)
    details = {
        "perturbation": pmeta,
        "stability_mix_applied": float(mix),
        "formula": "edge_trust = clip( (1-mix)*sqrt(vmin*gmean) + mix*sqrt(vmin*gmean)*(0.55+0.45*stab) )",
        "views_used": views_used,
    }
    return trust, details


def _percentile_norm_weights(orig: np.ndarray, *, lo_q: float = 5.0, hi_q: float = 95.0) -> tuple[np.ndarray, dict[str, float]]:
    """
    Unsupervised [0,1] normalization of positive edge weights using percentiles.

    norm = clip((w - p_lo) / max(p_hi - p_lo, eps), 0, 1) with p_lo, p_hi the lo_q/hi_q
    percentiles of ``orig`` (computed on the full edge list, no labels).
    """
    o = np.asarray(orig, dtype=np.float64)
    if o.size == 0:
        return o, {"p_lo": 0.0, "p_hi": 1.0, "lo_q": float(lo_q), "hi_q": float(hi_q)}
    p_lo = float(np.percentile(o, lo_q))
    p_hi = float(np.percentile(o, hi_q))
    span = max(p_hi - p_lo, 1e-12)
    norm = np.clip((o - p_lo) / span, 0.0, 1.0)
    meta = {"p_lo": p_lo, "p_hi": p_hi, "span": span, "lo_q": float(lo_q), "hi_q": float(hi_q)}
    return norm, meta


def apply_method1_refinement(
    edges_df: pd.DataFrame,
    edge_trust: np.ndarray,
    *,
    cfg: Method1RefinementConfig,
) -> pd.DataFrame:
    """
    Attach ``edge_weight_orig``, ``edge_trust`` (raw), ``edge_trust_calibrated``, blend columns, ``edge_weight_refined``.

    Trust calibration: ``trust_cal = clip(trust_raw, clip_lo, clip_hi) ** trust_gamma``.

    - ``multiplicative``: ``refined = orig * trust_cal``
    - ``softened``: ``mult = blend_floor + (1 - blend_floor) * trust_cal``; ``refined = orig * mult``
    - ``convex``: ``refined = convex_alpha * norm(orig) + (1 - convex_alpha) * trust_cal`` (see module doc)
    """
    e = edges_df.copy()
    wcol = cfg.weight_col
    if wcol not in e.columns:
        raise KeyError(wcol)
    orig = pd.to_numeric(e[wcol], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    trust_raw = np.asarray(edge_trust, dtype=np.float64)
    if len(trust_raw) != len(orig):
        raise ValueError("edge_trust length mismatch")
    lo, hi = cfg.clip_trust
    t_clip = np.clip(trust_raw, float(lo), float(hi))
    gamma = float(cfg.trust_gamma)
    if gamma <= 0:
        raise ValueError("trust_gamma must be positive")
    trust_cal = np.power(t_clip, gamma)

    rule = str(cfg.blend_rule).lower().strip()
    floor = float(np.clip(cfg.blend_floor, 0.0, 0.999999))
    if rule == "softened":
        mult = floor + (1.0 - floor) * trust_cal
        refined = orig * mult
    elif rule in {"multiplicative", "mult", "mul"}:
        mult = trust_cal
        refined = orig * mult
    elif rule == "convex":
        alpha = float(np.clip(cfg.convex_alpha, 0.0, 1.0))
        norm_o, _nmeta = _percentile_norm_weights(orig)
        refined = alpha * norm_o + (1.0 - alpha) * trust_cal
        mult = np.divide(
            refined,
            np.maximum(orig, 1e-12),
            out=np.zeros_like(refined),
            where=orig > 1e-12,
        )
        e["edge_weight_orig_norm_convex"] = norm_o
    else:
        raise ValueError(
            f"Unknown blend_rule {cfg.blend_rule!r} (use 'multiplicative', 'softened', or 'convex')"
        )

    e["edge_weight_orig"] = orig
    e["edge_trust"] = trust_raw
    e["edge_trust_calibrated"] = trust_cal
    e["edge_blend_multiplier"] = mult
    e["edge_weight_refined"] = refined
    return e


def save_method1_edge_refinement_artifacts(
    refined_edges_df: pd.DataFrame,
    *,
    output_dir: str | Path,
    cfg: Method1RefinementConfig,
    fit_summary: dict[str, Any],
) -> dict[str, str]:
    out = Path(output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    p_edges = out / "semantic_shard_step2_edges_refined.csv"
    p_cfg = out / "semantic_shard_method1_config.json"
    p_sum = out / "semantic_shard_method1_fit_summary.json"
    refined_edges_df.to_csv(p_edges, index=False)
    p_cfg.write_text(json.dumps(cfg.to_dict(), indent=2), encoding="utf-8")
    p_sum.write_text(json.dumps(fit_summary, indent=2), encoding="utf-8")
    return {"edges_refined_csv": str(p_edges), "config_json": str(p_cfg), "fit_summary_json": str(p_sum)}


def save_method1_calibration_variant_bundle(
    refined_edges_df: pd.DataFrame,
    *,
    bundle_dir: str | Path,
    cfg: Method1RefinementConfig,
    fit_summary: dict[str, Any],
) -> dict[str, str]:
    """
    Save a named calibration variant under ``bundle_dir/`` using the same filenames as production Step 2b.

    Intended layout: ``.../m1_soft_floor040_gamma075/semantic_shard_step2_edges_refined.csv`` plus JSON sidecars.
    """
    return save_method1_edge_refinement_artifacts(
        refined_edges_df, output_dir=bundle_dir, cfg=cfg, fit_summary=fit_summary
    )


def method1_fit_summary_table(refined_edges_df: pd.DataFrame) -> dict[str, Any]:
    t = refined_edges_df["edge_trust"]
    wr = refined_edges_df["edge_weight_refined"]
    wo = refined_edges_df["edge_weight_orig"]
    out: dict[str, Any] = {
        "n_edges": int(len(refined_edges_df)),
        "edge_trust_mean": float(t.mean()),
        "edge_trust_std": float(t.std()),
        "edge_trust_median": float(t.median()),
        "edge_trust_min": float(t.min()),
        "edge_trust_max": float(t.max()),
        "edge_weight_orig_mean": float(wo.mean()),
        "edge_weight_refined_mean": float(wr.mean()),
        "mean_shrink_factor": float((wr / np.maximum(wo, 1e-12)).mean()) if len(wr) else float("nan"),
    }
    if "edge_trust_calibrated" in refined_edges_df.columns:
        tc = refined_edges_df["edge_trust_calibrated"]
        out["edge_trust_calibrated_mean"] = float(tc.mean())
        out["edge_trust_calibrated_median"] = float(tc.median())
        out["edge_trust_calibrated_max"] = float(tc.max())
    return out


def run_method1_edge_refinement_pipeline(
    edges_df: pd.DataFrame,
    *,
    cfg: Method1RefinementConfig | None = None,
    output_dir: str | Path | None,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, str] | None]:
    """
    Full Method 1: features -> views -> local -> trust -> refined weights -> optional save.
    """
    cfg = cfg or Method1RefinementConfig()
    feat = build_method1_edge_feature_frame(edges_df, weight_col=cfg.weight_col)
    views = compute_method1_view_scores(feat, cfg=cfg)
    local = compute_method1_local_structure_features(feat, cfg=cfg)
    views = views.reset_index(drop=True)
    trust, tdetails = fit_or_compute_method1_edge_trust(views, local, cfg=cfg)
    refined = apply_method1_refinement(feat["edges_df"], trust, cfg=cfg)
    stats = method1_fit_summary_table(refined)
    blend_meta: dict[str, Any] = {
        "blend_rule": cfg.blend_rule,
        "trust_gamma": float(cfg.trust_gamma),
        "blend_floor": float(cfg.blend_floor),
        "convex_alpha": float(cfg.convex_alpha),
    }
    if str(cfg.blend_rule).lower().strip() == "convex":
        wo = pd.to_numeric(refined["edge_weight_orig"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        _norm, nmeta = _percentile_norm_weights(wo)
        blend_meta["convex_norm"] = "percentile_minmax_5_95"
        blend_meta["convex_percentile_meta"] = nmeta
    fit_summary: dict[str, Any] = {
        "trust_details": tdetails,
        "blend_calibration": blend_meta,
        "column_groups": {
            "infra_count_cols": feat["infra_count_cols"],
            "infra_idf_cols": feat["infra_idf_cols"],
            "infra_contrib_cols": feat["infra_contrib_cols"],
        },
        "stats": stats,
    }
    paths = None
    if output_dir is not None:
        paths = save_method1_edge_refinement_artifacts(
            refined, output_dir=output_dir, cfg=cfg, fit_summary=fit_summary
        )
    return refined, fit_summary, paths


def synthetic_method1_sanity_check() -> dict[str, Any]:
    """Minimal graph: no crash; trusts bounded."""
    edges = pd.DataFrame(
        [
            {
                "shard_a": "s0",
                "shard_b": "s1",
                "centroid_cosine": 0.9,
                "infra_score": 0.8,
                "temporal_score": 1.0,
                "shared_url_count": 1,
                "shared_url_idf_sum": 0.5,
                "infra_contrib_url": 0.3,
                "edge_weight": 1.2,
            },
            {
                "shard_a": "s1",
                "shard_b": "s2",
                "centroid_cosine": 0.2,
                "infra_score": 0.1,
                "temporal_score": 0.4,
                "shared_url_count": 1,
                "shared_url_idf_sum": 0.01,
                "infra_contrib_url": 0.1,
                "edge_weight": 0.35,
            },
            {
                "shard_a": "s0",
                "shard_b": "s2",
                "centroid_cosine": 0.5,
                "infra_score": 0.2,
                "temporal_score": 0.8,
                "shared_url_count": 0,
                "shared_url_idf_sum": 0.0,
                "infra_contrib_url": 0.0,
                "edge_weight": 0.5,
            },
        ]
    )
    test_cfg = Method1RefinementConfig(random_seed=42, n_perturb_passes=8)
    refined, _, _ = run_method1_edge_refinement_pipeline(edges, cfg=test_cfg, output_dir=None)
    t = refined["edge_trust"].to_numpy()
    assert np.all(t >= 0) and np.all(t <= 1.0), "trust out of bounds"
    return {"ok": True, "n_edges": len(refined), "trust_range": (float(t.min()), float(t.max()))}
