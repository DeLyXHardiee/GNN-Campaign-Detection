"""
Enrich GT pair samples with features used by candidate-family scorecard rules.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover

    def tqdm(iterable=None, **kwargs):  # type: ignore[misc]
        return iterable if iterable is not None else []

from seed_candidate_workflow.utils.gt_edge_structure_analysis import _load_embeddings
from seed_candidate_workflow.utils.pair_low_band_feature_discovery import (
    _build_artifact_df_maps,
    _build_body_rare_token_set,
    _compute_pair_features_row,
    _load_anchor_nodes_extended,
)
from seed_candidate_workflow.utils.pair_low_band_twohop_channel import (
    attach_twohop_channel_columns,
    extend_bool_terms_for_low_band_channels,
)
from seed_candidate_workflow.utils.pair_score_separation import (
    _load_email_text_catalog,
    _resolve_default_misp_json_path,
)


def _cosine_split_embeddings(
    eid_i: str,
    eid_j: str,
    id_to_emb: dict[str, np.ndarray],
    *,
    subj_dim: int | None = None,
) -> tuple[float, float, float]:
    """Return (full_concat_cosine, subject_cosine, body_cosine)."""
    vi = id_to_emb.get(eid_i)
    vj = id_to_emb.get(eid_j)
    if vi is None or vj is None:
        return float("nan"), float("nan"), float("nan")
    ni = np.linalg.norm(vi)
    nj = np.linalg.norm(vj)
    if ni <= 0 or nj <= 0:
        return float("nan"), float("nan"), float("nan")
    full = float(np.dot(vi, vj) / (ni * nj))
    if subj_dim is None:
        half = len(vi) // 2
        subj_dim = half if half > 0 else len(vi)
    si, bi = vi[:subj_dim], vi[subj_dim:]
    sj, bj = vj[:subj_dim], vj[subj_dim:]
    subj_cos = float("nan")
    body_cos = float("nan")
    ns_i, ns_j = np.linalg.norm(si), np.linalg.norm(sj)
    if ns_i > 0 and ns_j > 0:
        subj_cos = float(np.dot(si, sj) / (ns_i * ns_j))
    nb_i, nb_j = np.linalg.norm(bi), np.linalg.norm(bj)
    if nb_i > 0 and nb_j > 0:
        body_cos = float(np.dot(bi, bj) / (nb_i * nb_j))
    return full, subj_cos, body_cos


def enrich_gt_pair_dataframe(
    df: pd.DataFrame,
    *,
    anchor_run_dir: Path | None,
    pair_training_csv: Path | None,
    embeddings_json: Path | None,
    project_root: Path,
    misp_json: Path | None = None,
    admitting_evidence_dir: Path | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Add path/body/subject/sender/support/time/2-hop columns to a GT pair dataframe.
    """
    meta: dict[str, Any] = {"status": "ok", "warnings": []}
    if df.empty:
        return df.copy(), meta

    out = df.copy()
    pair_csv = pair_training_csv
    if pair_csv is None or not pair_csv.is_file():
        meta["warnings"].append("pair_training_csv missing; anchor/text enrichment limited.")
        pair_csv = project_root / "_missing_pair_training.csv"

    nodes, nodes_meta = _load_anchor_nodes_extended(
        pair_csv=pair_csv,
        project_root=project_root,
        anchor_run_dir=anchor_run_dir,
    )
    meta["anchor_nodes"] = nodes_meta
    if nodes_meta.get("status") != "ok":
        meta["status"] = "partial"
        meta["warnings"].append(f"anchor_nodes: {nodes_meta.get('status')}")

    misp_path = misp_json
    if misp_path is None:
        try:
            misp_path = _resolve_default_misp_json_path(project_root)
        except Exception:
            misp_path = None
    text_catalog: dict[str, dict[str, str]] = {}
    if misp_path is not None and misp_path.is_file():
        try:
            text_catalog, text_meta = _load_email_text_catalog(
                project_root=project_root,
                misp_json_path=misp_path,
                misp_translated_json_path=None,
            )
            meta["text_catalog"] = text_meta
        except Exception as exc:
            meta["warnings"].append(f"text_catalog: {exc}")

    id_to_emb: dict[str, np.ndarray] = {}
    subj_dim: int | None = None
    if embeddings_json is not None and embeddings_json.is_file():
        id_to_emb = _load_embeddings(embeddings_json)
        sample = next(iter(id_to_emb.values()), None)
        if sample is not None:
            subj_dim = len(sample) // 2

    df_maps = _build_artifact_df_maps(nodes) if nodes else {}
    n_docs = max(1, len(nodes))
    body_rare = _build_body_rare_token_set(text_catalog) if text_catalog else set()

    feature_rows: list[dict[str, Any]] = []
    pair_iter = tqdm(
        out.iterrows(),
        total=len(out),
        desc="Enriching pair features",
        unit="pair",
        leave=False,
    )
    for _, r in pair_iter:
        ei = str(r["email_i"])
        ej = str(r["email_j"])
        text_i = text_catalog.get(ei, {})
        text_j = text_catalog.get(ej, {})
        feats = _compute_pair_features_row(
            ei=ei,
            ej=ej,
            row=r,
            nodes=nodes,
            text_i=text_i,
            text_j=text_j,
            df_maps=df_maps,
            body_rare_tokens=body_rare,
            n_docs=n_docs,
        )
        _, subj_c, body_c = _cosine_split_embeddings(ei, ej, id_to_emb, subj_dim=subj_dim)
        feats["subject_cosine"] = subj_c
        feats["body_cosine"] = body_c
        if "semantic_cosine" in out.columns and np.isfinite(r.get("semantic_cosine", np.nan)):
            feats.setdefault("semantic_cosine_max", r.get("semantic_cosine"))
        feature_rows.append(feats)

    feat_df = pd.DataFrame(feature_rows)
    overlap_cols = [c for c in feat_df.columns if c in out.columns]
    if overlap_cols:
        meta["enrichment_columns_skipped_overlap"] = overlap_cols
        feat_df = feat_df.drop(columns=overlap_cols, errors="ignore")
    out = pd.concat([out.reset_index(drop=True), feat_df.reset_index(drop=True)], axis=1)
    out = out.loc[:, ~out.columns.duplicated(keep="last")]

    # HTML fingerprint direct share
    if "shared_html_fp" not in out.columns:
        out["shared_html_fp"] = 0
    sh = out["shared_html_fp"]
    if isinstance(sh, pd.DataFrame):
        sh = sh.iloc[:, 0]
    out["direct_shared_html_fp"] = sh.fillna(0).astype(int)

    # Alias for rule names
    out["shared_nontrivial_stem"] = out.get(
        "shared_stem_nontrivial", out.get("has_shared_stem", 0)
    )

    # 2-hop channel flags (optional admitting evidence)
    evidence_index: dict[tuple[str, str], list[dict[str, Any]]] | None = None
    if admitting_evidence_dir is not None and admitting_evidence_dir.is_dir():
        try:
            from seed_candidate_workflow.utils.pair_inspection_admitting_evidence import (
                load_admitting_evidence_index,
            )

            evidence_index, ev_meta = load_admitting_evidence_index(
                admitting_evidence_dir, project_root=project_root
            )
            meta["admitting_evidence"] = ev_meta
        except Exception as exc:
            meta["warnings"].append(f"admitting_evidence: {exc}")

    if evidence_index:
        chan_df = attach_twohop_channel_columns(out, evidence_index=evidence_index)
        for col in chan_df.columns:
            if col.startswith("twohop_via_") or col in ("twohop_channels", "twohop_primary_channel"):
                out[col] = chan_df[col].values
        meta["twohop_channel_source"] = "admitting_evidence"
    else:
        meta["twohop_channel_source"] = "approximation_where_noted"
        if "from_2hop" in out.columns:
            approx = (
                out["from_2hop"].fillna(False).astype(bool)
                & out["shared_html_fp"].fillna(0).astype(bool)
            )
            out["twohop_via_html_fp"] = approx
        meta["warnings"].append(
            "twohop_via_* channels use from_2hop∧shared_html_fp approximation except html_fp; "
            "pass admitting_evidence_dir for exact channel attribution."
        )

    # Quantile cutoffs for rarity-weighted support (for catalog thresholds)
    rw = pd.to_numeric(out.get("rarity_weighted_support_sum"), errors="coerce")
    rw_fin = rw[np.isfinite(rw)]
    if len(rw_fin) > 10:
        meta["rarity_weighted_support_quantiles"] = {
            "p50": float(np.quantile(rw_fin, 0.50)),
            "p75": float(np.quantile(rw_fin, 0.75)),
            "p90": float(np.quantile(rw_fin, 0.90)),
        }

    meta["n_rows"] = int(len(out))
    meta["n_feature_columns_added"] = int(len(feat_df.columns))
    return out, meta
