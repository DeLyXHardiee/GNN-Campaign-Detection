"""
Join pair_training_dataset.csv rows to duplicate-cluster membership and
aggregate duplicate / easy-edge pressure metrics.
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from seed_candidate_workflow.utils import graph_structure_helpers as gh


SIG_TYPES = (
    "strict_full_email",
    "content_subject_body",
    "near_template_subject_body_sender",
)


def _truthy(v: Any) -> bool:
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return False
    s = str(v).strip().lower()
    return s in ("1", "true", "yes", "t")


def _component_bucket(row: pd.Series) -> str:
    cross = _truthy(row.get("cross_seed_component_flag", False))
    same = _truthy(row.get("same_seed_component_flag", False))
    if cross:
        return "cross_component"
    if same:
        return "same_component"
    return "component_unknown"


def _cosine_bin(v: Any) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "semantic_cosine_nan"
    try:
        x = float(v)
    except (TypeError, ValueError):
        return "semantic_cosine_nan"
    if x >= 0.9:
        return "semantic_cosine_ge_0.9"
    if x >= 0.7:
        return "semantic_cosine_0.7_0.9"
    if x >= 0.5:
        return "semantic_cosine_0.5_0.7"
    return "semantic_cosine_lt_0.5"


def _load_cluster_maps(
    mem_path: Path,
) -> tuple[dict[str, dict[str, dict[str, Any]]], set[str]]:
    """
    Returns:
      maps[sig_type][external_id] = {cluster_id, group_size, cluster_rank_by_size, signature_hash12}
      all_dup_external_ids: union of external_ids appearing in any duplicate cluster row
    """
    mem = pd.read_parquet(mem_path)
    need = {"external_id", "signature_type", "cluster_id", "group_size"}
    miss = need - set(mem.columns)
    if miss:
        raise ValueError(f"membership parquet missing columns: {sorted(miss)}")

    maps: dict[str, dict[str, dict[str, Any]]] = {s: {} for s in SIG_TYPES}
    all_dup: set[str] = set()
    for sig in SIG_TYPES:
        sub = mem.loc[mem["signature_type"].astype(str) == sig]
        for _, r in sub.iterrows():
            eid = str(r["external_id"]).strip()
            maps[sig][eid] = {
                "cluster_id": str(r["cluster_id"]),
                "group_size": int(r["group_size"]),
                "cluster_rank_by_size": int(r["cluster_rank_by_size"])
                if pd.notna(r.get("cluster_rank_by_size"))
                else None,
                "signature_hash12": str(r["signature_hash12"])
                if pd.notna(r.get("signature_hash12"))
                else "",
            }
            all_dup.add(eid)
    return maps, all_dup


def _load_misp_loaded_ids(path: Path | None) -> set[str] | None:
    if path is None or not path.is_file():
        return None
    t = pd.read_parquet(path)
    if "external_id" not in t.columns:
        raise ValueError("misp_loaded_external_ids parquet must have column external_id")
    return set(t["external_id"].astype(str).str.strip())


def _load_graph_external_ids(meta_path: Path | None) -> set[str] | None:
    if meta_path is None or not meta_path.is_file():
        return None
    meta = gh.load_meta(meta_path)
    return set(gh.email_external_id_list(meta))


def _both_endpoints_in_graph(frame: pd.DataFrame, gids: set[str]) -> pd.Series:
    return frame["email_i"].astype(str).str.strip().isin(gids) & frame["email_j"].astype(str).str.strip().isin(
        gids
    )


def _attach_dup_flags_for_sig(
    df: pd.DataFrame,
    cmap: dict[str, dict[str, Any]],
    *,
    sig: str,
    misp_ids: set[str] | None,
) -> pd.DataFrame:
    out = df.copy()
    ei = out["email_i"].astype(str).str.strip()
    ej = out["email_j"].astype(str).str.strip()

    def _lookup(e: str) -> dict[str, Any] | None:
        return cmap.get(e)

    def _cid(e: str) -> Any:
        d = cmap.get(e)
        return d["cluster_id"] if d else None

    def _gsz(e: str) -> Any:
        d = cmap.get(e)
        return int(d["group_size"]) if d else np.nan

    ci = ei.map(_cid)
    cj = ej.map(_cid)
    gi = ei.map(_gsz)
    gj = ej.map(_gsz)

    out[f"cluster_id_i__{sig}"] = ci
    out[f"cluster_id_j__{sig}"] = cj
    in_i = ci.notna()
    in_j = cj.notna()
    same = in_i & in_j & (ci == cj)
    diff = in_i & in_j & (ci != cj)
    one = (in_i & ~in_j) | (~in_i & in_j)
    neither = ~in_i & ~in_j

    out[f"dup_same_cluster__{sig}"] = same.fillna(False)
    out[f"dup_both_diff_cluster__{sig}"] = diff.fillna(False)
    out[f"dup_one_side_only__{sig}"] = one.fillna(False)
    out[f"dup_neither__{sig}"] = neither.fillna(False)

    unk_i = pd.Series(False, index=out.index)
    unk_j = pd.Series(False, index=out.index)
    if misp_ids is not None:
        unk_i = ~ei.isin(misp_ids)
        unk_j = ~ej.isin(misp_ids)
    out[f"endpoint_not_in_misp_run__i__{sig}"] = unk_i
    out[f"endpoint_not_in_misp_run__j__{sig}"] = unk_j
    out[f"dup_unknown_misp_coverage__{sig}"] = unk_i | unk_j

    out[f"dup_group_size_max_endpoint__{sig}"] = pd.concat(
        [pd.to_numeric(gi, errors="coerce"), pd.to_numeric(gj, errors="coerce")], axis=1
    ).max(axis=1)
    return out


def _edges_potential_in_graph(
    mem: pd.DataFrame,
    sig: str,
    graph_ids: set[str],
) -> tuple[int, int]:
    """
    Sum over clusters of n*(n-1)/2 where n = number of cluster members present in graph_ids.
    Returns (n_clusters_with_n_ge_2, potential_edges).
    """
    sub = mem.loc[mem["signature_type"].astype(str) == sig, ["cluster_id", "external_id"]].copy()
    if sub.empty:
        return 0, 0
    sub["external_id"] = sub["external_id"].astype(str).str.strip()
    sub = sub.loc[sub["external_id"].isin(graph_ids)]
    potential = 0
    n_nonempty = 0
    for _, grp in sub.groupby("cluster_id"):
        n = int(grp["external_id"].nunique())
        if n >= 2:
            n_nonempty += 1
            potential += (n * (n - 1)) // 2
    return n_nonempty, potential


def _stratum_rows(df: pd.DataFrame, sig: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    dsc = f"dup_same_cluster__{sig}"
    if dsc not in df.columns:
        return rows

    def bucket(r: pd.Series) -> str:
        ps = str(r.get("pair_status", "")).lower()
        comp = _component_bucket(r)
        fs = _truthy(r.get("from_semantic", False))
        fc = _truthy(r.get("from_component", False))
        f2 = _truthy(r.get("from_2hop", False))
        fr = _truthy(r.get("from_rare_artifact", False))
        fseed = _truthy(r.get("from_seed", False))
        nsrc = int(fs) + int(fc) + int(f2) + int(fr) + int(fseed)
        sem_comp = fs and fc
        parts = [
            f"pair_status={ps}",
            f"component={comp}",
            f"n_prov_bools={nsrc}",
            f"semantic_and_component={int(sem_comp)}",
            f"from_semantic={int(fs)}",
            f"from_component={int(fc)}",
            f"from_2hop={int(f2)}",
        ]
        return "|".join(parts)

    df = df.copy()
    df["_stratum"] = df.apply(bucket, axis=1)
    for strat, g in df.groupby("_stratum", dropna=False):
        n = int(len(g))
        k = int(g[dsc].sum())
        rows.append(
            {
                "signature_type": sig,
                "stratum": strat,
                "n_rows": n,
                "n_dup_same_cluster": k,
                "frac_dup_same_cluster": float(k / n) if n else 0.0,
            }
        )
    return rows


def _top_clusters_positive(df: pd.DataFrame, sig: str, top_k: int = 20) -> list[dict[str, Any]]:
    col = f"dup_same_cluster__{sig}"
    if col not in df.columns:
        return []
    pos = df.loc[df["pair_status"].astype(str).str.lower() == "positive"].copy()
    pos = pos.loc[pos[col]]
    if pos.empty:
        return []
    ci = f"cluster_id_i__{sig}"
    cj = f"cluster_id_j__{sig}"
    cc = pos[ci].astype(str)
    vc = cc.value_counts()
    out: list[dict[str, Any]] = []
    cum = 0
    total = int(len(pos))
    for rank, (cid, cnt) in enumerate(vc.head(top_k).items(), start=1):
        cum += int(cnt)
        out.append(
            {
                "signature_type": sig,
                "rank": rank,
                "cluster_id": cid,
                "n_positive_pair_rows": int(cnt),
                "cumulative_positive_pair_rows": cum,
                "cumulative_frac_of_positive_dup_same_rows": float(cum / total) if total else 0.0,
            }
        )
    return out


def _email_pair_degree_positive(df: pd.DataFrame, sig: str) -> pd.DataFrame:
    col = f"dup_same_cluster__{sig}"
    pos = df.loc[(df["pair_status"].astype(str).str.lower() == "positive") & df[col]].copy()
    deg: dict[str, int] = defaultdict(int)
    for _, r in pos.iterrows():
        deg[str(r["email_i"]).strip()] += 1
        deg[str(r["email_j"]).strip()] += 1
    return pd.DataFrame([{"external_id": e, "positive_dup_same_pair_row_degree": c} for e, c in deg.items()])


def _cross_tab_semantic_and_shared(df: pd.DataFrame, sig: str) -> list[dict[str, Any]]:
    """dup_same_cluster (strict sig) x cosine bin x shared-attribute proxies."""
    if sig != "strict_full_email":
        return []
    dsc = f"dup_same_cluster__{sig}"
    if dsc not in df.columns:
        return []
    df = df.copy()
    if "semantic_cosine_max" in df.columns:
        df["_cos_bin"] = df["semantic_cosine_max"].map(_cosine_bin)
    else:
        df["_cos_bin"] = "semantic_cosine_nan"
    z = pd.Series(False, index=df.index)
    hs = df["has_shared_stem"].astype(bool) if "has_shared_stem" in df.columns else z
    hu = df["has_shared_url"].astype(bool) if "has_shared_url" in df.columns else z
    ha = df["has_shared_attachment"].astype(bool) if "has_shared_attachment" in df.columns else z
    if "shared_sender_count" in df.columns:
        try:
            ssc = pd.to_numeric(df["shared_sender_count"], errors="coerce").fillna(0) > 0
        except Exception:
            ssc = z.copy()
    else:
        ssc = z.copy()
    df["_shared_any"] = hs | hu | ha | ssc.astype(bool)

    rows: list[dict[str, Any]] = []
    for (dup_s, cbin, sh), g in df.groupby([dsc, "_cos_bin", "_shared_any"]):
        rows.append(
            {
                "dup_same_cluster_strict": bool(dup_s),
                "semantic_cosine_bin": str(cbin),
                "shared_proxy_any": bool(sh),
                "n_rows": int(len(g)),
                "n_positive": int((g["pair_status"].astype(str).str.lower() == "positive").sum()),
            }
        )
    return rows


def run_pair_duplicate_pressure(
    *,
    pair_csv: Path,
    membership_parquet: Path,
    out_dir: Path,
    graph_meta_json: Path | None = None,
    misp_loaded_ids_parquet: Path | None = None,
    training_rows_only: bool = True,
    apply_split: bool = False,
    pair_val_ratio: float = 0.1,
    pair_test_ratio: float = 0.1,
    pair_split_seed: int = 42,
    write_augmented_parquet: bool = False,
) -> dict[str, Any]:
    pair_csv = pair_csv.expanduser().resolve()
    membership_parquet = membership_parquet.expanduser().resolve()
    out_dir = out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(pair_csv)
    if training_rows_only:
        gi = pd.to_numeric(df.get("graph_email_idx_i"), errors="coerce")
        gj = pd.to_numeric(df.get("graph_email_idx_j"), errors="coerce")
        if gi is None or gj is None:
            raise ValueError("pair CSV missing graph_email_idx_* columns for training_rows_only")
        ok = gi.notna() & gj.notna()
        df = df.loc[ok].copy()

    gpath = Path(graph_meta_json).expanduser().resolve() if graph_meta_json else None
    graph_ids = _load_graph_external_ids(gpath)
    mip = Path(misp_loaded_ids_parquet).expanduser().resolve() if misp_loaded_ids_parquet else None
    misp_ids = _load_misp_loaded_ids(mip)

    maps, _ = _load_cluster_maps(membership_parquet)
    mem_full = pd.read_parquet(membership_parquet)

    enriched_frames: dict[str, pd.DataFrame] = {}
    for sig in SIG_TYPES:
        enriched_frames[sig] = _attach_dup_flags_for_sig(df, maps[sig], sig=sig, misp_ids=misp_ids)

    work = enriched_frames["strict_full_email"].copy()

    summary: dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "pair_csv": str(pair_csv),
        "membership_parquet": str(membership_parquet),
        "out_dir": str(out_dir),
        "training_rows_only": bool(training_rows_only),
        "n_pair_rows_analyzed": int(len(work)),
        "graph_meta_json": str(graph_meta_json) if graph_meta_json else None,
        "n_graph_emails": int(len(graph_ids)) if graph_ids is not None else None,
        "misp_loaded_ids_parquet": str(misp_loaded_ids_parquet) if misp_loaded_ids_parquet else None,
        "n_misp_loaded_ids": int(len(misp_ids)) if misp_ids is not None else None,
        "per_signature_type": {},
        "potential_vs_realized": {},
        "split_projection": None,
    }

    by_stratum_all: list[dict[str, Any]] = []
    top_clusters_all: list[dict[str, Any]] = []

    for sig in SIG_TYPES:
        w = enriched_frames[sig]
        dsc = f"dup_same_cluster__{sig}"
        per = {
            "n_dup_same_cluster_rows": int(w[dsc].sum()),
            "frac_dup_same_cluster_rows": float(w[dsc].mean()) if len(w) else 0.0,
            "n_dup_both_diff_cluster": int(w[f"dup_both_diff_cluster__{sig}"].sum()),
            "n_dup_one_side_only": int(w[f"dup_one_side_only__{sig}"].sum()),
            "n_dup_neither": int(w[f"dup_neither__{sig}"].sum()),
        }
        if misp_ids is not None:
            per["n_rows_any_endpoint_not_in_misp_run"] = int(w[f"dup_unknown_misp_coverage__{sig}"].sum())
        summary["per_signature_type"][sig] = per

        by_stratum_all.extend(_stratum_rows(w, sig))
        top_clusters_all.extend(_top_clusters_positive(w, sig, top_k=20))

    for sig in SIG_TYPES:
        w = enriched_frames[sig]
        dsc = f"dup_same_cluster__{sig}"
        if graph_ids is not None:
            n_cl, pot = _edges_potential_in_graph(mem_full, sig, graph_ids)
            mask = w[dsc] & _both_endpoints_in_graph(w, graph_ids)
            realized = int(mask.sum())
            summary["potential_vs_realized"][sig] = {
                "n_duplicate_clusters_intersecting_graph": n_cl,
                "E_duplicate_potential_edges_in_graph": int(pot),
                "E_pair_rows_dup_same_cluster_both_endpoints_in_graph": realized,
                "realized_over_potential": float(realized / pot) if pot else None,
            }
        else:
            summary["potential_vs_realized"][sig] = {
                "note": "graph_meta_json not provided; skipped potential_vs_realized",
            }

    deg_df = _email_pair_degree_positive(work, "strict_full_email")
    strict_mem = mem_full.loc[mem_full["signature_type"].astype(str) == "strict_full_email", ["external_id", "group_size"]].copy()
    if not strict_mem.empty:
        strict_mem["external_id"] = strict_mem["external_id"].astype(str).str.strip()
        mx = strict_mem.groupby("external_id", as_index=False)["group_size"].max().rename(
            columns={"group_size": "max_strict_duplicate_group_size"}
        )
        deg_df = deg_df.merge(mx, on="external_id", how="left")
    else:
        deg_df["max_strict_duplicate_group_size"] = np.nan
    p_deg = out_dir / "pair_duplicate_positive_email_degree_strict.csv"
    deg_df.sort_values("positive_dup_same_pair_row_degree", ascending=False).to_csv(p_deg, index=False)
    summary["positive_dup_same_email_degree_csv"] = str(p_deg)

    pair_status_block: dict[str, Any] = {}
    for sig in SIG_TYPES:
        w = enriched_frames[sig]
        dsc = f"dup_same_cluster__{sig}"
        pair_status_block[sig] = {}
        for ps in sorted(w["pair_status"].astype(str).str.lower().unique()):
            sub = w.loc[w["pair_status"].astype(str).str.lower() == ps]
            n = int(len(sub))
            if n == 0:
                continue
            k = int(sub[dsc].sum())
            pair_status_block[sig][ps] = {
                "n_rows": n,
                "n_dup_same_cluster": k,
                "frac_dup_same_cluster": float(k / n),
            }
    summary["by_pair_status"] = pair_status_block

    xtab = _cross_tab_semantic_and_shared(work, "strict_full_email")
    p_xtab = out_dir / "pair_duplicate_cross_tab_strict_semantic_shared.csv"
    pd.DataFrame(xtab).to_csv(p_xtab, index=False)
    summary["cross_tab_strict_semantic_shared_csv"] = str(p_xtab)

    p_strat = out_dir / "pair_duplicate_pressure_by_stratum.csv"
    pd.DataFrame(by_stratum_all).to_csv(p_strat, index=False)

    p_top = out_dir / "pair_duplicate_pressure_top_clusters.csv"
    pd.DataFrame(top_clusters_all).to_csv(p_top, index=False)

    if apply_split:
        n = len(work)
        rng = np.random.default_rng(int(pair_split_seed))
        perm = rng.permutation(n)
        n_test = int(np.floor(n * float(pair_test_ratio)))
        n_val = int(np.floor(n * float(pair_val_ratio)))
        i_test = perm[:n_test]
        i_val = perm[n_test : n_test + n_val]
        i_train = perm[n_test + n_val :]
        split_slices = {"train": i_train, "val": i_val, "test": i_test}
        split_block: dict[str, Any] = {
            "pair_val_ratio": pair_val_ratio,
            "pair_test_ratio": pair_test_ratio,
            "pair_split_seed": pair_split_seed,
            "splits": {},
        }
        for name, sl in split_slices.items():
            split_block["splits"][name] = {
                "n_rows": int(len(sl)),
                "per_signature_type": {},
            }
            for sig in SIG_TYPES:
                wsig = enriched_frames[sig].iloc[sl]
                dsc = f"dup_same_cluster__{sig}"
                nn = int(len(wsig))
                split_block["splits"][name]["per_signature_type"][sig] = {
                    "frac_dup_same_cluster": float(wsig[dsc].mean()) if nn else 0.0,
                    "n_dup_same_cluster": int(wsig[dsc].sum()),
                }
        summary["split_projection"] = split_block
        p_split = out_dir / "pair_duplicate_pressure_split_summary.json"
        p_split.write_text(json.dumps(split_block, indent=2), encoding="utf-8")
        summary["split_summary_json"] = str(p_split)

    p_sum = out_dir / "pair_duplicate_pressure_summary.json"
    p_sum.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    if write_augmented_parquet:
        aug = work.copy()
        for sig in SIG_TYPES:
            if sig == "strict_full_email":
                continue
            w2 = enriched_frames[sig]
            for c in w2.columns:
                if (c.startswith("cluster_id_") or c.startswith("dup_")) and f"__{sig}" in c:
                    aug[c] = w2[c].values
        p_aug = out_dir / "pair_duplicate_labeled_rows.parquet"
        aug.to_parquet(p_aug, index=False)
        summary["augmented_parquet"] = str(p_aug)

    summary["artifacts"] = {
        "pair_duplicate_pressure_summary_json": str(p_sum),
        "pair_duplicate_pressure_by_stratum_csv": str(p_strat),
        "pair_duplicate_pressure_top_clusters_csv": str(p_top),
    }
    return summary
