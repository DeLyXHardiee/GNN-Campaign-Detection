from __future__ import annotations

from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from seed_candidate_workflow.utils import graph_structure_helpers as gh
from seed_candidate_workflow.utils import raw_gnn_notebook as rn


def load_raw_hdbscan_params(raw_sweep_csv: Path) -> tuple[int, int | None]:
    if not raw_sweep_csv.exists():
        return 2, None
    df = pd.read_csv(raw_sweep_csv)
    if df.empty:
        return 2, None
    row = df.sort_values("completeness", ascending=False).iloc[0]
    mcs = int(row.get("min_cluster_size", 2))
    ms = row.get("min_samples")
    ms_val = None if pd.isna(ms) else int(ms)
    return mcs, ms_val


def build_raw_predictions(
    id_to_emb_raw: dict[str, np.ndarray],
    label_map: dict[str, Any],
    *,
    min_cluster_size: int,
    min_samples: int | None,
) -> tuple[list[str], np.ndarray, dict[str, int], dict[str, float]]:
    cluster_ids = sorted(set(map(str, label_map.keys())) & set(id_to_emb_raw.keys()))
    emb_map = {eid: id_to_emb_raw[eid] for eid in cluster_ids}
    sorted_ids, labels = rn.run_hdbscan_get_labels(
        emb_map,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
    )
    pred_map = rn.eid_label_map(sorted_ids, labels)
    ext = rn.external_scores_subset(sorted_ids, labels, label_map)
    return sorted_ids, labels, pred_map, ext


def campaign_fragmentation_df(
    campaign_to_members: dict[Any, list[str]],
    pred_map: dict[str, int],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for cid, members in campaign_to_members.items():
        m = [str(x) for x in members]
        assigned = [eid for eid in m if eid in pred_map]
        labs = [int(pred_map[eid]) for eid in assigned if int(pred_map[eid]) != -1]
        ct = Counter(labs)
        dominant = int(max(ct.values())) if ct else 0
        num_clusters = int(len(ct))
        dominant_fraction = float(dominant / max(1, len(m)))
        rows.append(
            {
                "campaign_id": cid,
                "campaign_size": int(len(m)),
                "n_members_with_prediction": int(len(assigned)),
                "n_members_non_noise": int(sum(1 for eid in assigned if int(pred_map[eid]) != -1)),
                "num_pred_clusters": num_clusters,
                "dominant_cluster_size": dominant,
                "dominant_fraction": dominant_fraction,
                "fragmentation_score": float(1.0 - dominant_fraction),
                "extra_clusters_beyond_dominant": int(max(num_clusters - 1, 0)),
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values(
        ["fragmentation_score", "campaign_size", "num_pred_clusters"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def campaign_split_counts(
    campaign_id: Any,
    campaign_to_members: dict[Any, list[str]],
    pred_map: dict[str, int],
) -> pd.DataFrame:
    m = [str(x) for x in campaign_to_members.get(campaign_id, [])]
    labs = [int(pred_map[eid]) for eid in m if eid in pred_map]
    ct = Counter(labs)
    rows = []
    for lab, n in ct.most_common():
        rows.append(
            {
                "campaign_id": campaign_id,
                "pred_cluster": int(lab),
                "n_members": int(n),
                "fraction_of_campaign": float(n / max(1, len(m))),
            }
        )
    return pd.DataFrame(rows)


def _pair_share_any(email_sets: dict[str, list[set[int]]], i: int, j: int, channels: Iterable[str]) -> bool:
    for ch in channels:
        if ch not in email_sets:
            continue
        if email_sets[ch][i] & email_sets[ch][j]:
            return True
    return False


def subgroup_bridge_tables(
    campaign_id: Any,
    campaign_to_members: dict[Any, list[str]],
    pred_map: dict[str, int],
    eid_to_row: dict[str, int],
    email_sets: dict[str, list[set[int]]],
    channels: list[str],
    id_to_emb_raw: dict[str, np.ndarray],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    members = [str(x) for x in campaign_to_members.get(campaign_id, [])]
    groups: dict[int, list[str]] = {}
    for eid in members:
        if eid not in pred_map:
            continue
        lab = int(pred_map[eid])
        if lab == -1:
            continue
        groups.setdefault(lab, []).append(eid)

    within_rows: list[dict[str, Any]] = []
    across_rows: list[dict[str, Any]] = []

    for lab, eids in sorted(groups.items(), key=lambda t: (len(t[1]), t[0]), reverse=True):
        idxs = [eid_to_row[e] for e in eids if e in eid_to_row]
        pair_list = list(combinations(idxs, 2))
        if not pair_list:
            continue
        rec: dict[str, Any] = {
            "campaign_id": campaign_id,
            "subgroup": int(lab),
            "subgroup_size": int(len(eids)),
            "n_pairs": int(len(pair_list)),
        }
        cos_vals = []
        for i, j in pair_list:
            ei = members[0]  # placeholder not used
            _ = ei
            # map rows back to external ids via inverse lookup for cosine
        inv = {v: k for k, v in eid_to_row.items()}
        for i, j in pair_list:
            a, b = inv.get(i), inv.get(j)
            if a in id_to_emb_raw and b in id_to_emb_raw:
                va = id_to_emb_raw[a]
                vb = id_to_emb_raw[b]
                na = np.linalg.norm(va) + 1e-12
                nb = np.linalg.norm(vb) + 1e-12
                cos_vals.append(float(np.dot(va, vb) / (na * nb)))
        rec["avg_raw_cos_within_subgroup"] = float(np.mean(cos_vals)) if cos_vals else np.nan
        for ch in channels:
            hit = sum(
                1
                for i, j in pair_list
                if (ch in email_sets and len(email_sets[ch][i] & email_sets[ch][j]) > 0)
            )
            rec[f"within_share_{ch}"] = float(hit / len(pair_list))
        within_rows.append(rec)

    inv = {v: k for k, v in eid_to_row.items()}
    for (la, ea), (lb, eb) in combinations(sorted(groups.items()), 2):
        ia = [eid_to_row[e] for e in ea if e in eid_to_row]
        ib = [eid_to_row[e] for e in eb if e in eid_to_row]
        cross_pairs = [(i, j) for i in ia for j in ib]
        if not cross_pairs:
            continue
        rec = {
            "campaign_id": campaign_id,
            "subgroup_a": int(la),
            "subgroup_b": int(lb),
            "size_a": int(len(ea)),
            "size_b": int(len(eb)),
            "cross_pair_count": int(len(cross_pairs)),
            "any_graph_bridge": bool(
                any(_pair_share_any(email_sets, i, j, channels) for i, j in cross_pairs)
            ),
        }
        cos_vals = []
        for i, j in cross_pairs:
            a, b = inv.get(i), inv.get(j)
            if a in id_to_emb_raw and b in id_to_emb_raw:
                va = id_to_emb_raw[a]
                vb = id_to_emb_raw[b]
                na = np.linalg.norm(va) + 1e-12
                nb = np.linalg.norm(vb) + 1e-12
                cos_vals.append(float(np.dot(va, vb) / (na * nb)))
        rec["avg_raw_cos_across_subgroups"] = float(np.mean(cos_vals)) if cos_vals else np.nan
        for ch in channels:
            hit = sum(
                1
                for i, j in cross_pairs
                if (ch in email_sets and len(email_sets[ch][i] & email_sets[ch][j]) > 0)
            )
            rec[f"cross_share_{ch}"] = float(hit / len(cross_pairs))
            rec[f"any_cross_share_{ch}"] = bool(hit > 0)
        across_rows.append(rec)

    return pd.DataFrame(within_rows), pd.DataFrame(across_rows)


def aggregate_bridgeability(
    top_campaign_ids: list[Any],
    campaign_to_members: dict[Any, list[str]],
    pred_map: dict[str, int],
    eid_to_row: dict[str, int],
    email_sets: dict[str, list[set[int]]],
    channels: list[str],
    id_to_emb_raw: dict[str, np.ndarray],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_across = []
    for cid in top_campaign_ids:
        _w, a = subgroup_bridge_tables(
            cid,
            campaign_to_members,
            pred_map,
            eid_to_row,
            email_sets,
            channels,
            id_to_emb_raw,
        )
        if not a.empty:
            all_across.append(a)
    if not all_across:
        return pd.DataFrame(), pd.DataFrame()
    across = pd.concat(all_across, ignore_index=True)
    campaign_level = []
    for cid, g in across.groupby("campaign_id"):
        rec = {"campaign_id": cid, "n_subgroup_pairs": int(len(g))}
        for ch in channels:
            col = f"any_cross_share_{ch}"
            rec[f"campaign_has_cross_{ch}"] = bool(g[col].any()) if col in g.columns else False
        rec["campaign_has_any_cross_bridge"] = bool(g["any_graph_bridge"].any())
        campaign_level.append(rec)
    camp_df = pd.DataFrame(campaign_level)
    agg_rows = []
    denom = max(1, len(camp_df))
    for ch in channels:
        c = f"campaign_has_cross_{ch}"
        agg_rows.append({"channel": ch, "fraction_campaigns_with_cross_bridge": float(camp_df[c].sum() / denom)})
    agg_rows.append(
        {
            "channel": "any_channel",
            "fraction_campaigns_with_cross_bridge": float(camp_df["campaign_has_any_cross_bridge"].sum() / denom),
        }
    )
    agg = pd.DataFrame(agg_rows)
    return camp_df, agg

