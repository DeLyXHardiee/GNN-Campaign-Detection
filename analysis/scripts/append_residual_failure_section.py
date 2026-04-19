"""Append residual failure analysis section to semantic_shard_oracle_headroom_analysis.ipynb."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NB = ROOT / "analysis" / "semantic_shard_oracle_headroom_analysis.ipynb"


def _src(s: str) -> list[str]:
    if not s.endswith("\n"):
        s += "\n"
    return [ln if ln.endswith("\n") else ln + "\n" for ln in s.splitlines() if True]


def main() -> None:
    nb = json.loads(NB.read_text(encoding="utf-8"))

    cells = [
        {
            "cell_type": "markdown",
            "id": "residual-md-title",
            "metadata": {},
            "source": _src(
                """# Residual failure analysis: what problem is left for a GNN to solve?

The **Step-2 shard graph baseline** (current weighted edges) is already strong: shards are very pure on GT, so most loss is **completeness / recall**—getting the right campaign pieces connected.

This section estimates **where recall is still lost**:

* Stage-1 **noise / singleton** shards vs **normal** shards  
* **Missing same-campaign links** (never in the candidate table vs **present but filtered** by weight)  
* **Near-miss semantic** pairs if cosine thresholds were relaxed  
* Whether off-component shards could **attach** to the dominant GT-connected component (semantic / direct edges)

**GT is used only for evaluation**, not for training. Outputs are diagnostics to choose among: better shard creation, edge rescoring, missing-edge completion near semantic thresholds, or **node-to-community** attachment.

**How to read outputs:** small summary tables + a few plots with explicit titles/axes; larger tables are written to CSV under `OUT_DIR`."""
            ),
        },
        {
            "cell_type": "markdown",
            "id": "residual-md-params",
            "metadata": {},
            "source": _src(
                """## Setup (residual section)

Reads `STEP2` graph artifacts from variables defined earlier (`nodes_df`, `edges_df`, `assignments_df`, `shard_summary`, …).

* **`BASELINE_MIN_EDGE_WEIGHT`**: edges with `edge_weight >=` this value are **active** in the connectivity model (default `0.0` = every row in `semantic_shard_step2_edges_weighted.csv` counts).  
* **`OPERATING_MIN_EDGE_WEIGHT`**: stricter cutoff used to classify **existing_edge_below_operating_threshold** in the missing-link breakdown (loaded from `semantic_shard_step2_graph_summary.json` when available, else falls back to the **median** edge weight, or `0` if missing).  
* **Singleton / noise shard**: `size <= 1` in `nodes_df` (Stage-1 shard size)."""
            ),
        },
        {
            "cell_type": "code",
            "id": "residual-code-setup",
            "metadata": {},
            "outputs": [],
            "source": _src(
                """from analysis.utils.semantic_shard_residual_failure_helpers import (
    active_edge_keys,
    campaign_fracture_summary_stats,
    classify_disconnected_same_campaign_pairs,
    community_attachment_summary,
    compute_campaign_fracture_table,
    cosine_threshold_new_edge_stats,
    edge_weight_retention_table,
    fractured_campaign_noise_summary,
    fractured_campaigns_at_edge_threshold,
    load_centroid_matrix,
    load_step2_config,
    off_main_shards_and_similarity,
    plot_attachment_stacked,
    plot_campaign_component_bar,
    plot_horizontal_category_bars,
    plot_largest_component_hist,
    plot_max_cos_histogram,
)

_step2_cfg = load_step2_config(STEP2_DIR)
_g = (_step2_cfg.get("graph_config_summary") or {}) if isinstance(_step2_cfg, dict) else {}
BASELINE_MIN_EDGE_WEIGHT = float(_g.get("min_edge_weight_active", _g.get("min_edge_weight", 0.0)) or 0.0)
# Stricter "operating" cutoff for diagnostics: summary JSON, else weight median
if edges_df is not None and not edges_df.empty and "edge_weight" in edges_df.columns:
    _med_w = float(edges_df["edge_weight"].astype(float).median())
else:
    _med_w = 0.0
OPERATING_MIN_EDGE_WEIGHT = float(_g.get("operating_min_edge_weight", _med_w))
SEMANTIC_CANDIDATE_MIN_COS = float(_g.get("semantic_min_cos", _g.get("semantic_min_cosine", 0.72)))

centroid_mat, shard_to_idx = load_centroid_matrix(STEP2_DIR, nodes_df)

display(
    Markdown(
        "**Resolved parameters:** "
        f"`BASELINE_MIN_EDGE_WEIGHT={BASELINE_MIN_EDGE_WEIGHT}`, "
        f"`OPERATING_MIN_EDGE_WEIGHT={OPERATING_MIN_EDGE_WEIGHT}`, "
        f"semantic candidate build cos ≈ `{SEMANTIC_CANDIDATE_MIN_COS}`, "
        f"centroids loaded: `{centroid_mat is not None}`"
    )
)"""
            ),
        },
        {
            "cell_type": "markdown",
            "id": "residual-md-1",
            "metadata": {},
            "source": _src(
                """## 1. Campaign fracture inventory

**Interpretation:** For each GT campaign, we connect shards using **active** edges (`edge_weight >= BASELINE_MIN_EDGE_WEIGHT`). If more than one connected component appears among that campaign's shards, the campaign is **fractured** in the current graph.

**Plot 1 (left):** how many campaigns have 1, 2, … components (bucketed tail if many).  
**Plot 2:** among fractured campaigns, how large is the **main** component as a fraction of shards (heavy left = one big piece + small tails; flat = severe splits)."""
            ),
        },
        {
            "cell_type": "code",
            "id": "residual-code-1",
            "metadata": {},
            "outputs": [],
            "source": _src(
                """fracture_df = compute_campaign_fracture_table(
    assignments_df,
    gt_label_map,
    edges_df,
    nodes_df,
    min_edge_weight=BASELINE_MIN_EDGE_WEIGHT,
    weight_col="edge_weight",
)
fracture_summary = campaign_fracture_summary_stats(fracture_df)
fracture_df.to_csv(OUT_DIR / "campaign_fracture_detail.csv", index=False)
fracture_summary.to_csv(OUT_DIR / "campaign_fracture_summary.csv", index=False)
display(fracture_summary)

fig = plot_campaign_component_bar(
    fracture_df, title="GT campaigns by number of shard-graph components (within campaign)"
)
fig.savefig(OUT_DIR / "residual_campaign_component_counts.png", dpi=120, bbox_inches="tight")
plt.show()
plt.close(fig)

frac_only = fracture_df[fracture_df["n_graph_components_in_campaign"] > 1]
if len(frac_only):
    fig2 = plot_largest_component_hist(
        frac_only, title="Largest component size (fraction of shards) — fractured campaigns only"
    )
    fig2.savefig(OUT_DIR / "residual_largest_component_fraction_hist.png", dpi=120, bbox_inches="tight")
    plt.show()
    plt.close(fig2)
else:
    display(Markdown("_No fractured campaigns at this baseline — histogram skipped._"))"""
            ),
        },
        {
            "cell_type": "markdown",
            "id": "residual-md-2",
            "metadata": {},
            "source": _src(
                """## 2. Noise / singleton contribution (fractured campaigns)

**Interpretation:** Among emails in **fractured** GT campaigns, what fraction sit in **singleton** shards (`size <= 1`)?

**Plots:** (1) two-bar counts normal vs noise shard emails; (2) stacked counts of fractured campaigns by noise involvement."""
            ),
        },
        {
            "cell_type": "code",
            "id": "residual-code-2",
            "metadata": {},
            "outputs": [],
            "source": _src(
                """noise_summary, noise_buckets = fractured_campaign_noise_summary(
    assignments_df, gt_label_map, fracture_df, nodes_df
)
noise_summary.to_csv(OUT_DIR / "noise_contribution_summary.csv", index=False)
display(noise_summary)

_gt = {str(k): v for k, v in gt_label_map.items()}
_frid = set(fracture_df.loc[fracture_df["n_graph_components_in_campaign"] > 1, "campaign_id"])
_ad = assignments_df.copy()
_ad["external_id"] = _ad["external_id"].astype(str)
_ad["shard_id"] = _ad["shard_id"].astype(str)
_ad = _ad[_ad["external_id"].isin(_gt)].copy()
_ad["campaign_id"] = _ad["external_id"].map(_gt)
_ad = _ad[_ad["campaign_id"].isin(_frid)].copy()
_sz = _ad["shard_id"].map(lambda s: int(dict(zip(nodes_df["shard_id"].astype(str), nodes_df["size"], strict=False)).get(str(s), 999)))
_ad["is_noise"] = _sz <= 1

fig, ax = plt.subplots(figsize=(5, 3.5))
counts = [int((~_ad["is_noise"]).sum()), int(_ad["is_noise"].sum())]
ax.bar(["emails in normal shards", "emails in noise/singleton shards"], counts, color=["#4c72b0", "#dd8452"])
ax.set_ylabel("Count (GT emails, fractured campaigns)")
ax.set_title("Fractured campaigns — email placement by shard size")
plt.tight_layout()
fig.savefig(OUT_DIR / "residual_noise_email_bars.png", dpi=120, bbox_inches="tight")
plt.show()
plt.close(fig)

if not noise_buckets.empty:
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.bar(noise_buckets["bucket"], noise_buckets["n_campaigns"], color="#8172b2")
    ax.set_ylabel("Fractured campaigns")
    ax.set_title("Fractured campaigns by noise involvement")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    fig.savefig(OUT_DIR / "residual_noise_stacked_groups.png", dpi=120, bbox_inches="tight")
    plt.show()
    plt.close(fig)"""
            ),
        },
        {
            "cell_type": "markdown",
            "id": "residual-md-3",
            "metadata": {},
            "source": _src(
                """## 3. Missing-link opportunity (same-campaign, different component)

**Interpretation:** Take each **unordered shard pair** that (i) belongs to the same GT campaign and (ii) lies in **different** baseline connected components. Classify:

* **existing_edge_below_operating_threshold** — row exists in `edges_df` but `edge_weight < OPERATING_MIN_EDGE_WEIGHT`  
* **no_edge_but_high_semantic_candidate** — no edge row; centroid cosine ≥ 0.85 (configurable)  
* **no_edge_but_some_direct_infra_support** — no edge row; any parsed infra-set overlap on `nodes_df`  
* **no_edge_no_direct_support** — else  
* **pair_anomaly_high_weight_but_disconnected** — should be rare (edge active in model but still disconnected ⇒ check weights / keys)

Large pair list: `missing_link_opportunity_pairs_detail.csv`."""
            ),
        },
        {
            "cell_type": "code",
            "id": "residual-code-3",
            "metadata": {},
            "outputs": [],
            "source": _src(
                """NEAR_MISS_PAIR_COS = 0.85
pair_detail, miss_summary = classify_disconnected_same_campaign_pairs(
    fracture_df,
    assignments_df,
    gt_label_map,
    edges_df,
    nodes_df,
    active_min_edge_weight=OPERATING_MIN_EDGE_WEIGHT,
    near_miss_cos=NEAR_MISS_PAIR_COS,
    weight_col="edge_weight",
    centroid_mat=centroid_mat,
    shard_to_idx=shard_to_idx,
)
miss_summary.to_csv(OUT_DIR / "missing_link_opportunity_breakdown.csv", index=False)
pair_detail.to_csv(OUT_DIR / "missing_link_opportunity_pairs_detail.csv", index=False)
display(miss_summary)

if not miss_summary.empty:
    fig = plot_horizontal_category_bars(
        miss_summary, title="Same-campaign disconnected shard pairs — category counts"
    )
    fig.savefig(OUT_DIR / "residual_missing_link_horizontal.png", dpi=120, bbox_inches="tight")
    plt.show()
    plt.close(fig)
else:
    display(Markdown("_No disconnected same-campaign pairs to classify._"))"""
            ),
        },
        {
            "cell_type": "markdown",
            "id": "residual-md-4",
            "metadata": {},
            "source": _src(
                """## 4. Cosine-threshold sweep (new pairwise opportunities)

**Interpretation:** For cosines τ in `{0.90, 0.85, 0.80}` and the **semantic candidate** floor from Step-2 (`SEMANTIC_CANDIDATE_MIN_COS`), count **new** shard pairs with cosine ≥ τ that are **not** already in the **baseline active** edge set. Bars/lines show how many new edges are GT-same vs cross vs ambiguous.

**Column `n_fractured_campaigns_with_new_same_bridge`:** fractured campaigns where at least one **new same-GT** pair would connect two different baseline components."""
            ),
        },
        {
            "cell_type": "code",
            "id": "residual-code-4",
            "metadata": {},
            "outputs": [],
            "source": _src(
                """if centroid_mat is None or not len(shard_to_idx):
    display(Markdown("_Skip cosine sweep: centroids not aligned with `nodes_df`._"))
    cos_sweep_df = pd.DataFrame()
else:
    _base_keys = active_edge_keys(
        edges_df, min_edge_weight=BASELINE_MIN_EDGE_WEIGHT, weight_col="edge_weight"
    )
    _taus = sorted(set([0.90, 0.85, 0.80, float(SEMANTIC_CANDIDATE_MIN_COS)]))
    cos_sweep_df = cosine_threshold_new_edge_stats(
        edges_df,
        centroid_mat,
        shard_to_idx,
        shard_summary,
        thresholds=_taus,
        baseline_active_keys=_base_keys,
        fracture_df=fracture_df,
        assignments_df=assignments_df,
        gt_label_map=gt_label_map,
        active_min_edge_weight=BASELINE_MIN_EDGE_WEIGHT,
        weight_col="edge_weight",
    )
    cos_sweep_df.to_csv(OUT_DIR / "cosine_threshold_edge_opportunities.csv", index=False)
    display(cos_sweep_df)

    if not cos_sweep_df.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        x = np.arange(len(cos_sweep_df))
        w = 0.25
        ax.bar(x - w, cos_sweep_df["n_new_same"], width=w, label="new same", color=COLOR_SAME)
        ax.bar(x, cos_sweep_df["n_new_cross"], width=w, label="new cross", color=COLOR_CROSS)
        ax.bar(x + w, cos_sweep_df["n_new_ambiguous"], width=w, label="new ambiguous", color=COLOR_AMBIG)
        ax.set_xticks(x)
        ax.set_xticklabels([str(round(t, 3)) for t in cos_sweep_df["threshold"]])
        ax.set_xlabel("Centroid cosine threshold τ")
        ax.set_ylabel("New edges vs baseline (not in active edge set)")
        ax.set_title("New candidate edges by τ — GT edge labels")
        ax.legend()
        plt.tight_layout()
        fig.savefig(OUT_DIR / "residual_cosine_sweep_grouped.png", dpi=120, bbox_inches="tight")
        plt.show()
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 3.8))
        ax.plot(cos_sweep_df["threshold"], cos_sweep_df["same_fraction_among_new_confident"], marker="o", label="same fraction (among new same+cross)", color=COLOR_SAME)
        ax.plot(cos_sweep_df["threshold"], cos_sweep_df["cross_fraction_among_new_confident"], marker="o", label="cross fraction (among new same+cross)", color=COLOR_CROSS)
        ax.set_xlabel("Threshold τ")
        ax.set_ylabel("Fraction of new confident edges")
        ax.set_ylim(0, 1.05)
        ax.legend()
        ax.set_title("Quality of new edges vs τ (same + cross only in denominator)")
        plt.tight_layout()
        fig.savefig(OUT_DIR / "residual_cosine_sweep_lines.png", dpi=120, bbox_inches="tight")
        plt.show()
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 3.5))
        ax.bar(
            [str(round(t, 3)) for t in cos_sweep_df["threshold"]],
            cos_sweep_df["n_fractured_campaigns_with_new_same_bridge"],
            color="#8172b2",
        )
        ax.set_xlabel("τ")
        ax.set_ylabel("Fractured campaigns with ≥1 new same bridge")
        ax.set_title("Could τ add a same-campaign edge between components?")
        plt.tight_layout()
        fig.savefig(OUT_DIR / "residual_cosine_sweep_reconnect.png", dpi=120, bbox_inches="tight")
        plt.show()
        plt.close(fig)"""
            ),
        },
        {
            "cell_type": "markdown",
            "id": "residual-md-5",
            "metadata": {},
            "source": _src(
                """## 5. Edge-weight threshold sweep

**Interpretation:** Retention / removal of labeled **same** vs **cross** edges as we require higher `edge_weight`. Shows whether true same-campaign links are concentrated at **low weights** (sensitive to stricter thresholds).

Fractured-campaign count vs threshold is saved in the same CSV as auxiliary columns if computed."""
            ),
        },
        {
            "cell_type": "code",
            "id": "residual-code-5",
            "metadata": {},
            "outputs": [],
            "source": _src(
                """if edges_df.empty or "edge_weight" not in edges_df.columns:
    display(Markdown("_Skip edge-weight sweep: no edge weights._"))
    sweep_df = pd.DataFrame()
else:
    ew = edges_df["edge_weight"].astype(float)
    sweep_thr = sorted(set(
        [float(ew.quantile(q)) for q in (0.0, 0.25, 0.5, 0.75, 0.9)]
        + [float(ew.min()), float(ew.max()), OPERATING_MIN_EDGE_WEIGHT, BASELINE_MIN_EDGE_WEIGHT]
    ))
    sweep_df = edge_weight_retention_table(
        edges_df, shard_summary, thresholds=sweep_thr, weight_col="edge_weight"
    )
    fc_extra = fractured_campaigns_at_edge_threshold(
        fracture_df,
        assignments_df,
        gt_label_map,
        edges_df,
        thresholds=sweep_thr,
        weight_col="edge_weight",
    )
    sweep_df = sweep_df.merge(fc_extra, on="threshold", how="left")
    sweep_df.to_csv(OUT_DIR / "edge_weight_threshold_retention.csv", index=False)
    display(sweep_df[["threshold", "frac_same_retained", "frac_cross_retained", "frac_same_removed", "frac_cross_removed", "n_fractured_campaigns"]])

    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.plot(sweep_df["threshold"], sweep_df["frac_same_retained"], marker="o", label="same retained", color=COLOR_SAME)
    ax.plot(sweep_df["threshold"], sweep_df["frac_cross_retained"], marker="o", label="cross retained", color=COLOR_CROSS)
    ax.set_xlabel("Minimum edge_weight to retain edge")
    ax.set_ylabel("Fraction of GT-labeled edges retained")
    ax.set_title("Same vs cross retention vs edge-weight floor")
    ax.legend()
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "residual_edge_weight_retention.png", dpi=168, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.plot(sweep_df["threshold"], sweep_df["frac_same_removed"], marker="o", label="same removed", color=COLOR_SAME)
    ax.plot(sweep_df["threshold"], sweep_df["frac_cross_removed"], marker="o", label="cross removed", color=COLOR_CROSS)
    ax.set_xlabel("Minimum edge_weight to retain edge")
    ax.set_ylabel("Fraction removed")
    ax.set_title("Same vs cross removal vs edge-weight floor")
    ax.legend()
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "residual_edge_weight_removal.png", dpi=168, bbox_inches="tight")
    plt.show()
    plt.close(fig)"""
            ),
        },
        {
            "cell_type": "markdown",
            "id": "residual-md-6",
            "metadata": {},
            "source": _src(
                """## 6. Community attachment (off–largest-component shards)

**Interpretation:** For shards in **fractured** campaigns that are **not** in the largest connected component, measure **max centroid cosine** to shards in that largest component and whether any **active** edge touches the target component.

**Noise vs normal** uses `size <= 1`. Off-main shard detail is in `community_attachment_off_main_detail.csv`."""
            ),
        },
        {
            "cell_type": "code",
            "id": "residual-code-6",
            "metadata": {},
            "outputs": [],
            "source": _src(
                """off_main = off_main_shards_and_similarity(
    fracture_df,
    assignments_df,
    gt_label_map,
    edges_df,
    nodes_df,
    active_min_edge_weight=BASELINE_MIN_EDGE_WEIGHT,
    weight_col="edge_weight",
    centroid_mat=centroid_mat,
    shard_to_idx=shard_to_idx,
)
attach_summary = community_attachment_summary(off_main)
off_main.to_csv(OUT_DIR / "community_attachment_off_main_detail.csv", index=False)
attach_summary.to_csv(OUT_DIR / "community_attachment_summary.csv", index=False)
display(attach_summary)

if not off_main.empty and centroid_mat is not None:
    fig = plot_max_cos_histogram(
        off_main, title="Max cosine to target (largest) component — off-main shards"
    )
    fig.savefig(OUT_DIR / "residual_attachment_max_cos_hist.png", dpi=168, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    fig2 = plot_attachment_stacked(
        off_main, title="Off-main shards — semantic match strength to target component"
    )
    fig2.savefig(OUT_DIR / "residual_attachment_stacked.png", dpi=168, bbox_inches="tight")
    plt.show()
    plt.close(fig2)
else:
    display(Markdown("_Skip attachment plots (no off-main rows or no centroids)._"))"""
            ),
        },
        {
            "cell_type": "markdown",
            "id": "residual-md-7",
            "metadata": {},
            "source": _src(
                """## 7. Decision summary (fill using numbers above)

1. **How much of remaining failure is due to noise/singleton shards?**  
   → See `noise_contribution_summary.csv` and §2 plots (`frac_fractured_emails_in_noise`).

2. **How much is due to missing same-campaign edges among normal shards?**  
   → §3 category counts (especially `no_edge_*` vs `existing_edge_below_operating_threshold`) and `missing_link_opportunity_breakdown.csv`.

3. **Do lower semantic thresholds add useful same edges or mostly cross bridges?**  
   → §4 `cosine_threshold_edge_opportunities.csv` — compare `n_new_same` vs `n_new_cross` and the line plot fractions.

4. **Are misses better explained by pairwise completion or community attachment?**  
   → §6 `community_attachment_summary.csv` — `frac_sem_match_ge_0.90` / `frac_with_any_direct_support` split by `group`.

5. **Most plausible next direction (pick one or combine):**  
   * Better Stage-1 / noise handling  
   * **Edge rescoring** or conditional validation of low-weight same signals  
   * **Missing-edge completion** near semantic τ  
   * **Node-to-community** attachment when off-main shards show high cosine to the dominant component  
   * **Limited headroom** if most failures are `no_edge_no_direct_support` and attachment cosines are low  

_Edit this cell after a run with one-sentence verdicts if helpful._"""
            ),
        },
    ]

    nb["cells"].extend(cells)
    NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print("Appended", len(cells), "cells to", NB)


if __name__ == "__main__":
    main()
