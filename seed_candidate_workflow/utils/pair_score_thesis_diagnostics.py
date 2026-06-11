"""Thesis-ready score statistics and separation metrics for learned pair scorers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from seed_candidate_workflow.utils.raw_gnn_notebook import load_ground_truth_structures
from seed_candidate_workflow.utils.scorer_diagnostics_core import safe_auroc

SLICE_ALL = "all_evaluated"
SLICE_SEED = "seed_positive"
SLICE_NON_SEED = "non_seed_candidate"


def safe_average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    try:
        from sklearn.metrics import average_precision_score
    except ImportError:
        return None
    if y_true.size < 2 or len(np.unique(y_true)) < 2:
        return None
    return float(average_precision_score(y_true, y_score))


def distribution_stats(scores: np.ndarray) -> dict[str, Any]:
    x = np.asarray(scores, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "std": None,
            "q1": None,
            "q3": None,
            "iqr": None,
            "min": None,
            "max": None,
        }
    q1, med, q3 = (float(v) for v in np.quantile(x, [0.25, 0.5, 0.75]))
    std = float(x.std(ddof=1)) if x.size > 1 else 0.0
    return {
        "count": int(x.size),
        "mean": float(np.mean(x)),
        "median": med,
        "std": std,
        "q1": q1,
        "q3": q3,
        "iqr": float(q3 - q1),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
    }


def _seed_mask(df: pd.DataFrame) -> np.ndarray:
    n = len(df)
    if "is_seed_pair" in df.columns:
        return df["is_seed_pair"].fillna(False).astype(bool).to_numpy()
    if "from_seed" in df.columns:
        return df["from_seed"].fillna(False).astype(bool).to_numpy()
    if "pair_status" in df.columns:
        return df["pair_status"].astype(str).str.lower().eq("positive").to_numpy()
    return np.zeros(n, dtype=bool)


def _campaign_masks(
    *,
    email_i: np.ndarray,
    email_j: np.ndarray,
    label_map: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(email_i)
    camp_i = np.array([label_map.get(str(email_i[k])) for k in range(n)], dtype=object)
    camp_j = np.array([label_map.get(str(email_j[k])) for k in range(n)], dtype=object)
    both = np.array(
        [camp_i[k] is not None and camp_j[k] is not None for k in range(n)],
        dtype=bool,
    )
    same = both & (camp_i == camp_j)
    cross = both & (camp_i != camp_j)
    return both, same, cross


def _slice_mask(slice_id: str, *, seed: np.ndarray, base: np.ndarray) -> np.ndarray:
    if slice_id == SLICE_ALL:
        return base
    if slice_id == SLICE_SEED:
        return base & seed
    if slice_id == SLICE_NON_SEED:
        return base & (~seed)
    raise ValueError(f"Unknown slice_id: {slice_id!r}")


def _summarize_slice(
    *,
    slice_id: str,
    slice_label: str,
    scores: np.ndarray,
    seed: np.ndarray,
    both: np.ndarray,
    same: np.ndarray,
    cross: np.ndarray,
    scored: np.ndarray,
) -> dict[str, Any]:
    m = _slice_mask(slice_id, seed=seed, base=both & scored)
    same_m = m & same
    cross_m = m & cross
    same_s = scores[same_m]
    cross_s = scores[cross_m]
    y = same[same_m | cross_m].astype(np.int32)
    s = scores[same_m | cross_m]
    auroc = safe_auroc(y, s)
    ap = safe_average_precision(y, s)
    return {
        "slice_id": slice_id,
        "slice_label": slice_label,
        "n_gt_covered_scored_pairs": int(m.sum()),
        "n_same_campaign": int(same_m.sum()),
        "n_cross_campaign": int(cross_m.sum()),
        "same_campaign": distribution_stats(same_s),
        "cross_campaign": distribution_stats(cross_s),
        "auroc_same_vs_cross": auroc,
        "average_precision_same_vs_cross": ap,
        "average_precision_class_imbalance_sensitive": True,
        "auroc_both_classes_present": (
            auroc is not None
            and int(same_m.sum()) > 0
            and int(cross_m.sum()) > 0
        ),
    }


def build_statistics_rows(slices: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sl in slices:
        for rel in ("same_campaign", "cross_campaign"):
            stats = sl[rel]
            rows.append(
                {
                    "slice_id": sl["slice_id"],
                    "slice_label": sl["slice_label"],
                    "campaign_relation": rel,
                    "count": stats["count"],
                    "mean_score": stats["mean"],
                    "median_score": stats["median"],
                    "std_score": stats["std"],
                    "q1_score": stats["q1"],
                    "q3_score": stats["q3"],
                    "iqr_score": stats["iqr"],
                    "min_score": stats["min"],
                    "max_score": stats["max"],
                }
            )
    return rows


def build_separation_rows(slices: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sl in slices:
        rows.append(
            {
                "slice_id": sl["slice_id"],
                "slice_label": sl["slice_label"],
                "n_same_campaign": sl["n_same_campaign"],
                "n_cross_campaign": sl["n_cross_campaign"],
                "auroc_same_vs_cross": sl["auroc_same_vs_cross"],
                "average_precision_same_vs_cross": sl["average_precision_same_vs_cross"],
                "average_precision_class_imbalance_sensitive": sl[
                    "average_precision_class_imbalance_sensitive"
                ],
                "auroc_both_classes_present": sl["auroc_both_classes_present"],
            }
        )
    return rows


def format_latex_table(
    *,
    slices: list[dict[str, Any]],
    score_col_label: str = "PU score",
    caption: str | None = None,
    label: str = "tab:pair-score-thesis-diagnostics",
) -> str:
    cap = caption or (
        "Learned pair score statistics on expanded-ground-truth--covered pairs "
        "(same vs.\\ cross campaign)."
    )
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{{cap}}}",
        rf"\label{{{label}}}",
        r"\small",
        r"\begin{tabular}{llrrrrrr}",
        r"\toprule",
        "Slice & Relation & $n$ & Mean & Median & Std & Q1 & Q3 & AUROC \\",
        r"\midrule",
    ]
    for sl in slices:
        auroc = sl.get("auroc_same_vs_cross")
        auroc_s = f"{auroc:.3f}" if auroc is not None else "---"
        ap = sl.get("average_precision_same_vs_cross")
        ap_s = f"{ap:.3f}" if ap is not None else "---"
        for i, rel in enumerate(("same_campaign", "cross_campaign")):
            st = sl[rel]
            rel_tex = "Same" if rel == "same_campaign" else "Cross"
            slice_tex = sl["slice_label"].replace("_", r"\_") if i == 0 else ""
            n = st["count"]
            mean = st["mean"]
            med = st["median"]
            std = st["std"]
            q1 = st["q1"]
            q3 = st["q3"]

            def _f(v: Any, nd: int = 3) -> str:
                if v is None:
                    return "---"
                return f"{float(v):.{nd}f}"

            auroc_cell = auroc_s if i == 0 else ""
            lines.append(
                f"{slice_tex} & {rel_tex} & {n:,} & {_f(mean)} & {_f(med)} & {_f(std)} "
                f"& {_f(q1)} & {_f(q3)} & {auroc_cell} \\\\"
            )
        lines.append(
            rf"\multicolumn{{9}}{{l}}{{\footnotesize AP (class-imbalance-sensitive): {ap_s}}} \\"
        )
        lines.append(r"\addlinespace")
    if lines[-1] == r"\addlinespace":
        lines.pop()
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines) + "\n"


def compute_thesis_pair_score_diagnostics(
    *,
    df: pd.DataFrame,
    scores: np.ndarray,
    gt_path: Path,
    score_col: str = "pu_score",
    run_dir: Path | None = None,
    pair_csv: Path | None = None,
    scoring_run_id: str | None = None,
) -> dict[str, Any]:
    gt_path = Path(gt_path).resolve()
    label_map, _eid_row, _camp = load_ground_truth_structures(gt_path)
    label_map = {str(k): v for k, v in label_map.items()}

    ei = df["email_i"].astype(str).values
    ej = df["email_j"].astype(str).values
    scored = np.isfinite(np.asarray(scores, dtype=float))
    seed = _seed_mask(df)
    both, same, cross = _campaign_masks(email_i=ei, email_j=ej, label_map=label_map)

    slice_specs = [
        (SLICE_ALL, "All evaluated"),
        (SLICE_SEED, "Seed-positive"),
        (SLICE_NON_SEED, "Non-seed candidate"),
    ]
    slices = [
        _summarize_slice(
            slice_id=sid,
            slice_label=slab,
            scores=scores,
            seed=seed,
            both=both,
            same=same,
            cross=cross,
            scored=scored,
        )
        for sid, slab in slice_specs
    ]
    for sl in slices:
        sl["same_campaign"]["campaign_relation"] = "same_campaign"
        sl["cross_campaign"]["campaign_relation"] = "cross_campaign"

    return {
        "run_dir": str(run_dir.resolve()) if run_dir else None,
        "pair_csv": str(pair_csv.resolve()) if pair_csv else None,
        "scoring_run_id": scoring_run_id,
        "gt_path": str(gt_path),
        "score_col": score_col,
        "score_col_note": (
            "Raw learned PU probability (pu_score) before seed edge_weight=1.0 "
            "and community edge_weight transform; matches community scoring input."
        ),
        "slices": slices,
        "statistics_rows": build_statistics_rows(slices),
        "separation_rows": build_separation_rows(slices),
    }


def write_thesis_pair_score_diagnostics(
    payload: dict[str, Any],
    *,
    output_dir: Path,
) -> dict[str, Path]:
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    stats_csv = output_dir / "thesis_pair_score_statistics.csv"
    sep_csv = output_dir / "thesis_pair_score_separation.csv"
    json_path = output_dir / "thesis_pair_score_diagnostics.json"
    tex_path = output_dir / "thesis_pair_score_statistics.tex"

    pd.DataFrame(payload["statistics_rows"]).to_csv(stats_csv, index=False)
    pd.DataFrame(payload["separation_rows"]).to_csv(sep_csv, index=False)

    latex = format_latex_table(slices=payload["slices"])
    payload_out = {**payload, "latex_table": latex}
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload_out, f, indent=2)
    tex_header = (
        "% Requires \\usepackage{booktabs} in your LaTeX preamble.\n"
        f"% Score column: {payload.get('score_col', 'pu_score')} on GT-covered pairs.\n\n"
    )
    tex_path.write_text(tex_header + latex, encoding="utf-8")

    return {
        "statistics_csv": stats_csv,
        "separation_csv": sep_csv,
        "json": json_path,
        "latex": tex_path,
    }


def run_thesis_pair_score_diagnostics(
    *,
    run_dir: Path,
    graph_pt: Path,
    pair_csv: Path,
    gt_path: Path,
    output_dir: Path | None = None,
    checkpoint_name: str = "best_model.pt",
    device: str = "cpu",
    to_undirected: bool = True,
    scoring_run_id: str | None = None,
    scored_pairs_csv: Path | None = None,
) -> dict[str, Any]:
    """Score pairs (or load pre-scored CSV) and write thesis diagnostics artifacts."""
    from src.pair_train import load_pair_training_dataframe

    from seed_candidate_workflow.utils.pair_model_inference import (
        load_pair_supervision_for_inference,
        score_pair_rows,
    )

    project_root = Path(__file__).resolve().parents[2]
    run_dir = Path(run_dir).resolve()
    graph_pt = Path(graph_pt).resolve()
    pair_csv = Path(pair_csv).resolve()
    gt_path = Path(gt_path).resolve()
    out_dir = (
        output_dir
        or (run_dir / "pair_score_separation" / "thesis_score_diagnostics")
    ).resolve()

    df, _stats = load_pair_training_dataframe(pair_csv)
    df_work = df.reset_index(drop=True).copy()

    if scored_pairs_csv is not None and Path(scored_pairs_csv).is_file():
        scored_df = pd.read_csv(scored_pairs_csv, low_memory=False)
        if "pu_score" not in scored_df.columns:
            raise ValueError(f"scored_pairs_csv missing pu_score column: {scored_pairs_csv}")
        key_cols = ["email_i", "email_j"]
        merged = df_work.merge(
            scored_df[key_cols + ["pu_score"]],
            on=key_cols,
            how="left",
            validate="many_to_one",
        )
        scores = pd.to_numeric(merged["pu_score"], errors="coerce").to_numpy(dtype=float)
        score_source = str(scored_pairs_csv)
    else:
        df_work["_row"] = np.arange(len(df_work), dtype=np.int64)
        edge_scores_csv = (run_dir / "edge_gnn_pair_scores.csv").resolve()
        if edge_scores_csv.is_file():
            from seed_candidate_workflow.utils.edge_gnn_score_inference import (
                scores_array_for_pair_dataframe,
            )

            scores, _diag = scores_array_for_pair_dataframe(
                edge_scores_csv, df_work, project_root=project_root
            )
            score_source = str(edge_scores_csv)
        else:
            bundle = load_pair_supervision_for_inference(
                run_dir=run_dir,
                graph_pt=graph_pt,
                checkpoint_name=checkpoint_name,
                device=device,
                to_undirected=to_undirected,
            )
            scores = score_pair_rows(
                model=bundle["model"],
                pair_scorer=bundle["pair_scorer"],
                data_cpu=bundle["data_cpu"],
                df_work=df_work,
                device=bundle["device"],
                fanout=bundle["fanout"],
                pair_batch_size=bundle["pair_batch_size"],
                max_unique_emails=bundle["max_unique_emails"],
                pair_feature_columns=bundle.get("pair_feature_columns"),
            )
            score_source = str(bundle["checkpoint_path"])

    payload = compute_thesis_pair_score_diagnostics(
        df=df_work,
        scores=scores,
        gt_path=gt_path,
        score_col="pu_score",
        run_dir=run_dir,
        pair_csv=pair_csv,
        scoring_run_id=scoring_run_id,
    )
    payload["score_source"] = score_source
    paths = write_thesis_pair_score_diagnostics(payload, output_dir=out_dir)
    payload["output_paths"] = {k: str(v) for k, v in paths.items()}
    return payload
