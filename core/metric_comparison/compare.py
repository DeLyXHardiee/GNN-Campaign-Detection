"""Orchestrate metric comparison and write metric_comparison/ outputs."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score

from clustering.clusteringMetrics import compute_external_metrics, extract_ground_truth_labels

from config.pipeline_config import GnnPathLayout, gnn_path_layout_from_pipeline, load_pipeline_config

from .artifacts import (
    load_campaign_artifact,
    pred_map_from_campaign_payload,
    strip_payload_meta,
)
from .plots import save_agreement_bar_chart, save_external_metrics_bar_chart


def _external_metrics_from_pred_map(
    pred_map: dict[str, int],
    ground_truth_labels: dict[str, Any],
) -> dict[str, Any]:
    true_labels: list[Any] = []
    predicted_labels: list[int] = []
    for eid, pl in pred_map.items():
        true = ground_truth_labels.get(eid)
        if true is None:
            continue
        true_labels.append(true)
        predicted_labels.append(int(pl))
    return compute_external_metrics(true_labels, predicted_labels)


def _agreement_on_intersection(
    pred_a: dict[str, int],
    pred_b: dict[str, int],
) -> dict[str, Any] | None:
    common = sorted(set(pred_a) & set(pred_b))
    n = len(common)
    if n < 2:
        return None
    y_a = [pred_a[eid] for eid in common]
    y_b = [pred_b[eid] for eid in common]
    return {
        "n_common_emails": n,
        "adjusted_rand_score": float(adjusted_rand_score(y_a, y_b)),
        "adjusted_mutual_info_score": float(
            adjusted_mutual_info_score(y_a, y_b, average_method="arithmetic")
        ),
    }


def _write_csv_summary(path: Path, summary: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    fs = summary.get("featureset") or {}
    gnn = summary.get("gnn") or {}
    agree = summary.get("agreement") or {}

    for label, block in ("featureset", fs), ("gnn", gnn):
        if not block:
            continue
        ext = block.get("external_metrics_vs_ground_truth") or {}
        rows.append(
            {
                "solution": label,
                "homogeneity": ext.get("homogeneity"),
                "completeness": ext.get("completeness"),
                "v_measure": ext.get("v_measure"),
                "n_samples": ext.get("n_samples"),
            }
        )
    if agree:
        rows.append(
            {
                "solution": "agreement_fs_vs_gnn",
                "adjusted_rand_score": agree.get("adjusted_rand_score"),
                "adjusted_mutual_info_score": agree.get("adjusted_mutual_info_score"),
                "n_common_emails": agree.get("n_common_emails"),
            }
        )

    if not rows:
        path.write_text("", encoding="utf-8")
        return str(path)

    fieldnames: list[str] = []
    for r in rows:
        for k in r:
            if k not in fieldnames:
                fieldnames.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    return str(path)


def run_metric_comparison_for_run(
    run_path: Path,
    *,
    ground_truth_path: str,
    path_layout: GnnPathLayout | None = None,
    dpi: int = 150,
) -> dict[str, Any]:
    """
    Read campaign artifacts, compute metrics, write ``metric_comparison/`` under ``run_path``.

    Raises ``FileNotFoundError`` if neither campaigns file exists.
    """
    run_path = Path(run_path).expanduser().resolve()
    cfg = load_pipeline_config()
    layout = path_layout or gnn_path_layout_from_pipeline(cfg)

    fs_artifact_path = run_path / "featureset_clustering" / "campaigns_featureset.json"
    gnn_artifact_path = run_path / layout.clustering_subdir / "campaigns_gnn.json"

    fs_raw = load_campaign_artifact(fs_artifact_path)
    gnn_raw = load_campaign_artifact(gnn_artifact_path)

    if fs_raw is None and gnn_raw is None:
        raise FileNotFoundError(
            f"No campaign artifacts found. Expected at least one of:\n"
            f"  {fs_artifact_path}\n"
            f"  {gnn_artifact_path}"
        )

    ground_truth_labels = extract_ground_truth_labels(ground_truth_path)

    out_dir = run_path / "metric_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "run_dir": str(run_path),
        "ground_truth_path": str(Path(ground_truth_path).resolve()),
        "artifacts": {
            "featureset": str(fs_artifact_path),
            "gnn": str(gnn_artifact_path),
        },
    }

    fs_metrics: dict[str, Any] | None = None
    gnn_metrics: dict[str, Any] | None = None
    fs_pred: dict[str, int] | None = None
    gnn_pred: dict[str, int] | None = None

    if fs_raw:
        fs_pred = pred_map_from_campaign_payload(fs_raw)
        fs_metrics = _external_metrics_from_pred_map(fs_pred, ground_truth_labels)
        summary["featureset"] = {
            "meta": strip_payload_meta(fs_raw),
            "external_metrics_vs_ground_truth": fs_metrics,
        }

    if gnn_raw:
        gnn_pred = pred_map_from_campaign_payload(gnn_raw)
        gnn_metrics = _external_metrics_from_pred_map(gnn_pred, ground_truth_labels)
        summary["gnn"] = {
            "meta": strip_payload_meta(gnn_raw),
            "external_metrics_vs_ground_truth": gnn_metrics,
        }

    agreement: dict[str, Any] | None = None
    if fs_pred and gnn_pred:
        agreement = _agreement_on_intersection(fs_pred, gnn_pred)
        if agreement is not None:
            summary["agreement"] = agreement

    json_path = out_dir / "comparison_summary.json"
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    csv_path = out_dir / "comparison_metrics.csv"
    _write_csv_summary(csv_path, summary)

    ext_png = out_dir / "external_metrics_comparison.png"
    plot_ext = save_external_metrics_bar_chart(
        featureset_metrics=fs_metrics,
        gnn_metrics=gnn_metrics,
        out_path=ext_png,
        dpi=dpi,
    )

    agree_png_path: str | None = None
    if agreement and agreement.get("n_common_emails", 0) >= 2:
        agree_png_path = save_agreement_bar_chart(
            ari=float(agreement["adjusted_rand_score"]),
            ami=float(agreement["adjusted_mutual_info_score"]),
            out_path=out_dir / "agreement_metrics.png",
            dpi=dpi,
        )

    result = {
        "output_dir": str(out_dir),
        "comparison_summary_json": str(json_path),
        "comparison_metrics_csv": str(csv_path),
        "external_metrics_plot": plot_ext,
        "agreement_plot": agree_png_path,
    }
    print(f"Metric comparison written under: {out_dir}")
    return result
