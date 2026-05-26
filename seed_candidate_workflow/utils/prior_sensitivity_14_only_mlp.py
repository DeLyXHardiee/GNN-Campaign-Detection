"""nnPU prior (pi) sensitivity grid for _14_only_mlp explicit pair scorer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_manifest(manifest_path: Path | None = None) -> dict[str, Any]:
    repo = Path(__file__).resolve().parents[2]
    path = manifest_path or (
        repo / "seed_candidate_workflow/configs/prior_sensitivity/prior_sensitivity_14_only_mlp.manifest.json"
    )
    path = Path(path).resolve()
    return json.loads(path.read_text(encoding="utf-8-sig"))


def prior_entries(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    priors = manifest.get("priors") or []
    if not isinstance(priors, list) or not priors:
        raise ValueError("manifest.priors must be a non-empty list")
    return [dict(p) for p in priors]


def build_14_only_mlp_training_cfg(
    *,
    pi: float,
    pair_dataset_csv: str,
    reference_training_config: Path,
    project_root: Path | None = None,
) -> dict[str, Any]:
    """Same architecture/split/optimizer settings as baseline _14_only_mlp; only pi varies."""
    repo = Path(project_root or Path(__file__).resolve().parents[2])
    ref = json.loads(Path(reference_training_config).read_text(encoding="utf-8-sig"))

    pipeline_training: dict[str, Any] = {}
    pcfg_path = repo / "pipeline_config.json"
    if pcfg_path.is_file():
        pipeline_training = dict(json.loads(pcfg_path.read_text(encoding="utf-8-sig")).get("training") or {})

    # pipeline training block supplies lr scheduler + checkpoint names; ref supplies pair-architecture keys.
    cfg: dict[str, Any] = {**pipeline_training, **ref}
    cfg.update(
        {
            "pu_class_prior": float(pi),
            "pi_p": float(pi),
            "pair_dataset_csv": str(pair_dataset_csv),
            "training_objective": "pair_supervision",
            "pair_encoder_backend": "explicit_only",
            "pair_scorer_use_embedding_features": False,
            "explicit_only_pair_features": True,
        }
    )
    for key, value in {
        "torch_seed": 42,
        "lr": 2e-4,
        "wd": 2e-5,
        "epochs": 30,
        "early_stopping_patience": 7,
        "lr_reduce_patience": 4,
        "lr_reduce_factor": 0.5,
        "lr_reduce_min": 1e-6,
        "model_save_name": "best_model.pt",
        "hidden": 128,
        "out_dim": 128,
        "layers": 2,
        "dropout": 0.0,
    }.items():
        cfg.setdefault(key, value)
    return cfg


def training_run_dir(repo: Path, run_id: str) -> Path:
    return (repo / "output/runs" / run_id).resolve()


def scoring_run_dir(repo: Path, scoring_run_id: str) -> Path:
    return (repo / "seed_candidate_workflow/output/scoring_runs" / scoring_run_id).resolve()


def community_sweep_csv(repo: Path, scoring_run_id: str, *, gt_slug: str = "ground_truth") -> Path:
    return (
        scoring_run_dir(repo, scoring_run_id)
        / "seed_candidate"
        / "community"
        / f"anchor_community_sweep__{gt_slug}.csv"
    )


def format_latex_prior_table(rows: list[dict[str, Any]], *, caption: str | None = None) -> str:
    cap = caption or (
        "Best community detection on expanded ground truth by nnPU prior "
        "(\\texttt{14\\_only\\_mlp} explicit MLP; Leiden/Louvain sweep)."
    )
    lines = [
        "% Requires \\usepackage{booktabs}",
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{{cap}}}",
        r"\label{tab:14-only-mlp-prior-sensitivity}",
        r"\small",
        r"\begin{tabular}{l l r r r r r r}",
        r"\toprule",
        r"$\pi$ & Algorithm & Threshold & Resolution & $H$ & $C$ & $V$ \\",
        r"\midrule",
    ]
    for row in rows:
        pi = row.get("pi")
        pi_s = f"{float(pi):.2f}" if pi is not None else "---"
        algo = str(row.get("algorithm") or row.get("method") or "---")
        thr = row.get("threshold")
        thr_s = f"{float(thr):.1f}" if thr is not None else "---"
        res = row.get("resolution")
        res_s = f"{float(res):.1f}" if res is not None else "---"

        def _f(key: str) -> str:
            v = row.get(key)
            return f"{float(v):.3f}" if v is not None else "---"

        lines.append(
            f"{pi_s} & {algo} & {thr_s} & {res_s} & {_f('homogeneity')} & {_f('completeness')} & {_f('v_measure')} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    return "\n".join(lines)
