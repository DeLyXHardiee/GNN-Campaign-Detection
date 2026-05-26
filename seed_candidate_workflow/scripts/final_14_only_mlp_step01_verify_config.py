#!/usr/bin/env python3
"""Step 1: verify timestamp / time-gating / pair-universe before final thesis training."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from seed_candidate_workflow.utils.final_14_only_mlp_thesis import (  # noqa: E402
    load_manifest,
    pair_universe_stats,
    repo_root,
    resolve_repo_path,
    steps_dir,
    verify_time_gating_disabled,
)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, default=None)
    args = p.parse_args()

    repo = repo_root()
    manifest = load_manifest(args.manifest)
    out_dir = steps_dir(repo, manifest)

    baseline_csv = resolve_repo_path(repo, str(manifest["baseline_pair_dataset_csv"]))
    cand_cfg = resolve_repo_path(repo, str(manifest["candidate_generation_config"]))

    baseline_stats = pair_universe_stats(baseline_csv)
    gating = verify_time_gating_disabled(cand_cfg)

    report = {
        "pair_universe_decision": {
            "regenerate_seed_candidate_graph": False,
            "time_gating_in_generation": False,
            "reason": (
                "Time gating is disabled on all enabled generators. "
                "Pair keys match baseline _13; only time_gap_seconds_min (log1p MISP gap) changes after materialize."
            ),
        },
        "time_gating_audit": gating,
        "baseline_pair_universe": baseline_stats,
        "expected_final_pair_universe": {
            "same_row_keys_as_baseline": True,
            "n_pairs": baseline_stats.get("n_pairs"),
            "n_seed_positive_pairs": baseline_stats.get("n_seed_positive_pairs"),
            "n_non_seed_candidate_pairs": baseline_stats.get("n_non_seed_candidate_pairs"),
        },
        "timestamp_feature_plan": {
            "mlp_column": "time_gap_seconds_min",
            "transform": "log1p(abs(ts_i - ts_j)) from MISP dates",
            "not_used": ["raw_unix_ts", "raw_seconds_in_mlp_input"],
        },
        "training_plan": manifest.get("training"),
        "nnpu_pi": manifest.get("nnpu_pi"),
    }

    final_summary = resolve_repo_path(repo, str(manifest.get("final_pair_materialize_summary_json") or ""))
    if final_summary.is_file():
        report["existing_materialize_summary"] = json.loads(final_summary.read_text(encoding="utf-8-sig"))

    p_json = out_dir / "step01_verify_config_report.json"
    p_md = out_dir / "step01_verify_config_report.md"
    p_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    md = [
        "# Step 1 — Final MLP preflight",
        "",
        f"- **Time gating disabled on all enabled gatable generators:** {gating.get('all_gating_disabled')}",
        f"- **Pair universe regeneration needed:** {report['pair_universe_decision']['regenerate_seed_candidate_graph']}",
        f"- **Baseline pairs:** {baseline_stats.get('n_pairs')} (seed+ {baseline_stats.get('n_seed_positive_pairs')}, non-seed cand {baseline_stats.get('n_non_seed_candidate_pairs')})",
        "",
        "Next: `final_14_only_mlp_step02_materialize.py`",
        "",
        f"Report: `{p_json}`",
    ]
    p_md.write_text("\n".join(md), encoding="utf-8")
    print(json.dumps({"step01_report_json": str(p_json), "step01_report_md": str(p_md)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
