from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def print_experiment_cli_summary(out: dict[str, Any]) -> None:
    m = out.get("manifest") or {}
    dry = bool(out.get("dry_run"))
    print("", flush=True)
    print(
        "experiment: dry run (paths resolved; graph setup and community not executed)." if dry else "experiment: complete",
        flush=True,
    )
    mj = str(out.get("manifest_json") or "").strip()
    if mj:
        print(f"  run_manifest: {mj}", flush=True)
    gbr = str(m.get("graph_bundle_root") or "").strip()
    gid = str(m.get("graph_id") or "").strip()
    if gbr and gid:
        print(f"  graph_bundle: {Path(gbr) / gid}", flush=True)
    rr = str(m.get("run_root") or "").strip()
    if rr:
        print(f"  scoring_run_dir: {rr}", flush=True)
    mode = str(m.get("mode") or "").strip()
    if mode:
        print(f"  mode: {mode}", flush=True)
    sm = m.get("score_mode")
    if sm:
        print(f"  score_mode: {sm}", flush=True)
    for row in out.get("community_results") or []:
        tgt = str(row.get("target") or "").strip() or "(target)"
        cr = row.get("community_result") or {}
        if not isinstance(cr, dict):
            continue
        if cr.get("dry_run"):
            print(f"  [{tgt}] community: (dry run — not executed)", flush=True)
            continue
        od = str(cr.get("output_dir") or "").strip()
        sj = str(cr.get("summary_json") or "").strip()
        sd = str(cr.get("scorer_diagnostics_json") or "").strip()
        if od:
            print(f"  [{tgt}] community_dir: {od}", flush=True)
        if sj:
            print(f"  [{tgt}] community_summary: {sj}", flush=True)
        if sd:
            print(f"  [{tgt}] scorer_diagnostics: {sd}", flush=True)


def build_target_result(*, target: str, edges_csv: str, community_result: dict[str, Any]) -> dict[str, Any]:
    return {
        "target": target,
        "inputs": {
            "edges_csv": str(edges_csv),
            "score_mode": community_result.get("score_mode"),
        },
        "artifacts": {
            "output_dir": community_result.get("output_dir"),
            "summary_json": community_result.get("summary_json"),
        },
        "metrics": community_result.get("metrics") or {},
        "community_result": community_result,
    }


def write_manifest(run_root: Path, manifest: dict[str, Any]) -> str:
    p_manifest = run_root / "run_manifest.json"
    p_manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return str(p_manifest)


def legacy_compatible_target_view(target_row: dict[str, Any]) -> dict[str, Any]:
    """Compatibility adapter for consumers expecting target/edges_csv/community_result."""
    out = {
        "target": target_row.get("target"),
        "community_result": target_row.get("community_result"),
    }
    inputs = dict(target_row.get("inputs") or {})
    if "edges_csv" in inputs:
        out["edges_csv"] = inputs["edges_csv"]
    return out
