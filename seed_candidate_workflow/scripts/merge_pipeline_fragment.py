#!/usr/bin/env python3
"""Deep-merge a pipeline fragment JSON into repo-root pipeline_config.json."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _deep_merge(base: dict[str, Any], patch: dict[str, Any]) -> None:
    for key, value in patch.items():
        if key.startswith("_"):
            continue
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "fragment",
        type=Path,
        help="Fragment JSON (e.g. seed_candidate_workflow/configs/experiments/pipeline_fragment.dedup_task_identity_10.json)",
    )
    ap.add_argument(
        "--pipeline-config",
        type=Path,
        default=None,
        help="Target pipeline_config.json (default: repo root)",
    )
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[2]
    cfg_path = args.pipeline_config or (repo / "pipeline_config.json")
    frag_path = args.fragment if args.fragment.is_absolute() else (repo / args.fragment)
    frag_path = frag_path.resolve()

    with open(cfg_path, encoding="utf-8-sig") as f:
        cfg = json.load(f)
    with open(frag_path, encoding="utf-8-sig") as f:
        frag = json.load(f)

    _deep_merge(cfg, frag)
    cfg_path.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    print(f"Merged {frag_path.name} -> {cfg_path}")
    if "run_id" in frag:
        print(f"  run_id: {frag['run_id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
