"""CLI: build email-level anchor graph from JSON config."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.utils.anchor_graph_helpers import build_anchor_graph


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "analysis" / "configs" / "anchor_graph.default.json",
        help="Path to anchor graph config JSON.",
    )
    args = p.parse_args()

    cfg_path = args.config.expanduser().resolve()
    if not cfg_path.is_file():
        raise SystemExit(f"Config not found: {cfg_path}")
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    out = build_anchor_graph(cfg)
    print("Wrote:", out["paths"])
    validation = (out.get("summary") or {}).get("validation") or {}
    if validation:
        print("Validation:", validation)


if __name__ == "__main__":
    main()

