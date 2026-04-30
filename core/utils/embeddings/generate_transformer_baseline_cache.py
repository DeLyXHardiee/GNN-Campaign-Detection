from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow direct execution: python core/utils/embeddings/generate_transformer_baseline_cache.py
_CORE_ROOT = Path(__file__).resolve().parents[2]
if str(_CORE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CORE_ROOT))

from config.pipeline_config import load_pipeline_config, resolve_project_path
from graph.common import parse_misp_events
from utils.embeddings import DEFAULT_OUTPUT_DIR, get_embeddings


def _default_misp_path() -> str:
    cfg = load_pipeline_config()
    datasets = cfg.get("datasets") or {}
    graph_cfg = cfg.get("graph") or {}
    raw = datasets.get("misp_json_path") or graph_cfg.get("misp_json_path")
    resolved = resolve_project_path(raw) if raw else ""
    if not resolved:
        raise ValueError(
            "Could not resolve default MISP JSON path from pipeline_config "
            "(datasets.misp_json_path or graph.misp_json_path)."
        )
    return resolved


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generate/cache untouched transformer subject+body embeddings once for "
            "clustering baseline reuse."
        )
    )
    parser.add_argument(
        "--misp-json",
        default=None,
        help=(
            "Path to MISP JSON file (default: datasets.misp_json_path from pipeline_config)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Output directory for embeddings cache (default: {DEFAULT_OUTPUT_DIR}).",
    )
    args = parser.parse_args()

    misp_json = (
        resolve_project_path(args.misp_json)
        if args.misp_json
        else _default_misp_path()
    )
    if not misp_json:
        raise ValueError("MISP JSON path is empty.")
    output_dir = str(Path(args.output_dir).expanduser().resolve())

    with open(misp_json, "r", encoding="utf-8") as f:
        misp_events = json.load(f)
    emails = parse_misp_events(misp_events)
    subj_vecs, body_vecs, subj_dim, body_dim = get_embeddings(
        emails,
        output_dir=output_dir,
    )
    n = len(subj_vecs) if subj_vecs else len(body_vecs)
    cache_path = Path(output_dir) / "embeddings.json"
    print(
        f"Transformer baseline cache ready: n_emails={n}, subj_dim={subj_dim}, "
        f"body_dim={body_dim}, cache={cache_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
