"""
Build heterograph + mapping + BERT overlay from MISP lake + semantic_supernode_clusters.csv.

Run from repo root with ``core`` on PYTHONPATH (same as other analysis scripts), e.g.::

    python analysis/scripts/build_semantic_supernode_collapsed_graph.py \\
        --misp-json data/misp/incidents-lake-misp.json \\
        --clusters-csv output/analysis/semantic_supernode_t097/semantic_supernode_clusters.csv \\
        --source-embeddings core/utils/embeddings/output/embeddings.json \\
        --out-dir core/graph/output \\
        --out-name my_run_hetero.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_script_dir = Path(__file__).resolve().parent
_repo_root = _script_dir.parent.parent
_core = _repo_root / "core"
if str(_core) not in sys.path:
    sys.path.insert(0, str(_core))

from config.pipeline_config import EmailFeatureProjectionSettings, graph_build_settings_from_pipeline, load_pipeline_config
from graph.semantic_supernode_collapse import build_semantic_supernode_graph


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--misp-json", type=Path, required=True)
    p.add_argument("--clusters-csv", type=Path, required=True)
    p.add_argument("--source-embeddings", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--out-name", type=str, required=True, help="e.g. my_stem_hetero.pt")
    p.add_argument(
        "--embeddings-overlay-parent",
        type=Path,
        default=None,
        help="Parent dir for semantic_supernode_embeddings_overlay/ (default: --out-dir)",
    )
    p.add_argument("--l2-normalize-mean-bert", action="store_true")
    p.add_argument(
        "--from-pipeline-config",
        action="store_true",
        help="Use graph.email_feature_projection / degree_node_filter / zero_email_timestamps from pipeline_config.json",
    )
    p.add_argument("--max-misp-events", type=int, default=None)
    args = p.parse_args()

    exclude = None
    deg = None
    proj = None
    zts = False
    if args.from_pipeline_config:
        cfg = load_pipeline_config(project_root=_repo_root)
        g = graph_build_settings_from_pipeline(cfg, project_root=_repo_root)
        exclude = g.exclude_node_types
        deg = g.degree_node_filter
        proj = g.email_feature_projection or EmailFeatureProjectionSettings()
        zts = g.zero_email_timestamps

    _g, gp, mp, mpath = build_semantic_supernode_graph(
        misp_json_path=args.misp_json,
        clusters_csv=args.clusters_csv,
        source_embeddings_json=args.source_embeddings,
        out_dir=args.out_dir,
        out_name=args.out_name,
        embeddings_overlay_dir=args.embeddings_overlay_parent or args.out_dir,
        l2_normalize_after_mean=args.l2_normalize_mean_bert,
        exclude_nodes=exclude,
        degree_node_filter=deg,
        email_feature_projection=proj,
        zero_email_timestamps=zts,
        max_misp_events=args.max_misp_events,
    )
    print(f"Graph: {gp}")
    print(f"Meta: {mp}")
    print(f"Mapping: {mpath}")


if __name__ == "__main__":
    main()
