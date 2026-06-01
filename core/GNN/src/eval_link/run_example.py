"""
Example script: run AUROC/AP and Recall@K analysis from saved graph + checkpoint.

Run from the GNN package root (core/GNN) so that 'src' resolves, e.g.:

    cd core/GNN
    python -m src.eval_link.run_example

Or with the data/checkpoint paths as arguments:

    python -m src.eval_link.run_example path/to/graph.pt best_model.pt

Edit DATA_PATH and CHECKPOINT_FILENAME below if you prefer to run without arguments.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PKG_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from src.eval_link import run_auroc_ap_from_run, run_recall_at_k_from_run


def main() -> None:
    parser = argparse.ArgumentParser(description="Run AUROC/AP and Recall@K seed_candidate_workflow.")
    parser.add_argument(
        "data_path",
        nargs="?",
        default=None,
        help="Path to HeteroData .pt file (default: relative path below)",
    )
    parser.add_argument(
        "checkpoint",
        nargs="?",
        default="best_model.pt",
        help="Checkpoint filename under models/ (default: best_model.pt)",
    )
    args = parser.parse_args()

    data_path = args.data_path or str(
        _PKG_ROOT.parent.parent / "graph" / "output" / "incidents-20260211-misp_hetero.pt"
    )
    checkpoint = args.checkpoint

    if not Path(data_path).expanduser().exists():
        print(f"Data file not found: {data_path}")
        print("Pass a valid data_path as first argument or edit DATA_PATH in the script.")
        sys.exit(1)

    print("Running AUROC/AP seed_candidate_workflow...")
    result_auroc = run_auroc_ap_from_run(
        data_path=data_path,
        filename=checkpoint,
        to_undirected=True,
    )
    print(f"  Metrics: {len(result_auroc['metrics'])} edge types")
    print(f"  Plots:  {result_auroc['plot_paths']}")
    print(f"  JSON:   {result_auroc['metrics_path']}")

    print("\nRunning Recall@K seed_candidate_workflow...")
    result_recall = run_recall_at_k_from_run(
        data_path=data_path,
        filename=checkpoint,
        K_list=[1, 10, 20, 40, 60, 80, 100],
        use_dot=False,
        to_undirected=True,
    )
    print(f"  Plot:   {result_recall['plot_path']}")
    print(f"  JSON:   {result_recall['metrics_path']}")

    print("\nDone. Outputs are under models/analysis_<checkpoint_stem>/.")


if __name__ == "__main__":
    main()
