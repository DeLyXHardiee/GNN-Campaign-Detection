"""
Load a saved HeteroData .pt graph and print per-node-type feature statistics.

Usage (from repo root):
  python core/graph/inspect_saved_graph_stats.py path/to/graph_hetero.pt

Helps debug pair-training loss blow-ups (NaNs, huge values, odd email dims).
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path


def _tensor_stats(x) -> dict[str, float | int | bool]:
    import torch

    if x is None:
        return {"error": "missing_x"}
    t = x.detach().float().cpu()
    if t.numel() == 0:
        return {"n_rows": 0, "n_cols": 0}
    finite = torch.isfinite(t)
    n_fin = int(finite.sum().item())
    n_nan = int(torch.isnan(t).sum().item())
    n_inf = int(torch.isinf(t).sum().item())
    if n_fin == 0:
        return {
            "n_rows": int(t.shape[0]),
            "n_cols": int(t.shape[1]),
            "n_finite": 0,
            "n_nan": n_nan,
            "n_inf": n_inf,
        }
    tf = t[finite]
    return {
        "n_rows": int(t.shape[0]),
        "n_cols": int(t.shape[1]),
        "n_finite": n_fin,
        "n_nan": n_nan,
        "n_inf": n_inf,
        "min": float(tf.min().item()),
        "max": float(tf.max().item()),
        "mean": float(tf.mean().item()),
        "std": float(tf.std(unbiased=False).item()),
        "abs_max": float(tf.abs().max().item()),
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Inspect saved HeteroData graph tensor stats.")
    p.add_argument("graph_pt", type=Path, help="Path to hetero .pt from torch.save")
    args = p.parse_args()
    path = args.graph_pt.expanduser().resolve()
    if not path.is_file():
        print(f"ERROR: file not found: {path}", file=sys.stderr)
        return 1

    try:
        import torch
    except ImportError as e:
        print(f"ERROR: torch required: {e}", file=sys.stderr)
        return 1

    try:
        obj = torch.load(str(path), map_location="cpu", weights_only=False)
    except TypeError:
        obj = torch.load(str(path), map_location="cpu")

    node_types = getattr(obj, "node_types", None)
    if node_types is None:
        print(f"ERROR: loaded object has no node_types: {type(obj)!r}", file=sys.stderr)
        return 1

    print(f"graph_path={path}")
    print(f"node_types={list(node_types)}")
    for ntype in node_types:
        store = obj[ntype]
        has_x = hasattr(store, "x") and getattr(store, "x", None) is not None
        n_nodes = int(getattr(store, "num_nodes", 0) or 0)
        if not has_x:
            print(f"\n[{ntype}] num_nodes={n_nodes} (no x)")
            continue
        x = store.x
        st = _tensor_stats(x)
        print(f"\n[{ntype}] num_nodes={n_nodes} x_shape={tuple(x.shape)}")
        for k, v in st.items():
            if k == "error":
                continue
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                print(f"  {k}={v}")
            else:
                print(f"  {k}={v}")

    if "email" in node_types and hasattr(obj["email"], "x") and obj["email"].x is not None:
        ec = int(obj["email"].x.size(1))
        print(
            f"\nNote: email x has {ec} columns after graph build (projection + normalize_graph). "
            f"Pair GNN uses lazy in_channels; this dim is the encoder input width for email."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
