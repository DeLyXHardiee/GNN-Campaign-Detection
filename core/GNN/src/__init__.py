"""Utilities for GNN training and evaluation."""

from .load_graph_data import load_hetero_pt, load_imdb
from .graph_diagnostics import print_connectivity_report
from .model import HeteroSAGE, DotPredictor, MLPredictor, DistMultPredictor

# Lazy imports: train, model_io, embed, loaders require torch-sparse/pyg-lib.
# Only load them when the user actually needs training or loaders.
def __getattr__(name):
    if name == "run_training":
        from .train import run_training
        return run_training
    if name in ("load_model_checkpoint", "load_full_run", "load_training_state", "save_model_checkpoint"):
        from . import model_io
        return getattr(model_io, name)
    if name in ("export_embeddings", "get_primary_embeddings", "embed_with_graph"):
        from . import embed
        return getattr(embed, name)
    if name == "make_link_loaders":
        from .loaders import make_link_loaders
        return make_link_loaders
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "load_hetero_pt",
    "load_imdb",
    "print_connectivity_report",
    "run_training",
    "HeteroSAGE",
    "DotPredictor",
    "MLPredictor",
    "DistMultPredictor",
    "load_model_checkpoint",
    "load_full_run",
    "load_training_state",
    "save_model_checkpoint",
    "export_embeddings",
    "get_primary_embeddings",
    "embed_with_graph",
    "make_link_loaders",
]
