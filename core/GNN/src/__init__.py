"""Utilities for GNN training and evaluation."""

from .load_graph_data import load_hetero_pt, load_imdb
from .model import HeteroSAGE, DotPredictor, MLPredictor, DistMultPredictor

try:
    from .train import run_training
except Exception as _e:  # pragma: no cover
    _train_import_error = _e

    def run_training(*_args, **_kwargs):
        raise ImportError(
            "run_training could not be imported. This usually means optional PyG "
            "dependencies (e.g. torch-sparse/pyg-lib) are missing."
        ) from _train_import_error

try:
    from .model_io import (
        load_model_checkpoint,
        load_full_run,
        load_training_state,
        save_model_checkpoint,
    )
except Exception as _e:  # pragma: no cover
    _model_io_import_error = _e

    def load_model_checkpoint(*_args, **_kwargs):
        raise ImportError("model_io could not be imported.") from _model_io_import_error

    def load_full_run(*_args, **_kwargs):
        raise ImportError("model_io could not be imported.") from _model_io_import_error

    def load_training_state(*_args, **_kwargs):
        raise ImportError("model_io could not be imported.") from _model_io_import_error

    def save_model_checkpoint(*_args, **_kwargs):
        raise ImportError("model_io could not be imported.") from _model_io_import_error

try:
    from .embed import export_embeddings, get_primary_embeddings, embed_with_graph
except Exception as _e:  # pragma: no cover
    _embed_import_error = _e

    def export_embeddings(*_args, **_kwargs):
        raise ImportError("embed could not be imported.") from _embed_import_error

    def get_primary_embeddings(*_args, **_kwargs):
        raise ImportError("embed could not be imported.") from _embed_import_error

    def embed_with_graph(*_args, **_kwargs):
        raise ImportError("embed could not be imported.") from _embed_import_error

try:
    from .loaders import make_link_loaders
except Exception as _e:  # pragma: no cover
    _loaders_import_error = _e

    def make_link_loaders(*_args, **_kwargs):
        raise ImportError(
            "make_link_loaders could not be imported. This usually means optional PyG "
            "dependencies (e.g. torch-sparse/pyg-lib) are missing."
        ) from _loaders_import_error

__all__ = [
    "load_hetero_pt",
    "load_imdb",
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
