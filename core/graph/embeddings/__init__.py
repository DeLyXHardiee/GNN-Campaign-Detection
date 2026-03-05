"""
Embedding component: create, load, and save subject/body BERT embeddings per email.

Runs independently (own output folder, CLI) or is used by the graph assembler.
Loads existing embeddings from cache and computes only missing ones per email.
"""
from __future__ import annotations

from pathlib import Path

from .embedder import (
    DEFAULT_OUTPUT_DIR,
    MODEL_NAME,
    get_embeddings,
    run_standalone,
)

__all__ = [
    "DEFAULT_OUTPUT_DIR",
    "MODEL_NAME",
    "get_embeddings",
    "run_standalone",
]
