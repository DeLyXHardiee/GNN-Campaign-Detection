"""
Load fixed per-email feature vectors from the heterogeneous graph checkpoint.

Saved ``email.x`` is usually **already projected** in this repo:
``concat(bert_proj(subject+body SBERT) -> 128 dim, other_proj -> 32 dim)`` → 160 dims.

Use ``feature_mode="projected_bert128"`` for the semantic block only, or
``projected_full`` for all 160. If the tensor matches raw assembler layout (large dim),
``raw_subject_body`` slices the concatenated subject+body SBERT block (equal half split,
same heuristic as ``graph_builder_pytorch._infer_email_embedding_dims``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from analysis.utils.graph_structure_helpers import external_id_to_row, load_hetero, load_meta

# Match core.graph.feature_projection layout for *raw* email rows
_SCALAR = 4
_HTML = 40
_BOOL = 7
_AUTH = 18
_TRAIL = _BOOL + _AUTH
_OTHER_TAIL = _HTML + _TRAIL


def _raw_bert_span(total_dim: int) -> tuple[int, int] | None:
    """(start, end) exclusive for concatenated subject+body in raw layout, or None."""
    text_dim = total_dim - _SCALAR - _OTHER_TAIL
    if text_dim <= 1:
        return None
    return _SCALAR, _SCALAR + text_dim


def _is_likely_projected_email_x(dim: int) -> bool:
    return dim in (128, 160) or (dim > 32 and dim < 256)


def load_graph_email_features_for_external_ids(
    graph_pt: Path | str,
    meta_json: Path | str | None,
    external_ids: list[str],
    *,
    feature_mode: str = "auto",
    to_undirected: bool = True,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """
    Map ``external_ids`` (in order) to feature rows from ``data['email'].x``.

    Returns
    -------
    features : ndarray [N, D]
        Zero-filled for missing IDs (caller should filter if needed).
    present_mask : bool ndarray [N]
        True where ``external_id`` was found in the graph.
    info : dict
        Layout / mode used, input dim, missing count, etc.
    """
    graph_pt = Path(graph_pt)
    meta_path = Path(meta_json) if meta_json is not None else graph_pt.with_suffix(".meta.json")

    data = load_hetero(graph_pt, to_undirected=to_undirected)
    ntypes = list(getattr(data, "node_types", []))
    if "email" not in data.node_types:
        raise ValueError(
            f"HeteroData has no 'email' node type (node_types={ntypes}). "
            f"Check GRAPH_PT: expected the full incidents hetero checkpoint, not a shard subgraph export."
        )
    ex = data["email"].x
    if ex is None:
        raise ValueError(
            "email.x is None: this checkpoint has email nodes but no feature tensor. "
            "Rebuild/save the graph from the pipeline so email.x is populated, or use a different .pt."
        )

    x = ex.detach().cpu().float().numpy()
    row_map = external_id_to_row(load_meta(meta_path))
    D_in = x.shape[1]

    mode = feature_mode
    if mode == "auto":
        if _is_likely_projected_email_x(D_in):
            mode = "projected_bert128" if D_in >= 128 else "projected_full"
        else:
            mode = "raw_subject_body"

    slices: tuple[int, int]
    if mode == "projected_bert128":
        if D_in < 128:
            raise ValueError(f"email.x dim {D_in} < 128; cannot take bert128 block")
        slices = (0, 128)
    elif mode in ("projected_full", "full"):
        slices = (0, D_in)
    elif mode == "raw_subject_body":
        span = _raw_bert_span(D_in)
        if span is None:
            raise ValueError(
                f"raw_subject_body: cannot infer SBERT span from email.x dim {D_in} "
                "(expected raw assembler layout)"
            )
        slices = span
    else:
        raise ValueError(
            f"Unknown feature_mode={feature_mode!r}; use auto, projected_bert128, "
            "projected_full, raw_subject_body"
        )

    sub = x[:, slices[0] : slices[1]]
    D = sub.shape[1]

    n = len(external_ids)
    out = np.zeros((n, D), dtype=np.float32)
    mask = np.zeros((n,), dtype=bool)
    missing = 0
    for i, eid in enumerate(external_ids):
        eid = str(eid)
        if eid not in row_map:
            missing += 1
            continue
        r = row_map[eid]
        out[i] = sub[r].astype(np.float32, copy=False)
        mask[i] = True

    info: dict[str, Any] = {
        "graph_pt": str(graph_pt.resolve()),
        "meta_json": str(meta_path.resolve()),
        "email_x_shape": list(x.shape),
        "feature_mode_resolved": mode,
        "slice": list(slices),
        "output_dim": int(D),
        "n_requested": int(n),
        "n_present": int(mask.sum()),
        "n_missing": int(missing),
    }
    return out, mask, info
