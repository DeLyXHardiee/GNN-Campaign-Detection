"""
Extract body and subject BERT (SBERT) embeddings from an existing graph file in
graph/output and store them in a JSON file. The script opens and parses the .pt
graph file directly. The graph must have been built with SBERT email features.
The cache can be parsed later when creating the graph to avoid recomputing embeddings.

Usage:
  python -m graph.extract_bert_embeddings <graph_path> [output_path]
  python -m graph.extract_bert_embeddings --graph path/to/graph.pt --output path/to/embeddings.json

Output format (JSON) for later parsing:
  {
    "model": "intfloat/multilingual-e5-large",
    "n_emails": int,
    "subj_dim": int,
    "body_dim": int,
    "subj_vecs": [[float, ...], ...],   // one vector per email, index = email node index
    "body_vecs": [[float, ...], ...]
  }
  Use load_embeddings(path) to read this file from code.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

SBERT_MODEL_NAME = "intfloat/multilingual-e5-large"
_SCALAR_COUNT = 4  # ts, len_body, n_urls, len_subject
_HTML_CSS_LEN = 40  # len(create_html_css_features({}, {}))
_EMAIL_NODE_TYPE = "email"


def _infer_text_dims(total_dim: int) -> tuple[int, int]:
    """Infer subj_dim and body_dim from total email feature dimension.

    Layout: [scalars (4), subject_emb, body_emb, html_css]. Subject and body
    use the same SBERT model so subj_dim == body_dim when both present.
    """
    text_dim = total_dim - _SCALAR_COUNT - _HTML_CSS_LEN
    if text_dim <= 0:
        return 0, 0
    half = text_dim // 2
    return half, half


def extract_embeddings_from_graph(graph_path: str | Path) -> tuple[list[list[float]], list[list[float]], int, int]:
    """Load the graph file directly and extract subject/body embedding slices from email node features.

    Layout in email.x: [ts, len_body, n_urls, len_subject, SBERT(subject), SBERT(body), html_css].
    Returns (subj_vecs, body_vecs, subj_dim, body_dim).
    """
    graph = torch.load(str(graph_path), weights_only=False)
    if _EMAIL_NODE_TYPE not in getattr(graph, "node_types", []):
        return [], [], 0, 0
    store = graph[_EMAIL_NODE_TYPE]
    if not hasattr(store, "x") or store.x is None or store.x.numel() == 0:
        return [], [], 0, 0

    x = store.x
    n_emails = x.shape[0]
    total_dim = x.shape[1]
    subj_dim, body_dim = _infer_text_dims(total_dim)
    if subj_dim <= 0 and body_dim <= 0:
        return [], [], 0, 0

    start_subj = _SCALAR_COUNT
    end_subj = start_subj + subj_dim
    start_body = end_subj
    end_body = start_body + body_dim

    subj_vecs: list[list[float]] = []
    body_vecs: list[list[float]] = []
    for i in range(n_emails):
        row = x[i]
        if subj_dim > 0:
            subj_vecs.append(row[start_subj:end_subj].tolist())
        if body_dim > 0:
            body_vecs.append(row[start_body:end_body].tolist())

    return subj_vecs, body_vecs, subj_dim, body_dim


def extract_and_save(graph_path: str | Path, output_path: str | Path) -> Path:
    """Load graph from graph/output, extract BERT embeddings, and write JSON cache."""
    graph_path = Path(graph_path)
    if not graph_path.exists():
        raise FileNotFoundError(f"Graph file not found: {graph_path}")

    print(f"Loading graph from {graph_path}...")
    subj_vecs, body_vecs, subj_dim, body_dim = extract_embeddings_from_graph(graph_path)
    n = len(subj_vecs) if subj_vecs else len(body_vecs)
    if n == 0:
        raise ValueError("No email node features found in graph (empty or no text embeddings).")

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": SBERT_MODEL_NAME,
        "n_emails": n,
        "subj_dim": subj_dim,
        "body_dim": body_dim,
        "subj_vecs": subj_vecs,
        "body_vecs": body_vecs,
    }
    with out.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=0, separators=(",", ":"))
    print(f"Wrote {out} (n_emails={n}, subj_dim={subj_dim}, body_dim={body_dim})")
    return out


def load_embeddings(path: str | Path) -> dict:
    """Load a previously saved embeddings JSON for use in the graph assembler.

    Returns dict with keys: model, n_emails, subj_dim, body_dim, subj_vecs, body_vecs.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Embeddings file not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract body and subject BERT embeddings from an existing graph file and save to JSON."
    )
    parser.add_argument(
        "graph_path",
        nargs="?",
        help="Path to saved graph file (.pt) in graph/output.",
    )
    parser.add_argument(
        "output_path",
        nargs="?",
        help="Output JSON path. Default: graph/embeddings/output/email_bert_embeddings.json",
    )
    parser.add_argument("--graph", dest="graph_opt", help="Path to graph .pt file (alternative to positional).")
    parser.add_argument("--output", dest="output_opt", help="Output JSON path (alternative to positional).")
    args = parser.parse_args()

    graph_path = args.graph_opt or args.graph_path
    if not graph_path:
        parser.error("Provide graph_path (positional or --graph).")
    out_path = args.output_opt or args.output_path
    if not out_path:
        _graph_dir = Path(__file__).resolve().parent.parent
        out_path = _graph_dir / "embeddings" / "output" / "email_bert_embeddings.json"

    extract_and_save(graph_path, out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
