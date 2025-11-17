# Copilot Instructions for GNN-Campaign-Detection

This repo builds a heterogeneous email graph from the TREC-07 dataset, optionally mirrors it to Memgraph, and lays groundwork for downstream GNN and clustering.

## Big Picture
- Pipeline stages (see `core/main.py`): CSV → MISP JSON → Hetero graph (PyG) → optional Memgraph mirror → GNN → clustering → metrics.
- Single source of truth for schema in `core/graph/graph_schema.py`. All builders render from a shared Graph IR produced by `core/graph/assembler.py`.
- Two graph backends:
  - PyTorch Geometric: `core/graph/graph_builder_pytorch.py` emits `HeteroData` and saves `.pt` + `.meta.json`.
  - Memgraph: `core/graph/graph_builder_memgraph.py` writes nodes/edges via Bolt (Neo4j driver) using the same schema.

## What the Graph Looks Like (current)
- Node types: `email`, `sender`, `receiver`, `week`, `url`, `domain`, `stem`, `email_domain`.
- Edge types:
  - `(email)-[has_sender]->(sender)` / `(email)-[has_receiver]->(receiver)`
  - `(email)-[in_week]->(week)`
  - `(email)-[has_url]->(url)` / `(url)-[has_domain]->(domain)` / `(url)-[has_stem]->(stem)`
  - `(sender|receiver)-[from_domain]->(email_domain)`
- Features & attrs (selected):
  - Email: `[len_body, n_urls, ts_minmax, len_body_z, len_subject, len_subject_z, optional TF-IDF of subject/body]`.
  - Domain/Stem/EmailDomain: small lexical vector via `compute_lexical_features`, docfreq counts.
  - Sender/Receiver/Url: simple numeric features plus docfreq where available.

## Key Conventions & Helpers
- Schema-driven: Update node/edge definitions in `graph_schema.py`; both PyG and Memgraph builders follow automatically via Graph IR.
- Graph IR: Assembled in `assembler.py` (indices, features, attributes, and edges). Don’t duplicate feature logic in builders.
- URL parsing & extraction: Use `core/utils/url_extractor.py` (`parse_url_components`, regex extraction). Avoid re-implementing.
- Dates → ISO week and UNIX ts: `core/graph/common.py` (`extract_week_key`, `to_unix_ts`).
- Freemail filter: `is_freemail_domain` prevents creating `email_domain` nodes for common freemail providers.
- Text vectors: TF‑IDF via scikit-learn is optional; tests accept empty vectors when sklearn is absent.

## Developer Workflows
- Environment (example, Windows/pwsh; see `venv_guide.txt`):
  ```pwsh
  conda create -n gnn-py310 python=3.10
  conda activate gnn-py310
  pip install -r requirements.txt
  ```
- Prepare data: ensure `data/csv/TREC-07.csv` exists. The converter filters `label == 1` and extracts URLs from body.
- Generate MISP JSON and build graph (runs two first stages by default):
  ```pwsh
  python -m core.main
  ```
  Outputs: `results/trec07_misp_hetero.pt` and `results/trec07_misp_hetero.meta.json`.
- Load into Memgraph (optional):
  ```pwsh
  docker compose -f memgraph-platform/docker-compose.yml up -d
  python -m core.main  # calls build_memgraph() after PyG build
  ```
  - Memgraph (Bolt) at `bolt://localhost:7687`, Memgraph Lab UI at `http://localhost:3000`.
  - Example queries: `memgraph-platform/queries.md`.
- Run tests (requires `data/misp/trec07_misp.json`; generate via `python -m core.main`):
  ```pwsh
  pytest -q
  ```
- Inspect graph metrics:
  ```pwsh
  python -m core.utils.graph_metrics results/trec07_misp_hetero.meta.json results/trec07_misp_hetero.pt
  ```

## Extending the Graph (pattern)
- Add a node/edge type: edit `graph_schema.py` first.
- Compute indices, features, and attrs in `assembler.py` (Graph IR). Prefer small numeric features; reuse `normalizer.py` and `common.py` helpers.
- Builders (PyG/Memgraph) will pick up the new types via the IR; avoid bespoke logic in builders unless necessary.
- Keep URL/email normalization in `common.py` to unify behavior across components.

## File Landmarks
- Orchestration: `core/main.py`.
- CSV → MISP: `core/misp/trec_to_misp.py` (uses `extract_urls_from_text`).
- Graph IR & schema: `core/graph/assembler.py`, `core/graph/graph_schema.py`.
- Backends: `core/graph/graph_builder_pytorch.py`, `core/graph/graph_builder_memgraph.py`.
- Utilities/metrics: `core/utils/url_extractor.py`, `core/utils/graph_metrics.py`.
- Tests: `core/tests/test_graph_schema_parity.py`, `core/tests/test_email_text_vectors.py`, `core/tests/test_url_extraction.py`.

## Notes
- `graph_info.md` describes broader/optional nodes/edges; current implementation reflects the schema above.
- If PyG or Torch are missing, import errors suggest installation commands. CPU-only wheels are fine for this pipeline.
