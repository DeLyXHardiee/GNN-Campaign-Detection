# Cluster visualization

After GNN and/or feature-set clustering, the pipeline can write a combined JSON file and start a small web UI (Docker) to inspect campaigns and emails.

## Artifacts

- `output/runs/<run>/clustering/campaigns_gnn.json` — best-algorithm campaign membership from GNN clustering.
- `output/runs/<run>/featureset_clustering/campaigns_featureset.json` — best configuration from the feature-set grid search.
- `output/runs/<run>/visualization/data.json` — merged payload for the UI (campaigns + email text from MISP).

## Pipeline step

Call `visualize_clusters()` from `core/main.py` (included at the end of `run_pipeline()`), or run it manually after clustering:

```python
from core.main import visualize_clusters
visualize_clusters()  # uses session run dir and MISP path from pipeline_config.json
```

Optional arguments:

- `run_dir=` — absolute path to an existing run folder (overrides session dir).
- `run_id=` — folder name or unique prefix under `output_runs_root` (e.g. `my_experiment` or `my_experiment (1)`). Do not pass both `run_dir` and `run_id`.
- `misp_json_path=` — override MISP JSON for email text.
- `include_attribute_similarity=` — set `False` to skip SBERT similarity (faster, smaller JSON).

## Attribute similarity (red → green)

When enabled (default), `visualization/data.json` includes `attribute_similarity` for the GNN and feature-set tabs. For each email in a campaign, **subject**, **body**, **senders**, **receivers**, and **date** are embedded with the same multilingual-e5 model as graph build (`passage:` prefix). The underlying metric is cosine similarity between that field’s vector and the **mean of the other members’** vectors in the same campaign. Raw scores in a tight cluster are often all very high (≈0.95+), so the UI uses **min–max scaling per campaign and per attribute** so the red→green range reflects **relative** differences among emails in that campaign. Requires `sentence-transformers` and the model download at runtime when building data.

## Configuration (`pipeline_config.json`)

```json
"visualization": {
  "enabled": true,
  "port": 8787,
  "compose_file": "docker-compose.visualization.yml",
  "include_attribute_similarity": true
}
```

Set `"enabled": false` to only write `visualization/data.json` without starting Docker. Set `"include_attribute_similarity": false` to skip SBERT similarity scores (smaller, faster JSON build).

## Docker Compose

From the repository root (after `data.json` exists under the run):

```bash
export RUN_DIR=/absolute/path/to/output/runs/your_run
export VIZ_PORT=8787
docker compose -f docker-compose.visualization.yml up --build
```

Open `http://localhost:8787/`. The **GNN** and **Feature set** tabs appear only when the corresponding campaign file exists and has at least one campaign.
