# MISP lake strict dedupe — downstream alignment

This note lists what must change after running [`data/misp/collapse_misp_lake_strict_duplicates.py`](../data/misp/collapse_misp_lake_strict_duplicates.py) so the pipeline stays consistent.

## 1. Produce artifacts

From the repo root (example paths):

```powershell
python data/misp/collapse_misp_lake_strict_duplicates.py `
  --input-json data/misp/incidents-lake-misp.json `
  --out-json data/misp/incidents-lake-misp.dedup_strict.json `
  --out-dir data/misp/misp_lake_dedup_strict `
  --ground-truth-in data/groundtruth/ground_truth.json `
  --ground-truth-out data/groundtruth/ground_truth.dedup_strict.json
```

Outputs:

- Deduped MISP array JSON (`--out-json`)
- `collapse_summary.json`, `collapse_manifest.json`, `collapsed_clusters.json`, `external_id_map.csv` (+ `.parquet` when pyarrow works)

## 2. `pipeline_config.json` paths

Point **both** MISP sources at the deduped file (see `run_preprocessing` / graph loader notes in [`core/main.py`](../core/main.py)):

- `preprocessing.misp_json_path` (if you use preprocessing to refresh exports)
- `graph.misp_json_path`
- `datasets.misp_json_path`

Set **`graph.hetero_graph_stem`** and **`graph.graph_pt_path_override`** in parallel to the baseline naming pattern, e.g. baseline `main_gnn_pu_1_no_ts` → dedup test **`main_gnn_pu_1_no_ts_dedup_strict`** so outputs are `core/graph/output/main_gnn_pu_1_no_ts_dedup_strict_hetero.pt` and `.meta.json`.

Set `datasets.featureset_base_name` to a new basename (e.g. `incidents-lake-misp-dedup-strict`) so cached feature-set artifacts do not silently reuse the old slice.

**Build graph from repo root** (with `core/main.py` entrypoint configured to call `run_graph_creation`):

```powershell
Set-Location "c:\Users\aar\Desktop\GNN-Campaign-Detection"
.\.venv311\Scripts\python.exe core\main.py
```

## 3. Ground truth

Use the remapped file from `--ground-truth-out` as `datasets.ground_truth_json`, or keep the original only if every `external_id` there still exists in the deduped MISP (generally false after collapse).

## 4. Regenerate downstream (order)

1. Heterogeneous graph: `core/graph/output/<stem>_hetero.pt` and `.meta.json` from the new MISP path.
2. Email features / embeddings that key on graph email order or `external_id`.
3. Seed/candidate workflow bundle for the graph stem (anchors, candidates, semantic shards as applicable).
4. `pair_training_dataset.csv` for that graph bundle.
5. Duplicate-pressure diagnostics: `seed_candidate_workflow/scripts/analyze_pair_training_duplicate_pressure.py` with the new pair CSV and refreshed `misp_loaded_external_ids` / cluster parquet if you re-ran `analyze_misp_duplicate_emails.py` on the deduped export.

Until steps 1–4 are rerun, pair-training duplicate-pressure numbers will still reflect the **old** multiplicity.
