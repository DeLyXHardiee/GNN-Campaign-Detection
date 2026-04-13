# Pipeline Readability Layer

Implementation-faithful overview of how the ML pipeline executes, which artifacts move between stages, and which configs/parameters control each step. Grounded in `core/main.py`, `core/GNN/*`, `config/pipeline_config.py`, `pipeline_config.json`, and related modules.

## 1) PIPELINE FLOW (HIGH LEVEL)

- **Primary orchestrator (`core/main.py`):** `run_pipeline()` executes:
  - `run_preprocessing_lake()` → stream incidents from lake API and write MISP JSON.
  - `create_feature_sets()` → generate FS1..FS7 feature-set JSON files from MISP events.
  - `run_featureset_clustering()` → run DBSCAN/MeanShift/HDBSCAN sweeps on feature-set embeddings.
  - `run_graph_creation(misp_json_path)` → build PyG hetero graph (`.pt` + `.meta.json`), optional Memgraph load.
  - `run_gnn()` → train GNN and save checkpoints/metrics/config under run directory.
  - `run_gnn_evaluation()` → AUROC/AP + Recall@K evaluation from graph + checkpoint.
  - `run_gnn_clustering()` → clustering sweeps on learned email embeddings; select locked params; write `campaigns_gnn.json`.
  - `run_metric_comparison()` → compare feature-set vs GNN campaign outputs against ground truth.
  - `visualize_clusters()` → build visualization JSON and optionally start Docker UI.

- **Current script entry behavior (`if __name__ == "__main__"` in `core/main.py`):**
  - Runs: `create_feature_sets()` → `run_featureset_clustering()` → `run_graph_creation(misp_path, to_memgraph=False)` → `run_gnn()` → `run_gnn_evaluation()` → `run_gnn_clustering()` → `run_metric_comparison()` → `visualize_clusters()`.
  - Uses hardcoded `misp_path = "preprocessing/output/incidents-lake-misp.json"` in that block.
  - `run_preprocessing_lake()` is commented out there.

- **Secondary/manual entry (`core/GNN/run_pipeline.py`):**
  - `main()` is structured with a large block of code stored as a triple-quoted string (inactive). The active tail calls `run_clustering_stage(...)` with symbols (`graph_path_str`, `run_dir_str`, `ground_truth_path_str`, etc.) that are only set inside that inactive/commented region—so this entrypoint expects the user to uncomment and wire path resolution before use. Train/eval calls remain commented in that block.

## 2) DATA FLOW

- **Raw incidents → MISP JSON**
  - `run_preprocessing_lake()` reads lake credentials (`LAKE_BASE_URL`, `LAKE_API_KEY`), table names, filters, limit.
  - Output: `preprocessing.misp_json_path` (current config: `core/preprocessing/output/incidents-lake-misp.json`).

- **MISP JSON → feature-set artifacts**
  - `create_feature_sets()` calls `run_featureset_extraction(misp_path=cfg.datasets.misp_json_path)`.
  - Outputs per input basename (`incidents-lake-misp`):
    - `core/feature_set_extraction/output/featuresets/incidents-lake-misp-FS1.json` … `-FS7.json`
    - helper caches: `core/feature_set_extraction/output/helpers/<base>_subject_idf.json`, `<base>_lsa.json`.

- **Feature-set JSONs → clustering outputs**
  - `run_featureset_clustering()` loads FS1..FS7 files and ground truth JSON.
  - Outputs under resolved run dir:
    - `featureset_clustering/results/dbscan_scores.txt`
    - `featureset_clustering/results/meanshift_scores.txt`
    - `featureset_clustering/results/hdbscan_scores.txt` (if enabled)
    - `featureset_clustering/campaigns_featureset.json` (best run only).

- **MISP JSON → hetero graph artifacts**
  - `run_graph_creation()` → `build_graph(...)`.
  - Outputs:
    - graph tensor: `<graph.output_dir>/<misp_base>_hetero.pt`
    - metadata: same basename `.meta.json` with node maps, feature shapes, `email_attrs.external_id`.
  - Optional additional sink: Memgraph via `build_memgraph(...)`.

- **Graph → model checkpoints + metrics**
  - `run_gnn()` → `run_train_stage()` → `run_training()`.
  - Outputs in run dir (`<output_runs_root>/<resolved_run_id>/`):
    - `<gnn.models_subdir>/best_model.pt` plus `model_epoch_<n>.pt`
    - `<gnn.metrics_csv>`
    - `<gnn.training_config_json>`
    - `<gnn.stage_result_json>`.

- **Graph + checkpoint → evaluation outputs**
  - AUROC/AP: `<run_dir>/<gnn.eval_auroc_ap_subdir>/...`
  - Recall@K: `<run_dir>/<gnn.eval_recall_at_k_subdir>/...`
  - Each writes `eval_config.json` + stage result JSON.

- **Graph + checkpoint + ground truth → GNN clustering outputs**
  - `run_gnn_clustering()` → per-algorithm sweep CSVs in `<run_dir>/<gnn.clustering_subdir>/<algo>/`.
  - Selects locked hyperparameter per algorithm; optionally runs locked param across epoch checkpoints.
  - Writes `<run_dir>/<gnn.clustering_subdir>/campaigns_gnn.json` + stage result JSON.
  - Optional plot stage writes PNGs under `<run_dir>/<gnn.clustering_subdir>/<gnn.clustering_plots_subdir>/`.

- **campaigns_featureset + campaigns_gnn + ground truth → comparison + viz**
  - `run_metric_comparison()` writes `<run_dir>/metric_comparison/comparison_summary.json`, `comparison_metrics.csv`, plots.
  - `visualize_clusters()` writes `<run_dir>/visualization/data.json`; may run Docker compose for UI.

## 3) CONFIG SURFACE (CRITICAL)

### Global/pipeline-level resolution

- `load_pipeline_config()` loads repo-root `pipeline_config.json`.
- Run directory allocation: `resolve_session_run_output_dir()`
  - precedence: env `PIPELINE_RUN_OUTPUT_DIR` → explicit run dir → cached session run dir → allocate unique under `output_runs_root`/`gnn.runs_parent`.
- Current top-level values (see `pipeline_config.json`):
  - `device: "cuda"`
  - `to_undirected: true`
  - `run_id: "test_metric_comparison"`
  - `output_runs_root: "output/runs"`.

### Preprocessing stage config (`run_preprocessing*`)

- `preprocessing`:
  - `incidents_csv_path`, `bodies_dir`, `misp_json_path`, `limit`, `category_allowlist`.
- `preprocessing_lake`:
  - `incidents_table`, `parsed_emails_table`, `start_date`, `end_date`, `limit`, `category_allowlist`.
- Current values include:
  - `misp_json_path`: `core/preprocessing/output/incidents-lake-misp.json`
  - `limit`: `0` (no row cap where `>0` triggers limit)
  - category allowlist: `["phishing", "scam"]`.

### Graph stage config (`run_graph_creation`)

- `graph` block → `graph_build_settings_from_pipeline()`:
  - `misp_json_path` (fallback to `datasets.misp_json_path`)
  - `max_misp_events` (positive int or null)
  - `output_dir`
  - `exclude_node_types` (list)
  - `degree_node_filter`: `enabled`, `strength [0,1]`, `target_node_types | null`, `min_degree`
  - `embeddings_output_dir | null`
  - `email_feature_projection`: `seed`, `bert_out_dim | null`, `other_out_dim | null`
  - `memgraph`: `enabled`, `uri`, `user`, `password`, `clear`, `create_indexes`.
- Current values:
  - `exclude_node_types: []`
  - `degree_node_filter.enabled: false`, `strength: 0.2`, `target_node_types: null`, `min_degree: 2`
  - `email_feature_projection: { seed: 42, bert_out_dim: 69, other_out_dim: 69 }`
  - `memgraph.enabled: false`
  - `max_misp_events: null`.

### GNN path/layout config

- `gnn` block used by `gnn_path_layout_from_pipeline()`:
  - `runs_parent`, `models_subdir`, `metrics_csv`, `training_config_json`,
  - `eval_auroc_ap_subdir`, `eval_recall_at_k_subdir`,
  - `clustering_subdir`, `clustering_plots_subdir`, `stage_result_json`.
- Current values:
  - runs parent effectively `output/runs` (top-level override)
  - models: `models`
  - metrics: `metrics.csv`
  - training config: `training_config.json`
  - clustering subdirs: `clustering`, plots `plots`.

### GNN training config (`training`)

- Passed into `run_training()` with explicit casting:
  - `torch_seed: 42`
  - `primary_ntype: "email"`
  - `hidden: 128`, `out_dim: 128`, `layers: 2`, `dropout: 0.3`
  - `neg_ratio: 1.0`, `batch_size: 256`, `fanout: [-1, -1]`
  - `val_ratio: 0.1`, `test_ratio: 0.1`, `epochs: 30`
  - `lr: 0.0005`, `wd: 0.0005`
  - `score_head: "dot"`
  - `early_stopping_patience: 5`
  - `lr_reduce_patience: 5`, `lr_reduce_factor: 0.5`, `lr_reduce_min: 0.0`
  - `supervised_edge_types: null`
  - `model_save_name: "best_model.pt"`
  - `contrastive_edges: null`, `contrastive_weight: 0.2`.
- Important hardcoded behavior:
  - `run_training()` resolves supervised edge types via `pick_supervised_edge_types(..., direction='both')` (hardcoded).

### GNN evaluation config (`evaluation`)

- AUROC/AP: `evaluation.auroc_ap` (currently `{}`; stage still writes this into `eval_config.json`).
- Recall@K (`evaluation.recall_at_k`):
  - `K_list: [1, 10, 20]`
  - `use_dot: true`.

### GNN clustering config (`gnn_clustering`)

- Selection thresholds:
  - `selection.min_coverage_ground_truth` (current `0.5`)
  - `selection.min_coverage_all` (current `0.5`, fallback to ground_truth threshold).
- Algorithm config (`gnn_clustering.config`):
  - DBSCAN: `enabled`, `epsilon_values`, `min_samples`
  - MeanShift: `enabled`, `quantile_values`, `n_samples`
  - HDBSCAN: `enabled`, `min_cluster_size_values`, optional `min_samples`.
- Current values:
  - DBSCAN `epsilon_values`: `[0.01, 0.05, 0.1, ..., 1.0]`, `min_samples: 5`
  - MeanShift `quantile_values`: `[0.01, 0.05, 0.1, ..., 1.0]`, `n_samples: 500`
  - HDBSCAN `min_cluster_size_values: [2]`, `min_samples` omitted in config (treated as `None`).

### Feature-set extraction + clustering config

- `datasets`:
  - `misp_json_path`: `core/preprocessing/output/incidents-lake-misp.json`
  - `ground_truth_json`: `data/groundtruth/ground_truth.json`
  - `featureset_base_name`: `incidents-lake-misp`.
- `featureset-clustering` block (preferred over legacy `clustering`):
  - shared: `max_tfidf_features`, `n_components_values`
  - outlier: `outlier_removal.enabled`, `outlier_removal.contamination`
  - dbscan: `eps_values`, `min_samples`
  - meanshift: `quantile_values`, `n_samples`
  - hdbscan: `enabled`, `min_cluster_size_values`, `min_samples`.
- Current values:
  - `max_tfidf_features: 500`
  - `n_components_values: [500]`
  - outlier removal disabled (`enabled: false`, contamination `0.05`)
  - DBSCAN eps sweep `[0.01..1.0]`, min_samples `5`
  - MeanShift quantile sweep `[0.01..1.0]`, n_samples `3000`
  - HDBSCAN enabled, min_cluster_size `[2]`, min_samples `null`.

### Visualization config

- `visualization`:
  - `enabled` (current `true`)
  - `port` (current `8787`)
  - `compose_file` (current `docker-compose.visualization.yml`)
  - `include_attribute_similarity` (current `true`).

## 4) GRAPH CONSTRUCTION DETAILS (`run_graph_creation`)

### Inputs

- Primary input: MISP JSON path
  - priority: function arg `misp_json_path` → `graph.misp_json_path` → `datasets.misp_json_path`.
- Optional in-memory input exists in lower-level `build_graph(misp_events=...)`, but `run_graph_creation` uses file path.
- Optional event cap: `max_misp_events` arg or config `graph.max_misp_events` (only applied when `>0`).

### Parameters controlling graph structure

- `exclude_nodes=settings.exclude_node_types` → removes entire node types + touching edges.
- `degree_node_filter=settings.degree_node_filter`:
  - active only when `enabled=true` and `strength>0`
  - supports `target_node_types` subset
  - removes high-degree nodes based on quantile threshold from `strength`
  - enforces `min_degree` floor.
- Node/edge schema from `DEFAULT_SCHEMA` (`core/graph/graph_schema.py`):
  - node types: `email`, `sender`, `receiver`, `url`, `domain`, `stem`, `email_domain`, `attachment`, `origin_ip`, `received_host`, `return_path_email`, `return_path_domain`
  - key edges include: `has_sender`, `has_receiver`, `has_url`, `has_domain`, `has_stem`, `sender_from_domain`, `receiver_from_domain`, plus attachment/origin/return-path edges.
- Collapse rules (`_collapse_graph_ir`) can merge child features into parent for 1-degree children per schema rules (implicit in `assemble_misp_graph_ir` + collapse loop).

### Feature engineering in graph build

- `parse_misp_events(...)` normalizes MISP attributes into:
  - `senders`, `receivers`, `urls`, `attachments`, `subject`, `body`, `html`, `css`, auth fields, tracking/cyrillic/symbol booleans, received hops, return path.
- Email raw feature vector assembled as:
  - scalars `[ts, len_body, n_urls, len_subject]`
  - SBERT(subject) + SBERT(body) from `utils.embeddings.get_embeddings` (cache-backed, model `intfloat/multilingual-e5-large`)
  - HTML/CSS vector (40 dims from `create_html_css_features`)
  - 7 boolean attrs
  - SPF/DKIM/DMARC one-hot (18 dims).
- Non-email node features include string length + optional lexical/docfreq attrs depending on node type.
- Email feature projection:
  - `EmailFeatureProjectionModule` applies linear projection to SBERT block and optional projection to structured block.
  - controlled by `email_feature_projection.seed`, `bert_out_dim`, `other_out_dim`.
  - current config sets both to 69, so output channel widths are explicit.
- Final graph normalized via `normalize_graph(data)` (details implicit in `graph/normalizer.py`).

### Optional behaviors

- **Memgraph output** (`build_memgraph`) controlled by `to_memgraph` arg or `graph.memgraph.enabled`.
  - Uses same assembled IR, same exclude/max_misp_events semantics.
  - Supports DB clear and index creation (`clear`, `create_indexes`).
  - Connection settings from args or `graph.memgraph.{uri,user,password}`.
- **Embeddings cache location** controlled by `graph.embeddings_output_dir`; defaults to `core/utils/embeddings/output` when null.

## 5) GNN SETUP (`run_gnn` + helpers)

### Model inputs

- Graph input path resolved by `resolve_gnn_paths(...)`:
  - explicit arg `graph_path` or default `default_hetero_graph_pt_path()` computed from current graph config.
- `load_hetero_pt(path, to_undirected=bool(cfg.to_undirected))`:
  - loads `HeteroData` from `.pt`
  - removes non-tensor `email.external_id` if present
  - applies `ToUndirected()` when `to_undirected=true`.

### Training parameters passed through

- `run_gnn()` passes `training_cfg` unchanged from config into `run_train_stage()`.
- `run_train_stage()` explicitly casts and forwards all training knobs into `run_training()`.
- `run_training()` sets optimizer `AdamW`, LR scheduler `ReduceLROnPlateau`, early stopping by validation loss, checkpoint every epoch 1 and multiples of 5 + best-model checkpoint.

### Device, directionality, paths

- Device:
  - `device_pref` from function arg override or `cfg.device` (current `cuda`)
  - resolved through `select_device(...)`.
- Directionality:
  - Graph direction conversion controlled by `cfg.to_undirected` (current `true`).
  - Supervised edge-direction selection is hardcoded `direction='both'` in `pick_supervised_edge_types`.
- Paths:
  - `run_dir` resolved by session allocator unless explicit
  - checkpoint default: `<run_dir>/<gnn.models_subdir>/<training.model_save_name>`
  - run folder naming from allocated run directory basename.

### Config vs hardcoded

- **From config:** device, to_undirected, all training/eval/clustering hyperparams, path layout names.
- **Hardcoded in code paths:**
  - supervised edge direction `'both'`
  - checkpoint-per-epoch save cadence (`epoch==1` or `%5==0`)
  - best model criterion = min validation loss
  - some default fallback filenames/subdirs if absent.

## 6) FEATURE SET + CLUSTERING PIPELINE

### Features extracted

- `run_featureset_extraction()` precomputes:
  - subject IDF cache (`*_subject_idf.json`)
  - LSA cache (`*_lsa.json`).
- Extracts 7 feature-set variants from same parsed MISP events:
  - `FS1`: `time, subject, body, origin, receiver, urls, attachments`
  - `FS2`: `time, subject, body, urls, origin, attachments` with omit keys incl. `sender_email`
  - `FS3`: `body, urls, origin` with omit list removing body scalar + LSA topics
  - `FS4`: `subject, body` with omit list removing URL/body scalar + subject TF-IDF summary keys
  - `FS5`: `subject, body, receiver, origin, urls, attachments` with broad omit list
  - `FS6`: `subject, time, body, origin, urls, attachments` with omit list
  - `FS7`: `subject, body, origin, urls` with omit keys `bow`, `sender_email`, `body`.
- Underlying per-email feature functions:
  - `extract_time_features`, `extract_subject_features`, `extract_body_based_features`,
  - `extract_origin_based_features`, `extract_recipient_based_features`,
  - `extract_url_based_features`, `extract_attachment_features`.

### Clustering methods and hyperparameters (feature-set stage)

- `run_featureset_clustering(...)` executes sweeps for:
  - DBSCAN (`run_db_scan_analysis`): `eps_values`, `min_samples`
  - MeanShift (`run_meanshift_analysis`): `quantile_values`, `n_samples`
  - HDBSCAN (`run_hdbscan_analysis`): `min_cluster_size_values`, `hdbscan_min_samples`
- Shared preprocessing knobs:
  - `max_tfidf_features`, `n_components_values` (TruncatedSVD),
  - `remove_outliers`, `outlier_contamination`,
  - `embeddings_output_dir`, `scaler_type="robust"`, `l2_normalize=True` (inside preprocessing).
- Best feature-set selection criterion:
  - prefer rows meeting both thresholds (`min_coverage_ground_truth`, `min_coverage_all`), then max `v_measure`.
- Threshold source for those coverage constraints:
  - from `gnn_clustering.selection` (reused by feature-set clustering wrapper).

### Clustering methods and hyperparameters (GNN stage)

- `run_clustering_stage()` reads `gnn_clustering.config` and runs enabled algorithms with their sweeps.
- For each algorithm, best locked param chosen by:
  - candidate rows with both coverage thresholds met (or fallback to all rows)
  - maximize `v_measure`.
- Locked-parameter checkpoint sweep:
  - scans `<models_dir>/model_epoch_*.pt`
  - re-runs clustering at locked param across epochs.
- Campaign export:
  - chooses best algorithm among locked params by highest `v_measure`
  - runs `fit_predict_labels(...)` and writes `campaigns_gnn.json`.

## 7) HIDDEN COMPLEXITY MAP

- `assemble_misp_graph_ir(...)`
  - **Does:** full MISP normalization → entity indexing → edge materialization → feature matrix assembly → collapse rules.
  - **Controlled by:** parsed MISP attributes, `DEFAULT_SCHEMA`, `embeddings_output_dir`, implicit collapse rules in schema.

- `parse_misp_events(...)`
  - **Does:** schema-driven extraction/coercion of MISP `Attribute` entries into normalized email dictionaries.
  - **Controlled by:** `DEFAULT_MISP_ATTRIBUTE_SCHEMA`, raw attribute `type/value`, auth parsing and URL side-effects.

- `build_graph(...)` / `build_hetero_graph_from_misp(...)`
  - **Does:** optional event truncation, node-type exclusion, degree filtering, email projection, graph normalization, persistence.
  - **Controlled by:** `exclude_nodes`, `degree_node_filter`, `max_misp_events`, `email_feature_projection`, `embeddings_output_dir`.

- `filter_graph_ir_by_degree(...)`
  - **Does:** quantile-threshold high-degree pruning with per-type reindexing of nodes/edges/email_attrs.
  - **Controlled by:** `strength`, `target_node_types`, `min_degree`, graph topology.

- `run_training(...)`
  - **Does:** split graph edges, create link loaders, build model+predictor, train/eval loop, scheduler, early stopping, checkpointing.
  - **Controlled by:** full `training_cfg`; plus hardcoded supervised edge direction `'both'`.

- `load_full_run(...)` and `load_model_checkpoint(...)` (implicit in `eval_stage_utils` and clustering helpers)
  - **Does:** reconstruct model/predictor/loaders/splits from checkpoint + graph metadata.
  - **Controlled by:** checkpoint file contents, graph metadata, selected device.

- `sweep_clustering_for_one_model(...)`
  - **Does:** extract email embeddings from model, run algorithm-specific sweeps, write per-model CSVs.
  - **Controlled by:** algorithm config (`epsilon_values`/`quantile_values`/`min_cluster_size_values`, min_samples, n_samples), ground truth labels.

- `run_locked_param_across_checkpoints(...)`
  - **Does:** fixes one selected hyperparameter and re-runs sweep across epoch checkpoints.
  - **Controlled by:** selected locked value, list of checkpoint files, clustering algorithm type.

- `preprocess_for_clustering(...)` (feature-set pipeline)
  - **Does:** mixed numeric/text/token/dict featurization, SBERT use for subject/body, sparse stacking, SVD, scaling, normalization.
  - **Controlled by:** `max_tfidf_features`, `n_components`, `token_list_fields`, `dict_feature_fields`, `scaler_type`, `l2_normalize`, `sbert_model_name`, `embeddings_output_dir`.

- `run_metric_comparison_for_run(...)`
  - **Does:** loads campaign artifacts, computes external metrics against ground truth, computes agreement (ARI/AMI), writes JSON/CSV/plots.
  - **Controlled by:** existence/content of `campaigns_featureset.json` and `campaigns_gnn.json`, `ground_truth_path`, `dpi`.

- `write_visualization_data_json(...)` (called by `visualize_clusters`; implementation in [`core/visualization/data_builder.py`](../core/visualization/data_builder.py))
  - **Does:** builds visualization payload joining run outputs with MISP source; optional attribute-similarity enrichment.
  - **Controlled by:** `run_dir`, `misp_json_path`, `include_attribute_similarity`.
