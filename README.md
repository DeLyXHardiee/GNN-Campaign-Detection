# GNN Campaign Detection

Research codebase for grouping phishing and scam emails into **campaigns** using graph structure, learned pair scoring, and community detection. The pipeline turns incident/MISP email data into a heterogeneous graph (emails linked to senders, URLs, domains, HTML fingerprints, return paths, and related artifacts), trains models that score whether two emails belong to the same campaign, and evaluates clusters against ground truth.

## What it does

1. **Ingest & preprocess** — Load emails from MISP/incident exports, parse bodies and headers, deduplicate identities, and build ground-truth campaign labels.
2. **Build graphs** — Construct PyTorch Geometric hetero graphs (`core/graph/`) with email-centric nodes and infrastructure/semantic edges.
3. **Seed–candidate workflow** — Generate candidate email pairs from shared evidence, train **positive–unlabeled (PU)** pair classifiers (explicit MLP and/or edge-GNN variants), score edges, and run **Louvain/Leiden** sweeps with expanded GT evaluation (`seed_candidate_workflow/`).
4. **Analyze** — Pair score separation, calibration, and community metrics (V-measure, homogeneity, completeness) for ablations and thesis experiments.

## Repository layout

| Path | Role |
|------|------|
| `core/` | Graph construction, GNN training, feature extraction, clustering utilities |
| `core/GNN/` | Pair-supervised training (`nnPU`), checkpoints under `output/runs/<run_id>/` |
| `seed_candidate_workflow/` | End-to-end experiment runner, graph bundles, scoring runs, configs |
| `data/` | MISP JSON, ground truth, dedup/collapse mappings (not committed in full) |
| `pipeline_config.json` | Central knobs: graph paths, training, PU prior, sampling, backends |
| `docs/` | Deeper pipeline notes and experiment write-ups |

## Quick start

**Environment** (Python 3.9+, GPU optional):

```bash
python -m venv .venv311
.venv311\Scripts\activate          # Windows
pip install torch
pip install torch-sparse torch_scatter --no-build-isolation
pip install -r requirements.txt
```

**Run an experiment** (graph setup → train → PU scoring → community sweep):

```bash
python seed_candidate_workflow/pipelines/run_experiment.py ^
  --config seed_candidate_workflow/configs/experiments/exp.best_b2_1to2_mlp.setup_gnn_score.json
```

Dry-run first to validate paths:

```bash
python seed_candidate_workflow/pipelines/run_experiment.py --config <config>.json --dry-run
```

**Pair score diagnostics** (after a training run exists):

```bash
python seed_candidate_workflow/scripts/run_pair_score_separation_analysis.py ^
  --run-dir output/runs/<run_id>/mlp ^
  --graph-pt core/graph/output/<graph_stem>_hetero.pt ^
  --gt-path data/groundtruth/ground_truth.json
```

## Further reading

- [Seed-candidate quick guide](seed_candidate_workflow/SEED_CANDIDATE_PIPELINE_QUICKGUIDE.md) — configs, output folders, modes
- [Best B2 (1:2) workflow](seed_candidate_workflow/docs/BEST_B2_1TO2_WORKFLOW.md) — current recommended training/scoring setup
- [Experiment runner](seed_candidate_workflow/docs/experiment_runner.md) — `setup_only` / `score_only` / `setup_gnn_score`
- [Graph schema](graph_info.md) — node and edge types
- [Pipeline overview](docs/PIPELINE_READABILITY_LAYER.md) — stage-by-stage data flow
