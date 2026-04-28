# Seed-Candidate Pipeline Quick Guide

This is a fast start guide for teammates who already know the method and just need to run, tweak configs, and find outputs.

## One command entrypoint

Use the experiment runner:

`python analysis/pipelines/run_experiment.py --config analysis/configs/experiments/<config>.json`

## Core idea

The run has two phases:

- graph setup (build/reuse artifacts under `graph_bundles`)
- scoring + community detection (write results under `scoring_runs`)

Controlled by:

- `experiment.mode`: `setup_only`, `score_only`, `setup_and_score`

## Where artifacts go

Configured by `artifacts` in your experiment config:

- graph bundle root: `artifacts.graph_bundle_root/<graph_id>/`
- scoring run root: `artifacts.scoring_output_root/<scoring_run_id>/`

Inside graph bundle (only enabled pieces are created):

- `anchor/`
- `seed/`
- `candidate/`
- `seed_candidate/`
- `pair_training/`
- `semantic_shard/` (if used)

Inside scoring run:

- `<target>/community/` (community sweeps + best files)
- `run_manifest.json` (full run record)

## Config fields you will edit most

In `analysis/configs/experiments/*.json`:

- `experiment.graph_id`
  - names/reuses the graph bundle folder
- `experiment.scoring_run_id`
  - names the scoring output folder
- `experiment.mode`
  - `setup_only`, `score_only`, `setup_and_score`
- `selection.score_targets`
  - what to run community/scoring on (`anchor`, `seed`, `candidate`, `seed_candidate`, `semantic_shard`)
- `scoring.score_mode`
  - `none`, `seed_candidate_handcrafted_v1`, `seed_candidate_pu_v1`, etc.
- `setup.enable`
  - toggle build stages (`anchor`, `seed`, `candidate`, `seed_candidate`, `pair_training`, `semantic_shard`)
- `setup.policy`
  - `on_missing`: `build` or `fail`
  - `on_present`: `reuse` or `rebuild`
- `community.sweep`
  - clustering method/sweep settings and ranking metric

## Which config to run (common cases)

- PU end-to-end:
  - `analysis/configs/experiments/exp03.seedcand.setupscore.pu.json`
- Seed-candidate unweighted baseline:
  - `analysis/configs/experiments/exp01.seedcand.setupscore.unweighted.json`
- Seed-candidate handcrafted score-only on existing base bundle:
  - `analysis/configs/experiments/exp02.seedcand.scoreonly.handcrafted_on_base.json`
- Anchor unweighted score-only on PU bundle:
  - `analysis/configs/experiments/exp04.anchor.scoreonly.unweighted_on_seedcand_pu.json`
- Seed-candidate unweighted score-only on PU bundle:
  - `analysis/configs/experiments/exp05.seedcand.scoreonly.unweighted_on_seedcand_pu.json`
- Seed-candidate handcrafted score-only on PU bundle:
  - `analysis/configs/experiments/exp06.seedcand.scoreonly.handcrafted_on_seedcand_pu.json`

## Copy/paste commands

PU full run:

`python analysis/pipelines/run_experiment.py --config analysis/configs/experiments/exp03.seedcand.setupscore.pu.json`

Unweighted seed-candidate full run:

`python analysis/pipelines/run_experiment.py --config analysis/configs/experiments/exp01.seedcand.setupscore.unweighted.json`

Handcrafted score-only (reuses existing graph bundle):

`python analysis/pipelines/run_experiment.py --config analysis/configs/experiments/exp02.seedcand.scoreonly.handcrafted_on_base.json`

Dry-run path check (no execution):

`python analysis/pipelines/run_experiment.py --config analysis/configs/experiments/exp03.seedcand.setupscore.pu.json --dry-run`

## Typical workflow for new experiment variants

1) Copy the closest config in `analysis/configs/experiments/`.
2) Change `experiment.graph_id` and `experiment.scoring_run_id`.
3) Set `experiment.mode`:
   - first run: `setup_and_score`
   - rerun with new scoring only: `score_only`
4) Set `selection.score_targets` and `scoring.score_mode`.
5) Tune `community.sweep` values.
6) Run with `--dry-run` once, then run normally.

## Score mode and target compatibility

Use compatible pairings:

- `seed_candidate_pu_v1` -> `seed_candidate`
- `seed_candidate_handcrafted_v1` -> `seed_candidate`
- `semantic_shard_handcrafted_v1` -> `semantic_shard`
- `semantic_shard_affine_v1` -> `semantic_shard`
- `none` -> unweighted topology baseline

If an incompatible target/mode is selected, config validation will fail fast.

## Quick troubleshooting

- Missing file in score-only run:
  - check that `experiment.graph_id` points to an existing bundle folder.
- Unexpected reuse/rebuild behavior:
  - check `setup.policy.on_present`.
- Outputs not where expected:
  - confirm `artifacts.graph_bundle_root` and `artifacts.scoring_output_root`.
- Unsure what the run resolved to:
  - open `<scoring_run_dir>/run_manifest.json`.
