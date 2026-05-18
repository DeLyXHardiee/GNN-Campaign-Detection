"""Hyperparameter tuning helpers for seed/candidate graph generation.

The tuner sits on top of ``seed_candidate_workflow/pipelines/run_experiment.py``,
sampling parameter dicts via Optuna TPE, patching the experiment config + the
underlying anchor/seed/candidate JSONs, invoking the runner per trial, and
appending one JSONL row per finished trial that doubles as the checkpoint.
"""
