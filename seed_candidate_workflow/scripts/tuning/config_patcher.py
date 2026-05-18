"""Apply a sampled ``params`` dict to per-trial JSON config copies.

Each trial gets its own directory under ``output/tuning/configs/<trial_id>/``
containing patched copies of the anchor, seed, candidate, and experiment JSONs.
The experiment JSON's ``setup.paths`` is rewritten to reference these copies so
``run_experiment.py`` reads the per-trial configs without any global state.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from seed_candidate_workflow.scripts.tuning.search_space import (
    PARAM_SPECS,
    PatchTarget,
    ParamSpec,
    applied_value,
    enabled_specs,
    spec_by_name,
)


_PATH_SEGMENT_RE = re.compile(
    r"""
    (?P<name>[^.\[\]]+)              # plain key
    (?P<selector>\[[^\]]+\])?        # optional [name=foo] selector
    """,
    re.VERBOSE,
)


def _split_path(path: str) -> list[tuple[str, str | None]]:
    """Tokenize a dotted path. Each token is (key, selector?).

    Examples:
      ``a.b.c``                       -> [("a", None), ("b", None), ("c", None)]
      ``seeds.generators[name=foo].x`` -> [("seeds", None), ("generators", "name=foo"), ("x", None)]
    """
    out: list[tuple[str, str | None]] = []
    for chunk in path.split("."):
        m = _PATH_SEGMENT_RE.fullmatch(chunk)
        if not m:
            raise ValueError(f"Bad path segment: {chunk!r} in {path!r}")
        name = m.group("name")
        sel = m.group("selector")
        if sel is not None:
            sel = sel[1:-1]
        out.append((name, sel))
    return out


def _select_list_item(items: list[Any], selector: str) -> dict[str, Any]:
    """Resolve a ``[key=value]`` selector against a list of dicts."""
    if "=" not in selector:
        raise ValueError(f"Selector must be key=value, got {selector!r}")
    key, value = selector.split("=", 1)
    key, value = key.strip(), value.strip()
    for item in items:
        if not isinstance(item, dict):
            continue
        if str(item.get(key)) == value:
            return item
    raise KeyError(f"No list item with {key}={value!r}")


def set_path(obj: dict[str, Any], path: str, value: Any) -> None:
    """Set ``obj[path] = value`` in-place, where ``path`` supports
    dotted keys and ``[key=value]`` list selectors.

    Intermediate dicts are created if missing; list selectors require the
    list and the matching item to already exist (we never invent generators).
    """
    tokens = _split_path(path)
    cur: Any = obj
    for i, (name, sel) in enumerate(tokens):
        is_last = i == len(tokens) - 1
        if not isinstance(cur, dict):
            raise TypeError(f"Cannot descend into non-dict at {name!r} (path={path!r})")
        if name not in cur:
            if is_last and sel is None:
                cur[name] = value
                return
            cur[name] = {} if sel is None else []
        nxt = cur[name]
        if sel is not None:
            if not isinstance(nxt, list):
                raise TypeError(f"{name!r} is not a list but a selector was given (path={path!r})")
            picked = _select_list_item(nxt, sel)
            if is_last:
                # selector + last token doesn't quite make sense for our usage;
                # treat as picking the dict to assign into. Reject for safety.
                raise ValueError(f"Path ends on a list selector: {path!r}")
            cur = picked
        else:
            if is_last:
                cur[name] = value
                return
            cur = nxt


def get_path(obj: dict[str, Any], path: str, *, default: Any = None) -> Any:
    """Read a dotted/selector path; return ``default`` if anything is missing."""
    tokens = _split_path(path)
    cur: Any = obj
    for name, sel in tokens:
        if not isinstance(cur, dict) or name not in cur:
            return default
        cur = cur[name]
        if sel is not None:
            if not isinstance(cur, list):
                return default
            try:
                cur = _select_list_item(cur, sel)
            except KeyError:
                return default
    return cur


# ---------------------------------------------------------------------------
# Hashing helpers (used for stable trial / bundle IDs)
# ---------------------------------------------------------------------------


def _canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _short_sha(text: str, n: int = 12) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:n]


def bundle_hash(params: dict[str, Any]) -> str:
    """Hash only the bundle-affecting subset of ``params`` (stable across runs)."""
    subset: dict[str, Any] = {}
    for spec in enabled_specs():
        if not spec.bundle_affecting:
            continue
        if spec.name in params:
            subset[spec.name] = params[spec.name]
    return _short_sha(_canonical_json(subset), 12)


def full_hash(params: dict[str, Any]) -> str:
    return _short_sha(_canonical_json(params), 16)


# ---------------------------------------------------------------------------
# Per-trial config materialization
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MaterializedTrialConfig:
    """Paths and identifiers for one trial's on-disk config bundle."""

    trial_dir: Path
    experiment_config_path: Path
    anchor_config_path: Path
    seed_config_path: Path
    candidate_config_path: Path
    graph_id: str
    scoring_run_id: str
    gnn_run_id: str
    bundle_hash: str
    full_hash: str
    run_mode: str  # "setup_gnn_score" or "score_only"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _resolve_input(project_root: Path, raw: str | Path) -> Path:
    p = Path(str(raw)).expanduser()
    if not p.is_absolute():
        p = project_root / p
    return p.resolve()


def materialize_trial_config(
    *,
    project_root: Path,
    study_name: str,
    trial_number: int,
    params: dict[str, Any],
    base_experiment_config_path: Path,
    tuning_root: Path,
) -> MaterializedTrialConfig:
    """Materialize per-trial JSON copies and return a manifest of paths/IDs.

    Bundle reuse decision is made in :func:`run_hyperparam_tuning`; this
    function only stages files and computes IDs/hashes.
    """
    base_exp = _read_json(base_experiment_config_path)
    setup_paths = dict((base_exp.get("setup") or {}).get("paths") or {})
    anchor_src = _resolve_input(project_root, setup_paths.get("anchor_config", ""))
    seed_src = _resolve_input(project_root, setup_paths.get("seed_config", ""))
    cand_src = _resolve_input(project_root, setup_paths.get("candidate_config", ""))
    for label, src in (("anchor", anchor_src), ("seed", seed_src), ("candidate", cand_src)):
        if not src.is_file():
            raise FileNotFoundError(
                f"Base experiment config references missing {label}_config: {src}"
            )

    bhash = bundle_hash(params)
    fhash = full_hash(params)
    trial_id = f"t{trial_number:04d}__{fhash}"
    trial_dir = (tuning_root / "configs" / study_name / trial_id).resolve()
    trial_dir.mkdir(parents=True, exist_ok=True)

    anchor_cfg = copy.deepcopy(_read_json(anchor_src))
    seed_cfg = copy.deepcopy(_read_json(seed_src))
    cand_cfg = copy.deepcopy(_read_json(cand_src))
    exp_cfg = copy.deepcopy(base_exp)

    target_to_cfg: dict[PatchTarget, dict[str, Any]] = {
        "anchor_graph": anchor_cfg,
        "anchor_seed": seed_cfg,
        "anchor_candidate": cand_cfg,
        "experiment_scoring": exp_cfg,
        "experiment_community": exp_cfg,
    }
    apply_report: list[dict[str, Any]] = []
    for name, raw_value in params.items():
        spec = spec_by_name(name)
        if spec is None or not spec.enabled:
            apply_report.append({"name": name, "status": "skipped_unknown_spec"})
            continue
        target_cfg = target_to_cfg.get(spec.target)
        if target_cfg is None:
            raise ValueError(f"Unknown target {spec.target!r} on spec {name!r}")
        value = applied_value(spec, raw_value)
        set_path(target_cfg, spec.path, value)
        apply_report.append(
            {"name": name, "target": spec.target, "path": spec.path, "value": value}
        )

    anchor_out = trial_dir / "anchor_graph.json"
    seed_out = trial_dir / "anchor_seed.json"
    cand_out = trial_dir / "anchor_candidate.json"
    _write_json(anchor_out, anchor_cfg)
    _write_json(seed_out, seed_cfg)
    _write_json(cand_out, cand_cfg)

    graph_id = f"tuner_{study_name}__{bhash}"
    gnn_run_id = graph_id
    scoring_run_id = f"tuner_{study_name}__{trial_id}"

    exp_cfg.setdefault("experiment", {})
    exp_cfg["experiment"]["graph_id"] = graph_id
    exp_cfg["experiment"]["scoring_run_id"] = scoring_run_id
    # Mode is decided by the caller (after checking the bundle cache);
    # default to setup_gnn_score so a fresh bundle is built.
    exp_cfg["experiment"]["mode"] = "setup_gnn_score"

    setup_section = dict(exp_cfg.get("setup") or {})
    setup_paths = dict(setup_section.get("paths") or {})
    setup_paths["anchor_config"] = str(anchor_out)
    setup_paths["seed_config"] = str(seed_out)
    setup_paths["candidate_config"] = str(cand_out)
    setup_section["paths"] = setup_paths
    exp_cfg["setup"] = setup_section

    experiment_out = trial_dir / "experiment.json"
    _write_json(experiment_out, exp_cfg)
    _write_json(trial_dir / "_apply_report.json", {
        "trial_number": trial_number,
        "study_name": study_name,
        "params_raw": params,
        "bundle_hash": bhash,
        "full_hash": fhash,
        "applied": apply_report,
    })

    return MaterializedTrialConfig(
        trial_dir=trial_dir,
        experiment_config_path=experiment_out,
        anchor_config_path=anchor_out,
        seed_config_path=seed_out,
        candidate_config_path=cand_out,
        graph_id=graph_id,
        scoring_run_id=scoring_run_id,
        gnn_run_id=gnn_run_id,
        bundle_hash=bhash,
        full_hash=fhash,
        run_mode="setup_gnn_score",
    )


def switch_to_score_only(experiment_config_path: Path) -> None:
    """Rewrite an already-materialized experiment JSON to ``score_only`` mode."""
    cfg = _read_json(experiment_config_path)
    cfg.setdefault("experiment", {})
    cfg["experiment"]["mode"] = "score_only"
    _write_json(experiment_config_path, cfg)
