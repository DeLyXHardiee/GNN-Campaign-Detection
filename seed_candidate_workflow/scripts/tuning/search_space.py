"""Search space for hyperparameter tuning.

A single source of truth for:

- which parameters are tunable
- their Optuna distribution (categorical / int / float, log scale, step)
- where each parameter lives in the per-stage JSON configs
- whether each parameter is "bundle-affecting" (changes the cached graph bundle
  + GNN training data) or "scoring-only" (re-uses an existing bundle)

To extend the search:

1. Add a :class:`ParamSpec` entry to :data:`PARAM_SPECS`.
2. Pick the right ``target`` so the patcher writes it into the correct file.
3. Set ``bundle_affecting`` to ``True`` if the bundle (or GNN training data)
   must be rebuilt when this knob changes; otherwise scoring/community-only
   trials can reuse a cached bundle.

The companion :mod:`config_patcher` consumes ``PARAM_SPECS`` to apply a
sampled ``params`` dict to per-trial JSON copies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal

import optuna


PatchTarget = Literal[
    "anchor_graph",
    "anchor_seed",
    "anchor_candidate",
    "experiment_scoring",
    "experiment_community",
]


@dataclass(frozen=True)
class ParamSpec:
    """One tunable hyperparameter."""

    #: Stable key under which the sampled value is recorded in the JSONL.
    name: str
    #: Which config file the value gets written into (or "experiment_*" for
    #: keys that live directly in the experiment JSON).
    target: PatchTarget
    #: Dotted path inside that file. List elements can be addressed via
    #: ``"seeds.generators[name=corroborated_v1].semantic_support.min_semantic_score"``;
    #: see :func:`config_patcher.set_path`.
    path: str
    #: Optuna distribution kind.
    kind: Literal["categorical", "int", "float", "loguniform"]
    #: For categorical/int/float. Always provide a finite range so resume from
    #: JSONL can validate.
    choices: tuple[Any, ...] | None = None
    low: float | int | None = None
    high: float | int | None = None
    step: float | int | None = None
    log: bool = False
    #: If True, changing this value invalidates a cached bundle (and the
    #: GNN trained on it). Scoring-only knobs (False) can reuse an existing
    #: bundle and just re-run the community sweep.
    bundle_affecting: bool = True
    #: Optional value mapper applied right before writing to the config
    #: (e.g. round to integer, clip, derive list, etc.).
    cast: Callable[[Any], Any] | None = None
    #: Whether the spec is currently enabled. Disabled specs are not
    #: suggested but still validated on resume so renames are detected.
    enabled: bool = True
    #: Free-form note shown in --list-params output.
    notes: str = ""


def _round_int(x: Any) -> int:
    return int(round(float(x)))


# ---------------------------------------------------------------------------
# Anchor graph channels (bundle-affecting)
# ---------------------------------------------------------------------------

ANCHOR_CHANNEL_SPECS: list[ParamSpec] = [
    ParamSpec(
        name="anchor.channels.semantic.top_k",
        target="anchor_graph",
        path="channels.channel_settings.semantic.top_k",
        kind="categorical",
        choices=(3000, 5000, 7500, 10000, 15000),
        bundle_affecting=True,
    ),
    ParamSpec(
        name="anchor.channels.semantic.min_cos",
        target="anchor_graph",
        path="channels.channel_settings.semantic.min_cos",
        kind="float",
        low=0.78,
        high=0.92,
        step=0.01,
        bundle_affecting=True,
    ),
    ParamSpec(
        name="anchor.channels.url.max_email_df",
        target="anchor_graph",
        path="channels.channel_settings.url.max_email_df",
        kind="categorical",
        choices=(1500, 2500, 3500, 5000),
        bundle_affecting=True,
    ),
    ParamSpec(
        name="anchor.channels.sender.max_email_df",
        target="anchor_graph",
        path="channels.channel_settings.sender.max_email_df",
        kind="categorical",
        choices=(1000, 2000, 3000, 4000),
        bundle_affecting=True,
    ),
    ParamSpec(
        name="anchor.channels.attachment.max_email_df",
        target="anchor_graph",
        path="channels.channel_settings.attachment.max_email_df",
        kind="categorical",
        choices=(750, 1500, 2500, 4000),
        bundle_affecting=True,
    ),
    ParamSpec(
        name="anchor.channels.domain.max_email_df",
        target="anchor_graph",
        path="channels.channel_settings.domain.max_email_df",
        kind="categorical",
        choices=(2000, 3000, 4500, 6000),
        bundle_affecting=True,
    ),
    ParamSpec(
        name="anchor.channels.stem.max_email_df",
        target="anchor_graph",
        path="channels.channel_settings.stem.max_email_df",
        kind="categorical",
        choices=(2000, 3000, 4500, 6000),
        bundle_affecting=True,
    ),
]


# ---------------------------------------------------------------------------
# Seed generation (corroborated_v1) — bundle-affecting
# ---------------------------------------------------------------------------

SEED_SPECS: list[ParamSpec] = [
    ParamSpec(
        name="seed.corroborated_v1.require_min_support_channels",
        target="anchor_seed",
        path="seeds.generators[name=corroborated_v1].require_min_support_channels",
        kind="categorical",
        choices=(2, 3),
        bundle_affecting=True,
    ),
    ParamSpec(
        name="seed.corroborated_v1.semantic_support.min_semantic_score",
        target="anchor_seed",
        path="seeds.generators[name=corroborated_v1].semantic_support.min_semantic_score",
        kind="float",
        low=0.85,
        high=0.97,
        step=0.01,
        bundle_affecting=True,
    ),
]


# ---------------------------------------------------------------------------
# Candidate generation — bundle-affecting
# ---------------------------------------------------------------------------

CANDIDATE_SPECS: list[ParamSpec] = [
    # semantic_reciprocal_v1
    ParamSpec(
        name="candidate.semantic_reciprocal_v1.semantic_top_k",
        target="anchor_candidate",
        path="candidates.generators[name=semantic_reciprocal_v1].config.semantic_top_k",
        kind="categorical",
        choices=(20, 50, 100, 200),
        bundle_affecting=True,
    ),
    ParamSpec(
        name="candidate.semantic_reciprocal_v1.semantic_min_cos",
        target="anchor_candidate",
        path="candidates.generators[name=semantic_reciprocal_v1].config.semantic_min_cos",
        kind="float",
        low=0.78,
        high=0.92,
        step=0.01,
        bundle_affecting=True,
    ),
    # rare_artifact_v1
    ParamSpec(
        name="candidate.rare_artifact_v1.min_artifact_idf",
        target="anchor_candidate",
        path="candidates.generators[name=rare_artifact_v1].config.min_artifact_idf",
        kind="float",
        low=0.3,
        high=1.2,
        step=0.05,
        bundle_affecting=True,
    ),
    ParamSpec(
        name="candidate.rare_artifact_v1.max_artifact_df",
        target="anchor_candidate",
        path="candidates.generators[name=rare_artifact_v1].config.max_artifact_df",
        kind="categorical",
        choices=(15, 25, 40, 60, 100),
        bundle_affecting=True,
    ),
    # component_expansion_v1
    ParamSpec(
        name="candidate.component_expansion_v1.min_artifact_idf",
        target="anchor_candidate",
        path="candidates.generators[name=component_expansion_v1].config.min_artifact_idf",
        kind="float",
        low=0.4,
        high=1.4,
        step=0.05,
        bundle_affecting=True,
    ),
    ParamSpec(
        name="candidate.component_expansion_v1.semantic_centroid_min_cos",
        target="anchor_candidate",
        path="candidates.generators[name=component_expansion_v1].config.semantic_centroid_min_cos",
        kind="float",
        low=0.78,
        high=0.92,
        step=0.01,
        bundle_affecting=True,
    ),
    ParamSpec(
        name="candidate.component_expansion_v1.semantic_email_cross_top_k",
        target="anchor_candidate",
        path="candidates.generators[name=component_expansion_v1].config.semantic_email_cross_top_k",
        kind="categorical",
        choices=(5, 10, 20, 40),
        bundle_affecting=True,
    ),
    # 2hop_bounded_v1
    ParamSpec(
        name="candidate.twohop_bounded_v1.max_total_pairs",
        target="anchor_candidate",
        path="candidates.generators[name=2hop_bounded_v1].config.max_total_pairs",
        kind="categorical",
        choices=(50_000, 100_000, 200_000, 400_000),
        bundle_affecting=True,
    ),
    ParamSpec(
        name="candidate.twohop_bounded_v1.semantic_min_cos",
        target="anchor_candidate",
        path="candidates.generators[name=2hop_bounded_v1].config.semantic_min_cos",
        kind="float",
        low=0.15,
        high=0.55,
        step=0.05,
        bundle_affecting=True,
    ),
]


# ---------------------------------------------------------------------------
# PU scorer params (live in the experiment JSON, scoring-only — bundle reusable)
# ---------------------------------------------------------------------------

SCORER_SPECS: list[ParamSpec] = [
    ParamSpec(
        name="scorer.pu.seed_edge_weight",
        target="experiment_scoring",
        path="scoring.params.pu.seed_edge_weight",
        kind="categorical",
        choices=(0.5, 1.0, 2.0, 5.0, 10.0),
        bundle_affecting=False,
    ),
    ParamSpec(
        name="scorer.pu.export_non_seed_min_pu_score",
        target="experiment_scoring",
        path="scoring.params.pu.export_non_seed_min_pu_score",
        kind="float",
        low=0.0,
        high=0.4,
        step=0.05,
        bundle_affecting=False,
    ),
]


# ---------------------------------------------------------------------------
# Community sweep (scoring-only)
# Note: the community sweep already searches over its internal grid and
# returns the best row. Tuning the GRID itself is rarely useful, but you may
# want to bias towards narrower/wider weight thresholds. Disabled by default.
# ---------------------------------------------------------------------------

COMMUNITY_SPECS: list[ParamSpec] = [
    ParamSpec(
        name="community.weight_threshold_max",
        target="experiment_community",
        path="community.sweep.weight_thresholds",
        kind="categorical",
        choices=(0.3, 0.4, 0.5, 0.6),
        bundle_affecting=False,
        enabled=False,
        notes="When enabled, picks the max value of the [0, w_max] threshold grid.",
        # The cast turns a single scalar into the actual list written to JSON.
        cast=lambda wmax: [round(0.0 + i * (float(wmax) / 4), 4) for i in range(5)],
    ),
]


# ---------------------------------------------------------------------------
# Combined registry — order matters for stable JSONL field ordering.
# ---------------------------------------------------------------------------

PARAM_SPECS: list[ParamSpec] = [
    *ANCHOR_CHANNEL_SPECS,
    *SEED_SPECS,
    *CANDIDATE_SPECS,
    *SCORER_SPECS,
    *COMMUNITY_SPECS,
]


def enabled_specs() -> list[ParamSpec]:
    return [s for s in PARAM_SPECS if s.enabled]


def spec_by_name(name: str) -> ParamSpec | None:
    for s in PARAM_SPECS:
        if s.name == name:
            return s
    return None


def bundle_affecting_names() -> list[str]:
    return [s.name for s in enabled_specs() if s.bundle_affecting]


def suggest_params(trial: "optuna.Trial") -> dict[str, Any]:
    """Sample one full parameter vector for ``trial``.

    Returns the **raw** sampled values (matching the Optuna distributions) so
    the JSONL roundtrips cleanly through ``study.add_trial`` on resume. The
    per-spec ``cast`` is applied later at config-write time
    (see :func:`config_patcher.apply_params`).
    """
    out: dict[str, Any] = {}
    for spec in enabled_specs():
        if spec.kind == "categorical":
            assert spec.choices is not None, f"{spec.name}: categorical needs choices"
            val: Any = trial.suggest_categorical(spec.name, list(spec.choices))
        elif spec.kind == "int":
            assert spec.low is not None and spec.high is not None
            val = trial.suggest_int(
                spec.name,
                int(spec.low),
                int(spec.high),
                step=int(spec.step) if spec.step is not None else 1,
                log=spec.log,
            )
        elif spec.kind == "float":
            assert spec.low is not None and spec.high is not None
            val = trial.suggest_float(
                spec.name,
                float(spec.low),
                float(spec.high),
                step=float(spec.step) if spec.step is not None and not spec.log else None,
                log=spec.log,
            )
        elif spec.kind == "loguniform":
            assert spec.low is not None and spec.high is not None
            val = trial.suggest_float(spec.name, float(spec.low), float(spec.high), log=True)
        else:
            raise ValueError(f"Unknown spec kind {spec.kind!r} for {spec.name!r}")
        out[spec.name] = val
    return out


def applied_value(spec: ParamSpec, raw_value: Any) -> Any:
    """Return the value that should actually be written into the config file."""
    return spec.cast(raw_value) if spec.cast is not None else raw_value


def distribution_for(spec: ParamSpec) -> optuna.distributions.BaseDistribution:
    """Return the Optuna distribution matching ``spec`` (for ``study.add_trial``)."""
    if spec.kind == "categorical":
        assert spec.choices is not None
        return optuna.distributions.CategoricalDistribution(list(spec.choices))
    if spec.kind == "int":
        assert spec.low is not None and spec.high is not None
        return optuna.distributions.IntDistribution(
            low=int(spec.low),
            high=int(spec.high),
            step=int(spec.step) if spec.step is not None else 1,
            log=spec.log,
        )
    if spec.kind == "float":
        assert spec.low is not None and spec.high is not None
        return optuna.distributions.FloatDistribution(
            low=float(spec.low),
            high=float(spec.high),
            step=float(spec.step) if spec.step is not None and not spec.log else None,
            log=spec.log,
        )
    if spec.kind == "loguniform":
        assert spec.low is not None and spec.high is not None
        return optuna.distributions.FloatDistribution(
            low=float(spec.low), high=float(spec.high), log=True
        )
    raise ValueError(f"Unknown spec kind {spec.kind!r}")
