"""
Configuration for Method 1 Version 2: unsupervised shard-edge plausibility (MLP).

No labels, no edge_weight as model input, no blend with handcrafted weights at inference.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class EdgePlausibilityV2Config:
    """Hyperparameters for training and scoring."""

    random_seed: int = 0
    run_id: str = "v2_default"

    # Model
    hidden_dim: int = 64
    hidden_dim2: int = 32
    activation: str = "gelu"  # "gelu" | "relu"

    # Training
    epochs: int = 30
    batch_size: int = 512
    learning_rate: float = 1e-3
    validation_fraction: float = 0.1
    early_stop_patience: int = 10
    lr_reduce_patience: int = 3
    lr_reduce_factor: float = 0.5
    min_learning_rate: float = 1e-5
    show_progress: bool = True
    # When True, save ``checkpoints/epoch_XXXX.pt`` after each epoch (for post-hoc diagnostics).
    save_every_epoch_checkpoint: bool = False
    ranking_margin: float = 0.1
    # Larger margin for safe_pos > hard_neg_hsli pairs (HS-LI false-bridge focus).
    ranking_margin_hsli: float = 0.18
    loss_weight_ranking: float = 1.0
    loss_weight_stability: float = 0.0
    loss_weight_hub: float = 0.05
    loss_weight_agreement_aux: float = 0.00  # tiny; ranking remains primary
    # Anti-compression: mild penalty when batch score std falls below a target (reduces late collapse).
    anti_compress_enabled: bool = True
    loss_weight_anti_compress: float = 0.08
    anti_compress_target_std: float = 0.08
    anti_compress_eps: float = 1e-5

    # Pair sampling (V2.1 precision + HS-LI split: dominant safe_pos > hard_neg_hsli)
    n_ranking_pairs_per_batch: int = 256
    quantile_bins: int = 5
    fraction_endpoint_hard_pairs: float = 0.35
    # ``buckets`` = precision-oriented multi-evidence ranking (default); ``legacy_teacher`` = agreement stratification.
    ranking_supervision_mode: str = "buckets"
    # Four-way mix (normalized to sum 1). Primary signal: safe positive vs HS-LI false-bridge.
    ranking_frac_pos_vs_hard_neg_hsli: float = 0.55
    ranking_frac_pos_vs_hard_neg_other: float = 0.15
    ranking_frac_pos_vs_strong_neg: float = 0.12
    ranking_frac_hard_neg_hsli_vs_strong_neg: float = 0.18
    bucket_min_per_class: int = 8
    bucket_relaxation_rounds: int = 8
    bucket_relaxation_factor: float = 0.92
    bucket_q_semantic_high: float = 0.65
    bucket_q_semantic_low: float = 0.35
    bucket_q_semantic_mid: float = 0.5
    bucket_q_view_infra_high: float = 0.72
    # Safe positives need view_infra at least this dataset quantile (multi-evidence / not sem-only).
    bucket_q_safe_min_infra: float = 0.58
    # False-bridge core: view_infra at or below this quantile value counts as "low infra" vs high semantic.
    bucket_q_false_bridge_max_infra: float = 0.40
    # Stricter "clearly low infra" cutoff for ``hard_neg_hsli`` core (quantile on view_infra; lower = tighter).
    bucket_q_hsli_core_infra_max: float = 0.32
    bucket_q_hub_high: float = 0.75
    bucket_q_hub_low_for_positive: float = 0.28
    bucket_q_local_support_high: float = 0.62
    bucket_q_local_support_mid: float = 0.45
    bucket_q_local_support_low: float = 0.4
    bucket_q_view_spread_low: float = 0.40
    # High spread → treat as unstable / low multi-view agreement (strong-negative cue).
    bucket_q_view_spread_high: float = 0.65
    bucket_q_mv_min_low: float = 0.35
    bucket_q_mv_min_floor_pos: float = 0.32
    bucket_q_idf_high: float = 0.65
    # Stricter IDF floor for optional "hard bridge" safe bucket (moderate sem, rare corroboration).
    bucket_q_idf_hard_bridge: float = 0.72
    bucket_q_agreement_backup_high: float = 0.72
    # When False (V2.1 default), teacher agreement is not used to define safe positives.
    bucket_use_backup_teacher: bool = False
    bucket_q_shared_url_high: float = 0.75
    # One-channel infra dominance (from features): high values → false-bridge cue when semantic is high.
    bucket_q_infra_dominance_high: float = 0.72

    # Perturbation
    feature_dropout_prob: float = 0.12
    feature_noise_std: float = 0.03
    view_dropout_prob: float = 0.2  # applied to 3-view block used for stability branch only if enabled
    use_view_dropout_in_stability: bool = True

    # Hub regularization: penalize high plausibility when infra dominance is high
    hub_dominance_threshold: float = 0.85

    # Paths (set by caller or notebook)
    output_root: str = "analysis/output/semantic_shard_edge_v2"

    # GT diagnostics only (no supervision): per-epoch ``diag_*`` gaps in ``training_history.json``
    # and ``v2_gt_score_separation*`` artifacts. When True, loads GT + shard assignments if not passed
    # to ``train_and_score_edge_plausibility`` (see ``gt_separation_*`` paths).
    log_gt_separation: bool = True
    gt_separation_gt_json: str | None = None
    gt_separation_assignments_csv: str | None = None
    # When set (e.g. same folder you pass to ``load_step2_artifacts``), also looks for
    # ``../semantic_shard_step1/semantic_shard_step1_assignments.csv`` next to that directory.
    gt_separation_step2_dir: str | None = None

    # Feature construction
    extra_edge_columns_exclude: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> EdgePlausibilityV2Config:
        d = dict(d)
        if "extra_edge_columns_exclude" in d and isinstance(d["extra_edge_columns_exclude"], list):
            d["extra_edge_columns_exclude"] = tuple(d["extra_edge_columns_exclude"])
        # Migrate saved configs from pre–HS-LI-split pair fractions
        if "ranking_frac_pos_vs_hard_neg_hsli" not in d and "ranking_frac_pos_vs_hard_neg" in d:
            oh = float(d["ranking_frac_pos_vs_hard_neg"])
            d["ranking_frac_pos_vs_hard_neg_hsli"] = 0.70 * oh
            d["ranking_frac_pos_vs_hard_neg_other"] = 0.30 * oh
        if "ranking_frac_hard_neg_hsli_vs_strong_neg" not in d and "ranking_frac_hard_vs_strong_neg" in d:
            d["ranking_frac_hard_neg_hsli_vs_strong_neg"] = float(d["ranking_frac_hard_vs_strong_neg"])
        d.pop("ranking_frac_pos_vs_hard_neg", None)
        d.pop("ranking_frac_hard_vs_strong_neg", None)
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def output_dir(self) -> Path:
        return Path(self.output_root).expanduser().resolve() / self.run_id
