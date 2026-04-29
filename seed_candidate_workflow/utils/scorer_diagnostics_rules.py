from __future__ import annotations

PROVENANCE_KEYS_DEFAULT: tuple[str, ...] = (
    "from_semantic",
    "from_rare_artifact",
    "from_2hop",
    "from_component",
    "source_count_eq_1",
    "source_count_eq_2",
    "source_count_ge_3",
    "same_seed_component_flag",
    "cross_seed_component_flag",
)

FEATURE_KEYS_DEFAULT: tuple[str, ...] = (
    "semantic_cosine_max",
    "rare_artifact_rarity_max",
    "twohop_rarity_max",
    "component_cosine_max",
    "time_gap_seconds_min",
)

SHARED_EVIDENCE_KEYS_DEFAULT: tuple[str, ...] = (
    "shared_url",
    "shared_sender",
    "shared_attachment",
    "shared_sender_domain",
    "shared_domain",
    "shared_stem",
)

BINARY_CONDITION_RULES_DEFAULT: tuple[str, ...] = (
    "from_semantic_AND_shared_sender",
    "from_semantic_AND_NOT_shared_sender",
    "from_2hop_AND_shared_sender",
    "from_2hop_AND_NOT_shared_sender",
    "from_component_AND_shared_sender",
    "from_component_AND_NOT_shared_sender",
    "from_semantic_AND_from_2hop",
    "from_semantic_AND_NOT_from_2hop",
    "from_2hop_AND_NOT_from_semantic",
    "from_component_AND_NOT_from_semantic",
    "from_component_AND_from_2hop",
    "shared_sender_AND_shared_stem",
    "shared_sender_AND_NOT_shared_stem",
    "shared_sender_domain_AND_NOT_shared_sender",
    "shared_sender_domain_AND_shared_sender",
)

SEMANTIC_BUCKET_RULES_DEFAULT: tuple[tuple[str, float | None, float | None], ...] = (
    ("semantic_lt_0_91", None, 0.91),
    ("semantic_0_91_to_0_93", 0.91, 0.93),
    ("semantic_0_93_to_0_95", 0.93, 0.95),
    ("semantic_ge_0_95", 0.95, None),
)

CANDIDATE_RULES_DEFAULT: tuple[str, ...] = (
    "likely_positive__from_semantic_AND_shared_sender",
    "likely_positive__from_semantic_AND_semantic_ge_0_93",
    "likely_positive__from_semantic_AND_shared_sender_AND_NOT_from_2hop",
    "likely_positive__shared_sender_AND_NOT_from_2hop",
    "likely_negative__from_2hop_AND_NOT_shared_sender",
    "likely_negative__from_2hop_AND_NOT_from_semantic",
    "likely_negative__from_component_AND_NOT_shared_sender",
    "likely_negative__shared_sender_domain_AND_NOT_shared_sender",
)

