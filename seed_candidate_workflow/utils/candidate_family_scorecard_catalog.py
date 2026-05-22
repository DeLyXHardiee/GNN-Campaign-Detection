"""
Curated candidate-family catalogs for scorecard screening (analysis-only).

``rich_v1`` expands semantic, path/template, body/subject, sender, support, 2-hop,
html-fingerprint, and time-aware families with small threshold grids.
"""

from __future__ import annotations

from typing import Any


def _fam(
    name: str,
    expr: str,
    category: str,
    *,
    mode: str = "hypothetical_add",
    provenance_column: str | None = None,
) -> dict[str, Any]:
    d: dict[str, Any] = {
        "family_name": name,
        "category": category,
        "mode": mode,
    }
    if provenance_column:
        d["provenance_column"] = provenance_column
    else:
        d["rule_expression"] = expr
    return d


def _grid_numeric(
    base: str,
    col: str,
    thresholds: list[float | str],
    category: str,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for thr in thresholds:
        thr_s = str(thr).replace(".", "_") if isinstance(thr, float) else str(thr)
        if base == col or not base:
            expr = f"{col}_ge_{thr_s}"
            name = expr
        else:
            name = f"{base}_ge_{thr_s}"
            expr = name
        out.append(_fam(name, expr, category))
    return out


def build_rich_v1_catalog() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Return (families, skipped_definitions).

    ``skipped_definitions`` lists families we cannot evaluate yet and why.
    """
    families: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    # --- A. Strong semantic support ---
    families.extend(
        [
            _fam("semantic_ge_0_93", "semantic_ge_0_93", "A_strong_semantic"),
            _fam("semantic_ge_0_93_AND_NOT_from_2hop", "semantic_ge_0_93_AND_NOT_from_2hop", "A_strong_semantic"),
            _fam(
                "semantic_ge_0_90_AND_support_count_excl_domain_and_root_stem_ge_1",
                "semantic_ge_0_90_AND_support_count_excl_domain_and_root_stem_ge_1",
                "A_strong_semantic",
            ),
        ]
    )
    families.extend(
        _grid_numeric(
            "semantic_ge_0_90_AND_path_token_jaccard_combined",
            "path_token_jaccard_combined",
            [0.2, 0.4, 0.6],
            "A_strong_semantic",
        )
    )
    families.extend(
        _grid_numeric(
            "semantic_ge_0_90_AND_sender_localpart_norm_jaccard",
            "sender_localpart_norm_jaccard",
            [0.5, 0.7, 0.9],
            "A_strong_semantic",
        )
    )
    families.extend(
        [
            _fam(
                "semantic_ge_0_90_AND_rarity_weighted_support_sum_ge_2",
                "semantic_ge_0_90_AND_rarity_weighted_support_sum_ge_2",
                "A_strong_semantic",
            ),
        ]
    )

    # --- B. Medium-semantic recovery ---
    families.extend(
        [
            _fam(
                "semantic_band_0_85_0_90_AND_support_count_excl_domain_and_root_stem_ge_1",
                "semantic_band_0_85_0_90_AND_support_count_excl_domain_and_root_stem_ge_1",
                "B_medium_semantic",
            ),
        ]
    )
    families.extend(
        _grid_numeric(
            "semantic_band_0_85_0_90_AND_path_token_jaccard_combined",
            "path_token_jaccard_combined",
            [0.2, 0.4, 0.6],
            "B_medium_semantic",
        )
    )
    families.extend(
        _grid_numeric(
            "semantic_band_0_85_0_90_AND_sender_localpart_norm_jaccard",
            "sender_localpart_norm_jaccard",
            [0.5, 0.7, 0.9],
            "B_medium_semantic",
        )
    )
    families.extend(
        _grid_numeric(
            "semantic_band_0_85_0_90_AND_body_token_jaccard",
            "body_token_jaccard",
            [0.15, 0.25, 0.35],
            "B_medium_semantic",
        )
    )
    families.extend(
        _grid_numeric(
            "semantic_band_0_85_0_90_AND_body_char4gram_jaccard",
            "body_char4gram_jaccard",
            [0.10, 0.17, 0.25],
            "B_medium_semantic",
        )
    )
    families.extend(
        _grid_numeric(
            "semantic_band_0_85_0_90_AND_subject_token_jaccard",
            "subject_token_jaccard",
            [0.15, 0.25, 0.35],
            "B_medium_semantic",
        )
    )

    # --- C. URL / path / template ---
    families.extend(
        _grid_numeric("path_token_jaccard_combined", "path_token_jaccard_combined", [0.2, 0.4, 0.6], "C_url_path")
    )
    families.extend(
        _grid_numeric("url_path_token_jaccard", "url_path_token_jaccard", [0.2, 0.4, 0.6], "C_url_path")
    )
    families.extend(
        _grid_numeric("stem_path_token_jaccard", "stem_path_token_jaccard", [0.2, 0.4, 0.6], "C_url_path")
    )
    families.extend(
        [
            _fam(
                "same_registrable_domain_AND_path_token_jaccard_combined_ge_0_4",
                "same_registrable_domain_AND_path_token_jaccard_combined_ge_0_4",
                "C_url_path",
            ),
            _fam(
                "shared_nontrivial_stem_AND_path_token_jaccard_combined_ge_0_4",
                "shared_nontrivial_stem_AND_path_token_jaccard_combined_ge_0_4",
                "C_url_path",
            ),
        ]
    )
    skipped.append(
        {
            "family_name": "same_query_key_set_or_overlap_ge_*",
            "category": "C_url_path",
            "reason": "query-key overlap feature not computed in scorecard enrichment yet",
        }
    )

    # --- D. Subject/body split ---
    families.extend(_grid_numeric("body_cosine", "body_cosine", [0.85, 0.90, 0.93], "D_subject_body"))
    families.extend(
        [
            _fam(
                "body_cosine_ge_0_90_AND_subject_cosine_lt_0_90",
                "body_cosine_ge_0_90_AND_subject_cosine_lt_0_90",
                "D_subject_body",
            ),
        ]
    )
    families.extend(_grid_numeric("body_token_jaccard", "body_token_jaccard", [0.15, 0.25, 0.35], "D_subject_body"))
    families.extend(
        _grid_numeric("body_char4gram_jaccard", "body_char4gram_jaccard", [0.10, 0.17, 0.25], "D_subject_body")
    )
    families.extend(
        [
            _fam(
                "subject_token_jaccard_ge_0_25_AND_body_token_jaccard_ge_0_25",
                "subject_token_jaccard_ge_0_25_AND_body_token_jaccard_ge_0_25",
                "D_subject_body",
            ),
            _fam(
                "subject_cosine_ge_0_90_AND_body_cosine_ge_0_90",
                "subject_cosine_ge_0_90_AND_body_cosine_ge_0_90",
                "D_subject_body",
            ),
        ]
    )
    families.extend(_grid_numeric("subject_cosine", "subject_cosine", [0.85, 0.90, 0.93], "D_subject_body"))

    # --- E. Sender / sender-family ---
    families.extend(
        [
            _fam("sender_exact_match", "sender_exact_match", "E_sender"),
            _fam("sender_localpart_exact_match", "sender_localpart_exact_match", "E_sender"),
            _fam(
                "sender_domain_exact_match_AND_semantic_ge_0_90",
                "sender_domain_exact_match_AND_semantic_ge_0_90",
                "E_sender",
            ),
            _fam(
                "sender_exact_match_AND_path_token_jaccard_combined_ge_0_4",
                "sender_exact_match_AND_path_token_jaccard_combined_ge_0_4",
                "E_sender",
            ),
        ]
    )
    families.extend(
        _grid_numeric(
            "sender_localpart_norm_jaccard",
            "sender_localpart_norm_jaccard",
            [0.5, 0.7, 0.9],
            "E_sender",
        )
    )

    # --- F. Support excluding weak channels ---
    families.extend(
        [
            _fam(
                "support_count_excl_domain_and_root_stem_ge_1",
                "support_count_excl_domain_and_root_stem_ge_1",
                "F_support",
            ),
            _fam(
                "support_count_excl_domain_and_root_stem_ge_2",
                "support_count_excl_domain_and_root_stem_ge_2",
                "F_support",
            ),
            _fam("strong_support_count_ge_1", "strong_support_count_ge_1", "F_support"),
            _fam(
                "shared_domain_without_other_support",
                "shared_domain_without_other_support",
                "F_support",
            ),
            _fam(
                "shared_url_or_stem_without_sender",
                "shared_url_or_stem_without_sender",
                "F_support",
            ),
        ]
    )
    families.extend(
        _grid_numeric(
            "rarity_weighted_support_sum",
            "rarity_weighted_support_sum",
            [1.0, 2.0, 3.0],
            "F_support",
        )
    )

    # --- G. 2-hop corroborated ---
    families.extend(
        [
            _fam("from_2hop_AND_shared_sender", "from_2hop_AND_shared_sender", "G_2hop"),
            _fam(
                "from_2hop_AND_semantic_ge_0_90_AND_NOT_twohop_via_html_fp",
                "from_2hop_AND_semantic_ge_0_90_AND_NOT_twohop_via_html_fp",
                "G_2hop",
            ),
            _fam(
                "from_2hop_AND_support_count_excl_domain_and_root_stem_ge_1",
                "from_2hop_AND_support_count_excl_domain_and_root_stem_ge_1",
                "G_2hop",
            ),
            _fam(
                "from_2hop_AND_same_seed_component_flag",
                "from_2hop_AND_same_seed_component_flag",
                "G_2hop",
            ),
            _fam(
                "from_2hop_AND_semantic_ge_0_90_AND_shared_sender",
                "from_2hop_AND_semantic_ge_0_90_AND_shared_sender",
                "G_2hop",
            ),
            _fam(
                "from_2hop_AND_semantic_ge_0_90_AND_shared_stem",
                "from_2hop_AND_semantic_ge_0_90_AND_shared_stem",
                "G_2hop",
            ),
        ]
    )
    for thr in (0.2, 0.4, 0.6):
        thr_s = str(thr).replace(".", "_")
        families.append(
            _fam(
                f"from_2hop_AND_path_token_jaccard_combined_ge_{thr_s}",
                f"from_2hop_AND_path_token_jaccard_combined_ge_{thr_s}",
                "G_2hop",
            )
        )

    # --- H. HTML fingerprint ---
    families.extend(
        [
            _fam("twohop_via_html_fp", "twohop_via_html_fp", "H_html_fp"),
            _fam(
                "twohop_via_html_fp_AND_semantic_ge_0_90",
                "twohop_via_html_fp_AND_semantic_ge_0_90",
                "H_html_fp",
            ),
            _fam(
                "twohop_via_html_fp_AND_shared_sender",
                "twohop_via_html_fp_AND_shared_sender",
                "H_html_fp",
            ),
            _fam(
                "twohop_via_html_fp_AND_same_seed_component_flag",
                "twohop_via_html_fp_AND_same_seed_component_flag",
                "H_html_fp",
            ),
            _fam("direct_shared_html_fp", "direct_shared_html_fp", "H_html_fp"),
            _fam(
                "direct_shared_html_fp_AND_semantic_ge_0_90",
                "direct_shared_html_fp_AND_semantic_ge_0_90",
                "H_html_fp",
            ),
        ]
    )

    # --- I. Time-aware ---
    for gap in ("3d", "7d", "14d"):
        families.extend(
            [
                _fam(
                    f"semantic_ge_0_90_AND_time_gap_le_{gap}",
                    f"semantic_ge_0_90_AND_time_gap_le_{gap}",
                    "I_time",
                ),
                _fam(
                    f"semantic_band_0_85_0_90_AND_shared_sender_AND_time_gap_le_{gap}",
                    f"semantic_band_0_85_0_90_AND_shared_sender_AND_time_gap_le_{gap}",
                    "I_time",
                ),
                _fam(
                    f"path_token_jaccard_combined_ge_0_4_AND_time_gap_le_{gap}",
                    f"path_token_jaccard_combined_ge_0_4_AND_time_gap_le_{gap}",
                    "I_time",
                ),
                _fam(
                    f"from_2hop_AND_time_gap_le_{gap}",
                    f"from_2hop_AND_time_gap_le_{gap}",
                    "I_time",
                ),
            ]
        )

    # --- Pipeline reference slices (existing union provenance) ---
    families.extend(
        [
            _fam(
                "semantic_mid_sender_support_v1",
                "",
                "Z_pipeline_existing",
                mode="provenance_slice",
                provenance_column="from_semantic_mid_sender_support",
            ),
            _fam(
                "semantic_mid_core_support_v1",
                "",
                "Z_pipeline_existing",
                mode="provenance_slice",
                provenance_column="from_semantic_mid_core_support",
            ),
            _fam(
                "shared_stem_highconf_v1",
                "",
                "Z_pipeline_existing",
                mode="provenance_slice",
                provenance_column="from_shared_stem_highconf",
            ),
            _fam(
                "2hop_bounded_v1",
                "",
                "Z_pipeline_existing",
                mode="provenance_slice",
                provenance_column="from_2hop",
            ),
            _fam(
                "semantic_reciprocal_v1",
                "",
                "Z_pipeline_existing",
                mode="provenance_slice",
                provenance_column="from_semantic",
            ),
            _fam(
                "component_expansion_v1",
                "",
                "Z_pipeline_existing",
                mode="provenance_slice",
                provenance_column="from_component",
            ),
            _fam(
                "body_token_jaccard_highconf_v1",
                "",
                "Z_pipeline_existing",
                mode="provenance_slice",
                provenance_column="from_body_token_jaccard_highconf",
            ),
            _fam(
                "body_char4gram_jaccard_highconf_v1",
                "",
                "Z_pipeline_existing",
                mode="provenance_slice",
                provenance_column="from_body_char4gram_jaccard_highconf",
            ),
            _fam(
                "semantic_mid_senderlocalpart_support_v1",
                "",
                "Z_pipeline_existing",
                mode="provenance_slice",
                provenance_column="from_semantic_mid_senderlocalpart_support",
            ),
        ]
    )

    # --- Legacy narrow set (for continuity) ---
    families.extend(
        [
            _fam("semantic_ge_0_90_AND_shared_sender", "semantic_ge_0_90_AND_shared_sender", "legacy"),
            _fam("semantic_ge_0_90_AND_shared_stem", "semantic_ge_0_90_AND_shared_stem", "legacy"),
            _fam("from_2hop_AND_semantic_ge_0_90", "from_2hop_AND_semantic_ge_0_90", "legacy"),
        ]
    )

    return families, skipped


CATALOG_REGISTRY: dict[str, Any] = {
    "rich_v1": build_rich_v1_catalog,
}
