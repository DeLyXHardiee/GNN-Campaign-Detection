"""Tests for rescued vs collapsed same-campaign unlabeled pair analysis."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from seed_candidate_workflow.utils.pair_same_unlabeled_rescued_collapsed import (
    assign_same_unlabeled_buckets,
    run_same_unlabeled_rescued_vs_collapsed_analysis,
)


def test_bool_series_missing_column():
    from seed_candidate_workflow.utils.pair_same_unlabeled_rescued_collapsed import _bool_series

    df = pd.DataFrame({"a": [1, 2]})
    s = _bool_series(df, "twohop_via_html_fp")
    assert len(s) == 2
    assert not bool(s.any())


def test_ranking_excludes_gt_campaign_columns():
    from seed_candidate_workflow.utils.pair_same_unlabeled_rescued_collapsed import (
        _filter_ranked_separator_rows,
        _is_ranking_excluded_column,
    )

    assert _is_ranking_excluded_column("gt_campaign_i")
    rows = [
        {
            "metric_group": "feature_mean",
            "metric_name": "gt_campaign_i",
            "abs_difference": 1.0,
            "favors": "rescued",
        },
        {
            "metric_group": "feature_mean",
            "metric_name": "body_token_jaccard",
            "abs_difference": 0.5,
            "favors": "rescued",
        },
    ]
    filtered = _filter_ranked_separator_rows(rows)
    assert len(filtered) == 1
    assert filtered[0]["metric_name"] == "body_token_jaccard"


def test_enrich_path_features_from_nodes():
    from seed_candidate_workflow.utils.pair_same_unlabeled_rescued_collapsed import (
        _enrich_path_features_from_nodes,
    )

    nodes = {
        "a": {"url_set": set(), "stem_set": {"/foo/bar"}},
        "b": {"url_set": set(), "stem_set": {"/foo/bar", "/foo/bar/baz"}},
    }
    df = pd.DataFrame({"email_i": ["a"], "email_j": ["b"]})
    out = _enrich_path_features_from_nodes(df, nodes)
    assert out["path_token_jaccard_combined"].notna().iloc[0]
    assert out["url_path_token_jaccard"].notna().all()
    assert out["stem_path_token_jaccard"].notna().all()


def test_merge_pair_features_from_eval():
    from seed_candidate_workflow.utils.pair_same_unlabeled_rescued_collapsed import (
        _merge_pair_features_from_eval,
    )

    df_ins = pd.DataFrame({"email_i": ["a", "b"], "email_j": ["b", "c"]})
    df_eval = pd.DataFrame(
        {
            "email_i": ["a", "b"],
            "email_j": ["b", "c"],
            "body_token_jaccard": [0.42, 0.11],
            "body_char4gram_jaccard": [0.55, 0.09],
        }
    )
    out = _merge_pair_features_from_eval(df_ins, df_eval)
    assert out["body_token_jaccard"].tolist() == [0.42, 0.11]
    assert out["body_char4gram_jaccard"].tolist() == [0.55, 0.09]


def test_feature_cols_excludes_boolean_columns():
    df = pd.DataFrame(
        {
            "semantic_cosine_max": [0.8, 0.7],
            "body_token_jaccard": [0.3, 0.1],
            "has_shared_html_fp": [True, False],
            "twohop_via_html_fp": [False, True],
        }
    )
    from seed_candidate_workflow.utils.pair_same_unlabeled_rescued_collapsed import (
        _is_continuous_feature_column,
    )

    assert _is_continuous_feature_column(df, "semantic_cosine_max")
    assert _is_continuous_feature_column(df, "body_token_jaccard")
    assert not _is_continuous_feature_column(df, "has_shared_html_fp")
    assert not _is_continuous_feature_column(df, "twohop_via_html_fp")


def test_assign_same_unlabeled_buckets_defaults():
    scores = np.array([0.05, 0.50, 0.90, np.nan])
    buckets = assign_same_unlabeled_buckets(scores)
    assert buckets[0] == "collapsed_same_unlabeled"
    assert buckets[1] == "mid_same_unlabeled"
    assert buckets[2] == "rescued_same_unlabeled"
    assert buckets[3] == "other"


def test_run_rescued_vs_collapsed_writes_artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    df_work = pd.DataFrame(
        {
            "email_i": ["a", "b", "c", "d"],
            "email_j": ["b", "c", "d", "e"],
            "pair_status": ["unlabeled"] * 4,
            "from_semantic": [True, False, True, False],
            "from_2hop": [False, True, False, True],
            "semantic_cosine_max": [0.85, 0.78, 0.82, 0.79],
            "source_count": [2, 1, 2, 1],
            "body_token_jaccard": [0.4, 0.1, 0.35, 0.05],
            "body_char4gram_jaccard": [0.5, 0.12, 0.4, 0.08],
        }
    )
    scores = np.array([0.05, 0.90, 0.05, 0.88])
    label_map = {"a": "C1", "b": "C1", "c": "C1", "d": "C1", "e": "C2"}
    gt_path = tmp_path / "ground_truth.json"
    gt_path.write_text("{}", encoding="utf-8")

    def _fake_inspection(**kwargs: object) -> pd.DataFrame:
        mask = kwargs["row_mask"]
        idx = np.where(mask)[0]
        rows = []
        for i in idx:
            sc = float(scores[i])
            bucket = assign_same_unlabeled_buckets(np.array([sc]))[0]
            rows.append(
                {
                    "email_i": df_work.iloc[i]["email_i"],
                    "email_j": df_work.iloc[i]["email_j"],
                    "score": sc,
                    "same_unlabeled_bucket": bucket,
                    "from_semantic": bool(df_work.iloc[i]["from_semantic"]),
                    "from_2hop": bool(df_work.iloc[i]["from_2hop"]),
                    "semantic_cosine_max": float(df_work.iloc[i]["semantic_cosine_max"]),
                    "source_count": int(df_work.iloc[i]["source_count"]),
                    "has_shared_html_fp": False,
                    "twohop_via_html_fp": i == 1,
                    "gt_relation": "same_campaign",
                }
            )
        return pd.DataFrame(rows)

    import seed_candidate_workflow.utils.pair_same_unlabeled_rescued_collapsed as rvc_mod

    class _FakePse:
        @staticmethod
        def _build_high_band_inspection_dataframe(**kwargs: object) -> pd.DataFrame:
            return _fake_inspection(**kwargs)

        @staticmethod
        def _summarize_group(**_: object) -> dict[str, object]:
            return {"provenance": {}, "feature_summaries": {}, "shared_evidence": {}}

        @staticmethod
        def _safe_float_stats(series: pd.Series) -> dict[str, object]:
            s = pd.to_numeric(series, errors="coerce").dropna()
            return {"mean": float(s.mean()) if len(s) else None, "n_non_null": int(len(s))}

        @staticmethod
        def _cmp_from_masks(**kwargs: object) -> dict[str, object]:
            cond = kwargs["cond_same"]
            base_r = kwargs["base_same"]
            base_c = kwargs["base_cross"]
            nr = max(int(base_r.sum()), 1)
            nc = max(int(base_c.sum()), 1)
            rf = float(cond[base_r].mean()) if base_r.any() else 0.0
            cf = float(cond[base_c].mean()) if base_c.any() else 0.0
            return {
                "rescued_fraction": rf,
                "collapsed_fraction": cf,
                "difference_same_minus_cross": rf - cf,
                "abs_difference": abs(rf - cf),
            }

        _PAIR_SHARED_CHANNEL_DEFS = ()
        _INSPECTION_PROVENANCE_COLS = ()
        _INSPECTION_FEATURE_COLS = ()

        @staticmethod
        def _pair_shared_evidence_detail(*_a: object, **_k: object) -> dict[str, bool]:
            return {"has_shared_html_fp": False}

    fake = _FakePse()
    monkeypatch.setattr(rvc_mod, "_PSE", fake)
    monkeypatch.setattr(rvc_mod, "_pse", lambda: fake)
    monkeypatch.setattr(
        rvc_mod,
        "attach_twohop_channel_columns",
        lambda df, **_: df,
    )

    nodes = {
        "a": {"url_set": set(), "stem_set": {"/login"}},
        "b": {"url_set": set(), "stem_set": {"/login"}},
        "c": {"url_set": set(), "stem_set": {"/other"}},
        "d": {"url_set": set(), "stem_set": {"/login"}},
    }
    out = run_same_unlabeled_rescued_vs_collapsed_analysis(
        df_work=df_work,
        scores=scores,
        gt_path=gt_path,
        label_map=label_map,
        out_dir=tmp_path,
        nodes_by_email=nodes,
        evidence_index={},
        email_text_by_eid=None,
    )
    assert Path(out["summary_path"]).is_file()
    assert "core_json" in str(out["summary_path"]).replace("\\", "/")
    from seed_candidate_workflow.utils.pair_score_separation_output_layout import ExportFlags

    flags = ExportFlags(emit_debug_json=True, emit_debug_csv=True)
    out2 = run_same_unlabeled_rescued_vs_collapsed_analysis(
        df_work=df_work,
        scores=scores,
        gt_path=gt_path,
        label_map=label_map,
        out_dir=tmp_path,
        nodes_by_email=nodes,
        evidence_index={},
        email_text_by_eid=None,
        export_flags=flags,
    )
    assert "debug_csv" in str(out2["table_path"]).replace("\\", "/")
    assert "debug_json" in str(out2["joint_path"]).replace("\\", "/")

    summary = __import__("json").loads(Path(out["summary_path"]).read_text(encoding="utf-8"))
    assert summary["status"] == "ok"
    counts = summary["marginal_comparison"]["counts"]
    assert counts["n_rescued_same_unlabeled"] == 1
    assert counts["n_collapsed_same_unlabeled"] == 2
    assert "rescued_vs_collapsed_recommendations" in summary
    assert "same_unlabeled_html_fp_frontier_analysis" in summary
    assert "feature_population_diagnostics" in summary
    assert "path_feature_population_diagnostics" in summary
    assert "body_vs_body_only_comparison" in summary
    assert "single_source_frontier_summary" in summary
    diag = summary["feature_population_diagnostics"]
    path_diag = summary["path_feature_population_diagnostics"]
    assert "body_token_jaccard" in diag["found_feature_columns"]
    assert diag["non_null_counts_by_cohort"]["all_same_unlabeled"]["body_token_jaccard"]["n_non_null"] >= 1
    assert path_diag["found_path_feature_columns"] == [
        "path_token_jaccard_combined",
        "url_path_token_jaccard",
        "stem_path_token_jaccard",
    ]
    assert path_diag["non_null_counts_by_cohort"]["all_same_unlabeled"]["path_token_jaccard_combined"][
        "n_non_null"
    ] >= 1
    top = summary["marginal_comparison"]["ranked_marginal_separators_top20"]
    assert all("gt_campaign" not in str(r.get("metric_name", "")) for r in top)
    hf_cohort = next(
        c
        for c in summary["same_unlabeled_html_fp_frontier_analysis"]["cohorts"]
        if c.get("cohort") == "collapsed_same_html_fp_frontier"
    )
    if hf_cohort.get("n_pairs", 0) > 0:
        bt = hf_cohort["feature_summaries"]["body_token_jaccard"]
        assert int(bt.get("n_non_null", 0)) >= 1
