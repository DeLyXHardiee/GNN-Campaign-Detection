# Pair score separation — output audit

Generated: `2026-05-20T17:35:02+00:00` (read-only audit; no pipeline changes).

## 1. Scope

This audit inventories artifacts under `<run_dir>/pair_score_separation/` produced by:

- `python seed_candidate_workflow/scripts/run_pair_score_separation_analysis.py`
- Rescued-vs-collapsed block inside the same run (`enable_rescued_collapsed_analysis`)
- Optional separate: `pair_low_band_feature_discovery` (writes into the same folder)

## 2. Example run on disk

Scanned: `C:\Users\aar\Desktop\GNN-Campaign-Detection\output\runs\main_gnn_pu_1_no_ts_dedup_task_identity_10\pair_score_separation`
- **55** files, **197.11 MB** total
- HTML: 11, JSON: 9, CSV: 18, PNG: 13
- Largest HTML: `pair_same_unlabeled_rescued_vs_collapsed.html` (41.4 MB)

## 3. Primary outputs (recommended front page)

- `pair_score_separation_summary.json`
- `plots/score_distribution_all_scored_pairs.png`
- `plots/score_distribution_same_campaign_<gt>.png`
- `plots/score_distribution_cross_campaign_<gt>.png`
- `pair_low_band_unlabeled_pairs_for_review.html`
- `pair_same_unlabeled_rescued_vs_collapsed_summary.json`
- `pair_same_unlabeled_rescued_for_review.html`
- `pair_same_unlabeled_collapsed_for_review.html`

## 4. Biggest problems

- Flat output directory (~45+ files per run) with no subfolders.
- Six+ overlapping JSON summaries (low/high marginal + joint + FP + twohop + rescued).
- Separator FEATURE_KEYS_DEFAULT omits body_only and path Jaccard — JSON summaries under-report new features.
- Low/high band HTML uses sparse meta-grid; rescued HTML shows Jaccards but is unreadable at scale.
- pair_same_unlabeled_rescued_vs_collapsed.html can exceed 40MB (all pairs, no TOC).
- Duplicate CSV+JSONL+HTML triple exports for each review cohort.
- 13 plots per GT when multiple --gt-path passed — filenames help but clutter top level plots/.

## 5. HTML feature visibility

| Page family | Readable? | Width issue | Body/path Jaccard in HTML |
|-------------|-----------|-------------|---------------------------|
| `pair_low_band_*_for_review.html` | good | low | missing: body_*, body_only_*, path_token_jaccard_combined, url_path_token_jaccard |
| `pair_high_band_false_positive_pairs_for_review.html` | good | low | missing: body_only_*, path_*, body_token_jaccard |
| `pair_same_unlabeled_rescued_vs_collapsed.html` | poor | high | missing: twohop channel badges in compact form only |

**Key finding:** Low/high band review HTML (`_write_pairs_for_review_html`) does **not** render `body_only_*`, `path_token_jaccard_combined`, or `url_path_token_jaccard` in the card header, even when columns exist on the pair-training table. Rescued-vs-collapsed HTML **does** show them, but packs 10+ metrics into one `meta-grid` row and often ships one giant file for all buckets.

## 6. JSON summary overlap

### `pair_score_separation_summary.json`
- Role: canonical run index
- Overlaps: per_gt plots paths, band_diagnostics

### `pair_low_band_separator_summary.json`
- Role: low marginal separators
- Overlaps: pair_score_separation_summary per_gt, pair_low_band_joint_separator_summary.json
- Stale risk: FEATURE_KEYS_DEFAULT lacks body/path jaccard

### `pair_high_band_separator_summary.json`
- Role: high marginal separators
- Overlaps: pair_high_band_joint, pair_high_band_false_positive_summary.json

### `pair_same_unlabeled_rescued_vs_collapsed_summary.json`
- Role: rescued vs collapsed — keep separate
- Overlaps: joint summary only for rule tables

## 7. Proposed folder layout (cleanup phase)

```
pair_score_separation/
  core_json/          # 1–2 summaries
  review_html/        # human review pages
  plots/
  debug_csv/          # tables + full pair dumps
  debug_json/         # separator / joint summaries
```

## 8. Cleanup order (recommended)

- 1. Reorganize output into review_html/, core_json/, plots/, debug_csv/ (no behavior change).
- 2. Unify HTML pair card: add body_only + path Jaccard block to _pair_card_html (reuse rescued template fields).
- 3. Split rescued_vs_collapsed combined HTML or add sidebar TOC; cap pairs per page.
- 4. Collapse JSON summaries into pair_score_separation_summary.json + optional debug bundle.
- 5. Extend FEATURE_KEYS_DEFAULT / band separator stats for body_only_* and path_token_jaccard_combined.

## 9. Full artifact catalog

See `pair_score_separation_output_inventory.csv` and `pair_score_separation_output_audit_summary.json` for per-file recommendations (keep_as_primary / debug / merge / remove).
