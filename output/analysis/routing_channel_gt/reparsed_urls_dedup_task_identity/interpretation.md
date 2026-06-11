# Routing channel GT purity (post-dedup, reparsed URLs lake)

**Data:** `incidents-lake-misp.reparsed_urls.dedup_task_identity.json` (4,970 incidents)  
**GT:** `ground_truth.json` ∪ `ground_truth.reparsed_urls.dedup_task_identity.json` (1,488 labeled emails)  
**Method:** `build_parsed_misp_channel_reports` in `thesis_graph_construction_diagnostics.py` — for each artifact value, count induced email–email pairs and classify same vs cross campaign among GT-covered pairs.

## Routing channels

| Channel | GT-covered pairs | Same-campaign | Cross-campaign | Cross % | Max artifact degree |
|---------|------------------:|--------------:|---------------:|--------:|--------------------:|
| return_path_email | 2,547 | 2,547 | 0 | **0.0%** | 92 |
| return_path_domain | 5,480 | 5,446 | 34 | **0.6%** | 302 |
| origin_ip | 4,129 | 1,199 | 2,930 | 71.0% | 197 |
| received_host | 15,819 | 6,377 | 9,442 | 59.7% | 438 |
| helo_host | 4,738 | 1,629 | 3,109 | 65.6% | 197 |

## Interpretation

- **Return-path fields are not noisy** under this diagnostic: shared `return_path_email` is perfectly same-campaign among GT-covered pairs; `return_path_domain` is ~99.4% same-campaign with only 34 cross pairs out of 5,480.
- **Hop/transit metadata is noisy**: `origin_ip`, `received_host`, and `helo_host` induce a majority of **cross-campaign** pairs (60–71% cross), with high max degrees (especially `received_host` at 438 emails sharing one host).
- Results match the standard (non-reparsed) dedup lake on routing fields; URL reparsing does not change routing-header statistics.

## Note on other analyses

`gt_edge_structure_analysis` with `anchor_run_dir` set reads core channels from `nodes.csv` only — routing marginals were all zero (`routing_channels_from_graph: false`). Use this MISP-parsed channel report (or load the heterograph with routing edges enabled) for routing validation.
