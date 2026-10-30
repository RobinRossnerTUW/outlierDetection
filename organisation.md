# Organisation

## Scripts

| What | Where |
|---|---|
| Test-set metrics (mAP, top-k, hard/soft/perc-k, R@k) for a checkpoint, optional feature cache | `outlierDetection/icdar23/eval_baseline.py` |
| Test-only eval of a checkpoint on a foreign WriterZoo test set, reusing the experiment's stored `pca.pkl` (no train re-encoding); writes `metrics.csv`, `page_feats.npz` | `outlierDetection/icdar23/eval_external_test.py` |
| Per-group breakdown of an `eval_external_test.py` run from `page_feats.npz` + `labels.csv` (CPU only): `per_group.csv`, `cross_ms.csv`, `within_ms.csv` | `outlierDetection/icdar23/eval_external_breakdown.py` |
| Add the new multi-hand labels to the old Patrick-data `labels.csv` -> `labels_v2.csv` (same 563 rows; `in_new`, `hands`, `n_hands`, `original_hand` (old hand, pva-msB split into -1/-2/-3), `foreign_hand` (proof-readers, Tibetan)) | `outlierDetection/icdar23/helpers/make_labels_v2.py` |
| Retrieval on two-hand Patrick-data pages from `page_feats.npz` + `labels_v2.csv` (CPU, venvOUT): Q1 original hand still found (gallery all 563 pages) vs single-hand reference, Q2 foreign hand trace (gallery = original hand's pages in new JSON, relevant = pages with the foreign hand); per group + ALL per query / per group | `outlierDetection/icdar23/eval_mixed_hands.py` |
| SGR rerank grid search (from feature cache or saved PCA-ed embeddings), writes best + full grid | `outlierDetection/icdar23/eval_sgr_grid.py` |
| **Contamination eval (SumPooling only)**: embeds patches on the fly, random + lines modes, contaminated / clean gallery, own + foreign-writer mAP, CVL + ICDAR | `DAS2026/icdar23/contam_sumpool.py` |
| **Robust methods under contamination** (`--methods basic`: GeoMedian, TrimmedMean20, KMeansReject, SumPool+SGR contaminated gallery; `sgr_clean`: SumPool+SGR clean gallery; `trim_eps`; `topk_ot`); run with venvDAS26 (POT) | `DAS2026/icdar23/contam_robust.py` |
| Robust pooling classes used by `contam_robust.py` (see "Robust pooling" below) | `DAS2026/icdar23/robust_pooling.py` |
| Rebuild `coordinates.json` for an existing patch set (needed for CVL lines mode) | `DAS2026/icdar23/helpers/make_coords_for_patches.py` |
| Clean the new Patrick-data `new_data/data.json.gz` (2026-10-22 version, multi-hand pages) -> `new_data/data_fixed.json`: adds `m-ms-id` back (new key is `m-id`), maps `.tif`/`.png` image names to the `.jpg` on disk, drops hands not declared in the hands XML | `outlierDetection/icdar23/helpers/fix_patrick_json.py` |
| Stage Patrick-data (MASLOTS Sanskrit hands): parse `data.json`, grayscale non-gray images, symlink as `<writer>_<page>.jpg`, write `labels.csv` | `DAS2026/icdar23/helpers/prepare_patrick_data.py` |
| SIFT patches without clustering for non-binarized images (Canny edge filter) | `DAS2026/icdar23/helpers/extract_patches_only_color.py` |
| Copy of the above with `--scale_csv` (one scale per page, columns `staged`, `scale`) | `DAS2026/icdar23/helpers/extract_patches_only_color_scale_csv.py` |
| Patch contact sheets over several `--scale` values (one random capped page per writer, one row per scale); output in `DAS2026/icdar23/experiments/maslots_skt_scale_sweep/` | `DAS2026/icdar23/helpers/scale_sweep_patches.py` |
| Scale for Patrick-data so a character is ~32 px high: `survey` (auto char height = area-weighted median ink-component height; jinendra split into width classes `psT-ms-<hand>_w<width>` because its resolution varies), `rulers` (ruler crops per group + `char_height.csv` for a manual check), `scales --auto` (-> `scales.csv`, group-median scale, 13 groups); output in `DAS2026/icdar23/experiments/maslots_skt_char_height/` | `DAS2026/icdar23/helpers/char_height_calibration.py` |
| Shared by contamination: foreign-page / patch / line selection | `DAS2026/icdar23/selection.py` |
| Shared by contamination: PAGE-XML parsing, patch-to-line assignment | `DAS2026/icdar23/page_xml.py` |
| Shared by contamination: foreign-writer mAP (`calc_foreign_map`) | `DAS2026/icdar23/results.py` |
| Legacy HDF5-based contamination pipeline (not updated, see caveats) | `DAS2026/icdar23/contam_main_robust.py`, `contam_main_OT.py`, `workers.py`, `config.py` |

## Robust pooling (`robust_pooling.py`)
All three aggregate a page's patches, then L2-normalise (`contam_robust.py` uses `pn_alpha=1.0`; legacy pipeline 0.4).

| Method | What it does | Parameters |
|---|---|---|
| TrimmedMean20 | drop the patches farthest from the page centroid, mean of the rest (hard filter + sum-pool) | `trim_frac=0.2`, always drops 20% |
| KMeansReject | k-means on the page, keep the larger cluster, mean of it (hard filter + sum-pool) | `k=2`, drops 0–50% (data-dependent); `min_cluster_frac` never triggers at k=2 |
| GeoMedian | Weiszfeld: iteratively reweighted mean with weights 1/distance, nothing dropped (soft down-weighting) | `max_iter=30`, `tol=1e-5` |

## SGR with a clean gallery (`sgr_clean`)
SGR is transductive (one graph over queries = gallery). For the clean gallery each query gets its own graph:
the clean pages with the query's own clean page replaced by its contaminated descriptor; only that query's row is kept.
Batched on GPU, ~15 min for CVL (20 trials, 6 levels, both modes); log in `contamination/robust_sgr_clean.log`.

## Data

| What | Where |
|---|---|
| CVL test patches (binarized, model's own test set) | `/data/rrossner/datasets/CVL/cvl_test_patches_bin` |
| Line coordinates for those patches (from `make_coords_for_patches.py`) | `DAS2026/cvl_test_patches_bin_coords/` |
| ICDAR17 binarized test pages (white ink on black; median stroke width 4 px) | `/data/rrossner/datasets/ICDAR17/binarized_test` |
| ICDAR17 test patches + their `coordinates.json` | `/data/rrossner/DAS2026/icdar/test_patches_coords` |
| Patrick-data (MASLOTS Sanskrit hands, 563 pages, 5 hands): copy of `/caa/Projects02/maslots/private/Patrick-data` without the duplicate tif/png; `gray/` (89 converted images), `staged/` (symlinks), `labels.csv` (writer id -> hand, manuscript, source) | `/data/rrossner/datasets/Patrick-data` |
| New Patrick-data labels (2026-10-22): `data.json.gz` + `maslots-skt-hands.xml`, 436 pages (subset of the old 563: 127 psT-ms B pages b_065b-b_260b gone), pva-msB hand split into `-1/-2/-3` + `-tibetan`, new `psT-ms-A/B-hand-corr` (proof-readers, only on pages also written by A/B); 47 two-hand pages after cleaning; 2 more pages had a junk second hand (`.../baseballcard.sty`), dropped in `data_fixed.json` | `/data/rrossner/datasets/Patrick-data/new_data` |
| Patrick-data patches (1.12M, `icdar17_test` params, max 2000/page, per-group scale from `scales.csv` (copied into the folder) so a character is ~32 px high; one folder per page with `<writer>_<page>_<n>.png` + `coordinates.json`); `writer_zoo` key `maslots_skt` / `test`. Replaced the earlier global `--scale 0.4` set on 2026-10-08 | `/data/rrossner/datasets/Patrick-data_patches` |

## Results

### CVL — `outlierDetection/icdar23/experiments/cvl_cluster_285k/`
| Subdir | Contents |
|---|---|
| `eval/` | `baseline.csv`, `baseline_rerank.csv` (best SGR), `page_feats.npz` (feature cache), `rerank_k2_l2_g0.15.csv` |
| `eval/sgr_grid/` | `sgr_grid_fixed.csv` (valid), `sgr_grid_BUGGY_before_sgr_fix.csv` (invalid, kept for reference) |
| `contamination/sumpool/random/`, `lines/` | contaminated-gallery runs; **`foreign_*` here predate the foreign-mAP fix — rerun** (own mAP is fine) |
| `contamination/sumpool/*_clean_gallery/` | clean-gallery runs (after rerun) |
| `contamination/sumpool/_check_5trials/` | 5-trial sanity check of the foreign-mAP fix (random, 0–100%) |
| `contamination/<method>/<mode>[_clean_gallery]/` | `contam_robust.py` results (`sumpool_ref`, `geomedian`, `trimmedmean20`, `kmeans_reject_k2`, `sumpool_sgr_*`, `trim_eps*`, `sumpool_top2_ot_m*`); source page excluded from own-writer ranking. `sumpool_sgr_*/<mode>` comes from `basic`, `sumpool_sgr_*/<mode>_clean_gallery` from `sgr_clean` |
| `pca_patch.pkl` | patch-level PCA (128-d, 200k train patches) for trim_eps / topk_ot |
| `contamination/sumpool/_diagnostics/` | lines-vs-random control experiment (`matched_control_*.csv`) |

### MASLOTS Sanskrit hands (Patrick-data) — `outlierDetection/icdar23/experiments/maslot_icdar19_params-2026-09-21-15-58/eval_maslots_skt/`
MaSLoT-trained model on `maslots_skt` test (scale 0.4 patches): mAP 0.741, top-1 0.979; `metrics.csv`, `page_feats.npz` (pages sorted by writer, page as in `labels.csv`), `run.log`. Hand and manuscript are confounded: `vibhūticandra` across its two manuscripts gets top-1 0.29 / 0.00.
Same model on the per-group-scale patches (character ~32 px): `../eval_maslots_skt_charscale/` (same files), mAP 0.767, top-1 0.995.
Two-hand pages (45 queries, new JSON 2026-10-22): `../eval_maslots_skt_charscale/mixed_hands/`. Q1 original hand: mAP 0.704, top-1 0.956 (per query), psT A/B mixed pages as good as their single-hand pages. Q2 foreign hand: at chance (A-corr mAP 0.576 vs 0.592 chance, B-corr 0.112 vs 0.087), i.e. the page descriptor carries no proof-reader signal. pva-msB-3 + Tibetan has no Q2 row (the other Tibetan pages are hand-1).

### ICDAR17 — `DAS2026/icdar23/experiments/test_repository-2026-05-08-12-23/`
Copy of the original experiment without raw patch files (h5/npy/index).
| Subdir | Contents |
|---|---|
| `eval/` | `baseline.csv` (0.6808), `baseline_rerank.csv` (best SGR), `sgr_grid/sgr_grid_fixed.csv` |
| `contamination/sumpool/{random,lines}[_clean_gallery]/` | current contamination results, 20 trials, 0–50%, own + foreign |
| `contamination_results_*`, `reranked/` | old pipeline outputs, **do not use** (lines misaligned, buggy SGR, old foreign exclusion) |

Every contamination folder: `trials.csv` / `summary.csv` (own writer), `foreign_trials.csv` / `foreign_summary.csv` (foreign writer).

## Fixed bugs (working copies only)
- `reranking/sgr.py` (both repos): in-place update in `gnn_propagate` corrupted SGR.
- `results.py` `calc_foreign_map`: source page is now removed from the ranking, not only from the relevant set.
- `helpers/extract_patches_only_color.py`: duplicate-keypoint check used a list (quadratic per page); now a set, same output.
- `robust_pooling.py` / `contam_robust.py basic`: per-page (n, D) temporaries (k-means, trimmed mean, page gather) made 48 forked workers spend ~96% of CPU in kernel page faults (ICDAR >22 h without one trial). Now matvec-based pooling (outputs within 3e-8), reused gather buffer, BLAS threads=1 set before numpy import, one job per (mode, trial, level), per-job logs in `contamination/_progress/`, `kill -USR1 <pid>` dumps a stack. ICDAR basic random/clean, 20 trials: ~50 min.
- `contam_robust.py topk_ot`: OT is compute-bound (~4.5 s per ICDAR pair, 3 m values); `mallopt` at import keeps freed (n, n) matrices in the heap (no re-faulting); logs per level / every 10% of OT pairs. ICDAR random/clean top-3/5/10, 128 workers on a loaded server: ~80 min per level (36k pairs), ~35 h for 5 trials. `--embed_cache` also works for topk_ot (CPU-only once `pca_patch.pkl` exists).
- `contam_robust.py` options: `--poolings` / `--no_sumpool_ref` (subset of `basic`); `--embed_only --embed_cache X.npy` embeds into a disk file and exits (frees the GPU), then `--embed_cache X.npy` runs `basic` memory-mapped from it (page cache, not counted by the RAM watchdog).
- Old HDF5 dumps stored patches in filename string order while line maps use `patch_idx` → old lines results invalid.
  `contam_sumpool.py` orders by `patch_idx`. The legacy `contam_main*.py` pipeline still reads those HDF5s.

See `contamination_findings.md` for the lines-vs-random analysis.
