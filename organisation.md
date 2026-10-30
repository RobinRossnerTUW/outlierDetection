# Organisation

## Character height adjustment
`DAS2026/icdar23/helpers/char_height_calibration.py` computes the scale that makes a character about `--target` px high (the SIFT window size), so patches show similar amounts of writing across manuscripts:

| Step | What it does |
|---|---|
| `survey` | Estimates each page's character height as the area-weighted median height of its ink components; splits jinendra pages into width classes because their resolution varies |
| `rulers` | Writes page crops with a px ruler and reference boxes per group, plus `char_height.csv` for a manual check |
| `scales` | Turns the heights into `scales.csv` (`scale = target / character height`); `--auto` uses the estimated height without the manual check |

`DAS2026/icdar23/helpers/extract_patches_only_color_scale_csv.py --scale_csv scales.csv` then applies the scales during patch extraction.

## Scripts

| What | Where |
|---|---|
| Test-set metrics (mAP, top-k, hard/soft/perc-k, R@k) for a checkpoint, optional feature cache | `outlierDetection/icdar23/eval_baseline.py` |
| Test-only eval of a checkpoint on a foreign WriterZoo test set, reusing the experiment's stored `pca.pkl` (no train re-encoding); writes `metrics.csv`, `page_feats.npz` | `outlierDetection/icdar23/eval_external_test.py` |
| Per-group breakdown of an `eval_external_test.py` run from `page_feats.npz` + `labels.csv`: `per_group.csv`, `cross_ms.csv`, `within_ms.csv` | `outlierDetection/icdar23/eval_external_breakdown.py` |
| Add multi-hand labels to the Patrick-data `labels.csv` -> `labels_v2.csv` | `outlierDetection/icdar23/helpers/make_labels_v2.py` |
| Retrieval on two-hand Patrick-data pages from `page_feats.npz` + `labels_v2.csv`: Q1 original hand still found, Q2 foreign hand trace | `outlierDetection/icdar23/eval_mixed_hands.py` |
| SGR rerank grid search (from feature cache or saved PCA-ed embeddings), writes best + full grid | `outlierDetection/icdar23/eval_sgr_grid.py` |
| **Contamination eval (SumPooling only)**: embeds patches on the fly, random + lines modes, contaminated / clean gallery, own + foreign-writer mAP, CVL + ICDAR | `DAS2026/icdar23/contam_sumpool.py` |
| **Robust methods under contamination** (`--methods basic`: GeoMedian, TrimmedMean20, KMeansReject, SumPool+SGR contaminated gallery; `sgr_clean`: SumPool+SGR clean gallery; `trim_eps`; `topk_ot`) | `DAS2026/icdar23/contam_robust.py` |
| Robust pooling classes used by `contam_robust.py` (see "Robust pooling" below) | `DAS2026/icdar23/robust_pooling.py` |
| Rebuild `coordinates.json` for an existing patch set (needed for CVL lines mode) | `DAS2026/icdar23/helpers/make_coords_for_patches.py` |
| Clean the Patrick-data `data.json.gz` -> `data_fixed.json`: restores `m-ms-id`, maps `.tif`/`.png` image names to `.jpg`, drops hands not declared in the hands XML | `outlierDetection/icdar23/helpers/fix_patrick_json.py` |
| Stage Patrick-data: parse `data.json`, grayscale non-gray images, symlink as `<writer>_<page>.jpg`, write `labels.csv` | `DAS2026/icdar23/helpers/prepare_patrick_data.py` |
| SIFT patches without clustering for non-binarized images (Canny edge filter) | `DAS2026/icdar23/helpers/extract_patches_only_color.py` |
| Copy of the above with `--scale_csv` (one scale per page, columns `staged`, `scale`) | `DAS2026/icdar23/helpers/extract_patches_only_color_scale_csv.py` |
| Patch contact sheets over several `--scale` values (one random capped page per writer, one row per scale) | `DAS2026/icdar23/helpers/scale_sweep_patches.py` |
| Shared by contamination: foreign-page / patch / line selection | `DAS2026/icdar23/selection.py` |
| Shared by contamination: PAGE-XML parsing, patch-to-line assignment | `DAS2026/icdar23/page_xml.py` |
| Shared by contamination: foreign-writer mAP (`calc_foreign_map`) | `DAS2026/icdar23/results.py` |
| Legacy HDF5-based contamination pipeline | `DAS2026/icdar23/contam_main_robust.py`, `contam_main_OT.py`, `workers.py`, `config.py` |

## Robust pooling (`robust_pooling.py`)
All three aggregate a page's patches, then L2-normalise.

| Method | What it does | Parameters |
|---|---|---|
| TrimmedMean20 | drop the patches farthest from the page centroid, mean of the rest | `trim_frac=0.2` |
| KMeansReject | k-means on the page, keep the larger cluster, mean of it | `k=2` |
| GeoMedian | Weiszfeld: iteratively reweighted mean with weights 1/distance, nothing dropped | `max_iter=30`, `tol=1e-5` |

## SGR with a clean gallery (`sgr_clean`)
SGR is transductive (one graph over queries = gallery). For the clean gallery each query gets its own graph:
the clean pages with the query's own clean page replaced by its contaminated descriptor; only that query's row is kept.
