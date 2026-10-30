"""
threshold_eval_icdar17.py
--------------------------
Open-set writer identification on ICDAR17/HistoricalWI using cosine-
similarity thresholding, rebuilt on top of the icdar24-master ("SAGHOG")
codebase with the supervised resnet56 + NetRVLAD architecture.

WHAT CHANGED VS threshold_eval_cvl.py / threshold_eval_iam.py
================================================================
1. Model / feature backend
   Old scripts used a bespoke `backbone.model.Model` (triplet-trained
   resnet56, plain sum-pooling + a pickled sklearn PCA) that doesn't exist
   in icdar24. This version loads the model exactly the way icdar24's own
   `main_finetune.py` does for `model_zoo.experiment == 'resnet56'`
   (`models.resnets.resnet56` + `finetune.netvlad.Wrapper` with
   `netvlad.random=True`, i.e. NetRVLAD) via icdar24_page_features.py.

2. Memory
   The old scripts extracted features for every patch in the *entire*
   dataset into one array before pooling per page -- for a NetRVLAD head
   (patch descriptor dim = num_clusters * 64, e.g. 6400-d) across a large
   corpus like ICDAR17/HistoricalWI, that's exactly what drives RAM past
   200GB. This version streams **one page at a time**: patches for a page
   are embedded, sum-pooled into a single page descriptor, and discarded
   before the next page starts. See icdar24_page_features.py for details.
   Peak memory is bounded by the single largest page's patch count, not
   the whole dataset.

3. Unused work removed
   The old scripts extracted TRAIN features purely to print dataset
   stats -- they were never used in calibration or the ratio sweep (both
   only ever touch the TEST set, split into permanent cal/eval writer
   subsets). This version drops that extraction entirely.

4. PCA
   The old scripts loaded a pre-fit `pca.pkl` from a specific training
   run. Here PCA (if enabled) is fit on the fly on the small
   [num_pages, dim] page-descriptor matrix -- exactly like icdar24's own
   `evaluators.validators.pca()` -- so there's no dependency on an
   external pickle, and it stays cheap (page-level, not patch-level).

5. Calibration/eval logic
   The calibration -> permanent cal/eval writer split -> unknown-ratio
   sweep logic is unchanged in spirit from the two original scripts (it
   already only worked with page-level descriptors, so it was never the
   memory problem). We keep the more careful version from
   threshold_eval_iam.py, which holds out a query page per known writer
   so writer_id_acc_known measures retrieval, not self-similarity.

REQUIREMENTS
============
- Must be run with icdar24_page_features.py importable (same directory,
  or on PYTHONPATH), and ICDAR24_ROOT (env var or edit below) pointing at
  your icdar24-master checkout so its internal modules resolve.
- A test CSV in icdar24's own format (columns: 'imgs' + int-encoded
  'writer'/'page'), built the same way as
  datasets_example/create_dataset_csvs.py's ICDAR2017_Test class -- point
  its `root` at your HistoricalWI test images and adjust the regex if
  your filenames differ from `<writer>-IMG_MAX_<page>...`.
- Needs opencv-python + doxapy (already icdar24 dependencies, used for
  the on-the-fly SIFT patch extraction in dataloading/wrapper.py).
"""

import logging
import os

import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score, precision_score, recall_score

from icdar24_page_features import (
    load_resnet56_netrvlad,
    extract_pages_from_csv,
    extract_pages_from_patch_dir,
    apply_pca,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ------------------------------------------------------------------ #
# CONFIG
# ------------------------------------------------------------------ #
# Point this at your icdar24-master checkout (or set env var ICDAR24_ROOT
# before running); only needed if this script doesn't already live inside it.
os.environ.setdefault("ICDAR24_ROOT", "/data/rrossner/outlierDetection/icdar24")

CHECKPOINT_PATH = "/data/rrossner/outlierDetection/icdar24/experiments/resnet56_0.5_writer-2026-07-02-10-34/model.pt"

# --- data source: pick ONE of the two modes ---
# "csv"     -> native icdar24 pipeline: full binarized page images, on-the-fly
#              SIFT patch extraction (matches how main_finetune.py evaluates
#              icdar2017/HistoricalWI test sets).
# "patches" -> pre-extracted patch PNGs on disk (e.g. from icdar24's own
#              utils/extract_patches_only.py, or however you already have
#              HistoricalWI patches laid out), grouped per page by regex.
DATA_MODE = "patches"  # "csv" or "patches"

TEST_CSV = "datasets/icdar2017-test-binarized.csv"   # used if DATA_MODE == "csv"

TEST_PATCH_DIR = "/data/rrossner/datasets/ICDAR17/binarized_test/patches"   # used if DATA_MODE == "patches"
import re
WRITER_RE = re.compile(r'^(\d+)-')                   # adjust to your patch filename convention
PAGE_RE   = re.compile(r'^\d+-IMG_MAX_(\d+)')

# --- model / architecture (must match training config, e.g.
#     config/finetune_maskfeat_icdar2017_resnet.yml) ---
NUM_CLUSTERS = 100
RANDOM_VLAD  = True     # True = NetRVLAD (as requested)
IN_DIM       = -1       # model_options.in_dim; -1 = no embed_fc reduction, dim stays 64
PATCH_SIZE   = 32
DEVICE       = "cuda"

# --- optional page-descriptor-level PCA (icdar24 eval_options convention) --- 
APPLY_PCA = True
PCA_DIM   = 512

# --- open-set calibration / sweep (unchanged from the CVL/IAM scripts) ---
CAL_FRACTION   = 0.2
VAL_FRACTION   = 0.2
UNKNOWN_RATIOS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
N_REPEATS      = 10
RANDOM_SEED    = 42
OUT_CSV        = "threshold_results_icdar17.csv"

# Cache page descriptors to disk so re-running the sweep (or debugging)
# doesn't require re-extracting patches. Delete the file to force recompute.
CACHE_PATH = "icdar17_test_page_descriptors.npz"
# ------------------------------------------------------------------ #


def build_prototypes(descs, writers):
    protos = {}
    for w in np.unique(writers):
        v = descs[writers == w].mean(axis=0)
        norm = np.linalg.norm(v)
        protos[w] = v / norm if norm > 0 else v
    return protos


def score_all(descs, prototypes):
    writer_order = np.array(sorted(prototypes.keys()))
    proto_mat = np.stack([prototypes[w] for w in writer_order])
    sims = descs @ proto_mat.T
    best_idx = np.argmax(sims, axis=1)
    return writer_order[best_idx], sims[np.arange(len(descs)), best_idx]


def calibrate_tau(known_scores, unknown_scores, n_candidates=500):
    """Sweep tau, pick value maximising balanced accuracy."""
    pooled = np.concatenate([known_scores, unknown_scores])
    taus = np.quantile(pooled, np.linspace(0.005, 0.995, n_candidates))
    y_true = np.concatenate([
        np.ones(len(known_scores), dtype=np.int64),
        np.zeros(len(unknown_scores), dtype=np.int64),
    ])
    best_tau, best_bal = taus[0], -1.0
    for tau in taus:
        y_pred = (pooled >= tau).astype(np.int64)
        bal = balanced_accuracy_score(y_true, y_pred)
        if bal > best_bal:
            best_bal, best_tau = bal, float(tau)
    return best_tau, round(best_bal, 4)


def compute_metrics(test_writers, final_pred, pred_w, unknown_set):
    is_truly_unk = np.isin(test_writers, list(unknown_set)).astype(np.int64)
    is_pred_unk = (final_pred == -1).astype(np.int64)
    known_mask = is_truly_unk == 0
    writer_acc = (
        float(np.mean(pred_w[known_mask] == test_writers[known_mask]))
        if known_mask.any() else 0.0
    )
    return {
        "bal_acc": round(balanced_accuracy_score(is_truly_unk, is_pred_unk), 4),
        "f1_unknown": round(f1_score(is_truly_unk, is_pred_unk, pos_label=1, zero_division=0), 4),
        "precision_unknown": round(precision_score(is_truly_unk, is_pred_unk, pos_label=1, zero_division=0), 4),
        "recall_unknown": round(recall_score(is_truly_unk, is_pred_unk, pos_label=1, zero_division=0), 4),
        "writer_id_acc_known": round(writer_acc, 4),
        "pred_unknown_rate": round(float(np.mean(is_pred_unk)), 4),
        "true_unknown_rate": round(float(np.mean(is_truly_unk)), 4),
    }


def main():
    import pandas as pd

    model = load_resnet56_netrvlad(
        CHECKPOINT_PATH,
        num_clusters=NUM_CLUSTERS,
        random_vlad=RANDOM_VLAD,
        in_dim=IN_DIM,
        device=DEVICE,
    )

    print("\nExtracting TEST page descriptors (memory-efficient, one page at a time)...")
    if DATA_MODE == "csv":
        test_descs, test_writers, _test_pages = extract_pages_from_csv(
            TEST_CSV, model, device=DEVICE, patch_size=PATCH_SIZE,
            cache_path=CACHE_PATH,
        )
    elif DATA_MODE == "patches":
        test_descs, test_writers, _test_pages = extract_pages_from_patch_dir(
            TEST_PATCH_DIR, model, writer_re=WRITER_RE, page_re=PAGE_RE,
            device=DEVICE, cache_path=CACHE_PATH,
        )
    else:
        raise ValueError(f"Unknown DATA_MODE: {DATA_MODE}")

    all_test_writers = np.unique(test_writers)
    print(f"Test : {len(test_descs)} pages, {len(all_test_writers)} writers, "
          f"descriptor dim={test_descs.shape[1]}")

    if APPLY_PCA:
        test_descs, _pca = apply_pca(test_descs, pca_dim=PCA_DIM)
        print(f"After PCA: descriptor dim={test_descs.shape[1]}")

    # ------------------------------------------------------------------
    # OPTION B: permanent calibration / evaluation split
    #
    # cal_writers  (20%, fixed) -> used ONLY to calibrate tau, never swept
    # eval_writers (80%, fixed) -> used ONLY in the ratio sweep, never for tau
    # ------------------------------------------------------------------
    rng_split = np.random.default_rng(RANDOM_SEED)
    n_cal = max(2, int(round(len(all_test_writers) * CAL_FRACTION)))
    cal_writers = set(rng_split.choice(all_test_writers, size=n_cal, replace=False).tolist())
    eval_writers = set(all_test_writers.tolist()) - cal_writers

    print(f"\nPermanent split (seed={RANDOM_SEED}):")
    print(f"  calibration writers : {len(cal_writers)}  (never evaluated)")
    print(f"  evaluation writers  : {len(eval_writers)}  (never used for tau)")

    # ------------------------------------------------------------------
    # CALIBRATE TAU on cal_writers only
    #
    # Within cal_writers:
    #   - pick ~20% as synthetic unknowns  -> should score LOW
    #   - remaining 80%: fit prototypes on 80% of their pages,
    #                    val on the other 20%  -> should score HIGH
    # ------------------------------------------------------------------
    rng_cal = np.random.default_rng(RANDOM_SEED + 1)
    cal_arr = np.array(sorted(cal_writers))
    n_cal_unk = max(1, int(round(len(cal_arr) * 0.2)))
    cal_unk = set(rng_cal.choice(cal_arr, size=n_cal_unk, replace=False).tolist())
    cal_known = cal_writers - cal_unk

    fit_idx, val_idx = [], []
    for w in cal_known:
        idx = np.where(test_writers == w)[0]
        shuffled = rng_cal.permutation(idx)
        n_val = max(1, int(round(len(idx) * VAL_FRACTION)))
        n_val = min(n_val, len(idx) - 1) if len(idx) > 1 else 0
        val_idx.extend(shuffled[:n_val].tolist())
        fit_idx.extend(shuffled[n_val:].tolist())

    fit_idx = np.array(fit_idx)
    val_idx = np.array(val_idx)
    unk_idx = np.where(np.isin(test_writers, list(cal_unk)))[0]

    print(f"\nTau calibration (cal_writers only):")
    print(f"  cal known writers   : {len(cal_known)}  ({n_cal_unk} held out as unknowns)")
    print(f"  fit pages           : {len(fit_idx)}")
    print(f"  known val pages     : {len(val_idx)}")
    print(f"  synthetic unk pages : {len(unk_idx)}")

    cal_prototypes = build_prototypes(test_descs[fit_idx], test_writers[fit_idx])
    _, known_scores = score_all(test_descs[val_idx], cal_prototypes)
    _, unk_scores = score_all(test_descs[unk_idx], cal_prototypes)

    print(f"  known val score  mean={known_scores.mean():.3f}  std={known_scores.std():.3f}")
    print(f"  unknown   score  mean={unk_scores.mean():.3f}  std={unk_scores.std():.3f}")

    tau, cal_bal = calibrate_tau(known_scores, unk_scores)
    print(f"\nCalibrated tau : {tau:.4f}  (cal balanced_acc={cal_bal})")

    # ------------------------------------------------------------------
    # SWEEP on eval_writers only (tau is now fixed, never changes)
    # ------------------------------------------------------------------
    eval_arr = np.array(sorted(eval_writers))
    eval_mask = np.isin(test_writers, eval_arr)
    eval_descs = test_descs[eval_mask]
    eval_w = test_writers[eval_mask]

    print(f"\nEvaluation writers : {len(eval_writers)}")
    print(f"Evaluation pages   : {len(eval_descs)}")
    print(f"Sweeping ratios    : {UNKNOWN_RATIOS}")
    print("-" * 70)

    rows = []
    for ratio in UNKNOWN_RATIOS:
        run_metrics = []
        for repeat in range(N_REPEATS):
            rng2 = np.random.default_rng(RANDOM_SEED + repeat * 100)
            n_unk = max(1, int(round(len(eval_arr) * ratio)))
            unk_set = set(rng2.choice(eval_arr, size=n_unk, replace=False).tolist())
            known_set = set(eval_arr.tolist()) - unk_set

            # Held-out split per known writer -- a writer's own queried page
            # must never have contributed to its own prototype, or
            # writer_id_acc_known would measure self-similarity, not retrieval.
            fit_idx2, held_idx = [], []
            for w in known_set:
                idx = np.where(eval_w == w)[0]
                shuffled = rng2.permutation(idx)
                n_val = max(1, int(round(len(idx) * VAL_FRACTION)))
                n_val = min(n_val, len(idx) - 1) if len(idx) > 1 else 0
                held_idx.extend(shuffled[:n_val].tolist())
                fit_idx2.extend(shuffled[n_val:].tolist())

            fit_idx2 = np.array(fit_idx2)
            held_idx = np.array(held_idx)
            unk_idx2 = np.where(np.isin(eval_w, list(unk_set)))[0]

            protos = build_prototypes(eval_descs[fit_idx2], eval_w[fit_idx2])

            # query set: held-out known pages + all unknown-writer pages
            query_idx = np.concatenate([held_idx, unk_idx2])
            query_descs = eval_descs[query_idx]
            query_writers = eval_w[query_idx]

            pred_w, max_sim = score_all(query_descs, protos)
            final_pred = pred_w.copy().astype(np.int64)
            final_pred[max_sim < tau] = -1

            m = compute_metrics(query_writers, final_pred, pred_w, unk_set)
            m.update({"ratio": ratio, "repeat": repeat, "tau": round(tau, 4)})
            run_metrics.append(m)

        avg = {
            k: round(float(np.mean([r[k] for r in run_metrics])), 4)
            for k in run_metrics[0] if k not in ("ratio", "repeat")
        }
        avg.update({"ratio": ratio, "n_runs": len(run_metrics)})
        rows.append(avg)

        print(
            f"ratio={ratio:.1f} | "
            f"bal_acc={avg['bal_acc']:.3f} | "
            f"f1_unk={avg['f1_unknown']:.3f} | "
            f"recall={avg['recall_unknown']:.3f} | "
            f"precision={avg['precision_unknown']:.3f} | "
            f"writer_id_acc={avg['writer_id_acc_known']:.3f}"
        )

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved: {OUT_CSV}")
    print(df[["ratio", "tau", "bal_acc", "f1_unknown", "recall_unknown",
              "precision_unknown", "writer_id_acc_known"]].to_string(index=False))


if __name__ == "__main__":
    main()
