"""
predict_labels_svm_icdar17.py
------------------------------
Open-set writer identification benchmark on ICDAR17/HistoricalWI using
per-writer exemplar (one-vs-rest) LinearSVC classifiers, built on the
icdar24-master ("SAGHOG") codebase with the supervised resnet56 + NetRVLAD
architecture.

THIS IS THE SELF-CONTAINED VERSION
===================================
There is no external "-1 unknown" labeled CSV and no submission file here.
Exactly like threshold_eval_icdar17.py, this script:
  1. extracts page descriptors for whichever ICDAR17/HistoricalWI writer
     population you point it at (train patches, test patches, or your own
     folder -- see PATCH_DIR below),
  2. permanently splits writers into a small calibration pool (never
     evaluated) and a large evaluation pool (never used for tau),
  3. calibrates a rejection threshold `tau` on the calibration pool by
     synthetically holding out some of ITS writers as fake "unknowns",
  4. sweeps a range of unknown-writer ratios on the evaluation pool,
     each time synthetically hiding a subset of eval writers as unknown
     and measuring how well the model (a) rejects genuinely unknown
     writers at the fixed tau and (b) still identifies known ones.

The only thing that differs from threshold_eval_icdar17.py is the scorer:
instead of nearest-cosine-to-mean-prototype, this fits one binary LinearSVC
per known writer (this writer vs. every other known writer) and scores by
max decision function -- see "Exemplar SVMs" below for why/how.

IMPORTANT: don't naively combine train + test patches as your PATCH_DIR.
ICDAR17 train and test writer IDs are numbered independently by the
dataset, so writer "5" in train and writer "5" in test are not the same
person. Point this at ONE split (or your own folder where you've already
made writer IDs globally unique) at a time.

EXEMPLAR SVMS
=============
For each known writer w, fit one binary LinearSVC:
    positive = writer w's (fit-split) pages
    negative = every other known writer's (fit-split) pages
At scoring time, every candidate page is run through all K classifiers;
the predicted writer is whichever gives the highest decision-function
margin, and that margin (not a probability) is compared against `tau` to
decide known-vs-unknown. This gives a real geometric margin to threshold
(rather than an often-miscalibrated softmax), and scales incrementally
(adding a writer later = one more small SVM, not retraining a shared
boundary).

NOTE ON FIT COST: fitting ~500+ small LinearSVCs per repeat, across
N_REPEATS x len(UNKNOWN_RATIOS) repeats, is real compute (unlike the
cosine-prototype version in threshold_eval_icdar17.py, which is nearly
free per repeat). If a first run is too slow, lower N_REPEATS.
"""

import logging
import os
import re

import numpy as np
from sklearn.svm import LinearSVC
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
os.environ.setdefault("ICDAR24_ROOT", "/data/rrossner/outlierDetection/icdar24")
CHECKPOINT_PATH = "/data/rrossner/outlierDetection/icdar24/experiments/resnet56_0.5_writer-2026-07-02-10-34/model.pt"

# --- data source: pick ONE of the two modes (see icdar24_page_features.py) ---
# "csv"     -> native icdar24 pipeline: full binarized page images, on-the-fly
#              SIFT patch extraction (matches main_finetune.py's eval path).
# "patches" -> pre-extracted patch PNGs on disk, grouped per page by regex
#              (e.g. icdar24's own utils/extract_patches_only.py output).
DATA_MODE = "patches"

CSV_PATH = "datasets/icdar2017-test-binarized.csv"   # used if DATA_MODE == "csv"

# Point this at ONE writer population -- train patches, test patches, or
# your own folder. See the module docstring above re: not combining splits.
PATCH_DIR = "/data/rrossner/datasets/ICDAR17/binarized_test/patches"
WRITER_RE = re.compile(r'^(\d+)-IMG_MAX_')
PAGE_RE   = re.compile(r'^\d+-IMG_MAX_(\d+)_\d+$')

# --- model / architecture (must match training config) ---
NUM_CLUSTERS = 100
RANDOM_VLAD = True      # True = NetRVLAD
IN_DIM = -1
PATCH_SIZE = 32
DEVICE = "cuda"

# --- optional page-descriptor-level PCA (icdar24 eval_options convention) ---
APPLY_PCA = True
PCA_DIM = 512

# --- permanent calibration/evaluation writer split (same protocol as
#     threshold_eval_icdar17.py) ---
CAL_FRACTION = 0.2
VAL_FRACTION = 0.2
UNKNOWN_RATIOS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
N_REPEATS = 3          # SVM fitting is much slower per-repeat than cosine-prototype; start low
RANDOM_SEED = 42
OUT_CSV = "svm_results_icdar17.csv"

# Threshold calibration objective:
#   "balanced_acc" -> best average of known-accept and unknown-reject
#   "f1"           -> best F1 for predicting "known"
THRESHOLD_MODE = "balanced_acc"

# --- per-writer exemplar SVM settings ---
SVM_C = 0.1
SVM_MAX_ITER = 10000
NEGATIVE_RATIO = 20       # sample at most NEGATIVE_RATIO * #positives negatives per writer
USE_CLASS_WEIGHT = True

# Cache page descriptors to disk so re-running the sweep (or trying a
# different SVM_C) doesn't require re-extracting patches.
CACHE_PATH = "icdar17_page_descriptors.npz"
# ------------------------------------------------------------------ #


# ----------------------------
# Exemplar / one-vs-rest SVMs
# ----------------------------
def fit_exemplar_svms(X, y, all_unknown_X=None, C=1.0, max_iter=10000,
                       negative_ratio=20, use_class_weight=True, seed=42):
    """
    Fit one binary SVM per writer:
      positive = this writer
      negative = all other known writers (+ optional extra negatives, e.g.
                 held-out synthetic-unknown pages during tau calibration)
    """
    rng = np.random.default_rng(seed)
    writers = np.unique(y)
    models = {}

    for w in writers:
        pos_mask = (y == w)
        X_pos = X[pos_mask]
        X_neg = X[~pos_mask]

        if all_unknown_X is not None and len(all_unknown_X) > 0:
            X_neg = np.concatenate([X_neg, all_unknown_X], axis=0)

        n_pos = len(X_pos)
        max_neg = max(n_pos * negative_ratio, n_pos)
        if len(X_neg) > max_neg:
            neg_sel = rng.choice(len(X_neg), size=max_neg, replace=False)
            X_neg = X_neg[neg_sel]

        X_bin = np.concatenate([X_pos, X_neg], axis=0)
        y_bin = np.concatenate(
            [np.ones(len(X_pos), dtype=np.int64), np.zeros(len(X_neg), dtype=np.int64)], axis=0
        )

        clf = LinearSVC(
            C=C, class_weight="balanced" if use_class_weight else None,
            max_iter=max_iter, dual="auto", random_state=seed,
        )
        clf.fit(X_bin, y_bin)
        models[int(w)] = clf

    return models, writers


def score_exemplar_svms(models, writers, X):
    """
    Returns:
      pred_writer : [M] predicted known writer (argmax over writer SVM scores)
      max_score   : [M] max SVM decision-function score
    """
    score_cols = [np.asarray(models[int(w)].decision_function(X)).reshape(-1, 1) for w in writers]
    score_mat = np.concatenate(score_cols, axis=1)
    pred_idx = np.argmax(score_mat, axis=1)
    max_score = score_mat[np.arange(len(X)), pred_idx]
    pred_writer = writers[pred_idx]
    return pred_writer, max_score


# ----------------------------
# Tau calibration (score-array based -- identical in spirit to
# threshold_eval_icdar17.py's calibrate_tau, just balanced_acc/f1 aware)
# ----------------------------
def calibrate_tau(known_scores, unknown_scores, mode="balanced_acc", n_candidates=500):
    pooled = np.concatenate([known_scores, unknown_scores])
    taus = np.quantile(pooled, np.linspace(0.005, 0.995, n_candidates))
    y_true = np.concatenate([
        np.ones(len(known_scores), dtype=np.int64),
        np.zeros(len(unknown_scores), dtype=np.int64),
    ])
    best_tau, best_score = taus[0], -1.0
    for tau in taus:
        y_pred = (pooled >= tau).astype(np.int64)
        score = (balanced_accuracy_score(y_true, y_pred) if mode == "balanced_acc"
                 else f1_score(y_true, y_pred, pos_label=1, zero_division=0))
        if score > best_score:
            best_score, best_tau = score, float(tau)
    return best_tau, round(best_score, 4)


def compute_metrics(query_writers, final_pred, pred_w, unknown_set):
    is_truly_unk = np.isin(query_writers, list(unknown_set)).astype(np.int64)
    is_pred_unk = (final_pred == -1).astype(np.int64)
    known_mask = is_truly_unk == 0
    writer_acc = (
        float(np.mean(pred_w[known_mask] == query_writers[known_mask]))
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
        CHECKPOINT_PATH, num_clusters=NUM_CLUSTERS, random_vlad=RANDOM_VLAD,
        in_dim=IN_DIM, device=DEVICE,
    )

    print("\nExtracting page descriptors (memory-efficient, one page at a time)...")
    if DATA_MODE == "csv":
        descs, writers, _pages = extract_pages_from_csv(
            CSV_PATH, model, device=DEVICE, patch_size=PATCH_SIZE, cache_path=CACHE_PATH,
        )
    elif DATA_MODE == "patches":
        descs, writers, _pages = extract_pages_from_patch_dir(
            PATCH_DIR, model, writer_re=WRITER_RE, page_re=PAGE_RE,
            device=DEVICE, cache_path=CACHE_PATH,
        )
    else:
        raise ValueError(f"Unknown DATA_MODE: {DATA_MODE}")

    all_writers = np.unique(writers)
    print(f"Pool : {len(descs)} pages, {len(all_writers)} writers, descriptor dim={descs.shape[1]}")

    if APPLY_PCA:
        descs, _pca = apply_pca(descs, pca_dim=PCA_DIM)
        print(f"After PCA: descriptor dim={descs.shape[1]}")

    # ------------------------------------------------------------------
    # Permanent calibration / evaluation writer split (identical protocol
    # to threshold_eval_icdar17.py, so results are directly comparable).
    # ------------------------------------------------------------------
    rng_split = np.random.default_rng(RANDOM_SEED)
    n_cal = max(2, int(round(len(all_writers) * CAL_FRACTION)))
    cal_writers = set(rng_split.choice(all_writers, size=n_cal, replace=False).tolist())
    eval_writers = set(all_writers.tolist()) - cal_writers

    print(f"\nPermanent split (seed={RANDOM_SEED}):")
    print(f"  calibration writers : {len(cal_writers)}  (never evaluated)")
    print(f"  evaluation writers  : {len(eval_writers)}  (never used for tau)")

    # ------------------------------------------------------------------
    # CALIBRATE TAU on cal_writers only.
    # Synthetic unknowns ARE used as extra negatives here (unlike the
    # eval sweep below) -- during calibration we want the SVMs to see
    # what "not a known writer" looks like, since that's exactly what
    # we're tuning tau against.
    # ------------------------------------------------------------------
    rng_cal = np.random.default_rng(RANDOM_SEED + 1)
    cal_arr = np.array(sorted(cal_writers))
    n_cal_unk = max(1, int(round(len(cal_arr) * 0.2)))
    cal_unk = set(rng_cal.choice(cal_arr, size=n_cal_unk, replace=False).tolist())
    cal_known = cal_writers - cal_unk

    fit_idx, val_idx = [], []
    for w in cal_known:
        idx = np.where(writers == w)[0]
        shuffled = rng_cal.permutation(idx)
        n_val = max(1, int(round(len(idx) * VAL_FRACTION)))
        n_val = min(n_val, len(idx) - 1) if len(idx) > 1 else 0
        val_idx.extend(shuffled[:n_val].tolist())
        fit_idx.extend(shuffled[n_val:].tolist())

    fit_idx = np.array(fit_idx)
    val_idx = np.array(val_idx)
    unk_idx = np.where(np.isin(writers, list(cal_unk)))[0]

    print(f"\nTau calibration (cal_writers only):")
    print(f"  cal known writers   : {len(cal_known)}  ({n_cal_unk} held out as unknowns)")
    print(f"  fit pages           : {len(fit_idx)}")
    print(f"  known val pages     : {len(val_idx)}")
    print(f"  synthetic unk pages : {len(unk_idx)}")

    cal_models, cal_writer_list = fit_exemplar_svms(
        descs[fit_idx], writers[fit_idx], all_unknown_X=descs[unk_idx],
        C=SVM_C, max_iter=SVM_MAX_ITER, negative_ratio=NEGATIVE_RATIO,
        use_class_weight=USE_CLASS_WEIGHT, seed=RANDOM_SEED,
    )
    _, known_scores = score_exemplar_svms(cal_models, cal_writer_list, descs[val_idx])
    _, unk_scores = score_exemplar_svms(cal_models, cal_writer_list, descs[unk_idx])

    print(f"  known val score  mean={known_scores.mean():.3f}  std={known_scores.std():.3f}")
    print(f"  unknown   score  mean={unk_scores.mean():.3f}  std={unk_scores.std():.3f}")

    tau, cal_score = calibrate_tau(known_scores, unk_scores, mode=THRESHOLD_MODE)
    print(f"\nCalibrated tau : {tau:.4f}  ({THRESHOLD_MODE}={cal_score})")

    # ------------------------------------------------------------------
    # SWEEP on eval_writers only (tau is now fixed, never changes).
    # Unlike calibration, unknown-writer pages are NOT used as SVM
    # negatives here -- they're purely held out, so this sweep measures
    # exactly what threshold_eval_icdar17.py measures for its scorer:
    # can a fixed tau reject writers the classifiers never saw at all.
    # ------------------------------------------------------------------
    eval_arr = np.array(sorted(eval_writers))
    eval_mask = np.isin(writers, eval_arr)
    eval_descs = descs[eval_mask]
    eval_w = writers[eval_mask]

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
            # must never have contributed to its own SVM's training, or
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

            models, writer_list = fit_exemplar_svms(
                eval_descs[fit_idx2], eval_w[fit_idx2], all_unknown_X=None,
                C=SVM_C, max_iter=SVM_MAX_ITER, negative_ratio=NEGATIVE_RATIO,
                use_class_weight=USE_CLASS_WEIGHT, seed=RANDOM_SEED + repeat,
            )

            query_idx = np.concatenate([held_idx, unk_idx2])
            query_descs = eval_descs[query_idx]
            query_writers = eval_w[query_idx]

            pred_w, max_score = score_exemplar_svms(models, writer_list, query_descs)
            final_pred = pred_w.copy().astype(np.int64)
            final_pred[max_score < tau] = -1

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