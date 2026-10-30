"""
threshold_eval_cvl_iam.py
--------------------------
Open-set writer identification on CVL and IAM using cosine-similarity
thresholding. Consolidates threshold_eval_cvl.py and threshold_eval_iam.py
into one script -- both datasets share the same backbone
(backbone.model.Model wrapping a triplet-trained resnet56 + a pickled
sklearn PCA) and now share the EXACT SAME calibration/sweep pipeline.
Only CONFIG (paths, filename regex, repeats) differs per dataset -- see
DATASET_CONFIGS below.

QUERY DESIGN: UNIFIED ON THE HELD-OUT (NO-LEAKAGE) VERSION
=============================================================
CVL's original script queried EVERY eval page using prototypes fit on
that SAME known writer's full page set -- a known writer's own page
could score against a prototype built partly from itself
(self-similarity leakage in writer_id_acc_known). IAM's script fixed
this by holding out a VAL_FRACTION slice of each known writer's pages as
queries and fitting prototypes only on the remainder.

CVL is now unified onto IAM's held-out design: for BOTH datasets, only a
VAL_FRACTION slice of each known writer's pages reaches the query set;
the rest are reserved to fit that writer's prototype. Unknown writers
still contribute 100% of their pages as queries (they never have a
prototype to reserve pages for). No dataset here uses the leakage-prone
design anymore.

RATIO / TRUE_UNKNOWN_RATE FIX (2026-07-22)
============================================
Because the held-out design queries only ~VAL_FRACTION of each known
writer's pages but ALL of each unknown writer's pages, the nominal
writer-level "ratio" swept from 0.1-0.9 does NOT equal the actual
fraction of QUERY PAGES that are unknown ("true_unknown_rate") -- unknown
writers are overrepresented in the query set relative to known ones.
Previously this made true_unknown_rate drift well above the nominal
ratio (e.g. ratio=0.1 landing around true_unknown_rate=0.26 for IAM).

TARGET_UNKNOWN_RATIOS (0.1 ... 0.9) are now treated as the *desired*
true_unknown_rate itself. For each target T we solve for how many
writers to mark unknown so the resulting query set actually lands on T,
using this eval split's REAL per-writer page counts (not an
assumed-uniform approximation -- IAM in particular has many writers with
as few as 2 pages, already called out as a wrinkle in the original
script). See n_unk_writers_for_target() below for the closed form
(identical to the one in threshold_eval_icdar17.py, since both now use
the same held-out query design).

Output CSVs keep the column name "ratio" for continuity, but it now
means the TARGET true_unknown_rate (0.1, 0.2, ..., 0.9). A new
"writer_unknown_frac" column records the actual writer-level fraction
used to hit that target. "true_unknown_rate" is still the measured value
and should now track "ratio" closely (small residual gaps remain --
rounding to a whole number of writers, and per-repeat sampling variance).
"""

import glob, re, pickle, os
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
from torchvision import transforms
from sklearn.preprocessing import normalize
from sklearn.metrics import balanced_accuracy_score, f1_score, precision_score, recall_score

from backbone import resnets
from backbone.model import Model
from dataloading.GenericDataset import FilepathImageDataset
from dataloading.regex import pil_loader
from page_encodings import SumPooling

# ------------------------------------------------------------------ #
# SHARED CONFIG
# ------------------------------------------------------------------ #
CAL_FRACTION          = 0.2
VAL_FRACTION          = 0.2
TARGET_UNKNOWN_RATIOS = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
RANDOM_SEED           = 42

# ------------------------------------------------------------------ #
# PER-DATASET CONFIG
#
# Both datasets now use the identical held-out query pipeline: a
# VAL_FRACTION slice of each KNOWN writer's pages is held out as queries;
# prototypes are fit only on the remainder. Query set = held-out known
# pages + all unknown-writer pages.
# ------------------------------------------------------------------ #
DATASET_CONFIGS = [
    {
        "name": "cvl",
        "experiment_dir": (
            "/data/rrossner/outlierDetection/icdar23/experiments/"
            "cvl_c32_triplet_m03_s8_100k-2026-06-11-15-02"
        ),
        "train_img_dir": "/data/rrossner/datasets/CVL/cvl_train_patches_bin",
        "test_img_dir":  "/data/rrossner/datasets/CVL/cvl_test_patches_bin",
        "writer_re": re.compile(r'^(\d+)-\d+-cropped_\d+'),
        "page_re":   re.compile(r'^\d+-(\d+)-cropped_\d+'),
        "num_clusters": 32,
        "model_dim": 64,
        "batch_size": 256,
        "n_repeats": 10,
        "out_csv": "threshold_results.csv",
    },
    {
        "name": "iam",
        "experiment_dir": (
            "/data/rrossner/outlierDetection/icdar23/experiments/"
            "iam_c32_triplet_m03_s8_300k-2026-06-29-14-24-40"
        ),
        "train_img_dir": "/data/rrossner/datasets/IAM/iam_train_patches",
        "test_img_dir":  "/data/rrossner/datasets/IAM/iam_test_patches",
        "writer_re": re.compile(r'^(\d+)_'),
        "page_re":   re.compile(r'^\d+_(\d+)_'),
        "num_clusters": 32,
        "model_dim": 64,
        "batch_size": 256,
        "n_repeats": 10,
        "out_csv": "threshold_results_iam.csv",
    },
]


# ------------------------------------------------------------------ #
# GENERIC HELPERS (dataset-agnostic; parameterized by cfg)
# ------------------------------------------------------------------ #
def parse_filename(path, writer_re, page_re):
    stem = Path(path).stem
    return int(writer_re.match(stem).group(1)), int(page_re.match(stem).group(1))


def load_model(model_path, num_clusters, model_dim):
    backbone = resnets.resnet56()
    model    = Model(backbone, dim=model_dim, num_clusters=num_clusters)
    ckpt     = torch.load(model_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval().cuda()
    print(f"Loaded model : {model_path}")
    return model


def extract_patch_features(model, img_dir, writer_re, page_re, batch_size):
    srcs   = sorted(glob.glob(f"{img_dir}/**/*.png", recursive=True))
    print(f"  {len(srcs)} patches in {img_dir}")
    ds     = FilepathImageDataset(srcs, pil_loader, transforms.ToTensor())
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, num_workers=8, pin_memory=True)
    all_feats = []
    with torch.no_grad():
        for imgs in tqdm(loader, desc="  extracting"):
            all_feats.append(F.normalize(model(imgs.cuda()), dim=1).cpu().numpy())
    feats   = np.concatenate(all_feats)
    writers = np.array([parse_filename(s, writer_re, page_re)[0] for s in srcs])
    pages   = np.array([parse_filename(s, writer_re, page_re)[1] for s in srcs])
    return feats, writers, pages


def build_page_descriptors(patch_feats, writers, pages, pca, pooling):
    label_np = np.stack([writers, pages], axis=1)
    unique   = list({(w, p) for w, p in label_np})
    page_descs, page_writers = [], []
    for w, p in tqdm(unique, desc="  pooling pages"):
        idx = np.where((label_np[:, 0] == w) & (label_np[:, 1] == p))[0]
        page_descs.append(pooling.encode([patch_feats[idx]]))
        page_writers.append(w)
    page_descs = normalize(pca.transform(np.concatenate(page_descs)), axis=1)
    return page_descs, np.array(page_writers)


def build_prototypes(descs, writers):
    protos = {}
    for w in np.unique(writers):
        v    = descs[writers == w].mean(axis=0)
        norm = np.linalg.norm(v)
        protos[w] = v / norm if norm > 0 else v
    return protos


def score_all(descs, prototypes):
    writer_order = np.array(sorted(prototypes.keys()))
    proto_mat    = np.stack([prototypes[w] for w in writer_order])
    sims         = descs @ proto_mat.T
    best_idx     = np.argmax(sims, axis=1)
    return writer_order[best_idx], sims[np.arange(len(descs)), best_idx]


def calibrate_tau(known_scores, unknown_scores, n_candidates=500):
    """Sweep tau, pick value maximising balanced accuracy."""
    pooled = np.concatenate([known_scores, unknown_scores])
    taus   = np.quantile(pooled, np.linspace(0.005, 0.995, n_candidates))
    y_true = np.concatenate([
        np.ones(len(known_scores),    dtype=np.int64),
        np.zeros(len(unknown_scores), dtype=np.int64),
    ])
    best_tau, best_bal = taus[0], -1.0
    for tau in taus:
        y_pred = (pooled >= tau).astype(np.int64)
        bal    = balanced_accuracy_score(y_true, y_pred)
        if bal > best_bal:
            best_bal, best_tau = bal, float(tau)
    return best_tau, round(best_bal, 4)


def compute_metrics(query_writers, final_pred, pred_w, unknown_set):
    is_truly_unk = np.isin(query_writers, list(unknown_set)).astype(np.int64)
    is_pred_unk  = (final_pred == -1).astype(np.int64)
    known_mask   = is_truly_unk == 0
    writer_acc   = (
        float(np.mean(pred_w[known_mask] == query_writers[known_mask]))
        if known_mask.any() else 0.0
    )
    return {
        "bal_acc":             round(balanced_accuracy_score(is_truly_unk, is_pred_unk), 4),
        "f1_unknown":          round(f1_score(is_truly_unk, is_pred_unk, pos_label=1, zero_division=0), 4),
        "precision_unknown":   round(precision_score(is_truly_unk, is_pred_unk, pos_label=1, zero_division=0), 4),
        "recall_unknown":      round(recall_score(is_truly_unk, is_pred_unk, pos_label=1, zero_division=0), 4),
        "writer_id_acc_known": round(writer_acc, 4),
        "pred_unknown_rate":   round(float(np.mean(is_pred_unk)), 4),
        "true_unknown_rate":   round(float(np.mean(is_truly_unk)), 4),
    }


# ------------------------------------------------------------------ #
# RATIO <-> TRUE_UNKNOWN_RATE INVERSION
# (identical to threshold_eval_icdar17.py -- both scripts now use the
# same held-out query design, so the same closed form applies)
# ------------------------------------------------------------------ #
def held_pages_if_known(n_pages, val_fraction):
    """Pages of a writer that reach the query set IF it's known."""
    if n_pages <= 1:
        return 0
    n_q = max(1, int(round(n_pages * val_fraction)))
    return min(n_q, n_pages - 1)


def n_unk_writers_for_target(target_true_unknown_rate, pages_per_writer, val_fraction):
    """
    Solve, in expectation and using this eval split's ACTUAL per-writer
    page counts, for how many writers to mark unknown so the query set's
    true_unknown_rate lands on `target_true_unknown_rate`.

        n_unk = T * N * H / (P * (1 - T) + T * H)

    where P = total pages (all-unknown basis), H = total held-query
    pages if every writer were known (all-known basis), N = num writers,
    T = target true_unknown_rate.
    """
    N = len(pages_per_writer)
    P = float(np.sum(pages_per_writer))
    H = float(sum(held_pages_if_known(p, val_fraction) for p in pages_per_writer))
    T = target_true_unknown_rate
    x = (T * N * H) / (P * (1 - T) + T * H)
    n_unk = int(round(x))
    return min(max(n_unk, 1), N - 1)


# ------------------------------------------------------------------ #
# PER-DATASET PIPELINE
# ------------------------------------------------------------------ #
def run_dataset(cfg):
    import pandas as pd

    print("=" * 70)
    print(f"DATASET: {cfg['name']}")
    print("=" * 70)

    model_path = os.path.join(cfg["experiment_dir"], "model.pt")
    pca_path   = os.path.join(cfg["experiment_dir"], "pca.pkl")

    model   = load_model(model_path, cfg["num_clusters"], cfg["model_dim"])
    pca     = pickle.load(open(pca_path, "rb"))
    pooling = SumPooling("l2", pn_alpha=0.4)
    print(f"Loaded PCA   : {pca_path}  (n_components={pca.n_components_})")

    print("\nExtracting TRAIN patch features...")
    tr_feats, tr_writers, tr_pages = extract_patch_features(
        model, cfg["train_img_dir"], cfg["writer_re"], cfg["page_re"], cfg["batch_size"])
    print("\nExtracting TEST patch features...")
    te_feats, te_writers, te_pages = extract_patch_features(
        model, cfg["test_img_dir"], cfg["writer_re"], cfg["page_re"], cfg["batch_size"])

    print("\nBuilding TRAIN page descriptors...")
    _train_descs, _train_writers = build_page_descriptors(tr_feats, tr_writers, tr_pages, pca, pooling)
    print("\nBuilding TEST page descriptors...")
    test_descs, test_writers = build_page_descriptors(te_feats, te_writers, te_pages, pca, pooling)

    all_test_writers = np.unique(test_writers)
    print(f"\nTrain : {len(_train_descs)} pages, {len(np.unique(_train_writers))} writers")
    print(f"Test  : {len(test_descs)} pages, {len(all_test_writers)} writers")

    # ------------------------------------------------------------------
    # OPTION B: permanent calibration / evaluation split
    # ------------------------------------------------------------------
    rng_split = np.random.default_rng(RANDOM_SEED)
    n_cal     = max(2, int(round(len(all_test_writers) * CAL_FRACTION)))
    cal_writers  = set(rng_split.choice(all_test_writers, size=n_cal, replace=False).tolist())
    eval_writers = set(all_test_writers.tolist()) - cal_writers

    print(f"\nPermanent split (seed={RANDOM_SEED}):")
    print(f"  calibration writers : {len(cal_writers)}  (never evaluated)")
    print(f"  evaluation writers  : {len(eval_writers)}  (never used for tau)")

    # ------------------------------------------------------------------
    # CALIBRATE TAU on cal_writers only (identical for both datasets)
    # ------------------------------------------------------------------
    rng_cal   = np.random.default_rng(RANDOM_SEED + 1)
    cal_arr   = np.array(sorted(cal_writers))
    n_cal_unk = max(1, int(round(len(cal_arr) * 0.2)))
    cal_unk   = set(rng_cal.choice(cal_arr, size=n_cal_unk, replace=False).tolist())
    cal_known = cal_writers - cal_unk

    fit_idx, val_idx = [], []
    for w in cal_known:
        idx      = np.where(test_writers == w)[0]
        shuffled = rng_cal.permutation(idx)
        n_val    = max(1, int(round(len(idx) * VAL_FRACTION)))
        n_val    = min(n_val, len(idx) - 1) if len(idx) > 1 else 0
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
    _, unk_scores   = score_all(test_descs[unk_idx],  cal_prototypes)

    print(f"  known val score  mean={known_scores.mean():.3f}  std={known_scores.std():.3f}")
    print(f"  unknown   score  mean={unk_scores.mean():.3f}  std={unk_scores.std():.3f}")

    tau, cal_bal = calibrate_tau(known_scores, unk_scores)
    print(f"\nCalibrated tau : {tau:.4f}  (cal balanced_acc={cal_bal})")

    # ------------------------------------------------------------------
    # SWEEP on eval_writers only (tau is now fixed, never changes)
    # ------------------------------------------------------------------
    eval_arr   = np.array(sorted(eval_writers))
    eval_mask  = np.isin(test_writers, eval_arr)
    eval_descs = test_descs[eval_mask]
    eval_w     = test_writers[eval_mask]

    # per-writer page counts within the eval split, aligned to eval_arr,
    # needed by the ratio<->true_unknown_rate inversion below.
    id_to_count      = dict(zip(*np.unique(eval_w, return_counts=True)))
    pages_per_writer = np.array([id_to_count[w] for w in eval_arr])

    print(f"\nEvaluation writers : {len(eval_writers)}")
    print(f"Evaluation pages   : {len(eval_descs)}")
    print(f"Sweeping targets   : {TARGET_UNKNOWN_RATIOS}")
    print("-" * 70)

    rows = []
    for target in TARGET_UNKNOWN_RATIOS:
        n_unk = n_unk_writers_for_target(target, pages_per_writer, VAL_FRACTION)
        writer_unknown_frac = n_unk / len(eval_arr)

        run_metrics = []
        for repeat in range(cfg["n_repeats"]):
            rng2 = np.random.default_rng(RANDOM_SEED + repeat * 100)
            unk_set   = set(rng2.choice(eval_arr, size=n_unk, replace=False).tolist())
            known_set = set(eval_arr.tolist()) - unk_set

            # Held-out split per known writer -- a writer's own queried
            # page must never have contributed to its own prototype, or
            # writer_id_acc_known would measure self-similarity, not
            # retrieval. Same design for both CVL and IAM now.
            fit_idx2, held_idx = [], []
            for w in known_set:
                idx      = np.where(eval_w == w)[0]
                shuffled = rng2.permutation(idx)
                n_val    = max(1, int(round(len(idx) * VAL_FRACTION)))
                n_val    = min(n_val, len(idx) - 1) if len(idx) > 1 else 0
                held_idx.extend(shuffled[:n_val].tolist())
                fit_idx2.extend(shuffled[n_val:].tolist())

            fit_idx2 = np.array(fit_idx2)
            held_idx = np.array(held_idx)
            unk_idx2 = np.where(np.isin(eval_w, list(unk_set)))[0]

            protos = build_prototypes(eval_descs[fit_idx2], eval_w[fit_idx2])

            # query set: held-out known pages + all unknown-writer pages
            query_idx     = np.concatenate([held_idx, unk_idx2])
            query_descs   = eval_descs[query_idx]
            query_writers = eval_w[query_idx]

            pred_w, max_sim = score_all(query_descs, protos)
            final_pred      = pred_w.copy().astype(np.int64)
            final_pred[max_sim < tau] = -1

            m = compute_metrics(query_writers, final_pred, pred_w, unk_set)
            m.update({"ratio": target, "repeat": repeat, "tau": round(tau, 4)})
            run_metrics.append(m)

        avg = {
            k: round(float(np.mean([r[k] for r in run_metrics])), 4)
            for k in run_metrics[0] if k not in ("ratio", "repeat")
        }
        avg.update({
            "ratio": target,
            "writer_unknown_frac": round(writer_unknown_frac, 4),
            "n_runs": len(run_metrics),
        })
        rows.append(avg)

        print(
            f"target={target:.1f} | writer_frac={writer_unknown_frac:.3f} | "
            f"true_unk={avg['true_unknown_rate']:.3f} | "
            f"bal_acc={avg['bal_acc']:.3f} | "
            f"f1_unk={avg['f1_unknown']:.3f} | "
            f"recall={avg['recall_unknown']:.3f} | "
            f"precision={avg['precision_unknown']:.3f} | "
            f"writer_id_acc={avg['writer_id_acc_known']:.3f}"
        )

    df = pd.DataFrame(rows)
    df.to_csv(cfg["out_csv"], index=False)
    print(f"\nSaved: {cfg['out_csv']}")
    print(df[["ratio", "writer_unknown_frac", "true_unknown_rate", "tau", "bal_acc",
              "f1_unknown", "recall_unknown", "precision_unknown",
              "writer_id_acc_known"]].to_string(index=False))


def main():
    for cfg in DATASET_CONFIGS:
        run_dataset(cfg)


if __name__ == "__main__":
    main()
