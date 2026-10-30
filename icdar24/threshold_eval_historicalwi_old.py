"""
threshold_eval_historicalwi.py
-------------------------------
Open-set writer identification on ICDAR2017 Historical-WI using
cosine-similarity thresholding.

MEMORY FIX (see chat): previously extracted+pooled every patch in
TRAIN_IMG_DIR just to print a count that's never used again, and
extracted+pooled every TEST patch in one giant array before indexing
per page. Both are now removed/streamed:
  - train pass is filename-only (no model, no images) since the
    descriptors were dead weight
  - test pass streams one page at a time: extract -> pool -> PCA ->
    keep only the small descriptor, discard patch tensors
Numerically identical results; peak RAM now bounded by the largest
single page instead of the whole dataset.
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
# CONFIG
# ------------------------------------------------------------------ #
EXPERIMENT_DIR = (
    "/data/rrossner/outlierDetection/icdar24/experiments/"
    "resnet56_0.5_writer-2026-07-02-10-34"
)
MODEL_PATH    = os.path.join(EXPERIMENT_DIR, "model.pt")
PCA_PATH      = os.path.join(EXPERIMENT_DIR, "pca.pkl")
TRAIN_IMG_DIR = "/data/rrossner/datasets/ICDAR17/binarized_train/patches"
TEST_IMG_DIR  = "/data/rrossner/datasets/ICDAR17/binarized_test/patches"

NUM_CLUSTERS     = 100
MODEL_DIM        = 64
BATCH_SIZE       = 256
PAGE_LOADER_WORKERS = 0   # per-page batches are small; 0 avoids process-spawn
                          # overhead across ~thousands of tiny DataLoaders.
                          # Bump up only if you profile this as I/O-bound.
CAL_FRACTION     = 0.2
VAL_FRACTION     = 0.2
UNKNOWN_RATIOS   = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
N_REPEATS        = 10
RANDOM_SEED      = 42
OUT_CSV          = "threshold_results_historicalwi.csv"
# ------------------------------------------------------------------ #

_SEP_RE = re.compile(r'^(\d+)([_-])')


def parse_filename(path):
    """
    Returns (writer: int, page: str).
    Train  {writerID}_{pageStr}_{patchIdx}.png
    Test   {writerID}-{pageStr}_{patchIdx}.png
    """
    stem = Path(path).stem
    m    = _SEP_RE.match(stem)
    writer = int(m.group(1))
    sep    = m.group(2)
    if sep == '_':
        parts = stem.split('_')
        page  = '_'.join(parts[1:-1])
    else:
        rest = stem[len(m.group(1)) + 1:]
        page = rest.rsplit('_', 1)[0]
    return writer, page


def load_model():
    backbone = resnets.resnet56()
    model    = Model(backbone, dim=MODEL_DIM, num_clusters=NUM_CLUSTERS)
    ckpt     = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval().cuda()
    print(f"Loaded model : {MODEL_PATH}")
    return model


def count_pages_writers(img_dir):
    """
    Filenames only -- no image loading, no model inference, no PCA.
    Train descriptors were never used past a summary print, so there's
    no reason to run the full (expensive) pipeline just for counts.
    """
    srcs   = sorted(glob.glob(f"{img_dir}/**/*.png", recursive=True))
    parsed = {parse_filename(s) for s in srcs}
    writers = {w for w, p in parsed}
    return len(parsed), len(writers)


def list_pages(img_dir):
    """
    Single cheap pass over filenames (strings only) grouping them into
    {(writer, page): [filepaths]}. No images opened, no model touched.
    """
    srcs = sorted(glob.glob(f"{img_dir}/**/*.png", recursive=True))
    print(f"  {len(srcs)} patches in {img_dir}")
    pages = {}
    for s in srcs:
        w, p = parse_filename(s)
        pages.setdefault((w, p), []).append(s)
    print(f"  {len(pages)} unique pages")
    return pages


def encode_one_page(model, pooling, pca, filepaths):
    """
    Runs inference + pooling + PCA for ONE page's patches only.
    Peak memory here is bounded by this page's patch count, not the
    whole dataset. Patch-level features are discarded once this
    returns -- only the small pooled+PCA'd descriptor survives.
    """
    ds     = FilepathImageDataset(filepaths, pil_loader, transforms.ToTensor())
    loader = torch.utils.data.DataLoader(
        ds, batch_size=BATCH_SIZE, num_workers=PAGE_LOADER_WORKERS, pin_memory=True
    )
    feats = []
    with torch.no_grad():
        for imgs in loader:
            feats.append(F.normalize(model(imgs.cuda()), dim=1).cpu().numpy())
    feats = np.concatenate(feats)                     # [n_patches_this_page, MODEL_DIM]
    desc  = pooling.encode([feats])                    # [1, pooled_dim]
    desc  = normalize(pca.transform(desc), axis=1)      # [1, pca_dim]
    return desc[0]


def build_page_descriptors_streaming(model, pooling, pca, img_dir):
    pages = list_pages(img_dir)
    descs, writers = [], []
    for (w, p), filepaths in tqdm(pages.items(), desc=f"  encoding pages ({Path(img_dir).name})"):
        descs.append(encode_one_page(model, pooling, pca, filepaths))
        writers.append(w)
    descs   = np.stack(descs).astype(np.float32)
    writers = np.array(writers, dtype=np.int64)
    return descs, writers


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


def compute_metrics(test_writers, final_pred, pred_w, unknown_set):
    is_truly_unk = np.isin(test_writers, list(unknown_set)).astype(np.int64)
    is_pred_unk  = (final_pred == -1).astype(np.int64)
    known_mask   = is_truly_unk == 0
    writer_acc   = (
        float(np.mean(pred_w[known_mask] == test_writers[known_mask]))
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


def main():
    import pandas as pd

    model   = load_model()
    pca     = pickle.load(open(PCA_PATH, "rb"))
    pooling = SumPooling("powernorm", pn_alpha=0.4)
    print(f"Loaded PCA   : {PCA_PATH}  (n_components={pca.n_components_})")

    print("\nCounting TRAIN pages/writers (filenames only -- descriptors unused downstream)...")
    n_train_pages, n_train_writers = count_pages_writers(TRAIN_IMG_DIR)
    print(f"Train : {n_train_pages} pages, {n_train_writers} writers")

    print("\nEncoding TEST pages (streaming, one page at a time)...")
    test_descs, test_writers = build_page_descriptors_streaming(model, pooling, pca, TEST_IMG_DIR)

    all_test_writers = np.unique(test_writers)
    print(f"Test  : {len(test_descs)} pages, {len(all_test_writers)} writers")
    # Expected: ~3600 test pages from 720 writers (5 pages each)

    # ------------------------------------------------------------------
    # Everything below is UNCHANGED -- only ever used test_descs/test_writers
    # ------------------------------------------------------------------
    rng_split = np.random.default_rng(RANDOM_SEED)
    n_cal     = max(2, int(round(len(all_test_writers) * CAL_FRACTION)))
    cal_writers  = set(rng_split.choice(all_test_writers, size=n_cal, replace=False).tolist())
    eval_writers = set(all_test_writers.tolist()) - cal_writers

    print(f"\nPermanent split (seed={RANDOM_SEED}):")
    print(f"  calibration writers : {len(cal_writers)}  (never evaluated)")
    print(f"  evaluation writers  : {len(eval_writers)}  (never used for tau)")

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
    np.set_printoptions(precision=8, suppress=False)
    print("known_scores[:20] :", known_scores[:20])
    print("unk_scores[:20]   :", unk_scores[:20])
    print(f"known: min={known_scores.min():.8f} max={known_scores.max():.8f} std={known_scores.std():.10f}")
    print(f"unk:   min={unk_scores.min():.8f} max={unk_scores.max():.8f} std={unk_scores.std():.10f}")

    import matplotlib.pyplot as plt
    plt.hist(known_scores, bins=50, alpha=0.5, label="known")
    plt.hist(unk_scores, bins=50, alpha=0.5, label="unknown")
    plt.legend()
    plt.savefig("score_hist.png")
    print(f"\nCalibrated tau : {tau:.4f}  (cal balanced_acc={cal_bal})")

    eval_arr  = np.array(sorted(eval_writers))
    eval_mask = np.isin(test_writers, eval_arr)
    eval_descs = test_descs[eval_mask]
    eval_w     = test_writers[eval_mask]

    print(f"\nEvaluation writers : {len(eval_writers)}")
    print(f"Evaluation pages   : {len(eval_descs)}")
    print(f"Sweeping ratios    : {UNKNOWN_RATIOS}")
    print("-" * 70)

    rows = []
    for ratio in UNKNOWN_RATIOS:
        run_metrics = []
        for repeat in range(N_REPEATS):
            rng2  = np.random.default_rng(RANDOM_SEED + repeat * 100)
            n_unk = max(1, int(round(len(eval_arr) * ratio)))
            unk_set   = set(rng2.choice(eval_arr, size=n_unk, replace=False).tolist())
            known_set = set(eval_arr.tolist()) - unk_set

            fit_idx, held_idx = [], []
            for w in known_set:
                idx      = np.where(eval_w == w)[0]
                shuffled = rng2.permutation(idx)
                n_val    = max(1, int(round(len(idx) * VAL_FRACTION)))
                n_val    = min(n_val, len(idx) - 1)
                held_idx.extend(shuffled[:n_val].tolist())
                fit_idx.extend(shuffled[n_val:].tolist())

            fit_idx  = np.array(fit_idx)
            held_idx = np.array(held_idx)
            unk_idx  = np.where(np.isin(eval_w, list(unk_set)))[0]

            protos = build_prototypes(eval_descs[fit_idx], eval_w[fit_idx])

            query_idx     = np.concatenate([held_idx, unk_idx])
            query_descs   = eval_descs[query_idx]
            query_writers = eval_w[query_idx]

            pred_w, max_sim = score_all(query_descs, protos)
            final_pred      = pred_w.copy().astype(np.int64)
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
    print(df[["ratio","tau","bal_acc","f1_unknown","recall_unknown",
              "precision_unknown","writer_id_acc_known"]].to_string(index=False))


if __name__ == "__main__":
    main()