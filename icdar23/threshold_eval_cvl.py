"""
threshold_eval_cvl.py
---------------------
Open-set writer identification on CVL using cosine-similarity thresholding.
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
    "/data/rrossner/outlierDetection/icdar23/experiments/"
    "cvl_c32_triplet_m03_s8_100k-2026-06-11-15-02"
)
MODEL_PATH    = os.path.join(EXPERIMENT_DIR, "model.pt")
PCA_PATH      = os.path.join(EXPERIMENT_DIR, "pca.pkl")
TRAIN_IMG_DIR = "/data/rrossner/datasets/CVL/cvl_train_patches_bin"
TEST_IMG_DIR  = "/data/rrossner/datasets/CVL/cvl_test_patches_bin"

WRITER_RE = re.compile(r'^(\d+)-\d+-cropped_\d+')
PAGE_RE   = re.compile(r'^\d+-(\d+)-cropped_\d+')

NUM_CLUSTERS     = 32
MODEL_DIM        = 64
BATCH_SIZE       = 256
CAL_FRACTION     = 0.2
VAL_FRACTION     = 0.2
UNKNOWN_RATIOS   = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
N_REPEATS        = 3
RANDOM_SEED      = 42
OUT_CSV          = "threshold_results.csv"
# ------------------------------------------------------------------ #


def parse_filename(path):
    stem = Path(path).stem
    return int(WRITER_RE.match(stem).group(1)), int(PAGE_RE.match(stem).group(1))


def load_model():
    backbone = resnets.resnet56()
    model    = Model(backbone, dim=MODEL_DIM, num_clusters=NUM_CLUSTERS)
    ckpt     = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval().cuda()
    print(f"Loaded model : {MODEL_PATH}")
    return model


def extract_patch_features(model, img_dir):
    srcs   = sorted(glob.glob(f"{img_dir}/**/*.png", recursive=True))
    print(f"  {len(srcs)} patches in {img_dir}")
    ds     = FilepathImageDataset(srcs, pil_loader, transforms.ToTensor())
    loader = torch.utils.data.DataLoader(ds, batch_size=BATCH_SIZE, num_workers=8, pin_memory=True)
    all_feats = []
    with torch.no_grad():
        for imgs in tqdm(loader, desc="  extracting"):
            all_feats.append(F.normalize(model(imgs.cuda()), dim=1).cpu().numpy())
    feats   = np.concatenate(all_feats)
    writers = np.array([parse_filename(s)[0] for s in srcs])
    pages   = np.array([parse_filename(s)[1] for s in srcs])
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
    pooling = SumPooling("l2", pn_alpha=0.4)
    print(f"Loaded PCA   : {PCA_PATH}  (n_components={pca.n_components_})")

    print("\nExtracting TRAIN patch features...")
    tr_feats, tr_writers, tr_pages = extract_patch_features(model, TRAIN_IMG_DIR)
    print("\nExtracting TEST patch features...")
    te_feats, te_writers, te_pages = extract_patch_features(model, TEST_IMG_DIR)

    print("\nBuilding TRAIN page descriptors...")
    _train_descs, _train_writers = build_page_descriptors(tr_feats, tr_writers, tr_pages, pca, pooling)
    print("\nBuilding TEST page descriptors...")
    test_descs, test_writers = build_page_descriptors(te_feats, te_writers, te_pages, pca, pooling)

    all_test_writers = np.unique(test_writers)
    print(f"\nTrain : {len(_train_descs)} pages, {len(np.unique(_train_writers))} writers")
    print(f"Test  : {len(test_descs)} pages, {len(all_test_writers)} writers")

    # ------------------------------------------------------------------
    # OPTION B: permanent calibration / evaluation split
    #
    # cal_writers  (20%, fixed) -> used ONLY to calibrate tau, never swept
    # eval_writers (80%, fixed) -> used ONLY in the ratio sweep, never for tau
    # ------------------------------------------------------------------
    rng_split = np.random.default_rng(RANDOM_SEED)
    n_cal     = max(2, int(round(len(all_test_writers) * CAL_FRACTION)))
    cal_writers  = set(rng_split.choice(all_test_writers, size=n_cal, replace=False).tolist())
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
    print(f"\nCalibrated tau : {tau:.4f}  (cal balanced_acc={cal_bal})")

    # ------------------------------------------------------------------
    # SWEEP on eval_writers only (tau is now fixed, never changes)
    # ------------------------------------------------------------------
    eval_arr = np.array(sorted(eval_writers))
    # restrict test_descs / test_writers to eval writers only
    eval_mask   = np.isin(test_writers, eval_arr)
    eval_descs  = test_descs[eval_mask]
    eval_w      = test_writers[eval_mask]

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

            # prototypes from known eval writers
            known_mask = np.isin(eval_w, list(known_set))
            protos     = build_prototypes(eval_descs[known_mask], eval_w[known_mask])

            pred_w, max_sim = score_all(eval_descs, protos)
            final_pred      = pred_w.copy().astype(np.int64)
            final_pred[max_sim < tau] = -1

            m = compute_metrics(eval_w, final_pred, pred_w, unk_set)
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