"""
threshold_eval_cvl_circleid_split.py
-------------------------------------
Open-set writer identification on CVL, CircleID-style. Compacted to match
the shape of the original threshold_eval_cvl_iam.py: one flat
run_dataset() per config, a single pooling method, a ratio sweep, one CSV.

SPLIT: the known/unseen writer split (and which known-writer pages the
backbone trained on vs. reserved) is computed deterministically -- no
RNG, no manifest file -- by the SAME rule main_new_circleid_split.py's
circleid_page_split() used to build the training set: writers[offset:
offset+N] (by sorted id) from whichever original folder was used as the
training source become "known"; each one's first K pages (by sorted page
id) are backbone_train_pages, the rest are backbone_holdout_pages. Every
writer in the OTHER original folder is a free fully-unseen writer. Keep
CIRCLEID_TRAIN_SOURCE / CIRCLEID_KNOWN_WRITER_COUNT /
CIRCLEID_TRAIN_PAGES_PER_WRITER / CIRCLEID_WRITER_OFFSET in sync with
whatever a given checkpoint was actually trained with, or this script's
idea of "which pages were backbone_train vs backbone_holdout" silently
stops matching that checkpoint.

Reads patches through WriterZoo directly (not glob+regex) so this script
sees the exact same files/labels the training run did.

SWEEP: eval_known's query pages are always exactly its
backbone_holdout_pages -- fixed, never resampled, since the backbone was
specifically trained to discriminate those writers' identities via
metric learning, and a "known" writer can't be relabeled synthetic-
unknown without that being meaningless. Only eval_unseen writers (never
touched by backbone training) can validly be marked unknown, so the
sweep varies HOW MANY of them are included per target true_unknown_rate,
via n_unseen_writers_for_target()'s closed form -- solved against the
FIXED known query page count and this split's real per-writer unseen
page counts, then per-repeat randomly sampled from eval_unseen.

PCA is fit fresh on exactly backbone_train_pages (never a stale pca.pkl).
Pooling is SumPooling only -- GMP was tried and clearly underperformed,
so it's dropped entirely here rather than kept around as dead machinery.

CROSS-DATASET UNSEEN WRITERS (IAM): since the backbone here is trained
only on CVL, IAM writers are a genuinely out-of-domain unknown -- unlike
the "bonus_unseen" writers pulled from CVL's other split, they never
appear in known_writers, backbone_train_pages, or backbone_holdout_pages,
and (per IAM_UNSEEN_ONLY_IN_EVAL) never enter cal_unseen either, so tau
calibration stays purely in-domain. They're folded in exclusively as
extra eval_unseen writers, so the sweep can also report how a CVL-only
backbone handles a writer distribution it never saw at all. IAM writer
ids are offset (IAM_WRITER_ID_OFFSET) before merging so they can never
collide with CVL's own small integer writer ids.
"""

import pickle, os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torchvision import transforms
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.metrics import balanced_accuracy_score, f1_score, precision_score, recall_score

from backbone import resnets
from backbone.model import Model
from dataloading.writer_zoo import WriterZoo
from page_encodings import SumPooling

# ------------------------------------------------------------------ #
# CONFIG
# ------------------------------------------------------------------ #
RANDOM_SEED = 42
LOAD_CHECKPOINT = True   # False = randomly initialized backbone (sanity-testing the pipeline only)

CIRCLEID_TRAIN_SOURCE           = "test_folder"  # "test_folder" -> cfg["test_zoo"] (option 2: N of 283)
                                                   # "train_folder" -> cfg["train_zoo"] (option 1: all 27)
CIRCLEID_KNOWN_WRITER_COUNT     = 100    # None = all remaining writers from writer_offset onward
CIRCLEID_TRAIN_PAGES_PER_WRITER = 3      # must match circleid_train_pages_per_writer
CIRCLEID_WRITER_OFFSET          = 100      # must match circleid_writer_offset for the checkpoint in use
CIRCLEID_CAL_KNOWN_COUNT        = 20     # of known writers, reserved for tau calibration only
CIRCLEID_CAL_UNSEEN_COUNT       = 30     # of unseen writers, reserved for tau calibration only

# --- Cross-dataset unseen writers (IAM) --------------------------------
INCLUDE_IAM_AS_UNSEEN   = True          # fold IAM writers into eval_unseen as free OOD unknowns
IAM_SPLITS_FOR_UNSEEN   = ["train", "test"]  # any non-empty subset of {"train", "test"}
IAM_WRITER_ID_OFFSET    = 100_000       # keeps IAM writer ids from ever colliding with CVL's
IAM_UNSEEN_ONLY_IN_EVAL = True          # True -> IAM writers never enter cal_unseen (tau calibration
                                         # stays in-domain); they only appear in eval_unseen for the sweep.

TARGET_UNKNOWN_RATIOS = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

FIT_PCA_FROM_TRAINING = True
PCA_N_COMPONENTS = 64
PCA_WHITEN = True

DATASET_CONFIGS = [
    {
        "name": "cvl",
        "experiment_dir": (
            "/data/rrossner/outlierDetection/icdar23/experiments/"
            "cvl_circle_id-2026-08-10-16-21"
        ),
        "train_zoo": {"dataset": "cvl", "set": "train"},
        "test_zoo":  {"dataset": "cvl", "set": "test"},
        "num_clusters": 32,
        "model_dim": 64,
        "batch_size": 256,
        "n_repeats": 10,
        "out_csv": "threshold_results_cvl_circleid_split.csv",
    },
]


# ------------------------------------------------------------------ #
# GENERIC HELPERS
# ------------------------------------------------------------------ #
def load_model(model_path, num_clusters, model_dim, load_checkpoint=True):
    backbone = resnets.resnet56()
    model    = Model(backbone, dim=model_dim, num_clusters=num_clusters)
    if load_checkpoint:
        ckpt = torch.load(model_path, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded model : {model_path}")
    else:
        print(f"LOAD_CHECKPOINT=False -- using a randomly initialized backbone.")
    model.eval().cuda()
    return model


def extract_patch_features(model, zoo_spec, batch_size):
    """Patches -> resnet backbone -> embedding, via WriterZoo directly
    (same path main_new_circleid_split.py's training run used)."""
    ds = (WriterZoo.get(**zoo_spec)
                    .TransformImages(transform=transforms.ToTensor())
                    .SelectLabels(label_names=['writer', 'page']))
    print(f"  {len(ds)} patches (dataset={zoo_spec['dataset']}, set={zoo_spec['set']})")
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, num_workers=8, pin_memory=True)
    all_feats, all_writers, all_pages = [], [], []
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="  extracting"):
            all_feats.append(F.normalize(model(imgs.cuda()), dim=1).cpu().numpy())
            w, p = (labels[1], labels[2]) if len(labels) == 3 else (labels[0], labels[1])
            all_writers.append(w)
            all_pages.append(p)
    return np.concatenate(all_feats), np.concatenate(all_writers), np.concatenate(all_pages)


def extract_iam_unseen_features(model, splits, writer_id_offset, batch_size):
    """Extract IAM patches to use as a free, fully out-of-domain unseen-
    writer pool. Writer ids get `writer_id_offset` added so they can
    never collide with CVL's own (small integer) writer ids once
    concatenated into the same desc_writers array -- IAM never
    participates in known_writers / backbone_train_pages /
    backbone_holdout_pages, it's purely extra eval_unseen material."""
    feats_list, writers_list, pages_list = [], [], []
    for split in splits:
        f, w, p = extract_patch_features(model, {"dataset": "iam", "set": split}, batch_size)
        feats_list.append(f)
        writers_list.append(w + writer_id_offset)
        pages_list.append(p)
    return (np.concatenate(feats_list), np.concatenate(writers_list), np.concatenate(pages_list))


def pool_pages_raw(patch_feats, writers, pages, pooling):
    """Patch embeddings -> one page descriptor via `pooling`, grouped by
    a single sort + boundary-detection pass (O(n log n)) -- the same
    technique main_new.py's encode_per_class() uses. A naive scan-per-
    unique-page approach is O(n_pages * n) and becomes minutes-to-hours
    slow once n is in the millions. Returns PRE-PCA descriptors (pooling
    already applies its own internal powernorm/L2 norm) plus writer AND
    page id per row."""
    n = len(writers)
    sort_idx = np.lexsort((pages, writers))
    w_sorted, p_sorted, f_sorted = writers[sort_idx], pages[sort_idx], patch_feats[sort_idx]
    keys = w_sorted.astype(np.int64) * 100_000 + p_sorted.astype(np.int64)
    boundaries = np.concatenate([[0], np.where(np.diff(keys))[0] + 1, [n]])

    page_descs, page_writers, page_pages = [], [], []
    for i in tqdm(range(len(boundaries) - 1), desc="  pooling pages"):
        s, e = boundaries[i], boundaries[i + 1]
        page_descs.append(pooling.encode([f_sorted[s:e]]))
        page_writers.append(int(w_sorted[s]))
        page_pages.append(int(p_sorted[s]))
    return np.concatenate(page_descs), np.array(page_writers), np.array(page_pages)


def fit_pca_on_training(raw_descs_train, n_components, whiten):
    n_components = min(n_components, raw_descs_train.shape[0], raw_descs_train.shape[1])
    return PCA(n_components=n_components, whiten=whiten, random_state=RANDOM_SEED).fit(raw_descs_train)


def apply_pca(raw_descs, pca):
    return normalize(pca.transform(raw_descs), axis=1)


def build_prototypes(descs, writers):
    protos = {}
    for w in np.unique(writers):
        v = descs[writers == w].mean(axis=0)
        n = np.linalg.norm(v)
        protos[w] = v / n if n > 0 else v
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
    y_true = np.concatenate([np.ones(len(known_scores), dtype=np.int64),
                              np.zeros(len(unknown_scores), dtype=np.int64)])
    best_tau, best_bal = taus[0], -1.0
    for tau in taus:
        bal = balanced_accuracy_score(y_true, (pooled >= tau).astype(np.int64))
        if bal > best_bal:
            best_bal, best_tau = bal, float(tau)
    return best_tau, round(best_bal, 4)


def compute_metrics(query_writers, final_pred, pred_w, unknown_set):
    is_truly_unk = np.isin(query_writers, list(unknown_set)).astype(np.int64)
    is_pred_unk  = (final_pred == -1).astype(np.int64)
    known_mask   = is_truly_unk == 0
    writer_acc   = (float(np.mean(pred_w[known_mask] == query_writers[known_mask]))
                    if known_mask.any() else 0.0)
    return {
        "bal_acc":             round(balanced_accuracy_score(is_truly_unk, is_pred_unk), 4),
        "f1_unknown":          round(f1_score(is_truly_unk, is_pred_unk, pos_label=1, zero_division=0), 4),
        "precision_unknown":   round(precision_score(is_truly_unk, is_pred_unk, pos_label=1, zero_division=0), 4),
        "recall_unknown":      round(recall_score(is_truly_unk, is_pred_unk, pos_label=1, zero_division=0), 4),
        "writer_id_acc_known": round(writer_acc, 4),
        "pred_unknown_rate":   round(float(np.mean(is_pred_unk)), 4),
        "true_unknown_rate":   round(float(np.mean(is_truly_unk)), 4),
    }


def page_key(w, p):
    return int(w) * 100_000 + int(p)


def keys_from_pairs(pairs):
    return {page_key(w, p) for w, p in pairs}


# ------------------------------------------------------------------ #
# CIRCLEID SPLIT (deterministic, no RNG, no file)
# ------------------------------------------------------------------ #
def pages_by_writer_from_arrays(writers, pages):
    out = {}
    for w, p in zip(writers.tolist(), pages.tolist()):
        out.setdefault(int(w), set()).add(int(p))
    return {w: sorted(ps) for w, ps in out.items()}


def compute_circleid_groups(pool_pages_by_writer, bonus_unseen_writers,
                             known_writer_count, train_pages_per_writer,
                             cal_known_count, cal_unseen_count, writer_offset=0,
                             extra_eval_unseen_writers=None):
    """extra_eval_unseen_writers (e.g. offset IAM writer ids) are unioned
    into eval_unseen AFTER the cal/eval slice is taken, so they're
    guaranteed to never land in cal_unseen -- explicit, not dependent on
    them sorting after cal_unseen_count's cutoff."""
    pool_writers = sorted(pool_pages_by_writer.keys())
    known_writers = (pool_writers[writer_offset:] if known_writer_count is None
                      else pool_writers[writer_offset:writer_offset + known_writer_count])
    known_set = set(known_writers)
    unseen_writers = sorted({w for w in pool_writers if w not in known_set} | set(bonus_unseen_writers))

    train_pages, holdout_pages = set(), set()
    for w in known_writers:
        pages = pool_pages_by_writer[w]
        train_pages.update((w, p) for p in pages[:train_pages_per_writer])
        holdout_pages.update((w, p) for p in pages[train_pages_per_writer:])

    cal_unseen  = set(unseen_writers[:cal_unseen_count])
    eval_unseen = set(unseen_writers[cal_unseen_count:])
    extra_eval_unseen_writers = set(extra_eval_unseen_writers or ())
    eval_unseen |= extra_eval_unseen_writers

    return {
        "known_writers":  set(known_writers),
        "unseen_writers": set(unseen_writers) | extra_eval_unseen_writers,
        "cal_known":  set(known_writers[:cal_known_count]),
        "eval_known": set(known_writers[cal_known_count:]),
        "cal_unseen":  cal_unseen,
        "eval_unseen": eval_unseen,
        "backbone_train_pages": train_pages,
        "backbone_holdout_pages": holdout_pages,
    }


def n_unseen_writers_for_target(target, known_query_page_count, unseen_pages_per_writer):
    """How many eval_unseen writers (each contributing ALL their pages)
    to combine with the FIXED known_query_page_count known pages so the
    query set's true_unknown_rate lands on `target`. Solved against this
    split's real average unseen-writer page count, then per-repeat
    randomly sampled -- same spirit as the original script's
    n_unk_writers_for_target(), simplified since only the unseen side is
    swept here (the known side's query pages are fixed by the split, not
    re-sampled per target)."""
    N = len(unseen_pages_per_writer)
    if N == 0 or target <= 0:
        return 0
    if target >= 1:
        return N
    target_unseen_pages = target * known_query_page_count / (1 - target)
    avg_pages = float(np.sum(unseen_pages_per_writer)) / N
    n_unk = int(round(target_unseen_pages / avg_pages)) if avg_pages > 0 else N
    return min(max(n_unk, 0), N)


# ------------------------------------------------------------------ #
# PER-DATASET PIPELINE
# ------------------------------------------------------------------ #
def run_dataset(cfg):
    print("=" * 70)
    print(f"DATASET: {cfg['name']}")
    print("=" * 70)

    model   = load_model(os.path.join(cfg["experiment_dir"], "model.pt"),
                          cfg["num_clusters"], cfg["model_dim"], load_checkpoint=LOAD_CHECKPOINT)
    pooling = SumPooling("l2", pn_alpha=0.4)

    print("\nExtracting patch features (train set)...")
    feats1, writers1, pages1 = extract_patch_features(model, cfg["train_zoo"], cfg["batch_size"])
    print("\nExtracting patch features (test set)...")
    feats2, writers2, pages2 = extract_patch_features(model, cfg["test_zoo"], cfg["batch_size"])

    if INCLUDE_IAM_AS_UNSEEN:
        print(f"\nExtracting patch features (IAM, splits={IAM_SPLITS_FOR_UNSEEN}) as extra unseen pool...")
        feats3, writers3, pages3 = extract_iam_unseen_features(
            model, IAM_SPLITS_FOR_UNSEEN, IAM_WRITER_ID_OFFSET, cfg["batch_size"])
        iam_unseen_writer_ids = set(np.unique(writers3).tolist())
        print(f"  {len(iam_unseen_writer_ids)} IAM writers -> {len(feats3)} patches "
              f"(offset ids {min(iam_unseen_writer_ids)}-{max(iam_unseen_writer_ids)})")
    else:
        feats3   = np.empty((0, feats1.shape[1]), dtype=feats1.dtype)
        writers3 = np.empty((0,), dtype=writers1.dtype)
        pages3   = np.empty((0,), dtype=pages1.dtype)
        iam_unseen_writer_ids = set()

    pages_train = pages_by_writer_from_arrays(writers1, pages1)
    pages_test  = pages_by_writer_from_arrays(writers2, pages2)
    if CIRCLEID_TRAIN_SOURCE == "test_folder":
        pool, bonus_unseen = pages_test, set(pages_train.keys())
    elif CIRCLEID_TRAIN_SOURCE == "train_folder":
        pool, bonus_unseen = pages_train, set(pages_test.keys())
    else:
        raise ValueError(f"CIRCLEID_TRAIN_SOURCE must be 'train_folder' or 'test_folder', got {CIRCLEID_TRAIN_SOURCE!r}")

    if INCLUDE_IAM_AS_UNSEEN and not IAM_UNSEEN_ONLY_IN_EVAL:
        # IAM writers allowed into calibration too -> route them through the
        # normal cal/eval slice, same as the existing bonus_unseen writers.
        bonus_unseen = set(bonus_unseen) | iam_unseen_writer_ids
        extra_eval_unseen = None
    else:
        extra_eval_unseen = iam_unseen_writer_ids  # eval-only, guaranteed to skip cal_unseen

    groups = compute_circleid_groups(
        pool, bonus_unseen, CIRCLEID_KNOWN_WRITER_COUNT, CIRCLEID_TRAIN_PAGES_PER_WRITER,
        CIRCLEID_CAL_KNOWN_COUNT, CIRCLEID_CAL_UNSEEN_COUNT, writer_offset=CIRCLEID_WRITER_OFFSET,
        extra_eval_unseen_writers=extra_eval_unseen,
    )

    kw = groups["known_writers"]
    print(f"\nCircleID split (source={CIRCLEID_TRAIN_SOURCE}, offset={CIRCLEID_WRITER_OFFSET}): "
          f"{len(kw)} known ({len(groups['cal_known'])} cal / {len(groups['eval_known'])} eval), "
          f"{len(groups['unseen_writers'])} unseen ({len(groups['cal_unseen'])} cal / {len(groups['eval_unseen'])} eval, "
          f"of which {len(iam_unseen_writer_ids & groups['eval_unseen'])} are IAM)")
    print(f"  known writer checksum: count={len(kw)}, min={min(kw)}, max={max(kw)}, sum={sum(kw)} "
          f"-- compare against main_new_circleid_split.py's training-time log for this checkpoint's run")

    all_feats   = np.concatenate([feats1, feats2, feats3])
    all_writers = np.concatenate([writers1, writers2, writers3])
    all_pages   = np.concatenate([pages1, pages2, pages3])

    raw_descs, desc_writers, desc_pages = pool_pages_raw(all_feats, all_writers, all_pages, pooling)
    desc_keys = np.array([page_key(w, p) for w, p in zip(desc_writers, desc_pages)])

    cal_known, eval_known   = groups["cal_known"], groups["eval_known"]
    cal_unseen, eval_unseen = groups["cal_unseen"], groups["eval_unseen"]
    train_keys   = keys_from_pairs(groups["backbone_train_pages"])
    holdout_keys = keys_from_pairs(groups["backbone_holdout_pages"])

    # PCA: fit on exactly the pages the backbone trained on
    if FIT_PCA_FROM_TRAINING:
        fit_idx = np.where(np.isin(desc_keys, list(train_keys)))[0]
        pca = fit_pca_on_training(raw_descs[fit_idx], PCA_N_COMPONENTS, PCA_WHITEN)
        print(f"PCA fit on {len(fit_idx)} backbone_train_pages, n_components={pca.n_components_}")
    else:
        pca = pickle.load(open(os.path.join(cfg["experiment_dir"], "pca.pkl"), "rb"))
        print(f"Loaded old PCA (n_components={pca.n_components_}) -- may not match this split")
    descs = apply_pca(raw_descs, pca)

    # tau calibration: cal_known (fit=train pages, val=holdout pages) + cal_unseen (all pages)
    cal_known_mask = np.isin(desc_writers, list(cal_known))
    cal_fit_idx = np.where(cal_known_mask & np.isin(desc_keys, list(train_keys)))[0]
    cal_val_idx = np.where(cal_known_mask & np.isin(desc_keys, list(holdout_keys)))[0]
    cal_unk_idx = np.where(np.isin(desc_writers, list(cal_unseen)))[0]

    cal_protos = build_prototypes(descs[cal_fit_idx], desc_writers[cal_fit_idx])
    _, known_scores = score_all(descs[cal_val_idx], cal_protos)
    _, unk_scores    = score_all(descs[cal_unk_idx], cal_protos)
    tau, cal_bal = calibrate_tau(known_scores, unk_scores)
    print(f"tau={tau:.4f} (cal balanced_acc={cal_bal}, "
          f"known score mean={known_scores.mean():.3f}, unknown score mean={unk_scores.mean():.3f})")

    # eval prototypes + FIXED known-side query pages (never swept, never resampled)
    eval_known_mask  = np.isin(desc_writers, list(eval_known))
    eval_fit_idx     = np.where(eval_known_mask & np.isin(desc_keys, list(train_keys)))[0]
    known_query_idx  = np.where(eval_known_mask & np.isin(desc_keys, list(holdout_keys)))[0]
    eval_protos = build_prototypes(descs[eval_fit_idx], desc_writers[eval_fit_idx])

    eval_unseen_arr = np.array(sorted(eval_unseen))
    unseen_pages_per_writer = np.array([int(np.sum(desc_writers == w)) for w in eval_unseen_arr])

    print(f"known query pages (fixed): {len(known_query_idx)}  |  "
          f"eval_unseen writers: {len(eval_unseen_arr)}, pages: {int(unseen_pages_per_writer.sum())}")
    print(f"Sweeping targets: {TARGET_UNKNOWN_RATIOS}")
    print("-" * 70)

    rows = []
    for target in TARGET_UNKNOWN_RATIOS:
        n_unk = n_unseen_writers_for_target(target, len(known_query_idx), unseen_pages_per_writer)

        run_metrics = []
        for repeat in range(cfg["n_repeats"]):
            rng = np.random.default_rng(RANDOM_SEED + repeat * 100 + int(target * 1000))
            chosen_unseen = (set(rng.choice(eval_unseen_arr, size=n_unk, replace=False).tolist())
                              if n_unk > 0 else set())
            unseen_query_idx = np.where(np.isin(desc_writers, list(chosen_unseen)))[0]

            query_idx     = np.concatenate([known_query_idx, unseen_query_idx])
            pred_w, max_sim = score_all(descs[query_idx], eval_protos)
            final_pred = pred_w.copy().astype(np.int64)
            final_pred[max_sim < tau] = -1

            m = compute_metrics(desc_writers[query_idx], final_pred, pred_w, chosen_unseen)
            m["n_iam_unseen_writers"] = len(chosen_unseen & iam_unseen_writer_ids)
            m.update({"ratio": target, "repeat": repeat, "tau": round(tau, 4)})
            run_metrics.append(m)

        avg = {k: round(float(np.mean([r[k] for r in run_metrics])), 4)
               for k in run_metrics[0] if k not in ("ratio", "repeat")}
        avg.update({"ratio": target, "n_unseen_writers_used": n_unk})
        rows.append(avg)

        print(f"target={target:.1f} | n_unk_writers={n_unk:3d} (iam={avg['n_iam_unseen_writers']:.1f}) | "
              f"true_unk={avg['true_unknown_rate']:.3f} | bal_acc={avg['bal_acc']:.3f} | "
              f"f1_unk={avg['f1_unknown']:.3f} | recall={avg['recall_unknown']:.3f} | "
              f"precision={avg['precision_unknown']:.3f} | writer_id_acc={avg['writer_id_acc_known']:.3f}")

    out_csv = cfg["out_csv"]
    if INCLUDE_IAM_AS_UNSEEN:
        out_csv = out_csv.replace(".csv", "_iam_unseen.csv")
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")
    print(df[["ratio", "n_unseen_writers_used", "n_iam_unseen_writers", "true_unknown_rate", "tau",
              "bal_acc", "f1_unknown", "recall_unknown", "precision_unknown",
              "writer_id_acc_known"]].to_string(index=False))


def main():
    for cfg in DATASET_CONFIGS:
        run_dataset(cfg)


if __name__ == "__main__":
    main()