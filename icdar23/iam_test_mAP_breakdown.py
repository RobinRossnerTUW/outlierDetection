"""
iam_test_mAP_breakdown.py
--------------------------
1. Sanity check: confirms no writer ID appears in both iam_train_patches and
   iam_test_patches (direct check against the actual folder names on disk,
   independent of whatever the split script intended).
2. Computes mAP at the per-page-query level and breaks it down by how many
   test pages that writer has -- writers with exactly 2 pages can only ever
   score AP=0.0 or AP=1.0 (no partial credit), so it's worth knowing how
   much of the aggregate number comes from those vs. writers with more pages.

EDIT THE CONFIG BLOCK BELOW before running.
"""

import os, glob, re, pickle
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
from torchvision import transforms
from sklearn.preprocessing import normalize

from backbone import resnets
from backbone.model import Model
from dataloading.GenericDataset import FilepathImageDataset
from dataloading.regex import pil_loader
from page_encodings import SumPooling

# ------------------------------------------------------------------ #
# CONFIG -- point these at your actual experiment / data paths
# ------------------------------------------------------------------ #
EXPERIMENT_DIR = "/data/rrossner/outlierDetection/icdar23/experiments/iam_c32_triplet_m03_s8_300k-2026-06-29-14-24-40"
MODEL_PATH = os.path.join(EXPERIMENT_DIR, "model.pt")
PCA_PATH   = os.path.join(EXPERIMENT_DIR, "pca.pkl")
TRAIN_DIR  = "/data/rrossner/datasets/IAM/iam_train_patches"
TEST_DIR   = "/data/rrossner/datasets/IAM/iam_test_patches"

NUM_CLUSTERS = 32
MODEL_DIM    = 64
BATCH_SIZE   = 256
OUT_CSV      = "iam_test_writer_ap.csv"

# matches the writer_pageidx_patchidx.png naming from the fixed extraction script
WRITER_RE = re.compile(r'^(\d+)_')
PAGE_RE   = re.compile(r'^\d+_(\d+)_')
# ------------------------------------------------------------------ #


def parse_filename(path):
    stem = Path(path).stem
    return int(WRITER_RE.match(stem).group(1)), int(PAGE_RE.match(stem).group(1))


def load_model():
    backbone = resnets.resnet56()
    model = Model(backbone, dim=MODEL_DIM, num_clusters=NUM_CLUSTERS)
    ckpt = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval().cuda()
    print(f"Loaded model: {MODEL_PATH}")
    return model


def extract_patch_features(model, img_dir):
    srcs = sorted(glob.glob(f"{img_dir}/**/*.png", recursive=True))
    print(f"  {len(srcs)} patches in {img_dir}")
    ds = FilepathImageDataset(srcs, pil_loader, transforms.ToTensor())
    loader = torch.utils.data.DataLoader(ds, batch_size=BATCH_SIZE, num_workers=8, pin_memory=True)
    feats = []
    with torch.no_grad():
        for imgs in tqdm(loader, desc="  extracting"):
            feats.append(F.normalize(model(imgs.cuda()), dim=1).cpu().numpy())
    feats = np.concatenate(feats)
    writers = np.array([parse_filename(s)[0] for s in srcs])
    pages   = np.array([parse_filename(s)[1] for s in srcs])
    return feats, writers, pages


def build_page_descriptors(patch_feats, writers, pages, pca, pooling):
    label_np = np.stack([writers, pages], axis=1)
    unique = list({(w, p) for w, p in label_np})
    descs, ws = [], []
    for w, p in tqdm(unique, desc="  pooling pages"):
        idx = np.where((label_np[:, 0] == w) & (label_np[:, 1] == p))[0]
        descs.append(pooling.encode([patch_feats[idx]]))
        ws.append(w)
    descs = normalize(pca.transform(np.concatenate(descs)), axis=1)
    return descs, np.array(ws)


def average_precision(sims_row, labels, query_writer, query_idx):
    order = np.argsort(-sims_row)
    order = order[order != query_idx]          # drop self-match
    rel = (labels[order] == query_writer).astype(np.int64)
    if rel.sum() == 0:
        return 0.0
    cum_rel = np.cumsum(rel)
    precision_at_k = cum_rel / (np.arange(len(rel)) + 1)
    return float((precision_at_k * rel).sum() / rel.sum())


def main():
    import pandas as pd

    # ---- writer-disjointness check, directly from folder names on disk ----
    train_writers = {os.path.basename(p).split('_')[0]
                      for p in glob.glob(f"{TRAIN_DIR}/*") if os.path.isdir(p)}
    test_writers = {os.path.basename(p).split('_')[0]
                     for p in glob.glob(f"{TEST_DIR}/*") if os.path.isdir(p)}
    overlap = train_writers & test_writers

    print(f"Train writers: {len(train_writers)}   Test writers: {len(test_writers)}")
    print(f"Overlap between train/test writer IDs: {len(overlap)}")
    if overlap:
        sample = sorted(overlap)[:20]
        print(f"  !! LEAKAGE DETECTED -- these writer IDs appear in BOTH train and test: {sample}"
              f"{' ...' if len(overlap) > 20 else ''}")
        print("  Stop here and re-check your split before trusting any mAP number.")
    else:
        print("  OK -- no writer ID overlap between train and test patch folders.")
    print()

    # ---- encode test pages ----
    model = load_model()
    pca = pickle.load(open(PCA_PATH, "rb"))
    pooling = SumPooling("l2", pn_alpha=0.4)
    print(f"Loaded PCA (n_components={pca.n_components_}) -- confirm this matches "
          f"the single pca_dim value you intended, not a swept-and-selected one.\n")

    feats, writers, pages = extract_patch_features(model, TEST_DIR)
    descs, desc_writers = build_page_descriptors(feats, writers, pages, pca, pooling)

    sims = descs @ descs.T

    aps = [average_precision(sims[i], desc_writers, desc_writers[i], i)
           for i in range(len(descs))]
    aps = np.array(aps)

    df = pd.DataFrame({"writer": desc_writers, "ap": aps})
    page_counts = df.groupby("writer").size().rename("n_test_pages")
    df = df.join(page_counts, on="writer")
    df.to_csv(OUT_CSV, index=False)
    print(f"Saved per-page AP to {OUT_CSV}\n")

    print(f"Overall mAP (page-level): {df['ap'].mean():.4f}")
    print(f"Total test pages: {len(df)}   Total test writers: {df['writer'].nunique()}\n")

    print("--- mAP broken down by # test pages for that writer ---")
    for n, group in df.groupby("n_test_pages"):
        n_writers = group['writer'].nunique()
        print(f"  {n:>2} pages | {n_writers:>3} writers | {len(group):>4} queries | "
              f"mean AP={group['ap'].mean():.4f} | "
              f"AP==1.0: {(group['ap'] == 1.0).mean()*100:5.1f}% | "
              f"AP==0.0: {(group['ap'] == 0.0).mean()*100:5.1f}%")

    n2 = int((page_counts == 2).sum())
    total_w = page_counts.shape[0]
    print(f"\nWriters with exactly 2 test pages: {n2}/{total_w} ({100*n2/total_w:.1f}%) "
          f"-- AP for these is binary (0.0 or 1.0), no partial credit.")


if __name__ == "__main__":
    main()
