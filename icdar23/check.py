"""
check_map.py
------------
Computes leave-one-out retrieval mAP on the full test set using the
CURRENT (fixed, powernorm) pipeline -- the real apples-to-apples check
against the known 66% mAP baseline from main_finetune.py's own eval.

Caches page descriptors to disk so you don't have to re-run the ~26min
extraction every time you want to re-check something.
"""

import glob, re, pickle, os
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torchvision import transforms
from sklearn.preprocessing import normalize

from backbone import resnets
from backbone.model import Model
from dataloading.GenericDataset import FilepathImageDataset
from dataloading.regex import pil_loader
from page_encodings import SumPooling

EXPERIMENT_DIR = (
    "/data/rrossner/outlierDetection/icdar24/experiments/"
    "resnet56_0.5_writer-2026-07-02-10-34"
)
MODEL_PATH   = f"{EXPERIMENT_DIR}/model.pt"
PCA_PATH     = f"{EXPERIMENT_DIR}/pca.pkl"
TEST_IMG_DIR = "/data/rrossner/datasets/ICDAR17/binarized_test/patches"

MODEL_DIM    = 64
NUM_CLUSTERS = 100
BATCH_SIZE   = 256
CACHE_DIR    = "map_check_cache"

_SEP_RE = re.compile(r'^(\d+)([_-])')

def parse_filename(path):
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


def list_pages(img_dir):
    srcs = sorted(glob.glob(f"{img_dir}/**/*.png", recursive=True))
    pages = defaultdict(list)
    for s in srcs:
        w, p = parse_filename(s)
        pages[(w, p)].append(s)
    return pages


def encode_one_page(model, pooling, pca, filepaths):
    ds     = FilepathImageDataset(filepaths, pil_loader, transforms.ToTensor())
    loader = torch.utils.data.DataLoader(ds, batch_size=BATCH_SIZE, num_workers=0, pin_memory=True)
    feats = []
    with torch.no_grad():
        for imgs in loader:
            feats.append(F.normalize(model(imgs.cuda()), dim=1).cpu().numpy())
    feats = np.concatenate(feats)
    desc  = pooling.encode([feats])
    desc  = normalize(pca.transform(desc), axis=1)[0]
    return desc


def get_test_descriptors():
    os.makedirs(CACHE_DIR, exist_ok=True)
    descs_path   = os.path.join(CACHE_DIR, "test_descs.npy")
    writers_path = os.path.join(CACHE_DIR, "test_writers.npy")

    if os.path.exists(descs_path) and os.path.exists(writers_path):
        print("Loading cached descriptors...")
        return np.load(descs_path), np.load(writers_path)

    backbone = resnets.resnet56()
    model    = Model(backbone, dim=MODEL_DIM, num_clusters=NUM_CLUSTERS)
    ckpt     = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval().cuda()

    pca     = pickle.load(open(PCA_PATH, "rb"))
    pooling = SumPooling("powernorm", pn_alpha=0.4)

    pages = list_pages(TEST_IMG_DIR)
    descs, writers = [], []
    for (w, p), filepaths in tqdm(pages.items(), desc="Encoding all test pages"):
        descs.append(encode_one_page(model, pooling, pca, filepaths))
        writers.append(w)

    descs   = np.stack(descs).astype(np.float32)
    writers = np.array(writers, dtype=np.int64)
    np.save(descs_path, descs)
    np.save(writers_path, writers)
    return descs, writers


def compute_map(descs, writers):
    """Standard leave-one-out mAP: for each query page, rank all OTHER
    pages by cosine similarity, treat same-writer pages as relevant."""
    descs = descs / (np.linalg.norm(descs, axis=1, keepdims=True) + 1e-12)
    sims  = descs @ descs.T
    n = len(writers)
    np.fill_diagonal(sims, -np.inf)   # exclude self-match

    aps = []
    for i in range(n):
        order    = np.argsort(-sims[i])
        relevant = (writers[order] == writers[i])
        n_rel    = relevant.sum()
        if n_rel == 0:
            continue   # writer has only 1 page in the set -- no positives possible
        hits = np.cumsum(relevant)
        precision_at_k = hits / (np.arange(n) + 1)
        ap = (precision_at_k * relevant).sum() / n_rel
        aps.append(ap)

    return float(np.mean(aps)), len(aps), n


if __name__ == "__main__":
    descs, writers = get_test_descriptors()
    print(f"\nLoaded {len(descs)} page descriptors, {len(np.unique(writers))} writers")

    mAP, n_queries, n_total = compute_map(descs, writers)
    print(f"\nLeave-one-out mAP: {mAP:.4f}  ({n_queries}/{n_total} pages used as valid queries)")
    print("Compare against known baseline: 66% mAP from main_finetune.py's own PageEvaluator")