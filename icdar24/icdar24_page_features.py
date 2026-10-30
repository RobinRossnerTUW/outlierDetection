"""
icdar24_page_features.py
-------------------------
Shared helpers for open-set writer-identification scripts that plug into the
icdar24-master ("SAGHOG") codebase.

WHY THIS FILE EXISTS
====================
The original threshold_eval_cvl.py / threshold_eval_iam.py scripts extracted
features for *every single patch in the whole dataset* into one big numpy
array before ever pooling per page:

    all_feats.append(...)                # every patch, every page
    feats = np.concatenate(all_feats)     # -> one giant array in RAM

For a small dataset (CVL/IAM, patches already cropped to disk) that's
annoying but survivable. For ICDAR17/HistoricalWI -- run through a
NetRVLAD head, where every patch is embedded into a `num_clusters * dim`
vector (e.g. 100*64 = 6400 floats) -- keeping all patches of the whole
corpus in memory at once is exactly what blows past 200GB.

The fix is simple: a page descriptor only needs the *sum-pooled* patch
features of that one page. So we never need more than one page's worth of
patch features in memory. This module extracts and immediately pools
descriptors **page by page**, discarding raw patch features as soon as a
page has been pooled. Only the pooled page descriptors (tiny) accumulate.

THREE EXTRACTION MODES
=======================
1. `extract_pages_from_csv(...)`
   The "native" icdar24 way. Reuses icdar24's own
   `dataloading.dataset.DocumentDataset` + `.EvalSIFTSampler(...)` dataset
   wrapper, which performs on-the-fly SIFT keypoint detection + patch
   cropping directly on the full (binarized) page image -- exactly what
   `main_finetune.py` / `evaluators.validators.PageEvaluator` do at test
   time for the icdar2017 / HistoricalWI dataset. Use this if you have (or
   can build, see datasets_example/create_dataset_csvs.py) a CSV with an
   `imgs` column of full-page image paths + `writer`/`page` label columns.

2. `extract_pages_from_patch_dir(...)`
   For the case where patches have *already* been extracted to disk per
   page (e.g. via icdar24's own `utils/extract_patches_only.py`, which
   writes patches into one sub-directory per page: `root/<page_stem>/
   <page_stem>_<idx>.png`), or any flat directory of patch images whose
   filenames encode writer/page via regex (same convention as the old
   threshold_eval_cvl.py / threshold_eval_iam.py scripts). This mode never
   builds a global patch index in memory -- it groups filepaths (cheap,
   just strings) by page and only loads pixel data for one page at a time.

3. `extract_pages_from_image_list(...)`
   For an arbitrary in-memory list of full-page image paths + per-image
   writer labels (e.g. a competition-style CSV with image_id/writer_id
   columns, including a "-1" unknown marker for open-set training rows).
   Internally reuses the same on-the-fly SIFT extraction as mode 1
   without needing a CSV file on disk. Used by predict_labels_svm_icdar17.py.

All three extraction functions return the same, small, in-memory result:
    descs   : np.ndarray [num_pages, num_clusters*dim]  (float32)
    writers : np.ndarray [num_pages]
    pages   : np.ndarray [num_pages]

MODEL
=====
`load_resnet56_netrvlad(...)` builds the supervised resnet56 backbone +
NetRVLAD head exactly as `main_finetune.py` does for
`model_zoo.experiment == 'resnet56'` with `netvlad.random == True`
(NetRVLAD, i.e. no alpha-init / no intra-normalization -- see
`finetune/netvlad.py`), and loads a checkpoint saved by that training code
(`utils.save_model`, i.e. a dict with a 'model_state_dict' key).

KNOWN COMPATIBILITY ISSUE: doxapy version
==========================================
Modes 1 and 3 (on-the-fly SIFT extraction) go through icdar24's own
`dataloading/wrapper.py`, which calls
`doxapy.Binarization.update_to_binary(algorithm, image)`. The current
PyPI release of `doxapy` (0.9.8, tested here) no longer exposes
`update_to_binary` as a `Binarization` classmethod -- it's now a
module-level function `doxapy.update_to_binary(...)`. If you hit
`AttributeError: type object 'doxapy.doxapy.Binarization' has no
attribute 'update_to_binary'`, either pin an older `doxapy` release that
still has the old API, or add this one-time shim before importing this
module:

    import doxapy
    doxapy.Binarization.update_to_binary = staticmethod(
        lambda algorithm, image: doxapy.update_to_binary(algorithm, image)
    )

Mode 2 (pre-extracted patch PNGs) does not use doxapy at all and is
unaffected.

This is an icdar24-master / doxapy version-skew issue, independent of
everything else in this file.

at) the icdar24-master checkout, since it imports that repo's internal
modules (`models`, `finetune`, `dataloading`, `page_encodings`, `utils`)
exactly like `main_finetune.py` does.
"""

import logging
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

# ------------------------------------------------------------------ #
# Point this at your icdar24-master checkout so the imports below
# resolve (models/, finetune/, dataloading/, page_encodings.py, utils/).
# Not needed if you run scripts from inside that repo already.
# ------------------------------------------------------------------ #
ICDAR24_ROOT = os.environ.get("ICDAR24_ROOT", "")
if ICDAR24_ROOT and ICDAR24_ROOT not in sys.path:
    sys.path.insert(0, ICDAR24_ROOT)

from models.resnets import resnet56              # noqa: E402
from finetune.netvlad import Wrapper             # noqa: E402
from page_encodings import SumPooling            # noqa: E402
from dataloading.wrapper import WrapableDataset  # noqa: E402


# ------------------------------------------------------------------ #
# MODEL
# ------------------------------------------------------------------ #
def load_resnet56_netrvlad(
    checkpoint_path,
    num_clusters=100,
    random_vlad=True,
    in_dim=-1,
    device="cuda",
    strict=False,
):
    """
    Build supervised resnet56 + NetRVLAD exactly as icdar24's
    main_finetune.py does for model_zoo.experiment == 'resnet56', then
    load a finetuned checkpoint (dict with 'model_state_dict').

    num_clusters / random_vlad should match the `netvlad:` block of the
    config the model was trained with, e.g. config/finetune_maskfeat_
    icdar2017_resnet.yml has `netvlad: {num_clusters: 100, random: True}`.
    random_vlad=True == NetRVLAD (no alpha-init, no intra-normalization).
    """
    backbone = resnet56()
    backbone.num_features = backbone.embed_dim = 64  # matches main_finetune.py

    wrapper_args = {
        "model_options": {"in_dim": in_dim},
        "netvlad": {"num_clusters": num_clusters, "random": random_vlad},
        "netvlad_pooling": False,  # resnet56 patch features are a single vector, not a feature map
    }
    model = Wrapper(backbone, wrapper_args)

    logging.info(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state_dict, strict=strict)
    if missing:
        logging.warning(f"Missing keys when loading checkpoint: {missing}")
    if unexpected:
        logging.warning(f"Unexpected keys when loading checkpoint: {unexpected}")

    model.eval()
    model.to(device)
    return model


def _make_pooling(pn_alpha=0.4):
    # Matches page_encodings.get_encoder() used throughout icdar24's own
    # training/eval code: sum-pool patch descriptors, then power-norm.
    return SumPooling(["powernorm"], pn_alpha=pn_alpha)


@torch.no_grad()
def _embed_and_pool(patches, model, pooling, device, chunk_size):
    """
    patches: CPU tensor [n_patches, 3, H, W] for exactly ONE page.
    Runs the model in chunks (bounded GPU/CPU memory), concatenates only
    this page's patch embeddings, pools immediately, and returns a single
    1D page descriptor. Nothing from this page is kept afterwards.
    """
    n = patches.shape[0]
    chunk_feats = []
    for start in range(0, n, chunk_size):
        chunk = patches[start:start + chunk_size].to(device, non_blocking=True)
        emb = model(chunk)  # Wrapper.forward already L2-normalizes
        chunk_feats.append(emb.detach().cpu().numpy())
        del chunk, emb

    page_patch_feats = np.concatenate(chunk_feats, axis=0)
    del chunk_feats

    page_desc = pooling.encode([page_patch_feats])[0]
    del page_patch_feats
    return page_desc


# ------------------------------------------------------------------ #
# MODE 1 -- native icdar24 pipeline: full-page images, on-the-fly SIFT
# patch extraction via dataloading's EvalSIFTSampler (needs 'doxapy'
# for binarization + opencv for SIFT, same deps icdar24 already needs).
# ------------------------------------------------------------------ #
def extract_pages_from_csv(
    csv_path,
    model,
    device="cuda",
    patch_size=32,
    num_samples=-1,
    pn_alpha=0.4,
    chunk_size=5000,
    num_workers=8,
    cache_path=None,
):
    """
    csv_path : CSV built the same way as icdar24's
               datasets_example/create_dataset_csvs.py (columns: 'imgs'
               + int-encoded 'writer'/'page'), e.g. the icdar2017-test-
               binarized.csv / HistoricalWI equivalent used by
               config/finetune_maskfeat_icdar2017_resnet.yml.

    Streams ONE PAGE AT A TIME: dataset.__getitem__ already returns *all*
    SIFT patches for a single page (that's how icdar24's own
    EvalSIFTSampler + PageEvaluator work), so a DataLoader with
    batch_size=1 naturally gives per-page batches. We chunk further
    within a page only to bound the single forward pass size.
    """
    if cache_path and Path(cache_path).exists():
        logging.info(f"Loading cached page descriptors from {cache_path}")
        data = np.load(cache_path)
        return data["descs"], data["writers"], data["pages"]

    from dataloading.dataset import DocumentDataset
    from utils.collate import collate_fn_evaluate

    d = DocumentDataset(csv_path)
    ds = d.SelectLabels(label_names=["writer", "page"]).EvalSIFTSampler(
        patch_size=patch_size, num_samples=num_samples
    )
    loader = torch.utils.data.DataLoader(
        ds, batch_size=1, shuffle=False,
        collate_fn=collate_fn_evaluate,
        num_workers=num_workers, pin_memory=True,
    )

    pooling = _make_pooling(pn_alpha)
    model.eval()

    descs, writers, pages = [], [], []
    n_skipped = 0
    for patches, label in tqdm(loader, desc=f"Extracting pages ({csv_path})"):
        if patches.numel() == 0:
            n_skipped += 1
            continue
        w, p = int(label[0][0]), int(label[0][1])
        try:
            page_desc = _embed_and_pool(patches, model, pooling, device, chunk_size)
        except Exception as e:
            logging.warning(f"Skipping page (writer={w}, page={p}) due to error: {e}")
            n_skipped += 1
            continue
        descs.append(page_desc)
        writers.append(w)
        pages.append(p)

    if n_skipped:
        logging.warning(f"Skipped {n_skipped} pages (no usable patches / errors)")

    descs = np.stack(descs).astype(np.float32)
    writers = np.array(writers)
    pages = np.array(pages)

    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache_path, descs=descs, writers=writers, pages=pages)
        logging.info(f"Cached page descriptors to {cache_path}")

    return descs, writers, pages


# ------------------------------------------------------------------ #
# MODE 2 -- pre-extracted patch PNGs on disk, grouped by page via
# filename regex (same convention as the old threshold_eval_cvl.py /
# threshold_eval_iam.py scripts, or the per-page sub-directories written
# by icdar24's own utils/extract_patches_only.py).
# ------------------------------------------------------------------ #
def _group_patches_by_page(root_dir, writer_re, page_re, recursive=True):
    """
    Cheap step: only glob + regex-match filenames (strings + ints), never
    touches pixel data. Returns dict {(writer, page): [filepaths, ...]}.
    """
    import glob

    pattern = f"{root_dir}/**/*.png" if recursive else f"{root_dir}/*.png"
    paths = sorted(glob.glob(pattern, recursive=recursive))
    if not paths:
        raise RuntimeError(f"No .png patches found under {root_dir}")

    groups = {}
    for p in paths:
        stem = Path(p).stem
        wm = writer_re.match(stem)
        pm = page_re.match(stem)
        if wm is None or pm is None:
            continue
        w, pg = int(wm.group(1)), int(pm.group(1))
        groups.setdefault((w, pg), []).append(p)

    if not groups:
        raise RuntimeError(
            f"Found {len(paths)} patch files under {root_dir} but none matched "
            f"the writer/page regex -- check WRITER_RE/PAGE_RE."
        )
    return groups


def extract_pages_from_patch_dir(
    root_dir,
    model,
    writer_re,
    page_re,
    device="cuda",
    pn_alpha=0.4,
    batch_size=256,
    chunk_size=5000,
    num_workers=4,
    cache_path=None,
):
    """
    root_dir  : directory containing pre-extracted patch PNGs (either flat,
                or one sub-directory per page as produced by icdar24's
                utils/extract_patches_only.py).
    writer_re, page_re : compiled regexes applied to Path(patch).stem,
                matched with `.match(...).group(1)` (same convention as
                the original threshold_eval_cvl.py / threshold_eval_iam.py
                WRITER_RE / PAGE_RE).

    Groups filepaths by page first (cheap - just strings), then loads and
    embeds ONE PAGE'S patches at a time, pools immediately, and discards
    them before moving to the next page. Peak memory is bounded by the
    largest single page's patch count, not the whole dataset.
    """
    if cache_path and Path(cache_path).exists():
        logging.info(f"Loading cached page descriptors from {cache_path}")
        data = np.load(cache_path)
        return data["descs"], data["writers"], data["pages"]

    from torchvision import transforms
    from dataloading.GenericDataset import FilepathImageDataset
    from dataloading.regex import pil_loader

    groups = _group_patches_by_page(root_dir, writer_re, page_re)
    logging.info(f"Found {len(groups)} pages under {root_dir}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    pooling = _make_pooling(pn_alpha)
    model.eval()

    descs, writers, pages = [], [], []
    for (w, p), filepaths in tqdm(sorted(groups.items()), desc=f"Extracting pages ({root_dir})"):
        page_ds = FilepathImageDataset(filepaths, pil_loader, transform)
        page_loader = torch.utils.data.DataLoader(
            page_ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True,
        )

        chunk_feats = []
        with torch.no_grad():
            for imgs in page_loader:
                imgs = imgs.to(device, non_blocking=True)
                emb = model(imgs)
                chunk_feats.append(emb.detach().cpu().numpy())
                del imgs, emb

        page_patch_feats = np.concatenate(chunk_feats, axis=0)
        del chunk_feats
        page_desc = pooling.encode([page_patch_feats])[0]
        del page_patch_feats

        descs.append(page_desc)
        writers.append(w)
        pages.append(p)

    descs = np.stack(descs).astype(np.float32)
    writers = np.array(writers)
    pages = np.array(pages)

    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache_path, descs=descs, writers=writers, pages=pages)
        logging.info(f"Cached page descriptors to {cache_path}")

    return descs, writers, pages


def _group_patches_by_key(root_dir, key_re, recursive=True):
    """
    Like _group_patches_by_page, but groups by a single arbitrary string
    key (e.g. an `image_id`) captured by one regex group, instead of a
    (writer, page) pair. Used when the true writer label (including a
    "-1" unknown marker) lives in an external CSV rather than being
    parseable straight out of the filename.
    """
    import glob

    pattern = f"{root_dir}/**/*.png" if recursive else f"{root_dir}/*.png"
    paths = sorted(glob.glob(pattern, recursive=recursive))
    if not paths:
        raise RuntimeError(f"No .png patches found under {root_dir}")

    groups = {}
    for p in paths:
        stem = Path(p).stem
        km = key_re.match(stem)
        if km is None:
            continue
        key = km.group(1)
        groups.setdefault(key, []).append(p)

    if not groups:
        raise RuntimeError(
            f"Found {len(paths)} patch files under {root_dir} but none matched "
            f"key_re -- check the regex."
        )
    return groups


def extract_pages_from_patch_dir_by_key(
    root_dir,
    model,
    key_re,
    device="cuda",
    pn_alpha=0.4,
    batch_size=256,
    chunk_size=5000,
    num_workers=4,
    cache_path=None,
):
    """
    Same memory-efficient one-page(image)-at-a-time pooling as
    extract_pages_from_patch_dir, but groups patches by an arbitrary
    string key (e.g. the `image_id` embedded in your patch filenames,
    like `<image_id>_<patch_idx>.png` or `root/<image_id>/<image_id>_
    <idx>.png`) instead of a (writer, page) pair parsed from the
    filename. Use this when the real label (e.g. a "-1" open-set
    unknown marker) lives in a CSV keyed by that same id, rather than
    being derivable from the patch filename itself.

    key_re : compiled regex with ONE capture group, applied to
             Path(patch).stem via `.match(...).group(1)`.

    Returns:
        descs : [num_keys, num_clusters*dim] float32
        keys  : [num_keys] array of the matched key strings (use these
                to look up the label for each descriptor in your CSV)
    """
    if cache_path and Path(cache_path).exists():
        logging.info(f"Loading cached page descriptors from {cache_path}")
        data = np.load(cache_path, allow_pickle=True)
        return data["descs"], data["keys"]

    from torchvision import transforms
    from dataloading.GenericDataset import FilepathImageDataset
    from dataloading.regex import pil_loader

    groups = _group_patches_by_key(root_dir, key_re)
    logging.info(f"Found {len(groups)} keys (images/pages) under {root_dir}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    pooling = _make_pooling(pn_alpha)
    model.eval()

    descs, keys = [], []
    for key, filepaths in tqdm(sorted(groups.items()), desc=f"Extracting pages ({root_dir})"):
        page_ds = FilepathImageDataset(filepaths, pil_loader, transform)
        page_loader = torch.utils.data.DataLoader(
            page_ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True,
        )

        chunk_feats = []
        with torch.no_grad():
            for imgs in page_loader:
                imgs = imgs.to(device, non_blocking=True)
                emb = model(imgs)
                chunk_feats.append(emb.detach().cpu().numpy())
                del imgs, emb

        page_patch_feats = np.concatenate(chunk_feats, axis=0)
        del chunk_feats
        page_desc = pooling.encode([page_patch_feats])[0]
        del page_patch_feats

        descs.append(page_desc)
        keys.append(key)

    descs = np.stack(descs).astype(np.float32)
    keys = np.array(keys, dtype=object)

    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache_path, descs=descs, keys=keys)
        logging.info(f"Cached page descriptors to {cache_path}")

    return descs, keys


# ------------------------------------------------------------------ #
# MODE 3 -- an arbitrary in-memory list of (full-page) image paths +
# per-image writer labels, e.g. for a competition-style CSV
# (image_id, writer_id) rather than icdar24's own dataset CSV format.
# Used by predict_labels_svm_icdar17.py. Reuses the exact same on-the-fly
# SIFT patch extraction (dataloading's EvalSIFTSampler) as MODE 1, but
# builds a tiny in-memory dataset instead of round-tripping through a CSV
# file, so writer labels (including a "-1" unknown marker) are passed
# through completely untouched -- no int-remapping ambiguity.
# ------------------------------------------------------------------ #
class _InMemoryImageDataset(WrapableDataset):
    """Minimal dataset satisfying the interface icdar24's dataset wrappers
    (SelectLabels, EvalSIFTSampler, ...) expect: `.imgs`, `.labels`,
    `.get_image(i)`. Subclassing WrapableDataset directly (same as
    icdar24's own DocumentDataset) gives us its `.SelectLabels(...)` /
    `.EvalSIFTSampler(...)` chaining for free.

    `writer` labels are stored exactly as given (so a "-1" unknown marker
    survives untouched); `page` is just the original list index, used
    only to realign results back to the caller's list afterwards (some
    images may be skipped if no SIFT keypoints survive filtering).
    """

    def __init__(self, image_paths, writer_labels):
        from PIL import Image

        assert len(image_paths) == len(writer_labels)
        self.imgs = list(image_paths)
        self.labels = {
            "writer": [int(w) for w in writer_labels],
            "page": list(range(len(image_paths))),
        }
        self._loader = lambda p: Image.open(p).convert("RGB")

    def __len__(self):
        return len(self.imgs)

    def get_image(self, index):
        return self._loader(self.imgs[index])

    def get_label(self, index):
        return (self.labels["writer"][index], self.labels["page"][index])


def extract_pages_from_image_list(
    image_paths,
    writer_labels,
    model,
    device="cuda",
    patch_size=32,
    num_samples=-1,
    pn_alpha=0.4,
    chunk_size=5000,
    num_workers=8,
    cache_path=None,
):
    """
    image_paths    : list of full-page image file paths.
    writer_labels  : list/array of int writer ids, same length, one per
                      image (use -1 for "explicitly unknown" rows in an
                      open-set training CSV -- it is passed through as-is).

    Returns:
        descs        : [num_pages_kept, num_clusters*dim] float32
        writers      : [num_pages_kept] int, exactly the labels you passed in
        orig_indices : [num_pages_kept] int, index into the *input* lists
                       (image_paths / writer_labels) each row corresponds
                       to -- use this to realign back to e.g. image_id,
                       since a small number of pages can be skipped (no
                       usable SIFT keypoints / load errors).
    """
    if cache_path and Path(cache_path).exists():
        logging.info(f"Loading cached page descriptors from {cache_path}")
        data = np.load(cache_path)
        return data["descs"], data["writers"], data["orig_indices"]

    from utils.collate import collate_fn_evaluate

    base = _InMemoryImageDataset(image_paths, writer_labels)

    ds = base.SelectLabels(label_names=["writer", "page"]).EvalSIFTSampler(
        patch_size=patch_size, num_samples=num_samples
    )
    loader = torch.utils.data.DataLoader(
        ds, batch_size=1, shuffle=False,
        collate_fn=collate_fn_evaluate,
        num_workers=num_workers, pin_memory=True,
    )

    pooling = _make_pooling(pn_alpha)
    model.eval()

    descs, writers, orig_indices = [], [], []
    n_skipped = 0
    for patches, label in tqdm(loader, desc="Extracting page descriptors"):
        if patches.numel() == 0:
            n_skipped += 1
            continue
        w, row_idx = int(label[0][0]), int(label[0][1])
        try:
            page_desc = _embed_and_pool(patches, model, pooling, device, chunk_size)
        except Exception as e:
            logging.warning(f"Skipping row {row_idx} due to error: {e}")
            n_skipped += 1
            continue
        descs.append(page_desc)
        writers.append(w)
        orig_indices.append(row_idx)

    if n_skipped:
        logging.warning(f"Skipped {n_skipped} / {len(image_paths)} images (no usable patches / errors)")

    descs = np.stack(descs).astype(np.float32)
    writers = np.array(writers)
    orig_indices = np.array(orig_indices)

    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache_path, descs=descs, writers=writers, orig_indices=orig_indices)
        logging.info(f"Cached page descriptors to {cache_path}")

    return descs, writers, orig_indices


# ------------------------------------------------------------------ #
# Optional page-descriptor-level PCA+whitening, matching icdar24's own
# evaluators.validators.pca() -- applied to the small [num_pages, dim]
# descriptor matrix, never to patch-level features, so this is cheap.
# ------------------------------------------------------------------ #
def apply_pca(descs, pca_dim=512, min_dim_for_pca=500):
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import normalize

    if pca_dim == -1 or min(descs.shape) <= min_dim_for_pca:
        logging.info("Skipping PCA (disabled or too few samples/dims)")
        return descs, None

    pca_dim = min(min(descs.shape), pca_dim)
    logging.info(f"Fitting PCA on page descriptors {descs.shape} -> {pca_dim}")
    pca = PCA(n_components=pca_dim, whiten=True)
    out = normalize(pca.fit_transform(descs), axis=1)
    return out.astype(np.float32), pca