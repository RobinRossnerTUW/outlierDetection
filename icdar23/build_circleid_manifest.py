"""
build_circleid_manifest.py
----------------------------
Builds ONE shared writer/page split manifest for the CircleID-style open-
set protocol, and materializes it as a filtered, symlinked patch
directory that main_new.py can point a WriterZoo dataset entry at.

Run this ONCE, before training. Pure filesystem work -- no model, no GPU.

WHY THIS EXISTS
================
threshold_eval_cvl_circleid_split.py computes its writer-group split
(known/unseen, cal/eval) with assign_writer_groups(). That part is
actually safe to duplicate: it's a pure function of (sorted writer ids,
seed), so as long as both scripts see the same universe of writer ids,
the WRITER-level groups will match exactly wherever it's called.

What is NOT safe to duplicate is the PAGE-level split within known
writers (which pages the backbone gets to train on vs. which pages stay
reserved for PCA fitting / tau calibration / evaluation queries). That
split depends on iteration order over each writer's page list, and
Python set iteration order isn't guaranteed to match between two
different scripts even with identical seeds. If the backbone's
"training pages" and the eval script's "held-out query pages" are
computed independently and happen to disagree even slightly, some pages
the backbone directly trained on could end up scored as if they were
held out -- reintroducing the exact leakage this whole split was built
to prevent.

So: compute the split ONCE, here, with a fully deterministic (sorted,
not set-based) iteration order, and save it. main_new.py consumes it
indirectly via a filtered symlink directory (see below). The eval script
should ideally be pointed at the same manifest too -- see the note at
the bottom of this file's printed output.

WHAT THIS PRODUCES
====================
1. circleid_split_manifest.json -- the full split (writer groups + every
   known writer's backbone_train_pages / backbone_holdout_pages + every
   unseen writer's pages), for reference and for the eval script to load.
2. cvl_backbone_train_patches/   -- symlinks to every patch file whose
   (writer, page) is in a KNOWN writer's backbone_train_pages. This is
   the ONLY data the backbone should ever be trained on. No unseen
   writer, no held-out known-writer page, is symlinked here.
3. cvl_backbone_holdout_patches/ -- symlinks to every patch file whose
   (writer, page) is in a KNOWN writer's backbone_holdout_pages. Not
   required by main_new.py, but handy to have materialized for sanity
   checks (e.g. confirming the eval script's PCA/tau fit pool only ever
   touches these files for known writers).
"""

import glob, re, json, os
import numpy as np
from pathlib import Path

# ------------------------------------------------------------------ #
# CONFIG -- keep RANDOM_SEED / the four fractions identical to
# threshold_eval_cvl_circleid_split.py so writer-level groups match.
# ------------------------------------------------------------------ #
RANDOM_SEED = 42
KNOWN_WRITER_FRACTION     = 0.68
CAL_KNOWN_FRACTION        = 0.15
CAL_UNSEEN_FRACTION       = 0.30
BACKBONE_HOLDOUT_FRACTION = 0.2   # same role as VAL_FRACTION in the eval
                                    # script -- fraction of each known
                                    # writer's pages reserved from
                                    # backbone training entirely.

# (short tag, directory) -- tag only used to guarantee unique output
# filenames if the two original folders ever share a basename.
IMG_DIRS = [
    ("train", "/data/rrossner/datasets/CVL/cvl_train_patches_bin"),
    ("test",  "/data/rrossner/datasets/CVL/cvl_test_patches_bin"),
]
WRITER_RE = re.compile(r'^(\d+)-\d+-cropped_\d+')
PAGE_RE   = re.compile(r'^\d+-(\d+)-cropped_\d+')

OUT_MANIFEST    = "circleid_split_manifest.json"
OUT_TRAIN_DIR   = "./cvl_backbone_train_patches"
OUT_HOLDOUT_DIR = "./cvl_backbone_holdout_patches"


# ------------------------------------------------------------------ #
def parse_filename(path):
    stem = Path(path).stem
    return int(WRITER_RE.match(stem).group(1)), int(PAGE_RE.match(stem).group(1))


def list_patch_files():
    """(path, tag, writer, page) for every patch across all configured
    image dirs, in deterministic (sorted-glob) order."""
    records = []
    for tag, d in IMG_DIRS:
        srcs = sorted(glob.glob(f"{d}/**/*.png", recursive=True))
        print(f"  {len(srcs)} patches in {d}")
        for s in srcs:
            w, p = parse_filename(s)
            records.append((s, tag, w, p))
    return records


def assign_writer_groups(all_writer_ids, seed):
    """Identical logic to threshold_eval_cvl_circleid_split.py's function
    of the same name -- copied here (not imported) so this script has no
    dependency on the eval script's module path/environment. Keep the
    two in sync if you ever change the fractions."""
    writers = np.array(sorted(all_writer_ids))
    rng0 = np.random.default_rng(seed)
    shuffled = rng0.permutation(writers)
    n_known = int(round(len(shuffled) * KNOWN_WRITER_FRACTION))
    known_writers  = shuffled[:n_known]
    unseen_writers = shuffled[n_known:]

    rng_k = np.random.default_rng(seed + 1)
    known_shuf  = rng_k.permutation(known_writers)
    n_cal_known = max(1, int(round(len(known_shuf) * CAL_KNOWN_FRACTION)))
    cal_known   = sorted(known_shuf[:n_cal_known].tolist())
    eval_known  = sorted(known_shuf[n_cal_known:].tolist())

    rng_u = np.random.default_rng(seed + 2)
    unseen_shuf  = rng_u.permutation(unseen_writers)
    n_cal_unseen = max(1, int(round(len(unseen_shuf) * CAL_UNSEEN_FRACTION)))
    cal_unseen   = sorted(unseen_shuf[:n_cal_unseen].tolist())
    eval_unseen  = sorted(unseen_shuf[n_cal_unseen:].tolist())

    return {
        "known_writers":  sorted(known_writers.tolist()),
        "unseen_writers": sorted(unseen_writers.tolist()),
        "cal_known":  cal_known,
        "eval_known": eval_known,
        "cal_unseen": cal_unseen,
        "eval_unseen": eval_unseen,
    }


def held_out_split(sorted_items, val_fraction, rng):
    """sorted_items: a SORTED list (deterministic input order matters --
    np.random.Generator.permutation over a fixed input order is
    reproducible; permuting a set's arbitrary iteration order is not).
    Returns (fit_items, held_items), both sorted for a stable manifest.
    """
    items = np.array(sorted_items)
    if len(items) <= 1:
        return items.tolist(), []
    shuffled = rng.permutation(items)
    n_val = max(1, int(round(len(items) * val_fraction)))
    n_val = min(n_val, len(items) - 1)
    return sorted(shuffled[n_val:].tolist()), sorted(shuffled[:n_val].tolist())


def symlink_patches(records, page_set, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    n = 0
    for path, tag, w, p in records:
        if (w, p) in page_set:
            dst = os.path.join(out_dir, f"{tag}__{os.path.basename(path)}")
            if not os.path.lexists(dst):
                os.symlink(os.path.abspath(path), dst)
            n += 1
    print(f"  symlinked {n} patch files -> {out_dir}")
    return n


def main():
    print("Listing patch files...")
    records = list_patch_files()
    writers_present = sorted({w for _, _, w, _ in records})
    print(f"Found {len(records)} patches, "
          f"{len({(w, p) for _, _, w, p in records})} unique pages, "
          f"{len(writers_present)} writers")

    groups = assign_writer_groups(writers_present, RANDOM_SEED)
    print(f"known  : {len(groups['known_writers']):3d}  "
          f"(cal_known={len(groups['cal_known'])}, eval_known={len(groups['eval_known'])})")
    print(f"unseen : {len(groups['unseen_writers']):3d}  "
          f"(cal_unseen={len(groups['cal_unseen'])}, eval_unseen={len(groups['eval_unseen'])})")

    pages_by_writer = {}
    for _, _, w, p in records:
        pages_by_writer.setdefault(w, set()).add(p)
    for w in pages_by_writer:
        pages_by_writer[w] = sorted(pages_by_writer[w])

    # One FIXED split per known writer -- backbone_train_pages is the
    # ONLY thing the backbone is ever allowed to see. This holds
    # regardless of whether the writer is in cal_known or eval_known;
    # both groups' pages get this same treatment.
    rng_bb = np.random.default_rng(RANDOM_SEED + 5)
    backbone_train_pages, backbone_holdout_pages = [], []
    for w in sorted(groups["known_writers"]):
        fit, held = held_out_split(pages_by_writer[w], BACKBONE_HOLDOUT_FRACTION, rng_bb)
        backbone_train_pages.extend([w, p] for p in fit)
        backbone_holdout_pages.extend([w, p] for p in held)

    unseen_pages = [
        [w, p] for w in sorted(groups["unseen_writers"]) for p in pages_by_writer[w]
    ]

    manifest = {
        "seed": RANDOM_SEED,
        "known_writer_fraction": KNOWN_WRITER_FRACTION,
        "cal_known_fraction": CAL_KNOWN_FRACTION,
        "cal_unseen_fraction": CAL_UNSEEN_FRACTION,
        "backbone_holdout_fraction": BACKBONE_HOLDOUT_FRACTION,
        **groups,
        "backbone_train_pages": backbone_train_pages,
        "backbone_holdout_pages": backbone_holdout_pages,
        "unseen_pages": unseen_pages,
    }
    with open(OUT_MANIFEST, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nbackbone_train_pages   : {len(backbone_train_pages)}  (known writers only)")
    print(f"backbone_holdout_pages : {len(backbone_holdout_pages)}  (known writers only, reserved)")
    print(f"unseen_pages           : {len(unseen_pages)}  (never touched by backbone training)")
    print(f"Saved manifest: {OUT_MANIFEST}")

    print("\nMaterializing filtered patch directories...")
    train_page_set   = {tuple(wp) for wp in backbone_train_pages}
    holdout_page_set = {tuple(wp) for wp in backbone_holdout_pages}
    symlink_patches(records, train_page_set, OUT_TRAIN_DIR)
    symlink_patches(records, holdout_page_set, OUT_HOLDOUT_DIR)

    print(
        "\nNext steps:\n"
        f"  1. Register a new WriterZoo dataset/set entry pointing at "
        f"{os.path.abspath(OUT_TRAIN_DIR)}, reusing the same writer/page "
        f"regex as your existing 'cvl' entries (the symlinked filenames "
        f"are prefixed 'train__'/'test__' but the writer-id/page-id "
        f"portion of each original filename is untouched, so the same "
        f"regex still parses them).\n"
        f"  2. Point your training config's 'trainset' at that new entry.\n"
        f"  3. Set no_val_split: true and skip_internal_test: true in "
        f"that config (see the corresponding main_new.py changes).\n"
        f"  4. Make sure 'checkpoint' is unset/null in that config -- you "
        f"want fresh weights, not a checkpoint trained on a different split.\n"
        f"  5. After training, point threshold_eval_cvl_circleid_split.py's "
        f"experiment_dir at this run's output and set LOAD_CHECKPOINT=True."
    )


if __name__ == "__main__":
    main()
