import cv2
import numpy as np
from joblib import Parallel, delayed
import multiprocessing
import os
import glob
import logging
import sklearn.preprocessing
from sklearn.decomposition import PCA
from tqdm import tqdm


# ======================================================================
# QUICK DEBUG CHECK
# ----------------------------------------------------------------------
# Set this to a list of (0-indexed) subfolder positions to restrict the
# run to just those writer subfolders inside --in_dir, e.g.:
#
#     DEBUG_SUBFOLDER_INDICES = [1, 2]
#
# This picks the 2nd and 3rd writer folders (alphabetically sorted) and
# only extracts patches from those -- useful for sanity-checking that
# output filenames/labels look right before running on the full dataset.
#
# Set to None to process every subfolder normally (full run).
# ======================================================================
DEBUG_SUBFOLDER_INDICES = None


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tif', '.tiff', '.TIF', '.TIFF'
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def black_pixels(img):
    return np.sum(img == 0)
def white_pixels(img):
    return np.sum(img==255)


def calc_features(input_img, arguments):
    img = cv2.imread(input_img)

    if arguments.scale != -1:
        scale = arguments.scale
        height = img.shape[0] * scale
        width = img.shape[1] * scale
        new_size_mask = (int(width), int(height))
        img = cv2.resize(img, new_size_mask, interpolation=cv2.INTER_CUBIC)

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img_gray.copy()

    sift = cv2.SIFT_create(sigma=args.sigma)
    kp = sift.detect(img_gray, None)
    _, img_bin = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)


    def filter_keypoint(point):
        # if 3 < point.size < 10:
        #     return point
        return point


    new_kp = []
    uniq_kp = []

    num_pixel = arguments.win_size ** 2 

    for p in kp:
        point = (int(p.pt[0]), int(p.pt[1]))

        roi = get_image_patch(img, point[0], point[1], arguments)

        if roi is None:
            continue

        if arguments.centered:
            if img_bin[point[1], point[0]] != 0:
                continue

        if arguments.black_pixel_thresh != -1:
            if black_pixels(roi) > (arguments.black_pixel_thresh * num_pixel):
                continue 

        if arguments.white_pixel_thresh != -1:
            if white_pixels(roi) > (arguments.white_pixel_thresh * num_pixel):
                continue


        if not point in uniq_kp and filter_keypoint(p):
            new_kp.append(p)
            uniq_kp.append(point)

    _, desc = sift.compute(img_gray, new_kp)

    if desc is None or len(desc) == 0:
        print(f'No descriptors found for {os.path.splitext(os.path.basename(input_img))[0]}, skipping')
        return []

    desc = sklearn.preprocessing.normalize(desc, norm='l1')
    desc = np.sign(desc) * np.sqrt(np.abs(desc))
    desc = sklearn.preprocessing.normalize(desc, norm='l2')

    print(f'{len(new_kp)} out of {len(kp)} for {os.path.splitext(os.path.basename(input_img))[0]}')
    kp_tupple = [(p, desc[i]) for i, p in enumerate(uniq_kp)]


    return kp_tupple


def get_page_index(filename):
    """
    Deterministic per-writer page index, independent of the original form-ID
    string. For IAM (and any dataset where page IDs aren't clean integers,
    e.g. 'a01-003x'), this avoids parsing the ID at all -- each writer's own
    pages are sorted alphabetically and numbered 0..N-1. Two different pages
    can never collide, regardless of what the original filenames look like.
    """
    parent = os.path.dirname(filename)
    siblings = sorted(
        os.path.splitext(f)[0] for f in os.listdir(parent) if is_image_file(f)
    )
    basename = os.path.splitext(os.path.basename(filename))[0]
    return siblings.index(basename)


def get_image_patch(img, px, py, arguments):
    half_win_size = int(arguments.win_size / 2)
    if not (half_win_size < px < img.shape[1] - half_win_size and half_win_size < py < img.shape[0] - half_win_size):
        # if px - half_win_size < 0 or px + half_win_size > img.shape[1] or py - half_win_size < 0 or py + half_win_size > img.shape[0]:
        return None

    roi = img[py - half_win_size:py + half_win_size, px - half_win_size:px + half_win_size]
    assert roi.shape == (half_win_size * 2, half_win_size * 2), 'shape of the roi is not (%d,%d). It is (%d,%d)' % \
                                                                (half_win_size * 2, half_win_size * 2,
                                                                 roi.shape[0], roi.shape[1])
    return roi


def extract_patches(filename, tup, args):
    if len(tup) == 0:
        print(f'Skipping {os.path.basename(filename)} - no keypoints')
        return

    if len(tup) > args.patches_per_page and args.patches_per_page != -1:
        idx = np.linspace(0, len(tup) - 1, args.patches_per_page, dtype=np.int32)
        tup = [tup[i] for i in idx]

    points, _ = zip(*tup)

    img = cv2.imread(filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if args.scale != -1:
        scale = args.scale
        height = img.shape[0] * scale
        width = img.shape[1] * scale
        new_size_mask = (int(width), int(height))
        img = cv2.resize(img, new_size_mask, interpolation=cv2.INTER_CUBIC)

    # NEW: writer ID comes from the parent folder (needed for datasets like IAM
    # where it's not in the filename). Page index is a clean sequential integer
    # per writer -- NOT parsed from the original form ID -- so naming quirks
    # like 'a01-003x' can never cause a collision or a downstream regex/int()
    # crash.
    writer_id = os.path.basename(os.path.dirname(filename))
    page_idx  = get_page_index(filename)
    stem      = f'{writer_id}_{page_idx:04d}'

    count = 0
    for p in points:
        roi = get_image_patch(img, p[0], p[1], args)
        if roi is None:
            continue

        out_path = os.path.join(args.out_dir[0], stem, stem + '_' + str(count) + '.png')

        count = count + 1
        cv2.imwrite(out_path, roi)


if __name__ == "__main__":
    import argparse
    import pickle

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s ')

    parser = argparse.ArgumentParser(description="extract patches from images")
    parser.add_argument('--in_dir', metavar='in_dir', dest='in_dir', type=str, nargs=1,
                        help='input directory', required=True)
    parser.add_argument('--out_dir', metavar='out_dir', dest='out_dir', type=str, nargs=1,
                        help='output directory', required=True)
    parser.add_argument('--win_size', metavar='win_size', dest='win_size', type=int, nargs='?',
                        help='size of the patch',
                        default=32)
    parser.add_argument('--patches_per_page', metavar='patches_per_page', dest='patches_per_page', type=int, nargs='?',
                        help='maximal number of patches per page (-1 for no limit)',
                        default=-1)
    parser.add_argument('--scale', type=float, help='scale images up or down',
                        default=-1)
    parser.add_argument('--sigma', type=float, help='blur factor for SIFT',
                        default=1.6)
    parser.add_argument('--black_pixel_thresh', type=float, help='if more black_pixel_thresh percent of the pixels are black -> discard',
                        default=0.5)
    parser.add_argument('--white_pixel_thresh', type=float, help='if more than white_pixel_thresh percent of the pixels are white -> discard',
                        default=0.5)
    parser.add_argument('--centered', type=bool, help='filter patches whose keypoints are not located on handwriting, only for binarized datasets',
                    default=True)

    
    args = parser.parse_args()


    assert os.path.exists(args.in_dir[0]), 'in_dir {} does not exist'.format(args.in_dir[0])


    if not os.path.exists(args.out_dir[0]):
        logging.info('creating directory %s' % args.out_dir[0])
        os.mkdir(args.out_dir[0])

    assert len(os.listdir(args.out_dir[0])) == 0, 'out_dir is not empty'

    assert args.win_size % 2 == 0, 'win_size must be even'

    num_cores = int(multiprocessing.cpu_count() / 2)
    num_cores = 10
    path_to_centers = ''

    # NEW: optionally restrict to a handful of writer subfolders for a quick check
    if DEBUG_SUBFOLDER_INDICES is not None:
        all_subfolders = sorted(
            d for d in os.listdir(args.in_dir[0])
            if os.path.isdir(os.path.join(args.in_dir[0], d))
        )
        selected_subfolders = [
            all_subfolders[i] for i in DEBUG_SUBFOLDER_INDICES if i < len(all_subfolders)
        ]
        logging.info(f'DEBUG MODE: restricting to subfolders {selected_subfolders} '
                     f'(out of {len(all_subfolders)} total)')

        files = []
        for sf in selected_subfolders:
            files.extend([
                f for f in glob.glob(os.path.join(args.in_dir[0], sf, '**', '*.*'), recursive=True)
                if os.path.isfile(f) and is_image_file(f)
            ])
    else:
        files = [f for f in glob.glob(args.in_dir[0] + '/**/*.*', recursive=True) if os.path.isfile(f) and is_image_file(f)]

    assert len(files) > 0, 'no images found'
    logging.info('Found {} images'.format(len(files)))

    logging.info('calculating features for images in %s (number of cores:%d)' % (args.in_dir, num_cores))
    results = []
    results = Parallel(n_jobs=num_cores, verbose=9)(delayed(calc_features)(f, args) for f in files)

    logging.info('collecting descriptors')
    desc_list = []
    kp_list = []
    fn_list = []

    for r, f in tqdm(zip(results, files)):
        if len(r) == 0:
            logging.warning('no keypoints found in file {} '.format(f))
        for kp, desc in r:
            kp_list.append(kp)
            desc_list.append(desc)
            fn_list.append(f)

    # filtered = []
    # if args.cluster:
    #     descs = np.array(desc_list)
    #     samples = min(descs.shape[0], 500000)

    #     pca = PCA(32, whiten=True)
    #     pca.fit(descs[np.linspace(0, descs.shape[0]-1, samples).astype('int')])

    #     center_path = '/data/mpeer/resources/icdar2017_train_sift_patches_binarized/centers.pkl' #args.cluster
    #     kmeans = pickle.load(open(center_path,'rb'))
    #     patches_files = {}
    #     feature_count = 0
    #     batch_size = 50000
    #     for r, f in zip(results, files):
    #         kp, desc = zip(*r)
                
    #         desc = np.array(desc)
    #         desc = pca.transform(desc)
    #         dist = kmeans.transform(desc)
    #         prediction = kmeans.predict(desc)

    #         dist = np.sort(dist)
    #         ratio = dist[:, 0] / dist[:, 1]
    #         idx = (ratio > 0.9).nonzero()[0]
    #         print(f'remove {idx.shape} keypoints')
    #         kp = np.delete(kp, idx).tolist()
    #         desc =  np.delete(desc, idx).tolist()

    #         filtered.append([(k,d) for k,d in zip(kp, desc)])
    #             # for p, f, c, r in zip(batch(kp_list), batch(fn_list), prediction, ratio):
    #             #     if r <= 0.9:
    #             #         feature_count += 1
    #             #         if f in patches_files:
    #             #             patches_files[f].append((p, c))
    #             #         else:
    #             #             patches_files[f] = [(p, c)]

    # descs = None
    # results = filtered
    logging.info('creating labels directories in output directory')
    for fn in set(fn_list):
        # NEW: matches the writer_id + sequential page_idx stem used in extract_patches()
        writer_id = os.path.basename(os.path.dirname(fn))
        page_idx  = get_page_index(fn)
        f_name    = f'{writer_id}_{page_idx:04d}'
        if not os.path.exists(os.path.join(args.out_dir[0], str(f_name))):
            os.mkdir(os.path.join(args.out_dir[0], str(f_name)))

    # logging.info('copying %i (all patches per page) image patches to %s ' % (kp_list, args.out_dir[0]))
    results = Parallel(n_jobs=num_cores, verbose=9)(
        delayed(extract_patches)(filename, tup, args) for filename, tup in zip(files, results))

    config_out_path = os.path.join(args.out_dir[0], 'db-creation-parameters.json')
    logging.info('writing config parameters to {}')
    with open(config_out_path, 'w') as f:
        import json

        json.dump(vars(args), f)

    # NEW: record original form ID for every writer_id/page_idx pair, so the
    # mapping back to IAM's raw filenames is never lost.
    page_index_map = {}
    for fn in set(fn_list):
        writer_id = os.path.basename(os.path.dirname(fn))
        page_idx  = get_page_index(fn)
        orig_form_id = os.path.splitext(os.path.basename(fn))[0]
        page_index_map.setdefault(writer_id, {})[f'{page_idx:04d}'] = orig_form_id

    map_out_path = os.path.join(args.out_dir[0], 'page_index_map.json')
    with open(map_out_path, 'w') as f:
        import json
        json.dump(page_index_map, f, indent=2)
    logging.info(f'wrote writer/page index -> original form ID mapping to {map_out_path}')

    out_files = [f for f in glob.glob(args.out_dir[0] + '/**/*.*', recursive=True) if os.path.isfile(f) and is_image_file(f)]
    logging.info(f'done - extracted {len(out_files)} patches')