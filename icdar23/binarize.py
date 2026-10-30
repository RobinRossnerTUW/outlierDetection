import cv2
import os
import glob
from tqdm import tqdm
import numpy as np

def binarize(in_dir, out_dir, blockSize=75, C=18, denoise=True, despeckle=True):
    os.makedirs(out_dir, exist_ok=True)
    files = glob.glob(f'{in_dir}/**/*.*', recursive=True)
    files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.tif', '.tiff'))]

    for f in tqdm(files):
        rel = os.path.relpath(f, in_dir)
        out_path = os.path.join(out_dir, rel)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)

        if denoise:
            img = cv2.GaussianBlur(img, (3, 3), 0)   # smooth faint paper texture before thresholding

        binary = cv2.adaptiveThreshold(img, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=blockSize,
            C=C)

        if despeckle:
            kernel = np.ones((2, 2), np.uint8)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        cv2.imwrite(out_path, binary)

if __name__ == '__main__':
    binarize('/data/rrossner/datasets/CVL/cvl_train_raw', '/data/rrossner/datasets/CVL/cvl_train_binarized')
    binarize('/data/rrossner/datasets/CVL/cvl_test_raw',  '/data/rrossner/datasets/CVL/cvl_test_binarized')