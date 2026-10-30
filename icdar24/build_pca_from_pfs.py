#!/usr/bin/env python
"""
build_pca_from_pfs.py — reconstructs the missing pca.pkl from the
already-saved pre-PCA descriptors (pfs.npy), reproducing exactly
what evaluators/validators.py::pca() did.
"""
import argparse, pickle
from pathlib import Path
import numpy as np
import yaml
from sklearn.decomposition import PCA

ap = argparse.ArgumentParser()
ap.add_argument('--experiment_dir', required=True)
ap.add_argument('--pca_dim', type=int, default=None,
                help='Override; else read eval_options.pca_dim from config.yaml')
args = ap.parse_args()

exp_dir = Path(args.experiment_dir)
pfs = np.load(exp_dir / 'pfs.npy')
print(f'Loaded pfs.npy: shape={pfs.shape}')

pca_dim = args.pca_dim
if pca_dim is None:
    cfg = yaml.safe_load(open(exp_dir / 'config.yaml'))
    pca_dim = cfg['eval_options']['pca_dim']
    print(f'Read pca_dim={pca_dim} from config.yaml')

pca_dim = min(min(pfs.shape), pca_dim)   # same clamp as the original pca()
print(f'Fitting PCA with shape {pca_dim}')

pca = PCA(pca_dim, whiten=True)          # same call as validators.py
pca.fit(pfs)

out_path = exp_dir / 'pca.pkl'
pickle.dump(pca, open(out_path, 'wb'))
print(f'Saved: {out_path}  (n_components_={pca.n_components_})')