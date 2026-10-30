"""
eval_sgr_grid.py - grid search over SGR reranking params on cached page descriptors.

Needs the --feat_cache npz written by eval_baseline.py (no re-encoding, PCA fitted once),
or --embeddings/--writers .npy of already PCA-ed page descriptors (e.g. the ICDAR experiment's).
Writes every config's metrics to <out_dir>/sgr_grid.csv and the best (by mAP)
to --out in the same format as eval_baseline.py.
"""

import os, csv, argparse, itertools

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from tqdm import tqdm

from evaluators.retrieval import Retrieval
from reranking.sgr import sgr_reranking
from eval_baseline import recall_at_k


def metrics(writer, distances):
    _, res = Retrieval.calc_map_from_distances(writer, distances)
    ks = range(1, 10)
    res.update(recall_at_k(writer, distances, ks))
    for k in ks:
        res[f'top-{k}'] = res[f'soft-{k}']
    return {k: float(v) for k, v in res.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--feat_cache', default=None, help='npz written by eval_baseline.py --feat_cache')
    parser.add_argument('--embeddings', default=None,
                        help='alternative to --feat_cache: .npy of final (PCA-ed, l2-normalised) test '
                             'page descriptors, e.g. embeddings_SumPooling-PN0p4_512.npy; no PCA is fitted')
    parser.add_argument('--writers', default=None, help='with --embeddings: .npy of page writer labels')
    parser.add_argument('--baseline_out', default=None, help='also write the no-rerank metrics to this csv')
    parser.add_argument('--out', required=True, help='csv for the best config')
    parser.add_argument('--grid_out', default=None, help='default: sgr_grid.csv next to --out')
    parser.add_argument('--ks', default=[1, 2, 3, 4], type=int, nargs='+')
    parser.add_argument('--layers', default=[1, 2, 3, 4], type=int, nargs='+')
    parser.add_argument('--gammas', default=[0.1, 0.2, 0.3, 0.5], type=float, nargs='+')
    parser.add_argument('--pca_dim', default=128, type=int)
    parser.add_argument('--seed', default=2174, type=int)
    args = parser.parse_args()

    np.random.seed(args.seed); torch.manual_seed(args.seed)

    if args.embeddings:
        X = normalize(np.load(args.embeddings), axis=1)
        writer = np.load(args.writers).tolist()
        pca_dim, source = X.shape[1], args.embeddings
    else:
        d = np.load(args.feat_cache)
        pfs, writer, train_pfs = d['test'], d['writer'].tolist(), d['train']
        pca_dim = min(min(pfs.shape), args.pca_dim)
        pca = PCA(pca_dim, whiten=True)
        pca.fit(train_pfs)
        X = normalize(pca.transform(pfs), axis=1)
        source = args.feat_cache
    X = torch.tensor(X, dtype=torch.float32)

    if args.baseline_out:
        res = metrics(writer, (1 - X @ X.t()).numpy())
        row = {'checkpoint': source, 'rerank': 'none', 'sgr_k': '', 'sgr_layers': '', 'sgr_gamma': '',
               'pca_dim': pca_dim, 'num_pages': len(writer), **res}
        with open(args.baseline_out, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys())); w.writeheader(); w.writerow(row)
        print(f"Baseline (no rerank): mAP={res['map']:.4f} top1={res['top1']:.4f} -> {args.baseline_out}")

    base = {'checkpoint': source, 'rerank': 'sgr', 'pca_dim': pca_dim, 'num_pages': len(writer)}
    rows = []
    grid = list(itertools.product(args.ks, args.layers, args.gammas))
    for k, layers, gamma in tqdm(grid, desc='SGR grid'):
        _, dist = sgr_reranking(X.clone(), k=k, layer=layers, gamma=gamma)
        res = metrics(writer, dist.numpy())
        rows.append({**base, 'sgr_k': k, 'sgr_layers': layers, 'sgr_gamma': gamma, **res})
        tqdm.write(f'k={k} layers={layers} gamma={gamma}: mAP={res["map"]:.4f} top1={res["top1"]:.4f}')

    grid_out = args.grid_out or os.path.join(os.path.dirname(os.path.abspath(args.out)), 'sgr_grid.csv')
    rows_sorted = sorted(rows, key=lambda r: r['map'], reverse=True)
    for path, data in [(grid_out, rows_sorted), (args.out, rows_sorted[:1])]:
        with open(path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(data)

    b = rows_sorted[0]
    print(f"\nBest: k={b['sgr_k']} layers={b['sgr_layers']} gamma={b['sgr_gamma']}  "
          f"mAP={b['map']:.4f}  top1={b['top1']:.4f}")
    print(f'Wrote {grid_out}\nWrote {args.out}')


if __name__ == '__main__':
    main()
