"""
eval_baseline.py - test-only evaluation of a trained checkpoint, writes all retrieval metrics to a CSV.

Reproduces main.test() (SumPooling-PN0p4 -> PCA(128, whiten) fitted on train pages -> l2 norm),
optionally followed by SGR reranking, and reports mAP, top-k, hard-k, soft-k, perc-k (P@k) and R@k.
"""

import os, csv, argparse, logging

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

from utils.utils import GPU, seed_everything, load_yaml
from evaluators.retrieval import Retrieval
from page_encodings import SumPooling
from backbone import resnets
from backbone.model import Model
from reranking.sgr import sgr_reranking
from main import encode_per_class


def recall_at_k(labels, distances, ks):
    labels = np.array(labels)
    recalls = {k: [] for k in ks}
    for i in range(len(labels)):
        idxs = np.argsort(distances[i]).flatten()
        idxs = idxs[idxs != i]  # drop the query itself
        rel = labels[idxs] == labels[i]
        n_rel = rel.sum()
        if n_rel == 0:
            continue
        for k in ks:
            recalls[k].append(rel[:k].sum() / n_rel)
    return {f'R@{k}': float(np.mean(v)) for k, v in recalls.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_dir', required=True, help='experiment dir containing config.yaml and model.pt')
    parser.add_argument('--checkpoint', default=None, help='default: <exp_dir>/model.pt')
    parser.add_argument('--out', required=True, help='output csv path')
    parser.add_argument('--rerank', action='store_true', help='apply SGR reranking')
    parser.add_argument('--sgr_k', default=4, type=int)
    parser.add_argument('--sgr_layers', default=1, type=int)
    parser.add_argument('--sgr_gamma', default=0.4, type=float)
    parser.add_argument('--pca_dim', default=128, type=int)
    parser.add_argument('--feat_cache', default=None,
                        help='npz file to store/reuse page descriptors (skips re-encoding on the second run)')
    parser.add_argument('--gpuid', default='0', type=str)
    parser.add_argument('--seed', default=2174, type=int)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s ')
    config = load_yaml(os.path.join(args.exp_dir, 'config.yaml'))
    checkpoint = args.checkpoint or os.path.join(args.exp_dir, 'model.pt')

    seed_everything(args.seed)

    if args.feat_cache and os.path.isfile(args.feat_cache):
        print(f'Loading cached page descriptors from {args.feat_cache}')
        d = np.load(args.feat_cache)
        pfs, writer, train_pfs = d['test'], d['writer'].tolist(), d['train']
    else:
        GPU.set(args.gpuid, 400)
        cudnn.benchmark = True

        backbone = getattr(resnets, config['model']['name'])()
        random = config['model'].get('encoding', None) == 'netrvlad'
        model = Model(backbone, dim=64, num_clusters=config['model']['num_clusters'], random=random)
        print(f'Loading model from {checkpoint}')
        model.load_state_dict(torch.load(checkpoint, map_location='cpu')['model_state_dict'])
        model = model.cuda().eval()

        sumps = [SumPooling('l2', pn_alpha=0.4)]
        print('Extracting test features...')
        (pfs,), writer = encode_per_class(model, config, poolings=sumps, use_trainset=False)
        print('Extracting train features for PCA fitting...')
        (train_pfs,), _ = encode_per_class(model, config, poolings=sumps, use_trainset=True)

        if args.feat_cache:
            np.savez(args.feat_cache, test=pfs, writer=np.array(writer), train=train_pfs)
            print(f'Saved page descriptors to {args.feat_cache}')

    pca_dim = min(min(pfs.shape), args.pca_dim)
    print(f'Fitting PCA (dim={pca_dim}) on train descriptors...')
    pca = PCA(pca_dim, whiten=True)
    pca.fit(train_pfs)
    pfs_tf = normalize(pca.transform(pfs), axis=1)

    if args.rerank:
        print(f'SGR reranking (k={args.sgr_k}, layers={args.sgr_layers}, gamma={args.sgr_gamma})...')
        _, distances = sgr_reranking(torch.tensor(pfs_tf, dtype=torch.float32),
                                     k=args.sgr_k, layer=args.sgr_layers, gamma=args.sgr_gamma)
        distances = distances.numpy()
    else:
        distances = Retrieval().compute_distances(pfs_tf)

    _, res = Retrieval.calc_map_from_distances(writer, distances)
    ks = range(1, 10)  # same range as Retrieval (hard/soft/perc)
    res.update(recall_at_k(writer, distances, ks))
    for k in ks:
        res[f'top-{k}'] = res[f'soft-{k}']  # top-k accuracy == at least one correct in top k

    row = {'checkpoint': checkpoint, 'rerank': 'sgr' if args.rerank else 'none',
           'sgr_k': args.sgr_k if args.rerank else '', 'sgr_layers': args.sgr_layers if args.rerank else '',
           'sgr_gamma': args.sgr_gamma if args.rerank else '', 'pca_dim': pca_dim, 'num_pages': len(writer)}
    row.update({k: float(v) for k, v in res.items()})

    with open(args.out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        w.writeheader()
        w.writerow(row)

    print(f"mAP: {res['map']:.4f}  Top-1: {res['top1']:.4f}  R@1: {res['R@1']:.4f}")
    print(f'Wrote {args.out}')


if __name__ == '__main__':
    main()
