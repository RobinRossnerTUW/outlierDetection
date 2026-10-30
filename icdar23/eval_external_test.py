"""
eval_external_test.py - test-only evaluation of a trained checkpoint on a test set it was not trained for.

Encodes the given WriterZoo test set (SumPooling-PN0p4), applies the PCA stored in the experiment dir
(pca.pkl, fitted on the experiment's own train pages, so the train set is not re-encoded), l2-normalises
and writes mAP, top-k, hard-k, soft-k, perc-k (P@k) and R@k to a CSV. Page descriptors go to an npz.
"""

import os, csv, argparse, logging
import pickle as pk

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from sklearn.preprocessing import normalize

from utils.utils import GPU, seed_everything, load_yaml
from evaluators.retrieval import Retrieval
from page_encodings import SumPooling
from backbone import resnets
from backbone.model import Model
from main_new import encode_per_class
from eval_baseline import recall_at_k


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_dir', required=True, help='experiment dir containing config.yaml, model.pt and pca.pkl')
    parser.add_argument('--dataset', required=True, help='WriterZoo dataset name of the external test set')
    parser.add_argument('--set', default='test')
    parser.add_argument('--out_dir', required=True, help='output dir for metrics.csv and page_feats.npz')
    parser.add_argument('--gpuid', default='0', type=str)
    parser.add_argument('--seed', default=2174, type=int)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s ')
    config = load_yaml(os.path.join(args.exp_dir, 'config.yaml'))
    config['testset'] = {'dataset': args.dataset, 'set': args.set}
    checkpoint = os.path.join(args.exp_dir, 'model.pt')
    os.makedirs(args.out_dir, exist_ok=True)

    seed_everything(args.seed)
    GPU.set(args.gpuid, 400)
    cudnn.benchmark = True

    backbone = getattr(resnets, config['model']['name'])()
    random = config['model'].get('encoding', None) == 'netrvlad'
    model = Model(backbone, dim=64, num_clusters=config['model']['num_clusters'], random=random)
    print(f'Loading model from {checkpoint}')
    model.load_state_dict(torch.load(checkpoint, map_location='cpu')['model_state_dict'])
    model = model.cuda().eval()

    print('Extracting test features...')
    (pfs,), writer = encode_per_class(model, config, poolings=[SumPooling('l2', pn_alpha=0.4)], use_trainset=False)

    pca = pk.load(open(os.path.join(args.exp_dir, 'pca.pkl'), 'rb'))
    print(f'Applying stored PCA (dim={pca.n_components_}, fitted on {pca.n_samples_} train pages)...')
    pfs_tf = normalize(pca.transform(pfs), axis=1)
    np.savez(os.path.join(args.out_dir, 'page_feats.npz'), raw=pfs, pca=pfs_tf, writer=np.array(writer))

    distances = Retrieval().compute_distances(pfs_tf)
    _, res = Retrieval.calc_map_from_distances(writer, distances)
    ks = range(1, 10)
    res.update(recall_at_k(writer, distances, ks))
    for k in ks:
        res[f'top-{k}'] = res[f'soft-{k}']  # top-k accuracy == at least one correct in top k

    row = {'checkpoint': checkpoint, 'dataset': f'{args.dataset}/{args.set}', 'pca_dim': pca.n_components_,
           'num_pages': len(writer)}
    row.update({k: float(v) for k, v in res.items()})
    out = os.path.join(args.out_dir, 'metrics.csv')
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        w.writeheader()
        w.writerow(row)

    print(f"mAP: {res['map']:.4f}  Top-1: {res['top1']:.4f}  R@1: {res['R@1']:.4f}")
    print(f'Wrote {out}')


if __name__ == '__main__':
    main()
