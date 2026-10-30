"""
Supervised writer retrieval baseline.

Run patch extraction first (no clustering needed):
    python helpers/extract_patches_only.py --in_dir <raw_docs> --out_dir <patches>

Then train:
    python baseline.py --config config/cvl_supervised.yml --gpuid 0

Notes:
- All 27 train writers are used for training (no val split).
- The final model (last epoch) is used — no best-checkpoint selection.
"""

import os, math, random, glob, re, logging, argparse
from pathlib import Path
from tqdm import tqdm

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch import optim
from torch.cuda.amp import autocast, GradScaler
from torchvision import transforms
from pytorch_metric_learning import samplers
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

from utils.utils import GPU, seed_everything, load_config, getLogger, save_model, cosine_scheduler
from dataloading.writer_zoo import WriterZoo
from dataloading.GenericDataset import FilepathImageDataset
from dataloading.regex import pil_loader
from evaluators.retrieval import Retrieval
from page_encodings import SumPooling
from aug import Erosion, Dilation
from utils.triplet_loss import TripletLoss
from backbone import resnets
from backbone.model import Model

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def encode_pages(model, args):
    """
    Extract per-patch features from the test set, pool them per page,
    and return (page_descriptors, page_writer_ids).
    """
    cfg     = args['testset']
    ds_meta = WriterZoo.datasets[cfg['dataset']]
    ds_set  = ds_meta['set'][cfg['set']]
    path    = os.path.join(ds_meta['basepath'], ds_set['path'])
    regex   = ds_set['regex']

    srcs = sorted(glob.glob(f'{path}/**/*.png', recursive=True))
    logging.info(f'Found {len(srcs)} test patches')

    rx_w    = regex['writer']
    rx_p    = regex['page']
    writers = [int('_'.join(re.search(rx_w, Path(f).name).groups())) for f in srcs]
    pages   = [int('_'.join(re.search(rx_p, Path(f).name).groups())) for f in srcs]

    tf = transforms.Compose([transforms.ToTensor()] + (
        [transforms.Grayscale(num_output_channels=3)] if args.get('grayscale') else []))

    np_w, np_p = np.array(writers), np.array(pages)
    pooling    = SumPooling('l2', pn_alpha=0.4)
    page_descs, page_writers = [], []

    for w, p in tqdm(set(zip(writers, pages)), desc='Encoding test pages'):
        idx    = np.where((np_w == w) & (np_p == p))[0]
        loader = torch.utils.data.DataLoader(
            FilepathImageDataset([srcs[i] for i in idx], pil_loader, tf),
            batch_size=args['test_batch_size'], num_workers=8)

        feats = []
        with torch.no_grad():
            for imgs in loader:
                f = model(imgs.cuda())
                feats.append(torch.nn.functional.normalize(f).cpu().numpy())

        page_descs.append(pooling.encode([np.concatenate(feats)]))
        page_writers.append(w)

    torch.cuda.empty_cache()
    return np.concatenate(page_descs), np.array(page_writers)


def run_test(model, args, logger):
    model.eval()
    descs, writers = encode_pages(model, args)

    pca_dim = min(descs.shape[0], descs.shape[1], args.get('pca_dim', 512))
    pca     = PCA(pca_dim, whiten=True)
    descs   = normalize(pca.fit_transform(descs), axis=1)

    res, _ = Retrieval().eval(descs, writers)
    print(f'Final — Test mAP: {res["map"]:.4f}  Top-1: {res["top1"]:.4f}')
    logger.log_value('Test-mAP',  res['map'])
    logger.log_value('Test-Top1', res['top1'])
    return res


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(model, train_ds, args, logger):
    scaler = GradScaler()

    opt = optim.Adam(
        model.parameters(),
        lr=args['optimizer_options']['base_lr'],
        weight_decay=args['optimizer_options']['wd'])

    epochs   = args['train_options']['epochs']
    n_iter   = math.ceil(args['train_options']['length_before_new_iter'] /
                         args['train_options']['batch_size'])
    schedule = cosine_scheduler(
        args['optimizer_options']['base_lr'],
        args['optimizer_options']['final_lr'],
        epochs, n_iter,
        warmup_epochs=args['optimizer_options']['warmup_epochs'])

    loss_fn = TripletLoss(margin=args['train_options']['margin'])

    writer_labels = np.array(train_ds.labels['writer'])
    sampler = samplers.MPerClassSampler(
        writer_labels,
        args['train_options']['sampler_m'],
        length_before_new_iter=args['train_options']['length_before_new_iter'])
    train_loader = torch.utils.data.DataLoader(
        train_ds, sampler=sampler,
        batch_size=args['train_options']['batch_size'],
        drop_last=True, num_workers=8)

    print(f'Training on {len(set(writer_labels))} writers, {len(train_ds)} patches')

    for epoch in range(epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}')

        for i, (imgs, labels) in enumerate(pbar):
            it = len(train_loader) * epoch + i
            lr = schedule[min(it, len(schedule) - 1)]
            for pg in opt.param_groups:
                pg['lr'] = lr

            imgs = imgs.cuda()
            w    = labels[0].cuda()

            with autocast():
                emb  = model(imgs)
                loss = loss_fn(emb, w, emb, w)

            pbar.set_postfix(loss=f'{loss.item():.4f}', lr=f'{lr:.2e}')
            logger.log_value('loss', loss.item())
            logger.log_value('lr',   lr)

            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2)
            scaler.step(opt)
            scaler.update()

        torch.cuda.empty_cache()

    return model, opt


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    Logger = getLogger(args['logger'])
    logger = Logger(os.path.join(args['log_dir'], args['super_fancy_new_name']), args=args)
    logger.log_options(args)

    backbone = getattr(resnets, args['model']['name'])()
    model    = Model(backbone, dim=64,
                     num_clusters=args['model']['num_clusters'],
                     random=False)
    model    = model.cuda()

    tfs = [transforms.ToTensor()]
    if args.get('grayscale'):
        tfs.append(transforms.Grayscale(num_output_channels=3))
    if args.get('data_augmentation') == 'morph':
        tfs += [transforms.RandomApply([Erosion()],  p=0.3),
                transforms.RandomApply([Dilation()], p=0.3)]
    transform = transforms.Compose(tfs)

    dataset = WriterZoo.get(**args['trainset'])
    dataset = (dataset
               .TransformImages(transform=transform)
               .SelectLabels(label_names=['writer', 'page']))

    print(f'Loaded {len(dataset)} patches from '
          f'{len(set(dataset.labels["writer"]))} writers')

    if args.get('checkpoint'):
        ckpt = torch.load(args['checkpoint'])
        model.load_state_dict(ckpt['model_state_dict'])
        print(f'Loaded checkpoint from {args["checkpoint"]}')

    if not args.get('only_test'):
        model, opt = train(model, dataset, args, logger)
        final_path = os.path.join(logger.log_dir, 'model_final.pt')
        save_model(model, opt, args['train_options']['epochs'], final_path)
        print(f'Saved final model to {final_path}')

    print('\n[Final evaluation]')
    run_test(model, args, logger)
    logger.finish()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

    parser = argparse.ArgumentParser()
    parser.add_argument('--config',     default='config/cvl.yml')
    parser.add_argument('--only_test',  action='store_true')
    parser.add_argument('--checkpoint', default=None, type=str)
    parser.add_argument('--gpuid',      default='5',  type=str)
    parser.add_argument('--seed',       default=42,   type=int)
    parsed = parser.parse_args()

    config = load_config(parsed)[0]
    print(config['model'])
    GPU.set(parsed.gpuid, 400)
    cudnn.benchmark = True
    seed_everything(parsed.seed)
    main(config)