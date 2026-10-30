"""
main_new.py

Standard is:
train_label: writer
use_test_as_validation: true
Use whole trainset, with no validation, train on entire train set for all epochs and pick that model.
"""


import logging, os, argparse, math, random, re, glob
import pickle as pk
from pathlib import Path
from tqdm import tqdm
import psutil

import numpy as np
import torch
from torch import optim
import torch.backends.cudnn as cudnn

from torchvision import transforms

from pytorch_metric_learning import samplers
from pytorch_metric_learning.losses import MultiSimilarityLoss, NTXentLoss
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from utils.utils import GPU, seed_everything, load_config, getLogger, save_model, cosine_scheduler
from utils.triplet_loss import TripletLoss

from dataloading.writer_zoo import WriterZoo
from dataloading.GenericDataset import FilepathImageDataset
from dataloading.regex import pil_loader

from evaluators.retrieval import Retrieval
from page_encodings import SumPooling, GMP, MaxPooling, LSEPooling

from aug import Erosion, Dilation

from backbone import resnets
from backbone.model import Model

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


def compute_page_features(_features, writer, pages):
    _labels = list(zip(writer, pages))
    labels_np = np.array(_labels)
    features_np = np.array(_features)
    writer = labels_np[:, 0]
    page = labels_np[:, 1]

    page_features = []
    page_writer = []
    for w, p in tqdm(set(_labels), 'Page Features'):
        idx = np.where((writer == w) & (page == p))
        page_features.append(features_np[idx])
        page_writer.append(w)

    return page_features, page_writer

"""
def encode_per_class(model, args, poolings=[], use_trainset=False):
    key = 'trainset' if use_trainset else 'testset'
    dataset_cfg = args[key]
    ds_info = WriterZoo.datasets[dataset_cfg['dataset']]['set'][dataset_cfg['set']]

    basepath = WriterZoo.datasets[dataset_cfg['dataset']]['basepath']
    path = os.path.join(basepath, ds_info['path'])
    regex = ds_info['regex']

    pfs_per_pooling = [[] for _ in poolings]
    regex_w = regex.get('writer')
    regex_p = regex.get('page')

    srcs = sorted(list(glob.glob(f'{path}/**/*.png', recursive=True)))
    logging.info(f'Found {len(srcs)} images in {"train" if use_trainset else "test"}set')

    writer = [int('_'.join(re.search(regex_w, Path(f).name).groups())) for f in srcs]
    page   = [int('_'.join(re.search(regex_p, Path(f).name).groups())) for f in srcs]

    labels = list(zip(writer, page))

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Grayscale(num_output_channels=3)
    ]) if args.get('grayscale') else transforms.ToTensor()

    np_writer = np.array(writer)
    np_page   = np.array(page)
    print(f'Found {len(set(labels))} pages.')

    writers = []
    ds = FilepathImageDataset(srcs, pil_loader, transform)
    loader = torch.utils.data.DataLoader(
        ds, num_workers=8, batch_size=args['test_batch_size'],
        pin_memory=True
    )

    all_feats = []
    for img in tqdm(loader, desc='Extracting features'):
        img = img.cuda()
        with torch.no_grad():
            feat = torch.nn.functional.normalize(model(img))
        all_feats.append(feat.detach().cpu().numpy())
    all_feats = np.concatenate(all_feats)

    for w, p in tqdm(set(labels), desc='Encoding pages'):
        idx = np.where((np_writer == w) & (np_page == p))[0]
        feats = all_feats[idx]
        for i, pooling in enumerate(poolings):
            pfs_per_pooling[i].append(pooling.encode([feats]))
        writers.append(w)

    torch.cuda.empty_cache()
    pfs_per_pooling = [np.concatenate(pfs) for pfs in pfs_per_pooling]
    return pfs_per_pooling, writers"""

def encode_per_class(model, args, poolings=[], use_trainset=False):
    key = 'trainset' if use_trainset else 'testset'
    dataset_cfg = args[key]
    ds_info = WriterZoo.datasets[dataset_cfg['dataset']]['set'][dataset_cfg['set']]
    basepath = WriterZoo.datasets[dataset_cfg['dataset']]['basepath']
    path = os.path.join(basepath, ds_info['path'])
    regex = ds_info['regex']

    pfs_per_pooling = [[] for _ in poolings]
    regex_w = regex.get('writer')
    regex_p = regex.get('page')

    srcs = sorted(list(glob.glob(f'{path}/**/*.png', recursive=True)))
    logging.info(f'Found {len(srcs)} images in {"train" if use_trainset else "test"}set')

    print(f'[DEBUG] Starting regex parsing of {len(srcs)} filenames...', flush=True)
    writer = np.array([int('_'.join(re.search(regex_w, Path(f).name).groups())) for f in srcs])
    print(f'[DEBUG] Writer regex done.', flush=True)
    page   = np.array([int('_'.join(re.search(regex_p, Path(f).name).groups())) for f in srcs])
    print(f'[DEBUG] Page regex done.', flush=True)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Grayscale(num_output_channels=3)
    ]) if args.get('grayscale') else transforms.ToTensor()

    # Sort srcs so each page's images are contiguous — O(N log N), ~15s for 6M items
    print(f'[DEBUG] Sorting {len(srcs)} images by (writer, page)...', flush=True)
    sort_idx      = np.lexsort((page, writer))   # primary: writer, secondary: page
    srcs_sorted   = [srcs[i]   for i in sort_idx]
    writer_sorted = writer[sort_idx]
    page_sorted   = page[sort_idx]
    print(f'[DEBUG] Sort done.', flush=True)

    # Compute page boundary indices in the sorted array
    page_ids   = writer_sorted.astype(np.int64) * 100_000 + page_sorted
    boundaries = np.concatenate([[0], np.where(np.diff(page_ids))[0] + 1, [len(srcs_sorted)]])
    page_ranges = [
        (int(writer_sorted[boundaries[i]]), int(page_sorted[boundaries[i]]),
         boundaries[i], boundaries[i + 1])
        for i in range(len(boundaries) - 1)
    ]
    print(f'[DEBUG] Found {len(page_ranges)} pages. Building DataLoader...', flush=True)

    ds = FilepathImageDataset(srcs_sorted, pil_loader, transform)
    loader = torch.utils.data.DataLoader(
        ds, num_workers=4, batch_size=args['test_batch_size'],
        pin_memory=False, prefetch_factor=2
    )
    print(f'[DEBUG] DataLoader ready. Starting streaming extraction...', flush=True)

    page_iter  = iter(page_ranges)
    cur_w, cur_p, _, cur_end = next(page_iter)
    cur_buf    = []
    writers_out = []
    global_idx = 0
    pages_done = 0

    for i, img_batch in enumerate(tqdm(loader, desc='Extracting & pooling features')):
        img_batch = img_batch.cuda()
        with torch.no_grad():
            feat = torch.nn.functional.normalize(model(img_batch)).detach().cpu().numpy()
        del img_batch

        batch_end   = global_idx + len(feat)
        feat_offset = 0

        while global_idx < batch_end:
            take = min(cur_end - global_idx, batch_end - global_idx)
            cur_buf.append(feat[feat_offset : feat_offset + take])
            global_idx  += take
            feat_offset += take

            if global_idx == cur_end:
                # Page complete — pool and immediately discard patch features
                page_feats = np.concatenate(cur_buf, axis=0)
                for j, pooling in enumerate(poolings):
                    pfs_per_pooling[j].append(pooling.encode([page_feats]))
                writers_out.append(cur_w)
                pages_done += 1
                del cur_buf, page_feats
                cur_buf = []

                if pages_done % 200 == 0:
                    mem = psutil.virtual_memory()
                    print(f'[DEBUG] pages done: {pages_done}/{len(page_ranges)}, '
                          f'RAM used: {mem.used/1e9:.1f}GB, available: {mem.available/1e9:.1f}GB',
                          flush=True)

                try:
                    cur_w, cur_p, _, cur_end = next(page_iter)
                except StopIteration:
                    break

        del feat

        if i % 500 == 0:
            mem = psutil.virtual_memory()
            print(f'[DEBUG] batch {i}/{len(loader)}, '
                  f'RAM used: {mem.used/1e9:.1f}GB, available: {mem.available/1e9:.1f}GB',
                  flush=True)

    print(f'[DEBUG] Extraction & pooling done. {pages_done} pages processed.', flush=True)
    torch.cuda.empty_cache()
    print(f'[DEBUG] Starting final pfs concatenation...', flush=True)
    pfs_per_pooling = [np.concatenate(pfs) for pfs in pfs_per_pooling]
    print(f'[DEBUG] Final concatenation done. Shape: {pfs_per_pooling[0].shape}', flush=True)
    return pfs_per_pooling, writers_out


def inference(model, ds, args):
    model.eval()
    loader = torch.utils.data.DataLoader(
        ds, num_workers=8, batch_size=args['test_batch_size'],
        pin_memory=True
    )

    feats, pages, writers = [], [], []

    for sample, labels in tqdm(loader, desc='Inference'):
        if len(labels) == 3:
            w, p = labels[1], labels[2]
        else:
            w, p = labels[0], labels[1]

        writers.append(w)
        pages.append(p)
        sample = sample.cuda()

        with torch.no_grad():
            emb = torch.nn.functional.normalize(model(sample))
        feats.append(emb.detach().cpu().numpy())

    feats   = np.concatenate(feats)
    writers = np.concatenate(writers)
    pages   = np.concatenate(pages)
    return feats, writers, pages


def get_loss(args):
    loss_name = args['train_options'].get('loss', 'triplet')

    if loss_name == 'triplet':
        margin = args['train_options'].get('margin', 0.1)
        print(f'Using TripletLoss margin={margin}')
        return TripletLoss(margin=margin), 'triplet'

    elif loss_name == 'multisimilarity':
        print('Using MultiSimilarityLoss')
        return MultiSimilarityLoss(), 'multisimilarity'

    elif loss_name == 'ntxent':
        print('Using NTXentLoss')
        return NTXentLoss(), 'ntxent'

    else:
        raise ValueError(f'Unknown loss: {loss_name}')


def test(model, logger, args, name='Test'):
    sumps    = [SumPooling('l2', pn_alpha=0.4)]
    poolings = ['SumPooling-PN0p4']

    print('Extracting test features...')
    pfs_per_pooling, writer = encode_per_class(model, args, poolings=sumps, use_trainset=False)

    print('Extracting train features for PCA fitting...')
    train_pfs_per_pooling, _ = encode_per_class(model, args, poolings=sumps, use_trainset=True)

    best_map     = -1
    best_top1    = -1
    best_pooling = ''
    table        = []
    extra_keys   = [f'{m}-{k}' for m in ('hard', 'soft', 'perc') for k in range(1, 10)]
    columns      = ['Pooling', 'mAP', 'Top1'] + extra_keys

    pca_dims = args.get('pca_dims', [512]) #CONFIG TODO CHANGE 128:CVL, 400:ICDAR2017 
    if not isinstance(pca_dims, list):
        pca_dims = [pca_dims]

    for i, pfs in enumerate(pfs_per_pooling):
        for pca_dim in pca_dims:

            if pca_dim != -1:
                # cap at min of both train and test sample counts
                actual_dim = min(pca_dim, min(pfs.shape), min(train_pfs_per_pooling[i].shape))
                print(f'Fitting PCA (dim={actual_dim}) on train descriptors...')
                pca = PCA(actual_dim, whiten=True)
                pca.fit(train_pfs_per_pooling[i])
                pfs_tf = normalize(pca.transform(pfs), axis=1)
                p = str(actual_dim)
            else:
                pfs_tf = pfs
                pca    = None
                p      = 'full'

            print('Calculating mAP...')
            res, csv_res = Retrieval().eval(pfs_tf, writer)

            meanavp = res['map']
            tag     = f'{poolings[i]}-{p}'

            if meanavp > best_map:
                best_map     = meanavp
                best_top1    = res['top1']
                best_pooling = tag
                if pca is not None:
                    pk.dump(pca, open(os.path.join(logger.log_dir, 'pca.pkl'), 'wb'))

            table.append([tag, meanavp, res['top1']] + [csv_res[k] for k in extra_keys])
            print(f'{tag}-{name} MAP: {meanavp}')
            print(f'{tag}-{name} Top-1: {res["top1"]}')
            for k in extra_keys:
                print(f'{tag}-{name} {k}: {csv_res[k]}')

    logger.log_table(table, 'Results', columns)
    logger.log_value('Best-mAP',  best_map)
    logger.log_value('Best-Top1', best_top1)
    print(f'Best-Pooling: {best_pooling}')


def get_optimizer(args, model):
    return optim.Adam(
        model.parameters(),
        lr=args['optimizer_options']['base_lr'],
        weight_decay=args['optimizer_options']['wd']
    )


def validate(model, val_ds, args):
    desc, writer, pages = inference(model, val_ds, args)
    print('Inference done')
    pfs, writer = compute_page_features(desc, writer, pages)

    pooling = SumPooling('powernorm')
    descs   = pooling.encode(pfs)

    _eval = Retrieval()
    res, _ = _eval.eval(descs, writer)
    return res['map']

def get_base_dataset(ds):
    """Unwrap dataset wrappers until we find the ImageFolder with .labels"""
    while not hasattr(ds, 'labels'):
        ds = ds.dataset
    return ds

def train_one_epoch(model, train_ds, loss_fn, loss_type, optimizer, scheduler, epoch, args, logger):
    model.train()
    model = model.cuda()

    sampler = samplers.MPerClassSampler(
        np.array(get_base_dataset(train_ds.dataset).labels[args['train_label']])[train_ds.indices],
        args['train_options']['sampler_m'],
        length_before_new_iter=args['train_options']['length_before_new_iter']
    )
    train_triplet_loader = torch.utils.data.DataLoader(
        train_ds, sampler=sampler,
        batch_size=args['train_options']['batch_size'],
        drop_last=True, num_workers=8,
        pin_memory=True, persistent_workers=True
    )

    pbar = tqdm(train_triplet_loader)
    pbar.set_description(f'Epoch {epoch} Training')
    iters = len(train_triplet_loader)
    logger.log_value('Epoch', epoch, commit=False)

    for i, (samples, label) in enumerate(pbar):
        it = iters * epoch + i
        for j, param_group in enumerate(optimizer.param_groups):
            param_group['lr'] = scheduler[min(it, len(scheduler) - 1)]
            if param_group.get('name') == 'lambda':
                param_group['lr'] *= args['optimizer_options'].get('gmp_lr_factor', 1.0)

        samples = samples.cuda()
        samples.requires_grad = True
        l = label[0].cuda()

        emb = model(samples)

        if loss_type == 'triplet':
            loss = loss_fn(emb, l, emb, l)
        else:
            loss = loss_fn(emb, l)

        logger.log_value('loss', loss.item())
        logger.log_value('lr', optimizer.param_groups[0]['lr'])

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1) # changed from 2 to 1 to replicate paper params
        optimizer.step()

    torch.cuda.empty_cache()
    return model


def train(model, train_ds, val_ds, args, logger, optimizer):
    epochs = args['train_options']['epochs']
    niter_per_ep = math.ceil(args['train_options']['length_before_new_iter'] / args['train_options']['batch_size'])
    lr_schedule = cosine_scheduler(
        args['optimizer_options']['base_lr'],
        args['optimizer_options']['final_lr'],
        epochs, niter_per_ep,
        warmup_epochs=args['optimizer_options']['warmup_epochs'],
        start_warmup_value=0
    )

    loss_fn, loss_type = get_loss(args)

    for epoch in range(epochs):
        model = train_one_epoch(model, train_ds, loss_fn, loss_type, optimizer, lr_schedule, epoch, args, logger)

    return model, optimizer


def prepare_logging(args):
    Logger = getLogger(args['logger'])
    logger = Logger(os.path.join(args['log_dir'], args['super_fancy_new_name']), args=args)
    logger.log_options(args)
    return logger


def train_val_split(dataset, prop=0.9):
    authors = list(set(dataset.labels['writer']))
    random.shuffle(authors)

    train_len    = math.floor(len(authors) * prop)
    train_authors = authors[:train_len]
    val_authors   = authors[train_len:]
    print(f'{len(train_authors)} authors for training - {len(val_authors)} authors for validation')

    train_idxs, val_idxs = [], []
    for i in tqdm(range(len(dataset)), desc='Splitting dataset'):
        w = dataset.get_label(i)[0]
        if w in train_authors:
            train_idxs.append(i)
        if w in val_authors:
            val_idxs.append(i)

    return torch.utils.data.Subset(dataset, train_idxs), torch.utils.data.Subset(dataset, val_idxs)


def main(args):
    logger = prepare_logging(args)
    logger.update_config(args)

    backbone = getattr(resnets, args['model']['name'], None)()
    if not backbone:
        print('Unknown backbone!')
        raise ValueError(f"Unknown backbone: {args['model']['name']}")

    print('----------')
    print(f'Using {type(backbone)} as backbone')
    print(f"Using {args['model'].get('encoding', 'netvlad')} as encoding.")
    print('----------')

    random_enc = args['model'].get('encoding', None) == 'netrvlad'
    model = Model(backbone, dim=64, num_clusters=args['model']['num_clusters'], random=random_enc)
    model.train()
    model = model.cuda()

    tfs = []
    if args.get('grayscale'):
        tfs.extend([transforms.ToTensor(), transforms.Grayscale(num_output_channels=3)])
    else:
        tfs.append(transforms.ToTensor())

    if args.get('data_augmentation') == 'morph':
        tfs.extend([
            transforms.RandomApply([Erosion()],   p=0.1), # Changed from 0.3 to 0.1 to replicate paper params
            transforms.RandomApply([Dilation()],  p=0.1), # Changed from 0.3 to 0.1 to replicate paper params
        ])

    transform = transforms.Compose(tfs)

    train_dataset = None
    if args['trainset']:
        d = WriterZoo.get(**args['trainset'])
        train_dataset = d.TransformImages(transform=transform).SelectLabels(label_names=['writer', 'page'])

    if args.get('use_test_as_validation', False):
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Grayscale(num_output_channels=3)
        ]) if args.get('grayscale') else transforms.ToTensor()

        val_ds   = WriterZoo.get(**args['testset'])
        val_ds   = val_ds.TransformImages(transform=test_transform).SelectLabels(label_names=['writer', 'page'])
        train_ds = torch.utils.data.Subset(train_dataset, range(len(train_dataset)))
        val_ds   = torch.utils.data.Subset(val_ds, range(len(val_ds)))
    else:
        train_ds, val_ds = train_val_split(train_dataset)

    optimizer = get_optimizer(args, model)

    if args['checkpoint']:
        print(f"Loading model from {args['checkpoint']}")
        checkpoint = torch.load(args['checkpoint'])
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if not args['only_test']:
        model, optimizer = train(model, train_ds, val_ds, args, logger, optimizer)

    save_model(model, optimizer, args['train_options']['epochs'], os.path.join(logger.log_dir, 'model.pt'))
    model.eval()
    test(model, logger, args, name='Test')
    logger.finish()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s ')
    parser = argparse.ArgumentParser()

    parser.add_argument('--config',     default='config/icdar2017.yml')
    parser.add_argument('--only_test',  default=False, action='store_true')
    parser.add_argument('--checkpoint', default=None, type=str, metavar='PATH')
    parser.add_argument('--no-cuda',    action='store_true', default=False)
    parser.add_argument('--gpuid',      default='0', type=str)
    parser.add_argument('--seed',       default=2174, type=int)

    args   = parser.parse_args()
    config = load_config(args)[0]
    config['only_test']  = args.only_test
    config['checkpoint'] = args.checkpoint

    GPU.set(args.gpuid, 400)
    cudnn.benchmark = True
    seed_everything(args.seed)
    main(config)