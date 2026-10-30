"""
main.py
"""

import logging, os, argparse, math, random, re, glob
import pickle as pk
from pathlib import Path
from tqdm import tqdm

import numpy as np
import torch
from torch import optim
import torch.backends.cudnn as cudnn

from torchvision import transforms

from pytorch_metric_learning import samplers
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from utils.utils import GPU, seed_everything, load_config, getLogger, save_model, cosine_scheduler

from dataloading.writer_zoo import WriterZoo
from dataloading.GenericDataset import FilepathImageDataset
from dataloading.regex import pil_loader

from evaluators.retrieval import Retrieval
from page_encodings import SumPooling, GMP, MaxPooling, LSEPooling

from aug import Erosion, Dilation
from pytorch_metric_learning.losses import MultiSimilarityLoss

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
    for w, p in tqdm(set(labels), desc='Encoding pages'):
        idx = np.where((np_writer == w) & (np_page == p))[0]
        fps = [srcs[i] for i in idx]

        ds = FilepathImageDataset(fps, pil_loader, transform)
        loader = torch.utils.data.DataLoader(ds, num_workers=4, batch_size=args['test_batch_size'])

        feats = []
        for img in loader:
            img = img.cuda()
            with torch.no_grad():
                feat = torch.nn.functional.normalize(model(img))
            feats.append(feat.detach().cpu().numpy())

        feats = np.concatenate(feats)

        for i, pooling in enumerate(poolings):
            pfs_per_pooling[i].append(pooling.encode([feats]))
        writers.append(w)

    torch.cuda.empty_cache()
    pfs_per_pooling = [np.concatenate(pfs) for pfs in pfs_per_pooling]

    return pfs_per_pooling, writers


def inference(model, ds, args):
    model.eval()
    loader = torch.utils.data.DataLoader(ds, num_workers=4, batch_size=args['test_batch_size'])

    feats = []
    pages = []
    writers = []

    for sample, labels in tqdm(loader, desc='Inference'):
        if len(labels) == 3:
            w,p = labels[1], labels[2]
        else:
            w,p = labels[0], labels[1]

        writers.append(w)
        pages.append(p)
        sample = sample.cuda()

        with torch.no_grad():
            emb = model(sample)
            emb = torch.nn.functional.normalize(emb)
        feats.append(emb.detach().cpu().numpy())
    
    feats = np.concatenate(feats)
    writers = np.concatenate(writers)
    pages = np.concatenate(pages)   

    return feats, writers, pages

def test(model, logger, args, name='Test'):

    sumps   = [SumPooling('l2', pn_alpha=0.4)]
    poolings = ['SumPooling-PN0p4']

    print('Extracting test features...')
    pfs_per_pooling, writer = encode_per_class(model, args, poolings=sumps, use_trainset=False)

    print('Extracting train features for PCA fitting...')
    train_pfs_per_pooling, _ = encode_per_class(model, args, poolings=sumps, use_trainset=True)

    best_map    = -1
    best_top1   = -1
    best_pooling = ''
    table        = []
    columns      = ['Pooling', 'mAP', 'Top1']

    for i, pfs in enumerate(pfs_per_pooling):
        for pca_dim in [128]: #TODO SELF CHANGE Changed from 512 to 128

            if pca_dim != -1:
                pca_dim = min(min(pfs.shape), pca_dim)
                print(f'Fitting PCA (dim={pca_dim}) on train descriptors...')

                pca = PCA(pca_dim, whiten=True)
                pca.fit(train_pfs_per_pooling[i])
                pfs_tf = normalize(pca.transform(pfs), axis=1)
            else:
                pfs_tf = pfs

            print('Calculating mAP...')
            res, _ = Retrieval().eval(pfs_tf, writer)

            p        = str(pca_dim) if pca_dim != -1 else 'full'
            meanavp  = res['map']
            tag      = f'{poolings[i]}-{p}'

            if meanavp > best_map:
                best_map     = meanavp
                best_top1    = res['top1']
                best_pooling = tag
                pk.dump(pca, open(os.path.join(logger.log_dir, 'pca.pkl'), 'wb'))

            table.append([tag, meanavp, res['top1']])
            print(f'{tag}-{name} MAP: {meanavp}')
            print(f'{tag}-{name} Top-1: {res["top1"]}')

    logger.log_table(table, 'Results', columns)
    logger.log_value('Best-mAP',  best_map)
    logger.log_value('Best-Top1', best_top1)
    print(f'Best-Pooling: {best_pooling}')

###########

def get_optimizer(args, model):
    optimizer = optim.Adam(model.parameters(), lr=args['optimizer_options']['base_lr'],
                    weight_decay=args['optimizer_options']['wd'])
    return optimizer

def validate(model, val_ds, args):
    desc, writer, pages = inference(model, val_ds, args)
    print('Inference done')
    pfs, writer = compute_page_features(desc, writer, pages)

    norm = 'powernorm'
    pooling = SumPooling(norm)
    descs = pooling.encode(pfs)

    _eval = Retrieval()
    res, _ = _eval.eval(descs, writer)
    meanavp = res['map']

    return meanavp


def train_one_epoch(model, train_ds, triplet_loss, optimizer, scheduler, epoch, args, logger):

    model.train()
    model = model.cuda()

    # set up the triplet stuff
    sampler = samplers.MPerClassSampler(np.array(train_ds.dataset.dataset.labels[args['train_label']])[train_ds.indices], args['train_options']['sampler_m'], length_before_new_iter=args['train_options']['length_before_new_iter']) #len(ds))
    train_triplet_loader = torch.utils.data.DataLoader(train_ds, sampler=sampler, batch_size=args['train_options']['batch_size'], drop_last=True, num_workers=32)
    
    pbar = tqdm(train_triplet_loader)
    pbar.set_description('Epoch {} Training'.format(epoch))
    iters = len(train_triplet_loader)
    logger.log_value('Epoch', epoch, commit=False)

    for i, (samples, label) in enumerate(pbar):
        it = iters * epoch + i
        for i, param_group in enumerate(optimizer.param_groups):
            if it > (len(scheduler) - 1):
                param_group['lr'] = scheduler[-1]
            else:
                param_group["lr"] = scheduler[it]
            
            if param_group.get('name', None) == 'lambda':
                param_group['lr'] *= args['optimizer_options']['gmp_lr_factor']
   
        samples = samples.cuda()
        samples.requires_grad=True

        l = label[0]

        l = l.cuda()

        emb = model(samples)

        loss = triplet_loss(emb, l)
        logger.log_value(f'loss', loss.item())
        logger.log_value(f'lr', optimizer.param_groups[0]['lr'])

        # compute gradient and update weights
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2)
        optimizer.step()

    torch.cuda.empty_cache()
    return model

def train(model, train_ds, val_ds, args, logger, optimizer):

    epochs = args['train_options']['epochs']

    niter_per_ep = math.ceil(args['train_options']['length_before_new_iter'] / args['train_options']['batch_size'])
    lr_schedule = cosine_scheduler(args['optimizer_options']['base_lr'], args['optimizer_options']['final_lr'], epochs, niter_per_ep, warmup_epochs=args['optimizer_options']['warmup_epochs'], start_warmup_value=0)

    loss = MultiSimilarityLoss()
    print('Using MultiSimilarity Loss')

    for epoch in range(epochs):
        model = train_one_epoch(model, train_ds, loss, optimizer, lr_schedule, epoch, args, logger)

    return model, optimizer

def prepare_logging(args):
    os.path.join(args['log_dir'], args['super_fancy_new_name'])
    Logger = getLogger(args["logger"])
    logger = Logger(os.path.join(args['log_dir'], args['super_fancy_new_name']), args=args)
    logger.log_options(args)
    return logger

def train_val_split(dataset, prop = 0.9):
    authors = list(set(dataset.labels['writer']))
    random.shuffle(authors)

    train_len = math.floor(len(authors) * prop)
    train_authors = authors[:train_len]
    val_authors = authors[train_len:]

    print(f'{len(train_authors)} authors for training - {len(val_authors)} authors for validation')

    train_idxs = []
    val_idxs = []

    for i in tqdm(range(len(dataset)), desc='Splitting dataset'):
        w = dataset.get_label(i)[0]
        if w in train_authors:
            train_idxs.append(i)
        if w in val_authors:
            val_idxs.append(i)

    train = torch.utils.data.Subset(dataset, train_idxs)
    val = torch.utils.data.Subset(dataset, val_idxs)

    return train, val

def main(args):
    logger = prepare_logging(args)
    logger.update_config(args)

    backbone = getattr(resnets, args['model']['name'], None)()
    if not backbone:
        print("Unknown backbone!")
        raise

    print('----------')
    print(f'Using {type(backbone)} as backbone')
    print(f'''Using {args['model'].get('encoding', 'netvlad')} as encoding.''')
    print('----------')

    random = args['model'].get('encoding', None) == 'netrvlad'
    model = Model(backbone, dim=64, num_clusters=args['model']['num_clusters'], random=random)
    model.train()
    model = model.cuda()

    tfs = []
   
    if args.get('grayscale', None):
        tfs.extend([
            transforms.ToTensor(),
            transforms.Grayscale(num_output_channels=3)
        ])
    else:
        tfs.append(transforms.ToTensor())

    if args.get('data_augmentation', None) == 'morph':
        tfs.extend([transforms.RandomApply(
            [Erosion()],
            p=0.3
        ),
        transforms.RandomApply(
            [Dilation()],
            p=0.3
        )])

    transform = transforms.Compose(tfs)

    train_dataset = None
    if args['trainset']:
        d = WriterZoo.get(**args['trainset'])
        train_dataset = d.TransformImages(transform=transform).SelectLabels(label_names=['writer', 'page'])
    
    if args.get('use_test_as_validation', False):
        val_ds = WriterZoo.get(**args['testset'])
        if args.get('grayscale', None):
            test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Grayscale(num_output_channels=3)
            ])
        else:
            test_transform = transforms.ToTensor()
        val_ds = val_ds.TransformImages(transform=test_transform).SelectLabels(label_names=['writer', 'page'])

        train_ds = torch.utils.data.Subset(train_dataset, range(len(train_dataset)))
        val_ds = torch.utils.data.Subset(val_ds, range(len(val_ds)))
    else:
        train_ds, val_ds = train_val_split(train_dataset)

    optimizer = get_optimizer(args, model)

    if args['checkpoint']:
        print(f'''Loading model from {args['checkpoint']}''')
        checkpoint = torch.load(args['checkpoint'])
        model.load_state_dict(checkpoint['model_state_dict'])    
        model.eval() 

        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])    

    if not args['only_test']:
        model, optimizer = train(model, train_ds, val_ds, args, logger, optimizer)

    # testing
    save_model(model, optimizer, args['train_options']['epochs'], os.path.join(logger.log_dir, 'model.pt'))
    model.eval()
    test(model, logger, args, name='Test')
    logger.finish()


if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s ')
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', default='config/icdar2017.yml')
    parser.add_argument('--only_test', default=False, action='store_true',
                        help='only test')
    parser.add_argument('--checkpoint', default=None, type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--gpuid', default='0', type=str,
                        help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--seed', default=2174, type=int,
                        help='seed')

    args = parser.parse_args()
        
    config = load_config(args)[0]

    GPU.set(args.gpuid, 400)
    cudnn.benchmark = True
    
    seed_everything(args.seed)
    
    main(config)