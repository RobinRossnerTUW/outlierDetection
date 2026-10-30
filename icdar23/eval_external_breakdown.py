"""
eval_external_breakdown.py - per-group breakdown of an eval_external_test.py run (CPU only, no model needed).

Reads page_feats.npz (PCA-ed page descriptors, pages sorted by writer then page) and the dataset's labels.csv
(columns writer, page, hand, ms_id) and writes to --out_dir:

  per_writer.csv    mAP / top-1 per writer, full gallery, label = writer
  per_group.csv     mAP / top-1 per manuscript+hand group, full gallery, label = writer; plus where the
                    nearest neighbour lands. Rows 'ALL (label=writer)' and 'ALL (label=manuscript)' are overall.
  cross_ms.csv      writers with several manuscripts: queries from one manuscript, that manuscript removed
                    from the gallery (can the writer be found in another manuscript?)
  within_ms.csv     manuscripts with several writers: gallery restricted to that manuscript
                    (can writers be told apart under identical imaging?)
"""

import os, csv, argparse

import numpy as np
from sklearn.metrics import pairwise_distances


def average_precision(dist_row, i, relevant, gallery=None):
    """AP of query i; relevant / gallery are boolean masks over all pages, the query itself is dropped."""
    idx = np.argsort(dist_row)
    idx = idx[idx != i]
    if gallery is not None:
        idx = idx[gallery[idx]]
    rel = relevant[idx]
    if rel.sum() == 0:
        return np.nan, np.nan
    ap = (np.cumsum(rel)[rel] / (np.flatnonzero(rel) + 1)).mean()
    return ap, float(rel[0])


def evaluate(D, queries, relevant_of, gallery=None):
    res = np.array([average_precision(D[i], i, relevant_of(i), gallery) for i in queries])
    return float(np.nanmean(res[:, 0])), float(np.nanmean(res[:, 1]))


def write_csv(path, rows):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f'Wrote {path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--feats', required=True, help='page_feats.npz written by eval_external_test.py')
    parser.add_argument('--labels', required=True, help='labels.csv of the dataset (writer, page, hand, ms_id)')
    parser.add_argument('--out_dir', default=None, help='default: directory of --feats')
    args = parser.parse_args()
    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.feats))
    os.makedirs(out_dir, exist_ok=True)

    d = np.load(args.feats)
    X, writer = d['pca'], d['writer']
    # encode_per_class returns pages sorted by (writer, page)
    rows = sorted(csv.DictReader(open(args.labels, encoding='utf-8')),
                  key=lambda r: (int(r['writer']), int(r['page'])))
    assert [int(r['writer']) for r in rows] == writer.tolist(), 'labels.csv does not match page_feats.npz'
    ms = np.array([r['ms_id'] for r in rows])
    hand = np.array([r['hand'] for r in rows])
    group = np.array([f'{m}/{h}' for m, h in zip(ms, hand)])

    D = pairwise_distances(X, metric='euclidean')
    everyone = np.arange(len(writer))
    same_writer = lambda i: writer == writer[i]
    nn = np.argmin(D + np.diag(np.full(len(D), np.inf)), axis=1)

    # ── per manuscript+hand group, full gallery ──────────────────────────────
    per_group = []
    for g in sorted(set(group)):
        q = np.flatnonzero(group == g)
        m_ap, top1 = evaluate(D, q, same_writer)
        nn_groups = {k: int((group[nn[q]] == k).sum()) for k in sorted(set(group[nn[q]]))}
        per_group.append({'group': g, 'writer': int(writer[q[0]]), 'pages': len(q), 'map': m_ap, 'top1': top1,
                          'nearest_neighbour_groups': '; '.join(f'{k}: {v}' for k, v in nn_groups.items())})
    for name, rel in (('ALL (label=writer)', same_writer), ('ALL (label=manuscript)', lambda i: ms == ms[i])):
        m_ap, top1 = evaluate(D, everyone, rel)
        per_group.append({'group': name, 'writer': '', 'pages': len(writer), 'map': m_ap, 'top1': top1,
                          'nearest_neighbour_groups': ''})
    write_csv(os.path.join(out_dir, 'per_group.csv'), per_group)

    # ── per writer, full gallery ─────────────────────────────────────────────
    per_writer = []
    for w in sorted(set(writer)):
        q = np.flatnonzero(writer == w)
        m_ap, top1 = evaluate(D, q, same_writer)
        per_writer.append({'writer': int(w), 'hand': hand[q[0]], 'manuscripts': '; '.join(sorted(set(ms[q]))),
                           'pages': len(q), 'map': m_ap, 'top1': top1})
    write_csv(os.path.join(out_dir, 'per_writer.csv'), per_writer)

    # ── same writer across manuscripts: query manuscript removed from gallery ─
    cross = []
    for w in sorted(set(writer)):
        for m in sorted(set(ms[writer == w])):
            if len(set(ms[writer == w])) < 2:
                continue
            q = np.flatnonzero((writer == w) & (ms == m))
            gallery = ms != m
            m_ap, top1 = evaluate(D, q, same_writer, gallery)
            cross.append({'writer': int(w), 'hand': hand[q[0]], 'query_ms': m, 'pages': len(q),
                          'map': m_ap, 'top1': top1, 'chance_top1': float((writer[gallery] == w).mean())})
    if cross:
        write_csv(os.path.join(out_dir, 'cross_ms.csv'), cross)

    # ── several writers in one manuscript: gallery restricted to it ──────────
    within = []
    for m in sorted(set(ms)):
        if len(set(writer[ms == m])) < 2:
            continue
        gallery = ms == m
        for w in sorted(set(writer[gallery])) + [None]:
            q = np.flatnonzero(gallery & (writer == w)) if w is not None else np.flatnonzero(gallery)
            m_ap, top1 = evaluate(D, q, same_writer, gallery)
            within.append({'ms_id': m, 'writer': 'ALL' if w is None else int(w),
                           'hand': 'ALL' if w is None else hand[q[0]], 'pages': len(q), 'map': m_ap, 'top1': top1})
    if within:
        write_csv(os.path.join(out_dir, 'within_ms.csv'), within)

    for r in per_group:
        print(f"{r['group']:28s} pages {r['pages']:4d}  mAP {r['map']:.4f}  top-1 {r['top1']:.4f}")


if __name__ == '__main__':
    main()
