"""
eval_mixed_hands.py - retrieval on Patrick-data pages written by two hands (CPU only, no model needed).

Reads page_feats.npz of an eval_external_test.py run on maslots_skt and labels_v2.csv (helpers/make_labels_v2.py).
Queries are the mixed pages (original_hand and foreign_hand set). Writes to --out_dir:

  q1_original.csv  Is the original writer still found? Gallery: all pages (also the ones not in the new JSON);
                   relevant: pages with the original hand. 'clean_*' columns: same metric for the single-hand
                   pages of that hand (new JSON only), as reference. nn_*: hand of the nearest neighbour.
  q2_foreign.csv   Does the foreign hand leave a trace? Gallery: pages of the original hand in the new JSON;
                   relevant: pages that also carry the foreign hand.
  per_page.csv     AP / top-1 / nearest neighbour of every query for Q1 and Q2.

Rows: one per original+foreign group, then 'ALL (per query)' (mean over queries) and 'ALL (per group)'
(mean over group means). Groups in --exclude_from_all (default: the Tibetan notes, 3 queries) only get their own row in Q2.
"""

import os, csv, argparse
from collections import Counter

import numpy as np
from sklearn.metrics import pairwise_distances

KS = range(1, 10)


def rank_metrics(D, i, relevant, gallery):
    """mAP / top-k / P@k / R@k of query i; relevant, gallery are boolean masks, the query itself is dropped."""
    idx = np.argsort(D[i])
    idx = idx[idx != i]
    idx = idx[gallery[idx]]
    rel = relevant[idx]
    n_rel = rel.sum()
    if n_rel == 0:
        return None
    res = {'map': float((np.cumsum(rel)[rel] / (np.flatnonzero(rel) + 1)).mean())}
    for k in KS:
        res[f'top-{k}'] = float(rel[:k].any())
    for k in KS:
        res[f'P@{k}'] = float(rel[:k].mean())
    for k in KS:
        res[f'R@{k}'] = float(rel[:k].sum() / n_rel)
    res['chance_top1'] = float(n_rel / len(idx))
    res['n_relevant'] = int(n_rel)
    res['gallery'] = len(idx)
    res['nn'] = int(idx[0])
    return res


def mean_of(results, keys):
    return {k: float(np.mean([r[k] for r in results])) for k in keys}


def table(per_query, groups, exclude, extra=None):
    """One row per group plus the two ALL rows; per_query maps group -> list of rank_metrics results."""
    keys = [k for k in next(iter(per_query.values()))[0] if k not in ('nn',)]
    rows = []
    for g in groups:
        row = {'group': g, 'queries': len(per_query[g])}
        row.update(mean_of(per_query[g], keys))
        if extra:
            row.update(extra(g))
        rows.append(row)
    used = [r for r in rows if r['group'] not in exclude]
    pooled = [q for g in groups if g not in exclude for q in per_query[g]]
    all_q = {'group': 'ALL (per query)', 'queries': len(pooled)}
    all_q.update(mean_of(pooled, keys))
    all_g = {'group': 'ALL (per group)', 'queries': len(pooled)}
    all_g.update({k: float(np.mean([r[k] for r in used])) for k in keys})
    return rows + [all_q, all_g]


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f'Wrote {path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--feats', required=True, help='page_feats.npz written by eval_external_test.py')
    parser.add_argument('--labels', required=True, help='labels_v2.csv written by helpers/make_labels_v2.py')
    parser.add_argument('--out_dir', default=None, help='default: <dir of --feats>/mixed_hands')
    parser.add_argument('--exclude_from_all', nargs='*', default=['pva-msB-hand-tibetan'],
                        help='foreign hands whose groups are left out of the Q2 ALL rows')
    args = parser.parse_args()
    out_dir = args.out_dir or os.path.join(os.path.dirname(os.path.abspath(args.feats)), 'mixed_hands')
    os.makedirs(out_dir, exist_ok=True)

    d = np.load(args.feats)
    X, writer = d['pca'], d['writer']
    # encode_per_class returns pages sorted by (writer, page)
    rows = sorted(csv.DictReader(open(args.labels, encoding='utf-8')),
                  key=lambda r: (int(r['writer']), int(r['page'])))
    assert [int(r['writer']) for r in rows] == writer.tolist(), 'labels does not match page_feats.npz'
    hands = [set(r['hands'].split(';')) for r in rows]
    in_new = np.array([r['in_new'] == '1' for r in rows])
    has = lambda h: np.array([h in s for s in hands])

    D = pairwise_distances(X, metric='euclidean')
    queries = [i for i, r in enumerate(rows) if r['original_hand'] and r['foreign_hand']]
    group_of = {i: f"{rows[i]['original_hand']} + {rows[i]['foreign_hand']}" for i in queries}
    groups = sorted(set(group_of.values()))
    exclude = {g for g in groups if g.split(' + ')[1] in args.exclude_from_all}
    everyone = np.ones(len(rows), bool)

    q1, q2, per_page = {g: [] for g in groups}, {g: [] for g in groups}, []
    for i in queries:
        orig, foreign = rows[i]['original_hand'], rows[i]['foreign_hand']
        r1 = rank_metrics(D, i, has(orig), everyone)
        r2 = rank_metrics(D, i, has(foreign), in_new & has(orig))
        q1[group_of[i]].append(r1)
        if r2 is not None:
            q2[group_of[i]].append(r2)
        per_page.append({'staged': rows[i]['staged'], 'source': rows[i]['source'], 'group': group_of[i],
                         'q1_ap': r1['map'], 'q1_top1': r1['top-1'], 'q1_nn': rows[r1['nn']]['hands'],
                         'q2_ap': r2['map'] if r2 else '', 'q2_top1': r2['top-1'] if r2 else '',
                         'q2_nn': rows[r2['nn']]['hands'] if r2 else ''})

    def q1_extra(g):
        """single-hand reference for the original hand + where the nearest neighbours of the mixed pages land"""
        orig = g.split(' + ')[0]
        clean = [i for i, r in enumerate(rows) if in_new[i] and r['hands'] == orig]
        res = [rank_metrics(D, i, has(orig), everyone) for i in clean]
        nn = Counter(rows[r['nn']]['hands'] for r in q1[g])
        return {'clean_queries': len(clean), 'clean_map': float(np.mean([r['map'] for r in res])),
                'clean_top-1': float(np.mean([r['top-1'] for r in res])),
                'nn_hands': '; '.join(f'{k}: {v}' for k, v in nn.most_common())}

    t1 = table(q1, groups, set(), q1_extra)
    t2 = table({g: v for g, v in q2.items() if v}, [g for g in groups if q2[g]], exclude)
    write_csv(os.path.join(out_dir, 'q1_original.csv'), t1)
    write_csv(os.path.join(out_dir, 'q2_foreign.csv'), t2)
    write_csv(os.path.join(out_dir, 'per_page.csv'), per_page)

    for name, t in (('Q1 original', t1), ('Q2 foreign', t2)):
        print(f'\n{name}')
        for r in t:
            clean = f"  clean mAP {r['clean_map']:.3f}" if 'clean_map' in r else ''
            print(f"  {r['group']:48s} q {r['queries']:3d}  mAP {r['map']:.3f}  top-1 {r['top-1']:.3f}  "
                  f"chance {r['chance_top1']:.3f}{clean}")


if __name__ == '__main__':
    main()
