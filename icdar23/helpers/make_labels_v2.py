"""
make_labels_v2.py - add the multi-hand labels of the new Patrick-data JSON to the old labels.csv.

Rows stay the old 563 staged pages (so page_feats.npz of an eval_external_test.py run still matches).
Added columns:
  in_new         1 if the page is in the new JSON (127 old psT-ms B pages are not)
  hands          ';'-joined hands (new JSON if in_new, else the old hand)
  n_hands
  original_hand  the page's hand from the old JSON, or its split in the new one (pva-msB-hand -> -1/-2/-3);
                 empty if a page has two such hands (pva-msB-hand-1 + -2)
  foreign_hand   the one hand that is neither old nor a split (proof-readers, Tibetan notes); empty otherwise

Usage
─────
    python helpers/make_labels_v2.py --root /data/rrossner/datasets/Patrick-data
"""

import os, csv, json, argparse

# hands of the old JSON that the new JSON splits into several scribes
SPLITS = {'pva-msB-hand': {'pva-msB-hand-1', 'pva-msB-hand-2', 'pva-msB-hand-3'}}


def key(root, ms, path):
    """(manuscript, image path below root without extension), lower-case."""
    rel = os.path.relpath(os.path.normpath(os.path.join(root, path)), root)
    return ms, os.path.splitext(rel)[0].lower()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True, help='Patrick-data folder')
    parser.add_argument('--new_json', default='new_data/data_fixed.json', help='output of fix_patrick_json.py')
    parser.add_argument('--out', default='labels_v2.csv')
    args = parser.parse_args()

    rows = list(csv.DictReader(open(os.path.join(args.root, 'labels.csv'), encoding='utf-8')))
    new = {key(args.root, rec['m-ms-id'], e['img']): e['hands']
           for rec in json.load(open(os.path.join(args.root, args.new_json), encoding='utf-8'))
           for e in rec['m-images']}
    old_hands = {r['hand'] for r in rows}
    known = old_hands | set().union(*SPLITS.values())

    found = set()
    for r in rows:
        k = key(args.root, r['ms_id'], r['source'])
        hands = new.get(k)
        r['in_new'] = int(hands is not None)
        if hands is None:
            hands = [r['hand']]
        else:
            found.add(k)
            assert all(h == r['hand'] or h in SPLITS.get(r['hand'], ()) or h not in known for h in hands), \
                f'{k}: old hand {r["hand"]}, new hands {hands}'
        original = [h for h in hands if h in known]
        foreign = [h for h in hands if h not in known]
        r['hands'] = ';'.join(hands)
        r['n_hands'] = len(hands)
        r['original_hand'] = original[0] if len(original) == 1 else ''
        r['foreign_hand'] = foreign[0] if len(foreign) == 1 else ''
    assert found == set(new), f'{len(set(new) - found)} pages of the new JSON are not in labels.csv'

    out = os.path.join(args.root, args.out)
    with open(out, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    n_query = sum(bool(r['original_hand'] and r['foreign_hand']) for r in rows)
    print(f'Wrote {out}: {len(rows)} pages, {len(found)} in new JSON, {n_query} mixed (original + foreign)')
