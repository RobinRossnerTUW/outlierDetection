"""
fix_patrick_json.py - clean the new MASLOTS data.json (new_data/data.json.gz) so prepare/eval scripts can use it.

  - reads the gzipped JSON list (new format: 'm-id' instead of 'm-ms-id'), adds 'm-ms-id' again
  - points every 'img' to the file that exists on disk (new JSON names .tif / .png, disk only has .jpg)
  - drops hands that are not declared as <handNote xml:id=...> in the hands XML (export junk such as a .sty path)
  - writes a plain JSON list and prints a report

Usage
─────
    python helpers/fix_patrick_json.py --root /data/rrossner/datasets/Patrick-data
"""

import os, re, gzip, json, argparse
from collections import Counter


def on_disk(root, rel):
    """Relative path of the image on disk, matched by name without extension (case-insensitive)."""
    folder, name = os.path.split(os.path.normpath(os.path.join(root, rel)))
    stem = os.path.splitext(name)[0].lower()
    match = [f for f in os.listdir(folder) if os.path.splitext(f)[0].lower() == stem]
    assert len(match) == 1, f'cannot resolve {rel}: {match}'
    return './' + os.path.relpath(os.path.join(folder, match[0]), root)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True, help='Patrick-data folder (contains data/ and new_data/)')
    parser.add_argument('--src', default='new_data/data.json.gz')
    parser.add_argument('--hands_xml', default='new_data/maslots-skt-hands.xml')
    parser.add_argument('--out', default='new_data/data_fixed.json')
    args = parser.parse_args()

    records = json.load(gzip.open(os.path.join(args.root, args.src), 'rt', encoding='utf-8'))
    known = set(re.findall(r'<handNote xml:id="([^"]+)"', open(os.path.join(args.root, args.hands_xml),
                                                                   encoding='utf-8').read()))

    dropped, moved, combos = Counter(), 0, Counter()
    for rec in records:
        rec['m-ms-id'] = rec['m-id']
        for entry in rec['m-images']:
            path = on_disk(args.root, entry['img'])
            moved += path != entry['img']
            entry['img'] = path
            for h in entry['hands']:
                if h not in known:
                    dropped[(rec['m-id'], h)] += 1
            entry['hands'] = [h for h in entry['hands'] if h in known]
            assert entry['hands'], f"{entry['img']} has no valid hand left"
            combos[(rec['m-id'], tuple(entry['hands']))] += 1

    out = os.path.join(args.root, args.out)
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f'Wrote {out}: {len(records)} manuscripts, {sum(combos.values())} pages, {moved} image paths changed')
    for (ms, h), n in dropped.items():
        print(f'  dropped unknown hand on {n} pages of {ms}: {h}')
    for (ms, hands), n in sorted(combos.items()):
        print(f'  {ms:8s} {n:4d}  {" + ".join(hands)}')
