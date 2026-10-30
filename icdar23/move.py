# split_cvl.py
from pathlib import Path
import os

src = Path('/data/rrossner/datasets/CVL/cvl-database-cropped-1-1')
train_dir = src.parent / 'cvl_train_raw'
test_dir  = src.parent / 'cvl_test_raw'
train_dir.mkdir(exist_ok=True)
test_dir.mkdir(exist_ok=True)

# Group files by writer ID
from collections import defaultdict
writers = defaultdict(list)
for f in src.glob('*-cropped.tif'):
    writer_id = f.name.split('-')[0]
    writers[writer_id].append(f)

for writer_id, files in writers.items():
    # Train writers have page 7 or 8; test writers don't
    pages = {f.name.split('-')[1] for f in files}
    target_dir = train_dir if ('7' in pages or '8' in pages) else test_dir
    for f in files:
        link = target_dir / f.name
        if not link.exists():
            os.symlink(f, link)

# Summary
print(f'Train writers: {len(set(f.name.split("-")[0] for f in train_dir.iterdir()))}')
print(f'Test  writers: {len(set(f.name.split("-")[0] for f in test_dir.iterdir()))}')