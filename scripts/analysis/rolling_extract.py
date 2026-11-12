"""Encode both rolling-ball models once and cache the result.

Each model is run on the colour it was trained on, which is the only
comparison Experiment 1 left defensible. The two datasets are frame-for-frame
matched, so window *i* is the same scene in both and the pairing survives.

Usage:
    uv run python scripts/analysis/rolling_extract.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rolling_common as R  # noqa: E402


def main():
    R.CACHE.mkdir(parents=True, exist_ok=True)
    dev = 'cuda'
    ds_red, norms = R.build_dataset('red')
    ds_green, _ = R.build_dataset('green', norms=norms)
    fit, val = R.split_indices(ds_red)
    print(f'{len(fit)} fit windows, {len(val)} held-out windows', flush=True)

    for colour, ds in (('red', ds_red), ('green', ds_green)):
        for split, idx in (('val', val), ('fit', fit)):
            f = R.CACHE / f'{colour}_{split}.npz'
            if f.exists():
                print(f'  {f.name} cached, skipping', flush=True)
                continue
            print(f'  encoding {colour}/{split} ({len(idx)} windows)...', flush=True)
            out = R.extract_multi({colour: R.load_model(colour, dev)}, ds, idx, device=dev)
            np.savez_compressed(f, **out[colour])
            print(f'  wrote {f.name}', flush=True)

    # the whole study rests on red window i and green window i being the same scene
    a, b = np.load(R.CACHE / 'red_val.npz'), np.load(R.CACHE / 'green_val.npz')
    for k in ('pos_traj', 'x', 'speed'):
        if not np.array_equal(a[k], b[k]):
            raise AssertionError(f'{k} differs between red and green — pairing is broken')
    print('pairing verified: ground truth identical across the two colours')


if __name__ == '__main__':
    main()
