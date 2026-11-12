"""Encode every model in the study once, and cache the result.

Two passes, one per colour. Within a pass the image pipeline runs once per
batch and every model sees that batch, which is what makes encoding seven
models affordable: the PNG decode, the 448->336 resize and the ImageNet
normalisation dominate the cost and are identical across models.

    red frames   <- red_init1/2/3, untrained
    green frames <- green_init1/2/3

Each model is encoded on the frames it was trained on, which is the only
comparison Experiment 1 left defensible. The untrained model goes on red
frames; it is the floor of the similarity scale, and its weights never saw
either colour.

Usage:
    uv run python scripts/analysis/flat2d_step2_extract.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from flat2d_common import (  # noqa: E402
    REPO, build_dataset, extract_multi, load_model, untrained_model,
    val_window_indices,
)

CACHE = REPO / 'outputs' / 'flat2d_cache'
N_FIT = 4000


def main():
    dev = 'cuda'
    CACHE.mkdir(parents=True, exist_ok=True)
    ds_red, norms = build_dataset('flat2d_ball_red')
    ds_green, _ = build_dataset('flat2d_ball_green', norms=norms)

    val_idx = val_window_indices(ds_red)
    fit_idx = np.sort(np.random.default_rng(0).choice(
        np.setdiff1d(np.arange(len(ds_red)), val_idx), size=N_FIT, replace=False))

    passes = [
        ('red', ds_red, {f'red{i}': lambda i=i: load_model('red', i, dev)
                         for i in (1, 2, 3)} | {'untrained': lambda: untrained_model(dev)}),
        ('green', ds_green, {f'green{i}': lambda i=i: load_model('green', i, dev)
                             for i in (1, 2, 3)}),
    ]

    for colour, ds, builders in passes:
        for split, idx in (('val', val_idx), ('fit', fit_idx)):
            todo = {n: b for n, b in builders.items()
                    if not (CACHE / f's2_{n}_{split}.npz').exists()}
            if not todo:
                print(f'{colour}/{split}: all cached, skipping', flush=True)
                continue
            print(f'{colour}/{split}: encoding {len(idx)} windows with '
                  f'{len(todo)} models -> {sorted(todo)}', flush=True)
            models = {n: b() for n, b in todo.items()}
            out = extract_multi(models, ds, idx, batch=32, device=dev)
            for n, d in out.items():
                np.savez_compressed(CACHE / f's2_{n}_{split}.npz', **d)
                print(f'  cached s2_{n}_{split}.npz', flush=True)
            del models

    print('done')


if __name__ == '__main__':
    main()
