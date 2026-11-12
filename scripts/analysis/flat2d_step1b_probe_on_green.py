"""Step 1 follow-up: is ball position recoverable from the green representation
by a probe fitted ON GREEN?

The main Step 1 run only asked whether a probe fitted on RED transfers to green.
That is a question about shared coordinates, and it answers 'no'. It does not
answer the prior question — whether the information survives at all in some
other direction of the 192-d space.

The affine recalibration reported in Step 1 is a weak proxy for this: it may
only rescale the single 1-D direction the red probe already chose. A probe
fitted directly on green embeddings is free to pick any direction, so it is the
honest test of 'is the information there'.

Fitted on green frames from TRAINING episodes, scored on green frames from
VALIDATION episodes — the same protocol as the red probe, so the two numbers
are directly comparable.

Usage:
    uv run python scripts/analysis/flat2d_step1b_probe_on_green.py
"""

import json
import os
import sys

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from flat2d_common import (  # noqa: E402
    HISTORY, REPO, build_dataset, cached_extract, load_model, val_window_indices,
)

OUT = REPO / 'outputs' / 'flat2d_step1'


def train_pool(ds):
    val = set(val_window_indices(ds).tolist())
    return np.array([i for i in range(len(ds)) if i not in val])


def fit_score(xtr, ytr, xte, yte):
    sc = StandardScaler().fit(xtr)
    reg = RidgeCV(alphas=np.logspace(-2, 5, 25)).fit(sc.transform(xtr), ytr)
    p = reg.predict(sc.transform(xte))
    return {'r2': float(r2_score(yte, p)),
            'mae': float(np.mean(np.abs(p - yte))),
            'pearson': float(np.corrcoef(p, yte)[0, 1]),
            'pred_span': float(p.max() - p.min())}


def main():
    dev = 'cuda'
    model = load_model('red', 1, device=dev)
    ds_red, norms = build_dataset('flat2d_ball_red')
    ds_green, _ = build_dataset('flat2d_ball_green', norms=norms)

    val_idx = val_window_indices(ds_red)
    tr_idx = np.sort(np.random.default_rng(0).choice(train_pool(ds_red), size=4000,
                                                     replace=False))

    data = {
        ('train', 'red'): cached_extract('mred1_train_red', lambda: model,
                                         lambda: ds_red, tr_idx, 32, dev),
        ('train', 'green'): cached_extract('mred1_train_green', lambda: model,
                                           lambda: ds_green, tr_idx, 32, dev),
        ('val', 'red'): cached_extract('mred1_val_red', lambda: model,
                                       lambda: ds_red, val_idx, 32, dev),
        ('val', 'green'): cached_extract('mred1_val_green', lambda: model,
                                         lambda: ds_green, val_idx, 32, dev),
    }
    yte = data[('val', 'red')]['x']
    print(f'val windows {len(yte)} | probe-fit windows {len(tr_idx)} '
          f'| true span {yte.max()-yte.min():.3f} image widths\n')

    res = {}
    for rep in ('cls', 'proj'):
        row = {}
        # four fit/test combinations; the diagonal is the honest 'is it there' test
        for ftag in ('red', 'green'):
            for ttag in ('red', 'green'):
                row[f'fit_{ftag}_test_{ttag}'] = fit_score(
                    data[('train', ftag)][rep][:, HISTORY], data[('train', ftag)]['x'],
                    data[('val', ttag)][rep][:, HISTORY], yte)
        res[rep] = row
        print(f'[{rep}]')
        for k, v in row.items():
            print(f'  {k:28s} R2 {v["r2"]:>9.4f}  MAE {v["mae"]:.4f}  '
                  f'r {v["pearson"]:+.4f}  span {v["pred_span"]:.3f}')
        print()

    with open(OUT / 'step1b_probe_matrix.json', 'w') as f:
        json.dump(res, f, indent=2)
    print('saved ->', OUT / 'step1b_probe_matrix.json')


if __name__ == '__main__':
    main()
