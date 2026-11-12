"""Render red's prediction of the GREEN ball's next position.

The numbers say red's predictor, fed green's state through an affine map,
predicts the next position as well as it does on its own state. This renders
it: red's decoder draws what red's predictor produced from green's state.

The ball comes out red, because red frames are the only thing that decoder was
ever trained to draw. What matters is whether it lands where the GREEN ball
actually goes.

Usage:
    uv run python scripts/analysis/rolling_step4_decode.py
"""

import io
import json
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rolling_common as R  # noqa: E402
from flat2d_decoder import train_decoder, apply_decoder, psnr, IMG  # noqa: E402
from rolling_decode_stages import ball_x, load_targets  # noqa: E402

OUT = R.REPO / 'outputs' / 'rolling_step4'


def main():
    dev = 'cuda'
    ds, _ = R.build_dataset('red')
    clips = ds.clip_indices
    fit_idx, val_idx = R.split_indices(ds)
    F = np.load(R.CACHE / 'red_fit.npz')
    V = np.load(R.CACHE / 'red_val.npz')
    x_true = V['pos_traj'][:, R.HISTORY, R.TRAVEL_AXIS]

    print('loading red target frames...', flush=True)
    tf = load_targets('red', fit_idx, clips)
    tv = load_targets('red', val_idx, clips)
    print('training red decoder on red predictions...', flush=True)
    dec, mu, sd = train_decoder(F['pred'], tf, dev)

    preds = {
        'red state': np.load(OUT / 'pred_red_on_red.npz')['pred'],
        'green state, mapped': np.load(OUT / 'pred_red_on_mapped_green.npz')['pred'],
        'green state, no map': np.load(OUT / 'pred_red_on_raw_green.npz')['pred'],
    }
    res, imgs = {}, {'tgt': tv, 'x_true': x_true}
    for name, p in preds.items():
        im = apply_decoder(dec, mu, sd, p, dev)
        bx = ball_x(im, 'red'); ok = ~np.isnan(bx)
        res[name] = {'psnr': psnr(im, tv), 'found': float(ok.mean()),
                     'mae_px': float(np.abs(bx[ok] - x_true[ok]).mean() * R.PX) if ok.any() else float('nan'),
                     'corr': float(np.corrcoef(bx[ok], x_true[ok])[0, 1]) if ok.sum() > 2 else float('nan')}
        imgs[name.replace(' ', '_').replace(',', '')] = im
        r = res[name]
        print(f'  {name:<22} ball found {100*r["found"]:3.0f}% | off by {r["mae_px"]:6.2f} px '
              f'| correlation {r["corr"]:+.4f}')
    np.savez_compressed(OUT / 'decode.npz', **imgs)
    json.dump(res, open(OUT / 'decode.json', 'w'), indent=2)
    print('saved ->', OUT)


if __name__ == '__main__':
    main()
