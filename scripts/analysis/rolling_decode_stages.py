"""Decode each stage of both rolling-ball models and measure the ball directly.

A linear readout and a trained decoder can disagree a lot about how much
position a representation holds, because the readout is confined to one
direction and the decoder is not. Both are reported; this script supplies the
decoder half.

Usage:
    uv run python scripts/analysis/rolling_decode_stages.py
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


def ball_x(imgs, colour, thresh=0.25):
    """Centroid x of the ball, in image widths.

    Has to know the ball's colour: the flat2d version looks for redness
    (R - (G+B)/2), which finds nothing at all in a green frame and silently
    returns NaN for every window.
    """
    c = {'red': 0, 'green': 1}[colour]
    other = [i for i in (0, 1, 2) if i != c]
    score = imgs[..., c] - 0.5 * (imgs[..., other[0]] + imgs[..., other[1]])
    out = np.full(len(imgs), np.nan)
    xs = (np.arange(IMG) + 0.5) / IMG
    for i, r in enumerate(score):
        m = r > thresh
        if m.sum() < 3:
            continue
        w = r[m]
        out[i] = float((np.broadcast_to(xs, r.shape)[m] * w).sum() / w.sum())
    return out


OUT = R.REPO / 'outputs' / 'rolling_step5'


def load_targets(colour, window_idx, clips):
    """Last frame of each window. Note start + HISTORY*FRAMESKIP, not start+HISTORY."""
    import lance
    tab = lance.dataset(str(R.REPO / '.stable_worldmodel' / 'datasets' /
                            f'{R.DATASETS[colour]}.lance')).to_table(
        columns=['episode_idx', 'step_idx', 'pixels']).to_pandas()
    key = {(int(e), int(s)): i for i, (e, s) in enumerate(zip(tab.episode_idx, tab.step_idx))}
    out = np.empty((len(window_idx), IMG, IMG, 3), np.float32)
    for n, w in enumerate(window_idx):
        ep, start = clips[int(w)]
        im = Image.open(io.BytesIO(tab.pixels.iloc[key[(int(ep), int(start) + R.HISTORY * R.FRAMESKIP)]]))
        out[n] = np.asarray(im.convert('RGB').resize((IMG, IMG), Image.BILINEAR), np.float32) / 255.0
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    dev = 'cuda'
    ds, _ = R.build_dataset('red')
    clips = ds.clip_indices
    fit_idx, val_idx = R.split_indices(ds)
    res, imgs = {}, {}
    for colour in ('red', 'green'):
        F = np.load(R.CACHE / f'{colour}_fit.npz')
        V = np.load(R.CACHE / f'{colour}_val.npz')
        x_true = V['pos_traj'][:, R.HISTORY, R.TRAVEL_AXIS]
        print(f'loading {colour} target frames...', flush=True)
        tf = load_targets(colour, fit_idx, clips)
        tv = load_targets(colour, val_idx, clips)
        res[colour] = {}
        for name, zf, zv in (('cls', F['cls'][:, R.HISTORY], V['cls'][:, R.HISTORY]),
                             ('truth', F['proj'][:, R.HISTORY], V['proj'][:, R.HISTORY]),
                             ('pred', F['pred'], V['pred'])):
            print(f'  training decoder from {colour}/{name}...', flush=True)
            dec, mu, sd = train_decoder(zf, tf, dev)
            im = apply_decoder(dec, mu, sd, zv, dev)
            bx = ball_x(im, colour); ok = ~np.isnan(bx)
            res[colour][name] = {
                'psnr': psnr(im, tv), 'found': float(ok.mean()),
                'mae_px': float(np.abs(bx[ok] - x_true[ok]).mean() * R.PX),
                'corr': float(np.corrcoef(bx[ok], x_true[ok])[0, 1])}
            r = res[colour][name]
            assert r['found'] > 0.9, f'{colour}/{name}: ball found in only {100*r["found"]:.0f}% of frames'
            print(f'    PSNR {r["psnr"]:5.1f} dB | ball off by {r["mae_px"]:6.2f} px '
                  f'| correlation {r["corr"]:+.4f}', flush=True)
            if colour == 'red':
                imgs[name] = im
        if colour == 'red':
            imgs['tgt'] = tv; imgs['x_true'] = x_true; imgs['val_idx'] = val_idx
    np.savez_compressed(OUT / 'decode_stages.npz', **imgs)
    json.dump(res, open(OUT / 'decode_stages.json', 'w'), indent=2)
    print('\nsaved ->', OUT)


if __name__ == '__main__':
    main()
