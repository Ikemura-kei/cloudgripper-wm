"""Decode the predictor's output and measure where the ball actually lands.

A readout is an indirect way to ask whether a predicted embedding still knows
where the ball is: if it fails, that could be the embedding's fault or the
readout's. Decoding to pixels and measuring the ball directly removes that
ambiguity.

Train one decoder from the projector space (the space the predictor works in)
to the image. Then decode two things for every held-out window:

    D(truth)  the real next embedding   -> tells us how good the decoder is
    D(pred)   the predicted one          -> tells us how good the PREDICTOR is

Then find the ball in each decoded image and compare it with the true position.
If D(truth) puts the ball in the right place but D(pred) does not, the
predictor is not strong enough. If both are off, the decoder is the problem.

Usage:
    uv run python scripts/analysis/flat2d_step5_decode_pred.py
"""

import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from flat2d_common import HISTORY, REPO, build_dataset, val_window_indices  # noqa: E402
from flat2d_decoder import load_targets, train_decoder, apply_decoder, psnr, IMG  # noqa: E402

CACHE = REPO / 'outputs' / 'flat2d_cache'
OUT = REPO / 'outputs' / 'flat2d_step5'
MODEL = 'red1'


def ball_x(imgs, thresh=0.25):
    """Centroid x of the red ball in each decoded image, in image widths.

    The ball is the only red thing in the scene, so 'how red is this pixel'
    (R minus the average of G and B) isolates it without any model.
    """
    redness = imgs[..., 0] - 0.5 * (imgs[..., 1] + imgs[..., 2])
    out = np.full(len(imgs), np.nan)
    xs = (np.arange(IMG) + 0.5) / IMG
    for i, r in enumerate(redness):
        m = r > thresh
        if m.sum() < 3:
            continue
        w = r[m]
        out[i] = float((np.broadcast_to(xs, r.shape)[m] * w).sum() / w.sum())
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    ds, _ = build_dataset('flat2d_ball_red')
    clips = ds.clip_indices
    val_idx = val_window_indices(ds)
    fit_idx = np.sort(np.random.default_rng(0).choice(
        np.setdiff1d(np.arange(len(ds)), val_idx), size=4000, replace=False))

    f = np.load(CACHE / f's2_{MODEL}_fit.npz')
    v = np.load(CACHE / f's2_{MODEL}_val.npz')
    # the predictor's space is `proj`, so the decoder has to be trained there
    z_fit = f['proj'][:, HISTORY]
    z_truth, z_pred = v['proj'][:, HISTORY], v['pred']
    x_true = v['pos_traj'][:, HISTORY, 0]
    x_prev = v['pos_traj'][:, HISTORY - 1, 0]

    print('loading target frames...', flush=True)
    tgt_fit = load_targets('red', fit_idx, clips)
    tgt_val = load_targets('red', val_idx, clips)
    print(f'training decoder: {MODEL} proj -> frame, on {len(fit_idx)} windows', flush=True)
    dec, mu, sd = train_decoder(z_fit, tgt_fit, dev)

    img_truth = apply_decoder(dec, mu, sd, z_truth, dev)
    img_pred = apply_decoder(dec, mu, sd, z_pred, dev)

    res = {}
    print(f'\n{"decoded from":<22}{"PSNR":>8}{"ball found":>12}{"err vs true x":>16}')
    for name, img in (('the real next embedding', img_truth),
                      ('the predicted one', img_pred)):
        bx = ball_x(img)
        ok = ~np.isnan(bx)
        err_px = np.abs(bx[ok] - x_true[ok]) * 448
        res[name] = {'psnr': psnr(img, tgt_val), 'found': float(ok.mean()),
                     'mae_px': float(err_px.mean()), 'median_px': float(np.median(err_px))}
        print(f'{name:<22}{res[name]["psnr"]:>8.1f}{100*ok.mean():>11.0f}%'
              f'{err_px.mean():>13.1f} px')
    step = float(np.mean(np.abs(x_true - x_prev)) * 448)
    print(f'\nfor reference, the ball moves {step:.1f} px per step')
    res['step_px'] = step

    np.savez_compressed(OUT / 'decode_pred.npz', tgt=tgt_val, truth=img_truth,
                        pred=img_pred, x_true=x_true, x_prev=x_prev,
                        bx_truth=ball_x(img_truth), bx_pred=ball_x(img_pred),
                        val_idx=val_idx)
    json.dump(res, open(OUT / 'decode_pred_scores.json', 'w'), indent=2)
    print('saved ->', OUT)


if __name__ == '__main__':
    main()
