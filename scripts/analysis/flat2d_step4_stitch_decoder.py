"""Experiment 4, visual version: render one model's state with another's decoder.

The numbers say a single linear map makes any trained model's state readable
by any other trained model's position readout. This does the same thing with
the pixel decoder, so the claim can be looked at instead of scored.

We train one decoder, D_red1, which turns red·1's embeddings into red frames.
Then we fit an affine map T from another model's embedding space into red·1's,
and render D_red1(T(z)). If the two models hold the same thing, the picture
should show the right scene with the ball in the right place -- drawn in red,
because that is the only thing D_red1 knows how to draw.

Usage:
    uv run python scripts/analysis/flat2d_step4_stitch_decoder.py
"""

import json
import os
import sys

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from flat2d_common import HISTORY, REPO, build_dataset, val_window_indices  # noqa: E402
from flat2d_decoder import Decoder, load_targets, train_decoder, apply_decoder, psnr, IMG  # noqa: E402

CACHE = REPO / 'outputs' / 'flat2d_cache'
OUT = REPO / 'outputs' / 'flat2d_step4'
REP = 'cls'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    ds, _ = build_dataset('flat2d_ball_red')
    clips = ds.clip_indices
    val_idx = val_window_indices(ds)
    fit_idx = np.sort(np.random.default_rng(0).choice(
        np.setdiff1d(np.arange(len(ds)), val_idx), size=4000, replace=False))

    models = ['red1', 'red2', 'green1']
    E = {m: {'f': np.load(CACHE / f's2_{m}_fit.npz')[REP][:, HISTORY],
             'v': np.load(CACHE / f's2_{m}_val.npz')[REP][:, HISTORY]} for m in models}

    print('loading target frames...', flush=True)
    tgt_fit = load_targets('red', fit_idx, clips)
    tgt_val = load_targets('red', val_idx, clips)

    print('training D_red1 (red·1 embeddings -> red frames)', flush=True)
    dec, mu, sd = train_decoder(E['red1']['f'], tgt_fit, dev)

    def stitch(src):
        """Affine map from src model's space into red·1's, fitted on the fit split."""
        sc = StandardScaler().fit(E[src]['f'])
        reg = Ridge(alpha=1.0).fit(sc.transform(E[src]['f']), E['red1']['f'])
        return reg.predict(sc.transform(E[src]['v']))

    panels, scores = {}, {}
    panels['own'] = apply_decoder(dec, mu, sd, E['red1']['v'], dev)
    panels['stitch_red2'] = apply_decoder(dec, mu, sd, stitch('red2'), dev)
    panels['stitch_green1'] = apply_decoder(dec, mu, sd, stitch('green1'), dev)
    panels['raw_green1'] = apply_decoder(dec, mu, sd, E['green1']['v'], dev)
    for k, v in panels.items():
        scores[k] = psnr(v, tgt_val)
        print(f'  {k:<16} PSNR {scores[k]:.2f} dB')

    np.savez_compressed(OUT / 'stitch_recon.npz', tgt=tgt_val, val_idx=val_idx, **panels)
    json.dump(scores, open(OUT / 'stitch_decoder_scores.json', 'w'), indent=2)
    print('saved ->', OUT)


if __name__ == '__main__':
    main()
