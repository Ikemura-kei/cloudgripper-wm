"""Did frameskip 5 fix the predictor? Decode each stage and measure the ball.

Repeats the Experiment 5 measurement on the rolling-ball model, which differs
from the Experiment 1-5 models in four ways: frameskip 5 instead of 1,
history 3 instead of 6, a 1.5x larger ball, and a narrower speed band.

Three decoders are trained on frozen embeddings, all on training-episode
windows and scored on held-out ones:

    from cls    the encoder's output
    from proj   the real next embedding, in the predictor's own space
    from pred   the predictor's forecast of that embedding

Then the red blob is located in each decoded image and compared with where the
ball actually is. The question is whether the third row now lands near the
truth, which it did not before.

Usage:
    uv run python scripts/analysis/rolling_decode_test.py
"""

import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import flat2d_common as C  # noqa: E402
from flat2d_decoder import train_decoder, apply_decoder, psnr, IMG  # noqa: E402
from flat2d_step5_decode_pred import ball_x  # noqa: E402

# the rolling-ball models were trained with these, not the Experiment 1-5 values
C.HISTORY = 3
C.IMG_SIZE = 224
FRAMESKIP = 5
DATASET = 'rolling_ball_red'
RUN = 'lewm_rolling_red-26-10-20-11-54-48-xsy'
EPOCH = 15
OUT = C.REPO / 'outputs' / 'rolling_check'
N_FIT = 4000


def load_targets(dataset, window_idx, clips):
    """Last frame of each window, as RGB in [0,1] at the decoder's resolution.

    Written here rather than reused from flat2d_decoder because that version
    hardcodes the Experiment 1-5 dataset, takes HISTORY from its own module
    (6, not 3), and assumes frameskip 1 -- so the last frame of a window is at
    `start + HISTORY * FRAMESKIP`, not `start + HISTORY`.
    """
    import io
    import lance
    from PIL import Image
    tab = lance.dataset(str(C.REPO / '.stable_worldmodel' / 'datasets' /
                            f'{dataset}.lance')).to_table(
        columns=['episode_idx', 'step_idx', 'pixels']).to_pandas()
    key = {(int(e), int(s)): i for i, (e, s) in
           enumerate(zip(tab.episode_idx, tab.step_idx))}
    out = np.empty((len(window_idx), IMG, IMG, 3), np.float32)
    for n, w in enumerate(window_idx):
        ep, start = clips[int(w)]
        frame = int(start) + C.HISTORY * FRAMESKIP
        im = Image.open(io.BytesIO(tab.pixels.iloc[key[(int(ep), frame)]]))
        out[n] = np.asarray(im.convert('RGB').resize((IMG, IMG), Image.BILINEAR),
                            np.float32) / 255.0
    return out


def build(name, norms=None):
    """Dataset with the rolling-ball frameskip and history."""
    import stable_worldmodel as swm, stable_pretraining as spt
    from stable_pretraining import data as dt
    from stable_worldmodel.data import column_normalizer
    ds = swm.data.load_dataset(
        str(C.REPO / '.stable_worldmodel' / 'datasets' / f'{name}.lance'),
        transform=None, num_steps=C.HISTORY + 1, frameskip=FRAMESKIP,
        keys_to_load=['pixels', 'action', 'state', 'ball_pos', 'ball_speed'])
    stats = dt.dataset_stats.ImageNet
    tfs = [dt.transforms.Compose(
        dt.transforms.ToImage(**stats, source='pixels', target='pixels'),
        dt.transforms.Resize(C.IMG_SIZE, source='pixels', target='pixels'))]
    norms = norms or {c: column_normalizer(ds, c, c) for c in ('action', 'state')}
    tfs.extend(norms.values())
    ds.transform = spt.data.transforms.Compose(*tfs)
    return ds, norms


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    import stable_worldmodel as swm
    m = swm.wm.utils.load_pretrained(
        str(C.CKPT_ROOT / RUN / f'weights_epoch_{EPOCH}.pt')).to(dev).eval()
    m.requires_grad_(False)

    ds, _ = build(DATASET)
    val_idx = C.val_window_indices(ds)
    fit_idx = np.sort(np.random.default_rng(0).choice(
        np.setdiff1d(np.arange(len(ds)), val_idx), size=N_FIT, replace=False))
    print(f'{len(val_idx)} held-out windows, {len(fit_idx)} for fitting', flush=True)

    print('encoding...', flush=True)
    F = C.extract(m, ds, fit_idx, batch=32, device=dev)
    V = C.extract(m, ds, val_idx, batch=32, device=dev)
    x_true = V['pos_traj'][:, C.HISTORY, 0]
    x_prev = V['pos_traj'][:, C.HISTORY - 1, 0]
    step_px = float(np.mean(np.abs(x_true - x_prev)) * 448)
    print(f'the ball advances {step_px:.1f} px per step at frameskip {FRAMESKIP}\n', flush=True)

    print('loading target frames...', flush=True)
    clips = ds.clip_indices
    tgt_fit = load_targets(DATASET, fit_idx, clips)
    tgt_val = load_targets(DATASET, val_idx, clips)

    sources = [('cls', F['cls'][:, C.HISTORY], V['cls'][:, C.HISTORY]),
               ('proj', F['proj'][:, C.HISTORY], V['proj'][:, C.HISTORY]),
               ('pred', F['pred'], V['pred'])]
    res, imgs = {}, {}
    for name, zf, zv in sources:
        print(f'training decoder from {name}...', flush=True)
        dec, mu, sd = train_decoder(zf, tgt_fit, dev)
        im = apply_decoder(dec, mu, sd, zv, dev)
        bx = ball_x(im); ok = ~np.isnan(bx)
        err = np.abs(bx[ok] - x_true[ok]) * 448
        res[name] = {'psnr': psnr(im, tgt_val), 'found': float(ok.mean()),
                     'mae_px': float(err.mean()), 'median_px': float(np.median(err)),
                     'corr_with_truth': float(np.corrcoef(bx[ok], x_true[ok])[0, 1])}
        imgs[name] = im
        r = res[name]
        print(f'  {name:<5} PSNR {r["psnr"]:5.1f} dB | ball off by {r["mae_px"]:6.1f} px '
              f'| correlation with the truth {r["corr_with_truth"]:+.3f}', flush=True)

    res['step_px'] = step_px
    res['_note'] = f'{RUN} epoch {EPOCH}, frameskip {FRAMESKIP}, history {C.HISTORY}'
    np.savez_compressed(OUT / 'rolling_decode.npz', tgt=tgt_val, x_true=x_true,
                        val_idx=val_idx, **imgs)
    json.dump(res, open(OUT / 'rolling_decode.json', 'w'), indent=2, default=float)
    print(f'\nfor comparison, the old frameskip-1 model gave: cls 2.2 px, '
          f'proj 41.9 px, pred 95.3 px (step was 10.2 px)')
    print('saved ->', OUT)


if __name__ == '__main__':
    main()
