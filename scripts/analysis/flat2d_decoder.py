"""Train pixel decoders from a frozen LeWM representation, per input colour.

A probe says whether a quantity is readable from the representation. A decoder
says what the representation still contains overall, in a form a reader can
look at. Both are needed here: the probe result (position recoverable from
either colour, but not with the other colour's readout) is a pair of numbers,
and the decoder turns the same claim into a picture.

Two decoders are trained, one per input colour, on frozen embeddings:

    D_red   : z_red   -> red frame
    D_green : z_green -> green frame

Applying D_red to z_green is then the visual analogue of the cross-colour probe
transfer, and the matched pair D_green(z_green) is the visual analogue of the
green-fitted probe.

Decoders are fitted on windows from the TRAINING episodes and every image shown
or scored comes from the VALIDATION episodes, matching the probe protocol so
the two results are directly comparable.

Usage:
    uv run python scripts/analysis/flat2d_decoder.py
"""

import io
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from flat2d_common import HISTORY, REPO, build_dataset, val_window_indices  # noqa: E402

OUT = REPO / 'outputs' / 'flat2d_step1'
CACHE = REPO / 'outputs' / 'flat2d_cache'
IMG = 112           # decode resolution; the ball is ~8 px across here
REP = 'cls'         # backbone representation


class Decoder(nn.Module):
    """192-d embedding -> IMG x IMG RGB, via a small transposed-conv stack."""

    def __init__(self, dim: int = 192, ch: int = 128):
        super().__init__()
        self.fc = nn.Linear(dim, ch * 7 * 7)
        self.ch = ch
        blocks = []
        c = ch
        for _ in range(4):                       # 7 -> 14 -> 28 -> 56 -> 112
            blocks += [nn.ConvTranspose2d(c, c // 2, 4, stride=2, padding=1),
                       nn.GroupNorm(8, c // 2), nn.SiLU()]
            c //= 2
        self.up = nn.Sequential(*blocks)
        self.out = nn.Conv2d(c, 3, 3, padding=1)

    def forward(self, z):
        h = self.fc(z).view(-1, self.ch, 7, 7)
        return torch.sigmoid(self.out(self.up(h)))


def load_targets(colour: str, window_idx: np.ndarray, clips) -> np.ndarray:
    """Last frame of each window, as float RGB in [0,1] at IMG resolution."""
    import lance
    tab = lance.dataset(str(REPO / '.stable_worldmodel' / 'datasets' /
                            f'flat2d_ball_{colour}.lance')).to_table(
        columns=['episode_idx', 'step_idx', 'pixels']).to_pandas()
    key = {(int(e), int(s)): i for i, (e, s) in
           enumerate(zip(tab.episode_idx, tab.step_idx))}
    out = np.empty((len(window_idx), IMG, IMG, 3), np.float32)
    for n, w in enumerate(window_idx):
        ep, start = clips[int(w)]
        blob = tab.pixels.iloc[key[(int(ep), int(start) + HISTORY)]]
        im = Image.open(io.BytesIO(blob)).convert('RGB').resize(
            (IMG, IMG), Image.BILINEAR)
        out[n] = np.asarray(im, np.float32) / 255.0
    return out


def train_decoder(z, y, dev, epochs=400, bs=128, seed=0):
    torch.manual_seed(seed)
    dec = Decoder(z.shape[1]).to(dev)
    opt = torch.optim.AdamW(dec.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    # standardise the embedding; the two colours live at very different scales
    mu, sd = z.mean(0, keepdims=True), z.std(0, keepdims=True) + 1e-6
    zt = torch.tensor((z - mu) / sd, dtype=torch.float32, device=dev)
    yt = torch.tensor(y.transpose(0, 3, 1, 2), dtype=torch.float32, device=dev)
    n = len(zt)
    for ep in range(epochs):
        perm = torch.randperm(n, device=dev)
        tot = 0.0
        for lo in range(0, n, bs):
            idx = perm[lo:lo + bs]
            loss = ((dec(zt[idx]) - yt[idx]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss) * len(idx)
        sched.step()
        if (ep + 1) % 100 == 0:
            print(f'    epoch {ep+1:3d}  mse {tot/n:.5f}', flush=True)
    return dec, mu, sd


@torch.no_grad()
def apply_decoder(dec, mu, sd, z, dev, bs=256):
    zt = torch.tensor((z - mu) / sd, dtype=torch.float32, device=dev)
    outs = [dec(zt[i:i + bs]).cpu().numpy() for i in range(0, len(zt), bs)]
    return np.concatenate(outs).transpose(0, 2, 3, 1)


def psnr(a, b):
    mse = ((a - b) ** 2).mean(axis=(1, 2, 3))
    return float(np.mean(10 * np.log10(1.0 / np.maximum(mse, 1e-12))))


def main():
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    ds, _ = build_dataset('flat2d_ball_red')
    clips = ds.clip_indices
    val_idx = val_window_indices(ds)
    tr_idx = np.sort(np.random.default_rng(0).choice(
        np.setdiff1d(np.arange(len(ds)), val_idx), size=4000, replace=False))

    emb = {(sp, c): np.load(CACHE / f'mred1_{sp}_{c}.npz')[REP][:, HISTORY]
           for sp in ('train', 'val') for c in ('red', 'green')}
    print('loading target frames...', flush=True)
    tgt = {('train', c): load_targets(c, tr_idx, clips) for c in ('red', 'green')}
    tgt.update({('val', c): load_targets(c, val_idx, clips) for c in ('red', 'green')})

    decs = {}
    for c in ('red', 'green'):
        print(f'training D_{c} on {len(tr_idx)} training-episode windows', flush=True)
        decs[c] = train_decoder(emb[('train', c)], tgt[('train', c)], dev)

    recon, scores = {}, {}
    for fit in ('red', 'green'):
        dec, mu, sd = decs[fit]
        for test in ('red', 'green'):
            r = apply_decoder(dec, mu, sd, emb[('val', test)], dev)
            recon[f'{fit}|{test}'] = r
            scores[f'D_{fit}(z_{test})'] = {
                'psnr_db': psnr(r, tgt[('val', test)]),
                'mse': float(((r - tgt[('val', test)]) ** 2).mean()),
            }
            print(f'  D_{fit}(z_{test}) -> PSNR {scores[f"D_{fit}(z_{test})"]["psnr_db"]:.2f} dB')

    np.savez_compressed(OUT / 'decoder_recon.npz',
                        val_idx=val_idx,
                        tgt_red=tgt[('val', 'red')], tgt_green=tgt[('val', 'green')],
                        **{k.replace('|', '_'): v for k, v in recon.items()})
    with open(OUT / 'decoder_scores.json', 'w') as f:
        json.dump(scores, f, indent=2)
    print('saved ->', OUT)


if __name__ == '__main__':
    main()
