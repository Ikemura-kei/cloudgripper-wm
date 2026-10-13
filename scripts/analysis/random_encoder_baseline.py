"""Control 1: does an UNTRAINED network locate the ball just as well?

A red ball on a plain floor is an easy visual feature, and random networks are
known to support surprisingly good probes. If a randomly initialised ViT-tiny
reaches the same sub-millimetre position accuracy, then our encoder result says
nothing about what training achieved. Same architecture, same config, same
probes, only the weights differ.
"""
import argparse
import copy
import os
import sys

os.environ.setdefault('MUJOCO_GL', 'egl')
import numpy as np
import torch
from transformers import ViTModel

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from probe_ball_latent import build_dataset, SETS

import stable_worldmodel as swm


@torch.no_grad()
def encode(encoder, projector, ds, n, history, batch, device, seed=0):
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(ds), size=min(n, len(ds)), replace=False))
    cls_, proj_, pos_ = [], [], []
    for lo in range(0, len(idx), batch):
        rows = [ds[int(i)] for i in idx[lo:lo + batch]]
        pix = torch.stack([r['pixels'] for r in rows]).to(device)
        b, t = pix.shape[:2]
        flat = pix.reshape(b * t, *pix.shape[2:]).to(
            next(encoder.parameters()).dtype)
        c = encoder(flat, interpolate_pos_encoding=True).last_hidden_state[:, 0]
        p = projector(c)
        cls_.append(c.reshape(b, t, -1).float().cpu().numpy())
        proj_.append(p.reshape(b, t, -1).float().cpu().numpy())
        pos_.append(np.stack([np.asarray(r['ball_pos']) for r in rows]))
    return np.concatenate(cls_), np.concatenate(proj_), np.concatenate(pos_)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', default='datasets/checkpoints/'
                    'lewm_ball-26-10-08-23-51-41-rzr/weights_epoch_100.pt')
    ap.add_argument('--windows-per-set', type=int, default=4000)
    ap.add_argument('--history', type=int, default=6)
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--img-size', type=int, default=336)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default='outputs/ball_probe/random_cache.npz')
    args = ap.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'

    trained = swm.wm.utils.load_pretrained(os.path.abspath(args.checkpoint))
    torch.manual_seed(args.seed)
    # same architecture and config, fresh random weights
    rand_enc = ViTModel(trained.encoder.config).to(dev).eval()
    rand_proj = copy.deepcopy(trained.projector).to(dev).eval()
    for mod in rand_proj.modules():                 # re-randomise the projector too
        if hasattr(mod, 'reset_parameters'):
            mod.reset_parameters()
    rand_enc.requires_grad_(False), rand_proj.requires_grad_(False)

    out, norms = {}, None
    for label, name in SETS.items():
        ds, norms = build_dataset(name, os.getcwd(), args.history,
                                  args.img_size, norms)
        c, p, pos = encode(rand_enc, rand_proj, ds, args.windows_per_set,
                           args.history, args.batch, dev)
        out[f'{label}|cls'], out[f'{label}|proj'], out[f'{label}|pos'] = c, p, pos
        print(f'{label:9s} {len(pos):5d} windows', flush=True)
    np.savez_compressed(args.out, **out)
    print('saved', args.out)


if __name__ == '__main__':
    main()
