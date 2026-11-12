"""At which layer do green and blue stop being interchangeable?

The encoder is a ViT-tiny trained from scratch on red balls only. Its first
layer reads the three input channels almost equally (32.7 / 33.6 / 33.6 % of
the weight), yet by the output a blue ball is squashed 3.2x harder than a green
one, and the same holds for all ten green/blue swap pairs in the colour set.
Somewhere in the twelve blocks that asymmetry gets built.

This walks the CLS token out through every block and measures, at each depth:

    squash     how much smaller that colour's cloud is than the red ball's,
               the layer-wise version of the gain reported elsewhere
    asymmetry  log( squash of the blue-heavy twin / squash of the green-heavy
               twin ) for each swap pair. Zero means the two are treated
               interchangeably; the pairs hold the same three numbers in
               different channels, so nothing but the encoder can separate them.

Reading the asymmetry curve against depth says where the channels stop being
equivalent, which is a sharper question than where colours in general separate.

Usage:
    uv run python scripts/analysis/colour_layers.py [--n 600]
"""

import argparse
import io
import json
import os
import sys

import lance
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import colour_common as C  # noqa: E402
import rolling_common as R  # noqa: E402

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)

# the ten green/blue twins, green-heavy first
GB_PAIRS = [('orange', 'rose'), ('yellow', 'magenta'), ('c23', 'c22'),
            ('green', 'blue'), ('c25', 'c24'), ('c21', 'c20'),
            ('c14', 'c13'), ('c17', 'c16'), ('spring', 'azure'), ('c28', 'c27')]


def window_frames(n):
    """The exact frames the rest of the study measures.

    Not raw dataset rows: a window's representation is taken at its LAST frame,
    start + HISTORY*FRAMESKIP, so sampling raw rows pulls in each episode's
    opening frames, which no window ever lands on. That alone moved blue's
    output squash from 25.5x to 15.9x, so it is not a detail.
    """
    ds = C.build_dataset(C.REFERENCE, norms=C.training_norms())
    _, test = C.map_split(ds)
    sel = test[np.linspace(0, len(test) - 1, n).astype(int)]
    return [ds.clip_indices[int(w)] for w in sel]


def load_frames(colour, clips):
    tab = lance.dataset(str(C.DATA / f'{C.dataset_name(colour)}.lance')).to_table(
        columns=['episode_idx', 'step_idx', 'pixels']).to_pandas()
    key = {(int(e), int(s)): i for i, (e, s) in
           enumerate(zip(tab.episode_idx, tab.step_idx))}
    out = []
    for ep, start in clips:
        blob = tab.pixels.iloc[key[(int(ep), int(start) + R.HISTORY * R.FRAMESKIP)]]
        im = Image.open(io.BytesIO(blob)).convert('RGB')
        out.append((np.asarray(im, np.float32) / 255.0 - MEAN) / STD)
    return torch.from_numpy(np.stack(out).transpose(0, 3, 1, 2))


@torch.no_grad()
def per_layer(model, x, device, batch=64):
    """Representation after every block, plus the final layernorm.

    Mean over the PATCH tokens, not the CLS token. At the patch-embedding stage
    the CLS token is a learned constant plus a position embedding, identical for
    every input, so its spread is zero and any ratio against red is 0/0. The
    patch tokens carry image content at every stage including that one, which is
    the whole point of starting the scan there.

    The last entry applies the encoder's final layernorm, so it matches the
    representation the rest of the study calls `cls`-stage input.
    """
    outs = None
    for lo in range(0, len(x), batch):
        b = x[lo:lo + batch].to(device).to(next(model.parameters()).dtype)
        o = model(b, interpolate_pos_encoding=True, output_hidden_states=True)
        reps = [h[:, 1:].mean(1).float().cpu().numpy() for h in o.hidden_states]
        reps.append(o.last_hidden_state[:, 0].float().cpu().numpy())   # the real output
        outs = reps if outs is None else [np.concatenate([a, c]) for a, c in zip(outs, reps)]
    return np.stack(outs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stats', default='imagenet',
                    choices=['imagenet', 'symmetric'],
                    help='which version of the study to operate on')
    ap.add_argument('--n', type=int, default=600)
    args = ap.parse_args()
    C.use_stats(args.stats)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    enc = R.load_model(C.REF_RUN[args.stats], dev).encoder

    clips = window_frames(args.n)
    print(f'{args.n} window last-frames, evenly spaced over the held-out episodes '
          f'— the same frames every other number in this study is computed on\n')

    spread = {}
    for colour in C.COLOURS:
        H = per_layer(enc, load_frames(colour, clips), dev)
        spread[colour] = np.linalg.norm(H - H.mean(1, keepdims=True), axis=2).mean(1)
        print(f'  {colour:<9} done', end='\r', flush=True)
    L = len(spread['red'])
    print(f'\n{L} stages: patch embedding + {L - 2} transformer blocks + the CLS '
          f'output used everywhere else in this study\n')

    sq = {c: spread['red'] / np.maximum(spread[c], 1e-9) for c in C.COLOURS}

    print('Squash factor by depth (how much smaller than the red-ball cloud)')
    print(f'{"stage":<8}' + ''.join(f'{c:>9}' for c in
                                    ('green', 'blue', 'yellow', 'magenta', 'orange', 'rose')))
    for l in range(L):
        tag = 'patch' if l == 0 else ('OUTPUT' if l == L - 1 else f'block{l}')
        print(f'{tag:<8}' + ''.join(f'{sq[c][l]:>9.2f}' for c in
                                    ('green', 'blue', 'yellow', 'magenta', 'orange', 'rose')))

    print('\nGreen/blue asymmetry by depth: log2(squash of blue twin / green twin)')
    print('0 means the two channels are interchangeable at that depth.')
    print(f'{"stage":<8}' + ''.join(f'{a[:3]+"/"+b[:3]:>10}' for a, b in GB_PAIRS) + f'{"median":>10}')
    asym = np.zeros((L, len(GB_PAIRS)))
    for l in range(L):
        for k, (a, b) in enumerate(GB_PAIRS):
            asym[l, k] = np.log2(sq[b][l] / sq[a][l])
        tag = 'patch' if l == 0 else ('OUTPUT' if l == L - 1 else f'block{l}')
        print(f'{tag:<8}' + ''.join(f'{asym[l, k]:>10.2f}' for k in range(len(GB_PAIRS)))
              + f'{np.median(asym[l]):>10.2f}')

    med = np.median(asym, axis=1)
    agree = (asym > 0).sum(axis=1)
    print(f'\n{"stage":<8}{"median asym":>13}{"pairs with blue costlier":>27}')
    for l in range(L):
        tag = 'patch' if l == 0 else ('OUTPUT' if l == L - 1 else f'block{l}')
        print(f'{tag:<8}{med[l]:>13.2f}{str(agree[l]) + "/10":>27}')

    (C.CACHE / 'layers.json').write_text(json.dumps(
        dict(n=args.n, layers=L, squash={c: sq[c].tolist() for c in C.COLOURS},
             pairs=GB_PAIRS, asym=asym.tolist()), indent=2))
    print(f'\nwrote {C.CACHE / "layers.json"}')


if __name__ == '__main__':
    main()
