"""Decode the colour-corrected embeddings back to pixels.

colour_maps.py says in numbers whether ball position survives the map. This
says it as a picture, which is the part a reader can check by eye.

One decoder is trained, on red embeddings only:

    D : z_red -> the red frame of that window

It never sees a recoloured embedding. Then for each colour we feed it

    D(z_c)                 the recoloured embedding untouched
    D(predicted map(z_c))  after a map built without that colour's data
    D(fitted map(z_c))     after that colour's own fitted map (the ceiling)

against D(z_red) and the true frame. The predicted map is the leave-one-colour-
out version from colour_maps.py: the map is built from the eight other colours,
so the column is an honest test of a colour the fit never saw.

Four colours are shown, chosen to span the squash: yellow needs 1.6x
amplification, cyan 34.6x.

Usage:
    uv run python scripts/analysis/colour_decode.py [--stage cls]
"""

import argparse
import io
import os
import sys

import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import RidgeCV

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import colour_common as C  # noqa: E402
import colour_style as S  # noqa: E402
import rolling_common as R  # noqa: E402
from flat2d_decoder import IMG, apply_decoder, psnr, train_decoder  # noqa: E402

S.use_style()
import matplotlib.pyplot as plt  # noqa: E402

ALPHAS = np.logspace(-5, 2, 29)
SHOW = ['yellow', 'green', 'blue', 'cyan']   # spans gain 1.6x .. 34.6x


def load_frames(colour, window_idx, clips):
    """Last frame of each window, float RGB in [0,1] at IMG resolution.

    The last context frame is at start + HISTORY*FRAMESKIP, not start+HISTORY —
    frameskip is 5 here. Getting that wrong silently decodes a different moment,
    so the lookup raises rather than falling back to something plausible.
    """
    import lance
    tab = lance.dataset(str(C.DATA / f'{C.dataset_name(colour)}.lance')).to_table(
        columns=['episode_idx', 'step_idx', 'pixels']).to_pandas()
    key = {(int(e), int(s)): i for i, (e, s) in
           enumerate(zip(tab.episode_idx, tab.step_idx))}
    out = np.empty((len(window_idx), IMG, IMG, 3), np.float32)
    for n, w in enumerate(window_idx):
        ep, start = clips[int(w)]
        k = (int(ep), int(start) + R.HISTORY * R.FRAMESKIP)
        if k not in key:
            raise KeyError(f'window {w}: no frame at {k}')
        im = Image.open(io.BytesIO(tab.pixels.iloc[key[k]])).convert('RGB')
        out[n] = np.asarray(im.resize((IMG, IMG), Image.BILINEAR), np.float32) / 255.0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stats', default='imagenet',
                    choices=['imagenet', 'symmetric'],
                    help='which version of the study to operate on')
    ap.add_argument('--stage', default='cls')
    ap.add_argument('--epochs', type=int, default=400)
    args = ap.parse_args()
    C.use_stats(args.stats)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'

    ds = C.build_dataset(C.REFERENCE, norms=C.training_norms())
    fit_w, test_w = C.map_split(ds)
    order = np.concatenate([fit_w, test_w])
    sp = np.load(C.CACHE / 'split.npz')
    fit, test = sp['fit'], sp['test']

    Z = {c: np.load(C.CACHE / f'{c}.npz')[args.stage][:, R.HISTORY].astype(np.float64)
         for c in C.COLOURS}
    ref = Z[C.REFERENCE]
    dim = ref.shape[1]
    eye = np.eye(dim)
    others = [c for c in C.COLOURS if c != C.REFERENCE]

    print('loading red target frames...', flush=True)
    frames = load_frames(C.REFERENCE, order, ds.clip_indices)
    print(f'training decoder on {len(fit)} red embeddings...', flush=True)
    dec, mu, sd = train_decoder(ref[fit], frames[fit], dev, epochs=args.epochs)

    maps = {}
    for c in others:
        m = RidgeCV(alphas=ALPHAS).fit(Z[c][fit], ref[fit])
        maps[c] = (m.coef_.copy(), m.intercept_.copy())

    def loo_map(colour):
        rest = [o for o in others if o != colour]
        D = np.stack([C.delta(o) for o in rest])
        P = np.linalg.pinv(D)
        A = np.tensordot(P, np.stack([maps[o][0] - eye for o in rest]), axes=(1, 0))
        Bv = P @ np.stack([maps[o][1] for o in rest])
        d = C.delta(colour)
        return eye + np.tensordot(d, A, axes=(0, 0)), d @ Bv

    apply_ = lambda Mb, z: z @ Mb[0].T + Mb[1]
    dec_of = lambda z: apply_decoder(dec, mu, sd, z, dev)

    ref_psnr = psnr(dec_of(ref[test]), frames[test])
    print(f'\nDecoder PSNR against the true red frame (held-out windows, dB)')
    print(f'  red embedding (reference) {ref_psnr:.2f}')
    print(f'{"colour":<9} {"no map":>9} {"predicted":>11} {"fitted":>9}')
    for c in others:
        z = Z[c][test]
        print(f'{c:<9} {psnr(dec_of(z), frames[test]):>9.2f} '
              f'{psnr(dec_of(apply_(loo_map(c), z)), frames[test]):>11.2f} '
              f'{psnr(dec_of(apply_(maps[c], z)), frames[test]):>9.2f}')

    # --- figure: two windows per colour, spread across the ball's travel -----
    pos = np.load(C.CACHE / f'{C.REFERENCE}.npz')['pos'][:, R.TRAVEL_AXIS]
    cand = test[np.argsort(pos[test])]
    use = cand[(np.array([0.2, 0.75]) * (len(cand) - 1)).astype(int)]

    cols = ['true frame', 'D(z_red)', 'no map', 'predicted map', 'fitted map']
    fig, axes = plt.subplots(len(SHOW) * 2, len(cols),
                             figsize=(1.5 * len(cols), 1.5 * len(SHOW) * 2))
    for bi, colour in enumerate(SHOW):
        z = Z[colour]
        panels = {
            'D(z_red)': dec_of(ref[use]),
            'no map': dec_of(z[use]),
            'predicted map': dec_of(apply_(loo_map(colour), z[use])),
            'fitted map': dec_of(apply_(maps[colour], z[use])),
        }
        for r, w in enumerate(use):
            row = axes[bi * 2 + r]
            for ci, cname in enumerate(cols):
                ax = row[ci]
                img = frames[w] if cname == 'true frame' else panels[cname][r]
                ax.imshow(np.clip(img, 0, 1))
                ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
                for spn in ax.spines.values():
                    spn.set_color(S.AXIS); spn.set_linewidth(0.6)
                if bi == 0 and r == 0:
                    ax.set_title(cname, fontsize=8, color=S.INK2, pad=5)
            if r == 0:
                row[0].set_ylabel(f'{colour}\nball', fontsize=8, color=S.INK2,
                                  rotation=0, ha='right', va='center', labelpad=20)
    fig.suptitle('A recoloured embedding, decoded through the red model’s decoder',
                 fontsize=10.5, fontweight='bold', y=0.998)
    fig.tight_layout(rect=(0, 0, 1, 0.982))
    fig.savefig(C.OUT / f'fig5_decoded_{args.stage}.png')
    plt.close(fig)
    print(f'\nwrote {C.OUT / f"fig5_decoded_{args.stage}.png"}')


if __name__ == '__main__':
    main()
