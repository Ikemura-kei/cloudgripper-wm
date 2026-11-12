"""Figure: pixel decoders applied within and across input colour."""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import flat2d_style as S  # noqa: E402
from flat2d_common import REPO  # noqa: E402

S.use_style()
import matplotlib.pyplot as plt  # noqa: E402

OUT = REPO / 'outputs' / 'flat2d_step1'


def main(n=5):
    z = np.load(OUT / 'decoder_recon.npz')
    sc = json.load(open(OUT / 'decoder_scores.json'))
    # Span the travel range rather than sampling at random, so the matched
    # decoder's tracking is visible and the transferred decoder's failure to
    # move is obvious against it.
    x = np.load(REPO / 'outputs' / 'flat2d_cache' / 'mred1_val_red.npz')['x']
    order = np.argsort(x)
    pick = order[np.linspace(0, len(order) - 1, n).astype(int)]

    rows = [
        ('Ground truth (red)',   z['tgt_red'],      None),
        ('$D_R(z^R)$  matched',  z['red_red'],      'D_red(z_red)'),
        ('$D_R(z^G)$  transferred', z['red_green'], 'D_red(z_green)'),
        ('Ground truth (green)', z['tgt_green'],    None),
        ('$D_G(z^G)$  matched',  z['green_green'],  'D_green(z_green)'),
    ]
    fig, axes = plt.subplots(len(rows), n, figsize=(1.72 * n + 2.3, 1.72 * len(rows)))
    for r, (label, imgs, key) in enumerate(rows):
        for c, p in enumerate(pick):
            ax = axes[r, c]
            ax.imshow(np.clip(imgs[p], 0, 1))
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            for sp in ax.spines.values():
                sp.set_color(S.RULE if hasattr(S, 'RULE') else '#c3c2b7')
                sp.set_linewidth(0.6)
            if c == 0:
                col = S.INK2 if key is None else (
                    S.RED if 'D_R' in label else S.GREEN)
                ax.set_ylabel(label, fontsize=9, color=col,
                              fontweight='600' if key else '400')
        if key:
            axes[r, n - 1].annotate(
                f'{sc[key]["psnr_db"]:.1f} dB', xy=(1.06, 0.5),
                xycoords='axes fraction', va='center', fontsize=10,
                fontweight='bold',
                color=S.INK if sc[key]['psnr_db'] > 30 else S.NEUTRAL)
    fig.suptitle('Decoding the frozen representation back to pixels — '
                 'matched decoder vs transferred decoder',
                 fontsize=11, fontweight='bold', color=S.INK, x=0.01, ha='left')
    fig.tight_layout(rect=[0, 0, 0.95, 0.955])
    fig.savefig(OUT / 'fig5_decoder.png')
    plt.close(fig)
    print('->', OUT / 'fig5_decoder.png')


if __name__ == '__main__':
    main()
