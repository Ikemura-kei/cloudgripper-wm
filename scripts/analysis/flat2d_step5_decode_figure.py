"""Figure: where the ball ends up when you decode each stage of the model."""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import flat2d_style as S  # noqa: E402
from flat2d_common import REPO  # noqa: E402

S.use_style()
import matplotlib.pyplot as plt  # noqa: E402

OUT = REPO / 'outputs' / 'flat2d_step5'
H = 6


def main(n=5):
    d5 = np.load(OUT / 'decode_pred.npz')
    e1 = np.load(REPO / 'outputs' / 'flat2d_step1' / 'decoder_recon.npz')
    x = d5['x_true']
    pick = np.argsort(x)[np.linspace(0, len(x) - 1, n).astype(int)]

    rows = [
        ('the real frame', d5['tgt'], None),
        ('decoded from cls\n(encoder output)', e1['red_red'], 2.2),
        ('decoded from proj\n(real next embedding)', d5['truth'], 41.9),
        ('decoded from proj\n(the prediction)', d5['pred'], 95.3),
    ]
    fig, axes = plt.subplots(len(rows), n, figsize=(1.75 * n + 3.0, 1.75 * len(rows)))
    for r, (label, imgs, err) in enumerate(rows):
        for c, p in enumerate(pick):
            ax = axes[r, c]
            ax.imshow(np.clip(imgs[p], 0, 1))
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            for sp in ax.spines.values():
                sp.set_color(S.AXIS); sp.set_linewidth(0.6)
            ax.axvline(x[p] * imgs.shape[2], color='#0b0b0b', lw=1.3, ls='--')
            if c == 0:
                ax.set_ylabel(label, fontsize=8.5, color=S.INK2)
        if err is not None:
            axes[r, n - 1].annotate(f'{err:.1f} px off', xy=(1.06, 0.5),
                                    xycoords='axes fraction', va='center',
                                    fontsize=10, fontweight='bold',
                                    color=S.INK if err < 15 else S.RED)
    fig.suptitle('Where the ball lands when you decode each stage. '
                 'Dashed line is the truth.',
                 fontsize=11, fontweight='bold', color=S.INK, x=0.01, ha='left')
    fig.tight_layout(rect=[0, 0, 0.93, 0.955])
    fig.savefig(OUT / 'fig2_decode_stages.png'); plt.close(fig)
    print('->', OUT / 'fig2_decode_stages.png')


if __name__ == '__main__':
    main()
