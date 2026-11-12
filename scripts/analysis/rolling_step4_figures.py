"""Figures for the Experiment 4 re-run."""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import flat2d_style as S  # noqa: E402
import rolling_common as R  # noqa: E402

S.use_style()
import matplotlib.pyplot as plt  # noqa: E402

OUT = R.REPO / 'outputs' / 'rolling_step4'


def fig_frames(n=5):
    z = np.load(OUT / 'decode.npz')
    d = json.load(open(OUT / 'decode.json'))
    x = z['x_true']
    pick = np.argsort(x)[np.linspace(0, len(x) - 1, n).astype(int)]
    rows = [
        ('the real frame', 'tgt', None, S.INK2),
        ("red's own state", 'red_state', d['red state']['mae_px'], S.RED),
        ("green's state, mapped", 'green_state_mapped', d['green state, mapped']['mae_px'], S.GREEN),
        ("green's state, no map", 'green_state_no_map', d['green state, no map']['mae_px'], S.NEUTRAL),
    ]
    fig, axes = plt.subplots(len(rows), n, figsize=(1.75 * n + 3.0, 1.75 * len(rows)))
    for r, (lab, key, err, col) in enumerate(rows):
        for c, p in enumerate(pick):
            ax = axes[r, c]
            ax.imshow(np.clip(z[key][p], 0, 1))
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            for sp in ax.spines.values(): sp.set_color(S.AXIS); sp.set_linewidth(0.6)
            ax.axvline(x[p] * z[key].shape[2], color='#0b0b0b', lw=1.3, ls='--')
            if c == 0:
                ax.set_ylabel(lab, fontsize=8.5, color=col,
                              fontweight='600' if err is not None else '400')
        if err is not None:
            axes[r, n - 1].annotate(f'{err:.2f} px off' if err < 10 else f'{err:.0f} px off',
                                    xy=(1.06, 0.5), xycoords='axes fraction', va='center',
                                    fontsize=10, fontweight='bold',
                                    color=S.INK if err < 10 else S.NEUTRAL)
    fig.suptitle("All three rows are red's predictor and red's decoder.\n"
                 "Only the state they were given differs. Dashed line is where the ball really goes.",
                 fontsize=10.5, fontweight='bold', color=S.INK, x=0.01, ha='left')
    fig.tight_layout(rect=[0, 0, 0.93, 0.93])
    fig.savefig(OUT / 'fig1_frames.png'); plt.close(fig)


def fig_numbers():
    r = json.load(open(OUT / 'results.json'))
    names = ['red state -> red predictor', 'T(green) -> red predictor',
             'green state -> green predictor', 'green state -> red predictor, no map']
    labels = ["red's state\nred's predictor", "green's state mapped\nred's predictor",
              "green's state\ngreen's predictor", "green's state raw\nred's predictor"]
    cols = [S.RED, S.GREEN, S.GREEN, S.NEUTRAL]
    fig, ax = plt.subplots(figsize=(7.6, 3.9))
    vals = [r[n]['px'] for n in names]
    bars = ax.bar(range(4), vals, 0.58, color=cols)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.9, f'{v:.1f}', ha='center', fontsize=9.5,
                fontweight='bold', color=S.INK)
    ax.axhline(r['step_px'], color=S.INK, ls='--', lw=1.3)
    ax.annotate('assume the ball did not move (33.4 px)', xy=(3.42, 34.6),
                ha='right', fontsize=8.5, color=S.INK2)
    ax.set_xticks(range(4)); ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel('error in predicting the next ball position (px)')
    ax.set_ylim(0, 44)
    ax.set_title("Red's predictor works on green's state once the axes are lined up",
                 loc='left', fontsize=10.5)
    S.despine(ax)
    fig.tight_layout()
    fig.savefig(OUT / 'fig2_numbers.png'); plt.close(fig)


if __name__ == '__main__':
    fig_frames(); fig_numbers()
    print('->', OUT)
