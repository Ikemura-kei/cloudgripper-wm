"""Figures for Experiment 2 (the similarity scale)."""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import flat2d_style as S  # noqa: E402
from flat2d_common import REPO  # noqa: E402

S.use_style()
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

OUT = REPO / 'outputs' / 'flat2d_step2'
MODELS = ['red1', 'red2', 'red3', 'green1', 'green2', 'green3', 'untrained']
LABEL = {'red1': 'red·1', 'red2': 'red·2', 'red3': 'red·3',
         'green1': 'green·1', 'green2': 'green·2', 'green3': 'green·3',
         'untrained': 'untrained'}
SEQ = LinearSegmentedColormap.from_list(
    'blue_seq', ['#fbfaf8', '#cde2fb', '#86b6ef', '#2a78d6', '#184f95', '#0d366b'])


def fig_matrix():
    M = np.load(OUT / 'cka_matrix_cls.npy')
    fig, ax = plt.subplots(figsize=(6.6, 5.4))
    im = ax.imshow(M, cmap=SEQ, vmin=0, vmax=1)
    ax.set_xticks(range(7)); ax.set_yticks(range(7))
    ax.set_xticklabels([LABEL[m] for m in MODELS], rotation=40, ha='right', fontsize=8.5)
    ax.set_yticklabels([LABEL[m] for m in MODELS], fontsize=8.5)
    for i in range(7):
        for j in range(7):
            ax.text(j, i, f'{M[i, j]:.3f}', ha='center', va='center', fontsize=7.6,
                    color='white' if M[i, j] > 0.55 else S.INK,
                    fontweight='bold' if i != j else 'normal')
    for k, col in ((2.5, S.INK), (5.5, S.INK)):
        ax.axhline(k, color=col, lw=1.4); ax.axvline(k, color=col, lw=1.4)
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label('linear CKA', fontsize=9); cb.outline.set_visible(False)
    ax.set_title('Every trained model agrees with every other, '
                 'whatever colour it saw', loc='left', fontsize=10.5, pad=12)
    fig.tight_layout()
    fig.savefig(OUT / 'fig1_cka_matrix.png'); plt.close(fig)


def fig_scale():
    M = np.load(OUT / 'cka_matrix_cls.npy')
    reds, greens, un = [0, 1, 2], [3, 4, 5], 6
    import itertools
    groups = [
        ('same colour\ndifferent init', [M[i, j] for i, j in
         list(itertools.combinations(reds, 2)) + list(itertools.combinations(greens, 2))], S.NEUTRAL),
        ('different colour\nsame init', [M[i, j] for i, j in zip(reds, greens)], S.GREEN),
        ('different colour\ndifferent init', [M[i, j] for i in reds for j in greens
                                              if (i, j) not in list(zip(reds, greens))], S.RED),
        ('trained vs\nuntrained', [M[i, un] for i in reds + greens], S.INK2),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.9),
                             gridspec_kw={'width_ratios': [1.25, 1]})

    ax = axes[0]
    for k, (name, vals, col) in enumerate(groups):
        ax.scatter([k] * len(vals), vals, s=46, color=col, zorder=3,
                   marker='o' if k < 3 else 'D', linewidths=0)
        ax.annotate(f'{np.mean(vals):.3f}', xy=(k, np.mean(vals) - 0.055),
                    fontsize=9, fontweight='bold', color=S.INK, ha='center')
    ax.set_xticks(range(4))
    ax.set_xticklabels([g[0] for g in groups], fontsize=8.5)
    ax.set_ylabel('linear CKA')
    ax.set_ylim(0.28, 1.08)
    ax.set_xlim(-0.6, 3.6)
    ax.set_title('The scale, and where each kind of pair sits on it',
                 loc='left', fontsize=10.5)
    S.despine(ax)

    # The three trained groups are indistinguishable at full scale — which is
    # the result — so a zoom is needed to show they genuinely overlap rather
    # than merely being drawn on top of each other.
    inset = ax.inset_axes([0.17, 0.17, 0.55, 0.42])
    for k, (name, vals, col) in enumerate(groups[:3]):
        inset.scatter([k] * len(vals), vals, s=34, color=col, zorder=3, linewidths=0)
        inset.plot([k - 0.26, k + 0.26], [np.mean(vals)] * 2, color=col, lw=2, zorder=4)
    lo = min(min(g[1]) for g in groups[:3]); hi = max(max(g[1]) for g in groups[:3])
    pad = (hi - lo) * 0.25
    inset.set_ylim(lo - pad, hi + pad)
    inset.set_xlim(-0.6, 2.6)
    inset.set_xticks(range(3)); inset.set_xticklabels(['same\ncolour', 'same\ninit', 'neither'],
                                                      fontsize=7)
    inset.tick_params(labelsize=7, left=False, labelleft=False,
                      right=True, labelright=True)
    inset.set_title('zoom: the three trained groups', fontsize=7.5, loc='left', pad=4)
    for sp in inset.spines.values():
        sp.set_color(S.AXIS)
    inset.grid(True, color=S.GRID, lw=0.5)

    ax = axes[1]
    probe = json.load(open(OUT / 'self_probe.json'))
    names = MODELS
    vals = [probe[m]['mae_px'] for m in names]
    cols = [S.RED] * 3 + [S.GREEN] * 3 + [S.INK2]
    bars = ax.barh(range(len(names)), vals, color=cols, height=0.62)
    for b, v in zip(bars, vals):
        ax.text(v + 0.7, b.get_y() + b.get_height() / 2, f'{v:.1f}', va='center',
                fontsize=8.5, fontweight='bold', color=S.INK)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([LABEL[m] for m in names], fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlabel('position readout error (px, 448-wide frame)')
    ax.set_xlim(0, 42)
    ax.axvline(np.mean(vals[:6]), color=S.AXIS, ls='--', lw=1.1)
    ax.annotate('trained mean 16.5 px', xy=(17.3, 5.6), fontsize=8, color=S.INK2)
    ax.set_title('Training buys 2.1× over random features', loc='left', fontsize=10.5)
    S.despine(ax)

    fig.suptitle('Experiment 2 — a measured scale for "similar"',
                 fontsize=11, fontweight='bold', color=S.INK, x=0.01, ha='left')
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(OUT / 'fig2_scale.png'); plt.close(fig)


if __name__ == '__main__':
    fig_matrix()
    fig_scale()
    print('->', OUT)
