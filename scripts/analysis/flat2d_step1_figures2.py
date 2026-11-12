"""Corrected Step 1 figures (supersedes fig2/fig3 from flat2d_step1_figures.py).

The first pass read the green representation in RED's PCA basis, where it looks
like a collapsed point, and concluded information was lost. Fitting a probe
directly on green shows that was wrong: position is recoverable at R^2 ~0.93,
and a single affine map carries green embeddings onto red ones at R^2 ~0.999.
The representation is displaced and compressed, not degenerate.

These figures show both readings side by side so the distinction is visible
rather than asserted.
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import flat2d_style as S  # noqa: E402
from flat2d_common import REPO, HISTORY  # noqa: E402

S.use_style()
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

OUT = REPO / 'outputs' / 'flat2d_step1'
CACHE = REPO / 'outputs' / 'flat2d_cache'
SEQ = LinearSegmentedColormap.from_list(
    'blue_seq', ['#b7d3f6', '#6da7ec', '#2a78d6', '#184f95', '#0d366b'])


def pca_project(fit_on, apply_to, k=2):
    mu = fit_on.mean(0)
    _, _, Vt = np.linalg.svd(fit_on - mu, full_matrices=False)
    return (apply_to - mu) @ Vt[:k].T


def fig_recoordination(rep='cls'):
    """Three views of the same embeddings, with every basis named explicitly."""
    red = np.load(CACHE / 'mred1_val_red.npz')
    grn = np.load(CACHE / 'mred1_val_green.npz')
    x = red['x']
    a, b = red[rep][:, HISTORY], grn[rep][:, HISTORY]

    def pcs(m, k=2):
        mu = m.mean(0)
        _, sv, Vt = np.linalg.svd(m - mu, full_matrices=False)
        var = sv**2 / (sv**2).sum()
        return mu, Vt[:k], var[:k]

    mu_r, V_r, var_r = pcs(a)
    mu_g, V_g, var_g = pcs(b)
    pa = (a - mu_r) @ V_r.T
    pb_r = (b - mu_r) @ V_r.T          # green read in red's components
    pb_g = (b - mu_g) @ V_g.T          # green read in its own components

    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.8))
    panels = [
        (pa, 'Red embeddings', "red's components", var_r, S.RED),
        (pb_r, 'Green embeddings', "red's components", var_r, S.GREEN),
        (pb_g, 'Green embeddings', "green's own components", var_g, S.GREEN),
    ]
    lim = np.abs(pa).max() * 1.15
    for i, (p, what, basis, var, col) in enumerate(panels):
        ax = axes[i]
        sc = ax.scatter(p[:, 0], p[:, 1], c=x, cmap=SEQ, s=7, linewidths=0)
        if i < 2:
            ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect('equal')
        else:
            ax.set_aspect('equal')
        ax.set_xlabel(f'PC1  ({var[0]*100:.0f}% of variance)')
        ax.set_ylabel(f'PC2  ({var[1]*100:.0f}%)')
        ax.set_title(what, loc='left', color=col, fontsize=10, pad=22)
        ax.annotate(f'projected onto {basis}', xy=(0.0, 1.022),
                    xycoords='axes fraction', fontsize=8.5, color=S.INK2)
        S.despine(ax)
        if i == 2:
            cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.03)
            cb.set_label('ball position $x$ (image widths)', fontsize=8)
            cb.outline.set_visible(False)
    axes[1].annotate('same axes as the left panel',
                     xy=(0.5, 0.05), xycoords='axes fraction', ha='center',
                     fontsize=8.5, color=S.INK, fontweight='bold')
    # The axis ticks (+-0.1 against +-1.5) already carry the scale difference;
    # an annotation here would sit on top of the data.

    fig.suptitle('Green embeddings occupy a small, shifted region of the red '
                 'representation — but keep the same ordered structure inside it',
                 fontsize=10.5, fontweight='bold', color=S.INK, x=0.01, ha='left')
    fig.tight_layout(rect=[0, 0.01, 0.99, 0.88])
    fig.savefig(OUT / 'fig2_collapse.png'); plt.close(fig)


def fig_probe_matrix():
    m = json.load(open(OUT / 'step1b_probe_matrix.json'))
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.5))
    combos = [('fit_red_test_red', 'fit red\ntest red'),
              ('fit_green_test_green', 'fit green\ntest green'),
              ('fit_red_test_green', 'fit red\ntest green'),
              ('fit_green_test_red', 'fit green\ntest red')]
    for ax, rep in zip(axes, ('cls', 'proj')):
        vals = [m[rep][k]['r2'] for k, _ in combos]
        labs = [l for _, l in combos]
        cols = [S.RED, S.GREEN, S.NEUTRAL, S.NEUTRAL]
        shown = [max(v, -1.0) for v in vals]   # floor the axis; annotate true value
        bars = ax.bar(range(4), shown, color=cols, width=0.6)
        for i, (bb, v) in enumerate(zip(bars, vals)):
            txt = f'{v:.3f}' if v > -1 else f'{v:,.0f}'
            ax.text(bb.get_x() + bb.get_width() / 2,
                    shown[i] + (0.04 if shown[i] >= 0 else -0.10),
                    txt, ha='center', va='bottom' if shown[i] >= 0 else 'top',
                    fontsize=8, color=S.INK, fontweight='bold')
        ax.axhline(0, color=S.AXIS, lw=1.0)
        ax.set_ylim(-1.35, 1.15)
        ax.set_xticks(range(4)); ax.set_xticklabels(labs, fontsize=8)
        ax.set_ylabel('R²  (floored at −1)')
        ax.set_title(rep, loc='left')
        ax.annotate('matched', xy=(0.5, 1.05), xycoords=('data', 'axes fraction'),
                    ha='center', fontsize=8, color=S.INK2)
        ax.annotate('transferred', xy=(2.5, 1.05), xycoords=('data', 'axes fraction'),
                    ha='center', fontsize=8, color=S.INK2)
        S.despine(ax)
    fig.suptitle('Step 1c — position is recoverable from either colour, but not with '
                 'the other colour\'s readout',
                 fontsize=10.5, fontweight='bold', color=S.INK, x=0.01, ha='left')
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    fig.savefig(OUT / 'fig3_decoded.png'); plt.close(fig)


if __name__ == '__main__':
    fig_recoordination()
    fig_probe_matrix()
    print('corrected figures ->', OUT)
