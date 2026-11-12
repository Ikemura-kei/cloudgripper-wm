"""Figures for Experiment 4 (model stitching)."""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import flat2d_style as S  # noqa: E402
from flat2d_common import REPO  # noqa: E402

S.use_style()
import matplotlib.pyplot as plt  # noqa: E402

OUT = REPO / 'outputs' / 'flat2d_step4'


def fig_numbers():
    d = json.load(open(OUT / 'stitching_cls.json'))
    s = d['summary']
    groups = [
        ('same colour\ndifferent seed', s['same_colour'], S.NEUTRAL),
        ('different\ncolour', s['different_colour'], S.GREEN),
        ('trained reads\nuntrained', s['read_untrained'], S.INK2),
        ('untrained reads\ntrained', s['untrained_reads'], S.INK2),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.8),
                             gridspec_kw={'width_ratios': [1.1, 1]})

    ax = axes[0]
    for k, (name, g, col) in enumerate(groups):
        ax.bar(k, g['stitched_mean'], 0.56, color=col,
               yerr=[[g['stitched_mean'] - g['stitched_min']],
                     [g['stitched_max'] - g['stitched_mean']]], capsize=4)
        ax.text(k, g['stitched_mean'] + 0.03, f'{g["stitched_mean"]:.3f}',
                ha='center', fontsize=9, fontweight='bold', color=S.INK)
    own = float(np.mean([v for k, v in d['own'].items() if k != 'untrained']))
    ax.axhline(own, color=S.AXIS, ls='--', lw=1.2)
    ax.annotate(f'a trained model reading its own state ({own:.3f})',
                xy=(3.45, own + 0.015), ha='right', fontsize=8, color=S.INK2)
    ax.set_xticks(range(4)); ax.set_xticklabels([g[0] for g in groups], fontsize=8.5)
    ax.set_ylabel('position readout R² after stitching')
    ax.set_ylim(0, 1.12)
    ax.set_title('After one linear map, any trained model can read any other',
                 loc='left', fontsize=10)
    S.despine(ax)

    ax = axes[1]
    names = [g[0].replace('\n', ' ') for g in groups]
    # each group's own baseline is the READER model's accuracy, which differs
    # between the two floor rows -- use the value the analysis computed
    recovered = [g[1]['recovered_mean'] for g in groups]
    before = [g[1]['direct_mean'] for g in groups]
    y = np.arange(4)
    ax.barh(y, [1] * 4, 0.44, color=S.NEUTRAL, alpha=0.22)
    ax.barh(y, recovered, 0.44, color=[g[2] for g in groups])
    for i, (a, b) in enumerate(zip(recovered, before)):
        ax.text(min(a, 1.0) + 0.02, i, f'{100*a:.0f}%', va='center',
                fontsize=9, fontweight='bold', color=S.INK)
        ax.text(0.0, i + 0.40, f'without the map: R² = {b:,.0f}', va='center',
                fontsize=7, color=S.INK2)
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.3); ax.set_xticks([0, 0.5, 1.0])
    ax.set_xticklabels(['0', '50%', '100%'])
    ax.set_xlabel('share of the reading model’s own accuracy')
    ax.set_title('How much is recovered', loc='left', fontsize=10)
    S.despine(ax)

    fig.suptitle('Experiment 4 — lining up the axes repairs the transfer',
                 fontsize=11, fontweight='bold', color=S.INK, x=0.01, ha='left')
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    fig.savefig(OUT / 'fig1_stitching.png'); plt.close(fig)


def fig_images(n=5):
    z = np.load(OUT / 'stitch_recon.npz')
    sc = json.load(open(OUT / 'stitch_decoder_scores.json'))
    x = np.load(REPO / 'outputs' / 'flat2d_cache' / 's2_red1_val.npz')['x']
    pick = np.argsort(x)[np.linspace(0, len(x) - 1, n).astype(int)]
    rows = [
        ('Ground truth', 'tgt', None, S.INK2),
        ('red·1 reads itself', 'own', 'own', S.RED),
        ('red·1 reads red·2', 'stitch_red2', 'stitch_red2', S.RED),
        ('red·1 reads green·1', 'stitch_green1', 'stitch_green1', S.GREEN),
        ('green·1 with no map', 'raw_green1', 'raw_green1', S.NEUTRAL),
    ]
    fig, axes = plt.subplots(len(rows), n, figsize=(1.72 * n + 2.6, 1.72 * len(rows)))
    for r, (label, key, skey, col) in enumerate(rows):
        for c, p in enumerate(pick):
            ax = axes[r, c]
            ax.imshow(np.clip(z[key][p], 0, 1))
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            for sp in ax.spines.values():
                sp.set_color(S.AXIS); sp.set_linewidth(0.6)
            if c == 0:
                ax.set_ylabel(label, fontsize=9, color=col,
                              fontweight='600' if skey else '400')
        if skey:
            axes[r, n - 1].annotate(f'{sc[skey]:.1f} dB', xy=(1.06, 0.5),
                                    xycoords='axes fraction', va='center',
                                    fontsize=10, fontweight='bold',
                                    color=S.INK if sc[skey] > 25 else S.NEUTRAL)
    fig.suptitle('Experiment 4 — one decoder, reading three different models',
                 fontsize=11, fontweight='bold', color=S.INK, x=0.01, ha='left')
    fig.tight_layout(rect=[0, 0, 0.95, 0.955])
    fig.savefig(OUT / 'fig2_stitch_images.png'); plt.close(fig)


if __name__ == '__main__':
    fig_numbers()
    if (OUT / 'stitch_recon.npz').exists():
        fig_images()
    print('->', OUT)
