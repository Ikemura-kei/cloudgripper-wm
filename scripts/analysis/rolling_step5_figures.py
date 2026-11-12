"""Figures for the Experiment 5 re-run."""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import flat2d_style as S  # noqa: E402
import rolling_common as R  # noqa: E402

S.use_style()
import matplotlib.pyplot as plt  # noqa: E402

OUT = R.REPO / 'outputs' / 'rolling_step5'
OLD = {'cls': 2.2, 'truth': 41.9, 'pred': 95.3, 'step': 10.2}


def fig_frames(n=5):
    z = np.load(OUT / 'decode_stages.npz')
    d = json.load(open(OUT / 'decode_stages.json'))['red']
    x = z['x_true']
    pick = np.argsort(x)[np.linspace(0, len(x) - 1, n).astype(int)]
    rows = [('the real frame', 'tgt', None),
            ('decoded from cls\n(encoder output)', 'cls', d['cls']['mae_px']),
            ('decoded from proj\n(real next embedding)', 'truth', d['truth']['mae_px']),
            ('decoded from proj\n(the prediction)', 'pred', d['pred']['mae_px'])]
    fig, axes = plt.subplots(len(rows), n, figsize=(1.75 * n + 2.6, 1.75 * len(rows)))
    for r, (lab, key, err) in enumerate(rows):
        for c, p in enumerate(pick):
            ax = axes[r, c]
            ax.imshow(np.clip(z[key][p], 0, 1))
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            for sp in ax.spines.values(): sp.set_color(S.AXIS); sp.set_linewidth(0.6)
            ax.axvline(x[p] * z[key].shape[2], color='#0b0b0b', lw=1.3, ls='--')
            if c == 0: ax.set_ylabel(lab, fontsize=8.5, color=S.INK2)
        if err is not None:
            axes[r, n - 1].annotate(f'{err:.2f} px off', xy=(1.06, 0.5),
                                    xycoords='axes fraction', va='center',
                                    fontsize=10, fontweight='bold', color=S.INK)
    fig.suptitle('Every stage now puts the ball on the line. Dashed line is the truth.',
                 fontsize=11, fontweight='bold', color=S.INK, x=0.01, ha='left')
    fig.tight_layout(rect=[0, 0, 0.94, 0.955])
    fig.savefig(OUT / 'fig1_stages.png'); plt.close(fig)


def fig_before_after():
    d = json.load(open(OUT / 'decode_stages.json'))
    res = json.load(open(OUT / 'results.json'))
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.8))

    ax = axes[0]
    stages = ['cls', 'truth', 'pred']
    labels = ['encoder\noutput', 'real next\nembedding', 'the\nprediction']
    old = [OLD[s] for s in stages]
    new = [d['red'][s]['mae_px'] for s in stages]
    w = 0.34
    k = np.arange(3)
    ax.bar(k - w/2, old, w, color=S.NEUTRAL, label='frameskip 1 (step 10.2 px)')
    ax.bar(k + w/2, new, w, color=S.RED, label='frameskip 5 (step 33.4 px)')
    for i, (a, b) in enumerate(zip(old, new)):
        ax.text(i - w/2, a * 1.12, f'{a:.1f}', ha='center', fontsize=8, fontweight='bold', color=S.INK)
        ax.text(i + w/2, b * 1.3, f'{b:.2f}', ha='center', fontsize=8, fontweight='bold', color=S.INK)
    ax.set_yscale('log'); ax.set_ylim(0.05, 400)
    ax.set_xticks(k); ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel('ball position error (px, log scale)')
    ax.legend(loc='upper left', fontsize=8)
    ax.set_title('Decoded ball position, before and after', loc='left', fontsize=10)
    S.despine(ax)

    ax = axes[1]
    probe = [res['position']['red'][s]['px'] for s in stages]
    dec = [d['red'][s]['mae_px'] for s in stages]
    ax.bar(k - w/2, probe, w, color=S.NEUTRAL, label='linear readout')
    ax.bar(k + w/2, dec, w, color=S.RED, label='trained decoder')
    for i, (a, b) in enumerate(zip(probe, dec)):
        ax.text(i - w/2, a + 0.6, f'{a:.1f}', ha='center', fontsize=8, fontweight='bold', color=S.INK)
        ax.text(i + w/2, b + 0.6, f'{b:.2f}', ha='center', fontsize=8, fontweight='bold', color=S.INK)
    ax.axhline(res['step_px'], color=S.AXIS, ls='--', lw=1.2)
    ax.annotate('one step of the ball (33.4 px)', xy=(2.4, 34.5), ha='right',
                fontsize=8, color=S.INK2)
    ax.set_xticks(k); ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel('ball position error (px)')
    ax.set_ylim(0, 40)
    ax.legend(loc='upper left', fontsize=8)
    ax.set_title('The two readouts disagree by 135x', loc='left', fontsize=10)
    S.despine(ax)

    fig.suptitle('Experiment 5, re-run — the predictor now tracks the ball',
                 fontsize=11, fontweight='bold', color=S.INK, x=0.01, ha='left')
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    fig.savefig(OUT / 'fig2_before_after.png'); plt.close(fig)


if __name__ == '__main__':
    fig_frames(); fig_before_after()
    print('->', OUT)
