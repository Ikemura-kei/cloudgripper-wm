"""Figures for Experiment 5 (dynamics)."""

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
TRAINED = ['red1', 'red2', 'red3', 'green1', 'green2', 'green3']
LAB = {m: m.replace('red', 'red·').replace('green', 'green·') for m in TRAINED}


def main():
    d = json.load(open(OUT / 'step5_results.json'))
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.7))

    # --- panel 1: position survives the encoder but not the prediction ----
    ax = axes[0]
    y = np.arange(len(TRAINED))
    truth = [d['position_readout'][m]['truth_px'] for m in TRAINED]
    pred = [d['position_readout'][m]['pred_px'] for m in TRAINED]
    cols = [S.RED] * 3 + [S.GREEN] * 3
    ax.barh(y - 0.19, truth, 0.36, color=cols, label='from the real next embedding')
    ax.barh(y + 0.19, pred, 0.36, color=cols, alpha=0.42,
            label='from the predicted one')
    for i, (a, b) in enumerate(zip(truth, pred)):
        ax.text(a + 1.5, i - 0.19, f'{a:.0f}', va='center', fontsize=7.5, color=S.INK)
        ax.text(b + 1.5, i + 0.19, f'{b:.0f}', va='center', fontsize=7.5, color=S.INK)
    ax.axvline(d['step_px'], color=S.INK, ls='--', lw=1.3)
    ax.annotate('one step of\nthe ball (10 px)', xy=(13, -0.62), fontsize=7.5,
                color=S.INK2, annotation_clip=False)
    ax.set_yticks(y); ax.set_yticklabels([LAB[m] for m in TRAINED], fontsize=8.5)
    ax.invert_yaxis(); ax.set_xlim(0, 92); ax.set_ylim(6.9, -1.1)
    ax.set_xlabel('position readout error (px)')
    ax.set_title('Where the ball is survives the encoder\nbut not the prediction',
                 loc='left', fontsize=9.5)
    ax.legend(loc='lower center', fontsize=7.5, ncol=1, bbox_to_anchor=(0.5, -0.02))
    S.despine(ax)

    # --- panel 2: predictor is accurate overall ---------------------------
    ax = axes[1]
    rel = [100 * d['predictor_accuracy'][m]['rel_err'] for m in TRAINED]
    ax.bar(range(6), rel, 0.6, color=cols)
    for i, v in enumerate(rel):
        ax.text(i, v + 0.15, f'{v:.1f}%', ha='center', fontsize=8,
                fontweight='bold', color=S.INK)
    ax.set_xticks(range(6))
    ax.set_xticklabels([LAB[m] for m in TRAINED], rotation=40, ha='right', fontsize=8)
    ax.set_ylabel('predictor error, % of embedding size')
    ax.set_ylim(0, 11)
    ax.set_title('The predictor itself is accurate,\nand equally so for both colours',
                 loc='left', fontsize=9.5)
    S.despine(ax)

    # --- panel 3: the stitch carries predictions ---------------------------
    ax = axes[2]
    s = d['stitch_dynamics']['summary']
    names = ['same colour\ndifferent seed', 'different\ncolour']
    keys = ['same colour, different seed', 'different colour']
    w = 0.33
    for k, (nm, key) in enumerate(zip(names, keys)):
        ax.bar(k - w / 2, s[key]['static'], w, color=S.NEUTRAL,
               label='static frames' if k == 0 else None)
        ax.bar(k + w / 2, s[key]['pred'], w, color=S.GREEN,
               label='predictions' if k == 0 else None)
        ax.text(k - w / 2, s[key]['static'] + 0.004, f'{s[key]["static"]:.4f}',
                ha='center', fontsize=7.5, fontweight='bold', color=S.INK)
        ax.text(k + w / 2, s[key]['pred'] + 0.004, f'{s[key]["pred"]:.4f}',
                ha='center', fontsize=7.5, fontweight='bold', color=S.INK)
    ax.set_xticks(range(2)); ax.set_xticklabels(names, fontsize=8.5)
    ax.set_ylim(0.99, 1.006)
    ax.set_ylabel('R² of the mapped embedding')
    ax.legend(loc='lower left', fontsize=7.5)
    ax.set_title('A map fitted on static frames\nalso carries the predictions',
                 loc='left', fontsize=9.5)
    S.despine(ax)

    fig.suptitle('Experiment 5 — the two models predict motion the same way, '
                 'including where they fail',
                 fontsize=11, fontweight='bold', color=S.INK, x=0.01, ha='left')
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    fig.savefig(OUT / 'fig1_dynamics.png'); plt.close(fig)
    print('->', OUT)


if __name__ == '__main__':
    main()
