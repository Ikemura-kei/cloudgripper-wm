"""Figures for Experiment 3 (consolidating the null)."""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import flat2d_style as S  # noqa: E402
from flat2d_common import REPO  # noqa: E402

S.use_style()
import matplotlib.pyplot as plt  # noqa: E402

OUT = REPO / 'outputs' / 'flat2d_step3'
REPS = ['cls', 'proj', 'pred']
NICE = {'cls': 'cls\n(backbone)', 'proj': 'proj\n(predictor input)', 'pred': 'pred\n(predictor output)'}
TRAINED = ['red1', 'red2', 'red3', 'green1', 'green2', 'green3']
LABEL = {'red1': 'red·1', 'red2': 'red·2', 'red3': 'red·3', 'green1': 'green·1',
         'green2': 'green·2', 'green3': 'green·3', 'untrained': 'untrained'}


def fig_cka(res):
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.9),
                             gridspec_kw={'width_ratios': [1, 1.15]})

    ax = axes[0]
    w = 0.3
    for k, rep in enumerate(REPS):
        r = res[rep]
        ax.bar(k - w / 2, r['within_mean'], w, yerr=r['within_sd'], capsize=3,
               color=S.NEUTRAL, label='same colour' if k == 0 else None)
        ax.bar(k + w / 2, r['across_mean'], w, yerr=r['across_sd'], capsize=3,
               color=S.GREEN, label='different colour' if k == 0 else None)
        ax.plot([k - 0.42, k + 0.42], [r['floor_vs_untrained']] * 2,
                color=S.INK2, ls='--', lw=1.4)
        ax.annotate(f'floor {r["floor_vs_untrained"]:.3f}',
                    xy=(k + 0.44, r['floor_vs_untrained']), ha='left', va='center',
                    fontsize=7.5, color=S.INK2)
        for off, val in ((-w / 2, r['within_mean']), (w / 2, r['across_mean'])):
            ax.annotate(f'{val:.4f}', xy=(k + off, val + 0.025), ha='center',
                        fontsize=7.4, color=S.INK, fontweight='bold', rotation=90)
    ax.set_xticks(range(3)); ax.set_xticklabels([NICE[r] for r in REPS], fontsize=8.5)
    ax.set_ylabel('linear CKA')
    ax.set_ylim(0, 1.24)
    ax.set_xlim(-0.55, 2.95)
    ax.legend(loc='lower left', ncol=1, fontsize=8)
    ax.set_title('Same-colour and cross-colour pairs agree equally,\n'
                 'in all three representations', loc='left', fontsize=10)
    S.despine(ax)

    ax = axes[1]
    for k, rep in enumerate(REPS):
        r = res[rep]
        stats = np.array(r['all_split_stats'])
        obs = abs(r['observed_stat'])
        ax.scatter(stats, [k] * len(stats), s=42, color=S.NEUTRAL,
                   linewidths=0, zorder=2,
                   label='arbitrary 3/3 split' if k == 0 else None)
        ax.scatter([obs], [k], s=110, marker='D', color=S.RED, zorder=4,
                   label='split by training colour' if k == 0 else None)
        ax.annotate(f'colour split is\n{r["rank_of_true_split"]}th of {r["n_splits"]}',
                    xy=(0.995, (k + 0.42) / 3.5), xycoords='axes fraction',
                    ha='right', va='center', fontsize=8, color=S.INK,
                    fontweight='bold')
    ax.set_yticks(range(3)); ax.set_yticklabels([r for r in REPS], fontsize=9)
    ax.set_xlabel('|mean within-group CKA − mean across-group CKA|')
    ax.set_xscale('log')
    ax.set_ylim(-0.6, 3.1)
    ax.legend(loc='upper left', fontsize=8)
    ax.set_title('Splitting the six models by colour separates them\n'
                 'less than almost any arbitrary split', loc='left', fontsize=10)
    S.despine(ax)

    fig.suptitle('Experiment 3 — the colour signature, looked for three ways',
                 fontsize=11, fontweight='bold', color=S.INK, x=0.01, ha='left')
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    fig.savefig(OUT / 'fig1_cka_reps.png'); plt.close(fig)


def fig_probes(res):
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.6))
    for ax, rep in zip(axes, ('cls', 'proj')):
        P = res['probes'][rep]
        names = TRAINED + ['untrained']
        real = [P[m]['real_out'] for m in names]
        shuf = [P[m]['shuffled_out'] for m in names]
        y = np.arange(len(names))
        cols = [S.RED] * 3 + [S.GREEN] * 3 + [S.INK2]
        ax.barh(y, real, 0.56, color=cols, label='true positions')
        ax.barh(y, shuf, 0.56, color=S.NEUTRAL, alpha=0.9,
                label='shuffled positions (control)')
        for i, v in enumerate(real):
            ax.text(v + 0.012, i, f'{v:.3f}', va='center', fontsize=8,
                    fontweight='bold', color=S.INK)
        ax.axvline(P['untrained']['real_out'], color=S.AXIS, ls='--', lw=1.2)
        ax.annotate('random-features\nbaseline', xy=(P['untrained']['real_out'] - 0.02, 2.1),
                    ha='right', fontsize=7.5, color=S.INK2)
        ax.set_yticks(y); ax.set_yticklabels([LABEL[m] for m in names], fontsize=8.5)
        ax.invert_yaxis()
        ax.set_xlim(-0.08, 1.12)
        ax.set_xlabel('probe R² on held-out episodes')
        ax.set_title(rep, loc='left', fontsize=10)
        if rep == 'cls':
            ax.legend(loc='lower right', fontsize=8)
        S.despine(ax)
    fig.suptitle('Experiment 3 — the probes read signal, not capacity, '
                 'but the untrained baseline is high',
                 fontsize=11, fontweight='bold', color=S.INK, x=0.01, ha='left')
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    fig.savefig(OUT / 'fig2_probe_control.png'); plt.close(fig)


if __name__ == '__main__':
    res = json.load(open(OUT / 'step3_results.json'))
    fig_cka(res)
    fig_probes(res)
    print('->', OUT)
