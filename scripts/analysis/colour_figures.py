"""Figures for the colour-generalisation study.

    fig1_colours.png    the ten balls, and how far the encoder squashes each
    fig2_survives.png   every colour's information survives: a fitted map puts
                        ball position back at the red model's own floor
    fig3_prediction.png the claim under test, faceted by method — no way of
                        predicting the map without that colour's data comes
                        close to fitting it
    fig4_why.png        the diagnosis: the gain spans 37x and is roughly
                        predictable, the direction of the map is not

Usage:
    uv run python scripts/analysis/colour_figures.py [--stage cls]
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import colour_common as C  # noqa: E402
import colour_style as S  # noqa: E402

S.use_style()
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter  # noqa: E402



def sw(colour):
    return tuple(v / 255 for v in C.COLOURS[colour])


def log_axis(ax):
    ax.set_yscale('log')
    ax.yaxis.set_major_locator(FixedLocator([1, 10, 100, 1000]))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}'))


def fig1(res):
    order = [r['colour'] for r in res['tableA']]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.3),
                             gridspec_kw={'width_ratios': [1.15, 1]})
    ax = axes[0]
    names = list(C.COLOURS)
    for i, n in enumerate(names):
        ax.add_patch(plt.Rectangle((i, 0), 0.84, 1, color=sw(n)))
        role = 'trained on' if n == C.REFERENCE else 'held out'
        ax.text(i + 0.42, -0.14, n, ha='center', va='top', fontsize=7.5,
                color=S.INK2, rotation=45)
        ax.text(i + 0.42, 1.10, role, ha='center', va='bottom', fontsize=6.5,
                color=S.MUTED)
    ax.set_xlim(-0.3, len(names)); ax.set_ylim(-0.95, 1.45); ax.axis('off')
    ax.set_title('Ten ball colours, one model (trained on red)', loc='left')

    ax = axes[1]
    # A scatter against RGB distance collides badly — four of the nine colours
    # sit within 0.1 of each other on x and within a factor of 1.5 on y. The
    # ordering is the message, so a sorted bar carries it without overlap.
    rows = sorted(res['tableA'], key=lambda r: r['gain'])
    y = np.arange(len(rows))
    ax.barh(y, [r['gain'] for r in rows], 0.68,
            color=[sw(r['colour']) for r in rows],
            edgecolor=S.INK2, linewidth=0.6, zorder=3)
    for i, r in enumerate(rows):
        ax.text(r['gain'] * 1.08, i, f'{r["gain"]:.1f}\u00d7', va='center',
                fontsize=7, color=S.INK2)
    ax.axvline(1.0, color=S.RULE, lw=1, ls='--', zorder=4)
    ax.set_yticks(y); ax.set_yticklabels([r['colour'] for r in rows], fontsize=7.5)
    ax.set_xscale('log')
    ax.set_xlim(0.55, 85)
    ax.xaxis.set_major_locator(FixedLocator([1, 3, 10, 30]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}\u00d7'))
    ax.set_xlabel('how much the map must amplify to undo the squash')
    ax.set_title('Other colours are squashed, by wildly different amounts', loc='left')
    S.despine(ax)
    fig.tight_layout(); fig.savefig(C.OUT / 'fig1_colours.png'); plt.close(fig)
    return order


def fig2(res, order):
    rows = {r['colour']: r for r in res['tableA']}
    x = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(10.5, 3.6))
    ax.bar(x - 0.21, [rows[c]['raw_px'] for c in order], 0.42, color=S.ACCENT,
           edgecolor=S.SURFACE, linewidth=1.0, label='no map', zorder=3)
    ax.bar(x + 0.21, [rows[c]['fitted_px'] for c in order], 0.42, color=S.CEILING,
           edgecolor=S.SURFACE, linewidth=1.0, label='after a fitted map', zorder=3)
    ax.axhline(res['base_px'], color=S.FLOOR, lw=1.4, ls=':', zorder=4)
    ax.text(len(order) - 0.4, res['base_px'] * 1.12,
            f'red ball: {res["base_px"]:.1f} px', fontsize=7.5, color=S.FLOOR,
            va='bottom', ha='right')
    for i, c in enumerate(order):
        # lift the label clear of the floor line when the bar sits below it
        ax.text(i + 0.21, max(rows[c]['fitted_px'], res['base_px']) * 1.17,
                f'{rows[c]["fitted_px"]:.1f}', ha='center', fontsize=6.8,
                color=S.INK2)
        # swatch sits *below* the tick labels: at -0.055 it lands on top of
        # them, since these labels are horizontal rather than rotated
        ax.add_patch(plt.Rectangle((i - 0.42, -0.145), 0.84, 0.035, color=sw(c),
                                   transform=ax.get_xaxis_transform(),
                                   clip_on=False, zorder=5))
    log_axis(ax)
    ax.set_ylim(top=max(r['raw_px'] for r in rows.values()) * 2.6)
    ax.set_xticks(x); ax.set_xticklabels(order)
    ax.set_ylabel('ball position error (px)')
    ax.set_title('The information survives every colour — once you fit that '
                 'colour a map', loc='left')
    ax.legend(ncol=2, loc='upper left', bbox_to_anchor=(0, 1.02))
    S.despine(ax)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(C.OUT / 'fig2_survives.png'); plt.close(fig)


def fig3(res, order):
    rows = {r['colour']: r for r in res['tableB']}
    methods = S.METHODS
    fig, axes = plt.subplots(1, len(methods), figsize=(13.5, 3.5), sharey=True)
    x = np.arange(len(order))
    for ax, m in zip(axes, methods):
        vals = [rows[c][f'{m}_px'] for c in order]
        is_ceiling = m == 'fitted'
        ax.bar(x, vals, 0.68, color=S.CEILING if is_ceiling else S.ACCENT,
               edgecolor=S.SURFACE, linewidth=0.8, zorder=3)
        ax.axhline(res['base_px'], color=S.FLOOR, lw=1.2, ls=':', zorder=4)
        ax.set_title(S.METHOD_LABEL[m], fontsize=8.5, loc='left')
        ax.set_xticks(x)
        ax.set_xticklabels(order, rotation=90, fontsize=6.5)
        for i, c in enumerate(order):
            ax.add_patch(plt.Rectangle((i - 0.34, -0.035), 0.68, 0.02, color=sw(c),
                                       transform=ax.get_xaxis_transform(),
                                       clip_on=False, zorder=5))
        log_axis(ax)
        S.despine(ax)
    axes[0].set_ylabel('ball position error (px)')
    axes[0].text(0.02, 0.96, f'red floor {res["base_px"]:.1f} px', fontsize=7,
                 color=S.FLOOR, transform=axes[0].transAxes, va='top')
    fig.suptitle('Only the last panel uses the held-out colour’s own data '
                 '— and only the last panel works',
                 fontsize=10, fontweight='bold', x=0.005, ha='left')
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(C.OUT / 'fig3_prediction.png'); plt.close(fig)


def fig4(res, order):
    rows = {r['colour']: r for r in res['tableB']}
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6),
                             gridspec_kw={'width_ratios': [1, 1.1]})

    ax = axes[0]
    y = np.arange(len(order))[::-1]
    ax.barh(y + 0.19, [rows[c]['reliab_cosM'] for c in order], 0.36,
            color=S.CEILING, label='the map agrees with itself\n(split-half)', zorder=3)
    ax.barh(y - 0.19, [rows[c]['linear_cosM'] for c in order], 0.36,
            color=S.ACCENT, label='the predicted map agrees\nwith the fitted one', zorder=3)
    ax.set_yticks(y); ax.set_yticklabels(order, fontsize=7.5)
    ax.set_xlim(0, 1); ax.set_xlabel('cosine similarity of the map matrix')
    ax.axvline(0, color=S.AXIS, lw=0.8)
    ax.set_title('The gap is not noise', loc='left')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.22), ncol=2, fontsize=7)
    S.despine(ax, keep=('left', 'bottom'))

    ax = axes[1]
    cosM = res['cosM']
    names = [c for c in res['colours'] if c != C.REFERENCE]
    M = np.array([[cosM[a][b] for b in names] for a in names])
    im = ax.imshow(M, cmap='BuPu', vmin=0, vmax=1)
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=90, fontsize=7)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=7)
    ax.grid(False)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f'{M[i, j]:.2f}', ha='center', va='center', fontsize=5.6,
                    color=S.SURFACE if M[i, j] > 0.55 else S.INK2)
    ax.set_title('Only colours close in RGB share a map', loc='left')
    fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03).set_label(
        'cosine between fitted maps', fontsize=7.5)
    fig.tight_layout(); fig.savefig(C.OUT / 'fig4_why.png'); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stats', default='imagenet',
                    choices=['imagenet', 'symmetric'],
                    help='which version of the study to operate on')
    ap.add_argument('--stage', default='cls')
    args = ap.parse_args()
    C.use_stats(args.stats)
    res = json.loads((C.CACHE / f'maps_{args.stage}.json').read_text())
    order = fig1(res)
    fig2(res, order); fig3(res, order); fig4(res, order)
    stf = C.CACHE / f'structure_{args.stage}.json'
    if stf.exists():
        fig7(json.loads(stf.read_text()))
    lyf = C.CACHE / 'layers.json'
    if lyf.exists():
        fig8(json.loads(lyf.read_text()))
    print(f'wrote figures to {C.OUT}')



def fig7(st):
    """Structure across thirty maps: low-dimensional, but colour is not the index."""
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.7),
                             gridspec_kw={'width_ratios': [1, 1.25]})

    ax = axes[0]
    for key, col, lab in (('evr_M', S.ACCENT, 'the matrix  Mᴄ − I'),
                          ('evr_b', S.CEILING, 'the shift  bᴄ')):
        ev = np.cumsum(st[key]) * 100
        k = np.arange(1, len(ev) + 1)
        ax.plot(k, ev, marker='o', ms=3.5, lw=1.7, color=col, label=lab, zorder=3)
        n90 = int(np.searchsorted(ev, 90) + 1)
        ax.scatter([n90], [ev[n90 - 1]], s=120, facecolor='none', edgecolor=col,
                   linewidth=1.6, zorder=4)
        ax.annotate(f'{n90} comps', (n90, ev[n90 - 1]), textcoords='offset points',
                    xytext=(9, -13 if key == 'evr_M' else 7), fontsize=7.5,
                    color=col)
    ax.axhline(90, color=S.RULE, lw=1, ls='--', zorder=2)
    ax.text(0.985, 90.8, '90%', fontsize=7, color=S.MUTED, ha='right',
            va='bottom', transform=ax.get_yaxis_transform())
    ax.set_xlim(0.5, 20); ax.set_ylim(0, 103)
    ax.set_xlabel('number of components')
    ax.set_ylabel('variance of the 29 maps explained (%)')
    ax.set_title('The maps span few directions', loc='left')
    ax.legend(loc='lower right')
    S.despine(ax)

    ax = axes[1]
    order = ['RGB (linear)', 'RBF kernel', '3-nearest', 'shared map',
             'true size, true shape']
    labels = ['linear\nin RGB', 'RBF kernel\non RGB', '3-nearest\nin RGB',
              'one shared map\n(ignores colour)', 'fitted map\n(ceiling)']
    for i, key in enumerate(order):
        v = np.array(list(st['results'][key].values()))
        ceiling = key == 'true size, true shape'
        ax.scatter([i] * len(v), v, s=26,
                   color=S.CEILING if ceiling else S.ACCENT,
                   edgecolor=S.SURFACE, linewidth=0.4, alpha=.85, zorder=3)
        ax.plot([i - 0.3, i + 0.3], [np.median(v)] * 2, color=S.INK, lw=1.8, zorder=4)
        ax.text(i, np.median(v) * 1.45, f'{np.median(v):.0f}', ha='center',
                fontsize=7, color=S.INK)
    ax.axhline(st['base_px'], color=S.FLOOR, lw=1.3, ls=':', zorder=2)
    ax.text(4.45, st['base_px'] * 0.63, f'red floor {st["base_px"]:.1f} px',
            fontsize=7, color=S.FLOOR, ha='right', va='top')
    ax.set_yscale('log')
    ax.yaxis.set_major_locator(FixedLocator([10, 100, 1000, 10000, 100000]))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}'))
    ax.set_xticks(range(len(order))); ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylabel('ball position error (px)')
    ax.set_title('But colour does not say which direction', loc='left')
    ax.text(0.98, 0.96, 'one dot per colour; bar = median', fontsize=6.8,
            color=S.MUTED, ha='right', va='top', transform=ax.transAxes)
    S.despine(ax)

    fig.tight_layout(); fig.savefig(C.OUT / 'fig7_structure.png'); plt.close(fig)

def fig8(ly):
    """Where in the encoder green and blue stop being interchangeable."""
    sq, asym, pairs = ly['squash'], np.array(ly['asym']), ly['pairs']
    L = ly['layers']
    x = np.arange(L)
    tick = ['patch'] + [str(i) for i in range(1, L - 1)] + ['out']
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))

    ax = axes[0]
    for c in ('orange', 'rose', 'yellow', 'magenta', 'green', 'blue'):
        ax.plot(x, sq[c], marker='o', ms=3, lw=1.6,
                color=tuple(v / 255 for v in C.COLOURS[c]),
                markeredgecolor=S.INK2, markeredgewidth=0.3, zorder=3)
        ax.annotate(c, (x[-1], sq[c][-1]), textcoords='offset points',
                    xytext=(7, 0), va='center', fontsize=7, color=S.INK2)
    ax.axhline(1.0, color=S.RULE, lw=1, ls='--', zorder=2)
    ax.text(0, 1.04, 'same room as a red ball', fontsize=6.8, color=S.MUTED, va='bottom')
    ax.set_yscale('log'); ax.set_xlim(-0.5, L + 2.2)
    ax.yaxis.set_major_locator(FixedLocator([1, 3, 10, 30]))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}\u00d7'))
    ax.set_xticks(x); ax.set_xticklabels(tick, fontsize=6.5)
    ax.set_xlabel('stage in the encoder (block number)')
    ax.set_ylabel('squash, vs the red ball')
    ax.set_title('Squashing starts around block 5', loc='left')
    S.despine(ax)

    ax = axes[1]
    for k in range(asym.shape[1]):
        ax.plot(x, asym[:, k], lw=1.1, color=S.ACCENT, alpha=.55, zorder=3)
    med = np.median(asym, axis=1)
    ax.plot(x, med, lw=2.6, color=S.CEILING, marker='o', ms=4, zorder=4, label='median of the 10 pairs')
    ax.axhline(0, color=S.AXIS, lw=1, zorder=2)
    # first depth at which every pair agrees that the blue twin costs more
    agree = (asym > 0).sum(axis=1)
    first = int(np.argmax(agree == asym.shape[1]))
    ax.axvline(first, color=S.FLOOR, lw=1.2, ls=':', zorder=2)
    ax.annotate(f'block {first}: all 10 pairs\nagree for the first time',
                (first, ax.get_ylim()[1] * 0.62), textcoords='offset points',
                xytext=(8, 0), fontsize=7, color=S.FLOOR, va='center')
    ax.set_xticks(x); ax.set_xticklabels(tick, fontsize=6.5)
    ax.set_xlabel('stage in the encoder (block number)')
    ax.set_ylabel('log\u2082( blue twin\u2019s squash / green twin\u2019s )')
    ax.set_title('Green and blue diverge from block 6 on', loc='left')
    ax.legend(loc='upper left')
    S.despine(ax)
    fig.tight_layout(); fig.savefig(C.OUT / 'fig8_layers.png'); plt.close(fig)


if __name__ == '__main__':
    main()
