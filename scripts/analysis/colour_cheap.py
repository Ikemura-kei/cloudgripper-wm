"""How cheap is it to adapt the red model to a new ball colour?

colour_maps.py establishes that a fitted map works and a predicted one does
not. That leaves the practical question: fitting needs data in the new colour,
so how much, and does it have to be paired?

Three tiers are measured, in increasing order of what they ask for:

    nothing      one map fitted on the other eight colours, applied blind
    unpaired     CORAL — match only the mean and covariance of the new
                 colour's embeddings to red's. Needs frames in the new colour
                 but no correspondence between them and red frames.
    paired       the fitted map, from N matched windows. Swept over N.

"Paired" means window i in the new colour is the same scene as window i in
red. In this environment that is free, because a dataset can simply be
re-rendered in another colour. Anywhere that re-rendering is impossible, the
unpaired number is the one that applies.

Usage:
    uv run python scripts/analysis/colour_cheap.py [--stage cls]
"""

import argparse
import json
import os
import sys

import numpy as np
import scipy.linalg as sla
from sklearn.linear_model import Ridge, RidgeCV

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import colour_common as C  # noqa: E402
import colour_style as S  # noqa: E402
import rolling_common as R  # noqa: E402

S.use_style()
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter  # noqa: E402

ALPHAS = np.logspace(-5, 2, 29)
SIZES = [25, 50, 100, 250, 500, 1000, 2000, 4614]
WINDOWS_PER_EPISODE = 36.1          # 7321 windows / 203 episodes


def coral(src, dst):
    """Align src onto dst using only their mean and covariance.

    The classic unpaired domain-adaptation baseline: whiten the source, then
    colour it with the target's covariance. No correspondence between rows is
    used anywhere, so this is what you can do when re-rendering the same scene
    in a new colour is impossible.
    """
    ms, md = src.mean(0), dst.mean(0)
    n = src.shape[1]
    Cs = np.cov(src, rowvar=False) + 1e-6 * np.eye(n)
    Cd = np.cov(dst, rowvar=False) + 1e-6 * np.eye(n)
    W = sla.sqrtm(np.linalg.inv(Cs)).real @ sla.sqrtm(Cd).real
    return lambda z: (z - ms) @ W + md


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stats', default='imagenet',
                    choices=['imagenet', 'symmetric'],
                    help='which version of the study to operate on')
    ap.add_argument('--stage', default='cls')
    args = ap.parse_args()
    C.use_stats(args.stats)

    sp = np.load(C.CACHE / 'split.npz')
    fit, test = sp['fit'], sp['test']
    Z = {c: np.load(C.CACHE / f'{c}.npz')[args.stage][:, R.HISTORY].astype(np.float64)
         for c in C.COLOURS}
    ref = Z[C.REFERENCE]
    pos = np.load(C.CACHE / f'{C.REFERENCE}.npz')['pos'].astype(np.float64)
    others = [c for c in C.COLOURS if c != C.REFERENCE]

    probe = RidgeCV(alphas=ALPHAS).fit(ref[fit], pos[fit])
    px = lambda z: float(np.abs(probe.predict(z) - pos[test])[:, R.TRAVEL_AXIS].mean() * R.PX)
    base = px(ref[test])
    print(f'red floor {base:.2f} px\n')

    # --- how much paired data? ----------------------------------------------
    curve = {c: [] for c in others}
    print('Paired data needed (ball position error, px)')
    print(f'{"windows":>8}{"episodes":>10} ' + ''.join(f'{c[:7]:>9}' for c in others))
    for n in SIZES:
        idx = fit[:n]
        row = []
        for c in others:
            m = Ridge(alpha=1e-3).fit(Z[c][idx], ref[idx])
            v = px(m.predict(Z[c][test]))
            curve[c].append(v)
            row.append(v)
        print(f'{n:>8}{n / WINDOWS_PER_EPISODE:>10.0f} ' + ''.join(f'{v:>9.1f}' for v in row))

    # --- unpaired, and no data at all ---------------------------------------
    print(f'\n{"colour":<9}{"no map":>10}{"shared map":>12}{"CORAL":>9}{"fitted":>9}')
    tiers = {}
    for c in others:
        rest = [o for o in others if o != c]
        sm = RidgeCV(alphas=ALPHAS).fit(
            np.concatenate([Z[o][fit] for o in rest]), np.tile(ref[fit], (len(rest), 1)))
        f = coral(Z[c][fit], ref[fit])
        m = RidgeCV(alphas=ALPHAS).fit(Z[c][fit], ref[fit])
        tiers[c] = dict(nomap=px(Z[c][test]), shared=px(sm.predict(Z[c][test])),
                        coral=px(f(Z[c][test])), fitted=px(m.predict(Z[c][test])))
        t = tiers[c]
        print(f'{c:<9}{t["nomap"]:>10.1f}{t["shared"]:>12.1f}{t["coral"]:>9.1f}{t["fitted"]:>9.1f}')

    # --- figure ---------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.7),
                             gridspec_kw={'width_ratios': [1.15, 1]})
    ax = axes[0]
    eps = [n / WINDOWS_PER_EPISODE for n in SIZES]
    # Nine end-labels collide into an unreadable stack, and each line is already
    # drawn in its own ball colour. Label only the slowest and fastest.
    ends = sorted(others, key=lambda c: curve[c][-1])
    for c in others:
        ax.plot(eps, curve[c], marker='o', ms=3.5, lw=1.4,
                color=tuple(v / 255 for v in C.COLOURS[c]),
                markeredgecolor=S.INK2, markeredgewidth=0.4, zorder=3)
    for c, dy in ((ends[-1], 5), (ends[0], -5)):
        ax.annotate(c, (eps[-1], curve[c][-1]), textcoords='offset points',
                    xytext=(7, dy), va='center', fontsize=7, color=S.INK2)
    ax.text(0.97, 0.95, 'one line per colour', fontsize=7, color=S.MUTED,
            ha='right', va='top', transform=ax.transAxes)
    ax.axhline(base, color=S.FLOOR, lw=1.3, ls=':', zorder=4)
    ax.text(eps[0], base * 0.86, f'red floor {base:.1f} px', fontsize=7.5,
            color=S.FLOOR, va='top')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlim(0.5, 220)
    for a, loc in ((ax.xaxis, [1, 3, 10, 30, 100]), (ax.yaxis, [5, 10, 20, 40])):
        a.set_major_locator(FixedLocator(loc))
        a.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}'))
    ax.set_xlabel('paired episodes in the new colour')
    ax.set_ylabel('ball position error (px)')
    ax.set_title('A handful of episodes is enough', loc='left')
    S.despine(ax)

    ax = axes[1]
    labels = ['no map', 'one shared map\n(no new data)', 'CORAL\n(unpaired data)',
              'fitted map\n(~14 paired episodes)']
    keys = ['nomap', 'shared', 'coral', 'fitted']
    vals = [[tiers[c][k] for c in others] for k in keys]
    at14 = [curve[c][SIZES.index(500)] for c in others]
    vals[3] = at14
    x = np.arange(len(keys))
    for i, v in enumerate(vals):
        ax.scatter([x[i]] * len(v), v, s=46,
                   color=[tuple(q / 255 for q in C.COLOURS[c]) for c in others],
                   edgecolor=S.INK2, linewidth=0.5, zorder=3)
        ax.plot([x[i] - 0.28, x[i] + 0.28], [np.median(v)] * 2, color=S.INK,
                lw=1.6, zorder=4)
    ax.axhline(base, color=S.FLOOR, lw=1.3, ls=':', zorder=2)
    ax.set_yscale('log')
    ax.yaxis.set_major_locator(FixedLocator([10, 100, 1000]))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}'))
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylabel('ball position error (px)')
    ax.set_title('Only paired data reaches the floor', loc='left')
    ax.text(0.97, 0.95, 'one dot per colour; bar = median', fontsize=6.8,
            color=S.MUTED, ha='right', va='top', transform=ax.transAxes)
    S.despine(ax)

    fig.tight_layout(); fig.savefig(C.OUT / 'fig6_cheap.png'); plt.close(fig)
    (C.CACHE / f'cheap_{args.stage}.json').write_text(json.dumps(
        dict(base_px=base, sizes=SIZES, curve=curve, tiers=tiers), indent=2))
    print(f'\nwrote {C.OUT / "fig6_cheap.png"}')


if __name__ == '__main__':
    main()
