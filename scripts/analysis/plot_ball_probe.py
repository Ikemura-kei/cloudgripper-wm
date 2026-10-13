"""Four panels comparing encoder and predictor readouts against ground truth."""

import argparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

AX, SETS = 1, ['train', 'held-out', 'OOD-slow', 'OOD-fast']
C = {'train': '#2a78d6', 'held-out': '#eb6834',
     'OOD-slow': '#1baf7a', 'OOD-fast': '#eda100'}
INK, INK2, GRID, SURF = '#0b0b0b', '#52514e', '#dcdcd8', '#fcfcfb'


def style(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=11, color=INK, pad=10, loc='left')
    ax.set_xlabel(xlabel, fontsize=9, color=INK2)
    ax.set_ylabel(ylabel, fontsize=9, color=INK2)
    ax.tick_params(colors=INK2, labelsize=8, length=3)
    ax.grid(True, color=GRID, lw=.7, zorder=0)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.set_facecolor(SURF)


def diagonal(ax):
    lo = min(ax.get_xlim()[0], ax.get_ylim()[0])
    hi = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([lo, hi], [lo, hi], color=INK2, ls='--', lw=1.2, zorder=1,
            label='perfect agreement')
    ax.set_xlim(lo, hi), ax.set_ylim(lo, hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', default='outputs/ball_probe/emb_cache.npz')
    ap.add_argument('--out', default='outputs/ball_probe/ball_comparison.png')
    ap.add_argument('--history', type=int, default=6)
    ap.add_argument('--points', type=int, default=900)
    args = ap.parse_args()
    d, H = np.load(args.cache), args.history

    # one probe, fitted on training frames only, pooled over all frame indices
    E, P = d['train|proj'], d['train|pos'][:, :, AX]
    x = E.reshape(-1, E.shape[-1])
    sc = StandardScaler().fit(x)
    rg = RidgeCV(alphas=np.logspace(-3, 5, 30)).fit(sc.transform(x), P.reshape(-1))
    read = lambda e: rg.predict(sc.transform(e))

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 10.5), facecolor=SURF)
    rng = np.random.default_rng(0)
    (a, b), (c, e) = axes

    for s in SETS:
        emb, pr, p = d[f'{s}|proj'], d[f'{s}|pred'], d[f'{s}|pos'][:, :, AX]
        k = rng.choice(len(p), size=min(args.points, len(p)), replace=False)
        # A: encoder reads the frame it is shown
        a.scatter(p[k, H] * 1000, read(emb[k, H]) * 1000, s=7, alpha=.45,
                  color=C[s], lw=0, label=s, zorder=3)
        # B: predictor forecasts that same frame from the 6 before it
        b.scatter(p[k, H] * 1000, read(pr[k]) * 1000, s=7, alpha=.45,
                  color=C[s], lw=0, label=s, zorder=3)
        # C: how far the ball moved in one step
        ts, ps = (p[:, H] - p[:, H - 1]) * 1000, (read(pr) - read(emb[:, H - 1])) * 1000
        c.scatter(ts[k], ps[k], s=7, alpha=.4, color=C[s], lw=0, label=s, zorder=3)
        c.plot(ts.mean(), ps.mean(), 'o', ms=11, color=C[s], mec=SURF, mew=2, zorder=5)

    style(a, 'A  Encoder: shown one image, asked where the ball is',
          'true position (mm)', 'estimated position (mm)')
    diagonal(a)
    a.legend(frameon=False, fontsize=8.5, markerscale=2.2, labelcolor=INK2, loc='upper left')

    style(b, 'B  Predictor: shown 6 images, asked where the ball goes next',
          'true position (mm)', 'predicted position (mm)')
    diagonal(b)

    style(c, 'C  One-step movement — should follow the dashed line',
          'how far the ball really moved (mm)', 'how far the model thinks it moved (mm)')
    c.set_xlim(-2, 16), c.set_ylim(-40, 40)
    c.plot([-2, 16], [-2, 16], color=INK2, ls='--', lw=1.2, zorder=4)
    c.annotate('large dots = average per dataset', xy=(.97, .05),
               xycoords='axes fraction', ha='right', fontsize=8, color=INK2)

    # D: the predictor's own loss, no probe involved
    miss = []
    for s in SETS:
        emb, pr = d[f'{s}|proj'], d[f'{s}|pred']
        miss.append(np.mean((pr - emb[:, H]) ** 2)
                    / np.mean((emb[:, H] - emb[:, H - 1]) ** 2) * 100)
    bars = e.bar(SETS, miss, color=[C[s] for s in SETS], width=.6, zorder=3)
    for bar, m in zip(bars, miss):
        e.text(bar.get_x() + bar.get_width() / 2, m * 1.35, f'{m:.2f}%',
               ha='center', fontsize=9.5, color=INK)
    style(e, 'D  Predictor error, measured without any probe',
          '', 'share of the frame-to-frame change it gets wrong')
    e.set_yscale('log'), e.set_ylim(.02, 200)
    e.set_yticks([.1, 1, 10, 100]), e.set_yticklabels(['0.1%', '1%', '10%', '100%'])

    fig.suptitle('What the world model knows about the ball, against ground truth',
                 fontsize=13.5, color=INK, y=.985)
    fig.tight_layout(rect=[0, 0, 1, .97])
    fig.savefig(args.out, dpi=150, facecolor=SURF)
    print('saved', args.out)
    for s, m in zip(SETS, miss):
        print(f'  {s:9s} misses {m:6.2f}% of the frame-to-frame change')


if __name__ == '__main__':
    main()
