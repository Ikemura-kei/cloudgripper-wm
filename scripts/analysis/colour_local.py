"""Does the colour -> map function look linear close up?

colour_directions.py showed the 29 maps are a curved 3-parameter family of RGB
plus estimation noise — nothing beyond colour is involved. That predicts the
function is smooth, and a smooth function is locally almost linear. So global
prediction could have failed purely because our colours were far apart: the
nearest pair in the main set is 78 units apart in RGB, out of a cube 441 across.

This samples a dense cloud around one interior colour and asks how the
interpolation error depends on how close the neighbours are. Points are drawn
uniformly inside a ball, so restricting to the nearest k of them gives a
smaller effective radius for free — one collection yields the whole curve.

If the error falls towards the ~7 px ceiling as the neighbourhood tightens, the
smooth-manifold account is confirmed and the practical recipe follows: to get a
map for a new colour, sample a few colours near it. If the error stays flat, the
function is not smooth in colour and the account is wrong.

The centre is interior on purpose: at least 170 units from both white and black
at every radius used, so the ball never stops being visible against the wall or
the floor.
"""

from __future__ import annotations

import numpy as np

CENTRE = np.array([128, 144, 112], dtype=float)
RADIUS = 60.0
N_LOCAL = 24
SEED = 11


def local_colours() -> dict[str, tuple[int, int, int]]:
    """The centre plus N_LOCAL points drawn uniformly inside the ball."""
    rng = np.random.default_rng(SEED)
    out = {'loc00': tuple(int(v) for v in CENTRE)}
    pts = []
    while len(pts) < N_LOCAL:
        v = rng.normal(size=3)
        v /= np.linalg.norm(v)
        r = RADIUS * rng.uniform() ** (1 / 3)          # uniform in the ball
        c = np.clip(np.round(CENTRE + v * r), 0, 255)
        if np.linalg.norm(c - 255) < 150 or np.linalg.norm(c) < 150:
            continue                                   # keep it visible
        if any(np.array_equal(c, p) for p in pts):
            continue
        pts.append(c)
    for i, c in enumerate(sorted(pts, key=lambda p: np.linalg.norm(p - CENTRE)), 1):
        out[f'loc{i:02d}'] = tuple(int(v) for v in c)
    return out


LOCAL = local_colours()

def _describe():
    d = np.array([np.linalg.norm(np.array(v, float) - CENTRE) for v in LOCAL.values()])
    P = np.array(list(LOCAL.values()), float)
    nn = [np.sort(np.linalg.norm(P - p, axis=1))[1] for p in P]
    print(f'{len(LOCAL)} colours around {tuple(int(v) for v in CENTRE)}, radius {RADIUS:.0f}')
    print(f'  distance from centre : {d.min():.0f} - {d.max():.0f}')
    print(f'  nearest-neighbour    : {min(nn):.0f} - {max(nn):.0f}  '
          f'(median {np.median(nn):.0f})')
    print(f'  for comparison, the main 30-colour set: nearest neighbour 78')
    print(f'  distance to white    : {np.linalg.norm(P - 255, axis=1).min():.0f} min')
    print(f'  distance to black    : {np.linalg.norm(P, axis=1).min():.0f} min')
    for k, v in LOCAL.items():
        print(f'    {k} {v}')



# --- the experiment ---------------------------------------------------------

def _enc(args):
    """Encode the local colours, plus the main set's reference, into a cache."""
    import colour_common as C
    import rolling_common as R
    C.use_stats(args.stats)
    cache = C.CACHE.parent / f'colour_cache_local_{args.stats}'
    cache.mkdir(parents=True, exist_ok=True)
    norms = C.training_norms()
    ds0 = C.build_dataset('loc00', norms=norms)
    fit, test = C.map_split(ds0)
    order = np.concatenate([fit, test])
    # subsample: the maps saturate well before 4614 windows, and 25 colours at
    # full length is an hour of GPU for no extra precision
    keep = np.linspace(0, len(order) - 1, args.n_win).astype(int)
    idx = order[keep]
    is_fit = keep < len(fit)
    np.savez(cache / 'split.npz', fit=np.where(is_fit)[0], test=np.where(~is_fit)[0])
    model = {'m': R.load_model(C.REF_RUN[args.stats], 'cuda')}
    print(f'{len(idx)} windows ({is_fit.sum()} fit / {(~is_fit).sum()} test), '
          f'encoder {C.REF_RUN[args.stats]}')
    ref_done = False
    for name in list(LOCAL) + ['red']:
        f = cache / f'{name}.npz'
        if f.exists():
            continue
        ds = C.build_dataset(name, norms=norms)
        out = R.extract_multi(model, ds, idx, device='cuda')['m']
        np.savez_compressed(f, cls=out['cls'], pos=out['pos'])
        print(f'  {name} encoded', end='\r', flush=True)
        ref_done = ref_done or name == 'red'
    print(f'\nwrote {cache}')
    return cache


def _analyse(args, cache):
    """Leave-one-colour-out, with the neighbourhood tightened step by step."""
    import colour_common as C
    import rolling_common as R
    from sklearn.linear_model import Ridge, RidgeCV
    ALPHAS = np.logspace(-5, 2, 29)

    sp = np.load(cache / 'split.npz')
    fit, test = sp['fit'], sp['test']
    Z = {n: np.load(cache / f'{n}.npz')['cls'][:, R.HISTORY].astype(np.float64)
         for n in list(LOCAL) + ['red']}
    pos = np.load(cache / 'red.npz')['pos'].astype(np.float64)
    ref = Z['red']                       # the map's target is still the red ball
    dim = ref.shape[1]
    eye = np.eye(dim)
    names = list(LOCAL)

    probe = RidgeCV(alphas=ALPHAS).fit(ref[fit], pos[fit])
    px = lambda z: float(np.abs(probe.predict(z) - pos[test])[:, R.TRAVEL_AXIS].mean() * R.PX)
    print(f'red floor {px(ref[test]):.2f} px\n')

    maps = {n: (lambda m: (m.coef_.copy(), m.intercept_.copy()))(
        Ridge(alpha=1e-3).fit(Z[n][fit], ref[fit])) for n in names}
    own = np.array([px(Z[n][test] @ maps[n][0].T + maps[n][1]) for n in names])
    raw = np.array([px(Z[n][test]) for n in names])
    print(f'per-colour fitted map (the ceiling): median {np.median(own):.1f} px')
    print(f'no map at all:                       median {np.median(raw):.1f} px\n')

    P = {n: np.array(LOCAL[n], float) for n in names}
    print('Leave one colour out, predicting its map from its k nearest neighbours')
    print(f'{"k":>3}{"median neighbour dist":>23}{"linear fit":>13}{"inverse-dist blend":>21}')
    rows = []
    for k in (3, 5, 8, 12, 18, 24):
        lin, idw, dists = [], [], []
        for n in names:
            rest = [o for o in names if o != n]
            d = np.array([np.linalg.norm(P[o] - P[n]) for o in rest])
            nb = np.argsort(d)[:k]
            dists.append(float(d[nb].max()))
            sel = [rest[i] for i in nb]
            X = np.stack([np.array(LOCAL[o], float) / 255 for o in sel])
            Mr = np.stack([(maps[o][0] - eye).ravel() for o in sel])
            br = np.stack([maps[o][1] for o in sel])
            z = Z[n][test]
            # a plain linear fit in RGB, using only these k neighbours
            A = np.column_stack([X, np.ones(len(X))])
            W, *_ = np.linalg.lstsq(A, np.column_stack([Mr, br]), rcond=None)
            q = np.r_[np.array(LOCAL[n], float) / 255, 1.0] @ W
            Mh, bh = q[:Mr.shape[1]].reshape(dim, dim), q[Mr.shape[1]:]
            lin.append(px(z @ (eye + Mh).T + bh))
            # and an inverse-distance blend of the same neighbours
            w = 1 / np.maximum(d[nb], 1e-9) ** 2
            w = w / w.sum()
            Mi = (w[:, None] * Mr).sum(0).reshape(dim, dim)
            bi = (w[:, None] * br).sum(0)
            idw.append(px(z @ (eye + Mi).T + bi))
        rows.append(dict(k=k, radius=float(np.median(dists)),
                         linear=float(np.median(lin)), idw=float(np.median(idw))))
        print(f'{k:>3}{np.median(dists):>22.0f}{np.median(lin):>12.1f} px'
              f'{np.median(idw):>18.1f} px')
    return dict(floor=px(ref[test]), own=float(np.median(own)),
                raw=float(np.median(raw)), rows=rows)


def _figure(res, stats):
    """Error against how far away the neighbours are."""
    import colour_common as C
    import colour_style as S
    S.use_style()
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter, NullLocator

    C.use_stats(stats)
    r = [x['radius'] for x in res['rows']]
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.plot(r, [x['linear'] for x in res['rows']], marker='o', ms=5, lw=2,
            color=S.CEILING, label='linear fit on the neighbours', zorder=4)
    ax.plot(r, [x['idw'] for x in res['rows']], marker='s', ms=5, lw=2,
            color=S.ACCENT, label='inverse-distance blend', zorder=3)
    ax.axhline(res['own'], color=S.FLOOR, lw=1.4, ls='--', zorder=2)
    ax.text(r[-1], res['own'] * 0.90, f"that colour's own fitted map: {res['own']:.1f} px",
            fontsize=7.5, color=S.FLOOR, ha='right', va='top')
    ax.axhline(res['floor'], color=S.MUTED, lw=1.1, ls=':', zorder=2)
    ax.text(r[0], res['floor'] * 1.06, f"red-ball floor: {res['floor']:.1f} px",
            fontsize=7.5, color=S.MUTED, va='bottom')
    for x in res['rows']:
        ax.annotate(f"k={x['k']}", (x['radius'], x['linear']),
                    textcoords='offset points', xytext=(0, 9), ha='center',
                    fontsize=7, color=S.INK2)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlim(25, 135)
    ax.set_ylim(5.2, 62)
    for a, loc in ((ax.xaxis, [30, 50, 80, 120]), (ax.yaxis, [6, 10, 20, 40])):
        a.set_major_locator(FixedLocator(loc))
        a.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}'))
        # a log axis draws its own minor ticks, which label themselves as
        # 4x10^1 and so on right over the ticks set above
        a.set_minor_locator(NullLocator())
        a.set_minor_formatter(NullFormatter())
    # where the main 30-colour set sits, for scale
    ax.axvline(78, color=S.RULE, lw=1, ls='-.', zorder=1)
    ax.annotate('spacing of the\nmain 30-colour set', (78, 56),
                textcoords='offset points', xytext=(6, 0), fontsize=7,
                color=S.MUTED, va='top')
    ax.set_xlabel('distance to the furthest neighbour used (RGB units)')
    ax.set_ylabel('ball position error (px)')
    ax.set_title('Predicting a colour\u2019s map from nearby colours', loc='left')
    ax.legend(loc='upper left')
    S.despine(ax)
    fig.tight_layout()
    fig.savefig(C.OUT / 'fig9_local.png')
    plt.close(fig)
    print(f"wrote {C.OUT / 'fig9_local.png'}")


def run():
    import argparse
    import json
    import colour_common as C
    ap = argparse.ArgumentParser()
    ap.add_argument('--stats', default='symmetric', choices=['imagenet', 'symmetric'])
    ap.add_argument('--n-win', type=int, default=3000)
    args = ap.parse_args()
    cache = _enc(args)
    res = _analyse(args, cache)
    (cache / 'local.json').write_text(json.dumps(res, indent=2))
    _figure(res, args.stats)
    print(f'\nwrote {cache / "local.json"}')


if __name__ == '__main__':
    import sys
    if '--describe' in sys.argv:
        _describe()
    else:
        run()
