"""What indexes the low-dimensional family of colour maps?

colour_structure.py found that the 29 maps span only about 7 directions out of
a possible 28, and that the colour value is not the coordinate on that surface.
So something else is. This script looks for it.

Two families of candidate are tried, and the difference between them is the
whole point:

  colour properties      RGB, saturation, lightness, hue. Known from the colour
                         alone, before any frame is rendered. These already
                         failed; they are carried along as the control.

  embedding properties   the shape of the new colour's embedding cloud — how
                         spread out it is, where its centre sits relative to
                         red's, the spectrum and axes of its covariance. These
                         need frames in the new colour, but *not* paired ones:
                         no red target is used to compute any of them.

If the embedding properties index the maps, then the map can be recovered from
unlabelled frames, which is a far weaker requirement than the paired data a
fitted map needs. That is the practically interesting outcome, and it is a real
possibility rather than a reach: CORAL already reaches 24-95 px using nothing
but the mean and covariance of that cloud.

Usage:
    uv run python scripts/analysis/colour_index.py [--stage cls]
"""

import argparse
import json
import os
import sys

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import colour_common as C  # noqa: E402
import rolling_common as R  # noqa: E402

ALPHAS = np.logspace(-5, 2, 29)
NEIG = 8


def colour_props(name):
    """Everything knowable from the RGB triple alone."""
    r, g, b = C.rgb(name) / 255.0
    mx, mn = max(r, g, b), min(r, g, b)
    sat = (mx - mn) / max(mx, 1e-9)
    light = (r + g + b) / 3
    # hue is circular, so it enters as a sine/cosine pair rather than an angle
    hue = np.arctan2(np.sqrt(3) * (g - b), 2 * r - g - b)
    return dict(R=r, G=g, B=b, saturation=sat, lightness=light, maxch=mx, minch=mn,
                hue_cos=np.cos(hue), hue_sin=np.sin(hue))


def embed_props(Zc, ref, rows):
    """Everything knowable from the new colour's own frames, unpaired.

    Nothing here touches the red embedding of the *same scene*; only red's
    overall statistics, which you already have from the model you own.
    """
    X, Y = Zc[rows], ref[rows]
    mx, my = X.mean(0), Y.mean(0)
    Cx = np.cov(X, rowvar=False)
    Cy = np.cov(Y, rowvar=False)
    ex, Vx = np.linalg.eigh(Cx)
    ey, Vy = np.linalg.eigh(Cy)
    ex, Vx = ex[::-1][:NEIG], Vx[:, ::-1][:, :NEIG]
    ey, Vy = ey[::-1][:NEIG], Vy[:, ::-1][:, :NEIG]
    d = mx - my
    out = dict(
        log_spread=np.log(np.linalg.norm(X - mx, axis=1).mean()),
        log_mean_norm=np.log(np.linalg.norm(mx)),
        log_centre_dist=np.log(np.linalg.norm(d) + 1e-9),
        cos_mean=float(mx @ my / (np.linalg.norm(mx) * np.linalg.norm(my))),
        cos_offset_mean=float(d @ my / (np.linalg.norm(d) * np.linalg.norm(my))),
    )
    for i in range(NEIG):
        out[f'log_eig{i + 1}'] = float(np.log(max(ex[i], 1e-12)))
        out[f'axis{i + 1}'] = float(abs(Vx[:, i] @ Vy[:, i]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stats', default='imagenet',
                    choices=['imagenet', 'symmetric'],
                    help='which version of the study to operate on')
    ap.add_argument('--stage', default='cls')
    ap.add_argument('--ncomp', type=int, default=7)
    args = ap.parse_args()
    C.use_stats(args.stats)

    sp = np.load(C.CACHE / 'split.npz')
    fit, test = sp['fit'], sp['test']
    Z = {c: np.load(C.CACHE / f'{c}.npz')[args.stage][:, R.HISTORY].astype(np.float64)
         for c in C.COLOURS}
    pos = np.load(C.CACHE / f'{C.REFERENCE}.npz')['pos'].astype(np.float64)
    ref = Z[C.REFERENCE]
    dim = ref.shape[1]
    eye = np.eye(dim)
    names = [c for c in C.COLOURS if c != C.REFERENCE]

    probe = RidgeCV(alphas=ALPHAS).fit(ref[fit], pos[fit])
    px = lambda z: float(np.abs(probe.predict(z) - pos[test])[:, R.TRAVEL_AXIS].mean() * R.PX)
    base = px(ref[test])

    maps = {}
    for c in names:
        m = RidgeCV(alphas=ALPHAS).fit(Z[c][fit], ref[fit])
        maps[c] = (m.coef_.copy(), m.intercept_.copy())

    cp = {c: colour_props(c) for c in names}
    ep = {c: embed_props(Z[c], ref, fit) for c in names}
    ckeys, ekeys = list(cp[names[0]]), list(ep[names[0]])
    print(f'stage={args.stage}  {len(names)} colours  floor {base:.2f} px')
    print(f'{len(ckeys)} colour properties, {len(ekeys)} embedding properties\n')

    # --- which candidate lines up with which direction? ---------------------
    Mflat = np.stack([(maps[c][0] - eye).ravel() for c in names])
    bflat = np.stack([maps[c][1] for c in names])
    pM = PCA(args.ncomp).fit(Mflat)
    YM = pM.transform(Mflat)
    evr = pM.explained_variance_ratio_

    print('Correlation of each map direction with each candidate (|r|, 29 colours)')
    print('Only candidates reaching |r| > 0.6 on some direction are listed.\n')
    cand = {**{f'colour: {k}': np.array([cp[c][k] for c in names]) for k in ckeys},
            **{f'embed:  {k}': np.array([ep[c][k] for c in names]) for k in ekeys}}
    hdr = '  '.join(f'PC{i + 1}' for i in range(args.ncomp))
    print(f'{"candidate":<26}{hdr}')
    print(f'{"(variance explained)":<26}' +
          '  '.join(f'{v * 100:>3.0f}%' for v in evr))
    print('-' * (26 + 6 * args.ncomp))
    rows = []
    for k, v in cand.items():
        r = np.array([abs(np.corrcoef(v, YM[:, i])[0, 1]) for i in range(args.ncomp)])
        rows.append((r.max(), k, r))
    for mx, k, r in sorted(rows, reverse=True):
        if mx > 0.6:
            print(f'{k:<26}' + '  '.join(f'{x:>4.2f}' for x in r))

    # --- can either family rebuild a working map? ---------------------------
    print(f'\nLeave-one-colour-out: rebuild the map from each family (px)')
    featsets = {
        'colour properties': lambda ns: np.array([[cp[c][k] for k in ckeys] for c in ns]),
        'embedding properties': lambda ns: np.array([[ep[c][k] for k in ekeys] for c in ns]),
        'both': lambda ns: np.array([[cp[c][k] for k in ckeys] + [ep[c][k] for k in ekeys]
                                     for c in ns]),
    }
    res = {k: {} for k in featsets}
    for c in names:
        rest = [o for o in names if o != c]
        Mr = np.stack([(maps[o][0] - eye).ravel() for o in rest])
        br = np.stack([maps[o][1] for o in rest])
        k = min(args.ncomp, len(rest) - 1)
        pm, pb = PCA(k).fit(Mr), PCA(k).fit(br)
        z = Z[c][test]
        for fname, fs in featsets.items():
            F, Ft = fs(rest), fs([c])
            mu, sd = F.mean(0), F.std(0) + 1e-9      # scales differ by orders of magnitude
            F, Ft = (F - mu) / sd, (Ft - mu) / sd
            Mh = pm.inverse_transform(RidgeCV(alphas=ALPHAS).fit(F, pm.transform(Mr)).predict(Ft))[0]
            bh = pb.inverse_transform(RidgeCV(alphas=ALPHAS).fit(F, pb.transform(br)).predict(Ft))[0]
            res[fname][c] = px(z @ (eye + Mh.reshape(dim, dim)).T + bh)
    own = {c: px(Z[c][test] @ maps[c][0].T + maps[c][1]) for c in names}
    print(f'{"":<24}{"median":>9}{"best":>9}{"worst":>11}')
    for fname in featsets:
        v = np.array(list(res[fname].values()))
        print(f'{fname:<24}{np.median(v):>9.1f}{v.min():>9.1f}{v.max():>11.1f}')
    v = np.array(list(own.values()))
    print(f'{"its own fitted map":<24}{np.median(v):>9.1f}{v.min():>9.1f}{v.max():>11.1f}')

    (C.CACHE / f'index_{args.stage}.json').write_text(json.dumps(
        dict(stage=args.stage, base_px=base, evr=evr.tolist(),
             corr={k: [float(x) for x in r] for _, k, r in rows},
             results={k: res[k] for k in featsets}, own=own), indent=2))
    print(f'\nwrote {C.CACHE / f"index_{args.stage}.json"}')


if __name__ == '__main__':
    main()
