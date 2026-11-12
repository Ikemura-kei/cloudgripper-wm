"""Do the thirty colour maps have a shape, and does colour index it?

colour_maps.py tested one hypothesis — that the map is linear in RGB — and
rejected it with nine colours. That is a statement about one functional form,
not about whether any structure exists. With thirty colours there are thirty
(M_c, b_c) pairs, which is enough to ask the more basic question directly:

    1. What kind of transform is M_c?  Its singular values say whether it is a
       uniform stretch plus a rotation, or something that treats different
       directions of the embedding very differently.
    2. How many degrees of freedom do the thirty maps span?  PCA over the
       stacked maps. If a handful of components carry them, the maps live on a
       low-dimensional surface and only the coordinates on it need predicting.
    3. Is colour the coordinate on that surface?  Leave one colour out, predict
       its position on the surface from its RGB, rebuild the map, and score it
       the way everything else in this study is scored — ball position error
       through the red model's own probe.

Step 3 subsumes the original test: "linear in RGB" is one of the regressors
tried, now with 29 training colours instead of 8.

Usage:
    uv run python scripts/analysis/colour_structure.py [--stage cls]
"""

import argparse
import json
import os
import sys

import numpy as np
from sklearn.decomposition import PCA
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import RidgeCV

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import colour_common as C  # noqa: E402
import rolling_common as R  # noqa: E402

ALPHAS = np.logspace(-5, 2, 29)


def colour_features(names):
    """RGB plus the things RGB does not expose directly.

    Saturation and lightness are included because the original ten colours were
    all fully saturated, so no model fitted on them could have discovered a
    dependence on saturation even if one existed.
    """
    X = np.stack([C.rgb(n) for n in names]) / 255.0
    mx, mn = X.max(1), X.min(1)
    sat = (mx - mn) / np.maximum(mx, 1e-9)
    light = X.mean(1)
    return np.column_stack([X, sat, light, mx, mn])


FEATURE_SETS = {
    'RGB (linear)': lambda n: colour_features(n)[:, :3],
    'RGB + products': lambda n: (lambda X: np.column_stack(
        [X, X[:, 0] * X[:, 1], X[:, 0] * X[:, 2], X[:, 1] * X[:, 2], X ** 2]))(
            colour_features(n)[:, :3]),
    'RGB + sat/light': lambda n: colour_features(n),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stats', default='imagenet',
                    choices=['imagenet', 'symmetric'],
                    help='which version of the study to operate on')
    ap.add_argument('--stage', default='cls')
    ap.add_argument('--ncomp', type=int, default=12)
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
    n = len(names)

    probe = RidgeCV(alphas=ALPHAS).fit(ref[fit], pos[fit])
    px = lambda z: float(np.abs(probe.predict(z) - pos[test])[:, R.TRAVEL_AXIS].mean() * R.PX)
    base = px(ref[test])
    print(f'stage={args.stage}  {n} colours (+ red as reference)  floor {base:.2f} px\n')

    maps = {}
    for c in names:
        m = RidgeCV(alphas=ALPHAS).fit(Z[c][fit], ref[fit])
        maps[c] = (m.coef_.copy(), m.intercept_.copy())
    spread = lambda X: float(np.linalg.norm(X - X.mean(0), axis=1).mean())
    gain = {c: spread(ref[test]) / spread(Z[c][test]) for c in names}

    # --- 1. what kind of transform is M_c? ----------------------------------
    print('1. Is M_c a uniform stretch, or direction-dependent?')
    print('   (singular values of M_c: all equal would mean stretch + rotation)')
    print(f'   {"colour":<9}{"gain":>7}{"top sv":>9}{"median sv":>11}{"sv ratio 90/10":>16}')
    svinfo = {}
    for c in sorted(names, key=lambda c: gain[c])[:4] + sorted(names, key=lambda c: gain[c])[-3:]:
        s = np.linalg.svd(maps[c][0], compute_uv=False)
        svinfo[c] = dict(top=float(s[0]), med=float(np.median(s)),
                         ratio=float(np.percentile(s, 90) / np.percentile(s, 10)))
        print(f'   {c:<9}{gain[c]:>6.1f}x{s[0]:>9.1f}{np.median(s):>11.2f}'
              f'{svinfo[c]["ratio"]:>16.1f}')

    # --- 2. how many degrees of freedom? ------------------------------------
    print('\n2. How many degrees of freedom do the maps span?')
    Mflat = np.stack([(maps[c][0] - eye).ravel() for c in names])
    Mnorm = np.stack([((maps[c][0] - eye) / np.linalg.norm(maps[c][0] - eye)).ravel()
                      for c in names])
    bflat = np.stack([maps[c][1] for c in names])
    for label, A in (('M_c - I (raw)', Mflat), ('M_c - I (scale removed)', Mnorm),
                     ('b_c', bflat)):
        p = PCA().fit(A)
        ev = np.cumsum(p.explained_variance_ratio_)
        k50, k90 = int(np.searchsorted(ev, .5) + 1), int(np.searchsorted(ev, .9) + 1)
        print(f'   {label:<24} 1st component {p.explained_variance_ratio_[0] * 100:>5.1f}% '
              f'| {k50:>2} comps for 50% | {k90:>2} for 90%  (of {n - 1} possible)')

    # --- 3. is colour the coordinate? ---------------------------------------
    # Leave one colour out, fit PCA on the other 29 maps, predict the held-out
    # colour's coordinates from its RGB, rebuild and score. Several regressors,
    # so a failure cannot be blamed on picking a bad functional form.
    print(f'\n3. Leave-one-colour-out: predict the map from colour alone')
    print(f'   (ball position error in px; the floor is {base:.2f}, '
          f'a colour’s own fitted map gets ~6-10)')
    hdr = list(FEATURE_SETS) + ['RBF kernel', '3-nearest', 'shared map']
    print(f'   {"":<11}' + ''.join(f'{h:>17}' for h in hdr))

    results = {h: {} for h in hdr}
    for c in names:
        rest = [o for o in names if o != c]
        Mr = np.stack([(maps[o][0] - eye).ravel() for o in rest])
        br = np.stack([maps[o][1] for o in rest])
        k = min(args.ncomp, len(rest) - 1)
        pM, pb = PCA(k).fit(Mr), PCA(k).fit(br)
        YM, Yb = pM.transform(Mr), pb.transform(br)
        z = Z[c][test]

        def score(Mhat, bhat):
            return px(z @ (eye + Mhat.reshape(dim, dim)).T + bhat)

        for h in list(FEATURE_SETS) + ['RBF kernel']:
            if h == 'RBF kernel':
                mk = lambda: KernelRidge(kernel='rbf', alpha=1e-2, gamma=1.0)
                F = colour_features(rest)[:, :3]
                Ft = colour_features([c])[:, :3]
            else:
                F, Ft = FEATURE_SETS[h](rest), FEATURE_SETS[h]([c])
                mk = lambda: RidgeCV(alphas=ALPHAS)
            Mhat = pM.inverse_transform(mk().fit(F, YM).predict(Ft))[0]
            bhat = pb.inverse_transform(mk().fit(F, Yb).predict(Ft))[0]
            results[h][c] = score(Mhat, bhat)

        d = np.array([np.linalg.norm(C.rgb(o) - C.rgb(c)) for o in rest])
        nn = np.argsort(d)[:3]
        w = (1 / np.maximum(d[nn], 1e-9)); w /= w.sum()
        results['3-nearest'][c] = score(
            sum(wi * Mr[j] for wi, j in zip(w, nn)),
            sum(wi * br[j] for wi, j in zip(w, nn)))

        sm = RidgeCV(alphas=ALPHAS).fit(
            np.concatenate([Z[o][fit] for o in rest]), np.tile(ref[fit], (len(rest), 1)))
        results['shared map'][c] = px(sm.predict(z))

    for stat, fn in (('median', np.median), ('best', np.min), ('worst', np.max)):
        print(f'   {stat:<11}' + ''.join(f'{fn(list(results[h].values())):>17.1f}' for h in hdr))

    # --- 4. the same test, with size and shape handled separately -----------
    # Flattened maps span a 109x range in Frobenius norm, so a PCA over them is
    # dominated by the few high-gain colours and a small coordinate error
    # rebuilds a map with a wildly wrong scale. Splitting each map into a
    # scalar size and a unit-norm shape removes that pathology. The ablation
    # rows then say which half, if either, colour actually predicts.
    print('\n4. Same test, but size and shape predicted separately')
    Dm = {c: maps[c][0] - eye for c in names}
    gM = {c: float(np.linalg.norm(Dm[c])) for c in names}
    gb = {c: float(np.linalg.norm(maps[c][1])) for c in names}
    print(f'   size of M_c-I spans {min(gM.values()):.0f} to {max(gM.values()):.0f} '
          f'({max(gM.values()) / min(gM.values()):.0f}x)')
    abl = {k: {} for k in ('both predicted', 'true size, predicted shape',
                           'predicted size, true shape', 'true size, true shape')}
    for c in names:
        rest = [o for o in names if o != c]
        F, Ft = colour_features(rest), colour_features([c])
        SM = np.stack([(Dm[o] / gM[o]).ravel() for o in rest])
        Sb = np.stack([maps[o][1] / gb[o] for o in rest])
        k = min(10, len(rest) - 1)
        pS, pb2 = PCA(k).fit(SM), PCA(k).fit(Sb)
        unit = lambda v: v / np.linalg.norm(v)
        Sh = unit(pS.inverse_transform(
            RidgeCV(alphas=ALPHAS).fit(F, pS.transform(SM)).predict(Ft))[0])
        bh = unit(pb2.inverse_transform(
            RidgeCV(alphas=ALPHAS).fit(F, pb2.transform(Sb)).predict(Ft))[0])
        gh = float(np.exp(RidgeCV(alphas=ALPHAS).fit(
            F, np.log([gM[o] for o in rest])).predict(Ft)[0]))
        gbh = float(np.exp(RidgeCV(alphas=ALPHAS).fit(
            F, np.log([gb[o] for o in rest])).predict(Ft)[0]))
        z = Z[c][test]
        sc = lambda g, S, g2, B: px(z @ (eye + g * S.reshape(dim, dim)).T + g2 * B)
        abl['both predicted'][c] = sc(gh, Sh, gbh, bh)
        abl['true size, predicted shape'][c] = sc(gM[c], Sh, gb[c], bh)
        abl['predicted size, true shape'][c] = sc(gh, Dm[c].ravel() / gM[c], gbh,
                                                  maps[c][1] / gb[c])
        abl['true size, true shape'][c] = sc(gM[c], Dm[c].ravel() / gM[c], gb[c],
                                             maps[c][1] / gb[c])
    print(f'   {"":<30}{"median":>9}{"best":>9}{"worst":>11}')
    for k2, v in abl.items():
        a = np.array(list(v.values()))
        print(f'   {k2:<30}{np.median(a):>9.1f}{a.min():>9.1f}{a.max():>11.1f}')
    results.update(abl)
    hdr = hdr + list(abl)

    # --- the gain on its own, now with thirty colours ------------------------
    print('\n   For contrast, the gain alone (one number, not the whole map):')
    y = np.log([gain[c] for c in names])
    for h, fs in FEATURE_SETS.items():
        err = []
        for i, c in enumerate(names):
            tr = [j for j in range(n) if j != i]
            m = RidgeCV(alphas=ALPHAS).fit(fs([names[j] for j in tr]), y[tr])
            err.append(abs(float(m.predict(fs([c]))[0]) - y[i]))
        print(f'     log gain ~ {h:<18} leave-one-out within '
              f'x{np.exp(np.mean(err)):.2f}  ({np.mean(err):.3f} nats)')

    pcaM = PCA().fit(Mflat); pcab = PCA().fit(bflat)
    out = dict(stage=args.stage, base_px=base, n_colours=n,
               gain={c: gain[c] for c in names},
               results={h: results[h] for h in hdr}, sv=svinfo,
               evr_M=pcaM.explained_variance_ratio_.tolist(),
               evr_b=pcab.explained_variance_ratio_.tolist())
    p = C.CACHE / f'structure_{args.stage}.json'
    p.write_text(json.dumps(out, indent=2))
    print(f'\nwrote {p}')


if __name__ == '__main__':
    main()
