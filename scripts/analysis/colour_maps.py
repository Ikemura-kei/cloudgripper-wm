"""Is the colour-correction map a predictable function of the colour change?

Background. The red model, shown a green ball, produces embeddings that an
affine map sends back to its red-ball embeddings almost perfectly. That map has
to be *fitted*, which needs paired data in the new colour — so on its own it
buys nothing. This script asks whether the map can instead be *predicted* from
the colour change, which would need no paired data and no new model.

The hypothesis
--------------
Write the colour change as dc = (c - c_red)/255, a 3-vector. Suppose the
correction map is affine in the embedding and linear in dc:

    f_c(z) = M_c z + b_c      M_c = I + sum_k dc_k A_k      b_c = sum_k dc_k b_k

At c = red, dc = 0 and f is the identity, so the model is exactly right at the
training colour by construction rather than by fitting. The unknowns are three
matrices A_k and three vectors b_k; three independent colour changes determine
them.

What gets measured
------------------
Table A  every colour gets its own fitted map. Does the information survive?
Table B  leave-one-colour-out: predict each colour's map from the other eight,
         four ways, and score against that colour's own fitted map.
Table C  why B comes out the way it does — the gain each colour needs, how
         reliable a fitted map is, and how much the predicted one resembles it.

The four ways of handling a recoloured embedding, weakest first:

    no map     feed it in untouched — what you get today
    linear     the hypothesis above, fitted on the other eight colours
    interp     inverse-distance-weighted blend of the other eight fitted maps
    shared     one map fitted jointly on the other eight colours' windows
    fitted     that colour's own map, fitted on its own data — the ceiling

Metrics. Embedding R2 against the red-ball embedding of the same scene, and the
one that matters downstream: ball position read out by a ridge probe trained
only on red embeddings, in pixels of a 448-wide frame.

Usage:
    uv run python scripts/analysis/colour_maps.py [--stage cls|proj]
"""

import argparse
import json
import os
import sys

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import colour_common as C  # noqa: E402
import rolling_common as R  # noqa: E402

ALPHAS = np.logspace(-5, 2, 29)


def cos(u, v):
    return float(u.ravel() @ v.ravel() / (np.linalg.norm(u) * np.linalg.norm(v)))


def spread(X):
    """Mean distance from the centroid — how much room the embeddings occupy."""
    return float(np.linalg.norm(X - X.mean(0), axis=1).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stats', default='imagenet',
                    choices=['imagenet', 'symmetric'],
                    help='which version of the study to operate on')
    ap.add_argument('--stage', default='cls', choices=['cls', 'proj'])
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
    others = [c for c in C.COLOURS if c != C.REFERENCE]
    print(f'stage={args.stage}  dim={dim}  {len(fit)} fit, {len(test)} test windows\n')

    # The probe stands in for whatever a user has already built on the red
    # model. It is trained on red embeddings and never sees a recoloured one.
    probe = RidgeCV(alphas=ALPHAS).fit(ref[fit], pos[fit])
    base_px = float(np.abs(probe.predict(ref[test]) - pos[test])[:, R.TRAVEL_AXIS].mean() * R.PX)

    def px(z):
        return float(np.abs(probe.predict(z) - pos[test])[:, R.TRAVEL_AXIS].mean() * R.PX)

    def r2(z):
        return float(r2_score(ref[test], z, multioutput='variance_weighted'))

    def fit_map(src, rows):
        m = RidgeCV(alphas=ALPHAS).fit(src[rows], ref[rows])
        return m.coef_.copy(), m.intercept_.copy()

    apply_ = lambda Mb, z: z @ Mb[0].T + Mb[1]

    # --- Table A -------------------------------------------------------------
    maps = {c: fit_map(Z[c], fit) for c in others}
    gain = {c: spread(ref[test]) / spread(Z[c][test]) for c in others}
    M_rr = fit_map(ref, fit)
    print(f'position probe on red embeddings: {base_px:.2f} px  (the floor)')
    print(f'sanity: a map fitted red->red is the identity to '
          f'{np.linalg.norm(M_rr[0] - eye) / np.sqrt(dim):.4f} per dim, '
          f'|b| = {np.linalg.norm(M_rr[1]):.4f}\n')

    print('Table A — every colour gets its own fitted map (held-out episodes)')
    print(f'{"colour":<9} {"RGB":<16} {"gain":>7} {"no map px":>10} '
          f'{"fitted R2":>10} {"fitted px":>10}')
    tableA = []
    for c in sorted(others, key=lambda c: gain[c]):
        z = Z[c][test]
        row = dict(colour=c, rgb=list(C.COLOURS[c]), gain=gain[c],
                   spread=1 / gain[c], raw_r2=r2(z), raw_px=px(z),
                   fitted_r2=r2(apply_(maps[c], z)), fitted_px=px(apply_(maps[c], z)))
        tableA.append(row)
        print(f'{c:<9} {str(C.COLOURS[c]):<16} {row["gain"]:>6.1f}x '
              f'{row["raw_px"]:>10.1f} {row["fitted_r2"]:>10.4f} {row["fitted_px"]:>10.2f}')

    # --- Table B: leave one colour out ---------------------------------------
    def linear_model(train):
        D = np.stack([C.delta(c) for c in train])
        P = np.linalg.pinv(D)
        A = np.tensordot(P, np.stack([maps[c][0] - eye for c in train]), axes=(1, 0))
        Bv = P @ np.stack([maps[c][1] for c in train])
        return A, Bv

    print('\nTable B — leave-one-colour-out: predict the map from the other eight')
    print(f'{"colour":<9} |{"no map":>10} |{"linear":>10} |{"interp":>10} '
          f'|{"shared":>10} |{"fitted":>10}   (ball position error, px)')
    tableB = []
    for c in sorted(others, key=lambda c: gain[c]):
        rest = [o for o in others if o != c]
        z = Z[c][test]

        A, Bv = linear_model(rest)
        d = C.delta(c)
        lin = (eye + np.tensordot(d, A, axes=(0, 0)), d @ Bv)

        w = 1 / np.maximum([np.linalg.norm(C.rgb(o) - C.rgb(c)) for o in rest], 1e-9) ** 2
        w = w / w.sum()
        interp = (sum(wi * maps[o][0] for wi, o in zip(w, rest)),
                  sum(wi * maps[o][1] for wi, o in zip(w, rest)))

        sm = RidgeCV(alphas=ALPHAS).fit(
            np.concatenate([Z[o][fit] for o in rest]), np.tile(ref[fit], (len(rest), 1)))
        shared = (sm.coef_.copy(), sm.intercept_.copy())

        variants = {'nomap': z, 'linear': apply_(lin, z), 'interp': apply_(interp, z),
                    'shared': apply_(shared, z), 'fitted': apply_(maps[c], z)}
        row = dict(colour=c, rgb=list(C.COLOURS[c]), gain=gain[c],
                   linear_cosM=cos(lin[0] - eye, maps[c][0] - eye),
                   linear_cosb=cos(lin[1], maps[c][1]))
        for k, zz in variants.items():
            row[f'{k}_px'] = px(zz)
            row[f'{k}_r2'] = r2(zz)
        tableB.append(row)
        print(f'{c:<9} |' + '|'.join(f'{row[k + "_px"]:>10.1f} '
                                     for k in ('nomap', 'linear', 'interp', 'shared', 'fitted')))

    # --- Table C: diagnosis ---------------------------------------------------
    # How reliable is a fitted map? Fit each one twice on disjoint halves of the
    # fit episodes. Nothing can match "the" fitted map better than it matches
    # itself, so this is the ceiling the cos columns are read against.
    half = len(fit) // 2
    print('\nTable C — why: is the map stable, and does the prediction resemble it?')
    print(f'{"colour":<9} {"gain":>7} {"cos(M) split-half":>18} {"cos(M) predicted":>17}')
    for row in tableB:
        c = row['colour']
        m1, m2 = fit_map(Z[c], fit[:half]), fit_map(Z[c], fit[half:])
        row['reliab_cosM'] = cos(m1[0] - eye, m2[0] - eye)
        print(f'{c:<9} {row["gain"]:>6.1f}x {row["reliab_cosM"]:>18.3f} '
              f'{row["linear_cosM"]:>17.3f}')

    # Do two colours' maps even point the same way? Cosine between the fitted
    # matrices, normalised, so only shape is compared and not scale.
    cosM = {a: {b: cos(maps[a][0], maps[b][0]) for b in others} for a in others}

    # Is the *scale* at least predictable, even if the shape is not? Leave one
    # colour out and predict log gain from RGB.
    X = np.stack([C.rgb(c) / 255 for c in others])
    y = np.log(np.array([gain[c] for c in others]))
    gain_fit = {}
    for name, feat in (('RGB', X),
                       ('RGB + products', np.column_stack(
                           [X, X[:, 0] * X[:, 1], X[:, 0] * X[:, 2], X[:, 1] * X[:, 2]]))):
        errs = []
        for i in range(len(others)):
            tr = [j for j in range(len(others)) if j != i]
            A_ = np.column_stack([feat[tr], np.ones(len(tr))])
            w_, *_ = np.linalg.lstsq(A_, y[tr], rcond=None)
            errs.append(abs(np.r_[feat[i], 1] @ w_ - y[i]))
        gain_fit[name] = float(np.mean(errs))
        print(f'\nlog gain ~ {name:<16} leave-one-out error {np.mean(errs):.3f} nats '
              f'-> gain predicted within x{np.exp(np.mean(errs)):.2f}')

    res = dict(stage=args.stage, base_px=base_px, tableA=tableA, tableB=tableB,
               cosM=cosM, gain_fit=gain_fit, colours=list(C.COLOURS))
    p = C.CACHE / f'maps_{args.stage}.json'
    p.write_text(json.dumps(res, indent=2))
    print(f'\nwrote {p}')


if __name__ == '__main__':
    main()
