"""Why do the 29 colour maps span ~7 directions?

PCA said the maps need about seven components for 90% of their variance, out of
a possible twenty-eight. That was reported as structure. Two things have to be
ruled out before it means anything.

A. Is seven more than noise would give?
   PCA on any 29 slightly-wobbly estimates returns *some* spectrum. The control
   is 29 maps that should all be identical: one colour, fitted on 29 disjoint
   slices of its own data. Whatever dimension that produces is the floor, and
   the colour arm is fitted on slices of the same size so the two are matched.

B. Is seven just three, bent?
   Colour has three numbers, so the maps can be at most a 3-parameter family.
   But PCA measures *flat* dimension: a curved 3-D sheet needs more than three
   linear components to describe, and seven is unremarkable for one. Intrinsic
   dimension estimators (TwoNN, Levina-Bickel) measure dimension without
   assuming flatness.

   With only 29 points those estimators are unreliable in the abstract, so they
   are calibrated here against synthetic data built to match: 29 points, the
   same ambient dimension, generated as a smooth nonlinear function of the same
   30 RGB triples. If the real maps score like that synthetic 3-D family, the
   question is answered and nothing beyond colour is involved.

Usage:
    uv run python scripts/analysis/colour_directions.py [--stats imagenet]
"""

import argparse
import json
import os
import sys

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import colour_common as C  # noqa: E402
import rolling_common as R  # noqa: E402

ALPHA = 1e-3


def pca_dims(X, levels=(0.5, 0.9)):
    """How many linear components to reach each fraction of the variance."""
    ev = np.cumsum(PCA().fit(X).explained_variance_ratio_)
    return [int(np.searchsorted(ev, L) + 1) for L in levels]


def twonn(X, discard=0.1):
    """Facco et al. (2017): intrinsic dimension from the ratio of the two
    nearest-neighbour distances. Needs no assumption that the data is flat."""
    D = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
    np.fill_diagonal(D, np.inf)
    r = np.sort(D, axis=1)[:, :2]
    mu = r[:, 1] / np.maximum(r[:, 0], 1e-300)
    mu = np.sort(mu[np.isfinite(mu) & (mu > 1)])
    n = len(mu)
    keep = int(n * (1 - discard))
    x = np.log(mu[:keep])
    y = -np.log(1.0 - (np.arange(1, n + 1) / (n + 1))[:keep])
    return float((x @ y) / (x @ x))          # slope through the origin


def mle_dim(X, k=5):
    """Levina & Bickel (2004) maximum-likelihood estimator, averaged over points."""
    D = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
    np.fill_diagonal(D, np.inf)
    T = np.sort(D, axis=1)[:, :k]
    out = []
    for row in T:
        if row[0] <= 0:
            continue
        out.append((k - 1) / np.sum(np.log(row[k - 1] / row[:k - 1])))
    return float(np.mean(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stats', default='imagenet', choices=['imagenet', 'symmetric'])
    ap.add_argument('--n-sub', type=int, default=250)
    args = ap.parse_args()
    C.use_stats(args.stats)

    sp = np.load(C.CACHE / 'split.npz')
    fit = sp['fit']
    Z = {c: np.load(C.CACHE / f'{c}.npz')['cls'][:, R.HISTORY].astype(np.float64)
         for c in C.COLOURS}
    ref = Z[C.REFERENCE]
    dim = ref.shape[1]
    eye = np.eye(dim)
    names = [c for c in C.COLOURS if c != C.REFERENCE]
    n = len(names)
    print(f'stats={args.stats}  {n} colours  embedding dim {dim}\n')

    fitmap = lambda src, rows: Ridge(alpha=ALPHA).fit(src[rows], ref[rows]).coef_ - eye

    # ---- A: noise floor -----------------------------------------------------
    # 29 disjoint slices, so no two noise maps share a single window.
    # Slices are drawn from ALL windows, not just the fit split: this arm is
    # about how much a map wobbles when the data under it changes, which is a
    # question about estimation, not about generalising to held-out episodes.
    k = args.n_sub
    allw = np.arange(len(ref))
    if n * k > len(allw):
        k = len(allw) // n
        print(f'(slice size reduced to {k} so all {n} slices stay disjoint)\n')
    rng = np.random.default_rng(0)
    perm = rng.permutation(allw)
    slices = [perm[i * k:(i + 1) * k] for i in range(n)]

    noise = np.stack([fitmap(Z['green'], s).ravel() for s in slices])
    colour = np.stack([fitmap(Z[c], slices[0]).ravel() for c in names])
    full = np.stack([fitmap(Z[c], fit).ravel() for c in names])

    print(f'A. Is seven more than noise?   ({k} windows per fit, disjoint slices)')
    print(f'{"":<44}{"50%":>6}{"90%":>6}')
    for lab, X in ((f'noise floor: one colour, {n} data slices', noise),
                   (f'colour: {n} colours, one shared slice', colour),
                   (f'colour: {n} colours, all {len(fit)} windows', full)):
        a, b = pca_dims(X)
        print(f'   {lab:<41}{a:>6}{b:>6}')

    # ---- B: intrinsic dimension --------------------------------------------
    # Calibration: 29 points that ARE a smooth 3-parameter family of the same
    # RGB triples, in the same ambient dimension. Whatever the estimators say
    # here is what "it is only colour, bent" looks like at this sample size.
    rgb = np.stack([C.rgb(c) / 255 for c in names])
    g = np.random.default_rng(1)
    W1 = g.normal(size=(3, 40))
    W2 = g.normal(size=(40, full.shape[1]))
    synth3 = np.tanh(rgb @ W1 * 2.0) @ W2            # curved 3-D family
    synth7 = g.normal(size=(n, 7)) @ g.normal(size=(7, full.shape[1]))   # flat 7-D
    synth29 = g.normal(size=(n, full.shape[1]))                          # unstructured

    print(f'\nB. Is seven just three, bent?   (intrinsic dimension, {n} points)')
    print(f'{"":<44}{"TwoNN":>8}{"MLE k=5":>9}{"PCA 90%":>9}')
    # A 3-D family carrying the same estimation noise the real maps carry.
    # Scaled so its noise-to-signal ratio matches: how far the noise maps sit
    # from their own mean, against how far the colour maps sit from theirs.
    sig = lambda X: float(np.linalg.norm(X - X.mean(0), axis=1).mean())
    ratio = sig(noise) / sig(colour)
    synth3n = synth3 + g.normal(size=synth3.shape) * (sig(synth3) * ratio
                                                      / np.sqrt(synth3.shape[1]))
    rows = (('CALIBRATION curved 3-D family of RGB', synth3),
            (f'CALIBRATION the same, plus matched noise ({ratio:.2f}x)', synth3n),
            ('CALIBRATION flat 7-D', synth7),
            ('CALIBRATION unstructured', synth29),
            ('CONTROL one colour, 29 data slices (pure noise)', noise),
            (f'the real maps, {k}-window fits (noise-matched)', colour),
            ('the real maps, full fits   M_c - I', full),
            ('the real shifts, full fits  b_c', np.stack(
                [Ridge(alpha=ALPHA).fit(Z[c][fit], ref[fit]).intercept_
                 for c in names])))
    for lab, X in rows:
        print(f'   {lab:<41}{twonn(X):>8.2f}{mle_dim(X):>9.2f}{pca_dims(X)[1]:>9}')

    res = dict(stats=args.stats, n_sub=k,
               pca={'noise': pca_dims(noise), 'colour_slice': pca_dims(colour),
                    'colour_full': pca_dims(full)},
               intrinsic={'synth3': [twonn(synth3), mle_dim(synth3)],
                          'synth7': [twonn(synth7), mle_dim(synth7)],
                          'maps': [twonn(full), mle_dim(full)]})
    (C.CACHE / 'directions.json').write_text(json.dumps(res, indent=2))
    print(f'\nwrote {C.CACHE / "directions.json"}')


if __name__ == '__main__':
    main()
