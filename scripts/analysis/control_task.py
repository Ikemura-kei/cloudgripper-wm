"""Control 2: can the probe fit DELIBERATELY WRONG answers?

A probe with enough capacity can memorise instead of read. Our k-NN scored
0.00 mm on training frames, which was memorisation (the query sits in the
library). The test: shuffle the position labels so they are wrong on purpose
and refit. An honest probe should now fail completely. Whatever accuracy
survives on shuffled labels is capacity, not signal.

Four rows per probe:
  real,     in-sample     -- what I reported before; inflated if the probe memorises
  real,     out-of-sample -- the honest number
  shuffled, in-sample     -- pure memorisation capacity; should be LOW for an honest probe
  shuffled, out-of-sample -- should be ~0 for any probe
"""
import argparse

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

AX = 1


def probes(x, y):
    sc = StandardScaler().fit(x)
    xs = sc.transform(x)
    return {
        'linear (ridge)': (sc, RidgeCV(alphas=np.logspace(-3, 5, 30)).fit(xs, y)),
        'k-NN (k=8)': (sc, KNeighborsRegressor(n_neighbors=8,
                                               weights='distance').fit(xs, y)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', default='outputs/ball_probe/emb_cache.npz')
    ap.add_argument('--rep', default='cls', choices=['cls', 'proj'])
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    d = np.load(args.cache)

    tr = d[f'train|{args.rep}'].reshape(-1, 192)
    tr_y = d['train|pos'][:, :, AX].reshape(-1) * 1000
    ho = d[f'held-out|{args.rep}'].reshape(-1, 192)
    ho_y = d['held-out|pos'][:, :, AX].reshape(-1) * 1000

    rng = np.random.default_rng(args.seed)
    shuf_y = rng.permutation(tr_y)   # same distribution, link to images broken

    print(f'representation: {args.rep}   '
          f'{len(tr)} train frames, {len(ho)} held-out frames\n')
    for name in ['linear (ridge)', 'k-NN (k=8)']:
        print(f'=== {name} ===')
        print(f"{'labels':10s} {'evaluated on':14s} {'MAE':>9s} {'R2':>8s}")
        for tag, y_fit in (('real', tr_y), ('shuffled', shuf_y)):
            sc, mdl = probes(tr, y_fit)[name]
            # in-sample: against the labels it was fitted to
            p = mdl.predict(sc.transform(tr))
            print(f'{tag:10s} {"train (in)":14s} '
                  f'{np.mean(np.abs(p - y_fit)):8.2f}mm {r2_score(y_fit, p):8.3f}')
            # out-of-sample: always against the REAL held-out positions
            p = mdl.predict(sc.transform(ho))
            print(f'{tag:10s} {"held-out (out)":14s} '
                  f'{np.mean(np.abs(p - ho_y)):8.2f}mm {r2_score(ho_y, p):8.3f}')
        print()


if __name__ == '__main__':
    main()
