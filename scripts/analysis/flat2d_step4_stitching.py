"""Experiment 4: can one model's readout read another model's state?

Experiment 2 found that a position readout fitted on one model scores about
-42000 on another, even when the two models were trained on identical data and
differ only in their random starting weights. So the raw answer is no.

But Experiment 1 showed why that happens: two networks can hold the same
information along different internal axes. If that is all that is going on
here, then lining the axes up should repair the transfer completely.

The test, for an ordered pair of models (A, B):

    1. fit an affine map  T : z_B -> z_A   on training-episode windows
    2. take A's position readout  P_A, fitted only on A
    3. score  P_A( T(z_B) )  on held-out windows

If that recovers A's own accuracy, the two models are holding the same thing
and only the coordinates differed. This is the model-stitching idea (Bansal,
Nakkiran & Barak 2021), with a linear stitch.

Three groups, same as before:
    same colour, different seed   - the ceiling for this test
    different colour              - the question
    trained vs untrained          - the floor

Usage:
    uv run python scripts/analysis/flat2d_step4_stitching.py
"""

import itertools
import json
import os
import sys

import numpy as np
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from flat2d_common import HISTORY, REPO  # noqa: E402

CACHE = REPO / 'outputs' / 'flat2d_cache'
OUT = REPO / 'outputs' / 'flat2d_step4'
TRAINED = ['red1', 'red2', 'red3', 'green1', 'green2', 'green3']
ALL = TRAINED + ['untrained']


def fit_map(src_fit, dst_fit, alpha=1.0):
    """Affine map from one embedding space to another, fitted on the fit split."""
    sc = StandardScaler().fit(src_fit)
    reg = Ridge(alpha=alpha).fit(sc.transform(src_fit), dst_fit)
    return lambda z: reg.predict(sc.transform(z))


def fit_probe(X, y):
    sc = StandardScaler().fit(X)
    reg = RidgeCV(alphas=np.logspace(-2, 5, 25)).fit(sc.transform(X), y)
    return lambda z: reg.predict(sc.transform(z))


def main(rep='cls'):
    OUT.mkdir(parents=True, exist_ok=True)
    fit = {m: np.load(CACHE / f's2_{m}_fit.npz') for m in ALL}
    val = {m: np.load(CACHE / f's2_{m}_val.npz') for m in ALL}
    E = {m: {'f': fit[m][rep][:, HISTORY], 'v': val[m][rep][:, HISTORY]} for m in ALL}
    yf, yv = fit['red1']['x'], val['red1']['x']

    own = {m: float(r2_score(yv, fit_probe(E[m]['f'], yf)(E[m]['v']))) for m in ALL}
    probes = {m: fit_probe(E[m]['f'], yf) for m in ALL}
    print(f'{len(yv)} held-out windows. Each model reading its own state:')
    print('  ' + '  '.join(f'{m} {own[m]:.3f}' for m in ALL) + '\n')

    rows = {}
    for a, b in itertools.permutations(ALL, 2):
        T = fit_map(E[b]['f'], E[a]['f'])
        direct = float(r2_score(yv, probes[a](E[b]['v'])))
        stitched = float(r2_score(yv, probes[a](T(E[b]['v']))))
        rows[f'{a}<-{b}'] = {'direct': direct, 'stitched': stitched,
                             'own_a': own[a], 'recovered': stitched / own[a]}

    def group(pairs, title):
        st = [rows[k]['stitched'] for k in pairs]
        di = [rows[k]['direct'] for k in pairs]
        rc = [rows[k]['recovered'] for k in pairs]
        print(f'{title}  (n={len(pairs)})')
        print(f'  stitched  R2  mean {np.mean(st):.4f}   range [{min(st):.4f}, {max(st):.4f}]')
        print(f'  recovered     mean {100*np.mean(rc):.1f}%  of the reader model\'s own score')
        print(f'  without the map, for comparison: mean R2 {np.mean(di):,.0f}\n')
        return {'stitched_mean': float(np.mean(st)), 'stitched_min': float(min(st)),
                'stitched_max': float(max(st)), 'recovered_mean': float(np.mean(rc)),
                'direct_mean': float(np.mean(di)), 'n': len(pairs)}

    reds, greens = TRAINED[:3], TRAINED[3:]
    same = [f'{a}<-{b}' for a, b in itertools.permutations(reds, 2)]
    same += [f'{a}<-{b}' for a, b in itertools.permutations(greens, 2)]
    cross = [f'{a}<-{b}' for a in reds for b in greens] + \
            [f'{a}<-{b}' for a in greens for b in reds]
    floor_in = [f'{a}<-untrained' for a in TRAINED]
    floor_out = [f'untrained<-{b}' for b in TRAINED]

    summary = {
        'same_colour': group(same, 'SAME COLOUR, different seed  (ceiling)'),
        'different_colour': group(cross, 'DIFFERENT COLOUR            (the question)'),
        'read_untrained': group(floor_in, 'TRAINED model reading an UNTRAINED one (floor)'),
        'untrained_reads': group(floor_out, 'UNTRAINED model reading a TRAINED one (floor)'),
    }

    with open(OUT / f'stitching_{rep}.json', 'w') as f:
        json.dump({'rep': rep, 'own': own, 'pairs': rows, 'summary': summary}, f, indent=2)
    print('saved ->', OUT / f'stitching_{rep}.json')


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'cls')
