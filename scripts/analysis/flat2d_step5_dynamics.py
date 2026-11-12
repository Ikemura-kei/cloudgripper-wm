"""Experiment 5: do the two models predict motion the same way?

Everything before this looked at a single frame. A world model is supposed to
roll a state forward, and two models could encode position identically yet
disagree about where the ball goes next.

The predictor sees the first 6 frames' embeddings and outputs its estimate of
the 7th, so each window gives three vectors in the same space:

    prev   proj[:, 5]   the last frame the predictor actually saw
    truth  proj[:, 6]   what the 7th frame's embedding really is
    pred                what the predictor said the 7th would be

Four tests:

1. IS THE PREDICTOR ANY GOOD? Compare `pred` against `truth` directly.
2. DOES THE PREDICTION STILL CONTAIN THE BALL? Read a position out of `pred`
   and out of `truth`, each with its own matched probe (Experiment 1 showed a
   probe fitted in one space does not transfer to another, so both must be
   fitted where they are used).
3. DOES EXPERIMENT 4's STITCH CARRY THE DYNAMICS? Fit the map between two
   models on their STATIC embeddings only, then apply it to the predictor
   outputs it has never seen.
4. IS SPEED HANDLED THE SAME WAY? Speed is invisible in a single frame, so it
   has to come from the step the predictor takes.

Usage:
    uv run python scripts/analysis/flat2d_step5_dynamics.py
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
OUT = REPO / 'outputs' / 'flat2d_step5'
TRAINED = ['red1', 'red2', 'red3', 'green1', 'green2', 'green3']
ALL = TRAINED + ['untrained']
REP = 'proj'
PX = 448


def parts(z):
    return {'prev': z[REP][:, HISTORY - 1], 'truth': z[REP][:, HISTORY],
            'pred': z['pred']}


def fit_ridge(X, y, cv=True):
    sc = StandardScaler().fit(X)
    reg = RidgeCV(alphas=np.logspace(-2, 5, 25)) if cv else Ridge(alpha=1.0)
    reg.fit(sc.transform(X), y)
    return lambda Z: reg.predict(sc.transform(Z))


def r2_multi(pred, true):
    return float(1 - ((pred - true) ** 2).sum() / ((true - true.mean(0)) ** 2).sum())


def groups():
    reds, greens = TRAINED[:3], TRAINED[3:]
    same = [f'{a}<-{b}' for a, b in itertools.permutations(reds, 2)] + \
           [f'{a}<-{b}' for a, b in itertools.permutations(greens, 2)]
    cross = [f'{a}<-{b}' for a in reds for b in greens] + \
            [f'{a}<-{b}' for a in greens for b in reds]
    floor = [f'{a}<-untrained' for a in TRAINED] + [f'untrained<-{b}' for b in TRAINED]
    return [('same colour, different seed', same), ('different colour', cross),
            ('involving the untrained model', floor)]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fit = {m: np.load(CACHE / f's2_{m}_fit.npz') for m in ALL}
    val = {m: np.load(CACHE / f's2_{m}_val.npz') for m in ALL}
    F = {m: parts(fit[m]) for m in ALL}
    V = {m: parts(val[m]) for m in ALL}
    g, gv = fit['red1'], val['red1']
    xn_f, xn_v = g['pos_traj'][:, HISTORY, 0], gv['pos_traj'][:, HISTORY, 0]
    xp_v = gv['pos_traj'][:, HISTORY - 1, 0]
    spd_f, spd_v = g['speed'], gv['speed']
    step_px = float(np.mean(xn_v - xp_v) * PX)
    res = {'step_px': step_px, 'n_val': int(len(xn_v))}
    print(f'{len(xn_v)} held-out windows. The ball advances {step_px:.1f} px per step.\n')

    # ---- 1. is the predictor any good at its own job? --------------------
    print('1. PREDICTOR ACCURACY  (how close is `pred` to `truth`?)')
    print(f'   {"model":<12}{"rel. error":>12}{"cosine":>10}{"gap truth-prev":>17}')
    t1 = {}
    for m in ALL:
        e = float(np.linalg.norm(V[m]['pred'] - V[m]['truth'], axis=1).mean())
        n = float(np.linalg.norm(V[m]['truth'], axis=1).mean())
        gap = float(np.linalg.norm(V[m]['truth'] - V[m]['prev'], axis=1).mean())
        c = float(np.mean(np.sum(V[m]['pred'] * V[m]['truth'], 1) /
                          (np.linalg.norm(V[m]['pred'], axis=1) *
                           np.linalg.norm(V[m]['truth'], axis=1))))
        t1[m] = {'rel_err': e / n, 'cosine': c, 'norm': n, 'gap_truth_prev': gap}
        print(f'   {m:<12}{e/n:>12.3f}{c:>10.4f}{gap:>17.2f}')
    res['predictor_accuracy'] = t1

    # ---- 2. can the ball still be found in the prediction? ---------------
    print(f'\n2. POSITION, read with a matched probe')
    print(f'   baseline "assume no movement": {step_px:.1f} px error\n')
    print(f'   {"model":<12}{"from truth":>12}{"err px":>9}{"from pred":>12}{"err px":>9}')
    t2 = {}
    for m in TRAINED:
        a = fit_ridge(F[m]['truth'], xn_f)(V[m]['truth'])
        b = fit_ridge(F[m]['pred'], xn_f)(V[m]['pred'])
        t2[m] = {'truth_r2': float(r2_score(xn_v, a)),
                 'truth_px': float(np.mean(np.abs(a - xn_v)) * PX),
                 'pred_r2': float(r2_score(xn_v, b)),
                 'pred_px': float(np.mean(np.abs(b - xn_v)) * PX)}
        print(f'   {m:<12}{t2[m]["truth_r2"]:>12.4f}{t2[m]["truth_px"]:>9.1f}'
              f'{t2[m]["pred_r2"]:>12.4f}{t2[m]["pred_px"]:>9.1f}')
    res['position_readout'] = t2

    # how much of the embedding's variance does the position direction hold?
    share = {}
    for m in ['red1', 'green1']:
        X = F[m]['truth']
        sc = StandardScaler().fit(X)
        r = Ridge(alpha=1.0).fit(sc.transform(X), xn_f)
        d = r.coef_ / np.linalg.norm(r.coef_)
        share[m] = float(np.var(sc.transform(X) @ d) / np.var(sc.transform(X), axis=0).sum())
        print(f'   position direction holds {100*share[m]:.3f}% of {m}\'s variance')
    res['position_variance_share'] = share

    # ---- 3. does the static stitch carry the predictions? ----------------
    print('\n3. STITCH, fitted on STATIC frames only, applied to predictions')
    t3 = {}
    for a, b in itertools.permutations(ALL, 2):
        T = fit_ridge(F[b]['prev'], F[a]['prev'], cv=False)
        dt, dm = V[a]['pred'] - V[a]['prev'], T(V[b]['pred']) - T(V[b]['prev'])
        cos = float(np.mean(np.sum(dt * dm, 1) / (np.linalg.norm(dt, axis=1) *
                                                  np.linalg.norm(dm, axis=1))))
        t3[f'{a}<-{b}'] = {'static': r2_multi(T(V[b]['prev']), V[a]['prev']),
                           'pred': r2_multi(T(V[b]['pred']), V[a]['pred']),
                           'step_cosine': cos}
    print(f'   {"group":<34}{"static":>10}{"predictions":>14}{"step direction":>17}')
    t3s = {}
    for name, keys in groups():
        t3s[name] = {k: float(np.mean([t3[q][k] for q in keys]))
                     for k in ('static', 'pred', 'step_cosine')}
        print(f'   {name:<34}{t3s[name]["static"]:>10.4f}{t3s[name]["pred"]:>14.4f}'
              f'{t3s[name]["step_cosine"]:>17.4f}')
    res['stitch_dynamics'] = {'pairs': t3, 'summary': t3s}

    # ---- 4. speed ---------------------------------------------------------
    print(f'\n4. SPEED, read from the step the predictor takes '
          f'({len(np.unique(spd_v))} distinct values in the held-out set)')
    t4 = {}
    for m in ALL:
        probe = fit_ridge(F[m]['pred'] - F[m]['prev'], spd_f)
        t4[m] = float(r2_score(spd_v, probe(V[m]['pred'] - V[m]['prev'])))
    print('   own:  ' + '  '.join(f'{m} {t4[m]:.3f}' for m in TRAINED))
    t4x = {}
    for a, b in itertools.permutations(ALL, 2):
        T = fit_ridge(F[b]['prev'], F[a]['prev'], cv=False)
        probe = fit_ridge(F[a]['pred'] - F[a]['prev'], spd_f)
        t4x[f'{a}<-{b}'] = float(r2_score(spd_v, probe(T(V[b]['pred']) - T(V[b]['prev']))))
    t4s = {}
    print(f'   {"group":<34}{"stitched speed R2":>20}')
    for name, keys in groups():
        t4s[name] = float(np.mean([t4x[k] for k in keys]))
        print(f'   {name:<34}{t4s[name]:>20.4f}')
    res['speed'] = {'own': t4, 'stitched': t4x, 'summary': t4s}

    json.dump(res, open(OUT / 'step5_results.json', 'w'), indent=2)
    print(f'\nsaved -> {OUT / "step5_results.json"}')


if __name__ == '__main__':
    main()
