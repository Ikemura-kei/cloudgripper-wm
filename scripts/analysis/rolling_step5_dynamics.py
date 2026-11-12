"""Experiment 5, re-run on the rolling-ball models.

The original Experiment 5 found that the predictor's output no longer told you
where the ball was: reading position out of it gave a 68 px error against a
10 px step, and the decoded ball correlated with the truth at +0.07. That was
measured on models trained at frameskip 1, where the ball barely moved between
the frames being related.

These models are trained at frameskip 5, where it advances 33 px. Same four
measurements, so the numbers are directly comparable.

Usage:
    uv run python scripts/analysis/rolling_step5_dynamics.py
"""

import json
import os
import sys

import numpy as np
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rolling_common as R  # noqa: E402

OUT = R.REPO / 'outputs' / 'rolling_step5'
COLOURS = ['red', 'green']


def parts(z):
    """prev / truth / pred, all in the predictor's own space."""
    return {'prev': z['proj'][:, R.HISTORY - 1], 'truth': z['proj'][:, R.HISTORY],
            'pred': z['pred']}


def fit_ridge(X, y, cv=True):
    sc = StandardScaler().fit(X)
    reg = RidgeCV(alphas=np.logspace(-2, 5, 25)) if cv else Ridge(alpha=1.0)
    reg.fit(sc.transform(X), y)
    return lambda Z: reg.predict(sc.transform(Z))


def r2_multi(p, t):
    return float(1 - ((p - t) ** 2).sum() / ((t - t.mean(0)) ** 2).sum())


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    F = {c: np.load(R.CACHE / f'{c}_fit.npz') for c in COLOURS}
    V = {c: np.load(R.CACHE / f'{c}_val.npz') for c in COLOURS}
    Fp = {c: parts(F[c]) for c in COLOURS}
    Vp = {c: parts(V[c]) for c in COLOURS}
    xn_f = F['red']['pos_traj'][:, R.HISTORY, R.TRAVEL_AXIS]
    xn_v = V['red']['pos_traj'][:, R.HISTORY, R.TRAVEL_AXIS]
    xp_v = V['red']['pos_traj'][:, R.HISTORY - 1, R.TRAVEL_AXIS]
    spd_f, spd_v = F['red']['speed'], V['red']['speed']
    step = float(np.mean(np.abs(xn_v - xp_v)) * R.PX)
    res = {'step_px': step, 'n_val': int(len(xn_v))}
    print(f'{len(xn_v)} held-out windows. The ball advances {step:.1f} px per step '
          f'(it was 10.2 px at frameskip 1).\n')

    print('1. PREDICTOR ACCURACY')
    print(f'   {"model":<8}{"rel. error":>12}{"cosine":>9}{"variance explained":>21}')
    t1 = {}
    for c in COLOURS:
        p, t = Vp[c]['pred'], Vp[c]['truth']
        e = float(np.linalg.norm(p - t, axis=1).mean())
        n = float(np.linalg.norm(t, axis=1).mean())
        cos = float(np.mean(np.sum(p * t, 1) / (np.linalg.norm(p, axis=1) * np.linalg.norm(t, axis=1))))
        t1[c] = {'rel_err': e / n, 'cosine': cos, 'var_explained': r2_multi(p, t)}
        print(f'   {c:<8}{e/n:>12.3f}{cos:>9.4f}{t1[c]["var_explained"]:>21.4f}')
    res['predictor'] = t1

    print(f'\n2. POSITION, read with a matched probe (baseline "it did not move" = {step:.1f} px)')
    print(f'   {"model":<8}{"from cls":>11}{"from truth":>13}{"from pred":>12}   (px)')
    t2 = {}
    for c in COLOURS:
        row = {}
        for name, zf, zv in (('cls', F[c]['cls'][:, R.HISTORY], V[c]['cls'][:, R.HISTORY]),
                             ('truth', Fp[c]['truth'], Vp[c]['truth']),
                             ('pred', Fp[c]['pred'], Vp[c]['pred'])):
            p = fit_ridge(zf, xn_f)(zv)
            row[name] = {'r2': float(r2_score(xn_v, p)),
                         'px': float(np.mean(np.abs(p - xn_v)) * R.PX)}
        t2[c] = row
        print(f'   {c:<8}{row["cls"]["px"]:>11.1f}{row["truth"]["px"]:>13.1f}'
              f'{row["pred"]["px"]:>12.1f}')
    res['position'] = t2

    print('\n3. STITCH fitted on STATIC frames, applied to PREDICTIONS')
    t3 = {}
    for a, b in (('red', 'green'), ('green', 'red')):
        T = fit_ridge(Fp[b]['prev'], Fp[a]['prev'], cv=False)
        dt_, dm = Vp[a]['pred'] - Vp[a]['prev'], T(Vp[b]['pred']) - T(Vp[b]['prev'])
        t3[f'{a}<-{b}'] = {
            'static': r2_multi(T(Vp[b]['prev']), Vp[a]['prev']),
            'pred': r2_multi(T(Vp[b]['pred']), Vp[a]['pred']),
            'step_cosine': float(np.mean(np.sum(dt_ * dm, 1) /
                                         (np.linalg.norm(dt_, axis=1) * np.linalg.norm(dm, axis=1)))),
        }
        r = t3[f'{a}<-{b}']
        print(f'   {a} reads {b:<7} static {r["static"]:.4f}   predictions {r["pred"]:.4f}'
              f'   step direction {r["step_cosine"]:+.4f}')
    res['stitch'] = t3

    print(f'\n4. SPEED from the predictor\'s step ({len(np.unique(spd_v))} distinct values)')
    t4 = {}
    for c in COLOURS:
        probe = fit_ridge(Fp[c]['pred'] - Fp[c]['prev'], spd_f)
        t4[c] = {'own': float(r2_score(spd_v, probe(Vp[c]['pred'] - Vp[c]['prev'])))}
        print(f'   {c:<8} own R2 {t4[c]["own"]:.4f}')
    for a, b in (('red', 'green'), ('green', 'red')):
        T = fit_ridge(Fp[b]['prev'], Fp[a]['prev'], cv=False)
        probe = fit_ridge(Fp[a]['pred'] - Fp[a]['prev'], spd_f)
        v = float(r2_score(spd_v, probe(T(Vp[b]['pred']) - T(Vp[b]['prev']))))
        t4[f'{a}<-{b}'] = v
        print(f'   {a} reads {b}: stitched speed R2 {v:.4f}')
    res['speed'] = t4

    json.dump(res, open(OUT / 'results.json', 'w'), indent=2)
    print('\n--- the same measurements at frameskip 1, for comparison ---')
    print('   predictor rel. error 7.6-8.2%, variance explained 0.992-0.993')
    print('   position: cls 16 px | truth 31 px | pred 68 px   (step was 10.2 px)')
    print('   stitch: static 0.9998, predictions 0.9985')
    print('   speed: own 0.60-0.72, stitched -0.33')
    print('saved ->', OUT / 'results.json')


if __name__ == '__main__':
    main()
