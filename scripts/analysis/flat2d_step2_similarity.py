"""Experiment 2: what do 'similar' and 'different' mean for these models?

A similarity score between two networks is uninterpretable on its own. It is
never 1 (independent runs never coincide) and never 0 (same architecture, same
input statistics), so it needs a scale. This experiment measures the two ends
of that scale using models that differ in a known way:

    CEILING  red_i vs red_j      identical data and data order, different
                                 initialisation only. The most two separately
                                 trained models of this family can agree.
    FLOOR    red_i vs untrained  same architecture, same inputs, no training.
                                 Whatever similarity comes for free.

Three measures, because they fail differently and a claim resting on one is
weak:

    CKA        linear Centered Kernel Alignment (Kornblith et al. 2019).
               Invariant to rotation and isotropic scaling, which is what
               Experiment 1 showed is needed: two representations there
               differed by an affine map yet scored at chance on raw distance.
    PRED       linear predictivity. Fit a ridge map from one model's embedding
               to the other's on the fit split, score R^2 on validation.
               Directional, so both directions are reported.
    PROBE      transfer of the ball-position readout. Fit the probe on model A,
               apply it to model B. The most task-relevant of the three, and
               the one Experiment 1 already used within a single model.

All models are encoded on the frames they were trained on, and every pair is
compared window-by-window on the same validation windows.

Usage:
    uv run python scripts/analysis/flat2d_step2_similarity.py
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
OUT = REPO / 'outputs' / 'flat2d_step2'
MODELS = ['red1', 'red2', 'red3', 'green1', 'green2', 'green3', 'untrained']


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Linear CKA between two sets of representations of the SAME inputs.

    args:
        X, Y: (n, p) and (n, q); row i of each must be the same window.
    returns:
        similarity in [0, 1]
    """
    X = X - X.mean(0, keepdims=True)
    Y = Y - Y.mean(0, keepdims=True)
    num = np.linalg.norm(Y.T @ X, ord='fro') ** 2
    den = (np.linalg.norm(X.T @ X, ord='fro')
           * np.linalg.norm(Y.T @ Y, ord='fro'))
    return float(num / den)


def predictivity(Xf, Yf, Xv, Yv, alpha=1.0) -> float:
    """R^2 of a ridge map X -> Y, fitted on the fit split, scored on validation."""
    sx = StandardScaler().fit(Xf)
    reg = Ridge(alpha=alpha).fit(sx.transform(Xf), Yf)
    pred = reg.predict(sx.transform(Xv))
    ss_res = ((pred - Yv) ** 2).sum()
    ss_tot = ((Yv - Yv.mean(0)) ** 2).sum()
    return float(1 - ss_res / ss_tot)


def probe_transfer(Xf, yf, Xv, yv) -> float:
    sc = StandardScaler().fit(Xf)
    reg = RidgeCV(alphas=np.logspace(-2, 5, 25)).fit(sc.transform(Xf), yf)
    return float(r2_score(yv, reg.predict(sc.transform(Xv))))


def main(rep='cls'):
    OUT.mkdir(parents=True, exist_ok=True)
    fit = {m: np.load(CACHE / f's2_{m}_fit.npz') for m in MODELS}
    val = {m: np.load(CACHE / f's2_{m}_val.npz') for m in MODELS}

    # every model was encoded on the same window indices, so ground truth must
    # agree across all of them — the pairing the whole comparison rests on
    x_ref = val['red1']['x']
    for m in MODELS:
        assert np.array_equal(val[m]['x'], x_ref), f'{m} is not window-aligned'
    print(f'{len(x_ref)} validation windows, {len(fit["red1"]["x"])} fit windows, '
          f'pairing verified across {len(MODELS)} models\n')

    E = {m: {'fit': fit[m][rep][:, HISTORY], 'val': val[m][rep][:, HISTORY]}
         for m in MODELS}
    yf, yv = fit['red1']['x'], val['red1']['x']

    res = {}
    for a, b in itertools.combinations(MODELS, 2):
        res[f'{a}|{b}'] = {
            'cka': linear_cka(E[a]['val'], E[b]['val']),
            'pred_a_to_b': predictivity(E[a]['fit'], E[b]['fit'], E[a]['val'], E[b]['val']),
            'pred_b_to_a': predictivity(E[b]['fit'], E[a]['fit'], E[b]['val'], E[a]['val']),
            'probe_a_to_b': probe_transfer(E[a]['fit'], yf, E[b]['val'], yv),
            'probe_b_to_a': probe_transfer(E[b]['fit'], yf, E[a]['val'], yv),
        }
    for m in MODELS:
        res[f'{m}|{m}'] = {'cka': 1.0,
                           'probe_self': probe_transfer(E[m]['fit'], yf, E[m]['val'], yv)}

    def show(title, pairs):
        print(title)
        print(f'  {"pair":<22}{"CKA":>8}{"PRED A→B":>11}{"PRED B→A":>11}'
              f'{"PROBE A→B":>11}{"PROBE B→A":>11}')
        for a, b in pairs:
            k = f'{a}|{b}' if f'{a}|{b}' in res else f'{b}|{a}'
            r = res[k]
            print(f'  {a}↔{b:<16}{r["cka"]:>8.4f}{r["pred_a_to_b"]:>11.4f}'
                  f'{r["pred_b_to_a"]:>11.4f}{r["probe_a_to_b"]:>11.4f}'
                  f'{r["probe_b_to_a"]:>11.4f}')
        print()

    reds = ['red1', 'red2', 'red3']
    show('CEILING — same data, different initialisation:',
         list(itertools.combinations(reds, 2)))
    show('FLOOR — trained vs untrained:', [(r, 'untrained') for r in reds])
    show('(preview, Experiment 3) matched-init red vs green:',
         [(f'red{i}', f'green{i}') for i in (1, 2, 3)])

    print('self-probe R² (each model reading its own colour):')
    for m in MODELS:
        print(f'  {m:<12}{res[f"{m}|{m}"]["probe_self"]:>8.4f}')

    with open(OUT / f'similarity_{rep}.json', 'w') as f:
        json.dump({'rep': rep, 'n_val': int(len(yv)), 'pairs': res}, f, indent=2)
    print(f'\nsaved -> {OUT / f"similarity_{rep}.json"}')


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'cls')
