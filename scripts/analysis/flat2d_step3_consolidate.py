"""Experiment 3: does the ball's colour leave any signature at all?

Experiment 2 found all six trained models mutually similar at CKA ~0.994 on
the `cls` representation, with no red/green structure. That is a null result,
and a null result from one measure on one representation is the weakest kind
of claim. This experiment tries to break it three ways.

1. OTHER REPRESENTATIONS. `cls` is the backbone output. `proj` is what the
   predictor actually consumes, and in Experiment 1 it behaved differently --
   the two colours' principal planes barely overlapped there. `pred` is the
   predictor's own output, one step ahead. If colour left a signature anywhere
   it could be in these.

2. AN EXACT TEST, honestly bounded. With three seeds per colour the colour
   split is one of only ten ways to divide six models into two groups of
   three, so the smallest attainable p-value is 0.1 and this test can never
   establish equivalence. What it can do is say where the true split ranks
   among all possible splits: if colour mattered, splitting by colour should
   separate the models better than an arbitrary split does.

3. A CAPACITY CONTROL ON THE PROBES. Experiment 2 showed an untrained network
   reads ball position at R^2 0.765, so the trained models' 0.95 needs a
   baseline. This adds the other half of the control: refit each probe against
   deliberately shuffled positions. Whatever a probe scores on shuffled labels
   is capacity rather than signal (Hewitt & Liang 2019).

Usage:
    uv run python scripts/analysis/flat2d_step3_consolidate.py
"""

import itertools
import json
import os
import sys

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from flat2d_common import HISTORY, REPO  # noqa: E402
from flat2d_step2_similarity import linear_cka  # noqa: E402

CACHE = REPO / 'outputs' / 'flat2d_cache'
OUT = REPO / 'outputs' / 'flat2d_step3'
TRAINED = ['red1', 'red2', 'red3', 'green1', 'green2', 'green3']
TRUE_RED = {0, 1, 2}


def embeddings(rep):
    """One (n, d) array per trained model, plus the untrained floor."""
    def take(z):
        a = z[rep]
        return a[:, HISTORY] if a.ndim == 3 else a
    val = {m: take(np.load(CACHE / f's2_{m}_val.npz')) for m in TRAINED + ['untrained']}
    return val


def group_stat(M, red_idx):
    """mean(CKA within group) - mean(CKA across groups) for a given split."""
    green_idx = [i for i in range(6) if i not in red_idx]
    within = [M[i, j] for i, j in itertools.combinations(sorted(red_idx), 2)]
    within += [M[i, j] for i, j in itertools.combinations(sorted(green_idx), 2)]
    across = [M[i, j] for i in red_idx for j in green_idx]
    return float(np.mean(within) - np.mean(across)), within, across


def permutation_rank(M):
    """Where the true colour split ranks among all 10 possible 3/3 splits."""
    splits = []
    for comb in itertools.combinations(range(6), 3):
        if 0 not in comb:          # fix model 0 to one side, removes the mirror
            continue
        t, _, _ = group_stat(M, set(comb))
        splits.append((set(comb), t))
    obs, _, _ = group_stat(M, TRUE_RED)
    stats = [abs(t) for _, t in splits]
    rank = int(np.sum(np.array(stats) >= abs(obs)))
    return obs, rank, len(splits), sorted(stats, reverse=True)


def probe_with_control(Xf, yf, Xv, yv, seed=0):
    """Real probe and a shuffled-label control, same fit procedure for both."""
    rng = np.random.default_rng(seed)
    out = {}
    for tag, target in (('real', yf), ('shuffled', rng.permutation(yf))):
        sc = StandardScaler().fit(Xf)
        reg = RidgeCV(alphas=np.logspace(-2, 5, 25)).fit(sc.transform(Xf), target)
        out[f'{tag}_in'] = float(r2_score(target, reg.predict(sc.transform(Xf))))
        out[f'{tag}_out'] = float(r2_score(yv, reg.predict(sc.transform(Xv))))
    out['selectivity'] = out['real_out'] - out['shuffled_out']
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    results = {}

    print('=' * 72)
    print('1. CKA by representation, and whether colour explains the structure')
    print('=' * 72)
    for rep in ('cls', 'proj', 'pred'):
        E = embeddings(rep)
        M = np.zeros((6, 6))
        for i, a in enumerate(TRAINED):
            for j, b in enumerate(TRAINED):
                M[i, j] = 1.0 if i == j else linear_cka(E[a], E[b])
        floor = float(np.mean([linear_cka(E[m], E['untrained']) for m in TRAINED]))
        obs, rank, n_splits, stats = permutation_rank(M)
        _, within, across = group_stat(M, TRUE_RED)
        results[rep] = {
            'matrix': M.tolist(), 'floor_vs_untrained': floor,
            'within_mean': float(np.mean(within)), 'within_sd': float(np.std(within)),
            'across_mean': float(np.mean(across)), 'across_sd': float(np.std(across)),
            'observed_stat': obs, 'rank_of_true_split': rank, 'n_splits': n_splits,
            'all_split_stats': stats,
        }
        print(f'\n[{rep}]  floor (vs untrained) {floor:.4f}')
        print(f'  same-colour pairs   mean {np.mean(within):.4f}  sd {np.std(within):.4f}  n={len(within)}')
        print(f'  cross-colour pairs  mean {np.mean(across):.4f}  sd {np.std(across):.4f}  n={len(across)}')
        print(f'  within - across     {obs:+.5f}')
        print(f'  |stat| of the true colour split ranks {rank}/{n_splits} among all 3/3 splits')
        print(f'  all split statistics (largest first): '
              f'{", ".join(f"{s:.5f}" for s in stats)}')

    print('\n' + '=' * 72)
    print('2. Probe capacity control (Hewitt & Liang): real vs shuffled labels')
    print('=' * 72)
    fit = {m: np.load(CACHE / f's2_{m}_fit.npz') for m in TRAINED + ['untrained']}
    val = {m: np.load(CACHE / f's2_{m}_val.npz') for m in TRAINED + ['untrained']}
    yf, yv = fit['red1']['x'], val['red1']['x']
    probes = {}
    for rep in ('cls', 'proj'):
        print(f'\n[{rep}]')
        print(f'  {"model":<12}{"real out":>10}{"shuffled out":>14}'
              f'{"shuffled in":>13}{"selectivity":>13}')
        probes[rep] = {}
        for m in TRAINED + ['untrained']:
            Xf, Xv = fit[m][rep][:, HISTORY], val[m][rep][:, HISTORY]
            r = probe_with_control(Xf, yf, Xv, yv)
            probes[rep][m] = r
            print(f'  {m:<12}{r["real_out"]:>10.4f}{r["shuffled_out"]:>14.4f}'
                  f'{r["shuffled_in"]:>13.4f}{r["selectivity"]:>13.4f}')
    results['probes'] = probes

    with open(OUT / 'step3_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nsaved -> {OUT / "step3_results.json"}')


if __name__ == '__main__':
    main()
