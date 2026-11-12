"""Is the green/blue ordering learned, or an artefact of the preprocessing?

Background. Every stored frame of this scene is byte-identical under a
green/blue channel swap — the only colours present are black, white and red,
and all three have G equal to B. So nothing in the *data* distinguishes the two
channels. Yet the trained encoder squashes the bluer member of all eighteen
channel-swap pairs harder than the greener one.

First explanation tried: the direction is unconstrained, so it is an accident of
initialisation. That predicts a second seed should flip roughly half the pairs.
It did not — 17 of 18 held, Spearman +0.965. The explanation was wrong.

Second explanation: the preprocessing breaks the symmetry before the network
sees anything. ImageNet normalisation uses per-channel means (0.485, 0.456,
0.406), so after it, green sits ~0.22 below blue in every pixel of every
training frame. That offset is a fixed constant, identical for every seed, which
would explain the reproducibility.

This script tests both by comparing two models at a time:

    --stats imagenet   red vs red_s2        (expected, and found: no flips)
    --stats symmetric  red_sym_s1 vs _s2    the real test

With symmetric normalisation (0.5 / 0.25 per channel) the network's input truly
is swap invariant. If the preprocessing was the cause, the asymmetry should now
be seed-dependent and flip about half the pairs. If it still holds, something
else is responsible and both explanations are wrong.

Both models are scored on the same windows, in a single pass over the data.

Usage:
    uv run python scripts/analysis/colour_seed_check.py --a red_sym_s1 --b red_sym_s2 \\
        --stats symmetric
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import colour_common as C  # noqa: E402
import rolling_common as R  # noqa: E402


def swap_pairs():
    """Every pair of colours in the set that is a channel permutation of the other."""
    rgb = {n: tuple(C.COLOURS[n]) for n in C.COLOURS}
    inv = {v: k for k, v in rgb.items()}
    out = []
    for i, j, lab in ((1, 2, 'G<->B'), (0, 1, 'R<->G'), (0, 2, 'R<->B')):
        seen = set()
        for n, v in rgb.items():
            w = list(v)
            w[i], w[j] = w[j], w[i]
            w = tuple(w)
            if w == v or w not in inv:
                continue
            key = tuple(sorted([n, inv[w]]))
            if key in seen:
                continue
            seen.add(key)
            a, b = n, inv[w]
            if rgb[a][j] > rgb[b][j]:          # second member = more of channel j
                a, b = b, a
            out.append((lab, a, b))
    return out


def spread(X):
    return float(np.linalg.norm(X - X.mean(0), axis=1).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--a', default='red')
    ap.add_argument('--b', default='red_s2')
    ap.add_argument('--stats', default='imagenet', choices=['imagenet', 'symmetric'])
    ap.add_argument('--n', type=int, default=2000)
    args = ap.parse_args()
    dev = 'cuda'

    sp = np.load(C.CACHE / 'split.npz')
    test = sp['test']
    pick = test[np.linspace(0, len(test) - 1, args.n).astype(int)]
    norms = C.training_norms()
    ds_ref = C.build_dataset(C.REFERENCE, norms=norms, stats=args.stats)
    fit_w, test_w = C.map_split(ds_ref)
    win = np.concatenate([fit_w, test_w])[pick]

    models = {args.a: R.load_model(args.a, dev), args.b: R.load_model(args.b, dev)}
    print(f'models: {args.a} vs {args.b}   normalisation: {args.stats}')
    print(f'{args.n} held-out windows, the same ones for both\n')
    s = {k: {} for k in models}
    for colour in C.COLOURS:
        ds = C.build_dataset(colour, norms=norms, stats=args.stats)
        out = R.extract_multi(models, ds, win, device=dev)     # both models, one pass
        for k in models:
            s[k][colour] = spread(out[k]['cls'][:, R.HISTORY])
        print(f'  {colour:<9} encoded', end='\r', flush=True)
    gain = {k: {c: s[k][C.REFERENCE] / max(s[k][c], 1e-9) for c in C.COLOURS}
            for k in models}

    pairs = swap_pairs()
    ga, gb = gain[args.a], gain[args.b]
    print(f'\n{"swap":<8}{"pair":<20}{args.a:>16}{args.b:>16}   agree?')
    flips, rows = 0, []
    for lab, x, y in pairs:
        r_a, r_b = ga[y] / ga[x], gb[y] / gb[x]
        agree = (r_a > 1) == (r_b > 1)
        flips += (not agree)
        rows.append(dict(swap=lab, a=x, b=y, ratio_a=r_a, ratio_b=r_b, agree=bool(agree)))
        print(f'{lab:<8}{x + " / " + y:<20}{r_a:>15.2f}x{r_b:>15.2f}x   '
              f'{"same" if agree else "FLIPPED"}')
    print(f'\n{len(pairs) - flips}/{len(pairs)} keep the same direction, {flips} flip.')

    # Separately from agreement: how strong is the asymmetry inside each model?
    # Symmetric normalisation could leave the direction intact but shrink it.
    print(f'\nStrength of the asymmetry within each model '
          f'(median |log2 ratio|, 0 = channels interchangeable):')
    for k, g in ((args.a, ga), (args.b, gb)):
        for lab in ('G<->B', 'R<->G', 'R<->B'):
            v = [abs(np.log2(g[y] / g[x])) for l, x, y in pairs if l == lab]
            print(f'   {k:<12}{lab:<8}{np.median(v):>6.2f}   '
                  f'(ratio {2 ** np.median(v):.2f}x)')

    a1 = np.array([ga[c] for c in C.COLOURS])
    a2 = np.array([gb[c] for c in C.COLOURS])
    from scipy.stats import spearmanr
    print(f'\nAcross all 30 colours:')
    print(f'  Spearman of the two gain orderings: {spearmanr(a1, a2).statistic:+.3f}')
    print(f'  Pearson on log gain:                {np.corrcoef(np.log(a1), np.log(a2))[0, 1]:+.3f}')

    (C.CACHE / f'seed_check_{args.stats}.json').write_text(json.dumps(
        dict(stats=args.stats, a=args.a, b=args.b, n=args.n,
             gain={k: gain[k] for k in models}, pairs=rows, flips=flips), indent=2))
    print(f'\nwrote {C.CACHE / f"seed_check_{args.stats}.json"}')


if __name__ == '__main__':
    main()
