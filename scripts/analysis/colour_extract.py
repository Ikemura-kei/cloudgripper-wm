"""Encode every colour dataset with the *red* model and cache the result.

One model throughout. The only thing that changes between the cached files is
the colour of the ball in the pixels that model was shown — so any difference
between two caches is caused by colour and nothing else. In particular there
is no second network here, so the seed nuisance that complicated the
cross-model comparison in Experiment 4 cannot arise.

Usage:
    uv run python scripts/analysis/colour_extract.py
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import colour_common as C  # noqa: E402
import rolling_common as R  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stats', default='imagenet',
                    choices=['imagenet', 'symmetric'],
                    help='which version of the study to encode')
    args = ap.parse_args()
    C.use_stats(args.stats)
    dev = 'cuda'
    ref_run = C.REF_RUN[args.stats]
    print(f'normalisation: {args.stats} | encoder: {ref_run} | cache: {C.CACHE}')
    print('loading training-set normalisers...', flush=True)
    norms = C.training_norms()
    model = {'red_model': R.load_model(ref_run, dev)}

    idx_ref = None
    for colour in C.COLOURS:
        f = C.CACHE / f'{colour}.npz'
        ds = C.build_dataset(colour, norms=norms)
        fit, test = C.map_split(ds)
        idx = np.concatenate([fit, test])
        if idx_ref is None:
            idx_ref, n_fit = idx, len(fit)
            np.savez(C.CACHE / 'split.npz', fit=np.arange(len(fit)),
                     test=np.arange(len(fit), len(idx)))
            print(f'{len(fit)} fit windows, {len(test)} test windows per colour',
                  flush=True)
        elif not np.array_equal(idx, idx_ref):
            raise AssertionError(f'{colour}: window layout differs from red')
        if f.exists():
            print(f'  {f.name} cached, skipping', flush=True)
            continue
        print(f'  encoding {colour} ({len(idx)} windows)...', flush=True)
        out = R.extract_multi(model, ds, idx, device=dev)['red_model']
        np.savez_compressed(f, **out)
        print(f'  wrote {f.name}', flush=True)

    # the whole study rests on window i being the same scene in every colour
    ref = np.load(C.CACHE / f'{C.REFERENCE}.npz')
    for colour in C.COLOURS:
        if colour == C.REFERENCE:
            continue
        other = np.load(C.CACHE / f'{colour}.npz')
        for k in ('pos_traj', 'x', 'speed'):
            if not np.array_equal(ref[k], other[k]):
                raise AssertionError(f'{colour}: {k} differs from {C.REFERENCE}')
    print(f'pairing verified: ground truth identical across all '
          f'{len(C.COLOURS)} colours')


if __name__ == '__main__':
    main()
