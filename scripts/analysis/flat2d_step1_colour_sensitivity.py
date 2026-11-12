"""Step 1: does a single trained model even notice the ball's colour?

QUESTION
    Take ONE model (red, init_seed=1). Feed it the red frames, then the green
    frames. The two inputs are the same physical scene — identical ball
    positions, speeds and pixel masks — differing only in hue. How far does the
    representation move?

WHY THIS COMES FIRST
    It needs no new training and no second model, so it carries none of the
    init/optimisation noise that the red-vs-green model comparison has to
    control for. If the encoder is already colour-invariant, that bounds what
    the two-model comparison can possibly show.

WHAT IS MEASURED
    1. Displacement ratio. ||e_red(i) - e_green(i)|| (same scene, recoloured)
       against ||e_red(i) - e_red(j)|| for i != j (different scene, same
       colour). A raw distance means nothing; the ratio says how large the
       colour perturbation is relative to the representation's own spread.
    2. Paired cosine similarity, against a random-pair baseline.
    3. Nearest-neighbour retrieval. For each green-encoded window, is its
       nearest neighbour among the red-encoded windows the *same* window? This
       is the sharpest form of the question: if colour were invisible, every
       query retrieves itself.
    4. Probe transfer. Fit a ridge probe from representation to ball position
       on red frames (training episodes only), then apply it unchanged to
       green frames. If the readout survives, the colour shift does not move
       the information the probe reads.

    Everything is reported for both `cls` (ViT backbone) and `proj` (projector
    output, which is what the predictor actually consumes).

Usage:
    uv run python scripts/analysis/flat2d_step1_colour_sensitivity.py
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from flat2d_common import (  # noqa: E402
    HISTORY, TRAVEL_AXIS, REPO, build_dataset, cached_extract, load_model,
    val_window_indices, check_pairing,
)

from sklearn.linear_model import RidgeCV  # noqa: E402
from sklearn.metrics import r2_score  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402


def train_window_indices(ds):
    """Complement of the validation windows — where probes are fitted."""
    val = set(val_window_indices(ds).tolist())
    return np.array([i for i in range(len(ds)) if i not in val])


def fit_probe(x, y):
    sc = StandardScaler().fit(x)
    return sc, RidgeCV(alphas=np.logspace(-2, 5, 25)).fit(sc.transform(x), y)


def apply_probe(sc, reg, x):
    return reg.predict(sc.transform(x))


def displacement_stats(a, b, rng):
    """Colour-induced displacement against the representation's own spread.

    args:
        a, b: (N, D) paired representations (same scenes, two colours)
        rng: numpy Generator for the scene-distance baseline
    returns:
        dict of summary statistics
    """
    d_colour = np.linalg.norm(a - b, axis=1)
    perm = rng.permutation(len(a))
    keep = perm != np.arange(len(a))
    d_scene = np.linalg.norm(a[keep] - a[perm[keep]], axis=1)

    an = a / np.linalg.norm(a, axis=1, keepdims=True)
    bn = b / np.linalg.norm(b, axis=1, keepdims=True)
    cos_pair = np.sum(an * bn, axis=1)
    cos_rand = np.sum(an[keep] * an[perm[keep]], axis=1)
    return {
        'dist_colour_mean': float(d_colour.mean()),
        'dist_scene_mean': float(d_scene.mean()),
        'ratio': float(d_colour.mean() / d_scene.mean()),
        'cos_paired_mean': float(cos_pair.mean()),
        'cos_paired_min': float(cos_pair.min()),
        'cos_random_mean': float(cos_rand.mean()),
        '_d_colour': d_colour,
        '_d_scene': d_scene,
        '_cos_pair': cos_pair,
        '_cos_rand': cos_rand,
    }


def nn_retrieval(a, b):
    """Top-1 self-retrieval: query b against gallery a, correct iff same index."""
    an = torch.tensor(a / np.linalg.norm(a, axis=1, keepdims=True), dtype=torch.float32)
    bn = torch.tensor(b / np.linalg.norm(b, axis=1, keepdims=True), dtype=torch.float32)
    sim = bn @ an.T
    top1 = sim.argmax(1).numpy()
    return float((top1 == np.arange(len(a))).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--init-seed', type=int, default=1)
    ap.add_argument('--model-dataset', default='red')
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--train-windows', type=int, default=4000)
    ap.add_argument('--out', default=str(REPO / 'outputs' / 'flat2d_step1'))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = load_model(args.model_dataset, args.init_seed, device=dev)

    ds_red, norms = build_dataset('flat2d_ball_red')
    ds_green, _ = build_dataset('flat2d_ball_green', norms=norms)
    assert len(ds_red) == len(ds_green), 'datasets must be window-aligned'

    val_idx = val_window_indices(ds_red)
    rng = np.random.default_rng(0)
    tr_pool = train_window_indices(ds_red)
    tr_idx = np.sort(rng.choice(tr_pool, size=min(args.train_windows, len(tr_pool)),
                                replace=False))

    print(f'model: {args.model_dataset} init{args.init_seed} | '
          f'val windows {len(val_idx)} | probe-fit windows {len(tr_idx)}')

    # --- extract: same windows, two colours, one model -------------------
    mdl = f'{args.model_dataset}{args.init_seed}'
    out = {}
    for tag, ds in (('red', ds_red), ('green', ds_green)):
        out[('val', tag)] = cached_extract(
            f'm{mdl}_val_{tag}', lambda: model, lambda ds=ds: ds, val_idx, args.batch, dev)
        print(f'  encoded val/{tag}', flush=True)
    out[('train', 'red')] = cached_extract(
        f'm{mdl}_train_red', lambda: model, lambda: ds_red, tr_idx, args.batch, dev)
    print('  encoded train/red (for probe fitting)', flush=True)

    check_pairing(out[('val', 'red')], out[('val', 'green')], 'val red vs green')
    print('  pairing verified: ball_pos and ball_speed identical across colours')

    results = {'model': f'{args.model_dataset}_init{args.init_seed}',
               'n_val_windows': int(len(val_idx)), 'n_probe_fit': int(len(tr_idx))}

    last = HISTORY  # the window's final frame
    for rep in ('cls', 'proj'):
        a = out[('val', 'red')][rep][:, last]
        b = out[('val', 'green')][rep][:, last]
        st = displacement_stats(a, b, np.random.default_rng(1))
        st['nn_self_retrieval'] = nn_retrieval(a, b)

        # probe transfer: fit on red frames, apply to both
        xtr = out[('train', 'red')][rep][:, last]
        ytr = out[('train', 'red')]['x']        # scalar: travel-axis position, last frame
        sc, reg = fit_probe(xtr, ytr)
        ytrue = out[('val', 'red')]['x']
        assert ytr.ndim == 1 and ytrue.ndim == 1, 'probe target must be scalar'
        pr_red = apply_probe(sc, reg, a)
        pr_green = apply_probe(sc, reg, b)
        st['probe_r2_red'] = float(r2_score(ytrue, pr_red))
        st['probe_r2_green'] = float(r2_score(ytrue, pr_green))
        st['probe_mae_red'] = float(np.mean(np.abs(pr_red - ytrue)))
        st['probe_mae_green'] = float(np.mean(np.abs(pr_green - ytrue)))
        st['_pred_red'], st['_pred_green'], st['_true'] = pr_red, pr_green, ytrue
        results[rep] = st

        print(f'\n[{rep}]')
        print(f'  colour displacement / scene spread : {st["ratio"]:.4f}')
        print(f'  cosine(paired)   {st["cos_paired_mean"]:.4f}  (min {st["cos_paired_min"]:.4f})')
        print(f'  cosine(random)   {st["cos_random_mean"]:.4f}')
        print(f'  NN self-retrieval (green->red)     : {st["nn_self_retrieval"]*100:.2f}%')
        print(f'  position probe R2  red {st["probe_r2_red"]:.4f} | green {st["probe_r2_green"]:.4f}')
        print(f'  position probe MAE red {st["probe_mae_red"]:.5f} | green {st["probe_mae_green"]:.5f} (image widths)')

    np.savez_compressed(
        os.path.join(args.out, 'step1_arrays.npz'),
        **{f'{rep}|{k.lstrip("_")}': v for rep in ('cls', 'proj')
           for k, v in results[rep].items() if k.startswith('_')},
        pos_true=out[('val', 'red')]['x'], speed=out[('val', 'red')]['speed'],
        val_idx=val_idx,
    )
    clean = {k: ({kk: vv for kk, vv in v.items() if not kk.startswith('_')}
                 if isinstance(v, dict) else v) for k, v in results.items()}
    with open(os.path.join(args.out, 'step1_results.json'), 'w') as f:
        json.dump(clean, f, indent=2)
    print(f'\nsaved -> {args.out}')


if __name__ == '__main__':
    main()
