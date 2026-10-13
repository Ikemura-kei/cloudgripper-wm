"""Measure what the LeWM predictor actually propagates, in millimetres.

The predictor's training loss is a mean squared error between its output and
the encoder's embedding of the next frame. That number is hard to interpret:
it is averaged isotropically over all 192 dimensions, while the only thing in
this scene that moves -- the ball -- occupies about 1% of the embedding's
magnitude. A loss can therefore look excellent while the physics it is
supposed to carry is wrong.

Two measurements, deliberately independent:

  1. The loss itself, normalised by how much the embedding actually changes
     from one frame to the next. This needs no probe and so cannot be blamed
     on probe error. It answers: of the frame-to-frame change the predictor
     is asked to produce, what fraction does it get wrong?

  2. A linear position probe applied to the predictor's output, which converts
     the prediction into millimetres and asks whether the predicted
     displacement tracks the real one.

Probe validity matters here and is easy to get wrong. A position probe fitted
on a single frame index is accurate only near that index -- 0.57 mm at the
fitted frame, 14 mm six frames away -- because each index sees a different
slice of the track. Fitting across all frame indices pooled gives a probe that
is flat to 0.8 mm everywhere, which is what makes a 5 mm displacement
measurable at all. This script pools.

Reads the embedding cache written by cache_emb (see --cache).
"""

import argparse

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

TRAVEL_AXIS = 1
SETS = ['train', 'held-out', 'OOD-slow', 'OOD-fast']


def fit_pooled_probe(emb, pos):
    """Position probe fitted over every frame index, not just one."""
    x = emb.reshape(-1, emb.shape[-1])
    scaler = StandardScaler().fit(x)
    reg = RidgeCV(alphas=np.logspace(-3, 5, 30)).fit(
        scaler.transform(x), pos.reshape(-1))
    return scaler, reg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', default='outputs/ball_probe/emb_cache.npz')
    ap.add_argument('--history', type=int, default=6)
    args = ap.parse_args()

    d = np.load(args.cache)
    H = args.history
    emb, pos = d['train|proj'], d['train|pos'][:, :, TRAVEL_AXIS]
    scaler, reg = fit_pooled_probe(emb, pos)
    read = lambda e: reg.predict(scaler.transform(e))

    # how much embedding distance the whole track spans, for context on (1)
    w = np.linalg.norm(reg.coef_ / scaler.scale_)
    span = (pos.max() - pos.min()) / w
    magnitude = np.sqrt(emb.var() * emb.shape[-1])
    print(f'Ball position spans {span:.3f} embedding units out of a typical '
          f'magnitude of {magnitude:.2f},\nso it accounts for '
          f'{span / magnitude * 100:.2f}% of the embedding. The MSE loss '
          f'weights all\n{emb.shape[-1]} dimensions equally.\n')

    print('(1) predictor loss, with no probe involved')
    print(f"{'set':10s} {'pred MSE':>10s} {'frame-to-frame':>15s} {'missed':>8s}")
    for s in SETS:
        e, pr = d[f'{s}|proj'], d[f'{s}|pred']
        mse = np.mean((pr - e[:, H]) ** 2)
        change = np.mean((e[:, H] - e[:, H - 1]) ** 2)
        print(f'{s:10s} {mse:10.5f} {change:15.5f} {mse / change * 100:7.2f}%')

    print('\n(2) the same prediction, read out in millimetres')
    print(f"{'set':10s} {'true step':>10s} {'pred step':>10s} {'spread':>9s} "
          f"{'corr':>7s} {'probe err':>10s}")
    for s in SETS:
        e, pr, p = d[f'{s}|proj'], d[f'{s}|pred'], d[f'{s}|pos'][:, :, TRAVEL_AXIS]
        true_step = p[:, H] - p[:, H - 1]
        pred_step = read(pr) - read(e[:, H - 1])
        print(f'{s:10s} {true_step.mean() * 1000:7.2f} mm '
              f'{pred_step.mean() * 1000:7.2f} mm {pred_step.std() * 1000:6.2f} mm '
              f'{np.corrcoef(true_step, pred_step)[0, 1]:7.3f} '
              f'{np.mean(np.abs(read(e[:, H]) - p[:, H])) * 1000:7.2f} mm')
    print('\n"probe err" is the probe reading the TRUE next embedding -- the '
          'floor below\nwhich this instrument cannot measure.')


if __name__ == '__main__':
    main()
