"""Experiment 4, re-run: can red's machinery run on green's state?

Experiment 4 fitted an affine map between two models' embeddings and showed
one model's position readout worked on the other's state. Experiment 5 then
applied that same map to the predictor outputs and compared them.

Both of those compare representations. This adds the stronger test: feed
green's context through the map and into RED'S OWN PREDICTOR, and ask whether
what comes out is the correct next state -- not whether it resembles green's
prediction, but whether it is right.

    green context -> T -> red's predictor -> where is the ball?

The datasets are frame-for-frame matched, so the correct answer is the same
for both colours and can be checked directly against ground truth.

Four conditions, so the stitch is bracketed:

    red ctx   -> red predictor            how well red does on its own state
    T(green)  -> red predictor            the question
    green ctx -> red predictor, no map    what the map is worth
    green ctx -> green predictor          how well green does on its own state

Usage:
    uv run python scripts/analysis/rolling_step4_stitching.py
"""

import json
import os
import sys

import numpy as np
import torch
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rolling_common as R  # noqa: E402

OUT = R.REPO / 'outputs' / 'rolling_step4'


def fit_map(src, dst, alpha=1.0):
    """Affine map between two embedding spaces, fitted on the fit split."""
    sc = StandardScaler().fit(src)
    reg = Ridge(alpha=alpha).fit(sc.transform(src), dst)
    return lambda z: reg.predict(sc.transform(z))


def fit_probe(X, y):
    sc = StandardScaler().fit(X)
    reg = RidgeCV(alphas=np.logspace(-2, 5, 25)).fit(sc.transform(X), y)
    return lambda Z: reg.predict(sc.transform(Z))


@torch.no_grad()
def run_predictor(model, ctx, act, device='cuda', batch=256):
    """Push a context through a model's predictor. ctx is (N, HISTORY, 192)."""
    out = []
    for lo in range(0, len(ctx), batch):
        c = torch.tensor(ctx[lo:lo + batch], dtype=torch.float32, device=device)
        a = torch.tensor(act[lo:lo + batch], dtype=torch.float32, device=device)
        ae = model.action_encoder(a)
        p = model.predict(c.to(ae.dtype), ae[:, :R.HISTORY])
        out.append(p[:, -1].float().cpu().numpy())
    return np.concatenate(out)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    dev = 'cuda'
    F = {c: np.load(R.CACHE / f'{c}_fit.npz') for c in ('red', 'green')}
    V = {c: np.load(R.CACHE / f'{c}_val.npz') for c in ('red', 'green')}
    xn_f = F['red']['pos_traj'][:, R.HISTORY, R.TRAVEL_AXIS]
    xn_v = V['red']['pos_traj'][:, R.HISTORY, R.TRAVEL_AXIS]
    xp_v = V['red']['pos_traj'][:, R.HISTORY - 1, R.TRAVEL_AXIS]
    step = float(np.mean(np.abs(xn_v - xp_v)) * R.PX)
    print(f'{len(xn_v)} held-out windows, the ball advances {step:.1f} px per step\n')

    # the map is fitted on static frames only, pooling every context position,
    # so it has never seen a prediction and is not tuned to one frame index
    src = np.concatenate([F['green']['proj'][:, t] for t in range(R.HISTORY + 1)])
    dst = np.concatenate([F['red']['proj'][:, t] for t in range(R.HISTORY + 1)])
    T_g2r = fit_map(src, dst)
    T_r2g = fit_map(dst, src)
    held = r2_score(np.concatenate([V['red']['proj'][:, t] for t in range(R.HISTORY + 1)]),
                    T_g2r(np.concatenate([V['green']['proj'][:, t] for t in range(R.HISTORY + 1)])),
                    multioutput='variance_weighted')
    print(f'the map green->red reproduces red\'s embeddings at R2 {held:.4f} on held-out windows\n')

    red = R.load_model('red', dev)
    green = R.load_model('green', dev)
    # the action is (HISTORY+1, frameskip * 5) = (4, 25) here, not (4, 5):
    # lewm.py sets action_encoder.input_dim = frameskip * dataset.get_dim('action').
    # Every entry is zero because the scene has no actuation, but the shape
    # still has to be right or the encoder's Conv1d rejects it.
    adim = red.action_encoder.input_dim
    act_f = np.zeros((len(xn_f), R.HISTORY + 1, adim), np.float32)
    act_v = np.zeros((len(xn_v), R.HISTORY + 1, adim), np.float32)

    def ctx(colour, split, mapped=None):
        z = (F if split == 'fit' else V)[colour]['proj'][:, :R.HISTORY]
        if mapped is None:
            return z
        flat = mapped(z.reshape(-1, z.shape[-1]))
        return flat.reshape(z.shape[0], R.HISTORY, -1)

    # the tag is explicit, not derived from the description: deriving it from
    # the first word collided ('green state -> red predictor, no map' and
    # 'green state -> green predictor' both produced pred_green_green.npz,
    # so the second silently overwrote the first)
    conditions = [
        ('red_on_red', 'red state -> red predictor', red, 'red', None),
        ('red_on_mapped_green', 'T(green) -> red predictor', red, 'green', T_g2r),
        ('red_on_raw_green', 'green state -> red predictor, no map', red, 'green', None),
        ('green_on_green', 'green state -> green predictor', green, 'green', None),
    ]
    res = {'step_px': step, 'map_r2': float(held)}
    print(f'{"condition":<40}{"R2":>9}{"error px":>11}')
    for tag, name, model, colour, mp in conditions:
        pf = run_predictor(model, ctx(colour, 'fit', mp), act_f, dev)
        pv = run_predictor(model, ctx(colour, 'val', mp), act_v, dev)
        # the readout is fitted on whatever this pipeline produces on the fit
        # split, then scored on held-out windows against the true next position
        probe = fit_probe(pf, xn_f)
        p = probe(pv)
        r2 = float(r2_score(xn_v, p)); px = float(np.mean(np.abs(p - xn_v)) * R.PX)
        res[name] = {'r2': r2, 'px': px}
        print(f'{name:<40}{r2:>9.4f}{px:>11.1f}')
        f = OUT / f'pred_{tag}.npz'
        assert not f.exists() or tag in str(f), 'tag collision'
        np.savez_compressed(f, pred=pv)

    print(f'\nbaseline "assume the ball did not move": {step:.1f} px')
    json.dump(res, open(OUT / 'results.json', 'w'), indent=2)
    print('saved ->', OUT / 'results.json')


if __name__ == '__main__':
    main()
