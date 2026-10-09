"""Probe a trained LeWM ball model's latent for ball position and speed.

LeWM has no decoder, so a prediction is a 192-d embedding rather than an image.
To say anything in metres we fit a small ridge regressor from embedding to the
recorded ground truth, then apply that same probe to embeddings from datasets
the model never saw — including two whose speeds lie outside the band it was
trained on.

Position and speed deliberately use different inputs. Position is visible in a
single frame; speed is not visible in any single frame at all, so a probe fed
one frame embedding would fail for reasons of physics rather than anything
about the model. Speed therefore gets three inputs that span time, which ask
different questions:

    diff   emb[t] - emb[t-1]        is displacement linear in the embedding?
    ctx    concat(emb[0..H-1])      can a linear map use the whole context?
    pred   predictor output         did the predictor's own state encode it?

Every probe is run against two representations. LeWM's encode() returns
projector(cls_token), but a projection head is trained to drop whatever the
objective does not need, so in self-supervised work the representation that
gets probed is the backbone's, not the head's. `cls` is the ViT CLS token and
`proj` is what the model actually predicts in; comparing them says whether the
head is discarding the factor or the backbone never had it.

A fourth speed estimator is derived rather than fitted: apply the *position*
probe at t and t-1 and divide by the control timestep. A probe fitted on speeds
0.045-0.20 cannot by construction extrapolate to 0.28, so a direct speed probe
conflates "the latent lacks speed" with "the linear map cannot reach that far".
The derived estimate has no such ceiling, and isolates the representation.

Fitted on the training episodes only; reported on all four sets. Because the
probe is linear, fitting the targets jointly or separately gives identical
coefficients, so they are fitted together and scored per target.

Usage:
    uv run python scripts/analysis/probe_ball_latent.py
    uv run python scripts/analysis/probe_ball_latent.py --windows-per-set 6000
"""

import argparse
import os

os.environ.setdefault('MUJOCO_GL', 'egl')

import numpy as np
import torch
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import stable_pretraining as spt
import stable_worldmodel as swm
from stable_pretraining import data as dt
from stable_worldmodel.data import column_normalizer


DEFAULT_CKPT = ('datasets/checkpoints/lewm_ball-26-10-08-23-51-41-rzr/'
                'weights_epoch_100.pt')
SETS = {
    'train': 'cloudgripper_mj_ball',
    'held-out': 'cloudgripper_mj_ball_heldout',
    'OOD-slow': 'cloudgripper_mj_ball_ood_slow',
    'OOD-fast': 'cloudgripper_mj_ball_ood_fast',
}
TRAVEL_AXIS = 1  # ball_pos is (x, y, z); it travels along y
CONTROL_DT = 0.05  # s per step; matches the env's control_timestep


def build_dataset(name, root, history, img_size, norms=None):
    """Load a ball dataset with the same transform the model was trained with."""
    ds = swm.data.load_dataset(
        f'{root}/.stable_worldmodel/datasets/{name}.lance',
        transform=None,
        num_steps=history + 1,
        frameskip=1,
        keys_to_load=['pixels', 'action', 'state', 'ball_pos', 'ball_speed'],
    )
    stats = dt.dataset_stats.ImageNet
    tfs = [dt.transforms.Compose(
        dt.transforms.ToImage(**stats, source='pixels', target='pixels'),
        dt.transforms.Resize(img_size, source='pixels', target='pixels'),
    )]
    # Actions are inert in this scene but the predictor still consumes them, so
    # they must be normalised exactly as in training — with the *training* set's
    # statistics, never each set's own, or the eval sets get a different mapping.
    norms = norms if norms is not None else {
        c: column_normalizer(ds, c, c) for c in ('action', 'state')
    }
    tfs.extend(norms.values())
    ds.transform = spt.data.transforms.Compose(*tfs)
    return ds, norms


@torch.no_grad()
def extract(model, ds, n_windows, history, batch, device, seed=0):
    """Encode a sample of windows, returning embeddings and ground truth."""
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(ds), size=min(n_windows, len(ds)), replace=False))

    frame_cls, frame_emb, pred_emb, pos, speed = [], [], [], [], []
    for lo in range(0, len(idx), batch):
        rows = [ds[int(i)] for i in idx[lo:lo + batch]]
        pix = torch.stack([r['pixels'] for r in rows]).to(device)
        act = torch.stack([torch.as_tensor(r['action']) for r in rows]).float().to(device)
        act = torch.nan_to_num(act, 0.0)

        b, t = pix.shape[:2]
        flat = pix.reshape(b * t, *pix.shape[2:]).to(
            next(model.encoder.parameters()).dtype)
        enc = model.encoder(flat, interpolate_pos_encoding=True)
        cls = enc.last_hidden_state[:, 0]              # backbone representation
        proj = model.projector(cls)                    # what the model predicts in
        cls = cls.reshape(b, t, -1).float()
        proj = proj.reshape(b, t, -1).float()
        act_emb = model.action_encoder(act)
        # the predictor's own view: what it produces from the full context
        nxt = model.predict(proj[:, :history].to(act_emb.dtype),
                            act_emb[:, :history])
        frame_cls.append(cls.cpu().numpy())
        frame_emb.append(proj.cpu().numpy())
        pred_emb.append(nxt[:, -1].float().cpu().numpy())
        pos.append(np.stack([np.asarray(r['ball_pos']) for r in rows]))
        speed.append(np.stack([np.asarray(r['ball_speed']).reshape(-1)[0] for r in rows]))

    return (np.concatenate(frame_cls), np.concatenate(frame_emb),
            np.concatenate(pred_emb), np.concatenate(pos),
            np.concatenate(speed))


def fit_probe(x_train, y_train):
    """Standardise, then ridge with alpha chosen by cross-validation."""
    scaler = StandardScaler().fit(x_train)
    model = RidgeCV(alphas=np.logspace(-2, 5, 25)).fit(scaler.transform(x_train), y_train)
    return scaler, model


def score(scaler, model, x, y):
    pred = model.predict(scaler.transform(x))
    return {
        'r2': r2_score(y, pred),
        'mae': float(np.mean(np.abs(pred - y))),
        'rel': float(np.mean(np.abs(pred - y)) / np.mean(np.abs(y))),
        'bias': float(np.mean(pred - y)),   # systematic under/over-read matters for OOD
        'pred': pred,
    }


def derived_speed(scaler, reg, d, H):
    """Speed from the position probe's own displacement -- no speed fitting."""
    e = d['cls']
    at = lambda k: reg.predict(scaler.transform(e[:, k]))
    return (at(H - 1) - at(H - 2)) / CONTROL_DT


def make_plots(results, data, derived, H, out):
    """Parity plots: the two estimators that are worth looking at."""
    order = list(data)
    colors = dict(zip(order, ['#2a6ebb', '#4caf50', '#c77dff', '#e8702a']))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    panels = [
        ('position [cls]', 'pos', 'ball position y (m)'),
        ('speed diff [cls]', 'speed', 'speed, fitted probe (m/s)'),
        (None, 'speed', 'speed, derived from position probe (m/s)'),
    ]
    for ax, (key, target, title) in zip(axes, panels):
        for label in order:
            d = data[label]
            if target == 'pos':
                y, yhat = d['pos'][:, H - 1, TRAVEL_AXIS], results[key][label]['pred']
            elif key is None:
                y, yhat = d['speed'], derived[label]
            else:
                y, yhat = d['speed'], results[key][label]['pred']
            ax.scatter(y, yhat, s=4, alpha=.35, color=colors[label], label=label,
                       linewidths=0)
        lo = min(ax.get_xlim()[0], ax.get_ylim()[0])
        hi = max(ax.get_xlim()[1], ax.get_ylim()[1])
        ax.plot([lo, hi], [lo, hi], 'k--', lw=1, zorder=0)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_xlabel('ground truth'); ax.set_ylabel('probe estimate')
        ax.set_title(title, fontsize=10)
    axes[0].legend(markerscale=3, fontsize=8, loc='upper left')
    fig.suptitle('Linear probes of the LeWM ball latent (ViT backbone CLS token)',
                 fontsize=11)
    fig.tight_layout()
    path = os.path.join(out, 'ball_probe.png')
    fig.savefig(path, dpi=140)
    print(f'saved plot to {path}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', default=DEFAULT_CKPT)
    ap.add_argument('--windows-per-set', type=int, default=4000)
    ap.add_argument('--history', type=int, default=6)
    ap.add_argument('--img-size', type=int, default=336)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--out', default='outputs/ball_probe')
    args = ap.parse_args()

    root = os.getcwd()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(args.out, exist_ok=True)

    # absolute: load_pretrained resolves relative names against its own
    # checkpoints cache dir, which silently doubles a repo-relative path
    ckpt = os.path.abspath(args.checkpoint)
    print(f'loading {ckpt}')
    model = swm.wm.utils.load_pretrained(ckpt).to(device).eval()
    model.requires_grad_(False)
    if hasattr(model, 'interpolate_pos_encoding'):
        model.interpolate_pos_encoding = True

    data, norms = {}, None
    for label, name in SETS.items():
        ds, norms = build_dataset(name, root, args.history, args.img_size, norms)
        cls, emb, pred, pos, spd = extract(model, ds, args.windows_per_set,
                                           args.history, args.batch, device)
        data[label] = dict(cls=cls, emb=emb, pred=pred, pos=pos, speed=spd)
        print(f'  {label:9s} {len(spd):5d} windows   '
              f'speed {spd.min():.4f}-{spd.max():.4f} m/s  (std {spd.std():.4f})')

    H = args.history
    # --- probe inputs -------------------------------------------------------
    def inputs(d):
        y_pos, y_spd = d['pos'][:, H - 1, TRAVEL_AXIS], d['speed']
        out = {}
        for tag in ('cls', 'emb'):
            e = d[tag]
            name = 'cls' if tag == 'cls' else 'proj'
            out[f'position [{name}]'] = (e[:, H - 1], y_pos)
            out[f'speed diff [{name}]'] = (e[:, H - 1] - e[:, H - 2], y_spd)
            out[f'speed ctx  [{name}]'] = (e[:, :H].reshape(len(e), -1), y_spd)
        out['speed pred [predictor]'] = (d['pred'], y_spd)
        return out

    probes = {}
    for key, (x, y) in inputs(data['train']).items():
        probes[key] = fit_probe(x, y)

    # --- report -------------------------------------------------------------
    results = {}
    for key in probes:
        scaler, reg = probes[key]
        unit = 'm' if key.startswith('position') else 'm/s'
        print(f'\n=== {key} ===')
        print(f"{'set':10s} {'R2':>9s} {'MAE':>12s} {'rel':>8s} {'bias':>12s}")
        results[key] = {}
        for label, d in data.items():
            x, y = inputs(d)[key]
            s = score(scaler, reg, x, y)
            results[key][label] = s
            print(f'{label:10s} {s["r2"]:9.3f} {s["mae"]:9.4f} {unit:>2s} '
                  f'{s["rel"] * 100:7.1f}% {s["bias"]:+9.4f} {unit:>2s}')

    # --- speed without fitting a speed probe --------------------------------
    pscaler, preg = probes['position [cls]']
    derived, unit = {}, 'm/s'
    print('\n=== speed derived from position [cls] (no speed probe fitted) ===')
    print(f"{'set':10s} {'R2':>9s} {'MAE':>12s} {'rel':>8s} {'bias':>12s}")
    for label, d in data.items():
        est = derived_speed(pscaler, preg, d, H)
        derived[label] = est
        mae = np.mean(np.abs(est - d['speed']))
        print(f'{label:10s} {r2_score(d["speed"], est):9.3f} '
              f'{mae:9.4f} {unit:>2s} {mae / d["speed"].mean() * 100:7.1f}% '
              f'{np.mean(est - d["speed"]):+9.4f} {unit:>2s}')

    # The readout floor: differencing two position estimates of error e over dt
    # gives e*sqrt(2)/dt. If the measured speed error sits at this floor, the
    # limit is probe precision, not missing information in the latent.
    pos_mae = results['position [cls]']['held-out']['mae']
    floor = pos_mae * np.sqrt(2) / CONTROL_DT
    print(f'\nposition readout {pos_mae:.4f} m  ->  1-step speed floor '
          f'{floor:.4f} m/s, averaged over {H - 1} displacements '
          f'{floor / np.sqrt(H - 1):.4f} m/s')

    make_plots(results, data, derived, H, args.out)

    np.savez(os.path.join(args.out, 'probe_results.npz'),
             **{f'derived|{s}': v for s, v in derived.items()},
             **{f'{k}|{s}|{m}': v[s][m]
                for k, v in results.items() for s in v for m in ('pred',)},
             **{f'true|{s}': data[s]['speed'] for s in data})
    print(f'\nsaved raw predictions to {args.out}/probe_results.npz')
    return results, data


if __name__ == '__main__':
    main()
