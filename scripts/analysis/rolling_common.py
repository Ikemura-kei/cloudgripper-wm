"""Shared settings for the rolling-ball study (the frameskip-5 rerun).

Deliberately separate from flat2d_common.py rather than parameterising it.
That module encodes the Experiment 1-5 setup — frameskip 1, history 6, images
stored at 448 and resized to 336 — and the published write-ups point at the
scripts that use it. Changing its constants in place would silently break the
reproduction path for results that are already written up.

What differs here:

    frameskip    5 instead of 1      the ball advances 33 px per step, not 10
    history      3 instead of 6      matching the upstream TwoRoom config
    resolution   224 native          no resize, so frames keep exactly three
                                     colours instead of antialiased edges
    ball         1.5x larger in world units
    data         2000 episodes per colour instead of 500

The two datasets are frame-for-frame matched, verified at collection: window
*i* is the same scene in both colours, with the ball in the same place.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

import stable_pretraining as spt
import stable_worldmodel as swm
from stable_pretraining import data as dt
from stable_worldmodel.data import column_normalizer

REPO = Path(__file__).resolve().parents[2]
CKPT_ROOT = REPO / '.stable_worldmodel' / 'checkpoints'
CACHE = REPO / 'outputs' / 'rolling_cache'

HISTORY = 3
IMG_SIZE = 224
FRAMESKIP = 5
TRAVEL_AXIS = 0          # flat2d ball_pos is (x, y) and it travels along x
CONTROL_DT = 0.05
PX = 448                 # errors are reported in pixels of a 448-wide frame,
                         # so they stay comparable with the earlier write-ups
TRAIN_SPLIT = 0.9
SPLIT_SEED = 3072
N_FIT = 4000

DATASETS = {'red': 'rolling_ball_red_default', 'green': 'rolling_ball_green_default'}
RUNS = {'red': 'lewm_rb_red_default-*', 'green': 'lewm_rb_green_default-*',
        # same data, same split seed, same everything but init_seed=2.
        # Used to ask which of the colour findings are properties of the
        # training setup and which are accidents of one initialisation.
        'red_s2': 'lewm_rb_red_s2-*',
        # as above, but trained with channel-symmetric normalisation instead of
        # ImageNet's, so the network's input really is green/blue swap invariant.
        'red_sym_s1': 'lewm_rb_red_sym_s1-*',
        'red_sym_s2': 'lewm_rb_red_sym_s2-*'}
FINAL_EPOCH = 30


def run_dir(colour: str) -> Path:
    hits = sorted(CKPT_ROOT.glob(RUNS[colour]))
    if len(hits) != 1:
        raise FileNotFoundError(f'expected one run for {colour}, got {hits}')
    return hits[0]


def load_model(colour: str, device: str = 'cuda', epoch: int = FINAL_EPOCH):
    m = swm.wm.utils.load_pretrained(
        str(run_dir(colour) / f'weights_epoch_{epoch}.pt')).to(device).eval()
    m.requires_grad_(False)
    return m


# Must match what a model was TRAINED with. 'imagenet' is the historical value
# and the default; 'symmetric' belongs to the runs that test whether the
# per-channel ImageNet means are what breaks the green/blue symmetry.
IMG_STATS = {
    'imagenet': None,                                           # filled in below
    'symmetric': {'mean': [0.5, 0.5, 0.5], 'std': [0.25, 0.25, 0.25]},
}


def build_dataset(colour: str, norms: dict | None = None, stats: str = 'imagenet'):
    """Dataset with the frameskip, resolution and normalisation a model was trained on."""
    ds = swm.data.load_dataset(
        str(REPO / '.stable_worldmodel' / 'datasets' / f'{DATASETS[colour]}.lance'),
        transform=None, num_steps=HISTORY + 1, frameskip=FRAMESKIP,
        keys_to_load=['pixels', 'action', 'state', 'ball_pos', 'ball_speed'])
    IMG_STATS['imagenet'] = dt.dataset_stats.ImageNet
    stats = IMG_STATS[stats]
    tfs = [dt.transforms.Compose(
        dt.transforms.ToImage(**stats, source='pixels', target='pixels'),
        dt.transforms.Resize(IMG_SIZE, source='pixels', target='pixels'))]
    norms = norms or {c: column_normalizer(ds, c, c) for c in ('action', 'state')}
    tfs.extend(norms.values())
    ds.transform = spt.data.transforms.Compose(*tfs)
    return ds, norms


def split_indices(ds):
    """(fit, val) window indices. Val episodes are the ones training held out."""
    clip_eps = [ep for ep, _ in ds.clip_indices]
    episodes = sorted(set(clip_eps))
    gen = torch.Generator().manual_seed(SPLIT_SEED)
    perm = torch.randperm(len(episodes), generator=gen).tolist()
    n_train = max(1, min(int(round(TRAIN_SPLIT * len(episodes))), len(episodes) - 1))
    train_eps = {episodes[i] for i in perm[:n_train]}
    val = np.array([i for i, ep in enumerate(clip_eps) if ep not in train_eps])
    train_pool = np.array([i for i, ep in enumerate(clip_eps) if ep in train_eps])
    fit = np.sort(np.random.default_rng(0).choice(
        train_pool, size=min(N_FIT, len(train_pool)), replace=False))
    return fit, val


@torch.no_grad()
def extract_multi(models: dict, ds, idx: np.ndarray, batch: int = 64,
                  device: str = 'cuda') -> dict:
    """Encode the same windows with several models, loading each batch once."""
    acc = {k: {'cls': [], 'proj': [], 'pred': []} for k in models}
    pos_, spd_ = [], []
    for lo in range(0, len(idx), batch):
        rows = [ds[int(i)] for i in idx[lo:lo + batch]]
        pix = torch.stack([r['pixels'] for r in rows]).to(device)
        act = torch.nan_to_num(
            torch.stack([torch.as_tensor(r['action']) for r in rows]).float().to(device), 0.0)
        b, t = pix.shape[:2]
        flat = pix.reshape(b * t, *pix.shape[2:])
        for name, m in models.items():
            f = flat.to(next(m.encoder.parameters()).dtype)
            cls = m.encoder(f, interpolate_pos_encoding=True).last_hidden_state[:, 0]
            proj = m.projector(cls)
            cls = cls.reshape(b, t, -1).float()
            proj = proj.reshape(b, t, -1).float()
            ae = m.action_encoder(act)
            nxt = m.predict(proj[:, :HISTORY].to(ae.dtype), ae[:, :HISTORY])
            acc[name]['cls'].append(cls.cpu().numpy())
            acc[name]['proj'].append(proj.cpu().numpy())
            acc[name]['pred'].append(nxt[:, -1].float().cpu().numpy())
        pos_.append(np.stack([np.asarray(r['ball_pos']) for r in rows]))
        spd_.append(np.stack([np.asarray(r['ball_speed']).reshape(-1)[0] for r in rows]))
    traj = np.concatenate(pos_)
    assert traj.ndim == 3 and traj.shape[1] == HISTORY + 1, traj.shape
    out = {}
    for name in models:
        out[name] = {k: np.concatenate(v) for k, v in acc[name].items()}
        out[name].update({'pos_traj': traj, 'pos': traj[:, HISTORY],
                          'x': traj[:, HISTORY, TRAVEL_AXIS],
                          'speed': np.concatenate(spd_)})
    return out
