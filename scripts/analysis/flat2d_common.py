"""Shared machinery for the flat2d red/green representation study.

The six models (red/green x init_seed 1/2/3) were trained with `seed: 3072`
throughout, so every one of them saw the *same* episode split. The red and
green datasets are frame-for-frame matched — identical `ball_pos`,
`ball_speed`, episode lengths and ball pixel masks, differing only in hue — so
window *i* of the red dataset and window *i* of the green dataset are the same
physical scene in two colours. Every comparison below depends on that pairing,
so `extract()` is always called with the same window indices for both, and
`check_pairing()` asserts the ground truth really does line up.

Analysis is run on the **validation episodes only**: all six models trained on
the same 450 episodes, so the remaining 50 are frames none of them has seen.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch

import stable_pretraining as spt
import stable_worldmodel as swm
from stable_pretraining import data as dt
from stable_worldmodel.data import column_normalizer


REPO = Path(__file__).resolve().parents[2]
CKPT_ROOT = REPO / '.stable_worldmodel' / 'checkpoints'

# Must match scripts/train/config/lewm_flat2d_ball.yaml at the time of training.
HISTORY = 6
IMG_SIZE = 336
TRAIN_SPLIT = 0.9
SPLIT_SEED = 3072
FINAL_EPOCH = 30

# flat2d ball_pos is (x, y) and the ball travels along x — note this differs
# from the MuJoCo ball env, whose ball_pos is (x, y, z) travelling along y
# (see TRAVEL_AXIS in probe_ball_latent.py).
TRAVEL_AXIS = 0
CONTROL_DT = 0.05  # s per step; matches Flat2DBall's control_timestep


def run_dir(dataset: str, init_seed: int) -> Path:
    """Locate a run's checkpoint directory by dataset and init seed.

    args:
        dataset: 'red' or 'green'
        init_seed: the run's init_seed
    returns:
        path to the run directory
    """
    pat = f'lewm_flat2d_ball_{dataset}_init{init_seed}-*'
    hits = sorted(CKPT_ROOT.glob(pat))
    if len(hits) != 1:
        raise FileNotFoundError(f'expected exactly one run matching {pat}, got {hits}')
    return hits[0]


def load_model(dataset: str, init_seed: int, device: str = 'cuda', epoch: int = FINAL_EPOCH):
    """Load a trained LeWM model, frozen and in eval mode."""
    d = run_dir(dataset, init_seed)
    m = swm.wm.utils.load_pretrained(str(d / f'weights_epoch_{epoch}.pt')).to(device).eval()
    m.requires_grad_(False)
    return m


def build_dataset(name: str, norms: dict | None = None):
    """Load a flat2d dataset with the transform the models were trained with.

    args:
        name: dataset stem under .stable_worldmodel/datasets
        norms: reuse another set's column normalizers. Both flat2d datasets
            have constant-zero `action` and `state`, so their statistics are
            identical either way, but reusing them makes that explicit rather
            than incidental.
    returns:
        (dataset, norms)
    """
    ds = swm.data.load_dataset(
        str(REPO / '.stable_worldmodel' / 'datasets' / f'{name}.lance'),
        transform=None,
        num_steps=HISTORY + 1,
        frameskip=1,
        keys_to_load=['pixels', 'action', 'state', 'ball_pos', 'ball_speed'],
    )
    stats = dt.dataset_stats.ImageNet
    tfs = [dt.transforms.Compose(
        dt.transforms.ToImage(**stats, source='pixels', target='pixels'),
        dt.transforms.Resize(IMG_SIZE, source='pixels', target='pixels'),
    )]
    norms = norms if norms is not None else {
        c: column_normalizer(ds, c, c) for c in ('action', 'state')
    }
    tfs.extend(norms.values())
    ds.transform = spt.data.transforms.Compose(*tfs)
    return ds, norms


def val_window_indices(ds) -> np.ndarray:
    """Window indices belonging to the validation episodes.

    Replicates scripts/train/lewm.py:split_by_episode with the same seed, so
    these are exactly the windows no model was trained on.
    """
    clip_eps = [ep for ep, _ in ds.clip_indices]
    episodes = sorted(set(clip_eps))
    gen = torch.Generator().manual_seed(SPLIT_SEED)
    perm = torch.randperm(len(episodes), generator=gen).tolist()
    n_train = int(round(TRAIN_SPLIT * len(episodes)))
    n_train = max(1, min(n_train, len(episodes) - 1))
    train_eps = {episodes[i] for i in perm[:n_train]}
    return np.array([i for i, ep in enumerate(clip_eps) if ep not in train_eps])


def sample_indices(ds, n: int, seed: int = 0) -> np.ndarray:
    """A reproducible sample of validation windows, sorted for cheap reads."""
    pool = val_window_indices(ds)
    rng = np.random.default_rng(seed)
    if n >= len(pool):
        return np.sort(pool)
    return np.sort(rng.choice(pool, size=n, replace=False))


def cached_extract(tag: str, model_fn, ds_fn, idx, batch=32, device='cuda',
                   cache_dir=None) -> dict:
    """extract(), memoised to disk under outputs/flat2d_cache.

    Encoding ~50k frames takes ~15 min and is CPU-bound in the resize, so the
    analysis steps reuse one cache rather than re-encoding per experiment.
    `model_fn`/`ds_fn` are callables so nothing is loaded on a cache hit.
    """
    cache_dir = Path(cache_dir or (REPO / 'outputs' / 'flat2d_cache'))
    cache_dir.mkdir(parents=True, exist_ok=True)
    f = cache_dir / f'{tag}.npz'
    if f.exists():
        z = np.load(f)
        return {k: z[k] for k in z.files}
    out = extract(model_fn(), ds_fn(), idx, batch, device)
    np.savez_compressed(f, **out)
    return out


@torch.no_grad()
def extract(model, ds, idx: np.ndarray, batch: int = 32, device: str = 'cuda') -> dict:
    """Encode the given windows, returning representations and ground truth.

    `idx` is passed in rather than drawn here so the caller can guarantee the
    red and green runs see the *same* windows — the whole study rests on that
    pairing.

    returns a dict with
        cls      (N, H+1, 192)  ViT CLS token — the backbone representation
        proj     (N, H+1, 192)  projector output — what the model predicts in
        pred     (N, 192)       predictor output for the step after the context
        pos_traj (N, H+1, 2)    ground-truth ball (x, y) for every frame
        pos      (N, 2)         ball (x, y) at the window's LAST frame, i.e. the
                                frame whose representation is cls[:, HISTORY]
        x        (N,)           ball position along the travel axis at that frame
        speed    (N,)           ground-truth ball speed (constant per episode)

    `pos` and `x` are reduced here rather than at the call site: the raw column
    is (H+1, 2) per window, and indexing it as if it were (2,) silently yields
    frame 0's coordinate pair instead of the last frame's x — which fits a
    probe that looks like it works and reports nonsense.
    """
    cls_, proj_, pred_, pos_, spd_ = [], [], [], [], []
    for lo in range(0, len(idx), batch):
        rows = [ds[int(i)] for i in idx[lo:lo + batch]]
        pix = torch.stack([r['pixels'] for r in rows]).to(device)
        act = torch.stack([torch.as_tensor(r['action']) for r in rows]).float().to(device)
        act = torch.nan_to_num(act, 0.0)

        b, t = pix.shape[:2]
        flat = pix.reshape(b * t, *pix.shape[2:]).to(next(model.encoder.parameters()).dtype)
        enc = model.encoder(flat, interpolate_pos_encoding=True)
        cls = enc.last_hidden_state[:, 0]
        proj = model.projector(cls)
        cls = cls.reshape(b, t, -1).float()
        proj = proj.reshape(b, t, -1).float()
        act_emb = model.action_encoder(act)
        nxt = model.predict(proj[:, :HISTORY].to(act_emb.dtype), act_emb[:, :HISTORY])

        cls_.append(cls.cpu().numpy())
        proj_.append(proj.cpu().numpy())
        pred_.append(nxt[:, -1].float().cpu().numpy())
        pos_.append(np.stack([np.asarray(r['ball_pos']) for r in rows]))
        spd_.append(np.stack([np.asarray(r['ball_speed']).reshape(-1)[0] for r in rows]))

    pos_traj = np.concatenate(pos_)                      # (N, H+1, 2)
    assert pos_traj.ndim == 3 and pos_traj.shape[1] == HISTORY + 1, pos_traj.shape
    pos_last = pos_traj[:, HISTORY]                      # (N, 2)
    return {
        'cls': np.concatenate(cls_),
        'proj': np.concatenate(proj_),
        'pred': np.concatenate(pred_),
        'pos_traj': pos_traj,
        'pos': pos_last,
        'x': pos_last[:, TRAVEL_AXIS],
        'speed': np.concatenate(spd_),
    }


@torch.no_grad()
def extract_multi(models: dict, ds, idx: np.ndarray, batch: int = 32,
                  device: str = 'cuda') -> dict:
    """Encode the same windows with several models, loading each batch once.

    The image pipeline (PNG decode, 448->336 resize, ImageNet normalise) is
    CPU-bound and dominates extraction, and it is identical for every model.
    Running one model at a time re-pays that cost per model; this pays it once
    per batch and fans the batch out across all of them.

    args:
        models: {name: LeWM model}, all sharing the architecture
        ds: dataset built by build_dataset()
        idx: window indices to encode
    returns:
        {name: extraction dict}, each shaped exactly like extract()'s output
    """
    acc = {k: {'cls': [], 'proj': [], 'pred': []} for k in models}
    pos_, spd_ = [], []
    for lo in range(0, len(idx), batch):
        rows = [ds[int(i)] for i in idx[lo:lo + batch]]
        pix = torch.stack([r['pixels'] for r in rows]).to(device)
        act = torch.stack([torch.as_tensor(r['action']) for r in rows]).float().to(device)
        act = torch.nan_to_num(act, 0.0)
        b, t = pix.shape[:2]
        flat = pix.reshape(b * t, *pix.shape[2:])

        for name, m in models.items():
            f = flat.to(next(m.encoder.parameters()).dtype)
            enc = m.encoder(f, interpolate_pos_encoding=True)
            cls = enc.last_hidden_state[:, 0]
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

    pos_traj = np.concatenate(pos_)
    pos_last = pos_traj[:, HISTORY]
    out = {}
    for name in models:
        out[name] = {
            'cls': np.concatenate(acc[name]['cls']),
            'proj': np.concatenate(acc[name]['proj']),
            'pred': np.concatenate(acc[name]['pred']),
            'pos_traj': pos_traj, 'pos': pos_last,
            'x': pos_last[:, TRAVEL_AXIS], 'speed': np.concatenate(spd_),
        }
    return out


def untrained_model(device: str = 'cuda', seed: int = 99):
    """A LeWM with the trained models' architecture and random weights.

    The floor of the similarity scale: whatever two networks share before any
    training, given the same architecture and the same input statistics.
    """
    import lightning as pl
    from hydra.utils import instantiate
    from omegaconf import OmegaConf, open_dict
    cfg = OmegaConf.load(REPO / 'scripts' / 'train' / 'config' / 'lewm_flat2d_ball.yaml')
    with open_dict(cfg):
        cfg.model.action_encoder.input_dim = 5
    pl.seed_everything(seed, workers=True)
    m = instantiate(cfg.model).to(device).eval()
    m.requires_grad_(False)
    return m


def check_pairing(a: dict, b: dict, label: str = '') -> None:
    """Assert two extractions really are the same scenes in two colours."""
    for k in ('pos_traj', 'pos', 'x', 'speed'):
        if not np.allclose(a[k], b[k], atol=0, rtol=0):
            raise AssertionError(
                f'{label}: {k} differs between the two extractions — the red/green '
                f'pairing is broken, every comparison downstream would be invalid.'
            )
