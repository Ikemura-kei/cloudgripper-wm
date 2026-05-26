"""Evaluate autoregressive prediction quality of a LeWM checkpoint.

Samples N_EPISODES windows from a dataset, rolls out the world model
autoregressively with real actions, and plots per-step embedding MSE
(mean ± 1 std) as a function of prediction horizon.

Usage:
    uv run python scripts/debug/eval_mse_curve.py \\
        checkpoint=/path/to/weights.pt \\
        dataset=/path/to/dataset.h5
"""

from pathlib import Path

import hydra
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch.utils.data import DataLoader

import stable_worldmodel as swm
from stable_worldmodel.data import column_normalizer
from stable_worldmodel.wm.utils import load_pretrained
from stable_pretraining import data as dt


# ------------------------------------------------------------------ #
#  Rollout helpers                                                     #
# ------------------------------------------------------------------ #

def _rollout(model, pixels, actions, n_context, device, extra=None):
    """Autoregressive rollout for LeWM.

    Returns pred_emb (T, D) and gt_emb (T, D).
    extra: dict of additional conditioning tensors (e.g. proprio), shape (1, T, D).
    """
    info = {
        'pixels': pixels[:, :n_context].unsqueeze(1).to(device),
        'action': actions[:, :n_context].unsqueeze(1).to(device),
    }
    if extra:
        for k, v in extra.items():
            info[k] = v[:, :n_context].unsqueeze(1).to(device)

    with torch.no_grad():
        info = model.rollout(info, actions.unsqueeze(1).to(device), history_size=n_context)
    pred_emb = info['predicted_emb'][0, 0]  # (n_context + n_pred + 1, D)

    gt_batch = {'pixels': pixels.to(device), 'action': actions.to(device)}
    if extra:
        for k, v in extra.items():
            gt_batch[k] = v.to(device)
    with torch.no_grad():
        model.encode(gt_batch)
    gt_emb = gt_batch['emb'][0]  # (T_total, D)

    return pred_emb, gt_emb


# ------------------------------------------------------------------ #
#  Episode window sampling (NaN-boundary detection)                   #
# ------------------------------------------------------------------ #

def _sample_episodes(cfg, n_total_steps, frameskip):
    dataset = swm.data.load_dataset(
        cfg.dataset, num_steps=n_total_steps, frameskip=frameskip,
        transform=None, keys_to_load=list(cfg.eval.keys_to_load),
    )
    imagenet = dt.dataset_stats.ImageNet
    transforms = [
        dt.transforms.ToImage(**imagenet, source='pixels', target='pixels'),
        dt.transforms.Resize(cfg.eval.image_size, source='pixels', target='pixels'),
    ]
    # z-score normalise every non-pixel key, matching training preprocessing
    for col in cfg.eval.keys_to_load:
        if not col.startswith('pixels'):
            transforms.append(column_normalizer(dataset, col, col))
    dataset.transform = dt.transforms.Compose(*transforms)

    # clip_indices is [(ep_idx, start), ...]; start==0 means beginning of episode.
    # All windows are guaranteed within a single episode by the dataset itself.
    ep_start_indices = [
        i for i, (_, start) in enumerate(dataset.clip_indices) if start == 0
    ]
    n_avail = len(ep_start_indices)
    n_want  = cfg.eval.n_episodes
    if n_avail < n_want:
        print(f'  Warning: only {n_avail} episodes long enough for {n_total_steps} steps '
              f'(requested {n_want})')
    selected = ep_start_indices[:n_want]
    batches = []
    for i in selected:
        item = dataset[i]
        # dataset[i] has no batch dim; add it so downstream code sees (1, T, ...)
        batches.append({
            k: (v.unsqueeze(0) if isinstance(v, torch.Tensor)
                else torch.as_tensor(v).unsqueeze(0))
            for k, v in item.items()
        })
    return batches


# ------------------------------------------------------------------ #
#  Main                                                                #
# ------------------------------------------------------------------ #

@hydra.main(version_base=None, config_path='./config', config_name='eval_mse_curve')
def run(cfg: DictConfig):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    print(f'Loading model from {cfg.checkpoint} …')
    model = load_pretrained(cfg.checkpoint).to(device).eval()
    model.requires_grad_(False)

    # infer history_size from the model itself so it always matches training
    if hasattr(model, 'predictor') and hasattr(model.predictor, 'num_frames'):
        n_context = model.predictor.num_frames
    elif hasattr(model, 'history_size'):
        n_context = model.history_size
    else:
        raise AttributeError(
            'Cannot infer n_context from model. '
            'Expected model.predictor.num_frames (LeWM) or model.history_size.'
        )
    print(f'  n_context (history_size) = {n_context}  [read from model]')

    # infer frameskip from model's expected action input dim vs raw dataset action dim
    if hasattr(model, 'action_encoder') and hasattr(model.action_encoder, 'input_dim'):
        model_action_input_dim = model.action_encoder.input_dim
        _probe = swm.data.load_dataset(
            cfg.dataset, num_steps=1, frameskip=1,
            transform=None, keys_to_load=['action'],
        )
        raw_action_dim = next(iter(DataLoader(_probe, batch_size=1)))['action'].shape[-1]
        frameskip = model_action_input_dim // raw_action_dim
        if frameskip * raw_action_dim != model_action_input_dim:
            raise ValueError(
                f'action_encoder.input_dim={model_action_input_dim} is not divisible by '
                f'raw_action_dim={raw_action_dim} — cannot infer frameskip'
            )
        print(f'  frameskip = {frameskip}  '
              f'[inferred from action_encoder.input_dim={model_action_input_dim} / raw_action_dim={raw_action_dim}]')
    else:
        frameskip = cfg.eval.frameskip
        print(f'  frameskip = {frameskip}  [from config — model has no action_encoder.input_dim]')

    n_pred_steps = cfg.eval.n_pred_steps

    # cap n_pred_steps to what the dataset episodes can actually support
    _probe_ds = swm.data.load_dataset(
        cfg.dataset, num_steps=1, frameskip=frameskip,
        transform=None, keys_to_load=['action'],
    )
    max_ep_effective = int(np.array(_probe_ds.lengths).max()) // frameskip
    max_pred = max_ep_effective - n_context
    if max_pred <= 0:
        raise ValueError(
            f'No episode is long enough for even a single prediction step. '
            f'Longest episode has {max_ep_effective} effective steps '
            f'(after frameskip={frameskip}), but n_context={n_context} alone requires {n_context}.'
        )
    if n_pred_steps > max_pred:
        print(f'  n_pred_steps capped {n_pred_steps} → {max_pred} '
              f'(max episode has {max_ep_effective} effective steps, n_context={n_context})')
        n_pred_steps = max_pred

    n_total = n_context + n_pred_steps

    print(f'Sampling {cfg.eval.n_episodes} episode windows '
          f'(context={n_context}, pred={n_pred_steps}) …')
    episodes = _sample_episodes(cfg, n_total, frameskip)
    if not episodes:
        raise RuntimeError(
            f'No episodes found with {n_total} steps (frameskip={frameskip}). '
            f'Try reducing eval.n_pred_steps.'
        )
    print(f'  Found {len(episodes)} windows')

    _CORE_KEYS = {'pixels', 'action'}

    all_mse = []
    for i, batch in enumerate(episodes):
        pixels  = batch['pixels']   # (1, n_total, C, H, W)
        actions = batch['action']   # (1, n_total, action_dim)
        extra   = {k: v for k, v in batch.items() if k not in _CORE_KEYS}

        pred_emb, gt_emb = _rollout(model, pixels, actions, n_context, device, extra)

        step_mse = []
        for t in range(n_context, min(n_context + n_pred_steps, gt_emb.shape[0])):
            pred_t = pred_emb[t] if t < pred_emb.shape[0] else pred_emb[-1]
            step_mse.append(F.mse_loss(pred_t, gt_emb[t]).item())
        all_mse.append(step_mse)

    all_mse  = np.array(all_mse)      # (N_EPISODES, N_PRED_STEPS)
    mean_mse = all_mse.mean(axis=0)
    std_mse  = all_mse.std(axis=0)
    steps    = np.arange(1, len(mean_mse) + 1)

    # ---- text table ------------------------------------------------ #
    print(f'\nEmbedding MSE over {len(all_mse)} episodes  '
          f'[{Path(cfg.checkpoint).parent.name}]')
    print(f'{"step":>5}  {"mean":>10}  {"std":>10}  {"min":>10}  {"max":>10}')
    print('-' * 52)
    for s, m, sd, mn, mx in zip(
        steps, mean_mse, std_mse, all_mse.min(axis=0), all_mse.max(axis=0)
    ):
        print(f'{s:>5}  {m:>10.6f}  {sd:>10.6f}  {mn:>10.6f}  {mx:>10.6f}')

    # ---- plot ------------------------------------------------------ #
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(steps, mean_mse, marker='o', markersize=4, label='mean MSE')
        ax.fill_between(steps, mean_mse - std_mse, mean_mse + std_mse,
                        alpha=0.25, label='±1 std')
        ax.set_xlabel('Prediction step')
        ax.set_ylabel('Embedding MSE')
        ax.set_title(
            f'Autoregressive prediction MSE — '
            f'context={n_context}  n={len(all_mse)} episodes\n'
            f'{Path(cfg.checkpoint).parent.name}'
        )
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        if cfg.output.path:
            plt.savefig(cfg.output.path, dpi=150)
            print(f'\nPlot saved → {cfg.output.path}')
        else:
            plt.show()

    except ImportError:
        print('\nmatplotlib not available — skipping plot')


if __name__ == '__main__':
    run()
