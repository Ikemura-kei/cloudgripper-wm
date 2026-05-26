"""Evaluate autoregressive prediction quality of a stable-worldmodel checkpoint.

Samples N_EPISODES windows from a dataset, rolls out the world model
autoregressively with real actions, and plots per-step embedding MSE
(mean ± 1 std) as a function of prediction horizon.

Compatible with any model that exposes rollout() + encode():
  - LeWM  (stable_worldmodel.wm.lewm)
  - PLDM  (stable_worldmodel.wm.pldm)
  - PreJEPA (set model.is_prejepa: true — one-step prediction only)

Usage:
    uv run python scripts/debug/eval_mse_curve.py \\
        checkpoint=/path/to/weights_epoch_92.pt \\
        dataset=/path/to/cloudgripper.lance
"""

import inspect
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch.utils.data import DataLoader

import stable_worldmodel as swm
from stable_worldmodel.wm.utils import load_pretrained
from stable_pretraining import data as dt


# ------------------------------------------------------------------ #
#  Rollout helpers                                                     #
# ------------------------------------------------------------------ #

def _rollout_lewm(model, pixels, actions, n_context, device):
    """Autoregressive rollout for LeWM / PLDM.

    Returns pred_emb (T, D) and gt_emb (T, D) where T = N_CONTEXT + N_PRED + 1.
    """
    info = {
        'pixels': pixels[:, :n_context].unsqueeze(1).to(device),
        'action': actions[:, :n_context].unsqueeze(1).to(device),
    }
    sig = inspect.signature(model.rollout)
    kwargs = {'history_size': n_context} if 'history_size' in sig.parameters else {}
    with torch.no_grad():
        info = model.rollout(info, actions.unsqueeze(1).to(device), **kwargs)
    pred_emb = info['predicted_emb'][0, 0]  # (n_context + n_pred + 1, D)

    gt_batch = {'pixels': pixels.to(device), 'action': actions.to(device)}
    with torch.no_grad():
        model.encode(gt_batch)
    gt_emb = gt_batch['emb'][0]  # (T_total, D)

    return pred_emb, gt_emb


def _rollout_prejepa(model, pixels, actions, n_context, device):
    """One-step prediction for PreJEPA (not autoregressive)."""
    batch = {'pixels': pixels.to(device), 'action': actions.to(device)}
    with torch.no_grad():
        out = model.encode(batch)
    emb     = out['emb']      # (1, T, D)
    act_emb = out['act_emb']  # (1, T, A)
    with torch.no_grad():
        pred_emb = model.predict(emb[:, :n_context], act_emb[:, :n_context])
    gt_emb   = emb[0, 1:]       # (T-1, D)
    pred_emb = pred_emb[0]      # (n_pred, D)
    L = min(pred_emb.shape[0], gt_emb.shape[0])
    return pred_emb[:L], gt_emb[:L]


# ------------------------------------------------------------------ #
#  Episode window sampling (NaN-boundary detection)                   #
# ------------------------------------------------------------------ #

def _sample_episodes(cfg, n_total_steps):
    dataset = swm.data.load_dataset(
        cfg.dataset, num_steps=n_total_steps, frameskip=cfg.eval.frameskip,
        transform=None, keys_to_load=list(cfg.eval.keys_to_load),
    )
    imagenet = dt.dataset_stats.ImageNet
    dataset.transform = dt.transforms.Compose(
        dt.transforms.ToImage(**imagenet, source='pixels', target='pixels'),
        dt.transforms.Resize(cfg.eval.image_size, source='pixels', target='pixels'),
    )

    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    samples = {}
    ep_ctr = -1
    after_boundary = True

    for b in loader:
        if torch.isnan(b['action']).any():
            after_boundary = True
            continue
        if after_boundary:
            ep_ctr += 1
            after_boundary = False
        if ep_ctr in samples:
            continue
        samples[ep_ctr] = {k: v for k, v in b.items()}
        if len(samples) == cfg.eval.n_episodes:
            break

    return list(samples.values())


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

    n_pred_steps = cfg.eval.n_pred_steps
    n_total      = n_context + n_pred_steps

    print(f'Sampling {cfg.eval.n_episodes} episode windows '
          f'(context={n_context}, pred={n_pred_steps}) …')
    episodes = _sample_episodes(cfg, n_total)
    print(f'  Found {len(episodes)} windows')

    rollout_fn = _rollout_prejepa if cfg.model.is_prejepa else _rollout_lewm

    all_mse = []
    for i, batch in enumerate(episodes):
        pixels  = batch['pixels']   # (1, n_total, C, H, W)
        actions = batch['action']   # (1, n_total, action_dim)

        pred_emb, gt_emb = rollout_fn(model, pixels, actions, n_context, device)

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
