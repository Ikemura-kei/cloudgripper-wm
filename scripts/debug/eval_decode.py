"""Train a CNN decoder on frozen LeWM embeddings, then evaluate autoregressive
image generation by decoding rollout predictions back to pixel space.

Two phases in one run:
  1. Decoder training  — freeze LeWM, train CNN decoder (embed → pixels) with
                         Lightning + optional WandB image logging + train/val split.
  2. Evaluation        — autoregressive rollout → decode → side-by-side MP4s.

If decoder.load_path is set the training phase is skipped.

Preprocessing is identical to scripts/train/lewm.py:
  ToImage (ImageNet stats) + Resize + z-score column_normalizer for every
  non-pixel key in keys_to_load (only pixels needed for decoder training).

Usage:
    uv run python scripts/debug/eval_decode.py \\
        checkpoint=/path/to/weights.pt \\
        dataset=/path/to/dataset \\
        'eval.keys_to_load=[pixels,action]'
"""

import datetime
import json
import random
import string
from pathlib import Path

import cv2
import hydra
import lightning as pl
import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from hydra.utils import instantiate
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf, open_dict
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader
from tqdm import tqdm

from stable_worldmodel.data import column_normalizer
from stable_worldmodel.wm.utils import load_pretrained
from stable_pretraining import data as dt


# ------------------------------------------------------------------ #
#  CNN Decoder  (matches train_decoder.py: 7×13 spatial base)         #
# ------------------------------------------------------------------ #

class CNNDecoder(nn.Module):
    """Upsamples a D-dim embedding back to (3, H, W).

    7×13 → 5× ConvTranspose2d(k=4,s=2,p=1) → 224×416,
    then bilinear interpolation to the actual target size.
    """
    def __init__(self, embed_dim: int = 192):
        super().__init__()
        self.proj = nn.Linear(embed_dim, 128 * 7 * 13)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(128, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128,  64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),  nn.ReLU(inplace=True),
            nn.ConvTranspose2d( 64,  32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),  nn.ReLU(inplace=True),
            nn.ConvTranspose2d( 32,  16, 4, stride=2, padding=1),
            nn.BatchNorm2d(16),  nn.ReLU(inplace=True),
            nn.ConvTranspose2d( 16,   3, 4, stride=2, padding=1),
        )

    def forward(self, z: torch.Tensor,
                target_size: tuple[int, int] | None = None) -> torch.Tensor:
        out = self.net(self.proj(z).view(z.size(0), 128, 7, 13))
        if target_size is not None:
            out = F.interpolate(out, size=target_size,
                                mode='bilinear', align_corners=False)
        return out


# ------------------------------------------------------------------ #
#  Lightning module  (matches train_decoder.py)                        #
# ------------------------------------------------------------------ #

_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class DecoderModule(pl.LightningModule):
    def __init__(self, world_model: nn.Module, decoder: CNNDecoder, lr: float):
        super().__init__()
        self.world_model = world_model   # frozen
        self.decoder = decoder
        self.lr = lr

    def _encode(self, batch: dict) -> torch.Tensor:
        with torch.no_grad():
            self.world_model.encode(batch)
        return rearrange(batch['emb'], 'b t d -> (b t) d')

    def _step(self, batch: dict, stage: str):
        pixels = batch['pixels']
        flat   = rearrange(pixels, 'b t c h w -> (b t) c h w')
        emb    = self._encode(batch)
        recon  = self.decoder(emb, target_size=(flat.shape[-2], flat.shape[-1]))
        loss   = F.mse_loss(recon, flat)
        self.log(f'{stage}/mse_loss', loss, on_epoch=True, on_step=(stage == 'train'), prog_bar=True)
        return loss, recon, flat

    def training_step(self, batch, batch_idx):
        loss, _, _ = self._step(batch, 'train')
        return loss

    def validation_step(self, batch, batch_idx):
        loss, recon, orig = self._step(batch, 'val')
        if batch_idx == 0 and isinstance(self.logger, WandbLogger):
            self._log_images(orig, recon)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.decoder.parameters(), lr=self.lr)

    def _log_images(self, orig: torch.Tensor, recon: torch.Tensor, n: int = 4):
        import wandb
        mean = _MEAN.squeeze(0).to(orig)
        std  = _STD.squeeze(0).to(orig)
        denorm = lambda t: (t * std + mean).clamp(0, 1)
        imgs = []
        for i in range(min(n, orig.size(0))):
            imgs.append(wandb.Image(denorm(orig[i]),  caption=f'orig_{i}'))
            imgs.append(wandb.Image(denorm(recon[i]), caption=f'recon_{i}'))
        self.logger.experiment.log({
            'val/reconstructions': imgs,
            'trainer/global_step': self.global_step,
        })


class SaveDecoderCallback(pl.Callback):
    def __init__(self, save_path: Path):
        self.save_path = Path(save_path)
        self.save_path.parent.mkdir(parents=True, exist_ok=True)

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch + 1
        path  = self.save_path.parent / f'decoder_epoch_{epoch}.pt'
        torch.save(pl_module.decoder.state_dict(), path)

    def on_train_end(self, trainer, pl_module):
        torch.save(pl_module.decoder.state_dict(), self.save_path)


# ------------------------------------------------------------------ #
#  LeWM loader  (handles model-only and full-training config.json)    #
# ------------------------------------------------------------------ #

def _load_lewm(checkpoint_path: str) -> nn.Module:
    ckpt = Path(checkpoint_path)
    with open(ckpt.parent / 'config.json') as f:
        raw = json.load(f)
    model_cfg = raw.get('model', raw)
    model = instantiate(OmegaConf.create(model_cfg))
    model.load_state_dict(torch.load(str(ckpt), map_location='cpu'))
    return model


# ------------------------------------------------------------------ #
#  Dataset helpers  (preprocessing identical to scripts/train/lewm.py) #
# ------------------------------------------------------------------ #

def _infer_frameskip(model: nn.Module, dataset_path: str) -> int:
    if not (hasattr(model, 'action_encoder') and
            hasattr(model.action_encoder, 'input_dim')):
        return 1
    model_dim = model.action_encoder.input_dim
    probe = swm.data.load_dataset(dataset_path, num_steps=1, frameskip=1,
                                   transform=None, keys_to_load=['action'])
    raw_dim = next(iter(DataLoader(probe, batch_size=1)))['action'].shape[-1]
    fs = model_dim // raw_dim
    if fs * raw_dim != model_dim:
        raise ValueError(
            f'action_encoder.input_dim={model_dim} not divisible by '
            f'raw_action_dim={raw_dim}'
        )
    return fs


def _make_eval_transform(dataset, image_size: int,
                          keys_to_load: list[str]) -> dt.transforms.Compose:
    """ToImage + Resize + z-score per non-pixel key — same as lewm.py training."""
    imagenet = dt.dataset_stats.ImageNet
    transforms = [
        dt.transforms.ToImage(**imagenet, source='pixels', target='pixels'),
        dt.transforms.Resize(image_size, source='pixels', target='pixels'),
    ]
    for col in keys_to_load:
        if not col.startswith('pixels'):
            transforms.append(column_normalizer(dataset, col, col))
    return dt.transforms.Compose(*transforms)


def _episode_start_batches(cfg, n_total: int, frameskip: int) -> list[dict]:
    """One batch per episode start (clip_indices start==0)."""
    dataset = swm.data.load_dataset(
        cfg.dataset, num_steps=n_total, frameskip=frameskip,
        transform=None, keys_to_load=list(cfg.eval.keys_to_load),
    )
    dataset.transform = _make_eval_transform(
        dataset, cfg.eval.image_size, list(cfg.eval.keys_to_load)
    )
    ep_starts = [i for i, (_, s) in enumerate(dataset.clip_indices) if s == 0]
    n_want = cfg.eval.n_episodes
    selected = ep_starts if n_want <= 0 else ep_starts[:n_want]
    if 0 < n_want < len(ep_starts):
        pass  # silence: already took the right subset
    elif 0 < n_want and len(ep_starts) < n_want:
        print(f'  Warning: only {len(ep_starts)} episodes available '
              f'(requested {n_want})')

    batches = []
    for i in selected:
        item = dataset[i]
        batches.append({
            k: (v.unsqueeze(0) if isinstance(v, torch.Tensor)
                else torch.as_tensor(v).unsqueeze(0))
            for k, v in item.items()
        })
    return batches


# ------------------------------------------------------------------ #
#  Visualization helpers                                               #
# ------------------------------------------------------------------ #

def _denorm(t: torch.Tensor) -> torch.Tensor:
    return (t * _STD.squeeze(0).to(t) + _MEAN.squeeze(0).to(t)).clamp(0, 1)


def _to_uint8(t: torch.Tensor) -> np.ndarray:
    return (_denorm(t) * 255).byte().permute(1, 2, 0).cpu().numpy()


def _make_frame(gt: np.ndarray, pred: np.ndarray,
                label: str, is_context: bool) -> np.ndarray:
    H, W, _ = gt.shape
    bar_h, sep_w = 24, 4
    color = (30, 200, 30) if is_context else (200, 30, 30)
    canvas = np.zeros((H + bar_h, W * 2 + sep_w, 3), dtype=np.uint8)
    canvas[:H, :W]          = gt
    canvas[:H, W:W + sep_w] = color
    canvas[:H, W + sep_w:]  = pred
    canvas[H:, :]           = (20, 20, 20)
    img  = Image.fromarray(canvas)
    draw = ImageDraw.Draw(img)
    draw.text((4, H + 4),             f'GT  | {label}', fill=color)
    draw.text((W + sep_w + 4, H + 4), f'PRED| {label}', fill=color)
    return np.array(img)


def _write_video(frames: list[np.ndarray], path: Path, fps: int) -> None:
    H, W, _ = frames[0].shape
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H))
    for f in frames:
        writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    writer.release()


# ------------------------------------------------------------------ #
#  Main                                                                #
# ------------------------------------------------------------------ #

@hydra.main(version_base=None, config_path='./config', config_name='eval_decode')
def run(cfg: DictConfig) -> None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    ts     = datetime.datetime.now().strftime('%y-%m-%d-%H-%M-%S')
    suffix = ''.join(random.choices(string.ascii_lowercase, k=3))
    with open_dict(cfg):
        cfg.output_name = f'{cfg.output_name}-{ts}-{suffix}'
        if cfg.decoder.save_path is None:
            save_dir = swm.data.utils.get_cache_dir(sub_folder='checkpoints') / cfg.output_name
            cfg.decoder.save_path = str(save_dir / 'decoder.pt')
    print(f'Run name:  {cfg.output_name}')
    print(f'Save path: {cfg.decoder.save_path}')

    # ---- Output directory -------------------------------------------- #
    # train+eval: folder named after output_name (matches WandB run)
    # eval only:  folder named after timestamp alone
    if cfg.decoder.load_path:
        out_dir = Path(cfg.output.eval_dir) / ts
    else:
        out_dir = Path(cfg.output.train_eval_dir) / cfg.output_name
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / 'config.yaml', 'w') as _f:
        OmegaConf.save(cfg, _f)
    print(f'Output dir: {out_dir}/')

    # ---- LeWM -------------------------------------------------------- #
    print(f'Loading LeWM from {cfg.checkpoint} …')
    model = _load_lewm(cfg.checkpoint).to(device).eval()
    model.requires_grad_(False)

    n_context = model.predictor.num_frames
    embed_dim = model.predictor.input_dim
    print(f'  n_context={n_context}  embed_dim={embed_dim}')

    frameskip = _infer_frameskip(model, cfg.dataset)
    print(f'  frameskip={frameskip}')

    # cap n_pred_steps to what the dataset supports
    _probe = swm.data.load_dataset(cfg.dataset, num_steps=1, frameskip=frameskip,
                                    transform=None, keys_to_load=['action'])
    max_eff  = int(np.array(_probe.lengths).max()) // frameskip
    max_pred = max_eff - n_context
    n_pred   = cfg.eval.n_pred_steps
    if n_pred > max_pred:
        print(f'  n_pred_steps capped {n_pred} → {max_pred}')
        n_pred = max_pred
    n_total = n_context + n_pred

    # ---- Decoder ----------------------------------------------------- #
    if cfg.decoder.load_path:
        print(f'\nLoading decoder from {cfg.decoder.load_path} …')
        decoder = CNNDecoder(embed_dim=embed_dim).to(device)
        decoder.load_state_dict(
            torch.load(cfg.decoder.load_path, map_location=device)
        )
    else:
        # Training dataset: pixels only — action not needed for encode()
        train_ds = swm.data.load_dataset(
            cfg.dataset, num_steps=cfg.decoder.num_steps,
            frameskip=frameskip, transform=None, keys_to_load=['pixels'],
        )
        imagenet = dt.dataset_stats.ImageNet
        train_ds.transform = dt.transforms.Compose(
            dt.transforms.ToImage(**imagenet, source='pixels', target='pixels'),
            dt.transforms.Resize(cfg.eval.image_size,
                                 source='pixels', target='pixels'),
        )

        rng = torch.Generator().manual_seed(cfg.decoder.seed)
        train_set, val_set = spt.data.random_split(
            train_ds,
            lengths=[cfg.decoder.train_split, 1 - cfg.decoder.train_split],
            generator=rng,
        )
        mp_ctx = ({'multiprocessing_context': 'fork'}
                  if cfg.decoder.num_workers > 0 else {})
        train_loader = DataLoader(
            train_set, batch_size=cfg.decoder.batch_size,
            shuffle=True, drop_last=True,
            num_workers=cfg.decoder.num_workers,
            pin_memory=(device.type == 'cuda'),
            generator=rng, **mp_ctx,
        )
        val_loader = DataLoader(
            val_set, batch_size=cfg.decoder.batch_size,
            num_workers=cfg.decoder.num_workers,
            pin_memory=(device.type == 'cuda'),
            **mp_ctx,
        )

        decoder = CNNDecoder(embed_dim=embed_dim)
        module  = DecoderModule(model, decoder, lr=cfg.decoder.lr)

        logger = None
        if cfg.wandb.enabled:
            logger = WandbLogger(**cfg.wandb.config)

        save_path = Path(cfg.decoder.save_path)
        trainer = pl.Trainer(
            max_epochs=cfg.decoder.epochs,
            accelerator='gpu' if device.type == 'cuda' else 'cpu',
            devices='auto',
            precision='16-mixed' if device.type == 'cuda' else '32',
            gradient_clip_val=1.0,
            logger=logger,
            num_sanity_val_steps=1,
            callbacks=[SaveDecoderCallback(save_path)],
            enable_checkpointing=False,
        )
        print(f'\nTraining decoder …')
        trainer.fit(module, train_loader, val_loader)
        decoder = module.decoder
        print(f'  Decoder saved → {save_path}')

    decoder = decoder.to(device).eval()
    decoder.requires_grad_(False)

    # ---- Evaluation -------------------------------------------------- #
    print(f'\nSampling episodes (n_total={n_total}, frameskip={frameskip}) …')
    episodes = _episode_start_batches(cfg, n_total, frameskip)
    if not episodes:
        raise RuntimeError('No episodes found — try reducing eval.n_pred_steps.')
    print(f'  Found {len(episodes)} episodes')

    _CORE = {'pixels', 'action'}

    for ep_idx, batch in enumerate(tqdm(episodes, desc='Generating videos')):
        pixels  = batch['pixels'].to(device)   # (1, n_total, 3, H, W)
        actions = batch['action'].to(device)   # (1, n_total, action_dim)
        extra   = {k: v.to(device) for k, v in batch.items() if k not in _CORE}

        info = {
            'pixels': pixels[:, :n_context].unsqueeze(1),
            'action': actions[:, :n_context].unsqueeze(1),
            **{k: v[:, :n_context].unsqueeze(1) for k, v in extra.items()},
        }
        with torch.no_grad():
            info = model.rollout(info, actions.unsqueeze(1),
                                 history_size=n_context)
        pred_emb = info['predicted_emb'][0, 0]   # (n_context + n_pred + 1, D)

        gt_batch = {'pixels': pixels, 'action': actions, **extra}
        with torch.no_grad():
            model.encode(gt_batch)
        gt_emb = gt_batch['emb'][0]              # (n_total, D)

        H_img, W_img = pixels.shape[-2], pixels.shape[-1]
        with torch.no_grad():
            decoded_gt   = decoder(gt_emb,   target_size=(H_img, W_img))
            decoded_pred = decoder(pred_emb, target_size=(H_img, W_img))

        frames = []
        for t in range(n_total):
            gt_np = _to_uint8(pixels[0, t])
            if t < n_context:
                pred_np = _to_uint8(decoded_gt[t])
                label, ctx = f't={t} CONTEXT (recon)', True
            else:
                pred_np = _to_uint8(decoded_pred[t])
                label, ctx = f't={t} PRED +{t - n_context + 1}', False
            frames.append(_make_frame(gt_np, pred_np, label, ctx))

        vid_path = out_dir / f'episode_{ep_idx:04d}.mp4'
        _write_video(frames, vid_path, cfg.output.fps)

    print(f'\nDone — {len(episodes)} videos + config saved to {out_dir}/')


if __name__ == '__main__':
    run()
