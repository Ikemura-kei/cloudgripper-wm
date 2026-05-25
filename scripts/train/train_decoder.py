"""Train a CNN decoder to reconstruct images from frozen LeWM encoder embeddings.

Loads the full LeWM model from a checkpoint, freezes it, and trains only
a CNN decoder to reconstruct images from the projected CLS token embeddings.

Usage:
    Edit DATASET_PATH and CHECKPOINT_PATH below, then:
    uv run python scripts/train/train_decoder.py
"""

# ============================================================
# Configuration — edit before running
# ============================================================
DATASET_PATH    = "/mimer/NOBACKUP/groups/softenable-codesign26/kei/cloudgripper-wm/data/cloudgripper.lance"     # absolute path to cloudgripper.lance
CHECKPOINT_PATH = "/mimer/NOBACKUP/groups/softenable-codesign26/kei/.stable_worldmodel/checkpoints/lewm/weights_epoch_92.pt"     # path to weights_epoch_N.pt from LeWM training
                            # e.g. ~/.stable-worldmodel/checkpoints/lewm-.../weights_epoch_50.pt

EMBED_DIM       = 192       # must match LeWM embed_dim (tiny ViT)
IMAGE_SIZE      = 224

N_STEPS         = 4         # wm.history_size + wm.num_preds  (3 + 1)
FRAMESKIP       = 1

BATCH_SIZE      = 32
LR              = 1e-3
MAX_EPOCHS      = 50
TRAIN_SPLIT     = 0.9
SEED            = 42
NUM_WORKERS     = 0         # keep 0 — Lance reader holds thread locks

WANDB_ENABLED   = False
WANDB_PROJECT   = "cloudgripper-wm"
WANDB_RUN_NAME  = "decoder"
# ============================================================

import datetime
import random
import string

import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from lightning.pytorch.loggers import WandbLogger
from stable_pretraining import data as dt
from stable_worldmodel.wm.utils import load_pretrained
from torch.utils.data import DataLoader


# ------------------------------------------------------------------ #
#  Decoder                                                             #
# ------------------------------------------------------------------ #

class CNNDecoder(nn.Module):
    """Upsamples a flat embedding back to a 3×224×224 image.

    Spatial path: Linear → (128,7,7) → 5× ConvTranspose2d → (3,224,224)
    Each ConvTranspose2d(k=4, s=2, p=1) doubles spatial dims:
        7 → 14 → 28 → 56 → 112 → 224
    Output is in ImageNet-normalised space (same as input) — no final activation.
    """

    def __init__(self, embed_dim: int = EMBED_DIM):
        super().__init__()
        self.proj = nn.Linear(embed_dim, 128 * 7 * 7)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(128, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64,  4, stride=2, padding=1),
            nn.BatchNorm2d(64),  nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64,  32,  4, stride=2, padding=1),
            nn.BatchNorm2d(32),  nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32,  16,  4, stride=2, padding=1),
            nn.BatchNorm2d(16),  nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16,  3,   4, stride=2, padding=1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:  # (N, D) → (N, 3, 224, 224)
        return self.net(self.proj(z).view(z.size(0), 128, 7, 7))


# ------------------------------------------------------------------ #
#  Lightning module                                                    #
# ------------------------------------------------------------------ #

_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class DecoderModule(pl.LightningModule):
    def __init__(self, world_model: nn.Module, decoder: CNNDecoder):
        super().__init__()
        self.world_model = world_model  # frozen
        self.decoder = decoder

    def _encode(self, batch: dict) -> torch.Tensor:
        """Run the full LeWM encode path, return flat embeddings (B*T, D)."""
        with torch.no_grad():
            self.world_model.encode(batch)        # adds 'emb': (B, T, D)
        return rearrange(batch['emb'], 'b t d -> (b t) d')

    def _step(self, batch: dict, stage: str):
        pixels = batch['pixels']                                    # (B, T, C, H, W)
        flat   = rearrange(pixels, 'b t c h w -> (b t) c h w')
        emb    = self._encode(batch)                                # (B*T, D)
        recon  = self.decoder(emb)                                  # (B*T, 3, H, W)
        loss   = F.mse_loss(recon, flat)
        self.log(f'{stage}/mse_loss', loss, on_epoch=True, on_step=False, prog_bar=True)
        return loss, recon, flat

    def training_step(self, batch, batch_idx):
        loss, _, _ = self._step(batch, 'train')
        return loss

    def validation_step(self, batch, batch_idx):
        loss, recon, orig = self._step(batch, 'val')
        if batch_idx == 0 and WANDB_ENABLED and isinstance(self.logger, WandbLogger):
            self._log_images(orig, recon)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.decoder.parameters(), lr=LR)

    def _log_images(self, orig: torch.Tensor, recon: torch.Tensor, n: int = 4):
        import wandb
        mean = _MEAN.to(orig)
        std  = _STD.to(orig)
        denorm = lambda t: (t * std + mean).clamp(0, 1)
        imgs = []
        for i in range(min(n, orig.size(0))):
            imgs.append(wandb.Image(denorm(orig[i]),  caption=f"orig_{i}"))
            imgs.append(wandb.Image(denorm(recon[i]), caption=f"recon_{i}"))
        self.logger.experiment.log({
            "val/reconstructions": imgs,
            "trainer/global_step": self.global_step,
        })


# ------------------------------------------------------------------ #
#  Main                                                                #
# ------------------------------------------------------------------ #

def main():
    torch.manual_seed(SEED)

    assert DATASET_PATH    != "???", "Set DATASET_PATH at the top of the script"
    assert CHECKPOINT_PATH != "???", "Set CHECKPOINT_PATH at the top of the script"

    # Dataset — pixels only, no action needed for reconstruction
    dataset = swm.data.load_dataset(
        DATASET_PATH, num_steps=N_STEPS, frameskip=FRAMESKIP,
        transform=None, keys_to_load=['pixels'],
    )
    imagenet = dt.dataset_stats.ImageNet
    dataset.transform = dt.transforms.Compose(
        dt.transforms.ToImage(**imagenet, source='pixels', target='pixels'),
        dt.transforms.Resize(IMAGE_SIZE, source='pixels', target='pixels'),
    )

    rng = torch.Generator().manual_seed(SEED)
    train_set, val_set = spt.data.random_split(
        dataset, [TRAIN_SPLIT, 1 - TRAIN_SPLIT], generator=rng,
    )
    train_loader = DataLoader(
        train_set, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, drop_last=True, pin_memory=True, generator=rng,
    )
    val_loader = DataLoader(
        val_set, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True,
    )

    # Load full LeWM model and freeze it
    world_model = load_pretrained(CHECKPOINT_PATH)
    world_model.eval().requires_grad_(False)

    decoder = CNNDecoder(embed_dim=EMBED_DIM)
    module  = DecoderModule(world_model, decoder)

    logger = None
    if WANDB_ENABLED:
        ts     = datetime.datetime.now().strftime('%y-%m-%d-%H-%M-%S')
        suffix = ''.join(random.choices(string.ascii_lowercase, k=3))
        logger = WandbLogger(project=WANDB_PROJECT, name=f"{WANDB_RUN_NAME}-{ts}-{suffix}")

    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator='gpu', devices='auto',
        precision='16-mixed',
        gradient_clip_val=1.0,
        logger=logger,
        num_sanity_val_steps=1,
    )
    trainer.fit(module, train_loader, val_loader)


if __name__ == '__main__':
    main()
