"""Visualize LeWM autoregressive trajectory prediction decoded to pixels.

Loads a sequence from the dataset, uses the first N_CONTEXT frames as context,
autoregressively rolls out the world model for N_PRED_STEPS using real actions,
decodes all embeddings with the CNN decoder, and writes a side-by-side MP4.

  Left panel : ground truth
  Right panel: decoded prediction (green = context reconstruction, red = WM prediction)

Usage:
    Edit the config block below, then:
    uv run python scripts/debug/visualize_prediction.py
"""

# ============================================================
# Configuration — edit before running
# ============================================================
DATASET_PATH       = "/mimer/NOBACKUP/groups/softenable-codesign26/kei/cloudgripper-wm/data/cloudgripper.lance"
LEWM_CHECKPOINT    = "/mimer/NOBACKUP/groups/softenable-codesign26/kei/.stable_worldmodel/checkpoints/lewm/weights_epoch_92.pt"
DECODER_CHECKPOINT = "???"   # path to decoder_epoch_N.pt
OUTPUT_DIR         = "predictions"   # directory; one traj_N.mp4 per trajectory

EMBED_DIM    = 192
IMAGE_SIZE   = 224
N_CONTEXT    = 3     # history_size — must match LeWM training config
N_PRED_STEPS = 20    # autoregressive steps to predict beyond context
N_TRAJ          = 5     # number of trajectories to visualize
EPISODE_INDICES = None  # None = auto-pick N_TRAJ distinct episodes in order
                        # or set to a list e.g. [0, 3, 7] to pick specific episodes
FRAMESKIP       = 1
FPS             = 4

BGR_DATASET  = True  # images stored as BGR; set False after fixing env.render()
# ============================================================

import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import stable_worldmodel as swm
import stable_pretraining as spt
from einops import rearrange
from hydra.utils import instantiate
from omegaconf import OmegaConf
from PIL import Image, ImageDraw
from stable_pretraining import data as dt
from torch.utils.data import DataLoader


# ------------------------------------------------------------------ #
#  Decoder (kept in sync with train_decoder.py)                        #
# ------------------------------------------------------------------ #

class CNNDecoder(nn.Module):
    def __init__(self, embed_dim: int = EMBED_DIM):
        super().__init__()
        self.proj = nn.Linear(embed_dim, 128 * 7 * 13)
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

    def forward(self, z: torch.Tensor, target_size: tuple[int, int] | None = None) -> torch.Tensor:
        out = self.net(self.proj(z).view(z.size(0), 128, 7, 13))
        if target_size is not None:
            out = F.interpolate(out, size=target_size, mode='bilinear', align_corners=False)
        return out


# ------------------------------------------------------------------ #
#  Model loading (same logic as train_decoder.py)                      #
# ------------------------------------------------------------------ #

def _load_lewm(checkpoint_path: str) -> nn.Module:
    ckpt = Path(checkpoint_path)
    with open(ckpt.parent / 'config.json') as f:
        raw = json.load(f)
    model_cfg = raw.get('model', raw)
    world_model = instantiate(OmegaConf.create(model_cfg))
    world_model.load_state_dict(torch.load(str(ckpt), map_location='cpu'))
    return world_model


# ------------------------------------------------------------------ #
#  Visualization helpers                                               #
# ------------------------------------------------------------------ #

_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def denorm(t: torch.Tensor) -> torch.Tensor:
    mean = _MEAN.to(t)
    std  = _STD.to(t)
    return (t * std + mean).clamp(0, 1)


def tensor_to_np(t: torch.Tensor, bgr: bool = False) -> np.ndarray:
    """(3, H, W) float → (H, W, 3) uint8 RGB."""
    arr = (denorm(t) * 255).byte().permute(1, 2, 0).cpu().numpy()
    if bgr:
        arr = arr[..., ::-1].copy()   # BGR → RGB for display
    return arr


def make_frame(
    gt: np.ndarray,
    pred: np.ndarray,
    label: str,
    is_context: bool,
) -> np.ndarray:
    """Stack gt | pred side-by-side with a label bar."""
    H, W, _ = gt.shape
    bar_h  = 24
    sep_w  = 4
    color  = (30, 200, 30) if is_context else (200, 30, 30)

    canvas = np.zeros((H + bar_h, W * 2 + sep_w, 3), dtype=np.uint8)
    canvas[:H, :W]           = gt
    canvas[:H, W:W + sep_w]  = color
    canvas[:H, W + sep_w:]   = pred
    canvas[H:, :]            = (20, 20, 20)   # dark label bar

    img  = Image.fromarray(canvas)
    draw = ImageDraw.Draw(img)
    draw.text((4, H + 4),          f"GT  | {label}", fill=color)
    draw.text((W + sep_w + 4, H + 4), f"PRED| {label}", fill=color)
    return np.array(img)


# ------------------------------------------------------------------ #
#  Main                                                                #
# ------------------------------------------------------------------ #

def _build_frames(
    pixels: torch.Tensor,       # (N_TOTAL, 3, H, W)
    decoded_gt: torch.Tensor,   # (N_TOTAL, 3, H, W)
    decoded_pred: torch.Tensor, # (N_CONTEXT+N_PRED_STEPS+1, 3, H, W)
) -> list[np.ndarray]:
    frames = []
    N_TOTAL = pixels.shape[0]
    for t in range(N_TOTAL):
        gt_np = tensor_to_np(pixels[t], bgr=BGR_DATASET)
        if t < N_CONTEXT:
            pred_np    = tensor_to_np(decoded_gt[t])
            label      = f"t={t} CONTEXT (recon)"
            is_context = True
        else:
            pred_np    = tensor_to_np(decoded_pred[t])
            label      = f"t={t} PREDICTION (step +{t - N_CONTEXT + 1})"
            is_context = False
        frames.append(make_frame(gt_np, pred_np, label, is_context))
    return frames


def _write_video(frames: list[np.ndarray], path: str) -> None:
    H_f, W_f, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(path, fourcc, FPS, (W_f, H_f))
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()


def main():
    assert DECODER_CHECKPOINT != "???", "Set DECODER_CHECKPOINT at the top of the script"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- models ---------------------------------------------------- #
    print("Loading LeWM …")
    world_model = _load_lewm(LEWM_CHECKPOINT).to(device).eval()

    print("Loading decoder …")
    decoder = CNNDecoder(embed_dim=EMBED_DIM).to(device).eval()
    decoder.load_state_dict(torch.load(DECODER_CHECKPOINT, map_location=device))

    # ---- dataset --------------------------------------------------- #
    N_TOTAL = N_CONTEXT + N_PRED_STEPS
    dataset = swm.data.load_dataset(
        DATASET_PATH, num_steps=N_TOTAL, frameskip=FRAMESKIP,
        transform=None, keys_to_load=['pixels', 'action'],
    )
    imagenet = dt.dataset_stats.ImageNet
    dataset.transform = dt.transforms.Compose(
        dt.transforms.ToImage(**imagenet, source='pixels', target='pixels'),
        dt.transforms.Resize(IMAGE_SIZE, source='pixels', target='pixels'),
    )

    # Detect episode boundaries from NaN actions (stable-worldmodel inserts NaN
    # at transitions between episodes). A window with any NaN action crosses a
    # boundary and is skipped. After every such window the next clean window
    # is the start of a new episode.
    want           = set(EPISODE_INDICES) if EPISODE_INDICES is not None else None
    loader         = DataLoader(dataset, batch_size=1, shuffle=False)
    samples        = {}   # local episode counter → batch
    ep_ctr         = -1
    after_boundary = True   # first clean window counts as a new episode
    print("Scanning dataset for episode windows …")
    for b in loader:
        if torch.isnan(b['action']).any():    # crosses an episode boundary
            after_boundary = True
            continue

        if after_boundary:                    # first clean window of a new episode
            ep_ctr        += 1
            after_boundary = False

        if ep_ctr in samples:                 # already have one for this episode
            continue
        if want is not None and ep_ctr not in want:
            continue

        samples[ep_ctr] = b
        if want is None and len(samples) == N_TRAJ:
            break
        if want is not None and len(samples) == len(want):
            break

    if want is not None:
        missing = want - samples.keys()
        assert not missing, f"Episodes not found or too short: {missing}"
    else:
        assert len(samples) == N_TRAJ, (
            f"Only {len(samples)} distinct episodes found (need {N_TRAJ})"
        )

    episode_order = EPISODE_INDICES if EPISODE_INDICES is not None else sorted(samples)
    samples_list  = [samples[ep] for ep in episode_order]

    # ---- per-trajectory loop --------------------------------------- #
    for traj_idx, (ep, batch) in enumerate(zip(episode_order, samples_list)):
        print(f"\n[{traj_idx + 1}/{len(samples_list)}] episode {ep}")

        pixels  = batch['pixels'].to(device)   # (1, N_TOTAL, 3, H, W)
        actions = batch['action'].to(device)   # (1, N_TOTAL, action_dim)
        H_img, W_img = pixels.shape[-2], pixels.shape[-1]

        # rollout
        info = {
            'pixels': pixels[:, :N_CONTEXT].unsqueeze(1),
            'action': actions[:, :N_CONTEXT].unsqueeze(1),
        }
        with torch.no_grad():
            info = world_model.rollout(info, actions.unsqueeze(1), history_size=N_CONTEXT)
        pred_emb = info['predicted_emb'][0, 0]   # (N_CONTEXT+N_PRED_STEPS+1, D)

        # encode all GT frames for context reconstruction baseline
        gt_batch = {'pixels': pixels, 'action': actions}
        with torch.no_grad():
            world_model.encode(gt_batch)
        gt_emb = gt_batch['emb'][0]   # (N_TOTAL, D)

        # decode
        with torch.no_grad():
            decoded_gt   = decoder(gt_emb,   target_size=(H_img, W_img))
            decoded_pred = decoder(pred_emb, target_size=(H_img, W_img))

        # build and write video
        frames   = _build_frames(pixels[0], decoded_gt, decoded_pred)
        out_path = str(out_dir / f"episode_{ep:04d}.mp4")
        _write_video(frames, out_path)
        print(f"  Saved → {out_path}  ({len(frames)} frames @ {FPS} fps)")


if __name__ == '__main__':
    main()
