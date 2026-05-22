"""Train DINO-WM (PreJEPA) on CloudGripper data.

Uses stable-worldmodel's prejepa training pipeline with CloudGripper config.
For other world models (LeWM, PLDM, etc.) add a separate script alongside this one.

Usage:
    uv run python scripts/train/prejepa.py dataset_name=/abs/path/to/collect.lance
    uv run python scripts/train/prejepa.py dataset_name=$(pwd)/data/collect.lance trainer.max_epochs=200
"""

import sys
from functools import partial
from pathlib import Path

import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from loguru import logger as logging
from omegaconf import OmegaConf, open_dict
from stable_worldmodel.data import column_normalizer as get_column_normalizer
from stable_worldmodel.wm.utils import save_pretrained
from torch.nn import functional as F
from torch.utils.data import DataLoader

# Add stable-worldmodel train scripts to path for shared helpers
_TRAIN_DIR = Path(__file__).parent.parent.parent / 'third_party/stable-worldmodel/scripts/train'
sys.path.insert(0, str(_TRAIN_DIR))
from prejepa import (  # noqa: E402
    SaveCkptCallback,
    VideoPipeline,
    dinowm_forward,
    get_img_preprocessor,
)


@hydra.main(version_base=None, config_path='./config', config_name='prejepa')
def run(cfg) -> None:
    # --- Dataset ---
    encoding_keys = list(cfg.wm.get('encoding', {}).keys())
    keys_to_load = ['pixels'] + encoding_keys

    logging.info(f'Loading dataset: {cfg.dataset_name}')
    dataset = swm.data.load_dataset(
        cfg.dataset_name,
        num_steps=cfg.n_steps,
        frameskip=cfg.frameskip,
        transform=None,
        keys_to_load=keys_to_load,
        keys_to_cache=encoding_keys,
    )

    normalizers = [
        get_column_normalizer(dataset, col, col)
        for col in cfg.wm.get('encoding', {})
    ]
    transform = spt.data.transforms.Compose(
        get_img_preprocessor('pixels', 'pixels', cfg.image_size),
        *normalizers,
    )
    dataset.transform = transform

    with open_dict(cfg) as cfg:
        cfg.extra_dims = {}
        for key in cfg.wm.get('encoding', {}):
            if key not in dataset.column_names:
                raise ValueError(f"Encoding key '{key}' not found in dataset columns: {dataset.column_names}")
            dim = dataset.get_dim(key)
            cfg.extra_dims[key] = dim if key != 'action' else dim * cfg.frameskip

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = spt.data.random_split(
        dataset, [cfg.train_split, 1 - cfg.train_split], generator=rnd_gen
    )

    train_loader = DataLoader(
        train_set, batch_size=cfg.batch_size, num_workers=cfg.num_workers,
        drop_last=True, persistent_workers=True, pin_memory=True,
        shuffle=True, generator=rnd_gen,
    )
    val_loader = DataLoader(
        val_set, batch_size=cfg.batch_size, num_workers=cfg.num_workers, pin_memory=True,
    )

    # --- Model ---
    encoder = hydra.utils.instantiate(cfg.model.encoder)
    encoder.eval()
    encoder.requires_grad_(False)

    is_cnn = hasattr(encoder.config, 'hidden_sizes')
    embed_dim = encoder.config.hidden_sizes[-1] if is_cnn else encoder.config.hidden_size
    num_patches = 1 if is_cnn else (cfg.image_size // cfg.patch_size) ** 2
    embed_dim += sum(cfg.wm.get('encoding', {}).values())

    with open_dict(cfg):
        cfg.model.predictor.dim = embed_dim
        cfg.model.predictor.num_patches = num_patches
        cfg.model.extra_encoders = {
            '_target_': 'torch.nn.ModuleDict',
            'modules': {
                key: {
                    '_target_': 'stable_worldmodel.wm.prejepa.module.Embedder',
                    'in_chans': cfg.extra_dims[key],
                    'emb_dim': int(cfg.wm.encoding[key]),
                }
                for key in cfg.wm.get('encoding', {})
            },
        }

    world_model = hydra.utils.instantiate(cfg.model, encoder=encoder)
    world_model = spt.Module(
        model=world_model,
        forward=partial(dinowm_forward, cfg=cfg),
        optim={'model_opt': {'modules': 'model', 'optimizer': dict(cfg.optimizer), 'scheduler': dict(cfg.scheduler)}},
    )

    # --- Training ---
    run_dir = Path(swm.data.utils.get_cache_dir(sub_folder='checkpoints'), cfg.get('subdir') or '')
    run_dir.mkdir(parents=True, exist_ok=True)

    with open(run_dir / 'config.yaml', 'w') as f:
        OmegaConf.save(cfg, f)

    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=[
            SaveCkptCallback(run_name=cfg.output_model_name, cfg=cfg.model, epoch_interval=5),
            pl.pytorch.callbacks.LearningRateMonitor(logging_interval='step'),
        ],
        num_sanity_val_steps=1,
        enable_checkpointing=True,
    )

    ckpt_path = run_dir / f'{cfg.output_model_name}_weights.ckpt'
    spt.Manager(
        trainer=trainer,
        module=world_model,
        data=spt.data.DataModule(train=train_loader, val=val_loader),
        ckpt_path=ckpt_path if ckpt_path.exists() else None,
    )()

    logging.success(f'Training complete. Model: {cfg.output_model_name}')


if __name__ == '__main__':
    run()
