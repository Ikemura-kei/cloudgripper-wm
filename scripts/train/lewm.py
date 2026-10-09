import datetime
import os
import random
import string
from pathlib import Path

import hydra
import lightning as pl
import stable_pretraining as spt
from stable_pretraining import data as dt
import stable_worldmodel as swm
import torch
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Subset

from stable_worldmodel.data import column_normalizer as get_column_normalizer
from stable_worldmodel.wm.loss import SIGReg
from lightning.pytorch.callbacks import Callback
from stable_worldmodel.wm.utils import save_pretrained

# Needed since we may use only one worker.
_orig_on_train_start = spt.Module.on_train_start

def _on_train_start_compat(self):
    _orig = self.optimizers
    def _list_wrap(*args, **kwargs):
        r = _orig(*args, **kwargs)
        return r if isinstance(r, (list, tuple)) else [r]
    self.optimizers = _list_wrap
    _orig_on_train_start(self)
    del self.optimizers

spt.Module.on_train_start = _on_train_start_compat


def get_img_preprocessor(source: str, target: str, img_size: int = 224):
    imagenet_stats = dt.dataset_stats.ImageNet
    to_image = dt.transforms.ToImage(
        **imagenet_stats, source=source, target=target
    )
    resize = dt.transforms.Resize(img_size, source=source, target=target)
    return dt.transforms.Compose(to_image, resize)


class SaveCkptCallback(Callback):
    """Callback to save model checkpoint after each epoch using save_pretrained."""

    def __init__(self, run_name, cfg, epoch_interval: int = 1):
        super().__init__()
        self.run_name = run_name
        self.cfg = cfg
        self.epoch_interval = epoch_interval

    def on_train_epoch_end(self, trainer, pl_module):
        super().on_train_epoch_end(trainer, pl_module)

        if trainer.is_global_zero:
            if (trainer.current_epoch + 1) % self.epoch_interval == 0:
                self._save(pl_module.model, trainer.current_epoch + 1)

            # save final epoch
            if (trainer.current_epoch + 1) == trainer.max_epochs:
                self._save(pl_module.model, trainer.current_epoch + 1)

    def _save(self, model, epoch):
        save_pretrained(
            model,
            run_name=self.run_name,
            config=self.cfg,
            config_key='model',
            filename=f'weights_epoch_{epoch}.pt',
        )


def _memory_budget_bytes():
    """RAM we are actually allowed, honouring a SLURM/cgroup cap if present."""
    budget = None
    try:
        for line in open('/proc/meminfo'):
            if line.startswith('MemAvailable'):
                budget = int(line.split()[1]) * 1024
                break
    except OSError:
        pass
    for path in ('/sys/fs/cgroup/memory.max',
                 '/sys/fs/cgroup/memory/memory.limit_in_bytes'):
        try:
            raw = open(path).read().strip()
        except OSError:
            continue
        if raw and raw != 'max':
            cap = int(raw)
            if 0 < cap < (1 << 60):  # cgroup v1 writes a sentinel when uncapped
                budget = cap if budget is None else min(budget, cap)
    return budget


def check_loader_memory(cfg, dataset, fraction: float = 0.6):
    """Refuse to start if the dataloader's in-flight batches will not fit in RAM.

    A sample here is several frames, not one image, so batches are much larger
    than in ordinary vision training, and with num_workers > 0 roughly
    num_workers * prefetch_factor of them are resident at once — each copied
    between the worker, shared memory and the main process. Two runs were
    killed mid-training by this (one local at 14.6 GB resident, one cluster
    job against a 32 GB --mem cap), both after the queue wait and several
    minutes of training, which is an expensive way to learn the number.

    The per-sample size is measured from a real sample rather than derived, so
    it accounts for whatever the transform actually produces.
    """
    def _bytes(sample, as_float32=False):
        total = 0
        for v in sample.values():
            if isinstance(v, torch.Tensor):
                n, size = v.numel(), v.element_size()
            elif hasattr(v, 'nbytes') and hasattr(v, 'dtype'):  # numpy
                n, size = v.size, v.itemsize
            else:
                continue
            total += n * (4 if as_float32 and size == 1 else size)
        return total

    per_sample = _bytes(dataset[0])
    # Peak is not the transformed sample: images are converted to float32 at
    # their *stored* resolution and only then resized, so a dataset stored
    # larger than img_size peaks well above its final size. The cube set is
    # stored at 480 and trained at 224 — 1.32 GB per batch mid-transform
    # against 0.29 GB after it, which is what a post-transform-only estimate
    # misses (and why a 32 GB cluster job died while the estimate said 15.6).
    tf = getattr(dataset, 'transform', None)
    if tf is not None:
        try:
            dataset.transform = None
            per_sample = max(per_sample, _bytes(dataset[0], as_float32=True))
        except Exception:
            pass
        finally:
            dataset.transform = tf
    workers = int(cfg.loader.num_workers)
    prefetch = int(cfg.loader.get('prefetch_factor', 2) or 2)
    # 3x: measured. 7 workers x prefetch 2 on a 303 MB batch predicts 4.2 GB,
    # resident was 14.6 GB — the gap is the worker -> shm -> main copies, plus
    # a page-locked duplicate when pin_memory is on.
    in_flight = max(1, workers * prefetch) * (3.0 if workers else 1.0)
    per_batch = per_sample * int(cfg.loader.batch_size)
    # each spawned worker is a fresh interpreter with torch imported
    overhead = workers * 0.5 * 1024 ** 3
    estimate = per_batch * in_flight + overhead

    budget = _memory_budget_bytes()
    gb = 1024 ** 3
    print(
        f'loader memory estimate: {per_batch/gb:.2f} GB/batch x {in_flight:.0f} in flight '
        f'+ {overhead/gb:.1f} GB worker overhead = {estimate/gb:.1f} GB'
        + (f' (budget {budget/gb:.1f} GB)' if budget else ' (budget unknown)')
    )
    if budget and estimate > fraction * budget:
        raise MemoryError(
            f'dataloader needs ~{estimate/gb:.1f} GB but only {budget/gb:.1f} GB is available. '
            f'Lower loader.num_workers (currently {workers}) or loader.batch_size '
            f'(currently {cfg.loader.batch_size}), or request more memory '
            f'(--mem on SLURM). Set fraction higher only if you know the estimate is pessimistic.'
        )


def split_by_episode(dataset, train_frac: float, generator):
    """Split a dataset's clip windows into train/val by *episode*.

    Splitting the dataset directly (e.g. with random_split) splits its clip
    windows, and consecutive windows overlap by span-1 frames — so windows
    from one episode land on both sides and the val set scores frames the
    model trained on. With 500 ball episodes that left every single episode
    represented in training, making the val loss a near-training metric that
    could not reveal memorisation. Splitting whole episodes instead means no
    val episode is seen during training at all.

    Always leaves at least one episode on each side, so train_frac=1.0 still
    produces a usable (if token) validation set rather than an empty loader.

    Args:
        dataset: dataset exposing ``clip_indices`` as (episode, start) pairs.
        train_frac: fraction of *episodes* used for training.
        generator: torch Generator, for a reproducible episode permutation.

    Returns:
        (train_subset, val_subset)
    """
    clip_eps = [ep for ep, _ in dataset.clip_indices]
    episodes = sorted(set(clip_eps))

    perm = torch.randperm(len(episodes), generator=generator).tolist()
    n_train = int(round(train_frac * len(episodes)))
    n_train = max(1, min(n_train, len(episodes) - 1))
    train_eps = {episodes[i] for i in perm[:n_train]}

    train_idx = [i for i, ep in enumerate(clip_eps) if ep in train_eps]
    val_idx = [i for i, ep in enumerate(clip_eps) if ep not in train_eps]

    print(
        f'Episode-level split: {n_train}/{len(episodes)} episodes to train '
        f'({len(train_idx)} windows), {len(episodes) - n_train} to val '
        f'({len(val_idx)} windows)'
    )
    return Subset(dataset, train_idx), Subset(dataset, val_idx)


def lejepa_forward(self, batch, stage):
    """encode observations, predict next states, compute losses."""

    cfg = self._lewm_cfg
    ctx_len = cfg.wm.history_size
    n_preds = cfg.wm.num_preds
    lambd = cfg.loss.sigreg.weight

    # Replace NaN values with 0 (occurs at sequence boundaries)
    batch['action'] = torch.nan_to_num(batch['action'], 0.0)

    output = self.model.encode(batch)

    emb = output['emb']  # (B, T, D)
    act_emb = output['act_emb']

    ctx_emb = emb[:, :ctx_len]
    ctx_act = act_emb[:, :ctx_len]

    tgt_emb = emb[:, n_preds:]  # label
    pred_emb = self.model.predict(ctx_emb, ctx_act)  # pred

    # LeWM loss
    output['pred_loss'] = (pred_emb - tgt_emb).pow(2).mean()
    output['sigreg_loss'] = self.sigreg(emb.transpose(0, 1))
    output['loss'] = output['pred_loss'] + lambd * output['sigreg_loss']

    losses_dict = {
        f'{stage}/{k}': v.detach() for k, v in output.items() if 'loss' in k
    }
    self.log_dict(losses_dict, on_step=True, sync_dist=True)
    return output


def train(cfg):
    """Core LeWM training loop, shared across dataset-specific entry points
    (e.g. lewm.py, lewm_mujoco.py) that only differ in Hydra config."""
    ts = datetime.datetime.now().strftime('%y-%m-%d-%H-%M-%S')
    suffix = ''.join(random.choices(string.ascii_lowercase, k=3))
    with open_dict(cfg):
        cfg.output_model_name = f"{cfg.output_model_name}-{ts}-{suffix}"

    #########################
    ##       dataset       ##
    #########################

    dataset_cfg = OmegaConf.to_container(cfg.data.dataset, resolve=True)
    dataset_name = dataset_cfg.pop('name')
    cache_dir = os.environ.get('LOCAL_DATASET_DIR', None)
    print(
        f'Loading dataset "{dataset_name}" from {"local cache: " + cache_dir if cache_dir else "default location"}'
    )
    dataset = swm.data.load_dataset(
        dataset_name, transform=None, cache_dir=cache_dir, **dataset_cfg
    )
    transforms = [
        get_img_preprocessor(
            source='pixels', target='pixels', img_size=cfg.img_size
        )
    ]

    with open_dict(cfg):
        for col in cfg.data.dataset.keys_to_load:
            if col.startswith('pixels'):
                continue

            normalizer = get_column_normalizer(dataset, col, col)
            transforms.append(normalizer)

        cfg.model.action_encoder.input_dim = (
            cfg.data.dataset.frameskip * dataset.get_dim('action')
        )

    transform = spt.data.transforms.Compose(*transforms)
    dataset.transform = transform

    check_loader_memory(cfg, dataset)

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = split_by_episode(dataset, cfg.train_split, rnd_gen)

    # 'spawn', not 'fork': the dataset (e.g. LanceDataset) may already have
    # started lancedb's internal background threads by this point, and
    # lancedb is fork-unsafe (see _force_spawn() in
    # stable_worldmodel/data/formats/lance.py) — forking leaves locks those
    # threads held stuck in the child, hanging the first worker to read a
    # row. Only matters once num_workers > 0 actually spawns workers.
    mp_ctx = {'multiprocessing_context': 'spawn'} if cfg.loader.num_workers > 0 else {}
    train = torch.utils.data.DataLoader(
        train_set,
        **cfg.loader,
        generator=rnd_gen,
        **mp_ctx,
    )
    val_cfg = {**cfg.loader}
    val_cfg['shuffle'] = False
    val_cfg['drop_last'] = False
    val = torch.utils.data.DataLoader(val_set, **val_cfg, **mp_ctx)

    ##############################
    ##       model / optim      ##
    ##############################

    world_model = hydra.utils.instantiate(cfg.model)

    total_steps = cfg.trainer.max_epochs * len(train)
    optimizers = {
        'model_opt': {
            'modules': 'model',
            'optimizer': dict(cfg.optimizer),
            'scheduler': {
                'type': 'LinearWarmupCosineAnnealingLR',
                'warmup_steps': max(1, int(0.01 * total_steps)),
                'max_steps': total_steps,
            },
            'interval': 'epoch',
        },
    }

    data_module = spt.data.DataModule(train=train, val=val)
    world_model = spt.Module(
        model=world_model,
        sigreg=SIGReg(**cfg.loss.sigreg.kwargs),
        # Plain function reference, not a functools.partial: spt.Module
        # installs this via `types.MethodType(forward, self)`
        # (stable_pretraining/module.py), which under 'spawn' multiprocessing
        # (see mp_ctx below) needs to pickle `m.__func__` as a named,
        # importable function — a partial has no __name__ and breaks that.
        # cfg is attached to the instance below instead of bound via partial.
        forward=lejepa_forward,
        optim=optimizers,
    )
    world_model._lewm_cfg = cfg

    ##########################
    ##       training       ##
    ##########################

    run_id = cfg.get('subdir') or ''
    run_dir = Path(
        swm.data.utils.get_cache_dir(sub_folder='checkpoints'), run_id
    )

    logger = None
    if cfg.wandb.enabled:
        logger = WandbLogger(**cfg.wandb.config)
        logger.log_hyperparams(OmegaConf.to_container(cfg))

    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / 'config.yaml', 'w') as f:
        OmegaConf.save(cfg, f)

    # Every epoch by default (unchanged), but each checkpoint is ~70 MB, so a
    # few hundred epochs is several GB — set ckpt_every in the config to thin
    # them out. The final epoch is always saved regardless of the interval.
    object_dump_callback = SaveCkptCallback(
        run_name=cfg.output_model_name,
        cfg=cfg,
        epoch_interval=cfg.get('ckpt_every', 1),
    )

    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=[object_dump_callback],
        num_sanity_val_steps=1,
        logger=logger,
        enable_checkpointing=True,
    )

    # Lightning writes full training state — weights, optimizer, LR scheduler
    # and epoch/step counters — to lightning_logs/version_*/checkpoints/*.ckpt.
    # The fallback path below can never match one: output_model_name has a
    # fresh timestamp appended at the top of this function, so .exists() is
    # always False and a crashed run silently restarted from epoch 0. Point
    # resume_from at such a .ckpt to actually continue one; dataloaders are
    # rebuilt from the current config either way, so loader settings changed
    # since the checkpoint (e.g. num_workers) still take effect.
    resume_from = cfg.get('resume_from', None)
    ckpt_path = (
        Path(resume_from) if resume_from
        else run_dir / f'{cfg.output_model_name}_weights.ckpt'
    )
    if resume_from and not ckpt_path.is_file():
        raise FileNotFoundError(f'resume_from={resume_from} does not exist')
    manager = spt.Manager(
        trainer=trainer,
        module=world_model,
        data=data_module,
        ckpt_path=ckpt_path if ckpt_path.exists() else None,
    )

    manager()
    return


@hydra.main(version_base=None, config_path='./config', config_name='lewm')
def run(cfg):
    train(cfg)


if __name__ == '__main__':
    run()
