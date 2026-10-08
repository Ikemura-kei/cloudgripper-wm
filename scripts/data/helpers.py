
from collections import defaultdict

import lance
import numpy as np
import torch
from loguru import logger as logging
from tqdm import tqdm


from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from stable_worldmodel.data.format import get_format


def _lance_path(output: str, lance_name:str) -> str:
    """Derive the Lance dataset path inside the output folder."""
    p = Path(output)
    return str(p / (lance_name + '.lance'))


def _dataset_path(output: str, name: str, format: str = 'lance') -> str:
    """Path the writer actually writes this dataset to.

    Non-lance formats get their own sibling path, so a video dataset and a
    lance dataset of the same name can coexist without one being mistaken for
    the other when counting what has already been collected.
    """
    p = _lance_path(output, name)
    return p if format == 'lance' else f'{p}_{format}'


def _config_path(output: str, fname: str = None, format: str = 'lance') -> Path:
    if fname is None:
        return Path(output) / f'config.yaml'
    # Qualify by format so switching format doesn't read back as a changed
    # config on resume — each format's dataset has its own config beside it.
    stem = fname if format == 'lance' else f'{fname}_{format}'
    return Path(output) / f'{stem}_config.yaml'


def _count_existing_episodes(output: str, lance_name:str, format: str = 'lance') -> int:
    """Return number of episodes already collected, or 0 if absent.

    Must be told the format: the video writer stores episodes as a directory
    of per-column .npz files beside the lance path, so checking the lance path
    for a video run would always report 0 and silently re-collect everything.
    """
    path = Path(_dataset_path(output, lance_name, format))
    if not path.exists():
        return 0
    if format == 'video':
        ep_len = path / 'ep_len.npz'
        if not ep_len.exists():
            return 0
        try:
            with np.load(ep_len) as d:
                return int(len(d[next(iter(d))]))
        except Exception:
            return 0
    try:
        col = lance.dataset(str(path)).to_table(columns=["episode_idx"]).column("episode_idx").to_pylist()
        return max(col) + 1 if col else 0
    except Exception:
        return 0


def _save_config(cfg: DictConfig, output: str, fname: str, format: str = 'lance') -> None:
    config_path = _config_path(output, fname, format)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, config_path)
    logging.info(f'Config saved → {config_path}')


def _collect_materialized(
    world,
    path: str,
    episodes: int,
    seed: int,
    format: str = 'lance',
    options: dict | None = None,
) -> None:
    """Same rollout/write behavior as ``World.collect()``, except every
    episode is fully rolled out (and rendered) in the calling thread
    *before* any of it is handed to the writer.

    ``options`` is forwarded to ``envs.reset()`` exactly like
    ``World.collect()``'s own ``options`` — e.g. ``{'variation': [...]}``
    to pick which parts of the env's variation space get resampled each
    episode (see the target env's ``DEFAULT_VARIATIONS`` for what happens
    when this is left as ``None``).

    ``World.collect()`` streams a lazy generator straight into
    ``writer.write_episodes()``. For the Lance format, lancedb pulls that
    generator's batches from its own background event-loop thread (and,
    past the first batch, from a Tokio worker-pool thread that can differ
    call to call) — see ``lancedb/background_loop.py``. Env stepping
    inside that generator calls ``render()``, and MuJoCo's GL/EGL context
    is bound to whichever OS thread first made it current; a later pull
    from a different thread hits ``EGL_BAD_ACCESS`` /
    ``GLX ... BadAccess`` on ``eglMakeCurrent``/``glXMakeCurrent``. This
    was latent but unobserved as long as every env truncated in lockstep
    at a fixed ``max_episode_steps``, since ``write_episodes`` only ever
    saw one giant first batch; per-env early truncation (see
    ``CloudgripperMuJoCoCube.truncate_episode``) makes envs reset
    individually at different times, which is what actually exercises the
    later, differently-threaded pulls.

    Materializing here means ``write_episodes()`` only ever iterates
    already-built plain Python/numpy data — no env code, hence no GL
    calls, run during the writer's (possibly cross-thread) pull.
    """
    buffers = [defaultdict(list) for _ in range(world.num_envs)]

    def on_step(w):
        for col, data in w.infos.items():
            if col.startswith('_'):
                continue
            if not isinstance(data, (np.ndarray, torch.Tensor)):
                continue
            if data.ndim > 1 and data.shape[1] == 1:
                if isinstance(data, torch.Tensor):
                    data = data.squeeze(1)
                else:
                    data = np.squeeze(data, axis=1)
            for i in range(w.num_envs):
                val = data[i]
                if isinstance(val, torch.Tensor):
                    val = val.detach().cpu().numpy()
                elif isinstance(val, np.ndarray):
                    val = val.copy()
                buffers[i][col].append(val)

    all_episodes = []
    with tqdm(total=episodes, desc='Recording') as pbar:
        # 'wait', not 'auto': under 'auto' a chunk returns as soon as `episodes`
        # rollouts have *finished*, and short episodes finish first — so when
        # episode length varies with a sampled variation (ball speed, say), the
        # fast ones are over-represented and slow ones still in flight are
        # thrown away. Measured on a [0.045, 0.20] m/s band: 285 episodes in the
        # fastest tenth of the range against 4 in the slowest. 'wait' freezes
        # each env as it finishes and returns once all are done, so every env
        # contributes exactly one episode and the sampled distribution survives.
        for env_idx, _ in world._run_iter(
            episodes=episodes, seed=seed, options=options, mode='wait', on_step=on_step
        ):
            ep = {k: list(v) for k, v in buffers[env_idx].items()}
            buffers[env_idx].clear()
            if 'action' in ep:
                ep['action'].append(ep['action'].pop(0))
            pbar.update(1)
            all_episodes.append(ep)

    with get_format(format).open_writer(path) as w:
        w.write_episodes(all_episodes)


def _check_config_compatibility(
    cfg: DictConfig, output: str, fname: str = None, format: str = 'lance'
) -> None:
    """Raise if a saved config exists and anything other than `episodes` changed.

    Pass the same `fname`/`format` the config was saved under — looking in the
    wrong place makes this a silent no-op that lets a resume mix incompatible
    settings into one dataset.
    """
    config_path = _config_path(output, fname, format)
    if not config_path.exists():
        return
    saved = OmegaConf.to_container(OmegaConf.load(config_path), resolve=False)
    current = OmegaConf.to_container(cfg, resolve=False)
    saved.pop('episodes', None)
    current.pop('episodes', None)
    if current != saved:
        all_keys = set(saved) | set(current)
        changed = [k for k in all_keys if saved.get(k) != current.get(k)]
        raise ValueError(
            f'Config mismatch on resume — changed keys: {changed}. '
            f'Use a different output path or delete the existing dataset to start fresh.'
        )

