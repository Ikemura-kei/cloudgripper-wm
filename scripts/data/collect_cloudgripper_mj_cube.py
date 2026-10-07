"""Collect random trajectories from the CloudGripper MuJoCo simulation.

Usage:
    uv run python scripts/data/collect_cloudgripper_mujoco.py
"""

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import hydra

import stable_worldmodel as swm
from loguru import logger as logging
from omegaconf import DictConfig
from hydra.utils import instantiate
from helpers import _lance_path, _count_existing_episodes, _check_config_compatibility, _save_config, _collect_materialized

import environments.cloudgripper_mj  # noqa: F401  (triggers gymnasium registration)


@hydra.main(version_base=None, config_path='./config', config_name='cloudgripper_mj_cube')
def run(cfg: DictConfig) -> None:
    lance_out = _lance_path(cfg.output, cfg.output_name)

    format = 'lance'
    if cfg.as_video:
        format = "video"
        lance_out += f"_{format}" 

    n_existing = _count_existing_episodes(cfg.output, cfg.output_name)
    if n_existing > 0:
        _check_config_compatibility(cfg, cfg.output)
    to_collect = max(0, cfg.episodes - n_existing)

    if n_existing > 0:
        logging.info(
            f'Dataset exists: {n_existing} episodes. '
            f'Target: {cfg.episodes}. Collecting {to_collect} more.'
        )

    if to_collect == 0:
        logging.info('Target episode count already reached, nothing to collect.')
        return

    seed_start = cfg.seed + n_existing
    _save_config(cfg, cfg.output, cfg.output_name)

    variation = cfg.get('variation', None)
    options = {'variation': list(variation)} if variation is not None else None

    world = swm.World(
        "cloudgripper_mj/Cube-v0",
        num_envs=cfg.num_envs,
        image_shape=tuple(cfg.world.image_shape),
        max_episode_steps=cfg.world.max_episode_steps,
        height=cfg.world.height,
        width=cfg.world.width,
        mode="task",
    )
    policy = instantiate(cfg.policy)
    world.set_policy(policy)

    try:
        collected = 0
        while collected < to_collect:
            chunk = min(cfg.num_envs, to_collect - collected)
            seed = seed_start + collected
            if hasattr(policy, 'reset'):
                policy.reset()
            _collect_materialized(world, path=lance_out, episodes=chunk, seed=seed, format=format, options=options)
            collected += chunk
            logging.info(
                f'Collected {n_existing + collected}/{cfg.episodes} episodes → {lance_out}'
            )
    finally:
        world.close()
    

if __name__ == "__main__":
    run()
