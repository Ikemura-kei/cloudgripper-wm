"""Collect ball-crossing episodes from the CloudGripper MuJoCo simulation.

Each episode is one constant-speed left->right sweep of the ball, with the
speed resampled per episode. Intended as a minimal, near-single-factor dataset
for latent-space studies of world models.

Usage:
    uv run python scripts/data/collect_cloudgripper_mj_ball.py
    uv run python scripts/data/collect_cloudgripper_mj_ball.py variation="[ball.speed,ball.color]"
"""

import os

# Must be set before mujoco is imported (directly or via environments.cloudgripper_mj
# below), since it picks the GL backend at import/first-use time. Default GLFW/GLX
# offscreen rendering is prone to "X Error ... GLX ... BadAccess" crashes once envs
# reset asynchronously (each env truncates independently, so resets no longer stay
# in lockstep across the pool) — EGL sidesteps X11/GLX entirely. setdefault() so an
# explicitly exported MUJOCO_GL still wins.
os.environ.setdefault("MUJOCO_GL", "egl")

import hydra

import stable_worldmodel as swm
from loguru import logger as logging
from omegaconf import DictConfig
from hydra.utils import instantiate
from helpers import _dataset_path, _count_existing_episodes, _check_config_compatibility, _save_config, _collect_materialized

import environments.cloudgripper_mj  # noqa: F401  (triggers gymnasium registration)
from environments.cloudgripper_mj.cloudgripper_mj_ball import CloudgripperMuJoCoBall


@hydra.main(version_base=None, config_path='./config', config_name='cloudgripper_mj_ball')
def run(cfg: DictConfig) -> None:
    format = 'video' if cfg.as_video else 'lance'
    lance_out = _dataset_path(cfg.output, cfg.output_name, format)

    n_existing = _count_existing_episodes(cfg.output, cfg.output_name, format)
    if n_existing > 0:
        _check_config_compatibility(cfg, cfg.output, cfg.output_name, format)
    to_collect = max(0, cfg.episodes - n_existing)

    if n_existing > 0:
        logging.info(
            f'Dataset exists: {n_existing} episodes. '
            f'Target: {cfg.episodes}. Collecting {to_collect} more.'
        )

    if to_collect == 0:
        logging.info('Target episode count already reached, nothing to collect.')
        return

    # A chunk consumes more seeds than the `chunk` episodes it stores: World's
    # rollout seeds one episode per env up front, then — once episodes end at
    # different times, which is always here since a fast ball crosses sooner —
    # resets each finished env with the *next* seed in the sequence and starts
    # another episode. Stepping the cursor by `chunk` would hand the next
    # chunk a seed the previous one already drew from, duplicating a whole
    # episode's variation. Step past every seed a chunk could touch instead
    # (num_envs initial + at most one per episode it stores), and derive the
    # resume point from chunks already done so the same holds across runs
    # (num_envs can't change on resume — _check_config_compatibility rejects
    # that).
    seed_stride = 2 * cfg.num_envs
    chunks_done = -(-n_existing // cfg.num_envs)  # ceil
    seed_cursor = cfg.seed + chunks_done * seed_stride
    _save_config(cfg, cfg.output, cfg.output_name, format)

    variation = cfg.get('variation', None)
    options = {'variation': list(variation)} if variation is not None else None

    speed_range = tuple(cfg.world.speed_range)

    # A crossing at the slowest sampled speed must fit inside the step cap,
    # or those episodes get truncated mid-sweep and the dataset quietly stops
    # being "one complete crossing per episode" — easy to miss, and it only
    # bites the slow bands.
    slowest = CloudgripperMuJoCoBall.crossing_steps(speed_range[0])
    if slowest > cfg.world.max_episode_steps:
        raise ValueError(
            f'speed_range={speed_range}: a crossing at {speed_range[0]} m/s takes '
            f'{slowest} steps but max_episode_steps is {cfg.world.max_episode_steps}. '
            f'Raise world.max_episode_steps to at least {slowest}.'
        )

    world = swm.World(
        "cloudgripper_mj/Ball-v0",
        num_envs=cfg.num_envs,
        image_shape=tuple(cfg.world.image_shape),
        max_episode_steps=cfg.world.max_episode_steps,
        height=cfg.world.height,
        width=cfg.world.width,
        speed_range=speed_range,
    )
    policy = instantiate(cfg.policy)
    world.set_policy(policy)

    try:
        collected = 0
        while collected < to_collect:
            chunk = min(cfg.num_envs, to_collect - collected)
            if hasattr(policy, 'reset'):
                policy.reset()
            _collect_materialized(world, path=lance_out, episodes=chunk, seed=seed_cursor, format=format, options=options)
            collected += chunk
            seed_cursor += seed_stride
            logging.info(
                f'Collected {n_existing + collected}/{cfg.episodes} episodes → {lance_out}'
            )
    finally:
        world.close()


if __name__ == "__main__":
    run()
