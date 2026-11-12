"""Collect ball-crossing episodes from the flat 2D scene.

Each episode is one constant-speed left->right sweep of the ball across a
plain wall/floor background, with the speed resampled per episode. The
pixel-space content of the whole dataset is three RGB colours plus the ball's
position, which makes it the most stripped-down dataset in the repo for
latent-space studies of world models — the flat sibling of
``collect_cloudgripper_mj_ball.py``.

Frames are stored as PNG rather than the Lance writer's default JPEG, so the
three colours survive storage byte-exact (see ``lossless_lance.py``).

Usage:
    uv run python scripts/data/collect_flat2d_ball.py
    uv run python scripts/data/collect_flat2d_ball.py variation="[ball.speed,ball.color]"

To collect two datasets that differ *only* in a colour, override
``world.<element>_color`` and keep everything else — the seed above all — the
same. Those are fixed colours rather than sampled ones, so they consume no
variation draws and the two runs stay synchronised episode for episode:

    uv run python scripts/data/collect_flat2d_ball.py \\
        output_name=flat2d_ball_red   world.ball_color=[255,0,0]
    uv run python scripts/data/collect_flat2d_ball.py \\
        output_name=flat2d_ball_green world.ball_color=[0,255,0]
"""

import hydra

import stable_worldmodel as swm
from loguru import logger as logging
from omegaconf import DictConfig
from hydra.utils import instantiate
from helpers import _dataset_path, _count_existing_episodes, _check_config_compatibility, _save_config, _collect_materialized
from lossless_lance import LosslessLanceWriter

import environments.flat2d  # noqa: F401  (triggers gymnasium registration)
from environments.flat2d.flat2d_ball import Flat2DBall


@hydra.main(version_base=None, config_path='./config', config_name='flat2d_ball')
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

    # A chunk consumes more seeds than the `chunk` episodes it stores — see
    # collect_cloudgripper_mj_ball.py for the full reasoning. Same situation
    # here: a fast ball crosses sooner, so envs reset out of lockstep and each
    # finished env draws the next seed in the sequence.
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
    slowest = Flat2DBall.crossing_steps(speed_range[0])
    if slowest > cfg.world.max_episode_steps:
        raise ValueError(
            f'speed_range={speed_range}: a crossing at {speed_range[0]} widths/s takes '
            f'{slowest} steps but max_episode_steps is {cfg.world.max_episode_steps}. '
            f'Raise world.max_episode_steps to at least {slowest}.'
        )

    world = swm.World(
        "flat2d/Ball-v0",
        num_envs=cfg.num_envs,
        image_shape=tuple(cfg.world.image_shape),
        max_episode_steps=cfg.world.max_episode_steps,
        # 'nearest': any resize at all must not blend the three colours into
        # intermediate ones. With image_shape equal to the render size this is
        # a no-op, but it keeps the guarantee if someone changes one of them.
        image_resample=cfg.world.image_resample,
        height=cfg.world.height,
        width=cfg.world.width,
        speed_range=speed_range,
        wall_color=list(cfg.world.wall_color),
        floor_color=list(cfg.world.floor_color),
        ball_color=list(cfg.world.ball_color),
        ball_colors=(None if cfg.world.get('ball_colors', None) is None
                     else [list(c) for c in cfg.world.ball_colors]),
        zero_action=cfg.world.zero_action,
        ball_radius=cfg.world.ball_radius,
    )
    policy = instantiate(cfg.policy)
    world.set_policy(policy)

    # PNG, not the Lance writer's JPEG — JPEG ringing around the ball's edge
    # would turn three colours into hundreds. Only used for the lance format;
    # the video format is MP4-encoded and lossy either way.
    writer_factory = (
        (lambda path: LosslessLanceWriter(path)) if format == 'lance' else None
    )

    try:
        collected = 0
        while collected < to_collect:
            chunk = min(cfg.num_envs, to_collect - collected)
            if hasattr(policy, 'reset'):
                policy.reset()
            _collect_materialized(
                world,
                path=lance_out,
                episodes=chunk,
                seed=seed_cursor,
                format=format,
                options=options,
                writer_factory=writer_factory,
            )
            collected += chunk
            seed_cursor += seed_stride
            logging.info(
                f'Collected {n_existing + collected}/{cfg.episodes} episodes → {lance_out}'
            )
    finally:
        world.close()


if __name__ == "__main__":
    run()
