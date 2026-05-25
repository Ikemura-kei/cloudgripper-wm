"""Collect data from CloudGripper robots using a random policy.

Usage:
    uv run python scripts/data/collect.py
    uv run python scripts/data/collect.py robots=[robot1,robot2] episodes=100
    uv run python scripts/data/collect.py use_mock=true output=data/mock.lance

Appending to an existing dataset is automatic — just point to the same output
path. The seed is offset by the number of already-collected episodes so new
episodes are never duplicates of existing ones.
"""

from pathlib import Path

import hydra
import lance
from loguru import logger as logging
from omegaconf import DictConfig
from stable_worldmodel.policy import RandomPolicy

from cloudgripper_wm.world import CloudGripperWorld


def _count_existing_episodes(path: str) -> int:
    """Return number of episodes already in the Lance dataset, or 0 if absent."""
    if not Path(path).exists():
        return 0
    try:
        col = lance.dataset(path).to_table(columns=["episode_idx"]).column("episode_idx").to_pylist()
        return max(col) + 1 if col else 0
    except Exception:
        return 0


@hydra.main(version_base=None, config_path='./config', config_name='cloudgripper')
def run(cfg: DictConfig) -> None:
    n_existing = _count_existing_episodes(cfg.output)
    if n_existing:
        logging.info(f'Appending to existing dataset ({n_existing} episodes already collected)')
    seed_start = cfg.seed + n_existing

    world = CloudGripperWorld(
        robot_names=list(cfg.robots),
        use_mock=cfg.use_mock,
        **cfg.world,
    )

    world.set_policy(RandomPolicy(seed=seed_start))

    max_retries = 1000
    for ep in range(cfg.episodes):
        seed = seed_start + ep
        for attempt in range(1, max_retries + 1):
            try:
                world.collect(cfg.output, episodes=1, seed=seed)
                break
            except Exception as exc:
                logging.warning(f'Episode {n_existing + ep + 1} attempt {attempt}/{max_retries} failed: {exc}')
                if attempt == max_retries:
                    raise RuntimeError(
                        f'Episode {n_existing + ep + 1} failed after {max_retries} attempts'
                    ) from exc
                seed += cfg.episodes  # offset so retries don't collide with other episodes' seeds
        logging.info(f'Episode {n_existing + ep + 1} saved (run ep {ep + 1}/{cfg.episodes}) → {cfg.output}')

    world.close()
    logging.success(f'Collected {cfg.episodes} episodes → {cfg.output}')


if __name__ == '__main__':
    run()
