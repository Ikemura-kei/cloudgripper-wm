"""Collect data from CloudGripper robots using a random policy.

Usage:
    uv run python scripts/data/collect.py
    uv run python scripts/data/collect.py robots=[robot1,robot2] episodes=100
    uv run python scripts/data/collect.py use_mock=true output=data/mock.lance
"""

import hydra
from loguru import logger as logging
from omegaconf import DictConfig
from stable_worldmodel.policy import RandomPolicy

from cloudgripper_wm.world import CloudGripperWorld


@hydra.main(version_base=None, config_path='./config', config_name='cloudgripper')
def run(cfg: DictConfig) -> None:
    world = CloudGripperWorld(
        robot_names=list(cfg.robots),
        use_mock=cfg.use_mock,
        **cfg.world,
    )

    world.set_policy(RandomPolicy(seed=cfg.seed))

    for ep in range(cfg.episodes):
        world.collect(cfg.output, episodes=1, seed=cfg.seed + ep)
        logging.info(f'Episode {ep + 1}/{cfg.episodes} saved → {cfg.output}')

    world.close()
    logging.success(f'Collected {cfg.episodes} episodes → {cfg.output}')


if __name__ == '__main__':
    run()
