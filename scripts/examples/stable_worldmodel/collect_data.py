#!/usr/bin/env -S uv run 

from pathlib import Path
import os

import stable_worldmodel as swm
from stable_worldmodel.envs.pusht import WeakPolicy


def main():
    WS_DIR = 'STABLEWM_HOME'
    SEED = 42

    # Store collected data in workspace
    root = Path(os.environ.get(WS_DIR, Path.home() / '.stable_worldmodel'))
    dataset_path = root / 'datasets' / 'tutorial_pusht.lance'

    # init pushT environment with policy for data collection
    world = swm.World(
        'swm/PushT-v1',
        num_envs=4,
        image_shape=(64, 64),
        max_episode_steps=100
    )
    world.set_policy(
        WeakPolicy(dist_constraint=100, seed=SEED)
    )

    # collect dataset
    world.collect(
        path=dataset_path,
        episodes=64,
        seed=SEED,
        options = {
            'variation' : [
                'agent.start_position',
                'block.start_position',
                'block.angle',
                'agent.color',
                'block.color'
            ]
        }
    )

    world.close()
    print(f'wrote {dataset_path}')


if __name__ == '__main__':
    main()